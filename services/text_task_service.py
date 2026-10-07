"""Durable receipts for bound text requests, independent of HTTP lifetime.

Like image tasks, callers choose the request identity before sending. The
receipt stores results and cursors, never access tokens or uploaded image data.
An interrupted upstream operation is queried, never automatically resubmitted.
"""
from __future__ import annotations

import heapq
import itertools
import threading
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from services.config import DATA_DIR
from services.conversation_binding_service import (
    ConversationBindingError,
    NON_TEXT_RESULT_FIELD,
    OBSERVED_CONTENT_TYPES,
    RECOVERY_CONVERSATION_COVERAGE_VERSION_FIELD,
    RECOVERY_CONVERSATION_SCAN_FIELD,
    TextRecoveryReason,
    safe_scan_failures,
    safe_recovery_read_error,
    TURN_END_EVIDENCE_FIELD,
    conversation_binding_service,
    is_recovery_image_pointer,
)


from services.task_store import TaskStore, recovery_control
from services.request_context import current_request, AdmissionLost


class TextTaskCapacityError(RuntimeError):
    pass


def bind_waiting_public_sessions(store, db, receipts):
    """Resolve accepted successor cursors only from a successful original turn."""
    owned = {(owner, request_id): receipt for kind, owner, request_id, receipt in receipts if kind == "text"}
    fields = ("provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id")
    for kind, owner, request_id, receipt in receipts:
        if kind != "text" or receipt.get("status") != "queued" or not receipt.get("_public_previous_waiting"):
            continue
        previous = owned.get((owner, receipt.get("_previous_request_id"))) or {}
        reason = "CHAT_PREVIOUS_REQUEST_PENDING"
        if previous.get("status") == "succeeded" and previous.get("upstream_outcome") != "unknown":
            if not all(isinstance(previous.get(key), str) and previous[key] for key in fields):
                reason = "CHAT_CONTINUATION_UNAVAILABLE"
            elif (receipt.get("_requested_account_identity")
                  and receipt["_requested_account_identity"] != previous["provider_account_identity"]):
                receipt.update(status="failed", error_code="CHAT_ACCOUNT_SELECTION_CONFLICT", upstream_outcome="not_sent")
                reason = "CHAT_ACCOUNT_SELECTION_CONFLICT"
            else:
                receipt.update({key: previous[key] for key in fields})
                receipt["_submission_parent_message_id"] = previous["parent_message_id"]
                receipt.pop("_public_previous_waiting", None)
                receipt.pop("_public_previous_waiting_reason", None)
                store.write_receipt(db, kind, owner, request_id, receipt)
                continue
        elif previous.get("status") in {"unknown", "failed"}:
            reason = "CHAT_PREVIOUS_REQUEST_UNKNOWN" if previous.get("status") == "unknown" or previous.get("upstream_outcome") == "unknown" else "CHAT_PREVIOUS_REQUEST_FAILED"
        receipt["_public_previous_waiting_reason"] = reason
        store.write_receipt(db, kind, owner, request_id, receipt)


def _retained_size(value, seen=None):
    """Estimate the Python heap retained by one scheduled request body."""
    seen = seen if seen is not None else set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(
            _retained_size(key, seen) + _retained_size(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(_retained_size(item, seen) for item in value)
    return size


class ContinuationExecutor:
    """Finish ready products before starting more first-turn gallery requests."""
    DEFAULT_MAX_OUTSTANDING = 32
    DEFAULT_MAX_RETAINED_BYTES = 256 * 1024 * 1024

    def __init__(
        self,
        max_workers=4,
        *,
        max_outstanding=DEFAULT_MAX_OUTSTANDING,
        max_retained_bytes=DEFAULT_MAX_RETAINED_BYTES,
    ):
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bound-text")
        self.limit = max_workers
        self.max_outstanding = max(1, int(max_outstanding))
        self.max_retained_bytes = max(1, int(max_retained_bytes))
        self.lock = threading.Lock()
        self.queue = []
        self.sequence = itertools.count()
        self.active = 0
        self.outstanding = 0
        self.retained_bytes = 0
        self.closed = False

    def submit(self, function, *args):
        future = Future()
        body = args[-1] if args and isinstance(args[-1], dict) else {}
        priority = 0 if body.get("conversation_id") else 1
        retained_bytes = sum(_retained_size(item) for item in args)
        with self.lock:
            if self.closed:
                raise RuntimeError("executor shut down")
            if (
                self.outstanding >= self.max_outstanding
                or self.retained_bytes + retained_bytes > self.max_retained_bytes
            ):
                raise TextTaskCapacityError("text task capacity exceeded")
            self.outstanding += 1
            self.retained_bytes += retained_bytes
            entry = (
                priority, next(self.sequence), future, function, args, retained_bytes,
            )
            heapq.heappush(self.queue, entry)
            if self.active < self.limit:
                self.active += 1
                try:
                    self.pool.submit(self._drain)
                except BaseException:
                    self.active -= 1
                    self.queue.remove(entry)
                    heapq.heapify(self.queue)
                    self.outstanding -= 1
                    self.retained_bytes -= retained_bytes
                    raise
        return future

    def _drain(self):
        while True:
            with self.lock:
                if not self.queue:
                    self.active -= 1
                    return
                _, _, future, function, args, retained_bytes = heapq.heappop(self.queue)
            try:
                if future.set_running_or_notify_cancel():
                    future.set_result(function(*args))
            except BaseException as exc:
                future.set_exception(exc)
            finally:
                with self.lock:
                    self.outstanding -= 1
                    self.retained_bytes -= retained_bytes

    def shutdown(self, wait=True):
        with self.lock:
            self.closed = True
        self.pool.shutdown(wait=wait)


class TextTaskService:
    RECOVERY_LEASE_SECONDS = 60.0
    RECOVERY_BASE_BACKOFF_SECONDS = 30.0
    RECOVERY_MAX_BACKOFF_SECONDS = 15.0 * 60.0
    UNRECOVERABLE_MIN_AGE_SECONDS = 15.0 * 60.0
    UNRECOVERABLE_QUALIFIED_READS = 3
    _INTERNAL_RECEIPT_FIELDS = frozenset({
        "boot", "recovery_claim_id", "recovery_claimed_at", "recovery_lease_until",
        RECOVERY_CONVERSATION_SCAN_FIELD, RECOVERY_CONVERSATION_COVERAGE_VERSION_FIELD,
        NON_TEXT_RESULT_FIELD,
    })
    _SAFE_RECOVERY_CODES = frozenset({
        "CONVERSATION_BINDING_UNAVAILABLE",
        "CONVERSATION_BINDING_UNSUPPORTED",
        "CONVERSATION_BINDING_CONTRACT_INVALID",
        "CONVERSATION_BINDING_MISMATCH",
        "CONVERSATION_OUTCOME_UNKNOWN",
        "RECOVERY_TRANSPORT_FAILED",
        "RECOVERY_RATE_LIMITED",
        "RECOVERY_AUTH_REQUIRED",
        "RECOVERY_READ_FAILED",
    })
    _SUCCESS_RECOVERY_FIELDS = frozenset({
        # Binding and conversation identity stay rooted in the original
        # receipt unless the original receipt never captured a conversation.
        # Only an exact-account request lookup may fill that missing anchor.
        "content", "conversation_id", "parent_message_id",
        "request_parent_message_id", "binding_status",
    })
    _SAFE_RECOVERY_REASONS = frozenset(reason.value for reason in TextRecoveryReason)
    _UNRECOVERABLE_REASONS = frozenset({
        TextRecoveryReason.CONVERSATION_NOT_FOUND.value,
        TextRecoveryReason.REQUEST_MESSAGE_NOT_FOUND.value,
        TextRecoveryReason.REQUEST_BRANCH_SUPERSEDED.value,
        TextRecoveryReason.REQUEST_RESULT_NOT_FOUND.value,
        TextRecoveryReason.REQUEST_RESULT_TERMINAL_EMPTY.value,
        TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value,
    })
    _VALID_CONVERSATION_REASONS = frozenset({
        TextRecoveryReason.REQUEST_MESSAGE_NOT_FOUND.value,
        TextRecoveryReason.REQUEST_PARENT_MISMATCH.value,
        TextRecoveryReason.REQUEST_BRANCH_AMBIGUOUS.value,
        TextRecoveryReason.REQUEST_BRANCH_SUPERSEDED.value,
        TextRecoveryReason.REQUEST_RESULT_INCOMPLETE.value,
        TextRecoveryReason.REQUEST_CONVERSATION_ADVANCED.value,
        TextRecoveryReason.REQUEST_RESULT_NOT_FOUND.value,
        TextRecoveryReason.REQUEST_RESULT_TERMINAL_EMPTY.value,
    })

    def __init__(self, path: Path, runner=None, executor=None, *, clock=None, recovery_reader=None, admission=None):
        self.path = path
        self.store = TaskStore(path)
        self.admission = admission
        self.runner = runner or conversation_binding_service.complete_text
        self.executor = executor or ContinuationExecutor()
        self.clock = clock or time.time
        self.recovery_reader = recovery_reader or conversation_binding_service.read_text_request
        self.boot = uuid.uuid4().hex
        if self.store.path.exists():
            self._refresh_queued_forward_models()

    def _refresh_queued_forward_models(self):
        # Upgrade only the derived dispatch model of unclaimed waiting calls.
        # Read private inputs outside the DB transaction; a concurrent claim
        # invalidates the final compare-and-set. Original hashes/inputs stay put.
        from services.durable_forward import SEARCH_RECOVERY_PROTOCOLS, dispatch_model
        with self._db() as db:
            rows = db.execute("SELECT owner,id,request_hash,receipt FROM requests").fetchall()
        for owner, request_id, request_hash, raw in rows:
            receipt = json.loads(raw)
            if (receipt.get("status") != "queued" or receipt.get("_claim_id")
                    or receipt.get("_submission_started") or not receipt.get("_input_ref")
                    or receipt.get("_forward_protocol") not in SEARCH_RECOVERY_PROTOCOLS):
                continue
            try:
                body = self.store.load_input(receipt["_input_ref"])
                if not isinstance(body, dict):
                    continue
                if self._submission_identity(owner, body) != (request_id, request_hash):
                    continue  # Existing admission input validation owns this failure.
                model = dispatch_model(body)
            except (OSError, ValueError, TypeError, KeyError, ConversationBindingError):
                continue
            if model != receipt.get("model"):
                with self._db() as db:
                    db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=? AND receipt=?",
                               (json.dumps({**receipt, "model": model}), owner, request_id, raw))

    def _now(self):
        return float(self.clock())

    @contextmanager
    def _db(self):
        with self.store.connect() as db:
            yield db

    @staticmethod
    def _public_not_submitted(receipt):
        """Whether this terminal receipt proves the upstream text turn was not sent.

        This is deliberately narrower than a false private flag: a claim can be
        queued or in flight before its send guard runs, and a previously sent
        request retains its send sequence. Only the terminal parent-mismatch
        failure that occurred before that guard may be published as known not
        submitted.
        """
        required_identity = (
            "request_id", "client_conversation_id", "provider_binding_id",
            "provider_account_identity", "conversation_id", "parent_message_id",
        )
        return (
            receipt.get("status") == "failed"
            and receipt.get("error_code") == "CONVERSATION_BINDING_MISMATCH"
            and receipt.get("_submission_started") is False
            and receipt.get("_executing") is False
            and "_last_sent_sequence" not in receipt
            and all(isinstance(receipt.get(field), str) and receipt[field].strip() for field in required_identity)
        )

    @staticmethod
    def _public(receipt):
        from services.public_chat_service import project_text_execution
        result = {k: v for k, v in receipt.items() if k not in TextTaskService._INTERNAL_RECEIPT_FIELDS and not k.startswith("_")}
        if receipt.get("status") == "succeeded":
            # Older successful retries retained their previous active error.
            # Correct the read projection without rewriting original diagnostics.
            result.update(error_code=None, waiting=None, upstream_outcome="completed")
        result.pop("execution", None)
        result["recovery_control"] = recovery_control(receipt)
        if (receipt.get("_route", "chat") == "chat" and receipt.get("_operation", "text") == "text"
                and not receipt.get("_forward_protocol")):
            result["execution"] = project_text_execution(receipt)
        # Do not trust an arbitrary persisted public projection. This field is
        # emitted only from the exact internal terminal-state predicate above.
        result.pop("upstream_submission_started", None)
        result.pop("bound_successor_resume_retryable", None)
        if TextTaskService._public_not_submitted(receipt):
            result["upstream_submission_started"] = False
        # Derive diagnostics only from validated private records. Never trust a
        # stored projection or arbitrary exception content at the public edge.
        if receipt.get("_scheduling") is not None:
            result["scheduling"] = receipt["_scheduling"]
        result.pop("recovery_scan_progress", None)
        result.pop("recovery_last_read_error", None)
        read_error = safe_recovery_read_error(receipt.get("recovery_last_read_error"))
        if read_error is not None:
            result["recovery_last_read_error"] = read_error
        scan = TextTaskService._safe_recovery_scan(receipt.get(RECOVERY_CONVERSATION_SCAN_FIELD))
        if scan:
            failures = scan.get("failed_reads", {})
            result["recovery_scan_progress"] = {
                "list_complete": scan["coverage_complete"], "listed_total": scan["next_offset"],
                "window_candidates": len(scan["conversation_ids"]), "window_checked": scan["next_index"],
                "window_pending": len(scan["conversation_ids"]) - scan["next_index"],
                "matches": len(scan["matches"]),
                "failed_reads": [{**row["error"], "attempts": row["attempts"], "next_at": row["next_at"]}
                                 for row in failures.values()],
            }
        if receipt.get("_public_session_ref"):
            result["conversation"] = {"client_conversation_id": receipt["_public_session_ref"],
                                      "previous_request_id": receipt.get("_previous_request_id"),
                                      "protocol": "sequential-v1"}
        evidence = TextTaskService._verified_terminal_empty(receipt)
        if evidence:
            result["terminal_empty"] = {"verified": True, "original_request_id": receipt["request_id"],
                                        "observed_at": evidence["observed_at"],
                                        "same_conversation_continuation": True}
        if receipt.get("_terminal_empty_correction_of"):
            result["correction_of_request_id"] = receipt["_terminal_empty_correction_of"]
        if receipt.get("_supersedes_request_id"):
            result["supersedes_request_id"] = receipt["_supersedes_request_id"]
        if receipt.get("_derived_input"):
            result["derived_input"] = receipt["_derived_input"]
        completion = receipt.get("_completion")
        if isinstance(completion, dict):
            result["completion"] = {key: completion[key] for key in (
                "state", "reason", "replacement_id", "selected_id", "max_extra_requests", "conversation_mode",
                "automatic_empty_retry") if key in completion}
        return result

    @staticmethod
    def _cancelled_completion_child(receipt, original):
        state = (original or {}).get("_completion") or {}
        return bool(original and original.get("status") == "succeeded"
                    and state.get("selected_id") == original.get("request_id")
                    and state.get("replacement_id") == receipt.get("request_id")
                    and receipt.get("_completion_of") == original.get("request_id")
                    and receipt.get("status") == "failed"
                    and receipt.get("error_code") == "COMPLETION_ORIGINAL_RECOVERED"
                    and receipt.get("upstream_outcome") == "not_sent"
                    and not receipt.get("_submission_started"))

    @staticmethod
    def _verified_terminal_empty(receipt):
        """Only an exact, persisted original-turn proof can open correction."""
        evidence = receipt.get(TURN_END_EVIDENCE_FIELD)
        if (receipt.get("route") != "chat" or not receipt.get("_public_session_ref")
                or receipt.get("status") not in {"unknown", "failed"}
                or (receipt.get("status") == "failed" and (
                    receipt.get("error_code") != "RESULT_UNRECOVERABLE" or receipt.get("upstream_outcome") != "unknown"))
                or receipt.get("recovery_reason") != TextRecoveryReason.REQUEST_RESULT_TERMINAL_EMPTY.value
                or receipt.get("_upstream_terminal") is not True
                or receipt.get("recovery_requires_new_conversation") is True
                or not isinstance(evidence, dict)
                or set(evidence) != {"conversation_id", "request_message_id", "final_message_id", "observed_at"}
                or any(not isinstance(evidence.get(k), str) or not evidence[k].strip()
                       for k in ("conversation_id", "request_message_id", "final_message_id"))
                or evidence["request_message_id"] != receipt.get("request_message_id")
                or evidence["final_message_id"] == evidence["request_message_id"]
                or evidence["conversation_id"] != receipt.get("conversation_id")
                or type(evidence["observed_at"]) not in {int, float}
                or not math.isfinite(evidence["observed_at"]) or evidence["observed_at"] <= 0
                or any(not isinstance(receipt.get(k), str) or not receipt[k]
                       for k in ("provider_binding_id", "provider_account_identity", "client_conversation_id", "model"))):
            return None
        return evidence

    @classmethod
    def _verified_retryable_empty(cls, receipt):
        """An ended-empty turn OR a verified empty response after local completion.

        The latter is a one-retry policy, not evidence of remote cancellation.
        Legacy responses without stream diagnostics need the existing bounded
        unchanged-result observation; arbitrary timeouts are insufficient.
        """
        ended = cls._verified_terminal_empty(receipt)
        if ended:
            return {**ended, "retry_parent_message_id": ended["final_message_id"]}
        evidence = receipt.get("_empty_reply_evidence")
        if (receipt.get("route") != "chat" or not receipt.get("_public_session_ref")
                or receipt.get("_forward_protocol") or receipt.get("_operation", "text") != "text"
                or receipt.get("status") != "unknown" or receipt.get("_executing")
                or receipt.get("content") or receipt.get("recovery_requires_new_conversation")
                or not cls._safe_empty_reply_evidence(evidence, receipt)
                or any(not isinstance(receipt.get(k), str) or not receipt[k] for k in (
                    "provider_binding_id", "provider_account_identity", "client_conversation_id", "model"))):
            return None
        timeline = receipt.get("_execution_timeline") or []
        stages = {item.get("stage") for item in timeline if isinstance(item, dict)}
        stream = next((item for item in reversed(timeline) if isinstance(item, dict)
                       and item.get("stage") == "stream_finished"), {})
        closed = (stream.get("stream_end") == "done" and not stream.get("sse_error_event")
                  and not stream.get("sse_parse_errors"))
        legacy_observed = (not stream and type(receipt.get("_result_wait_ended_at")) in {int, float}
                           and (receipt.get("_result_no_progress_reads") or 0) >= 3)
        if (not {"send_call_started", "response_headers_received", "task_finished"} <= stages
                or not (closed or legacy_observed)):
            return None
        return evidence

    @staticmethod
    def _safe_empty_reply_evidence(evidence, receipt):
        fields = {"conversation_id", "request_message_id", "final_message_id", "retry_parent_message_id", "observed_at"}
        if (not isinstance(evidence, dict) or set(evidence) != fields
                or any(not isinstance(evidence.get(k), str) or not 1 <= len(evidence[k]) <= 200
                       for k in fields - {"observed_at"})
                or evidence["conversation_id"] != receipt.get("conversation_id")
                or evidence["request_message_id"] != receipt.get("request_message_id")
                or evidence["final_message_id"] == evidence["request_message_id"]
                or evidence["retry_parent_message_id"] == evidence["request_message_id"]
                or type(evidence["observed_at"]) not in {int, float}
                or not math.isfinite(evidence["observed_at"]) or evidence["observed_at"] <= 0):
            return None
        return dict(evidence)

    @classmethod
    def _fresh_terminal_empty_continuation(cls, previous, correction, recovered):
        """Revalidate the original empty branch and the chosen retry head."""
        if any(item.get(flag) is True for item in (previous or {}, correction)
               for flag in ("_recovery_suppressed", "_recovery_paused")):
            return False
        saved = cls._verified_retryable_empty(previous or {})
        if not saved or not isinstance(recovered, dict):
            return False
        validated, error_code, _, reason = cls._safe_recovery_result(recovered, previous)
        fresh_base = {k: v for k, v in previous.items() if k not in {
            TURN_END_EVIDENCE_FIELD, "_empty_reply_evidence", "_upstream_terminal"}}
        fresh = cls._verified_retryable_empty({**fresh_base, **(validated or {}), "recovery_reason": reason})
        return bool(
            error_code == "UPSTREAM_OUTCOME_UNKNOWN"
            and reason in {TextRecoveryReason.REQUEST_RESULT_TERMINAL_EMPTY.value,
                           TextRecoveryReason.REQUEST_RESULT_INCOMPLETE.value,
                           TextRecoveryReason.REQUEST_CONVERSATION_ADVANCED.value}
            and isinstance(fresh, dict)
            and all(fresh[key] == saved[key] for key in (
                "conversation_id", "request_message_id", "final_message_id", "retry_parent_message_id"))
            and fresh["observed_at"] >= saved["observed_at"]
            and correction.get("_terminal_empty_correction_of") == previous.get("request_id")
            and correction.get("_previous_request_id") == previous.get("request_id")
            and correction.get("route") == "chat"
            and correction.get("model") == previous.get("model")
            and correction.get("_public_session_ref") == previous.get("_public_session_ref")
            and correction.get("client_conversation_id") == previous.get("client_conversation_id")
            and correction.get("parent_message_id") == saved["retry_parent_message_id"]
            and correction.get("request_message_id") != previous.get("request_message_id")
            and all(previous.get(key) == correction.get(key) for key in (
                "provider_binding_id", "provider_account_identity", "conversation_id"))
        )

    @classmethod
    def _recovery_due(cls, receipt, now):
        if receipt.get("_forward_protocol"):
            from services.durable_forward import chat_recovery_supported
            if not chat_recovery_supported(receipt):
                return False
        next_at = receipt.get("recovery_next_at")
        lease_until = receipt.get("recovery_lease_until")
        if lease_until is not None and now < float(lease_until):
            return False
        return next_at is None or now >= float(next_at)

    @classmethod
    def _recovery_backoff(cls, attempt):
        exponent = max(0, min(int(attempt) - 1, 5))
        return min(cls.RECOVERY_MAX_BACKOFF_SECONDS, cls.RECOVERY_BASE_BACKOFF_SECONDS * (2 ** exponent))

    @classmethod
    def _safe_recovery_code(cls, exc):
        code = str(getattr(exc, "code", "")).strip().upper()
        return code if code in cls._SAFE_RECOVERY_CODES else "RECOVERY_READ_FAILED"

    @classmethod
    def _recovery_failure(cls, exc):
        evidence = safe_recovery_read_error(getattr(exc, "recovery_read_error", None)) or {}
        status = getattr(exc, "status_code", None) or evidence.get("http_status")
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is None:
            retry_after = evidence.get("retry_after_seconds")
        safe_retry_after = (
            int(retry_after)
            if isinstance(retry_after, (int, float))
            and not isinstance(retry_after, bool)
            and math.isfinite(retry_after) and 0 <= retry_after <= 2147483647
            else None
        )
        if status == 429:
            return "RECOVERY_RATE_LIMITED", safe_retry_after
        if status in {401, 403}:
            return "RECOVERY_AUTH_REQUIRED", None
        exc_type = type(exc)
        type_name = f"{exc_type.__module__}.{exc_type.__name__}".lower()
        if (
            (isinstance(exc, (ConnectionError, OSError)) and not isinstance(exc, TimeoutError))
            or any(marker in type_name for marker in ("curl_cffi", "connection", "network"))
        ):
            return "RECOVERY_TRANSPORT_FAILED", None
        return cls._safe_recovery_code(exc), safe_retry_after if evidence else None

    @classmethod
    def _safe_recovery_reason(cls, value):
        raw = value.value if isinstance(value, TextRecoveryReason) else str(value or "").strip().upper()
        return raw if raw in cls._SAFE_RECOVERY_REASONS else None

    @classmethod
    def _safe_recovery_result(cls, recovered, receipt=None):
        if not isinstance(recovered, dict):
            return None, "RECOVERY_INVALID_RESULT", "read_text_result", None
        status = recovered.get("status")
        if status == "failed":
            evidence = recovered.get(NON_TEXT_RESULT_FIELD)
            receipt = receipt or {}
            if (recovered.get("error_code") != "CHAT_RESPONSE_NOT_TEXT"
                    or recovered.get("recovery_reason") != TextRecoveryReason.REQUEST_RESULT_NON_TEXT.value
                    or recovered.get("binding_status") != "bound"
                    or not isinstance(evidence, dict)
                    or set(evidence) != {"conversation_id", "request_message_id", "final_message_id", "artifacts"}
                    or any(not isinstance(evidence.get(key), str) or not evidence[key].strip()
                           for key in ("conversation_id", "request_message_id", "final_message_id"))
                    or evidence["request_message_id"] != receipt.get("request_message_id")
                    or evidence["final_message_id"] == evidence["request_message_id"]
                    or evidence["final_message_id"] != recovered.get("parent_message_id")
                    or evidence["conversation_id"] != recovered.get("conversation_id")
                    or (receipt.get("conversation_id") and evidence["conversation_id"] != receipt["conversation_id"])
                    or any(not receipt.get(key) or recovered.get(key) != receipt[key]
                           for key in ("provider_binding_id", "provider_account_identity", "client_conversation_id"))):
                return None, "RECOVERY_INVALID_RESULT", "read_text_result", None
            artifacts = evidence["artifacts"]
            if (not isinstance(artifacts, list) or not artifacts or any(
                not isinstance(artifact, dict) or set(artifact) != {"tool_message_id", "asset_pointer"}
                or not isinstance(artifact["tool_message_id"], str) or not artifact["tool_message_id"].strip()
                or artifact["tool_message_id"] in {evidence["request_message_id"], evidence["final_message_id"]}
                or not is_recovery_image_pointer(artifact["asset_pointer"])
                for artifact in artifacts
            )):
                return None, "RECOVERY_INVALID_RESULT", "read_text_result", None
            return {
                "status": "failed", "error_code": "CHAT_RESPONSE_NOT_TEXT", "binding_status": "bound",
                "conversation_id": evidence["conversation_id"], "parent_message_id": evidence["final_message_id"],
                NON_TEXT_RESULT_FIELD: evidence,
                "result": {"type": "non_text", "artifact_type": "image", "artifact_count": len(artifacts)},
            }, None, None, None
        if status == "succeeded":
            content = recovered.get("content")
            parent_message_id = recovered.get("parent_message_id")
            if (recovered.get("binding_status") != "bound"
                    or not isinstance(content, str) or not content.strip()
                    or not isinstance(parent_message_id, str) or not parent_message_id.strip()):
                return None, "RECOVERY_INVALID_RESULT", "read_text_result", None
            result = {key: recovered[key] for key in cls._SUCCESS_RECOVERY_FIELDS if key in recovered}
            if ((receipt or {}).get("_chat_recovery") or {}).get("kind") == "search":
                search = recovered.get("_search_result")
                if (not isinstance(search, dict)
                        or set(search) != {"conversation_id", "status", "answer", "sources", "assistant_message_id", "create_time"}
                        or search["conversation_id"] != recovered.get("conversation_id")
                        or search["assistant_message_id"] != parent_message_id
                        or search["answer"] != content or search["status"] != "finished_successfully"
                        or type(search["create_time"]) not in {int, float} or not math.isfinite(search["create_time"])
                        or not isinstance(search["sources"], list)
                        or any(not isinstance(source, dict) or set(source) != {"title", "url", "snippet", "source_type"}
                               or any(not isinstance(value, str) for value in source.values())
                               for source in search["sources"])):
                    return None, "RECOVERY_INVALID_RESULT", "read_text_result", None
                result["_search_result"] = search
            scan = cls._safe_recovery_scan(recovered.get(RECOVERY_CONVERSATION_SCAN_FIELD))
            if scan is not None:
                result[RECOVERY_CONVERSATION_SCAN_FIELD] = scan
            return result, None, None, None
        if status in {"running", "unknown"}:
            anchor = {}
            for key in ("conversation_id", "parent_message_id", "request_parent_message_id"):
                value = recovered.get(key)
                if value is not None:
                    if not isinstance(value, str):
                        return None, "RECOVERY_INVALID_RESULT", "read_text_result", None
                    if key != "request_parent_message_id" and not value.strip():
                        return None, "RECOVERY_INVALID_RESULT", "read_text_result", None
                    anchor[key] = value
            empty = recovered.get("_empty_reply_evidence")
            from services.generation_completion import retry_evidence
            retry = retry_evidence({**(receipt or {}), "_retry_cursor": recovered.get("_retry_cursor")})
            anchor["_retry_cursor"] = retry
            if empty is not None:
                valid_empty = cls._safe_empty_reply_evidence(empty, receipt or {})
                if (not valid_empty or any(not (receipt or {}).get(k) or recovered.get(k) != receipt[k]
                        for k in ("provider_binding_id", "provider_account_identity", "client_conversation_id"))):
                    return None, "RECOVERY_INVALID_RESULT", "read_text_result", None
                anchor["_empty_reply_evidence"] = valid_empty
            observation = cls._safe_result_observation(recovered.get("_original_result_observation"), receipt or {})
            if observation is not None and recovered.get("recovery_reason") == TextRecoveryReason.REQUEST_RESULT_INCOMPLETE.value:
                anchor["_original_result_observation"] = observation
            scan = cls._safe_recovery_scan(recovered.get(RECOVERY_CONVERSATION_SCAN_FIELD))
            if scan is not None:
                anchor[RECOVERY_CONVERSATION_SCAN_FIELD] = scan
            ended = recovered.get(TURN_END_EVIDENCE_FIELD)
            if ended is not None:
                receipt = receipt or {}
                if (status != "unknown"
                        or recovered.get("recovery_reason") != TextRecoveryReason.REQUEST_RESULT_TERMINAL_EMPTY.value
                        or not isinstance(ended, dict)
                        or set(ended) != {"conversation_id", "request_message_id", "final_message_id", "observed_at"}
                        or any(not isinstance(ended.get(k), str) or not ended[k].strip()
                               for k in ("conversation_id", "request_message_id", "final_message_id"))
                        or ended["request_message_id"] != receipt.get("request_message_id")
                        or ended["final_message_id"] == ended["request_message_id"]
                        or ended["conversation_id"] != recovered.get("conversation_id")
                        or receipt.get("conversation_id") and ended["conversation_id"] != receipt["conversation_id"]
                        or type(ended["observed_at"]) not in {int, float} or not math.isfinite(ended["observed_at"])
                        or ended["observed_at"] <= 0
                        or any(not receipt.get(k) or recovered.get(k) != receipt[k]
                               for k in ("provider_binding_id", "provider_account_identity", "client_conversation_id"))):
                    return None, "RECOVERY_INVALID_RESULT", "read_text_result", None
                anchor.update({TURN_END_EVIDENCE_FIELD: dict(ended), "_upstream_terminal": True, "_turn_reserved": False})
            return anchor or None, "UPSTREAM_OUTCOME_UNKNOWN", "read_text_result", cls._safe_recovery_reason(recovered.get("recovery_reason"))
        return None, "RECOVERY_INVALID_RESULT", "read_text_result", None

    @staticmethod
    def _safe_result_observation(value, receipt):
        if (not isinstance(value, dict)
                or set(value) != {"conversation_id", "request_message_id", "nodes", "upstream_updated_at"}
                or not value.get("conversation_id")
                or value["conversation_id"] != receipt.get("conversation_id")
                or not value.get("request_message_id")
                or value["request_message_id"] != receipt.get("request_message_id")):
            return None
        nodes = value["nodes"]
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= 128:
            return None
        ids, normalized = set(), []
        legacy_fields = {"id", "role", "status", "end_turn", "text_chars"}
        for raw in nodes:
            if not isinstance(raw, dict):
                return None
            # The first candidate measured pure text only. Normalize it so a
            # reader upgrade does not manufacture progress or reset its clock.
            node = ({**raw, "content_type": "text", "content_items": 0, "finished_items": 0}
                    if set(raw) == legacy_fields else raw)
            if (set(node) != legacy_fields | {"content_type", "content_items", "finished_items"}
                    or not isinstance(node["id"], str) or not 1 <= len(node["id"]) <= 200
                    or node["id"] in ids or node["id"] == value["request_message_id"]
                    or node["role"] not in {"assistant", "tool"}
                    or node["status"] not in {"in_progress", "running", "pending", "queued", "finished_successfully"}
                    or type(node["end_turn"]) is not bool
                    or type(node["text_chars"]) is not int or not 0 <= node["text_chars"] <= 100_000_000
                    or not isinstance(node["content_type"], str) or node["content_type"] not in OBSERVED_CONTENT_TYPES
                    or type(node["content_items"]) is not int or not 0 <= node["content_items"] <= 65_664
                    or type(node["finished_items"]) is not int or not 0 <= node["finished_items"] <= 128):
                return None
            ids.add(node["id"])
            normalized.append(dict(node))
        updated = value["upstream_updated_at"]
        if updated is not None and (type(updated) not in (int, float) or not math.isfinite(updated) or updated <= 0):
            return None
        return {**value, "nodes": normalized}

    @staticmethod
    def _safe_recovery_scan(value):
        if value == {}:
            return {}
        if not isinstance(value, dict):
            return None
        identity = value.get("identity")
        conversation_ids = value.get("conversation_ids")
        next_offset = value.get("next_offset")
        coverage_complete = value.get("coverage_complete")
        time_order_valid = value.get("time_order_valid")
        last_update_time = value.get("last_update_time")
        next_index = value.get("next_index")
        matches = value.get("matches")
        if (
            set(value) - {"failed_reads"} != {
                "identity", "conversation_ids", "next_offset", "coverage_complete",
                "time_order_valid", "last_update_time", "next_index", "matches",
            }
            or not isinstance(identity, dict)
            or set(identity) != {
                "provider_binding_id", "provider_account_identity",
                "client_conversation_id", "request_message_id",
            }
            or any(
                not isinstance(item, str) or not item or len(item) > 300
                for item in identity.values()
            )
            or not isinstance(conversation_ids, list) or len(conversation_ids) > 100
            or any(not isinstance(item, str) or not item or len(item) > 200 for item in conversation_ids)
            or len(set(conversation_ids)) != len(conversation_ids)
            or not isinstance(next_offset, int) or isinstance(next_offset, bool)
            or next_offset < len(conversation_ids)
            or not isinstance(coverage_complete, bool)
            or not isinstance(time_order_valid, bool)
            or (last_update_time is not None and (
                isinstance(last_update_time, bool)
                or not isinstance(last_update_time, (int, float))
                or not math.isfinite(last_update_time)
                or last_update_time <= 0
            ))
            or (not time_order_valid and last_update_time is not None)
            or not isinstance(next_index, int) or isinstance(next_index, bool)
            or next_index < 0 or next_index > len(conversation_ids)
            or not isinstance(matches, list) or len(matches) > 2
        ):
            return None
        failures = safe_scan_failures(value.get("failed_reads", {}), conversation_ids[next_index:])
        if failures is None:
            return None
        for match in matches:
            if (
                not isinstance(match, dict)
                or set(match) != {"conversation_id", "request_parent_message_id"}
                or match.get("conversation_id") not in conversation_ids
                or not isinstance(match.get("request_parent_message_id"), str)
                or len(match["request_parent_message_id"]) > 200
            ):
                return None
        return {
            "identity": dict(identity),
            "conversation_ids": list(conversation_ids),
            "next_offset": next_offset,
            "coverage_complete": coverage_complete,
            "time_order_valid": time_order_valid,
            "last_update_time": last_update_time,
            "next_index": next_index,
            "matches": [dict(match) for match in matches],
            **({"failed_reads": failures} if failures else {}),
        }

    @staticmethod
    def _bounded_chat_wait(receipt):
        return (receipt.get("_route", "chat") == "chat"
                and receipt.get("_operation", "text") == "text"
                and not receipt.get("_forward_protocol")
                and receipt.get("_recovery_suppressed") is not True
                and receipt.get("_recovery_paused") is not True)

    @classmethod
    def _end_execution_wait(cls, receipt, now, *, observed=False):
        """End local waiting, not the upstream turn or its original-ID recovery.

        The existing age and qualified-read policy bounds Chat execution
        occupancy. A timeout, failed GET or arbitrary failed receipt alone
        cannot do so. Persist this decision so later failed reads and restart
        cannot reopen a released slot.
        """
        if not cls._bounded_chat_wait(receipt) or receipt.get("_execution_wait_ended_at") is not None:
            return False
        if not (receipt.get("status") == "unknown"
                or receipt.get("status") == "failed" and receipt.get("error_code") == "RESULT_UNRECOVERABLE"
                and receipt.get("upstream_outcome") == "unknown"):
            return False
        if receipt.get("recovery_claim_id") or receipt.get("_upstream_terminal") is True:
            # Proven empty terminal results already release execution capacity
            # and retain their separate explicit continuation contract.
            return False
        if not all(receipt.get(key) for key in (
                "request_message_id", "provider_binding_id", "provider_account_identity",
                "client_conversation_id")):
            return False
        claim_until = receipt.get("_claim_until")
        if receipt.get("_claim_id") and (type(claim_until) not in (int, float)
                or not math.isfinite(claim_until) or claim_until > now):
            return False
        if receipt.get("_executing") is True and not receipt.get("_claim_id"):
            return False  # A legacy executor without a lease cannot be fenced.
        if receipt.get("recovery_reason") == TextRecoveryReason.REQUEST_RESULT_INCOMPLETE.value:
            # A stale upstream label can end the foreground result wait, but
            # is never evidence to release the account turn. Only a fresh,
            # validated original GET may make this decision, not a timer/read
            # of old stored observations. Keep UNKNOWN and the original lease.
            progress = receipt.get("_result_last_progress_at") or receipt.get("_result_observation_started_at")
            reads = receipt.get("_result_no_progress_reads")
            if (not observed or receipt.get("_result_wait_ended_at") is not None
                    or type(progress) not in (int, float) or not math.isfinite(progress)
                    or now - progress < cls.UNRECOVERABLE_MIN_AGE_SECONDS
                    or type(reads) is not int or reads < cls.UNRECOVERABLE_QUALIFIED_READS):
                return False
            receipt.update(_result_wait_ended_at=now, updated_at=now)
            return True
        created = receipt.get("created_at")
        reads = receipt.get("recovery_no_result_reads")
        if (type(created) not in (int, float) or not math.isfinite(created)
                or now - created < cls.UNRECOVERABLE_MIN_AGE_SECONDS
                or type(reads) is not int or reads < cls.UNRECOVERABLE_QUALIFIED_READS
                or receipt.get("recovery_reason") not in cls._UNRECOVERABLE_REASONS):
            return False
        receipt.update(status="failed", error_code="RESULT_UNRECOVERABLE", upstream_outcome="unknown",
                       recovery_retryable=True, finished_at=receipt.get("finished_at") or now,
                       _execution_wait_ended_at=now, _turn_reserved=False, updated_at=now,
                       _claim_id=None, _claim_until=None, _executing=False)
        return True

    @classmethod
    def _end_no_final_recovery(cls, receipt, now):
        """Bound legacy Chat recovery without claiming that its upstream ended.

        Work-owned requests retain their existing completion investigation.
        Older internal callers have no such owner: ending execution occupancy
        alone must not leave ordinary reads scheduling upstream GETs forever.
        The existing explicit recovery endpoint may still make one fresh read.
        """
        if (not cls._bounded_chat_wait(receipt)
                or not receipt.get("_execution_wait_ended_at")
                or receipt.get("_attempt_finished_at")
                or any(receipt.get(key) for key in (
                    "_work_key", "_completion", "_completion_of", "_retry_cursor",
                    "_upstream_terminal", "recovery_claim_id", "_claim_id", "_executing"))
                or receipt.get("status") != "failed"
                or receipt.get("error_code") != "RESULT_UNRECOVERABLE"
                or receipt.get("upstream_outcome") != "unknown"
                or receipt.get("recovery_reason") != TextRecoveryReason.REQUEST_RESULT_NOT_FOUND.value
                or not all(isinstance(receipt.get(key), str) and receipt[key].strip() for key in (
                    "request_message_id", "provider_binding_id", "provider_account_identity",
                    "client_conversation_id", "conversation_id"))
                or type(receipt.get("recovery_no_result_reads")) is not int
                or receipt["recovery_no_result_reads"] < cls.UNRECOVERABLE_QUALIFIED_READS):
            return False
        receipt.update(_attempt_finished_at=now, _attempt_reason="ORIGINAL_RESULT_NO_FINAL",
                       recovery_next_at=None, updated_at=now)
        return True

    def _finish_recovery(
        self, owner, request_id, claim_id, recovered=None, error_code=None,
        phase=None, recovery_reason=None, retry_after_seconds=None, *, count_unrecoverable=False,
        read_error=None,
    ):
        now = self._now()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            if not row:
                return {"request_id": request_id, "status": "not_found"}
            current = json.loads(row[0])
            if current.get("status") == "succeeded" or (
                current.get("status") == "failed" and current.get("error_code") == "CHAT_RESPONSE_NOT_TEXT"
            ):
                # A late read cannot roll back an authoritative terminal
                # result, even if another writer did not clear the lease.
                return self._public(current)
            if current.get("recovery_claim_id") != claim_id:
                return self._public(current)
            if error_code:
                attempt = max(1, int(current.get("recovery_attempt", 1)))
                qualified_reads = int(current.get("recovery_no_result_reads") or 0)
                coverage_version = (recovered or {}).get(
                    RECOVERY_CONVERSATION_COVERAGE_VERSION_FIELD
                )
                if coverage_version == 1 and current.get(
                        RECOVERY_CONVERSATION_COVERAGE_VERSION_FIELD) != 1:
                    qualified_reads = 0
                created_at = float(current.get("created_at") or now)
                qualified_read_recorded = (
                    (count_unrecoverable or self._bounded_chat_wait(current))
                    and recovery_reason in self._UNRECOVERABLE_REASONS
                    and now - created_at >= self.UNRECOVERABLE_MIN_AGE_SECONDS
                )
                if qualified_read_recorded:
                    qualified_reads += 1
                if recovery_reason in {
                    TextRecoveryReason.CONVERSATION_NOT_FOUND.value,
                    TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value,
                }:
                    requires_new_conversation = True
                elif recovery_reason in self._VALID_CONVERSATION_REASONS:
                    requires_new_conversation = False
                else:
                    requires_new_conversation = bool(
                        current.get("recovery_requires_new_conversation")
                    )
                recovered_anchor = {
                    key: value for key, value in (recovered or {}).items()
                    if (
                        key in {RECOVERY_CONVERSATION_SCAN_FIELD, TURN_END_EVIDENCE_FIELD, "_upstream_terminal", "_turn_reserved"}
                        or key == "_retry_cursor" and not current.get("_completion", {}).get("replacement_id")
                        or key == "_empty_reply_evidence" and not current.get("_completion", {}).get("replacement_id")
                        or key == RECOVERY_CONVERSATION_COVERAGE_VERSION_FIELD
                        or key in {"conversation_id", "parent_message_id", "request_parent_message_id"}
                        and not current.get(key)
                    )
                }
                scan_incomplete = (
                    recovery_reason
                    == TextRecoveryReason.REQUEST_CONVERSATION_SCAN_INCOMPLETE.value
                )
                changes = {
                    **recovered_anchor,
                    "recovery_next_at": now + (
                        retry_after_seconds
                        if retry_after_seconds is not None
                        else self._recovery_backoff(
                            qualified_reads if qualified_read_recorded else (1 if scan_incomplete else attempt)
                        )
                    ),
                    "recovery_error_code": error_code,
                    "recovery_last_read_error": safe_recovery_read_error(read_error),
                    "recovery_phase": phase or current.get("recovery_phase") or "read_text_request",
                    "recovery_retry_after_seconds": retry_after_seconds,
                    "recovery_reason": recovery_reason,
                    "recovery_no_result_reads": qualified_reads,
                    # This marker describes the latest qualified read. A prior
                    # 404 must not force a new chat after the same chat becomes
                    # readable again.
                    "recovery_requires_new_conversation": requires_new_conversation,
                }
            elif (recovered or {}).get("status") == "queued":
                # Legacy multi-image recovery has saved every submitted slot.
                # Its unsent remainder competes in the original admission queue.
                changes = {**recovered, "error_code": None, "recovery_next_at": None,
                           "recovery_error_code": None, "recovery_phase": None, "finished_at": None}
            elif (recovered or {}).get("status") == "failed":
                changes = {
                    **recovered,
                    "finished_at": now,
                    "upstream_outcome": "completed",
                    "recovery_next_at": None,
                    "recovery_error_code": "CHAT_RESPONSE_NOT_TEXT",
                    "recovery_phase": "read_text_result",
                    "recovery_retry_after_seconds": None,
                    "recovery_reason": TextRecoveryReason.REQUEST_RESULT_NON_TEXT.value,
                    "recovery_retryable": False,
                    "recovery_requires_new_conversation": False,
                }
            else:
                changes = {
                    **(recovered or {}),
                    "status": "succeeded",
                    "upstream_outcome": "completed",
                    "recovery_retryable": False,
                    "finished_at": now,
                    "error_code": None,
                    "waiting": None,
                    "recovery_next_at": None,
                    "recovery_error_code": None,
                    "recovery_phase": None,
                    "recovery_retry_after_seconds": None,
                    "recovery_reason": None,
                    "recovery_requires_new_conversation": False,
                }
            if not error_code:
                changes["recovery_last_read_error"] = None
            observation = self._safe_result_observation((recovered or {}).get("_original_result_observation"), current)
            observed = bool(error_code == "UPSTREAM_OUTCOME_UNKNOWN"
                            and recovery_reason == TextRecoveryReason.REQUEST_RESULT_INCOMPLETE.value
                            and observation is not None)
            if observed:
                previous = self._safe_result_observation(current.get("_original_result_observation"), current)
                started = current.get("_result_observation_started_at")
                progress = current.get("_result_last_progress_at")
                initial = previous is None or type(started) not in (int, float) or not math.isfinite(started)
                changed = previous is not None and previous["nodes"] != observation["nodes"]
                count = current.get("_result_no_progress_reads")
                changes.update(_original_result_observation=observation, _result_last_checked_at=now,
                               _result_observation_started_at=now if initial else started,
                               _result_last_progress_at=now if changed else None if initial else progress,
                               _result_no_progress_reads=0 if initial or changed else (count if type(count) is int and count >= 0 else 0) + 1)
            if error_code == "UPSTREAM_OUTCOME_UNKNOWN" and recovery_reason == TextRecoveryReason.REQUEST_CONVERSATION_ADVANCED.value:
                # A completed external successor ends our foreground wait;
                # it does not authorize old-cursor continuation, archive,
                # capacity release, or adoption of that successor's result.
                changes.update(_result_last_checked_at=now,
                               _result_wait_ended_at=current.get("_result_wait_ended_at") or now)
            if error_code == "UPSTREAM_OUTCOME_UNKNOWN" and recovery_reason == TextRecoveryReason.REQUEST_RESULT_NOT_FOUND.value:
                # The original user turn was read successfully but has no
                # result yet. Keep this distinct from a failed/unavailable
                # read so the bounded same-session completion policy can
                # use its fresh cursor after the investigation window.
                changes["_result_last_checked_at"] = now
            updated = {**current, **changes, "recovery_claim_id": None,
                       "recovery_claimed_at": None, "recovery_lease_until": None,
                       "updated_at": now}
            # A process can die after sending and persisting its cursor. Once
            # the original result is recovered, fence only its expired sending
            # lease; otherwise the stale execution flag prevents completion.
            # Keep a live or unverifiable lease and all original send evidence.
            old_claim_until = current.get("_claim_until")
            if (updated.get("status") == "succeeded"
                    and current.get("_submission_started") is True
                    and current.get("_executing") is True
                    and isinstance(current.get("_claim_id"), str)
                    and current["_claim_id"]
                    and type(old_claim_until) in (int, float)
                    and math.isfinite(old_claim_until)
                    and old_claim_until <= now):
                updated.update(_claim_id=None, _claim_until=None, _executing=False)
            self._end_execution_wait(updated, now, observed=observed)
            self._end_no_final_recovery(updated, now)
            if (error_code and updated.get("_attempt_reason") == "ORIGINAL_RESULT_NO_FINAL"
                    and recovery_reason == TextRecoveryReason.REQUEST_RESULT_NOT_FOUND.value):
                updated["recovery_next_at"] = None
            # Empty-response recovery is part of the original logical task.
            # Reserve at most one child, and never auto-retry that child again.
            if (getattr(self.admission, "generation_completion", None) is not None and not updated.get("_completion_of")
                    and not updated.get("_completion") and self._verified_retryable_empty(updated)
                    and not updated.get("_recovery_suppressed")):
                updated["_completion"] = {"state": "checking_original", "started_at": now,
                    "allow_unconfirmed_retry": False, "max_extra_requests": 1, "next_at": now,
                    "automatic_empty_retry": True}
            if updated.get(RECOVERY_CONVERSATION_SCAN_FIELD) == {} or not error_code:
                updated.pop(RECOVERY_CONVERSATION_SCAN_FIELD, None)
            db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=? AND receipt=?",
                       (json.dumps(updated), owner, request_id, row[0]))
            return self._public(updated)

    def _update_recovery_claim(self, owner, receipt, **changes):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            request_id = receipt["request_id"]
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            current = json.loads(row[0]) if row else {}
            if not current.get("recovery_claim_id") or current["recovery_claim_id"] != receipt.get("recovery_claim_id"):
                raise AdmissionLost("original image recovery claim changed")
            current.update(changes, updated_at=self._now())
            db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(current), owner, request_id))
            return current

    def set_public_session_archived(self, owner: str, request_id: str, archived: bool) -> dict[str, object]:
        """Change one completed owner session by its original final request ID."""
        with self._db() as db:
            receipt = self.store.read_receipt(db, "text", owner, request_id)
            if not receipt or receipt.get("route") != "chat" or not receipt.get("_public_session_ref"):
                raise ConversationBindingError("product session not found", code="CHAT_REQUEST_NOT_FOUND")
            if receipt.get("status") != "succeeded" or db.execute(
                "SELECT 1 FROM requests WHERE owner=? AND json_extract(receipt,'$._previous_request_id')=? LIMIT 1",
                (owner, request_id),
            ).fetchone():
                raise ConversationBindingError("product session is not complete", code="CHAT_SESSION_NOT_TERMINAL")
            required = ("provider_binding_id", "provider_account_identity", "client_conversation_id",
                        "conversation_id", "parent_message_id")
            if any(not isinstance(receipt.get(key), str) or not receipt[key] for key in required):
                raise ConversationBindingError("original conversation cursor unavailable", code="CHAT_SESSION_UNCONFIRMED")
        conversation_binding_service.set_archived(receipt, archived)
        return {"request_id": request_id, "archived": archived,
                "conversation": {"client_conversation_id": receipt["_public_session_ref"], "protocol": "sequential-v1"}}

    def archive_public_session(self, owner: str, request_id: str) -> dict[str, object]:
        return self.set_public_session_archived(owner, request_id, True)

    def restore_public_session(self, owner: str, request_id: str) -> dict[str, object]:
        return self.set_public_session_archived(owner, request_id, False)

    def read(self, owner: str, request_id: str, *, allow_unrecoverable_retry: bool = False,
             _explicit_ended_recheck: bool = False):
        from services.pool_admission import unknown_text_result
        recovery_claim = None
        now = self._now()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            if row:
                previous = json.loads(row[0])
                ended_wait = self._end_execution_wait(previous, now)
                if self._end_no_final_recovery(previous, now) or ended_wait:
                    db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?",
                               (json.dumps(previous), owner, request_id))
                    row = (json.dumps(previous),)
                if previous.get("_recovery_suppressed") is True:
                    # An explicit operator stop preserves the original
                    # receipt and makes reads observational until the marker is
                    # removed. Never claim or reissue this UNKNOWN here.
                    row = (json.dumps(previous),)
                elif (previous["status"] == "failed"
                        and previous.get("error_code") == "CONVERSATION_BINDING_UNAVAILABLE"
                        and not previous.get("conversation_id") and not previous.get("parent_message_id")
                        and ((previous.get("provider_binding_id") and previous.get("provider_account_identity"))
                             or (not previous.get("provider_binding_id") and not previous.get("provider_account_identity")))):
                    # This legacy failure occurred after selecting an account
                    # but before obtaining its text token / sending a turn.
                    # A pre-binding failure has neither identity field; it is
                    # equally known-not-submitted and can reuse this receipt.
                    # Keep the same request hash and message id for recovery.
                    previous = {**previous, "status": "not_started", "updated_at": now}
                    db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(previous), owner, request_id))
                    row = (json.dumps(previous),)
                if (previous.get("_recovery_suppressed") is not True
                        and previous["status"] == "queued" and previous["boot"] != self.boot and not previous.get("_input_ref")):
                    # The atomic running claim never happened. Preserve identity
                    # and wait for the caller to supply the exact original body.
                    previous = {**previous, "status": "not_started", "updated_at": now}
                    db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(previous), owner, request_id))
                    row = (json.dumps(previous),)
                if (previous.get("_recovery_suppressed") is not True
                        and previous["status"] == "running" and previous["boot"] != self.boot and not previous.get("_claim_id")):
                    # Another process/restart cannot establish that the original
                    # write failed. Preserve its cursor and make the receipt
                    # eligible for the read-only recovery path.
                    previous = {**previous, "status": "unknown", "error_code": "CONVERSATION_OUTCOME_UNKNOWN", "updated_at": now}
                    db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(previous), owner, request_id))
                    row = (json.dumps(previous),)
                completion = previous.get("_completion") or {}
                # A caller can recheck an original after repairing a failed
                # read. Keep the ended attempt and automatic query stop; this
                # grants one ordinary, paced read, never a generation retry.
                completion_recheck = bool(
                    _explicit_ended_recheck and previous.get("_attempt_finished_at")
                    and previous.get("_attempt_reason") == "COMPLETION_ORIGINAL_READ_UNAVAILABLE"
                    and isinstance(completion, dict) and completion.get("state") == "needs_attention"
                    and completion.get("reason") == "COMPLETION_ORIGINAL_READ_UNAVAILABLE"
                    and not completion.get("replacement_id") and not completion.get("selected_id")
                    and not previous.get("_completion_of")
                )
                no_final_recheck = bool(
                    _explicit_ended_recheck and previous.get("_attempt_finished_at")
                    and previous.get("_attempt_reason") == "ORIGINAL_RESULT_NO_FINAL"
                    and not previous.get("_completion") and not previous.get("_completion_of")
                )
                ended_recheck = completion_recheck or no_final_recheck
                if (previous.get("_recovery_suppressed") is not True
                        and previous.get("_recovery_paused") is not True
                        and (not previous.get("_attempt_finished_at") or ended_recheck)
                        and unknown_text_result(previous)
                        and previous.get("request_message_id")
                        and previous.get("provider_binding_id")
                        and previous.get("provider_account_identity")
                        and self._recovery_due(previous, now)):
                    attempt = int(previous.get("recovery_attempt", 0)) + 1
                    claim_id = uuid.uuid4().hex
                    recovery_claim = claim_id, {
                        **previous,
                        "recovery_attempt": attempt,
                        "recovery_next_at": now + self.RECOVERY_LEASE_SECONDS,
                        "recovery_claim_id": claim_id,
                        "recovery_claimed_at": now,
                        "recovery_lease_until": now + self.RECOVERY_LEASE_SECONDS,
                        "recovery_error_code": None,
                        "recovery_phase": "read_text_request",
                        "recovery_reason": None,
                        "updated_at": now,
                    }
                    db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?",
                               (json.dumps(recovery_claim[1]), owner, request_id))
        if not row:
            return {"request_id": request_id, "status": "not_found"}
        if recovery_claim:
            # Account pacing and a slow original response can outlive the
            # initial lease. Keep that same claim alive so an ordinary client
            # read cannot start a duplicate GET while the background read runs.
            recovery_done = threading.Event()
            def keep_recovery_claim():
                while not recovery_done.wait(max(0.01, self.RECOVERY_LEASE_SECONDS / 3)):
                    try:
                        self._update_recovery_claim(owner, recovery_claim[1],
                                                    recovery_lease_until=self._now() + self.RECOVERY_LEASE_SECONDS)
                    except Exception:
                        return
            lease_thread = threading.Thread(target=keep_recovery_claim, name="original-read-lease", daemon=True)
            lease_started = False
            try:
                lease_thread.start()
                lease_started = True
                from services import durable_image_forward
                if durable_image_forward.supported(recovery_claim[1]):
                    recovered = durable_image_forward.recover(self, owner, recovery_claim[1])
                    result = self._finish_recovery(owner, request_id, recovery_claim[0], recovered)
                    if self.admission is not None:
                        self.admission.wake()
                    return result
                recovered = self.recovery_reader(recovery_claim[1])
                safe_result, error_code, phase, recovery_reason = self._safe_recovery_result(recovered, recovery_claim[1])
                if not error_code and recovery_claim[1].get("_forward_protocol"):
                    from services.durable_forward import recovered_chat_wire
                    if recovered.get("status") == "succeeded":
                        safe_result.update(recovered_chat_wire(self, recovery_claim[1], {**safe_result, "status": "succeeded"}))
                    else:
                        # A non-text upstream result is not a completed text
                        # compatibility response. Preserve its authoritative
                        # result without replaying the original model call.
                        safe_result = {**(safe_result or {}), "_upstream_terminal": True, "_turn_reserved": False,
                                       "_wire_head": None, "_wire_error_status": 422,
                                       "_wire_error_payload": {"error": {"code": "CHAT_RESPONSE_NOT_TEXT"}}}
                result = self._finish_recovery(
                    owner, request_id, recovery_claim[0], safe_result, error_code, phase,
                    recovery_reason, count_unrecoverable=allow_unrecoverable_retry,
                )
            except ConversationBindingError as exc:
                recovery_scan = self._safe_recovery_scan(getattr(exc, "recovery_scan", None))
                recovery_evidence = {}
                if recovery_scan is not None:
                    recovery_evidence[RECOVERY_CONVERSATION_SCAN_FIELD] = recovery_scan
                if getattr(exc, "recovery_coverage_version", None) == 1:
                    recovery_evidence[RECOVERY_CONVERSATION_COVERAGE_VERSION_FIELD] = 1
                recovery_error_code, retry_after_seconds = self._recovery_failure(exc)
                result = self._finish_recovery(
                    owner, request_id, recovery_claim[0],
                    recovery_evidence or None,
                    error_code=recovery_error_code,
                    phase=(safe_recovery_read_error(getattr(exc, "recovery_read_error", None)) or {}).get("phase"),
                    read_error=getattr(exc, "recovery_read_error", None),
                    recovery_reason=self._safe_recovery_reason(getattr(exc, "recovery_reason", "")),
                    retry_after_seconds=retry_after_seconds,
                    count_unrecoverable=allow_unrecoverable_retry,
                )
            except Exception as exc:
                # Never persist exception text: it may contain a token, prompt,
                # or provider response. The next bounded recovery window is
                # enough to make the request retryable without hammering GET.
                recovery_error_code, retry_after_seconds = self._recovery_failure(exc)
                result = self._finish_recovery(
                    owner, request_id, recovery_claim[0],
                    error_code=recovery_error_code,
                    retry_after_seconds=retry_after_seconds,
                    recovery_reason=None, count_unrecoverable=allow_unrecoverable_retry,
                )
            finally:
                recovery_done.set()
                if lease_started:
                    lease_thread.join(timeout=1)
            return self._authorize_unrecoverable(owner, request_id, result) if allow_unrecoverable_retry else result
        result = self._public(json.loads(row[0]))
        # This advertisement is a local proof only, never an implicit retry or
        # upstream read. The explicit endpoint revalidates it transactionally.
        if self.admission is not None and self._known_unsent_category_successor(json.loads(row[0]), now):
            try:
                with self._db() as db:
                    current = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
                    if current:
                        current_receipt = json.loads(current[1])
                        self._validated_unsent_successor_input(db, owner, current[0], current_receipt, self._now())
                        result = self._public(current_receipt)
                        result.update(bound_successor_resume_retryable=True, upstream_outcome="not_submitted",
                                      upstream_submission_started=False)
            except (ConversationBindingError, OSError, ValueError, TypeError, KeyError):
                pass
        return self._authorize_unrecoverable(owner, request_id, result) if allow_unrecoverable_retry else result

    def recover(self, owner: str, request_id: str, allow_unrecoverable_retry: bool = False,
                *, explicit_ended_recheck: bool = False):
        return self.read(
            owner, request_id,
            allow_unrecoverable_retry=bool(allow_unrecoverable_retry),
            _explicit_ended_recheck=explicit_ended_recheck,
        )

    def _authorize_unrecoverable(self, owner, request_id, observed):
        if observed.get("recovery_control", {}).get("state") in {"paused", "pausing"}:
            return observed
        if observed.get("error_code") == "RESULT_UNRECOVERABLE":
            return observed
        if observed.get("status") != "unknown":
            return observed
        now = self._now()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id),
            ).fetchone()
            if not row:
                return {"request_id": request_id, "status": "not_found"}
            current = json.loads(row[0])
            if (current.get("status") == "succeeded" or current.get("error_code") == "RESULT_UNRECOVERABLE"
                    or current.get("recovery_claim_id") or current.get("_executing") is True
                    or current.get("_recovery_paused") is True):
                return self._public(current)
            created_at = float(current.get("created_at") or now)
            if (
                current.get("status") != "unknown"
                or now - created_at < self.UNRECOVERABLE_MIN_AGE_SECONDS
                or int(current.get("recovery_no_result_reads") or 0) < self.UNRECOVERABLE_QUALIFIED_READS
            ):
                return self._public(current)
            updated = {
                **current,
                "status": "failed",
                "error_code": "RESULT_UNRECOVERABLE",
                "upstream_outcome": "unknown",
                "recovery_retryable": True,
                "recovery_requires_new_conversation": bool(
                    current.get("recovery_requires_new_conversation")
                ),
                "recovery_next_at": current.get("recovery_next_at") or now + self.RECOVERY_BASE_BACKOFF_SECONDS,
                "recovery_claim_id": None,
                "recovery_claimed_at": None,
                "recovery_lease_until": None,
                "updated_at": now,
                "finished_at": now,
            }
            self._end_execution_wait(updated, now)
            db.execute(
                "UPDATE requests SET receipt=? WHERE owner=? AND id=? AND receipt=?",
                (json.dumps(updated), owner, request_id, row[0]),
            )
            return self._public(updated)

    def _update(self, owner, request_id, *, classify_before_send=False, **changes):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            receipt = json.loads(row[0])
            context = current_request.get()
            if context is not None and (context.kind, context.owner, context.request_id) == ("text", owner, request_id) and receipt.get("_claim_id") != context.claim:
                raise AdmissionLost("original text claim changed")
            if receipt.get("status") == "succeeded" or (
                receipt.get("status") == "failed" and receipt.get("error_code") == "CHAT_RESPONSE_NOT_TEXT"
            ):
                # A late runner callback must not reopen a recovered terminal
                # result or replace its original result/cursor evidence.
                return
            if receipt.get("_execution_wait_ended_at") is not None:
                # A late runner cannot reopen local execution or overwrite the
                # original evidence. Only the exact-ID recovery path may adopt
                # its subsequently verified upstream result.
                return
            if classify_before_send and receipt.get("_submission_started") is False:
                changes.update(upstream_outcome="not_sent", _turn_reserved=False)
            if changes.get("status") == "succeeded":
                # Success ends the active failure; original_failure_* and the
                # execution timeline remain available as historical evidence.
                changes.update(error_code=None, waiting=None, upstream_outcome="completed")
            receipt = {**receipt, **changes, "updated_at": self._now()}
            db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(receipt), owner, request_id))

    @staticmethod
    def _submission_identity(owner: str, body: dict) -> tuple[str, str]:
        request_id = str(body.get("client_request_id") or "").strip()
        if not owner or not request_id or len(request_id) > 200:
            raise ConversationBindingError(
                "request identity is required",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
            )
        def immutable(value):
            # Public multimodal requests are validated into bytes before they
            # reach the durable service. Hash the bytes without storing another
            # copy of the image in the receipt or relying on JSON serialization.
            if isinstance(value, (bytes, bytearray)):
                data = bytes(value)
                return {
                    "$bytes_sha256": hashlib.sha256(data).hexdigest(),
                    "$bytes_length": len(data),
                }
            if isinstance(value, dict):
                return {str(key): immutable(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [immutable(item) for item in value]
            return value

        request_hash = hashlib.sha256(
            json.dumps(immutable(body), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return request_id, request_hash

    SUPERSEDE_FIELDS = frozenset({"supersedes_request_id", "client_request_id", "provider_binding_id",
        "provider_account_identity", "client_conversation_id", "conversation_id", "parent_message_id"})

    def submission_input(self, owner, body):
        """Resolve an explicit internal reference; callers cannot replace original content."""
        if "supersedes_request_id" not in body:
            if "derived_input" in body:
                raise ConversationBindingError("derived input requires original request", code="CHAT_DERIVED_INPUT_INVALID")
            return body
        request_id = body.get("client_request_id")
        with self._db() as db:
            existing = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            conflict = "CONVERSATION_REQUEST_CONFLICT" if existing else "CHAT_SUPERSEDE_INVALID"
            def reject():
                raise ConversationBindingError("original successor input cannot be changed", code=conflict)
            derived = "derived_input" in body
            if (set(body) != self.SUPERSEDE_FIELDS | ({"derived_input"} if derived else set()) or not owner
                    or any(not isinstance(body.get(k), str) or not body[k].strip() or body[k] != body[k].strip() for k in self.SUPERSEDE_FIELDS)
                    or len(request_id) > 200 or len(body["supersedes_request_id"]) > 200
                    or request_id == body["supersedes_request_id"]):
                reject()
            row = existing or db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                                         (owner, body["supersedes_request_id"])).fetchone()
            if not row:
                reject()
            receipt = json.loads(row[1])
            try:
                retained = self.store.load_input(receipt["_input_ref"])
                if self._submission_identity(owner, retained) != (receipt["request_id"], row[0]):
                    reject()
            except (OSError, ValueError, TypeError, KeyError):
                reject()
            expected = {**retained, "client_request_id": request_id, "supersedes_request_id": body["supersedes_request_id"]}
            if derived and not existing:
                from services.category_directory_derivation import derive_category_parent_input
                expected, _ = derive_category_parent_input(retained, request_id, body["supersedes_request_id"], body["derived_input"])
            if existing and retained.get("supersedes_request_id") != body["supersedes_request_id"]:
                reject()
            if existing and ("derived_input" in retained) != derived:
                reject()
            if any(expected.get(k) != v for k, v in body.items()):
                reject()
            return expected

    @staticmethod
    def _supersede_order_link(previous, successor):
        """A server-registered successor releases only its own predecessor's order head."""
        return bool(previous and successor.get("_supersedes_request_id") == previous.get("request_id")
            and successor.get("request_id") != previous.get("request_id")
            and successor.get("_supersedes_request_message_id") == previous.get("request_message_id")
            and successor.get("_supersedes_input_hash") and successor.get("_input_ref")
            and not previous.get("_public_session_ref") and not successor.get("_public_session_ref")
            and successor.get("_route", "chat") == previous.get("_route", "chat") == "chat"
            and successor.get("_operation", "text") == previous.get("_operation", "text") == "text"
            and not successor.get("_forward_protocol") and not previous.get("_forward_protocol")
            and all(successor.get(k) == previous.get(k) and previous.get(k) for k in (
                "provider_binding_id", "provider_account_identity", "conversation_id", "client_conversation_id", "model"))
            and successor.get("_submission_parent_message_id") == (
                previous.get("_submission_parent_message_id") or previous.get("parent_message_id")))

    @staticmethod
    def _validate_supersede(store, db, owner, successor, now):
        """Recheck the immutable original and live claims in the existing transaction."""
        def reject(code):
            raise ConversationBindingError("explicit original-request successor cannot advance", code=code)
        previous_id = successor.get("_supersedes_request_id")
        row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                         (owner, previous_id)).fetchone()
        if not row:
            reject("CHAT_SUPERSEDE_INVALID")
        previous = json.loads(row[1])
        if previous.get("status") == "succeeded" or previous.get("_upstream_terminal"):
            reject("CHAT_SUPERSEDE_ORIGINAL_FOUND")
        if (not isinstance(previous.get("request_message_id"), str) or not previous["request_message_id"]
                or not TextTaskService._supersede_order_link(previous, successor)):
            reject("CHAT_SUPERSEDE_INVALID")
        for claim_key, until_key in (("_claim_id", "_claim_until"), ("recovery_claim_id", "recovery_lease_until")):
            until = previous.get(until_key)
            if previous.get(claim_key) and (type(until) not in (int, float) or not math.isfinite(until) or until > now):
                reject("CHAT_SUPERSEDE_PREDECESSOR_BUSY")
        if previous.get("_executing") or previous.get("_turn_reserved"):
            reject("CHAT_SUPERSEDE_PREDECESSOR_BUSY")
        if (previous.get("status") != "failed" or previous.get("error_code") != "RESULT_UNRECOVERABLE"
                or previous.get("upstream_outcome") != "unknown"
                or previous.get("recovery_reason") != "REQUEST_MESSAGE_NOT_FOUND"
                or previous.get("_recovery_suppressed") or previous.get("_supersedes_request_id")
                or type(previous.get("_execution_wait_ended_at")) not in (int, float)
                or not math.isfinite(previous["_execution_wait_ended_at"])):
            reject("CHAT_SUPERSEDE_INVALID")
        if row[0] != successor.get("_supersedes_input_hash"):
            reject("CHAT_SUPERSEDE_INVALID")
        try:
            original_body = store.load_input(previous["_input_ref"])
            identity = TextTaskService._submission_identity(owner, original_body)
        except (OSError, ValueError, TypeError, KeyError):
            reject("CHAT_SUPERSEDE_INVALID")
        if identity != (previous_id, row[0]):
            reject("CHAT_SUPERSEDE_INVALID")
        if successor.get("_derived_input"):
            if (previous.get("original_http_status") != 413
                    or previous.get("original_failure_phase") != "stream_open"
                    or previous.get("original_upstream_request_stage") != "conversation"
                    or previous.get("original_exception_category") != "http"
                    or previous.get("_submission_started") is not True):
                reject("CHAT_DERIVED_INPUT_INVALID")
            from services.category_directory_derivation import derive_category_parent_input
            derived_body, audit = derive_category_parent_input(original_body, successor["request_id"], previous_id,
                                                               {"kind": successor["_derived_input"].get("kind")})
            if successor["_derived_input"] != {**audit, "original_input_hash": row[0]}:
                reject("CHAT_DERIVED_INPUT_INVALID")
            saved = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", (owner, successor["request_id"])).fetchone()
            if saved and TextTaskService._submission_identity(owner, derived_body)[1] != saved[0]:
                reject("CHAT_DERIVED_INPUT_INVALID")
        # The submission root is stable even when a result advances the live cursor.
        if any(original_body.get(k) != successor.get(k) for k in (
                "provider_binding_id", "provider_account_identity", "conversation_id", "client_conversation_id")):
            reject("CHAT_SUPERSEDE_INVALID")
        if original_body.get("parent_message_id") != successor.get("_submission_parent_message_id"):
            reject("CHAT_SUPERSEDE_INVALID")
        other = db.execute("SELECT id FROM requests WHERE owner=? AND json_extract(receipt,'$._supersedes_request_id')=? AND id<>? LIMIT 1",
                           (owner, previous_id, successor["request_id"])).fetchone()
        if other:
            reject("CHAT_SUPERSEDE_CONFLICT")
        return previous

    def _prepare_supersede(self, db, owner, body, receipt):
        if "supersedes_request_id" not in body:
            return
        previous_id = body["supersedes_request_id"]
        def reject(code):
            raise ConversationBindingError("explicit original-request successor cannot advance", code=code)
        if (not isinstance(previous_id, str) or not previous_id.strip() or len(previous_id) > 200
                or previous_id != previous_id.strip() or previous_id == body.get("client_request_id")
                or body.get("_public_session_ref") or body.get("_public_route") or body.get("_forward")
                or body.get("_editable") or body.get("_route", "chat") != "chat"
                or body.get("_operation", "text") != "text"):
            reject("CHAT_SUPERSEDE_INVALID")
        row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", (owner, previous_id)).fetchone()
        if not row:
            reject("CHAT_SUPERSEDE_INVALID")
        original = json.loads(row[1])
        if "derived_input" in body:
            from services.category_directory_derivation import derive_category_parent_input
            try:
                retained = self.store.load_input(original["_input_ref"])
                expected, audit = derive_category_parent_input(retained, body["client_request_id"], previous_id, body["derived_input"])
            except (OSError, ValueError, TypeError, KeyError):
                reject("CHAT_DERIVED_INPUT_INVALID")
            if self._submission_identity(owner, expected) != self._submission_identity(owner, body):
                reject("CHAT_DERIVED_INPUT_INVALID")
            receipt["_derived_input"] = {**audit, "original_input_hash": row[0]}
        else:
            comparison = {k: v for k, v in body.items() if k != "supersedes_request_id"}
            comparison["client_request_id"] = previous_id
            if self._submission_identity(owner, comparison) != (previous_id, row[0]):
                reject("CHAT_SUPERSEDE_INVALID")
        receipt.update({k: body.get(k) for k in ("provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id")})
        receipt.update(_supersedes_request_id=previous_id, _supersedes_input_hash=row[0],
                       _supersedes_request_message_id=original.get("request_message_id"),
                       _submission_parent_message_id=body.get("parent_message_id"))
        # This placeholder is used only for validation, never saved or exposed.
        candidate = {**receipt, "_input_ref": True}
        self._validate_supersede(self.store, db, owner, candidate, self._now())
        from services.pool_admission import unfinished
        # Do not move an intervening queued/unknown turn ahead of this explicit successor.
        for kind, other_owner, other_id, other in self.store.receipts(db):
            if other_owner != owner or (kind == "text" and other_id == previous_id):
                continue
            same_chat = (other.get("provider_account_identity") == receipt.get("provider_account_identity")
                         and other.get("conversation_id") == receipt.get("conversation_id"))
            if unfinished(kind, other) and (same_chat or other.get("client_conversation_id") == body.get("client_conversation_id")):
                reject("CHAT_SUPERSEDE_CONFLICT")

    def validate_submission(self, owner: str, body: dict):
        """Read-only conflict check before any optional external review call."""
        body = self.submission_input(owner, body)
        request_id, request_hash = self._submission_identity(owner, body)
        with self._db() as db:
            previous = db.execute(
                "SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                (owner, request_id),
            ).fetchone()
            if not previous and "supersedes_request_id" in body:
                from services.durable_forward import dispatch_model
                self._prepare_supersede(db, owner, body, {"request_id": request_id,
                    "client_conversation_id": body["client_conversation_id"], "model": dispatch_model(body)})
        if previous and previous[0] != request_hash:
            raise ConversationBindingError(
                "request identity already has different input",
                code="CONVERSATION_REQUEST_CONFLICT",
            )
        return self._public(json.loads(previous[1])) if previous else None

    def _continue_public_session(self, db, owner, body, receipt):
        """Validate the predecessor and bind this turn atomically, without a second queue.

        Same-ID retries are handled before here. One predecessor has one successor,
        across processes and restarts. A completed legacy receipt may be explicitly
        continued by its owner; no old receipt or input hash is rewritten.
        """
        if not body.get("_public_session_ref"):
            return
        group = body["client_conversation_id"]
        previous_id = body.get("_previous_request_id")
        members = {request_id: json.loads(raw) for request_id, raw in db.execute(
            "SELECT id,receipt FROM requests WHERE owner=? AND json_extract(receipt,'$.client_conversation_id')=?",
            (owner, group),
        )}
        def reject(code):
            raise ConversationBindingError("sequential Chat request cannot advance", code=code)
        if not previous_id:
            if members:
                reject("CHAT_PREVIOUS_REQUEST_REQUIRED")
        else:
            previous = self.store.read_receipt(db, "text", owner, previous_id)
            if previous is None or previous.get("route") != "chat":
                reject("CHAT_PREVIOUS_REQUEST_NOT_FOUND")
            # A new session cannot steal an existing sequential session. A legacy
            # successful request has no session reference and is an explicit anchor.
            if ((previous.get("_public_session_ref") and previous.get("client_conversation_id") != group)
                    or (members and previous_id not in members)):
                reject("CHAT_CONVERSATION_CONFLICT")
            successors = [json.loads(raw) for raw, in db.execute(
                "SELECT receipt FROM requests WHERE owner=? AND json_extract(receipt,'$._previous_request_id')=?",
                (owner, previous_id))]
            if any(not self._cancelled_completion_child(candidate, previous) for candidate in successors):
                reject("CHAT_CONVERSATION_CONFLICT")
            terminal_empty = bool(body.get("_continue_after_terminal_empty"))
            evidence = self._verified_retryable_empty(previous) if terminal_empty else None
            retry_attempt = bool(body.get("_continue_after_failed_attempt"))
            if retry_attempt:
                from services.generation_completion import retry_evidence
                state = previous.get("_completion") or {}
                evidence = retry_evidence(previous)
                if (not evidence or state.get("replacement_id") != receipt["request_id"]
                        or body.get("_completion_of") != previous_id or state.get("selected_id")
                        or previous.get("model") != receipt.get("model")):
                    reject("COMPLETION_ORIGINAL_CURSOR_UNCONFIRMED")
            if terminal_empty and (not evidence or previous.get("model") != receipt.get("model")):
                reject("CHAT_TERMINAL_EMPTY_UNVERIFIED")
            if previous.get("status") != "succeeded" and not evidence:
                if (self.admission is not None and previous.get("status") in {"queued", "running"}
                        and previous.get("_public_session_ref") and not terminal_empty
                        and previous.get("upstream_outcome") != "unknown"):
                    receipt.update(_public_session_ref=body["_public_session_ref"],
                                   _previous_request_id=previous_id, _public_previous_waiting=True,
                                   _public_previous_waiting_reason="CHAT_PREVIOUS_REQUEST_PENDING")
                    return
                reject("CHAT_PREVIOUS_REQUEST_PENDING")
            anchors = ("provider_binding_id", "provider_account_identity", "conversation_id")
            if not all(isinstance(previous.get(key), str) and previous[key] for key in anchors):
                reject("CHAT_CONTINUATION_UNAVAILABLE")
            if (receipt.get("_requested_account_identity")
                    and receipt["_requested_account_identity"] != previous["provider_account_identity"]):
                reject("CHAT_ACCOUNT_SELECTION_CONFLICT")
            receipt.update({key: previous[key] for key in anchors})
            parent = evidence["retry_parent_message_id"] if evidence else previous.get("parent_message_id")
            if not isinstance(parent, str) or not parent:
                reject("CHAT_CONTINUATION_UNAVAILABLE")
            receipt["parent_message_id"] = parent
            # The actual last-user parent is filled by the final upstream payload;
            # it differs from the submission parent for multi-message input.
            receipt["_previous_request_id"] = previous_id
            if not previous.get("_public_session_ref"):
                receipt["_legacy_session_anchor"] = previous_id
            receipt["_submission_parent_message_id"] = parent
            if evidence and not retry_attempt:
                receipt["_terminal_empty_correction_of"] = previous_id
        receipt["_public_session_ref"] = body["_public_session_ref"]

    def submit(self, owner: str, body: dict, *, source: str | None = None):
        from services.durable_forward import dispatch_model
        body = self.submission_input(owner, body)
        from services.workflow_scheduling import normalize_scheduling
        if "_scheduling" in body:
            body = {**body, "_scheduling": normalize_scheduling(body["_scheduling"])}
        if body.get("_scheduling") is not None and self.admission is None:
            raise ValueError("SCHEDULING_UNAVAILABLE")
        request_id, request_hash = self._submission_identity(owner, body)
        receipt = {"request_id": request_id, "client_conversation_id": body["client_conversation_id"],
                   "_route": body.get("_route", "chat"), "_operation": body.get("_operation", "text"),
                   "_forward_protocol": "editable_file" if body.get("_editable") else (body.get("_forward") or {}).get("protocol"),
                   "_editable_task_id": (body.get("_editable") or {}).get("task_id"),
                   "_editable_kind": (body.get("_editable") or {}).get("kind"),
                   "_expected_sends": int(body.get("_expected_sends") or 1),
                   "_previous_response_id": ((body.get("_forward") or {}).get("payload") or {}).get("previous_response_id"),
                   "route": str(body.get("_public_route") or ""),
                   "model": dispatch_model(body),
                   "request_message_id": str(uuid.uuid4()),
                   "request_parent_message_id": str(body.get("parent_message_id") or "").strip(),
                   "status": "queued", "boot": self.boot, "created_at": self._now(), "updated_at": self._now(),
                   "_execution_timeline": [{"stage": "accepted", "at": self._now()}]}
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            if previous and previous[0] != request_hash:
                raise ConversationBindingError("request identity already has different input", code="CONVERSATION_REQUEST_CONFLICT")
            schedule = not previous
            previous_receipt = json.loads(previous[1]) if previous else None
            public_chat_submission = str(body.get("_public_route") or "") == "chat"
            safe_capacity_retry = bool(
                previous_receipt
                and public_chat_submission
                and previous_receipt.get("route") == "chat"
                and previous_receipt.get("status") == "failed"
                and previous_receipt.get("error_code") == "TEXT_TASK_CAPACITY_EXCEEDED"
                and previous_receipt.get("upstream_outcome") == "not_sent"
            )
            if previous and (
                (previous_receipt["status"] == "not_started" and not public_chat_submission)
                or safe_capacity_retry
            ):
                receipt = {
                    **previous_receipt,
                    "status": "queued",
                    "model": dispatch_model(body),
                    "boot": self.boot,
                    "updated_at": self._now(),
                }
                if safe_capacity_retry:
                    for field in ("error_code", "upstream_outcome", "finished_at"):
                        receipt.pop(field, None)
                if not receipt.get("_input_ref"):
                    # Original ID/hash already matched. Retain the actual input
                    # of a legacy explicitly-unsent retry before accepting it.
                    receipt.update(_input_ref=self.store.save_input(body),
                                   _sequence=self.store.next_sequence(db),
                                   _source=source or "key:" + owner,
                                   _input_bytes=_retained_size(body),
                                   _turn_reserved=False, _submission_started=False)
                receipt.setdefault("_execution_timeline", [{"stage": "accepted", "at": self._now()}])
                db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(receipt), owner, request_id))
                schedule = True
            elif not previous:
                if body.get("_requested_account_ref") is not None:
                    from services.account_service import account_service
                    accounts = self.admission.accounts if self.admission is not None else account_service
                    receipt["_requested_account_ref"] = body["_requested_account_ref"]
                    receipt["_requested_account_identity"] = (accounts.resolve_image_account(body["_requested_account_ref"])
                        if body.get("_operation") == "image" else accounts.resolve_public_chat_account(body["_requested_account_ref"]))
                    if (body.get("provider_account_identity")
                            and body["provider_account_identity"] != receipt["_requested_account_identity"]):
                        raise ConversationBindingError("account selection conflicts with binding",
                                                       code="CHAT_ACCOUNT_SELECTION_CONFLICT")
                self._prepare_supersede(db, owner, body, receipt)
                self._continue_public_session(db, owner, body, receipt)
                from services.generation_completion import attach_replacement
                attach_replacement(self.store, db, "text", owner, request_id, body, receipt)
                receipt.update({"_input_ref": self.store.save_input(body),
                                "_sequence": self.store.next_sequence(db),
                                "_source": source or "key:" + owner,
                                "_input_bytes": _retained_size(body),
                                "_turn_reserved": False,
                                "_submission_started": False})
                if (receipt.get("_public_session_ref") and not receipt.get("_completion_of")
                        and receipt.get("_route") == "chat" and receipt.get("_operation") == "text"
                        and not receipt.get("_forward_protocol")):
                    receipt["_automatic_generation_recovery"] = True
                for field in ("provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id"):
                    if body.get(field):
                        receipt[field] = body[field]
                from services.workflow_scheduling import prepare_receipt
                from services.work_lifecycle import ensure_work
                prepare_receipt(receipt, body.get("_scheduling"), receipt["_source"])
                ensure_work(self.store, db, "text", owner, request_id, receipt,
                            source=receipt["_source"], scheduling=receipt.get("_scheduling"))
                db.execute("INSERT INTO requests VALUES(?,?,?,?)", (owner, request_id, request_hash, json.dumps(receipt)))
        if schedule and self.admission is not None:
            self.admission.wake()
        elif schedule:
            try:
                self.executor.submit(self._run, owner, request_id, {**body, "_request_message_id": receipt["request_message_id"]})
            except TextTaskCapacityError:
                # The receipt already exists, so overload is a durable known
                # rejection that can be queried safely without another submit.
                self._update(
                    owner,
                    request_id,
                    status="failed",
                    error_code="TEXT_TASK_CAPACITY_EXCEEDED",
                    upstream_outcome="not_sent",
                    finished_at=self._now(),
                )
            except Exception:
                # No upstream call was scheduled, so this is a known rejection.
                self._update(owner, request_id, status="failed", error_code="CONVERSATION_SCHEDULING_FAILED")
        return self.read(owner, request_id)

    @staticmethod
    def _known_unsent_category_successor(receipt, now):
        timeline = receipt.get("_execution_timeline")
        if (not isinstance(timeline, list) or not timeline
                or any(not isinstance(item, dict) or item.get("stage") not in {"accepted", "execution_claimed"} for item in timeline)
                or receipt.get("_last_sent_sequence") is not None
                or type(receipt.get("send_count", 0)) is not int or receipt.get("send_count", 0) != 0
                or type(receipt.get("_bound_successor_resume_count", 0)) is not int
                or receipt.get("_bound_successor_resume_count", 0) != 0):
            return False
        for claim, until in (("_claim_id", "_claim_until"), ("recovery_claim_id", "recovery_lease_until")):
            expires = receipt.get(until)
            if (expires is not None and (type(expires) not in (int, float) or not math.isfinite(expires) or expires > now)
                    or receipt.get(claim) and expires is None):
                return False
        return bool(receipt.get("_route", "chat") == "chat" and receipt.get("_operation", "text") == "text"
            and not receipt.get("_public_session_ref") and not receipt.get("_forward_protocol")
            and receipt.get("status") == "failed" and receipt.get("error_code") == "CHAT_SUPERSEDE_CURSOR_CHANGED"
            and receipt.get("upstream_outcome") == "not_sent"
            and all(receipt.get(key) is False for key in ("_submission_started", "_turn_reserved", "_executing"))
            and not any(receipt.get(key) for key in ("_recovery_paused", "_recovery_suppressed", "_execution_wait_ended_at", "_attempt_finished_at"))
            and isinstance(receipt.get("_derived_input"), dict)
            and receipt["_derived_input"].get("kind") == "category_directory_parent_v1"
            and all(isinstance(receipt.get(key), str) and receipt[key] for key in
                ("_input_ref", "_supersedes_request_id", "request_id", "request_message_id")))

    def _validated_unsent_successor_input(self, db, owner, request_hash, receipt, now):
        if not self._known_unsent_category_successor(receipt, now):
            raise ConversationBindingError("successor is not known unsent", code="CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE")
        body = self.store.load_input(receipt["_input_ref"])
        if self._submission_identity(owner, body) != (receipt["request_id"], request_hash):
            raise ConversationBindingError("retained successor input changed", code="CONVERSATION_REQUEST_CONFLICT")
        previous = self._validate_supersede(self.store, db, owner, receipt, now)
        if previous.get("_recovery_paused") or previous.get("_recovery_suppressed"):
            raise ConversationBindingError("original recovery is paused", code="CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE")
        return body

    def resume_unsent_successor(self, owner, request_id, envelope):
        """One explicit atomic requeue of the same proved-unsent category request.

        No original outcome or identity is rewritten. The normal runner repeats
        its original-node checks at both the read and final send boundaries.
        """
        if self.admission is None:
            raise ConversationBindingError("durable admission is unavailable", code="CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE")
        if (set(envelope) != self.SUPERSEDE_FIELDS | {"derived_input"}
                or envelope.get("client_request_id") != request_id
                or envelope.get("derived_input") != {"kind": "category_directory_parent_v1"}):
            raise ConversationBindingError("original successor envelope required", code="CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE")
        expanded = self.submission_input(owner, envelope)
        identity = self._submission_identity(owner, expanded)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            if not row:
                raise ConversationBindingError("successor not found", code="CHAT_REQUEST_NOT_FOUND")
            if identity != (request_id, row[0]):
                raise ConversationBindingError("retained successor input changed", code="CONVERSATION_REQUEST_CONFLICT")
            receipt = json.loads(row[1])
            if receipt.get("_bound_successor_resume_count") == 1:
                return self._public(receipt)  # Concurrent/restart duplicates consume no second resume.
            try:
                self._validated_unsent_successor_input(db, owner, row[0], receipt, self._now())
            except (OSError, ValueError, TypeError, KeyError):
                raise ConversationBindingError("retained successor input unavailable", code="CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE") from None
            updated = dict(receipt)
            updated["_bound_successor_resume_failure"] = {key: receipt[key] for key in (
                "status", "error_code", "upstream_outcome", "finished_at", "updated_at", "_execution_timeline") if key in receipt}
            for key in ("error_code", "upstream_outcome", "finished_at", "waiting", "_ready_at", "_claim_id", "_claim_until"):
                updated.pop(key, None)
            updated.update(status="queued", boot=self.boot, updated_at=self._now(),
                _bound_successor_resume_count=1, _executing=False, _turn_reserved=False, _submission_started=False)
            updated["_execution_timeline"] = [*receipt["_execution_timeline"], {"stage": "known_unsent_successor_resumed", "at": self._now()}][-32:]
            self.store.write_receipt(db, "text", owner, request_id, updated)
        self.admission.wake()
        return self._public(updated)

    @staticmethod
    def _known_unsent_terminal_empty_correction(receipt, now):
        timeline = receipt.get("_execution_timeline") or []
        claim_until = receipt.get("_claim_until")
        sent_stages = {"send_guard_passed", "send_call_started", "response_headers_received", "first_output"}
        if (not isinstance(timeline, list)
                or any(not isinstance(item, dict) for item in timeline)
                or (claim_until is not None and (type(claim_until) not in {int, float}
                                                 or not math.isfinite(claim_until)))):
            return False
        return bool(
            receipt.get("route") == "chat"
            and receipt.get("_route", "chat") == "chat"
            and receipt.get("_operation", "text") == "text"
            and receipt.get("status") == "failed"
            and receipt.get("error_code") == "CHAT_TERMINAL_EMPTY_UNVERIFIED"
            and receipt.get("upstream_outcome") == "not_sent"
            and receipt.get("_submission_started") is False
            and receipt.get("_turn_reserved") is False
            and receipt.get("_executing") is False
            and not any(item.get("stage") in sent_stages for item in timeline)
            and bool(receipt.get("_input_ref"))
            and bool(receipt.get("_terminal_empty_correction_of"))
            and (claim_until or 0) <= now
        )

    def resume_unsent_terminal_empty(self, owner: str, request_id: str):
        """Explicitly requeue one proved-unsent correction using only its saved input.

        This is never called on startup or by ordinary result reads. A fresh
        original-result GET precedes the atomic state transition; execution
        repeats that GET before any model send.
        """
        now = self._now()
        with self._db() as db:
            row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                             (owner, request_id)).fetchone()
            if not row:
                raise ConversationBindingError("request not found", code="CHAT_REQUEST_NOT_FOUND")
            request_hash, raw = row
            receipt = json.loads(raw)
            if receipt.get("_terminal_empty_correction_of") and receipt.get("status") in {
                "queued", "running", "succeeded", "unknown",
            }:
                return self._public(receipt)
            if not self._known_unsent_terminal_empty_correction(receipt, now):
                raise ConversationBindingError("correction is not known unsent", code="CHAT_UNSENT_CORRECTION_NOT_RESUMABLE")
            previous = self.store.read_receipt(db, "text", owner, receipt["_terminal_empty_correction_of"])
        if (receipt.get("_recovery_paused") is True or (previous or {}).get("_recovery_paused") is True
                or (previous or {}).get("_recovery_suppressed") is True or not self._verified_retryable_empty(previous or {})):
            raise ConversationBindingError("original terminal proof changed", code="CHAT_TERMINAL_EMPTY_UNVERIFIED")
        try:
            body = self.store.load_input(receipt["_input_ref"])
            if self._submission_identity(owner, body) != (request_id, request_hash):
                raise ValueError("original input identity changed")
        except (OSError, ValueError, TypeError, KeyError, ConversationBindingError):
            raise ConversationBindingError("original input unavailable", code="CHAT_ORIGINAL_INPUT_UNAVAILABLE") from None
        try:
            proven = self.recovery_reader(previous)
        except Exception:
            raise ConversationBindingError("original result read unavailable", code="CHAT_TERMINAL_EMPTY_READ_UNAVAILABLE") from None
        if not self._fresh_terminal_empty_continuation(previous, receipt, proven):
            raise ConversationBindingError("original terminal proof changed", code="CHAT_TERMINAL_EMPTY_UNVERIFIED")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            current_row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                                     (owner, request_id)).fetchone()
            if not current_row or current_row[0] != request_hash:
                raise ConversationBindingError("original request changed", code="CHAT_REQUEST_CONFLICT")
            current = json.loads(current_row[1])
            if current.get("_terminal_empty_correction_of") == receipt["_terminal_empty_correction_of"] and current.get("status") in {
                "queued", "running", "succeeded", "unknown",
            }:
                return self._public(current)
            latest_previous = self.store.read_receipt(db, "text", owner, receipt["_terminal_empty_correction_of"])
            if (current_row[1] != raw
                    or not self._known_unsent_terminal_empty_correction(current, self._now())
                    or not self._fresh_terminal_empty_continuation(latest_previous, current, proven)):
                raise ConversationBindingError("original terminal proof changed", code="CHAT_TERMINAL_EMPTY_UNVERIFIED")
            updated = dict(current)
            for key in ("error_code", "upstream_outcome", "finished_at", "waiting", "_ready_at",
                        "_claim_id", "_claim_until"):
                updated.pop(key, None)
            updated.update(status="queued", boot=self.boot, updated_at=self._now(),
                           _executing=False, _turn_reserved=False, _submission_started=False)
            timeline = list(updated.get("_execution_timeline") or [])
            timeline.append({"stage": "known_unsent_resumed", "at": self._now()})
            updated["_execution_timeline"] = timeline[-32:]
            self.store.write_receipt(db, "text", owner, request_id, updated)
        if self.admission is not None:
            self.admission.wake()
        else:
            self.executor.submit(self._run, owner, request_id,
                                 {**body, "_request_message_id": updated["request_message_id"]})
        return self._public(updated)

    def _run(self, owner, request_id, body):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            receipt = json.loads(row[0])
            claim = body.get("_admission_claim")
            if claim:
                if receipt.get("_claim_id") != claim or receipt["status"] != "running":
                    return
            elif receipt["status"] != "queued" or receipt["boot"] != self.boot:
                return
            receipt = {**receipt, "status": "running", "started_at": self._now()}
            db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(receipt), owner, request_id))
            if receipt.get("_requested_account_identity"):
                body = {**body, "_requested_account_identity": receipt["_requested_account_identity"]}
            if receipt.get("_public_session_ref"):
                # Keep the immutable submitted envelope unchanged in storage.
                # Only the execution copy receives server-owned account/cursors.
                body = {**body, **{key: receipt[key] for key in (
                    "provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id",
                ) if receipt.get(key)}}
        def progress(cursor):
            self._update(owner, request_id, **cursor)
        if body.get("_forward"):
            from services.durable_forward import run
            return run(self, owner, request_id, body)
        try:
            if receipt.get("_same_session_retry_of"):
                from services.generation_completion import retry_evidence, same_session_retry
                with self.store.connect() as db:
                    original = self.store.read_receipt(db, "text", owner, receipt["_same_session_retry_of"])
                try:
                    fresh = self.recovery_reader(original) if same_session_retry(original, receipt) else None
                    validated, error, _, _ = self._safe_recovery_result(fresh, original)
                    if validated and not error:
                        # This is a verified original read, not a late runner
                        # callback. It may finish a locally retired attempt.
                        with self.store.transaction() as db:
                            recovered = self.store.read_receipt(db, "text", owner, original["request_id"])
                            if (same_session_retry(recovered, receipt) and recovered.get("status") != "succeeded"
                                    and recovered.get("request_message_id") == original.get("request_message_id")):
                                recovered.update(validated)
                                recovered.update(status="succeeded", upstream_outcome="completed",
                                                 error_code=None, waiting=None, _turn_reserved=False)
                                self.store.write_receipt(db, "text", owner, original["request_id"], recovered)
                    saved = retry_evidence(original or {})
                    current = retry_evidence({**(original or {}), "_retry_cursor": (validated or {}).get("_retry_cursor")})
                    if not saved or not current or any(current[k] != saved[k] for k in (
                            "conversation_id", "request_message_id", "retry_parent_message_id")):
                        raise ValueError("original branch changed")
                except Exception:
                    self._update(owner, request_id, status="failed", error_code="COMPLETION_ORIGINAL_CURSOR_UNCONFIRMED",
                                 upstream_outcome="not_sent", _turn_reserved=False, finished_at=self._now())
                    return
                body = {**body, "_failed_retry_original": original}
            if receipt.get("_supersedes_request_id"):
                with self.store.transaction() as db:
                    previous = self._validate_supersede(self.store, db, owner, receipt, self._now())
                body = {**body, "_supersedes_request_message_id": previous["request_message_id"]}
            if receipt.get("_terminal_empty_correction_of"):
                with self._db() as db:
                    previous = self.store.read_receipt(db, "text", owner, receipt["_terminal_empty_correction_of"])
                original_evidence = self._verified_retryable_empty(previous or {})
                try:
                    proven = self.recovery_reader(previous) if original_evidence else None
                except Exception:
                    # A failed exact-account read proves nothing about the
                    # original result. Keep the new turn unsent and retry its
                    # existing durable queue entry when the reader returns.
                    changes = {"error_code": "CHAT_TERMINAL_EMPTY_READ_UNAVAILABLE", "upstream_outcome": "not_sent",
                               "_turn_reserved": False, "_executing": False,
                               "waiting": {"reason": "previous_result_unverified"}}
                    if self.admission is not None:
                        changes.update(status="queued", _ready_at=self._now() + self.RECOVERY_BASE_BACKOFF_SECONDS,
                                       _claim_id=None, _claim_until=None)
                    else:
                        changes.update(status="failed", finished_at=self._now())
                    self._update(owner, request_id, **changes)
                    return
                if not self._fresh_terminal_empty_continuation(previous, receipt, proven):
                    self._update(owner, request_id, status="failed", error_code="CHAT_TERMINAL_EMPTY_UNVERIFIED",
                                 upstream_outcome="not_sent", _turn_reserved=False, finished_at=self._now())
                    return
                # Repeat the exact cursor check inside the binding lock, at
                # the final upstream send boundary (manual browser turns may
                # have advanced the original conversation since admission).
                body = {**body, "_empty_retry_original": previous, "_empty_retry_receipt": receipt}
            if receipt.get("_legacy_session_anchor"):
                with self._db() as db:
                    previous = self.store.read_receipt(db, "text", owner, receipt["_legacy_session_anchor"])
                try:
                    proven = self.recovery_reader(previous)
                except Exception:
                    proven = None
                if not proven or proven.get("status") != "succeeded":
                    changes = {"error_code": "CHAT_LEGACY_ANCHOR_UNVERIFIED", "upstream_outcome": "not_sent",
                               "_turn_reserved": False, "_executing": False}
                    if self.admission is not None:
                        changes.update(status="queued", _ready_at=self._now() + self.RECOVERY_BASE_BACKOFF_SECONDS,
                                       _claim_id=None, _claim_until=None,
                                       waiting={"reason": "previous_result_unverified"})
                    else:
                        changes.update(status="failed")
                    self._update(owner, request_id, **changes)
                    return
                if any(proven.get(key) != receipt.get(key) for key in (
                    "provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id",
                )):
                    self._update(owner, request_id, status="failed", error_code="CHAT_LEGACY_ANCHOR_CHANGED",
                                 upstream_outcome="not_sent", _turn_reserved=False)
                    return
            if body.get("_editable"):
                from services.editable_file_task_service import editable_file_task_service
                result = editable_file_task_service.run_admitted(body)
            else:
                result = self.runner(body, on_cursor=progress)
            if receipt.get("_supersedes_request_id"):
                result = {**result, "upstream_outcome": "completed", "error_code": None, "waiting": None}
            # The runner has finished its upstream work. Publish the saved result
            # and release this execution marker together: a client may complete
            # its work as soon as it reads success, before telemetry/finally run.
            self._update(owner, request_id, **{**result, "status": "succeeded",
                                              "_executing": False, "finished_at": self._now()})
            context = current_request.get()
            if context is not None and hasattr(context, "record_stage"):
                context.record_stage("artifact_saved")
        except ConversationBindingError as exc:
            cursor = {k: getattr(exc, k) for k in ("provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id") if getattr(exc, k, "")}
            if exc.code.startswith("CHAT_SUPERSEDE_"):
                transient = exc.code in {"CHAT_SUPERSEDE_READ_UNAVAILABLE", "CHAT_SUPERSEDE_PREDECESSOR_BUSY"}
                changes = {"error_code": exc.code, "upstream_outcome": "not_sent", "_turn_reserved": False,
                           "_executing": False, "_claim_id": None, "_claim_until": None}
                if transient and self.admission is not None:
                    changes.update(status="queued", _ready_at=self._now()+self.RECOVERY_BASE_BACKOFF_SECONDS,
                                   waiting={"reason": "superseded_original_unverified"})
                else:
                    changes.update(status="failed", finished_at=self._now())
                self._update(owner, request_id, **changes)
                return
            if exc.code == "CHAT_ARCHIVE_RESTORE_UNCONFIRMED":
                changes = {"error_code": exc.code, "upstream_outcome": "not_sent", "_turn_reserved": False,
                           "_executing": False, "waiting": {"reason": "archive_restore"}}
                if self.admission is not None:
                    changes.update(status="queued", _ready_at=self._now()+self.RECOVERY_BASE_BACKOFF_SECONDS,
                                   _claim_id=None, _claim_until=None)
                else:
                    changes.update(status="failed", finished_at=self._now())
                self._update(owner, request_id, **cursor, **changes)
                return
            diagnostic = {
                key: value for key in (
                    "original_failure_phase", "original_http_status", "original_exception_category",
                    "original_upstream_request_stage",
                    "original_upstream_error_form", "original_upstream_rejected_field",
                )
                if (value := getattr(exc, key, None)) is not None and value != ""
            }
            self._update(owner, request_id, **cursor, **diagnostic,
                         classify_before_send=exc.code != "CONVERSATION_OUTCOME_UNKNOWN",
                         status="unknown" if exc.code == "CONVERSATION_OUTCOME_UNKNOWN" else "failed",
                         error_code=exc.code, finished_at=self._now())
        except Exception:
            self._update(owner, request_id, status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                         original_failure_phase="runner", original_exception_category="other", finished_at=self._now())


text_task_service = TextTaskService(DATA_DIR / "text_tasks.sqlite3")
