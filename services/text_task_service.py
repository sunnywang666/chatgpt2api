"""Durable receipts for bound text requests, independent of HTTP lifetime.

Like image tasks, callers choose the request identity before sending. The
receipt stores results and cursors, never access tokens or uploaded image data.
An interrupted upstream operation is queried, never automatically resubmitted.
"""
from __future__ import annotations

import heapq
import itertools
import threading
from copy import deepcopy
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
    public_no_final_guard_diagnostic,
    safe_scan_failures,
    safe_no_final_guard_diagnostic,
    safe_recovery_read_error,
    TURN_END_EVIDENCE_FIELD,
    conversation_binding_service,
    is_recovery_image_pointer,
)


from services.task_store import TaskStore, recovery_control
from services.request_context import current_request, AdmissionLost
from services.account_service import account_service
from utils.upstream_error import safe_upstream_response_ids


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
    _BOUNDED_RECOVERY_STOPS = {
        TextRecoveryReason.REQUEST_RESULT_NOT_FOUND.value: "ORIGINAL_RESULT_NO_FINAL",
        TextRecoveryReason.REQUEST_MESSAGE_NOT_FOUND.value: "ORIGINAL_REQUEST_MESSAGE_NOT_FOUND",
    }
    _UNRECOVERABLE_REASONS = frozenset({
        TextRecoveryReason.CONVERSATION_NOT_FOUND.value,
        TextRecoveryReason.REQUEST_MESSAGE_NOT_FOUND.value,
        TextRecoveryReason.REQUEST_BRANCH_SUPERSEDED.value,
        TextRecoveryReason.REQUEST_RESULT_NOT_FOUND.value,
        TextRecoveryReason.REQUEST_RESULT_TERMINAL_EMPTY.value,
        TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value,
    })
    _RECONCILED_QUEUED_SUCCESSOR = "_reconciled_queued_successor"
    _RECONCILE_LINK_FIELDS = (
        "_supersedes_request_id", "_previous_request_id", "_completion_of",
        "_same_session_retry_of", "_terminal_empty_correction_of",
        "_codex_fallback_of", "_codex_continue_of", "_codex_rejected_retry_of",
        "_codex_origin_request_id", "_codex_response_id", "_previous_response_id",
        "_missing_successor_resume", "continue_before_queued_image_id",
        "_derived_input", "_supersedes_no_final",
    )
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
    def _queued_read_snapshot(receipt):
        """Whether a queued original can be projected without a write lease.

        Keep this deliberately narrower than the ordinary queued state. A
        caller may only skip ``BEGIN IMMEDIATE`` once the receipt proves that
        it has durable input but no send, execution, or recovery state which a
        read could correct or advance.
        """
        outcome = receipt.get("upstream_outcome")
        return bool(
            receipt.get("status") == "queued"
            and isinstance(receipt.get("_input_ref"), str) and bool(receipt["_input_ref"])
            and receipt.get("_submission_started") is False
            and receipt.get("_turn_reserved") is False
            and receipt.get("_executing") is not True
            and receipt.get("_claim_id") is None
            and receipt.get("recovery_claim_id") is None
            and receipt.get("_last_sent_sequence") is None
            and type(receipt.get("send_count", 0)) is int and receipt.get("send_count", 0) == 0
            and outcome in (None, "not_sent", "not_submitted")
            and receipt.get("error_code") is None
            and not receipt.get("upstream_unfinished")
            and not any(receipt.get(field) for field in (
                "_recovery_paused", "_recovery_suppressed", "_completion", "_completion_of",
            ))
            and all(receipt.get(field) is None for field in (
                "recovery_next_at", "_attempt_finished_at", "_execution_wait_ended_at",
            ))
        )

    @staticmethod
    def _retire_completed_recovery_claim(receipt, now):
        """Retire only a fully proved, expired original-read recovery lease."""
        finite = lambda value: type(value) in (int, float) and math.isfinite(value)
        claim_id = receipt.get("recovery_claim_id")
        claimed_at = receipt.get("recovery_claimed_at")
        lease_until = receipt.get("recovery_lease_until")
        execution_claim = receipt.get("_claim_id")
        finished_at = receipt.get("finished_at")
        if not finite(now):
            return False
        if not (
            receipt.get("status") == "succeeded"
            and receipt.get("upstream_outcome") == "completed"
            and not receipt.get("error_code")
            and isinstance(receipt.get("content"), str) and bool(receipt["content"])
            and finite(finished_at) and 0 < finished_at <= now
            and receipt.get("_executing") is False
            and receipt.get("_turn_reserved") is False
            and not receipt.get("_completion") and not receipt.get("_completion_of")
            and not receipt.get("_recovery_paused") and not receipt.get("_recovery_suppressed")
            and receipt.get("_recovery_claim_id") is None
            and receipt.get("_recovery_claim_until") is None
            and (receipt.get("upstream_unfinished") is None or receipt.get("upstream_unfinished") is False)
            and receipt.get("recovery_error_code") is None and receipt.get("recovery_reason") is None
            and receipt.get("_route") == "chat" and receipt.get("_operation") == "text"
            and receipt.get("_forward_protocol") is None and receipt.get("_forward") is None
            and not receipt.get("_image_recovery")
            and receipt.get("recovery_phase") == "read_text_request"
            and isinstance(claim_id, str) and bool(claim_id)
            and finite(claimed_at) and finite(lease_until)
            and claimed_at <= lease_until <= now
            and (not execution_claim or finite(receipt.get("_claim_until"))
                 and receipt["_claim_until"] <= now)
        ):
            return False
        receipt.update(
            recovery_claim_id=None,
            recovery_claimed_at=None,
            recovery_lease_until=None,
            updated_at=now,
        )
        return True

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
        from utils.upstream_error import safe_upstream_error
        result = {k: v for k, v in receipt.items() if k not in TextTaskService._INTERNAL_RECEIPT_FIELDS and not k.startswith("_")}
        # A cancellation proof is always recomputed from the request hash and
        # durable ProductWork parent below. Never echo a persisted projection.
        result.pop("cancelled_unsent", None)
        # Revalidate owner-scoped HTTP diagnostics on every read.  Receipts can
        # outlive code versions and must never turn arbitrary stored content
        # into a public error message.
        result.pop("original_upstream_error", None)
        detail = safe_upstream_error(receipt.get("original_upstream_error"))
        if detail:
            result["original_upstream_error"] = detail
        result.pop("original_upstream_response_ids", None)
        response_ids = safe_upstream_response_ids(receipt.get("original_upstream_response_ids"))
        if response_ids and receipt.get("status") != "succeeded":
            result["original_upstream_response_ids"] = response_ids
        if receipt.get("status") == "succeeded":
            # Older successful retries retained their previous active error.
            # Correct the read projection without rewriting original diagnostics.
            result.update(error_code=None, waiting=None, upstream_outcome="completed")
        # A persisted public-looking value is never trusted. The diagnostic is
        # emitted only for the matching active error code from a fixed private
        # enum record, so a later successful or unrelated receipt cannot leak
        # a stale failure explanation.
        result.pop("no_final_guard", None)
        no_final_guard = public_no_final_guard_diagnostic(
            receipt.get("_no_final_guard"), receipt.get("error_code"), receipt.get("status"),
        )
        if no_final_guard is not None:
            result["no_final_guard"] = no_final_guard
        result.pop("execution", None)
        result["recovery_control"] = recovery_control(receipt)
        if receipt.get("_product_work"):
            result["product_work"] = {"protocol": "product-work-v1", **receipt["_product_work"]}
        if receipt.get("_codex_origin_request_id"):
            fallback = {
                "protocol": str(receipt.get("_codex_fallback_protocol") or "codex-responses-v1"),
                "predecessor_request_id": receipt["_codex_origin_request_id"],
                "parent_request_id": (receipt.get("_codex_continue_of") or receipt.get("_codex_rejected_retry_of")
                                      or receipt.get("_codex_fallback_of")),
                "client_conversation_id": receipt.get("client_conversation_id"),
                "model": receipt.get("model"),
            }
            if receipt.get("_requested_account_ref"):
                fallback["account_ref"] = receipt["_requested_account_ref"]
            if receipt.get("_codex_reasoning_effort"):
                fallback["reasoning_effort"] = receipt["_codex_reasoning_effort"]
            if receipt.get("_codex_response_id"):
                fallback["response_id"] = receipt["_codex_response_id"]
            result["codex_fallback"] = fallback
        result.pop("recovery_automatic_stopped", None)
        result.pop("recovery_stop_reason", None)
        if (receipt.get("_attempt_finished_at")
                and receipt.get("_attempt_reason") in TextTaskService._BOUNDED_RECOVERY_STOPS.values()
                and receipt.get("status") == "failed"
                and receipt.get("upstream_outcome") == "unknown"
                and not receipt.get("_completion")):
            result.update(recovery_automatic_stopped=True, recovery_stop_reason=receipt["_attempt_reason"])
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
        if receipt.get("_supersedes_no_final"):
            result["continue_after_no_final"] = True
        if receipt.get("_derived_input"):
            result["derived_input"] = receipt["_derived_input"]
        missing_attempt = TextTaskService._missing_successor_attempt(receipt)
        if missing_attempt:
            result["missing_successor_resume"] = {
                "attempt": missing_attempt["attempt"],
                "state": missing_attempt["state"],
                "request_message_id": missing_attempt["request_message_id"],
            }
        completion = receipt.get("_completion")
        if isinstance(completion, dict):
            result["completion"] = {key: completion[key] for key in (
                "state", "reason", "replacement_id", "selected_id", "max_extra_requests", "conversation_mode",
                "automatic_empty_retry") if key in completion}
        return result

    def _public_with_product_cancellation(self, db, owner, request_hash, receipt):
        result = self._public(receipt)
        try:
            from services.product_work import cancelled_unsent_public
            proof = cancelled_unsent_public(self.store, db, owner, request_hash, receipt)
        except (OSError, ValueError, TypeError, KeyError):
            proof = None
        if proof:
            result["cancelled_unsent"] = proof
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

    @staticmethod
    def _provably_expired_execution_claim(receipt, now):
        claim, deadline = receipt.get("_claim_id"), receipt.get("_claim_until")
        return bool(
            isinstance(claim, str) and claim
            and type(deadline) in (int, float) and math.isfinite(deadline) and deadline <= now
        )

    @classmethod
    def _historical_model_not_found(cls, receipt, now):
        """Return true only for an inactive receipt with a proved model rejection."""
        from services.conversation_binding_service import is_model_not_found_rejection
        timeline = receipt.get("_execution_timeline")
        if (not is_model_not_found_rejection(receipt)
                or receipt.get("status") not in {"unknown", "failed"}
                or receipt.get("status") == "failed" and receipt.get("error_code") != "RESULT_UNRECOVERABLE"
                or "content" in receipt or "result" in receipt
                or receipt.get("_submission_started") is not True
                or receipt.get("_executing") is True
                or receipt.get("_claim_id") is not None and not cls._provably_expired_execution_claim(receipt, now)
                or receipt.get("recovery_claim_id") is not None
                or receipt.get("_recovery_paused") is True
                or receipt.get("_recovery_suppressed") is True
                or not isinstance(timeline, list)
                or not any(isinstance(item, dict) and item.get("stage") in {
                    "send_call_started", "response_headers_received"
                } for item in timeline)):
            return False
        return True

    @classmethod
    def _correct_historical_model_not_found(cls, receipt, now):
        if not cls._historical_model_not_found(receipt, now):
            return False
        # Keep immutable input, cursor/identity, original diagnostics, timeline,
        # and prior recovery observations.  This changes only the current local
        # projection so no future GET can claim a known rejection.
        receipt.update(
            status="failed",
            error_code="MODEL_NOT_FOUND",
            upstream_outcome="rejected",
            recovery_retryable=False,
            recovery_requires_new_conversation=False,
            recovery_next_at=None,
            waiting=None,
            _turn_reserved=False,
            _executing=False,
            finished_at=receipt.get("finished_at") or now,
            updated_at=now,
        )
        # Match the existing recovery fence: only a finite, expired executor
        # lease can be removed, so an old worker cannot finish over this repair.
        if receipt.get("_claim_id") is not None:
            receipt.update(_claim_id=None, _claim_until=None)
        return True

    def _historical_codex_request_rejection(self, receipt, now):
        """Recognize one inactive native-Codex 4xx receipt with no wire bytes."""
        wire_error = receipt.get("_wire_error_payload")
        error = wire_error.get("error") if isinstance(wire_error, dict) else None
        timeline = receipt.get("_execution_timeline")
        if (
            receipt.get("_route") != "codex"
            or receipt.get("_operation") != "text"
            or receipt.get("_forward_protocol") != "codex"
            or receipt.get("status") != "unknown"
            or receipt.get("error_code") != "codex_request_rejected"
            or receipt.get("upstream_outcome") != "unknown"
            or receipt.get("_submission_started") is not True
            or receipt.get("_upstream_terminal") is not True
            or receipt.get("_turn_reserved") is not True
            or receipt.get("_executing") is not False
            or not self._provably_expired_execution_claim(receipt, now)
            or receipt.get("recovery_claim_id") is not None
            or receipt.get("_recovery_claim_id") is not None
            or receipt.get("recovery_lease_until") is not None
            or receipt.get("_recovery_claim_until") is not None
            or receipt.get("_recovery_paused") is True
            or receipt.get("_recovery_suppressed") is True
            or receipt.get("_wire_head") is not None
            or type(receipt.get("_wire_size")) is not int
            or receipt["_wire_size"] != 0
            or not isinstance(receipt.get("_wire_output"), str)
            or receipt.get("_codex_response_id")
            or any(field in receipt for field in ("content", "result", "data", "result_file_ids", "result_sediment_ids"))
            or type(receipt.get("_wire_error_status")) is not int
            or not 400 <= receipt["_wire_error_status"] < 500
            or receipt["_wire_error_status"] in {408, 429}
            or not isinstance(error, dict)
            or error.get("code") != "codex_request_rejected"
            or error.get("type") != "invalid_request_error"
            or not isinstance(timeline, list)
            or not {"upstream_terminal", "task_finished"}.issubset(
                item.get("stage") for item in timeline if isinstance(item, dict)
            )
        ):
            return False
        try:
            with self.store.output_file(receipt["_wire_output"]) as handle:
                return handle.read(1) == b""
        except (OSError, TypeError, ValueError):
            # A missing, unsafe, or unreadable private file is not evidence of
            # an empty response, so leave the original outcome untouched.
            return False

    def _correct_historical_codex_request_rejection(self, receipt, now):
        if not self._historical_codex_request_rejection(receipt, now):
            return False
        # Keep the immutable input, upstream error, response identifiers, and
        # execution history. This only reconciles the stale local projection;
        # it never reads or resends upstream work.
        receipt.update(
            status="failed",
            upstream_outcome="rejected",
            recovery_retryable=False,
            recovery_requires_new_conversation=False,
            recovery_next_at=None,
            waiting=None,
            _turn_reserved=False,
            _executing=False,
            finished_at=receipt.get("finished_at") or now,
            updated_at=now,
        )
        receipt.update(_claim_id=None, _claim_until=None)
        return True

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

    @staticmethod
    def _generic_recovery_read_error(exc):
        """Persist only typed evidence when a direct bound read raises.

        Direct known-conversation reads do not pass through the bounded scan,
        so they need the same redacted shape as scan failures.  Never retain
        ``str(exc)``: provider exceptions can contain credentials or prompts.
        """
        status = getattr(exc, "status_code", None)
        status = status if type(status) is int and 100 <= status <= 599 else None
        retry_after = getattr(exc, "retry_after", None)
        retry_after = (int(retry_after) if isinstance(retry_after, (int, float))
                       and not isinstance(retry_after, bool) and math.isfinite(retry_after)
                       and 0 <= retry_after <= 2147483647 else None)
        name = type(exc).__name__.lower()
        category = (
            "timeout" if isinstance(exc, TimeoutError) or "timeout" in name else
            "transport" if isinstance(exc, ConnectionError)
            or any(part in name for part in ("connection", "proxy", "dns", "ssl")) else
            "parse" if isinstance(exc, (ValueError, TypeError, KeyError)) else
            "http" if status is not None else "other"
        )
        return {
            "phase": "conversation_detail", "category": category,
            "http_status": status, "retry_after_seconds": retry_after,
            "candidate_ref": None,
        }

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
    def _qualified_stalled_incomplete_capacity_boundary(cls, receipt, now):
        """Return whether an observed stagnant original frees only account capacity.

        This is a read-only planner predicate. It does not establish an upstream
        terminal result, alter the receipt, release its physical conversation,
        or change recovery cadence. A fresh original observation that progresses
        clears the persisted wait marker before this predicate is consulted.
        """
        if type(now) not in (int, float) or not math.isfinite(now) or now <= 0:
            return False
        if (not cls._bounded_chat_wait(receipt)
                or receipt.get("status") != "unknown"
                or receipt.get("recovery_reason") != TextRecoveryReason.REQUEST_RESULT_INCOMPLETE.value
                or receipt.get("upstream_outcome") != "unknown"
                or receipt.get("_attempt_finished_at") is not None
                or receipt.get("_submission_started") is not True
                or type(receipt.get("_turn_reserved")) is not bool
                or receipt.get("_executing") is not False
                or receipt.get("recovery_claim_id") is not None
                or receipt.get("recovery_lease_until") is not None):
            return False
        if (receipt.get("_recovery_claim_id") is not None
                or receipt.get("_recovery_claim_until") is not None):
            return False
        if (receipt.get("_upstream_terminal") is not None and receipt.get("_upstream_terminal") is not False
                or any(receipt.get(field) is not None and receipt.get(field) is not False
                       for field in ("_recovery_suppressed", "_recovery_paused", "_forward_protocol"))):
            return False
        # Old durable Chat rows omitted route/operation; preserve that existing
        # Chat/text default while rejecting any explicit incompatible value.
        if receipt.get("_route", "chat") != "chat" or receipt.get("_operation", "text") != "text":
            return False
        if any(type(receipt.get(field)) is not str
               or not 1 <= len(receipt[field]) <= 300
               or receipt[field] != receipt[field].strip()
               for field in ("request_id", "request_message_id", "provider_binding_id",
                             "provider_account_identity", "client_conversation_id",
                             "conversation_id", "model")):
            return False
        claim_id, claim_until = receipt.get("_claim_id"), receipt.get("_claim_until")
        if claim_id is None:
            if claim_until is not None:
                return False
        elif (not isinstance(claim_id, str) or not claim_id
              or type(claim_until) not in (int, float) or not math.isfinite(claim_until)
              or claim_until > now):
            return False
        observation = cls._safe_result_observation(receipt.get("_original_result_observation"), receipt)
        wait_ended = receipt.get("_result_wait_ended_at")
        started = receipt.get("_result_observation_started_at")
        last_progress = receipt.get("_result_last_progress_at")
        last_checked = receipt.get("_result_last_checked_at")
        reads = receipt.get("_result_no_progress_reads")
        if (observation is None
                or type(wait_ended) not in (int, float) or not math.isfinite(wait_ended) or wait_ended <= 0
                or type(started) not in (int, float) or not math.isfinite(started) or started <= 0
                or last_progress is not None and (type(last_progress) not in (int, float)
                                                   or not math.isfinite(last_progress) or last_progress <= 0)
                or type(last_checked) not in (int, float) or not math.isfinite(last_checked) or last_checked <= 0
                or type(reads) is not int or reads < cls.UNRECOVERABLE_QUALIFIED_READS):
            return False
        progress = last_progress if last_progress is not None else started
        if not (started <= progress <= wait_ended <= last_checked <= now):
            return False
        # A later snapshot clock cannot rehabilitate an early/stale marker:
        # the marker itself must have been recorded after the existing window.
        if (wait_ended - progress < cls.UNRECOVERABLE_MIN_AGE_SECONDS
                or now - progress < cls.UNRECOVERABLE_MIN_AGE_SECONDS):
            return False
        return True

    @classmethod
    def _end_execution_wait(cls, receipt, now, *, observed=False):
        """End local waiting, not the upstream turn or its original-ID recovery.

        The existing age and qualified-read policy ends foreground waiting.
        A timeout, failed GET or arbitrary failed receipt alone cannot do so.
        An incomplete-result marker is not proof of an ended upstream turn;
        local capacity is assessed separately from its persisted strict
        observation evidence. Persist this decision across later failed reads
        and restart.
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
            # does not prove the upstream turn ended. The foreground marker
            # keeps UNKNOWN and the original lease; a separate planner-only
            # predicate may free local account capacity only after strict,
            # persisted original-observation evidence.
            progress = receipt.get("_result_last_progress_at") or receipt.get("_result_observation_started_at")
            reads = receipt.get("_result_no_progress_reads")
            if (not observed or receipt.get("_result_wait_ended_at") is not None
                    or type(progress) not in (int, float) or not math.isfinite(progress)
                    or now - progress < cls.UNRECOVERABLE_MIN_AGE_SECONDS
                    or type(reads) is not int or reads < cls.UNRECOVERABLE_QUALIFIED_READS):
                return False
            receipt.update(_result_wait_ended_at=now, updated_at=now)
            return True
        created = cls._recovery_started_at(receipt, now)
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
        """Bound original Chat reads without claiming that its upstream ended.

        A missing original node is distinct from a known empty result branch.
        Work ownership does not require querying an absent node forever; keep
        that work and conversation protected rather than closing or archiving
        them. Active completion investigators retain their existing contract.
        The existing explicit recovery endpoint may still make one fresh read.
        """
        reason = receipt.get("recovery_reason")
        stop_reason = cls._BOUNDED_RECOVERY_STOPS.get(reason)
        if (not cls._bounded_chat_wait(receipt)
                or not receipt.get("_execution_wait_ended_at")
                or receipt.get("_attempt_finished_at")
                or any(receipt.get(key) for key in (
                    "_completion", "_completion_of",
                    "_upstream_terminal", "recovery_claim_id", "_claim_id", "_executing"))
                or receipt.get("status") != "failed"
                or receipt.get("error_code") != "RESULT_UNRECOVERABLE"
                or receipt.get("upstream_outcome") != "unknown"
                or stop_reason is None
                # A generic work owner retains its original completion
                # contract. Product ownership has its own durable parent and
                # may stop polling the exact no-result original after the
                # qualified-read budget; that stop neither closes nor archives
                # the product work.
                or (reason == TextRecoveryReason.REQUEST_RESULT_NOT_FOUND.value
                    and receipt.get("_work_key"))
                or not all(isinstance(receipt.get(key), str) and receipt[key].strip() for key in (
                    "request_message_id", "provider_binding_id", "provider_account_identity",
                    "client_conversation_id", "conversation_id"))
                or type(receipt.get("recovery_no_result_reads")) is not int
                or receipt["recovery_no_result_reads"] < cls.UNRECOVERABLE_QUALIFIED_READS):
            return False
        # A saved branch cursor is evidence for an explicit later recheck,
        # not an active completion owner. Preserve it without polling forever.
        receipt.update(_attempt_finished_at=now, _attempt_reason=stop_reason,
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
            missing_attempt = self._missing_successor_attempt(current)
            if current.get("status") == "succeeded" or (
                current.get("status") == "failed" and current.get("error_code") == "CHAT_RESPONSE_NOT_TEXT"
            ):
                # A late read cannot roll back an authoritative terminal
                # result. Retire only this reader's still-current recovery
                # lease; a late reader must never clear a newer worker's.
                current_claim = current.get("recovery_claim_id")
                if current_claim and current_claim == claim_id:
                    current = {
                        **current,
                        "recovery_claim_id": None,
                        "recovery_claimed_at": None,
                        "recovery_lease_until": None,
                        "updated_at": now,
                    }
                    db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=? AND receipt=?",
                               (json.dumps(current), owner, request_id, row[0]))
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
                created_at = self._recovery_started_at(current, now)
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
            if missing_attempt:
                # The recovered branch belongs to the new private message ID.
                # Its request-parent cursor must not overwrite the original
                # submitted successor's historical cursor.
                changes.pop("request_parent_message_id", None)
                if not error_code or (recovered or {}).get("status") == "failed":
                    source = (missing_attempt.get("result_source")
                              if missing_attempt.get("state") == "recovered" else "replacement")
                    changes["_missing_successor_resume"] = {
                        **missing_attempt, "state": "recovered", "recovered_at": now,
                        "result_source": source,
                    }
            if not error_code:
                changes["recovery_last_read_error"] = None
            recovery_view = self._missing_successor_recovery_receipt(current) or current
            observation = self._safe_result_observation(
                (recovered or {}).get("_original_result_observation"), recovery_view,
            )
            observed = bool(error_code == "UPSTREAM_OUTCOME_UNKNOWN"
                            and recovery_reason == TextRecoveryReason.REQUEST_RESULT_INCOMPLETE.value
                            and observation is not None)
            if observed:
                previous = self._safe_result_observation(
                    current.get("_original_result_observation"), recovery_view,
                )
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
            if observed and changed:
                # A verified original-branch change puts the local account
                # boundary back on hold. It does not alter UNKNOWN, the
                # original turn, or recovery timing.
                updated.pop("_result_wait_ended_at", None)
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
            if (error_code and updated.get("_attempt_reason") in self._BOUNDED_RECOVERY_STOPS.values()
                    and recovery_reason in self._BOUNDED_RECOVERY_STOPS):
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

    def archive_product_work(self, owner, request_id, archived):
        """Archive the original bound conversation under its durable parent guard."""
        from services.request_context import current_archive_guard
        from services.work_lifecycle import WorkLifecycleError
        guard = current_archive_guard.get()
        if guard is None or archived is not True:
            raise WorkLifecycleError("PRODUCT_WORK_ARCHIVE_GUARD_REQUIRED")
        guard()
        with self._db() as db:
            receipt = self.store.read_receipt(db, "text", owner, request_id)
        required = ("_product_work_key", "provider_binding_id", "provider_account_identity",
                    "client_conversation_id", "conversation_id", "parent_message_id")
        if not receipt or any(not receipt.get(k) for k in required):
            raise WorkLifecycleError("PRODUCT_WORK_ARCHIVE_CURSOR_UNAVAILABLE")
        conversation_binding_service.set_archived(receipt, archived)
        return {"request_id": request_id, "conversation_id": receipt["conversation_id"], "archived": archived}

    def restore_public_session(self, owner: str, request_id: str) -> dict[str, object]:
        return self.set_public_session_archived(owner, request_id, False)

    def read_snapshot(self, owner: str, request_id: str):
        """Return the persisted original without joining its recovery work.

        PoolAdmission owns automatic original-result recovery independently of
        HTTP polling. In particular, UNKNOWN and failed snapshots must neither
        acquire the writer lock nor claim an upstream read or reopen a stopped
        attempt. Explicit recovery keeps using read/recover below.
        """
        with self._db() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                             (owner, request_id)).fetchone()
            if row is None:
                return {"request_id": request_id, "status": "not_found"}
            receipt = json.loads(row[1])
            # Match ``read``'s strictly evidenced native-Codex rejection
            # projection without changing the stored original. Snapshot is an
            # observational GET: the predicate may read the retained empty
            # wire file, but it never claims recovery or writes this receipt.
            now = self._now()
            self._correct_historical_codex_request_rejection(receipt, now)
            return self._local_read_projection(db, owner, row[0], receipt, now)

    def _local_read_projection(self, db, owner, request_hash, receipt, now):
        result = self._public_with_product_cancellation(db, owner, request_hash, receipt)
        # Keep the existing known-unsent eligibility advertisement. Its local
        # input/predecessor proof is observational; the explicit resume endpoint
        # still revalidates it under a writer before admitting any work.
        if self.admission is not None and (
                self._known_unsent_category_successor(receipt, now)
                or self._known_unsent_no_final_successor(receipt, now)):
            try:
                self._validated_unsent_successor_input(db, owner, request_hash, receipt, now)
                result.update(bound_successor_resume_retryable=True, upstream_outcome="not_submitted",
                              upstream_submission_started=False)
            except (ConversationBindingError, OSError, ValueError, TypeError, KeyError):
                pass
        return result

    def read(self, owner: str, request_id: str, *, allow_unrecoverable_retry: bool = False,
             _explicit_ended_recheck: bool = False):
        from services.pool_admission import unknown_text_result
        recovery_claim = None
        recovery_target = None
        now = self._now()
        # Queued originals and saved answers that need no maintenance can be
        # projected without competing with admission for the SQLite writer
        # lock. Any recovery or completed-lease cleanup still rereads under
        # the existing write transaction before changing the receipt.
        with self._db() as db:
            snapshot = db.execute(
                "SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", (owner, request_id),
            ).fetchone()
        if snapshot:
            receipt = json.loads(snapshot[1])
            completed = (receipt.get("status") == "succeeded"
                         and isinstance(receipt.get("content"), str) and bool(receipt["content"].strip())
                         # Probe a copy: an eligible cleanup must be persisted
                         # transactionally below, never only in the response.
                         and not self._retire_completed_recovery_claim(dict(receipt), now))
            if self._queued_read_snapshot(receipt) or completed:
                if receipt.get("_product_unsent_cancelled"):
                    return self.read_snapshot(owner, request_id)
                return self._public(receipt)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            if row:
                previous = json.loads(row[0])
                corrected = self._correct_historical_model_not_found(previous, now)
                if not corrected:
                    corrected = self._correct_historical_codex_request_rejection(previous, now)
                ended_wait = False if corrected else self._end_execution_wait(previous, now)
                if corrected or self._end_no_final_recovery(previous, now) or ended_wait:
                    db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?",
                               (json.dumps(previous), owner, request_id))
                    row = (json.dumps(previous),)
                if self._retire_completed_recovery_claim(previous, now):
                    db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=? AND receipt=?",
                               (json.dumps(previous), owner, request_id, row[0]))
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
                    and previous.get("_attempt_reason") in self._BOUNDED_RECOVERY_STOPS.values()
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
                    recovery_target = self._missing_successor_recovery_receipt(recovery_claim[1]) or recovery_claim[1]
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
                target = recovery_target or recovery_claim[1]
                recovered = self.recovery_reader(target)
                safe_result, error_code, phase, recovery_reason = self._safe_recovery_result(recovered, target)
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
                recovery_error = self._generic_recovery_read_error(exc)
                result = self._finish_recovery(
                    owner, request_id, recovery_claim[0],
                    error_code=recovery_error_code,
                    phase=recovery_error["phase"], read_error=recovery_error,
                    retry_after_seconds=retry_after_seconds,
                    recovery_reason=None, count_unrecoverable=allow_unrecoverable_retry,
                )
            finally:
                recovery_done.set()
                if lease_started:
                    lease_thread.join(timeout=1)
            return self._authorize_unrecoverable(owner, request_id, result) if allow_unrecoverable_retry else result
        observed = json.loads(row[0])
        result = self._public(observed)
        if observed.get("_product_unsent_cancelled"):
            result = self.read_snapshot(owner, request_id)
        # This advertisement is a local proof only, never an implicit retry or
        # upstream read. The explicit endpoint revalidates it transactionally.
        if self.admission is not None and self._known_unsent_category_successor(json.loads(row[0]), now):
            try:
                with self._db() as db:
                    current = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
                    if current:
                        result = self._local_read_projection(db, owner, current[0], json.loads(current[1]), self._now())
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
            created_at = self._recovery_started_at(current, now)
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
                changes.update(error_code=None, waiting=None, upstream_outcome="completed", _no_final_guard=None)
            if "original_upstream_response_ids" in changes:
                original_ids = safe_upstream_response_ids(receipt.get("original_upstream_response_ids"))
                incoming_ids = safe_upstream_response_ids(changes["original_upstream_response_ids"])
                # A response trace belongs to the original failed HTTP
                # response. Recovery/read failures are separate observations
                # and cannot replace it; malformed input is discarded.
                if original_ids or not incoming_ids:
                    changes.pop("original_upstream_response_ids")
                else:
                    changes["original_upstream_response_ids"] = incoming_ids
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
    REJECTED_SUPERSEDE_FIELDS = frozenset({"supersedes_request_id", "client_request_id", "client_conversation_id"})
    MISSING_NODE_RETRY_FIELD = "continue_before_queued_image_id"
    MISSING_NODE_RETRY_FIELDS = REJECTED_SUPERSEDE_FIELDS | frozenset({MISSING_NODE_RETRY_FIELD})

    @staticmethod
    def _rejected_chat_successor_source(receipt, *, allow_no_final_successor=False):
        """Only a proved pre-conversation model rejection can start this path."""
        from services.conversation_binding_service import is_model_not_found_rejection
        return bool(
            receipt.get("status") == "failed"
            and receipt.get("error_code") == "MODEL_NOT_FOUND"
            and receipt.get("upstream_outcome") == "rejected"
            and receipt.get("_submission_started") is True
            and receipt.get("_route", "chat") == "chat"
            and receipt.get("_operation", "text") == "text"
            and not receipt.get("_forward_protocol")
            and receipt.get("_executing") is False
            and receipt.get("_turn_reserved") is False
            and not any(receipt.get(key) for key in (
                "recovery_claim_id", "_recovery_claim_id", "_recovery_paused", "_recovery_suppressed",
                "upstream_unfinished", "_codex_fallback_of",
            ))
            and (allow_no_final_successor or not receipt.get("_supersedes_request_id"))
            and not any(key in receipt for key in ("content", "result", "data"))
            and not receipt.get("result_file_ids") and not receipt.get("result_sediment_ids")
            and is_model_not_found_rejection(receipt)
        )

    @classmethod
    def _rejected_no_final_successor_source(cls, store, db, owner, receipt, now):
        """Recognize one typed rejection of one proved no-final successor.

        The model retry is deliberately not a general successor-of-successor
        feature.  Its rejected parent must be the sole, server-registered,
        explicit no-final child of an original UNKNOWN receipt, with both
        retained inputs and hashes still proving that exact relationship.
        """
        if (not cls._rejected_chat_successor_source(receipt, allow_no_final_successor=True)
                or receipt.get("_supersedes_no_final") is not True):
            return False
        parent_id = receipt.get("request_id")
        original_id = receipt.get("_supersedes_request_id")
        if (not isinstance(parent_id, str) or not parent_id
                or not isinstance(original_id, str) or not original_id or parent_id == original_id):
            return False
        original_row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                                  (owner, original_id)).fetchone()
        parent_row = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?",
                                (owner, parent_id)).fetchone()
        if not original_row or not parent_row or receipt.get("_supersedes_input_hash") != original_row[0]:
            return False
        try:
            original = json.loads(original_row[1])
            original_body = store.load_input(original["_input_ref"])
            parent_body = store.load_input(receipt["_input_ref"])
        except (OSError, ValueError, TypeError, KeyError):
            return False
        if (original.get("request_id") != original_id
                or original.get("status") != "failed"
                or original.get("error_code") != "RESULT_UNRECOVERABLE"
                or original.get("upstream_outcome") != "unknown"
                or original.get("_supersedes_request_id")
                or original.get("_derived_input")
                or original.get("_upstream_terminal")
                or original.get("_executing") is not False
                or original.get("_turn_reserved") is not False
                or original.get("upstream_unfinished")
                or any(key in original for key in ("content", "result", "data"))
                or original.get("result_file_ids") or original.get("result_sediment_ids")
                or not cls._no_final_successor_source(original, now)
                or cls._submission_identity(owner, original_body) != (original_id, original_row[0])):
            return False
        for claim_key, until_key in (("_claim_id", "_claim_until"),
                                     ("recovery_claim_id", "recovery_lease_until"),
                                     ("_recovery_claim_id", "_recovery_claim_until")):
            until = original.get(until_key)
            if (original.get(claim_key) and (type(until) not in (int, float)
                                             or not math.isfinite(until) or until > now)):
                return False
            if until is not None and (type(until) not in (int, float) or not math.isfinite(until)):
                return False
        if (original_body.get("supersedes_request_id") or original_body.get("continue_after_no_final") is not None
                or original_body.get("derived_input") is not None):
            return False
        expected_parent = {**original_body, "client_request_id": parent_id,
                           "supersedes_request_id": original_id, "continue_after_no_final": True}
        if (parent_body != expected_parent
                or cls._submission_identity(owner, parent_body) != (parent_id, parent_row[0])
                or not cls._supersede_order_link(original, receipt)):
            return False
        children = db.execute(
            "SELECT id FROM requests WHERE owner=? AND "
            "(json_extract(receipt,'$._supersedes_request_id')=? OR json_extract(receipt,'$._codex_fallback_of')=?)",
            (owner, original_id, original_id),
        ).fetchall()
        return len(children) == 1 and children[0][0] == parent_id

    @classmethod
    def _rejected_missing_node_successor_source(cls, store, db, owner, receipt, now):
        """Recognize the one rejected child of one stopped missing-node root.

        This is deliberately separate from ``_rejected_no_final_successor_source``:
        a missing original user node is not evidence of a no-final result
        branch.  It can only support one typed-rejection retry when every
        durable identity in R -> P still proves the exact original branch.
        """
        if (not cls._rejected_chat_successor_source(receipt, allow_no_final_successor=True)
                or receipt.get("_supersedes_no_final") is True
                or receipt.get(cls.MISSING_NODE_RETRY_FIELD) is not None):
            return False
        parent_id = receipt.get("request_id")
        original_id = receipt.get("_supersedes_request_id")
        if (not isinstance(parent_id, str) or not parent_id
                or not isinstance(original_id, str) or not original_id or parent_id == original_id):
            return False
        original_row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                                  (owner, original_id)).fetchone()
        parent_row = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?",
                                (owner, parent_id)).fetchone()
        if not original_row or not parent_row or receipt.get("_supersedes_input_hash") != original_row[0]:
            return False
        try:
            original = json.loads(original_row[1])
            original_body = store.load_input(original["_input_ref"])
            parent_body = store.load_input(receipt["_input_ref"])
        except (OSError, ValueError, TypeError, KeyError):
            return False
        if (original.get("request_id") != original_id
                or original.get("status") != "failed"
                or original.get("error_code") != "RESULT_UNRECOVERABLE"
                or original.get("upstream_outcome") != "unknown"
                or original.get("recovery_reason") != "REQUEST_MESSAGE_NOT_FOUND"
                or original.get("_attempt_reason") != "ORIGINAL_REQUEST_MESSAGE_NOT_FOUND"
                or original.get("_supersedes_request_id")
                or original.get("_supersedes_no_final") is True
                or original.get("_derived_input")
                or original.get("_upstream_terminal")
                or original.get("_executing") is not False
                or original.get("_turn_reserved") is not False
                or original.get("upstream_unfinished")
                or any(original.get(key) for key in (
                    "_recovery_paused", "_recovery_suppressed", "_work_key", "_completion", "_completion_of",
                ))
                or any(key in original for key in ("content", "result", "data"))
                or original.get("result_file_ids") or original.get("result_sediment_ids")
                or type(original.get("_attempt_finished_at")) not in (int, float)
                or not math.isfinite(original["_attempt_finished_at"])
                or original["_attempt_finished_at"] > now
                or type(original.get("_execution_wait_ended_at")) not in (int, float)
                or not math.isfinite(original["_execution_wait_ended_at"])
                or original["_execution_wait_ended_at"] > now
                or type(original.get("recovery_no_result_reads")) is not int
                or original["recovery_no_result_reads"] < cls.UNRECOVERABLE_QUALIFIED_READS
                or cls._submission_identity(owner, original_body) != (original_id, original_row[0])):
            return False
        for claim_key, until_key in (("_claim_id", "_claim_until"),
                                     ("recovery_claim_id", "recovery_lease_until"),
                                     ("_recovery_claim_id", "_recovery_claim_until")):
            until = original.get(until_key)
            if (original.get(claim_key) and (type(until) not in (int, float)
                                             or not math.isfinite(until) or until > now)):
                return False
            if until is not None and (type(until) not in (int, float) or not math.isfinite(until)):
                return False
        if (original_body.get("supersedes_request_id") or original_body.get("continue_after_no_final") is not None
                or original_body.get("derived_input") is not None
                or original_body.get(cls.MISSING_NODE_RETRY_FIELD) is not None):
            return False
        expected_parent = {**original_body, "client_request_id": parent_id,
                           "supersedes_request_id": original_id}
        if (parent_body != expected_parent
                or cls._submission_identity(owner, parent_body) != (parent_id, parent_row[0])
                or not cls._supersede_order_link(original, receipt)):
            return False
        children = db.execute(
            "SELECT id FROM requests WHERE owner=? AND "
            "(json_extract(receipt,'$._supersedes_request_id')=? OR json_extract(receipt,'$._codex_fallback_of')=?)",
            (owner, original_id, original_id),
        ).fetchall()
        return len(children) == 1 and children[0][0] == parent_id

    @classmethod
    def _queued_image_for_missing_node_retry(cls, store, db, owner, original, successor, now):
        """Return the one unchanged queued image explicitly bound to C, if safe."""
        image_id = successor.get(cls.MISSING_NODE_RETRY_FIELD)
        if not isinstance(image_id, str) or not image_id or len(image_id) > 200:
            return None
        image = store.read_receipt(db, "image", owner, image_id)
        if not image:
            return None
        exact = ("_product_work_key", "provider_binding_id", "provider_account_identity",
                 "client_conversation_id", "conversation_id")
        if (any(not isinstance(original.get(key), str) or not original[key]
                or image.get(key) != original[key] for key in exact)
                or type(original.get("_sequence")) is not int
                or type(image.get("_sequence")) is not int
                or image["_sequence"] <= original["_sequence"]
                or image.get("status") != "queued"
                or image.get("upstream_outcome") not in (None, "not_sent", "not_submitted")
                or image.get("_submission_started") is not False
                or image.get("upstream_submission_started") not in (None, False)
                or image.get("_turn_reserved") is not False
                or image.get("_executing") is True
                or image.get("_claim_id") is not None
                or image.get("recovery_claim_id") is not None
                or image.get("_last_sent_sequence") is not None
                or type(image.get("send_count", 0)) is not int or image.get("send_count", 0) != 0
                or image.get("request_message_id")
                or image.get("result_file_ids") or image.get("result_sediment_ids") or image.get("data")
                or image.get("upstream_unfinished")
                or image.get("error_code") is not None
                or any(image.get(key) for key in ("_recovery_paused", "_recovery_suppressed", "_completion", "_completion_of"))
                or not isinstance(image.get("_input_ref"), str) or not image["_input_ref"]):
            return None
        try:
            from services.image_task_service import _request_hash
            frozen = store.load_input(image["_input_ref"])
            payload = frozen["payload"]
            image_identity = frozen["identity"]
        except (ImportError, OSError, ValueError, TypeError, KeyError):
            return None
        if (not isinstance(payload, dict) or not isinstance(image_identity, dict)
                or frozen.get("mode") != image.get("mode")
                or image_identity.get("role") != "admin"
                or image_identity.get("_trusted_source") != image.get("_source")
                or image.get("request_hash") != _request_hash(image["mode"], payload)
                or any(payload.get(key) != image.get(key) for key in exact[1:])):
            return None
        from services.pool_admission import unfinished
        for kind, other_owner, other_id, other in store.receipts(db):
            if (other_owner != owner
                    or (kind == "text" and other_id in {original.get("request_id"), successor.get("request_id")})
                    or (kind == "image" and other_id == image_id)):
                continue
            same_conversation = all(other.get(key) == original.get(key) for key in exact[1:])
            if same_conversation and unfinished(kind, other):
                return None
        return image

    @classmethod
    def _missing_node_model_rejection_retry(cls, store, db, owner, receipt, now):
        """Validate C as the sole queued retry of typed rejected P after missing R."""
        parent_id = receipt.get("_supersedes_request_id")
        request_id = receipt.get("request_id")
        image_id = receipt.get(cls.MISSING_NODE_RETRY_FIELD)
        # This runs while the pool builds every queue view.  Keep ordinary
        # rejected/no-final rows on the constant-time path; only C can carry
        # the marker that warrants receipt/input traversal.
        if (not isinstance(image_id, str) or not image_id
                or not isinstance(parent_id, str) or not parent_id
                or not isinstance(request_id, str) or not request_id):
            return False
        parent_row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                                (owner, parent_id)).fetchone()
        if not parent_row or receipt.get("_supersedes_input_hash") != parent_row[0]:
            return False
        try:
            parent = json.loads(parent_row[1])
            parent_body = store.load_input(parent["_input_ref"])
        except (OSError, ValueError, TypeError, KeyError):
            return False
        if not cls._rejected_missing_node_successor_source(store, db, owner, parent, now):
            return False
        original_id = parent.get("_supersedes_request_id")
        original = store.read_receipt(db, "text", owner, original_id) if isinstance(original_id, str) else None
        if not original or cls._queued_image_for_missing_node_retry(store, db, owner, original, receipt, now) is None:
            return False
        row = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
        expected = cls._rejected_successor_body(
            parent, parent_body, parent_id, request_id, receipt.get("client_conversation_id"),
            missing_node_retry=True, queued_image_id=receipt.get(cls.MISSING_NODE_RETRY_FIELD),
        )
        if not expected:
            return False
        if isinstance(receipt.get("_input_ref"), str):
            try:
                body = store.load_input(receipt["_input_ref"])
            except (OSError, ValueError, TypeError, KeyError):
                return False
            if body != expected:
                return False
        elif any(receipt.get(key) != expected.get(key) for key in (
                "provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id",
                "client_conversation_id", cls.MISSING_NODE_RETRY_FIELD)):
            return False
        children = db.execute(
            "SELECT id FROM requests WHERE owner=? AND "
            "(json_extract(receipt,'$._supersedes_request_id')=? OR json_extract(receipt,'$._codex_fallback_of')=?)",
            (owner, parent_id, parent_id),
        ).fetchall()
        return bool((row is None or cls._submission_identity(owner, expected) == (request_id, row[0]))
                    and ((row is None and not children) or (len(children) == 1 and children[0][0] == request_id))
                    and cls._supersede_order_link(parent, receipt))

    @classmethod
    def _missing_node_model_rejection_retry_consumed(cls, store, db, owner, receipt):
        """Recognize a registered R -> P -> C missing-node retry structurally.

        This deliberately ignores live status, claims, recovery fields, and
        cursor movement.  Once the immutable inputs and submission roots show
        that C was registered, later observations of R/P cannot turn C into a
        new rejected request eligible for another successor or Codex fallback.
        """
        parent_id = receipt.get("_supersedes_request_id")
        request_id = receipt.get("request_id")
        image_id = receipt.get(cls.MISSING_NODE_RETRY_FIELD)
        if (not isinstance(parent_id, str) or not parent_id
                or not isinstance(request_id, str) or not request_id
                or not isinstance(image_id, str) or not image_id):
            return False
        parent_row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                                (owner, parent_id)).fetchone()
        if not parent_row or receipt.get("_supersedes_input_hash") != parent_row[0]:
            return False
        try:
            parent = json.loads(parent_row[1])
        except (TypeError, ValueError):
            return False
        original_id = parent.get("_supersedes_request_id")
        if (parent.get("_supersedes_no_final") is True
                or not isinstance(original_id, str) or not original_id):
            return False
        original_row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                                  (owner, original_id)).fetchone()
        if not original_row or parent.get("_supersedes_input_hash") != original_row[0]:
            return False
        try:
            original = json.loads(original_row[1])
            original_body = store.load_input(original["_input_ref"])
            parent_body = store.load_input(parent["_input_ref"])
            body = store.load_input(receipt["_input_ref"])
        except (OSError, ValueError, TypeError, KeyError):
            return False
        image = store.read_receipt(db, "image", owner, image_id)
        if not image:
            return False
        try:
            from services.image_task_service import _request_hash
            image_body = store.load_input(image["_input_ref"])
            image_payload = image_body["payload"]
            image_identity = image_body["identity"]
        except (ImportError, OSError, ValueError, TypeError, KeyError):
            return False
        row = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
        root_children = db.execute(
            "SELECT id FROM requests WHERE owner=? AND "
            "(json_extract(receipt,'$._supersedes_request_id')=? OR json_extract(receipt,'$._codex_fallback_of')=?)",
            (owner, original_id, original_id),
        ).fetchall()
        parent_children = db.execute(
            "SELECT id FROM requests WHERE owner=? AND "
            "(json_extract(receipt,'$._supersedes_request_id')=? OR json_extract(receipt,'$._codex_fallback_of')=?)",
            (owner, parent_id, parent_id),
        ).fetchall()
        expected_parent = {**original_body, "client_request_id": parent_id,
                           "supersedes_request_id": original_id}
        expected = {**deepcopy(parent_body), "client_request_id": request_id,
                    "supersedes_request_id": parent_id,
                    cls.MISSING_NODE_RETRY_FIELD: image_id}
        return bool(
            row
            and original.get("request_id") == original_id
            and original.get("_supersedes_request_id") is None
            and original.get("_supersedes_no_final") is not True
            and original.get("_derived_input") is None
            and cls._submission_identity(owner, original_body) == (original_id, original_row[0])
            and parent.get("request_id") == parent_id
            and parent_body == expected_parent
            and cls._submission_identity(owner, parent_body) == (parent_id, parent_row[0])
            and receipt.get("_supersedes_input_hash") == parent_row[0]
            and body == expected
            and cls._submission_identity(owner, body) == (request_id, row[0])
            and parent.get("_supersedes_request_message_id") == original.get("request_message_id")
            and parent.get("_submission_parent_message_id") == original_body.get("parent_message_id")
            and receipt.get("_supersedes_request_message_id") == parent.get("request_message_id")
            and receipt.get("_submission_parent_message_id") == parent_body.get("parent_message_id")
            and all(parent.get(key) == receipt.get(key) == original_body.get(key) for key in (
                "provider_binding_id", "provider_account_identity", "conversation_id", "client_conversation_id"))
            and len(root_children) == 1 and root_children[0][0] == parent_id
            and len(parent_children) == 1 and parent_children[0][0] == request_id
            # A4 is identified by its owner-local receipt and immutable input,
            # never by a later queued/running/terminal status.  This lets the
            # lifecycle retain the relation after A4 has been claimed while
            # still refusing a changed task or a same-id task from another owner.
            and image.get("id") == image_id and image.get("owner_id") == owner
            and isinstance(image_payload, dict) and isinstance(image_identity, dict)
            and image_body.get("mode") == image.get("mode")
            and image_identity.get("role") == "admin"
            and image_identity.get("_trusted_source") == image.get("_source")
            and image.get("request_hash") == _request_hash(image["mode"], image_payload)
            and image.get("_product_work_key") == original.get("_product_work_key") == parent.get("_product_work_key") == receipt.get("_product_work_key")
            and all(image.get(key) == original_body.get(key) for key in (
                "provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id"))
            and all(image_payload.get(key) == image.get(key) for key in (
                "provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id"))
            and all(type(item.get("_sequence")) is int for item in (original, parent, image, receipt))
            and original["_sequence"] < parent["_sequence"] < image["_sequence"] < receipt["_sequence"]
        )

    @classmethod
    def _no_final_model_rejection_retry(cls, store, db, owner, receipt, now):
        """Recognize an already-registered no-final model-retry lineage.

        This is intentionally structural rather than live eligibility.  Once C
        exists, a later result or claim on R/P must stop further fallbacks and
        missing-node resends; it must never make the consumed C look like a
        fresh standalone model rejection.
        """
        parent_id = receipt.get("_supersedes_request_id")
        request_id = receipt.get("request_id")
        if not isinstance(parent_id, str) or not parent_id or not isinstance(request_id, str) or not request_id:
            return False
        parent_row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                                (owner, parent_id)).fetchone()
        if not parent_row or receipt.get("_supersedes_input_hash") != parent_row[0]:
            return False
        try:
            parent = json.loads(parent_row[1])
        except (TypeError, ValueError):
            return False
        original_id = parent.get("_supersedes_request_id")
        if (parent.get("_supersedes_no_final") is not True
                or not isinstance(original_id, str) or not original_id):
            return False
        original_row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                                  (owner, original_id)).fetchone()
        if not original_row or parent.get("_supersedes_input_hash") != original_row[0]:
            return False
        try:
            original = json.loads(original_row[1])
            original_body = store.load_input(original["_input_ref"])
            parent_body = store.load_input(parent["_input_ref"])
            body = store.load_input(receipt["_input_ref"])
        except (OSError, ValueError, TypeError, KeyError):
            return False
        row = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
        expected_parent = {**original_body, "client_request_id": parent_id,
                           "supersedes_request_id": original_id, "continue_after_no_final": True}
        expected = {**deepcopy(parent_body), "client_request_id": request_id,
                    "supersedes_request_id": parent_id}
        expected.pop("continue_after_no_final", None)
        return bool(
            row
            and original.get("request_id") == original_id
            and original.get("_supersedes_request_id") is None
            and original.get("_derived_input") is None
            and cls._submission_identity(owner, original_body) == (original_id, original_row[0])
            and parent.get("request_id") == parent_id
            and parent_body == expected_parent
            and cls._submission_identity(owner, parent_body) == (parent_id, parent_row[0])
            and receipt.get("_supersedes_input_hash") == parent_row[0]
            and body == expected
            and cls._submission_identity(owner, body) == (request_id, row[0])
            # Do not use live cursor fields here. A late upstream result may
            # advance R/P after C was registered; these links are instead the
            # durable frozen inputs and their receipt-time submission roots.
            and parent.get("_supersedes_input_hash") == original_row[0]
            and parent.get("_supersedes_request_message_id") == original.get("request_message_id")
            and parent.get("_submission_parent_message_id") == original_body.get("parent_message_id")
            and receipt.get("_supersedes_request_message_id") == parent.get("request_message_id")
            and receipt.get("_submission_parent_message_id") == parent_body.get("parent_message_id")
            and all(parent.get(key) == receipt.get(key) == original_body.get(key) for key in (
                "provider_binding_id", "provider_account_identity", "conversation_id", "client_conversation_id"))
        )

    @classmethod
    def _native_no_final_successor_source(cls, store, db, owner, receipt, now, *, successor_id, successor_sequence):
        """Return the proved R -> P -> N chain eligible for one native N -> C continuation.

        ``N`` is already a submitted model-retry whose user node exists but
        has no final answer.  This is deliberately narrower than the
        structural lineage recognizer: registration and execution must still
        reject a late R result, a live N claim, or a second direct child.
        """
        if (not isinstance(successor_id, str) or not successor_id
                or receipt.get("_supersedes_no_final") is True
                or not cls._no_final_model_rejection_retry(store, db, owner, receipt, now)):
            return None
        parent_id = receipt.get("_supersedes_request_id")
        parent_row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, parent_id)).fetchone()
        if not parent_row:
            return None
        try:
            parent = json.loads(parent_row[0])
        except (TypeError, ValueError):
            return None
        if (not cls._rejected_no_final_successor_source(store, db, owner, parent, now)
                or any(parent.get(key) for key in ("_claim_id", "recovery_claim_id", "_recovery_claim_id",
                                                   "_work_key", "_completion", "_completion_of",
                                                   "_recovery_paused", "_recovery_suppressed"))
                or any(parent.get(key) is not None for key in ("_claim_until", "recovery_lease_until",
                                                               "_recovery_claim_until"))):
            return None
        parent_children = db.execute(
            "SELECT id FROM requests WHERE owner=? AND "
            "(json_extract(receipt,'$._supersedes_request_id')=? OR json_extract(receipt,'$._codex_fallback_of')=?)",
            (owner, parent_id, parent_id),
        ).fetchall()
        if len(parent_children) != 1 or parent_children[0][0] != receipt.get("request_id"):
            return None
        original_id = parent.get("_supersedes_request_id")
        original = store.read_receipt(db, "text", owner, original_id) if isinstance(original_id, str) else None
        if not original:
            return None
        if (receipt.get("status") != "failed"
                or receipt.get("error_code") != "RESULT_UNRECOVERABLE"
                or receipt.get("upstream_outcome") != "unknown"
                or receipt.get("recovery_reason") != "REQUEST_RESULT_NOT_FOUND"
                or not cls._no_final_successor_source(receipt, now)
                or receipt.get("_submission_started") is not True
                or receipt.get("_upstream_terminal")
                or receipt.get("_derived_input")
                or receipt.get("_public_session_ref") or receipt.get("_public_route")
                or receipt.get("_forward") or receipt.get("_forward_protocol")
                or receipt.get("_executing") is not False or receipt.get("_turn_reserved") is not False
                or any(key in receipt for key in ("content", "result", "data"))
                or receipt.get("result_file_ids") or receipt.get("result_sediment_ids")
                or any(receipt.get(key) for key in ("_work_key", "_completion", "_completion_of",
                                                    "_recovery_paused", "_recovery_suppressed"))):
            return None
        for claim_key, until_key in (("_claim_id", "_claim_until"), ("recovery_claim_id", "recovery_lease_until"),
                                     ("_recovery_claim_id", "_recovery_claim_until")):
            if receipt.get(claim_key) or receipt.get(until_key) is not None:
                return None
        root_parent = original.get("request_parent_message_id") or original.get("_submission_parent_message_id")
        native_parent = receipt.get("request_parent_message_id")
        if not all(isinstance(value, str) and value for value in (
                original.get("request_message_id"), root_parent, parent.get("request_message_id"),
                receipt.get("request_message_id"), native_parent)):
            return None
        children = db.execute(
            "SELECT id FROM requests WHERE owner=? AND "
            "(json_extract(receipt,'$._supersedes_request_id')=? OR json_extract(receipt,'$._codex_fallback_of')=?)",
            (owner, receipt["request_id"], receipt["request_id"]),
        ).fetchall()
        if children and (len(children) != 1 or children[0][0] != successor_id):
            return None
        sequence = (original.get("_sequence"), parent.get("_sequence"), receipt.get("_sequence"), successor_sequence)
        if (any(type(value) is not int or value <= 0 for value in sequence)
                or not (sequence[0] < sequence[1] < sequence[2] < sequence[3])):
            return None
        return {"original": original, "parent": parent}

    @classmethod
    def _rejected_successor_body(cls, original, retained, previous_id, request_id, client_conversation_id,
                                 *, model_rejection_retry=False, missing_node_retry=False,
                                 queued_image_id=None):
        """Freeze a Chat retry to the rejected original's exact identity/input."""
        if (not cls._rejected_chat_successor_source(
                    original, allow_no_final_successor=model_rejection_retry or missing_node_retry)
                or (not model_rejection_retry and not missing_node_retry and (retained.get("supersedes_request_id")
                                                    or retained.get("derived_input") is not None
                                                    or retained.get("continue_after_no_final") is not None))
                or (model_rejection_retry and retained.get("continue_after_no_final") is not True)
                or (missing_node_retry and (retained.get("derived_input") is not None
                                             or retained.get("continue_after_no_final") is not None
                                             or retained.get(cls.MISSING_NODE_RETRY_FIELD) is not None))
                or (missing_node_retry and not isinstance(queued_image_id, str))
                or retained.get("client_conversation_id") != client_conversation_id
                or original.get("client_conversation_id") != client_conversation_id):
            return None
        bound = {}
        for field in ("provider_binding_id", "provider_account_identity"):
            value = original.get(field)
            retained_value = retained.get(field)
            if not isinstance(value, str) or not value or retained_value not in (None, "", value):
                return None
            bound[field] = value
        conversation, parent = original.get("conversation_id"), original.get("parent_message_id")
        if conversation or parent:
            if (not isinstance(conversation, str) or not conversation or not isinstance(parent, str) or not parent
                    or retained.get("conversation_id") != conversation or retained.get("parent_message_id") != parent):
                return None
            bound.update(conversation_id=conversation, parent_message_id=parent)
        elif retained.get("conversation_id") not in {None, ""} or retained.get("parent_message_id") not in {None, ""}:
            return None
        frozen = deepcopy(retained)
        if model_rejection_retry:
            frozen.pop("continue_after_no_final", None)
        result = {**frozen, **bound, "client_request_id": request_id,
                  "supersedes_request_id": previous_id,
                  "client_conversation_id": client_conversation_id}
        if missing_node_retry:
            result[cls.MISSING_NODE_RETRY_FIELD] = queued_image_id
        return result

    def submission_input(self, owner, body):
        """Resolve an explicit internal reference; callers cannot replace original content."""
        if "supersedes_request_id" not in body:
            if "continue_after_no_final" in body:
                raise ConversationBindingError("no-final continuation requires an original request", code="CHAT_SUPERSEDE_INVALID")
            if "derived_input" in body:
                raise ConversationBindingError("derived input requires original request", code="CHAT_DERIVED_INPUT_INVALID")
            return body
        request_id = body.get("client_request_id")
        with self._db() as db:
            existing = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            conflict = "CONVERSATION_REQUEST_CONFLICT" if existing else "CHAT_SUPERSEDE_INVALID"
            def reject():
                raise ConversationBindingError("original successor input cannot be changed", code=conflict)
            missing_node_retry = set(body) == self.MISSING_NODE_RETRY_FIELDS
            rejected_minimal = set(body) == self.REJECTED_SUPERSEDE_FIELDS or missing_node_retry
            if rejected_minimal:
                if (not owner or any(not isinstance(body.get(key), str) or not body[key].strip()
                                     or body[key] != body[key].strip() for key in self.REJECTED_SUPERSEDE_FIELDS)
                        or len(request_id) > 200 or len(body["supersedes_request_id"]) > 200
                        or request_id == body["supersedes_request_id"]):
                    reject()
                original_row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                                          (owner, body["supersedes_request_id"])).fetchone()
                if not original_row:
                    reject()
                try:
                    if existing:
                        saved_receipt = json.loads(existing[1])
                        saved = self.store.load_input(saved_receipt["_input_ref"])
                        if (saved_receipt.get("_supersedes_request_id") != body["supersedes_request_id"]
                                or self._submission_identity(owner, saved) != (request_id, existing[0])
                                or saved.get("client_conversation_id") != body["client_conversation_id"]
                                # The image is part of C's immutable four-field
                                # identity.  A same ID is a read only when it
                                # names this exact frozen image continuation.
                                or bool(saved.get(self.MISSING_NODE_RETRY_FIELD)) != missing_node_retry
                                or (missing_node_retry and saved.get(self.MISSING_NODE_RETRY_FIELD)
                                    != body.get(self.MISSING_NODE_RETRY_FIELD))):
                            reject()
                        # A same-ID retry is a read of its frozen child. Its
                        # parent can legitimately become terminal after submit.
                        return saved
                    original = json.loads(original_row[1])
                    retained = self.store.load_input(original["_input_ref"])
                    if self._submission_identity(owner, retained) != (body["supersedes_request_id"], original_row[0]):
                        reject()
                    missing_node_parent = self._rejected_missing_node_successor_source(
                        self.store, db, owner, original, self._now())
                    # A typed rejection P after a missing R has exactly one
                    # continuation shape: C must bind its own queued image.
                    # Do not let that P re-enter the historic direct-rejection
                    # envelope with the fourth field silently omitted.
                    if missing_node_parent and not missing_node_retry:
                        reject()
                    direct_rejection = (not missing_node_retry
                                        and self._rejected_chat_successor_source(original)
                                        and not missing_node_parent)
                    model_rejection_retry = (not direct_rejection and self._rejected_no_final_successor_source(
                        self.store, db, owner, original, self._now()))
                    missing_node_model_retry = (not direct_rejection and not model_rejection_retry and missing_node_retry
                        and missing_node_parent)
                    if not (direct_rejection or model_rejection_retry or missing_node_model_retry):
                        reject()
                    # Conversely, an explicit image marker cannot be attached
                    # to an ordinary direct/no-final rejection: it has no A4
                    # ordering contract and must never be silently dropped.
                    if missing_node_retry and not missing_node_model_retry:
                        reject()
                    expected = self._rejected_successor_body(
                        original, retained, body["supersedes_request_id"], request_id, body["client_conversation_id"],
                        model_rejection_retry=model_rejection_retry,
                        missing_node_retry=missing_node_model_retry,
                        queued_image_id=body.get(self.MISSING_NODE_RETRY_FIELD),
                    )
                    if expected is None:
                        reject()
                except (OSError, ValueError, TypeError, KeyError):
                    reject()
                return expected
            derived = "derived_input" in body
            no_final = "continue_after_no_final" in body
            attributes_upgrade = body.get("derived_input") == {"kind": "attributes_required_only_v4"}
            if (set(body) != self.SUPERSEDE_FIELDS | ({"derived_input"} if derived else set()) | ({"continue_after_no_final"} if no_final else set()) or not owner
                    or no_final and (body["continue_after_no_final"] is not True or derived and not attributes_upgrade)
                    or attributes_upgrade and not no_final
                    or any(not isinstance(body.get(k), str) or not body[k].strip() or body[k] != body[k].strip() for k in self.SUPERSEDE_FIELDS)
                    or len(request_id) > 200 or len(body["supersedes_request_id"]) > 200
                    or request_id == body["supersedes_request_id"]):
                reject()
            row = existing or db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                                         (owner, body["supersedes_request_id"])).fetchone()
            if not row:
                reject()
            receipt = json.loads(row[1])
            # A typed pre-conversation model rejection has exactly one narrow
            # envelope: clients cannot restate even matching bound identity.
            if self._rejected_chat_successor_source(receipt):
                reject()
            try:
                retained = self.store.load_input(receipt["_input_ref"])
                if self._submission_identity(owner, retained) != (receipt["request_id"], row[0]):
                    reject()
            except (OSError, ValueError, TypeError, KeyError):
                reject()
            expected = {**retained, "client_request_id": request_id, "supersedes_request_id": body["supersedes_request_id"]}
            if no_final and not existing:
                expected["continue_after_no_final"] = True
            if derived and not existing:
                from services.category_directory_derivation import derive_retained_input
                expected, _ = derive_retained_input(retained, request_id, body["supersedes_request_id"], body["derived_input"],
                                                    no_final=no_final)
            if existing and retained.get("supersedes_request_id") != body["supersedes_request_id"]:
                reject()
            if existing and ("derived_input" in retained) != derived:
                reject()
            if existing and ("continue_after_no_final" in retained) != no_final:
                reject()
            if any(expected.get(k) != v for k, v in body.items()):
                reject()
            return expected

    @classmethod
    def _validate_reconciled_queued_successor(cls, store, db, owner, successor, previous, previous_hash, original_body, now):
        """Revalidate the frozen, pre-accepted queued child at both send guards."""
        def reject():
            raise ConversationBindingError("queued successor reconciliation is no longer valid",
                                           code="CHAT_QUEUED_SUCCESSOR_NOT_RECONCILABLE")
        request_id = successor.get("request_id")
        row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                         (owner, request_id)).fetchone()
        if not row:
            reject()
        try:
            stored = json.loads(row[1])
            body = store.load_input(successor.get("_input_ref"))
        except (OSError, ValueError, TypeError, KeyError):
            reject()
        if (stored.get("request_id") != request_id
                or cls._submission_identity(owner, body) != (request_id, row[0])
                or cls._submission_identity(owner, original_body) != (previous.get("request_id"), previous_hash)
                or body != {**original_body, "client_request_id": request_id}
                or "supersedes_request_id" in body):
            reject()
        required = ("provider_binding_id", "provider_account_identity", "client_conversation_id",
                    "conversation_id", "_source", "_scheduling_owner", "model")
        if any(not previous.get(key) or successor.get(key) != previous.get(key) for key in required):
            reject()
        if previous.get("_source") != "internal:listing" or previous.get("_scheduling_owner") != "internal:listing":
            reject()
        sequence = (previous.get("_sequence"), successor.get("_sequence"))
        if (any(type(value) is not int or value <= 0 for value in sequence)
                or not sequence[0] < sequence[1]):
            reject()
        if (previous.get("_route", "chat") != "chat" or previous.get("_operation", "text") != "text"
                or previous.get("_forward_protocol") or previous.get("_public_session_ref")
                or previous.get("_expected_sends") != 1
                or previous.get("status") != "failed" or previous.get("error_code") != "RESULT_UNRECOVERABLE"
                or previous.get("upstream_outcome") != "unknown"
                or previous.get("recovery_reason") != "REQUEST_MESSAGE_NOT_FOUND"
                or previous.get("_attempt_reason") != "ORIGINAL_REQUEST_MESSAGE_NOT_FOUND"
                or previous.get("_submission_started") is not True
                or type(previous.get("_attempt_finished_at")) not in (int, float)
                or not math.isfinite(previous["_attempt_finished_at"])
                or type(previous.get("_execution_wait_ended_at")) not in (int, float)
                or not math.isfinite(previous["_execution_wait_ended_at"])
                or previous.get("_executing") or previous.get("_turn_reserved")
                or previous.get("_upstream_terminal") or previous.get("_recovery_paused")
                or previous.get("_recovery_suppressed") or previous.get("_work_key")
                or previous.get("_completion") or previous.get("_completion_of")
                or previous.get("_supersedes_request_id")
                or any(previous.get(key) is not None for key in ("content", "result", "data"))):
            reject()
        for claim, until in (("_claim_id", "_claim_until"), ("recovery_claim_id", "recovery_lease_until"),
                             ("_recovery_claim_id", "_recovery_claim_until")):
            if previous.get(claim) or previous.get(until) is not None:
                reject()

    @staticmethod
    def _supersede_order_link(previous, successor, *, rejected_model=None):
        """A server-registered successor releases only its own predecessor's order head."""
        if rejected_model is None:
            rejected_model = (TextTaskService._rejected_chat_successor_source(previous)
                              or previous.get("_supersedes_no_final") is True
                              and TextTaskService._rejected_chat_successor_source(
                                  previous, allow_no_final_successor=True))
        audit = successor.get("_derived_input") or {}
        attributes_upgrade = isinstance(audit, dict) and audit.get("kind") == "attributes_required_only_v4"
        model_upgrade = bool(attributes_upgrade
            and successor.get("_supersedes_no_final") is True
            and previous and previous.get("model") == audit.get("original_model") == "gpt-5-6-instant"
            and audit.get("original_thinking_effort") == "standard"
            and successor.get("model") == audit.get("model") == "gpt-5-6-thinking"
            and audit.get("thinking_effort") == "high")
        shared = bool(previous and successor.get("_supersedes_request_id") == previous.get("request_id")
            and successor.get("request_id") != previous.get("request_id")
            and successor.get("_supersedes_input_hash") and successor.get("_input_ref")
            and not previous.get("_public_session_ref") and not successor.get("_public_session_ref")
            and successor.get("_route", "chat") == previous.get("_route", "chat") == "chat"
            and successor.get("_operation", "text") == previous.get("_operation", "text") == "text"
            and not successor.get("_forward_protocol") and not previous.get("_forward_protocol")
            and (model_upgrade if attributes_upgrade else successor.get("model") == previous.get("model")))
        if not shared:
            return False
        if rejected_model:
            shared_binding = all(successor.get(k) == previous.get(k) and previous.get(k) for k in (
                "provider_binding_id", "provider_account_identity", "client_conversation_id"))
            if not shared_binding:
                return False
            if not previous.get("conversation_id") and not previous.get("parent_message_id"):
                return bool(not successor.get("conversation_id") and not successor.get("parent_message_id")
                            and not successor.get("_supersedes_request_message_id")
                            and not successor.get("_submission_parent_message_id"))
            return bool(successor.get("_supersedes_request_message_id") == previous.get("request_message_id")
                        and all(successor.get(k) == previous.get(k) and previous.get(k) for k in (
                            "conversation_id",))
                        and successor.get("_submission_parent_message_id") == previous.get("parent_message_id"))
        return bool(successor.get("_supersedes_request_message_id") == previous.get("request_message_id")
                    and all(successor.get(k) == previous.get(k) and previous.get(k) for k in (
                        "provider_binding_id", "provider_account_identity", "conversation_id", "client_conversation_id"))
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
        direct_rejected_model = TextTaskService._rejected_chat_successor_source(previous)
        model_rejection_retry = (not direct_rejected_model
                                  and TextTaskService._rejected_no_final_successor_source(
                                      store, db, owner, previous, now))
        missing_node_model_retry = (not direct_rejected_model and not model_rejection_retry
            and successor.get(TextTaskService.MISSING_NODE_RETRY_FIELD) is not None
            and TextTaskService._rejected_missing_node_successor_source(store, db, owner, previous, now))
        native_no_final_retry = (successor.get("_supersedes_no_final") is True
            and TextTaskService._native_no_final_successor_source(
                store, db, owner, previous, now, successor_id=successor.get("request_id"),
                successor_sequence=successor.get("_sequence")))
        rejected_model = direct_rejected_model or model_rejection_retry or missing_node_model_retry
        if (not rejected_model and (not isinstance(previous.get("request_message_id"), str)
                                    or not previous["request_message_id"])
                or not TextTaskService._supersede_order_link(previous, successor, rejected_model=rejected_model)):
            reject("CHAT_SUPERSEDE_INVALID")
        for claim_key, until_key in (("_claim_id", "_claim_until"), ("recovery_claim_id", "recovery_lease_until")):
            until = previous.get(until_key)
            # A terminal typed rejection retains the scheduler claim audit
            # fields until admission cleanup. It is retired once neither
            # execution nor a turn is reserved; recovery leases remain live.
            retired_terminal_claim = (rejected_model and claim_key == "_claim_id"
                                      and previous.get("status") == "failed"
                                      and previous.get("_executing") is False
                                      and previous.get("_turn_reserved") is False)
            if (not retired_terminal_claim and previous.get(claim_key)
                    and (type(until) not in (int, float) or not math.isfinite(until) or until > now)):
                reject("CHAT_SUPERSEDE_PREDECESSOR_BUSY")
        if previous.get("_executing") or previous.get("_turn_reserved"):
            reject("CHAT_SUPERSEDE_PREDECESSOR_BUSY")
        if (not rejected_model and not native_no_final_retry and (previous.get("status") != "failed" or previous.get("error_code") != "RESULT_UNRECOVERABLE"
                or previous.get("upstream_outcome") != "unknown"
                or (not TextTaskService._no_final_successor_source(previous, now)
                    if successor.get("_supersedes_no_final") else previous.get("recovery_reason") != "REQUEST_MESSAGE_NOT_FOUND")
                or previous.get("_recovery_suppressed") or previous.get("_supersedes_request_id")
                or type(previous.get("_execution_wait_ended_at")) not in (int, float)
                or not math.isfinite(previous["_execution_wait_ended_at"]))):
            reject("CHAT_SUPERSEDE_INVALID")
        if ((rejected_model or successor.get("_supersedes_no_final") is True)
                and account_service.bound_text_recovery_auth_unavailable(
                    previous.get("provider_binding_id", ""))):
            # This is a local, read-only account-state check.  A stopped
            # original branch is not authority to reuse an invalidated bound
            # account or create a successor that cannot pass token admission.
            reject("CHAT_SUPERSEDE_AUTH_REQUIRED")
        if row[0] != successor.get("_supersedes_input_hash"):
            reject("CHAT_SUPERSEDE_INVALID")
        if (successor.get(TextTaskService.MISSING_NODE_RETRY_FIELD) is not None
                and not TextTaskService._missing_node_model_rejection_retry(store, db, owner, successor, now)):
            reject("CHAT_SUPERSEDE_INVALID")
        try:
            original_body = store.load_input(previous["_input_ref"])
            identity = TextTaskService._submission_identity(owner, original_body)
        except (OSError, ValueError, TypeError, KeyError):
            reject("CHAT_SUPERSEDE_INVALID")
        if identity != (previous_id, row[0]):
            reject("CHAT_SUPERSEDE_INVALID")
        if successor.get(TextTaskService._RECONCILED_QUEUED_SUCCESSOR) is True:
            TextTaskService._validate_reconciled_queued_successor(
                store, db, owner, successor, previous, row[0], original_body, now,
            )
        elif TextTaskService._RECONCILED_QUEUED_SUCCESSOR in successor:
            reject("CHAT_SUPERSEDE_INVALID")
        if successor.get("_derived_input"):
            attributes_upgrade = successor["_derived_input"].get("kind") == "attributes_required_only_v4"
            if attributes_upgrade:
                if successor.get("_supersedes_no_final") is not True or not TextTaskService._no_final_successor_source(previous, now):
                    reject("CHAT_DERIVED_INPUT_INVALID")
            elif (previous.get("original_http_status") != 413
                    or previous.get("original_failure_phase") != "stream_open"
                    or previous.get("original_upstream_request_stage") != "conversation"
                    or previous.get("original_exception_category") != "http"
                    or previous.get("_submission_started") is not True):
                reject("CHAT_DERIVED_INPUT_INVALID")
            from services.category_directory_derivation import derive_retained_input
            derived_body, audit = derive_retained_input(original_body, successor["request_id"], previous_id,
                                                       {"kind": successor["_derived_input"].get("kind")},
                                                       no_final=successor.get("_supersedes_no_final") is True)
            if successor["_derived_input"] != {**audit, "original_input_hash": row[0]}:
                reject("CHAT_DERIVED_INPUT_INVALID")
            saved = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", (owner, successor["request_id"])).fetchone()
            if saved and TextTaskService._submission_identity(owner, derived_body)[1] != saved[0]:
                reject("CHAT_DERIVED_INPUT_INVALID")
        if rejected_model:
            expected = TextTaskService._rejected_successor_body(previous, original_body, previous_id,
                                                                 successor.get("request_id"), successor.get("client_conversation_id"),
                                                                 model_rejection_retry=model_rejection_retry,
                                                                 missing_node_retry=missing_node_model_retry,
                                                                 queued_image_id=successor.get(TextTaskService.MISSING_NODE_RETRY_FIELD))
            if expected is None:
                reject("CHAT_SUPERSEDE_INVALID")
            if isinstance(successor.get("_input_ref"), str):
                try:
                    successor_body = store.load_input(successor["_input_ref"])
                except (OSError, ValueError, TypeError, KeyError):
                    reject("CHAT_SUPERSEDE_INVALID")
                if successor_body != expected:
                    reject("CHAT_SUPERSEDE_INVALID")
            elif any(successor.get(key) != expected.get(key) for key in (
                    "provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id",
                    "client_conversation_id")):
                reject("CHAT_SUPERSEDE_INVALID")
        else:
            # The submission root is stable even when a result advances the live cursor.
            if any(original_body.get(k) != successor.get(k) for k in (
                    "provider_binding_id", "provider_account_identity", "conversation_id", "client_conversation_id")):
                reject("CHAT_SUPERSEDE_INVALID")
            if original_body.get("parent_message_id") != successor.get("_submission_parent_message_id"):
                reject("CHAT_SUPERSEDE_INVALID")
        other = db.execute("SELECT id FROM requests WHERE owner=? AND id<>? AND (json_extract(receipt,'$._supersedes_request_id')=? OR json_extract(receipt,'$._codex_fallback_of')=?) LIMIT 1",
                           (owner, successor["request_id"], previous_id, previous_id)).fetchone()
        if other:
            reject("CHAT_SUPERSEDE_CONFLICT")
        return previous

    def reconcile_queued_successor(self, owner, request_id, original_request_id, cursor, *, source, dry_run=True):
        """Attach one proven, accepted-but-unsent listing receipt to its original.

        The retained child input is never rebuilt.  This narrow repair exists for
        an internal listing grant that was accepted before its Provider lineage
        metadata was recorded; normal callers must use ``submit`` instead.
        """
        def reject(code="CHAT_QUEUED_SUCCESSOR_NOT_RECONCILABLE"):
            raise ConversationBindingError("queued successor cannot be reconciled", code=code)
        if source != "internal:listing" or type(dry_run) is not bool:
            reject()
        fields = ("provider_binding_id", "provider_account_identity", "client_conversation_id",
                  "conversation_id", "parent_message_id")
        if (not isinstance(original_request_id, str) or not original_request_id
                or original_request_id == request_id or set(cursor) != set(fields)
                or any(not isinstance(cursor.get(key), str) or not cursor[key].strip() for key in fields)):
            reject()
        with self._db() as db:
            if dry_run:
                db.execute("BEGIN")
            else:
                db.execute("BEGIN IMMEDIATE")
            try:
                child_row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                                       (owner, request_id)).fetchone()
                parent_row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                                        (owner, original_request_id)).fetchone()
                if not child_row or not parent_row:
                    reject()
                child = json.loads(child_row[1])
                parent = json.loads(parent_row[1])
                existing = child.get(self._RECONCILED_QUEUED_SUCCESSOR)
                if existing is True:
                    # A lost HTTP acknowledgement may be retried after this
                    # queued child has sent and its live receipt cursor has
                    # advanced.  Verify the retained immutable inputs and
                    # the relation that was already committed, rather than
                    # treating the acknowledgement as a new reconciliation.
                    try:
                        parent_body = self.store.load_input(parent.get("_input_ref"))
                        child_body = self.store.load_input(child.get("_input_ref"))
                    except (OSError, ValueError, TypeError, KeyError):
                        reject("CHAT_QUEUED_SUCCESSOR_CONFLICT")
                    if (child.get("request_id") != request_id or parent.get("request_id") != original_request_id
                            or self._submission_identity(owner, parent_body) != (original_request_id, parent_row[0])
                            or self._submission_identity(owner, child_body) != (request_id, child_row[0])
                            or child_body != {**parent_body, "client_request_id": request_id}
                            or "supersedes_request_id" in child_body
                            or any(parent_body.get(key) != cursor[key] or child_body.get(key) != cursor[key]
                                   for key in fields)
                            or child.get("_supersedes_request_id") != original_request_id
                            or child.get("_supersedes_input_hash") != parent_row[0]
                            or child.get("_supersedes_request_message_id") != parent.get("request_message_id")
                            or child.get("_submission_parent_message_id") != parent_body.get("parent_message_id")):
                        reject("CHAT_QUEUED_SUCCESSOR_CONFLICT")
                    result = {"request_id": request_id, "original_request_id": original_request_id,
                              "type": "queued_successor_reconciliation", "eligible": True,
                              "reconciled": True, "dry_run": dry_run}
                    db.execute("ROLLBACK")
                    return result
                if existing is not None or any(child.get(key) for key in self._RECONCILE_LINK_FIELDS):
                    reject("CHAT_QUEUED_SUCCESSOR_CONFLICT")
                if any(child.get(key) != cursor[key] or parent.get(key) != cursor[key] for key in fields):
                    reject()
                if (child.get("request_id") != request_id or parent.get("request_id") != original_request_id
                        or child.get("status") != "queued" or child.get("_submission_started") is not False
                        or child.get("_executing") not in (None, False) or child.get("_turn_reserved") is not False
                        or child.get("_last_sent_sequence") is not None
                        or child.get("send_count") is not None and (
                            type(child.get("send_count")) is not int or child["send_count"] != 0)
                        or child.get("error_code") is not None or child.get("upstream_outcome") not in (None, "")
                        or child.get("upstream_unfinished") not in (None, False)
                        or any(child.get(key) is not None for key in ("content", "result", "data"))
                        or any(child.get(key) for key in ("_upstream_terminal", "_recovery_paused", "_recovery_suppressed"))):
                    reject()
                timeline = child.get("_execution_timeline")
                sent_stages = {"send_guard_passed", "send_call_started", "response_headers_received", "first_output", "task_finished"}
                if (not isinstance(timeline, list) or len(timeline) != 1
                        or not isinstance(timeline[0], dict) or timeline[0].get("stage") != "accepted"
                        or any(not isinstance(item, dict) or item.get("stage") in sent_stages for item in timeline)):
                    reject()
                if any(child.get(key) is not None for key in (
                    "_attempt_finished_at", "_execution_wait_ended_at", "recovery_next_at", "recovery_lease_until",
                    "recovery_reason", "recovery_phase", "recovery_error_code", "recovery_last_read_error",
                    "recovery_retry_after_seconds",
                )):
                    reject()
                for claim, until in (("_claim_id", "_claim_until"), ("recovery_claim_id", "recovery_lease_until"),
                                     ("_recovery_claim_id", "_recovery_claim_until")):
                    if child.get(claim) or child.get(until) is not None:
                        reject()
                if (any(not parent.get(key) or child.get(key) != parent.get(key) for key in (
                        "provider_binding_id", "provider_account_identity", "client_conversation_id",
                        "conversation_id", "_source", "_scheduling_owner", "model"))):
                    reject()
                if parent.get("_source") != source or parent.get("_scheduling_owner") != source:
                    reject()
                sequence = (parent.get("_sequence"), child.get("_sequence"))
                if (any(type(value) is not int or value <= 0 for value in sequence)
                        or not sequence[0] < sequence[1]):
                    reject()
                children = db.execute(
                    "SELECT id FROM requests WHERE owner=? AND "
                    "(json_extract(receipt,'$._supersedes_request_id')=? OR json_extract(receipt,'$._codex_fallback_of')=?) LIMIT 1",
                    (owner, original_request_id, original_request_id),
                ).fetchone()
                child_children = db.execute(
                    "SELECT 1 FROM requests WHERE owner=? AND "
                    "(json_extract(receipt,'$._supersedes_request_id')=? OR json_extract(receipt,'$._codex_fallback_of')=?) LIMIT 1",
                    (owner, request_id, request_id),
                ).fetchone()
                if children or child_children:
                    reject("CHAT_QUEUED_SUCCESSOR_CONFLICT")
                from services.pool_admission import physical_conversation_key, unfinished
                physical = physical_conversation_key(parent)
                if not physical:
                    reject()
                for kind, other_owner, other_id, other in self.store.receipts(db):
                    if (other_owner == owner and other_id not in {request_id, original_request_id}
                            and physical_conversation_key(other) == physical and unfinished(kind, other)):
                        reject("CHAT_QUEUED_SUCCESSOR_CONFLICT")
                candidate = {
                    **child,
                    "_supersedes_request_id": original_request_id,
                    "_supersedes_input_hash": parent_row[0],
                    "_supersedes_request_message_id": parent.get("request_message_id"),
                    "_submission_parent_message_id": parent.get("_submission_parent_message_id") or parent.get("parent_message_id"),
                    self._RECONCILED_QUEUED_SUCCESSOR: True,
                }
                self._validate_supersede(self.store, db, owner, candidate, self._now())
                result = {"request_id": request_id, "original_request_id": original_request_id,
                          "type": "queued_successor_reconciliation", "eligible": True,
                          "reconciled": not dry_run, "dry_run": dry_run}
                if dry_run:
                    db.execute("ROLLBACK")
                    return result
                candidate["updated_at"] = self._now()
                self.store.write_receipt(db, "text", owner, request_id, candidate)
            except Exception:
                db.execute("ROLLBACK")
                raise
        if self.admission is not None:
            self.admission.wake()
        return result

    @staticmethod
    def _no_final_successor_source(previous, now):
        """Local eligibility only; the original branch is checked again before send."""
        ended, reads = previous.get("_attempt_finished_at"), previous.get("recovery_no_result_reads")
        cursor = previous.get("_retry_cursor")
        if cursor is not None:
            # The bounded original-read path keeps an exact empty-branch cursor
            # as evidence for a later explicit recheck. It has no active
            # completion owner, but only the text proof for this exact receipt
            # can coexist with an explicit no-final successor.
            from services.generation_completion import retry_evidence
            proof = retry_evidence(previous)
            if proof is None or "source" in proof:
                return False
        reason = previous.get("recovery_reason")
        read_error = safe_recovery_read_error(previous.get("recovery_last_read_error"))
        error_code = previous.get("recovery_error_code")
        latest_read_failure = (
            reason is None
            and error_code in {"RECOVERY_TRANSPORT_FAILED", "RECOVERY_RATE_LIMITED",
                               "RECOVERY_AUTH_REQUIRED", "RECOVERY_READ_FAILED"}
            and (read_error is not None or error_code == "RECOVERY_READ_FAILED")
        )
        # A stopped no-final proof survives an unavailable explicit recheck.
        # Old receipts recorded generic read failures before the typed detail
        # field existed; their exact stopped proof remains usable. New generic
        # failures always persist that detail above. A fresh, validated branch
        # reason still wins either form, and the binding service rechecks the
        # original branch again before any send.
        # A fresh, validated recovery reason is evidence about the branch and
        # must still supersede that historical local qualification.
        return bool(previous.get("_attempt_reason") == "ORIGINAL_RESULT_NO_FINAL"
            and (reason == "REQUEST_RESULT_NOT_FOUND" or latest_read_failure)
            and type(ended) in (int, float) and math.isfinite(ended) and 0 < ended <= now
            and type(reads) is int and reads >= 3
            and not any(previous.get(k) for k in ("_work_key", "_completion", "_completion_of",
                "_recovery_paused", "_recovery_suppressed")))

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
        direct_rejected_model = self._rejected_chat_successor_source(original)
        model_rejection_retry = (not direct_rejected_model and self._rejected_no_final_successor_source(
            self.store, db, owner, original, self._now()))
        missing_node_model_retry = (not direct_rejected_model and not model_rejection_retry
            and self.MISSING_NODE_RETRY_FIELD in body
            and self._rejected_missing_node_successor_source(self.store, db, owner, original, self._now()))
        native_sequence = None
        native_no_final_source = None
        if body.get("continue_after_no_final") is True:
            acceptance_sequence = self.store.runtime(db, "acceptance_sequence", 0)
            native_sequence = (acceptance_sequence + 1
                               if type(acceptance_sequence) is int and acceptance_sequence >= 0 else None)
            native_no_final_source = self._native_no_final_successor_source(
                self.store, db, owner, original, self._now(), successor_id=body.get("client_request_id"),
                successor_sequence=native_sequence)
        native_no_final_retry = bool(native_no_final_source)
        if native_no_final_retry:
            # The transaction owns the next sequence; retain this exact value
            # for the candidate revalidation and final insert below.
            receipt["_sequence"] = native_sequence
        rejected_model = direct_rejected_model or model_rejection_retry or missing_node_model_retry
        if rejected_model:
            try:
                retained = self.store.load_input(original["_input_ref"])
            except (OSError, ValueError, TypeError, KeyError):
                reject("CHAT_SUPERSEDE_INVALID")
            expected = self._rejected_successor_body(
                original, retained, previous_id, body["client_request_id"], body["client_conversation_id"],
                model_rejection_retry=model_rejection_retry,
                missing_node_retry=missing_node_model_retry,
                queued_image_id=body.get(self.MISSING_NODE_RETRY_FIELD),
            )
            if expected is None or body != expected:
                reject("CHAT_SUPERSEDE_INVALID")
        elif native_no_final_retry:
            try:
                retained = self.store.load_input(original["_input_ref"])
            except (OSError, ValueError, TypeError, KeyError):
                reject("CHAT_SUPERSEDE_INVALID")
            expected = {**retained, "client_request_id": body["client_request_id"],
                        "supersedes_request_id": previous_id, "continue_after_no_final": True}
            if body != expected:
                reject("CHAT_SUPERSEDE_INVALID")
        elif "derived_input" in body:
            from services.category_directory_derivation import derive_retained_input
            try:
                retained = self.store.load_input(original["_input_ref"])
                expected, audit = derive_retained_input(retained, body["client_request_id"], previous_id, body["derived_input"],
                                                       no_final=body.get("continue_after_no_final") is True)
            except (OSError, ValueError, TypeError, KeyError):
                reject("CHAT_DERIVED_INPUT_INVALID")
            if self._submission_identity(owner, expected) != self._submission_identity(owner, body):
                reject("CHAT_DERIVED_INPUT_INVALID")
            receipt["_derived_input"] = {**audit, "original_input_hash": row[0]}
        else:
            comparison = {k: v for k, v in body.items()
                          if k not in {"supersedes_request_id", "continue_after_no_final"}}
            comparison["client_request_id"] = previous_id
            if self._submission_identity(owner, comparison) != (previous_id, row[0]):
                reject("CHAT_SUPERSEDE_INVALID")
        receipt.update({k: body.get(k) for k in ("provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id")})
        receipt.update(_supersedes_request_id=previous_id, _supersedes_input_hash=row[0])
        if missing_node_model_retry:
            receipt[self.MISSING_NODE_RETRY_FIELD] = body[self.MISSING_NODE_RETRY_FIELD]
        if rejected_model:
            if original.get("conversation_id"):
                receipt.update(_supersedes_request_message_id=original.get("request_message_id"),
                               _submission_parent_message_id=body.get("parent_message_id"))
        else:
            receipt.update(_supersedes_request_message_id=original.get("request_message_id"),
                           _submission_parent_message_id=body.get("parent_message_id"))
        if body.get("continue_after_no_final") is True:
            receipt["_supersedes_no_final"] = True
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
                # The rejected parent is the only registered no-final child
                # of this exact original.  Its model retry cannot bypass an
                # unrelated unfinished turn, but the retired original itself
                # remains an UNKNOWN audit record and must not block this one
                # bounded retry.
                if ((model_rejection_retry or missing_node_model_retry) and kind == "text"
                        and other_id == original.get("_supersedes_request_id")):
                    continue
                if (native_no_final_retry and kind == "text" and other_id in {
                        original.get("_supersedes_request_id"),
                        native_no_final_source["parent"].get("_supersedes_request_id"),
                }):
                    continue
                if (missing_node_model_retry and kind == "image"
                        and other_id == receipt.get(self.MISSING_NODE_RETRY_FIELD)):
                    continue
                if kind == "text" and self._completed_historical_successor(db, owner, other_id, other, original, self._now()):
                    continue
                reject("CHAT_SUPERSEDE_CONFLICT")
        return native_sequence if native_no_final_retry else None

    @classmethod
    def _completed_historical_successor(cls, db, owner, previous_id, previous, original, now):
        """A completed registered child already released this historical head.

        Keep the old UNKNOWN unchanged, just as the admission planner does;
        accepting a later explicit successor must not revive the old barrier.
        Here require a terminal child and older ordering, not merely a claim.
        """
        if (previous.get("request_id") != previous_id
                or previous.get("status") != "failed" or previous.get("error_code") != "RESULT_UNRECOVERABLE"
                or previous.get("upstream_outcome") != "unknown"
                or previous.get("recovery_reason") != "REQUEST_MESSAGE_NOT_FOUND"):
            return False
        rows = db.execute("SELECT id,receipt FROM requests WHERE owner=? AND "
                          "json_extract(receipt,'$._supersedes_request_id')=? LIMIT 2",
                          (owner, previous_id)).fetchall()
        if len(rows) != 1:
            return False
        child = json.loads(rows[0][1])
        if (child.get("request_id") != rows[0][0]
                or child.get("status") != "succeeded" or child.get("upstream_outcome") != "completed"
                or child.get("_upstream_terminal") is not True or not cls._supersede_order_link(previous, child)):
            return False
        sequence = [r.get("_sequence") for r in (previous, child, original)]
        if not all(type(v) is int and v > 0 for v in sequence) or not sequence[0] < sequence[1] < sequence[2]:
            return False
        if any(not original.get(k) or original[k] != child.get(k) for k in
               ("provider_binding_id", "provider_account_identity", "conversation_id", "client_conversation_id")):
            return False
        for item in (previous, child):
            if any(item.get(k) for k in ("_executing", "_turn_reserved", "upstream_unfinished",
                    "_recovery_paused", "_recovery_suppressed", "_work_key", "_completion", "_completion_of", "_retry_cursor")):
                return False
            for claim, until in (("_claim_id", "_claim_until"), ("recovery_claim_id", "recovery_lease_until"),
                                 ("_recovery_claim_id", "_recovery_claim_until")):
                expiry = item.get(until)
                if (expiry is not None and (type(expiry) not in (int, float) or not math.isfinite(expiry) or expiry > now)
                        or item.get(claim) and expiry is None):
                    return False
        old = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?",
                         (owner, previous_id)).fetchone()
        return bool(old and child.get("_supersedes_input_hash") == old[0])

    def validate_submission(self, owner: str, body: dict, *, source=None):
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
        if previous:
            saved = json.loads(previous[1])
            if saved.get("_product_work_key") and source is not None and saved.get("_source") != source:
                from services.work_lifecycle import WorkLifecycleError
                raise WorkLifecycleError("PRODUCT_WORK_SOURCE_CONFLICT", 403)
            return self._public(saved)
        return None

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

    def submit(self, owner: str, body: dict, *, source: str | None = None, _resume_absent_successor: bool = False):
        from services.product_work import locking_product_conversations
        body = self.submission_input(owner, body)
        with locking_product_conversations([body] if body.get("product_work") else []):
            return self._submit_prepared(owner, body, source=source,
                                         _resume_absent_successor=_resume_absent_successor)

    def submit_codex_fallback(self, owner, original_id, client_request_id, model, account_ref, reasoning_effort=None, *, source=None):
        """Create one native Responses successor from an immutable rejected Chat input."""
        if self.admission is None:
            from services.work_lifecycle import WorkLifecycleError
            raise WorkLifecycleError("CODEX_FALLBACK_ADMISSION_REQUIRED", 503)
        from services.codex_text_fallback import build_body
        with self._db() as db:
            body = build_body(self.store, db, owner, original_id, client_request_id, model, account_ref, reasoning_effort)
        return self._submit_prepared(owner, body, source=source)

    def codex_fallback_options(self, owner, original_id):
        """Read validated fallback eligibility and the already-cached Codex catalog."""
        if self.admission is None:
            from services.work_lifecycle import WorkLifecycleError
            raise WorkLifecycleError("CODEX_FALLBACK_ADMISSION_REQUIRED", 503)
        from services.codex_text_fallback import fallback_options_context
        with self._db() as db:
            existing = fallback_options_context(self.store, db, owner, original_id)
        if existing:
            return {"original_request_id": original_id, "eligible": False,
                    "reason": "CODEX_FALLBACK_ALREADY_CREATED", "existing_request_id": existing, "items": []}
        catalog = self.admission.codex.management_models()
        items = []
        for item in catalog.get("items", []):
            if not isinstance(item, dict) or item.get("route") != "codex":
                continue
            accounts = []
            for account in item.get("accounts", []):
                if not isinstance(account, dict) or not isinstance(account.get("account_ref"), str):
                    continue
                projected = {key: account[key] for key in ("account_ref", "state", "reason", "quota_bucket") if key in account}
                if "dispatchable_now" in account:
                    projected["dispatchable_now"] = account["dispatchable_now"]
                accounts.append(projected)
            items.append({key: item[key] for key in ("id", "route", "capabilities", "state", "reasoning_efforts") if key in item}
                         | {"accounts": accounts})
        return {"original_request_id": original_id, "eligible": True, "items": items}

    def submit_codex_continue(self, owner, parent_id, client_request_id, messages, step_ref=None, *, source=None):
        if self.admission is None:
            from services.work_lifecycle import WorkLifecycleError
            raise WorkLifecycleError("CODEX_FALLBACK_ADMISSION_REQUIRED", 503)
        from services.codex_text_fallback import build_continue_body
        with self._db() as db:
            body = build_continue_body(self.store, db, owner, parent_id, client_request_id, messages, step_ref)
        return self._submit_prepared(owner, body, source=source)

    def submit_codex_rejected_retry(self, owner, parent_id, client_request_id, *, source=None):
        """Durably register the one explicit retry of an empty-wire Codex rejection."""
        if self.admission is None:
            from services.work_lifecycle import WorkLifecycleError
            raise WorkLifecycleError("CODEX_FALLBACK_ADMISSION_REQUIRED", 503)
        from services.codex_text_fallback import build_rejected_retry_body
        with self._db() as db:
            body = build_rejected_retry_body(self.store, db, owner, parent_id, client_request_id)
        return self._submit_prepared(owner, body, source=source)

    def _submit_prepared(self, owner, body, *, source=None, _resume_absent_successor=False):
        from services.durable_forward import dispatch_model
        from services.workflow_scheduling import normalize_scheduling
        if "_scheduling" in body:
            body = {**body, "_scheduling": normalize_scheduling(body["_scheduling"])}
        if body.get("_scheduling") is not None and self.admission is None:
            raise ValueError("SCHEDULING_UNAVAILABLE")
        codex_markers = (
            body.get("_codex_fallback_of"), body.get("_codex_continue_of"), body.get("_codex_rejected_retry_of"),
        )
        if any(codex_markers) and self.admission is None:
            from services.work_lifecycle import WorkLifecycleError
            raise WorkLifecycleError("CODEX_FALLBACK_ADMISSION_REQUIRED", 503)
        codex_product_bridge = (
            body.get("_route") == "codex"
            and ((body.get("_forward") or {}).get("protocol") == "codex")
            and bool(body.get("_codex_origin_request_id"))
            and sum(bool(marker) for marker in codex_markers) == 1
        )
        if "product_work" in body:
            from services.product_work import normalize_product_work, require_product_source
            from services.work_lifecycle import WorkLifecycleError
            body = {**body, "product_work": normalize_product_work(body["product_work"])}
            if body["product_work"] is None:
                raise WorkLifecycleError("PRODUCT_WORK_INVALID", 400)
            if source is None and body.get("supersedes_request_id"):
                # Dedicated original-successor resume has no new caller lane;
                # inherit only the owner-scoped, validated original relationship.
                with self._db() as db:
                    original = self.store.read_receipt(db, "text", owner, body["supersedes_request_id"])
                if original and original.get("_product_work") == body["product_work"]:
                    source = original.get("_source")
            # source is set by the authenticated API, never a body field.
            require_product_source({"role": "admin", "_trusted_source": source})
            if self.admission is None:
                raise WorkLifecycleError("PRODUCT_WORK_UNAVAILABLE", 503)
            if (body.get("_route", "chat") != "chat" or body.get("_public_session_ref")
                    or body.get("_scheduling") or body.get("_forward") or body.get("_editable")):
                if not codex_product_bridge:
                    raise WorkLifecycleError("PRODUCT_WORK_LEGACY_REQUIRED", 400)

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
            if previous and _resume_absent_successor:
                # The dedicated missing-row recovery can accept exactly once.
                # A concurrent insert or a lost reply is read back, never reset.
                return self._public(json.loads(previous[1]))
            schedule = not previous
            previous_receipt = json.loads(previous[1]) if previous else None
            if (previous_receipt and previous_receipt.get("_product_work_key")
                    and source is not None and previous_receipt.get("_source") != source):
                from services.work_lifecycle import WorkLifecycleError
                raise WorkLifecycleError("PRODUCT_WORK_SOURCE_CONFLICT", 403)
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
                if body.get("_codex_fallback_of"):
                    from services.codex_text_fallback import validate_prepared
                    validate_prepared(self.store, db, owner, body)
                    receipt.update(
                        _codex_fallback_of=body["_codex_fallback_of"],
                        _codex_origin_request_id=body["_codex_origin_request_id"],
                        _codex_fallback_input_hash=body["_codex_fallback_input_hash"],
                        _codex_fallback_protocol=body["_codex_fallback_protocol"],
                    )
                if body.get("_codex_continue_of"):
                    from services.codex_text_fallback import validate_prepared_continue
                    validate_prepared_continue(self.store, db, owner, body)
                    receipt.update(
                        _codex_continue_of=body["_codex_continue_of"],
                        _codex_origin_request_id=body["_codex_origin_request_id"],
                        _codex_fallback_input_hash=body["_codex_fallback_input_hash"],
                        _codex_fallback_protocol=body["_codex_fallback_protocol"],
                    )
                if body.get("_codex_rejected_retry_of"):
                    from services.codex_text_fallback import validate_prepared_rejected_retry
                    validate_prepared_rejected_retry(self.store, db, owner, body)
                    receipt.update(
                        _codex_rejected_retry_of=body["_codex_rejected_retry_of"],
                        _codex_origin_request_id=body["_codex_origin_request_id"],
                        _codex_fallback_input_hash=body["_codex_fallback_input_hash"],
                        _codex_fallback_protocol=body["_codex_fallback_protocol"],
                    )
                if any(codex_markers):
                    effort = (((body.get("_forward") or {}).get("payload") or {}).get("reasoning") or {}).get("effort")
                    if isinstance(effort, str):
                        receipt["_codex_reasoning_effort"] = effort
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
                expected_native_sequence = self._prepare_supersede(db, owner, body, receipt)
                self._continue_public_session(db, owner, body, receipt)
                from services.generation_completion import attach_replacement
                attach_replacement(self.store, db, "text", owner, request_id, body, receipt)
                assigned_sequence = self.store.next_sequence(db)
                if expected_native_sequence is not None and assigned_sequence != expected_native_sequence:
                    raise ConversationBindingError("native no-final acceptance order changed", code="CHAT_SUPERSEDE_INVALID")
                receipt.update({"_input_ref": self.store.save_input(body),
                                "_sequence": assigned_sequence,
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
                from services.product_work import ensure_product_work
                product = body.get("product_work")
                previous_id = (receipt.get("_supersedes_request_id") or receipt.get("_previous_request_id")
                               or receipt.get("_completion_of") or receipt.get("_codex_fallback_of")
                               or receipt.get("_codex_continue_of") or receipt.get("_codex_rejected_retry_of"))
                original = self.store.read_receipt(db, "text", owner, previous_id) if previous_id else None
                if original and original.get("_product_work"):
                    from services.work_lifecycle import WorkLifecycleError
                    if product is not None and product != original["_product_work"]:
                        if (not codex_product_bridge
                                or product.get("product_ref") != original["_product_work"].get("product_ref")):
                            raise WorkLifecycleError("PRODUCT_WORK_BINDING_CONFLICT")
                    else:
                        product = original["_product_work"]
                    if not codex_product_bridge and receipt.get("_route") == "codex":
                        raise WorkLifecycleError("PRODUCT_WORK_LEGACY_REQUIRED", 400)
                    if source is not None and source != original.get("_source"):
                        raise WorkLifecycleError("PRODUCT_WORK_SOURCE_CONFLICT")
                    receipt["_source"] = original["_source"]
                    prepare_receipt(receipt, body.get("_scheduling"), receipt["_source"])
                    if self.admission is None:
                        raise WorkLifecycleError("PRODUCT_WORK_UNAVAILABLE", 503)
                parent = ensure_product_work(self.store, db, "text", owner, request_id, receipt,
                                             product, source=receipt["_source"])
                if parent is None:
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

    @staticmethod
    def _known_unsent_no_final_successor(receipt, now):
        """Recognize one stopped plain no-final child that never sent.

        This remains separate from the category-derived predicate: an absent
        derivation is not evidence that an arbitrary cursor-changed row is
        safe to resume.  The original is revalidated transactionally below,
        then the normal runner repeats its live no-final tail check before
        any upstream send.
        """
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
            and receipt.get("_derived_input") is None and receipt.get("_supersedes_no_final") is True
            and all(isinstance(receipt.get(key), str) and receipt[key] for key in
                ("_input_ref", "_supersedes_request_id", "request_id", "request_message_id")))

    @staticmethod
    def _strict_successor_sequence(previous, successor):
        """Keep a plain no-final resume on its original, ordered lineage."""
        parent_sequence = previous.get("_sequence")
        child_sequence = successor.get("_sequence")
        return (type(parent_sequence) is int and parent_sequence > 0
                and type(child_sequence) is int and child_sequence > 0
                and parent_sequence < child_sequence)

    def _validated_unsent_successor_input(self, db, owner, request_hash, receipt, now):
        plain_no_final = self._known_unsent_no_final_successor(receipt, now)
        if not plain_no_final and not self._known_unsent_category_successor(receipt, now):
            raise ConversationBindingError("successor is not known unsent", code="CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE")
        body = self.store.load_input(receipt["_input_ref"])
        if self._submission_identity(owner, body) != (receipt["request_id"], request_hash):
            raise ConversationBindingError("retained successor input changed", code="CONVERSATION_REQUEST_CONFLICT")
        previous = self._validate_supersede(self.store, db, owner, receipt, now)
        if plain_no_final and (not self._strict_successor_sequence(previous, receipt)
                               or not self._no_final_successor_source(previous, now)):
            raise ConversationBindingError("original no-final proof is unavailable", code="CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE")
        if previous.get("_recovery_paused") or previous.get("_recovery_suppressed"):
            raise ConversationBindingError("original recovery is paused", code="CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE")
        return body

    def resume_unsent_successor(self, owner, request_id, envelope):
        """Resume a proved-unsent category or plain no-final row, or a still-absent v4 ID.

        No original outcome or identity is rewritten. The normal runner repeats
        its original-node checks at both the read and final send boundaries.
        """
        if self.admission is None:
            raise ConversationBindingError("durable admission is unavailable", code="CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE")
        if (set(envelope) == self.SUPERSEDE_FIELDS | {"derived_input", "continue_after_no_final"}
                and envelope.get("client_request_id") == request_id
                and envelope.get("continue_after_no_final") is True
                and envelope.get("derived_input") == {"kind": "attributes_required_only_v4"}):
            # A rejected initial submit has no durable row and cannot have sent:
            # submission always commits its row before admission is awakened.
            # Reuse that transaction, immutable derivation and fresh send guard.
            # Existing rows are only returned, including UNKNOWN/failed rows.
            return self.submit(owner, envelope, _resume_absent_successor=True)
        category_envelope = (set(envelope) == self.SUPERSEDE_FIELDS | {"derived_input"}
            and envelope.get("derived_input") == {"kind": "category_directory_parent_v1"})
        plain_no_final_envelope = (set(envelope) == self.SUPERSEDE_FIELDS | {"continue_after_no_final"}
            and envelope.get("continue_after_no_final") is True)
        if (envelope.get("client_request_id") != request_id
                or not (category_envelope or plain_no_final_envelope)):
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
            failure = {key: receipt[key] for key in (
                "status", "error_code", "upstream_outcome", "finished_at", "updated_at", "_execution_timeline") if key in receipt}
            guard = safe_no_final_guard_diagnostic(receipt.get("_no_final_guard"))
            if guard is not None:
                failure["_no_final_guard"] = guard
            updated["_bound_successor_resume_failure"] = failure
            updated.pop("_no_final_guard", None)
            for key in ("error_code", "upstream_outcome", "finished_at", "waiting", "_ready_at", "_claim_id", "_claim_until"):
                updated.pop(key, None)
            updated.update(status="queued", boot=self.boot, updated_at=self._now(),
                _bound_successor_resume_count=1, _executing=False, _turn_reserved=False, _submission_started=False)
            updated["_execution_timeline"] = [*receipt["_execution_timeline"], {"stage": "known_unsent_successor_resumed", "at": self._now()}][-32:]
            self.store.write_receipt(db, "text", owner, request_id, updated)
        self.admission.wake()
        return self._public(updated)

    @staticmethod
    def _missing_successor_attempt(receipt):
        """Return the private one-use resend record, if it is well formed.

        ``request_message_id`` remains the ID of the original submitted
        successor.  The retry needs its own upstream user-node ID, otherwise a
        late appearance of the original node is indistinguishable from a new
        send.  This marker is deliberately part of the existing receipt: it is
        committed before a transport can start and survives a process crash.
        """
        value = receipt.get("_missing_successor_resume")
        if not isinstance(value, dict) or set(value) - {
            "version", "attempt", "state", "request_message_id",
            "original_request_message_id", "submission_parent_message_id",
            "attempt_submission_parent_message_id",
            "request_parent_message_id", "accepted_at", "submitted_at",
            "recovered_at", "result_source", "prior",
        }:
            return None
        if (value.get("version") != 1 or value.get("attempt") != 1
                or value.get("state") not in {"reserved", "submitted", "recovered"}
                or any(not isinstance(value.get(key), str) or not value[key].strip()
                       for key in ("request_message_id", "original_request_message_id",
                                   "submission_parent_message_id"))
                or value["original_request_message_id"] != receipt.get("request_message_id")
                or value["request_message_id"] == value["original_request_message_id"]
                or not isinstance(value.get("prior"), dict)):
            return None
        for key in ("accepted_at", "submitted_at", "recovered_at"):
            if key in value and (type(value[key]) not in (int, float) or not math.isfinite(value[key])):
                return None
        parent = value.get("request_parent_message_id")
        if parent is not None and (not isinstance(parent, str) or not parent.strip()):
            return None
        attempt_parent = value.get("attempt_submission_parent_message_id")
        if attempt_parent is not None and (not isinstance(attempt_parent, str) or not attempt_parent.strip()):
            return None
        if value.get("result_source") is not None and value["result_source"] not in {"original", "replacement"}:
            return None
        return value

    @classmethod
    def _recovery_started_at(cls, receipt, now):
        """Return the current recovery window's durable start time.

        A missing-successor resend retains the original receipt verbatim in
        ``prior``.  Its own finite original-result reads must begin at the
        reservation, rather than inheriting the old stopped receipt's age.
        """
        marker = cls._missing_successor_attempt(receipt)
        started = marker.get("accepted_at") if marker else receipt.get("created_at")
        return float(started) if type(started) in (int, float) and math.isfinite(started) else now

    @classmethod
    def _missing_successor_recovery_receipt(cls, receipt):
        marker = cls._missing_successor_attempt(receipt)
        if not marker or marker["state"] != "submitted":
            return None
        # The exact new user node is the only acceptable post-send recovery
        # target.  Keep the original receipt unchanged so its old node remains
        # auditable and a late old-node answer is still checked at send time.
        return {
            **receipt,
            "request_message_id": marker["request_message_id"],
            "request_parent_message_id": marker.get("request_parent_message_id")
                or marker["submission_parent_message_id"],
            "_submission_parent_message_id": marker.get("attempt_submission_parent_message_id")
                or marker["submission_parent_message_id"],
        }

    def _update_missing_successor_cursor(self, owner, request_id, marker_id, cursor):
        """Keep new-attempt cursor evidence private to its one-use marker."""
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            current = json.loads(row[0]) if row else None
            marker = self._missing_successor_attempt(current or {})
            if (not marker or marker["request_message_id"] != marker_id
                    or marker["state"] not in {"reserved", "submitted"}):
                return
            actual_parent = cursor.get("request_parent_message_id")
            actual_submission_parent = cursor.get("_submission_parent_message_id")
            if ((isinstance(actual_parent, str) and actual_parent.strip())
                    or (isinstance(actual_submission_parent, str) and actual_submission_parent.strip())):
                marker = {
                    **marker,
                    **({"request_parent_message_id": actual_parent}
                       if isinstance(actual_parent, str) and actual_parent.strip() else {}),
                    **({"attempt_submission_parent_message_id": actual_submission_parent}
                       if isinstance(actual_submission_parent, str) and actual_submission_parent.strip() else {}),
                }
                current["_missing_successor_resume"] = marker
                current["updated_at"] = self._now()
                db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(current), owner, request_id))

    def _mark_missing_successor_recovered(self, owner, request_id, source="replacement"):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            current = json.loads(row[0]) if row else None
            marker = self._missing_successor_attempt(current or {})
            if marker and marker["state"] in {"reserved", "submitted"}:
                current["_missing_successor_resume"] = {**marker, "state": "recovered", "recovered_at": self._now(),
                                                          "result_source": source}
                current["updated_at"] = self._now()
                db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(current), owner, request_id))

    def _validated_missing_successor_input(self, db, owner, request_id, request_hash, receipt, envelope, now):
        if (set(envelope) != self.SUPERSEDE_FIELDS
                or envelope.get("client_request_id") != request_id
                or any(not isinstance(envelope.get(key), str) or not envelope[key].strip()
                       or envelope[key] != envelope[key].strip() for key in self.SUPERSEDE_FIELDS)):
            raise ConversationBindingError("missing-successor envelope is invalid", code="CHAT_MISSING_SUCCESSOR_NOT_RESUMABLE")
        body = self.store.load_input(receipt.get("_input_ref"))
        if (not isinstance(body, dict) or self._submission_identity(owner, body) != (request_id, request_hash)
                or any(body.get(key) != envelope[key] for key in self.SUPERSEDE_FIELDS)
                or "derived_input" in body or self.MISSING_NODE_RETRY_FIELD in body):
            raise ConversationBindingError("retained successor input changed", code="CHAT_MISSING_SUCCESSOR_INPUT_UNAVAILABLE")
        if (self._no_final_model_rejection_retry(self.store, db, owner, receipt, now)
                or receipt.get(self.MISSING_NODE_RETRY_FIELD)):
            raise ConversationBindingError("model-rejection retry has no missing-successor resend",
                                           code="CHAT_MISSING_SUCCESSOR_NOT_RESUMABLE")
        parent = receipt.get("_submission_parent_message_id")
        if not isinstance(parent, str) or not parent:
            # Old receipts can use the public parent only when there is no
            # distinct actual request parent to confuse the retry cursor.
            parent = receipt.get("parent_message_id")
            if (not isinstance(parent, str) or not parent
                    or receipt.get("request_parent_message_id") not in (None, "", parent)):
                raise ConversationBindingError("successor parent is ambiguous", code="CHAT_MISSING_SUCCESSOR_CURSOR_CHANGED")
        if body.get("parent_message_id") != parent:
            raise ConversationBindingError("retained successor parent changed", code="CHAT_MISSING_SUCCESSOR_CURSOR_CHANGED")
        marker = self._missing_successor_attempt(receipt)
        if "_missing_successor_resume" in receipt and marker is None:
            raise ConversationBindingError("missing-successor marker is invalid", code="CHAT_MISSING_SUCCESSOR_NOT_RESUMABLE")
        if marker:
            return body, marker
        timeline = receipt.get("_execution_timeline")
        if (receipt.get("_route", "chat") != "chat" or receipt.get("_operation", "text") != "text"
                or receipt.get("_forward_protocol") or receipt.get("_public_session_ref")
                or receipt.get("_expected_sends") != 1
                or receipt.get("_derived_input")
                or receipt.get("_submission_started") is not True
                or receipt.get("status") != "failed" or receipt.get("error_code") != "RESULT_UNRECOVERABLE"
                or receipt.get("upstream_outcome") != "unknown"
                or receipt.get("recovery_reason") != "REQUEST_MESSAGE_NOT_FOUND"
                or receipt.get("_attempt_reason") != "ORIGINAL_REQUEST_MESSAGE_NOT_FOUND"
                or not isinstance(receipt.get("_attempt_finished_at"), (int, float))
                or receipt.get("_execution_wait_ended_at") is None
                or receipt.get("_executing") or receipt.get("_turn_reserved")
                or receipt.get("_work_key") or receipt.get("_completion") or receipt.get("_completion_of")
                or receipt.get("_recovery_paused") or receipt.get("_recovery_suppressed")
                or not isinstance(timeline, list)
                or not isinstance(receipt.get("request_message_id"), str) or not receipt["request_message_id"]):
            raise ConversationBindingError("successor is not an ended missing-node attempt", code="CHAT_MISSING_SUCCESSOR_NOT_RESUMABLE")
        for claim, until in (("_claim_id", "_claim_until"), ("recovery_claim_id", "recovery_lease_until")):
            value = receipt.get(claim)
            expiry = receipt.get(until)
            if value and (type(expiry) not in (int, float) or not math.isfinite(expiry) or expiry > now):
                raise ConversationBindingError("successor is busy", code="CHAT_MISSING_SUCCESSOR_NOT_RESUMABLE")
        # An expired original-result read cannot be allowed to finish over the
        # newly reserved attempt.  Clear only a provably expired lease.
        if receipt.get("recovery_claim_id"):
            receipt.pop("recovery_claim_id", None)
            receipt.pop("recovery_claimed_at", None)
            receipt.pop("recovery_lease_until", None)
        previous = self._validate_supersede(self.store, db, owner, receipt, now)
        if previous.get("status") == "succeeded" or previous.get("_upstream_terminal"):
            raise ConversationBindingError("original predecessor recovered", code="CHAT_MISSING_SUCCESSOR_NOT_RESUMABLE")
        child = db.execute("SELECT 1 FROM requests WHERE owner=? AND json_extract(receipt,'$._supersedes_request_id')=? LIMIT 1",
                           (owner, request_id)).fetchone()
        if child:
            raise ConversationBindingError("successor already has a child", code="CHAT_MISSING_SUCCESSOR_NOT_RESUMABLE")
        return body, None

    def resume_missing_successor(self, owner, request_id, envelope):
        """Reserve one new upstream user node for a stopped missing successor.

        This is intentionally separate from ordinary same-ID submission.  The
        frozen input and Provider request ID stay unchanged; only a new private
        upstream message UUID may be sent, and it is never regenerated.
        """
        if self.admission is None:
            raise ConversationBindingError("durable admission is unavailable", code="CHAT_MISSING_SUCCESSOR_NOT_RESUMABLE")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            if not row:
                raise ConversationBindingError("successor not found", code="CHAT_REQUEST_NOT_FOUND")
            receipt = json.loads(row[1])
            try:
                _, marker = self._validated_missing_successor_input(
                    db, owner, request_id, row[0], receipt, envelope, self._now())
            except (OSError, ValueError, TypeError, KeyError):
                raise ConversationBindingError("retained successor input unavailable", code="CHAT_MISSING_SUCCESSOR_INPUT_UNAVAILABLE") from None
            if marker:
                return self._public(receipt)
            now = self._now()
            parent = receipt.get("_submission_parent_message_id") or receipt.get("parent_message_id")
            marker = {
                "version": 1, "attempt": 1, "state": "reserved",
                "request_message_id": str(uuid.uuid4()),
                "original_request_message_id": receipt["request_message_id"],
                "submission_parent_message_id": parent,
                "accepted_at": now,
                "prior": deepcopy(receipt),
            }
            updated = {**receipt, "status": "queued", "boot": self.boot, "updated_at": now,
                       "_submission_started": False, "_turn_reserved": False, "_executing": False,
                       "_claim_id": None, "_claim_until": None,
                       "_execution_wait_ended_at": None, "_attempt_finished_at": None,
                       "_attempt_reason": None, "recovery_no_result_reads": 0,
                       "recovery_attempt": 0, "recovery_next_at": None,
                       "recovery_error_code": None, "recovery_phase": None,
                       "recovery_retry_after_seconds": None, "recovery_reason": None,
                       "recovery_last_read_error": None,
                       "recovery_requires_new_conversation": False,
                       "_missing_successor_resume": marker}
            updated["_execution_timeline"] = [*(receipt.get("_execution_timeline") or []),
                                                {"stage": "missing_successor_resume_reserved", "at": now}][-32:]
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
        # A model-rejection recovery keeps the rejected predecessor's frozen
        # root in its durable receipt, even though this one upstream attempt
        # is sent from that predecessor's verified no-final tail.
        preserve_submission_parent = False
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
            missing_attempt = self._missing_successor_attempt(receipt)
            if missing_attempt:
                if missing_attempt["state"] not in {"reserved", "submitted"}:
                    return
                # Never rewrite the frozen body.  These are execution-only
                # pointers for the one reserved upstream attempt.
                body = {**body, "_request_message_id": missing_attempt["request_message_id"],
                        "parent_message_id": missing_attempt["submission_parent_message_id"],
                        "_missing_successor_original_receipt": missing_attempt["prior"],
                        "_missing_successor_attempt_message_id": missing_attempt["request_message_id"]}
        def progress(cursor):
            if missing_attempt:
                self._update_missing_successor_cursor(owner, request_id,
                                                      missing_attempt["request_message_id"], cursor)
                cursor = {k: v for k, v in cursor.items()
                          if k not in {"request_parent_message_id", "_submission_parent_message_id"}}
            if (receipt.get("_supersedes_no_final") or preserve_submission_parent) and not missing_attempt:
                # The immutable predecessor link keeps its submitted root;
                # request_parent_message_id records the actual fresh tool tail.
                cursor = {k: v for k, v in cursor.items() if k != "_submission_parent_message_id"}
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
                    no_final_original = None
                    missing_node_original = None
                    native_no_final_chain = self._native_no_final_successor_source(
                        self.store, db, owner, previous, self._now(), successor_id=receipt["request_id"],
                        successor_sequence=receipt.get("_sequence"))
                    if (previous.get("_supersedes_no_final") is True
                            and self._rejected_chat_successor_source(previous, allow_no_final_successor=True)):
                        no_final_original = self.store.read_receipt(
                            db, "text", owner, previous.get("_supersedes_request_id"))
                    if self._missing_node_model_rejection_retry(self.store, db, owner, receipt, self._now()):
                        missing_node_original = self.store.read_receipt(
                            db, "text", owner, previous.get("_supersedes_request_id"))
                # A pre-conversation MODEL_NOT_FOUND retry has no upstream
                # message to guard. It still revalidates the frozen original
                # and bound identity above, then starts a new conversation.
                if previous.get("conversation_id"):
                    # A missing-node resume must guard the current submitted
                    # successor, not its historical predecessor.
                    guard_message_id = (receipt["request_message_id"] if missing_attempt
                                        else previous["request_message_id"])
                    body = {**body, "_supersedes_request_message_id": guard_message_id}
                if (previous.get("_supersedes_no_final") is True
                        and self._rejected_chat_successor_source(previous, allow_no_final_successor=True)):
                    actual_parent = previous.get("request_parent_message_id")
                    if not isinstance(actual_parent, str) or not actual_parent:
                        raise ConversationBindingError("model-rejection retry cursor is unavailable",
                                                       code="CHAT_SUPERSEDE_CURSOR_CHANGED")
                    # The durable retry body remains the rejected parent's
                    # frozen input.  A no-final predecessor may have sent from
                    # a later verified tail, so use that cursor only in this
                    # execution copy after the full chain revalidation above.
                    body = {**body, "parent_message_id": actual_parent}
                    preserve_submission_parent = True
                    if not no_final_original:
                        raise ConversationBindingError("model-rejection ancestor is unavailable",
                                                       code="CHAT_SUPERSEDE_INVALID")
                    # Reuse the binding layer's no-final resend guard.  It
                    # checks R's full user/tool tail, P's absence, and C's
                    # absence both at the initial GET and the pre-send GET;
                    # this is execution-only state and never changes C's
                    # frozen retained input.
                    body.update(_missing_successor_no_final_predecessor=no_final_original,
                                _missing_successor_expected_tail=actual_parent)
                if native_no_final_chain:
                    root, rejected = native_no_final_chain["original"], native_no_final_chain["parent"]
                    body["_native_no_final_prefix"] = {
                        "root_request_message_id": root["request_message_id"],
                        "root_parent_message_id": root.get("request_parent_message_id") or root.get("_submission_parent_message_id"),
                        "rejected_request_message_id": rejected["request_message_id"],
                        "native_request_message_id": previous["request_message_id"],
                        "native_parent_message_id": previous["request_parent_message_id"],
                    }
                if missing_node_original:
                    root_user = missing_node_original.get("request_message_id")
                    if not isinstance(root_user, str) or not root_user:
                        raise ConversationBindingError("missing-node retry root is unavailable",
                                                       code="CHAT_SUPERSEDE_INVALID")
                    body["_missing_node_retry_root_request_message_id"] = root_user
                if missing_attempt and receipt.get("_supersedes_no_final"):
                    expected_tail = receipt.get("request_parent_message_id")
                    if not isinstance(expected_tail, str) or not expected_tail:
                        raise ConversationBindingError("missing-successor no-final tail is unavailable",
                                                       code="CHAT_MISSING_SUCCESSOR_CURSOR_CHANGED")
                    body.update(_missing_successor_no_final_predecessor=previous,
                                _missing_successor_expected_tail=expected_tail)
                if receipt.get("_supersedes_no_final") and not missing_attempt:
                    body["_supersedes_no_final_original"] = previous
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
            missing_result_source = (result.pop("_missing_successor_result_source", "replacement")
                                     if missing_attempt else None)
            # A returned runner result is the terminal local proof for this
            # turn.  Persist that before the admission wrapper's bookkeeping:
            # callers completing the associated work in this small interval
            # must not be told that this already-saved turn is still executing.
            if missing_attempt:
                # The runner/read result belongs to the private replacement
                # node.  Its current cursor and submission root stay in the
                # marker, leaving the canonical successor's history intact.
                result.pop("request_parent_message_id", None)
                result.pop("_submission_parent_message_id", None)
            self._update(owner, request_id, **{**result, "status": "succeeded", "finished_at": self._now(),
                                              "_executing": False, "_turn_reserved": False})
            if missing_attempt:
                self._mark_missing_successor_recovered(owner, request_id, missing_result_source)
            context = current_request.get()
            if context is not None and hasattr(context, "record_stage"):
                context.record_stage("artifact_saved")
        except ConversationBindingError as exc:
            cursor = {k: getattr(exc, k) for k in ("provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id") if getattr(exc, k, "")}
            if (missing_attempt and exc.code == "CHAT_SUPERSEDE_ORIGINAL_FOUND"):
                # The original, already-submitted user node appeared during
                # the final binding-locked check.  A replacement has not been
                # sent, so consume this reservation and return to exact
                # original-ID recovery rather than labelling the historic
                # UNKNOWN as an unsent failure or queuing another dispatch.
                self._mark_missing_successor_recovered(owner, request_id, "original")
                self._update(owner, request_id, status="unknown",
                             error_code="CONVERSATION_OUTCOME_UNKNOWN", upstream_outcome="unknown",
                             _turn_reserved=False, _executing=False,
                             _claim_id=None, _claim_until=None, finished_at=None,
                             recovery_next_at=self._now(), **cursor)
                return
            if exc.code.startswith("CHAT_SUPERSEDE_"):
                transient = exc.code in {"CHAT_SUPERSEDE_READ_UNAVAILABLE", "CHAT_SUPERSEDE_PREDECESSOR_BUSY"}
                changes = {"error_code": exc.code, "upstream_outcome": "not_sent", "_turn_reserved": False,
                           "_executing": False, "_claim_id": None, "_claim_until": None}
                guard = safe_no_final_guard_diagnostic(getattr(exc, "no_final_guard", None))
                changes["_no_final_guard"] = guard
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
                    "original_upstream_error",
                    "original_upstream_response_ids",
                )
                if (value := getattr(exc, key, None)) is not None and value != ""
            }
            model_not_found = exc.code == "MODEL_NOT_FOUND"
            self._update(
                owner, request_id, **cursor, **diagnostic,
                classify_before_send=exc.code not in {"CONVERSATION_OUTCOME_UNKNOWN", "MODEL_NOT_FOUND"},
                status="unknown" if exc.code == "CONVERSATION_OUTCOME_UNKNOWN" else "failed",
                error_code=exc.code,
                finished_at=self._now(),
                **({
                    # The typed upstream HTTP response proves a send even if a
                    # legacy caller did not record its send boundary.
                    "_submission_started": True,
                    "upstream_outcome": "rejected",
                    "recovery_retryable": False,
                    "recovery_requires_new_conversation": False,
                    "recovery_next_at": None,
                    "waiting": None,
                    "_turn_reserved": False,
                    "_executing": False,
                } if model_not_found else {}),
            )
        except Exception:
            self._update(owner, request_id, status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                         original_failure_phase="runner", original_exception_category="other", finished_at=self._now())


text_task_service = TextTaskService(DATA_DIR / "text_tasks.sqlite3")
