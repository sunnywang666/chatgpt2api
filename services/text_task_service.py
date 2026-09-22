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
    RECOVERY_CONVERSATION_COVERAGE_VERSION_FIELD,
    RECOVERY_CONVERSATION_SCAN_FIELD,
    TextRecoveryReason,
    TURN_END_EVIDENCE_FIELD,
    conversation_binding_service,
    is_recovery_image_pointer,
)


from services.task_store import TaskStore
from services.request_context import current_request, AdmissionLost


class TextTaskCapacityError(RuntimeError):
    pass


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
    def _public(receipt):
        result = {k: v for k, v in receipt.items() if k not in TextTaskService._INTERNAL_RECEIPT_FIELDS and not k.startswith("_")}
        if receipt.get("_public_session_ref"):
            result["conversation"] = {"client_conversation_id": receipt["_public_session_ref"],
                                      "previous_request_id": receipt.get("_previous_request_id"),
                                      "protocol": "sequential-v1"}
        return result

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
        status = getattr(exc, "status_code", None)
        retry_after = getattr(exc, "retry_after", None)
        safe_retry_after = (
            int(retry_after)
            if isinstance(retry_after, (int, float))
            and not isinstance(retry_after, bool)
            and retry_after >= 0
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
        return cls._safe_recovery_code(exc), None

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
            set(value) != {
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
        }

    def _finish_recovery(
        self, owner, request_id, claim_id, recovered=None, error_code=None,
        phase=None, recovery_reason=None, retry_after_seconds=None, *, count_unrecoverable=False,
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
                    count_unrecoverable
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
                    "recovery_next_at": None,
                    "recovery_error_code": None,
                    "recovery_phase": None,
                    "recovery_retry_after_seconds": None,
                    "recovery_reason": None,
                    "recovery_requires_new_conversation": False,
                }
            updated = {**current, **changes, "recovery_claim_id": None,
                       "recovery_claimed_at": None, "recovery_lease_until": None,
                       "updated_at": now}
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

    def read(self, owner: str, request_id: str, *, allow_unrecoverable_retry: bool = False):
        from services.pool_admission import unknown_text_result
        recovery_claim = None
        now = self._now()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            if row:
                previous = json.loads(row[0])
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
                if (previous.get("_recovery_suppressed") is not True
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
            try:
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
            return self._authorize_unrecoverable(owner, request_id, result) if allow_unrecoverable_retry else result
        result = self._public(json.loads(row[0]))
        return self._authorize_unrecoverable(owner, request_id, result) if allow_unrecoverable_retry else result

    def recover(self, owner: str, request_id: str, allow_unrecoverable_retry: bool = False):
        return self.read(
            owner, request_id,
            allow_unrecoverable_retry=bool(allow_unrecoverable_retry),
        )

    def _authorize_unrecoverable(self, owner, request_id, observed):
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
            if current.get("status") == "succeeded" or current.get("error_code") == "RESULT_UNRECOVERABLE":
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
            db.execute(
                "UPDATE requests SET receipt=? WHERE owner=? AND id=? AND receipt=?",
                (json.dumps(updated), owner, request_id, row[0]),
            )
            return self._public(updated)

    def _update(self, owner, request_id, **changes):
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

    def validate_submission(self, owner: str, body: dict):
        """Read-only conflict check before any optional external review call."""
        request_id, request_hash = self._submission_identity(owner, body)
        with self._db() as db:
            previous = db.execute(
                "SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                (owner, request_id),
            ).fetchone()
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
            if db.execute("SELECT 1 FROM requests WHERE owner=? AND json_extract(receipt,'$._previous_request_id')=? LIMIT 1",
                          (owner, previous_id)).fetchone():
                reject("CHAT_CONVERSATION_CONFLICT")
            if previous.get("status") != "succeeded":
                reject("CHAT_PREVIOUS_REQUEST_PENDING")
            anchors = ("provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id")
            if not all(isinstance(previous.get(key), str) and previous[key] for key in anchors):
                reject("CHAT_CONTINUATION_UNAVAILABLE")
            receipt.update({key: previous[key] for key in anchors})
            # The actual last-user parent is filled by the final upstream payload;
            # it differs from the submission parent for multi-message input.
            receipt["_previous_request_id"] = previous_id
            if not previous.get("_public_session_ref"):
                receipt["_legacy_session_anchor"] = previous_id
            receipt["_submission_parent_message_id"] = previous["parent_message_id"]
        receipt["_public_session_ref"] = body["_public_session_ref"]

    def submit(self, owner: str, body: dict, *, source: str | None = None):
        from services.durable_forward import dispatch_model
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
                   "status": "queued", "boot": self.boot, "created_at": self._now(), "updated_at": self._now()}
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
                db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(receipt), owner, request_id))
                schedule = True
            elif not previous:
                self._continue_public_session(db, owner, body, receipt)
                receipt.update({"_input_ref": self.store.save_input(body),
                                "_sequence": self.store.next_sequence(db),
                                "_source": source or "key:" + owner,
                                "_input_bytes": _retained_size(body),
                                "_turn_reserved": False,
                                "_submission_started": False})
                for field in ("provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id"):
                    if body.get(field):
                        receipt[field] = body[field]
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
            self._update(owner, request_id, **{**result, "status": "succeeded", "finished_at": self._now()})
        except ConversationBindingError as exc:
            cursor = {k: getattr(exc, k) for k in ("provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id") if getattr(exc, k, "")}
            diagnostic = {
                key: value for key in (
                    "original_failure_phase", "original_http_status", "original_exception_category",
                    "original_upstream_request_stage",
                    "original_upstream_error_form", "original_upstream_rejected_field",
                )
                if (value := getattr(exc, key, None)) is not None and value != ""
            }
            self._update(owner, request_id, **cursor, **diagnostic,
                         status="unknown" if exc.code == "CONVERSATION_OUTCOME_UNKNOWN" else "failed",
                         error_code=exc.code, finished_at=self._now())
        except Exception:
            self._update(owner, request_id, status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                         original_failure_phase="runner", original_exception_category="other", finished_at=self._now())


text_task_service = TextTaskService(DATA_DIR / "text_tasks.sqlite3")
