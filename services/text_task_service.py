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
import os
import sqlite3
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from services.config import DATA_DIR
from services.conversation_binding_service import ConversationBindingError, conversation_binding_service


class ContinuationExecutor:
    """Finish ready products before starting more first-turn gallery requests."""
    def __init__(self, max_workers=4):
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bound-text")
        self.limit = max_workers
        self.lock = threading.Lock()
        self.queue = []
        self.sequence = itertools.count()
        self.active = 0
        self.closed = False

    def submit(self, function, *args):
        future = Future()
        body = args[-1] if args and isinstance(args[-1], dict) else {}
        priority = 0 if body.get("conversation_id") else 1
        with self.lock:
            if self.closed:
                raise RuntimeError("executor shut down")
            heapq.heappush(self.queue, (priority, next(self.sequence), future, function, args))
            if self.active < self.limit:
                self.active += 1
                self.pool.submit(self._drain)
        return future

    def _drain(self):
        while True:
            with self.lock:
                if not self.queue:
                    self.active -= 1
                    return
                _, _, future, function, args = heapq.heappop(self.queue)
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(function(*args))
                except BaseException as exc:
                    future.set_exception(exc)

    def shutdown(self, wait=True):
        with self.lock:
            self.closed = True
        self.pool.shutdown(wait=wait)


class TextTaskService:
    RECOVERY_LEASE_SECONDS = 60.0
    RECOVERY_BASE_BACKOFF_SECONDS = 30.0
    RECOVERY_MAX_BACKOFF_SECONDS = 15.0 * 60.0
    _INTERNAL_RECEIPT_FIELDS = frozenset({
        "boot", "recovery_claim_id", "recovery_claimed_at", "recovery_lease_until",
    })
    _SAFE_RECOVERY_CODES = frozenset({
        "CONVERSATION_BINDING_UNAVAILABLE",
        "CONVERSATION_BINDING_UNSUPPORTED",
        "CONVERSATION_BINDING_CONTRACT_INVALID",
        "CONVERSATION_BINDING_MISMATCH",
        "CONVERSATION_OUTCOME_UNKNOWN",
    })
    _SUCCESS_RECOVERY_FIELDS = frozenset({
        # Binding and conversation identity stay rooted in the original
        # receipt. Only the recovered answer/cursor may advance.
        "content", "parent_message_id", "binding_status",
    })

    def __init__(self, path: Path, runner=None, executor=None, *, clock=None, recovery_reader=None):
        self.path = path
        self.runner = runner or conversation_binding_service.complete_text
        self.executor = executor or ContinuationExecutor()
        self.clock = clock or time.time
        self.recovery_reader = recovery_reader or conversation_binding_service.read_text_request
        self.boot = uuid.uuid4().hex

    def _now(self):
        return float(self.clock())

    @contextmanager
    def _db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=10)
        os.chmod(self.path, 0o600)
        db.execute("CREATE TABLE IF NOT EXISTS requests (owner TEXT, id TEXT, request_hash TEXT, receipt TEXT, PRIMARY KEY(owner,id))")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _public(receipt):
        return {k: v for k, v in receipt.items() if k not in TextTaskService._INTERNAL_RECEIPT_FIELDS}

    @classmethod
    def _recovery_due(cls, receipt, now):
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
    def _safe_recovery_result(cls, recovered):
        if not isinstance(recovered, dict):
            return None, "RECOVERY_INVALID_RESULT", "read_text_result"
        status = recovered.get("status")
        if status == "succeeded":
            content = recovered.get("content")
            parent_message_id = recovered.get("parent_message_id")
            if (recovered.get("binding_status") != "bound"
                    or not isinstance(content, str) or not content.strip()
                    or not isinstance(parent_message_id, str) or not parent_message_id.strip()):
                return None, "RECOVERY_INVALID_RESULT", "read_text_result"
            return {key: recovered[key] for key in cls._SUCCESS_RECOVERY_FIELDS if key in recovered}, None, None
        if status in {"running", "unknown"}:
            return None, "UPSTREAM_OUTCOME_UNKNOWN", "read_text_result"
        return None, "RECOVERY_INVALID_RESULT", "read_text_result"

    def _finish_recovery(self, owner, request_id, claim_id, recovered=None, error_code=None, phase=None):
        now = self._now()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            if not row:
                return {"request_id": request_id, "status": "not_found"}
            current = json.loads(row[0])
            if current.get("status") == "succeeded":
                # A late read result can never roll back an authoritative
                # success, even if another writer did not clear the lease.
                return self._public(current)
            if current.get("recovery_claim_id") != claim_id:
                return self._public(current)
            if error_code:
                attempt = max(1, int(current.get("recovery_attempt", 1)))
                changes = {
                    "recovery_next_at": now + self._recovery_backoff(attempt),
                    "recovery_error_code": error_code,
                    "recovery_phase": phase or "read_text_request",
                }
            else:
                changes = {
                    **(recovered or {}),
                    "status": "succeeded",
                    "finished_at": now,
                    "error_code": None,
                    "recovery_next_at": None,
                    "recovery_error_code": None,
                    "recovery_phase": None,
                }
            updated = {**current, **changes, "recovery_claim_id": None,
                       "recovery_claimed_at": None, "recovery_lease_until": None,
                       "updated_at": now}
            db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=? AND receipt=?",
                       (json.dumps(updated), owner, request_id, row[0]))
            return self._public(updated)

    def read(self, owner: str, request_id: str):
        recovery_claim = None
        now = self._now()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            if row:
                previous = json.loads(row[0])
                if (previous["status"] == "failed"
                        and previous.get("error_code") == "CONVERSATION_BINDING_UNAVAILABLE"
                        and previous.get("provider_binding_id")
                        and previous.get("provider_account_identity")
                        and not previous.get("conversation_id") and not previous.get("parent_message_id")):
                    # This legacy failure occurred after selecting an account
                    # but before obtaining its text token / sending a turn.
                    # Keep the same request hash and message id for recovery.
                    previous = {**previous, "status": "not_started", "updated_at": now}
                    db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(previous), owner, request_id))
                    row = (json.dumps(previous),)
                if previous["status"] == "queued" and previous["boot"] != self.boot:
                    # The atomic running claim never happened. Preserve identity
                    # and wait for the caller to supply the exact original body.
                    previous = {**previous, "status": "not_started", "updated_at": now}
                    db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(previous), owner, request_id))
                    row = (json.dumps(previous),)
                if previous["status"] == "running" and previous["boot"] != self.boot:
                    # Another process/restart cannot establish that the original
                    # write failed. Preserve its cursor and make the receipt
                    # eligible for the read-only recovery path.
                    previous = {**previous, "status": "unknown", "error_code": "CONVERSATION_OUTCOME_UNKNOWN", "updated_at": now}
                    db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(previous), owner, request_id))
                    row = (json.dumps(previous),)
                if (previous["status"] == "unknown" and previous.get("conversation_id")
                        and previous.get("request_message_id") and self._recovery_due(previous, now)):
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
                        "updated_at": now,
                    }
                    db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?",
                               (json.dumps(recovery_claim[1]), owner, request_id))
        if not row:
            return {"request_id": request_id, "status": "not_found"}
        if recovery_claim:
            try:
                recovered = self.recovery_reader(recovery_claim[1])
                safe_result, error_code, phase = self._safe_recovery_result(recovered)
                return self._finish_recovery(owner, request_id, recovery_claim[0], safe_result, error_code, phase)
            except ConversationBindingError as exc:
                return self._finish_recovery(owner, request_id, recovery_claim[0],
                                             error_code=self._safe_recovery_code(exc),
                                             phase="read_text_request")
            except Exception:
                # Never persist exception text: it may contain a token, prompt,
                # or provider response. The next bounded recovery window is
                # enough to make the request retryable without hammering GET.
                return self._finish_recovery(owner, request_id, recovery_claim[0],
                                             error_code="RECOVERY_READ_FAILED",
                                             phase="read_text_request")
        return self._public(json.loads(row[0]))

    def _update(self, owner, request_id, **changes):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            receipt = json.loads(row[0])
            if receipt.get("status") == "succeeded":
                # A late runner progress/error callback must not roll back an
                # authoritative recovery result or its advanced cursor.
                return
            receipt = {**receipt, **changes, "updated_at": self._now()}
            db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(receipt), owner, request_id))

    def submit(self, owner: str, body: dict):
        request_id = str(body.get("client_request_id") or "").strip()
        if not owner or not request_id or len(request_id) > 200:
            raise ConversationBindingError("request identity is required", code="CONVERSATION_BINDING_CONTRACT_INVALID")
        request_hash = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        receipt = {"request_id": request_id, "client_conversation_id": body["client_conversation_id"],
                   "request_message_id": str(uuid.uuid4()),
                   "status": "queued", "boot": self.boot, "created_at": self._now(), "updated_at": self._now()}
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            if previous and previous[0] != request_hash:
                raise ConversationBindingError("request identity already has different input", code="CONVERSATION_REQUEST_CONFLICT")
            schedule = not previous
            if previous and json.loads(previous[1])["status"] == "not_started":
                receipt = {**json.loads(previous[1]), "status": "queued", "boot": self.boot, "updated_at": self._now()}
                db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(receipt), owner, request_id))
                schedule = True
            elif not previous:
                db.execute("INSERT INTO requests VALUES(?,?,?,?)", (owner, request_id, request_hash, json.dumps(receipt)))
        if schedule:
            try:
                self.executor.submit(self._run, owner, request_id, {**body, "_request_message_id": receipt["request_message_id"]})
            except Exception:
                # No upstream call was scheduled, so this is a known rejection.
                self._update(owner, request_id, status="failed", error_code="CONVERSATION_SCHEDULING_FAILED")
        return self.read(owner, request_id)

    def _run(self, owner, request_id, body):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            receipt = json.loads(row[0])
            if receipt["status"] != "queued" or receipt["boot"] != self.boot:
                return
            receipt = {**receipt, "status": "running", "started_at": self._now()}
            db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(receipt), owner, request_id))
        def progress(cursor):
            self._update(owner, request_id, **cursor)
        try:
            result = self.runner(body, on_cursor=progress)
            self._update(owner, request_id, **{**result, "status": "succeeded", "finished_at": self._now()})
        except ConversationBindingError as exc:
            cursor = {k: getattr(exc, k) for k in ("provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id") if getattr(exc, k, "")}
            self._update(owner, request_id, **cursor,
                         status="unknown" if exc.code == "CONVERSATION_OUTCOME_UNKNOWN" else "failed",
                         error_code=exc.code, finished_at=self._now())
        except Exception:
            self._update(owner, request_id, status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN", finished_at=self._now())


text_task_service = TextTaskService(DATA_DIR / "text_tasks.sqlite3")
