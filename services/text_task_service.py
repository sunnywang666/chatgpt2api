"""Durable receipts for bound text requests, independent of HTTP lifetime.

Like image tasks, callers choose the request identity before sending. The
receipt stores results and cursors, never access tokens or uploaded image data.
An interrupted upstream operation is queried, never automatically resubmitted.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from services.config import DATA_DIR
from services.conversation_binding_service import ConversationBindingError, conversation_binding_service


class TextTaskService:
    def __init__(self, path: Path, runner=None, executor=None):
        self.path = path
        self.runner = runner or conversation_binding_service.complete_text
        self.executor = executor or ThreadPoolExecutor(max_workers=4, thread_name_prefix="bound-text")
        self.boot = uuid.uuid4().hex

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
        return {k: v for k, v in receipt.items() if k != "boot"}

    def read(self, owner: str, request_id: str):
        with self._db() as db:
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
        if not row:
            return {"request_id": request_id, "status": "not_found"}
        receipt = json.loads(row[0])
        if receipt["status"] in ("queued", "running") and receipt["boot"] != self.boot:
            # Another process/restart cannot establish that the original write
            # failed. Keep its cursor and require authoritative recovery.
            receipt = {**receipt, "status": "unknown", "error_code": "CONVERSATION_OUTCOME_UNKNOWN"}
        if receipt["status"] == "unknown" and receipt.get("conversation_id") and receipt.get("request_message_id"):
            try:
                recovered = conversation_binding_service.read_text_request(receipt)
                if recovered.get("status") == "succeeded":
                    self._update(owner, request_id, **recovered)
                    receipt = {**receipt, **recovered}
            except Exception:
                pass  # Query failure never authorizes a new message.
        return self._public(receipt)

    def _update(self, owner, request_id, **changes):
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            receipt = {**json.loads(row[0]), **changes, "updated_at": time.time()}
            db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (json.dumps(receipt), owner, request_id))

    def submit(self, owner: str, body: dict):
        request_id = str(body.get("client_request_id") or "").strip()
        if not owner or not request_id or len(request_id) > 200:
            raise ConversationBindingError("request identity is required", code="CONVERSATION_BINDING_CONTRACT_INVALID")
        request_hash = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        receipt = {"request_id": request_id, "client_conversation_id": body["client_conversation_id"],
                   "request_message_id": str(uuid.uuid4()),
                   "status": "queued", "boot": self.boot, "created_at": time.time(), "updated_at": time.time()}
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
            if previous and previous[0] != request_hash:
                raise ConversationBindingError("request identity already has different input", code="CONVERSATION_REQUEST_CONFLICT")
            if not previous:
                db.execute("INSERT INTO requests VALUES(?,?,?,?)", (owner, request_id, request_hash, json.dumps(receipt)))
        if not previous:
            try:
                self.executor.submit(self._run, owner, request_id, {**body, "_request_message_id": receipt["request_message_id"]})
            except Exception:
                # No upstream call was scheduled, so this is a known rejection.
                self._update(owner, request_id, status="failed", error_code="CONVERSATION_SCHEDULING_FAILED")
        return self.read(owner, request_id)

    def _run(self, owner, request_id, body):
        self._update(owner, request_id, status="running", started_at=time.time())
        def progress(cursor):
            self._update(owner, request_id, **cursor)
        try:
            result = self.runner(body, on_cursor=progress)
            self._update(owner, request_id, **{**result, "status": "succeeded", "finished_at": time.time()})
        except ConversationBindingError as exc:
            cursor = {k: getattr(exc, k) for k in ("provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id") if getattr(exc, k, "")}
            self._update(owner, request_id, **cursor,
                         status="unknown" if exc.code == "CONVERSATION_OUTCOME_UNKNOWN" else "failed",
                         error_code=exc.code, finished_at=time.time())
        except Exception:
            self._update(owner, request_id, status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN", finished_at=time.time())


text_task_service = TextTaskService(DATA_DIR / "text_tasks.sqlite3")
