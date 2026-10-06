"""Storage helpers for the existing text and image receipts.

Both receipt types participate in the text-task SQLite transaction. Private
inputs are fsynced before acceptance and are never part of a public receipt.
This module does not select accounts, execute work, or retry upstream requests.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import stat
import threading
import time
import uuid


# An expression index keeps the scheduler from reparsing every saved image just
# to decide whether it needs that receipt. Be conservative: only ordinary,
# finished successes can be omitted. Retry/completion children remain visible
# as reverse links, and exact predecessors are fetched separately by primary key.
_ADMISSION_IMAGE_NEEDED = """CASE WHEN
    json_extract(receipt,'$.status')='success'
    AND coalesce(json_extract(receipt,'$.upstream_unfinished'),0)=0
    AND coalesce(json_extract(receipt,'$.upstream_outcome'),'')!='unknown'
    AND (coalesce(json_type(receipt,'$._completion'),'')!='object'
         OR json_extract(receipt,'$._completion.state') IN ('completed','result_ready'))
    AND json_extract(receipt,'$._same_session_retry_of') IS NULL
    AND json_extract(receipt,'$._completion_of') IS NULL
    THEN 0 ELSE 1 END"""


def pending_image_result_ids(receipt):
    """Request-scoped observations that still require an authoritative settle read."""
    value = receipt.get("_pending_image_result_ids")
    if not isinstance(value, dict):
        return {}
    result = {field: list(dict.fromkeys(item.strip() for item in value.get(field, [])
              if isinstance(item, str) and item.strip())) if isinstance(value.get(field), list) else []
              for field in ("file_ids", "sediment_ids")}
    return result if any(result.values()) else {}


def recovery_control(receipt):
    """A read pause never certifies that an already claimed operation stopped."""
    paused = receipt.get("_recovery_paused") is True
    in_flight = bool(receipt.get("recovery_claim_id") or receipt.get("_executing")
                     or receipt.get("status") == "running")
    stopped = receipt.get("_recovery_suppressed") is True
    return {"state": "pausing" if paused and in_flight else "paused" if paused else "stopped" if stopped else "active",
            "in_flight": in_flight,
            "operator_stopped": stopped}


def _pack(value):
    if isinstance(value, (bytes, bytearray)):
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if isinstance(value, dict):
        return ["dict", [[key, _pack(item)] for key, item in value.items()]]
    if isinstance(value, (tuple, list)):
        return ["tuple" if isinstance(value, tuple) else "list", [_pack(item) for item in value]]
    if value is None or type(value) in (bool, int, float, str):
        return ["scalar", value]
    raise ValueError("unsupported durable input type")


def _unpack(value):
    tag, data = value
    if tag == "bytes":
        return base64.b64decode(data, validate=True)
    if tag == "dict":
        return {key: _unpack(item) for key, item in data}
    if tag in {"tuple", "list"}:
        items = [_unpack(item) for item in data]
        return tuple(items) if tag == "tuple" else items
    if tag == "scalar" and (data is None or type(data) in (bool, int, float, str)):
        return data
    raise ValueError("invalid durable input")


class TaskStore:
    def set_recovery_paused(self, kind, owner, request_id, paused):
        # Serialize with both original-result claim paths. Never touch leases,
        # outcome, retry times, original input, or conversation/slot protection.
        with self.transaction() as db:
            receipt = self.read_receipt(db, kind, owner, request_id)
            if receipt is None:
                return None
            if (receipt.get("_recovery_paused") is True) != paused:
                receipt["_recovery_paused"] = paused
                receipt["_recovery_control_updated_at"] = time.time()
                self.write_receipt(db, kind, owner, request_id, receipt)
            return recovery_control(receipt)

    def __init__(self, path: Path):
        self.path = Path(path)
        self.input_dir = self.path.parent / (self.path.stem + "_inputs")
        self._schema_lock = threading.Lock()
        self._ready = False

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Create with private permissions, including before SQLite first opens it.
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.fchmod(fd, 0o600)
        os.close(fd)
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def initialize(self):
        with self._schema_lock:
            if self._ready:
                return
            db = self._connect()
            try:
                with db:
                    db.execute("BEGIN IMMEDIATE")
                    db.execute("CREATE TABLE IF NOT EXISTS requests (owner TEXT, id TEXT, request_hash TEXT, receipt TEXT, PRIMARY KEY(owner,id))")
                    db.execute("CREATE TABLE IF NOT EXISTS image_requests (task_key TEXT PRIMARY KEY, receipt TEXT NOT NULL)")
                    db.execute("CREATE TABLE IF NOT EXISTS task_runtime (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
                    for table in ("requests", "image_requests"):
                        db.execute(f"CREATE INDEX IF NOT EXISTS {table}_status ON {table} (json_extract(receipt,'$.status'))")
                    db.execute("CREATE INDEX IF NOT EXISTS image_requests_admission ON image_requests ("
                               + _ADMISSION_IMAGE_NEEDED + ")")
            finally:
                db.close()
            self._ready = True

    @contextmanager
    def connect(self):
        self.initialize()
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            yield db

    @staticmethod
    def runtime(db, name, default=None):
        row = db.execute("SELECT value FROM task_runtime WHERE name=?", (name,)).fetchone()
        return json.loads(row[0]) if row else default

    @staticmethod
    def set_runtime(db, name, value):
        db.execute("INSERT INTO task_runtime VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                   (name, json.dumps(value)))

    @classmethod
    def next_sequence(cls, db):
        value = cls.runtime(db, "acceptance_sequence", 0) + 1
        cls.set_runtime(db, "acceptance_sequence", value)
        return value

    def save_input(self, body):
        if self.input_dir.is_symlink():
            raise ValueError("private input directory is not a real directory")
        self.input_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.input_dir, 0o700)
        name = uuid.uuid4().hex + ".json"
        fd = os.open(self.input_dir / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_pack(body), handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(self.input_dir, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return name

    def load_input(self, name):
        if (not isinstance(name, str) or len(name) != 37 or not name.endswith(".json")
                or any(c not in "0123456789abcdef" for c in name[:-5]) or self.input_dir.is_symlink()):
            raise ValueError("invalid private input reference")
        fd = os.open(self.input_dir / name, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
                raise ValueError("private input permissions are invalid")
            return _unpack(json.load(handle))

    def create_output(self):
        # Same private directory and durability boundary as accepted inputs.
        self.input_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        name = uuid.uuid4().hex + ".bin"
        with self.output_file(name, create=True) as handle:
            handle.flush()
            os.fsync(handle.fileno())
        fd = os.open(self.input_dir, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return name

    @contextmanager
    def output_file(self, name, *, create=False, append=False):
        if (not isinstance(name, str) or len(name) != 36 or not name.endswith(".bin")
                or any(c not in "0123456789abcdef" for c in name[:-4]) or self.input_dir.is_symlink()):
            raise ValueError("invalid private output reference")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL if create else os.O_WRONLY | os.O_APPEND if append else os.O_RDONLY
        fd = os.open(self.input_dir / name, flags | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "ab" if append or create else "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
                raise ValueError("private output permissions are invalid")
            yield handle

    @staticmethod
    def receipts(db, *, statuses=None, include_pending_completion=False, include_upstream_unfinished=False):
        # A scheduler may prefilter candidates before decoding saved image
        # payloads. Keep full receipts and the caller's authoritative predicates;
        # public reads, lineage and admission still use the unfiltered default.
        conditions, values = [], []
        if statuses is not None:
            values = list(statuses)
            if values:
                conditions.append("json_extract(receipt,'$.status') IN (" + ",".join("?" for _ in values) + ")")
        if include_pending_completion:
            conditions.append("(json_type(receipt,'$._completion')='object' AND "
                              "coalesce(json_extract(receipt,'$._completion.state'),'') NOT IN ('completed','result_ready'))")
        if include_upstream_unfinished:
            # Legacy terminal-looking receipts can still reserve a physical
            # turn. A status prefilter must not hide their unfinished marker.
            conditions.append("json_extract(receipt,'$.upstream_unfinished')=1")
        where = " WHERE (" + " OR ".join(conditions) + ")" if conditions else " WHERE 0" if statuses is not None else ""
        for owner, request_id, raw in db.execute("SELECT owner,id,receipt FROM requests" + where, values):
            yield "text", owner, request_id, json.loads(raw)
        for key, raw in db.execute("SELECT task_key,receipt FROM image_requests" + where, values):
            receipt = json.loads(raw)
            yield "image", receipt["owner_id"], receipt["id"], receipt

    @staticmethod
    def read_receipt(db, kind, owner, request_id):
        if kind == "text":
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
        elif kind == "image":
            row = db.execute("SELECT receipt FROM image_requests WHERE task_key=?", (owner + ":" + request_id,)).fetchone()
        else:
            raise ValueError("unsupported receipt kind")
        return json.loads(row[0]) if row else None

    @classmethod
    def admission_receipts(cls, db):
        """Claim snapshot without decoding unrelated, completed image results.

        Text history still participates in public-session ordering unchanged.
        Image dependencies retain the full original payload for fingerprint and
        cursor checks, including selected same-session retry results. The caller
        holds the transaction; ordinary receipt/history reads are unchanged.
        """
        for owner, request_id, raw in db.execute("SELECT owner,id,receipt FROM requests"):
            yield "text", owner, request_id, json.loads(raw)
        images = {}
        for (raw,) in db.execute("SELECT receipt FROM image_requests WHERE (" + _ADMISSION_IMAGE_NEEDED + ")=1"):
            receipt = json.loads(raw)
            images[(receipt["owner_id"], receipt["id"])] = receipt
        pending = list(images.items())
        for (owner, _), receipt in pending:
            thread = receipt.get("_image_thread") or {}
            completion = receipt.get("_completion") or {}
            references = [receipt.get(name) for name in (
                "_previous_request_id", "_terminal_empty_correction_of", "_supersedes_request_id",
                "_same_session_retry_of", "_completion_of")]
            references += [thread.get(name) for name in (
                "previous_task_id", "edit_source_task_id", "origin_task_id")]
            references.append(completion.get("selected_id"))
            for request_id in references:
                if not isinstance(request_id, str) or not request_id or (owner, request_id) in images:
                    continue
                previous = cls.read_receipt(db, "image", owner, request_id)
                if previous is not None:
                    images[(owner, request_id)] = previous
                    pending.append(((owner, request_id), previous))
        for (owner, request_id), receipt in images.items():
            yield "image", owner, request_id, receipt

    @staticmethod
    def work_receipts(db, kind, owner, work_key, conversation_id=None):
        """Read a superset of work members without decoding unrelated images.

        Keep all physical-conversation aliases and all same-owner legacy rows;
        the lifecycle's existing Python predicates remain authoritative.
        This only narrows reads, never rewrites saved receipt payloads.
        """
        for row_kind, table, owner_column in (
                ("text", "requests", "owner"),
                ("image", "image_requests", "json_extract(receipt,'$.owner_id')")):
            clauses = ["json_extract(receipt,'$._work_key')=?"]
            values = [work_key]
            if conversation_id:
                clauses.append("json_extract(receipt,'$.conversation_id')=?")
                values.append(conversation_id)
            if row_kind == kind:
                # A non-text/empty work key is a broad legacy candidate. Do not
                # reproduce _reference precedence or Python truthiness in SQL.
                clauses.append(f"({owner_column}=? AND (json_type(receipt,'$._work_key') IS NOT 'text' "
                               "OR json_extract(receipt,'$._work_key')=''))")
                values.append(owner)
            columns = "owner,id,receipt" if row_kind == "text" else "task_key,receipt"
            for row in db.execute(f"SELECT {columns} FROM {table} WHERE (" + " OR ".join(clauses) + ")", values):
                receipt = json.loads(row[-1])
                if row_kind == "text":
                    yield row_kind, row[0], row[1], receipt
                else:
                    yield row_kind, receipt["owner_id"], receipt["id"], receipt

    @staticmethod
    def write_receipt(db, kind, owner, request_id, receipt):
        raw = json.dumps(receipt, ensure_ascii=False, separators=(",", ":"))
        if kind == "text":
            db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (raw, owner, request_id))
        elif kind == "image":
            db.execute("INSERT INTO image_requests VALUES(?,?) ON CONFLICT(task_key) DO UPDATE SET receipt=excluded.receipt",
                       (owner + ":" + request_id, raw))
        else:
            raise ValueError("unsupported receipt kind")
