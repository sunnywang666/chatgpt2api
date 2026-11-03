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
    version = receipt.get("_recovery_control_version", 0)
    return {"state": "pausing" if paused and in_flight else "paused" if paused else "stopped" if stopped else "active",
            "in_flight": in_flight,
            "operator_stopped": stopped,
            "version": version if type(version) is int and version >= 0 else None}


_SEND_EVIDENCE_STAGES = frozenset({
    "send_guard_passed", "send_call_started", "response_headers_received", "first_output", "task_finished",
    "stream_finished", "stream_handoff", "artifact_saved", "upstream_terminal",
})


def send_hold_control(receipt):
    """Public projection of a narrow, pre-send-only operator hold."""
    version = receipt.get("_send_hold_version", 0)
    expired = receipt.get("error_code") == "WAIT_DEADLINE_EXCEEDED"
    return {
        "state": "expired" if expired else "held" if receipt.get("_send_hold") is True else "active",
        "version": version if type(version) is int and version >= 0 else None,
    }


class SendHoldConflict(Exception):
    def __init__(self, control):
        super().__init__("Send hold changed; no update was applied")
        self.control = control


class SendHoldRejected(Exception):
    def __init__(self, code, control):
        super().__init__(code)
        self.code = code
        self.control = control


class ReceiptProjection(dict):
    """Detached receipt state that loads a large result only if it is used."""

    _OMITTED = frozenset(("content", "data"))

    def __init__(self, value, omitted_types, loader):
        super().__init__(value)
        # Keep only actual omitted members.  The overwhelmingly common
        # metadata lookup now needs one cheap dict lookup, not two helper
        # predicates and a set membership check.
        self._omitted_types = {
            key: value_type for key, value_type in omitted_types.items()
            if value_type is not None
        }
        self._loader = loader
        self._hydrated = False

    def _hydrate(self):
        if self._hydrated:
            return
        full = self._loader()
        if not isinstance(full, dict):
            raise ValueError("canonical receipt is missing during projection hydration")
        # Every compact field is already a detached exact value.  Do not
        # replace it: callers can mutate nested scheduling/timeline state
        # before a later payload access or write.  Hydration contributes only
        # the two intentionally omitted result fields.
        for key in self._OMITTED:
            if key in full:
                dict.__setitem__(self, key, full[key])
            else:
                dict.pop(self, key, None)
        self._hydrated = True
        self._omitted_types.clear()

    def __contains__(self, key):
        if self._omitted_types.get(key) is not None and not dict.__contains__(self, key):
            return True
        return dict.__contains__(self, key)

    def __getitem__(self, key):
        value_type = self._omitted_types.get(key)
        if value_type == "null":
            return None
        if value_type is not None:
            self._hydrate()
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        value_type = self._omitted_types.get(key)
        if value_type == "null":
            return None
        if value_type is not None:
            self._hydrate()
        return dict.get(self, key, default)

    def __setitem__(self, key, value):
        self._hydrate()
        return dict.__setitem__(self, key, value)

    def update(self, *args, **kwargs):
        self._hydrate()
        return dict.update(self, *args, **kwargs)

    def pop(self, key, *args):
        self._hydrate()
        return dict.pop(self, key, *args)

    def popitem(self):
        self._hydrate()
        return dict.popitem(self)

    def clear(self):
        self._hydrate()
        return dict.clear(self)

    def setdefault(self, key, default=None):
        self._hydrate()
        return dict.setdefault(self, key, default)

    def __delitem__(self, key):
        self._hydrate()
        return dict.__delitem__(self, key)

    def __iter__(self):
        self._hydrate()
        return dict.__iter__(self)

    def __len__(self):
        # Presence is indexed separately, so truth testing a receipt never
        # needs to pull a result body into the claim transaction.
        return dict.__len__(self) + sum(
            1 for key in self._omitted_types
            if not dict.__contains__(self, key))

    def keys(self):
        self._hydrate()
        return dict.keys(self)

    def items(self):
        self._hydrate()
        return dict.items(self)

    def values(self):
        self._hydrate()
        return dict.values(self)

    def copy(self):
        self._hydrate()
        return dict.copy(self)

    def __eq__(self, other):
        self._hydrate()
        if isinstance(other, ReceiptProjection):
            other._hydrate()
        return dict.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

    def __or__(self, other):
        self._hydrate()
        return dict.__or__(self, other)

    def __ror__(self, other):
        self._hydrate()
        return dict.__ror__(self, other)

    def __ior__(self, other):
        self._hydrate()
        return dict.__ior__(self, other)

    def __repr__(self):
        self._hydrate()
        return dict.__repr__(self)


# SQLite 3.45 does not substitute an indexed JSON-subtyped function result;
# it rereads the canonical receipt to recompute json_remove.  These consumers
# decode ordinary JSON text, so index and select the explicit TEXT expression.
_CLAIM_METADATA = "CAST(json_remove(receipt,'$.content','$.data') AS TEXT)"
_CLAIM_STATUS = "json_extract(receipt,'$.status')"
_CONTENT_TYPE = "json_type(receipt,'$.content')"
_DATA_TYPE = "json_type(receipt,'$.data')"


def _has_send_evidence(receipt):
    timeline = receipt.get("_execution_timeline")
    nonzero_or_invalid = lambda value: value is not None and (type(value) is not int or value != 0)
    return bool(
        receipt.get("_submission_started") is True
        or receipt.get("_last_sent_sequence") is not None
        or nonzero_or_invalid(receipt.get("_send_sequence"))
        or nonzero_or_invalid(receipt.get("send_count"))
        or receipt.get("upstream_submission_started") is True
        or receipt.get("upstream_outcome") not in (None, "not_sent", "not_submitted")
        or receipt.get("upstream_unfinished") is True
        or any(receipt.get(field) is not None for field in ("content", "result", "data"))
        or isinstance(timeline, list) and any(
            isinstance(item, dict) and item.get("stage") in _SEND_EVIDENCE_STAGES for item in timeline
        )
    )


def _send_hold_eligible(receipt):
    """A deliberately strict proof that this original has not crossed send."""
    timeline = receipt.get("_execution_timeline")
    return bool(
        receipt.get("_route", "chat") == "chat"
        and receipt.get("_operation", "text") == "text"
        and not receipt.get("_forward_protocol") and not receipt.get("_forward")
        and receipt.get("status") in {"queued", "not_started"}
        and isinstance(receipt.get("_input_ref"), str) and bool(receipt["_input_ref"])
        and receipt.get("_submission_started") is False
        and receipt.get("_turn_reserved") is False
        and receipt.get("_executing") is not True
        and receipt.get("_claim_id") is None and receipt.get("_claim_until") in (None, 0)
        and receipt.get("recovery_claim_id") is None and receipt.get("_recovery_claim_id") is None
        and receipt.get("recovery_lease_until") is None and receipt.get("_recovery_claim_until") is None
        and receipt.get("error_code") is None
        and isinstance(timeline, list) and all(isinstance(item, dict) for item in timeline)
        and not _has_send_evidence(receipt)
    )


class RecoveryControlConflict(Exception):
    def __init__(self, control):
        super().__init__("Recovery control changed; no update was applied")
        self.control = control


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
    def set_recovery_paused(self, kind, owner, request_id, paused, expected_version=None):
        # Serialize with both original-result claim paths. Never touch leases,
        # outcome, retry times, original input, or conversation/slot protection.
        with self.transaction() as db:
            receipt = self.read_receipt(db, kind, owner, request_id)
            if receipt is None:
                return None
            control = recovery_control(receipt)
            version = control["version"]
            if version is None or (expected_version is not None and (
                    type(expected_version) is not int or expected_version != version)):
                raise RecoveryControlConflict(control)
            if (receipt.get("_recovery_paused") is True) != paused:
                receipt["_recovery_paused"] = paused
                receipt["_recovery_control_updated_at"] = time.time()
            # Even a repeated pause is a new operator decision: a previous
            # conditional restore must not undo it. Other receipt writes do
            # not advance this control-only version.
            receipt["_recovery_control_version"] = version + 1
            self.write_receipt(db, kind, owner, request_id, receipt)
            return recovery_control(receipt)

    def set_text_send_hold(self, owner, request_id, held, expected_version):
        """Atomically hold/release one exact original before its first send."""
        with self.transaction() as db:
            receipt = self.read_receipt(db, "text", owner, request_id)
            if receipt is None:
                return None
            control = send_hold_control(receipt)
            version = control["version"]
            if (version is None or type(expected_version) is not int
                    or expected_version != version):
                raise SendHoldConflict(control)
            if control["state"] == "expired":
                raise SendHoldRejected("SEND_HOLD_WAIT_DEADLINE_EXCEEDED", control)
            if not _send_hold_eligible(receipt):
                if (receipt.get("status") == "running" or receipt.get("_claim_id") is not None
                        or receipt.get("_executing") is True or receipt.get("_turn_reserved") is True):
                    code = "SEND_HOLD_IN_FLIGHT"
                elif _has_send_evidence(receipt) or receipt.get("status") in {"unknown", "succeeded"}:
                    code = "SEND_HOLD_ALREADY_SUBMITTED"
                else:
                    code = "SEND_HOLD_NOT_ELIGIBLE"
                raise SendHoldRejected(code, control)
            deadline = (receipt.get("_scheduling") or {}).get("wait_deadline")
            if deadline:
                from services.workflow_scheduling import timestamp
                if time.time() >= timestamp(deadline):
                    raise SendHoldRejected("SEND_HOLD_WAIT_DEADLINE_EXCEEDED", {**control, "state": "expired"})
            if receipt.get("_send_hold") is not held:
                receipt["_send_hold"] = held
                receipt["_send_hold_updated_at"] = time.time()
            receipt["_send_hold_version"] = version + 1
            self.write_receipt(db, "text", owner, request_id, receipt)
            return send_hold_control(receipt)

    def __init__(self, path: Path):
        self.path = Path(path)
        self.input_dir = self.path.parent / (self.path.stem + "_inputs")
        self._schema_lock = threading.Lock()
        self._ready = False

    def _connect(self, *, busy_timeout_secs=30.0):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Create with private permissions, including before SQLite first opens it.
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.fchmod(fd, 0o600)
        os.close(fd)
        if (not isinstance(busy_timeout_secs, (int, float)) or isinstance(busy_timeout_secs, bool)
                or busy_timeout_secs <= 0):
            raise ValueError("busy timeout must be a positive number")
        db = sqlite3.connect(self.path, timeout=float(busy_timeout_secs))
        db.execute(f"PRAGMA busy_timeout={max(1, int(float(busy_timeout_secs) * 1000))}")
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
                    # The send guard must find existing ProductWork owners,
                    # including terminal/paused ones. Index its exact candidate
                    # predicate so each send need not parse unrelated saved
                    # image payloads while holding SQLite's writer lock.
                    for table in ("requests", "image_requests"):
                        db.execute(
                            f"CREATE INDEX IF NOT EXISTS {table}_product_conversation "
                            f"ON {table}(json_extract(receipt,'$.conversation_id')) "
                            "WHERE json_type(receipt,'$._product_work_key') IS NOT NULL "
                            "AND json_type(receipt,'$._product_work_key') != 'null'")
                    # An all-receipt covering expression index stores the
                    # scheduling projection of every canonical receipt.  Raw
                    # SQL writers stay lossless: SQLite updates this index as
                    # part of their normal canonical-receipt write, with no
                    # side table, status filter, or legacy-row backfill.
                    for table, identity in (("requests", "owner,id"), ("image_requests", "task_key")):
                        index = f"{table}_claim_receipt_projection"
                        definition = (f"CREATE INDEX {index} ON {table}"
                                      f"({_CLAIM_STATUS},{identity},{_CLAIM_METADATA},{_CONTENT_TYPE},{_DATA_TYPE})")
                        existing = db.execute("SELECT sql FROM sqlite_master WHERE name=?", (index,)).fetchone()
                        if existing is not None and existing[0] != definition:
                            # Upgrade both legacy expressions in this existing
                            # schema transaction. Canonical rows stay untouched;
                            # rollback restores the old indexes if either fails.
                            db.execute(f"DROP INDEX {index}")
                        if existing is None or existing[0] != definition:
                            db.execute(definition)
            finally:
                db.close()
            self._ready = True

    @contextmanager
    def connect(self, *, busy_timeout_secs=30.0):
        self.initialize()
        db = self._connect(busy_timeout_secs=busy_timeout_secs)
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self, *, busy_timeout_secs=30.0):
        with self.connect(busy_timeout_secs=busy_timeout_secs) as db:
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

    def claim_receipts(self, db, *, statuses=None, include_pending_completion=False,
                       include_upstream_unfinished=False):
        """Return every receipt from its covering scheduling projection.

        ``content`` and ``data`` retain their ordinary mapping behavior, but
        their canonical values are loaded only when a caller actually reads or
        mutates either field.  Other fields are exact compact copies, including
        result IDs and recovery state.
        """
        values = () if statuses is None else tuple(statuses)
        conditions = []
        if values:
            # SQLite 3.45 can recompute the JSON-subtyped status expression
            # from the canonical row when IN is part of an OR scan. In that
            # path, extract status from the indexed TEXT projection instead.
            # Keep the direct status index seek for status-only selection.
            status = (f"json_extract({_CLAIM_METADATA},'$.status')"
                      if include_pending_completion or include_upstream_unfinished else _CLAIM_STATUS)
            conditions.append(status + " IN (" + ",".join("?" for _ in values) + ")")
        # Read exceptional occupancy from the existing covering projection too.
        # Referring to canonical receipt here would parse every saved result
        # while the final send guard holds the writer and account locks.
        if include_pending_completion:
            conditions.append(f"(json_type({_CLAIM_METADATA},'$._completion')='object' AND "
                              f"coalesce(json_extract({_CLAIM_METADATA},'$._completion.state'),'') "
                              "NOT IN ('completed','result_ready'))")
        if include_upstream_unfinished:
            conditions.append(f"json_extract({_CLAIM_METADATA},'$.upstream_unfinished')=1")
        where = (" WHERE (" + " OR ".join(conditions) + ")" if conditions
                 else " WHERE 0" if statuses is not None else "")
        return self._projected_receipts(db, where, values)

    def successor_conflict_receipts(self, db, owner, original_id, original, external_request_ids):
        """Prefilter all conflict classes, including completed and foreign-owner rows.

        validate_successor remains authoritative. No status filter is safe:
        a completed duplicate child or registered external message still fences
        a successor. Only unrelated metadata is excluded here.
        """
        external_request_ids = tuple(external_request_ids)
        field = lambda name: f"json_extract({_CLAIM_METADATA},'$.{name}')"
        where = (f" WHERE (({{owner}}=? AND ({field('_supersedes_request_id')}=? OR "
                 f"{field('_codex_fallback_of')}=?)) OR {field('_product_work_key')}=? OR "
                 f"({field('provider_account_identity')}=? AND "
                 f"({field('conversation_id')}=? OR {field('request_message_id')} IN (" +
                 ",".join("?" for _ in external_request_ids) + "))))")
        values = (owner, original_id, original_id,
                  original['_product_work_key'], original['provider_account_identity'],
                  original['conversation_id'], *external_request_ids)
        return self._projected_receipts(db, where, values)

    def _projected_receipts(self, db, where, values):
        rows = []

        def load_canonical(kind, owner, request_id):
            # Claim-time access stays on the current transaction.  A caller
            # may also keep a detached projection after its read connection
            # closes; reopen only in that case, never while the writer is live.
            try:
                return self.read_receipt(db, kind, owner, request_id)
            except sqlite3.ProgrammingError:
                with self.connect() as read_db:
                    return self.read_receipt(read_db, kind, owner, request_id)

        def projected(kind, owner, request_id, compact_receipt, content, data):
            if content in (None, "null") and data in (None, "null"):
                if content == "null":
                    compact_receipt["content"] = None
                if data == "null":
                    compact_receipt["data"] = None
                return kind, owner, request_id, compact_receipt
            return (kind, owner, request_id, ReceiptProjection(
                compact_receipt, {"content": content, "data": data},
                lambda kind=kind, owner=owner, request_id=request_id:
                load_canonical(kind, owner, request_id)))

        # The former table scans were rowid ordered.  Asking SQLite to restore
        # that order after an expression-index read makes it sort large compact
        # JSON values.  Read rowid from the covering index and restore the same
        # order in Python using only small scalar references.
        text_sql = (f"SELECT rowid,owner,id,{_CLAIM_METADATA},{_CONTENT_TYPE},{_DATA_TYPE}" +
                    " FROM requests INDEXED BY requests_claim_receipt_projection" + where.format(owner="owner"))
        text_rows = list(db.execute(text_sql, values))
        text_rows.sort(key=lambda row: row[0])
        for _rowid, owner, request_id, raw, content, data in text_rows:
            rows.append(projected("text", owner, request_id, json.loads(raw), content, data))
        del text_rows
        image_sql = (f"SELECT rowid,task_key,{_CLAIM_METADATA},{_CONTENT_TYPE},{_DATA_TYPE}" +
                     " FROM image_requests INDEXED BY image_requests_claim_receipt_projection" +
                     where.format(owner=f"json_extract({_CLAIM_METADATA},'$.owner_id')"))
        image_rows = list(db.execute(image_sql, values))
        image_rows.sort(key=lambda row: row[0])
        for _rowid, _task_key, raw, content, data in image_rows:
            compact_receipt = json.loads(raw)
            rows.append(projected("image", compact_receipt["owner_id"], compact_receipt["id"], compact_receipt, content, data))
        return rows

    @staticmethod
    def product_work_conversation_receipts(db, receipt):
        """Yield only ProductWork candidates that can share ``receipt``'s conversation.

        The caller remains responsible for the authoritative
        ``_same_conversation`` check.  SQL deliberately keeps malformed
        identity values as candidates, so its prefilter cannot turn an
        identity/binding edge case into a missed ProductWork fence.
        """
        conversation = receipt.get("conversation_id")
        if not conversation:
            return
        conditions = [
            "json_type(receipt,'$._product_work_key') IS NOT NULL",
            "json_type(receipt,'$._product_work_key') != 'null'",
            "json_extract(receipt,'$.conversation_id')=?",
        ]
        values = [conversation]
        identity = receipt.get("provider_account_identity")
        binding = receipt.get("provider_binding_id")
        if isinstance(identity, str) and identity:
            identity_filter = ["json_extract(receipt,'$.provider_account_identity')=?",
                               "json_type(receipt,'$.provider_account_identity') IS NOT 'text'"]
            identity_values = [identity]
            if isinstance(binding, str) and binding:
                identity_filter.append("(json_extract(receipt,'$.provider_account_identity')='' "
                                       "AND json_extract(receipt,'$.provider_binding_id')=?)")
                identity_values.append(binding)
            conditions.append("(" + " OR ".join(identity_filter) + ")")
            values.extend(identity_values)
        elif not identity and isinstance(binding, str) and binding:
            # With no usable identity on this receipt, _same_conversation
            # falls back to the exact binding even if the candidate has one.
            conditions.append("json_extract(receipt,'$.provider_binding_id')=?")
            values.append(binding)
        where = " WHERE " + " AND ".join(conditions)
        for owner, request_id, raw in db.execute("SELECT owner,id,receipt FROM requests" + where, values):
            yield "text", owner, request_id, json.loads(raw)
        for _key, raw in db.execute("SELECT task_key,receipt FROM image_requests" + where, values):
            candidate = json.loads(raw)
            yield "image", candidate["owner_id"], candidate["id"], candidate

    @staticmethod
    def read_receipt(db, kind, owner, request_id):
        if kind == "text":
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
        elif kind == "image":
            row = db.execute("SELECT receipt FROM image_requests WHERE task_key=?", (owner + ":" + request_id,)).fetchone()
        else:
            raise ValueError("unsupported receipt kind")
        return json.loads(row[0]) if row else None

    @staticmethod
    def write_receipt(db, kind, owner, request_id, receipt):
        if isinstance(receipt, ReceiptProjection):
            receipt._hydrate()
        raw = json.dumps(receipt, ensure_ascii=False, separators=(",", ":"))
        if kind == "text":
            db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?", (raw, owner, request_id))
        elif kind == "image":
            db.execute("INSERT INTO image_requests VALUES(?,?) ON CONFLICT(task_key) DO UPDATE SET receipt=excluded.receipt",
                       (owner + ":" + request_id, raw))
        else:
            raise ValueError("unsupported receipt kind")
