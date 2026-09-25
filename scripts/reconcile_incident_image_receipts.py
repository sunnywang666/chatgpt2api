"""Copy explicitly named JSON-only image receipts into the existing SQLite ledger.

One-time repair for a Provider image deployed from the legacy JSON writer.
All Provider writers must be stopped before --apply, and the SQLite Provider
must start only after the transaction commits. Dry run is read-only. This
script never submits, polls, or changes an existing receipt.
An UNKNOWN is refused unless its exact original ID is separately named for
quarantine; then only the SQLite copy gains the reversible suppression marker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


class ReconciliationError(ValueError):
    pass


def _json_file(path: Path) -> tuple[str, list[dict]]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    document = json.loads(raw)
    items = document.get("tasks") if isinstance(document, dict) else document
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ReconciliationError("invalid legacy image task file")
    return digest, items


def _identity(item: dict) -> tuple[str, str]:
    owner, task_id = item.get("owner_id"), item.get("id")
    if not isinstance(owner, str) or not owner or not isinstance(task_id, str) or not task_id:
        raise ReconciliationError("legacy image receipt has no original identity")
    return owner, task_id


def _projection(item: dict) -> tuple[object, ...]:
    data = item.get("data")
    identity_fields = (
        "status", "binding_status", "request_hash", "provider_binding_id",
        "provider_account_identity", "client_conversation_id", "conversation_id",
        "parent_message_id",
    )
    return (*(item.get(field) for field in identity_fields),
            hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest())


def _plan(db: sqlite3.Connection, items: list[dict], task_ids: list[str],
          quarantine_unknown_ids: set[str]) -> tuple[str, list[dict]]:
    if len(task_ids) != len(set(task_ids)) or not task_ids:
        raise ReconciliationError("task IDs must be distinct and nonempty")
    if not quarantine_unknown_ids <= set(task_ids):
        raise ReconciliationError("UNKNOWN quarantine IDs must be explicitly selected")
    row = db.execute("SELECT value FROM task_runtime WHERE name='image_json_imported'").fetchone()
    if row is None or json.loads(row[0]) is not True:
        raise ReconciliationError("legacy JSON import state is not confirmed")

    legacy: dict[tuple[str, str], dict] = {}
    by_id: dict[str, tuple[str, str]] = {}
    for item in items:
        key = _identity(item)
        if key in legacy or (key[1] in by_id and by_id[key[1]] != key):
            raise ReconciliationError("duplicate original image identity in JSON")
        legacy[key] = item
        by_id[key[1]] = key
    if any(task_id not in by_id for task_id in task_ids):
        raise ReconciliationError("one or more named original IDs are absent from JSON")
    selected = {by_id[task_id] for task_id in task_ids}

    persisted: dict[tuple[str, str], dict] = {}
    persisted_ids: dict[str, set[tuple[str, str]]] = {}
    for task_key, raw in db.execute("SELECT task_key,receipt FROM image_requests"):
        item = json.loads(raw)
        key = _identity(item)
        if task_key != f"{key[0]}:{key[1]}" or key in persisted:
            raise ReconciliationError("SQLite image identity is inconsistent")
        persisted[key] = item
        persisted_ids.setdefault(key[1], set()).add(key)
    if any(persisted_ids.get(task_id, set()) - {by_id[task_id]}
           for task_id in task_ids):
        raise ReconciliationError("selected task ID belongs to another SQLite owner")
    for key in legacy.keys() & persisted.keys():
        if _projection(legacy[key]) != _projection(persisted[key]):
            raise ReconciliationError("an existing image receipt differs between ledgers")

    for key in selected:
        item = legacy[key]
        unknown = item.get("status") == "error" and "UNKNOWN" in str(item.get("error_code") or "").upper()
        if unknown != (key[1] in quarantine_unknown_ids):
            raise ReconciliationError("selected original outcome is UNKNOWN or quarantine target is not UNKNOWN")
        if unknown and key in persisted and persisted[key].get("_recovery_suppressed") is not True:
            raise ReconciliationError("existing UNKNOWN receipt is not quarantined")

    json_only = legacy.keys() - persisted.keys()
    if not json_only and selected <= persisted.keys():
        return "already_present", []
    if json_only != selected:
        raise ReconciliationError("JSON-only original IDs differ from the explicit selection")

    pending: list[dict] = []
    for key in sorted(selected):
        item = legacy[key]
        status, data = item.get("status"), item.get("data")
        if not isinstance(item.get("request_hash"), str) or not item["request_hash"]:
            raise ReconciliationError("selected image receipt lacks its request hash")
        if status == "success":
            if not isinstance(data, list) or not data or not all(
                isinstance(result, dict) and isinstance(result.get("url"), str) and result["url"]
                for result in data
            ):
                raise ReconciliationError("selected success has no saved image result")
        elif status == "error":
            if data not in (None, []):
                raise ReconciliationError("selected error contains an unclassified image result")
        else:
            raise ReconciliationError("selected receipt is not terminal")
        copy = dict(item)
        if key[1] in quarantine_unknown_ids:
            # JSON stays untouched. Only this SQLite copy is reversibly stopped
            # so the new scheduler does not auto-read or retry the original.
            copy["_recovery_suppressed"] = True
        pending.append(copy)
    return "pending", pending


def reconcile(data_root: Path, task_ids: list[str], *, apply: bool = False,
              expected_json_sha256: str = "",
              quarantine_unknown_ids: set[str] | None = None) -> dict:
    root = data_root.resolve(strict=True)
    json_path = root / "image_tasks.json"
    db_path = root / "text_tasks.sqlite3"
    if not db_path.is_file():
        raise ReconciliationError("existing SQLite receipt database is missing")
    digest, items = _json_file(json_path)
    if apply and (len(expected_json_sha256) != 64 or digest != expected_json_sha256):
        raise ReconciliationError("legacy JSON digest changed or was not supplied")
    mode = "rw" if apply else "ro"
    db = sqlite3.connect(f"file:{db_path.as_posix()}?mode={mode}", uri=True, timeout=30)
    try:
        db.execute("PRAGMA busy_timeout=30000")
        if apply:
            db.execute("BEGIN IMMEDIATE")
        else:
            db.execute("PRAGMA query_only=ON")
        state, pending = _plan(db, items, task_ids, quarantine_unknown_ids or set())
        if apply and pending:
            if _json_file(json_path)[0] != digest:
                raise ReconciliationError("legacy JSON changed during reconciliation")
            for item in pending:
                db.execute(
                    "INSERT INTO image_requests(task_key,receipt) VALUES(?,?)",
                    (f"{item['owner_id']}:{item['id']}", json.dumps(item, ensure_ascii=False, separators=(",", ":"))),
                )
            if _json_file(json_path)[0] != digest:
                raise ReconciliationError("legacy JSON changed before commit")
            db.commit()
        elif apply:
            db.rollback()
        return {
            "applied": apply and bool(pending),
            "state": state,
            "count": len(pending),
            "quarantined_count": sum(item["id"] in (quarantine_unknown_ids or set()) for item in pending),
            "json_sha256": digest,
            "selected": [
                {"id_sha256": hashlib.sha256(item["id"].encode()).hexdigest()[:16],
                 "status": item["status"], "results": len(item.get("data") or []),
                 "quarantined": item["id"] in (quarantine_unknown_ids or set())}
                for item in pending
            ],
        }
    except Exception:
        if apply:
            db.rollback()
        raise
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--quarantine-unknown-task-id", action="append", default=[])
    parser.add_argument("--expected-json-sha256", default="")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        result = reconcile(args.data_root, args.task_id, apply=args.apply,
                           expected_json_sha256=args.expected_json_sha256,
                           quarantine_unknown_ids=set(args.quarantine_unknown_task_id))
    except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
        print(json.dumps({"error": "IMAGE_RECEIPT_RECONCILIATION_REFUSED", "reason": str(exc)}))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
