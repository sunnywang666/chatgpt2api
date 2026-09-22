"""Read one original SQLite receipt without invoking Provider or upstream recovery.

Run against an authorized snapshot or the existing readable live SQLite/WAL pair.
No immutable=1 shortcut: that could hide committed WAL rows. No credentials,
request text, account IDs, raw conversation IDs or paths are printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3


def reference(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:16]


def number(value):
    return value if type(value) in {int, float} and math.isfinite(value) and value >= 0 else None


def choice(value, allowed):
    return value if isinstance(value, str) and value in allowed else "unknown"


def summarize(receipt):
    """Allowlist operational evidence; this report never decides a model outcome."""
    status = choice(receipt.get("status"), {"queued", "running", "not_started", "unknown", "failed", "succeeded"})
    report = {"found": True, "observational_only": True, "status": status,
              "recovery_suppressed": receipt.get("_recovery_suppressed") is True,
              "has_original_conversation": bool(receipt.get("conversation_id")),
              "has_original_message": bool(receipt.get("request_message_id")),
              "recovery_attempt": number(receipt.get("recovery_attempt")),
              "next_recovery_at": number(receipt.get("recovery_next_at")),
              "lease_until": number(receipt.get("recovery_lease_until"))}
    scan = receipt.get("_recovery_conversation_scan")
    if not isinstance(scan, dict):
        report["scan"] = None
        return report
    ids, index = scan.get("conversation_ids"), scan.get("next_index")
    if (not isinstance(ids, list) or len(ids) > 100 or any(not isinstance(v, str) for v in ids)
            or type(index) is not int or not 0 <= index <= len(ids)):
        report["scan"] = {"invalid": True}
        return report
    failures = scan.get("failed_reads")
    failures = failures if isinstance(failures, dict) else {}
    candidates = []
    for position, cid in enumerate(ids):
        row = failures.get(cid)
        row = row if isinstance(row, dict) else {}
        error = row.get("error")
        error = error if isinstance(error, dict) else {}
        http = error.get("http_status")
        candidates.append({"position": position, "candidate_ref": reference(cid),
                           "checked": position < index,
                           "deferred": cid in failures,
                           "failure_kind": choice(error.get("category"), {"http", "timeout", "transport", "parse", "other"}),
                           "http_status": http if type(http) is int and 100 <= http <= 599 else None,
                           "attempts": number(row.get("attempts")), "next_at": number(row.get("next_at")),
                           "retry_after_seconds": number(error.get("retry_after_seconds"))})
    matches = scan.get("matches")
    report["scan"] = {"list_complete": scan.get("coverage_complete") is True,
                      "listed_total": number(scan.get("next_offset")), "checked_in_window": index,
                      "unread_in_window": len(ids) - index,
                      "provisional_matches": len(matches) if isinstance(matches, list) else None,
                      "candidates": candidates}
    return report


def audit(database, owner, request_id):
    # Opening in ro mode cannot create a missing database or change receipt data.
    with sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True, timeout=2) as db:
        db.execute("PRAGMA query_only=ON")
        row = db.execute("SELECT CASE WHEN length(receipt)<=1048576 THEN receipt END "
                         "FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
    if row is None:
        return {"found": False, "observational_only": True}
    if row[0] is None:
        return {"found": True, "observational_only": True, "invalid_receipt": True}
    value = json.loads(row[0])
    if not isinstance(value, dict):
        return {"found": True, "observational_only": True, "invalid_receipt": True}
    return summarize(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--request-id", required=True)
    args = parser.parse_args()
    try:
        result = audit(args.database, args.owner, args.request_id)
    except (sqlite3.Error, OSError, ValueError):
        # No raw exception: it can contain private paths or data.
        print(json.dumps({"error": "READ_ONLY_AUDIT_UNAVAILABLE"}))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
