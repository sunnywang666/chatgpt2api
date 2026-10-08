"""Inspect or explicitly stop recovery for one named original listing Chat turn.

Dry-run is the default and opens the existing SQLite receipt database read-only.
Applying a user-approved abandonment preserves the original outcome and all
request evidence; it only adds the existing reversible recovery-suppression
marker. Releasing even one turn can let an entire existing account queue send
over time. This tool never submits a model request or rewrites a queued task.
Removing the marker later cannot undo model requests that used the freed slot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time


SUPPRESSION_REASON = "user_authorized_abandon_unknown_result"


class SuppressionRefused(ValueError):
    pass


def account_ref(identity: object) -> str:
    value = str(identity or "")
    return hashlib.sha256(value.encode()).hexdigest()[:24] if value else ""


def _receipt(db: sqlite3.Connection, request_id: str) -> tuple[str, str, dict]:
    rows = db.execute("SELECT owner,receipt FROM requests WHERE id=?", (request_id,)).fetchall()
    if len(rows) != 1:
        raise SuppressionRefused("original request ID is absent or ambiguous")
    owner, raw = rows[0]
    receipt = json.loads(raw)
    if not isinstance(receipt, dict) or receipt.get("request_id", request_id) != request_id:
        raise SuppressionRefused("original receipt identity changed")
    return owner, raw, receipt


def _validate(request_id: str, receipt: dict, expected_account_ref: str, now: float,
              decision_ref: str) -> bool:
    if not isinstance(request_id, str) or not request_id or len(request_id) > 200:
        raise SuppressionRefused("one exact original request ID is required")
    if account_ref(receipt.get("provider_account_identity")) != expected_account_ref:
        raise SuppressionRefused("original account identity changed")
    status = receipt.get("status")
    if not ((status == "unknown" and receipt.get("error_code") == "CONVERSATION_OUTCOME_UNKNOWN")
            or (status == "failed" and receipt.get("error_code") == "RESULT_UNRECOVERABLE"
                and receipt.get("upstream_outcome") == "unknown")):
        raise SuppressionRefused("original outcome changed")
    if receipt.get("upstream_outcome") not in (None, "unknown") or receipt.get("_upstream_terminal") is True:
        raise SuppressionRefused("original upstream outcome changed")
    if (receipt.get("_source") != "internal:listing" or receipt.get("_route", "chat") != "chat"
            or receipt.get("_operation", "text") != "text" or receipt.get("_forward_protocol")):
        raise SuppressionRefused("original route or source changed")
    if (receipt.get("original_failure_phase") != "stream_open"
            or receipt.get("original_http_status") != 404
            or receipt.get("original_exception_category") != "http"
            or receipt.get("original_upstream_request_stage") != "conversation"
            or receipt.get("recovery_error_code") != "CONVERSATION_BINDING_MISMATCH"
            or receipt.get("recovery_reason") != "REQUEST_MESSAGE_NOT_FOUND"):
        raise SuppressionRefused("original HTTP or recovery evidence changed")
    if any(not receipt.get(key) for key in (
            "provider_binding_id", "provider_account_identity", "client_conversation_id",
            "conversation_id", "parent_message_id", "request_message_id", "_input_ref")):
        raise SuppressionRefused("original identity or input reference is incomplete")
    if receipt.get("_submission_started") is not True:
        raise SuppressionRefused("original send evidence changed")
    stages = [item for item in receipt.get("_execution_timeline", []) if isinstance(item, dict)]
    send_indices = [i for i, item in enumerate(stages) if item.get("stage") == "send_call_started"]
    if not send_indices or not any(
            i > send_indices[0] and item.get("stage") == "response_headers_received"
            and item.get("status_code") == 404 and item.get("upstream_request_id")
            for i, item in enumerate(stages)):
        raise SuppressionRefused("original POST response evidence changed")
    if receipt.get("_executing") is True or (receipt.get("_claim_id")
            and (not isinstance(receipt.get("_claim_until"), (int, float))
                 or float(receipt["_claim_until"]) > now)):
        raise SuppressionRefused("original execution claim is active or uncertain")
    if receipt.get("recovery_claim_id"):
        raise SuppressionRefused("original recovery claim is active or uncertain")
    if receipt.get("_recovery_suppressed") is True:
        if (receipt.get("_recovery_suppressed_reason") == SUPPRESSION_REASON
                and (not decision_ref or receipt.get("_recovery_suppressed_decision_ref") == decision_ref)):
            return True
        raise SuppressionRefused("original recovery was already stopped for another reason")
    return False


def _waiting_impact(db: sqlite3.Connection, owner: str, request_id: str, original: dict) -> dict:
    bound = unbound = linked = 0
    bound_listing = unbound_listing = 0
    bound_ready_input = unbound_ready_input = 0
    identity = original["provider_account_identity"]
    for waiting_owner, waiting_id, raw in db.execute("SELECT owner,id,receipt FROM requests"):
        if waiting_id == request_id:
            continue
        waiting = json.loads(raw)
        if (waiting.get("status") not in {"queued", "not_started"}
                or waiting.get("_route", "chat") != "chat" or waiting.get("_operation", "text") != "text"):
            continue
        if waiting_owner == owner and (
                waiting.get("client_conversation_id") == original.get("client_conversation_id")
                or waiting.get("conversation_id") == original.get("conversation_id")
                or waiting.get("parent_message_id") == original.get("request_message_id")
                or waiting.get("_previous_request_id") == request_id):
            linked += 1
        waiting_identity = waiting.get("provider_account_identity")
        if waiting_identity == identity:
            bound += 1
            bound_listing += waiting.get("_source") == "internal:listing"
            bound_ready_input += bool(waiting.get("_input_ref"))
        elif not waiting_identity:
            unbound += 1
            unbound_listing += waiting.get("_source") == "internal:listing"
            unbound_ready_input += bool(waiting.get("_input_ref"))
        else:
            continue
    return {"bound_waiters": bound, "bound_listing_waiters": bound_listing,
            "bound_waiters_with_input": bound_ready_input,
            "unbound_waiters": unbound, "unbound_listing_waiters": unbound_listing,
            "unbound_waiters_with_input": unbound_ready_input,
            "same_original_conversation_waiters": linked,
            "potential_existing_text_sends": bound + unbound}


def inspect_or_suppress(db_path: Path, request_id: str, expected_account_ref: str, *,
                        apply: bool = False, decision_ref: str = "",
                        expected_status: str = "", expected_bound_waiters: int | None = None,
                        expected_unbound_waiters: int | None = None,
                        clock=time.time) -> dict:
    if not db_path.is_file():
        raise SuppressionRefused("existing receipt database is missing")
    if apply and (not re.fullmatch(r"[A-Za-z0-9_.:-]{3,80}", decision_ref)
                  or expected_status not in {"unknown", "failed"}
                  or expected_bound_waiters is None or expected_unbound_waiters is None
                  or expected_bound_waiters < 0 or expected_unbound_waiters < 0):
        raise SuppressionRefused("apply requires a decision reference and exact dry-run expectations")
    mode = "rw" if apply else "ro"
    db = sqlite3.connect(f"file:{db_path.resolve(strict=True).as_posix()}?mode={mode}", uri=True, timeout=30)
    try:
        db.execute("PRAGMA busy_timeout=30000")
        if not apply:
            db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN IMMEDIATE" if apply else "BEGIN")
        owner, raw, receipt = _receipt(db, request_id)
        now = float(clock())
        already_suppressed = _validate(request_id, receipt, expected_account_ref, now, decision_ref)
        impact = _waiting_impact(db, owner, request_id, receipt)
        result = {"request_id": request_id, "account_ref": expected_account_ref,
                  "original_status": receipt["status"], "original_error_code": receipt["error_code"],
                  "upstream_outcome": receipt.get("upstream_outcome") or "unknown",
                  "mode": "apply" if apply else "dry-run", "already_suppressed": already_suppressed,
                  **({"decision_ref": decision_ref} if apply else {}),
                  **impact}
        if apply and not already_suppressed:
            if (receipt["status"] != expected_status
                    or impact["bound_waiters"] != expected_bound_waiters
                    or impact["unbound_waiters"] != expected_unbound_waiters
                    or impact["same_original_conversation_waiters"]):
                raise SuppressionRefused("original status, queue impact, or session dependency changed")
            updated = {**receipt, "_recovery_suppressed": True,
                       "_recovery_suppressed_reason": SUPPRESSION_REASON,
                       "_recovery_suppressed_decision_ref": decision_ref,
                       "_recovery_suppressed_at": now, "updated_at": now}
            cursor = db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=? AND receipt=?",
                                (json.dumps(updated, ensure_ascii=False, separators=(",", ":")),
                                 owner, request_id, raw))
            if cursor.rowcount != 1:
                raise SuppressionRefused("original receipt changed during the transaction")
            result["suppression_written"] = True
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    if apply:
        with sqlite3.connect(f"file:{db_path.resolve(strict=True).as_posix()}?mode=ro", uri=True) as check:
            _, _, saved = _receipt(check, request_id)
        if (saved.get("_recovery_suppressed") is not True
                or saved.get("_recovery_suppressed_reason") != SUPPRESSION_REASON
                or saved.get("_recovery_suppressed_decision_ref") != decision_ref):
            raise SuppressionRefused("post-commit original ID readback failed")
        result["readback_suppressed"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--expected-account-ref", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--decision-ref", default="")
    parser.add_argument("--expected-status", default="")
    parser.add_argument("--expected-bound-waiters", type=int)
    parser.add_argument("--expected-unbound-waiters", type=int)
    args = parser.parse_args()
    try:
        result = inspect_or_suppress(
            args.data_root / "text_tasks.sqlite3", args.request_id, args.expected_account_ref,
            apply=args.apply, decision_ref=args.decision_ref, expected_status=args.expected_status,
            expected_bound_waiters=args.expected_bound_waiters,
            expected_unbound_waiters=args.expected_unbound_waiters,
        )
    except (SuppressionRefused, sqlite3.Error, ValueError) as exc:
        parser.exit(2, f"refused: {exc}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
