import json
from pathlib import Path
import sqlite3
import tempfile

import pytest

from scripts.suppress_original_listing_recovery import (
    SuppressionRefused, account_ref, inspect_or_suppress,
)
from services.pool_admission import unknown_text_result


REQUEST_ID = "synthetic-original-request"
ACCOUNT = "synthetic-account-identity"
ACCOUNT_REF = account_ref(ACCOUNT)


@pytest.fixture
def ledger():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "text_tasks.sqlite3"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE requests (owner TEXT, id TEXT, request_hash TEXT, receipt TEXT, PRIMARY KEY(owner,id))")
            original = {
                "status": "unknown", "error_code": "CONVERSATION_OUTCOME_UNKNOWN",
                "upstream_outcome": "unknown", "_source": "internal:listing", "_route": "chat",
                "_operation": "text", "original_failure_phase": "stream_open",
                "original_http_status": 404, "original_exception_category": "http",
                "original_upstream_request_stage": "conversation",
                "recovery_error_code": "CONVERSATION_BINDING_MISMATCH",
                "recovery_reason": "REQUEST_MESSAGE_NOT_FOUND", "provider_binding_id": "binding-a",
                "provider_account_identity": ACCOUNT, "client_conversation_id": "product-a",
                "conversation_id": "conversation-a", "parent_message_id": "parent-a",
                "request_message_id": "message-a", "_input_ref": "original-input.json",
                "_submission_started": True, "_claim_id": "expired-claim", "_claim_until": 900,
                "_executing": False, "recovery_claim_id": None,
                "_execution_timeline": [
                    {"stage": "send_call_started", "at": 1},
                    {"stage": "response_headers_received", "at": 2, "status_code": 404,
                     "upstream_request_id": "upstream-a"},
                ],
                "private_note": "SECRET_PROMPT",
            }
            db.execute("INSERT INTO requests VALUES(?,?,?,?)",
                       ("owner-a", REQUEST_ID, "hash-a", json.dumps(original)))
            for index, identity in enumerate((ACCOUNT, ACCOUNT, "")):
                waiting = {"status": "queued", "_source": "internal:listing", "_route": "chat",
                           "_operation": "text", "provider_account_identity": identity,
                           "client_conversation_id": f"product-{index+1}",
                           "conversation_id": f"conversation-{index+1}",
                           "parent_message_id": f"parent-{index+1}", "_input_ref": f"waiting-{index}.json",
                           "_submission_started": False}
                db.execute("INSERT INTO requests VALUES(?,?,?,?)",
                           ("owner-b", f"waiting-{index}", f"hash-{index}", json.dumps(waiting)))
        yield path


def read(path, request_id=REQUEST_ID):
    with sqlite3.connect(path) as db:
        return json.loads(db.execute("SELECT receipt FROM requests WHERE id=?", (request_id,)).fetchone()[0])


def change(path, request_id=REQUEST_ID, **fields):
    with sqlite3.connect(path) as db:
        receipt = read(path, request_id)
        receipt.update(fields)
        db.execute("UPDATE requests SET receipt=? WHERE id=?", (json.dumps(receipt), request_id))


def apply(path, **kwargs):
    return inspect_or_suppress(
        path, REQUEST_ID, ACCOUNT_REF, apply=True, decision_ref="DLV-INCIDENT-1",
        expected_status="unknown", expected_bound_waiters=2,
        expected_unbound_waiters=1, clock=lambda: 1000, **kwargs,
    )


def test_dry_run_is_read_only_and_reports_full_account_queue_impact(ledger):
    before = ledger.read_bytes()
    result = inspect_or_suppress(ledger, REQUEST_ID, ACCOUNT_REF, clock=lambda: 1000)
    assert result["mode"] == "dry-run"
    assert result["bound_waiters"] == result["bound_listing_waiters"] == 2
    assert result["unbound_waiters"] == result["unbound_listing_waiters"] == 1
    assert result["potential_existing_text_sends"] == 3
    assert result["same_original_conversation_waiters"] == 0
    assert ledger.read_bytes() == before
    assert "SECRET_PROMPT" not in json.dumps(result)
    assert ACCOUNT not in json.dumps(result)


def test_apply_only_adds_traceable_suppression_and_is_idempotent(ledger):
    before = read(ledger)
    assert unknown_text_result(before)
    result = apply(ledger)
    assert result["suppression_written"] is True
    assert result["readback_suppressed"] is True
    after = read(ledger)
    assert not unknown_text_result(after)
    for key, value in before.items():
        assert after[key] == value
    assert after["_recovery_suppressed"] is True
    assert after["_recovery_suppressed_reason"] == "user_authorized_abandon_unknown_result"
    assert after["_recovery_suppressed_decision_ref"] == "DLV-INCIDENT-1"
    again = apply(ledger)
    assert again["already_suppressed"] is True
    assert "suppression_written" not in again
    assert inspect_or_suppress(ledger, REQUEST_ID, ACCOUNT_REF, clock=lambda: 1000)["already_suppressed"] is True
    assert read(ledger) == after


def test_unrecoverable_failed_receipt_still_preserves_unknown_upstream_outcome(ledger):
    change(ledger, status="failed", error_code="RESULT_UNRECOVERABLE",
           upstream_outcome="unknown")
    result = inspect_or_suppress(
        ledger, REQUEST_ID, ACCOUNT_REF, apply=True, decision_ref="DLV-INCIDENT-1",
        expected_status="failed", expected_bound_waiters=2,
        expected_unbound_waiters=1, clock=lambda: 1000,
    )
    assert result["readback_suppressed"] is True
    assert read(ledger)["status"] == "failed"
    assert read(ledger)["error_code"] == "RESULT_UNRECOVERABLE"
    assert read(ledger)["upstream_outcome"] == "unknown"


@pytest.mark.parametrize("fields", [
    {"status": "succeeded"},
    {"_claim_until": 1100},
    {"recovery_claim_id": "live-recovery"},
    {"recovery_reason": "REQUEST_RESULT_INCOMPLETE"},
    {"original_http_status": 422},
])
def test_changed_outcome_claim_or_evidence_refuses_without_writing(ledger, fields):
    change(ledger, **fields)
    before = read(ledger)
    with pytest.raises(SuppressionRefused):
        apply(ledger)
    assert read(ledger) == before


def test_account_or_queue_impact_change_refuses_without_writing(ledger):
    with pytest.raises(SuppressionRefused):
        inspect_or_suppress(ledger, REQUEST_ID, account_ref("other"), clock=lambda: 1000)
    with pytest.raises(SuppressionRefused, match="queue impact"):
        inspect_or_suppress(
            ledger, REQUEST_ID, ACCOUNT_REF, apply=True, decision_ref="DLV-INCIDENT-1",
            expected_status="unknown", expected_bound_waiters=1,
            expected_unbound_waiters=1, clock=lambda: 1000,
        )
    assert read(ledger).get("_recovery_suppressed") is not True


def test_same_original_session_waiter_refuses_apply(ledger):
    change(ledger, "waiting-0", client_conversation_id="product-a")
    with sqlite3.connect(ledger) as db:
        db.execute("UPDATE requests SET owner=? WHERE id=?", ("owner-a", "waiting-0"))
    dry_run = inspect_or_suppress(ledger, REQUEST_ID, ACCOUNT_REF, clock=lambda: 1000)
    assert dry_run["same_original_conversation_waiters"] == 1
    with pytest.raises(SuppressionRefused, match="session dependency"):
        apply(ledger)
    assert read(ledger).get("_recovery_suppressed") is not True


def test_unrelated_id_and_other_decision_ref_are_rejected(ledger):
    with pytest.raises(SuppressionRefused, match="original outcome changed"):
        inspect_or_suppress(ledger, "waiting-0", ACCOUNT_REF, clock=lambda: 1000)
    apply(ledger)
    with pytest.raises(SuppressionRefused, match="another reason"):
        inspect_or_suppress(ledger, REQUEST_ID, ACCOUNT_REF, apply=True,
                            decision_ref="DLV-OTHER", expected_status="unknown",
                            expected_bound_waiters=2, expected_unbound_waiters=1,
                            clock=lambda: 1000)
