"""Regression coverage for sent text receipts that retained an old unsent outcome."""

import json

import pytest

from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_expired_execution_wait import waiter
from test.test_stalled_text_diagnostics import incomplete_reader, stalled
from test.test_unknown_turn_recovery import migration


def receipt(service, request_id="old-0"):
    with service.store.connect() as db:
        return service.store.read_receipt(db, "text", "owner", request_id)


def update_receipt(service, request_id="old-0", **changes):
    with service.store.transaction() as db:
        saved = service.store.read_receipt(db, "text", "owner", request_id)
        saved.update(changes)
        service.store.write_receipt(db, "text", "owner", request_id, saved)


def submit_ordinary_text(service):
    service.submit("owner", {
        "client_request_id": "ordinary", "client_conversation_id": "ordinary-session",
        "model": "fixture-text", "messages": [{"role": "user", "content": "fixture input"}],
    })


def historical_stalled_not_sent(tmp_path):
    service, admission, backend, legacy = migration(tmp_path)
    update_receipt(service, upstream_outcome="not_sent", _submission_started=True,
                   _turn_reserved=True, _executing=False)
    stalled(service, admission, backend)
    return service, admission, backend, legacy[0]


def test_ordinary_before_send_replaces_stale_not_sent_only_after_the_guard(tmp_path):
    service, admission, _, _ = migration(tmp_path, count=0)
    submit_ordinary_text(service)
    update_receipt(service, "ordinary", upstream_outcome="not_sent")

    context = admission.claim_next()
    assert context and context.request_id == "ordinary"
    context.before_send()

    saved = receipt(service, "ordinary")
    assert saved["_submission_started"] is True
    assert saved["upstream_outcome"] == "unknown"


def test_failed_before_send_leaves_stale_not_sent_and_never_calls_runner(tmp_path):
    service, admission, _, _ = migration(tmp_path, count=0)
    calls = []

    def runner(body, _on_cursor):
        current_request.get().before_send()
        calls.append(body)
        return {"content": "must not run"}

    service.runner = runner
    admission.register("text", lambda context, body: service._run(context.owner, context.request_id, body))
    submit_ordinary_text(service)
    update_receipt(service, "ordinary", upstream_outcome="not_sent")
    context = admission.claim_next()
    assert context and context.request_id == "ordinary"

    accounts_path = tmp_path / "accounts.json"
    accounts = json.loads(accounts_path.read_text())
    for account in accounts:
        account["status"] = "异常"
    accounts_path.write_text(json.dumps(accounts))
    admission.execute(context)

    saved = receipt(service, "ordinary")
    assert calls == []
    assert saved["_submission_started"] is False
    assert saved["upstream_outcome"] == "not_sent"


def test_unknown_callback_repairs_sent_stale_outcome_but_cannot_reopen_success(tmp_path):
    service, _, _, _ = migration(tmp_path)
    update_receipt(service, upstream_outcome="not_sent", _submission_started=True)

    service._update("owner", "old-0", status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN")
    repaired = receipt(service)
    assert repaired["status"] == "unknown"
    assert repaired["error_code"] == "CONVERSATION_OUTCOME_UNKNOWN"
    assert repaired["upstream_outcome"] == "unknown"

    service._update("owner", "old-0", status="succeeded", content="late original", upstream_outcome="completed")
    service._update("owner", "old-0", status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                    upstream_outcome="unknown")
    terminal = receipt(service)
    assert terminal["status"] == "succeeded"
    assert terminal["content"] == "late original"
    assert terminal["upstream_outcome"] == "completed"


def test_unknown_callback_uses_its_merged_submission_and_error_state(tmp_path):
    service, _, _, _ = migration(tmp_path)
    update_receipt(service, upstream_outcome="not_sent", _submission_started=False,
                   error_code="CONVERSATION_BINDING_UNAVAILABLE")

    service._update("owner", "old-0", status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                    _submission_started=True)

    repaired = receipt(service)
    assert repaired["_submission_started"] is True
    assert repaired["error_code"] == "CONVERSATION_OUTCOME_UNKNOWN"
    assert repaired["upstream_outcome"] == "unknown"


@pytest.mark.parametrize("case, changes", [
    ("truly_unsent", {"_submission_started": False}),
    ("non_chat_route", {"_route": "codex"}),
    ("forward_protocol", {"_forward_protocol": "openai_v1_chat_completions"}),
])
def test_unknown_callback_does_not_correct_unsent_or_nonordinary_text(tmp_path, case, changes):
    service, _, _, _ = migration(tmp_path)
    update_receipt(service, upstream_outcome="not_sent", **{"_submission_started": True, **changes})

    service._update("owner", "old-0", status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN")

    saved = receipt(service)
    assert saved["status"] == "unknown"
    assert saved["upstream_outcome"] == "not_sent"


@pytest.mark.parametrize("reads", [0, 2, 3])
def test_sent_unknown_history_with_stale_not_sent_releases_only_account_capacity(tmp_path, reads):
    service, admission, backend, original = historical_stalled_not_sent(tmp_path)
    # Capacity uses the qualified observation window; successor creation keeps
    # its separate unchanged-read requirement.
    update_receipt(service, _result_no_progress_reads=reads)
    saved = receipt(service)

    assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(saved, admission.clock.now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert saved["status"] == "unknown"
    assert saved["error_code"] == "CONVERSATION_OUTCOME_UNKNOWN"
    assert saved["upstream_outcome"] == "not_sent"
    for key in ("request_id", "request_message_id", "conversation_id", "provider_account_identity"):
        assert saved[key] == original[key]
    waiter(service, "independent-next", "independent")
    assert admission.claim_next().request_id == "independent-next"
    assert backend.mock_calls == []


@pytest.mark.parametrize("case, changes", [
    ("missing_stall_evidence", {"_result_no_progress_reads": None}),
    ("incomplete_recovery_lease", {"recovery_claim_id": "active", "recovery_lease_until": 999999}),
    ("truly_unsent", {"_submission_started": False}),
    ("missing_unknown_error", {"error_code": "RESULT_UNRECOVERABLE"}),
    ("not_sent_failure", {"status": "failed"}),
    ("other_outcome", {"upstream_outcome": "rejected"}),
])
def test_stale_not_sent_compatibility_keeps_all_other_stalled_guards(tmp_path, case, changes):
    service, admission, backend, _ = historical_stalled_not_sent(tmp_path)
    if case == "incomplete_recovery_lease":
        changes = {**changes, "recovery_lease_until": admission.clock.now + 1}
    update_receipt(service, **changes)
    saved = receipt(service)

    assert not TextTaskService._qualified_stalled_incomplete_capacity_boundary(saved, admission.clock.now)
    assert saved["upstream_outcome"] == changes.get("upstream_outcome", "not_sent")
    assert backend.mock_calls == []


def test_stale_not_sent_qualified_original_read_lease_keeps_capacity_released(tmp_path):
    service, admission, backend, _ = historical_stalled_not_sent(tmp_path)
    now = admission.clock.now
    update_receipt(service, recovery_claim_id="read-claim", recovery_claimed_at=now,
                   recovery_lease_until=now + 10, recovery_phase="read_text_request")
    saved = receipt(service)
    assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(saved, now)
    assert saved["upstream_outcome"] == "not_sent" and saved["status"] == "unknown"
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert backend.mock_calls == []
