"""Owner-scoped pre-send hold is durable and never pauses original reads."""
import json
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from api import ai
from api.company_requests import PREFIX, company_identity
from services.request_context import current_request
from services.task_store import SendHoldConflict, SendHoldRejected, TaskStore
from services.text_task_service import TextTaskService
from test.test_company_requests import CONNECTOR, body, company
from test.test_text_task_service import QueuedExecutor


def _raw(service, owner, request_id):
    with service.store.connect() as db:
        return service.store.read_receipt(db, "text", owner, request_id)


def test_company_send_hold_is_owner_scoped_durable_and_projects_publicly(company):
    owner = company_identity("company", "employee", CONNECTOR)["id"]
    company.text_tasks.admission = Mock()
    accepted = company.client.post(PREFIX + "/api/chat-requests", headers=company.headers(), json=body())
    assert accepted.status_code == 202, accepted.text
    path = PREFIX + "/api/chat-requests/original-chat/send-hold"
    held = company.client.post(path, headers=company.headers(), json={"state": "held", "expected_version": 0})
    assert held.status_code == 200, held.text
    assert held.json()["send_hold"] == {"state": "held", "version": 1}
    assert held.headers["cache-control"] == "private, no-store"
    assert company.client.post(path, headers=company.headers(user="other"), json={"state": "active", "expected_version": 1}).status_code == 404
    assert company.client.post(path, headers=company.headers(), json={"state": "held"}).status_code == 422
    assert company.client.post(path, headers=company.headers(), json={"state": "held", "owner": owner}).status_code == 422
    read = company.client.get(PREFIX + "/api/chat-requests/original-chat", headers=company.headers())
    assert read.status_code == 200 and read.json()["send_hold"] == {"state": "held", "version": 1}
    restarted = TextTaskService(company.text_tasks.path)
    assert _raw(restarted, owner, "original-chat")["_send_hold"] is True
    released = company.client.post(path, headers=company.headers(), json={"state": "active", "expected_version": 1})
    assert released.status_code == 200 and released.json()["send_hold"]["version"] == 2


def test_send_hold_api_requires_admission_and_old_executor_respects_durable_hold(company):
    owner = company_identity("company", "employee", CONNECTOR)["id"]
    accepted = company.client.post(PREFIX + "/api/chat-requests", headers=company.headers(), json=body())
    assert accepted.status_code == 202, accepted.text
    path = PREFIX + "/api/chat-requests/original-chat/send-hold"
    rejected = company.client.post(path, headers=company.headers(), json={"state": "held", "expected_version": 0})
    assert rejected.status_code == 503 and rejected.json()["detail"]["code"] == "SEND_HOLD_ADMISSION_REQUIRED"
    company.text_tasks.store.set_text_send_hold(owner, "original-chat", True, expected_version=0)
    company.queue.run()
    held = _raw(company.text_tasks, owner, "original-chat")
    assert held["status"] == "queued" and held["_send_hold"] is True
    company.runner.assert_not_called()


def test_legacy_bound_send_hold_uses_authenticated_owner_only(company, monkeypatch):
    service = TextTaskService(company.text_tasks.path, runner=company.runner, executor=company.queue)
    service.admission = Mock()
    monkeypatch.setattr(ai, "text_task_service", service)
    owner = company.auth.authenticate(company.admin)["id"]
    service.submit(owner, {"client_request_id": "legacy", "client_conversation_id": "legacy-session", "model": "gpt-text",
                           "messages": [{"role": "user", "content": "private"}]})
    path = "/api/conversation-bindings/text-requests/legacy/send-hold"
    held = company.client.post(path, headers={"Authorization": "Bearer " + company.admin}, json={"state": "held", "expected_version": 0})
    assert held.status_code == 200, held.text
    assert _raw(service, owner, "legacy")["_send_hold"] is True
    assert company.client.post(path, headers={"Authorization": "Bearer " + company.ordinary},
                               json={"state": "active", "expected_version": 1}).status_code == 404


def test_legacy_api_hold_then_release_drives_real_pool_claim_and_same_id_send(company, monkeypatch, tmp_path):
    from test.test_pool_admission import build

    clock = lambda: 1000.0
    pool_root = tmp_path / "real-pool"
    pool_root.mkdir()
    (pool_root / "accounts.json").write_text(json.dumps([{
        "access_token": "fixture-token", "account_id": "upstream-0", "provider_account_identity": "account-0",
        "type": "Plus", "status": "正常", "quota": 999, "source_type": "web",
        "conversation_binding_ids": ["binding-0"],
        "capacity_observed_at": datetime.now(timezone.utc).isoformat(),
    }]))
    _accounts, store, admission = build(pool_root, clock)
    calls = []

    def runner(_body, *, on_cursor):
        context = current_request.get()
        context.before_send()
        calls.append((context.owner, context.request_id))
        return {"content": "saved original"}

    service = TextTaskService(store.path, runner=runner, admission=admission, clock=clock)
    admission.register("text", lambda context, body: service._run(context.owner, context.request_id, body))
    monkeypatch.setattr(ai, "text_task_service", service)
    owner = company.auth.authenticate(company.admin)["id"]
    service.submit(owner, {"client_request_id": "legacy-real", "client_conversation_id": "legacy-real-session",
                           "model": "fixture-text", "messages": [{"role": "user", "content": "private"}]})
    with service.store.connect() as db:
        request_hash = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", (owner, "legacy-real")).fetchone()[0]
    path = "/api/conversation-bindings/text-requests/legacy-real/send-hold"
    held = company.client.post(path, headers={"Authorization": "Bearer " + company.admin},
                               json={"state": "held", "expected_version": 0})
    assert held.status_code == 200, held.text
    assert admission.claim_next() is None and calls == []
    released = company.client.post(path, headers={"Authorization": "Bearer " + company.admin},
                                   json={"state": "active", "expected_version": 1})
    assert released.status_code == 200, released.text
    context = admission.claim_next()
    assert (context.owner, context.request_id) == (owner, "legacy-real")
    admission.execute(context)
    with service.store.connect() as db:
        assert db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", (owner, "legacy-real")).fetchone()[0] == request_hash
    assert calls == [(owner, "legacy-real")]
    assert _raw(service, owner, "legacy-real")["status"] == "succeeded"


@pytest.mark.parametrize("changes,code", [
    ({"_submission_started": True}, "SEND_HOLD_ALREADY_SUBMITTED"),
    ({"status": "unknown", "upstream_outcome": "unknown"}, "SEND_HOLD_ALREADY_SUBMITTED"),
    ({"status": "succeeded", "upstream_outcome": "completed", "content": "saved"}, "SEND_HOLD_ALREADY_SUBMITTED"),
    ({"status": "running", "_claim_id": "live", "_claim_until": 9999999999, "_executing": True, "_turn_reserved": True}, "SEND_HOLD_IN_FLIGHT"),
])
def test_send_hold_rejects_sent_unknown_success_or_live_claim_without_mutation(tmp_path, changes, code):
    # This checks the hold transaction, not an independently executing sender.
    # Retain the accepted input without racing the real background executor.
    queue = QueuedExecutor()
    runner = Mock(side_effect=AssertionError("rejection fixture must not generate"))
    service = TextTaskService(tmp_path / "tasks.sqlite3", runner=runner, executor=queue)
    service.submit("owner", {"client_request_id": "original", "client_conversation_id": "original-session", "model": "gpt-text",
                              "messages": [{"role": "user", "content": "private"}]})
    with service.store.transaction() as db:
        receipt = service.store.read_receipt(db, "text", "owner", "original")
        receipt.update(changes)
        service.store.write_receipt(db, "text", "owner", "original", receipt)
    before = _raw(service, "owner", "original")
    with pytest.raises(SendHoldRejected) as error:
        service.store.set_text_send_hold("owner", "original", True, expected_version=0)
    assert error.value.code == code
    assert _raw(service, "owner", "original") == before
    runner.assert_not_called()
    assert len(queue.calls) == 1


def test_send_hold_cas_cannot_allow_a_late_release(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    with store.transaction() as db:
        db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("owner", "original", "hash", json.dumps({
            "request_id": "original", "status": "queued", "_route": "chat", "_operation": "text",
            "_input_ref": "retained", "_submission_started": False, "_turn_reserved": False,
            "_executing": False, "_claim_id": None, "_execution_timeline": [{"stage": "accepted"}],
        })))
    first = store.set_text_send_hold("owner", "original", True, expected_version=0)
    repeated = store.set_text_send_hold("owner", "original", True, expected_version=first["version"])
    with pytest.raises(SendHoldConflict):
        store.set_text_send_hold("owner", "original", False, expected_version=first["version"])
    assert repeated["version"] == 2


def test_send_hold_release_rejects_elapsed_deadline_before_dispatcher_sweeps(tmp_path, monkeypatch):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    with store.transaction() as db:
        db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("owner", "original", "hash", json.dumps({
            "request_id": "original", "status": "queued", "_route": "chat", "_operation": "text",
            "_input_ref": "retained", "_submission_started": False, "_turn_reserved": False,
            "_executing": False, "_claim_id": None, "_execution_timeline": [{"stage": "accepted"}],
            "_scheduling": {"wait_deadline": "1970-01-01T00:16:41+00:00"},
        })))
    monkeypatch.setattr("services.task_store.time.time", lambda: 1000.0)
    held = store.set_text_send_hold("owner", "original", True, expected_version=0)
    monkeypatch.setattr("services.task_store.time.time", lambda: 1002.0)
    with pytest.raises(SendHoldRejected) as error:
        store.set_text_send_hold("owner", "original", False, expected_version=held["version"])
    assert error.value.code == "SEND_HOLD_WAIT_DEADLINE_EXCEEDED"
    assert error.value.control["state"] == "expired"
    with store.connect() as db:
        assert store.read_receipt(db, "text", "owner", "original")["_send_hold"] is True


def test_send_hold_does_not_pause_an_independent_unknown_original_read(tmp_path):
    from test.test_unknown_turn_recovery import migration

    service, _admission, backend, _legacy = migration(tmp_path)
    service.submit("owner", {"client_request_id": "held", "client_conversation_id": "held-session",
                             "model": "fixture-text", "messages": [{"role": "user", "content": "private"}]})
    service.store.set_text_send_hold("owner", "held", True, expected_version=0)
    result = service.read("owner", "old-0")
    assert result["status"] in {"unknown", "succeeded"}
    assert backend._get_conversation.called
    held = _raw(service, "owner", "held")
    assert held["_send_hold"] is True and held.get("_recovery_paused") is not True
