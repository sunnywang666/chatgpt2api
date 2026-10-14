"""Legacy internal Chat polling must stop without fabricating upstream outcome."""
from unittest.mock import Mock

import pytest

from services.text_task_service import TextTaskService
from services.conversation_binding_service import ConversationBindingService
from test.test_pool_admission import build
from test.test_text_task_service import ManualClock, QueuedExecutor


@pytest.fixture(params=[False, True], ids=["no-cursor", "saved-cursor"])
def legacy(tmp_path, request):
    clock = ManualClock()
    reader = Mock(return_value={"status": "unknown", "recovery_reason": "REQUEST_RESULT_NOT_FOUND"})
    tasks = TextTaskService(tmp_path / "text_tasks.sqlite3", clock=clock,
                            executor=QueuedExecutor(), recovery_reader=reader)
    tasks.submit("owner", {"client_request_id": "original", "client_conversation_id": "session",
                           "messages": [{"role": "user", "content": "original input"}]})
    tasks._update("owner", "original", status="unknown", upstream_outcome="unknown",
                  error_code="CONVERSATION_OUTCOME_UNKNOWN", provider_binding_id="binding",
                  provider_account_identity="account", conversation_id="chat", parent_message_id="parent",
                  request_parent_message_id="parent", _submission_started=True,
                  _last_sent_sequence=0, _executing=False, _claim_id=None, _claim_until=None)
    if request.param:
        proof = {"conversation_id": "chat", "request_message_id": raw(tasks)["request_message_id"],
                 "retry_parent_message_id": "original-user", "observed_at": clock()}
        tasks._update("owner", "original", _retry_cursor=proof)
        reader.return_value["_retry_cursor"] = proof
    return tasks, clock, reader


def raw(tasks):
    with tasks.store.connect() as db:
        return tasks.store.read_receipt(db, "text", "owner", "original")


def exhaust(tasks, clock):
    for _ in range(3):
        clock.advance(1000)
        tasks.read("owner", "original")


def test_no_final_budget_stops_normal_reads_and_survives_restart(legacy):
    tasks, clock, reader = legacy
    before = raw(tasks)
    exhaust(tasks, clock)
    stopped = raw(tasks)
    assert reader.call_count == 3
    assert stopped["_attempt_reason"] == "ORIGINAL_RESULT_NO_FINAL"
    assert stopped["recovery_next_at"] is None
    assert stopped["_turn_reserved"] is False
    assert stopped["status"] == "failed" and stopped["upstream_outcome"] == "unknown"
    for key in ("request_id", "request_message_id", "conversation_id", "provider_binding_id",
                "provider_account_identity", "client_conversation_id", "parent_message_id", "_last_sent_sequence"):
        assert stopped[key] == before[key]
    restarted = TextTaskService(tasks.path, clock=clock, executor=QueuedExecutor(), recovery_reader=reader)
    clock.advance(100000)
    for service in (tasks, restarted):
        result = service.read("owner", "original")
        assert result["execution"]["attempt_state"] == "ended"
        assert result["execution"]["resources"]["account_turn"] == "released"
    assert reader.call_count == 3 and raw(tasks)["recovery_no_result_reads"] == 3
    assert raw(tasks).get("_retry_cursor") == before.get("_retry_cursor")
    assert len(tasks.executor.calls) == 1  # No new generation after the original submit.
    if before.get("_retry_cursor"):
        assert tasks._no_final_successor_source(stopped, clock())


def test_existing_unbounded_receipt_is_closed_without_one_more_upstream_read(legacy):
    tasks, clock, reader = legacy
    clock.advance(10000)
    tasks._update("owner", "original", status="failed", error_code="RESULT_UNRECOVERABLE",
                  recovery_reason="REQUEST_RESULT_NOT_FOUND", recovery_no_result_reads=127,
                  recovery_attempt=132, _execution_wait_ended_at=2000, _turn_reserved=False)
    before = raw(tasks)
    tasks.read("owner", "original")
    assert reader.call_count == 0
    assert raw(tasks)["recovery_no_result_reads"] == 127
    assert raw(tasks)["_attempt_reason"] == "ORIGINAL_RESULT_NO_FINAL"
    assert raw(tasks).get("_retry_cursor") == before.get("_retry_cursor")
    assert raw(tasks)["upstream_outcome"] == "unknown"
    assert len(tasks.executor.calls) == 1


def test_explicit_recheck_is_one_read_and_preserves_stop_if_still_missing(legacy):
    tasks, clock, reader = legacy
    exhaust(tasks, clock)
    tasks.recover("owner", "original", explicit_ended_recheck=True)
    assert reader.call_count == 4 and raw(tasks)["recovery_next_at"] is None
    clock.advance(10000)
    tasks.read("owner", "original")
    assert reader.call_count == 4 and raw(tasks)["upstream_outcome"] == "unknown"


def test_explicit_recheck_adopts_late_final_without_new_submission(legacy):
    tasks, clock, reader = legacy
    exhaust(tasks, clock)
    reader.return_value = {"status": "succeeded", "binding_status": "bound", "content": "late original answer",
                           "conversation_id": "chat", "parent_message_id": "original-final"}
    result = tasks.recover("owner", "original", explicit_ended_recheck=True)
    assert result["status"] == "succeeded" and result["content"] == "late original answer"
    assert result["upstream_outcome"] == "completed"
    assert "recovery_automatic_stopped" not in result
    assert raw(tasks)["_last_sent_sequence"] == 0
    assert len(tasks.executor.calls) == 1  # Only the original submit was queued.


@pytest.mark.parametrize("guard", [
    {"_work_key": "work"}, {"_product_work_key": "product"},
    {"_completion": {"state": "checking_original"}}, {"_completion_of": "other"},
    {"_upstream_terminal": True}, {"recovery_claim_id": "live"}, {"_claim_id": "live"},
    {"_executing": True}, {"_forward_protocol": "responses"}, {"_recovery_paused": True},
    {"_recovery_suppressed": True}, {"recovery_reason": "REQUEST_RESULT_INCOMPLETE"},
    {"recovery_reason": "REQUEST_PARENT_MISMATCH"}, {"recovery_no_result_reads": 2},
    {"recovery_no_result_reads": True}, {"upstream_outcome": "completed"},
    {"conversation_id": None}, {"request_message_id": ""},
])
def test_does_not_close_other_recovery_contracts(legacy, guard):
    tasks, clock, _ = legacy
    receipt = {**raw(tasks), "status": "failed", "error_code": "RESULT_UNRECOVERABLE",
               "_execution_wait_ended_at": 2000, "recovery_reason": "REQUEST_RESULT_NOT_FOUND",
               "recovery_no_result_reads": 3, **guard}
    assert tasks._end_no_final_recovery(receipt, 3000) is False
    assert not receipt.get("_attempt_finished_at")


def test_background_recovery_skips_closed_original(legacy, tmp_path):
    tasks, clock, reader = legacy
    exhaust(tasks, clock)
    _, _, admission = build(tmp_path, clock)
    callback = Mock()
    admission.recoveries["text"] = callback
    clock.advance(100000)
    admission.recover_one()
    callback.assert_not_called()
    assert reader.call_count == 3


def test_failed_explicit_read_preserves_stop_and_cooldown(legacy):
    tasks, clock, reader = legacy
    exhaust(tasks, clock)
    reader.side_effect = RuntimeError("controlled read failure")
    tasks.recover("owner", "original", explicit_ended_recheck=True)
    assert reader.call_count == 4
    stopped = raw(tasks)
    assert stopped["_attempt_reason"] == "ORIGINAL_RESULT_NO_FINAL"
    assert stopped["recovery_next_at"] > clock()
    tasks.recover("owner", "original", explicit_ended_recheck=True)
    assert reader.call_count == 4
    clock.advance(10000)
    tasks.read("owner", "original")
    assert reader.call_count == 4


def test_bound_text_explicit_recovery_route_can_recheck_once(legacy, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api import ai

    tasks, clock, reader = legacy
    exhaust(tasks, clock)
    monkeypatch.setattr(ai, "text_task_service", tasks)
    monkeypatch.setattr(ai, "require_identity", lambda *_: {"id": "owner", "role": "admin"})
    app = FastAPI()
    app.include_router(ai.create_router())
    with TestClient(app) as client:
        response = client.post("/api/conversation-bindings/text-requests/original/recover", json={})
    assert response.status_code == 200 and reader.call_count == 4
    assert response.json()["recovery_automatic_stopped"] is True
    assert response.json()["recovery_stop_reason"] == "ORIGINAL_RESULT_NO_FINAL"
    clock.advance(10000)
    tasks.read("owner", "original")
    assert reader.call_count == 4


def test_public_stop_flag_requires_derived_proof(legacy):
    tasks, _, _ = legacy
    receipt = {**raw(tasks), "recovery_automatic_stopped": True,
               "recovery_stop_reason": "ORIGINAL_RESULT_NO_FINAL"}
    result = tasks._public(receipt)
    assert "recovery_automatic_stopped" not in result and "recovery_stop_reason" not in result


@pytest.fixture(params=[None, "_work_key", "_product_work_key"])
def missing_node(legacy, request):
    tasks, clock, reader = legacy
    backend = Mock()
    backend._get_conversation.return_value = {
        "conversation_id": "chat", "current_node": "parent", "mapping": {
            "parent": {"message": {"id": "parent", "author": {"role": "assistant"},
                                   "status": "finished_successfully", "end_turn": True}},
        },
    }
    reader.side_effect = lambda receipt: ConversationBindingService._read_text_request_result(backend, receipt)
    tasks._update("owner", "original", **({request.param: "original-work"} if request.param else {}),
                  original_http_status=404, original_failure_phase="stream_open", _retry_cursor=None)
    return tasks, clock, reader, backend


def test_missing_node_budget_stops_queries_but_keeps_unknown_and_work(missing_node, tmp_path):
    tasks, clock, reader, _ = missing_node
    before = raw(tasks)
    exhaust(tasks, clock)
    stopped = raw(tasks)
    assert reader.call_count == 3
    assert stopped["_attempt_reason"] == "ORIGINAL_REQUEST_MESSAGE_NOT_FOUND"
    assert stopped["recovery_next_at"] is None
    assert stopped["status"] == "failed" and stopped["upstream_outcome"] == "unknown"
    assert stopped.get("_upstream_terminal") is not True
    assert not tasks._no_final_successor_source(stopped, clock())
    for key in ("request_id", "request_message_id", "conversation_id", "provider_binding_id",
                "provider_account_identity", "client_conversation_id", "parent_message_id",
                "_input_ref", "_last_sent_sequence", "_work_key", "_product_work_key", "original_http_status"):
        assert stopped.get(key) == before.get(key)
    restarted = TextTaskService(tasks.path, clock=clock, executor=QueuedExecutor(), recovery_reader=reader)
    clock.advance(100000)
    result = restarted.read("owner", "original")
    assert result["recovery_automatic_stopped"] is True
    assert result["recovery_stop_reason"] == "ORIGINAL_REQUEST_MESSAGE_NOT_FOUND"
    assert result["execution"]["resources"]["conversation"] == "protected"
    _, _, admission = build(tmp_path, clock)
    callback = Mock()
    admission.recoveries["text"] = callback
    admission.recover_one()
    callback.assert_not_called()
    assert reader.call_count == 3 and len(tasks.executor.calls) == 1


def test_old_missing_node_receipt_stops_without_extra_get(missing_node):
    tasks, clock, reader, _ = missing_node
    clock.advance(10000)
    tasks._update("owner", "original", status="failed", error_code="RESULT_UNRECOVERABLE",
                  recovery_reason="REQUEST_MESSAGE_NOT_FOUND", recovery_no_result_reads=127,
                  recovery_attempt=132, _execution_wait_ended_at=2000, _turn_reserved=False)
    tasks.read("owner", "original")
    reader.assert_not_called()
    assert raw(tasks)["recovery_no_result_reads"] == 127
    assert raw(tasks)["_attempt_reason"] == "ORIGINAL_REQUEST_MESSAGE_NOT_FOUND"


def test_missing_node_explicit_recheck_and_late_final(missing_node):
    tasks, clock, reader, _ = missing_node
    exhaust(tasks, clock)
    tasks.recover("owner", "original", explicit_ended_recheck=True)
    assert reader.call_count == 4 and raw(tasks)["recovery_next_at"] is None
    reader.side_effect = RuntimeError("controlled read failure")
    tasks.recover("owner", "original", explicit_ended_recheck=True)
    assert reader.call_count == 5 and raw(tasks)["recovery_next_at"] > clock()
    tasks.recover("owner", "original", explicit_ended_recheck=True)
    assert reader.call_count == 5
    clock.advance(10000)
    tasks.read("owner", "original")
    assert reader.call_count == 5
    reader.side_effect = None
    reader.return_value = {"status": "succeeded", "binding_status": "bound", "content": "late original answer",
                           "conversation_id": "chat", "parent_message_id": "original-final"}
    result = tasks.recover("owner", "original", explicit_ended_recheck=True)
    assert reader.call_count == 6 and result["content"] == "late original answer"
    assert result["status"] == "succeeded" and result["upstream_outcome"] == "completed"
    assert "recovery_automatic_stopped" not in result
    assert len(tasks.executor.calls) == 1


@pytest.mark.parametrize("case", ["invalid_mapping", "active", "transport"])
def test_missing_node_budget_excludes_bad_reads_and_active_generation(missing_node, case):
    tasks, clock, reader, backend = missing_node
    if case == "invalid_mapping":
        backend._get_conversation.return_value["mapping"] = []
    elif case == "active":
        backend._get_conversation.return_value["mapping"]["parent"]["message"].update(
            status="in_progress", end_turn=False)
    else:
        backend._get_conversation.side_effect = TimeoutError("controlled transport failure")
    exhaust(tasks, clock)
    result = raw(tasks)
    assert reader.call_count == 3
    assert result.get("recovery_no_result_reads", 0) == 0
    assert not result.get("_attempt_finished_at")


@pytest.mark.parametrize("guard", [
    {"_completion": {"state": "checking_original"}}, {"_completion_of": "parent"},
    {"recovery_claim_id": "live"}, {"_claim_id": "live"}, {"_executing": True},
    {"_recovery_paused": True}, {"_recovery_suppressed": True},
])
def test_missing_node_stop_keeps_active_owners_and_leases(legacy, guard):
    tasks, _, _ = legacy
    receipt = {**raw(tasks), "status": "failed", "error_code": "RESULT_UNRECOVERABLE",
               "_execution_wait_ended_at": 2000, "recovery_reason": "REQUEST_MESSAGE_NOT_FOUND",
               "recovery_no_result_reads": 3, **guard}
    assert not tasks._end_no_final_recovery(receipt, 3000)
    assert not receipt.get("_attempt_finished_at")
