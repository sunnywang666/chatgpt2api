"""A stagnant original turn ends result waiting, never ownership or recovery."""
import copy
import json
import sqlite3
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import ai
from services.conversation_binding_service import ConversationBindingService
from services.public_chat_service import project_public_chat_receipt
from services.text_task_service import TextTaskService
from test.test_expired_execution_wait import saved, patch, waiter
from test.test_pool_admission import build
from test.test_unknown_turn_recovery import document, migration


def test_explicit_recovery_reports_real_sqlite_writer_contention_without_claim(tmp_path, monkeypatch):
    service, _, backend, _ = migration(tmp_path)
    before = saved(service)
    def read_original(row):
        doc = document(row)
        doc["mapping"][doc["current_node"]]["message"]["content"]["parts"] = ["original answer"]
        return ConversationBindingService._read_text_request_result(backend, row, document=doc)
    reader = Mock(side_effect=read_original)
    service.recovery_reader = reader
    connect = service.store._connect

    def short_busy_timeout(**kwargs):
        db = connect(**kwargs)
        db.execute("PRAGMA busy_timeout=10")
        return db

    monkeypatch.setattr(service.store, "_connect", short_busy_timeout)
    monkeypatch.setattr(ai, "text_task_service", service)
    monkeypatch.setattr(ai, "require_identity", lambda *_args, **_kwargs: {"id": "owner"})
    app = FastAPI()
    app.include_router(ai.create_router())
    client = TestClient(app)
    with sqlite3.connect(service.store.path) as writer:
        writer.execute("BEGIN IMMEDIATE")
        response = client.post("/api/conversation-bindings/text-requests/old-0/recover", json={})
        assert response.status_code == 503
        assert response.json()["detail"] == {
            "code": "TEXT_RECOVERY_LEASE_BUSY", "retryable": True,
            "lease_acquired": False, "upstream_read_started": False,
        }
        assert saved(service) == before
        reader.assert_not_called()
        backend._get_conversation.assert_not_called()
    response = client.post("/api/conversation-bindings/text-requests/old-0/recover", json={})
    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"
    assert response.json()["content"] == "original answer"
    assert reader.call_count == 1
    assert saved(service)["request_message_id"] == before["request_message_id"]


def test_explicit_recovery_does_not_translate_other_database_errors(tmp_path, monkeypatch):
    service, _, _, _ = migration(tmp_path)
    original_db = service._db
    calls = 0

    @contextmanager
    def failing_db():
        nonlocal calls
        calls += 1
        with original_db() as db:
            if calls == 2:
                class BrokenBegin:
                    def execute(self, _sql):
                        raise sqlite3.OperationalError("disk I/O error")
                yield BrokenBegin()
            else:
                yield db

    monkeypatch.setattr(service, "_db", failing_db)
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        service.recover("owner", "old-0")


def mixed_chain(doc):
    """Synthetic bodies reproduce only the live GET's verified envelope shapes."""
    final = doc["current_node"]
    parent = doc["mapping"][final]["parent"]
    envelopes = [
        ("code", "assistant", {"content_type": "code", "text": "PRIVATE_CODE"}),
        ("output", "tool", {"content_type": "execution_output", "text": "PRIVATE_TOOL"}),
        ("thought", "assistant", {"content_type": "thoughts", "thoughts": [
            {"summary": "PRIVATE_SUMMARY", "content": "", "chunks": [], "finished": False}]}),
        ("recap", "assistant", {"content_type": "reasoning_recap", "content": "PRIVATE_RECAP"}),
    ]
    for suffix, role, content in envelopes:
        node_id = final + "-" + suffix
        doc["mapping"][node_id] = {"parent": parent, "message": {
            "id": node_id, "author": {"role": role}, "status": "finished_successfully",
            "end_turn": False, "content": content}}
        parent = node_id
    doc["mapping"][final]["parent"] = parent


@pytest.mark.parametrize("reasoning", ["thought", "recap"])
def test_completed_final_after_stale_reasoning_recovers_original_without_resend(tmp_path, reasoning):
    service, admission, backend, legacy = migration(tmp_path)
    patch(service, _turn_reserved=True)
    original = saved(service)

    def read(row):
        doc = document(row)
        mixed_chain(doc)
        final = doc["current_node"]
        doc["mapping"][final + "-" + reasoning]["message"]["status"] = "in_progress"
        doc["mapping"][final]["message"]["content"]["parts"] = ["original final answer"]
        return ConversationBindingService._read_text_request_result(backend, row, document=doc)

    service.recovery_reader = read
    result = service.read("owner", "old-0")
    assert result["status"] == "succeeded"
    assert result["content"] == "original final answer"
    recovered = saved(service)
    assert all(recovered.get(key) == original.get(key) for key in (
        "request_message_id", "request_parent_message_id", "conversation_id", "provider_account_identity", "_input_ref"))
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert backend.mock_calls == []


@pytest.mark.parametrize("case", ["active_tool", "active_code", "active_final", "active_sibling"])
def test_stale_reasoning_allowance_does_not_accept_unfinished_or_branched_result(tmp_path, case):
    service, _, backend, _ = migration(tmp_path)
    row = saved(service)
    doc = document(row)
    mixed_chain(doc)
    final = doc["current_node"]
    doc["mapping"][final + "-thought"]["message"]["status"] = "in_progress"
    doc["mapping"][final]["message"]["content"]["parts"] = ["not sufficient by itself"]
    if case == "active_sibling":
        doc["mapping"]["sibling"] = {"parent": row["request_message_id"], "message": {
            "id": "sibling", "author": {"role": "assistant"}, "status": "in_progress",
            "end_turn": False, "content": {"content_type": "text", "parts": [""]}}}
    else:
        target = final + ("-output" if case == "active_tool" else "-code" if case == "active_code" else "")
        doc["mapping"][target]["message"].update(status="in_progress", end_turn=False)
    result = ConversationBindingService._read_text_request_result(backend, row, document=doc)
    assert result["status"] != "succeeded"
    assert "content" not in result


def completed_tool_chain(doc, reasoning_status=None):
    mixed_chain(doc)
    final = doc["current_node"]
    code = doc["mapping"][final + "-code"]["message"]
    code.update(status="in_progress", end_turn=None, recipient="python")
    tool = doc["mapping"][final + "-output"]["message"]
    tool["author"]["name"] = "python"
    tool["metadata"] = {"is_complete": True}
    if reasoning_status:
        node_id = final + "-tool-thought"
        doc["mapping"][node_id] = {"parent": final + "-code", "message": {
            "id": node_id, "author": {"role": "assistant"}, "recipient": "all",
            "status": reasoning_status, "end_turn": False,
            "content": {"content_type": "thoughts", "thoughts": []}}}
        doc["mapping"][final + "-output"]["parent"] = node_id
    doc["mapping"][final]["message"]["content"]["parts"] = ["original calculated stock review"]
    return final


@pytest.mark.parametrize("outcome", ["complete", "read_error", "empty"])
def test_explicit_recheck_of_ended_original_keeps_automatic_stop_and_never_resends(tmp_path, outcome):
    service, admission, backend, _ = migration(tmp_path)
    reason = "COMPLETION_ORIGINAL_READ_UNAVAILABLE"
    state = {"state": "needs_attention", "reason": reason, "next_at": None,
             "allow_unconfirmed_retry": False, "automatic_failure_retry": True, "max_extra_requests": 1}
    patch(service, _attempt_finished_at=1, _attempt_reason=reason, _completion=state)
    calls = []
    def read(row):
        calls.append(row["request_id"])
        if outcome == "read_error":
            raise ConnectionError("synthetic unavailable original")
        doc = document(row)
        if outcome == "complete":
            completed_tool_chain(doc)
        return ConversationBindingService._read_text_request_result(backend, row, document=doc)
    service.recovery_reader = read
    assert service.read("owner", "old-0")["status"] == "unknown"
    assert service.recover("owner", "old-0")["status"] == "unknown"
    assert calls == []
    result = service.recover("owner", "old-0", explicit_ended_recheck=True)
    assert calls == ["old-0"]
    assert result["status"] == ("succeeded" if outcome == "complete" else "unknown")
    row = saved(service)
    assert row["_attempt_finished_at"] == 1 and row["_attempt_reason"] == reason
    assert row["_completion"] == state
    if outcome == "complete":
        assert row["content"] == "original calculated stock review"
    else:
        assert row["recovery_next_at"] > admission.clock.now
        service.recover("owner", "old-0", explicit_ended_recheck=True)
    service.read("owner", "old-0")
    assert calls == ["old-0"] and backend.mock_calls == []


@pytest.mark.parametrize("case", ["paused", "suppressed", "cooldown", "lease", "child", "selected",
                                    "child_receipt", "other_reason", "other_state", "missing_completion"])
def test_explicit_ended_original_recheck_keeps_pause_cooldown_and_successor_guards(tmp_path, case):
    service, admission, _, _ = migration(tmp_path)
    reason = "COMPLETION_ORIGINAL_READ_UNAVAILABLE"
    state = {"state": "needs_attention", "reason": reason, "next_at": None}
    changes = {"_attempt_finished_at": 1, "_attempt_reason": reason, "_completion": state}
    if case == "paused": changes["_recovery_paused"] = True
    if case == "suppressed": changes["_recovery_suppressed"] = True
    if case == "cooldown": changes["recovery_next_at"] = admission.clock.now + 100
    if case == "lease": changes["recovery_lease_until"] = admission.clock.now + 100
    if case == "child": state["replacement_id"] = "existing-child"
    if case == "selected": state["selected_id"] = "selected-result"
    if case == "child_receipt": changes["_completion_of"] = "original-root"
    if case == "other_reason": changes["_attempt_reason"] = "COMPLETION_ATTEMPT_EXHAUSTED"
    if case == "other_state": state["state"] = "replacement_pending"
    if case == "missing_completion": changes["_completion"] = None
    patch(service, **changes)
    service.recovery_reader = lambda row: pytest.fail("must not read or submit")
    assert service.recover("owner", "old-0", explicit_ended_recheck=True)["status"] == "unknown"


@pytest.mark.parametrize("reasoning_status", [None, "in_progress", "finished_successfully"])
def test_completed_final_after_stale_paired_tool_call_recovers_original_without_resend(tmp_path, reasoning_status):
    service, admission, backend, _ = migration(tmp_path)
    patch(service, _turn_reserved=True)
    original = saved(service)

    def read(row):
        doc = document(row)
        completed_tool_chain(doc, reasoning_status)
        return ConversationBindingService._read_text_request_result(backend, row, document=doc)

    service.recovery_reader = read
    result = service.read("owner", "old-0")
    assert result["status"] == "succeeded"
    assert result["content"] == "original calculated stock review"
    recovered = saved(service)
    assert all(recovered.get(key) == original.get(key) for key in (
        "request_message_id", "request_parent_message_id", "conversation_id",
        "provider_account_identity", "provider_binding_id", "_input_ref"))
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert backend.mock_calls == []


@pytest.mark.parametrize("case", [
    "missing_recipient", "different_tool", "tool_active", "tool_failed",
    "tool_not_complete", "wrong_tool_content", "wrong_tool_id", "code_end_turn",
    "code_final", "second_active_code", "active_sibling", "final_active", "empty_final",
    "generic_recipient", "malformed_tool_node",
])
@pytest.mark.parametrize("reasoning_status", [None, "in_progress"])
def test_stale_tool_call_requires_paired_complete_output_and_nonempty_final(tmp_path, case, reasoning_status):
    service, _, backend, _ = migration(tmp_path)
    row = saved(service)
    doc = document(row)
    final = completed_tool_chain(doc, reasoning_status)
    code = doc["mapping"][final + "-code"]["message"]
    tool = doc["mapping"][final + "-output"]["message"]
    if case == "missing_recipient":
        code.pop("recipient")
    elif case == "generic_recipient":
        code["recipient"] = tool["author"]["name"] = " all "
    elif case == "malformed_tool_node":
        doc["mapping"][final + "-output"] = 42
    elif case == "different_tool":
        tool["author"]["name"] = "another-tool"
    elif case in {"tool_active", "tool_failed"}:
        tool["status"] = "in_progress" if case == "tool_active" else "failed"
    elif case == "tool_not_complete":
        tool["metadata"]["is_complete"] = False
    elif case == "wrong_tool_content":
        tool["content"] = {"content_type": "text", "parts": ["not tool output"]}
    elif case == "wrong_tool_id":
        tool["id"] = "another-id"
    elif case == "code_end_turn":
        code["end_turn"] = True
    elif case == "code_final":
        code["channel"] = "final"
    elif case == "second_active_code":
        tool.update(author={"role": "assistant"}, status="in_progress",
                    content={"content_type": "code", "text": "pending call"}, recipient="python")
    elif case == "active_sibling":
        doc["mapping"]["sibling"] = {"parent": final + "-code", "message": {
            "id": "sibling", "author": {"role": "tool", "name": "python"},
            "status": "in_progress", "content": {"content_type": "execution_output", "text": ""}}}
    elif case == "final_active":
        doc["mapping"][final]["message"].update(status="in_progress", end_turn=False)
    else:
        doc["mapping"][final]["message"]["content"]["parts"] = [""]
    result = ConversationBindingService._read_text_request_result(backend, row, document=doc)
    assert result["status"] != "succeeded"
    assert "content" not in result and "_turn_end_evidence" not in result
    assert "_empty_reply_evidence" not in result


@pytest.mark.parametrize("case", ["branch", "wrong_id", "failed", "end_turn", "final_channel",
                                  "tool_recipient", "code", "recap", "nested_thought"])
def test_tool_thought_bridge_requires_one_nonterminal_assistant_thought(tmp_path, case):
    service, _, backend, _ = migration(tmp_path)
    row = saved(service)
    doc = document(row)
    final = completed_tool_chain(doc, "in_progress")
    node_id = final + "-tool-thought"
    thought = doc["mapping"][node_id]["message"]
    if case in {"branch", "nested_thought"}:
        sibling = copy.deepcopy(doc["mapping"][node_id])
        sibling["parent"] = node_id
        sibling["message"]["id"] = "extra-thought"
        doc["mapping"]["extra-thought"] = sibling
        if case == "nested_thought":
            doc["mapping"][final + "-output"]["parent"] = "extra-thought"
    elif case == "wrong_id":
        thought["id"] = "wrong-id"
    elif case == "failed":
        thought["status"] = "failed"
    elif case == "end_turn":
        thought["end_turn"] = True
    elif case == "final_channel":
        thought["channel"] = "final"
    elif case == "tool_recipient":
        thought["recipient"] = "python"
    else:
        thought["content"] = {"content_type": case, "text": "not a thought bridge"}
    result = ConversationBindingService._read_text_request_result(backend, row, document=doc)
    assert result["status"] != "succeeded"
    assert "content" not in result and "_turn_end_evidence" not in result
    assert "_empty_reply_evidence" not in result


def external_continuation(doc):
    original = doc["current_node"]
    doc["mapping"][original]["message"].update(status="in_progress", end_turn=None)
    doc["mapping"][original]["message"]["content"]["parts"] = [""]
    doc["mapping"]["external-user"] = {"parent": original, "message": {
        "id": "external-user", "author": {"role": "user"}, "content": {"content_type": "text", "parts": ["PRIVATE_LATER_INPUT"]}}}
    doc["mapping"]["external-final"] = {"parent": "external-user", "message": {
        "id": "external-final", "author": {"role": "assistant"}, "status": "finished_successfully",
        "end_turn": True, "channel": "final", "content": {"content_type": "text", "parts": ["PRIVATE_LATER_ANSWER"]}}}
    doc["current_node"] = "external-final"
    return doc


def test_completed_external_turn_is_reported_without_adopting_answer_or_releasing_original(tmp_path):
    service, admission, backend, legacy = migration(tmp_path)
    def read(row):
        doc = external_continuation(document(row))
        result = ConversationBindingService._read_text_request_result(None, row, document=doc)
        assert result["status"] == "unknown" and result["recovery_reason"] == "REQUEST_CONVERSATION_ADVANCED"
        assert "content" not in result and "_turn_end_evidence" not in result
        return result
    service.recovery_reader = read
    result = service.read("owner", "old-0")
    assert result["status"] == "unknown" and result["recovery_reason"] == "REQUEST_CONVERSATION_ADVANCED"
    assert result["execution"]["wait_state"] == "ended"
    row = saved(service)
    assert "PRIVATE_LATER" not in json.dumps(row)
    assert all(row.get(key) == legacy[0].get(key) for key in (
        "request_message_id", "parent_message_id", "request_parent_message_id",
        "conversation_id", "provider_account_identity"))
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    restarted = TextTaskService(service.path, admission=admission, clock=admission.clock, recovery_reader=read)
    assert restarted.read("owner", "old-0")["recovery_reason"] == "REQUEST_CONVERSATION_ADVANCED"
    assert backend.mock_calls == []


@pytest.mark.parametrize("case", ["later_running", "other_current", "ambiguous", "wrong_id", "malformed_current"])
def test_external_progress_requires_verified_same_chain_completed_successor(tmp_path, case):
    service, _, _, _ = migration(tmp_path)
    row = saved(service)
    doc = external_continuation(document(row))
    if case == "later_running":
        doc["mapping"]["external-final"]["message"].update(status="in_progress", end_turn=None)
    elif case == "other_current":
        doc["current_node"] = "missing"
    elif case == "ambiguous":
        doc["mapping"]["other"] = {"parent": row["request_message_id"], "message": {"id": "other", "author": {"role": "user"}}}
    elif case == "malformed_current":
        doc["current_node"] = ["external-final"]
    else:
        doc["mapping"]["external-user"]["message"]["id"] = "wrong"
    result = ConversationBindingService._read_text_request_result(None, row, document=doc)
    assert result["recovery_reason"] == "REQUEST_RESULT_INCOMPLETE"
    assert "content" not in result and "_turn_end_evidence" not in result


@pytest.mark.parametrize("later", ["in_progress", "timeout"])
def test_advanced_read_persists_only_branch_ids_and_invalidates_on_later_observation(tmp_path, monkeypatch, later):
    service, admission, backend, _ = migration(tmp_path)
    monkeypatch.setattr("services.conversation_binding_service.time.time", service.clock)
    original = saved(service)
    doc = external_continuation(document(original))
    service.recovery_reader = lambda receipt: ConversationBindingService._read_text_request_result(
        None, receipt, document=doc)
    service.recover("owner", "old-0")
    current = saved(service)
    proof = current["_completed_successor_evidence"]
    assert proof == {"conversation_id": original["conversation_id"],
        "request_message_id": original["request_message_id"],
        "successor_request_message_id": "external-user", "successor_final_message_id": "external-final",
        "successor_request_parent_message_id": doc["mapping"]["external-user"]["parent"],
        "observed_at": service.clock()}
    assert current["status"] == "unknown" and current["upstream_outcome"] == "unknown"
    assert current.get("_upstream_terminal") is not True and "_turn_end_evidence" not in current
    assert "PRIVATE_LATER" not in json.dumps(current) and "content" not in current
    restarted = TextTaskService(service.path, admission=admission, clock=service.clock,
                                recovery_reader=service.recovery_reader)
    assert saved(restarted)["_completed_successor_evidence"] == proof
    # An unrelated/manual later user is only observed; no ownership, slot or
    # final-result permission is inferred by the reader itself.
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    doc["mapping"]["external-final"]["message"].update(status="in_progress", end_turn=False)
    if later == "timeout":
        def fail_read(_receipt):
            raise TimeoutError("fixture original read deadline")
        restarted.recovery_reader = fail_read
    service.clock.now = current["recovery_next_at"] + 1
    restarted.recover("owner", "old-0")
    assert "_completed_successor_evidence" not in saved(restarted)
    assert saved(restarted)["status"] == "unknown" and backend.mock_calls == []


@pytest.mark.parametrize("case", ["multiple_users", "active_tool", "stale_current"])
def test_advanced_diagnostic_does_not_supply_unambiguous_owned_successor_proof(tmp_path, case):
    service, _, _, _ = migration(tmp_path)
    original = saved(service)
    doc = external_continuation(document(original))
    if case == "multiple_users":
        doc["mapping"]["middle-user"] = {"parent": doc["mapping"]["external-user"]["parent"],
            "message": {"id": "middle-user", "author": {"role": "user"}}}
        doc["mapping"]["external-user"]["parent"] = "middle-user"
    elif case == "active_tool":
        doc["mapping"]["active-tool"] = {"parent": doc["mapping"]["external-user"]["parent"],
            "message": {"id": "active-tool", "author": {"role": "tool"}, "status": "in_progress"}}
        doc["mapping"]["external-user"]["parent"] = "active-tool"
    else:
        doc["mapping"]["future-user"] = {"parent": "external-final",
            "message": {"id": "future-user", "author": {"role": "user"}}}
    result = ConversationBindingService._read_text_request_result(None, original, document=doc)
    assert result["recovery_reason"] == "REQUEST_CONVERSATION_ADVANCED"
    assert "_completed_successor_evidence" not in result
    assert "_turn_end_evidence" not in result and "content" not in result


@pytest.mark.parametrize("output", ["missing", "active", "wrong_tool", "complete", "thought_bridge"])
@pytest.mark.parametrize("segment", ["original", "successor"])
def test_advanced_proof_requires_completed_output_for_stale_assistant_tool_call(tmp_path, output, segment):
    service, _, _, _ = migration(tmp_path)
    original = saved(service)
    doc = external_continuation(document(original))
    call_id = doc["mapping"]["external-user"]["parent"]
    next_id = "external-user"
    if segment == "successor":
        call_id, next_id = "child-call", "external-final"
        doc["mapping"][call_id] = {"parent": "external-user", "message": {
            "id": call_id, "author": {"role": "assistant"}}}
        doc["mapping"][next_id]["parent"] = call_id
    call = doc["mapping"][call_id]["message"]
    call.update(status="in_progress", end_turn=False, channel="analysis", recipient="python",
                content={"content_type": "code", "text": "fixture_call()"})
    if output != "missing":
        parent = call_id
        if output == "thought_bridge":
            doc["mapping"]["bridge"] = {"parent": parent, "message": {
                "id": "bridge", "author": {"role": "assistant"}, "status": "in_progress",
                "channel": "analysis", "content": {"content_type": "thoughts", "thoughts": []}}}
            parent = "bridge"
        doc["mapping"]["tool-output"] = {"parent": parent, "message": {
            "id": "tool-output", "author": {"role": "tool", "name": "other" if output == "wrong_tool" else "python"},
            "status": "in_progress" if output == "active" else "finished_successfully",
            "content": {"content_type": "execution_output", "text": "fixture output"},
            "metadata": {"is_complete": True}}}
        doc["mapping"][next_id]["parent"] = "tool-output"
    result = ConversationBindingService._read_text_request_result(None, original, document=doc)
    assert result["recovery_reason"] == (
        "REQUEST_CONVERSATION_ADVANCED" if segment == "original" or output in {"complete", "thought_bridge"}
        else "REQUEST_RESULT_INCOMPLETE")
    assert ("_completed_successor_evidence" in result) == (output in {"complete", "thought_bridge"})
    assert "_turn_end_evidence" not in result and "content" not in result


@pytest.mark.parametrize("case", ["account", "binding", "client", "original", "conversation", "same_id",
                                  "extra_field", "nan_time", "bool_time", "reason", "status", "binding_status",
                                  "empty_parent", "tail_empty", "tail_duplicate", "tail_extra"])
def test_advanced_successor_proof_rejects_identity_and_shape_drift(tmp_path, case):
    service, _, _, _ = migration(tmp_path)
    original = saved(service)
    result = ConversationBindingService._read_text_request_result(
        None, original, document=external_continuation(document(original)))
    proof = result["_completed_successor_evidence"]
    if case in {"account", "binding", "client"}:
        result[{"account": "provider_account_identity", "binding": "provider_binding_id",
                "client": "client_conversation_id"}[case]] = "wrong"
    elif case == "original": proof["request_message_id"] = "wrong"
    elif case == "conversation": proof["conversation_id"] = "wrong"
    elif case == "same_id": proof["successor_request_message_id"] = proof["request_message_id"]
    elif case == "extra_field": proof["content"] = "must not store"
    elif case == "empty_parent": proof["successor_request_parent_message_id"] = ""
    elif case == "tail_empty": proof["tail_steps"] = []
    elif case == "tail_duplicate": proof["tail_steps"] = [
        {"request_message_id": "tail", "final_message_id": "answer"},
        {"request_message_id": "tail", "final_message_id": "answer-2"}]
    elif case == "tail_extra": proof["tail_steps"] = [
        {"request_message_id": "tail", "final_message_id": "answer", "content": "private"}]
    elif case == "nan_time": proof["observed_at"] = float("nan")
    elif case == "bool_time": proof["observed_at"] = True
    elif case == "reason": result["recovery_reason"] = "REQUEST_RESULT_INCOMPLETE"
    elif case == "binding_status": result["binding_status"] = "bound"
    else: result["status"] = "running"
    recovered, code, _, _ = TextTaskService._safe_recovery_result(result, original)
    assert recovered is None and code == "RECOVERY_INVALID_RESULT"


def incomplete_reader(backend, mutate=None):
    def read(row):
        doc = document(row)
        message = doc["mapping"]["final-user-0"]["message"]
        message.update(status="in_progress", end_turn=None, update_time=10)
        message["content"]["parts"] = [""]
        if mutate:
            mutate(doc, message)
        return ConversationBindingService._read_text_request_result(backend, row, document=doc)
    return read


def test_original_phase_diagnostic_is_private_bounded_and_not_completion_proof(tmp_path):
    service, _, backend, _ = migration(tmp_path)
    def mutate(doc, message):
        mixed_chain(doc)
        message.update(channel="final", recipient="all", metadata={
            "channel": "final", "is_complete": True,
            "finish_details": {"type": "stop"}, "token": "PRIVATE_TOKEN"})
        message["content"]["parts"] = ["PRIVATE_PARTIAL_ANSWER"]
        thought = doc["mapping"][doc["current_node"] + "-thought"]["message"]
        thought.update(channel="analysis", recipient="PRIVATE_TOOL_NAME", metadata={
            "channel": "PRIVATE_CHANNEL", "is_complete": "PRIVATE_VALUE",
            "finish_details": {"type": "PRIVATE_FINISH"}})
    service.recovery_reader = incomplete_reader(backend, mutate)
    result = service.read("owner", "old-0")
    row = saved(service)
    phase = row["_original_result_observation"]["phase"]
    assert phase["current_node_relation"] == "observed"
    assert phase["current_node_id"] == "final-user-0"
    assert phase["nodes"][-1] == {
        "id": "final-user-0", "channel": "final", "metadata_channel": "final",
        "recipient": "all", "is_complete": True, "finish_type": "stop"}
    assert phase["nodes"][-3] == {
        "id": "final-user-0-thought", "channel": "analysis", "metadata_channel": "other",
        "recipient": "other", "is_complete": None, "finish_type": "other"}
    assert "PRIVATE_" not in json.dumps(row)
    assert "_original_result_observation" not in result
    assert result["status"] == "unknown" and "content" not in result
    assert row.get("_upstream_terminal") is not True
    assert backend.mock_calls == []


@pytest.mark.parametrize("current,relation,retained", [
    (None, "missing", None), ("user-0", "request", "user-0"),
    ("PRIVATE_OTHER_NODE", "other", None), (["PRIVATE_INVALID_NODE"], "other", None),
])
def test_phase_current_node_retains_only_original_branch_ids(tmp_path, current, relation, retained):
    service, _, backend, _ = migration(tmp_path)
    result = incomplete_reader(backend, lambda doc, _message: doc.update(current_node=current))(saved(service))
    observation = result["_original_result_observation"]
    phase = observation["phase"]
    assert phase["current_node_relation"] == relation and phase["current_node_id"] == retained
    assert "PRIVATE_" not in json.dumps(observation)
    assert TextTaskService._safe_result_observation(observation, saved(service)) == observation


@pytest.mark.parametrize("legacy", [False, True])
def test_phase_changes_and_reader_upgrade_do_not_reset_original_progress(tmp_path, legacy):
    service, admission, backend, _ = migration(tmp_path)
    service.recovery_reader = incomplete_reader(backend)
    service.read("owner", "old-0")
    before = saved(service)
    if legacy:
        observation = copy.deepcopy(before["_original_result_observation"])
        observation.pop("phase")
        patch(service, _original_result_observation=observation)
    def mutate(doc, message):
        message.update(channel="commentary", metadata={"is_complete": True, "channel": "final"})
        doc["current_node"] = "PRIVATE_OTHER_NODE"
    service.recovery_reader = incomplete_reader(backend, mutate)
    admission.clock.now += 301
    result = service.read("owner", "old-0")
    after = saved(service)
    assert result["status"] == "unknown"
    assert after["_result_no_progress_reads"] == before["_result_no_progress_reads"] + 1
    for key in ("_result_observation_started_at", "_result_last_progress_at", "request_message_id", "conversation_id"):
        assert after.get(key) == before.get(key)
    assert after["_original_result_observation"]["nodes"] == before["_original_result_observation"]["nodes"]
    assert after["_original_result_observation"]["phase"] != before["_original_result_observation"]["phase"]
    assert backend.mock_calls == []


@pytest.mark.parametrize("case", ["extra", "node_id", "node_count", "channel", "recipient", "complete", "finish", "current"])
def test_phase_validator_rejects_unscoped_or_unbounded_fields(tmp_path, case):
    service, _, backend, _ = migration(tmp_path)
    row = saved(service)
    observation = incomplete_reader(backend)(row)["_original_result_observation"]
    phase = observation["phase"]
    if case == "extra": phase["body"] = "PRIVATE_BODY"
    elif case == "node_id": phase["nodes"][0]["id"] = "OTHER_NODE"
    elif case == "node_count": phase["nodes"] *= 2
    elif case == "channel": phase["nodes"][0]["metadata_channel"] = {"token": "PRIVATE_TOKEN"}
    elif case == "recipient": phase["nodes"][0]["recipient"] = "PRIVATE_RECIPIENT"
    elif case == "complete": phase["nodes"][0]["is_complete"] = 1
    elif case == "finish": phase["nodes"][0]["finish_type"] = "PRIVATE_FINISH"
    else: phase["current_node_id"] = "OTHER_NODE"
    assert TextTaskService._safe_result_observation(observation, row) is None


def stalled(service, admission, backend):
    service.recovery_reader = incomplete_reader(backend)
    for index in range(4):
        if index:
            admission.clock.now += 301
        result = service.read("owner", "old-0")
    return result


def remove_legacy_outcome(service, stored_outcome):
    if not stored_outcome:
        with service.store.transaction() as db:
            receipt = service.store.read_receipt(db, "text", "owner", "old-0")
            receipt.pop("upstream_outcome")
            service.store.write_receipt(db, "text", "owner", "old-0", receipt)


def test_actual_submitted_exception_without_outcome_can_finish_stalled_local_wait(tmp_path):
    service, admission, backend, _ = migration(tmp_path, count=0)
    service.submit("owner", {"client_request_id": "old-0", "client_conversation_id": "client-0",
                             "model": "fixture-text", "parent_message_id": "parent",
                             "messages": [{"role": "user", "content": "original input"}]})
    def interrupted(context, body):
        context.before_send()
        admission.update_claim(context, conversation_id="conversation-0")
        raise ConnectionError("synthetic interrupted original stream")
    admission.register("text", interrupted)
    admission.execute(admission.claim_next())
    original = saved(service)
    assert original["status"] == "unknown"
    assert original["error_code"] == "CONVERSATION_OUTCOME_UNKNOWN"
    assert "upstream_outcome" not in original
    assert original["_submission_started"] is True and original["_executing"] is False
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    calls = []
    def incomplete(row):
        calls.append(row["request_id"])
        doc = document(row)
        doc["mapping"][doc["current_node"]]["message"].update(status="in_progress", end_turn=False)
        return ConversationBindingService._read_text_request_result(backend, row, document=doc)
    service.recovery_reader = incomplete
    admission.clock.now = original["_claim_until"] + 1
    for _ in range(4):
        service.read("owner", "old-0")
        admission.clock.now += 301
    row = saved(service)
    assert row["status"] == "unknown" and "upstream_outcome" not in row
    assert row["_result_no_progress_reads"] == 3
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert row["request_message_id"] == original["request_message_id"]
    assert calls == ["old-0"] * 4 and backend.mock_calls == []
    waiter(service, "same-conversation-next", "client-0")
    assert admission.claim_next() is None


@pytest.mark.parametrize("stored_outcome", [True, False])
def test_stalled_original_releases_local_capacity_without_losing_late_original_result(tmp_path, monkeypatch, stored_outcome):
    service, admission, backend, legacy = migration(tmp_path)
    remove_legacy_outcome(service, stored_outcome)
    patch(service, _submission_started=True, _turn_reserved=True, _executing=False,
          _execution_timeline=[{"stage": "send_call_started", "at": 3},
                               {"stage": "response_headers_received", "at": 4},
                               {"stage": "first_output", "at": 5}, {"stage": "task_finished", "at": 20}])
    result = stalled(service, admission, backend)
    # Match the admission clock while projecting an expired persisted send
    # lease; production uses the same wall-clock domain for both paths.
    patch(service, _claim_id="expired-send", _claim_until=admission.clock.now - 1)
    monkeypatch.setattr("services.public_chat_service.time.time", lambda: admission.clock.now)
    result = service.read("owner", "old-0")
    row = saved(service)
    assert result["status"] == "unknown"
    assert row["_result_no_progress_reads"] == 3
    assert row["_result_wait_ended_at"] == admission.clock.now
    assert "_execution_wait_ended_at" not in row
    assert row["_turn_reserved"] is True
    assert row["recovery_no_result_reads"] == 0
    assert ("upstream_outcome" in row) is stored_outcome
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    execution = project_public_chat_receipt(result)["execution"]
    assert execution["phase"] == "stalled"
    assert execution["wait_state"] == "ended"
    assert execution["send_state"] == "response_received"
    assert execution["local_state"] == "ended"
    assert execution["upstream_outcome"] == "unknown"
    assert execution["observation_started_at"] < execution["last_checked_at"]
    assert "last_progress_at" not in execution  # First empty observation is not real progress.
    assert execution["resources"] == {"local_worker": "idle", "account_turn": "released", "conversation": "protected"}
    restarted = TextTaskService(service.path, admission=admission, clock=admission.clock,
                                recovery_reader=incomplete_reader(backend))
    assert restarted.read("owner", "old-0")["execution"] == execution
    assert saved(service)["_result_wait_ended_at"] == row["_result_wait_ended_at"]
    def complete(original):
        doc = document(original)
        doc["mapping"]["final-user-0"]["message"]["content"]["parts"] = ["late original result"]
        return ConversationBindingService._read_text_request_result(backend, original, document=doc)
    restarted.recovery_reader = complete
    admission.clock.now = row["recovery_next_at"] + 1
    result = restarted.read("owner", "old-0")
    assert result["status"] == "succeeded" and result["content"] == "late original result"
    assert result["execution"]["wait_state"] == "completed"
    assert result["execution"]["resources"]["account_turn"] == "released"
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    for key in ("request_id", "request_message_id", "provider_account_identity", "conversation_id"):
        assert saved(service)[key] == legacy[0][key]
    assert saved(service)["_result_wait_ended_at"] == row["_result_wait_ended_at"]
    assert backend.mock_calls == []  # The reader receives controlled GET documents; no model POST.


def test_first_incomplete_observation_ends_capacity_wait_on_clock_without_another_get(tmp_path):
    service, admission, backend, legacy = migration(tmp_path)
    patch(service, _route="chat", _operation="text", _submission_started=True,
          _turn_reserved=True, _executing=False)
    reader = Mock(side_effect=incomplete_reader(backend))
    service.recovery_reader = reader

    assert service.read("owner", "old-0")["status"] == "unknown"
    observed = saved(service)
    started = observed["_result_observation_started_at"]
    assert observed["_result_no_progress_reads"] == 0
    assert "_result_wait_ended_at" not in observed
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1

    waiter(service, "same-conversation-next", "client-0")
    waiter(service, "independent-next", "independent")
    admission.clock.now = started + TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS - 1
    assert admission.claim_next() is None
    assert "_result_wait_ended_at" not in saved(service)

    admission.clock.now = started + TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS
    # The existing background claim sweep applies the timeout before it looks
    # for queued work; it does not invoke recovery_reader or start another
    # original GET.
    context = admission.claim_next()
    assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(saved(service), admission.clock.now)
    assert context and context.request_id == "independent-next"
    timed = saved(service)
    assert timed["_result_wait_ended_at"] == admission.clock.now
    assert timed["status"] == "unknown"
    assert timed["request_message_id"] == legacy[0]["request_message_id"]
    assert timed["conversation_id"] == legacy[0]["conversation_id"]
    assert reader.call_count == 1
    assert backend.mock_calls == []

    # The same physical conversation remains protected, while unrelated work
    # can claim the released generation capacity.
    with service.store.transaction() as db:
        independent = service.store.read_receipt(db, "text", "owner", "independent-next")
        independent.update(status="succeeded", upstream_outcome="completed",
                           _attempt_finished_at=admission.clock.now, _executing=False,
                           _turn_reserved=False, _claim_id=None, _claim_until=None)
        service.store.write_receipt(db, "text", "owner", "independent-next", independent)

    # A later changed exact observation continues recovery but cannot reclaim
    # the generation slot released by the first-observation clock.
    service.recovery_reader = incomplete_reader(
        backend, lambda _doc, message: message["content"].update(parts=["new verified text"]),
    )
    admission.clock.now = max(admission.clock.now, timed["recovery_next_at"]) + 1
    assert service.read("owner", "old-0")["status"] == "unknown"
    progressed = saved(service)
    assert progressed["_result_wait_ended_at"] == timed["_result_wait_ended_at"]
    assert progressed["_result_last_progress_at"] == admission.clock.now
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0

    def complete(original):
        doc = document(original)
        doc["mapping"]["final-user-0"]["message"]["content"]["parts"] = ["late original result"]
        return ConversationBindingService._read_text_request_result(backend, original, document=doc)

    service.recovery_reader = complete
    admission.clock.now = max(admission.clock.now, progressed["recovery_next_at"]) + 1
    result = service.read("owner", "old-0")
    assert result["status"] == "succeeded"
    assert result["content"] == "late original result"
    assert saved(service)["request_message_id"] == legacy[0]["request_message_id"]
    assert backend.mock_calls == []


def test_first_incomplete_observation_clock_never_releases_a_live_sender(tmp_path):
    service, admission, backend, _ = migration(tmp_path)
    patch(service, _route="chat", _operation="text", _submission_started=True,
          _turn_reserved=True, _executing=False)
    service.recovery_reader = incomplete_reader(backend)
    assert service.read("owner", "old-0")["status"] == "unknown"
    observed = saved(service)
    patch(service, _executing=True, _claim_id="live-sender",
          _claim_until=observed["_result_observation_started_at"] + 1800)

    admission.clock.now = observed["_result_observation_started_at"] + TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    current = saved(service)
    assert "_result_wait_ended_at" not in current
    waiter(service, "independent-next", "independent")
    assert admission.claim_next() is None
    assert backend.mock_calls == []


@pytest.mark.parametrize("turn_reserved", [True, False])
@pytest.mark.parametrize("stored_outcome", [True, False])
def test_stalled_original_keeps_same_conversation_blocked_but_claims_independent_work(tmp_path, turn_reserved, stored_outcome):
    service, admission, backend, _ = migration(tmp_path)
    remove_legacy_outcome(service, stored_outcome)
    patch(service, _submission_started=True, _turn_reserved=turn_reserved, _executing=False)
    stalled(service, admission, backend)
    assert saved(service)["_turn_reserved"] is turn_reserved
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    # The physical conversation/order resource remains occupied independently
    # of whether this historical receipt stored turn reservation as true/false.
    waiter(service, "same-conversation-next", "client-0")
    assert admission.claim_next() is None
    waiter(service, "independent-next", "independent")
    assert admission.claim_next().request_id == "independent-next"


@pytest.mark.parametrize("stored_outcome", [True, False])
def test_stalled_original_progress_keeps_capacity_released_before_an_independent_claim_sends(tmp_path, stored_outcome):
    service, admission, backend, _ = migration(tmp_path)
    remove_legacy_outcome(service, stored_outcome)
    patch(service, _submission_started=True, _turn_reserved=True, _executing=False)
    stalled(service, admission, backend)
    waiter(service, "independent-next", "independent")
    context = admission.claim_next()
    assert context and context.request_id == "independent-next"
    service.recovery_reader = incomplete_reader(
        backend, lambda _doc, message: message["content"].update(parts=["verified new text"]),
    )
    admission.clock.now = saved(service)["recovery_next_at"] + 1
    admission.update_claim(context, _claim_until=admission.clock.now + admission.CLAIM_SECONDS)
    assert service.read("owner", "old-0")["status"] == "unknown"
    assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(saved(service), admission.clock.now)
    context.before_send()
    with service.store.connect() as db:
        candidate = service.store.read_receipt(db, "text", "owner", "independent-next")
    assert candidate["status"] == "running" and candidate["_submission_started"] is True


@pytest.mark.parametrize("stored_outcome", [True, False])
def test_stalled_original_allows_an_independent_claim_to_send_while_capacity_stays_released(tmp_path, stored_outcome):
    service, admission, backend, _ = migration(tmp_path)
    remove_legacy_outcome(service, stored_outcome)
    patch(service, _submission_started=True, _turn_reserved=True, _executing=False)
    stalled(service, admission, backend)
    waiter(service, "independent-next", "independent")
    context = admission.claim_next()
    assert context and context.request_id == "independent-next"
    context.before_send()
    with service.store.connect() as db:
        candidate = service.store.read_receipt(db, "text", "owner", "independent-next")
    assert candidate["status"] == "running" and candidate["_submission_started"] is True


@pytest.mark.parametrize("progress", [False, True])
def test_stalled_original_read_lease_releases_capacity_but_keeps_conversation_protected(tmp_path, progress):
    service, admission, backend, original = migration(tmp_path)
    patch(service, _submission_started=True, _turn_reserved=False, _executing=False)
    stalled(service, admission, backend)
    admission.clock.now = saved(service)["recovery_next_at"] + 1
    entered, release = Event(), Event()

    def read(row):
        entered.set()
        assert release.wait(5)
        mutate = (lambda _doc, message: message["content"].update(parts=["new text"])) if progress else None
        return incomplete_reader(backend, mutate)(row)

    service.recovery_reader = read
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(service.read, "owner", "old-0")
        try:
            assert entered.wait(5)
            row = saved(service)
            assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(row, admission.clock.now)
            assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
            assert row["status"] == "unknown"
            for key in ("request_id", "request_message_id", "conversation_id", "provider_account_identity"):
                assert row[key] == original[0][key]
            waiter(service, "same-next", "client-0")
            assert admission.claim_next() is None
            waiter(service, "independent-next", "independent")
            context = admission.claim_next()
            assert context and context.request_id == "independent-next"
            context.before_send()
        finally:
            release.set()
        assert pending.result(5)["status"] == "unknown"
    row = saved(service)
    assert row["recovery_claim_id"] is None and row["recovery_lease_until"] is None
    assert row.get("_result_wait_ended_at") is not None
    assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(row, admission.clock.now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    assert backend.mock_calls == []


@pytest.mark.parametrize("changes", [
    {"recovery_claim_id": " "},
    {"recovery_claimed_at": None},
    {"recovery_claimed_at": float("nan")},
    {"recovery_claimed_at": 1},
    {"recovery_claimed_at": 9999999999},
    {"recovery_lease_until": float("inf")},
    {"recovery_lease_until": 1},
    {"recovery_phase": "other"},
    {"_recovery_claim_id": "legacy"},
    {"_executing": True},
])
def test_stalled_read_lease_requires_complete_current_idle_evidence(tmp_path, changes):
    service, admission, backend, _ = migration(tmp_path)
    patch(service, _submission_started=True, _turn_reserved=False, _executing=False)
    stalled(service, admission, backend)
    now = admission.clock.now
    patch(service, recovery_claim_id="read-claim", recovery_claimed_at=now,
          recovery_lease_until=now + 10, recovery_phase="read_text_request")
    patch(service, **changes)
    assert not TextTaskService._qualified_stalled_incomplete_capacity_boundary(saved(service), now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1


@pytest.mark.parametrize("marker", ["submitted", "reserved", "malformed"])
def test_stalled_successor_read_lease_cannot_use_original_read_exception(tmp_path, marker):
    service, admission, backend, _ = migration(tmp_path)
    patch(service, _submission_started=True, _turn_reserved=False, _executing=False)
    stalled(service, admission, backend)
    now = admission.clock.now
    original_id = saved(service)["request_message_id"]
    successor = {"version": 1, "attempt": 1, "state": marker,
                 "request_message_id": "new-attempt", "original_request_message_id": original_id,
                 "submission_parent_message_id": "parent", "prior": {}}
    if marker == "malformed":
        successor = {"state": "submitted"}
    patch(service, recovery_claim_id="read-claim", recovery_claimed_at=now,
          recovery_lease_until=now + 10, recovery_phase="read_text_request",
          _missing_successor_resume=successor)
    assert not TextTaskService._qualified_stalled_incomplete_capacity_boundary(saved(service), now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1


def test_stalled_read_claim_cannot_predate_last_observation(tmp_path):
    service, admission, backend, _ = migration(tmp_path)
    patch(service, _submission_started=True, _turn_reserved=False, _executing=False)
    stalled(service, admission, backend)
    wait_ended = saved(service)["_result_wait_ended_at"]
    admission.clock.now += 10
    patch(service, _result_last_checked_at=admission.clock.now,
          recovery_claim_id="read-claim", recovery_claimed_at=wait_ended + 1,
          recovery_lease_until=admission.clock.now + 10, recovery_phase="read_text_request")
    assert not TextTaskService._qualified_stalled_incomplete_capacity_boundary(saved(service), admission.clock.now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1


def test_stalled_expired_original_read_lease_keeps_only_capacity_released(tmp_path):
    service, admission, backend, original = migration(tmp_path)
    patch(service, _submission_started=True, _turn_reserved=False, _executing=False)
    stalled(service, admission, backend)
    now = admission.clock.now
    patch(service, recovery_claim_id="expired-original-read", recovery_claimed_at=now - 20,
          recovery_lease_until=now - 1, recovery_phase="read_text_request")

    row = saved(service)
    assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(row, now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert row["status"] == "unknown"
    assert row["error_code"] == "CONVERSATION_OUTCOME_UNKNOWN"
    for key in ("request_id", "request_message_id", "conversation_id", "provider_account_identity"):
        assert row[key] == original[0][key]
    assert backend.mock_calls == []


@pytest.mark.parametrize("changes", [
    {"recovery_phase": "other"},
    {"_missing_successor_resume": {"state": "submitted"}},
    {"recovery_claimed_at": 9999999999},
])
def test_stalled_expired_recovery_lease_keeps_original_read_guards(tmp_path, changes):
    service, admission, backend, _ = migration(tmp_path)
    patch(service, _submission_started=True, _turn_reserved=False, _executing=False)
    stalled(service, admission, backend)
    now = admission.clock.now
    patch(service, **{
        "recovery_claim_id": "expired-original-read",
        "recovery_claimed_at": now - 20,
        "recovery_lease_until": now - 1,
        "recovery_phase": "read_text_request",
        **changes,
    })

    assert not TextTaskService._qualified_stalled_incomplete_capacity_boundary(saved(service), now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    assert backend.mock_calls == []


@pytest.mark.parametrize("conflict", [None, "completed", "rejected", "not_submitted", "", False])
def test_stalled_legacy_compatibility_does_not_override_explicit_outcomes(tmp_path, conflict):
    service, admission, backend, _ = migration(tmp_path)
    remove_legacy_outcome(service, False)
    patch(service, _submission_started=True, _turn_reserved=False, _executing=False)
    stalled(service, admission, backend)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    patch(service, upstream_outcome=conflict)
    before = saved(service)
    assert not TextTaskService._qualified_stalled_incomplete_capacity_boundary(before, admission.clock.now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    assert saved(service) == before


@pytest.mark.parametrize("error_code", [None, "RESULT_UNRECOVERABLE", "MODEL_NOT_FOUND"])
def test_stalled_missing_outcome_requires_the_original_unknown_error(tmp_path, error_code):
    service, admission, backend, _ = migration(tmp_path)
    remove_legacy_outcome(service, False)
    patch(service, _submission_started=True, _turn_reserved=False, _executing=False)
    stalled(service, admission, backend)
    patch(service, error_code=error_code)
    assert not TextTaskService._qualified_stalled_incomplete_capacity_boundary(saved(service), admission.clock.now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1


@pytest.mark.parametrize("case", [
    "local_executor", "live_send_lease", "malformed_send_lease", "orphaned_send_lease",
    "active_recovery", "legacy_recovery_claim", "legacy_recovery_lease", "missing_identity", "missing_observation",
    "missing_turn_reservation", "deleted_turn_reservation", "deleted_executing",
    "fake_turn_reservation", "fake_submission_flag",
    "nan_observation_time", "early_wait_marker_later_now", "future_wait_end", "future_last_checked",
    "future_started", "future_progress",
])
@pytest.mark.parametrize("stored_outcome", [True, False])
def test_stalled_local_capacity_boundary_fails_closed_without_complete_idle_evidence(tmp_path, case, stored_outcome):
    service, admission, backend, _ = migration(tmp_path)
    remove_legacy_outcome(service, stored_outcome)
    patch(service, _submission_started=True, _turn_reserved=True, _executing=False)
    stalled(service, admission, backend)
    baseline = saved(service)
    assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(baseline, admission.clock.now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    observed = baseline
    if case == "local_executor":
        changes = {"_executing": True}
    elif case == "live_send_lease":
        changes = {"_claim_id": "live-send", "_claim_until": admission.clock.now + 1}
    elif case == "malformed_send_lease":
        changes = {"_claim_id": "missing-lease", "_claim_until": None}
    elif case == "orphaned_send_lease":
        changes = {"_claim_id": None, "_claim_until": admission.clock.now - 1}
    elif case == "active_recovery":
        changes = {"recovery_claim_id": "live-recovery", "recovery_lease_until": admission.clock.now + 1}
    elif case == "legacy_recovery_claim":
        changes = {"_recovery_claim_id": "legacy-recovery"}
    elif case == "legacy_recovery_lease":
        changes = {"_recovery_claim_until": admission.clock.now + 1}
    elif case == "missing_identity":
        changes = {"provider_binding_id": None}
    elif case == "missing_observation":
        changes = {"_original_result_observation": None}
    elif case == "missing_turn_reservation":
        changes = {"_turn_reserved": None}
    elif case in {"deleted_turn_reservation", "deleted_executing"}:
        changes = {}
    elif case == "fake_turn_reservation":
        changes = {"_turn_reserved": 0}
    elif case == "fake_submission_flag":
        changes = {"_submission_started": 1}
    elif case == "nan_observation_time":
        changes = {"_result_last_checked_at": float("nan")}
    elif case == "early_wait_marker_later_now":
        changes = {"_result_wait_ended_at": observed["_result_observation_started_at"] + 899}
    elif case == "future_wait_end":
        changes = {"_result_wait_ended_at": admission.clock.now + 1}
    elif case == "future_last_checked":
        changes = {"_result_last_checked_at": admission.clock.now + 1}
    elif case == "future_started":
        changes = {"_result_observation_started_at": admission.clock.now + 1}
    else:
        changes = {"_result_last_progress_at": admission.clock.now + 1}
    patch(service, **changes)
    if case in {"deleted_turn_reservation", "deleted_executing"}:
        field = "_turn_reserved" if case == "deleted_turn_reservation" else "_executing"
        with service.store.transaction() as db:
            receipt = service.store.read_receipt(db, "text", "owner", "old-0")
            receipt.pop(field, None)
            service.store.write_receipt(db, "text", "owner", "old-0", receipt)
    receipt = saved(service)
    assert not TextTaskService._qualified_stalled_incomplete_capacity_boundary(receipt, admission.clock.now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    waiter(service, "independent-next", "independent")
    assert admission.claim_next() is None


def test_stalled_capacity_boundary_survives_verified_progress_and_late_success_after_restart(tmp_path):
    service, admission, backend, legacy = migration(tmp_path)
    # Migration data predates the explicit idle markers; this capacity scenario
    # supplies the stored flags required by the strict production predicate.
    patch(service, _submission_started=True, _turn_reserved=True, _executing=False)
    stalled(service, admission, backend)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    wait_ended = saved(service)["_result_wait_ended_at"]
    admission.clock.now = saved(service)["recovery_next_at"] + 1
    service.recovery_reader = incomplete_reader(
        backend, lambda _doc, message: message["content"].update(parts=["new verified text"]),
    )
    result = service.read("owner", "old-0")
    row = saved(service)
    assert result["status"] == "unknown"
    assert row["_result_last_progress_at"] == admission.clock.now
    assert row["_result_wait_ended_at"] == wait_ended
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    waiter(service, "same-conversation-next", "client-0")
    assert admission.claim_next() is None
    restarted = TextTaskService(service.path, admission=admission, clock=admission.clock,
                                recovery_reader=incomplete_reader(backend))
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    def complete(original):
        doc = document(original)
        doc["mapping"]["final-user-0"]["message"]["content"]["parts"] = ["late original result"]
        return ConversationBindingService._read_text_request_result(backend, original, document=doc)
    restarted.recovery_reader = complete
    admission.clock.now = row["recovery_next_at"] + 1
    result = restarted.read("owner", "old-0")
    assert result["status"] == "succeeded" and result["content"] == "late original result"
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    for key in ("request_id", "request_message_id", "provider_account_identity", "conversation_id"):
        assert saved(restarted)[key] == legacy[0][key]


def test_only_original_branch_progress_updates_statistics_without_restarting_capacity_window(tmp_path):
    service, admission, backend, _ = migration(tmp_path)
    service.recovery_reader = incomplete_reader(backend)
    service.read("owner", "old-0")
    first = saved(service)["_result_observation_started_at"]
    admission.clock.now += 901
    def unrelated(doc, message):
        message["update_time"] = 99999
        doc["mapping"]["unrelated"] = {"parent": "elsewhere", "message": {"id": "unrelated"}}
    service.recovery_reader = incomplete_reader(backend, unrelated)
    service.read("owner", "old-0")
    assert saved(service)["_result_observation_started_at"] == first
    assert saved(service)["_result_last_progress_at"] is None
    assert saved(service)["_result_no_progress_reads"] == 1
    wait_ended = saved(service)["_result_wait_ended_at"]
    admission.clock.now += 901
    service.recovery_reader = incomplete_reader(backend, lambda _doc, message: message["content"].update(parts=["new verified text"]))
    service.read("owner", "old-0")
    assert saved(service)["_result_last_progress_at"] == admission.clock.now
    assert saved(service)["_result_no_progress_reads"] == 0
    assert saved(service)["_result_wait_ended_at"] == wait_ended
    assert saved(service)["_result_observation_started_at"] == first


@pytest.mark.parametrize("failure", ["transport", "429", "401", "mismatch", "branch", "invalid", "foreign_current"])
def test_unqualified_reads_do_not_advance_observation_or_extend_capacity_window(tmp_path, failure):
    service, admission, backend, _ = migration(tmp_path)
    service.recovery_reader = incomplete_reader(backend)
    service.read("owner", "old-0")
    original = saved(service)
    admission.clock.now += 901
    def read(row):
        if failure in {"transport", "429", "401"}:
            error = TimeoutError("private diagnostic must not leak")
            error.status_code = {"transport": None, "429": 429, "401": 401}[failure]
            raise error
        def mutate(doc, message):
            if failure == "mismatch":
                doc["mapping"]["user-0"]["parent"] = "different-parent"
            elif failure == "branch":
                doc["mapping"]["sibling"] = copy.deepcopy(doc["mapping"]["final-user-0"])
                doc["mapping"]["sibling"]["message"]["id"] = "sibling"
            elif failure == "foreign_current":
                del doc["mapping"]["user-0"]
            else:
                message["content"]["parts"] = [{"secret": "not text"}]
        return incomplete_reader(backend, mutate)(row)
    service.recovery_reader = read
    service.read("owner", "old-0")
    row = saved(service)
    # An unverified status/branch shape is not original progress. It must not
    # clear or restart the established observation baseline.
    for key in ("_result_last_checked_at", "_result_last_progress_at", "_result_no_progress_reads"):
        assert row[key] == original[key]
    # The existing trusted observation can end the local window on the clock;
    # this failed read neither counts as evidence nor extends that deadline.
    assert row["_original_result_observation"] == original["_original_result_observation"]
    assert row["_result_observation_started_at"] == original["_result_observation_started_at"]
    assert row["_result_wait_ended_at"] == admission.clock.now
    assert row["status"] == "unknown"
    assert not TextTaskService._end_execution_wait(row, admission.clock.now)


def test_parallel_reader_counts_once_and_old_claim_cannot_overwrite_observation(tmp_path):
    service, admission, backend, _ = migration(tmp_path)
    service.recovery_reader = incomplete_reader(backend)
    service.read("owner", "old-0")
    admission.clock.now += 901
    entered, release = Event(), Event()
    claims = []
    def read(row):
        claims.append(row["recovery_claim_id"])
        entered.set()
        assert release.wait(5)
        return incomplete_reader(backend)(row)
    service.recovery_reader = read
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(service.read, "owner", "old-0")
        assert entered.wait(5)
        service.read("owner", "old-0")
        release.set()
        pending.result(5)
    assert len(claims) == 1
    assert saved(service)["_result_no_progress_reads"] == 1
    before = saved(service)
    service._finish_recovery("owner", "old-0", claims[0], {"content": "stale overwrite"})
    assert saved(service) == before


def test_public_execution_is_allowlisted_and_legacy_send_is_unknown():
    raw = {"request_id": "legacy", "status": "unknown", "finished_at": 1,
           "provider_account_identity": "PRIVATE_ACCOUNT", "_execution_timeline": [{"stage": "PRIVATE_STAGE", "at": 2}],
           "execution": {"send_state": "not_sent", "secret": "PRIVATE_BODY"}}
    receipt = project_public_chat_receipt(TextTaskService._public(raw))
    assert receipt["execution"]["send_state"] == "unknown"
    assert receipt["execution"]["upstream_outcome"] == "unknown"
    assert "PRIVATE" not in json.dumps(receipt)
    receipt = project_public_chat_receipt({"request_id": "x", "execution": {
        "send_state": "response_received", "sent_at": float("inf"), "last_progress_at": True,
        "secret": "PRIVATE_BODY", "resources": {"account_turn": "held", "secret": "PRIVATE_ACCOUNT"}}})
    assert receipt["execution"] == {"send_state": "response_received", "resources": {"account_turn": "held"}}


@pytest.mark.parametrize("part", ["code", "output", "thought", "recap"])
def test_mixed_original_chain_observes_progress_then_stall_without_exposing_bodies(tmp_path, part):
    service, admission, backend, _ = migration(tmp_path)
    def shape(doc, message):
        mixed_chain(doc)
        content = doc["mapping"][doc["current_node"] + "-" + part]["message"]["content"]
        if progressed[0]:
            if part == "thought":
                content["thoughts"][0]["finished"] = True
            else:
                content["content" if part == "recap" else "text"] += " growth"
    progressed = [False]
    service.recovery_reader = incomplete_reader(backend, shape)
    result = service.read("owner", "old-0")
    assert result["execution"]["unchanged_reads"] == 0
    assert "PRIVATE" not in json.dumps(saved(service))
    progressed[0] = True
    admission.clock.now += 901
    service.read("owner", "old-0")
    assert saved(service)["_result_last_progress_at"] == admission.clock.now
    for _ in range(3):
        admission.clock.now += 301
        result = service.read("owner", "old-0")
    assert result["execution"]["phase"] == "stalled"
    assert result["execution"]["resources"]["account_turn"] == "held"
    assert "PRIVATE" not in json.dumps(result)
    assert all(key not in json.dumps(result) for key in ("text_chars", "content_items", "finished_items", "nodes"))


@pytest.mark.parametrize("malformed", ["media", "object_output", "unknown_thought_chunk", "later_user", "cycle", "kind_change"])
def test_mixed_observation_rejects_unsupported_or_ambiguous_branch(tmp_path, malformed):
    service, admission, backend, _ = migration(tmp_path)
    def mutate(doc, message):
        mixed_chain(doc)
        output = doc["mapping"][doc["current_node"] + "-output"]["message"]
        if malformed == "media":
            output["content"] = {"content_type": "image", "parts": ["PRIVATE_IMAGE"]}
        elif malformed == "object_output":
            output["content"]["text"] = {"PRIVATE": "BODY"}
        elif malformed == "unknown_thought_chunk":
            thought = doc["mapping"][doc["current_node"] + "-thought"]["message"]["content"]["thoughts"][0]
            thought["chunks"] = [{"PRIVATE": "BODY"}]
        elif malformed == "later_user":
            output["author"]["role"] = "user"
        elif malformed == "cycle":
            doc["mapping"]["user-0"]["parent"] = doc["current_node"]
        else:
            output["content"]["content_type"] = []
    service.recovery_reader = incomplete_reader(backend, mutate)
    result = service.read("owner", "old-0")
    assert "observation_started_at" not in result["execution"]
    assert "_result_wait_ended_at" not in saved(service)


def test_pure_text_observation_upgrade_preserves_progress_clock(tmp_path):
    service, admission, backend, _ = migration(tmp_path)
    service.recovery_reader = incomplete_reader(backend)
    service.read("owner", "old-0")
    original = saved(service)
    observation = original["_original_result_observation"]
    for node in observation["nodes"]:
        for key in ("content_type", "content_items", "finished_items"):
            node.pop(key)
    patch(service, _original_result_observation=observation)
    admission.clock.now += 301
    service.read("owner", "old-0")
    row = saved(service)
    assert row["_result_observation_started_at"] == original["_result_observation_started_at"]
    assert row["_result_last_progress_at"] is None
    assert row["_result_no_progress_reads"] == 1


@pytest.mark.parametrize("failure, expected_code", [
    (lambda: ConversationBindingError("bounded read unavailable", code="RECOVERY_READ_FAILED"),
     "RECOVERY_READ_FAILED"),
    (lambda: type("RateLimitError", (RuntimeError,), {"status_code": 429})("rate limited"),
     "RECOVERY_RATE_LIMITED"),
])
def test_stalled_capacity_boundary_survives_later_transient_original_read_failure(
        tmp_path, failure, expected_code):
    service, admission, backend, _ = migration(tmp_path)
    patch(service, _submission_started=True, _turn_reserved=True, _executing=False)
    stalled(service, admission, backend)
    before = saved(service)
    assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(before, admission.clock.now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0

    def unavailable(_receipt):
        raise failure()

    service.recovery_reader = unavailable
    admission.clock.now = before["recovery_next_at"] + 1
    result = service.read("owner", "old-0")
    after = saved(service)

    assert result["status"] == "unknown"
    assert after["recovery_error_code"] == expected_code
    assert after["recovery_reason"] == "REQUEST_RESULT_INCOMPLETE"
    assert after["recovery_claim_id"] is None
    assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(after, admission.clock.now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert backend.mock_calls == []


def test_stalled_recovery_fences_expired_execution_after_fresh_original_observation(tmp_path):
    service, admission, backend, _ = migration(tmp_path)
    patch(service, _submission_started=True, _turn_reserved=True, _executing=False)
    stalled(service, admission, backend)
    before = saved(service)
    assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(before, admission.clock.now)

    patch(service, _executing=True, _claim_id="expired-original-send",
          _claim_until=admission.clock.now - 1)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1

    service.recovery_reader = incomplete_reader(backend)
    admission.clock.now = saved(service)["recovery_next_at"] + 1
    service.read("owner", "old-0")
    after = saved(service)

    assert after["status"] == "unknown"
    assert after["_executing"] is False
    assert after["_claim_id"] is None and after["_claim_until"] is None
    assert after["_result_no_progress_reads"] > before["_result_no_progress_reads"]
    assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(after, admission.clock.now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert backend.mock_calls == []


def test_stalled_recovery_failure_does_not_fence_expired_sender_without_observation(tmp_path):
    service, admission, backend, _ = migration(tmp_path)
    patch(service, _submission_started=True, _turn_reserved=True, _executing=False)
    stalled(service, admission, backend)
    before = saved(service)
    patch(service, _executing=True, _claim_id="expired-original-send",
          _claim_until=admission.clock.now - 1)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1

    def rate_limited(_receipt):
        raise type("RateLimitError", (RuntimeError,), {"status_code": 429})("rate limited")

    service.recovery_reader = rate_limited
    admission.clock.now = before["recovery_next_at"] + 1
    service.read("owner", "old-0")
    after = saved(service)

    assert after["recovery_error_code"] == "RECOVERY_RATE_LIMITED"
    assert after["_executing"] is True
    assert after["_claim_id"] == "expired-original-send"
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    assert backend.mock_calls == []


@pytest.mark.parametrize("claim_id, claim_until", [
    ("live-original-send", 10_000),
    (None, None),
])
def test_stalled_recovery_does_not_fence_live_or_malformed_sender_lease(
        tmp_path, claim_id, claim_until):
    service, admission, backend, _ = migration(tmp_path)
    patch(service, _submission_started=True, _turn_reserved=True, _executing=False)
    stalled(service, admission, backend)
    before = saved(service)
    if claim_until is not None:
        claim_until = admission.clock.now + 1_000
    patch(service, _executing=True, _claim_id=claim_id, _claim_until=claim_until)

    service.recovery_reader = incomplete_reader(backend)
    admission.clock.now = before["recovery_next_at"] + 1
    service.read("owner", "old-0")
    after = saved(service)

    assert after["_executing"] is True
    assert after["_claim_id"] == claim_id and after["_claim_until"] == claim_until
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    assert backend.mock_calls == []


def test_read_failure_does_not_restore_missing_stalled_reason_or_release_capacity(tmp_path):
    service, admission, backend, _ = migration(tmp_path)
    patch(service, _submission_started=True, _turn_reserved=True, _executing=False)
    stalled(service, admission, backend)
    before = saved(service)
    patch(service, recovery_reason=None)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1

    def unavailable(_receipt):
        raise ConnectionError("transient read failure")

    service.recovery_reader = unavailable
    admission.clock.now = before["recovery_next_at"] + 1
    service.read("owner", "old-0")
    after = saved(service)

    assert after["recovery_reason"] is None
    assert after["recovery_error_code"] == "RECOVERY_TRANSPORT_FAILED"
    assert not TextTaskService._qualified_stalled_incomplete_capacity_boundary(after, admission.clock.now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    assert backend.mock_calls == []


def test_completed_exact_read_failure_releases_only_local_capacity_and_preserves_recovery(tmp_path):
    service, admission, backend, legacy = migration(tmp_path)
    patch(service, _submission_started=True, _turn_reserved=False, _executing=False)

    # Establish a real original observation first. Its age is reused below;
    # the later GET failure does not restart this clock.
    service.recovery_reader = incomplete_reader(backend)
    assert service.read("owner", "old-0")["status"] == "unknown"
    observed = saved(service)
    first_observation_at = observed["_result_observation_started_at"]
    admission.clock.now += TextTaskService.RECOVERY_LEASE_SECONDS + 1
    patch(service, recovery_attempt=9)

    def unavailable(row):
        assert row["recovery_claim_id"]
        error = ConnectionError("synthetic original GET unavailable")
        error._original_result_get_attempted = True
        raise error

    service.recovery_reader = unavailable
    assert service.read("owner", "old-0")["status"] == "unknown"
    row = saved(service)
    # A high attempt no longer stretches normal sent-Chat original-result
    # recovery past five minutes. The completed exact GET error also releases
    # only local capacity immediately.
    assert row["recovery_attempt"] == 10
    assert row["recovery_next_at"] == admission.clock.now + 300
    # A persisted pre-fix +900 schedule for this same original GET becomes
    # due at five minutes from its completed read; explicit Retry-After and
    # unsent work keep their existing schedules.
    legacy_due = {**row, "recovery_next_at": admission.clock.now + 900,
                  "updated_at": admission.clock.now,
                  "recovery_retry_after_seconds": None}
    assert not TextTaskService._recovery_due(legacy_due, admission.clock.now + 299)
    assert TextTaskService._recovery_due(legacy_due, admission.clock.now + 300)
    from services.public_chat_service import project_public_chat_receipt
    public = service._public(legacy_due)
    assert public["recovery_next_at"] == admission.clock.now + 300
    assert project_public_chat_receipt(public)["recovery"]["next_at"] == admission.clock.now + 300
    assert legacy_due["recovery_next_at"] == admission.clock.now + 900
    legacy_no_outcome = dict(legacy_due)
    legacy_no_outcome.pop("upstream_outcome", None)
    assert TextTaskService._recovery_due(legacy_no_outcome, admission.clock.now + 300)
    retry_after_due = {**legacy_due, "recovery_retry_after_seconds": 900}
    assert not TextTaskService._recovery_due(retry_after_due, admission.clock.now + 300)
    assert service._public(retry_after_due)["recovery_next_at"] == admission.clock.now + 900
    unsent_due = {**legacy_due, "_submission_started": False}
    assert not TextTaskService._recovery_due(unsent_due, admission.clock.now + 300)
    assert service._public(unsent_due)["recovery_next_at"] == admission.clock.now + 900
    assert row["_result_read_capacity_released_at"] == admission.clock.now
    assert row["_result_observation_started_at"] == first_observation_at
    assert TextTaskService._bounded_unreadable_result_capacity_boundary(row, admission.clock.now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert not TextTaskService._no_final_successor_source(row, admission.clock.now)
    assert not TextTaskService._incomplete_compact_successor_source(row, admission.clock.now)

    # The marker survives a SQLite restart, does not end the original turn,
    # and only frees a different physical chat/account for normal claiming.
    _, _, restarted_admission = build(tmp_path, admission.clock)
    restarted = TextTaskService(service.path, admission=restarted_admission, clock=admission.clock)
    assert restarted_admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    waiter(restarted, "same-conversation-next", "client-0")
    waiter(restarted, "independent-next", "independent")
    restarted.submit("owner", {
        "client_request_id": "other-account-next", "client_conversation_id": "other-account-session",
        "provider_binding_id": "binding-1", "provider_account_identity": "account-1",
        "model": "fixture-text", "messages": [{"role": "user", "content": "other account work"}],
    })
    context = restarted_admission.claim_next()
    assert context and context.request_id == "independent-next"
    other_context = restarted_admission.claim_next()
    assert other_context and other_context.request_id == "other-account-next"
    # The claim assertions above exercise unrelated scheduling. Retire those
    # synthetic holders before checking the original marker by itself.
    with restarted.store.transaction() as db:
        for request_id in ("independent-next", "other-account-next"):
            holder = restarted.store.read_receipt(db, "text", "owner", request_id)
            holder.update(status="succeeded", upstream_outcome="completed",
                          _attempt_finished_at=admission.clock.now, _executing=False,
                          _turn_reserved=False, _claim_id=None, _claim_until=None)
            restarted.store.write_receipt(db, "text", "owner", request_id, holder)
    for key in ("request_id", "request_message_id", "provider_binding_id",
                "provider_account_identity", "client_conversation_id", "conversation_id"):
        assert row[key] == legacy[0][key]
    assert row["_turn_reserved"] is False
    assert row.get("_attempt_finished_at") is None

    # A status-only running result is not verifiable branch progress. It keeps
    # the old observation and local release marker; it never restarts the
    # five-minute result-read cadence or the existing progress clock.
    def active_without_observation(original):
        doc = document(original)
        del doc["mapping"][original["request_message_id"]]
        active = doc["mapping"][doc["current_node"]]["message"]
        active.update(status="in_progress", end_turn=None)
        return ConversationBindingService._read_text_request_result(backend, original, document=doc)

    restarted.recovery_reader = active_without_observation
    admission.clock.now = row["recovery_next_at"] + 1
    assert restarted.read("owner", "old-0")["status"] == "unknown"
    after_running = saved(restarted)
    assert after_running["_result_read_capacity_released_at"] == row["_result_read_capacity_released_at"]
    assert after_running["_result_observation_started_at"] == first_observation_at
    assert TextTaskService._bounded_unreadable_result_capacity_boundary(after_running, admission.clock.now)
    assert restarted_admission.resource_snapshot()["chat_turn"]["inflight"] == 0

    # A later typed GET failure also preserves the original marker and
    # progress clock. It cannot manufacture a new 60-second hold or push the
    # normal result-read cadence back to the old 900-second cap.
    restarted.recovery_reader = unavailable
    admission.clock.now = after_running["recovery_next_at"] + 1
    assert restarted.read("owner", "old-0")["status"] == "unknown"
    after_repeat_failure = saved(restarted)
    assert after_repeat_failure["_result_read_capacity_released_at"] == row["_result_read_capacity_released_at"]
    assert after_repeat_failure["_result_observation_started_at"] == first_observation_at
    assert after_repeat_failure["recovery_next_at"] == admission.clock.now + 300
    assert TextTaskService._bounded_unreadable_result_capacity_boundary(after_repeat_failure, admission.clock.now)
    assert restarted_admission.resource_snapshot()["chat_turn"]["inflight"] == 0

    # A changed, identity-safe observation clears the GET-failure marker. The
    # separate absolute observation window has now ended, so generation
    # capacity stays released while exact-result recovery continues.
    restarted.recovery_reader = incomplete_reader(
        backend, lambda _doc, message: message["content"].update(parts=["new verified text"]),
    )
    admission.clock.now = after_repeat_failure["recovery_next_at"] + 1
    assert restarted.read("owner", "old-0")["status"] == "unknown"
    progressed = saved(restarted)
    assert "_result_read_capacity_released_at" not in progressed
    assert progressed["_result_last_progress_at"] == admission.clock.now
    assert not TextTaskService._bounded_unreadable_result_capacity_boundary(progressed, admission.clock.now)
    assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(progressed, admission.clock.now)
    assert restarted_admission.resource_snapshot()["chat_turn"]["inflight"] == 0

    # An exact late success remains admissible under the original identity and
    # never schedules a new POST.
    def complete(original):
        doc = document(original)
        doc["mapping"]["final-user-0"]["message"]["content"]["parts"] = ["late original result"]
        return ConversationBindingService._read_text_request_result(backend, original, document=doc)

    restarted.recovery_reader = complete
    admission.clock.now = progressed["recovery_next_at"] + 1
    result = restarted.read("owner", "old-0")
    assert result["status"] == "succeeded"
    assert result["content"] == "late original result"
    after = saved(restarted)
    before_stale_failure = copy.deepcopy(after)
    restarted._finish_recovery(
        "owner", "old-0", "obsolete-reader", error_code="RECOVERY_TRANSPORT_FAILED",
        read_error={"phase": "conversation_detail", "category": "transport",
                    "http_status": None, "retry_after_seconds": None, "candidate_ref": None},
    )
    assert saved(restarted) == before_stale_failure
    assert backend.mock_calls == []


def test_verified_progress_before_capacity_deadline_still_clears_completed_get_failure_marker(tmp_path):
    service, admission, backend, _ = migration(tmp_path)
    patch(service, _submission_started=True, _turn_reserved=False, _executing=False)
    service.recovery_reader = incomplete_reader(backend)
    assert service.read("owner", "old-0")["status"] == "unknown"
    started = saved(service)["_result_observation_started_at"]
    admission.clock.now += TextTaskService.RECOVERY_LEASE_SECONDS + 1

    def unavailable(_row):
        error = ConnectionError("synthetic original GET unavailable")
        error._original_result_get_attempted = True
        raise error

    service.recovery_reader = unavailable
    assert service.read("owner", "old-0")["status"] == "unknown"
    failed_read = saved(service)
    assert TextTaskService._bounded_unreadable_result_capacity_boundary(failed_read, admission.clock.now)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    service.recovery_reader = incomplete_reader(
        backend, lambda _doc, message: message["content"].update(parts=["new verified text"]),
    )
    admission.clock.now = failed_read["recovery_next_at"] + 1
    assert admission.clock.now - started < TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS
    assert service.read("owner", "old-0")["status"] == "unknown"
    progressed = saved(service)
    assert "_result_read_capacity_released_at" not in progressed
    assert "_result_wait_ended_at" not in progressed
    assert progressed["_result_last_progress_at"] == admission.clock.now
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    assert backend.mock_calls == []


def test_capacity_marker_rejects_preflight_and_unknown_leases(tmp_path):
    service, admission, _backend, _ = migration(tmp_path)
    patch(service, _submission_started=True, _turn_reserved=False, _executing=False)
    before = saved(service)
    # A credential/local preparation failure has no proved GET and must not
    # create the release marker, even though it is a typed auth recovery error.
    claim = "current-reader"
    patch(service, recovery_claim_id=claim, recovery_claimed_at=admission.clock.now,
          recovery_lease_until=admission.clock.now + 60)
    service._finish_recovery(
        "owner", "old-0", claim, error_code="RECOVERY_AUTH_REQUIRED",
        recovery_reason=None,
        read_error={"phase": "conversation_detail", "category": "other",
                    "http_status": None, "retry_after_seconds": None, "candidate_ref": None},
    )
    assert "_result_read_capacity_released_at" not in saved(service)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1

    # Begin from a genuine completed exact GET failure, then prove that sender
    # claims and unknown/non-reader recovery leases cannot borrow its release.
    patch(service, recovery_claim_id="exact", recovery_claimed_at=admission.clock.now,
          recovery_lease_until=admission.clock.now + 60)
    service._finish_recovery(
        "owner", "old-0", "exact", error_code="RECOVERY_TRANSPORT_FAILED",
        recovery_reason=None,
        original_result_get_attempted=True,
        read_error={"phase": "conversation_detail", "category": "transport",
                    "http_status": None, "retry_after_seconds": None, "candidate_ref": None},
    )
    released = saved(service)
    assert TextTaskService._bounded_unreadable_result_capacity_boundary(released, admission.clock.now)
    unsafe = (
        {"_executing": True},
        {"_claim_id": "live-sender", "_claim_until": admission.clock.now + 1},
        {"_claim_id": "malformed-sender", "_claim_until": None},
        {"recovery_claim_id": "foreign", "recovery_claimed_at": admission.clock.now,
         "recovery_lease_until": admission.clock.now + 1, "recovery_phase": "send_text"},
        {"_missing_successor_resume": {"state": "recovered"}},
    )
    for changes in unsafe:
        assert not TextTaskService._bounded_unreadable_result_capacity_boundary(
            {**released, **changes}, admission.clock.now)
    exact_lease = {**released, "recovery_claim_id": "reader", "recovery_claimed_at": admission.clock.now,
                   "recovery_lease_until": admission.clock.now + 60, "recovery_phase": "read_text_request"}
    assert TextTaskService._bounded_unreadable_result_capacity_boundary(exact_lease, admission.clock.now)
    assert before["request_message_id"] == released["request_message_id"]
