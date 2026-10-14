"""One-use recovery for a submitted successor whose original user node vanished."""
from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from services.request_context import current_request
from test.test_bound_text_archive import bound_chat
from test.test_explicit_unknown_successor import row, task, update
from test.test_no_final_successor import graph, no_final, user_leaf_graph


def stopped_successor(task):
    h = task
    h.service.submit("owner", h.envelope)
    saved = row(h.service, "successor")
    # The outbound root and the actual user parent differ for a batched turn.
    # The resume must use only the saved submission root, never this old cursor.
    update(h.service, "successor", status="failed", error_code="RESULT_UNRECOVERABLE",
           upstream_outcome="unknown", recovery_reason="REQUEST_MESSAGE_NOT_FOUND",
           _attempt_reason="ORIGINAL_REQUEST_MESSAGE_NOT_FOUND", _attempt_finished_at=900,
           _execution_wait_ended_at=900, _submission_started=True, _last_sent_sequence=0,
           _executing=False, _turn_reserved=False, _claim_id=None, _claim_until=None,
           send_count=1, request_parent_message_id="old-batched-parent",
           _submission_parent_message_id=saved["parent_message_id"],
           _execution_timeline=[{"stage": "accepted"}, {"stage": "send_call_started"}])
    return h


def install_bound_send(h, chat, monkeypatch, sent):
    import services.conversation_binding_service as binding
    chat.document["is_archived"] = False
    original_get = binding.OpenAIBackendAPI._get_conversation

    def get(backend, conversation_id, **_kwargs):
        return original_get(backend, conversation_id)

    monkeypatch.setattr(binding.OpenAIBackendAPI, "_get_conversation", get)

    def events(backend, **kwargs):
        # This is the actual binding pre-send path, including its final check
        # and admission transport boundary.
        backend.text_pre_send_check(None)
        current_request.get().before_send()
        sent.append(deepcopy(kwargs))
        attempt = backend.text_request_message_id
        assert kwargs["parent_message_id"] == "previous-answer"
        backend.text_cursor_callback({"request_parent_message_id": "new-batched-parent",
                                      "_submission_parent_message_id": "previous-answer"})
        chat.document["mapping"][attempt] = {"parent": "previous-answer", "children": ["new-answer"], "message": {
            "id": attempt, "author": {"role": "user"}, "status": "finished_successfully"}}
        chat.document["mapping"]["new-answer"] = {"parent": attempt, "children": [], "message": {
            "id": "new-answer", "author": {"role": "assistant"}, "status": "finished_successfully",
            "end_turn": True, "channel": "final", "content": {"content_type": "text", "parts": ["fresh answer"]}}}
        chat.document["current_node"] = "new-answer"
        yield {"type": "conversation.delta", "conversation_id": "original-chat", "delta": "fresh answer"}

    monkeypatch.setattr(binding, "conversation_events", events)
    h.service.runner = chat.service.complete_text


def test_same_id_one_use_dispatch_keeps_frozen_input_and_old_evidence(task, bound_chat, monkeypatch):
    h = stopped_successor(task)
    before = deepcopy(row(h.service, "successor"))
    frozen = h.service.store.load_input(before["_input_ref"])
    sent = []
    install_bound_send(h, bound_chat, monkeypatch, sent)

    accepted = h.service.resume_missing_successor("owner", "successor", h.envelope)
    reserved = row(h.service, "successor")
    marker = reserved["_missing_successor_resume"]
    assert accepted["status"] == "queued"
    assert accepted["missing_successor_resume"] == {"attempt": 1, "state": "reserved",
                                                       "request_message_id": marker["request_message_id"]}
    assert reserved["request_message_id"] == before["request_message_id"]
    assert marker["original_request_message_id"] == before["request_message_id"]
    assert marker["prior"]["_attempt_reason"] == "ORIGINAL_REQUEST_MESSAGE_NOT_FOUND"
    assert marker["prior"]["request_parent_message_id"] == "old-batched-parent"
    assert h.service.store.load_input(reserved["_input_ref"]) == frozen

    h.admission.execute(h.admission.claim_next())
    result = row(h.service, "successor")
    assert len(sent) == 1 and sent[0]["messages"] == frozen["messages"], {
        key: result.get(key) for key in ("status", "error_code", "upstream_outcome", "_execution_timeline", "_missing_successor_resume")}
    assert result["status"] == "succeeded" and result["content"] == "fresh answer"
    assert result["request_message_id"] == before["request_message_id"]
    assert result["request_parent_message_id"] == "old-batched-parent"
    assert result["_missing_successor_resume"]["state"] == "recovered"
    assert result["_missing_successor_resume"]["request_parent_message_id"] == "new-batched-parent"
    assert h.service.resume_missing_successor("owner", "successor", h.envelope)["status"] == "succeeded"
    assert h.admission.claim_next() is None and len(sent) == 1


def test_lost_response_reads_exact_new_node_without_second_dispatch(task):
    h = stopped_successor(task)
    h.service.resume_missing_successor("owner", "successor", h.envelope)
    original_id = row(h.service, "successor")["request_message_id"]
    attempt_id = row(h.service, "successor")["_missing_successor_resume"]["request_message_id"]
    calls = []

    def uncertain(body, on_cursor):
        current_request.get().before_send()
        calls.append(body["_request_message_id"])
        raise RuntimeError("lost after transport")

    h.service.runner = uncertain
    h.admission.execute(h.admission.claim_next())
    uncertain_row = row(h.service, "successor")
    assert uncertain_row["status"] == "unknown"
    assert uncertain_row["_missing_successor_resume"]["state"] == "submitted"
    assert calls == [attempt_id]

    def recover(receipt):
        assert receipt["request_message_id"] == attempt_id
        assert receipt["request_parent_message_id"] == "previous-answer"
        return {"status": "succeeded", "binding_status": "bound", "content": "recovered",
                "conversation_id": "original-chat", "parent_message_id": "answer-after-crash"}

    h.service.recovery_reader = Mock(side_effect=recover)
    recovered = h.service.read("owner", "successor")
    saved = row(h.service, "successor")
    assert recovered["status"] == "succeeded" and recovered["content"] == "recovered"
    assert saved["request_message_id"] == original_id
    assert saved["request_parent_message_id"] == "old-batched-parent"
    assert saved["_missing_successor_resume"]["state"] == "recovered"
    assert h.admission.claim_next() is None and calls == [attempt_id]


@pytest.mark.parametrize("tail", ["tool", "user"])
def test_no_final_resume_keeps_frozen_root_but_rechecks_live_tail(task, bound_chat, monkeypatch, tail):
    h = no_final(task)
    h.service.submit("owner", h.envelope)
    successor = row(h.service, "successor")
    update(h.service, "successor", status="failed", error_code="RESULT_UNRECOVERABLE",
           upstream_outcome="unknown", recovery_reason="REQUEST_MESSAGE_NOT_FOUND",
           _attempt_reason="ORIGINAL_REQUEST_MESSAGE_NOT_FOUND", _attempt_finished_at=900,
           _execution_wait_ended_at=900, _submission_started=True, _last_sent_sequence=0,
           _executing=False, _turn_reserved=False, _claim_id=None, _claim_until=None,
           send_count=1, request_parent_message_id=(h.old["request_message_id"] if tail == "user" else "tool"),
           _submission_parent_message_id=successor["parent_message_id"],
           _execution_timeline=[{"stage": "accepted"}, {"stage": "send_call_started"}])
    expected_tail = h.old["request_message_id"] if tail == "user" else "tool"
    bound_chat.document = user_leaf_graph(h) if tail == "user" else graph(h)
    bound_chat.document["is_archived"] = False
    import services.conversation_binding_service as binding
    original_get = binding.OpenAIBackendAPI._get_conversation
    monkeypatch.setattr(binding.OpenAIBackendAPI, "_get_conversation",
                        lambda backend, conversation_id, **_kwargs: original_get(backend, conversation_id))
    sent = []

    def events(backend, **kwargs):
        backend.text_pre_send_check(None)
        current_request.get().before_send()
        sent.append(deepcopy(kwargs))
        attempt = backend.text_request_message_id
        assert kwargs["parent_message_id"] == expected_tail
        backend.text_cursor_callback({"request_parent_message_id": expected_tail,
                                      "_submission_parent_message_id": expected_tail})
        bound_chat.document["mapping"][attempt] = {"parent": expected_tail, "children": ["new-answer"], "message": {
            "id": attempt, "author": {"role": "user"}, "status": "finished_successfully"}}
        bound_chat.document["mapping"]["new-answer"] = {"parent": attempt, "children": [], "message": {
            "id": "new-answer", "author": {"role": "assistant"}, "status": "finished_successfully",
            "end_turn": True, "channel": "final", "content": {"content_type": "text", "parts": ["no-final answer"]}}}
        bound_chat.document["current_node"] = "new-answer"
        yield {"type": "conversation.delta", "conversation_id": "original-chat", "delta": "no-final answer"}

    monkeypatch.setattr(binding, "conversation_events", events)
    h.service.runner = bound_chat.service.complete_text
    envelope = {key: h.envelope[key] for key in h.service.SUPERSEDE_FIELDS}
    h.service.resume_missing_successor("owner", "successor", envelope)
    reserved = row(h.service, "successor")
    assert reserved["_missing_successor_resume"]["submission_parent_message_id"] == "previous-answer"
    h.admission.execute(h.admission.claim_next())
    result = row(h.service, "successor")
    assert len(sent) == 1 and result["status"] == "succeeded"
    assert result["request_parent_message_id"] == expected_tail
    assert result["_missing_successor_resume"]["attempt_submission_parent_message_id"] == expected_tail


def test_replacement_read_budget_starts_at_reservation_and_stops(task):
    h = stopped_successor(task)
    update(h.service, "successor", recovery_no_result_reads=11, recovery_attempt=11,
           recovery_next_at=10_000)
    h.service.resume_missing_successor("owner", "successor", h.envelope)
    reserved = row(h.service, "successor")
    assert reserved["recovery_no_result_reads"] == 0
    assert reserved["recovery_attempt"] == 0
    assert reserved["_missing_successor_resume"]["prior"]["recovery_no_result_reads"] == 11

    def uncertain(body, on_cursor):
        current_request.get().before_send()
        raise RuntimeError("lost after transport")
    h.service.runner = uncertain
    h.admission.execute(h.admission.claim_next())
    h.service.recovery_reader = Mock(return_value={
        "status": "unknown", "recovery_reason": "REQUEST_MESSAGE_NOT_FOUND",
        "conversation_id": "original-chat", "parent_message_id": "previous-answer",
        "request_parent_message_id": "previous-answer",
    })
    h.clock.now += 901
    for index in range(3):
        h.service.read("owner", "successor")
        saved = row(h.service, "successor")
        if index < 2:
            assert saved.get("recovery_next_at")
            h.clock.now = saved["recovery_next_at"] + 1
    stopped = row(h.service, "successor")
    assert h.service.recovery_reader.call_count == 3
    assert stopped["recovery_no_result_reads"] == 3
    assert stopped["_attempt_reason"] == "ORIGINAL_REQUEST_MESSAGE_NOT_FOUND"
    assert stopped["_attempt_finished_at"]
    h.service.read("owner", "successor")
    assert h.service.recovery_reader.call_count == 3
    h.service.recover("owner", "successor", explicit_ended_recheck=True)
    assert h.service.recovery_reader.call_count == 4


def test_final_preflight_late_original_result_returns_to_original_id_recovery(task, bound_chat, monkeypatch):
    h = stopped_successor(task)
    sent = []
    install_bound_send(h, bound_chat, monkeypatch, sent)
    h.service.resume_missing_successor("owner", "successor", h.envelope)
    original_id = row(h.service, "successor")["request_message_id"]

    import services.conversation_binding_service as binding
    def events(backend, **_kwargs):
        # The initial read passed; the binding-locked final check sees the
        # original user node appear before any replacement transport.
        bound_chat.document["mapping"][original_id] = {"parent": "previous-answer", "children": [], "message": {
            "id": original_id, "author": {"role": "user"}, "status": "finished_successfully"}}
        backend.text_pre_send_check(None)
        current_request.get().before_send()
        sent.append(True)
        yield {"type": "conversation.delta", "conversation_id": "original-chat", "delta": "never"}
    monkeypatch.setattr(binding, "conversation_events", events)

    h.admission.execute(h.admission.claim_next())
    paused = row(h.service, "successor")
    assert not sent and paused["status"] == "unknown"
    assert paused["upstream_outcome"] == "unknown"
    assert paused["_missing_successor_resume"]["state"] == "recovered"
    assert paused["_missing_successor_resume"]["result_source"] == "original"

    def late_original(receipt):
        assert receipt["request_message_id"] == original_id
        return {"status": "succeeded", "binding_status": "bound", "content": "late original",
                "conversation_id": "original-chat", "parent_message_id": "late-answer"}
    h.service.recovery_reader = Mock(side_effect=late_original)
    recovered = h.service.read("owner", "successor")
    assert recovered["content"] == "late original"
    assert h.service.recovery_reader.call_count == 1


def test_initial_late_incomplete_original_returns_to_original_id_recovery(task, bound_chat, monkeypatch):
    h = stopped_successor(task)
    h.service.resume_missing_successor("owner", "successor", h.envelope)
    original_id = row(h.service, "successor")["request_message_id"]
    bound_chat.document["is_archived"] = False
    bound_chat.document["mapping"][original_id] = {"parent": "previous-answer", "children": [], "message": {
        "id": original_id, "author": {"role": "user"}, "status": "finished_successfully"}}
    import services.conversation_binding_service as binding
    original_get = binding.OpenAIBackendAPI._get_conversation
    monkeypatch.setattr(binding.OpenAIBackendAPI, "_get_conversation",
                        lambda backend, conversation_id, **_kwargs: original_get(backend, conversation_id))
    monkeypatch.setattr(binding.ConversationBindingService, "_read_text_request_result", staticmethod(
        lambda _backend, receipt, **_kwargs: {"status": "unknown"}
        if receipt["request_message_id"] == original_id else {"status": "succeeded", "content": "replacement"},
    ))
    sent = []
    monkeypatch.setattr(binding, "conversation_events", lambda *_args, **_kwargs: sent.append(True))
    h.service.runner = bound_chat.service.complete_text
    h.admission.execute(h.admission.claim_next())
    paused = row(h.service, "successor")
    assert not sent and paused["status"] == "unknown"
    assert paused["_missing_successor_resume"]["result_source"] == "original"

    def late_original(receipt):
        assert receipt["request_message_id"] == original_id
        return {"status": "succeeded", "binding_status": "bound", "content": "late original",
                "conversation_id": "original-chat", "parent_message_id": "late-answer"}
    h.service.recovery_reader = Mock(side_effect=late_original)
    assert h.service.read("owner", "successor")["content"] == "late original"


@pytest.mark.parametrize("when,shape", [("initial", "mapping_key"), ("final", "embedded_alias")])
def test_generic_resume_refuses_existing_replacement_node_before_transport(task, bound_chat, monkeypatch, when, shape):
    h = stopped_successor(task)
    h.service.resume_missing_successor("owner", "successor", h.envelope)
    attempt = row(h.service, "successor")["_missing_successor_resume"]["request_message_id"]
    bound_chat.document["is_archived"] = False
    def add_replacement():
        key = attempt if shape == "mapping_key" else "replacement-alias"
        # Keep this node off the checked terminal root.  Passing now depends
        # on the replacement UUID/key-or-embedded-ID fence itself.
        bound_chat.document["mapping"][key] = {"parent": "unrelated-parent", "children": [], "message": {
            "id": attempt, "author": {"role": "user"}, "status": "finished_successfully"}}
    if when == "initial":
        add_replacement()
    import services.conversation_binding_service as binding
    original_get = binding.OpenAIBackendAPI._get_conversation
    monkeypatch.setattr(binding.OpenAIBackendAPI, "_get_conversation",
                        lambda backend, conversation_id, **_kwargs: original_get(backend, conversation_id))
    sent = []
    def events(backend, **_kwargs):
        if when == "final":
            add_replacement()
            backend.text_pre_send_check(None)
        sent.append(True)
        yield {"type": "conversation.delta", "conversation_id": "original-chat", "delta": "unexpected"}
    monkeypatch.setattr(binding, "conversation_events", events)
    h.service.runner = bound_chat.service.complete_text
    h.admission.execute(h.admission.claim_next())
    assert not sent
    saved = row(h.service, "successor")
    assert saved["_missing_successor_resume"]["state"] == "reserved"


@pytest.mark.parametrize("change", ["active_claim", "child", "late_answer", "wrong_parent", "not_submitted"])
def test_resume_rejects_active_or_ambiguous_state(task, change):
    h = stopped_successor(task)
    if change == "active_claim":
        update(h.service, "successor", _claim_id="active", _claim_until=2000)
    elif change == "child":
        with h.service.store.transaction() as db:
            db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("owner", "child", "fixture-hash",
                       json.dumps({"request_id": "child", "_supersedes_request_id": "successor"})))
    elif change == "late_answer":
        update(h.service, "successor", status="succeeded", content="late")
    elif change == "not_submitted":
        update(h.service, "successor", _submission_started=False)
    else:
        update(h.service, "successor", _submission_parent_message_id="wrong")
    with pytest.raises(Exception):
        h.service.resume_missing_successor("owner", "successor", h.envelope)
    assert "_missing_successor_resume" not in row(h.service, "successor")


def test_ordinary_same_id_submit_never_consumes_missing_successor_retry(task):
    h = stopped_successor(task)
    before = deepcopy(row(h.service, "successor"))
    assert h.service.submit("owner", h.envelope)["status"] == "failed"
    assert row(h.service, "successor") == before and h.admission.claim_next() is None
