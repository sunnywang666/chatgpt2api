"""An explicit legacy text continuation keeps the original task and context."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from services.conversation_binding_service import ConversationBindingError, ConversationBindingService
from services.request_context import current_request
from test.test_explicit_unknown_successor import task, row, update
from test.test_bound_text_archive import bound_chat
from test.test_pool_admission import build


def no_final(h):
    update(h.service, recovery_reason="REQUEST_RESULT_NOT_FOUND", recovery_no_result_reads=3,
           _attempt_reason="ORIGINAL_RESULT_NO_FINAL", _attempt_finished_at=h.clock(),
           request_parent_message_id="previous-answer")
    h.old = row(h.service)
    h.envelope["continue_after_no_final"] = True
    return h


def graph(h):
    user = h.old["request_message_id"]
    def node(key, parent, children, role, content, end=None):
        return {"parent": parent, "children": children, "message": {"id": key,
            "author": {"role": role}, "status": "finished_successfully", "end_turn": end,
            "channel": None, "content": content}}
    return {"conversation_id": "original-chat", "current_node": "tool", "is_archived": False,
        "mapping": {
            user: node(user, "previous-answer", ["code"], "user", {"content_type": "text", "parts": ["original input"]}),
            "code": node("code", user, ["tool"], "assistant", {"content_type": "code", "text": "inspect input"}, False),
            "tool": node("tool", "code", [], "tool", {"content_type": "multimodal_text", "parts": ["retained tool context"]})}}


def user_leaf_graph(h):
    document = graph(h)
    user = h.old["request_message_id"]
    document["mapping"] = {user: document["mapping"][user]}
    document["mapping"][user]["children"] = []
    document["current_node"] = user
    return document


def test_explicit_mode_preserves_original_and_immutable_input_after_restart(task):
    h = no_final(task)
    first = h.service.submit("owner", h.envelope)
    assert first["status"] == "queued"
    original_body = h.service.store.load_input(row(h.service, "successor")["_input_ref"])
    assert original_body == {**h.body, "client_request_id": "successor", "supersedes_request_id": "old",
                             "continue_after_no_final": True}
    _, store, admission = build(h.root, h.clock)
    service = type(h.service)(store.path, runner=h.service.runner, clock=h.clock, admission=admission)
    admission.register("text", lambda ctx, body: service._run(ctx.owner, ctx.request_id, body))
    admission.execute(admission.claim_next())
    assert len(h.sent) == 1 and h.sent[0]["_supersedes_no_final_original"] == h.old
    assert service.submit("owner", h.envelope)["status"] == "succeeded"
    assert row(service) == h.old
    with pytest.raises(ConversationBindingError):
        service.submit("owner", {**h.envelope, "client_request_id": "second"})


@pytest.mark.parametrize("changes", [
    {"recovery_no_result_reads": 2}, {"_attempt_finished_at": None}, {"_attempt_finished_at": float("inf")},
    {"_attempt_reason": "other"}, {"_execution_wait_ended_at": None}, {"_work_key": "work"},
    {"_completion": {"state": "checking"}}, {"_retry_cursor": {"anything": True}},
    {"_recovery_paused": True}, {"_executing": True}, {"_turn_reserved": True},
])
def test_unqualified_original_cannot_enqueue(task, changes):
    h = no_final(task); update(h.service, **changes)
    with pytest.raises(ConversationBindingError):
        h.service.submit("owner", h.envelope)
    assert row(h.service, "successor") is None and not h.sent


def test_mode_is_explicit_and_cannot_be_changed_on_same_id(task):
    h = no_final(task)
    without = {k: v for k, v in h.envelope.items() if k != "continue_after_no_final"}
    with pytest.raises(ConversationBindingError): h.service.submit("owner", without)
    h.service.submit("owner", h.envelope)
    with pytest.raises(ConversationBindingError): h.service.submit("owner", without)
    from api.ai import ConversationBindingTextRequest
    assert ConversationBindingTextRequest.model_validate(h.envelope).model_dump(exclude_unset=True) == h.envelope
    for bad in ({**h.envelope, "continue_after_no_final": False},
                {**h.envelope, "continue_after_no_final": "true"},
                {**h.body, "continue_after_no_final": True},
                {**h.envelope, "derived_input": {"kind": "any"}}):
        with pytest.raises(ValidationError): ConversationBindingTextRequest.model_validate(bad)


@pytest.mark.parametrize("change", ["archived", "parent", "head", "fork", "later_user", "code_running",
                                  "tool_running", "final", "end_turn", "children", "alias"])
def test_graph_changes_fail_closed(task, change):
    h = no_final(task); d = graph(h)
    assert ConversationBindingService._no_final_successor_tail(d, h.old)[0] == "tool"
    if change == "archived": d["is_archived"] = True
    elif change == "parent": d["mapping"][h.old["request_message_id"]]["parent"] = "wrong"
    elif change == "head": d["current_node"] = "code"
    elif change == "fork": d["mapping"]["other"] = {"parent": h.old["request_message_id"]}
    elif change == "later_user": d["mapping"]["tool"]["message"]["author"]["role"] = "user"
    elif change.endswith("running"): d["mapping"][change.split("_")[0]]["message"]["status"] = "in_progress"
    elif change == "final": d["mapping"]["code"]["message"]["content"] = {"content_type": "text", "parts": ["partial answer"]}
    elif change == "end_turn": d["mapping"]["code"]["message"]["end_turn"] = True
    elif change == "children": d["mapping"]["tool"]["children"] = ["missing"]
    elif change == "alias": d["mapping"]["tool"]["message"]["id"] = "alias"
    with pytest.raises(ConversationBindingError): ConversationBindingService._no_final_successor_tail(d, h.old)


@pytest.mark.parametrize("tail,drift", [
    ("tool", None), ("tool", "end_turn"), ("tool", "tool_content"), ("tool", "code_content"),
    ("user", None), ("user", "user_content"), ("user", "late_answer"), ("user", "in_progress"),
    ("user", "head"), ("user", "unlisted_child"), ("user", "archived"), ("user", "parent"),
    ("user", "alias"),
])
def test_fresh_tail_is_used_and_rechecked_without_rewriting_original(task, bound_chat, monkeypatch, tail, drift):
    import services.conversation_binding_service as module
    h, chat = no_final(task), bound_chat
    chat.document = user_leaf_graph(h) if tail == "user" else graph(h)
    expected_tail = h.old["request_message_id"] if tail == "user" else "tool"
    h.service.runner = chat.service.complete_text
    original_get = module.OpenAIBackendAPI._get_conversation
    reads = []
    def read(backend, cid, **kw):
        reads.append(cid)
        return original_get(backend, cid)
    monkeypatch.setattr(module.OpenAIBackendAPI, "_get_conversation", read)
    sends = []
    def events(backend, **kwargs):
        if drift == "end_turn":
            chat.document["mapping"]["code"]["message"]["end_turn"] = True
        elif drift == "tool_content":
            chat.document["mapping"]["tool"]["message"]["content"]["parts"] = ["changed context"]
        elif drift == "code_content":
            chat.document["mapping"]["code"]["message"]["content"]["text"] = "changed code"
        elif drift == "user_content":
            chat.document["mapping"][expected_tail]["message"]["content"]["parts"] = ["changed input"]
        elif drift == "late_answer":
            chat.document["mapping"][expected_tail]["children"] = ["late"]
            chat.document["mapping"]["late"] = {"parent": expected_tail, "children": [], "message": {
                "id": "late", "author": {"role": "assistant"}, "status": "finished_successfully",
                "end_turn": True, "content": {"content_type": "text", "parts": ["original answer"]}}}
            chat.document["current_node"] = "late"
        elif drift == "in_progress":
            chat.document["mapping"][expected_tail]["message"]["status"] = "in_progress"
        elif drift == "head":
            chat.document["current_node"] = "different"
        elif drift == "unlisted_child":
            chat.document["mapping"]["hidden"] = {"parent": expected_tail}
        elif drift == "archived":
            chat.document["is_archived"] = True
        elif drift == "parent":
            chat.document["mapping"][expected_tail]["parent"] = "different"
        elif drift == "alias":
            chat.document["mapping"][expected_tail]["message"]["id"] = "alias"
        backend.text_pre_send_check(None)
        current_request.get().before_send()
        sends.append(kwargs)
        assert kwargs["parent_message_id"] == expected_tail and kwargs["conversation_id"] == "original-chat"
        assert kwargs["messages"] == h.body["messages"] and kwargs["model"] == h.body["model"]
        backend.text_cursor_callback({"request_parent_message_id": expected_tail, "_submission_parent_message_id": expected_tail})
        yield {"type": "conversation.delta", "conversation_id": "original-chat", "delta": "new answer"}
    monkeypatch.setattr(module, "conversation_events", events)
    monkeypatch.setattr(ConversationBindingService, "_read_text_request_result", staticmethod(
        lambda backend, receipt: {"status": "succeeded", "content": "new answer", "parent_message_id": "new-answer"}))
    h.service.submit("owner", h.envelope)
    h.admission.execute(h.admission.claim_next())
    result = row(h.service, "successor")
    assert reads == ["original-chat", "original-chat"]
    assert row(h.service) == h.old
    assert result["_submission_parent_message_id"] == "previous-answer"
    if drift:
        assert not sends and result["upstream_outcome"] == "not_sent"
        assert result["status"] == "failed" and result["error_code"] == "CHAT_SUPERSEDE_CURSOR_CHANGED"
    else:
        assert len(sends) == 1 and result["status"] == "succeeded"
        assert result["request_parent_message_id"] == expected_tail
        assert h.service.submit("owner", h.envelope)["status"] == "succeeded"
        with pytest.raises(ConversationBindingError):
            h.service.submit("owner", {**h.envelope, "client_request_id": "second"})
        assert len(sends) == 1


def test_no_final_two_processes_keep_one_successor_and_one_claim(task):
    from test.test_explicit_unknown_successor import test_two_processes_cannot_register_two_successors_and_only_one_can_claim
    test_two_processes_cannot_register_two_successors_and_only_one_can_claim(no_final(task))
