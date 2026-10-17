"""One bounded N -> C continuation after an R -> P -> N native no-final chain.

All identities and conversation nodes are synthetic.  C keeps N's frozen input,
but only sends from N's freshly re-read physical tool tail.
"""
from copy import deepcopy

import pytest

from services.conversation_binding_service import ConversationBindingError, ConversationBindingService
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_bound_text_archive import bound_chat
from test.test_rejected_chat_successor import (
    prove_no_final_model_rejection,
    receipt,
    task,
    update,
)


def native_chain(task):
    retry = prove_no_final_model_rejection(task)
    task.service.submit("owner", retry, source="internal:content")
    root = receipt(task.service)
    rejected = receipt(task.service, "rejected-parent")
    native = receipt(task.service, "model-retry")
    update(task.service, "old", request_parent_message_id="previous-answer")
    update(task.service, "model-retry", status="failed", error_code="RESULT_UNRECOVERABLE",
           upstream_outcome="unknown", recovery_reason="REQUEST_RESULT_NOT_FOUND",
           _attempt_reason="ORIGINAL_RESULT_NO_FINAL", _attempt_finished_at=500,
           recovery_no_result_reads=4, _execution_wait_ended_at=500,
           _submission_started=True, _last_sent_sequence=0, _executing=False,
           _turn_reserved=False, _claim_id=None, _claim_until=None,
           recovery_claim_id=None, recovery_lease_until=None,
           request_parent_message_id="native-root-tail")
    root = receipt(task.service)
    native = receipt(task.service, "model-retry")
    body = task.service.store.load_input(native["_input_ref"])
    envelope = {key: body[key] for key in TextTaskService.SUPERSEDE_FIELDS}
    envelope.update(client_request_id="native-no-final-child", supersedes_request_id="model-retry",
                    continue_after_no_final=True)
    return root, rejected, native, envelope


def graph(root, native):
    def node(message_id, parent, children, role, content, *, end=None, channel=None):
        return {"parent": parent, "children": children, "message": {
            "id": message_id, "author": {"role": role}, "status": "finished_successfully",
            "end_turn": end, "channel": channel, "content": content,
        }}
    root_user, native_user = root["request_message_id"], native["request_message_id"]
    return {"conversation_id": "original-chat", "current_node": "native-tail", "is_archived": False,
            "mapping": {
                root_user: node(root_user, "previous-answer", ["root-code"], "user",
                                {"content_type": "text", "parts": ["root input"]}),
                "root-code": node("root-code", root_user, ["native-root-tail"], "assistant",
                                  {"content_type": "code", "text": "root tool"}, end=False, channel="analysis"),
                "native-root-tail": node("native-root-tail", "root-code", [native_user], "tool",
                                         {"content_type": "multimodal_text", "parts": ["root tool"]}, end=False),
                native_user: node(native_user, "native-root-tail", ["native-code"], "user",
                                  {"content_type": "text", "parts": ["native input"]}),
                "native-code": node("native-code", native_user, ["native-tail"], "assistant",
                                    {"content_type": "code", "text": "native tool"}, end=False, channel="analysis"),
                "native-tail": node("native-tail", "native-code", [], "tool",
                                    {"content_type": "multimodal_text", "parts": ["native tool"]}, end=False),
            }}


def setup_bound_runner(task, bound_chat, monkeypatch, *, drift=None):
    root, rejected, native, envelope = native_chain(task)
    bound_chat.document = graph(root, native)
    sent = []
    import services.conversation_binding_service as binding
    backend_type = binding.OpenAIBackendAPI
    original_get = backend_type._get_conversation
    monkeypatch.setattr(backend_type, "_get_conversation", lambda self, chat, _send=None: original_get(self, chat))

    def events(backend, **kwargs):
        if drift == "root_branch":
            root_user = root["request_message_id"]
            bound_chat.document["mapping"][root_user]["children"].append("late-root-final")
            bound_chat.document["mapping"]["late-root-final"] = {
                "parent": root_user, "children": [], "message": {"id": "late-root-final",
                "author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True,
                "content": {"content_type": "text", "parts": ["late root result"]}}}
        elif drift == "rejected_appears":
            bound_chat.document["mapping"][rejected["request_message_id"]] = {
                "parent": "native-root-tail", "children": [], "message": {"id": rejected["request_message_id"],
                "author": {"role": "user"}, "status": "finished_successfully",
                "content": {"content_type": "text", "parts": ["P appeared"]}}}
        elif drift == "native_final":
            native_user = native["request_message_id"]
            bound_chat.document["mapping"][native_user]["children"].append("late-native-final")
            bound_chat.document["mapping"]["late-native-final"] = {
                "parent": native_user, "children": [], "message": {"id": "late-native-final",
                "author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True,
                "content": {"content_type": "text", "parts": ["late native result"]}}}
        elif drift == "new_user":
            bound_chat.document["mapping"]["native-tail"]["children"].append("new-user")
            bound_chat.document["mapping"]["new-user"] = {
                "parent": "native-tail", "children": [], "message": {"id": "new-user",
                "author": {"role": "user"}, "status": "finished_successfully",
                "content": {"content_type": "text", "parts": ["new input"]}}}
        backend.text_pre_send_check(None)
        current_request.get().before_send()
        sent.append(deepcopy(kwargs))
        yield {"type": "conversation.delta", "conversation_id": "original-chat", "delta": "C answer"}

    monkeypatch.setattr(binding, "conversation_events", events)
    monkeypatch.setattr(ConversationBindingService, "_read_text_request_result", staticmethod(
        lambda backend, receipt: {"status": "succeeded", "content": "C answer", "parent_message_id": "C-tail"}))
    task.service.runner = bound_chat.service.complete_text
    return root, rejected, native, envelope, sent


def test_native_no_final_successor_sends_once_from_native_tail_and_is_idempotent(task, bound_chat, monkeypatch):
    root, rejected, native, envelope, sent = setup_bound_runner(task, bound_chat, monkeypatch)
    before = {name: deepcopy(receipt(task.service, name)) for name in ("old", "rejected-parent", "model-retry")}
    frozen = task.service.store.load_input(native["_input_ref"])

    accepted = task.service.submit("owner", envelope, source="internal:content")
    assert accepted["status"] == "queued"
    child = receipt(task.service, "native-no-final-child")
    assert child["_supersedes_input_hash"]
    assert task.service.store.load_input(child["_input_ref"]) == {
        **frozen, "client_request_id": "native-no-final-child", "supersedes_request_id": "model-retry",
        "continue_after_no_final": True,
    }
    context = task.admission.claim_next()
    assert context and context.request_id == "native-no-final-child"
    task.admission.execute(context)
    assert len(sent) == 1 and sent[0]["parent_message_id"] == "native-tail"
    assert receipt(task.service, "native-no-final-child")["status"] == "succeeded"
    restarted = TextTaskService(task.service.store.path, runner=task.service.runner, admission=task.admission)
    replay = restarted.submit("owner", envelope)
    assert replay["request_id"] == "native-no-final-child" and replay["status"] == "succeeded" and len(sent) == 1
    for name, saved in before.items():
        assert receipt(task.service, name) == saved
    with pytest.raises(ConversationBindingError):
        task.service.submit("owner", {**envelope, "client_request_id": "second-child"}, source="internal:content")
    assert len(sent) == 1


def test_native_no_final_validation_is_read_only_and_uses_one_predicted_sequence(task):
    _, _, _, envelope = native_chain(task)
    with task.service.store.connect() as db:
        before_runtime = task.service.store.runtime(db, "acceptance_sequence", 0)
        before_runtime_rows = db.execute("SELECT name,value FROM task_runtime ORDER BY name").fetchall()
        before_rows = db.execute("SELECT id,request_hash,receipt FROM requests ORDER BY id").fetchall()
    assert task.service.validate_submission("owner", envelope, source="internal:content") is None
    with task.service.store.connect() as db:
        assert task.service.store.runtime(db, "acceptance_sequence", 0) == before_runtime
        assert db.execute("SELECT name,value FROM task_runtime ORDER BY name").fetchall() == before_runtime_rows
        assert db.execute("SELECT id,request_hash,receipt FROM requests ORDER BY id").fetchall() == before_rows
    task.service.submit("owner", envelope, source="internal:content")
    with task.service.store.connect() as db:
        assert task.service.store.runtime(db, "acceptance_sequence", 0) == before_runtime + 1
    assert receipt(task.service, "native-no-final-child")["_sequence"] == before_runtime + 1


def test_native_no_final_child_cannot_recursively_create_d(task):
    _, _, native, envelope = native_chain(task)
    task.service.submit("owner", envelope, source="internal:content")
    child = receipt(task.service, "native-no-final-child")
    update(task.service, "native-no-final-child", status="failed", error_code="RESULT_UNRECOVERABLE",
           upstream_outcome="unknown", recovery_reason="REQUEST_RESULT_NOT_FOUND",
           _attempt_reason="ORIGINAL_RESULT_NO_FINAL", _attempt_finished_at=500,
           recovery_no_result_reads=4, _execution_wait_ended_at=500,
           _submission_started=True, _last_sent_sequence=0, _executing=False,
           _turn_reserved=False, _claim_id=None, _claim_until=None,
           recovery_claim_id=None, recovery_lease_until=None,
           request_parent_message_id="child-native-tail")
    child_body = task.service.store.load_input(child["_input_ref"])
    exact_d = {key: child_body[key] for key in TextTaskService.SUPERSEDE_FIELDS}
    exact_d.update(client_request_id="native-no-final-grandchild",
                   supersedes_request_id="native-no-final-child", continue_after_no_final=True)
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", exact_d, source="internal:content")
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"
    assert receipt(task.service, "native-no-final-grandchild") is None


@pytest.mark.parametrize("drift", ["root_branch", "rejected_appears", "native_final", "new_user"])
def test_native_no_final_successor_rechecks_prefix_and_native_tail_before_send(task, bound_chat, monkeypatch, drift):
    _, _, _, envelope, sent = setup_bound_runner(task, bound_chat, monkeypatch, drift=drift)
    task.service.submit("owner", envelope, source="internal:content")
    context = task.admission.claim_next()
    assert context and context.request_id == "native-no-final-child"
    task.admission.execute(context)
    child = receipt(task.service, "native-no-final-child")
    assert not sent and child["status"] == "failed" and child["upstream_outcome"] == "not_sent"
    assert child["error_code"] in {"CHAT_SUPERSEDE_CURSOR_CHANGED", "CHAT_SUPERSEDE_ORIGINAL_FOUND"}


@pytest.mark.parametrize("request_id,changes", [
    ("old", {"_sequence": 0}),
    ("rejected-parent", {"_sequence": 1}),
    ("rejected-parent", {"_claim_id": "retained", "_claim_until": 0}),
    ("model-retry", {"_sequence": True}),
    ("model-retry", {"_claim_id": "live", "_claim_until": 9999}),
    ("model-retry", {"_recovery_paused": True}),
    ("model-retry", {"_supersedes_input_hash": "changed"}),
    ("model-retry", {"content": "late native", "upstream_outcome": "completed"}),
])
def test_native_no_final_successor_refuses_broken_or_live_lineage(task, request_id, changes):
    _, _, _, envelope = native_chain(task)
    update(task.service, request_id, **changes)
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", envelope, source="internal:content")
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"
    assert receipt(task.service, "native-no-final-child") is None


def test_native_no_final_prefix_rejects_an_intermediate_code_parent(task):
    root, rejected, native, _ = native_chain(task)
    document = graph(root, native)
    native_user = native["request_message_id"]
    document["mapping"]["root-code"]["children"] = [native_user]
    document["mapping"][native_user]["parent"] = "root-code"
    prefix = {
        "root_request_message_id": root["request_message_id"],
        "root_parent_message_id": "previous-answer",
        "rejected_request_message_id": rejected["request_message_id"],
        "native_request_message_id": native_user,
        "native_parent_message_id": "root-code",
    }
    with pytest.raises(ConversationBindingError) as error:
        ConversationBindingService._native_no_final_prefix(document, "original-chat", prefix)
    assert error.value.code == "CHAT_SUPERSEDE_CURSOR_CHANGED"
