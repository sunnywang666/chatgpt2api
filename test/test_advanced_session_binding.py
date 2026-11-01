"""An isolated retry must leave the source discussion untouched at the send edge."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import task_results
from services import conversation_binding_service as module
from services.conversation_binding_service import ConversationBindingService, ConversationBindingError
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_advanced_session_successor import advanced
from test.test_explicit_unknown_successor import task, row
from test.test_no_final_successor import product_no_final


def source():
    original = dict(request_id="old", request_message_id="old-user", provider_binding_id="old-binding",
                    provider_account_identity="account", client_conversation_id="old-client",
                    conversation_id="old-chat", parent_message_id="gallery", request_parent_message_id="gallery")
    def node(parent, role, text, *, final=False, thoughts=False):
        return {"parent": parent, "message": {"author": {"role": role},
            "status": "finished_successfully" if final or role == "user" else "in_progress",
            "end_turn": final, "channel": "final" if final else None,
            "content": {"content_type": "thoughts" if thoughts else "text", "parts": [text]}}}
    mapping = {"gallery": node(None, "assistant", "saved gallery", final=True),
        "old-user": node("gallery", "user", "old prompt"),
        "thoughts": node("old-user", "assistant", "", thoughts=True),
        "discussion": node("thoughts", "user", "Why is the prompt long?"),
        "discussion-final": node("discussion", "assistant", "Prompt discussion", final=True)}
    for key, value in mapping.items():
        value["message"]["id"] = key
    document = dict(conversation_id="old-chat", current_node="discussion-final", is_archived=False, mapping=mapping)
    result = ConversationBindingService._read_text_request_result(None, original, document=document)
    proof = dict(protocol="advanced-session-successor-v1", original_request_id="old",
        original_request_message_id="old-user", original_binding_id="old-binding",
        original_account_identity="account", original_conversation_id="old-chat",
        original_head_message_id="discussion-final", completed_successor_evidence=result["_completed_successor_evidence"])
    return original, document, proof


@pytest.mark.parametrize("change", ["head", "archived", "conversation", "late_original", "new_user", "active_tool", "proof", "account"])
def test_fresh_guard_rejects_changed_source(change):
    original, document, proof = source()
    if change == "head": document["current_node"] = "thoughts"
    elif change == "archived": document["is_archived"] = True
    elif change == "conversation": document["conversation_id"] = "other"
    elif change == "late_original":
        document["mapping"]["thoughts"]["message"].update(status="finished_successfully", end_turn=True,
            channel="final", content={"content_type": "text", "parts": ["late original answer"]})
    elif change == "new_user":
        document["mapping"]["new-user"] = {"parent": "discussion-final", "message": {
            "id": "new-user", "author": {"role": "user"}, "content": {"content_type": "text", "parts": ["new turn"]}}}
    elif change == "active_tool":
        document["mapping"]["thoughts"]["message"].update(recipient="python")
    elif change == "account": original["provider_account_identity"] = "other-account"
    else: proof["completed_successor_evidence"]["successor_request_message_id"] = "foreign"
    with pytest.raises(ConversationBindingError):
        ConversationBindingService._check_advanced_session_document(None, document, original, proof)


@pytest.mark.parametrize("drift_at_send", [False, True])
def test_real_binding_path_rechecks_old_graph_and_sends_only_new_conversation(monkeypatch, drift_at_send):
    original, document, proof = source()
    original_document = deepcopy(document)
    backend = SimpleNamespace(close=Mock(), set_conversation_archived=Mock())
    queries, sends, locks = [], [], []
    def get(conversation_id, **_kwargs):
        queries.append(conversation_id)
        if conversation_id == "old-chat": return deepcopy(document)
        assert conversation_id == "new-chat"
        return {"conversation_id": "new-chat", "current_node": "new-final", "is_archived": False,
            "mapping": {"new-user": {"parent": "new-root", "message": {"id": "new-user", "author": {"role": "user"}}},
                "new-final": {"parent": "new-user", "message": {"id": "new-final", "author": {"role": "assistant"},
                    "status": "finished_successfully", "end_turn": True, "channel": "final",
                    "content": {"content_type": "text", "parts": ["new business result"]}}}}}
    backend._get_conversation = get
    monkeypatch.setattr(module, "OpenAIBackendAPI", lambda **_kwargs: backend)
    accounts = module.account_service
    monkeypatch.setattr(accounts, "create_text_conversation_binding", lambda **kw: ("new-binding", "account"))
    monkeypatch.setattr(accounts, "get_bound_text_access_token", lambda *_a, **_k: "test-token")
    monkeypatch.setattr(accounts, "get_bound_account_identity", lambda _binding: "account")
    monkeypatch.setattr(accounts, "mark_text_used", lambda _token: None)
    monkeypatch.setattr(accounts, "conversation_binding_lock", lambda b, c: (locks.append((b, c)) or nullcontext()))
    def events(active_backend, **kwargs):
        assert kwargs["conversation_id"] == "" and kwargs["parent_message_id"] == ""
        if drift_at_send: document["current_node"] = "moved-by-external-client"
        active_backend.text_pre_send_check(lambda *_a, **_k: None)
        sends.append(kwargs)
        yield {"type": "conversation.delta", "conversation_id": "new-chat", "delta": "new business result"}
    monkeypatch.setattr(module, "conversation_events", events)
    body = {"model": "test-model", "messages": [{"role": "user", "content": "complete context"}],
        "client_conversation_id": "new-client", "provider_account_identity": "account", "_text_only_binding": True,
        "_requested_account_identity": "account", "_request_message_id": "new-user", "product_work": {},
        "_advanced_session": proof, "_advanced_session_original": original}
    if drift_at_send:
        with pytest.raises(ConversationBindingError, match="advanced original conversation changed"):
            ConversationBindingService().complete_text(body)
        assert not sends
    else:
        result = ConversationBindingService().complete_text(body)
        assert result["content"] == "new business result" and result["conversation_id"] == "new-chat"
        assert len(sends) == 1 and document == original_document
    assert queries[:2] == ["old-chat", "old-chat"]
    assert ("old-binding", "old-client") in locks and ("new-binding", "new-client") in locks
    backend.set_conversation_archived.assert_not_called()
    backend.close.assert_called_once()


@pytest.mark.parametrize(("reject_before_post", "restart_before_reclaim"), [
    pytest.param(False, False, id="first-claim-succeeds"),
    pytest.param(True, False, id="pre-send-rejection-reclaims-binding"),
    pytest.param(True, True, id="pre-send-rejection-restarts-service-before-reclaim"),
])
def test_submit_claim_binding_final_send_guard_and_saved_result(
        advanced, monkeypatch, reject_before_post, restart_before_reclaim):
    """A pre-send rejection reclaims one allocated binding without a second POST."""
    h = advanced
    _, document, _ = source()
    replacements = {
        "old-chat": "original-chat", "gallery": "previous-answer",
        "old-user": h.old["request_message_id"],
        "discussion-final": "external-final", "discussion": "external-user",
    }

    def translate(value):
        if isinstance(value, str):
            return replacements.get(value, value)
        if isinstance(value, dict):
            return {replacements.get(key, key): translate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [translate(item) for item in value]
        return value

    document = translate(document)
    original_document = deepcopy(document)
    backend = SimpleNamespace(close=Mock(), set_conversation_archived=Mock())
    queries, sends, allocated, event_attempts = [], [], [], []
    reject_first_send = [False]

    def get(conversation_id, **_kwargs):
        queries.append(conversation_id)
        if conversation_id == "original-chat":
            return deepcopy(document)
        assert conversation_id == "new-chat"
        user = backend.text_request_message_id
        return {
            "conversation_id": "new-chat", "current_node": "new-final", "is_archived": False,
            "mapping": {
                user: {"parent": "new-root", "message": {"id": user, "author": {"role": "user"}}},
                "new-final": {"parent": user, "message": {
                    "id": "new-final", "author": {"role": "assistant"},
                    "status": "finished_successfully", "end_turn": True, "channel": "final",
                    "content": {"content_type": "text", "parts": ["new business result"]},
                }},
            },
        }

    backend._get_conversation = get
    monkeypatch.setattr(module, "OpenAIBackendAPI", lambda **_kwargs: backend)
    for accounts in [module.account_service, h.admission.accounts]:
        monkeypatch.setattr(accounts, "get_bound_account_identity", lambda _binding: "original-account")

    def create_text_conversation_binding(**kwargs):
        allocated.append(kwargs)
        return "new-binding", "original-account"

    monkeypatch.setattr(module.account_service, "create_text_conversation_binding", create_text_conversation_binding)
    monkeypatch.setattr(module.account_service, "get_bound_text_access_token", lambda *_a, **_k: "fixture-token")
    monkeypatch.setattr(module.account_service, "mark_text_used", lambda *_args: None)
    monkeypatch.setattr(module.account_service, "conversation_binding_lock", lambda *_args: nullcontext())

    real_rows = h.admission._rows

    def rows_at_send_guard():
        rows = real_rows()
        if not reject_first_send[0]:
            return rows
        return [{**account, "status": "禁用"} for account in rows]

    monkeypatch.setattr(h.admission, "_rows", rows_at_send_guard)

    def events(active_backend, **kwargs):
        event_attempts.append(None)
        receipt = row(h.service, "isolated")
        assert receipt["provider_binding_id"] == "new-binding"
        assert not receipt.get("conversation_id")
        assert kwargs["conversation_id"] == "" and kwargs["parent_message_id"] == ""
        active_backend.text_pre_send_check(lambda *_a, **_k: None)
        if reject_before_post and len(event_attempts) == 1:
            # This is evaluated by PoolAdmission's real final send guard,
            # after allocation but before the mocked transport POST.
            reject_first_send[0] = True
        current_request.get().before_send_with_marker()
        reject_first_send[0] = False
        sends.append("POST")
        yield {"type": "conversation.delta", "conversation_id": "new-chat", "delta": "new business result"}

    monkeypatch.setattr(module, "conversation_events", events)
    h.service.runner = ConversationBindingService().complete_text
    h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    frozen_child_input = h.service.store.load_input(row(h.service, "isolated")["_input_ref"])
    gallery_before = row(h.service, "gallery")

    first_claim = h.admission.claim_next()
    assert first_claim is not None
    h.admission.execute(first_claim)
    reject_first_send[0] = False
    if reject_before_post:
        after_rejection = row(h.service, "isolated")
        assert after_rejection["status"] == "queued"
        assert after_rejection["upstream_outcome"] == "not_sent"
        assert after_rejection["provider_binding_id"] == "new-binding"
        assert not after_rejection.get("conversation_id") and after_rejection["_submission_started"] is False
        assert h.service.store.load_input(after_rejection["_input_ref"]) == frozen_child_input
        assert "provider_binding_id" not in frozen_child_input and sends == [] and len(allocated) == 1

        if restart_before_reclaim:
            h.service = TextTaskService(h.service.path, runner=ConversationBindingService().complete_text,
                                        clock=h.clock, admission=h.admission)
            h.admission.register("text", lambda ctx, body: h.service._run(ctx.owner, ctx.request_id, body))
        h.clock.now += 2
        claim = h.admission.claim_next()
        assert claim is not None and claim.claim != first_claim.claim
        h.admission.execute(claim)
    receipt = row(h.service, "isolated")
    assert receipt["status"] == "succeeded", receipt.get("error_code")
    assert receipt["content"] == "new business result"
    assert receipt["provider_binding_id"] == "new-binding" and receipt["conversation_id"] == "new-chat"
    assert receipt["parent_message_id"] == "new-final" and receipt["_submission_started"] is True
    original_reads = 4 if reject_before_post else 2
    assert sends == ["POST"] and len(allocated) == 1 and queries[:original_reads] == ["original-chat"] * original_reads
    assert row(h.service) == h.old and row(h.service, "gallery") == gallery_before and document == original_document

    monkeypatch.setattr(task_results, "text_task_service", h.service)
    monkeypatch.setattr(task_results, "require_identity", lambda *_args, **_kwargs: {"role": "admin", "id": "owner"})
    app = FastAPI()
    app.include_router(task_results.create_router())
    with TestClient(app) as client:
        result = client.get("/api/task-results/text/isolated")
    assert result.status_code == 200, result.text
    result_receipt = result.json()["receipt"]
    assert result_receipt["provider_binding_id"] == "new-binding"
    assert result_receipt["advanced_session"]["provider_binding_id"] == "new-binding"
    backend.set_conversation_archived.assert_not_called()
    assert backend.close.call_count == (2 if reject_before_post else 1)
