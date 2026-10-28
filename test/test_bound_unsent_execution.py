"""Exercise the bound retry through its real runner and final send guards."""
from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import Mock

import pytest

from services.conversation_binding_service import ConversationBindingService
from services.request_context import current_request
from test.test_unsent_bound_retry import retry, read


@pytest.mark.parametrize("drift", [False, True])
def test_verified_unsent_retry_checks_the_branch_again_at_send(retry, monkeypatch, drift):
    import services.conversation_binding_service as binding

    service, admission, store, _clock, original_body, _envelope, _sent, _before, _root = retry
    with store.transaction() as db:
        original = store.read_receipt(db, "text", "owner", "P")
        original["request_message_id"] = "P-user"
        store.write_receipt(db, "text", "owner", "P", original)
    original = read(service, "P")
    document = {
        "conversation_id": "conversation-0", "is_archived": False, "current_node": "B",
        "mapping": {
            "root": {"parent": None, "children": ["A"], "message": {
                "id": "root", "author": {"role": "user"}}},
            "A": {"parent": "root", "children": ["B"], "message": {
                "id": "A", "author": {"role": "assistant"}, "status": "finished_successfully",
                "end_turn": True, "content": {"content_type": "text", "parts": ["previous answer"]}}},
            "B": {"parent": "A", "children": [], "message": {
                "id": "B", "author": {"role": "assistant"}, "status": "finished_successfully",
                "end_turn": True, "content": {"content_type": "text", "parts": ["previous answer tail"]}}},
        },
    }
    reads, sends = [], []

    class Backend:
        def __init__(self, access_token):
            assert access_token == "isolated-token"

        def _get_conversation(self, conversation_id, **_kwargs):
            assert conversation_id == "conversation-0"
            reads.append(conversation_id)
            return deepcopy(document)

        def close(self):
            pass

    accounts = Mock()
    accounts.get_bound_account_identity.return_value = "account-0"
    accounts.get_bound_text_access_token.return_value = "isolated-token"
    accounts.conversation_binding_lock.side_effect = lambda *_: nullcontext()
    monkeypatch.setattr(binding, "account_service", accounts)
    monkeypatch.setattr(binding, "OpenAIBackendAPI", Backend)

    def events(backend, **kwargs):
        # Change only the reverse edge after the runner's initial read. The
        # apparent head and explicit children list still look unchanged.
        if drift:
            document["mapping"]["hidden"] = {"parent": "A", "children": [], "message": {
                "id": "hidden", "author": {"role": "assistant"}, "status": "finished_successfully",
                "end_turn": True}}
        backend.text_pre_send_check(None)
        current_request.get().before_send()
        sends.append(deepcopy(kwargs))
        yield {"type": "conversation.delta", "conversation_id": "conversation-0", "delta": "new retry answer"}

    monkeypatch.setattr(binding, "conversation_events", events)
    monkeypatch.setattr(ConversationBindingService, "_read_text_request_result", staticmethod(
        lambda _backend, _receipt, **_kwargs: {
            "status": "succeeded", "content": "new retry answer", "parent_message_id": "retry-final",
            "_upstream_terminal": True, "upstream_outcome": "completed",
        }))
    service.runner = ConversationBindingService().complete_text
    result = service.retry_bound_attempt("owner", "P")
    child_id = result["attempt"]["request_id"]
    assert child_id != "P" and result["result_ready"] is False
    assert read(service, "P") == original  # Previous answers are never adopted as P.
    assert read(service, child_id)["parent_message_id"] == "B"
    context = admission.claim_next()
    assert context and context.request_id == child_id
    admission.execute(context)
    child = read(service, child_id)
    assert reads == ["conversation-0"] * 3  # Resolve, runner, final pre-send.
    assert read(service, "P") == original
    if drift:
        assert sends == []
        assert child["upstream_outcome"] == "not_sent"
        assert child["error_code"] == "CHAT_BOUND_RETRY_CURSOR_UNAVAILABLE"
    else:
        assert len(sends) == 1
        assert sends[0]["parent_message_id"] == "B"
        assert sends[0]["messages"] == original_body["messages"]
        assert child["status"] == "succeeded" and child["content"] == "new retry answer"
        repeated = service.retry_bound_attempt("owner", "P")
        assert repeated["existing"] and repeated["result_ready"]
        assert repeated["attempt"]["request_id"] == child_id
        assert reads == ["conversation-0"] * 3 and len(sends) == 1
