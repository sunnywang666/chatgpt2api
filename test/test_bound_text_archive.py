"""Original bound Chat continuation, with isolated upstream and SQLite data."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from services.conversation_binding_service import ConversationBindingError, ConversationBindingService
from services.openai_backend_api import OpenAIBackendAPI
from services.text_task_service import TextTaskService
from test.test_text_task_service import QueuedExecutor


@pytest.fixture
def bound_chat(monkeypatch):
    import services.conversation_binding_service as module
    state = SimpleNamespace(document={
        "conversation_id": "original-chat", "current_node": "previous-answer", "is_archived": True,
        "mapping": {"previous-answer": {"message": {"id": "previous-answer",
            "author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True}}},
    }, calls=[], read_error=None, patch_error=None, after_patch=None)

    class Backend:
        base_url = "https://chatgpt.com"
        set_conversation_archived = OpenAIBackendAPI.set_conversation_archived

        def __init__(self, access_token):
            assert access_token == "fixture-token"
            self.session = SimpleNamespace(patch=self.patch)

        def _headers(self, path, extra):
            return extra

        def _get_conversation(self, chat):
            assert chat == "original-chat"
            state.calls.append("read")
            if state.read_error:
                raise state.read_error
            return deepcopy(state.document)

        def patch(self, url, *, headers, json, timeout):
            assert url.endswith("/conversation/original-chat") and json == {"is_archived": False}
            state.calls.append("restore")
            if state.patch_error:
                raise state.patch_error
            state.document["is_archived"] = False
            if state.after_patch:
                state.after_patch(state.document)
            return SimpleNamespace(status_code=200)

        def get_conversation_parent_message_id(self, chat):
            return "new-answer"

        def close(self):
            pass

    accounts = Mock()
    accounts.get_bound_account_identity.return_value = "original-account"
    accounts.get_bound_text_access_token.return_value = "fixture-token"
    accounts.conversation_binding_lock.side_effect = lambda *_: nullcontext()
    monkeypatch.setattr(module, "account_service", accounts)
    monkeypatch.setattr(module, "OpenAIBackendAPI", Backend)

    def events(backend, **kwargs):
        state.calls.append("send")
        assert kwargs["conversation_id"] == "original-chat"
        assert kwargs["parent_message_id"] == "previous-answer"
        yield {"type": "conversation.delta", "conversation_id": "original-chat", "delta": "answer"}

    monkeypatch.setattr(module, "conversation_events", events)
    state.accounts = accounts
    state.body = {"client_request_id": "new-original", "provider_binding_id": "original-binding",
        "provider_account_identity": "original-account", "client_conversation_id": "product-session",
        "conversation_id": "original-chat", "parent_message_id": "previous-answer",
        "messages": [{"role": "user", "content": "next authorized work"}]}
    state.service = ConversationBindingService()
    return state


def test_internal_archived_chat_is_verified_restored_then_sent(bound_chat):
    h = bound_chat
    result = h.service.complete_text(h.body)
    assert h.calls == ["read", "read", "restore", "read", "send"]
    assert result["content"] == "answer" and result["conversation_id"] == "original-chat"
    assert result["provider_account_identity"] == "original-account"
    h.accounts.create_conversation_binding.assert_not_called()


def test_unarchived_internal_chat_only_checks_cursor_then_sends(bound_chat):
    h = bound_chat
    h.document["is_archived"] = False
    h.service.complete_text(h.body)
    assert h.calls == ["read", "send"]


@pytest.mark.parametrize("value", [None, "false", 0])
def test_unknown_visibility_does_not_guess_chat_is_open(bound_chat, value):
    h = bound_chat
    h.document["is_archived"] = value
    with pytest.raises(ConversationBindingError) as failure:
        h.service.complete_text(h.body)
    assert failure.value.code == "CHAT_ARCHIVE_RESTORE_UNCONFIRMED"
    assert h.calls == ["read"]


@pytest.mark.parametrize("change", ["chat", "cursor", "missing_parent", "account"])
def test_binding_drift_never_restores_or_sends(bound_chat, change):
    h = bound_chat
    if change == "chat":
        h.document["conversation_id"] = "another-chat"
    elif change == "cursor":
        h.document["current_node"] = "another-turn"
    elif change == "missing_parent":
        h.document["mapping"] = {}
    else:
        h.accounts.get_bound_account_identity.return_value = "another-account"
    with pytest.raises(ConversationBindingError):
        h.service.complete_text(h.body)
    assert "restore" not in h.calls and "send" not in h.calls


@pytest.mark.parametrize("failure", ["read_timeout", "restore_timeout", "restore_unconfirmed", "cursor_after_restore"])
def test_unconfirmed_restore_cannot_submit_and_persists_not_sent(bound_chat, tmp_path, failure):
    h = bound_chat
    if failure == "read_timeout":
        h.read_error = TimeoutError("fixture read")
    elif failure == "restore_timeout":
        h.patch_error = TimeoutError("fixture restore")
    elif failure == "restore_unconfirmed":
        h.after_patch = lambda d: d.update(is_archived=True)
    else:
        h.after_patch = lambda d: d.update(current_node="foreign-turn")
    queue = QueuedExecutor()
    tasks = TextTaskService(tmp_path / "tasks.sqlite3", h.service.complete_text, queue)
    tasks.submit("owner", h.body, source="internal:listing")
    queue.run()
    with tasks._db() as db:
        receipt = tasks.store.read_receipt(db, "text", "owner", "new-original")
    assert receipt["status"] == "failed" and receipt["upstream_outcome"] == "not_sent"
    assert receipt["error_code"] == "CHAT_ARCHIVE_RESTORE_UNCONFIRMED"
    assert receipt["conversation_id"] == "original-chat"
    assert "send" not in h.calls


def test_old_failed_unknown_is_not_replayed_by_archive_fix(bound_chat, tmp_path):
    h = bound_chat
    queue = QueuedExecutor()
    path = tmp_path / "tasks.sqlite3"
    tasks = TextTaskService(path, h.service.complete_text, queue,
        recovery_reader=Mock(side_effect=AssertionError("no recovery mutation")))
    tasks.submit("owner", h.body, source="internal:listing")
    tasks._update("owner", "new-original", status="failed", error_code="RESULT_UNRECOVERABLE",
        upstream_outcome="unknown", _execution_wait_ended_at=1, _executing=False, _turn_reserved=False)
    queue.run()
    result = tasks.submit("owner", h.body, source="internal:listing")
    assert result["status"] == "failed" and result["upstream_outcome"] == "unknown"
    assert h.calls == [] and queue.calls == []
