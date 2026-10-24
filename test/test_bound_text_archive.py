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

        def _get_conversation(self, chat, *, deadline_monotonic=None):
            assert chat == "original-chat"
            state.calls.append("read")
            if state.read_error:
                raise state.read_error
            return deepcopy(state.document)

        def patch(self, url, *, headers, json, timeout, _account_request_deadline_monotonic):
            assert 0 < timeout <= 60
            assert _account_request_deadline_monotonic > 0
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
    assert h.calls == ["read", "restore", "read", "send"]
    assert result["content"] == "answer" and result["conversation_id"] == "original-chat"
    assert result["provider_account_identity"] == "original-account"
    h.accounts.create_conversation_binding.assert_not_called()


def test_unarchived_internal_chat_only_checks_cursor_then_sends(bound_chat):
    h = bound_chat
    h.document["is_archived"] = False
    h.service.complete_text(h.body)
    assert h.calls == ["read", "send"]


def test_request_cannot_supply_restore_precheck_document(bound_chat):
    h = bound_chat
    h.body["_restore_precheck_document"] = {
        "conversation_id": "foreign-chat", "current_node": "foreign-parent", "is_archived": False,
    }
    h.service.complete_text(h.body)
    assert h.calls == ["read", "restore", "read", "send"]


@pytest.mark.parametrize("read_interval", [45, 90])
def test_restore_progresses_with_exhausted_read_burst(bound_chat, tmp_path, monkeypatch, read_interval):
    """Real pacing must not repeat the initial GET before applying restore."""
    import time
    from services.account_request_pacing import AccountRequestClock
    from services.config import config

    h = bound_chat
    now = [10000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(time, "time", lambda: 1700000000 + now[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(type(config), "account_request_interval_secs", property(lambda _: 0))
    monkeypatch.setattr(type(config), "account_conversation_read_interval_secs", property(lambda _: read_interval))
    monkeypatch.setattr(type(config), "account_conversation_read_burst", property(lambda _: 1))
    clock = AccountRequestClock("fixture", tmp_path / "clock.json")

    class Response:
        status_code = 200
        headers = {}

        def json(self):
            return deepcopy(h.document)

        def close(self):
            pass

    def transport(method, url, **options):
        assert url == "https://fixture.invalid/backend-api/conversation/original-chat"
        assert method in {"GET", "PATCH"}
        h.calls.append("read" if method == "GET" else "restore")
        if method == "PATCH":
            assert options["json"] == {"is_archived": False}
            h.document["is_archived"] = False
        return Response()

    backend = object.__new__(OpenAIBackendAPI)
    backend.base_url = "https://fixture.invalid"
    backend._headers = lambda _path, headers: headers
    backend.session = SimpleNamespace(
        get=lambda url, **options: clock.request(transport, "GET", url, **options),
        patch=lambda url, **options: clock.request(transport, "PATCH", url, **options),
    )
    backend.close = lambda: None
    backend.get_conversation_parent_message_id = lambda _chat: "new-answer"
    monkeypatch.setattr("services.conversation_binding_service.OpenAIBackendAPI", lambda **_: backend)

    if read_interval == 90:
        # The bounded readback can still run out of budget. Do not send until
        # the next original-ID attempt independently confirms the applied PATCH.
        with pytest.raises(ConversationBindingError) as failure:
            h.service.complete_text(h.body)
        assert failure.value.code == "CHAT_ARCHIVE_RESTORE_UNCONFIRMED"
        assert h.calls == ["read", "restore"]
        assert h.document["is_archived"] is False
    result = h.service.complete_text(h.body)
    assert result["content"] == "answer"
    assert h.calls == ["read", "restore", "read", "send"]
    assert now[0] == 10000 + read_interval


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


def test_applied_restore_timeout_requeues_original_across_restart(bound_chat, tmp_path, monkeypatch):
    import json
    from services.request_context import current_request
    from test.test_pool_admission import build

    h = bound_chat
    now = [1000.0]
    (tmp_path / "accounts.json").write_text(json.dumps([{
        "access_token": "fixture-token", "account_id": "fixture-upstream",
        "provider_account_identity": "original-account", "type": "Plus", "status": "正常",
        "quota": 999, "source_type": "web", "conversation_binding_ids": ["original-binding"],
    }]))
    _, store, admission = build(tmp_path, lambda: now[0])
    tasks = TextTaskService(store.path, runner=h.service.complete_text, admission=admission, clock=lambda: now[0])
    admission.register("text", lambda ctx, body: tasks._run(ctx.owner, ctx.request_id, body))
    h.after_patch = lambda _doc: setattr(h, "read_error", TimeoutError("fixture readback timeout"))
    tasks.submit("owner", h.body, source="internal:listing")
    first = admission.claim_next()
    admission.execute(first)
    with store.connect() as db:
        before = store.read_receipt(db, "text", "owner", "new-original")
        request_hash = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", ("owner", "new-original")).fetchone()[0]
    assert before["status"] == "queued" and before["upstream_outcome"] == "not_sent"
    assert before["_submission_started"] is False
    assert before.get("_last_sent_sequence") is None
    assert before["_claim_id"] is None and before["_executing"] is False
    assert h.document["is_archived"] is False
    assert h.calls == ["read", "restore", "read"]

    # New service/admission instances reuse the persisted receipt and input.
    h.read_error = None
    now[0] += 31
    _, restarted_store, restarted = build(tmp_path, lambda: now[0])
    resumed = TextTaskService(restarted_store.path, runner=h.service.complete_text,
                             admission=restarted, clock=lambda: now[0])
    restarted.register("text", lambda ctx, body: resumed._run(ctx.owner, ctx.request_id, body))
    def events(_backend, **kwargs):
        current_request.get().before_send()
        h.calls.append("send")
        yield {"type": "conversation.delta", "conversation_id": "original-chat", "delta": "answer"}
    monkeypatch.setattr("services.conversation_binding_service.conversation_events", events)
    second = restarted.claim_next()
    assert second.request_id == first.request_id and second.claim != first.claim
    restarted.execute(second)
    with restarted_store.connect() as db:
        after = restarted_store.read_receipt(db, "text", "owner", "new-original")
        assert db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", ("owner", "new-original")).fetchone()[0] == request_hash
    assert after["status"] == "succeeded" and after["_submission_started"] is True
    assert after["_last_sent_sequence"] == (before.get("_send_sequence") or 0)
    assert h.calls == ["read", "restore", "read", "read", "send"]
    for key in ("request_id", "provider_binding_id", "provider_account_identity", "client_conversation_id", "request_message_id", "_sequence"):
        assert after[key] == before[key]
    assert resumed.submit("owner", h.body, source="internal:listing")["status"] == "succeeded"
    assert restarted.claim_next() is None
