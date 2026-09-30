"""Explicit internal successors use original durable input; no live accounts."""
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from services.conversation_binding_service import ConversationBindingError, ConversationBindingService
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_bound_text_archive import bound_chat
from test.test_pool_admission import build, Clock, claim_worker


def row(service, request_id="old"):
    with service.store.connect() as db:
        return service.store.read_receipt(db, "text", "owner", request_id)


def update(service, request_id="old", **changes):
    with service.store.transaction() as db:
        r = service.store.read_receipt(db, "text", "owner", request_id)
        r.update(changes)
        service.store.write_receipt(db, "text", "owner", request_id, r)


@pytest.fixture
def task(tmp_path):
    (tmp_path / "accounts.json").write_text(json.dumps([{"access_token": "fixture", "account_id": "physical",
        "provider_account_identity": "original-account", "conversation_binding_ids": ["original-binding"],
        "type": "Plus", "status": "正常", "quota": 99, "source_type": "web"}]))
    clock = Clock()
    _, store, admission = build(tmp_path, clock)
    sent = []
    def runner(body, on_cursor):
        current_request.get().before_send()
        sent.append(deepcopy(body))
        return {"content": "new answer", "parent_message_id": "new-answer"}
    service = TextTaskService(store.path, runner=runner, clock=clock, admission=admission)
    admission.register("text", lambda ctx, body: service._run(ctx.owner, ctx.request_id, body))
    body = {"client_request_id": "old", "client_conversation_id": "product-session",
        "provider_binding_id": "original-binding", "provider_account_identity": "original-account",
        "conversation_id": "original-chat", "parent_message_id": "previous-answer", "model": "fixture-text",
        "image_model": "gpt-image-2", "thinking_effort": "high",
        "messages": [{"role": "user", "content": "original retained instructions"}]}
    service.submit("owner", body, source="internal:content")
    update(service, status="failed", error_code="RESULT_UNRECOVERABLE", upstream_outcome="unknown",
        recovery_reason="REQUEST_MESSAGE_NOT_FOUND", _execution_wait_ended_at=500,
        _submission_started=True, _last_sent_sequence=0, _executing=False, _claim_id=None,
        _claim_until=None, _turn_reserved=False, send_count=1)
    envelope = {k: v for k, v in body.items() if k in TextTaskService.SUPERSEDE_FIELDS}
    envelope.update(client_request_id="successor", supersedes_request_id="old")
    return SimpleNamespace(service=service, admission=admission, clock=clock, sent=sent,
        envelope=envelope, body=body, old=row(service), root=tmp_path)


def test_minimal_reference_persists_exact_input_and_idempotent_success(task):
    h = task
    first = h.service.submit("owner", h.envelope, source="internal:content")
    assert first["supersedes_request_id"] == "old" and first["status"] == "queued"
    r = row(h.service, "successor")
    body = h.service.store.load_input(r["_input_ref"])
    assert body == {**h.body, "client_request_id": "successor", "supersedes_request_id": "old"}
    assert h.service.submit("owner", h.envelope) == first
    ctx = h.admission.claim_next()
    assert ctx.request_id == "successor"
    h.admission.execute(ctx)
    result = h.service.submit("owner", h.envelope)
    assert result["status"] == "succeeded" and result["content"] == "new answer"
    assert len(h.sent) == 1 and h.sent[0]["messages"] == h.body["messages"]
    assert row(h.service) == h.old
    assert h.admission.claim_next() is None


def test_restart_claims_new_request_once_and_does_not_reopen_old_order_head(task):
    h = task
    h.service.submit("owner", h.envelope)
    _, store, restarted = build(h.root, h.clock)
    service = TextTaskService(store.path, runner=h.service.runner, clock=h.clock, admission=restarted)
    restarted.register("text", lambda ctx, body: service._run(ctx.owner, ctx.request_id, body))
    ctx = restarted.claim_next()
    assert ctx.request_id == "successor" and h.admission.claim_next() is None
    restarted.execute(ctx)
    assert len(h.sent) == 1 and row(service) == h.old
    # A later normal same-session turn can advance only after the successor finished.
    service.submit("owner", {**h.body, "client_request_id": "later", "parent_message_id": "new-answer"})
    assert restarted.claim_next().request_id == "later"


@pytest.mark.parametrize("field", ["messages", "model", "image_model", "thinking_effort", "_public_session_ref", "_route"])
def test_content_cannot_be_rebuilt_or_overridden(task, field):
    with pytest.raises(ConversationBindingError, match="cannot be changed") as error:
        task.service.submit("owner", {**task.envelope, field: task.body.get(field)})
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"
    assert row(task.service, "successor") is None


@pytest.mark.parametrize("field", ["provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id", "parent_message_id"])
def test_original_identity_cannot_change(task, field):
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", {**task.envelope, field: "different"})
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"
    assert row(task.service) == task.old


def test_foreign_owner_and_changed_same_id_are_rejected(task):
    with pytest.raises(ConversationBindingError):
        task.service.submit("foreign", task.envelope)
    task.service.submit("owner", task.envelope)
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", {**task.envelope, "parent_message_id": "different"})
    assert error.value.code == "CONVERSATION_REQUEST_CONFLICT"
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", {**task.envelope, "client_request_id": "second"})
    assert error.value.code == "CHAT_SUPERSEDE_CONFLICT"


@pytest.mark.parametrize("changes,code", [
    ({"status": "unknown"}, "CHAT_SUPERSEDE_INVALID"),
    ({"_execution_wait_ended_at": None}, "CHAT_SUPERSEDE_INVALID"),
    ({"recovery_reason": "REQUEST_RESULT_INCOMPLETE"}, "CHAT_SUPERSEDE_INVALID"),
    ({"_recovery_suppressed": True}, "CHAT_SUPERSEDE_INVALID"),
    ({"_claim_id": "live", "_claim_until": 2000}, "CHAT_SUPERSEDE_PREDECESSOR_BUSY"),
    ({"recovery_claim_id": "reader", "recovery_lease_until": 2000}, "CHAT_SUPERSEDE_PREDECESSOR_BUSY"),
    ({"status": "succeeded", "content": "late answer"}, "CHAT_SUPERSEDE_ORIGINAL_FOUND"),
])
def test_predecessor_eligibility(task, changes, code):
    update(task.service, **changes)
    before = row(task.service)
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", task.envelope)
    assert error.value.code == code and row(task.service) == before


def test_missing_or_corrupt_retained_input_cannot_be_accepted(task):
    with task.service.store.connect() as db:
        db.execute("UPDATE requests SET request_hash='incorrect' WHERE id='old'")
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", task.envelope)
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"


def test_ordinary_unknown_still_blocks_and_intervening_turn_is_not_skipped(task):
    h = task
    h.service.submit("owner", {**h.body, "client_request_id": "ordinary-next"})
    assert h.admission.claim_next() is None
    with pytest.raises(ConversationBindingError) as error:
        h.service.submit("owner", h.envelope)
    assert error.value.code == "CHAT_SUPERSEDE_CONFLICT"
    assert row(h.service) == h.old


def submit_worker(root, envelope, ready, start, results):
    service = TextTaskService(Path(root) / "text_tasks.sqlite3", admission=Mock())
    ready.put(True)
    start.wait(10)
    try:
        service.submit("owner", envelope)
        results.put("accepted")
    except ConversationBindingError as exc:
        results.put(exc.code)


def test_two_processes_cannot_register_two_successors_and_only_one_can_claim(task):
    h = task
    mp = multiprocessing.get_context("spawn")
    ready, results, start = mp.Queue(), mp.Queue(), mp.Event()
    workers = [mp.Process(target=submit_worker, args=(str(h.root), {**h.envelope, "client_request_id": name}, ready, start, results))
               for name in ("a", "b")]
    for p in workers: p.start()
    for _ in workers: ready.get(timeout=20)
    start.set()
    assert sorted(results.get(timeout=20) for _ in workers) == ["CHAT_SUPERSEDE_CONFLICT", "accepted"]
    for p in workers:
        p.join(20)
        assert p.exitcode == 0
    ready, results, start = mp.Queue(), mp.Queue(), mp.Event()
    workers = [mp.Process(target=claim_worker, args=(str(h.root), ready, start, results)) for _ in range(2)]
    for p in workers: p.start()
    for _ in workers: ready.get(timeout=20)
    start.set()
    claims = [results.get(timeout=20) for _ in workers]
    assert sum(c is not None for c in claims) == 1
    for p in workers:
        p.join(20)
        assert p.exitcode == 0
    assert row(h.service) == h.old


@pytest.mark.parametrize("late", ["found", "claim"])
def test_send_edge_rechecks_original_after_runner_started(task, late):
    h = task
    def runner(body, on_cursor):
        if late == "found":
            update(h.service, status="succeeded", content="late original")
        else:
            update(h.service, _claim_id="active", _claim_until=2000)
        current_request.get().before_send()
        h.sent.append(body)
    h.service.runner = runner
    h.service.submit("owner", h.envelope)
    h.admission.execute(h.admission.claim_next())
    r = row(h.service, "successor")
    assert r["status"] == ("failed" if late == "found" else "queued")
    assert r["upstream_outcome"] == "not_sent" and not r["_submission_started"]
    assert not h.sent


@pytest.mark.parametrize("change,code", [
    ("old_user", "CHAT_SUPERSEDE_ORIGINAL_FOUND"),
    ("inner_user", "CHAT_SUPERSEDE_ORIGINAL_FOUND"),
    ("parent", "CHAT_SUPERSEDE_CURSOR_CHANGED"),
    ("parent_unfinished", "CHAT_SUPERSEDE_CURSOR_CHANGED"),
    ("child", "CHAT_SUPERSEDE_CURSOR_CHANGED"),
    ("read_failure", "CHAT_SUPERSEDE_READ_UNAVAILABLE"),
])
def test_original_chat_is_checked_before_restore_or_send(task, bound_chat, change, code):
    h, chat = task, bound_chat
    h.service.runner = chat.service.complete_text
    old_user = h.old["request_message_id"]
    if change in {"old_user", "inner_user"}:
        chat.document["mapping"][old_user if change == "old_user" else "alias"] = {"message": {"id": old_user}}
    elif change == "parent": chat.document["current_node"] = "different"
    elif change == "parent_unfinished": chat.document["mapping"]["previous-answer"]["message"]["end_turn"] = False
    elif change == "child": chat.document["mapping"]["child"] = {"parent": "previous-answer"}
    else: chat.read_error = TimeoutError("read failure")
    h.service.submit("owner", h.envelope)
    h.admission.execute(h.admission.claim_next())
    r = row(h.service, "successor")
    assert r["error_code"] == code and r["upstream_outcome"] == "not_sent"
    assert r["status"] == ("queued" if change == "read_failure" else "failed")
    assert "send" not in chat.calls and "restore" not in chat.calls
    assert row(h.service) == h.old


def test_http_schema_preserves_normal_defaults_and_requires_reference_only(task):
    from api.ai import ConversationBindingTextRequest
    original = ConversationBindingTextRequest.model_validate(task.body).model_dump()
    original.pop("supersedes_request_id")
    assert original == task.body
    assert ConversationBindingTextRequest.model_validate(task.envelope).model_dump(exclude_unset=True) == task.envelope
    for invalid in ({"client_conversation_id": "x"}, {**task.envelope, "messages": task.body["messages"]}):
        with pytest.raises(ValidationError):
            ConversationBindingTextRequest.model_validate(invalid)


@pytest.mark.parametrize("outcome", ["success", "late_user", "late_parent", "429", "timeout"])
def test_real_paced_send_checks_after_cooldown_and_saves_only_own_result(task, bound_chat, monkeypatch, outcome):
    from services import account_request_pacing as pacing
    from services.openai_backend_api import OpenAIBackendAPI
    from services.config import config
    import services.conversation_binding_service as binding
    h, chat = task, bound_chat
    now = [1000.0]
    fake_time = SimpleNamespace(monotonic=lambda: now[0], time=lambda: now[0],
                               sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(pacing, "time", fake_time)
    monkeypatch.setattr(type(config), "account_request_interval_secs", property(lambda _: 2))
    monkeypatch.setattr(type(config), "account_message_interval_secs", property(lambda _: 60))
    pace = pacing.AccountRequestClock("fixture", h.root / "clock.json")
    pace.cooldown_until = 1010
    pace._save()
    sends = []
    backend_type = binding.OpenAIBackendAPI
    original_get = backend_type._get_conversation
    def get(backend, chat_id, *, _send=None):
        if _send is None:
            return original_get(backend, chat_id)
        return OpenAIBackendAPI._get_conversation(backend, chat_id, _send=_send)
    monkeypatch.setattr(backend_type, "_get_conversation", get)
    monkeypatch.setattr(backend_type, "_image_active_timeout", lambda self, value: value, raising=False)
    def events(backend, **kwargs):
        def raw(method, url, **kw):
            sends.append((method, now[0]))
            if method == "GET":
                assert now[0] == 1010 and kw["timeout"] == 20
                if outcome == "late_user":
                    chat.document["mapping"]["alias"] = {"message": {"id": h.old["request_message_id"]}}
                elif outcome == "late_parent":
                    chat.document["current_node"] = "foreign"
                elif outcome == "timeout":
                    raise TimeoutError("synthetic original GET timeout")
                return SimpleNamespace(status_code=429 if outcome == "429" else 200,
                    headers={"Retry-After": "120", "x-request-id": "preflight-fixture"},
                    json=lambda: deepcopy(chat.document), text="rate limit", close=lambda: None)
            assert method == "POST" and now[0] == 1012
            user = backend.text_request_message_id
            backend.text_request_parent_message_id = "previous-answer"
            backend.text_cursor_callback({"request_parent_message_id": "previous-answer", "_submission_parent_message_id": "previous-answer"})
            chat.document["mapping"][user] = {"parent": "previous-answer", "message": {"id": user, "author": {"role": "user"}}}
            chat.document["mapping"]["new-answer"] = {"parent": user, "message": {"id": "new-answer", "author": {"role": "assistant"},
                "status": "finished_successfully", "end_turn": True, "channel": "final", "content": {"content_type": "text", "parts": ["own successor answer"]}}}
            chat.document["current_node"] = "new-answer"
            return SimpleNamespace(status_code=200, headers={}, close=lambda: None, iter_lines=lambda: iter(()))
        pace.request(raw, "POST", "https://chatgpt.com/backend-api/conversation", json={"model": "fixture-text"},
                     _account_request_preflight=backend.text_pre_send_check, _account_request_deadline_monotonic=1030)
        yield {"type": "conversation.delta", "conversation_id": "original-chat", "delta": "ignored stream fragment"}
    monkeypatch.setattr(binding, "conversation_events", events)
    h.service.runner = chat.service.complete_text
    h.service.submit("owner", h.envelope)
    h.admission.execute(h.admission.claim_next())
    r = row(h.service, "successor")
    if outcome == "success":
        assert sends == [("GET", 1010), ("POST", 1012)]
        assert r["status"] == "succeeded" and r["content"] == "own successor answer"
        assert r["_submission_started"] and r["_upstream_terminal"]
        assert pace.last_turn_started == 1012
    else:
        assert sends == [("GET", 1010)]
        assert r["upstream_outcome"] == "not_sent" and not r["_submission_started"]
        assert not any(x["stage"] in {"send_guard_passed", "send_call_started"} for x in r["_execution_timeline"])
        assert r["status"] == ("queued" if outcome in {"429", "timeout"} else "failed")
        assert pace.last_turn_started is None and pace.next_turn == 0
    if outcome == "429":
        saved_clock = pacing.AccountRequestClock("fixture", h.root / "clock.json")
        assert saved_clock.cooldown_until == 1130 and saved_clock.rate_failures == 1
        assert r["rate_limit"]["phase"] == "conversation_preflight"
        assert r["rate_limit"]["retry_after_seconds"] == 120
    assert row(h.service) == h.old


def test_preflight_deadline_does_not_enter_network_or_mark_submission(task, monkeypatch):
    from services import account_request_pacing as pacing
    from services.request_context import executing
    h = task
    monkeypatch.setattr(pacing, "time", SimpleNamespace(monotonic=lambda: 1000, time=lambda: 1000))
    h.service.submit("owner", h.envelope)
    ctx = h.admission.claim_next()
    pace = pacing.AccountRequestClock()
    pace.cooldown_until = 1100
    sender, preflight = Mock(), Mock()
    with executing(ctx), pytest.raises(pacing.AccountRequestDeadlineExceeded):
        pace.request(sender, "POST", "https://chatgpt.com/backend-api/conversation",
                     _account_request_preflight=preflight, _account_request_deadline_monotonic=1010)
    sender.assert_not_called()
    preflight.assert_not_called()
    assert not row(h.service, "successor")["_submission_started"]
    assert pace.last_turn_started is None


def test_http_auth_idempotency_and_no_client_content_override(task, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock
    import api.ai as api
    app = FastAPI()
    app.include_router(api.create_router())
    monkeypatch.setattr(api, "text_task_service", task.service)
    monkeypatch.setattr(api, "require_identity", lambda token: {"id": "owner", "role": "admin" if token == "internal" else "user"})
    monkeypatch.setattr(api, "require_chat_text_policy", lambda _: None)
    review = AsyncMock()
    monkeypatch.setattr(api, "filter_or_log", review)
    with TestClient(app) as client:
        url = "/api/conversation-bindings/text"
        assert client.post(url, json=task.envelope, headers={"Authorization": "ordinary"}).status_code == 501
        assert client.post(url, json={**task.envelope, "messages": []}, headers={"Authorization": "internal"}).status_code == 422
        first = client.post(url, json=task.envelope, headers={"Authorization": "internal"})
        assert first.status_code == 200, first.text
        assert first.json()["supersedes_request_id"] == "old"
        assert client.post(url, json=task.envelope, headers={"Authorization": "internal"}).json() == first.json()
        changed = client.post(url, json={**task.envelope, "parent_message_id": "other"}, headers={"Authorization": "internal"})
        assert changed.status_code == 409 and changed.json()["detail"]["code"] == "CONVERSATION_REQUEST_CONFLICT"
        assert review.await_count == 1
        assert "original retained instructions" in review.await_args.args[1]
        assert "original retained instructions" not in first.text
        assert "_input_ref" not in first.text


def test_unsent_read_failure_resumes_same_id_after_restart(task, bound_chat):
    h, chat = task, bound_chat
    h.service.runner = chat.service.complete_text
    chat.read_error = TimeoutError("temporary")
    h.service.submit("owner", h.envelope)
    h.admission.execute(h.admission.claim_next())
    before = row(h.service, "successor")
    assert before["status"] == "queued" and before["upstream_outcome"] == "not_sent"
    h.clock.now = before["_ready_at"] + 1
    _, store, admission = build(h.root, h.clock)
    def runner(body, on_cursor):
        current_request.get().before_send()
        return {"content": "same successor result"}
    tasks = TextTaskService(store.path, runner=runner, clock=h.clock, admission=admission)
    admission.register("text", lambda ctx, body: tasks._run(ctx.owner, ctx.request_id, body))
    admission.execute(admission.claim_next())
    after = row(tasks, "successor")
    assert after["request_message_id"] == before["request_message_id"]
    assert after["status"] == "succeeded" and after["upstream_outcome"] == "completed"
    assert after["error_code"] is None and after["waiting"] is None
    assert row(tasks) == h.old
