import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from services.pool_admission import PoolAdmission
from services.request_context import current_request
from services.text_task_service import TextTaskService
from services import durable_forward
from test.test_pool_admission import build, Clock
from test.test_company_requests import company, body, PREFIX


@pytest.fixture
def runtime(tmp_path):
    account = {"access_token": "fixture-only-token", "account_id": "physical-account", "provider_account_identity": "account-0",
               "source_type": "web", "type": "Plus", "status": "正常", "quota": 99, "conversation_binding_ids": ["binding-0"]}
    (tmp_path / "accounts.json").write_text(json.dumps([account]))
    clock = Clock()
    accounts, store, admission = build(tmp_path, clock)
    service = TextTaskService(store.path, admission=admission, clock=clock)
    admission.register("text", lambda ctx, payload: service._run(ctx.owner, ctx.request_id, payload))
    return SimpleNamespace(accounts=accounts, store=store, admission=admission, service=service, clock=clock, account=account)


def request(request_id="original-wire"):
    return SimpleNamespace(headers={"x-client-request-id": request_id, "session-id": "original-session", "authorization": "NEVER_PERSIST_SECRET"}, state=SimpleNamespace())


def test_wire_original_input_restart_and_repeat_returns_one_generation(runtime, monkeypatch):
    from services.protocol import openai_v1_chat_complete
    calls = []
    def handle(payload):
        current_request.get().before_send()
        calls.append(payload)
        return {"choices": [{"message": {"content": "original answer"}}], "_account_email": "PRIVATE"}
    monkeypatch.setattr(openai_v1_chat_complete, "handle", handle)
    who, data = {"id": "key-user", "role": "user"}, {"model": "fixture-text", "messages": [{"role": "user", "content": "original input"}]}
    envelope = durable_forward.envelope(who, data, request(), "openai_v1_chat_complete")
    runtime.service.submit("key-user", envelope)
    restarted = TextTaskService(runtime.store.path, admission=runtime.admission)
    runtime.admission.register("text", lambda ctx, p: restarted._run(ctx.owner, ctx.request_id, p))
    runtime.admission.execute(runtime.admission.claim_next())
    for _ in range(2):
        result = asyncio.run(durable_forward.respond(who, data, request(), "openai_v1_chat_complete", service=restarted))
        assert result.status_code == 200
        assert b"original answer" in result.body
        assert b"PRIVATE" not in result.body
    assert len(calls) == 1
    receipt = durable_forward.raw_receipt(restarted, "key-user", "original-wire")
    saved = runtime.store.load_input(receipt["_input_ref"])
    assert "NEVER_PERSIST_SECRET" not in json.dumps(saved)
    assert not any(k.startswith("_") for k in restarted.read("key-user", "original-wire"))


def test_wire_consumer_disconnect_does_not_cancel_execution(runtime, monkeypatch):
    from services.protocol import openai_v1_chat_complete
    entered, release = threading.Event(), threading.Event()
    calls = []
    def handle(payload):
        current_request.get().before_send()
        calls.append(payload)
        entered.set()
        release.wait(3)
        return {"choices": []}
    monkeypatch.setattr(openai_v1_chat_complete, "handle", handle)
    who, data = {"id": "key-user", "role": "user"}, {"model": "fixture-text", "stream": False}
    runtime.admission.start()
    async def disconnect():
        subscriber = asyncio.create_task(durable_forward.respond(who, data, request(), "openai_v1_chat_complete", service=runtime.service))
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(.01)
        assert entered.is_set()
        subscriber.cancel()
        with pytest.raises(asyncio.CancelledError):
            await subscriber
        assert runtime.service.read("key-user", "original-wire")["status"] == "running"
        release.set()
        return await durable_forward.respond(who, data, request(), "openai_v1_chat_complete", service=runtime.service)
    try:
        assert asyncio.run(disconnect()).status_code == 200
        assert len(calls) == 1
    finally:
        release.set()
        runtime.admission.stop()


def test_company_key_and_internal_http_entries_share_durable_claims(company, tmp_path, monkeypatch):
    from services.account_service import AccountService
    from services.storage.json_storage import JSONStorageBackend
    from api import ai
    row = {"access_token": "fixture-token", "account_id": "physical", "provider_account_identity": "account-0",
           "source_type": "web", "type": "Plus", "status": "正常", "quota": 99, "conversation_binding_ids": ["binding-0"]}
    path = tmp_path / "pool-only.json"
    path.write_text(json.dumps([row]))
    accounts = AccountService(JSONStorageBackend(path))
    admission = PoolAdmission(company.text_tasks.store, accounts, settings=lambda: {"image_account_concurrency": 4, "codex_max_concurrency": 4},
                              model_types=lambda _: {"Plus"}, pacing=lambda a, now: {"next_at": now})
    company.text_tasks.admission = admission
    company.text_tasks.runner = lambda payload, on_cursor: current_request.get().before_send() or {"content": "done"}
    monkeypatch.setattr(ai, "text_task_service", company.text_tasks)
    admission.register("text", lambda ctx, p: company.text_tasks._run(ctx.owner, ctx.request_id, p))
    assert company.client.post(PREFIX + "/api/chat-requests", headers=company.headers(), json=body()).status_code == 202
    first = admission.claim_next()
    ordinary = {"Authorization": "Bearer " + company.ordinary, "X-Workbench-Consumer": "happy"}
    assert company.client.post("/api/chat-requests", headers=ordinary, json={**body(), "client_request_id": "ordinary"}).status_code == 202
    for lane in ("wb-to-ozon", "ozon-to-wb"):
        result = company.client.post("/api/conversation-bindings/text", headers={"Authorization": "Bearer " + company.admin, "X-Workbench-Consumer": lane},
                                     json={"client_request_id": lane, "client_conversation_id": lane, "model": "gpt-text", "messages": body()["messages"]})
        assert result.status_code == 200
    assert admission.claim_next() is None
    snapshot = admission.resource_snapshot()
    assert snapshot["queue"]["queued"] == 3
    assert set(snapshot["queue"]["by_source"]) == {"key:" + company.old_key["id"], "internal:wb-to-ozon", "internal:ozon-to-wb"}
    admission.execute(first)
    for _ in range(3):
        claimed = admission.claim_next()
        assert claimed is not None
        admission.execute(claimed)
    assert admission.resource_snapshot()["queue"]["queued"] == 0


def native_runtime(runtime, monkeypatch, response):
    from services.codex_service import CodexService
    from test.test_codex_service import observation, FakeSession, SessionFactory
    account = runtime.account
    runtime.accounts.update_account(account["access_token"], {"codex_observation": observation()})
    monkeypatch.setattr(runtime.accounts, "refresh_codex_access_token", lambda token, **kw: token)
    session = FakeSession(post_response=response)
    native = CodexService(runtime.accounts, SessionFactory([session]))
    runtime.admission.codex = native
    monkeypatch.setattr("services.codex_service.codex_service", native)
    who = {"id": "native-user", "role": "user"}
    payload = {"model": "gpt-5.6-codex", "input": [], "reasoning": {"effort": "high"}}
    saved = durable_forward.envelope(who, payload, request(), "codex")
    runtime.service.submit(who["id"], saved)
    return native, session, who, payload


def test_native_codex_uses_original_durable_claim_and_exact_payload(runtime, monkeypatch):
    from test.test_codex_service import FakeResponse
    native, session, who, payload = native_runtime(runtime, monkeypatch, FakeResponse(payload={"id": "response-original", "output": []}))
    claimed = runtime.admission.claim_next()
    assert claimed is not None
    assert runtime.admission.resource_snapshot()["codex"]["inflight"] == 1
    runtime.admission.execute(claimed)
    original = runtime.service.read(who["id"], "original-wire")
    assert original["status"] == "succeeded"
    assert json.loads(session.calls[0][2]["data"]) == payload
    assert original["provider_account_identity"] == "account-0"
    assert runtime.admission.resource_snapshot()["codex"]["inflight"] == 0
    result = asyncio.run(durable_forward.respond(who, payload, request(), "codex", service=runtime.service))
    assert result.status_code == 200
    assert len(session.calls) == 1


def test_native_upstream_sse_limit_keeps_layer_and_cooldown_evidence(runtime, monkeypatch):
    from test.test_codex_service import FakeResponse
    response = FakeResponse(chunks=[b'data: {"type":"response.failed","response":{"id":"response-limited","error":{"code":"rate_limit_exceeded","retry_after_seconds":400}}}\n\n'], content_type="text/event-stream")
    response.headers["x-request-id"] = "upstream-original-429"
    native, session, who, payload = native_runtime(runtime, monkeypatch, response)
    runtime.admission.execute(runtime.admission.claim_next())
    receipt = runtime.service.read(who["id"], "original-wire")
    limit = receipt["rate_limit"]
    assert limit["layer"] == "upstream_codex"
    assert limit["origin"] == "sse_rate_limit"
    assert limit["upstream_request_id"] == "upstream-original-429"
    assert limit["retry_after_seconds"] == 400
    assert limit["cooldown_until"] > limit["observed_at"] + 399
    assert runtime.admission.resource_snapshot()["accounts"][0]["codex"]["capacity"] == 0
    assert len(session.calls) == 1


def test_duplicate_send_inside_original_runner_is_fenced(runtime):
    from services.request_context import AdmissionLost
    payload = {"client_request_id": "single", "client_conversation_id": "same", "model": "fixture-text", "messages": []}
    runtime.service.submit("owner", payload)
    ctx = runtime.admission.claim_next()
    ctx.before_send()
    with pytest.raises(AdmissionLost):
        ctx.before_send()


def test_legacy_multi_image_slots_cannot_escape_claim_or_redraw_a_slot(runtime, monkeypatch):
    from services.protocol.conversation import ConversationRequest, stream_image_outputs_with_pool
    from services.request_context import executing, AdmissionLost
    saved = durable_forward.envelope({"id": "owner", "role": "user"}, {"model": "gpt-image-2", "n": 2}, request(), "openai_v1_image_generations", operation="image")
    runtime.service.submit("owner", saved)
    ctx = runtime.admission.claim_next()
    sent = []
    def generate(req, index, total):
        assert current_request.get() is ctx
        ctx.before_send()
        with pytest.raises(AdmissionLost):
            ctx.before_send()
        sent.append(index)
        return [index]
    monkeypatch.setattr("services.protocol.conversation._generate_single_image", generate)
    with executing(ctx):
        assert list(stream_image_outputs_with_pool(ConversationRequest(prompt="test", model="gpt-image-2", n=2))) == [1, 2]
    assert sent == [1, 2]
