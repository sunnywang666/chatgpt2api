import asyncio
import json
import threading
from datetime import datetime, timezone
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
def runtime(tmp_path, monkeypatch):
    from services import log_service
    logs = log_service.LogService(tmp_path / "calls.jsonl")
    monkeypatch.setattr(log_service, "log_service", logs)
    account = {"access_token": "fixture-only-token", "account_id": "physical-account", "provider_account_identity": "account-0",
               "source_type": "web", "type": "Plus", "status": "正常", "quota": 99, "conversation_binding_ids": ["binding-0"],
               "limits_progress": [{"feature_name": "image_gen", "remaining": 99}],
               "capacity_observed_at": datetime.now(timezone.utc).isoformat()}
    (tmp_path / "accounts.json").write_text(json.dumps([account]))
    clock = Clock()
    accounts, store, admission = build(tmp_path, clock)
    service = TextTaskService(store.path, admission=admission, clock=clock)
    admission.register("text", lambda ctx, payload: service._run(ctx.owner, ctx.request_id, payload))
    return SimpleNamespace(accounts=accounts, store=store, admission=admission, service=service, clock=clock, account=account, logs=logs)


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
    entries = runtime.logs.list(type="call")
    assert len(entries) == 1
    assert entries[0]["detail"]["request_id"] == "original-wire"
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
        response = await durable_forward.respond(who, data, request(), "openai_v1_chat_complete", service=runtime.service)
        for _ in range(100):
            if runtime.logs.list(type="call"):
                break
            await asyncio.sleep(.01)
        return response
    try:
        assert asyncio.run(disconnect()).status_code == 200
        assert len(calls) == 1
        assert len(runtime.logs.list(type="call")) == 1
        assert runtime.logs.list(type="call")[0]["detail"]["status"] == "success"
    finally:
        release.set()
        runtime.admission.stop()


def test_company_key_and_internal_http_entries_share_durable_claims(company, tmp_path, monkeypatch):
    from services.account_service import AccountService
    from services.storage.json_storage import JSONStorageBackend
    from api import ai
    row = {"access_token": "fixture-token", "account_id": "physical", "provider_account_identity": "account-0",
           "source_type": "web", "type": "Plus", "status": "正常", "quota": 99, "conversation_binding_ids": ["binding-0"],
               "limits_progress": [{"feature_name": "image_gen", "remaining": 99}],
               "capacity_observed_at": datetime.now(timezone.utc).isoformat()}
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


def test_selected_native_codex_receipt_binds_exact_account_before_final_post(runtime, monkeypatch):
    """The public ref is persisted privately; the controlled upstream never sees it."""
    from services.codex_service import CodexService
    from test.test_codex_service import FakeResponse, FakeSession, SessionFactory, observation

    account = runtime.account
    runtime.accounts.update_account(account["access_token"], {"codex_observation": observation()})
    monkeypatch.setattr(runtime.accounts, "refresh_codex_access_token", lambda token, **_kwargs: token)
    session = FakeSession(post_response=FakeResponse(payload={"id": "response-selected", "output": []}))
    native = CodexService(runtime.accounts, SessionFactory([session]))
    runtime.admission.codex = native
    monkeypatch.setattr("services.codex_service.codex_service", native)

    who = {"id": "native-user", "role": "user"}
    account_ref = runtime.accounts.pool_account_ref(account)
    payload = {"model": "gpt-5.6-codex", "input": [], "account_ref": account_ref}
    saved = durable_forward.envelope(who, payload, request("selected-codex"), "codex")
    runtime.service.submit(who["id"], saved)
    receipt = durable_forward.raw_receipt(runtime.service, who["id"], "selected-codex")
    assert receipt["_requested_account_ref"] == account_ref
    assert receipt["_requested_account_identity"] == "account-0"
    assert "account_ref" not in runtime.store.load_input(receipt["_input_ref"])["_forward"]["payload"]

    claimed = runtime.admission.claim_next()
    assert claimed is not None
    runtime.admission.execute(claimed)
    assert json.loads(session.calls[0][2]["data"]) == {"model": "gpt-5.6-codex", "input": []}
    final = durable_forward.raw_receipt(runtime.service, who["id"], "selected-codex")
    assert final["provider_account_identity"] == "account-0"
    assert len(session.calls) == 1


def test_native_codex_router_keeps_multimodal_wire_durable_and_does_not_replay_unknown(runtime, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api import codex as codex_api
    from services.codex_service import CodexService
    from test.test_codex_service import FakeResponse, FakeSession, SessionFactory, observation

    runtime.accounts.update_account(runtime.account["access_token"], {"codex_observation": observation()})
    monkeypatch.setattr(runtime.accounts, "refresh_codex_access_token", lambda token, **_kwargs: token)
    completed_wire = {
        "id": "resp-multimodal-completed",
        "object": "response",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "图片识别和文案完成"}],
        }],
    }
    completed_session = FakeSession(post_response=FakeResponse(payload=completed_wire))
    unknown_session = FakeSession(post_error=TimeoutError("simulated upstream timeout"))
    native = CodexService(runtime.accounts, SessionFactory([completed_session, unknown_session]))
    runtime.admission.codex = native
    monkeypatch.setattr("services.codex_service.codex_service", native)
    monkeypatch.setattr(codex_api, "codex_service", native)
    monkeypatch.setattr("services.text_task_service.text_task_service", runtime.service)
    identity = {"id": "native-router-user", "role": "user"}
    monkeypatch.setattr(codex_api, "require_identity", lambda _token: identity)
    monkeypatch.setattr(codex_api, "require_codex_policy", lambda _identity, _payload: None)

    app = FastAPI()
    app.include_router(codex_api.create_router())
    payload = {
        "model": "gpt-5.6-codex",
        "account_ref": runtime.accounts.pool_account_ref(runtime.account),
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "识别这张图片并写商品文案"},
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
            ],
        }],
    }
    headers = {
        "Authorization": "Bearer fixture-only",
        "X-Client-Request-ID": "native-multimodal-original",
        "Session-ID": "native-multimodal-session",
    }

    runtime.admission.start()
    try:
        with TestClient(app) as client:
            first = client.post("/codex/v1/responses", headers=headers, json=payload)
            repeated = client.post("/codex/v1/responses", headers=headers, json=payload)

            assert first.status_code == 200
            assert first.json() == completed_wire
            assert repeated.status_code == 200
            assert repeated.json() == completed_wire
            assert len(completed_session.calls) == 1
            forwarded = json.loads(completed_session.calls[0][2]["data"])
            assert forwarded["input"] == payload["input"]
            assert "account_ref" not in forwarded
            completed_receipt = durable_forward.raw_receipt(runtime.service, identity["id"], headers["X-Client-Request-ID"])
            assert completed_receipt["_requested_account_ref"] == payload["account_ref"]
            assert completed_receipt["provider_account_identity"] == "account-0"

            unknown_headers = {**headers, "X-Client-Request-ID": "native-multimodal-unknown"}
            first_unknown = client.post("/codex/v1/responses", headers=unknown_headers, json=payload)
            repeated_unknown = client.post("/codex/v1/responses", headers=unknown_headers, json=payload)

            assert first_unknown.status_code == 409
            assert repeated_unknown.status_code == 409
            assert len(unknown_session.calls) == 1
            unknown_receipt = runtime.service.read(identity["id"], "native-multimodal-unknown")
            assert unknown_receipt["status"] == "unknown"
    finally:
        runtime.admission.stop()


def test_admin_codex_fallback_reuses_rejected_gallery_input_and_projects_final_text(runtime, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api import ai
    from services.codex_service import CodexService
    from test.test_codex_service import FakeResponse, FakeSession, SessionFactory, observation

    original = {
        "client_request_id": "chat-model-not-found",
        "client_conversation_id": "gallery-session-without-physical-cursor",
        "model": "gpt-5-6-thinking",
        "product_work": {"product_ref": "product-1", "step_ref": "gallery"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "识别图片并生成商品文案"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": "high"}},
        ]}],
    }
    runtime.service.submit("admin-owner", original, source="internal:listing")
    runtime.service._update("admin-owner", original["client_request_id"], status="failed", error_code="MODEL_NOT_FOUND",
                            upstream_outcome="rejected", _submission_started=True)
    runtime.accounts.update_account(runtime.account["access_token"], {"codex_observation": observation()})
    monkeypatch.setattr(runtime.accounts, "refresh_codex_access_token", lambda token, **_kwargs: token)
    wire = {"id": "resp-fallback-final", "object": "response", "status": "completed", "output": [{
        "type": "reasoning", "encrypted_content": "opaque-reasoning", "summary": [],
    }, {
        "type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": "最终商品文案"}],
    }]}
    session = FakeSession(post_response=FakeResponse(payload=wire))
    continued_session = FakeSession(post_response=FakeResponse(payload={"id": "resp-continued", "status": "completed", "output": [{
        "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "后续文案"}],
    }]}))
    native = CodexService(runtime.accounts, SessionFactory([session, continued_session]))
    runtime.admission.codex = native
    monkeypatch.setattr("services.codex_service.codex_service", native)

    app = FastAPI()
    app.include_router(ai.create_router())
    monkeypatch.setattr(ai, "text_task_service", runtime.service)
    monkeypatch.setattr(ai, "require_identity", lambda _token: {"id": "admin-owner", "role": "admin"})
    monkeypatch.setattr(ai, "trusted_source", lambda _identity, _request: "internal:listing")
    fallback = {
        "client_request_id": "codex-fallback-1",
        "model": "gpt-5.6-codex",
        "account_ref": runtime.accounts.pool_account_ref(runtime.account),
        "reasoning_effort": "high",
    }
    with TestClient(app) as client:
        first = client.post("/api/conversation-bindings/text-requests/chat-model-not-found/codex-fallback",
                            headers={"Authorization": "admin"}, json=fallback)
        repeated = client.post("/api/conversation-bindings/text-requests/chat-model-not-found/codex-fallback",
                               headers={"Authorization": "admin"}, json=fallback)
        conflict = client.post("/api/conversation-bindings/text-requests/chat-model-not-found/codex-fallback",
                               headers={"Authorization": "admin"}, json={**fallback, "client_request_id": "codex-fallback-2"})
        parameter_conflict = client.post("/api/conversation-bindings/text-requests/chat-model-not-found/codex-fallback",
                                         headers={"Authorization": "admin"}, json={**fallback, "model": "another-native-model"})
        assert first.status_code == repeated.status_code == 200
        assert conflict.status_code == 409
        assert parameter_conflict.status_code == 409
        assert first.json()["codex_fallback"] == {
            "protocol": "codex-responses-v1",
            "predecessor_request_id": "chat-model-not-found",
            "parent_request_id": "chat-model-not-found",
            "client_conversation_id": first.json()["codex_fallback"]["client_conversation_id"],
            "model": "gpt-5.6-codex",
            "account_ref": fallback["account_ref"],
            "reasoning_effort": "high",
        }

        claimed = runtime.admission.claim_next()
        assert claimed is not None and claimed.request_id == "codex-fallback-1"
        runtime.admission.execute(claimed)
        readback = client.get("/api/conversation-bindings/text-requests/codex-fallback-1", headers={"Authorization": "admin"})
        continued = {"client_request_id": "codex-continuation-1", "step_ref": "copy", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "补充颜色信息后重写"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BB==", "detail": "low"}},
        ]}]}
        next_result = client.post("/api/conversation-bindings/text-requests/codex-fallback-1/codex-continue",
                                  headers={"Authorization": "admin"}, json=continued)
        next_repeat = client.post("/api/conversation-bindings/text-requests/codex-fallback-1/codex-continue",
                                  headers={"Authorization": "admin"}, json=continued)
        next_conflict = client.post("/api/conversation-bindings/text-requests/codex-fallback-1/codex-continue",
                                    headers={"Authorization": "admin"}, json={**continued, "client_request_id": "other-branch"})
        assert next_result.status_code == next_repeat.status_code == 200
        assert next_conflict.status_code == 409
        continued_claim = runtime.admission.claim_next()
        assert continued_claim is not None and continued_claim.request_id == "codex-continuation-1"
        runtime.admission.execute(continued_claim)
        continued_readback = client.get("/api/conversation-bindings/text-requests/codex-continuation-1", headers={"Authorization": "admin"})

    assert len(session.calls) == 1
    forwarded = json.loads(session.calls[0][2]["data"])
    assert forwarded == {
        "model": "gpt-5.6-codex",
        "reasoning": {"effort": "high"},
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "识别图片并生成商品文案"},
            {"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "high"},
        ]}],
    }
    raw = durable_forward.raw_receipt(runtime.service, "admin-owner", "codex-fallback-1")
    saved = runtime.store.load_input(raw["_input_ref"])
    assert raw["client_conversation_id"] == native._affinity_key(
        saved["_forward"]["identity"], saved["_forward"]["headers"], forwarded)
    assert readback.status_code == 200
    assert readback.json()["content"] == "最终商品文案"
    assert readback.json()["codex_fallback"] == {
        "protocol": "codex-responses-v1",
        "predecessor_request_id": "chat-model-not-found",
        "parent_request_id": "chat-model-not-found",
        "client_conversation_id": first.json()["codex_fallback"]["client_conversation_id"],
        "model": "gpt-5.6-codex",
        "account_ref": fallback["account_ref"],
        "reasoning_effort": "high",
        "response_id": "resp-fallback-final",
    }
    assert len(continued_session.calls) == 1
    continued_payload = json.loads(continued_session.calls[0][2]["data"])
    assert continued_payload["model"] == "gpt-5.6-codex"
    assert continued_payload["reasoning"] == {"effort": "high"}
    assert continued_payload["input"] == [
        {"role": "user", "content": [
            {"type": "input_text", "text": "识别图片并生成商品文案"},
            {"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "high"},
        ]},
        {"type": "reasoning", "encrypted_content": "opaque-reasoning", "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "最终商品文案"}]},
        {"role": "user", "content": [
            {"type": "input_text", "text": "补充颜色信息后重写"},
            {"type": "input_image", "image_url": "data:image/png;base64,BB==", "detail": "low"},
        ]},
    ]
    assert readback.json()["product_work"] == {"protocol": "product-work-v1", "product_ref": "product-1", "step_ref": "gallery"}
    assert continued_readback.json()["content"] == "后续文案"
    assert continued_readback.json()["product_work"] == {"protocol": "product-work-v1", "product_ref": "product-1", "step_ref": "copy"}
    assert continued_readback.json()["codex_fallback"]["predecessor_request_id"] == "chat-model-not-found"
    assert continued_readback.json()["codex_fallback"]["parent_request_id"] == "codex-fallback-1"
    assert continued_readback.json()["codex_fallback"]["response_id"] == "resp-continued"
    assert "messages" not in readback.text and "data:image" not in readback.text


@pytest.mark.parametrize("history_retained", [True, False])
def test_bound_codex_fallback_preserves_complete_vision_history_or_refuses(runtime, monkeypatch, history_retained):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api import ai
    from services.codex_service import CodexService
    from test.test_codex_service import FakeResponse, FakeSession, SessionFactory, observation

    who = "admin-owner"
    scope = {"client_conversation_id": "original-chat-client", "provider_binding_id": "binding-0",
             "provider_account_identity": "account-0", "conversation_id": "original-chat"}
    root_body = {"client_request_id": "retained-root", "client_conversation_id": scope["client_conversation_id"],
                 "model": "gpt-5-6-thinking", "messages": [{"role": "user", "content": [
                     {"type": "text", "text": "先识别原图"},
                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": "high"}},
                 ]}]}
    if history_retained:
        root_receipt = {**scope, "request_id": "retained-root", "parent_message_id": "prior-answer",
                        "status": "succeeded", "upstream_outcome": "completed", "content": "原图识别结果",
                        "_route": "chat", "_operation": "text", "_input_ref": runtime.store.save_input(root_body)}
        with runtime.store.transaction() as db:
            db.execute("INSERT INTO requests VALUES(?,?,?,?)", (who, "retained-root",
                       TextTaskService._submission_identity(who, root_body)[1], json.dumps(root_receipt)))
    original = {**scope, "client_request_id": "bound-rejected", "parent_message_id": "prior-answer",
                "model": "gpt-5-6-thinking", "messages": [{"role": "user", "content": "结合原图生成文案"}]}
    runtime.service.submit(who, original)
    runtime.service._update(who, "bound-rejected", status="failed", error_code="MODEL_NOT_FOUND",
                            upstream_outcome="rejected", _submission_started=True)
    before = durable_forward.raw_receipt(runtime.service, who, "bound-rejected")
    runtime.accounts.update_account(runtime.account["access_token"], {"codex_observation": observation()})
    monkeypatch.setattr(runtime.accounts, "refresh_codex_access_token", lambda token, **_kwargs: token)
    session = FakeSession(post_response=FakeResponse(payload={
        "id": "resp-bound", "status": "completed", "output": [{"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "最终文案"}]}],
    }))
    native = CodexService(runtime.accounts, SessionFactory([session]))
    runtime.admission.codex = native
    monkeypatch.setattr("services.codex_service.codex_service", native)
    app = FastAPI()
    app.include_router(ai.create_router())
    monkeypatch.setattr(ai, "text_task_service", runtime.service)
    monkeypatch.setattr(ai, "require_identity", lambda _token: {"id": who, "role": "admin"})
    monkeypatch.setattr(ai, "trusted_source", lambda _identity, _request: "internal:listing")
    with TestClient(app) as client:
        response = client.post("/api/conversation-bindings/text-requests/bound-rejected/codex-fallback",
                               headers={"Authorization": "admin"}, json={"client_request_id": "bound-codex",
                               "model": "gpt-5.6-codex", "account_ref": runtime.accounts.pool_account_ref(runtime.account)})
    assert durable_forward.raw_receipt(runtime.service, who, "bound-rejected") == before
    if not history_retained:
        assert response.status_code == 409
        assert "CODEX_FALLBACK_CONTEXT_UNAVAILABLE" in response.text
        assert runtime.admission.claim_next() is None
        assert session.calls == []
        return
    assert response.status_code == 200
    claimed = runtime.admission.claim_next()
    assert claimed.request_id == "bound-codex"
    runtime.admission.execute(claimed)
    assert len(session.calls) == 1
    payload = json.loads(session.calls[0][2]["data"])
    assert payload["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "先识别原图"},
            {"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "high"}]},
        {"role": "assistant", "content": [{"type": "input_text", "text": "原图识别结果"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "结合原图生成文案"}]},
    ]
    assert "conversation_id" not in payload and "parent_message_id" not in payload
    assert runtime.service.read(who, "bound-codex")["content"] == "最终文案"


def test_codex_fallback_rejects_missing_chat_history_and_never_replays_unknown(runtime, monkeypatch):
    from services.codex_service import CodexService
    from services.conversation_binding_service import ConversationBindingError
    from test.test_codex_service import FakeResponse, FakeSession, SessionFactory, observation

    original = {"client_request_id": "old", "client_conversation_id": "first-gallery", "model": "gpt-5-6-thinking",
                "conversation_id": "submitted-chat-cursor", "messages": [{"role": "user", "content": "standalone prompt"}]}
    runtime.service.submit("owner", original)
    runtime.service._update("owner", "old", status="failed", error_code="MODEL_NOT_FOUND",
                            upstream_outcome="rejected", _submission_started=True, conversation_id="runtime-chat-cursor")
    with pytest.raises(ConversationBindingError) as unavailable:
        runtime.service.submit_codex_fallback("owner", "old", "blocked", "gpt-5.6-codex",
                                              runtime.accounts.pool_account_ref(runtime.account))
    assert unavailable.value.code == "CODEX_FALLBACK_CONTEXT_UNAVAILABLE"

    original.pop("conversation_id")
    with runtime.service.store.transaction() as db:
        receipt = runtime.service.store.read_receipt(db, "text", "owner", "old")
        receipt["_input_ref"] = runtime.service.store.save_input(original)
        runtime.service.store.write_receipt(db, "text", "owner", "old", receipt)
        db.execute("UPDATE requests SET request_hash=? WHERE owner='owner' AND id='old'",
                   (TextTaskService._submission_identity("owner", original)[1],))
    from services.work_lifecycle import WorkLifecycleError
    legacy = TextTaskService(runtime.store.path, clock=runtime.clock)
    with pytest.raises(WorkLifecycleError) as no_admission:
        legacy.submit_codex_fallback("owner", "old", "no-admission", "gpt-5.6-codex",
                                     runtime.accounts.pool_account_ref(runtime.account))
    assert no_admission.value.code == "CODEX_FALLBACK_ADMISSION_REQUIRED" and no_admission.value.status == 503
    runtime.accounts.update_account(runtime.account["access_token"], {"codex_observation": observation()})
    monkeypatch.setattr(runtime.accounts, "refresh_codex_access_token", lambda token, **_kwargs: token)
    unknown_session = FakeSession(post_error=TimeoutError("simulated unknown post outcome"))
    native = CodexService(runtime.accounts, SessionFactory([unknown_session]))
    runtime.admission.codex = native
    monkeypatch.setattr("services.codex_service.codex_service", native)
    account_ref = runtime.accounts.pool_account_ref(runtime.account)
    runtime.service.submit_codex_fallback("owner", "old", "unknown-child", "gpt-5.6-codex", account_ref)
    claimed = runtime.admission.claim_next()
    assert claimed is not None and claimed.request_id == "unknown-child"
    runtime.admission.execute(claimed)
    assert runtime.service.read("owner", "unknown-child")["status"] == "unknown"
    assert runtime.service.submit_codex_fallback("owner", "old", "unknown-child", "gpt-5.6-codex", account_ref)["status"] == "unknown"
    assert len(unknown_session.calls) == 1

    from services.codex_text_fallback import build_body
    with runtime.service.store.connect() as db:
        sibling = build_body(runtime.store, db, "owner", "old", "different-child", "gpt-5.6-codex", account_ref)
    unknown_saved = runtime.store.load_input(durable_forward.raw_receipt(runtime.service, "owner", "unknown-child")["_input_ref"])
    assert sibling["client_conversation_id"] != unknown_saved["client_conversation_id"]
    assert sibling["_forward"]["headers"]["session-id"] != unknown_saved["_forward"]["headers"]["session-id"]


def test_codex_fallback_projects_only_completed_sse_response(runtime):
    from services.codex_text_fallback import project_completed_wire

    output = runtime.store.create_output()
    completed = {"id": "resp-sse-final", "status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": "完整终态"},
    ]}]}
    wire = (
        b'data: {"type":"response.output_text.delta","delta":"incomplete"}\n\n'
        + ("data: " + json.dumps({"type": "response.completed", "response": completed}) + "\n\n").encode()
        + b"data: [DONE]\n\n"
    )
    with runtime.store.output_file(output, append=True) as handle:
        handle.write(wire)
    assert project_completed_wire(runtime.service, {
        "_codex_origin_request_id": "old", "_upstream_terminal": True,
        "_wire_output": output, "_wire_size": len(wire),
    }) == {"_codex_response_id": "resp-sse-final", "content": "完整终态"}


@pytest.mark.parametrize(("wire", "saved_response_id", "code"), [
    ({"id": "resp-reasoning-only", "status": "completed", "output": [
        {"type": "reasoning", "encrypted_content": "opaque", "summary": []},
    ]}, None, "CODEX_FALLBACK_EMPTY_RESPONSE"),
    ({"id": "resp-actual", "status": "completed", "output": [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "final"}]},
    ]}, "resp-expected", "CODEX_FALLBACK_RESPONSE_INVALID"),
])
def test_codex_fallback_terminal_invalid_response_fails_without_replay_or_continuation(
        runtime, monkeypatch, wire, saved_response_id, code):
    from services.codex_service import CodexService
    from services.conversation_binding_service import ConversationBindingError
    from test.test_codex_service import FakeResponse, FakeSession, SessionFactory, observation

    original = {"client_request_id": "rejected-original", "client_conversation_id": "first-gallery",
                "model": "gpt-5-6-thinking", "messages": [{"role": "user", "content": "describe image"}]}
    runtime.service.submit("owner", original)
    runtime.service._update("owner", "rejected-original", status="failed", error_code="MODEL_NOT_FOUND",
                            upstream_outcome="rejected", _submission_started=True)
    runtime.accounts.update_account(runtime.account["access_token"], {"codex_observation": observation()})
    monkeypatch.setattr(runtime.accounts, "refresh_codex_access_token", lambda token, **_kwargs: token)
    session = FakeSession(post_response=FakeResponse(payload=wire))
    native = CodexService(runtime.accounts, SessionFactory([session]))
    runtime.admission.codex = native
    monkeypatch.setattr("services.codex_service.codex_service", native)
    account_ref = runtime.accounts.pool_account_ref(runtime.account)

    runtime.service.submit_codex_fallback("owner", "rejected-original", "invalid-child", "gpt-5.6-codex", account_ref)
    if saved_response_id is not None:
        with runtime.store.transaction() as db:
            receipt = runtime.store.read_receipt(db, "text", "owner", "invalid-child")
            receipt["upstream_response_id"] = saved_response_id
            runtime.store.write_receipt(db, "text", "owner", "invalid-child", receipt)
    claimed = runtime.admission.claim_next()
    assert claimed is not None and claimed.request_id == "invalid-child"
    runtime.admission.execute(claimed)

    failed = runtime.service.read("owner", "invalid-child")
    assert failed["status"] == "failed"
    assert failed["error_code"] == code
    assert len(session.calls) == 1
    assert runtime.admission.claim_next() is None

    if code == "CODEX_FALLBACK_EMPTY_RESPONSE":
        # A pre-fix receipt may have been persisted as succeeded without text;
        # it must remain unavailable as a continuation parent.
        runtime.service._update("owner", "invalid-child", status="succeeded", error_code=None,
                                upstream_outcome="completed", _codex_response_id=wire["id"])
    with pytest.raises(ConversationBindingError) as unavailable:
        runtime.service.submit_codex_continue("owner", "invalid-child", "blocked-child",
                                              [{"role": "user", "content": "continue"}])
    assert unavailable.value.code == "CODEX_FALLBACK_PARENT_UNAVAILABLE"
    assert runtime.admission.claim_next() is None
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
    from services.protocol.conversation import ConversationRequest, ImageOutput, stream_image_outputs_with_pool
    from services.request_context import executing, AdmissionLost
    saved = durable_forward.envelope({"id": "owner", "role": "user"}, {"model": "gpt-image-2", "n": 2}, request(), "openai_v1_image_generations", operation="image")
    runtime.service.submit("owner", saved)
    ctx = runtime.admission.claim_next()
    assert ctx is not None
    sent = []
    def generate(req, index, total):
        assert current_request.get() is ctx
        ctx.before_send()
        with pytest.raises(AdmissionLost):
            ctx.before_send()
        sent.append(index)
        return [ImageOutput(kind="result", model=req.model, index=index, total=total, data=[{"url": "https://image.example/" + str(index)}])]
    monkeypatch.setattr("services.protocol.conversation._generate_single_image", generate)
    with executing(ctx):
        assert [item.index for item in stream_image_outputs_with_pool(ConversationRequest(prompt="test", model="gpt-image-2", n=2))] == [1, 2]
    assert sent == [1, 2]
