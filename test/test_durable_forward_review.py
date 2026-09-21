"""PR43 independent review regressions: real rejection, logs and read limits."""
import asyncio
import importlib
import json
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from services import durable_forward
from services.request_context import current_request
from test.test_reliable_pool_routes import runtime, request, native_runtime


WHO = {"id": "key-user", "name": "fixture key", "role": "user"}
PROTOCOLS = [
    ("openai_v1_chat_complete", "/v1/chat/completions", "text"),
    ("openai_v1_response", "/v1/responses", "text"),
    ("anthropic_v1_messages", "/v1/messages", "text"),
    ("openai_search", "/v1/search", "text"),
    ("openai_v1_image_generations", "/v1/images/generations", "image"),
    ("openai_v1_image_edit", "/v1/images/edits", "image"),
]


def submit(runtime, payload, protocol="openai_v1_chat_complete", operation="text"):
    saved = durable_forward.envelope(WHO, payload, request(), protocol, operation=operation)
    runtime.service.submit(WHO["id"], saved)
    return runtime.admission.claim_next()


async def response_bytes(runtime, payload, protocol="openai_v1_chat_complete", operation="text"):
    result = await durable_forward.respond(WHO, payload, request(), protocol, operation=operation, service=runtime.service)
    if hasattr(result, "body"):
        return result.status_code, result.body
    return result.status_code, b"".join([part async for part in result.body_iterator])


@pytest.mark.parametrize("stream", [False, True])
def test_real_empty_chat_is_one_terminal_400_without_upstream_or_reclaim(runtime, monkeypatch, stream):
    from services.protocol import openai_v1_chat_complete
    upstream = Mock(side_effect=AssertionError("invalid input must not reach upstream"))
    monkeypatch.setattr(openai_v1_chat_complete, "text_backend", upstream)
    payload = {"model": "fixture-text", "messages": [], "stream": stream}
    runtime.admission.execute(submit(runtime, payload))
    for _ in range(2):
        runtime.clock.now += 100
        assert runtime.admission.claim_next() is None
        status, data = asyncio.run(response_bytes(runtime, payload))
        assert status == 400
        assert "messages or prompt is required" in json.loads(data)["error"]["message"]
    original = durable_forward.raw_receipt(runtime.service, WHO["id"], "original-wire")
    assert original["status"] == "failed"
    assert original["error_code"] == "PROTOCOL_REQUEST_REJECTED"
    assert original["_submission_started"] is False
    upstream.assert_not_called()
    assert len(runtime.logs.list(type="call")) == 1


def test_real_codex_local_busy_waits_then_executes_original_once(runtime, monkeypatch):
    from test.test_codex_service import FakeResponse
    native, session, who, payload = native_runtime(runtime, monkeypatch, FakeResponse(payload={"id": "original", "output": []}))
    native._inflight.add(runtime.account["access_token"])
    runtime.admission.execute(runtime.admission.claim_next())
    original = durable_forward.raw_receipt(runtime.service, who["id"], "original-wire")
    assert original["status"] == "queued"
    assert original["_submission_started"] is False
    assert session.calls == []
    native._inflight.clear()
    runtime.clock.now += 2
    runtime.admission.execute(runtime.admission.claim_next())
    assert runtime.service.read(who["id"], "original-wire")["status"] == "succeeded"
    assert len(session.calls) == 1


@pytest.mark.parametrize("protocol,endpoint,operation", PROTOCOLS)
@pytest.mark.parametrize("stream", [False, True])
def test_each_compatibility_executor_logs_once_after_output_and_never_on_resubscribe(runtime, monkeypatch, protocol, endpoint, operation, stream):
    result = {"type": "message_delta", "choices": [], "_account_email": "fixture@example.test", "_conversation_id": "original-conversation"}
    calls = []
    def handler(payload):
        current_request.get().before_send()
        calls.append(True)
        return iter([result]) if stream else result
    monkeypatch.setattr(importlib.import_module("services.protocol." + protocol), "handle", handler)
    payload = {"model": "gpt-image-2" if operation == "image" else "fixture-text", "prompt": "fixture", "stream": stream}
    runtime.admission.execute(submit(runtime, payload, protocol, operation))
    for _ in range(2):
        status, data = asyncio.run(response_bytes(runtime, payload, protocol, operation))
        assert status == 200
        assert b"fixture@example.test" not in data
    entries = runtime.logs.list(type="call")
    assert len(entries) == len(calls) == 1
    detail = entries[0]["detail"]
    assert detail["status"] == "success"
    assert detail["endpoint"] == endpoint
    assert detail["key_id"] == "key-user"
    assert detail["request_id"] == "original-wire"
    assert detail["provider_account_identity"] == "account-0"
    assert detail["account_email"] == "fixture@example.test"
    assert detail["conversation_id"] == "original-conversation"
    assert "NEVER_PERSIST_SECRET" not in json.dumps(entries)
    assert "流式调用结束" in entries[0]["summary"] if stream else "调用完成" in entries[0]["summary"]


@pytest.mark.parametrize("protocol,endpoint,operation", PROTOCOLS)
def test_each_compatibility_protocol_rejection_keeps_400_and_one_failure_log(runtime, monkeypatch, protocol, endpoint, operation):
    def reject(payload):
        raise HTTPException(400, detail={"error": "invalid fixture"})
    monkeypatch.setattr(importlib.import_module("services.protocol." + protocol), "handle", reject)
    payload = {"model": "gpt-image-2" if operation == "image" else "fixture-text"}
    runtime.admission.execute(submit(runtime, payload, protocol, operation))
    for _ in range(2):
        status, data = asyncio.run(response_bytes(runtime, payload, protocol, operation))
        assert status == 400
        assert json.loads(data)["error"]["message"] == "invalid fixture"
        assert runtime.admission.claim_next() is None
    entries = runtime.logs.list(type="call")
    assert len(entries) == 1
    assert entries[0]["detail"]["status"] == "failed"
    assert entries[0]["detail"]["endpoint"] == endpoint


def test_first_stream_event_rejection_keeps_http_400(runtime, monkeypatch):
    def reject(payload):
        raise HTTPException(400, detail="stream validation failed")
        yield  # A real generator whose failure happens on first next().
    monkeypatch.setattr(importlib.import_module("services.protocol.openai_v1_chat_complete"), "handle", reject)
    payload = {"model": "fixture-text", "stream": True}
    runtime.admission.execute(submit(runtime, payload))
    status, data = asyncio.run(response_bytes(runtime, payload))
    assert status == 400
    assert json.loads(data)["error"]["message"] == "stream validation failed"
    assert runtime.admission.claim_next() is None


def test_failed_stream_keeps_prefix_and_logs_once_without_claiming_upstream_recovery(runtime, monkeypatch):
    calls, closed = [], []
    def incomplete(payload):
        try:
            current_request.get().before_send()
            calls.append(True)
            yield {"choices": [{"delta": {"content": "original prefix"}}], "_account_email": "fixture@example.test"}
            raise TimeoutError("NEVER_LOG_UPSTREAM_SECRET")
        finally:
            closed.append(True)
    monkeypatch.setattr(importlib.import_module("services.protocol.openai_v1_chat_complete"), "handle", incomplete)
    reader = Mock(side_effect=AssertionError("compatibility receipt has no recoverable upstream cursor"))
    runtime.service.recovery_reader = reader
    payload = {"model": "fixture-text", "stream": True}
    runtime.admission.execute(submit(runtime, payload))
    for _ in range(2):
        original = runtime.service.read(WHO["id"], "original-wire", allow_unrecoverable_retry=True)
        assert original["status"] == "unknown"
        runtime.admission.recover_one()
        status, data = asyncio.run(response_bytes(runtime, payload))
        assert status == 200
        assert b"original prefix" in data and b"[DONE]" not in data
    reader.assert_not_called()
    assert len(calls) == len(closed) == 1
    assert runtime.admission.claim_next() is None
    logs = runtime.logs.list(type="call")
    assert len(logs) == 1 and logs[0]["detail"]["status"] == "failed"
    assert "NEVER_LOG_UPSTREAM_SECRET" not in json.dumps(logs)
