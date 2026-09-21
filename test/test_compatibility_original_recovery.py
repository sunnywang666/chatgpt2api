"""Real compatibility formatters, synthetic upstream SSE/GET and isolated store."""
import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from services import durable_forward
from services.conversation_binding_service import ConversationBindingService
from services.openai_backend_api import OpenAIBackendAPI
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_reliable_pool_routes import runtime, request


WHO = {"id": "key-user", "role": "user"}
PROTOCOLS = sorted(durable_forward.TEXT_RECOVERY_PROTOCOLS)


def payload(protocol, stream=True):
    data = {"model": "fixture-text", "stream": stream}
    if protocol == "openai_v1_response":
        data["input"] = "original input"
    else:
        data["messages"] = [{"role": "user", "content": "original input"}]
    if protocol == "anthropic_v1_messages":
        data["max_tokens"] = 100
    return data


def original(runtime):
    return durable_forward.raw_receipt(runtime.service, WHO["id"], "original-wire")


def upstream(runtime, monkeypatch, *, done=False, transport_error=False, text="original prefix", text_chunks=None):
    sent = []
    from services.protocol import conversation, openai_v1_chat_complete, openai_v1_response, anthropic_v1_messages
    class Backend(OpenAIBackendAPI):
        def __init__(self, access_token="fixture-only-token"):
            self.access_token = access_token
            self.text_request_message_id = current_request.get().receipt()["request_message_id"]

        def stream_conversation(self, messages, model, **kwargs):
            assert self.retain_bound_conversation is True
            data = self._conversation_payload(messages, model, "America/New_York")
            # The actual cursor is durable before the one model send.
            assert original(runtime)["_submission_parent_message_id"] == data["parent_message_id"]
            current_request.get().before_send()
            sent.append(data)
            for value in text_chunks or [text]:
                yield json.dumps({"conversation_id": "original-conversation", "message": {
                    "id": "original-assistant", "author": {"role": "assistant"},
                    "status": "in_progress", "end_turn": False,
                    "content": {"content_type": "text", "parts": [value]},
                }})
            if transport_error:
                raise ConnectionError("NEVER_PERSIST_PRIVATE_TRANSPORT_DETAIL")
            if done:
                yield "[DONE]"

        def close(self):
            pass
    monkeypatch.setattr(conversation, "OpenAIBackendAPI", Backend)
    monkeypatch.setattr(conversation, "account_service", runtime.accounts)
    monkeypatch.setattr(openai_v1_chat_complete, "text_backend", lambda _: Backend())
    monkeypatch.setattr(openai_v1_response, "text_backend", lambda _: Backend())
    monkeypatch.setattr(anthropic_v1_messages, "OpenAIBackendAPI", Backend)
    monkeypatch.setattr(anthropic_v1_messages, "account_service", runtime.accounts)
    monkeypatch.setattr(openai_v1_chat_complete.chat_completion_cache, "_settings", lambda: {"enabled": False})
    return sent


def execute(runtime, protocol, stream=True):
    data = payload(protocol, stream)
    runtime.service.submit(WHO["id"], durable_forward.envelope(WHO, data, request(), protocol))
    runtime.admission.execute(runtime.admission.claim_next())
    return data


def result_document(receipt, text="original prefix and completed answer"):
    user = receipt["request_message_id"]
    return {"conversation_id": "original-conversation", "mapping": {
        user: {"parent": receipt.get("request_parent_message_id") or "parent", "message": {
            "id": user, "author": {"role": "user"}, "content": {"content_type": "text", "parts": ["original input"]}}},
        "original-assistant": {"parent": user, "message": {
            "id": "original-assistant", "author": {"role": "assistant"}, "status": "finished_successfully",
            "end_turn": True, "channel": "final", "content": {"content_type": "text", "parts": [text]}}},
    }}


def reader(receipt):
    assert receipt["provider_account_identity"] == "account-0"
    backend = Mock()
    backend._get_conversation.return_value = result_document(receipt)
    return ConversationBindingService._read_text_request_result(backend, receipt)


async def response_bytes(runtime, data, protocol):
    response = await durable_forward.respond(WHO, data, request(), protocol, service=runtime.service)
    content = response.body if hasattr(response, "body") else b"".join([part async for part in response.body_iterator])
    return response.status_code, content


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("transport_error", [False, True])
def test_restart_gets_original_full_answer_and_never_resends(runtime, monkeypatch, protocol, stream, transport_error):
    sent = upstream(runtime, monkeypatch, transport_error=transport_error)
    data = execute(runtime, protocol, stream)
    before = original(runtime)
    assert before["status"] == "unknown"
    assert before["conversation_id"] == "original-conversation"
    assert before["request_message_id"] == sent[0]["messages"][-1]["id"]
    assert sent[0]["history_and_training_disabled"] is False
    assert runtime.admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    with runtime.store.output_file(before["_wire_output"]) as handle:
        prefix = handle.read()
    assert b"[DONE]" not in prefix and b"message_stop" not in prefix and b'"stop"' not in prefix
    # Reopening the same database models process replacement; read ownership
    # still belongs to the existing recovery lease and original account.
    runtime.service = TextTaskService(runtime.store.path, admission=runtime.admission, clock=runtime.clock, recovery_reader=reader)
    runtime.admission.recoveries["text"] = runtime.service.read
    runtime.admission.recover_one()
    after = original(runtime)
    assert after["status"] == "succeeded", after
    assert after["_wire_output"] != before["_wire_output"]
    for key in ("request_id", "request_message_id", "provider_binding_id", "provider_account_identity", "client_conversation_id"):
        assert before[key] == after[key]
    assert runtime.admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    for _ in range(2):
        status, content = asyncio.run(response_bytes(runtime, data, protocol))
        assert status == 200
        assert b"original prefix and completed answer" in content
        if before.get("_wire_identity", {}).get("id"):
            assert before["_wire_identity"]["id"].encode() in content
        assert b"message_stop" in content if stream and protocol == "anthropic_v1_messages" else b"[DONE]" in content if stream else True
    with runtime.store.output_file(before["_wire_output"]) as handle:
        assert handle.read() == prefix
    assert len(sent) == len(runtime.logs.list(type="call")) == 1
    assert runtime.admission.claim_next() is None
    assert "NEVER_PERSIST" not in json.dumps(after)


def test_two_recovery_workers_claim_once_and_live_subscriber_cannot_splice_files(runtime, monkeypatch):
    sent = upstream(runtime, monkeypatch)
    protocol = "openai_v1_chat_complete"
    data = execute(runtime, protocol)
    entered, release = threading.Event(), threading.Event()
    def blocked_reader(receipt):
        entered.set()
        assert release.wait(3)
        return reader(receipt)
    recovering = Mock(side_effect=blocked_reader)
    runtime.service.recovery_reader = recovering
    second = TextTaskService(runtime.store.path, clock=runtime.clock, recovery_reader=recovering)
    async def check():
        runtime.service._update(WHO["id"], "original-wire", recovery_next_at=runtime.clock.now + 30)
        response = await durable_forward.respond(WHO, data, request(), protocol, service=runtime.service)
        runtime.clock.now += 31
        with ThreadPoolExecutor() as executor:
            work = executor.submit(runtime.service.read, WHO["id"], "original-wire")
            assert entered.wait(2)
            assert second.read(WHO["id"], "original-wire")["status"] == "unknown"
            release.set()
            assert work.result(timeout=2)["status"] == "succeeded"
        # This subscriber opened against the old prefix. A new one receives
        # the complete original response; the old one never mixes generations.
        assert b"".join([part async for part in response.body_iterator]) == b""
    asyncio.run(check())
    assert recovering.call_count == len(sent) == 1


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_original_upstream_done_keeps_normal_success(runtime, monkeypatch, protocol):
    sent = upstream(runtime, monkeypatch, done=True)
    execute(runtime, protocol)
    assert original(runtime)["status"] == "succeeded"
    assert len(sent) == 1


def test_active_original_get_keeps_unknown_and_occupancy(runtime, monkeypatch):
    sent = upstream(runtime, monkeypatch)
    execute(runtime, "openai_v1_chat_complete")
    def active(receipt):
        document = result_document(receipt)
        document["mapping"]["original-assistant"]["message"]["status"] = "in_progress"
        return ConversationBindingService._read_text_request_result(Mock(), receipt, document=document)
    runtime.service.recovery_reader = active
    assert runtime.service.read(WHO["id"], "original-wire")["status"] == "unknown"
    assert runtime.admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    assert runtime.admission.claim_next() is None
    assert len(sent) == 1


def test_overdue_recovery_rotates_before_repeating_short_backoff_scan(runtime):
    for request_id in ("first", "second", "third"):
        runtime.service.submit(WHO["id"], {"client_request_id": request_id, "client_conversation_id": request_id,
                                          "model": "fixture-text", "messages": []})
        runtime.service._update(WHO["id"], request_id, status="unknown", provider_binding_id="binding-0",
                                provider_account_identity="account-0", recovery_next_at=runtime.clock.now)
    reads = []
    def read(owner, request_id):
        reads.append(request_id)
        # A long GET plus the shortest existing backoff can make this receipt
        # due again before older rows have received any read opportunity.
        runtime.clock.now += 60
        runtime.service._update(owner, request_id, recovery_next_at=runtime.clock.now + 30)
    runtime.admission.recoveries["text"] = read
    for _ in range(3):
        runtime.admission.recover_one()
    assert reads == ["first", "second", "third"]


def test_native_partial_stream_keeps_real_response_cursor_without_claiming_a_get_route(runtime, monkeypatch):
    from test.test_codex_service import FakeResponse
    from test.test_reliable_pool_routes import native_runtime
    response = FakeResponse(chunks=[
        b'data: {"type":"response.created","response":{"id":"original-native-response","status":"in_progress"}}\n\n',
        b'data: {"type":"response.output_text.delta","response_id":"original-native-response","delta":"prefix"}\n\n',
    ], content_type="text/event-stream")
    response.headers["x-request-id"] = "original-native-upstream-request"
    native, session, who, data = native_runtime(runtime, monkeypatch, response)
    runtime.admission.execute(runtime.admission.claim_next())
    receipt = runtime.service.read(who["id"], "original-wire")
    assert receipt["status"] == "unknown"
    assert receipt["upstream_response_id"] == "original-native-response"
    assert receipt["upstream_request_id"] == "original-native-upstream-request"
    assert len(session.calls) == 1
    assert runtime.admission.claim_next() is None


def test_anthropic_recovery_preserves_tool_id_already_emitted_before_output_failure(runtime, monkeypatch):
    text = '<tool_calls><tool_call><tool_name>lookup</tool_name><parameters>{"key":"fixture"}</parameters></tool_call></tool_calls>'
    sent = upstream(runtime, monkeypatch, done=True, text=text)
    protocol = "anthropic_v1_messages"
    data = payload(protocol)
    data["tools"] = [{"name": "lookup", "input_schema": {"type": "object"}}]
    update, failed = runtime.service._update, []
    def fail_after_tool_id(owner, request_id, **changes):
        result = update(owner, request_id, **changes)
        if "_wire_size" in changes and not failed and original(runtime).get("_wire_identity", {}).get("tool_ids"):
            failed.append(True)
            raise OSError("synthetic output interruption")
        return result
    monkeypatch.setattr(runtime.service, "_update", fail_after_tool_id)
    runtime.service.submit(WHO["id"], durable_forward.envelope(WHO, data, request(), protocol))
    runtime.admission.execute(runtime.admission.claim_next())
    before = original(runtime)
    assert before["status"] == "unknown"
    tool_id = before["_wire_identity"]["tool_ids"]["0"]["id"]
    assert runtime.admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    runtime.service.recovery_reader = lambda r: ConversationBindingService._read_text_request_result(
        Mock(), r, document=result_document(r, text))
    assert runtime.service.read(WHO["id"], "original-wire")["status"] == "succeeded"
    _, wire = asyncio.run(response_bytes(runtime, data, protocol))
    assert tool_id.encode() in wire and b"message_stop" in wire
    assert len(sent) == 1


@pytest.mark.parametrize("names,emitted", [(("lookup",), 1), (("lookup", "inspect"), 2),
                                          (("lookup", "lookup"), 2), (("lookup", "inspect"), 1)])
def test_anthropic_whitespace_chunks_preserve_every_emitted_tool_identity(runtime, monkeypatch, names, emitted):
    text = "<tool_calls>" + "".join(
        '<tool_call><tool_name>' + name + '</tool_name><parameters>{"index":' + str(index) + '}</parameters></tool_call>'
        for index, name in enumerate(names)) + "</tool_calls>"
    sent = upstream(runtime, monkeypatch, done=True, text_chunks=[" \n", " \n" + text])
    protocol, data = "anthropic_v1_messages", payload("anthropic_v1_messages")
    data["tools"] = [{"name": name, "input_schema": {"type": "object"}} for name in dict.fromkeys(names)]
    update, failed = runtime.service._update, []

    def disconnect_after_tools(owner, request_id, **changes):
        result = update(owner, request_id, **changes)
        receipt = original(runtime)
        saved = receipt.get("_wire_identity", {}).get("tool_ids", {})
        if "_wire_size" in changes and len(saved) == emitted and not failed:
            with runtime.store.output_file(receipt["_wire_output"]) as handle:
                sent_prefix = handle.read()
            if all(tool["id"].encode() in sent_prefix for tool in saved.values()):
                failed.append(True)
                raise OSError("synthetic output interruption after emitted tools")
        return result

    monkeypatch.setattr(runtime.service, "_update", disconnect_after_tools)
    runtime.service.submit(WHO["id"], durable_forward.envelope(WHO, data, request(), protocol))
    runtime.admission.execute(runtime.admission.claim_next())
    before = original(runtime)
    assert before["status"] == "unknown"
    saved = before["_wire_identity"]["tool_ids"]
    # The whitespace opened text block zero before the tool markup arrived.
    assert list(saved) == [str(index + 1) for index in range(emitted)]
    with runtime.store.output_file(before["_wire_output"]) as handle:
        prefix = handle.read()
    assert all(tool["id"].encode() in prefix for tool in saved.values())
    runtime.service.recovery_reader = lambda r: ConversationBindingService._read_text_request_result(
        Mock(), r, document=result_document(r, text))
    assert runtime.service.read(WHO["id"], "original-wire")["status"] == "succeeded"
    _, wire = asyncio.run(response_bytes(runtime, data, protocol))
    events = [json.loads(line[6:]) for line in wire.decode().splitlines() if line.startswith("data: ")]
    tools = [event["content_block"] for event in events
             if event.get("type") == "content_block_start" and event["content_block"]["type"] == "tool_use"]
    assert [tool["name"] for tool in tools] == list(names)
    assert [tool["id"] for tool in tools[:emitted]] == [saved[str(i + 1)]["id"] for i in range(emitted)]
    after = original(runtime)
    assert after["_wire_output"] != before["_wire_output"]
    with runtime.store.output_file(before["_wire_output"]) as handle:
        assert handle.read() == prefix
    assert len(sent) == len(runtime.logs.list(type="call")) == 1
