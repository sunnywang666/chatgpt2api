"""Search protocol recovery through real formatters and original-branch reads."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from services import durable_forward
from services.conversation_binding_service import ConversationBindingService
from services.openai_backend_api import OpenAIBackendAPI, SEARCH_MODEL
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_reliable_pool_routes import runtime, request


WHO = {"id": "search-user", "role": "user"}
CASES = [("openai_search", False), ("openai_v1_chat_complete", False), ("openai_v1_chat_complete", True),
         ("openai_v1_response", False), ("openai_v1_response", True)]


def original(runtime):
    return durable_forward.raw_receipt(runtime.service, WHO["id"], "original-wire")


def document(receipt, status="finished_successfully"):
    user = receipt["request_message_id"]
    def message(mid, role, text, **extra):
        return {"id": mid, "author": {"role": role}, "content": {"content_type": "text", "parts": [text]}, **extra}
    return {"conversation_id": "original-search", "current_node": "later-answer", "mapping": {
        user: {"parent": "resolved-server-root", "message": message(user, "user", "original query")},
        "original-answer": {"parent": user, "message": message("original-answer", "assistant", "original search answer",
            status=status, end_turn=status == "finished_successfully", channel="final", create_time=10,
            metadata={"citations": [{"title": "Original source", "url": "https://original.example/source"}]})},
        "later-user": {"parent": "original-answer", "message": message("later-user", "user", "different query")},
        "later-answer": {"parent": "later-user", "message": message("later-answer", "assistant", "WRONG LATEST ANSWER",
            status="finished_successfully", end_turn=True, channel="final", create_time=20,
            metadata={"citations": [{"url": "https://wrong.example/source"}]})},
    }}


def upstream(runtime, monkeypatch, *, done=False, transport_error=False, with_cursor=True):
    from services.protocol import openai_search, web_search_tool, openai_v1_chat_complete
    from services import conversation_binding_service
    observed = SimpleNamespace(prepares=[], sends=[], reads=[], lists=[], closes=0)
    class Response:
        status_code = 200
        headers = {}
        def json(self):
            return {"conduit_token": "synthetic-conduit"}
        def iter_lines(self):
            if with_cursor:
                yield b'data: {"conversation_id":"original-search"}'
            if transport_error:
                raise ConnectionError("PRIVATE_TRANSPORT_DETAIL")
            if done:
                yield b'data: [DONE]'
        def close(self):
            observed.closes += 1
    class Session:
        def post(self, url, **kwargs):
            if url.endswith("/prepare"):
                observed.prepares.append(kwargs["json"])
            else:
                assert original(runtime)["_chat_recovery"]["kind"] == "search"
                current_request.get().before_send()
                observed.sends.append(kwargs["json"])
            return Response()
    class Backend(OpenAIBackendAPI):
        def __init__(self, access_token="fixture-only-token"):
            self.access_token = access_token
            self.base_url = "https://synthetic.invalid"
            self.text_request_message_id = original(runtime)["request_message_id"]
            self.session = Session()
        def _headers(self, *args):
            return {}
        def _image_headers(self, *args):
            return {}
        def _bootstrap(self):
            pass
        def _get_chat_requirements(self):
            return {}
        def _get_conversation(self, conversation_id, **kwargs):
            assert conversation_id == "original-search"
            observed.reads.append(conversation_id)
            return document(original(runtime))
        _get_search_conversation = _get_conversation
        def _list_recent_conversations(self, **kwargs):
            observed.lists.append(kwargs)
            return [{"id": "original-search", "update_time": runtime.clock.now}]
        def close(self):
            pass
    for module in (openai_search, web_search_tool):
        monkeypatch.setattr(module, "OpenAIBackendAPI", Backend)
        monkeypatch.setattr(module, "account_service", runtime.accounts)
    monkeypatch.setattr(openai_v1_chat_complete.chat_completion_cache, "_settings", lambda: {"enabled": False})
    monkeypatch.setattr(conversation_binding_service, "OpenAIBackendAPI", Backend)
    monkeypatch.setattr(conversation_binding_service, "account_service", runtime.accounts)
    def read(receipt):
        assert receipt["provider_account_identity"] == "account-0"
        assert receipt["provider_binding_id"] == "binding-0"
        return ConversationBindingService().read_text_request(receipt)
    observed.reader = read
    return observed


def payload(protocol, stream):
    if protocol == "openai_search":
        data = {"prompt": "original query"}
    else:
        data = {"model": "fixture-text", "stream": stream, "tools": [{"type": "web_search"}]}
        if protocol == "openai_v1_response":
            data["input"] = "original query"
        else:
            data["messages"] = [{"role": "user", "content": "original query"}]
    return data


def execute(runtime, protocol, stream):
    data = payload(protocol, stream)
    runtime.service.submit(WHO["id"], durable_forward.envelope(WHO, data, request(), protocol))
    runtime.admission.execute(runtime.admission.claim_next())
    return data


async def response_bytes(runtime, data, protocol):
    response = await durable_forward.respond(WHO, data, request(), protocol, service=runtime.service)
    wire = response.body if hasattr(response, "body") else b"".join([part async for part in response.body_iterator])
    return response.status_code, wire


@pytest.mark.parametrize("protocol,stream", CASES)
@pytest.mark.parametrize("transport_error", [False, True])
def test_search_restart_recovers_original_answer_sources_and_ids(runtime, monkeypatch, protocol, stream, transport_error):
    observed = upstream(runtime, monkeypatch, transport_error=transport_error)
    data = execute(runtime, protocol, stream)
    before = original(runtime)
    assert before["status"] == "unknown"
    assert before["model"] == SEARCH_MODEL
    assert before["conversation_id"] == "original-search"
    assert before["request_message_id"] == observed.prepares[0]["partial_query"]["id"] == observed.sends[0]["messages"][0]["id"]
    assert runtime.admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    with runtime.store.output_file(before["_wire_output"]) as handle:
        prefix = handle.read()
    if protocol == "openai_v1_response" and stream:
        assert before["_wire_identity"]["search_id"].encode() in prefix
        assert b"response.completed" not in prefix
    runtime.service = TextTaskService(runtime.store.path, admission=runtime.admission, clock=runtime.clock, recovery_reader=observed.reader)
    runtime.admission.recoveries["text"] = runtime.service.read
    runtime.admission.recover_one()
    after = original(runtime)
    assert after["status"] == "succeeded", after
    assert after["_wire_output"] != before["_wire_output"]
    for key in ("request_id", "request_message_id", "provider_binding_id", "provider_account_identity", "client_conversation_id"):
        assert after[key] == before[key]
    for _ in range(2):
        status, wire = asyncio.run(response_bytes(runtime, data, protocol))
        assert status == 200
        assert b"original search answer" in wire and b"https://original.example/source" in wire
        assert b"WRONG LATEST" not in wire and b"wrong.example" not in wire
        for key in ("id", "search_id", "item_id"):
            if before.get("_wire_identity", {}).get(key):
                assert before["_wire_identity"][key].encode() in wire
    with runtime.store.output_file(before["_wire_output"]) as handle:
        assert handle.read() == prefix
    assert len(observed.prepares) == len(observed.sends) == len(observed.reads) == len(runtime.logs.list(type="call")) == 1
    assert runtime.admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert "PRIVATE_TRANSPORT_DETAIL" not in json.dumps(after)


def test_search_without_stream_cursor_uses_bounded_original_account_lookup(runtime, monkeypatch):
    observed = upstream(runtime, monkeypatch, with_cursor=False)
    data = execute(runtime, "openai_search", False)
    assert original(runtime)["status"] == "unknown"
    assert not original(runtime).get("conversation_id")
    runtime.service.recovery_reader = observed.reader
    assert runtime.service.read(WHO["id"], "original-wire")["status"] == "unknown"
    assert original(runtime)["_recovery_conversation_scan"]["matches"][0]["conversation_id"] == "original-search"
    runtime.clock.now = original(runtime)["recovery_next_at"] + 1
    assert runtime.service.read(WHO["id"], "original-wire")["status"] == "succeeded"
    status, wire = asyncio.run(response_bytes(runtime, data, "openai_search"))
    assert status == 200 and b"original search answer" in wire
    assert len(observed.sends) == len(observed.lists) == 1
    assert len(observed.reads) == 2


@pytest.mark.parametrize("protocol,stream", CASES)
def test_search_waits_for_actual_upstream_model_quota(runtime, protocol, stream):
    runtime.accounts.update_account(runtime.account["access_token"], {
        "limits_progress": [{"feature_name": SEARCH_MODEL, "remaining": 0}],
    })
    runtime.service.submit(WHO["id"], durable_forward.envelope(WHO, payload(protocol, stream), request(), protocol))
    assert runtime.admission.claim_next() is None
    assert original(runtime)["status"] == "queued"
    runtime.accounts.update_account(runtime.account["access_token"], {"limits_progress": []})
    assert runtime.admission.claim_next() is not None


def test_search_recovery_preserves_ids_after_answer_download_before_output_finished(runtime, monkeypatch):
    observed = upstream(runtime, monkeypatch, done=True)
    update, interrupted = runtime.service._update, []
    def fail_after_output(owner, request_id, **changes):
        result = update(owner, request_id, **changes)
        receipt = original(runtime)
        if "_wire_size" in changes and not interrupted and receipt.get("_wire_identity", {}).get("item_id"):
            with runtime.store.output_file(receipt["_wire_output"]) as handle:
                if receipt["_wire_identity"]["item_id"].encode() in handle.read():
                    interrupted.append(True)
                    raise OSError("synthetic output interruption")
        return result
    monkeypatch.setattr(runtime.service, "_update", fail_after_output)
    data = execute(runtime, "openai_v1_response", True)
    before = original(runtime)
    assert before["status"] == "unknown" and interrupted
    runtime.service.recovery_reader = observed.reader
    assert runtime.service.read(WHO["id"], "original-wire")["status"] == "succeeded"
    _, wire = asyncio.run(response_bytes(runtime, data, "openai_v1_response"))
    for key in ("id", "search_id", "item_id"):
        assert before["_wire_identity"][key].encode() in wire
    assert b"original search answer" in wire
    assert len(observed.sends) == 1 and len(observed.reads) == 2


@pytest.mark.parametrize("protocol,stream", CASES)
def test_search_normal_success_also_uses_original_branch(runtime, monkeypatch, protocol, stream):
    observed = upstream(runtime, monkeypatch, done=True)
    data = execute(runtime, protocol, stream)
    assert original(runtime)["status"] == "succeeded"
    status, wire = asyncio.run(response_bytes(runtime, data, protocol))
    assert status == 200 and b"original search answer" in wire and b"WRONG LATEST" not in wire
    assert len(observed.sends) == len(observed.reads) == 1


@pytest.mark.parametrize("status", ["in_progress", "finished_partial_completion"])
def test_stable_or_timed_out_search_fragments_never_complete(monkeypatch, status):
    from services import openai_backend_api
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr(openai_backend_api.time, "time", lambda: clock.now)
    monkeypatch.setattr(openai_backend_api.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
    backend.search_request_message_id = "original-user"
    reads = []
    def get(_):
        reads.append(clock.now)
        return document({"request_message_id": "original-user"}, status)
    monkeypatch.setattr(backend, "_get_search_conversation", get)
    with pytest.raises(RuntimeError, match="timed out"):
        backend._wait_search_result("original-search", 4, 1)
    assert reads == [0, 1, 2, 3]
