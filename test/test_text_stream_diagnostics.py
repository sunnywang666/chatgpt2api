"""Persist structural stream evidence, never raw upstream error bodies."""
import json
from types import SimpleNamespace

import pytest

from services.conversation_binding_service import ConversationBindingError
from services.openai_backend_api import ChatRequirements, OpenAIBackendAPI, StreamHardTimeoutError
from services.public_chat_service import project_text_execution, safe_public_execution
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_reliable_pool_routes import runtime


class Response:
    def __init__(self, lines, error=None):
        self.lines, self.error, self.closed = lines, error, False

    def iter_lines(self):
        yield from self.lines
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


@pytest.mark.parametrize("end", ["done", "eof", "consumer_closed", "transport_error", "hard_timeout"])
def test_stream_end_persists_across_restart_with_only_safe_structural_evidence(runtime, end):
    lines = [b'event: error', b'data: {"error":"PRIVATE_TOKEN_AND_BODY"}', b'data: PRIVATE_BAD_JSON']
    if end == "done":
        lines.append(b'data: [DONE]')
    error = ConnectionError("PRIVATE_TRANSPORT") if end == "transport_error" else (
        StreamHardTimeoutError("PRIVATE_TIMEOUT") if end == "hard_timeout" else None)
    response = Response(lines, error)

    def run(body):
        current_request.get().before_send()
        iterator = OpenAIBackendAPI._iter_sse_payloads_capped(None, response, 60, observe_text=True)
        if end == "consumer_closed":
            next(iterator)
            iterator.close()
        else:
            list(iterator)
        raise ConversationBindingError("result remains unconfirmed", code="CONVERSATION_OUTCOME_UNKNOWN")

    runtime.admission.register("text", lambda ctx, body: run(body))
    runtime.service.submit("owner", {"client_request_id": "stream-original", "client_conversation_id": "stream-session", "model": "fixture-text",
                                     "messages": [{"role": "user", "content": "fixture input"}]})
    runtime.admission.execute(runtime.admission.claim_next())
    restarted = TextTaskService(runtime.store.path, admission=runtime.admission, clock=runtime.clock)
    with restarted.store.connect() as db:
        row = restarted.store.read_receipt(db, "text", "owner", "stream-original")
    stages = [s for s in row["_execution_timeline"] if s["stage"] == "stream_finished"]
    assert len(stages) == 1
    assert stages[0]["stream_end"] == end
    assert stages[0]["sse_error_event"] is True
    assert stages[0]["sse_data_count"] == (1 if end == "consumer_closed" else 3 if end == "done" else 2)
    assert stages[0]["sse_parse_errors"] == (0 if end == "consumer_closed" else 1)
    public = project_text_execution(row)
    assert public["stream_end"] == end and public["stream_ended_at"] > 0
    assert public["sse_error_event"] is True
    assert response.closed
    assert "PRIVATE" not in json.dumps(stages)
    assert "PRIVATE" not in json.dumps(public)


def test_failed_diagnostic_write_does_not_replace_original_transport_exception():
    def unavailable(*args, **kwargs):
        raise OSError("diagnostic storage unavailable")
    token = current_request.set(SimpleNamespace(record_stage=unavailable))
    try:
        response = Response([], ConnectionError("original transport failure"))
        with pytest.raises(ConnectionError, match="original transport failure"):
            list(OpenAIBackendAPI._iter_sse_payloads_capped(None, response, 60, observe_text=True))
    finally:
        current_request.reset(token)


def test_public_stream_diagnostics_reject_arbitrary_values():
    assert safe_public_execution({"stream_end": "PRIVATE", "stream_ended_at": float("inf"),
                                  "sse_data_count": True, "sse_parse_errors": -1,
                                  "sse_error_event": "PRIVATE", "sse_error_category": "PRIVATE", "error_body": "PRIVATE"}) == {}


@pytest.mark.parametrize("code,category", [
    ("rate_limit_exceeded", "rate_limit"), ("insufficient_quota", "quota"),
    ("authentication_error", "auth"), ("server_error", "upstream"),
    ("PRIVATE_UNKNOWN_CODE", "unknown"), (None, "unknown"),
])
@pytest.mark.parametrize("named_event", [False, True])
def test_sse_error_retains_only_structured_category_across_restart(runtime, code, category, named_event):
    error = {"type": code, "message": "PRIVATE_BODY rate_limit_exceeded"}
    response = Response(([b'event: error'] if named_event else []) + [
        ("data: " + json.dumps(error if named_event else {"error": error})).encode(), b'', b'data: [DONE]'])

    def run(body):
        current_request.get().before_send()
        list(OpenAIBackendAPI._iter_sse_payloads_capped(None, response, 60, observe_text=True))
        raise ConversationBindingError("unconfirmed", code="CONVERSATION_OUTCOME_UNKNOWN")

    runtime.admission.register("text", lambda ctx, body: run(body))
    runtime.service.submit("owner", {"client_request_id": "category-original", "client_conversation_id": "category-session", "model": "fixture-text",
                                    "messages": [{"role": "user", "content": "fixture"}]})
    runtime.admission.execute(runtime.admission.claim_next())
    restarted = TextTaskService(runtime.store.path, admission=runtime.admission, clock=runtime.clock)
    with restarted.store.connect() as db:
        row = restarted.store.read_receipt(db, "text", "owner", "category-original")
    stage = next(e for e in row["_execution_timeline"] if e["stage"] == "stream_finished")
    public = project_text_execution(row)
    assert stage["sse_error_category"] == public["sse_error_category"] == category
    assert public["sse_error_event"] is True
    assert "PRIVATE" not in json.dumps(stage) and "PRIVATE" not in json.dumps(public)


@pytest.mark.parametrize("end", ["done", "eof", "transport_error"])
def test_image_stream_records_its_own_end_before_result_collection(monkeypatch, end):
    monkeypatch.setattr("services.openai_backend_api.account_service.get_account", lambda token: {})
    backend = OpenAIBackendAPI(access_token="fixture-token")
    monkeypatch.setattr(backend, "_bootstrap", lambda: None)
    monkeypatch.setattr(backend, "_get_chat_requirements", lambda: ChatRequirements(token="fixture"))
    monkeypatch.setattr(backend, "_prepare_image_conversation", lambda *a, **kw: "conduit")
    lines = [b'data: {"private_image_body":"PRIVATE_IMAGE_CONTENT"}']
    if end == "done":
        lines.append(b'data: [DONE]')
    response = Response(lines, ConnectionError("PRIVATE_TRANSPORT") if end == "transport_error" else None)
    monkeypatch.setattr(backend, "_start_image_generation", lambda *a, **kw: response)
    stages = []
    token = current_request.set(SimpleNamespace(record_stage=lambda stage, **fields: stages.append((stage, fields))))
    try:
        if end == "transport_error":
            with pytest.raises(ConnectionError, match="PRIVATE_TRANSPORT"):
                list(backend._stream_picture_conversation("draw", "gpt-image-2", []))
        else:
            list(backend._stream_picture_conversation("draw", "gpt-image-2", []))
    finally:
        current_request.reset(token)
    assert len(stages) == 1
    stage, evidence = stages[0]
    assert stage == "stream_finished" and evidence["stream_end"] == end
    assert evidence["sse_parse_errors"] == 0 and evidence["sse_error_event"] is False
    assert response.closed
    assert "PRIVATE" not in json.dumps(stages)


def test_sse_terminal_observations_preserve_unknown_and_never_store_nodes_or_content(runtime):
    request_node = "PRIVATE_ORIGINAL_USER_NODE"
    message = {"id": "PRIVATE_ASSISTANT_NODE", "author": {"role": "assistant"},
               "status": "finished_successfully", "end_turn": True, "channel": "final",
               "metadata": {"parent_id": request_node},
               "content": {"parts": ["PRIVATE_TEXT https://private.test/?token=PRIVATE_TOKEN"]}}
    events = [{"v": {"message": message}}, {"message": message},
              {"message": {**message, "parent_id": "PRIVATE_DIFFERENT_PARENT"}},
              {"message": {**message, "id": ""}},
              {"message": {**message, "end_turn": "true"}},
              {"message": {**message, "status": "in_progress"}},
              {"message": {**message, "author": {"role": "tool"}}},
              {"p": "/message/end_turn", "v": True}]
    payloads = [json.dumps(event) for event in events] + ["[DONE]"]
    response = Response([("data: " + payload).encode() for payload in payloads])

    def run(body):
        current_request.get().before_send()
        assert list(OpenAIBackendAPI._iter_sse_payloads_capped(
            None, response, 60, observe_text=True, request_message_id=request_node)) == payloads
        raise ConversationBindingError("still needs original branch verification", code="CONVERSATION_OUTCOME_UNKNOWN")

    runtime.admission.register("text", lambda ctx, body: run(body))
    runtime.service.submit("owner", {"client_request_id": "terminal-evidence", "client_conversation_id": "terminal-session",
                                     "model": "fixture-text", "messages": [{"role": "user", "content": "fixture"}]})
    runtime.admission.execute(runtime.admission.claim_next())
    restarted = TextTaskService(runtime.store.path, admission=runtime.admission, clock=runtime.clock)
    with restarted.store.connect() as db:
        row = restarted.store.read_receipt(db, "text", "owner", "terminal-evidence")
    evidence = next(e for e in row["_execution_timeline"] if e["stage"] == "stream_finished")
    assert evidence["sse_message_snapshot_events"] == 7
    assert evidence["sse_terminal_assistant_events"] == 4
    assert evidence["sse_terminal_id_events"] == 3
    assert evidence["sse_terminal_final_channel_events"] == 4
    assert evidence["sse_terminal_parent_events"] == 4
    assert evidence["sse_terminal_direct_parent_match_events"] == 2
    assert row["status"] != "succeeded"
    assert sum(e["stage"] == "send_guard_passed" for e in row["_execution_timeline"]) == 1
    assert "PRIVATE" not in json.dumps(evidence)


def test_sse_done_or_patch_alone_is_not_a_terminal_message_snapshot():
    from utils.helper import iter_sse_payloads
    observation = {}
    response = Response([b'data: {"p":"/message/end_turn","v":true}', b'data: [DONE]'])
    list(iter_sse_payloads(response, observation=observation, request_message_id="original"))
    assert observation["stream_end"] == "done"
    assert observation["sse_terminal_assistant_events"] == 0
    assert observation["sse_terminal_direct_parent_match_events"] == 0


def test_image_stream_passes_original_node_to_safe_observation(monkeypatch):
    monkeypatch.setattr("services.openai_backend_api.account_service.get_account", lambda token: {})
    backend = OpenAIBackendAPI(access_token="fixture-token")
    backend.image_request_message_id = "PRIVATE_IMAGE_USER_NODE"
    monkeypatch.setattr(backend, "_bootstrap", lambda: None)
    monkeypatch.setattr(backend, "_get_chat_requirements", lambda: ChatRequirements(token="fixture"))
    monkeypatch.setattr(backend, "_prepare_image_conversation", lambda *a, **kw: "conduit")
    response = Response([b'data: {"message":{"id":"PRIVATE_ASSISTANT","author":{"role":"assistant"},"status":"finished_successfully","end_turn":true,"parent_id":"PRIVATE_IMAGE_USER_NODE"}}', b'data: [DONE]'])
    monkeypatch.setattr(backend, "_start_image_generation", lambda *a, **kw: response)
    stages = []
    token = current_request.set(SimpleNamespace(record_stage=lambda stage, **fields: stages.append((stage, fields))))
    try:
        list(backend._stream_picture_conversation("draw", "gpt-image-2", []))
    finally:
        current_request.reset(token)
    assert stages[0][1]["sse_terminal_direct_parent_match_events"] == 1
    assert "PRIVATE" not in json.dumps(stages)


@pytest.mark.parametrize("strict_final", [True, False])
def test_native_curl_silent_image_stream_does_not_hold_result_or_timeout(monkeypatch, tmp_path, strict_final):
    """Real curl, real account clock, no upstream: server never sends a tail."""
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlsplit
    from curl_cffi import requests
    from services import account_request_pacing as pacing
    from services.config import ConfigStore
    from services.protocol.conversation import ConversationRequest, stream_image_outputs
    from services.openai_backend_api import ImageStreamHardTimeoutError
    from test.test_multi_image_results import _conversation

    asset = "file_00000000" + "a" * 24
    document = _conversation([asset])
    document["mapping"]["request"]["message"]["id"] = "request"
    document["mapping"]["tool"]["message"].update(id="tool", status="finished_successfully")
    if strict_final:
        document["mapping"]["final"] = {"parent": "tool", "message": {
            "id": "final", "author": {"role": "assistant"}, "status": "finished_successfully",
            "end_turn": True, "channel": "final"}}
        document["current_node"] = "final"
    release, received = threading.Event(), []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *_args):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.append(("POST", self.path))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            payload = {"conversation_id": "conv-1", "message": document["mapping"]["tool"]["message"]}
            self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
            self.wfile.flush()
            release.wait(5)
            self.close_connection = True
        def do_GET(self):
            received.append(("GET", self.path))
            if self.path.endswith("/conv-1"):
                body = json.dumps(document).encode()
            elif self.path.endswith("/download"):
                body = json.dumps({"download_url": "https://chatgpt.com/fixture.png"}).encode()
            elif self.path == "/fixture.png":
                body = b"saved-original-image"
            elif self.path == "/slow-read":
                time.sleep(1.65)
                body = b"independent-read"
            else:
                raise AssertionError("unexpected query: " + self.path)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    session = requests.Session(trust_env=False)
    raw_send, responses = session.request, []
    def local_send(method, url, **kwargs):
        assert urlsplit(url).hostname == "chatgpt.com"
        response = raw_send(method, f"http://127.0.0.1:{server.server_port}" + urlsplit(url).path, **kwargs)
        if kwargs.get("stream"):
            responses.append(response)
        return response
    session.request = local_send
    (tmp_path / "config.json").write_text(json.dumps({"auth-key": "test-only"}))
    settings = ConfigStore(tmp_path / "config.json")
    settings.update({"account_request_interval_secs": 0, "account_message_interval_secs": 5,
                     "account_conversation_read_interval_secs": 0})
    monkeypatch.setattr(pacing, "config", settings)
    monkeypatch.setattr(pacing, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pacing, "_clocks", {})
    # Parallel curl-version runs must not share the repository image index.
    monkeypatch.setattr("services.config.DATA_DIR", tmp_path)
    monkeypatch.setattr("services.image_storage_service.config", settings)
    monkeypatch.setattr("services.image_storage_service.image_storage_service.index_file",
                        tmp_path / "image_index.json")
    pacing.pace_account_session(session, {"account_id": "fixture-only"}, "fixture-token")
    backend = object.__new__(OpenAIBackendAPI)
    backend.access_token, backend.account, backend.session = "fixture-token", {}, session
    backend.base_url, backend.image_request_message_id, backend.progress_callback = "https://chatgpt.com", "request", None
    backend._bootstrap = lambda: None
    backend._get_chat_requirements = lambda: ChatRequirements(token="fixture")
    backend._prepare_image_conversation = lambda *a, **k: "fixture"
    backend._image_headers = lambda *a: {}
    backend._headers = lambda *a: {}
    backend._image_active_timeout = lambda seconds: min(seconds, 1.5)
    monkeypatch.setattr("services.openai_backend_api.account_service.require_image_account", lambda *a: None)
    class ShortImageDeadline:
        image_poll_timeout_secs = .35
        def __getattr__(self, name):
            return getattr(settings, name)
    monkeypatch.setattr("services.openai_backend_api.config", ShortImageDeadline())
    try:
        start = time.monotonic()
        if strict_final:
            outputs = list(stream_image_outputs(backend, ConversationRequest(prompt="fixture", model="gpt-image-2")))
            assert any(output.kind == "result" and output.data for output in outputs)
            assert time.monotonic() - start < 1.2, "must return before the native stream's 1.5-second cap"
            assert ("GET", "/fixture.png") in received
        else:
            with pytest.raises(ImageStreamHardTimeoutError):
                list(stream_image_outputs(backend, ConversationRequest(prompt="fixture", model="gpt-image-2")))
            assert time.monotonic() - start < 1.2, "watchdog must wake the silent queue"
            assert ("GET", "/fixture.png") not in received
        assert not release.is_set(), "server has not sent EOF or a later body chunk"
        assert len(responses) == 1
        if strict_final:
            assert not responses[0].stream_task.done()
            read = session.get("https://chatgpt.com/slow-read", timeout=2.5)
            assert read.content == b"independent-read", "SSE's 1.5-second cap must not leak into another GET"
        responses[0].stream_task.result(timeout=2)
        deadline = time.monotonic() + 1
        while responses[0].curl._curl is not None and time.monotonic() < deadline:
            time.sleep(.005)
        assert responses[0].curl._curl is None
        assert len([x for x in received if x[0] == "POST"]) == 1
    finally:
        release.set()
        session.close()
        server.shutdown()
        server.server_close()
