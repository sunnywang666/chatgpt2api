"""Regression coverage for bounded streamed HTTP error diagnostics."""
import json
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from curl_cffi import requests

from services.conversation_binding_service import (
    ConversationBindingError,
    _text_http_diagnostic,
)
from services.protocol.conversation import (
    ConversationRequest,
    ImageGenerationError,
    ImageOutput,
    _generate_single_image,
    stream_text_deltas,
)
from services.text_task_service import TextTaskService
from utils.helper import (
    _UPSTREAM_STREAM_ERROR_BODY_LIMIT,
    UpstreamHTTPError,
    ensure_ok,
)
from utils.upstream_error import (
    safe_upstream_error,
    safe_upstream_response_ids,
    upstream_response_ids_from_headers,
)


class _QueuedExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        self.calls.append((function, args))

    def run(self):
        function, args = self.calls.pop(0)
        function(*args)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        pass

    def do_GET(self):
        if self.path == "/stall":
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "28")
            self.end_headers()
            self.wfile.write(b"{")
            self.wfile.flush()
            time.sleep(2)
            return
        routes = {
            "/json": (404, b'{"error":{"code":"conversation_missing","type":"invalid_request","message":"conversation no longer exists"}}', "application/json"),
            "/text": (404, b"upstream route is unavailable", "text/plain"),
            "/empty": (404, b"", "text/plain"),
            "/long": (404, b'{"detail":"' + b"A" * (_UPSTREAM_STREAM_ERROR_BODY_LIMIT * 2) + b'"}', "application/json"),
            "/sse": (200, b"data: first SSE payload\n\ndata: [DONE]\n\n", "text/event-stream"),
        }
        status, body, content_type = routes.get(self.path, (404, b"missing", "text/plain"))
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def loopback_server():
    class _DaemonServer(ThreadingHTTPServer):
        daemon_threads = True

    server = _DaemonServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _stream(url):
    return requests.get(url, stream=True, timeout=2)


def _assert_stream_closed(response):
    deadline = time.monotonic() + 3
    while not response._stream_closed and time.monotonic() < deadline:
        time.sleep(0.02)
    assert response._stream_closed is True


def test_streamed_non_2xx_reads_bounded_json_text_and_empty_bodies(loopback_server):
    cases = [
        ("/json", {"error": {"code": "conversation_missing", "type": "invalid_request", "message": "conversation no longer exists"}}),
        ("/text", "upstream route is unavailable"),
        ("/empty", ""),
    ]
    for path, expected in cases:
        response = _stream(loopback_server + path)
        with pytest.raises(UpstreamHTTPError) as raised:
            ensure_ok(response, "/backend-api/conversation")
        assert raised.value.status_code == 404
        assert raised.value.body == expected


def test_streamed_non_2xx_body_is_limited_and_signals_existing_transport_cleanup(loopback_server):
    response = _stream(loopback_server + "/long")
    with pytest.raises(UpstreamHTTPError) as raised:
        ensure_ok(response, "/backend-api/conversation")
    assert isinstance(raised.value.body, str)
    assert len(raised.value.body) <= _UPSTREAM_STREAM_ERROR_BODY_LIMIT
    _assert_stream_closed(response)


def test_streamed_error_body_read_times_out_without_blocking_on_response_close(loopback_server):
    response = _stream(loopback_server + "/stall")
    started = time.monotonic()
    with pytest.raises(UpstreamHTTPError) as raised:
        ensure_ok(response, "/backend-api/conversation")
    elapsed = time.monotonic() - started
    assert raised.value.body == "{"
    assert elapsed < 1.5
    # curl_cffi's close() waits on an unfinished stream task.  The bounded
    # reader sets its existing cancellation event instead of starting that
    # unbounded wait on the request thread.
    assert response.quit_now.is_set()
    assert response._stream_closed is False
    _assert_stream_closed(response)


def test_successful_sse_is_not_consumed_by_ensure_ok(loopback_server):
    response = _stream(loopback_server + "/sse")
    ensure_ok(response, "/backend-api/conversation")
    assert b"".join(response.iter_content()) == b"data: first SSE payload\n\ndata: [DONE]\n\n"


def test_failed_http_response_captures_only_allowlisted_response_ids():
    from requests import Response

    response = Response()
    response.status_code = 404
    response._content = b'{"detail":"Model not found"}'
    response.headers.update({
        "X-Request-Id": "req_0123456789abcdef",
        "CF-Ray": "0123456789abcdef-SJC",
        "Authorization": "Bearer PRIVATE_TOKEN",
        "Set-Cookie": "session=PRIVATE_COOKIE",
        "X-Trace": "PRIVATE_TRACE",
    })
    with pytest.raises(UpstreamHTTPError) as raised:
        ensure_ok(response, "/backend-api/conversation")
    assert raised.value.response_ids == {
        "request_id": "req_0123456789abcdef",
        "cf_ray": "0123456789abcdef-SJC",
    }
    assert "PRIVATE" not in json.dumps(raised.value.response_ids)

    no_header_response = Response()
    no_header_response.status_code = 404
    no_header_response._content = b'{"detail":"Model not found"}'
    with pytest.raises(UpstreamHTTPError) as without_ids:
        ensure_ok(no_header_response, "/backend-api/conversation")
    assert without_ids.value.response_ids == {}


def test_response_id_capture_rejects_malformed_conflicting_and_publicly_tampered_values():
    assert upstream_response_ids_from_headers({
        "x-request-id": "req_0123456789abcdef",
        "openai-request-id": "req_fedcba9876543210",
        "cf-ray": "0123456789abcdef-SJC",
    }) == {"cf_ray": "0123456789abcdef-SJC"}
    assert upstream_response_ids_from_headers({
        "x-request-id": "PRIVATE_TOKEN",
        "cf-ray": "not-a-ray",
    }) == {}
    assert upstream_response_ids_from_headers({
        "openai-request-id": "req_0123456789abcdef",
    }) == {"request_id": "req_0123456789abcdef"}
    assert safe_upstream_response_ids({
        "request_id": "req_0123456789abcdef",
        "authorization": "PRIVATE_TOKEN",
    }) == {}
    assert safe_upstream_response_ids({"request_id": "req_0123456789abcdef "}) == {}


def test_image_stream_failure_logs_safe_exception_text_and_keeps_structured_details():
    private_token = "TOKEN_PRIVATE_VALUE"
    private_prompt = "PROMPT_PRIVATE_VALUE"
    upstream = UpstreamHTTPError(
        "/backend-api/conversation?access_token=" + private_token,
        429,
        {"error": {"message": "prompt=" + private_prompt, "token": private_token}},
        retry_after=12,
    )
    logged = []

    class FakeBackend:
        image_submission_started = False

        def __init__(self, *, access_token):
            self.access_token = access_token

        def close(self):
            pass

    def stream(*_args):
        raise upstream
        yield

    with (
        mock.patch("services.protocol.conversation.account_service.get_available_access_token", return_value="fixture-token"),
        mock.patch("services.protocol.conversation.account_service.get_account", return_value={"email": "fixture@example.test"}),
        mock.patch("services.protocol.conversation.account_service.mark_image_result"),
        mock.patch("services.protocol.conversation.OpenAIBackendAPI", FakeBackend),
        mock.patch("services.protocol.conversation.stream_image_outputs", side_effect=stream),
        mock.patch("services.protocol.conversation.logger.warning", side_effect=logged.append),
    ):
        with pytest.raises(ImageGenerationError) as raised:
            _generate_single_image(ConversationRequest(model="gpt-image-2", prompt="fixture"), 1, 1)

    assert raised.value.status_code == 429
    assert raised.value.retry_after == 12
    assert upstream.body["error"]["message"] == "prompt=" + private_prompt
    assert str(upstream) == "/backend-api/conversation failed: status=429"
    assert logged[0]["event"] == "image_stream_fail"
    assert logged[0]["error"] == str(upstream)
    serialized = json.dumps(logged)
    assert private_token not in serialized
    assert private_prompt not in serialized
    assert private_token not in str(raised.value)
    assert private_prompt not in str(raised.value)


def test_text_stream_retries_only_structured_invalid_token_code():
    initial = mock.Mock(access_token="old-token")
    upstream = UpstreamHTTPError("/backend-api/conversation", 401, {
        "error": {"code": "token_invalidated", "message": "private upstream body"},
    })

    def events(backend, **_kwargs):
        if backend.access_token == "old-token":
            raise upstream
        yield {"type": "conversation.delta", "delta": "ok"}

    with (
        mock.patch("services.protocol.conversation.OpenAIBackendAPI", side_effect=lambda access_token: mock.Mock(access_token=access_token)),
        mock.patch("services.protocol.conversation.conversation_events", side_effect=events),
        mock.patch("services.protocol.conversation.account_service.refresh_access_token", return_value="new-token") as refresh,
        mock.patch("services.protocol.conversation.account_service.remove_invalid_token") as remove,
        mock.patch("services.protocol.conversation.account_service.mark_text_used"),
    ):
        assert list(stream_text_deltas(initial, ConversationRequest(model="auto", prompt="fixture"))) == ["ok"]

    refresh.assert_called_once_with("old-token", force=True, event="text_stream")
    remove.assert_not_called()


def test_text_stream_does_not_retry_upstream_message_without_invalid_token_code():
    initial = mock.Mock(access_token="old-token")
    upstream = UpstreamHTTPError("/backend-api/conversation", 401, {
        "error": {"code": "other_error", "message": "token_invalidated"},
    })

    with (
        mock.patch("services.protocol.conversation.OpenAIBackendAPI", return_value=mock.Mock(access_token="old-token")),
        mock.patch("services.protocol.conversation.conversation_events", side_effect=upstream),
        mock.patch("services.protocol.conversation.account_service.refresh_access_token") as refresh,
        mock.patch("services.protocol.conversation.account_service.remove_invalid_token") as remove,
    ):
        with pytest.raises(UpstreamHTTPError):
            list(stream_text_deltas(initial, ConversationRequest(model="auto", prompt="fixture")))

    refresh.assert_not_called()
    remove.assert_not_called()


def test_image_stream_retries_structured_invalid_token_code_before_submission():
    upstream = UpstreamHTTPError("/backend-api/conversation", 401, {
        "error": {"code": "token_revoked", "message": "private upstream body"},
    })

    class FakeBackend:
        image_submission_started = False

        def __init__(self, *, access_token):
            self.access_token = access_token

        def close(self):
            pass

    def stream(backend, *_args):
        if backend.access_token == "old-token":
            raise upstream
        yield ImageOutput(kind="result", model="gpt-image-2", index=1, total=1, data=[])

    with (
        mock.patch("services.protocol.conversation.account_service.get_available_access_token", side_effect=["old-token", "new-token"]),
        mock.patch("services.protocol.conversation.account_service.get_account", return_value={}),
        mock.patch("services.protocol.conversation.account_service.mark_image_result"),
        mock.patch("services.protocol.conversation.account_service.refresh_access_token", return_value="new-token") as refresh,
        mock.patch("services.protocol.conversation.account_service.remove_invalid_token") as remove,
        mock.patch("services.protocol.conversation.OpenAIBackendAPI", FakeBackend),
        mock.patch("services.protocol.conversation.stream_image_outputs", side_effect=stream),
    ):
        result = _generate_single_image(ConversationRequest(model="gpt-image-2", prompt="fixture"), 1, 1)

    assert len(result) == 1
    refresh.assert_called_once_with("old-token", force=True, event="image_stream")
    remove.assert_not_called()


def test_image_stream_invalid_token_code_never_retries_after_submission():
    upstream = UpstreamHTTPError("/backend-api/conversation", 401, {
        "error": {"code": "token_invalidated", "message": "private upstream body"},
    })
    backend = mock.Mock(image_submission_started=True)

    def stream(*_args):
        raise upstream
        yield

    with (
        mock.patch("services.protocol.conversation.account_service.get_available_access_token", return_value="fixture-token"),
        mock.patch("services.protocol.conversation.account_service.get_account", return_value={}),
        mock.patch("services.protocol.conversation.account_service.mark_image_result"),
        mock.patch("services.protocol.conversation.account_service.refresh_access_token") as refresh,
        mock.patch("services.protocol.conversation.account_service.remove_invalid_token") as remove,
        mock.patch("services.protocol.conversation.OpenAIBackendAPI", return_value=backend),
        mock.patch("services.protocol.conversation.stream_image_outputs", side_effect=stream),
    ):
        with pytest.raises(ImageGenerationError) as raised:
            _generate_single_image(ConversationRequest(model="gpt-image-2", prompt="fixture"), 1, 1)

    assert raised.value.upstream_submitted is True
    refresh.assert_not_called()
    remove.assert_not_called()


def test_http_detail_is_redacted_persisted_and_revalidated_at_owner_read():
    secret = "https://private.example.test/path?access_token=TOKEN_PRIVATE"
    diagnostic = _text_http_diagnostic(UpstreamHTTPError(
        "/backend-api/conversation", 404,
        {"error": {"code": "conversation_missing", "message": "request failed at " + secret}},
    ))
    error = ConversationBindingError(
        "upstream failed", original_failure_phase="stream_open", original_http_status=404,
        original_exception_category="http", original_upstream_error=diagnostic["original_upstream_error"],
    )
    assert secret not in json.dumps(error.original_upstream_error)
    receipt = {
        "request_id": "request", "status": "unknown", "original_upstream_error": {
            "code": "conversation_missing", "message": "request failed at " + secret,
        },
    }
    public = TextTaskService._public(receipt)
    assert secret not in json.dumps(public)
    assert public["original_upstream_error"]["code"] == "conversation_missing"
    assert public["original_upstream_error"]["message"] == "request failed at [redacted]"


def test_http_detail_preserves_ordinary_private_resource_code():
    diagnostic = _text_http_diagnostic(UpstreamHTTPError(
        "/backend-api/conversation", 404,
        {"error": {"code": "private_resource", "type": "invalid_request", "message": "resource is unavailable"}},
    ))
    assert diagnostic["original_upstream_error"] == {
        "code": "private_resource", "type": "invalid_request", "message": "resource is unavailable",
    }
    assert safe_upstream_error(diagnostic["original_upstream_error"])["code"] == "private_resource"


def test_http_detail_survives_runner_persistence_and_sqlite_restart(tmp_path):
    secret = "https://private.example.test/path?access_token=TOKEN_PRIVATE"
    queue = _QueuedExecutor()
    from services.conversation_binding_service import ConversationBindingService

    service = TextTaskService(
        Path(tmp_path) / "text.sqlite3", ConversationBindingService().complete_text, queue,
        recovery_reader=lambda _receipt: {"status": "running"},
    )
    body = {
        "client_request_id": "stream-http-404", "client_conversation_id": "session",
        "provider_binding_id": "binding", "provider_account_identity": "account",
        "messages": [{"role": "user", "content": "private input"}],
    }
    upstream = UpstreamHTTPError(
        "/backend-api/conversation", 404,
        {"error": {"code": "conversation_missing", "message": "request failed at " + secret}},
    )
    backend = mock.Mock()
    with (
        mock.patch("services.conversation_binding_service.account_service.get_bound_account_identity", return_value="account"),
        mock.patch("services.conversation_binding_service.account_service.get_bound_text_access_token", return_value="TOKEN"),
        mock.patch("services.conversation_binding_service.account_service.conversation_binding_lock", return_value=nullcontext()),
        mock.patch("services.conversation_binding_service.OpenAIBackendAPI", return_value=backend),
        mock.patch("services.conversation_binding_service.conversation_events", side_effect=upstream),
    ):
        assert service.submit("owner", body)["status"] == "queued"
        queue.run()

    restarted = TextTaskService(service.store.path, recovery_reader=lambda _receipt: {"status": "running"})
    with restarted.store.connect() as db:
        stored = restarted.store.read_receipt(db, "text", "owner", body["client_request_id"])
    assert stored["status"] == "unknown"
    assert stored["original_upstream_error"] == {
        "code": "conversation_missing", "message": "request failed at [redacted]",
    }
    assert secret not in restarted.store.path.read_text(errors="ignore")
    assert TextTaskService._public(stored)["original_upstream_error"] == stored["original_upstream_error"]
