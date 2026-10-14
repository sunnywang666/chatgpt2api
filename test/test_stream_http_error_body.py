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
from services.text_task_service import TextTaskService
from utils.helper import (
    _UPSTREAM_STREAM_ERROR_BODY_LIMIT,
    UpstreamHTTPError,
    ensure_ok,
)
from utils.upstream_error import safe_upstream_error


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
