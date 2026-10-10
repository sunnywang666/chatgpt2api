from __future__ import annotations

import base64
import hashlib
import json
import unittest
import threading
import tempfile
import time
from pathlib import Path
from contextlib import nullcontext
from unittest import mock
from types import SimpleNamespace

from services import account_request_pacing as pacing

from services.openai_backend_api import ChatRequirements, ConversationArchiveCursorMismatch, OpenAIBackendAPI, StreamHardTimeoutError
from services.conversation_binding_service import (
    ConversationBindingError,
    ConversationBindingService,
    RECOVERY_CONVERSATION_SCAN_FIELD,
    TextRecoveryReason,
)
from utils.helper import UpstreamHTTPError
from services.protocol.conversation import (
    ConversationRequest,
    ImageGenerationError,
    ImageOutput,
    _generate_bound_single_image,
)


class AccountRequestPacingTests(unittest.TestCase):
    def setUp(self):
        pacing._clocks.clear()
        self.real_monotonic = time.monotonic
        self.now = 1000.0
        self.calls = []
        self.addCleanup(mock.patch.stopall)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        mock.patch.object(pacing, "DATA_DIR", Path(directory.name)).start()
        mock.patch.object(pacing.time, "monotonic", side_effect=lambda: self.now).start()
        mock.patch.object(pacing.time, "time", side_effect=lambda: 1000000 + self.now).start()
        mock.patch.object(pacing.time, "sleep", side_effect=self.advance).start()
        mock.patch.object(pacing, "config", SimpleNamespace(
            account_request_interval_secs=5.0, account_message_interval_secs=30.0)).start()

    def advance(self, delay):
        self.now += delay

    def session(self, account_id="account-a", token="token", statuses=None):
        replies = iter(statuses or [(200, {})] * 10)

        def send(method, url, **kwargs):
            self.calls.append((account_id, self.now, method, url))
            code, headers = next(replies)
            return SimpleNamespace(status_code=code, headers=headers)

        session = SimpleNamespace(request=send)
        pacing.pace_account_session(session, {"account_id": account_id}, token)
        return session

    def test_text_image_and_poll_share_one_clock_across_clients_and_refreshed_tokens(self):
        first = self.session(token="old-token")
        second = self.session(token="new-token")
        first.request("POST", "https://chatgpt.com/backend-api/conversation")
        second.request("GET", "https://chatgpt.com/backend-api/tasks")
        second.request("POST", "https://chatgpt.com/backend-api/f/conversation")
        first.request("GET", "https://chatgpt.com/backend-api/conversation/original")
        self.assertEqual([row[1] for row in self.calls], [1000, 1005, 1030, 1035])

    def test_429_cools_down_other_clients_honors_retry_after_and_never_replays(self):
        first = self.session(statuses=[(429, {"Retry-After": "120"})])
        second = self.session()
        response = first.request("POST", "https://chatgpt.com/backend-api/conversation")
        self.assertEqual(response.status_code, 429)
        self.assertEqual(len(self.calls), 1)
        second.request("GET", "https://chatgpt.com/backend-api/tasks")
        self.assertEqual(self.calls[-1][1], 1120)

    def test_repeated_limits_increase_cooldown_instead_of_immediate_retry(self):
        session = self.session(statuses=[(429, {}), (429, {}), (200, {})])
        for _ in range(3):
            session.request("GET", "https://chatgpt.com/backend-api/tasks")
        self.assertEqual([row[1] for row in self.calls], [1000, 1060, 1180])

    def test_other_account_and_object_storage_are_not_blocked_by_account_cooldown(self):
        first = self.session(statuses=[(429, {}), (200, {})])
        second = self.session(account_id="account-b")
        first.request("GET", "https://chatgpt.com/backend-api/tasks")
        second.request("GET", "https://chatgpt.com/backend-api/tasks")
        first.request("GET", "https://storage.example.test/image.png")
        self.assertEqual([row[1] for row in self.calls], [1000, 1000, 1000])

    def test_invalid_retry_after_does_not_disable_cooldown(self):
        for value in (None, "NaN", "Infinity", "-1", "bad"):
            self.assertEqual(pacing.retry_after_seconds(value), 0)
        self.assertGreater(pacing.retry_after_seconds("Fri, 01 Jan 2100 00:00:00 GMT"), 0)

    def test_restart_preserves_message_interval_and_shared_cooldown(self):
        first = self.session(statuses=[(429, {"Retry-After": "120"})])
        first.request("POST", "https://chatgpt.com/backend-api/conversation")
        self.advance(10)
        pacing._clocks.clear()
        restarted = self.session(token="refreshed", statuses=[(429, {}), (200, {})])
        restarted.request("GET", "https://chatgpt.com/backend-api/tasks")
        restarted.request("POST", "https://chatgpt.com/backend-api/conversation")
        self.assertEqual([row[1] for row in self.calls], [1000, 1120, 1240])
        self.assertEqual(next(iter(pacing._clocks.values())).rate_failures, 2)

    def test_restart_after_unknown_send_does_not_reset_reserved_message_interval(self):
        first = self.session()
        first.request("POST", "https://chatgpt.com/backend-api/conversation")
        self.advance(10)
        pacing._clocks.clear()
        self.session().request("POST", "https://chatgpt.com/backend-api/conversation")
        self.assertEqual([row[1] for row in self.calls], [1000, 1030])

    def test_successful_poll_does_not_reset_generation_rate_backoff(self):
        session = self.session(statuses=[(429, {}), (200, {}), (429, {}), (200, {})])
        session.request("POST", "https://chatgpt.com/backend-api/conversation")
        session.request("GET", "https://chatgpt.com/backend-api/tasks")
        session.request("POST", "https://chatgpt.com/backend-api/conversation")
        session.request("GET", "https://chatgpt.com/backend-api/tasks")
        self.assertEqual([row[1] for row in self.calls], [1000, 1060, 1070, 1190])

    def test_message_stream_preserves_send_spacing_without_holding_account_for_whole_reply(self):
        # The current production clock reserves send starts, not whole SSE
        # lifetimes. PoolAdmission owns active capacity and conversation order.
        clock = pacing.AccountRequestClock()
        entered = threading.Event()
        starts = []
        close = mock.Mock()
        first = SimpleNamespace(status_code=200, headers={}, close=close, iter_lines=lambda: iter([]))
        clock.request(lambda *a, **k: first, "POST", "https://chatgpt.com/backend-api/conversation", stream=True)
        def send(*args, **kwargs):
            starts.append(self.now)
            entered.set()
            return SimpleNamespace(status_code=200, headers={})
        worker = threading.Thread(target=lambda: clock.request(
            send, "POST", "https://chatgpt.com/backend-api/f/conversation"))
        worker.start()
        try:
            self.assertTrue(entered.wait(1), "an independent turn may start while the first stream is still open")
            close.assert_not_called()
            self.assertEqual(starts, [1030.0], "starting another turn must still obey message spacing")
            read = mock.Mock(return_value=SimpleNamespace(status_code=200, headers={}))
            clock.request(read, "GET", "https://chatgpt.com/backend-api/conversation/original")
            read.assert_called_once()
            self.assertEqual(self.now, 1035.0, "original-result reads retain request spacing")
            first.close()
            first.close()  # Watchdog and generator cleanup can both close it.
        finally:
            first.close()
            worker.join(1)
        self.assertFalse(worker.is_alive())

    def test_http_200_stream_rate_error_cools_whole_account_and_releases_turn(self):
        clock = pacing.AccountRequestClock()
        response = SimpleNamespace(status_code=200, headers={"x-request-id": "original-upstream-id"}, close=lambda: None,
            iter_lines=lambda: iter([b'data: {"error":{"code":"rate_limit_exceeded","message":"Too many requests"}}']))
        clock.request(lambda *a, **k: response, "POST", "https://chatgpt.com/backend-api/conversation", stream=True)
        with mock.patch.object(pacing.logger, "warning") as log:
            list(response.iter_lines())
        evidence = log.call_args.args[0]
        self.assertEqual(evidence["upstream_request_id"], "original-upstream-id")
        self.assertEqual(evidence["origin"], "sse_rate_limit")
        self.assertEqual(evidence["layer"], "upstream_chatgpt")
        self.assertEqual(clock.rate_failures, 1)
        self.assertEqual(clock.cooldown_until, 1060)
        self.assertTrue(clock.turn_lock.acquire(blocking=False))
        clock.turn_lock.release()
        self.assertFalse(pacing.rate_limited_event('data: {"message":{"content":"Too many requests"}}'))

    def test_failed_request_does_not_leak_account_turn_lock_or_replay(self):
        clock = pacing.AccountRequestClock()
        send = mock.Mock(side_effect=TimeoutError("no response"))
        with self.assertRaises(TimeoutError):
            clock.request(send, "POST", "https://chatgpt.com/backend-api/conversation", stream=True)
        self.assertEqual(send.call_count, 1)
        self.assertTrue(clock.turn_lock.acquire(blocking=False))
        clock.turn_lock.release()

    def test_deadline_held_turn_lock_prevents_send_and_releases_nothing(self):
        clock = pacing.AccountRequestClock()

        class HeldLock:
            def __init__(self):
                self.timeouts = []

            def acquire(self, *, timeout=None, **_kwargs):
                self.timeouts.append(timeout)
                return False

        held_lock = HeldLock()
        clock.turn_lock = held_lock
        send = mock.Mock()
        with self.assertRaises(pacing.AccountRequestDeadlineExceeded):
            clock.request(
                send, "POST", "https://chatgpt.com/backend-api/conversation",
                timeout=2,
                _account_request_deadline_monotonic=self.now + 2,
            )
        send.assert_not_called()
        self.assertEqual(held_lock.timeouts, [2])

    def test_deadline_expires_during_cooldown_without_submission_or_lock_leak(self):
        clock = pacing.AccountRequestClock()
        clock.cooldown_until = self.now + 30
        send = mock.Mock()
        before_send = mock.Mock()
        with self.assertRaises(pacing.AccountRequestDeadlineExceeded):
            clock.request(
                send, "POST", "https://chatgpt.com/backend-api/conversation",
                timeout=5,
                _account_request_before_send=before_send,
                _account_request_deadline_monotonic=self.now + 5,
            )
        send.assert_not_called()
        before_send.assert_not_called()
        self.assertEqual(clock.cooldown_until, self.now + 30)
        self.assertTrue(clock.turn_lock.acquire(blocking=False))
        clock.turn_lock.release()

    def test_metadata_requests_keep_network_timeout_after_pacing_wait(self):
        clock = pacing.AccountRequestClock()
        pacing.config.account_request_interval_secs = 10
        sent = []
        def send(_method, _url, **kwargs):
            sent.append((self.now, kwargs["timeout"]))
            return SimpleNamespace(status_code=200, headers={})
        for _ in range(3):
            clock.request(send, "GET", "https://chatgpt.com/backend-api/me", timeout=20)
        self.assertEqual(sent, [(1000, 20), (1010, 20), (1020, 20)])

    def test_actual_backend_session_get_and_post_use_shared_account_pacing(self):
        def send(_session, method, url, **kwargs):
            self.calls.append((method, self.now))
            return SimpleNamespace(status_code=200, headers={})
        with mock.patch("services.openai_backend_api.account_service.get_account", return_value={"account_id":"same-account"}), \
                mock.patch("services.openai_backend_api.requests.Session.request", new=send):
            first = OpenAIBackendAPI(access_token="token-1")
            second = OpenAIBackendAPI(access_token="token-2")
            try:
                first.session.post("https://chatgpt.com/backend-api/conversation")
                second.session.get("https://chatgpt.com/backend-api/tasks")
                second.session.post("https://chatgpt.com/backend-api/f/conversation")
            finally:
                first.close()
                second.close()
        self.assertEqual(self.calls, [("POST",1000), ("GET",1005), ("POST",1030)])

    def test_text_stream_watchdog_closes_a_stalled_response(self):
        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend._bootstrap = mock.Mock()
        backend._get_chat_requirements = mock.Mock(return_value=ChatRequirements(token="requirements"))
        backend._chat_target = mock.Mock(return_value=("/backend-api/conversation", "UTC"))
        backend._conversation_headers = mock.Mock(return_value={})
        closed = threading.Event()

        class Response:
            status_code = 200
            headers = {}

            def close(self):
                closed.set()

            def iter_lines(self):
                closed.wait(1)
                if False:
                    yield b""

        response = Response()
        backend.session = SimpleNamespace(post=mock.Mock(return_value=response))
        with (
            mock.patch("services.openai_backend_api.TEXT_STREAM_HARD_CAP_SECS", 0.01),
            mock.patch("services.openai_backend_api.time.monotonic", self.real_monotonic),
        ):
            with self.assertRaises(StreamHardTimeoutError):
                next(backend.stream_conversation(prompt="hello"))
        self.assertTrue(closed.is_set())

    def test_text_stream_uses_utf8_postfields_without_losing_pacing_model_or_input(self):
        from curl_cffi import CurlOpt
        from curl_cffi.requests.session import set_curl_options

        class CapturingCurl:
            def __init__(self):
                self.options = []

            def setopt(self, option, value):
                self.options.append((option, value))

        class Response:
            status_code = 200
            headers = {}

            def close(self):
                pass

            def iter_lines(self):
                return iter([b"data: [DONE]"])

        class OfflineSession:
            def __init__(self):
                self.calls = []
                self.postfields = []
                self.request = self._raw_request

            def post(self, url, **kwargs):
                return self.request("POST", url, **kwargs)

            def _raw_request(self, method, url, **kwargs):
                self.calls.append((method, url, kwargs))
                if method == "POST":
                    curl = CapturingCurl()
                    set_curl_options(
                        curl,
                        method,
                        url,
                        data=kwargs.get("data"),
                        json=kwargs.get("json"),
                        params_list=[None, None],
                        headers_list=[None, kwargs.get("headers")],
                        cookies_list=[None, None],
                        proxies_list=[None, None],
                        verify_list=[None, None],
                        stream=False,
                    )
                    self.postfields.append(next(value for option, value in curl.options if option == CurlOpt.POSTFIELDS))
                return Response()

        payload = {
            "model": "gpt-5-6-instant",
            "messages": [{"id": "original-user", "content": {"parts": ["中文🌟"]}}],
            "parent_message_id": "frozen-parent",
            "nested": {"text": "保留原输入"},
        }
        canonical_before = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload_before = json.loads(json.dumps(payload, ensure_ascii=False))
        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.com"
        backend._bootstrap = mock.Mock()
        backend._get_chat_requirements = mock.Mock(return_value=ChatRequirements(token="requirements"))
        backend._chat_target = mock.Mock(return_value=("/backend-api/conversation", "UTC"))
        backend._conversation_headers = mock.Mock(return_value={"Content-Type": "application/json"})
        backend._conversation_payload = mock.Mock(return_value=payload)
        backend.text_pre_send_check = lambda read: read("GET", "https://chatgpt.com/backend-api/conversation/original")
        backend.session = OfflineSession()
        pacing.pace_account_session(backend.session, {"account_id": "utf8-fixture"}, "fixture-token")

        with mock.patch.object(pacing.logger, "info") as log:
            self.assertEqual(list(backend.stream_conversation(messages=[{"role": "user", "content": "unused"}],
                                                              model="gpt-5-6-instant")), ["[DONE]"])

        self.assertEqual(payload, payload_before)
        self.assertEqual(
            hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).digest(),
            hashlib.sha256(canonical_before).digest(),
        )
        body = backend.session.postfields[0]
        self.assertEqual(json.loads(body.decode("utf-8")), payload)
        self.assertIn("中文🌟".encode("utf-8"), body)
        self.assertNotIn("\\u4e2d".encode(), body)
        self.assertLess(len(body), len(canonical_before))
        post = next(kwargs for method, _, kwargs in backend.session.calls if method == "POST")
        self.assertNotIn("json", post)
        self.assertNotIn("_account_request_model", post)
        starts = [entry.args[0] for entry in log.call_args_list
                  if entry.args and entry.args[0].get("event") == "account_message_start"]
        self.assertEqual([entry["model"] for entry in starts], ["gpt-5-6-instant"])
        self.assertEqual([method for method, _, _ in backend.session.calls], ["GET", "POST"])

    def test_text_stream_preflight_rejection_does_not_reach_utf8_transport(self):
        class OfflineSession:
            def __init__(self):
                self.calls = []
                self.request = self._raw_request

            def post(self, url, **kwargs):
                return self.request("POST", url, **kwargs)

            def _raw_request(self, method, url, **kwargs):
                self.calls.append((method, url, kwargs))
                raise AssertionError("preflight rejection must stop before any transport send")

        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.com"
        backend._bootstrap = mock.Mock()
        backend._get_chat_requirements = mock.Mock(return_value=ChatRequirements(token="requirements"))
        backend._chat_target = mock.Mock(return_value=("/backend-api/conversation", "UTC"))
        backend._conversation_headers = mock.Mock(return_value={"Content-Type": "application/json"})
        backend._conversation_payload = mock.Mock(return_value={"model": "gpt-5-6-instant", "messages": []})
        backend.text_pre_send_check = mock.Mock(side_effect=ConversationArchiveCursorMismatch("newer original"))
        backend.session = OfflineSession()
        pacing.pace_account_session(backend.session, {"account_id": "utf8-preflight"}, "fixture-token")

        with self.assertRaises(ConversationArchiveCursorMismatch):
            list(backend.stream_conversation(model="gpt-5-6-instant"))
        backend.text_pre_send_check.assert_called_once()
        self.assertEqual(backend.session.calls, [])

    def test_json_model_keeps_precedence_over_transport_model_hint(self):
        clock = pacing.AccountRequestClock("model-precedence")
        sent = mock.Mock(return_value=SimpleNamespace(status_code=200, headers={}))
        with mock.patch.object(pacing.logger, "info") as log:
            clock.request(sent, "POST", "https://chatgpt.com/backend-api/conversation",
                          json={"model": "json-model"}, _account_request_model="transport-model")
        self.assertNotIn("_account_request_model", sent.call_args.kwargs)
        starts = [entry.args[0] for entry in log.call_args_list
                  if entry.args and entry.args[0].get("event") == "account_message_start"]
        self.assertEqual([entry["model"] for entry in starts], ["json-model"])

    def test_transport_model_hint_is_retained_on_429_attribution(self):
        clock = pacing.AccountRequestClock("utf8-429-model")
        sent = mock.Mock(return_value=SimpleNamespace(status_code=429, headers={}))
        with mock.patch.object(pacing.logger, "warning") as log:
            response = clock.request(
                sent,
                "POST",
                "https://chatgpt.com/backend-api/conversation",
                data="{}",
                _account_request_model="gpt-5-6-instant",
            )
        self.assertEqual(response.status_code, 429)
        self.assertNotIn("_account_request_model", sent.call_args.kwargs)
        limited = [entry.args[0] for entry in log.call_args_list
                   if entry.args and entry.args[0].get("event") == "account_rate_limited"]
        self.assertEqual([entry["model"] for entry in limited], ["gpt-5-6-instant"])

    def test_non_chat_transport_strips_private_model_hint(self):
        class Session:
            def __init__(self):
                self.raw_request = mock.Mock(return_value=SimpleNamespace(status_code=200, headers={}))
                self.request = self.raw_request

        session = Session()
        pacing.pace_account_session(session, {"account_id": "non-chat-model-hint"}, "fixture-token")
        session.request("POST", "https://proxy.example/conversation", data="{}", _account_request_model="fixture-model")
        self.assertNotIn("_account_request_model", session.raw_request.call_args.kwargs)


class ConversationContinuationPayloadTests(unittest.TestCase):
    def test_b_request_uses_chat_instant_without_changing_content_pool_default(self):
        backend = object.__new__(OpenAIBackendAPI)
        backend.image_upstream_model = "gpt-5-6-instant"
        with mock.patch("services.openai_backend_api.config", SimpleNamespace(
            default_upstream_model_name="gpt-5.6-sol-wm", default_thinking_effort="extended")):
            self.assertEqual(backend._image_model_settings("gpt-image-2"), ("gpt-5-6-instant", ""))
            content_backend = object.__new__(OpenAIBackendAPI)
            self.assertEqual(content_backend._image_model_settings("gpt-image-2"), ("gpt-5.6-sol-wm", "extended"))

    def test_image_conversation_falls_back_from_removed_f_route(self) -> None:
        class FakeResponse:
            def __init__(self, status_code: int) -> None:
                self.status_code = status_code
                self.text = ""
                self.headers = {}
                self.closed = False

            def close(self) -> None:
                self.closed = True

            def json(self):
                return {}

        class FakeSession:
            def __init__(self) -> None:
                self.responses = [FakeResponse(404), FakeResponse(200)]
                self.calls = []

            def post(self, url, **kwargs):
                self.calls.append((url, kwargs))
                callback = kwargs.pop("_account_request_before_send", None)
                if callable(callback):
                    callback()
                return self.responses[len(self.calls) - 1]

        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend.session = FakeSession()
        backend._image_headers = lambda path, *_args: {"x-test-path": path}
        backend.access_token = "fixture-capability-token"
        with mock.patch("services.openai_backend_api.account_service.require_image_account") as capability:
            response = backend._start_image_generation(
                "make an image",
                ChatRequirements(token="requirements"),
                "conduit",
                "gpt-image-2",
                conversation_id="conversation-1",
                parent_message_id="message-1",
            )
        self.assertEqual(capability.call_args_list, [mock.call("fixture-capability-token", "gpt-image-2")] * 2)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(backend.session.responses[0].closed)
        self.assertEqual(
            [call[0] for call in backend.session.calls],
            [
                "https://chatgpt.test/backend-api/f/conversation",
                "https://chatgpt.test/backend-api/conversation",
            ],
        )
        self.assertEqual(backend.session.calls[0][1]["json"], backend.session.calls[1][1]["json"])

    def test_continuation_sends_exact_conversation_and_parent(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)

        payload = backend._conversation_payload(
            [{"role": "user", "content": "continue"}],
            "auto",
            "UTC",
            conversation_id="conversation-1",
            parent_message_id="message-1",
        )

        self.assertEqual(payload["conversation_id"], "conversation-1")
        self.assertEqual(payload["parent_message_id"], "message-1")

    def test_bound_multimodal_parts_preserve_interleaving_and_cursor(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend.access_token = "fixture-token"
        backend.retain_bound_conversation = True
        backend._bootstrap = mock.Mock()
        backend._get_chat_requirements = mock.Mock(return_value=ChatRequirements(token="fixture"))
        backend._chat_target = mock.Mock(return_value=("/backend-api/conversation", "UTC"))
        backend._conversation_headers = mock.Mock(return_value={})
        response = SimpleNamespace(status_code=200, headers={}, close=mock.Mock())
        backend.session = SimpleNamespace(post=mock.Mock(return_value=response))
        backend._iter_sse_payloads_capped = mock.Mock(return_value=iter(["fixture event"]))
        backend._upload_image = mock.Mock(side_effect=[
            {"file_id": "first", "width": 10, "height": 20, "file_size": 30,
             "mime_type": "image/png", "file_name": "image_1.png"},
            {"file_id": "second", "width": 11, "height": 21, "file_size": 31,
             "mime_type": "image/jpeg", "file_name": "image_2.jpg"},
        ])

        events = list(backend.stream_conversation(
            [{"role": "user", "content": [
                {"type": "text", "text": "label first"},
                {"type": "image", "data": b"first-bytes", "mime": "image/png"},
                {"type": "text", "text": "label second"},
                {"type": "image", "data": b"second-bytes", "mime": "image/jpeg"},
                {"type": "text", "text": "final instruction"},
            ]}],
            "gpt-5-6-thinking", thinking_effort="high",
            conversation_id="conversation-1", parent_message_id="message-1",
        ))

        self.assertEqual(events, ["fixture event"])
        backend.session.post.assert_called_once()
        call = backend.session.post.call_args
        self.assertEqual(call.args, ("https://chatgpt.test/backend-api/conversation",))
        self.assertEqual(call.kwargs["_account_request_model"], "gpt-5-6-thinking")
        payload = json.loads(call.kwargs["data"])
        response.close.assert_called_once()

        self.assertEqual(payload["conversation_id"], "conversation-1")
        self.assertEqual(payload["parent_message_id"], "message-1")
        self.assertEqual(payload["model"], "gpt-5-6-thinking")
        self.assertEqual(payload["thinking_effort"], "extended")
        self.assertEqual(
            payload["messages"][0]["content"]["parts"],
            [
                "label first",
                {"content_type": "image_asset_pointer", "asset_pointer": "file-service://first",
                 "width": 10, "height": 20, "size_bytes": 30},
                "label second",
                {"content_type": "image_asset_pointer", "asset_pointer": "file-service://second",
                 "width": 11, "height": 21, "size_bytes": 31},
                "final instruction",
            ],
        )
        self.assertEqual(
            [item["id"] for item in payload["messages"][0]["metadata"]["attachments"]],
            ["first", "second"],
        )

    def test_multimodal_conversion_keeps_text_only_and_single_image_forms(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)
        backend.access_token = "fixture-token"
        backend._upload_image = mock.Mock(return_value={
            "file_id": "one", "width": 1, "height": 2, "file_size": 3,
            "mime_type": "image/png", "file_name": "image_1.png",
        })

        messages = backend._api_messages_to_conversation_messages([
            {"role": "system", "content": "plain text"},
            {"role": "user", "content": [{"type": "text", "text": "zero images"}]},
            {"role": "user", "content": [{"type": "image", "data": b"one", "mime": "image/png"}]},
        ])

        self.assertEqual(messages[0]["content"], {"content_type": "text", "parts": ["plain text"]})
        self.assertEqual(messages[1]["content"], {"content_type": "text", "parts": ["zero images"]})
        self.assertEqual(messages[2]["content"]["parts"], [
            {"content_type": "image_asset_pointer", "asset_pointer": "file-service://one",
             "width": 1, "height": 2, "size_bytes": 3},
        ])

        backend._upload_image.reset_mock()
        payload = backend._conversation_payload(
            [{"role": "user", "content": [
                {"type": "text", "text": "continue "},
                {"type": "text", "text": "without images"},
            ]}],
            "gpt-5-6-thinking", "UTC", thinking_effort="high",
            conversation_id="conversation-1", parent_message_id="message-1",
        )
        self.assertEqual(payload["messages"][0]["content"],
                         {"content_type": "text", "parts": ["continue without images"]})
        self.assertEqual(payload["conversation_id"], "conversation-1")
        self.assertEqual(payload["parent_message_id"], "message-1")
        self.assertEqual(payload["model"], "gpt-5-6-thinking")
        self.assertEqual(payload["thinking_effort"], "extended")
        backend._upload_image.assert_not_called()

    def test_ten_images_keep_labels_upload_bytes_and_attachment_order(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)
        backend.access_token = "fixture-token"
        backend.text_request_message_id = "original-request"
        refs = [{"file_id": f"file-{i}", "width": i + 1, "height": i + 2,
                 "file_size": 1, "mime_type": "image/png", "file_name": f"image_{i + 1}.png"}
                for i in range(10)]
        backend._upload_image = mock.Mock(side_effect=refs)
        content = [{"type": "text", "text": ""}]
        for i in range(10):
            content.extend([{"type": "text", "text": f"label-{i}"},
                            {"type": "image", "data": bytes([i]), "mime": "image/png"}])
        content.extend([{"type": "text", "text": "review the final image"},
                        {"type": "text", "text": ""}])

        message = backend._api_messages_to_conversation_messages(
            [{"role": "user", "content": content}])[0]

        self.assertEqual(message["id"], "original-request")
        parts = message["content"]["parts"]
        self.assertEqual(len(parts), 21)
        self.assertEqual(parts[-1], "review the final image")
        for i in range(10):
            self.assertEqual(parts[i * 2], f"label-{i}")
            self.assertEqual(parts[i * 2 + 1]["asset_pointer"], f"file-service://file-{i}")
            self.assertEqual(parts[i * 2 + 1]["width"], i + 1)
            self.assertEqual(parts[i * 2 + 1]["height"], i + 2)
        self.assertEqual(backend._upload_image.call_count, 10)
        for i, call in enumerate(backend._upload_image.call_args_list):
            self.assertEqual(base64.b64decode(call.args[0].split(",", 1)[1]), bytes([i]))
            self.assertEqual(call.args[1], f"image_{i + 1}.png")
        self.assertEqual([item["id"] for item in message["metadata"]["attachments"]],
                         [ref["file_id"] for ref in refs])

    def test_image_upload_failure_never_sends_a_partial_conversation(self) -> None:
        image_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "first"},
            {"type": "image", "data": image_bytes, "mime": "image/png"},
            {"type": "text", "text": "second"},
            {"type": "image", "data": image_bytes, "mime": "image/png"},
        ]}]

        for failing_step in ("init", "put", "confirm"):
            with self.subTest(failing_step=failing_step):
                backend = object.__new__(OpenAIBackendAPI)
                backend.base_url = "https://chatgpt.test"
                backend.user_agent = "fixture"
                backend.access_token = "fixture-token"
                backend._bootstrap = mock.Mock()
                backend._get_chat_requirements = mock.Mock(return_value=ChatRequirements(token="fixture"))
                backend._chat_target = mock.Mock(return_value=("/backend-api/conversation", "UTC"))
                backend._headers = mock.Mock(return_value={})
                backend._image_request_options = mock.Mock(return_value={})
                calls = []
                upload_count = 0

                def send(method, url, **kwargs):
                    nonlocal upload_count
                    calls.append((method, url))
                    if url == "https://chatgpt.test/backend-api/files":
                        upload_count += 1
                        step = "init"
                    elif method == "PUT":
                        step = "put"
                        self.assertEqual(kwargs["data"], image_bytes)
                    elif url.endswith("/uploaded"):
                        step = "confirm"
                    else:
                        self.fail(f"unexpected upstream request: {method} {url}")
                    if upload_count == 2 and step == failing_step:
                        raise RuntimeError("fixture upload failure")
                    return SimpleNamespace(status_code=200, headers={}, json=lambda: {
                        "file_id": f"file-{upload_count}",
                        "upload_url": f"https://storage.test/file-{upload_count}",
                    })

                backend.session = SimpleNamespace(
                    post=lambda url, **kwargs: send("POST", url, **kwargs),
                    put=lambda url, **kwargs: send("PUT", url, **kwargs),
                )
                with self.assertRaisesRegex(RuntimeError, "fixture upload failure"):
                    list(backend.stream_conversation(
                        messages, "gpt-5-6-thinking", thinking_effort="high",
                        conversation_id="conversation-1", parent_message_id="message-1"))

                self.assertEqual(upload_count, 2)
                self.assertIn(("POST", "https://chatgpt.test/backend-api/files/file-1/uploaded"), calls)
                self.assertNotIn(("POST", "https://chatgpt.test/backend-api/conversation"), calls)

    def test_partial_continuation_cursor_fails_closed(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)

        with self.assertRaisesRegex(RuntimeError, "requires parent_message_id"):
            backend._conversation_payload(
                [{"role": "user", "content": "continue"}],
                "auto",
                "UTC",
                conversation_id="conversation-1",
            )

    def test_bound_image_uses_exact_account_and_advances_parent(self) -> None:
        request = ConversationRequest(
            model="gpt-image-2",
            prompt="continue",
            provider_binding_id="cb-account-a",
            provider_account_identity="account-opaque-a",
            client_conversation_id="workbench-conversation-1",
            conversation_id="conversation-1",
            parent_message_id="message-1",
            retain_conversation=True,
            upstream_model="gpt-5-6-instant",
        )

        class FakeBackend:
            def __init__(self, access_token: str) -> None:
                self.access_token = access_token
                self.progress_callback = None

            def _get_conversation(self, conversation_id: str, **_kwargs) -> dict:
                self.test_case.assertEqual(conversation_id, "conversation-1")
                return {
                    "conversation_id": conversation_id,
                    "current_node": "message-1",
                    "mapping": {
                        "message-1": {"children": [], "message": {
                            "id": "message-1", "author": {"role": "assistant"},
                            "status": "finished_successfully", "end_turn": True,
                            "channel": "final",
                        }},
                    },
                }

            def get_conversation_parent_message_id(self, conversation_id: str) -> str:
                self.test_case.assertEqual(conversation_id, "conversation-1")
                self.test_case.assertEqual(self.image_upstream_model, "gpt-5-6-instant")
                self.test_case.assertTrue(self.retain_bound_conversation)
                return "message-2"

            def close(self) -> None:
                pass

        FakeBackend.test_case = self
        output = ImageOutput(
            kind="result",
            model="gpt-image-2",
            index=1,
            total=1,
            data=[{"url": "image.png"}],
            conversation_id="conversation-1",
        )

        def stream(backend, sent_request, *_args):
            self.assertEqual(sent_request.parent_message_id, "message-1")
            self.assertTrue(callable(backend.image_pre_send_check))
            backend.image_pre_send_check()
            return iter([output])

        with (
            mock.patch(
                "services.protocol.conversation.account_service.acquire_bound_image_access_token",
                return_value="token-a",
            ) as acquire,
            mock.patch(
                "services.protocol.conversation.account_service.get_bound_account_identity",
                return_value="account-opaque-a",
            ),
            mock.patch(
                "services.protocol.conversation.account_service.get_available_access_token",
                side_effect=AssertionError("bound image must not round-robin"),
            ),
            mock.patch(
                "services.protocol.conversation.account_service.get_account",
                return_value={"email": "a@example.test"},
            ),
            mock.patch(
                "services.protocol.conversation.account_service.conversation_binding_lock",
                return_value=nullcontext(),
            ),
            mock.patch("services.protocol.conversation.account_service.mark_image_result"),
            mock.patch("services.protocol.conversation.account_service.release_image_slot"),
            mock.patch("services.protocol.conversation.OpenAIBackendAPI", FakeBackend),
            mock.patch(
                "services.protocol.conversation.stream_image_outputs",
                side_effect=stream,
            ),
        ):
            result = _generate_bound_single_image(request, 1, 1)

        acquire.assert_called_once_with("cb-account-a", image_model="gpt-image-2")
        self.assertEqual(result[0].provider_account_identity, "account-opaque-a")
        self.assertEqual(result[0].provider_binding_id, "cb-account-a")
        self.assertEqual(result[0].conversation_id, "conversation-1")
        self.assertEqual(result[0].parent_message_id, "message-2")

    def test_bound_image_without_thread_rejects_later_user_before_submit(self) -> None:
        request = ConversationRequest(
            model="gpt-image-2",
            prompt="continue",
            provider_binding_id="cb-account-a",
            provider_account_identity="account-opaque-a",
            client_conversation_id="workbench-conversation-1",
            conversation_id="conversation-1",
            parent_message_id="completed-revision-reply",
            retain_conversation=True,
        )
        submitted = []
        documents = [
            {
                "conversation_id": "conversation-1",
                "current_node": "completed-revision-reply",
                "mapping": {
                    "completed-revision-reply": {"children": [], "message": {
                        "id": "completed-revision-reply", "author": {"role": "assistant"},
                        "status": "finished_successfully", "end_turn": True, "channel": "final",
                    }},
                },
            },
            {
                "conversation_id": "conversation-1",
                "current_node": "later-user",
                "mapping": {
                    "completed-revision-reply": {"children": ["later-user"], "message": {
                        "id": "completed-revision-reply", "author": {"role": "assistant"},
                        "status": "finished_successfully", "end_turn": True, "channel": "final",
                    }},
                    "later-user": {"parent": "completed-revision-reply", "children": [], "message": {
                        "id": "later-user", "author": {"role": "user"},
                    }},
                },
            },
        ]

        class FakeBackend:
            def __init__(self, access_token: str) -> None:
                self.access_token = access_token
                self.progress_callback = None

            def _get_conversation(self, conversation_id: str, **_kwargs) -> dict:
                self.test_case.assertEqual(conversation_id, "conversation-1")
                return documents.pop(0)

            def close(self) -> None:
                pass

        FakeBackend.test_case = self

        def stream(backend, _request, *_args):
            backend.image_pre_send_check()
            submitted.append(True)
            return iter(())

        with (
            mock.patch(
                "services.protocol.conversation.account_service.acquire_bound_image_access_token",
                return_value="token-a",
            ),
            mock.patch(
                "services.protocol.conversation.account_service.get_bound_account_identity",
                return_value="account-opaque-a",
            ),
            mock.patch(
                "services.protocol.conversation.account_service.get_account",
                return_value={"email": "a@example.test"},
            ),
            mock.patch(
                "services.protocol.conversation.account_service.conversation_binding_lock",
                return_value=nullcontext(),
            ),
            mock.patch("services.protocol.conversation.account_service.mark_image_result"),
            mock.patch("services.protocol.conversation.account_service.release_image_slot"),
            mock.patch("services.protocol.conversation.OpenAIBackendAPI", FakeBackend),
            mock.patch("services.protocol.conversation.stream_image_outputs", side_effect=stream),
        ):
            with self.assertRaises(ImageGenerationError) as captured:
                _generate_bound_single_image(request, 1, 1)

        self.assertEqual(captured.exception.code, "CONVERSATION_BINDING_MISMATCH")
        self.assertFalse(captured.exception.upstream_submitted)
        self.assertEqual(submitted, [])

    def test_bound_image_without_thread_accepts_completed_image_tool_parent(self) -> None:
        request = ConversationRequest(
            model="gpt-image-2",
            prompt="continue",
            provider_binding_id="cb-account-a",
            provider_account_identity="account-opaque-a",
            client_conversation_id="workbench-conversation-1",
            conversation_id="conversation-1",
            parent_message_id="completed-image-tool",
            retain_conversation=True,
        )

        class FakeBackend:
            def __init__(self, access_token: str) -> None:
                self.access_token = access_token
                self.progress_callback = None

            def _get_conversation(self, conversation_id: str, **_kwargs) -> dict:
                self.test_case.assertEqual(conversation_id, "conversation-1")
                return {
                    "conversation_id": conversation_id,
                    "current_node": "completed-image-tool",
                    "mapping": {
                        "original-user": {"children": ["image-call"], "message": {
                            "id": "original-user", "author": {"role": "user"},
                            "status": "finished_successfully",
                        }},
                        "image-call": {"parent": "original-user", "children": ["completed-image-tool"], "message": {
                            "id": "image-call", "author": {"role": "assistant"},
                            "status": "finished_successfully", "end_turn": False,
                            "recipient": "image_gen", "content": {"content_type": "code"},
                        }},
                        "completed-image-tool": {"parent": "image-call", "children": [], "message": {
                            "id": "completed-image-tool", "author": {"role": "tool", "name": "image_gen"},
                            "status": "finished_successfully", "end_turn": True, "recipient": "all",
                            "content": {"content_type": "multimodal_text", "parts": [{
                                "content_type": "image_asset_pointer",
                                "asset_pointer": "file-service://generated-image",
                            }]},
                        }},
                    },
                }

            def get_conversation_parent_message_id(self, conversation_id: str) -> str:
                self.test_case.assertEqual(conversation_id, "conversation-1")
                return "next-parent"

            def close(self) -> None:
                pass

        FakeBackend.test_case = self
        output = ImageOutput(
            kind="result", model="gpt-image-2", index=1, total=1,
            data=[{"url": "image.png"}], conversation_id="conversation-1",
        )

        def stream(backend, sent_request, *_args):
            self.assertEqual(sent_request.parent_message_id, "completed-image-tool")
            backend.image_pre_send_check()
            return iter([output])

        with (
            mock.patch(
                "services.protocol.conversation.account_service.acquire_bound_image_access_token",
                return_value="token-a",
            ),
            mock.patch(
                "services.protocol.conversation.account_service.get_bound_account_identity",
                return_value="account-opaque-a",
            ),
            mock.patch(
                "services.protocol.conversation.account_service.get_account",
                return_value={"email": "a@example.test"},
            ),
            mock.patch(
                "services.protocol.conversation.account_service.conversation_binding_lock",
                return_value=nullcontext(),
            ),
            mock.patch("services.protocol.conversation.account_service.mark_image_result"),
            mock.patch("services.protocol.conversation.account_service.release_image_slot"),
            mock.patch("services.protocol.conversation.OpenAIBackendAPI", FakeBackend),
            mock.patch("services.protocol.conversation.stream_image_outputs", side_effect=stream),
        ):
            result = _generate_bound_single_image(request, 1, 1)

        self.assertEqual(result[0].parent_message_id, "next-parent")

    def test_bound_image_without_thread_rejects_missing_or_unfinished_reply_before_submit(self) -> None:
        request = ConversationRequest(
            model="gpt-image-2",
            prompt="continue",
            provider_binding_id="cb-account-a",
            provider_account_identity="account-opaque-a",
            client_conversation_id="workbench-conversation-1",
            conversation_id="conversation-1",
            parent_message_id="completed-revision-reply",
            retain_conversation=True,
        )
        documents = {
            "missing_parent": {"conversation_id": "conversation-1", "current_node": "completed-revision-reply", "mapping": {}},
            "malformed_mapping": {
                "conversation_id": "conversation-1", "current_node": "completed-revision-reply",
                "mapping": {
                    "completed-revision-reply": {"children": [], "message": {
                        "id": "completed-revision-reply", "author": {"role": "assistant"},
                        "status": "finished_successfully", "end_turn": True,
                    }},
                    "unexpected-shape": "not-a-conversation-node",
                },
            },
            "unfinished_parent": {
                "conversation_id": "conversation-1", "current_node": "completed-revision-reply",
                "mapping": {"completed-revision-reply": {"children": [], "message": {
                    "id": "completed-revision-reply", "author": {"role": "assistant"},
                    "status": "in_progress", "end_turn": False,
                }}},
            },
        }
        submitted = []

        class FakeBackend:
            document = None

            def __init__(self, access_token: str) -> None:
                self.access_token = access_token
                self.progress_callback = None

            def _get_conversation(self, conversation_id: str, **_kwargs) -> dict:
                self.test_case.assertEqual(conversation_id, "conversation-1")
                return self.document

            def close(self) -> None:
                pass

        FakeBackend.test_case = self

        def stream(_backend, _request, *_args):
            submitted.append(True)
            return iter(())

        with (
            mock.patch(
                "services.protocol.conversation.account_service.acquire_bound_image_access_token",
                return_value="token-a",
            ),
            mock.patch(
                "services.protocol.conversation.account_service.get_bound_account_identity",
                return_value="account-opaque-a",
            ),
            mock.patch(
                "services.protocol.conversation.account_service.get_account",
                return_value={"email": "a@example.test"},
            ),
            mock.patch(
                "services.protocol.conversation.account_service.conversation_binding_lock",
                return_value=nullcontext(),
            ),
            mock.patch("services.protocol.conversation.account_service.mark_image_result"),
            mock.patch("services.protocol.conversation.account_service.release_image_slot"),
            mock.patch("services.protocol.conversation.OpenAIBackendAPI", FakeBackend),
            mock.patch("services.protocol.conversation.stream_image_outputs", side_effect=stream),
        ):
            for case, document in documents.items():
                with self.subTest(case=case):
                    FakeBackend.document = document
                    with self.assertRaises(ImageGenerationError) as captured:
                        _generate_bound_single_image(request, 1, 1)
                    self.assertEqual(captured.exception.code, "CONVERSATION_BINDING_MISMATCH")
                    self.assertFalse(captured.exception.upstream_submitted)
                    self.assertEqual(submitted, [])

    def test_unavailable_bound_account_fails_without_fallback(self) -> None:
        request = ConversationRequest(
            model="gpt-image-2",
            provider_binding_id="cb-account-a",
            provider_account_identity="account-opaque-a",
            client_conversation_id="workbench-conversation-1",
            conversation_id="conversation-1",
            parent_message_id="message-1",
            retain_conversation=True,
        )
        with (
            mock.patch(
                "services.protocol.conversation.account_service.acquire_bound_image_access_token",
                side_effect=RuntimeError("conversation binding unavailable"),
            ),
            mock.patch(
                "services.protocol.conversation.account_service.get_available_access_token",
                side_effect=AssertionError("must not fall back"),
            ),
        ):
            with self.assertRaises(ImageGenerationError) as captured:
                _generate_bound_single_image(request, 1, 1)

        self.assertEqual(captured.exception.code, "CONVERSATION_BINDING_UNAVAILABLE")

    def test_text_binding_is_created_once_then_reused(self) -> None:
        service = ConversationBindingService()

        class FakeBackend:
            def __init__(self, access_token: str) -> None:
                self.access_token = access_token

            def _get_conversation(self, conversation_id: str) -> dict:
                return {"conversation_id": conversation_id, "current_node": "message-2", "is_archived": False}

            def get_conversation_parent_message_id(self, conversation_id: str) -> str:
                return "message-2" if conversation_id == "conversation-1" else ""

            def close(self) -> None:
                pass

        events = [
            {
                "type": "conversation.delta",
                "delta": "ok",
                "conversation_id": "conversation-1",
            }
        ]
        with (
            mock.patch(
                "services.conversation_binding_service.account_service.create_conversation_binding",
                return_value=("cb-account-a", "account-opaque-a", "token-a"),
            ) as create_binding,
            mock.patch(
                "services.conversation_binding_service.account_service.release_image_slot"
            ),
            mock.patch(
                "services.conversation_binding_service.account_service.get_bound_text_access_token",
                return_value="token-a",
            ) as bound_token,
            mock.patch(
                "services.conversation_binding_service.account_service.get_bound_account_identity",
                return_value="account-opaque-a",
            ),
            mock.patch(
                "services.conversation_binding_service.account_service.conversation_binding_lock",
                return_value=nullcontext(),
            ),
            mock.patch("services.conversation_binding_service.account_service.mark_text_used"),
            mock.patch("services.conversation_binding_service.OpenAIBackendAPI", FakeBackend),
            mock.patch(
                "services.conversation_binding_service.conversation_events",
                side_effect=(iter(events), iter(events)),
            ) as conversation_events,
        ):
            first = service.complete_text(
                {
                    "messages": [{"role": "user", "content": "first"}],
                    "client_conversation_id": "workbench-conversation-1",
                }
            )
            second = service.complete_text(
                {
                    "messages": [{"role": "user", "content": "second"}],
                    "provider_binding_id": first["provider_binding_id"],
                    "provider_account_identity": first["provider_account_identity"],
                    "client_conversation_id": "workbench-conversation-1",
                    "conversation_id": first["conversation_id"],
                    "parent_message_id": first["parent_message_id"],
                }
            )

        create_binding.assert_called_once()
        self.assertEqual(bound_token.call_count, 2)
        self.assertEqual(first["provider_binding_id"], "cb-account-a")
        self.assertEqual(first["provider_account_identity"], "account-opaque-a")
        self.assertEqual(second["provider_binding_id"], "cb-account-a")
        self.assertEqual(second["conversation_id"], "conversation-1")
        self.assertEqual(second["parent_message_id"], "message-2")
        second_call = conversation_events.call_args_list[1].kwargs
        self.assertEqual(second_call["conversation_id"], "conversation-1")
        self.assertEqual(second_call["parent_message_id"], "message-2")

    def test_project_binding_can_create_independent_conversations_on_same_account(self) -> None:
        service = ConversationBindingService()

        class FakeBackend:
            def __init__(self, access_token: str) -> None:
                self.access_token = access_token

            def get_conversation_parent_message_id(self, conversation_id: str) -> str:
                return "message-1"

            def close(self) -> None:
                pass

        events = [{"type": "conversation.delta", "delta": "ok", "conversation_id": "conversation-2"}]
        with (
            mock.patch(
                "services.conversation_binding_service.account_service.create_conversation_binding",
                side_effect=AssertionError("project binding must not be recreated"),
            ),
            mock.patch(
                "services.conversation_binding_service.account_service.get_bound_account_identity",
                return_value="account-opaque-a",
            ),
            mock.patch(
                "services.conversation_binding_service.account_service.get_bound_text_access_token",
                return_value="token-a",
            ),
            mock.patch(
                "services.conversation_binding_service.account_service.conversation_binding_lock",
                return_value=nullcontext(),
            ) as binding_lock,
            mock.patch("services.conversation_binding_service.account_service.mark_text_used"),
            mock.patch("services.conversation_binding_service.OpenAIBackendAPI", FakeBackend),
            mock.patch(
                "services.conversation_binding_service.conversation_events",
                return_value=iter(events),
            ),
        ):
            result = service.complete_text(
                {
                    "messages": [{"role": "user", "content": "new conversation"}],
                    "provider_binding_id": "cb-account-a",
                    "provider_account_identity": "account-opaque-a",
                    "client_conversation_id": "workbench-conversation-2",
                }
            )

        binding_lock.assert_called_once_with("cb-account-a", "workbench-conversation-2")
        self.assertEqual(result["conversation_id"], "conversation-2")
        self.assertEqual(result["provider_account_identity"], "account-opaque-a")

    def test_provider_account_identity_mismatch_fails_closed(self) -> None:
        service = ConversationBindingService()
        with mock.patch(
            "services.conversation_binding_service.account_service.get_bound_account_identity",
            return_value="account-opaque-a",
        ):
            with self.assertRaises(ConversationBindingError) as captured:
                service.complete_text(
                    {
                        "messages": [{"role": "user", "content": "continue"}],
                        "provider_binding_id": "cb-account-a",
                        "provider_account_identity": "account-opaque-b",
                        "client_conversation_id": "workbench-conversation-1",
                        "conversation_id": "conversation-1",
                        "parent_message_id": "message-1",
                    }
                )

        self.assertEqual(captured.exception.code, "CONVERSATION_BINDING_MISMATCH")

    def test_initial_unknown_returns_new_account_binding_for_authoritative_storage(self) -> None:
        service = ConversationBindingService()

        class FakeBackend:
            def __init__(self, access_token: str) -> None:
                self.access_token = access_token

            def get_conversation_parent_message_id(self, conversation_id: str) -> str:
                return "message-1"

            def close(self) -> None:
                pass

        with (
            mock.patch(
                "services.conversation_binding_service.account_service.create_conversation_binding",
                return_value=("cb-account-a", "account-opaque-a", "token-a"),
            ),
            mock.patch(
                "services.conversation_binding_service.account_service.release_image_slot"
            ),
            mock.patch(
                "services.conversation_binding_service.account_service.get_bound_text_access_token",
                return_value="token-a",
            ),
            mock.patch(
                "services.conversation_binding_service.account_service.conversation_binding_lock",
                return_value=nullcontext(),
            ),
            mock.patch("services.conversation_binding_service.OpenAIBackendAPI", FakeBackend),
            mock.patch(
                "services.conversation_binding_service.conversation_events",
                return_value=iter(
                    [{"type": "conversation.done", "conversation_id": "conversation-1"}]
                ),
            ),
        ):
            with self.assertRaises(ConversationBindingError) as captured:
                service.complete_text(
                    {
                        "messages": [{"role": "user", "content": "first"}],
                        "client_conversation_id": "workbench-conversation-1",
                    }
                )

        self.assertEqual(captured.exception.code, "CONVERSATION_OUTCOME_UNKNOWN")
        self.assertEqual(captured.exception.provider_binding_id, "cb-account-a")
        self.assertEqual(captured.exception.original_failure_phase, "result_check")
        self.assertEqual(captured.exception.original_exception_category, "empty_result")
        self.assertEqual(
            captured.exception.provider_account_identity, "account-opaque-a"
        )
        self.assertEqual(captured.exception.conversation_id, "conversation-1")


class TextResultRecoveryTests(unittest.TestCase):
    cursor = {
        "provider_binding_id": "binding-one", "provider_account_identity": "account-one",
        "client_conversation_id": "client-one", "conversation_id": "conversation-one",
        "parent_message_id": "user-one",
    }

    def document(self):
        return {"conversation_id": "conversation-one", "current_node": "answer-one", "mapping": {
            "user-one": {"parent": None, "message": {"id": "user-one", "author": {"role": "user"}}},
            "answer-one": {"parent": "user-one", "message": {"id": "answer-one", "author": {"role": "assistant"},
                "status": "finished_successfully", "end_turn": True, "channel": "final",
                "content": {"content_type": "text", "parts": ['{"name_ru":"Набор"}']}}},
        }}

    def request_receipt(self, **changes):
        receipt = {
            "provider_binding_id": "binding-one",
            "provider_account_identity": "account-one",
            "client_conversation_id": "client-one",
            "conversation_id": "conversation-one",
            "parent_message_id": "prior-answer",
            "request_message_id": "request-user",
        }
        receipt.update(changes)
        return receipt

    def request_document(self):
        return {
            "conversation_id": "conversation-one",
            "current_node": "later-answer",
            "mapping": {
                "request-user": {
                    "parent": "prior-answer",
                    "message": {"id": "request-user", "author": {"role": "user"}},
                },
                "original-answer": {
                    "parent": "request-user",
                    "message": {
                        "id": "original-answer", "author": {"role": "assistant"},
                        "status": "finished_successfully", "end_turn": True, "channel": "final",
                        "content": {"content_type": "text", "parts": ["original answer"]},
                    },
                },
                "later-user": {
                    "parent": "original-answer",
                    "message": {"id": "later-user", "author": {"role": "user"}},
                },
                "later-answer": {
                    "parent": "later-user",
                    "message": {
                        "id": "later-answer", "author": {"role": "assistant"},
                        "status": "finished_successfully", "end_turn": True, "channel": "final",
                        "content": {"content_type": "text", "parts": ["later answer"]},
                    },
                },
            },
        }

    def test_limited_bound_account_reads_and_archives_without_sending_a_new_turn(self):
        from services.account_service import AccountService
        from services.storage.json_storage import JSONStorageBackend

        with tempfile.TemporaryDirectory() as directory:
            accounts = AccountService(JSONStorageBackend(Path(directory) / "accounts.json"))
            accounts.add_account_items([{
                "access_token": "limited-token", "source_type": "web", "type": "Plus", "status": "限流",
            }])
            accounts.refresh_access_token = lambda token, **_kwargs: token
            with accounts._lock:
                binding = accounts._conversation_binding_for_token_locked("limited-token")
                identity = accounts._provider_account_identity_for_token_locked("limited-token")

            calls = []

            class FakeBackend:
                def __init__(self, *, access_token):
                    self.access_token = access_token
                    calls.append(("open", access_token))

                def _get_conversation(self, _conversation_id):
                    calls.append(("read", self.access_token))
                    return TextResultRecoveryTests().request_document()

                def set_conversation_archived(self, conversation_id, parent_message_id, archived):
                    calls.append(("archive", conversation_id, parent_message_id, archived))
                    return {"archived": archived}

                def close(self):
                    calls.append(("close", self.access_token))

            receipt = self.request_receipt(
                provider_binding_id=binding,
                provider_account_identity=identity,
            )
            service = ConversationBindingService()
            with (
                mock.patch("services.conversation_binding_service.account_service", accounts),
                mock.patch("services.conversation_binding_service.OpenAIBackendAPI", FakeBackend),
            ):
                result = service.read_text_request(receipt)
                archived = service.set_archived({
                    "provider_binding_id": binding,
                    "provider_account_identity": identity,
                    "client_conversation_id": "client-one",
                    "conversation_id": "conversation-one",
                    "parent_message_id": "prior-answer",
                }, True)
                with self.assertRaises(ConversationBindingError):
                    service.complete_text({
                        "provider_binding_id": binding,
                        "provider_account_identity": identity,
                        "client_conversation_id": "client-one",
                        "model": "auto",
                        "messages": [{"role": "user", "content": "must not send"}],
                    })

            self.assertEqual(result["status"], "succeeded")
            self.assertTrue(archived["archived"])
            self.assertIn(("read", "limited-token"), calls)
            self.assertIn(("archive", "conversation-one", "prior-answer", True), calls)
            # The explicit message path was rejected before it opened another
            # upstream client or emitted a replacement request.
            self.assertEqual(calls.count(("open", "limited-token")), 2)
            self.assertEqual(accounts.get_account("limited-token")["status"], "限流")

    def test_request_recovery_reads_completed_original_after_later_user_turn(self):
        backend = mock.Mock()
        backend._get_conversation.return_value = self.request_document()
        result = ConversationBindingService._read_text_request_result(backend, self.request_receipt())
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["content"], "original answer")
        self.assertEqual(result["parent_message_id"], "original-answer")
        self.assertNotEqual(result["parent_message_id"], "later-answer")

    def test_request_recovery_uses_persisted_request_parent_after_timeout_cursor_advance(self):
        backend = mock.Mock()
        backend._get_conversation.return_value = self.request_document()
        receipt = self.request_receipt(request_parent_message_id="prior-answer", parent_message_id="later-answer")
        result = ConversationBindingService._read_text_request_result(backend, receipt)
        self.assertEqual(result["content"], "original answer")
        self.assertEqual(receipt["parent_message_id"], "later-answer")

    def test_request_recovery_accepts_proven_submission_root_when_batched_parent_was_omitted(self):
        document = self.request_document()
        document["mapping"]["submission-root"] = {"parent": None, "message": None}
        document["mapping"]["request-user"]["parent"] = "submission-root"
        receipt = self.request_receipt(
            request_parent_message_id="intermediate-context",
            _submission_parent_message_id="submission-root",
        )
        backend = mock.Mock()
        result = ConversationBindingService._read_text_request_result(
            backend, receipt, document=document,
        )
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["content"], "original answer")
        self.assertEqual(result["parent_message_id"], "original-answer")

    def test_request_recovery_rejects_submission_root_without_omitted_batch_parent(self):
        document = self.request_document()
        document["mapping"]["submission-root"] = {"parent": None, "message": None}
        document["mapping"]["request-user"]["parent"] = "submission-root"
        receipt = self.request_receipt(
            request_parent_message_id="intermediate-context",
            _submission_parent_message_id="submission-root",
        )
        backend = mock.Mock()
        for candidate in ("intermediate-context", "submission-root"):
            altered = self.request_document()
            altered["mapping"].update(document["mapping"])
            if candidate == "intermediate-context":
                altered["mapping"][candidate] = {"parent": "submission-root", "message": None}
            else:
                altered["mapping"].pop(candidate)
            with self.subTest(candidate=candidate), self.assertRaises(ConversationBindingError) as mismatch:
                ConversationBindingService._read_text_request_result(
                    backend, receipt, document=altered,
                )
            self.assertEqual(mismatch.exception.recovery_reason, TextRecoveryReason.REQUEST_PARENT_MISMATCH.value)

    def test_request_recovery_keeps_divergent_sibling_finals_unknown(self):
        document = self.request_document()
        for node_id in ("later-user", "later-answer"):
            document["mapping"].pop(node_id)
        document["current_node"] = "answer-b"
        for node_id, text in (("answer-a", "branch A"), ("answer-b", "branch B")):
            document["mapping"][node_id] = {
                "parent": "request-user",
                "message": {
                    "id": node_id, "author": {"role": "assistant"},
                    "status": "finished_successfully", "end_turn": True, "channel": "final",
                    "content": {"content_type": "text", "parts": [text]},
                },
            }
        backend = mock.Mock()
        backend._get_conversation.return_value = document
        with self.assertRaises(ConversationBindingError) as captured:
            ConversationBindingService._read_text_request_result(backend, self.request_receipt())
        self.assertEqual(captured.exception.code, "CONVERSATION_BINDING_MISMATCH")
        self.assertEqual(captured.exception.recovery_reason, TextRecoveryReason.REQUEST_BRANCH_AMBIGUOUS.value)

    def test_request_recovery_rejects_missing_anchor_and_parent_drift(self):
        document = self.request_document()
        document["mapping"].pop("request-user")
        backend = mock.Mock()
        backend._get_conversation.return_value = document
        with self.assertRaises(ConversationBindingError) as missing:
            ConversationBindingService._read_text_request_result(backend, self.request_receipt())
        self.assertEqual(missing.exception.recovery_reason, TextRecoveryReason.REQUEST_MESSAGE_NOT_FOUND.value)

        document = self.request_document()
        backend._get_conversation.return_value = document
        with self.assertRaises(ConversationBindingError) as drift:
            ConversationBindingService._read_text_request_result(
                backend, self.request_receipt(parent_message_id="different-parent")
            )
        self.assertEqual(drift.exception.recovery_reason, TextRecoveryReason.REQUEST_PARENT_MISMATCH.value)

    def test_missing_request_anchor_waits_for_explicit_active_current_node(self):
        backend = mock.Mock()
        for role in ("assistant", "tool"):
            for status in ("in_progress", "running", "pending", "queued"):
                backend._get_conversation.return_value = {
                    "conversation_id": "conversation-one",
                    "current_node": "active-node",
                    "mapping": {
                        "active-node": {
                            "message": {
                                "id": "active-node",
                                "author": {"role": role},
                                "status": status,
                            },
                        },
                    },
                }

                result = ConversationBindingService._read_text_request_result(
                    backend, self.request_receipt(),
                )

                self.assertEqual(result["status"], "running")
                self.assertEqual(
                    result["recovery_reason"],
                    TextRecoveryReason.REQUEST_RESULT_INCOMPLETE.value,
                )

    def test_request_recovery_distinguishes_empty_branch_from_invalid_mapping(self):
        document = self.request_document()
        for node_id in ("original-answer", "later-user", "later-answer"):
            document["mapping"].pop(node_id)
        document["current_node"] = "request-user"
        backend = mock.Mock()
        backend._get_conversation.return_value = document

        empty = ConversationBindingService._read_text_request_result(
            backend, self.request_receipt(),
        )

        self.assertEqual(empty["status"], "unknown")
        self.assertEqual(
            empty["recovery_reason"], TextRecoveryReason.REQUEST_RESULT_NOT_FOUND.value,
        )

        for invalid_document in (
            {"conversation_id": "conversation-one"},
            {"conversation_id": "conversation-one", "mapping": []},
        ):
            backend._get_conversation.return_value = invalid_document
            with self.assertRaises(ConversationBindingError) as invalid:
                ConversationBindingService._read_text_request_result(
                    backend, self.request_receipt(),
                )
            self.assertEqual(
                invalid.exception.code, "CONVERSATION_BINDING_CONTRACT_INVALID",
            )
            self.assertNotEqual(
                invalid.exception.recovery_reason,
                TextRecoveryReason.REQUEST_MESSAGE_NOT_FOUND.value,
            )

    def test_request_recovery_account_and_conversation_mismatch_are_distinct(self):
        service = ConversationBindingService()
        receipt = self.request_receipt()
        with mock.patch(
            "services.conversation_binding_service.account_service.get_bound_account_identity",
            return_value="other-account",
        ):
            with self.assertRaises(ConversationBindingError) as account:
                service.read_text_request(receipt)
        self.assertEqual(account.exception.code, "CONVERSATION_BINDING_MISMATCH")
        self.assertEqual(account.exception.recovery_reason, TextRecoveryReason.ACCOUNT_IDENTITY_MISMATCH.value)

        backend = mock.Mock()
        document = self.request_document()
        document["conversation_id"] = "other-conversation"
        backend._get_conversation.return_value = document
        with (
            mock.patch(
                "services.conversation_binding_service.account_service.get_bound_account_identity",
                return_value="account-one",
            ),
            mock.patch(
                "services.conversation_binding_service.account_service.get_bound_text_access_token",
                return_value="synthetic-token",
            ),
            mock.patch(
                "services.conversation_binding_service.account_service.conversation_binding_lock",
                return_value=nullcontext(),
            ),
            mock.patch("services.conversation_binding_service.OpenAIBackendAPI", return_value=backend),
        ):
            with self.assertRaises(ConversationBindingError) as conversation:
                service.read_text_request(receipt)
        self.assertEqual(conversation.exception.code, "CONVERSATION_BINDING_MISMATCH")
        self.assertEqual(conversation.exception.recovery_reason, TextRecoveryReason.CONVERSATION_ID_MISMATCH.value)

    def test_request_recovery_distinguishes_running_from_terminal_empty_answer(self):
        for patch, expected_reason in (
            ({"status": "in_progress"}, TextRecoveryReason.REQUEST_RESULT_INCOMPLETE.value),
            (
                {"content": {"content_type": "text", "parts": []}},
                TextRecoveryReason.REQUEST_RESULT_TERMINAL_EMPTY.value,
            ),
            (
                {"content": {"content_type": "multimodal_text", "parts": [{"type": "image"}]}},
                TextRecoveryReason.REQUEST_RESULT_TERMINAL_EMPTY.value,
            ),
        ):
            document = self.request_document()
            document["mapping"].pop("later-user")
            document["mapping"].pop("later-answer")
            document["current_node"] = "original-answer"
            document["mapping"]["original-answer"]["message"].update(patch)
            backend = mock.Mock()
            backend._get_conversation.return_value = document
            result = ConversationBindingService._read_text_request_result(backend, self.request_receipt())
            self.assertEqual(
                result["status"],
                "running" if expected_reason == TextRecoveryReason.REQUEST_RESULT_INCOMPLETE.value else "unknown",
            )
            self.assertEqual(result["recovery_reason"], expected_reason)
            self.assertNotIn("content", result)

    def test_backend_tasks_strict_schema_rejects_missing_or_malformed_tasks(self):
        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend._headers = lambda *_args, **_kwargs: {}
        response = SimpleNamespace(
            status_code=200, text="{}", headers={}, json=lambda: {},
        )
        backend.session = SimpleNamespace(get=mock.Mock(return_value=response))

        self.assertEqual(backend._query_backend_tasks(conversation_id="chat"), [])
        with self.assertRaisesRegex(RuntimeError, "missing the tasks list"):
            backend._query_backend_tasks(conversation_id="chat", strict_schema=True)

        response.json = lambda: {"tasks": {"bad": "shape"}}
        self.assertEqual(backend._query_backend_tasks(conversation_id="chat"), [])
        with self.assertRaisesRegex(RuntimeError, "invalid tasks list"):
            backend._query_backend_tasks(conversation_id="chat", strict_schema=True)

        response.json = lambda: {"tasks": ["bad-task"]}
        with self.assertRaisesRegex(RuntimeError, "invalid task"):
            backend._query_backend_tasks(conversation_id="chat", strict_schema=True)

    def test_recent_conversations_strict_schema_rejects_unknown_or_malformed_lists(self):
        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend._headers = lambda *_args, **_kwargs: {}
        response = SimpleNamespace(
            status_code=200, text="{}", headers={}, json=lambda: {},
        )
        backend.session = SimpleNamespace(get=mock.Mock(return_value=response))

        self.assertEqual(backend._list_recent_conversations(), [])
        with self.assertRaisesRegex(RuntimeError, "missing the conversation list"):
            backend._list_recent_conversations(strict_schema=True)

        response.json = lambda: {"items": {"bad": "shape"}}
        self.assertEqual(backend._list_recent_conversations(), [])
        with self.assertRaisesRegex(RuntimeError, "invalid conversation list"):
            backend._list_recent_conversations(strict_schema=True)

        response.json = lambda: {"items": [{"title": "missing id"}]}
        with self.assertRaisesRegex(RuntimeError, "without an id"):
            backend._list_recent_conversations(strict_schema=True)

        response.json = lambda: {
            "items": [{"id": f"conversation-{index}"} for index in range(21)],
        }
        with self.assertRaisesRegex(RuntimeError, "exceeds the requested limit"):
            backend._list_recent_conversations(limit=20, strict_schema=True)

        response.json = lambda: {"items": []}
        backend._list_recent_conversations(limit=20, offset=40, strict_schema=True)
        self.assertIn("offset=40&limit=20", backend.session.get.call_args.args[0])

    def test_oversized_recent_conversation_response_reads_no_details(self):
        receipt = self.request_receipt()
        receipt.pop("conversation_id")
        backend = mock.Mock()
        backend._list_recent_conversations.side_effect = RuntimeError(
            "recent conversations response exceeds the requested limit",
        )

        with self.assertRaises(ConversationBindingError) as failed:
            ConversationBindingService._locate_text_request_conversation(
                backend, receipt,
            )

        self.assertEqual(failed.exception.code, "RECOVERY_READ_FAILED")
        self.assertRegex(str(failed.exception.__cause__), "requested limit")
        backend._get_conversation.assert_not_called()

    def test_missing_conversation_is_recovered_only_by_one_exact_request_node(self):
        service = ConversationBindingService()
        receipt = self.request_receipt()
        receipt.pop("conversation_id")
        receipt.pop("parent_message_id")
        backend = mock.Mock()
        backend._list_recent_conversations.return_value = [
            {"id": "unrelated-conversation"},
            {"id": "conversation-one"},
        ]
        unrelated = self.request_document()
        unrelated["conversation_id"] = "unrelated-conversation"
        unrelated["mapping"].pop("request-user")
        backend._get_conversation.side_effect = [
            unrelated, self.request_document(), self.request_document(),
        ]

        with (
            mock.patch(
                "services.conversation_binding_service.account_service.get_bound_account_identity",
                return_value="account-one",
            ),
            mock.patch(
                "services.conversation_binding_service.account_service.get_bound_text_access_token",
                return_value="original-paid-account-token",
            ) as get_token,
            mock.patch(
                "services.conversation_binding_service.account_service.conversation_binding_lock",
                return_value=nullcontext(),
            ),
            mock.patch("services.conversation_binding_service.OpenAIBackendAPI", return_value=backend) as backend_type,
        ):
            with self.assertRaises(ConversationBindingError) as incomplete:
                service.read_text_request(receipt)
            self.assertEqual(
                incomplete.exception.recovery_reason,
                TextRecoveryReason.REQUEST_CONVERSATION_SCAN_INCOMPLETE.value,
            )
            receipt[RECOVERY_CONVERSATION_SCAN_FIELD] = incomplete.exception.recovery_scan
            recovered = service.read_text_request(receipt)

        self.assertEqual(recovered["status"], "succeeded")
        self.assertEqual(recovered["content"], "original answer")
        self.assertEqual(recovered["conversation_id"], "conversation-one")
        self.assertEqual(recovered["parent_message_id"], "original-answer")
        self.assertEqual(get_token.call_count, 2)
        get_token.assert_called_with("binding-one", model="auto")
        self.assertEqual(backend_type.call_count, 2)
        for backend_call in backend_type.call_args_list:
            self.assertEqual(backend_call.kwargs["access_token"], get_token.return_value)
        backend._list_recent_conversations.assert_called_once_with(
            limit=ConversationBindingService.RECOVERY_RECENT_CONVERSATION_LIMIT,
            offset=0,
            timeout_secs=10.0,
            strict_schema=True,
        )
        self.assertEqual(backend._get_conversation.call_count, 3)
        for call in backend._get_conversation.call_args_list:
            self.assertGreater(call.kwargs["timeout_secs"], 0)
            self.assertLessEqual(
                call.kwargs["timeout_secs"],
                ConversationBindingService.RECOVERY_SCAN_TIMEOUT_SECONDS,
            )

    def test_missing_conversation_zero_match_is_unattributable_but_multiple_is_a_mismatch(self):
        receipt = self.request_receipt()
        receipt.pop("conversation_id")
        receipt.pop("parent_message_id")

        empty_backend = mock.Mock()
        empty_backend._list_recent_conversations.return_value = []
        with self.assertRaises(ConversationBindingError) as empty:
            ConversationBindingService._locate_text_request_conversation(
                empty_backend, receipt,
            )
        self.assertEqual(
            empty.exception.recovery_reason,
            TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value,
        )
        self.assertFalse(empty.exception.conversation_id)

        duplicate_backend = mock.Mock()
        duplicate_backend._list_recent_conversations.return_value = [
            {"id": "conversation-one"}, {"id": "conversation-two"},
        ]
        first = self.request_document()
        second = self.request_document()
        second["conversation_id"] = "conversation-two"
        duplicate_backend._get_conversation.side_effect = [first, second]
        with self.assertRaises(ConversationBindingError) as duplicate:
            ConversationBindingService._locate_text_request_conversation(
                duplicate_backend, receipt,
            )
        self.assertEqual(duplicate.exception.code, "CONVERSATION_BINDING_MISMATCH")
        self.assertFalse(duplicate.exception.recovery_reason)
        self.assertFalse(duplicate.exception.conversation_id)

    def test_missing_conversation_lookup_preserves_the_original_request_parent(self):
        receipt = self.request_receipt(
            conversation_id="",
            parent_message_id="",
            request_parent_message_id="different-parent",
        )
        backend = mock.Mock()
        backend._list_recent_conversations.return_value = [
            {"id": "conversation-one"},
        ]
        backend._get_conversation.return_value = self.request_document()

        with self.assertRaises(ConversationBindingError) as incomplete:
            ConversationBindingService._locate_text_request_conversation(
                backend, receipt,
            )
        receipt[RECOVERY_CONVERSATION_SCAN_FIELD] = incomplete.exception.recovery_scan
        with self.assertRaises(ConversationBindingError) as mismatch:
            ConversationBindingService._locate_text_request_conversation(
                backend, receipt,
            )

        self.assertEqual(
            mismatch.exception.recovery_reason,
            TextRecoveryReason.REQUEST_PARENT_MISMATCH.value,
        )
        self.assertEqual(mismatch.exception.conversation_id, "conversation-one")

    def test_missing_conversation_lookup_accepts_proven_submission_root(self):
        receipt = self.request_receipt(
            conversation_id="", parent_message_id="",
            request_parent_message_id="intermediate-context",
            _submission_parent_message_id="submission-root",
        )
        document = self.request_document()
        document["mapping"]["submission-root"] = {"parent": None, "message": None}
        document["mapping"]["request-user"]["parent"] = "submission-root"
        backend = mock.Mock()
        backend._list_recent_conversations.return_value = [{"id": "conversation-one"}]
        backend._get_conversation.return_value = document
        with self.assertRaises(ConversationBindingError) as incomplete:
            ConversationBindingService._locate_text_request_conversation(backend, receipt)
        receipt[RECOVERY_CONVERSATION_SCAN_FIELD] = incomplete.exception.recovery_scan
        located, located_document = ConversationBindingService._locate_text_request_conversation(
            backend, receipt,
        )
        result = ConversationBindingService._read_text_request_result(
            backend, located, document=located_document,
        )
        self.assertEqual(located["request_parent_message_id"], "submission-root")
        self.assertEqual(result["content"], "original answer")

    def test_missing_conversation_root_request_keeps_an_explicit_empty_parent(self):
        receipt = self.request_receipt()
        receipt.pop("conversation_id")
        receipt.pop("parent_message_id")
        document = self.request_document()
        document["mapping"]["request-user"]["parent"] = None
        backend = mock.Mock()
        backend._list_recent_conversations.return_value = [
            {"id": "conversation-one"},
        ]
        backend._get_conversation.return_value = document

        with self.assertRaises(ConversationBindingError) as incomplete:
            ConversationBindingService._locate_text_request_conversation(backend, receipt)
        receipt[RECOVERY_CONVERSATION_SCAN_FIELD] = incomplete.exception.recovery_scan
        located, located_document = ConversationBindingService._locate_text_request_conversation(
            backend, receipt,
        )
        result = ConversationBindingService._read_text_request_result(
            backend, located, document=located_document,
        )

        self.assertIn("request_parent_message_id", located)
        self.assertEqual(located["request_parent_message_id"], "")
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["content"], "original answer")
        self.assertEqual(backend._get_conversation.call_count, 2)

    def test_missing_conversation_scan_has_one_total_time_budget(self):
        receipt = self.request_receipt()
        receipt.pop("conversation_id")
        backend = mock.Mock()
        backend._list_recent_conversations.return_value = [
            {"id": "conversation-one"},
        ]

        with (
            mock.patch(
                "services.conversation_binding_service.time.monotonic",
                side_effect=[100.0, 101.0, 121.0],
            ),
            self.assertRaises(ConversationBindingError) as incomplete,
        ):
            ConversationBindingService._locate_text_request_conversation(
                backend, receipt,
            )

        self.assertEqual(
            incomplete.exception.recovery_reason,
            TextRecoveryReason.REQUEST_CONVERSATION_SCAN_INCOMPLETE.value,
        )
        self.assertEqual(incomplete.exception.recovery_scan["next_index"], 0)
        backend._list_recent_conversations.assert_called_once_with(
            limit=ConversationBindingService.RECOVERY_RECENT_CONVERSATION_LIMIT,
            offset=0,
            timeout_secs=10.0,
            strict_schema=True,
        )
        backend._get_conversation.assert_not_called()

    def test_missing_conversation_scan_accepts_real_iso_update_time_and_stops_at_dispatch(self):
        receipt = self.request_receipt(
            created_at=ConversationBindingService._timestamp("2026-09-17T15:15:38.243Z"),
        )
        receipt.pop("conversation_id")
        receipt.pop("parent_message_id")
        backend = mock.Mock()
        backend._list_recent_conversations.return_value = [
            {
                "id": f"conversation-{index}",
                "update_time": "2026-09-17T13:30:26.273935Z",
            }
            for index in range(20)
        ]
        backend._get_conversation.side_effect = [
            {"conversation_id": f"conversation-{index}", "mapping": {}}
            for index in range(20)
        ]

        with self.assertRaises(ConversationBindingError) as complete:
            ConversationBindingService._locate_text_request_conversation(backend, receipt)

        self.assertEqual(
            complete.exception.recovery_reason,
            TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value,
        )
        backend._list_recent_conversations.assert_called_once_with(
            limit=20, offset=0, timeout_secs=10.0, strict_schema=True,
        )
        self.assertEqual(backend._get_conversation.call_count, 20)
        self.assertEqual(
            ConversationBindingService._conversation_update_time(
                {"update_time": "2026-09-17T13:30:26.273935Z"}
            ),
            1789651826.273935,
        )
        self.assertEqual(
            ConversationBindingService._conversation_update_time({"update_time": "123.5"}),
            123.5,
        )
        self.assertIsNone(
            ConversationBindingService._conversation_update_time(
                {"update_time": "2026-09-17T13:30:26"}
            )
        )

    def test_missing_timestamps_continue_beyond_each_hundred_candidate_window(self):
        receipt = self.request_receipt()
        receipt.pop("conversation_id")
        receipt.pop("parent_message_id")
        backend = mock.Mock()

        def list_page(*, offset, **_kwargs):
            if offset >= 120:
                return []
            return [
                {"id": f"conversation-{offset + index}"}
                for index in range(20)
            ]

        backend._list_recent_conversations.side_effect = list_page
        backend._get_conversation.side_effect = lambda conversation_id, **_kwargs: {
            "conversation_id": conversation_id,
            "mapping": {},
        }

        with self.assertRaises(ConversationBindingError) as first:
            ConversationBindingService._locate_text_request_conversation(backend, receipt)
        self.assertEqual(
            first.exception.recovery_reason,
            TextRecoveryReason.REQUEST_CONVERSATION_SCAN_INCOMPLETE.value,
        )
        self.assertEqual(first.exception.recovery_scan["next_offset"], 100)
        self.assertEqual(first.exception.recovery_scan["conversation_ids"], [])
        receipt[RECOVERY_CONVERSATION_SCAN_FIELD] = first.exception.recovery_scan

        with self.assertRaises(ConversationBindingError) as complete:
            ConversationBindingService._locate_text_request_conversation(backend, receipt)

        self.assertEqual(
            complete.exception.recovery_reason,
            TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value,
        )
        self.assertEqual(
            [call.kwargs["offset"] for call in backend._list_recent_conversations.call_args_list],
            [0, 20, 40, 60, 80, 100, 120],
        )
        self.assertEqual(backend._get_conversation.call_count, 120)

    def test_missing_conversation_lookup_failures_do_not_become_no_result_evidence(self):
        receipt = self.request_receipt()
        receipt.pop("conversation_id")
        backend = mock.Mock()
        backend._list_recent_conversations.side_effect = RuntimeError(
            "recent conversation schema invalid",
        )

        with self.assertRaises(ConversationBindingError) as failed:
            ConversationBindingService._locate_text_request_conversation(
                backend, receipt,
            )
        self.assertEqual(failed.exception.code, "RECOVERY_READ_FAILED")
        self.assertFalse(failed.exception.recovery_reason)

    def test_missing_conversation_scan_retries_failed_item_without_losing_progress(self):
        receipt = self.request_receipt()
        receipt.pop("conversation_id")
        receipt.pop("parent_message_id")
        backend = mock.Mock()
        backend._list_recent_conversations.return_value = [
            {"id": "conversation-one"},
            {"id": "conversation-two"},
            {"id": "conversation-three"},
        ]
        unrelated = self.request_document()
        unrelated["mapping"].pop("request-user")
        first = {**unrelated, "conversation_id": "conversation-one"}
        second = {**unrelated, "conversation_id": "conversation-two"}
        third = {**unrelated, "conversation_id": "conversation-three"}
        backend._get_conversation.side_effect = [
            first,
            UpstreamHTTPError("/backend-api/conversation/conversation-two", 429, {}),
            second,
            third,
        ]

        with self.assertRaises(ConversationBindingError) as failed:
            ConversationBindingService._locate_text_request_conversation(backend, receipt)
        self.assertEqual(failed.exception.code, "RECOVERY_READ_FAILED")
        self.assertEqual(failed.exception.recovery_scan["next_index"], 1)
        receipt[RECOVERY_CONVERSATION_SCAN_FIELD] = failed.exception.recovery_scan

        with self.assertRaises(ConversationBindingError) as complete:
            ConversationBindingService._locate_text_request_conversation(backend, receipt)
        self.assertEqual(
            complete.exception.recovery_reason,
            TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value,
        )
        self.assertEqual(
            [call.args[0] for call in backend._get_conversation.call_args_list],
            ["conversation-one", "conversation-two", "conversation-two", "conversation-three"],
        )
        backend._list_recent_conversations.assert_called_once()

    def test_missing_conversation_scan_keeps_progress_on_detail_schema_failure(self):
        receipt = self.request_receipt()
        receipt.pop("conversation_id")
        receipt.pop("parent_message_id")
        backend = mock.Mock()
        backend._list_recent_conversations.return_value = [
            {"id": "conversation-one"}, {"id": "conversation-two"},
        ]
        unrelated = self.request_document()
        unrelated["conversation_id"] = "conversation-one"
        unrelated["mapping"].pop("request-user")
        backend._get_conversation.side_effect = [
            unrelated,
            {"conversation_id": "conversation-two", "mapping": "invalid"},
        ]

        with self.assertRaises(ConversationBindingError) as failed:
            ConversationBindingService._locate_text_request_conversation(backend, receipt)

        self.assertEqual(failed.exception.code, "CONVERSATION_BINDING_CONTRACT_INVALID")
        self.assertEqual(failed.exception.recovery_scan["next_index"], 1)

    def test_missing_conversation_scan_does_not_treat_detail_404_as_absence(self):
        receipt = self.request_receipt()
        receipt.pop("conversation_id")
        receipt.pop("parent_message_id")
        backend = mock.Mock()
        backend._list_recent_conversations.return_value = [
            {"id": "deleted-conversation"}, {"id": "unrelated-conversation"},
        ]
        unrelated = self.request_document()
        unrelated["conversation_id"] = "unrelated-conversation"
        unrelated["mapping"].pop("request-user")
        backend._get_conversation.side_effect = [
            UpstreamHTTPError("/backend-api/conversation/deleted-conversation", 404, {}),
            unrelated,
        ]

        with self.assertRaises(ConversationBindingError) as complete:
            ConversationBindingService._locate_text_request_conversation(backend, receipt)

        self.assertEqual(complete.exception.code, "RECOVERY_READ_FAILED")
        self.assertFalse(complete.exception.recovery_reason)
        # A missing candidate remains unresolved, but cannot prevent reading
        # later candidates. The checked count excludes the unread 404.
        self.assertEqual(complete.exception.recovery_scan["next_index"], 1)
        self.assertEqual(backend._get_conversation.call_count, 2)
        self.assertEqual(complete.exception.recovery_scan["conversation_ids"],
                         ["unrelated-conversation", "deleted-conversation"])
        failure = complete.exception.recovery_scan["failed_reads"]["deleted-conversation"]
        self.assertEqual(failure["error"]["http_status"], 404)
        self.assertEqual(failure["attempts"], 1)
        self.assertEqual(complete.exception.recovery_scan["matches"], [])

    def test_missing_conversation_scan_identity_change_starts_a_fresh_snapshot(self):
        receipt = self.request_receipt()
        receipt.pop("conversation_id")
        receipt.pop("parent_message_id")
        stale_identity = ConversationBindingService._recovery_scan_identity(receipt)
        receipt[RECOVERY_CONVERSATION_SCAN_FIELD] = {
            "identity": dict(stale_identity),
            "conversation_ids": ["stale-conversation"],
            "next_index": 1,
            "matches": [],
        }
        receipt["provider_binding_id"] = "different-binding"
        backend = mock.Mock()
        backend._list_recent_conversations.return_value = []

        with self.assertRaises(ConversationBindingError) as complete:
            ConversationBindingService._locate_text_request_conversation(backend, receipt)

        self.assertEqual(
            complete.exception.recovery_reason,
            TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value,
        )
        backend._list_recent_conversations.assert_called_once()
        backend._get_conversation.assert_not_called()

    def test_missing_conversation_exact_reread_rejects_a_changed_request_parent(self):
        receipt = self.request_receipt()
        receipt.pop("conversation_id")
        receipt.pop("parent_message_id")
        first = self.request_document()
        changed = self.request_document()
        changed["mapping"]["request-user"]["parent"] = "changed-parent"
        backend = mock.Mock()
        backend._list_recent_conversations.return_value = [
            {"id": "conversation-one"},
        ]
        backend._get_conversation.side_effect = [first, changed]

        with self.assertRaises(ConversationBindingError) as incomplete:
            ConversationBindingService._locate_text_request_conversation(backend, receipt)
        receipt[RECOVERY_CONVERSATION_SCAN_FIELD] = incomplete.exception.recovery_scan
        with self.assertRaises(ConversationBindingError) as mismatch:
            ConversationBindingService._locate_text_request_conversation(backend, receipt)

        self.assertEqual(
            mismatch.exception.recovery_reason,
            TextRecoveryReason.REQUEST_PARENT_MISMATCH.value,
        )
        self.assertEqual(mismatch.exception.parent_message_id, "changed-parent")

    def test_request_recovery_distinguishes_missing_conversation_from_other_read_failures(self):
        service = ConversationBindingService()
        receipt = self.request_receipt()
        backend = mock.Mock()
        backend._get_conversation.side_effect = UpstreamHTTPError(
            "/backend-api/conversation/chat", 404, {},
        )
        with (
            mock.patch(
                "services.conversation_binding_service.account_service.get_bound_account_identity",
                return_value="account-one",
            ),
            mock.patch(
                "services.conversation_binding_service.account_service.get_bound_text_access_token",
                return_value="synthetic-token",
            ),
            mock.patch(
                "services.conversation_binding_service.account_service.conversation_binding_lock",
                return_value=nullcontext(),
            ),
            mock.patch("services.conversation_binding_service.OpenAIBackendAPI", return_value=backend),
        ):
            with self.assertRaises(ConversationBindingError) as missing:
                service.read_text_request(receipt)

        self.assertEqual(missing.exception.code, "CONVERSATION_OUTCOME_UNKNOWN")
        self.assertEqual(
            missing.exception.recovery_reason,
            TextRecoveryReason.CONVERSATION_NOT_FOUND.value,
        )

    def test_read_only_completed_answer_and_scope_checks(self):
        backend = mock.Mock()
        backend._get_conversation.return_value = self.document()
        result = ConversationBindingService._read_text_result(backend, self.cursor)
        self.assertEqual(result["content"], '{"name_ru":"Набор"}')
        self.assertEqual(result["parent_message_id"], "answer-one")
        backend._get_conversation.assert_called_once_with("conversation-one")
        backend.stream_conversation.assert_not_called()
        for mutate in ("foreign", "branch", "new-user"):
            document = self.document()
            if mutate == "foreign": document["conversation_id"] = "other"
            elif mutate == "branch": document["mapping"]["answer-one"]["parent"] = "other"
            else: document["mapping"]["answer-one"]["message"]["author"]["role"] = "user"
            backend._get_conversation.return_value = document
            with self.assertRaises(ConversationBindingError):
                ConversationBindingService._read_text_result(backend, self.cursor)

    def test_partial_analysis_and_unfinished_text_are_not_success(self):
        for patch in ({"channel": "analysis"}, {"end_turn": False}, {"status": "in_progress"}, {"content": {"content_type": "text", "parts": []}}):
            document = self.document()
            document["mapping"]["answer-one"]["message"].update(patch)
            backend = mock.Mock()
            backend._get_conversation.return_value = document
            result = ConversationBindingService._read_text_result(backend, self.cursor)
            self.assertEqual(result["status"], "running")
            self.assertNotIn("content", result)

    def test_stream_timeout_reads_saved_answer_without_a_second_generation(self):
        def timed_out(*_args, **_kwargs):
            yield {"type": "conversation.event", "conversation_id": "conversation-one"}
            raise TimeoutError("stream timeout")
        backend = mock.Mock()
        backend.get_conversation_parent_message_id.return_value = "user-one"
        backend._get_conversation.return_value = self.document()
        with (
            mock.patch("services.conversation_binding_service.account_service.create_conversation_binding", return_value=("binding-one", "account-one", "synthetic-token")),
            mock.patch("services.conversation_binding_service.account_service.release_image_slot"),
            mock.patch("services.conversation_binding_service.account_service.get_bound_text_access_token", return_value="synthetic-token"),
            mock.patch("services.conversation_binding_service.account_service.conversation_binding_lock", return_value=nullcontext()),
            mock.patch("services.conversation_binding_service.OpenAIBackendAPI", return_value=backend),
            mock.patch("services.conversation_binding_service.conversation_events", side_effect=timed_out) as generation,
        ):
            result = ConversationBindingService().complete_text({"client_conversation_id": "client-one", "messages": [{"role": "user", "content": "copy"}]})
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(generation.call_count, 1)
        backend._get_conversation.assert_called_once_with("conversation-one")

    def test_direct_read_cooldown_exits_without_lingering_send_or_clearing_cooldown(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        import time
        from services.account_request_pacing import AccountRequestClock, AccountRequestDeadlineExceeded
        from services.openai_backend_api import OpenAIBackendAPI

        with TemporaryDirectory() as directory:
            clock = AccountRequestClock("fixture", Path(directory) / "clock.json")
            clock.next_conversation_read = time.monotonic() + 900
            clock.conversation_read_rate_failures = 6
            clock._save()
            raw = mock.Mock(side_effect=AssertionError("cooling read must not reach transport"))
            backend = object.__new__(OpenAIBackendAPI)
            backend.base_url = "https://fixture.invalid"
            backend._headers = lambda path, headers: headers
            backend.session = SimpleNamespace(
                get=lambda url, **kw: clock.request(raw, "GET", url, **kw), close=lambda: None,
            )
            before = clock.next_conversation_read
            with (
                mock.patch("services.conversation_binding_service.account_service.get_bound_account_identity", return_value="account-one"),
                mock.patch("services.conversation_binding_service.account_service.get_bound_text_access_token", return_value="synthetic-token"),
                mock.patch("services.conversation_binding_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.conversation_binding_service.OpenAIBackendAPI", return_value=backend),
            ):
                started = time.monotonic()
                with self.assertRaises(AccountRequestDeadlineExceeded):
                    ConversationBindingService().read_text(self.cursor)
                self.assertLess(time.monotonic() - started, 1)
            raw.assert_not_called()
            self.assertEqual(clock.ordinary_read_queue, [])
            self.assertFalse(clock.lock.locked())
            self.assertAlmostEqual(clock.next_conversation_read, before, delta=0.001)
            self.assertEqual(clock.conversation_read_rate_failures, 6)

    def test_admin_cursor_route_reads_the_same_bound_account(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.testclient import TestClient
        from api.ai import create_router
        app = FastAPI()
        app.include_router(create_router())
        backend = mock.Mock()
        backend._get_conversation.return_value = self.document()
        with (
            mock.patch("api.ai.require_identity", return_value={"id": "admin", "role": "admin"}) as identity,
            mock.patch("services.conversation_binding_service.account_service.get_bound_account_identity", return_value="account-one"),
            mock.patch("services.conversation_binding_service.account_service.get_bound_text_access_token", return_value="synthetic-token"),
            mock.patch("services.conversation_binding_service.account_service.conversation_binding_lock", return_value=nullcontext()),
            mock.patch("services.conversation_binding_service.OpenAIBackendAPI", return_value=backend),
        ):
            with TestClient(app) as client:
                result = client.get("/api/conversation-bindings/text", params=self.cursor, headers={"Authorization": "synthetic"})
                self.assertEqual(result.status_code, 200, result.text)
                self.assertEqual(result.json()["status"], "succeeded")
                identity.assert_called_with("synthetic")
                wrong = client.get("/api/conversation-bindings/text", params={**self.cursor, "provider_account_identity": "other"})
                self.assertEqual(wrong.status_code, 409)
                identity.side_effect = HTTPException(status_code=401)
                self.assertEqual(client.get("/api/conversation-bindings/text", params=self.cursor).status_code, 401)
        backend._get_conversation.assert_called_once_with("conversation-one", deadline_monotonic=mock.ANY,
                                                         connect_timeout_secs=10.0)

    def test_direct_read_reconnects_once_with_same_cursor_and_deadline(self):
        from curl_cffi.requests.exceptions import SSLError
        from curl_cffi import CurlInfo
        backend = mock.Mock()
        response = SimpleNamespace(status_code=0, infos={CurlInfo.NUM_CONNECTS:1,
            CurlInfo.APPCONNECT_TIME:0., CurlInfo.SIZE_DOWNLOAD_T:0})
        backend._get_conversation.side_effect = [SSLError("private TLS detail", code=35, response=response), self.document()]
        with (
            mock.patch("services.conversation_binding_service.account_service.get_bound_account_identity", return_value="account-one"),
            mock.patch("services.conversation_binding_service.account_service.get_bound_text_access_token", return_value="synthetic-token"),
            mock.patch("services.conversation_binding_service.account_service.conversation_binding_lock", return_value=nullcontext()),
            mock.patch("services.conversation_binding_service.OpenAIBackendAPI", return_value=backend),
        ):
            result = ConversationBindingService().read_text(self.cursor)
        self.assertEqual(result["content"], '{"name_ru":"Набор"}')
        self.assertEqual(backend._get_conversation.call_count, 2)
        first, second = backend._get_conversation.call_args_list
        self.assertEqual(first.args, second.args)
        self.assertEqual(first.kwargs, {k: v for k, v in second.kwargs.items() if k != "minimum_budget_secs"})
        self.assertEqual(second.kwargs["minimum_budget_secs"], 10.0)
        backend.stream_conversation.assert_not_called()
        backend.close.assert_called_once()

    def test_direct_read_retry_rechecks_connection_window_after_pacing(self):
        from curl_cffi import CurlInfo
        from curl_cffi.requests.exceptions import SSLError
        import services.account_request_pacing as pacing
        from services.openai_backend_api import OpenAIBackendAPI

        for initial_wait in (0., 29.):
            with self.subTest(initial_wait=initial_wait), tempfile.TemporaryDirectory() as directory:
                now = [100.]
                fake_time = SimpleNamespace(monotonic=lambda: now[0], time=lambda: 1700000000 + now[0],
                                            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
                error = SSLError("private first failure", code=35, response=SimpleNamespace(status_code=0,
                    infos={CurlInfo.NUM_CONNECTS: 1, CurlInfo.APPCONNECT_TIME: 0., CurlInfo.SIZE_DOWNLOAD_T: 0}))
                sends = []
                response = SimpleNamespace(status_code=200, headers={}, json=self.document, close=lambda: None)
                def raw(method, url, **kwargs):
                    self.assertNotIn("_account_request_minimum_budget_secs", kwargs)
                    sends.append((now[0], kwargs["timeout"]))
                    if len(sends) == 1:
                        now[0] += 10
                        raise error
                    return response
                with mock.patch.object(pacing, "time", fake_time), \
                     mock.patch.object(pacing, "config", SimpleNamespace(account_request_interval_secs=0,
                                                                       account_conversation_read_interval_secs=30)):
                    clock = pacing.AccountRequestClock("fixture", Path(directory)/"clock.json")
                    clock.next_conversation_read = now[0] + initial_wait
                    clock._save()
                    backend = object.__new__(OpenAIBackendAPI)
                    backend.base_url = "https://fixture.invalid"
                    backend._headers = lambda path, headers: headers
                    closed = mock.Mock()
                    backend.session = SimpleNamespace(
                        get=lambda url, **kwargs: clock.request(raw, "GET", url, **kwargs), close=closed)
                    with (
                        mock.patch("services.conversation_binding_service.time", fake_time),
                        mock.patch("services.conversation_binding_service.account_service.get_bound_account_identity", return_value="account-one"),
                        mock.patch("services.conversation_binding_service.account_service.get_bound_text_access_token", return_value="synthetic-token"),
                        mock.patch("services.conversation_binding_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                        mock.patch("services.conversation_binding_service.OpenAIBackendAPI", return_value=backend),
                    ):
                        if initial_wait:
                            with self.assertRaises(SSLError) as caught:
                                ConversationBindingService().read_text(self.cursor)
                            self.assertIs(caught.exception, error)
                            self.assertEqual(len(sends), 1)
                            self.assertEqual(clock.next_conversation_read, 159.)
                        else:
                            result = ConversationBindingService().read_text(self.cursor)
                            self.assertEqual(result["status"], "succeeded")
                            self.assertEqual([at for at, _ in sends], [100., 130.])
                        self.assertLess(now[0], 160.)
                        self.assertTrue(all(at + budget <= 160. for at, budget in sends))
                        self.assertEqual(clock.ordinary_read_queue, [])
                        self.assertFalse(clock.lock.locked())
                        self.assertEqual(clock.rate_failures, 0)
                        self.assertEqual(clock.conversation_read_rate_failures, 0)
                        closed.assert_called_once()

    def test_direct_read_preserves_first_error_when_retry_window_expires(self):
        from curl_cffi import CurlInfo
        from curl_cffi.requests.exceptions import SSLError
        from services.account_request_pacing import AccountReadRetryBudgetInsufficient
        first = SSLError("private first failure", code=35, response=SimpleNamespace(status_code=0,
            infos={CurlInfo.NUM_CONNECTS: 1, CurlInfo.APPCONNECT_TIME: 0., CurlInfo.SIZE_DOWNLOAD_T: 0}))
        backend = mock.Mock()
        backend._get_conversation.side_effect = [first, AccountReadRetryBudgetInsufficient()]
        with (
            mock.patch("services.conversation_binding_service.account_service.get_bound_account_identity", return_value="account-one"),
            mock.patch("services.conversation_binding_service.account_service.get_bound_text_access_token", return_value="synthetic-token"),
            mock.patch("services.conversation_binding_service.account_service.conversation_binding_lock", return_value=nullcontext()),
            mock.patch("services.conversation_binding_service.OpenAIBackendAPI", return_value=backend),
        ):
            with self.assertRaises(SSLError) as caught:
                ConversationBindingService().read_text(self.cursor)
        self.assertIs(caught.exception, first)
        self.assertEqual(backend._get_conversation.call_count, 2)
        backend.stream_conversation.assert_not_called()
        backend.close.assert_called_once()

    def test_direct_read_does_not_retry_certificate_http_or_ambiguous_timeouts(self):
        from curl_cffi import CurlInfo
        from curl_cffi.requests.exceptions import RequestException
        for code, status, infos, expected in (
            (35, 0, {}, False), (60, 0, {}, False), (28, 0, {}, False),
            (35, 401, {}, False), (35, 403, {}, False), (35, 429, {}, False),
            (35, 0, {CurlInfo.HTTP_CONNECTCODE: 407}, False),
            (35, 0, {CurlInfo.NUM_CONNECTS: 1, CurlInfo.APPCONNECT_TIME: 0., CurlInfo.SIZE_DOWNLOAD_T: 0}, True),
            (35, 0, {CurlInfo.NUM_CONNECTS: 0, CurlInfo.APPCONNECT_TIME: 0., CurlInfo.SIZE_DOWNLOAD_T: 0}, False),
            (35, 0, {CurlInfo.NUM_CONNECTS: 1, CurlInfo.APPCONNECT_TIME: 0., CurlInfo.SIZE_DOWNLOAD_T: 5}, False),
            (28, 0, {CurlInfo.NUM_CONNECTS: 1, CurlInfo.APPCONNECT_TIME: 0., CurlInfo.SIZE_DOWNLOAD_T: 0}, True),
            (28, 0, {CurlInfo.NUM_CONNECTS: 0, CurlInfo.APPCONNECT_TIME: 0., CurlInfo.SIZE_DOWNLOAD_T: 0}, False),
            (28, 200, {CurlInfo.NUM_CONNECTS: 1, CurlInfo.APPCONNECT_TIME: 0., CurlInfo.SIZE_DOWNLOAD_T: 5}, False),
            (28, 0, {CurlInfo.NUM_CONNECTS: 1, CurlInfo.APPCONNECT_TIME: False, CurlInfo.SIZE_DOWNLOAD_T: 0}, False),
        ):
            with self.subTest(code=code, status=status, expected=expected):
                error = RequestException("private", code=code, response=SimpleNamespace(status_code=status, infos=infos))
                self.assertEqual(ConversationBindingService._retryable_direct_read_connection(error), expected)

    def test_direct_read_retry_is_bounded_and_never_outlives_original_budget(self):
        from curl_cffi.requests.exceptions import SSLError
        from curl_cffi import CurlInfo
        for expired in (False, True):
            with self.subTest(expired=expired):
                now = [100.]
                error = SSLError("private", code=35, response=SimpleNamespace(status_code=0,
                    infos={CurlInfo.NUM_CONNECTS:1, CurlInfo.APPCONNECT_TIME:0., CurlInfo.SIZE_DOWNLOAD_T:0}))
                backend = mock.Mock()
                def fail(*a, **kw):
                    if expired: now[0] = 161.
                    raise error
                backend._get_conversation.side_effect = fail
                with (
                    mock.patch("services.conversation_binding_service.time", SimpleNamespace(monotonic=lambda:now[0])),
                    mock.patch("services.conversation_binding_service.account_service.get_bound_account_identity", return_value="account-one"),
                    mock.patch("services.conversation_binding_service.account_service.get_bound_text_access_token", return_value="synthetic-token"),
                    mock.patch("services.conversation_binding_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                    mock.patch("services.conversation_binding_service.OpenAIBackendAPI", return_value=backend),
                ):
                    with self.assertRaises(SSLError) as caught:
                        ConversationBindingService().read_text(self.cursor)
                self.assertIs(caught.exception, error)
                self.assertEqual(backend._get_conversation.call_count, 1 if expired else 2)
                backend.close.assert_called_once()

    def test_direct_read_retry_respects_cooldown_arriving_after_first_connection(self):
        from curl_cffi import CurlInfo
        from curl_cffi.requests.exceptions import SSLError
        from services.account_request_pacing import AccountRequestClock, AccountRequestDeadlineExceeded
        with tempfile.TemporaryDirectory() as directory:
            clock = AccountRequestClock("fixture", Path(directory)/"clock.json")
            def fail_connection(*args, **kwargs):
                with clock.lock:
                    clock.limited(900, retry_after_present=True)
                raise SSLError("private", code=35, response=SimpleNamespace(status_code=0,
                    infos={CurlInfo.NUM_CONNECTS:1, CurlInfo.APPCONNECT_TIME:0., CurlInfo.SIZE_DOWNLOAD_T:0}))
            raw = mock.Mock(side_effect=fail_connection)
            backend = object.__new__(OpenAIBackendAPI)
            backend.base_url = "https://fixture.invalid"
            backend._headers = lambda path, headers: headers
            backend.session = SimpleNamespace(
                get=lambda url, **kw: clock.request(raw, "GET", url, **kw), close=lambda: None)
            with (
                mock.patch("services.conversation_binding_service.account_service.get_bound_account_identity", return_value="account-one"),
                mock.patch("services.conversation_binding_service.account_service.get_bound_text_access_token", return_value="synthetic-token"),
                mock.patch("services.conversation_binding_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.conversation_binding_service.OpenAIBackendAPI", return_value=backend),
            ):
                started = time.monotonic()
                with self.assertRaises(AccountRequestDeadlineExceeded):
                    ConversationBindingService().read_text(self.cursor)
                self.assertLess(time.monotonic()-started, 1)
            raw.assert_called_once()
            self.assertEqual(clock.ordinary_read_queue, [])
            self.assertFalse(clock.lock.locked())
            self.assertGreater(clock.cooldown_until-time.monotonic(), 899)


if __name__ == "__main__":
    unittest.main()

class ProductConversationArchiveTests(unittest.TestCase):
    def backend(self, documents):
        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.com"
        backend._get_conversation = mock.Mock(side_effect=documents)
        backend._headers = mock.Mock(return_value={})
        backend.session = mock.Mock()
        backend.session.patch.return_value.status_code = 200
        return backend

    def test_archive_preserves_chat_and_checks_authoritative_readback(self):
        backend = self.backend([{"mapping": {"original": {}}, "current_node": "original", "is_archived": False}, {"current_node": "original", "is_archived": True}])
        result = backend.archive_conversation("chat-a", "original")
        self.assertTrue(result["archived"])
        self.assertEqual(backend.session.patch.call_args.kwargs["json"], {"is_archived": True})
        self.assertEqual(backend._get_conversation.call_count, 2)

    def test_recovery_reads_already_archived_chat_without_repeating_patch(self):
        backend = self.backend([{"mapping": {"original": {}}, "current_node": "original", "is_archived": True}])
        self.assertTrue(backend.archive_conversation("chat-a", "original")["archived"])
        backend.session.patch.assert_not_called()

    def test_review_reversal_restores_same_archived_chat_with_readback(self):
        backend = self.backend([{"mapping": {"original": {}}, "current_node": "original", "is_archived": True}, {"current_node": "original", "is_archived": False}])
        result = backend.set_conversation_archived("chat-a", "original", False)
        self.assertFalse(result["archived"])
        self.assertEqual(backend.session.patch.call_args.kwargs["json"], {"is_archived": False})

    def test_missing_original_turn_never_archives_a_different_chat(self):
        backend = self.backend([{"mapping": {"other": {}}}])
        with self.assertRaises(RuntimeError):
            backend.archive_conversation("chat-a", "original")
        backend.session.patch.assert_not_called()

    def test_current_node_drift_blocks_patch_and_already_archived_shortcut(self):
        for archived in (False, True):
            with self.subTest(archived=archived):
                backend = self.backend([{"mapping": {"original": {}, "newer": {}},
                    "current_node": "newer", "is_archived": archived}])
                with self.assertRaisesRegex(RuntimeError, "cursor changed"):
                    backend.set_conversation_archived("chat-a", "original", archived)
                backend.session.patch.assert_not_called()

    def test_current_node_drift_after_patch_refuses_success(self):
        backend = self.backend([{"mapping": {"original": {}}, "current_node": "original", "is_archived": True},
            {"mapping": {"original": {}, "newer": {}}, "current_node": "newer", "is_archived": False}])
        with self.assertRaisesRegex(RuntimeError, "cursor changed"):
            backend.set_conversation_archived("chat-a", "original", False)
        backend.session.patch.assert_called_once()

    def public_archive(self, backend, archived=True, account_identity="account"):
        body = {"provider_binding_id": "binding", "provider_account_identity": "account",
                "client_conversation_id": "work", "conversation_id": "chat-a",
                "parent_message_id": "original", "_public_session_ref": "session"}
        backend.close = mock.Mock()
        accounts = mock.Mock()
        accounts.get_bound_account_identity.return_value = account_identity
        accounts.get_bound_text_access_token.return_value = "test-token"
        accounts.conversation_binding_lock.return_value = nullcontext()
        with mock.patch("services.conversation_binding_service.account_service", accounts), \
             mock.patch("services.conversation_binding_service.OpenAIBackendAPI", return_value=backend):
            return ConversationBindingService().set_archived(body, archived)

    def test_public_archive_and_restore_use_one_preflight_and_one_readback(self):
        for desired in (True, False):
            with self.subTest(desired=desired):
                backend = self.backend([
                    {"mapping": {"original": {}}, "current_node": "original", "is_archived": not desired},
                    {"current_node": "original", "is_archived": desired},
                ])
                result = self.public_archive(backend, desired)
                self.assertIs(result["archived"], desired)
                self.assertEqual(backend._get_conversation.call_count, 2)
                backend.session.patch.assert_called_once()
                backend.close.assert_called_once()

    def test_public_archive_already_done_uses_one_read_and_no_patch(self):
        backend = self.backend([{"mapping": {"original": {}}, "current_node": "original", "is_archived": True}])
        self.assertTrue(self.public_archive(backend)["archived"])
        backend._get_conversation.assert_called_once_with("chat-a")
        backend.session.patch.assert_not_called()

    def test_public_archive_preflight_cursor_drift_keeps_binding_mismatch(self):
        for mapping in ({"original": {}}, {}):
            with self.subTest(mapping=mapping):
                backend = self.backend([{"mapping": mapping, "current_node": "newer", "is_archived": False}])
                with self.assertRaises(ConversationBindingError) as error:
                    self.public_archive(backend)
                self.assertEqual(error.exception.code, "CONVERSATION_BINDING_MISMATCH")
                backend.session.patch.assert_not_called()
                backend._get_conversation.assert_called_once()

    def test_public_archive_other_failures_are_not_preflight_mismatch(self):
        cases = [
            ([{"mapping": {}, "current_node": "original"}], 0),
            ([RuntimeError("read unavailable")], 0),
            ([{"mapping": {"original": {}}, "current_node": "original", "is_archived": False},
              {"current_node": "newer", "is_archived": True}], 1),
        ]
        for documents, patches in cases:
            with self.subTest(documents=documents):
                backend = self.backend(documents)
                with self.assertRaises(RuntimeError) as error:
                    self.public_archive(backend)
                self.assertNotIsInstance(error.exception, ConversationBindingError)
                self.assertEqual(backend.session.patch.call_count, patches)
                self.assertEqual(backend._get_conversation.call_count, len(documents))

    def test_public_archive_changed_account_does_not_read_or_patch(self):
        backend = self.backend([])
        with self.assertRaises(ConversationBindingError) as error:
            self.public_archive(backend, account_identity="other-account")
        self.assertEqual(error.exception.code, "CONVERSATION_BINDING_MISMATCH")
        backend._get_conversation.assert_not_called()
        backend.session.patch.assert_not_called()
