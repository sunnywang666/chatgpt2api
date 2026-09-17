from __future__ import annotations

import unittest
import threading
import tempfile
from pathlib import Path
from contextlib import nullcontext
from unittest import mock
from types import SimpleNamespace

from services import account_request_pacing as pacing

from services.openai_backend_api import ChatRequirements, OpenAIBackendAPI
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

    def test_message_stream_serializes_other_turn_but_allows_original_result_read(self):
        clock = pacing.AccountRequestClock()
        entered = threading.Event()
        attempted = threading.Event()
        first = SimpleNamespace(status_code=200, headers={}, close=lambda: None, iter_lines=lambda: iter([]))
        clock.request(lambda *a, **k: first, "POST", "https://chatgpt.com/backend-api/conversation", stream=True)
        def send(*args, **kwargs):
            entered.set()
            return SimpleNamespace(status_code=200, headers={})
        def second_turn():
            attempted.set()
            clock.request(send, "POST", "https://chatgpt.com/backend-api/f/conversation")
        worker = threading.Thread(target=second_turn)
        worker.start()
        try:
            self.assertTrue(attempted.wait(1))
            clock.request(lambda *a, **k: SimpleNamespace(status_code=200, headers={}), "GET", "https://chatgpt.com/backend-api/conversation/original")
            self.assertFalse(entered.is_set(), "HTTP headers alone cannot release the account turn")
            first.close()
            first.close()  # Watchdog and generator cleanup can both close it.
            self.assertTrue(entered.wait(1))
        finally:
            first.close()
            worker.join(1)
        self.assertFalse(worker.is_alive())

    def test_http_200_stream_rate_error_cools_whole_account_and_releases_turn(self):
        clock = pacing.AccountRequestClock()
        response = SimpleNamespace(status_code=200, headers={}, close=lambda: None,
            iter_lines=lambda: iter([b'data: {"error":{"code":"rate_limit_exceeded","message":"Too many requests"}}']))
        clock.request(lambda *a, **k: response, "POST", "https://chatgpt.com/backend-api/conversation", stream=True)
        list(response.iter_lines())
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
                return self.responses[len(self.calls) - 1]

        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.test"
        backend.session = FakeSession()
        backend._image_headers = lambda path, *_args: {"x-test-path": path}

        response = backend._start_image_generation(
            "make an image",
            ChatRequirements(token="requirements"),
            "conduit",
            "gpt-image-2",
            conversation_id="conversation-1",
            parent_message_id="message-1",
        )

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
                return_value=iter([output]),
            ),
        ):
            result = _generate_bound_single_image(request, 1, 1)

        acquire.assert_called_once_with("cb-account-a", image_model="gpt-image-2")
        self.assertEqual(result[0].provider_account_identity, "account-opaque-a")
        self.assertEqual(result[0].provider_binding_id, "cb-account-a")
        self.assertEqual(result[0].conversation_id, "conversation-1")
        self.assertEqual(result[0].parent_message_id, "message-2")

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

    def test_missing_conversation_zero_or_multiple_exact_matches_are_unattributable(self):
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
        self.assertEqual(
            duplicate.exception.recovery_reason,
            TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value,
        )
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
            timeout_secs=10.0,
            strict_schema=True,
        )
        backend._get_conversation.assert_not_called()

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

    def test_missing_conversation_scan_treats_detail_404_as_a_completed_candidate(self):
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

        self.assertEqual(
            complete.exception.recovery_reason,
            TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value,
        )
        self.assertEqual(backend._get_conversation.call_count, 2)

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

    def test_get_route_authenticates_then_reads_the_same_bound_account(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.testclient import TestClient
        from api.ai import create_router
        app = FastAPI()
        app.include_router(create_router())
        backend = mock.Mock()
        backend._get_conversation.return_value = self.document()
        with (
            mock.patch("api.ai.require_identity", return_value={}) as identity,
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
        backend._get_conversation.assert_called_once_with("conversation-one")


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
        backend = self.backend([{"mapping": {"original": {}}, "is_archived": False}, {"is_archived": True}])
        result = backend.archive_conversation("chat-a", "original")
        self.assertTrue(result["archived"])
        self.assertEqual(backend.session.patch.call_args.kwargs["json"], {"is_archived": True})
        self.assertEqual(backend._get_conversation.call_count, 2)

    def test_recovery_reads_already_archived_chat_without_repeating_patch(self):
        backend = self.backend([{"mapping": {"original": {}}, "is_archived": True}])
        self.assertTrue(backend.archive_conversation("chat-a", "original")["archived"])
        backend.session.patch.assert_not_called()

    def test_missing_original_turn_never_archives_a_different_chat(self):
        backend = self.backend([{"mapping": {"other": {}}}])
        with self.assertRaises(RuntimeError):
            backend.archive_conversation("chat-a", "original")
        backend.session.patch.assert_not_called()
