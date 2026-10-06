from __future__ import annotations

import json
import tempfile
import time
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

from services.account_request_pacing import AccountRequestClock, AccountRequestDeadlineExceeded, pace_account_session
from services.image_task_service import ImageTaskService, _authoritative_image_failure, _failure_details, _public_task
from services.image_thread import ImageThreadError
from services.openai_backend_api import (
    ChatRequirements,
    ImageActiveDeadlineExceeded,
    ImageContentPolicyError,
    ImagePollTimeoutError,
    ImageStreamHardTimeoutError,
    OpenAIBackendAPI,
)
from services.protocol.conversation import (
    ConversationRequest,
    ImageGenerationError,
    ImageOutput,
    _generate_bound_single_image,
)


OWNER = {"id": "owner-1", "name": "Owner", "role": "admin"}
OTHER_OWNER = {"id": "owner-2", "name": "Other", "role": "user"}


def write_policy_task(path: Path, **overrides):
    task = {
        "id": "policy-task", "owner_id": "owner-1", "status": "error",
        "mode": "generate", "model": "gpt-image-2",
        "provider_binding_id": "binding-1",
        "provider_account_identity": "account-1",
        "client_conversation_id": "client-chat-1",
        "conversation_id": "conversation-1", "parent_message_id": "old-failure-assistant",
        "request_message_id": "original-request", "binding_status": "bound",
        "error_code": "content_policy_violation", "error": "original policy failure",
        "upstream_unfinished": False, "created_ts": 100.0,
        "created_at": "2026-09-16 00:00:00", "updated_at": "2099-01-01 00:00:00",
    }
    task.update(overrides)
    path.write_text(json.dumps({"tasks": [task]}), encoding="utf-8")


def manual_image_document(*, latest_active=False, divergent=False, broken=False):
    mapping = {
        "anchor-1": {
            "message": {"author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True},
        },
        "original-request": {
            "parent": "anchor-1",
            "message": {"id": "original-request", "author": {"role": "user"}, "create_time": 100.0},
        },
        "policy-result": {
            "parent": "original-request",
            "message": {"author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True},
        },
        "manual-old": {
            "parent": "policy-result" if not divergent else "anchor-1",
            "message": {"author": {"role": "user"}, "create_time": 200.0},
        },
        "old-image": {
            "parent": "manual-old",
            "message": {"author": {"role": "tool"}},
        },
        "old-finished": {
            "parent": "old-image",
            "message": {"author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True},
        },
        "manual-latest": {
            "parent": "old-finished",
            "message": {"author": {"role": "user"}, "create_time": 300.0},
        },
        "latest-image": {
            "parent": "manual-latest",
            "message": {"author": {"role": "tool"}},
        },
        "latest-finished": {
            "parent": "latest-image",
            "message": {
                "author": {"role": "assistant"},
                "status": "in_progress" if latest_active else "finished_successfully",
                "end_turn": not latest_active,
            },
        },
    }
    if broken:
        mapping["latest-image"]["parent"] = "missing-parent"
    return {"conversation_id": "conversation-1", "current_node": "latest-finished", "mapping": mapping}


class AdoptionBackend:
    document = manual_image_document()
    reads = 0
    downloads = 0
    resolved = []

    def __init__(self, access_token=None, proxy_url=None):
        self.access_token = access_token

    def _get_conversation(self, _conversation_id):
        type(self).reads += 1
        return type(self).document

    def _extract_image_tool_records(self, _document, request_message_id):
        records = {
            "manual-old": [{"message_id": "old-image", "create_time": 210.0, "file_ids": ["old-file"], "sediment_ids": []}],
            "manual-latest": [{"message_id": "latest-image", "create_time": 310.0, "file_ids": [], "sediment_ids": ["latest-file"]}],
        }
        return records.get(request_message_id, [])

    def _query_backend_tasks(self, **kwargs):
        if kwargs.get("strict_schema") is not True:
            raise AssertionError("adoption must require a strict tasks schema")
        return []

    def resolve_conversation_image_urls(self, conversation_id, file_ids, sediment_ids, **kwargs):
        type(self).resolved.append((conversation_id, file_ids, sediment_ids, kwargs))
        return ["https://example.test/manual.png"]

    def download_image_bytes(self, _urls):
        type(self).downloads += 1
        return [b"manual-image-bytes"]

    def close(self):
        return None


def failure_cursor_document(*, change=None):
    mapping = {
        "anchor": {
            "message": {"author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True},
        },
        "original-request": {
            "parent": "anchor",
            "message": {"id": "original-request", "author": {"role": "user"}},
        },
        "fresh-terminal": {
            "parent": "original-request",
            "message": {
                "id": "fresh-terminal", "author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True,
                "content": {"content_type": "text", "parts": ["Image request blocked."]},
            },
        },
    }
    document = {"conversation_id": "conversation-1", "current_node": "fresh-terminal", "mapping": mapping,
                "is_archived": False}
    if change == "later_user":
        mapping["later-user"] = {"parent": "fresh-terminal", "message": {"author": {"role": "user"}}}
        document["current_node"] = "later-user"
    elif change == "branch":
        mapping["sibling"] = {"parent": "original-request", "message": {"author": {"role": "assistant"}}}
    elif change == "imageasset":
        mapping["fresh-terminal"]["message"]["content"] = {"content_type": "image_asset_pointer", "asset_pointer": "file-service://generated"}
    elif change == "running":
        mapping["fresh-terminal"]["message"].update(status="in_progress", end_turn=False)
    elif change == "wrongchat":
        document["conversation_id"] = "other-conversation"
    return document


class FailureCursorBackend:
    document = failure_cursor_document()
    reads = 0

    def __init__(self, access_token=None, proxy_url=None):
        self.access_token = access_token

    def _get_conversation(self, _conversation_id):
        type(self).reads += 1
        return type(self).document

    @staticmethod
    def _has_image_asset_pointer(payload):
        return OpenAIBackendAPI._has_image_asset_pointer(payload)

    def close(self):
        return None


def wait_for_task(service: ImageTaskService, identity: dict[str, object], task_id: str, status: str, timeout: float = 2.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        result = service.list_tasks(identity, [task_id])
        last = (result.get("items") or [None])[0]
        if last and last.get("status") == status:
            return last
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} did not reach {status}, last={last}")


class ImageTaskServiceTests(unittest.TestCase):
    def test_failure_continuation_returns_only_fresh_exact_terminal_cursor_without_mutating_receipt(self):
        cases = (
            (None, {}, True),
            (None, {"error_code": "NO_IMAGE_GENERATED"}, True),
            (None, {"upstream_outcome": "unknown"}, True),
            (None, {"_completion": {
                "state": "needs_attention", "reason": "COMPLETION_ORIGINAL_ONLY", "next_at": None,
                "read_only_original": True, "max_extra_requests": 0, "allow_unconfirmed_retry": False,
            }}, True),
            (None, {"_completion": {
                "state": "needs_attention", "reason": "COMPLETION_ORIGINAL_ONLY", "next_at": None,
                "read_only_original": True, "max_extra_requests": 0, "allow_unconfirmed_retry": True,
            }}, False),
            (None, {"_completion": {
                "state": "needs_attention", "reason": "COMPLETION_ORIGINAL_ONLY", "next_at": None,
                "read_only_original": True, "max_extra_requests": 0, "allow_unconfirmed_retry": False,
                "replacement_id": "child",
            }}, False),
            ("later_user", {"upstream_outcome": "unknown"}, False),
            ("branch", {"upstream_outcome": "unknown"}, False),
            ("wrongchat", {}, False),
            ("imageasset", {"upstream_outcome": "unknown"}, False),
            ("running", {}, False),
            (None, {"upstream_outcome": "unknown", "upstream_unfinished": True}, False),
            (None, {"error_code": "CONVERSATION_OUTCOME_UNKNOWN", "upstream_unfinished": False}, False),
            (None, {"request_message_id": ""}, False),
        )
        for change, overrides, eligible in cases:
            with self.subTest(change=change, overrides=overrides), tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "image_tasks.json"
                write_policy_task(path, **overrides)
                service = self.make_service(path)
                if "_completion" in overrides:
                    # Legacy JSON import intentionally drops completion state;
                    # exercise the actual SQLite receipt shape used after the
                    # admin original-result read has been persisted.
                    with service._transaction():
                        service._tasks["owner-1:policy-task"]["_completion"] = overrides["_completion"]
                        service._save_locked()
                if change is None and eligible:
                    # An expired scheduler claim is evidence to recover, not
                    # an active writer. The task itself is terminal.
                    with service._transaction():
                        task = service._tasks["owner-1:policy-task"]
                        task.update(
                            _claim_id="expired-claim",
                            _claim_until=time.time() - 1,
                            request_parent_message_id="anchor",
                        )
                        service._save_locked()
                FailureCursorBackend.document = failure_cursor_document(change=change)
                FailureCursorBackend.reads = 0
                with (
                    mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                    mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="read-token"),
                    mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                    mock.patch("services.openai_backend_api.OpenAIBackendAPI", FailureCursorBackend),
                ):
                    with service.store.connect() as db:
                        before = service.store.read_receipt(db, "image", OWNER["id"], "policy-task")
                    result = service.failure_continuation(OWNER, "policy-task")
                    with service.store.connect() as db:
                        after = service.store.read_receipt(db, "image", OWNER["id"], "policy-task")
                self.assertEqual(after, before)
                if eligible:
                    self.assertEqual(result, {
                        "source_task_id": "policy-task",
                        "source_request_message_id": "original-request",
                        "provider_binding_id": "binding-1",
                        "provider_account_identity": "account-1",
                        "client_conversation_id": "client-chat-1",
                        "conversation_id": "conversation-1",
                        "parent_message_id": "fresh-terminal",
                    })
                    self.assertEqual(FailureCursorBackend.reads, 1)
                else:
                    self.assertIsNone(result)

    def test_failure_continuation_refuses_account_change_without_provider_read(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path)
            service = self.make_service(path)
            FailureCursorBackend.document = failure_cursor_document()
            FailureCursorBackend.reads = 0
            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="other-account"),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", FailureCursorBackend),
            ):
                self.assertIsNone(service.failure_continuation(OWNER, "policy-task"))
            self.assertEqual(FailureCursorBackend.reads, 0)

    def test_failure_continuation_refuses_an_active_terminal_claim_or_reserved_turn(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path)
            service = self.make_service(path)
            FailureCursorBackend.document = failure_cursor_document()
            FailureCursorBackend.reads = 0
            with service._transaction():
                task = service._tasks["owner-1:policy-task"]
                task.update(_claim_id="live-claim", _claim_until=time.time() + 60)
                service._save_locked()
            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", FailureCursorBackend),
            ):
                self.assertIsNone(service.failure_continuation(OWNER, "policy-task"))
            self.assertEqual(FailureCursorBackend.reads, 0)
            with service._transaction():
                task = service._tasks["owner-1:policy-task"]
                task.update(_claim_id="expired-claim", _claim_until=time.time() - 1, _turn_reserved=True)
                service._save_locked()
            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", FailureCursorBackend),
            ):
                self.assertIsNone(service.failure_continuation(OWNER, "policy-task"))
            self.assertEqual(FailureCursorBackend.reads, 0)

    def test_attachment_download_observations_are_bounded_and_follow_actual_bytes(self):
        from services.request_context import executing
        for failure in (False, True):
            with self.subTest(failure=failure):
                events = []
                context = mock.Mock()
                context.record_stage.side_effect = lambda stage, **extra: events.append((stage, extra))
                backend = object.__new__(OpenAIBackendAPI)
                backend._image_request_options = mock.Mock(return_value={"timeout": 120})
                backend.session = mock.Mock()

                def download(url, **_kwargs):
                    self.assertEqual(events, [("attachment_download_started", {"image_count": 2})])
                    if failure:
                        raise TimeoutError("signed URL must never enter timing evidence")
                    return mock.Mock(status_code=200, content=b"same-original-image")

                backend.session.get.side_effect = download
                urls = ["https://storage.test/one?secret=hidden", "https://storage.test/two?secret=hidden"]
                with executing(context):
                    if failure:
                        with self.assertRaises(TimeoutError):
                            backend.download_image_bytes(urls)
                    else:
                        self.assertEqual(backend.download_image_bytes(urls), [b"same-original-image"])
                expected = [("attachment_download_started", {"image_count": 2})]
                if not failure:
                    expected.append(("attachment_download_finished", {"image_count": 1}))
                self.assertEqual(events, expected)
                self.assertEqual(backend.session.get.call_count, 1 if failure else 2)

    def test_attachment_observation_failure_cannot_repeat_download_or_change_result(self):
        from services.request_context import executing
        backend = object.__new__(OpenAIBackendAPI)
        backend._image_request_options = mock.Mock(return_value={"timeout": 120})
        backend.session = mock.Mock()
        backend.session.get.return_value = mock.Mock(status_code=200, content=b"original-image")
        context = mock.Mock()
        context.record_stage.side_effect = RuntimeError("diagnostic storage unavailable")
        with executing(context):
            self.assertEqual(backend.download_image_bytes(["https://storage.test/original"]), [b"original-image"])
            self.assertEqual(backend.download_image_bytes([]), [])
        self.assertEqual(backend.session.get.call_count, 1)
        self.assertEqual(context.record_stage.call_count, 2)

    def test_image_external_transfers_consume_deadline_without_account_clock(self):
        from services.openai_backend_api import requests
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch("services.account_request_pacing.DATA_DIR", Path(tmp)), \
             mock.patch("services.account_request_pacing._clocks", {}), \
             mock.patch("services.openai_backend_api.time.time", return_value=1000), \
             mock.patch("services.account_request_pacing.time.monotonic", return_value=100), \
             mock.patch.object(requests.Session, "request", autospec=True) as transport:
            transport.return_value = mock.Mock(status_code=200, content=b"generated-image")
            backend = object.__new__(OpenAIBackendAPI)
            backend.session = requests.Session()
            pace_account_session(backend.session, {"account_id": "fixture"}, "fixture-token")
            callback = lambda _step: None
            callback.active_deadline_at = 1010
            backend.progress_callback = callback
            self.assertEqual(backend.download_image_bytes(["https://storage.test/result"]), [b"generated-image"])
            backend.session.put("https://storage.test/upload", data=b"input", **backend._image_request_options(120))
            for call in transport.call_args_list:
                self.assertEqual(call.kwargs["timeout"], 10)
                self.assertFalse(any(key.startswith("_account_request") for key in call.kwargs))
            self.assertEqual(transport.call_count, 2)
            with self.assertRaises(AccountRequestDeadlineExceeded):
                backend.session.get("https://storage.test/result", timeout=10, _account_request_deadline_monotonic=99)
            self.assertEqual(transport.call_count, 2)
            backend.close()

    def test_image_result_reads_cannot_wait_past_active_deadline_in_account_clock(self):
        operations = (
            lambda backend: backend._get_conversation("original"),
            lambda backend: backend._query_backend_tasks("original"),
            lambda backend: backend._get_file_download_url("generated"),
            lambda backend: backend._get_attachment_download_url("original", "generated"),
            lambda backend: backend.download_image_bytes(["https://images.test/generated"]),
        )
        for operation in operations:
            with self.subTest(operation=operation), mock.patch("services.openai_backend_api.time.time", return_value=1000), \
                 mock.patch("services.account_request_pacing.time.monotonic", return_value=100), \
                 mock.patch("services.account_request_pacing.time.sleep") as sleep:
                backend = object.__new__(OpenAIBackendAPI)
                backend.base_url = "https://provider.test"
                backend._headers = lambda *_args: {}
                callback = lambda _step: None
                callback.active_deadline_at = 1010
                backend.progress_callback = callback
                clock = AccountRequestClock()
                clock.cooldown_until = 120
                transport = mock.Mock(return_value=mock.Mock(status_code=200, headers={}, content=b"image"))
                transport.return_value.json.return_value = {"tasks": []}
                backend.session = mock.Mock()
                backend.session.get.side_effect = lambda url, **kwargs: clock.request(transport, "GET", url, **kwargs)
                with self.assertRaises(AccountRequestDeadlineExceeded):
                    operation(backend)
                transport.assert_not_called()
                sleep.assert_not_called()
                self.assertFalse(clock.lock.locked())

    def test_image_result_read_reduces_transport_timeout_after_cooldown(self):
        now = [100.0]
        with mock.patch("services.openai_backend_api.time.time", side_effect=lambda: now[0] + 900), \
             mock.patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
             mock.patch("services.account_request_pacing.time.sleep", side_effect=lambda secs: now.__setitem__(0, now[0] + secs)):
            backend = object.__new__(OpenAIBackendAPI)
            backend.base_url = "https://provider.test"
            backend._headers = lambda *_args: {}
            callback = lambda _step: None
            callback.active_deadline_at = 1010
            backend.progress_callback = callback
            clock = AccountRequestClock()
            clock.cooldown_until = 104
            transport = mock.Mock(return_value=mock.Mock(status_code=200, headers={}))
            transport.return_value.json.return_value = {"current_node": "original"}
            backend.session = mock.Mock()
            backend.session.get.side_effect = lambda url, **kwargs: clock.request(transport, "GET", url, **kwargs)
            self.assertEqual(backend._get_conversation("original"), {"current_node": "original"})
            self.assertEqual(transport.call_args.kwargs["timeout"], 6)
            # This fixture stops at the clock; the real paced Session consumes
            # the connection option downstream (covered by the native tests).
            self.assertEqual(transport.call_args.kwargs.pop("_account_request_connect_timeout_secs"), 10)
            self.assertFalse(any(key.startswith("_account_request") for key in transport.call_args.kwargs))

    def test_image_original_reads_default_to_ten_second_connection_sub_budget(self):
        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://provider.test"
        backend._headers = lambda *_args: {}
        backend.session = mock.Mock()
        backend.session.get.return_value = mock.Mock(status_code=200)
        backend.session.get.return_value.json.return_value = {"current_node": "original"}
        for total in (60, 6):
            self.assertEqual(backend._get_conversation("original", timeout_secs=total), {"current_node": "original"})
            options = backend.session.get.call_args.kwargs
            self.assertEqual(options["timeout"], total)
            self.assertEqual(options["_account_request_connect_timeout_secs"], 10)

    def test_image_original_read_paced_session_caps_connection_inside_total(self):
        from curl_cffi import requests
        from services.config import config
        from services.account_request_pacing import pace_account_session
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch("services.account_request_pacing.DATA_DIR", Path(tmp)), \
             mock.patch("services.account_request_pacing._clocks", {}), \
             mock.patch.dict(config.data, {"account_request_interval_secs": 0,
                                           "account_conversation_read_interval_secs": 0}), \
             mock.patch.object(requests.Session, "request", autospec=True) as transport:
            transport.return_value = mock.Mock(status_code=200, headers={}, content=b'{}')
            transport.return_value.json.return_value = {"current_node": "original"}
            backend = object.__new__(OpenAIBackendAPI)
            backend.base_url = "https://chatgpt.com"
            backend._headers = lambda *_args: {}
            backend.session = requests.Session()
            pace_account_session(backend.session, {"account_id": "fixture"}, "fixture-token")
            try:
                for total in (60, 6):
                    backend._get_conversation("original", timeout_secs=total)
                    timeout = transport.call_args.kwargs["timeout"]
                    self.assertEqual(timeout, (min(10, total), max(0, total - 10)))
                    self.assertFalse(any(k.startswith("_account_request") for k in transport.call_args.kwargs))
            finally:
                backend.close()

    def test_image_preflight_raw_read_does_not_receive_pacing_kwargs(self):
        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://provider.test"
        backend._headers = lambda *_args: {}
        callback = lambda _step: None
        callback.active_deadline_at = time.time() + 10
        backend.progress_callback = callback
        response = mock.Mock(status_code=200)
        response.json.return_value = {"current_node": "original"}
        def send(method, url, *, headers, timeout):
            self.assertEqual(method, "GET")
            self.assertLessEqual(timeout, 10)
            return response
        self.assertEqual(backend._get_conversation("original", _send=send), {"current_node": "original"})
        response.close.assert_called_once()

    def test_image_pacing_wait_uses_active_budget_not_network_timeout(self):
        for active_seconds, expected_timeout in ((120, 30), (45, 10), (30, None)):
            now = [100.0]
            with self.subTest(active_seconds=active_seconds), \
                 mock.patch("services.openai_backend_api.time.time", side_effect=lambda: now[0] + 900), \
                 mock.patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
                 mock.patch("services.account_request_pacing.time.sleep", side_effect=lambda secs: now.__setitem__(0, now[0] + secs)):
                backend = object.__new__(OpenAIBackendAPI)
                backend.base_url = "https://provider.test"
                backend._headers = lambda *_args: {}
                callback = lambda _step: None
                callback.active_deadline_at = 1000 + active_seconds
                backend.progress_callback = callback
                clock = AccountRequestClock()
                clock.next_request = 135
                transport = mock.Mock(return_value=mock.Mock(status_code=200, headers={}))
                transport.return_value.json.return_value = {"current_node": "original"}
                backend.session = mock.Mock()
                backend.session.get.side_effect = lambda url, **kwargs: clock.request(transport, "GET", url, **kwargs)
                if expected_timeout is None:
                    with self.assertRaises(AccountRequestDeadlineExceeded):
                        backend._get_conversation("original", timeout_secs=30)
                    transport.assert_not_called()
                else:
                    self.assertEqual(backend._get_conversation("original", timeout_secs=30), {"current_node": "original"})
                    self.assertEqual(now[0], 135)
                    self.assertEqual(transport.call_args.kwargs["timeout"], expected_timeout)
                self.assertFalse(clock.lock.locked())

    def test_image_local_read_pacing_preserves_active_budget_but_not_http_time(self):
        now = [100.0]
        with mock.patch("services.openai_backend_api.time.time", side_effect=lambda: now[0] + 900), \
             mock.patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
             mock.patch("services.account_request_pacing.time.sleep", side_effect=lambda secs: now.__setitem__(0, now[0] + secs)):
            backend = object.__new__(OpenAIBackendAPI)
            backend.base_url = "https://provider.test"
            backend._headers = lambda *_args: {}
            callback = lambda _step: None
            callback.active_deadline_at = 1300
            callback.local_pacing_wait_secs = 0
            def credit(seconds):
                callback.active_deadline_at += seconds
                callback.local_pacing_wait_secs += seconds
            callback.record_local_pacing_wait = credit
            backend.progress_callback = callback
            clock = AccountRequestClock()
            response = mock.Mock(status_code=200, headers={})
            response.json.return_value = {"current_node": "original"}
            def transport(*args, **kwargs):
                self.assertEqual(kwargs.pop("_account_request_connect_timeout_secs"), 10)
                self.assertFalse(any(k.startswith('_account_request') for k in kwargs))
                now[0] += 1
                return response
            backend.session = mock.Mock()
            backend.session.get.side_effect = lambda url, **kwargs: clock.request(transport, "GET", url, **kwargs)
            for _ in range(8):
                clock.next_conversation_read = now[0] + 60
                backend._get_conversation("original")
            self.assertEqual(callback.local_pacing_wait_secs, 480)
            self.assertEqual(backend._image_active_timeout(300), 292)

    def test_image_provider_cooldown_is_not_credited_as_local_wait(self):
        with mock.patch("services.openai_backend_api.time.time", return_value=1000), \
             mock.patch("services.account_request_pacing.time.monotonic", return_value=100):
            backend = object.__new__(OpenAIBackendAPI)
            callback = lambda _step: None
            callback.active_deadline_at = 1010
            callback.record_local_pacing_wait = mock.Mock()
            backend.progress_callback = callback
            clock = AccountRequestClock()
            clock.cooldown_until = 160
            transport = mock.Mock()
            with self.assertRaises(AccountRequestDeadlineExceeded):
                clock.request(transport, "GET", "https://provider.test/conversation/original", **backend._image_request_options(60))
            transport.assert_not_called()
            callback.record_local_pacing_wait.assert_not_called()
            self.assertEqual(clock.cooldown_until, 160)

    @mock.patch("services.openai_backend_api.account_service.require_image_account")
    def test_final_local_deadline_after_reservation_remains_known_unsent(self, _capability):
        from services.request_context import executing
        now = [100.0]
        backend = object.__new__(OpenAIBackendAPI)
        backend.access_token = "synthetic-fixture"
        backend.base_url = "https://provider.test"
        backend.image_request_message_id = "original"
        backend.image_submission_started = False
        backend.retain_bound_conversation = True
        backend._image_model_settings = lambda _model: ("gpt-image", "")
        backend._image_headers = lambda *_args: {}
        callback = lambda _step: None
        callback.active_deadline_at = 1001
        callback.record_submission_started = lambda: now.__setitem__(0, 102)
        backend.progress_callback = callback
        clock = AccountRequestClock()
        transport = mock.Mock()
        backend.session = mock.Mock()
        backend.session.post.side_effect = lambda url, **kwargs: clock.request(transport, "POST", url, **kwargs)
        context = mock.Mock(owner="owner", request_id="request")
        context.log_fields.return_value = {}
        with mock.patch("services.openai_backend_api.time.time", side_effect=lambda: now[0] + 900), \
             mock.patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), executing(context):
            with self.assertRaises(AccountRequestDeadlineExceeded):
                backend._start_image_generation("cat", ChatRequirements(token="fixture"), "conduit", "gpt-image-2")
        transport.assert_not_called()
        context.record_stage.assert_not_called()
        self.assertFalse(backend.image_submission_started)

    @mock.patch("services.openai_backend_api.account_service.require_image_account")
    def test_endpoint_404_fallback_local_deadline_is_known_unsent(self, _capability):
        backend = object.__new__(OpenAIBackendAPI)
        backend.access_token = "synthetic-fixture"
        backend.base_url = "https://provider.test"
        backend.image_request_message_id = "original"
        backend.image_submission_started = False
        backend.retain_bound_conversation = True
        backend._image_model_settings = lambda _model: ("gpt-image", "")
        backend._image_headers = lambda *_args: {}
        backend.progress_callback = lambda _step: None
        first = mock.Mock(status_code=404)
        calls = []
        def post(url, **kwargs):
            calls.append(url)
            kwargs['_account_request_before_send']()
            if len(calls) == 1:
                return first
            raise AccountRequestDeadlineExceeded("local wait before fallback transport")
        backend.session = mock.Mock()
        backend.session.post.side_effect = post
        with self.assertRaises(AccountRequestDeadlineExceeded):
            backend._start_image_generation("cat", ChatRequirements(token="fixture"), "conduit", "gpt-image-2")
        self.assertEqual(len(calls), 2)
        first.close.assert_called_once()
        self.assertFalse(backend.image_submission_started)

    @mock.patch("services.openai_backend_api.account_service.require_image_account", return_value={"provider_account_identity": "fixture"})
    def test_generation_post_records_submission_boundary_before_network_call(self, _capability):
        backend = object.__new__(OpenAIBackendAPI)
        backend.access_token = "synthetic-fixture"
        backend.base_url = "https://chatgpt.example.test"
        backend.image_request_message_id = "request-message-1"
        backend.image_submission_started = False
        backend.retain_bound_conversation = True
        backend._image_model_settings = lambda _model: ("gpt-image", "")
        backend._image_headers = lambda *_args: {}
        observed = []

        def progress_callback(_step):
            return None

        def record_submission_started():
            observed.append("persisted")

        progress_callback.record_submission_started = record_submission_started
        backend.progress_callback = progress_callback

        class Session:
            def post(self, *_args, **kwargs):
                kwargs["_account_request_before_send"]()
                self.assert_boundary()
                raise RuntimeError("generation POST timed out")

            def assert_boundary(self):
                if observed != ["persisted"] or backend.image_submission_started is not True:
                    raise AssertionError("submission boundary was not recorded before POST")

        backend.session = Session()

        with self.assertRaisesRegex(RuntimeError, "generation POST timed out"):
            backend._start_image_generation(
                "cat", ChatRequirements(token="requirements"), "conduit", "gpt-image-2",
                conversation_id="conversation-1", parent_message_id="parent-1",
            )

        self.assertTrue(backend.image_submission_started)
        self.assertEqual(observed, ["persisted"])

    def test_upload_bootstrap_and_prepare_timeouts_use_one_remaining_budget(self):
        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.example.test"
        backend.user_agent = "test-agent"
        backend.image_submission_started = False
        backend.retain_bound_conversation = True
        backend._headers = lambda *_args: {}
        backend._image_headers = lambda *_args: {}
        backend._bootstrap_headers = lambda: {}
        backend._image_model_settings = lambda _model: ("gpt-image", "")
        callback = lambda _step: None
        callback.active_deadline_at = time.time() + 2.0
        backend.progress_callback = callback
        observed_timeouts = []

        class Response:
            status_code = 200
            text = ""
            content = b""

            def __init__(self, payload=None):
                self.payload = payload or {}

            def json(self):
                return self.payload

        class Session:
            def get(self, *_args, **kwargs):
                observed_timeouts.append(kwargs["timeout"])
                return Response()

            def post(self, url, **kwargs):
                observed_timeouts.append(kwargs["timeout"])
                if url.endswith("/backend-api/files"):
                    return Response({"upload_url": "https://upload.test/object", "file_id": "file-1"})
                if url.endswith("/uploaded"):
                    return Response()
                return Response({"conduit_token": "conduit"})

            def put(self, *_args, **kwargs):
                observed_timeouts.append(kwargs["timeout"])
                return Response()

        backend.session = Session()
        backend._decode_image_base64 = lambda _image: b"image-bytes"

        with mock.patch("services.openai_backend_api.Image.open") as image_open:
            image_open.return_value.size = (100, 100)
            image_open.return_value.format = "PNG"
            backend._bootstrap()
            backend._upload_image("encoded", "input.png")
            backend._prepare_image_conversation(
                "draw a cat", ChatRequirements("requirements"), "gpt-image-2",
            )
            backend._get_conversation("conversation-1")

        self.assertEqual(len(observed_timeouts), 6)
        self.assertTrue(all(0 < timeout <= 2.0 for timeout in observed_timeouts))

    @mock.patch("services.openai_backend_api.account_service.require_image_account", return_value={"provider_account_identity": "fixture"})
    def test_generation_post_is_not_called_when_submission_boundary_cannot_be_persisted(self, _capability):
        backend = object.__new__(OpenAIBackendAPI)
        backend.access_token = "synthetic-fixture"
        backend.base_url = "https://chatgpt.example.test"
        backend.image_request_message_id = "request-message-1"
        backend.image_submission_started = False
        backend.retain_bound_conversation = True
        backend._image_model_settings = lambda _model: ("gpt-image", "")
        backend._image_headers = lambda *_args: {}

        def progress_callback(_step):
            return None

        def record_submission_started():
            raise OSError("receipt save failed")

        progress_callback.record_submission_started = record_submission_started
        backend.progress_callback = progress_callback
        class Session:
            def __init__(self):
                self.send_called = False

            def post(self, *_args, **kwargs):
                kwargs["_account_request_before_send"]()
                self.send_called = True

        session = Session()
        backend.session = session

        with self.assertRaisesRegex(OSError, "receipt save failed"):
            backend._start_image_generation(
                "cat", ChatRequirements(token="requirements"), "conduit", "gpt-image-2",
                conversation_id="conversation-1", parent_message_id="parent-1",
            )

        self.assertFalse(session.send_called)
        self.assertFalse(backend.image_submission_started)

    def test_poll_successful_read_clears_a_prior_rate_limit(self):
        from utils.helper import UpstreamHTTPError

        backend = object.__new__(OpenAIBackendAPI)
        backend._query_backend_tasks = lambda **_kwargs: []
        backend._extract_image_tool_records = lambda *_args: []
        backend._find_content_policy_error_in_conversation = lambda *_args: ""
        reads = iter([
            UpstreamHTTPError("conversation", 429, {}, retry_after=0),
            {"mapping": {}},
        ])

        def get_conversation(_conversation_id):
            value = next(reads, {"mapping": {}})
            if isinstance(value, BaseException):
                raise value
            return value

        backend._get_conversation = get_conversation
        with (
            mock.patch.dict(
                "services.openai_backend_api.config.data",
                {"image_poll_initial_wait_secs": 0, "image_poll_interval_secs": 0.5},
            ),
            mock.patch("services.openai_backend_api.random.uniform", return_value=0),
        ):
            with self.assertRaises(ImagePollTimeoutError) as raised:
                backend._poll_image_results(
                    "conversation-1", 0.02, request_message_id="request-1",
                )

        self.assertIsNone(getattr(raised.exception, "status_code", None))
        self.assertIsNone(getattr(raised.exception, "retry_after", None))

    def test_bound_bootstrap_failure_is_known_not_submitted_with_existing_chat(self):
        class Backend:
            image_submission_started = False
            image_request_message_id = "request-message-1"

            def __init__(self, access_token=None):
                self.access_token = access_token

            def stream_conversation(self, **_kwargs):
                raise RuntimeError("bootstrap connection timed out")

            def get_conversation_parent_message_id(self, _conversation_id):
                return "parent-before-request"

            def close(self):
                return None

        request = ConversationRequest(
            model="gpt-image-2", prompt="cat", provider_binding_id="cb_account_a",
            provider_account_identity="account_opaque_a",
            client_conversation_id="workbench-conversation-1",
            conversation_id="conversation-1", parent_message_id="parent-before-request",
            retain_conversation=True,
        )
        with (
            mock.patch("services.protocol.conversation.account_service.get_bound_account_identity", return_value="account_opaque_a"),
            mock.patch("services.protocol.conversation.account_service.acquire_bound_image_access_token", return_value="bound-token"),
            mock.patch("services.protocol.conversation.account_service.get_account", return_value={"email": "account@example.test"}),
            mock.patch("services.protocol.conversation.account_service.conversation_binding_lock", return_value=nullcontext()),
            mock.patch("services.protocol.conversation.account_service.mark_image_result"),
            mock.patch("services.protocol.conversation.OpenAIBackendAPI", Backend),
        ):
            with self.assertRaises(ImageGenerationError) as raised:
                _generate_bound_single_image(request, 1, 1)

        self.assertEqual(raised.exception.code, "IMAGE_GENERATION_NOT_SUBMITTED")
        self.assertIs(raised.exception.upstream_submitted, False)
        self.assertEqual(raised.exception.conversation_id, "conversation-1")
        self.assertIn("timed out", str(raised.exception))

    def test_admitted_bound_image_never_releases_another_legacy_slot(self):
        from services.request_context import executing
        for outcome in ("success", "stream_error", "setup_error"):
            with self.subTest(outcome=outcome):
                request = ConversationRequest(
                    model="gpt-image-2", prompt="cat", provider_binding_id="binding-1",
                    provider_account_identity="account-1", client_conversation_id="client-1",
                    retain_conversation=True,
                )
                backend = mock.Mock()
                backend.image_submission_started = True
                backend.image_request_message_id = "message-1"
                backend.get_conversation_parent_message_id.return_value = "answer-1"
                output = ImageOutput(kind="result", model="gpt-image-2", index=1, total=1,
                                     data=[{"b64_json": "image"}], conversation_id="chat-1")
                with (
                    executing(mock.Mock()),
                    mock.patch("services.protocol.conversation.account_service.get_bound_account_identity", return_value="account-1"),
                    mock.patch("services.protocol.conversation.account_service.acquire_bound_image_access_token", return_value="token") as acquire,
                    mock.patch("services.protocol.conversation.account_service.get_account", return_value={}),
                    mock.patch("services.protocol.conversation.account_service.conversation_binding_lock", return_value=nullcontext()),
                    mock.patch("services.protocol.conversation.account_service.mark_image_result") as mark,
                    mock.patch("services.protocol.conversation.account_service.release_image_slot") as release,
                    mock.patch("services.protocol.conversation.OpenAIBackendAPI", return_value=backend,
                               side_effect=RuntimeError("setup failed") if outcome == "setup_error" else None),
                    mock.patch("services.protocol.conversation.stream_image_outputs", return_value=iter([output]),
                               side_effect=RuntimeError("stream failed") if outcome == "stream_error" else None),
                ):
                    if outcome == "success":
                        self.assertEqual(_generate_bound_single_image(request, 1, 1)[0].conversation_id, "chat-1")
                    else:
                        with self.assertRaises((RuntimeError, ImageGenerationError)):
                            _generate_bound_single_image(request, 1, 1)
                    acquire.assert_called_once_with("binding-1", image_model="gpt-image-2", reserve_slot=False)
                    if outcome == "setup_error":
                        mark.assert_not_called()
                    else:
                        mark.assert_called_once_with("token", outcome == "success", release_slot=False)
                    release.assert_not_called()

    def test_unsent_admission_rejection_does_not_consume_image_capacity(self):
        from services.request_context import AdmissionLost, executing
        for pool_managed in (False, True):
            for submitted in (False, True, None):
                with self.subTest(pool_managed=pool_managed, submitted=submitted):
                    request = ConversationRequest(
                        model="gpt-image-2", prompt="cat", provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="client-1",
                        retain_conversation=True,
                    )
                    backend = mock.Mock(image_submission_started=submitted, image_request_message_id="message-1")
                    with (
                        executing(mock.Mock() if pool_managed else None),
                        mock.patch("services.protocol.conversation.account_service.get_bound_account_identity", return_value="account-1"),
                        mock.patch("services.protocol.conversation.account_service.acquire_bound_image_access_token", return_value="token"),
                        mock.patch("services.protocol.conversation.account_service.get_account", return_value={}),
                        mock.patch("services.protocol.conversation.account_service.conversation_binding_lock", return_value=nullcontext()),
                        mock.patch("services.protocol.conversation.account_service.mark_image_result") as mark,
                        mock.patch("services.protocol.conversation.account_service.release_image_slot") as release,
                        mock.patch("services.protocol.conversation.OpenAIBackendAPI", return_value=backend),
                        mock.patch("services.protocol.conversation.stream_image_outputs",
                                   side_effect=AdmissionLost("original image capability is unavailable before send")),
                    ):
                        with self.assertRaises(ImageGenerationError) as raised:
                            _generate_bound_single_image(request, 1, 1)
                        self.assertIs(raised.exception.upstream_submitted, submitted)
                        if submitted is False:
                            mark.assert_not_called()
                            if pool_managed:
                                release.assert_not_called()
                            else:
                                release.assert_called_once_with("token")
                        else:
                            mark.assert_called_once_with("token", False, **({"release_slot": False} if pool_managed else {}))
                            release.assert_not_called()
                        backend.close.assert_called_once()

    def test_bound_post_submission_timeout_remains_unknown(self):
        class Backend:
            image_submission_started = True
            image_request_message_id = "request-message-1"

            def __init__(self, access_token=None):
                self.access_token = access_token

            def stream_conversation(self, **_kwargs):
                raise RuntimeError("generation POST response timed out")

            def get_conversation_parent_message_id(self, _conversation_id):
                return "parent-before-request"

            def close(self):
                return None

        request = ConversationRequest(
            model="gpt-image-2", prompt="cat", provider_binding_id="cb_account_a",
            provider_account_identity="account_opaque_a",
            client_conversation_id="workbench-conversation-1",
            conversation_id="conversation-1", parent_message_id="parent-before-request",
            retain_conversation=True,
        )
        with (
            mock.patch("services.protocol.conversation.account_service.get_bound_account_identity", return_value="account_opaque_a"),
            mock.patch("services.protocol.conversation.account_service.acquire_bound_image_access_token", return_value="bound-token"),
            mock.patch("services.protocol.conversation.account_service.get_account", return_value={"email": "account@example.test"}),
            mock.patch("services.protocol.conversation.account_service.conversation_binding_lock", return_value=nullcontext()),
            mock.patch("services.protocol.conversation.account_service.mark_image_result"),
            mock.patch("services.protocol.conversation.OpenAIBackendAPI", Backend),
        ):
            with self.assertRaises(ImageGenerationError) as raised:
                _generate_bound_single_image(request, 1, 1)

        self.assertEqual(raised.exception.code, "CONVERSATION_OUTCOME_UNKNOWN")
        self.assertIs(raised.exception.upstream_submitted, True)

    def test_active_deadline_during_preparation_is_known_not_submitted(self):
        class Backend:
            image_submission_started = False
            image_request_message_id = "request-message-1"

            def __init__(self, access_token=None):
                self.access_token = access_token

            def stream_conversation(self, **_kwargs):
                raise ImageActiveDeadlineExceeded("active preparation deadline exhausted")

            def get_conversation_parent_message_id(self, _conversation_id):
                return "parent-before-request"

            def close(self):
                return None

        callback = lambda _step: None
        callback.start_active_attempt = lambda: None
        callback.active_deadline_at = time.time() - 1
        request = ConversationRequest(
            model="gpt-image-2", prompt="cat", provider_binding_id="binding-1",
            provider_account_identity="account-1", client_conversation_id="client-1",
            conversation_id="conversation-1", parent_message_id="parent-before-request",
            retain_conversation=True, progress_callback=callback,
        )
        with (
            mock.patch("services.protocol.conversation.account_service.get_bound_account_identity", return_value="account-1"),
            mock.patch("services.protocol.conversation.account_service.acquire_bound_image_access_token", return_value="token"),
            mock.patch("services.protocol.conversation.account_service.get_account", return_value={}),
            mock.patch("services.protocol.conversation.account_service.conversation_binding_lock", return_value=nullcontext()),
            mock.patch("services.protocol.conversation.account_service.mark_image_result"),
            mock.patch("services.protocol.conversation.OpenAIBackendAPI", Backend),
        ):
            with self.assertRaises(ImageGenerationError) as raised:
                _generate_bound_single_image(request, 1, 1)

        self.assertEqual(raised.exception.code, "IMAGE_GENERATION_NOT_SUBMITTED")
        self.assertIs(raised.exception.upstream_submitted, False)

    def test_known_not_submitted_receipt_is_retryable_without_a_new_chat_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"

            def handler(payload):
                payload["progress_callback"]("bootstrapping")
                raise ImageGenerationError(
                    "bootstrap connection timed out",
                    code="IMAGE_GENERATION_NOT_SUBMITTED",
                    provider_binding_id="cb_account_a",
                    provider_account_identity="account_opaque_a",
                    conversation_id="conversation-1",
                    parent_message_id="parent-before-request",
                    upstream_submitted=False,
                )

            service = self.make_service(path, handler)
            service.submit_generation(
                OWNER, client_task_id="not-submitted-task", prompt="cat",
                model="gpt-image-2", size=None, provider_binding_id="cb_account_a",
                provider_account_identity="account_opaque_a",
                client_conversation_id="workbench-conversation-1",
                conversation_id="conversation-1", parent_message_id="parent-before-request",
                retain_conversation=True,
            )
            task = wait_for_task(service, OWNER, "not-submitted-task", "error")

            self.assertEqual(task["error_code"], "RESULT_UNRECOVERABLE")
            self.assertEqual(task["error"], "bootstrap connection timed out")
            self.assertEqual(task["progress"], "bootstrapping")
            self.assertEqual(task["last_recovery_failure"]["phase"], "bootstrap")
            self.assertFalse(task["upstream_submission_started"])
            self.assertFalse(task["upstream_unfinished"])
            self.assertEqual(task["upstream_outcome"], "not_submitted")
            self.assertTrue(task["recovery_retryable"])
            self.assertFalse(task["recovery_requires_new_conversation"])

            restarted = self.make_service(path)
            reloaded = restarted.list_tasks(OWNER, ["not-submitted-task"])["items"][0]
            self.assertEqual(reloaded, task)
            with mock.patch("services.image_task_service.threading.Thread") as thread:
                resumed = restarted.resume_poll(
                    OWNER, "not-submitted-task", 30, "http://content-provider", True,
                )
            self.assertEqual(resumed, task)
            thread.assert_not_called()

    def test_restart_does_not_infer_non_submission_for_an_alternate_image_route(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            path.write_text(json.dumps({"tasks": [{
                "id": "codex-route-task", "owner_id": "owner-1", "status": "running",
                "mode": "generate", "model": "codex-gpt-image-2",
                "created_at": "2026-09-17 00:00:00", "updated_at": "2026-09-17 00:00:00",
                "provider_binding_id": "binding-1", "provider_account_identity": "account-1",
                "client_conversation_id": "client-1", "upstream_unfinished": True,
                "progress": "generating",
            }]}), encoding="utf-8")

            service = self.make_service(path)
            task = service.list_tasks(OWNER, ["codex-route-task"])["items"][0]

        self.assertEqual(task["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
        self.assertTrue(task["upstream_unfinished"])
        self.assertNotIn("upstream_submission_started", task)
        self.assertNotIn("recovery_retryable", task)

    def test_restart_recovers_a_persisted_pre_submission_boundary_without_polling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            path.write_text(json.dumps({"tasks": [{
                "id": "pre-submit-restart", "owner_id": "owner-1", "status": "running",
                "mode": "generate", "model": "gpt-image-2",
                "created_at": "2026-09-17 00:00:00", "updated_at": "2026-09-17 00:00:00",
                "provider_binding_id": "binding-1", "provider_account_identity": "account-1",
                "client_conversation_id": "client-1", "conversation_id": "conversation-1",
                "parent_message_id": "parent-1", "upstream_unfinished": True,
                "upstream_submission_started": False, "progress": "bootstrapping",
            }]}), encoding="utf-8")

            service = self.make_service(path)
            task = service.list_tasks(OWNER, ["pre-submit-restart"])["items"][0]
            with mock.patch("services.image_task_service.threading.Thread") as thread:
                resumed = service.resume_poll(
                    OWNER, "pre-submit-restart", 30, "http://content-provider", True,
                )

        self.assertEqual(task["error_code"], "RESULT_UNRECOVERABLE")
        self.assertEqual(task["progress"], "bootstrapping")
        self.assertFalse(task["upstream_submission_started"])
        self.assertFalse(task["upstream_unfinished"])
        self.assertTrue(task["recovery_retryable"])
        self.assertFalse(task["recovery_requires_new_conversation"])
        self.assertEqual(resumed, task)
        thread.assert_not_called()

    def test_four_unknown_generations_keep_slots_and_fifth_waits_until_terminal(self):
        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch(
            "services.image_task_service.config", image_account_concurrency=4
        ):
            calls = []
            def handler(payload):
                calls.append(payload["prompt"])
                error = RuntimeError("query round expired")
                error.code = "CONVERSATION_OUTCOME_UNKNOWN"
                error.conversation_id = payload["conversation_id"]
                raise error
            path = Path(tmp_dir) / "images.json"
            service = self.make_service(path, handler)
            for number in range(4):
                service.submit_generation(OWNER, client_task_id=str(number), prompt=str(number),
                    model="gpt-image-2", size=None, provider_binding_id="binding",
                    provider_account_identity="account", client_conversation_id=str(number),
                    conversation_id="chat-" + str(number), parent_message_id="parent", retain_conversation=True)
                wait_for_task(service, OWNER, str(number), "error")
            restarted = self.make_service(path, handler)
            self.assertEqual(sum(t["upstream_unfinished"] for t in restarted._tasks.values()), 4)
            restarted.submit_generation(OWNER, client_task_id="fifth", prompt="fifth",
                model="gpt-image-2", size=None, provider_binding_id="binding",
                provider_account_identity="account", client_conversation_id="fifth",
                conversation_id="chat-fifth", parent_message_id="parent", retain_conversation=True)
            time.sleep(0.05)
            self.assertEqual(calls, ["0", "1", "2", "3"])
            self.assertEqual(restarted.list_tasks(OWNER, ["fifth"])["items"][0]["status"], "queued")
            restarted._update_task("owner-1:0", status="success", upstream_unfinished=False)
            wait_for_task(restarted, OWNER, "fifth", "error")
            self.assertEqual(calls, ["0", "1", "2", "3", "fifth"])

    def test_unknown_receipt_is_not_deleted_by_retention_and_poll_cooldown_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "images.json"
            path.write_text(json.dumps({"tasks": [{"id":"old", "owner_id":"owner-1",
                "status":"error", "provider_binding_id":"binding", "provider_account_identity":"account",
                "client_conversation_id":"product", "conversation_id":"chat", "parent_message_id":"parent",
                "error_code":"CONVERSATION_OUTCOME_UNKNOWN", "updated_at":"2020-01-01 00:00:00",
                "next_poll_at":time.time()+600}]}))
            service = self.make_service(path)
            with mock.patch("services.image_task_service.threading.Thread") as thread:
                receipt = service.resume_poll(OWNER, "old")
                self.assertEqual(receipt["status"], "error")
                self.assertTrue(receipt["upstream_unfinished"])
                self.assertEqual(receipt["recovery_status"], "request_message_id_required")
                thread.assert_not_called()

    def test_request_owned_model_is_durable_and_cannot_change_on_duplicate_submit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            captured = []
            def handler(payload):
                captured.append(payload["upstream_model"])
                return {"data": [{"url": "https://example.test/image.png"}]}
            service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler)
            request = dict(client_task_id="b-instant", prompt="product", model="gpt-image-2", size=None,
                           upstream_model="gpt-5-6-instant")
            service.submit_generation(OWNER, **request)
            task = wait_for_task(service, OWNER, "b-instant", "success")
            self.assertEqual(task["upstream_model"], "gpt-5-6-instant")
            service.submit_generation(OWNER, **request)
            with self.assertRaisesRegex(ValueError, "different immutable request"):
                service.submit_generation(OWNER, **{**request, "upstream_model": "gpt-5.6-sol-wm"})
            self.assertEqual(captured, ["gpt-5-6-instant"])

    def test_image_task_creates_an_independent_session_on_the_bound_account(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            captured = {}

            def handler(payload):
                captured.update(payload)
                return {
                    "data": [{"url": "http://example.test/image.png"}],
                    "_provider_binding_id": "cb_account_a",
                    "_provider_account_identity": "account_opaque_a",
                    "_conversation_id": "conversation-1",
                    "_parent_message_id": "message-2",
                }

            service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler)
            service.submit_generation(
                OWNER,
                client_task_id="bound-task",
                prompt="continue",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
                provider_binding_id="cb_account_a",
                provider_account_identity="account_opaque_a",
                client_conversation_id="workbench-conversation-1",
                retain_conversation=True,
            )

            task = wait_for_task(service, OWNER, "bound-task", "success")

            self.assertEqual(captured["provider_binding_id"], "cb_account_a")
            self.assertEqual(captured["provider_account_identity"], "account_opaque_a")
            self.assertEqual(captured["client_conversation_id"], "workbench-conversation-1")
            self.assertEqual(captured["conversation_id"], "")
            self.assertEqual(captured["parent_message_id"], "")
            self.assertTrue(captured["retain_conversation"])
            self.assertEqual(task["provider_binding_id"], "cb_account_a")
            self.assertEqual(task["provider_account_identity"], "account_opaque_a")
            self.assertEqual(task["client_conversation_id"], "workbench-conversation-1")
            self.assertEqual(task["image_session_id"], "conversation-1")
            self.assertEqual(task["image_session_parent_id"], "message-2")
            self.assertNotIn("conversation_id", task)
            self.assertNotIn("parent_message_id", task)

    def make_service(self, path: Path, handler=None) -> ImageTaskService:
        return ImageTaskService(
            path,
            generation_handler=handler or (lambda _payload: {"data": [{"url": "http://example.test/image.png"}]}),
            edit_handler=handler or (lambda _payload: {"data": [{"url": "http://example.test/edit.png"}]}),
            retention_days_getter=lambda: 30,
        )

    def test_public_result_stage_never_publishes_private_assets_or_hides_stops(self):
        cached = {"output_ref": "private-output-ref", "coverage": {"file_ids": ["private-asset-id"]}}
        recovering = {"status": "error", "error_code": "CONVERSATION_OUTCOME_UNKNOWN",
                      "conversation_id": "original", "request_message_id": "request",
                      "result_file_ids": ["private-asset-id"], "_pending_image_output": cached}
        cases = [
            ({"status": "queued"}, "queued"),
            ({"status": "running", "upstream_submission_started": False}, "preparing"),
            ({"status": "running", "upstream_outcome": "unknown", "conversation_id": "original"}, "submission_unconfirmed"),
            ({"status": "running", "upstream_submission_started": True}, "submitted"),
            ({"status": "running", "_pending_image_result_ids": {"file_ids": ["private-asset-id"]}}, "assets_discovered"),
            ({"status": "running", "result_file_ids": ["private-asset-id"]}, "assets_discovered"),
            (recovering, "downloaded_waiting_original_confirmation"),
            ({**recovering, "_recovery_paused": True}, "needs_attention"),
            ({**recovering, "_recovery_suppressed": True}, "needs_attention"),
            ({**recovering, "error_code": "content_policy_violation"}, "needs_attention"),
            ({"status": "success", "data": []}, "needs_attention"),
            ({"status": "success", "data": [{"url": "confirmed-result"}]}, "result_ready"),
        ]
        for receipt, expected in cases:
            with self.subTest(expected=expected, receipt=receipt):
                public = _public_task(receipt)
                self.assertEqual(public["result_stage"], expected)
                self.assertNotIn("private-", json.dumps(public))

    def test_first_qualified_assets_observation_is_atomic_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path, error_code="CONVERSATION_OUTCOME_UNKNOWN", upstream_unfinished=True)
            service = self.make_service(path)
            key = "owner-1:policy-task"
            service._update_task(key, _pending_image_result_ids={"file_ids": ["pending"]})
            self.assertNotIn("_first_qualified_image_assets_observed_at", service._tasks[key])
            service._update_task(key, result_file_ids=["original-file"], upstream_unfinished=True)
            self.assertNotIn("_first_qualified_image_assets_observed_at", service._tasks[key])
            # Merely ending an old attempt is not a newly observed asset read.
            service._update_task(key, upstream_unfinished=False)
            self.assertNotIn("_first_qualified_image_assets_observed_at", service._tasks[key])
            service._update_task(key, result_file_ids=["original-file"], result_sediment_ids=["original-file", "sediment"])
            first = service._tasks[key]["_first_qualified_image_assets_observed_at"]
            self.assertGreater(first, 0)
            self.assertEqual(service._tasks[key]["_first_qualified_image_asset_id_count"], 2)
            restarted = self.make_service(path)
            restarted._update_task(key, result_file_ids=["original-file"], result_sediment_ids=["sediment"])
            task = restarted._tasks[key]
            self.assertEqual(task["_first_qualified_image_assets_observed_at"], first)
            observations = [e for e in task["_execution_timeline"] if e["stage"] == "qualified_image_assets_observed"]
            self.assertEqual(observations, [{"stage": "qualified_image_assets_observed", "at": first, "image_count": 2}])
            self.assertNotIn("original-file", json.dumps(observations))

    def test_historical_assets_without_observation_keep_unknown_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path, error_code="CONVERSATION_OUTCOME_UNKNOWN", result_file_ids=["old-file"])
            service = self.make_service(path)
            service._update_task("owner-1:policy-task", progress="receiving_image")
            restarted = self.make_service(path)
            self.assertNotIn("_first_qualified_image_assets_observed_at", restarted._tasks["owner-1:policy-task"])

    def test_progress_update_reads_current_original_without_loading_or_rewriting_history(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path)
            service = self.make_service(path)
            key = "owner-1:policy-task"
            with service.store.transaction() as db:
                original = service.store.read_receipt(db, "image", "owner-1", "policy-task")
                original["external_revision"] = "newer-durable-value"
                service.store.write_receipt(db, "image", "owner-1", "policy-task", original)
                history = {"id": "history", "owner_id": "owner-1", "status": "success",
                           "data": [{"b64_json": "unrelated-historical-image" + "a" * 1_000_000}]}
                service.store.write_receipt(db, "image", "owner-1", "history", history)
                raw_history = db.execute("SELECT receipt FROM image_requests WHERE task_key='owner-1:history'").fetchone()[0]
            loads = json.loads
            def bounded_load(raw, *args, **kwargs):
                self.assertNotIn("unrelated-historical-image", raw)
                return loads(raw, *args, **kwargs)
            with mock.patch("services.image_task_service.json.loads", side_effect=bounded_load), \
                    mock.patch.object(service.store, "write_receipt", wraps=service.store.write_receipt) as writes:
                service._update_task(key, progress="receiving_image")
                self.assertEqual(writes.call_count, 1)
            with service.store.connect() as db:
                updated = service.store.read_receipt(db, "image", "owner-1", "policy-task")
                self.assertEqual(updated["external_revision"], "newer-durable-value")
                self.assertEqual(updated["progress"], "receiving_image")
                self.assertEqual(db.execute("SELECT receipt FROM image_requests WHERE task_key='owner-1:history'").fetchone()[0], raw_history)
            # A missing durable receipt must not be resurrected from _tasks.
            with service.store.transaction() as db:
                db.execute("DELETE FROM image_requests WHERE task_key=?", (key,))
            service._update_task(key, progress="late-callback")
            self.assertNotIn(key, service._tasks)
            with service.store.connect() as db:
                self.assertIsNone(service.store.read_receipt(db, "image", "owner-1", "policy-task"))

    def test_nested_progress_update_preserves_outer_transaction_changes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path)
            service = self.make_service(path)
            with service._transaction():
                service._tasks["owner-1:other"] = {"id": "other", "owner_id": "owner-1", "status": "success"}
                service._update_task("owner-1:policy-task", progress="receiving_image")
            with service.store.connect() as db:
                self.assertEqual(service.store.read_receipt(db, "image", "owner-1", "other")["status"], "success")
                self.assertEqual(service.store.read_receipt(db, "image", "owner-1", "policy-task")["progress"], "receiving_image")

    def test_active_attempt_budget_starts_after_account_slot_wait_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            entered = threading.Event()

            def handler(payload):
                entered.set()
                callback = payload["progress_callback"]
                callback.start_active_attempt()
                self.assertGreater(callback.active_deadline_at, time.time())
                callback.record_submission_started()
                error = RuntimeError("post response interrupted")
                error.code = "CONVERSATION_OUTCOME_UNKNOWN"
                error.conversation_id = "conversation-1"
                error.request_message_id = callback.request_message_id
                error.upstream_submitted = True
                raise error

            service = self.make_service(path, handler)
            with service._transaction():
                for index in range(10):
                    service._tasks[f"owner-1:occupant-{index}"] = {
                        "id": f"occupant-{index}", "owner_id": "owner-1", "status": "error",
                        "provider_account_identity": "account-1", "upstream_unfinished": True,
                        "created_at": "2026-09-17 00:00:00", "updated_at": "2026-09-17 00:00:00",
                    }
                service._save_locked()
            queued_at = time.time()
            service.submit_generation(
                OWNER, client_task_id="budget-task", prompt="cat", model="gpt-image-2", size=None,
                provider_binding_id="binding-1", provider_account_identity="account-1",
                client_conversation_id="client-1", retain_conversation=True,
            )
            time.sleep(0.05)
            queued = service.list_tasks(OWNER, ["budget-task"])["items"][0]
            self.assertNotIn("active_attempt_started_at", queued)
            for index in range(10):
                service._update_task(f"owner-1:occupant-{index}", upstream_unfinished=False)
            self.assertTrue(entered.wait(1))
            task = wait_for_task(service, OWNER, "budget-task", "error")

            self.assertGreaterEqual(task["active_attempt_started_at"], queued_at + 0.04)
            self.assertAlmostEqual(
                task["active_attempt_deadline_at"] - task["active_attempt_started_at"],
                300.0,
                delta=0.01,
            )
            restarted = self.make_service(path)
            reloaded = restarted.list_tasks(OWNER, ["budget-task"])["items"][0]
            self.assertEqual(reloaded["active_attempt_started_at"], task["active_attempt_started_at"])
            self.assertEqual(reloaded["active_attempt_deadline_at"], task["active_attempt_deadline_at"])

    def test_local_pacing_credit_is_persisted_without_resetting_active_budget(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tasks.json"
            def handler(payload):
                callback = payload["progress_callback"]
                callback.start_active_attempt()
                callback.record_local_pacing_wait(60)
                callback.record_local_pacing_wait(420)
                return {"data": [{"url": "http://example.test/image.png"}]}
            service = self.make_service(path, handler)
            service.submit_generation(OWNER, client_task_id="paced", prompt="cat", model="gpt-image-2", size=None)
            wait_for_task(service, OWNER, "paced", "success")
            restarted = self.make_service(path)
            with restarted._transaction():
                task = restarted._tasks["owner-1:paced"]
                self.assertEqual(task["active_local_pacing_wait_secs"], 480)
                self.assertAlmostEqual(task["active_attempt_deadline_at"] - task["active_attempt_started_at"], 780)
    def test_bound_account_and_chat_lock_waits_precede_active_budget(self):
        marks = {}

        def progress_callback(_step):
            return None

        def start_active_attempt():
            marks["active_start"] = time.time()
            progress_callback.active_deadline_at = marks["active_start"] + 300

        progress_callback.start_active_attempt = start_active_attempt
        progress_callback.active_deadline_at = None
        request = ConversationRequest(
            model="gpt-image-2", prompt="cat", provider_binding_id="binding-1",
            provider_account_identity="account-1", client_conversation_id="client-1",
            retain_conversation=True, progress_callback=progress_callback,
        )

        def acquire(*_args, **_kwargs):
            time.sleep(0.02)
            marks["account_ready"] = time.time()
            return "token"

        class DelayedLock:
            def __enter__(self):
                time.sleep(0.02)
                marks["chat_lock_ready"] = time.time()

            def __exit__(self, *_args):
                return False

        class Backend:
            image_submission_started = True
            image_request_message_id = "request-1"

            def __init__(self, access_token=None):
                self.access_token = access_token

            def get_conversation_parent_message_id(self, _conversation_id):
                return "assistant-1"

            def close(self):
                return None

        def stream_outputs(_backend, stream_request, _index, _total):
            self.assertGreaterEqual(marks["active_start"], marks["account_ready"])
            self.assertGreaterEqual(marks["active_start"], marks["chat_lock_ready"])
            self.assertAlmostEqual(
                stream_request.progress_callback.active_deadline_at - marks["active_start"],
                300,
                delta=0.01,
            )
            return iter([
                ImageOutput(
                    kind="result", model="gpt-image-2", index=1, total=1,
                    data=[{"b64_json": "image"}], conversation_id="conversation-1",
                ),
            ])

        with (
            mock.patch("services.protocol.conversation.account_service.get_bound_account_identity", return_value="account-1"),
            mock.patch("services.protocol.conversation.account_service.acquire_bound_image_access_token", side_effect=acquire),
            mock.patch("services.protocol.conversation.account_service.get_account", return_value={"email": "account@example.test"}),
            mock.patch("services.protocol.conversation.account_service.conversation_binding_lock", return_value=DelayedLock()),
            mock.patch("services.protocol.conversation.account_service.mark_image_result"),
            mock.patch("services.protocol.conversation.OpenAIBackendAPI", Backend),
            mock.patch("services.protocol.conversation.stream_image_outputs", side_effect=stream_outputs),
        ):
            outputs = _generate_bound_single_image(request, 1, 1)

        self.assertEqual(outputs[0].conversation_id, "conversation-1")

    def test_actual_submission_replaces_stale_not_submitted_outcome(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            observed = {}

            def handler(payload):
                # The same request previously returned to admission before
                # sending because its original conversation was unreadable.
                service._update_task(
                    "owner-1:resumed-send", upstream_outcome="not_submitted",
                    upstream_submission_started=False,
                    error_code="IMAGE_THREAD_PREVIOUS_UNCONFIRMED",
                    waiting={"reason": "archive_restore"}, recovery_retryable=True,
                )
                callback = payload["progress_callback"]
                callback.record_submission_started()
                observed.update(service.list_tasks(OWNER, ["resumed-send"])["items"][0])
                error = ConnectionError("response lost after submission")
                error.code = "CONVERSATION_OUTCOME_UNKNOWN"
                error.upstream_submitted = True
                raise error

            service = self.make_service(path, handler)
            service.submit_generation(
                OWNER, client_task_id="resumed-send", prompt="cat", model="gpt-image-2", size=None,
                provider_binding_id="binding-1", provider_account_identity="account-1",
                client_conversation_id="client-1", retain_conversation=True,
            )
            failed = wait_for_task(service, OWNER, "resumed-send", "error")
            self.assertEqual(observed["upstream_outcome"], "unknown")
            self.assertTrue(observed["upstream_submission_started"])
            self.assertFalse(observed.get("error_code"))
            self.assertFalse(observed.get("waiting"))
            self.assertFalse(observed.get("recovery_retryable"))
            self.assertEqual(failed["upstream_outcome"], "unknown")
            self.assertEqual(failed["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
            restarted = self.make_service(path)
            reloaded = restarted.list_tasks(OWNER, ["resumed-send"])["items"][0]
            self.assertEqual(reloaded["upstream_outcome"], "unknown")
            self.assertTrue(reloaded["upstream_submission_started"])

    def test_generated_result_download_failure_resumes_download_without_poll_or_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"

            def handler(payload):
                callback = payload["progress_callback"]
                callback.start_active_attempt()
                callback.record_conversation_id("conversation-1")
                callback.record_submission_started()
                callback.record_result_ids(["file-generated"], [])
                error = ConnectionError("download proxy token=secret")
                error.code = "CONVERSATION_OUTCOME_UNKNOWN"
                error.conversation_id = "conversation-1"
                error.request_message_id = callback.request_message_id
                error.upstream_submitted = True
                raise error

            service = self.make_service(path, handler)
            service.submit_generation(
                OWNER, client_task_id="download-task", prompt="cat", model="gpt-image-2", size=None,
                provider_binding_id="binding-1", provider_account_identity="account-1",
                client_conversation_id="client-1", retain_conversation=True,
            )
            failed = wait_for_task(service, OWNER, "download-task", "error")
            self.assertEqual(failed["recovery_phase"], "download_image_result")
            self.assertEqual(failed["recovery_error_code"], "RECOVERY_TRANSPORT_FAILED")
            self.assertEqual(failed["upstream_outcome"], "generated")
            self.assertFalse(failed["upstream_unfinished"])
            self.assertNotIn("token=secret", failed["error"])
            first_assets_observed = service._tasks["owner-1:download-task"]["_first_qualified_image_assets_observed_at"]
            self.assertGreater(first_assets_observed, 0)
            service._update_task("owner-1:download-task", active_attempt_deadline_at=time.time() - 1)
            failed = service.list_tasks(OWNER, ["download-task"])["items"][0]
            active_started_at = failed["active_attempt_started_at"]
            active_deadline_at = failed["active_attempt_deadline_at"]
            service = self.make_service(path)
            from services.pool_admission import PoolAdmission
            # A restarted scheduler must recover the original captured result,
            # including its new atomic read claim, without a client resubmit.
            service.admission = PoolAdmission(service.store, None)
            service.admission.recoveries["image"] = lambda owner, task_id: service.resume_poll({"id": owner, "role": "user"}, task_id)

            class DownloadBackend:
                polls = 0
                reads = 0

                def __init__(self, access_token=None, proxy_url=None):
                    self.access_token = access_token

                def _get_conversation(self, _conversation_id):
                    type(self).reads += 1
                    raise AssertionError("download recovery must not reread generation state")

                def _poll_image_results(self, *_args, **_kwargs):
                    type(self).polls += 1
                    raise AssertionError("download recovery must not poll generation")

                def resolve_conversation_image_urls(self, _conversation_id, file_ids, sediment_ids, **_kwargs):
                    if file_ids != ["file-generated"] or sediment_ids != []:
                        raise AssertionError("download recovery changed the persisted generated result IDs")
                    return ["https://provider.test/generated.png"]

                def download_image_bytes(self, _urls):
                    return [b"generated"]

                def get_conversation_parent_message_id(self, _conversation_id):
                    return "result-parent"

                def close(self):
                    return None

            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="token"),
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", DownloadBackend),
            ):
                service.admission.recover_one()
                succeeded = wait_for_task(service, OWNER, "download-task", "success")
                # Success commits before the recovery worker clears its claim.
                # Keep fixtures alive through that cleanup instead of removing
                # the temporary database underneath the worker.
                for worker in threading.enumerate():
                    if worker.name == "image-resume-download-task":
                        worker.join(timeout=2)
                        self.assertFalse(worker.is_alive())

            self.assertEqual(succeeded["image_session_parent_id"], "result-parent")
            self.assertEqual(succeeded["active_attempt_started_at"], active_started_at)
            self.assertEqual(succeeded["active_attempt_deadline_at"], active_deadline_at)
            self.assertEqual(DownloadBackend.polls, 0)
            self.assertEqual(DownloadBackend.reads, 0)
            self.assertEqual(service._tasks["owner-1:download-task"]["_first_qualified_image_assets_observed_at"], first_assets_observed)

    def test_complete_final_recovery_reuses_proof_but_tool_leaf_keeps_publication_check(self):
        from copy import deepcopy
        from services.config import config
        for case in ("complete", "new_second_image", "partial_download", "final_external_successor", "download_failure", "pending_missing", "parent_changed", "running", "read_429", "post_download_changed", "settle_failure"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "tasks.json"
                write_policy_task(path, error_code="CONVERSATION_OUTCOME_UNKNOWN", error="",
                    upstream_unfinished=True, _image_thread={"protocol": "image-thread-v1"},
                    _image_thread_request_parent="anchor",
                    **({"_pending_image_result_ids": {"file_ids": ["missing" if case == "pending_missing" else "original-image"], "sediment_ids": []}}
                       if case in ("pending_missing", "new_second_image", "settle_failure") else {}))
                service = self.make_service(path)
                service._update_task("owner-1:policy-task", _image_thread={"id": "original-thread"},
                                     _image_thread_request_parent="anchor")
                document = {"conversation_id": "conversation-1", "current_node": "final", "mapping": {
                    "original-request": {"parent": "anchor", "message": {
                        "id": "original-request", "author": {"role": "user"}}},
                    "image": {"parent": "original-request", "message": {
                        "id": "image", "author": {"role": "tool"}, "status": "finished_successfully",
                        "metadata": {"async_task_type": "image_gen"},
                        "content": {"parts": [{"content_type": "image_asset_pointer", "asset_pointer": "file-service://original-image"}]}}},
                    "final": {"parent": "image", "message": {"id": "final", "author": {"role": "assistant"},
                        "status": "finished_successfully", "end_turn": True}}}}
                if case == "parent_changed": document["mapping"]["original-request"]["parent"] = "other"
                if case in ("running", "settle_failure"): document["mapping"]["final"]["message"]["status"] = "in_progress"
                if case == "post_download_changed":
                    # A finished tool leaf is weaker than a complete assistant
                    # final: keep the original late-drift rejection regression.
                    document['mapping']['code'] = {'parent': 'original-request', 'message': {
                        'id': 'code', 'author': {'role': 'assistant'}, 'status': 'finished_successfully'}}
                    document['mapping']['image']['parent'] = 'code'
                    del document['mapping']['final']
                    document['current_node'] = 'image'
                expected_files = ["original-image"]
                if case in ("new_second_image", "partial_download"):
                    expected_files.append("second-image")
                    document["mapping"]["image"]["message"]["content"]["parts"].append(
                        {"content_type": "image_asset_pointer", "asset_pointer": "file-service://second-image"})
                calls = {"read": 0, "poll": 0, "download": 0}
                class Backend(OpenAIBackendAPI):
                    def __init__(self, **kwargs): pass
                    def _get_conversation(self, cid):
                        self_test.assertEqual(cid, "conversation-1")
                        calls["read"] += 1
                        if case == "read_429":
                            error = RuntimeError("rate limited")
                            error.status_code = 429
                            raise error
                        result = deepcopy(document)
                        if case == "post_download_changed" and calls["read"] > 1:
                            result["mapping"]["original-request"]["parent"] = "other"
                        if case == "settle_failure" and calls["read"] > 1:
                            result["mapping"]["final"]["message"].update(status="finished_successfully",
                                content={"content_type": "text", "parts": ["Something went wrong while generating your image."]})
                        return result
                    def _poll_image_results(self, cid, timeout, **kwargs):
                        calls["poll"] += 1
                        self_test.assertEqual(kwargs["initial_document"], document)
                        raise ImagePollTimeoutError("not confirmed", cid)
                    def resolve_conversation_image_urls(self, cid, files, sediments, **kwargs):
                        self_test.assertEqual((cid, files, sediments), ("conversation-1", expected_files, []))
                        if case == 'partial_download' and calls['download'] == 0: files = files[:1]
                        return ["https://original.test/" + f + ".png" for f in files]
                    def download_image_bytes(self, urls):
                        calls["download"] += 1
                        if case == 'download_failure': raise TimeoutError('original attachment unavailable')
                        if case == 'final_external_successor':
                            document['mapping']['external'] = {'parent': 'final', 'message': {
                                'id': 'external', 'author': {'role': 'user'}}}
                            document['current_node'] = 'external'
                        return [f.encode() for f in expected_files[:len(urls)]]
                    def close(self): pass
                self_test = self
                with (
                    mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                    mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="token"),
                    mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                    mock.patch("services.openai_backend_api.OpenAIBackendAPI", Backend),
                    mock.patch("services.protocol.conversation.format_image_result", return_value={"data": [{"url": "saved"}]}) as publish,
                    mock.patch("services.protocol.conversation._observe_image_terminal") as observe,
                    mock.patch("services.image_task_service.time.sleep") as sleep,
                    mock.patch.dict(config.data, {"image_settle_enabled": True, "image_settle_secs": 30}),
                ):
                    service._run_resume_poll("owner-1:policy-task", "conversation-1", 60,
                        "http://localhost", OWNER, "generate", "gpt-image-2", False, False)
                    if case == 'partial_download':
                        incomplete = service._tasks['owner-1:policy-task']
                        self.assertEqual(incomplete['status'], 'error')
                        self.assertFalse(incomplete.get('data'))
                        self.assertEqual(incomplete['result_file_ids'], expected_files)
                        self.assertTrue(incomplete.get('_pending_image_output'))
                        publish.assert_not_called()
                        # Same original receipt, same assets. The incomplete
                        # private cache cannot masquerade as a complete result.
                        service._run_resume_poll('owner-1:policy-task', 'conversation-1', 60,
                            'http://localhost', OWNER, 'generate', 'gpt-image-2', False, False)
                row = service._tasks["owner-1:policy-task"]
                if case == 'partial_download':
                    self.assertEqual(calls, {'read': 2, 'poll': 0, 'download': 2})
                    observe.assert_called_once_with()
                    sleep.assert_not_called()
                elif case in ("complete", "new_second_image", "final_external_successor", "download_failure", "post_download_changed"):
                    self.assertEqual(calls, {"read": 2 if case == 'post_download_changed' else 1, "poll": 0, "download": 1})
                    observe.assert_called_once_with()
                    sleep.assert_not_called()
                else:
                    self.assertEqual(calls, {"read": 2 if case in ("pending_missing", "settle_failure") else 1, "poll": 0 if case in ("read_429", "settle_failure") else 1, "download": 0})
                    if case in ("pending_missing", "settle_failure"): sleep.assert_called_once_with(30)
                    else: sleep.assert_not_called()
                    observe.assert_not_called()
                if case in ("complete", "new_second_image", "partial_download", "final_external_successor"):
                    self.assertEqual(row["status"], "success")
                    self.assertEqual(row["parent_message_id"], "final")
                    self.assertEqual(row["result_file_ids"], expected_files)
                    publish.assert_called_once()
                    self.assertEqual(len(publish.call_args.args[0]), len(expected_files))
                else:
                    self.assertEqual(row["status"], "error")
                    publish.assert_not_called()
                    self.assertFalse(row.get("data"))
                    if case == "post_download_changed": self.assertTrue(row["_pending_image_output"])

    def test_downloaded_original_is_private_and_reused_after_confirmation_failure(self):
        from services.request_context import executing
        for alteration in ("unchanged", "asset_drift", "corrupt", "confirmation_still_fails"):
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "image_tasks.json"
                write_policy_task(path, error_code="CONVERSATION_OUTCOME_UNKNOWN", error="",
                                  result_file_ids=["original-file"], result_sediment_ids=[],
                                  _image_thread={"protocol": "image-thread-v1"})
                service = self.make_service(path)
                service._update_task("owner-1:policy-task", _image_thread={"id": "original-thread"})
                calls = {"download": 0, "resolve": 0, "confirm": 0}
                fail_confirmation = [True]
                timing_context = mock.Mock(kind="timing-fixture")

                class Backend:
                    def __init__(self, **_kwargs): pass
                    def resolve_conversation_image_urls(self, *_args, **_kwargs):
                        calls["resolve"] += 1
                        return ["https://provider.test/original.png"]
                    def download_image_bytes(self, _urls):
                        calls["download"] += 1
                        return [b"original-image-bytes"]
                    def _get_conversation(self, _conversation_id): return {}
                    def close(self): pass

                def confirm(*_args, **_kwargs):
                    calls["confirm"] += 1
                    if fail_confirmation[0]:
                        error = RuntimeError("read rate limit")
                        error.status_code = 429
                        raise error
                    return "verified-parent"

                def resume(current):
                    with executing(timing_context):
                        current._run_resume_poll("owner-1:policy-task", "conversation-1", 60,
                                                 "http://localhost", OWNER, "generate", "gpt-image-2", False, False)

                with (
                    mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                    mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="token"),
                    mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                    mock.patch("services.openai_backend_api.OpenAIBackendAPI", Backend),
                    mock.patch("services.image_task_service.finished_parent", side_effect=confirm),
                    mock.patch("services.protocol.conversation.format_image_result", return_value={"data": [{"url": "http://localhost/verified.png"}]}) as publish,
                ):
                    resume(service)
                    self.assertEqual(calls["download"], 1)
                    publish.assert_not_called()
                    public = service.list_tasks(OWNER, ["policy-task"])["items"][0]
                    self.assertFalse(public.get("data"))
                    self.assertNotIn("_pending_image_output", public)
                    cached = service._tasks["owner-1:policy-task"]["_pending_image_output"]
                    with service.store.output_file(cached["output_ref"]) as handle:
                        self.assertIn(b"b64_json", handle.read())
                    service = self.make_service(path)
                    if alteration == "asset_drift":
                        service._update_task("owner-1:policy-task", result_file_ids=["different-file"])
                    elif alteration == "corrupt":
                        with service.store.output_file(cached["output_ref"], append=True) as handle:
                            handle.write(b"corrupt")
                    elif alteration == "confirmation_still_fails":
                        resume(service)
                        self.assertEqual(calls["download"], 1)
                        publish.assert_not_called()
                    fail_confirmation[0] = False
                    resume(service)
                    succeeded = service.list_tasks(OWNER, ["policy-task"])["items"][0]
                    self.assertEqual(succeeded["status"], "success")
                    self.assertEqual(succeeded["image_session_parent_id"], "verified-parent")
                    self.assertEqual(calls["download"], 2 if alteration in {"asset_drift", "corrupt"} else 1)
                    self.assertEqual(calls["resolve"], calls["download"])
                    self.assertEqual(calls["confirm"], 3 if alteration == "confirmation_still_fails" else 2)
                    publish.assert_called_once()
                    self.assertIsNone(service._tasks["owner-1:policy-task"]["_pending_image_output"])
                    expected_reuses = 2 if alteration == "confirmation_still_fails" else 1 if alteration == "unchanged" else 0
                    self.assertEqual(timing_context.record_stage.call_args_list,
                                     [mock.call("attachment_download_cache_reused", image_count=1)] * expected_reuses)

    def test_pending_image_ids_survive_restart_without_skipping_settle_or_releasing_capacity(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            def handler(payload):
                callback = payload["progress_callback"]
                callback.record_submission_started()
                callback.record_conversation_id("conversation-1")
                callback.record_pending_result_ids(["pending-file"], [])
                error = ImagePollTimeoutError("settle budget exhausted", "conversation-1")
                error.code = "CONVERSATION_OUTCOME_UNKNOWN"
                error.upstream_submitted = True
                raise error
            service = self.make_service(path, handler)
            service.submit_generation(
                OWNER, client_task_id="pending-task", prompt="cat", model="gpt-image-2", size=None,
                provider_binding_id="binding-1", provider_account_identity="account-1",
                client_conversation_id="client-1", retain_conversation=True,
            )
            failed = wait_for_task(service, OWNER, "pending-task", "error")
            self.assertTrue(failed["upstream_unfinished"])
            self.assertEqual(failed["recovery_phase"], "read_image_request")
            service = self.make_service(path)
            self.assertEqual(service._tasks["owner-1:pending-task"]["_pending_image_result_ids"],
                             {"file_ids": ["pending-file"], "sediment_ids": []})
            from services.pool_admission import image_generation_active
            self.assertTrue(image_generation_active("image", service._tasks["owner-1:pending-task"], False))
            calls = []
            from services.openai_backend_api import OpenAIBackendAPI as RealBackend
            from services.config import config
            original_message = service._tasks["owner-1:pending-task"]["request_message_id"]
            observations = []
            class PendingBackend(RealBackend):
                reads = 0
                def __init__(self, access_token=None, proxy_url=None):
                    pass
                def _get_conversation(self, _conversation_id):
                    PendingBackend.reads += 1
                    observations.append("read")
                    return {"observation": PendingBackend.reads, "current_node": "image", "mapping": {
                        original_message: {"message": {"author": {"role": "user"}}},
                        "image": {"parent": original_message, "message": {
                            "author": {"role": "tool"}, "metadata": {"async_task_type": "image_gen"},
                            "content": {"parts": ["file-service://pending-file", "file-service://second-file"]}}}}}
                def _poll_image_results(self, _conversation_id, _timeout, **kwargs):
                    if not kwargs.get("require_fresh_result_ids"):
                        raise AssertionError("pending IDs must be observed again")
                    if not kwargs.get("initial_document"):
                        raise AssertionError("reuse this recovery attempt's fresh snapshot")
                    self_test.assertEqual(kwargs["initial_document"]["observation"], PendingBackend.reads)
                    self_test.assertEqual(observations[-2:], ["sleep", "read"])
                    calls.append(("poll", kwargs["initial_file_ids"]))
                    return super()._poll_image_results(_conversation_id, _timeout, **kwargs)
                def resolve_conversation_image_urls(self, _conversation_id, files, sediments, **kwargs):
                    calls.append(("resolve", files))
                    return ["https://provider.test/" + file + ".png" for file in files]
                def download_image_bytes(self, _urls):
                    calls.append(("download", []))
                    return [b"image-result" for _ in _urls]
                def get_conversation_parent_message_id(self, _conversation_id):
                    return "result-parent"
                def close(self):
                    pass
            self_test = self
            original_sleep = time.sleep
            def record_sleep(seconds):
                if threading.current_thread().name.startswith("image-resume-"):
                    observations.append("sleep")
                original_sleep(seconds)
            with (
                mock.patch("services.image_task_service.time.sleep", side_effect=record_sleep),
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="token"),
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", PendingBackend),
                mock.patch.dict(config.data, {"image_settle_enabled": True, "image_settle_secs": 0.5,
                    "image_check_before_hit_enabled": True, "image_poll_initial_wait_secs": 10}),
            ):
                with mock.patch.dict(config.data, {"image_settle_secs": 30}), mock.patch(
                    "services.image_task_service.time.sleep", side_effect=lambda seconds: observations.append("sleep")
                ) as sleep:
                    service._run_resume_poll("owner-1:pending-task", "conversation-1", 5,
                                             "http://provider", OWNER, "generate", "gpt-image-2", False, False)
                    sleep.assert_called_once_with(5)
                    self.assertEqual(PendingBackend.reads, 1)
                    self.assertEqual(observations, ["read", "sleep"])
                    self.assertEqual(calls, [])
                    self.assertEqual(service._tasks["owner-1:pending-task"]["status"], "error")
                    self.assertEqual(service._tasks["owner-1:pending-task"]["_pending_image_result_ids"],
                                     {"file_ids": ["pending-file"], "sediment_ids": []})
                service._update_task("owner-1:pending-task", next_poll_at=0)
                service.resume_poll(OWNER, "pending-task", 5, "http://provider")
                still_pending = wait_for_task(service, OWNER, "pending-task", "error")
                for worker in threading.enumerate():
                    if worker.name == "image-resume-pending-task":
                        worker.join(timeout=2)
                        self.assertFalse(worker.is_alive())
                self.assertEqual(calls, [("poll", ["pending-file"])])
                self.assertEqual(PendingBackend.reads, 3)
                self.assertTrue(still_pending["upstream_unfinished"])
                service = self.make_service(path)
                service._update_task("owner-1:pending-task", next_poll_at=0,
                                     _attempt_finished_at=time.time(), error_code="RESULT_UNRECOVERABLE")
                from services.pool_admission import PoolAdmission
                service.admission = PoolAdmission(service.store, None)
                service.admission.recoveries["image"] = lambda owner, task_id: service.resume_poll(
                    {"id": owner, "role": "user"}, task_id, 5, "http://provider")
                service.admission.recover_one()
                succeeded = wait_for_task(service, OWNER, "pending-task", "success")
                # A visible success precedes the admission worker's final claim
                # release. Keep its SQLite directory until that worker exits.
                for worker in threading.enumerate():
                    if worker.name == "image-resume-pending-task":
                        worker.join(timeout=2)
                        self.assertFalse(worker.is_alive())
            self.assertEqual(calls[1], ("poll", ["pending-file", "second-file"]))
            self.assertEqual(calls[2][0], "resolve")
            self.assertEqual(calls[3][0], "download")
            self.assertEqual(PendingBackend.reads, 5)
            self.assertIsNone(service._tasks["owner-1:pending-task"].get("_pending_image_result_ids"))
            self.assertEqual(succeeded["image_session_parent_id"], "result-parent")

    def test_successful_handler_clears_pending_observations(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            def handler(payload):
                payload["progress_callback"].record_pending_result_ids(["observed-file"], [])
                return {"data": [{"url": "https://provider.test/finished.png"}]}
            service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler)
            service.submit_generation(OWNER, client_task_id="pending-success", prompt="cat",
                                      model="gpt-image-2", size=None)
            wait_for_task(service, OWNER, "pending-success", "success")
            self.assertIsNone(service._tasks["owner-1:pending-success"].get("_pending_image_result_ids"))

    def test_captured_result_failure_keeps_cause_without_claiming_download_failed(self):
        for error_type, expected in (
            (ImageStreamHardTimeoutError, "RECOVERY_TIMED_OUT"),
            (ImageActiveDeadlineExceeded, "RECOVERY_TIMED_OUT"),
            (RuntimeError, "RECOVERY_RESULT_INCOMPLETE"),
        ):
            with self.subTest(error_type=error_type), tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "image_tasks.json"
                def handler(payload):
                    callback = payload["progress_callback"]
                    callback.start_active_attempt()
                    callback.record_submission_started()
                    callback.record_conversation_id("conversation-1")
                    callback.record_result_ids(["file-generated"], [])
                    error = error_type("Authorization: bearer-secret URL=https://private.test/signed")
                    error.code = "CONVERSATION_OUTCOME_UNKNOWN"
                    error.conversation_id = "conversation-1"
                    error.request_message_id = callback.request_message_id
                    error.upstream_submitted = True
                    raise error
                service = self.make_service(path, handler)
                service.submit_generation(
                    OWNER, client_task_id="captured-task", prompt="cat", model="gpt-image-2", size=None,
                    provider_binding_id="binding-1", provider_account_identity="account-1",
                    client_conversation_id="client-1", retain_conversation=True,
                )
                failed = wait_for_task(service, OWNER, "captured-task", "error")
                self.assertEqual(failed["recovery_error_code"], expected)
                self.assertEqual(failed["recovery_phase"], "download_image_result")
                detail = failed["last_recovery_failure"]
                self.assertEqual(detail["phase"], "handler_operation")
                self.assertEqual(detail["type"], error_type.__name__)
                self.assertNotIn("bearer-secret", json.dumps(failed))
                self.assertNotIn("private.test", json.dumps(failed))
                restored = self.make_service(path).list_tasks(OWNER, ["captured-task"])["items"][0]
                self.assertEqual(restored["last_recovery_failure"], detail)

    def test_handler_failure_reports_observed_phase_separately_from_recovery_action(self):
        for step, expected_phase in (
            ("preparing_conversation", "prepare_conversation"),
            ("starting_generation", "start_image_generation"),
            ("generating", "stream_image_generation"),
            ("image_stream_resolve_start", "resolve_image_result"),
            ("receiving_image", "receive_image_result"),
            (None, "handler_operation"),
            ("unrecognized_step", "handler_operation"),
        ):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "image_tasks.json"
                def handler(payload):
                    callback = payload["progress_callback"]
                    callback.record_submission_started()
                    callback.record_conversation_id("conversation-1")
                    if step is not None:
                        callback(step)
                    raise ImageGenerationError(
                        "Authorization: bearer-secret URL=https://private.test/signed",
                        conversation_id="conversation-1",
                        request_message_id=callback.request_message_id,
                        upstream_submitted=True,
                    )
                service = self.make_service(path, handler)
                service.submit_generation(
                    OWNER, client_task_id="phase-task", prompt="cat", model="gpt-image-2", size=None,
                    provider_binding_id="binding-1", provider_account_identity="account-1",
                    client_conversation_id="client-1", retain_conversation=True,
                )
                failed = wait_for_task(service, OWNER, "phase-task", "error")
                self.assertEqual(failed["last_recovery_failure"]["phase"], expected_phase)
                self.assertEqual(failed["recovery_phase"], "read_image_request")
                self.assertEqual(failed["recovery_error_code"], "RECOVERY_READ_FAILED")
                self.assertEqual(failed["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
                self.assertNotIn("bearer-secret", json.dumps(failed))
                self.assertNotIn("private.test", json.dumps(failed))
                restored = self.make_service(path).list_tasks(OWNER, ["phase-task"])["items"][0]
                self.assertEqual(restored["last_recovery_failure"], failed["last_recovery_failure"])

    def test_restart_captured_result_does_not_invent_a_download_failure(self):
        for prior_code in (None, "RECOVERY_TIMED_OUT"):
            with self.subTest(prior_code=prior_code), tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "image_tasks.json"
                detail = {"phase": "collect_image_result", "type": "TimeoutError", "status_code": None, "at": 1000}
                write_policy_task(path, status="running", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                    result_file_ids=["original-file"], recovery_error_code=prior_code, last_recovery_failure=detail)
                restored = self.make_service(path).list_tasks(OWNER, ["policy-task"])["items"][0]
                self.assertEqual(restored["recovery_error_code"], prior_code or "RECOVERY_RESULT_INCOMPLETE")
                self.assertEqual(restored["last_recovery_failure"], detail)
                self.assertEqual(restored["recovery_phase"], "download_image_result")

    def test_binding_diagnosis_survives_protocol_and_public_readback_without_text(self):
        for message, reason in [
            ("conversation binding unavailable: bound account missing", "bound_account_missing"),
            ("conversation binding unavailable: bound account cannot generate images", "bound_image_capability_unavailable"),
            ("secret token or response body", None),
        ]:
            with self.subTest(reason=reason):
                request = ConversationRequest(model="gpt-image-2", prompt="mug", provider_binding_id="binding",
                                              provider_account_identity="account", client_conversation_id="client")
                with mock.patch("services.protocol.conversation.account_service.get_bound_account_identity", side_effect=RuntimeError(message)), \
                        mock.patch("services.protocol.conversation.OpenAIBackendAPI") as backend, \
                        self.assertRaises(ImageGenerationError) as failure:
                    _generate_bound_single_image(request, 1, 1)
                backend.assert_not_called()
                detail = _failure_details(failure.exception, "select_image_account")
                self.assertEqual(detail.get("binding_reason"), reason)
                self.assertNotIn(message, json.dumps(detail))
                with tempfile.TemporaryDirectory() as tmp_dir:
                    path = Path(tmp_dir) / "image_tasks.json"
                    write_policy_task(path, last_recovery_failure=detail)
                    public = self.make_service(path).list_tasks(OWNER, ["policy-task"])["items"][0]
                    self.assertEqual(public["last_recovery_failure"].get("binding_reason"), reason)

    def test_public_failure_details_drop_untrusted_persisted_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path, last_recovery_failure={
                "phase": "https://private.test/token", "type": "token=private",
                "status_code": "secret", "at": float("nan"),
                "body": "Authorization: private", "url": "https://private.test",
                "code": "https://private.test/token",
                "binding_reason": "https://private.test/token",
            })
            public = self.make_service(path).list_tasks(OWNER, ["policy-task"])["items"][0]
            self.assertEqual(public["last_recovery_failure"], {
                "phase": "unknown", "type": "Error", "status_code": None, "at": None,
            })
            self.assertNotIn("private", json.dumps(public))

    def test_original_image_confirmation_reason_survives_restart_without_exception_text(self):
        for code in ("IMAGE_THREAD_UPSTREAM_CHANGED", "IMAGE_THREAD_TURN_UNCONFIRMED"):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as tmp_dir:
                error = ImageThreadError(code)
                error.args = ("Authorization: private",)
                detail = _failure_details(error, "confirm_image_turn")
                path = Path(tmp_dir) / "image_tasks.json"
                write_policy_task(path, result_file_ids=["original-file"],
                    recovery_error_code="RECOVERY_THREAD_UNCONFIRMED", last_recovery_failure=detail)
                restored = self.make_service(path)
                public = restored.list_tasks(OWNER, ["policy-task"])["items"][0]
                self.assertEqual(public["last_recovery_failure"]["code"], code)
                self.assertEqual(restored._tasks["owner-1:policy-task"]["result_file_ids"], ["original-file"])
                self.assertNotIn("private", json.dumps(public))

    def test_generated_download_rate_limit_and_auth_keep_phase_and_failure_type(self):
        for status, expected_code, retry_after in (
            (429, "RECOVERY_RATE_LIMITED", 47),
            (403, "RECOVERY_AUTH_REQUIRED", None),
        ):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp_dir:
                def handler(payload):
                    callback = payload["progress_callback"]
                    callback.start_active_attempt()
                    callback.record_conversation_id("conversation-1")
                    callback.record_submission_started()
                    callback.record_result_ids(["file-generated"], [])
                    error = RuntimeError("download failed bearer=secret")
                    error.code = "CONVERSATION_OUTCOME_UNKNOWN"
                    error.status_code = status
                    if retry_after is not None:
                        error.retry_after = retry_after
                    error.conversation_id = "conversation-1"
                    error.request_message_id = callback.request_message_id
                    error.upstream_submitted = True
                    raise error

                service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler)
                service.submit_generation(
                    OWNER, client_task_id="download-task", prompt="cat", model="gpt-image-2", size=None,
                    provider_binding_id="binding-1", provider_account_identity="account-1",
                    client_conversation_id="client-1", retain_conversation=True,
                )
                failed = wait_for_task(service, OWNER, "download-task", "error")

                self.assertEqual(failed["recovery_phase"], "download_image_result")
                self.assertEqual(failed["recovery_error_code"], expected_code)
                self.assertEqual(failed["upstream_outcome"], "generated")
                self.assertFalse(failed["upstream_unfinished"])
                self.assertIn("Generated image is preserved", failed["error"])
                self.assertNotIn("bearer=secret", failed["error"])
                if retry_after is None:
                    self.assertNotIn("recovery_retry_after_seconds", failed)
                else:
                    self.assertEqual(failed["recovery_retry_after_seconds"], retry_after)
                    self.assertAlmostEqual(
                        failed["next_poll_at"] - time.time(), retry_after, delta=2,
                    )

    def test_bound_wrapper_preserves_download_http_failure_in_durable_task(self):
        from utils.helper import UpstreamHTTPError
        for status, code in ((429, "RECOVERY_RATE_LIMITED"), (401, "RECOVERY_AUTH_REQUIRED"),
                             (403, "RECOVERY_AUTH_REQUIRED")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp_dir:
                backend = mock.Mock(image_submission_started=True, image_request_message_id="request-1")
                backend.get_conversation_parent_message_id.return_value = ""
                def stream(_backend, request, *_args):
                    callback = request.progress_callback
                    callback.record_conversation_id("conversation-1")
                    callback.record_submission_started()
                    callback.record_result_ids(["file-generated"], [])
                    raise UpstreamHTTPError("download URL", status, {"private": "secret"},
                                            retry_after=123 if status == 429 else None)
                    yield
                def handler(payload):
                    request = ConversationRequest(prompt="fixture", model="gpt-image-2",
                        progress_callback=payload["progress_callback"], provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="client-1",
                        retain_conversation=True)
                    return _generate_bound_single_image(request, 1, 1)
                path = Path(tmp_dir) / "image_tasks.json"
                service = self.make_service(path, handler)
                with mock.patch("services.protocol.conversation.account_service") as accounts, \
                     mock.patch("services.protocol.conversation.OpenAIBackendAPI", return_value=backend), \
                     mock.patch("services.protocol.conversation.stream_image_outputs", side_effect=stream) as generate:
                    accounts.get_bound_account_identity.return_value = "account-1"
                    accounts.acquire_bound_image_access_token.return_value = "fixture-token"
                    accounts.get_account.return_value = {}
                    accounts.conversation_binding_lock.return_value = nullcontext()
                    service.submit_generation(OWNER, client_task_id="download-task", prompt="cat",
                        model="gpt-image-2", size=None, provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="client-1",
                        retain_conversation=True)
                    failed = wait_for_task(service, OWNER, "download-task", "error")
                self.assertEqual(failed["recovery_phase"], "download_image_result")
                self.assertEqual(failed["recovery_error_code"], code)
                self.assertEqual(failed["last_recovery_failure"]["status_code"], status)
                self.assertEqual(failed["upstream_outcome"], "generated")
                self.assertFalse(failed["upstream_unfinished"])
                self.assertEqual(service._tasks["owner-1:download-task"]["result_file_ids"], ["file-generated"])
                self.assertNotIn("secret", json.dumps(failed))
                if status == 429:
                    self.assertEqual(failed["recovery_retry_after_seconds"], 123)
                generate.assert_called_once()

    def test_expired_active_deadline_bypasses_old_schedule_and_bounds_empty_reads(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            error = RuntimeError("image outcome unknown")
            error.code = "CONVERSATION_OUTCOME_UNKNOWN"
            error.provider_binding_id = "binding-1"
            error.provider_account_identity = "account-1"
            error.conversation_id = "conversation-1"
            error.parent_message_id = "request-1"
            error.request_message_id = "request-1"
            error.upstream_submitted = True
            service = self.make_service(
                Path(tmp_dir) / "image_tasks.json",
                lambda _payload: (_ for _ in ()).throw(error),
            )
            service.submit_generation(
                OWNER, client_task_id="deadline-task", prompt="cat", model="gpt-image-2", size=None,
                provider_binding_id="binding-1", provider_account_identity="account-1",
                client_conversation_id="client-1", retain_conversation=True,
            )
            wait_for_task(service, OWNER, "deadline-task", "error")
            service._update_task(
                "owner-1:deadline-task",
                active_attempt_deadline_at=time.time() - 1,
                next_poll_at=time.time() + 900,
            )

            class EmptyBackend:
                polls = 0

                def __init__(self, access_token=None, proxy_url=None):
                    self.access_token = access_token

                def _get_conversation(self, _conversation_id):
                    return {
                        "current_node": "request-1",
                        "mapping": {
                            "request-1": {
                                "parent": "prior-turn",
                                "message": {"author": {"role": "user"}},
                            },
                        },
                    }

                def _query_backend_tasks(self, **kwargs):
                    if kwargs.get("strict_schema") is not True:
                        raise AssertionError("deadline recovery requires a strict tasks read")
                    return []

                def _poll_image_results(self, *_args, **_kwargs):
                    type(self).polls += 1
                    return [], []

                def _extract_image_tool_records(self, *_args, **_kwargs):
                    return []

                def close(self):
                    return None

            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="token"),
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", EmptyBackend),
            ):
                # The expired deadline bypasses the old 15-minute schedule and
                # starts the first exact-request read immediately.
                service.resume_poll(OWNER, "deadline-task", 5, "http://content-provider")
                for expected in (1, 2, 3):
                    task = wait_for_task(service, OWNER, "deadline-task", "error")
                    self.assertEqual(task["recovery_no_result_reads"], expected)
                    if expected == 3:
                        break
                    self.assertAlmostEqual(task["next_poll_at"] - time.time(), 5, delta=2)
                    service._update_task("owner-1:deadline-task", next_poll_at=0)
                    service.resume_poll(OWNER, "deadline-task", 5, "http://content-provider")

            self.assertEqual(task["error_code"], "RESULT_UNRECOVERABLE")
            self.assertFalse(task["upstream_unfinished"])
            self.assertEqual(EmptyBackend.polls, 3)

    def test_expired_deadline_never_bypasses_provider_retry_after(self):
        for phase in ("read_image_request", "download_image_result"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp_dir:
                error = RuntimeError("image outcome unknown")
                error.code = "CONVERSATION_OUTCOME_UNKNOWN"
                error.provider_binding_id = "binding-1"
                error.provider_account_identity = "account-1"
                error.conversation_id = "conversation-1"
                error.parent_message_id = "assistant-1"
                error.request_message_id = "request-1"
                error.upstream_submitted = True
                service = self.make_service(
                    Path(tmp_dir) / "image_tasks.json",
                    lambda _payload: (_ for _ in ()).throw(error),
                )
                service.submit_generation(
                    OWNER, client_task_id="cooldown-task", prompt="cat", model="gpt-image-2", size=None,
                    provider_binding_id="binding-1", provider_account_identity="account-1",
                    client_conversation_id="client-1", retain_conversation=True,
                )
                wait_for_task(service, OWNER, "cooldown-task", "error")
                cooldown_until = time.time() + 3600
                updates = {
                    "active_attempt_deadline_at": time.time() - 1,
                    "next_poll_at": cooldown_until,
                    "recovery_error_code": "RECOVERY_RATE_LIMITED",
                    "recovery_retry_after_seconds": 3600,
                    "recovery_phase": phase,
                }
                if phase == "download_image_result":
                    updates.update(
                        result_file_ids=["file-generated"],
                        upstream_outcome="generated",
                        upstream_unfinished=False,
                    )
                service._update_task("owner-1:cooldown-task", **updates)

                before = service.list_tasks(OWNER, ["cooldown-task"])["items"][0]
                with mock.patch("services.image_task_service.threading.Thread") as thread:
                    result = service.resume_poll(
                        OWNER, "cooldown-task", 5, "http://content-provider",
                    )

                thread.assert_not_called()
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["recovery_error_code"], "RECOVERY_RATE_LIMITED")
                self.assertEqual(result["recovery_phase"], phase)
                self.assertEqual(result["recovery_retry_after_seconds"], 3600)
                self.assertEqual(result["next_poll_at"], before["next_poll_at"])
                self.assertGreater(result["next_poll_at"], time.time() + 3500)

    def test_deadline_recovery_started_marker_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            error = RuntimeError("image outcome unknown")
            error.code = "CONVERSATION_OUTCOME_UNKNOWN"
            error.provider_binding_id = "binding-1"
            error.provider_account_identity = "account-1"
            error.conversation_id = "conversation-1"
            error.parent_message_id = "assistant-1"
            error.request_message_id = "request-1"
            error.upstream_submitted = True
            service = self.make_service(
                path, lambda _payload: (_ for _ in ()).throw(error),
            )
            service.submit_generation(
                OWNER, client_task_id="deadline-marker-task", prompt="cat", model="gpt-image-2", size=None,
                provider_binding_id="binding-1", provider_account_identity="account-1",
                client_conversation_id="client-1", retain_conversation=True,
            )
            wait_for_task(service, OWNER, "deadline-marker-task", "error")
            next_poll_at = time.time() + 900
            service._update_task(
                "owner-1:deadline-marker-task",
                active_attempt_deadline_at=time.time() - 1,
                deadline_recovery_started=True,
                recovery_error_code="RECOVERY_READ_FAILED",
                recovery_phase="read_image_request",
                recovery_retry_after_seconds=None,
                next_poll_at=next_poll_at,
            )

            restarted = self.make_service(path)
            self.assertTrue(
                restarted._tasks["owner-1:deadline-marker-task"]["deadline_recovery_started"]
            )
            with mock.patch("services.image_task_service.threading.Thread") as thread:
                result = restarted.resume_poll(
                    OWNER, "deadline-marker-task", 5, "http://content-provider",
                )

            thread.assert_not_called()
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["next_poll_at"], next_poll_at)

    def test_expired_deadline_preserves_running_request_and_transport_failures(self):
        for scenario, expected_code in (
            ("running", "RECOVERY_TIMED_OUT"),
            ("transport", "RECOVERY_TRANSPORT_FAILED"),
        ):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp_dir:
                error = RuntimeError("image outcome unknown")
                error.code = "CONVERSATION_OUTCOME_UNKNOWN"
                error.provider_binding_id = "binding-1"
                error.provider_account_identity = "account-1"
                error.conversation_id = "conversation-1"
                error.parent_message_id = "assistant-1"
                error.request_message_id = "request-1"
                error.upstream_submitted = True
                service = self.make_service(
                    Path(tmp_dir) / "image_tasks.json",
                    lambda _payload: (_ for _ in ()).throw(error),
                )
                service.submit_generation(
                    OWNER, client_task_id="deadline-task", prompt="cat", model="gpt-image-2", size=None,
                    provider_binding_id="binding-1", provider_account_identity="account-1",
                    client_conversation_id="client-1", retain_conversation=True,
                )
                wait_for_task(service, OWNER, "deadline-task", "error")
                service._update_task(
                    "owner-1:deadline-task",
                    active_attempt_deadline_at=time.time() - 1,
                    next_poll_at=time.time() + 900,
                )

                class RecoveryBackend:
                    def __init__(self, access_token=None, proxy_url=None):
                        self.access_token = access_token

                    def _get_conversation(self, _conversation_id):
                        if scenario == "transport":
                            raise ConnectionError("proxy bearer=secret")
                        return {
                            "current_node": "assistant-1",
                            "mapping": {
                                "request-1": {"message": {"author": {"role": "user"}}},
                                "assistant-1": {
                                    "parent": "request-1",
                                    "message": {
                                        "author": {"role": "assistant"},
                                        "status": "in_progress",
                                    },
                                },
                            },
                        }

                    def _query_backend_tasks(self, **kwargs):
                        if kwargs.get("strict_schema") is not True:
                            raise AssertionError("deadline recovery requires a strict tasks read")
                        return [{"status": "running"}]

                    def _poll_image_results(self, *_args, **_kwargs):
                        raise ImagePollTimeoutError("still running", "conversation-1")

                    def close(self):
                        return None

                with (
                    mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                    mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="token"),
                    mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                    mock.patch("services.openai_backend_api.OpenAIBackendAPI", RecoveryBackend),
                ):
                    service.resume_poll(OWNER, "deadline-task", 5, "http://content-provider")
                    task = wait_for_task(service, OWNER, "deadline-task", "error")

                self.assertEqual(task["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
                self.assertEqual(task["recovery_error_code"], expected_code)
                self.assertEqual(task.get("recovery_no_result_reads", 0), 0)
                self.assertTrue(task["upstream_unfinished"])
                self.assertAlmostEqual(task["next_poll_at"] - time.time(), 30, delta=2)
                self.assertNotIn("bearer=secret", task["error"])

    def test_duplicate_submit_uses_existing_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            calls = 0

            def handler(_payload):
                nonlocal calls
                calls += 1
                time.sleep(0.05)
                return {"data": [{"url": "http://example.test/image.png"}]}

            service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler)
            first = service.submit_generation(
                OWNER,
                client_task_id="task-1",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )
            second = service.submit_generation(
                OWNER,
                client_task_id="task-1",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )

            self.assertEqual(first["id"], "task-1")
            self.assertEqual(second["id"], "task-1")
            task = wait_for_task(service, OWNER, "task-1", "success")
            self.assertEqual(task["data"][0]["url"], "http://example.test/image.png")
            self.assertEqual(calls, 1)

    def test_duplicate_task_id_with_changed_request_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = self.make_service(Path(tmp_dir) / "image_tasks.json")
            service.submit_generation(
                OWNER,
                client_task_id="task-immutable",
                prompt="cat",
                model="gpt-image-2",
                size=None,
            )

            with self.assertRaisesRegex(ValueError, "different immutable request"):
                service.submit_generation(
                    OWNER,
                    client_task_id="task-immutable",
                    prompt="dog",
                    model="gpt-image-2",
                    size=None,
                )
            wait_for_task(service, OWNER, "task-immutable", "success")

    def test_unknown_resume_uses_the_same_bound_account_and_session(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            error = RuntimeError("ChatGPT 生图超时")
            error.code = "CONVERSATION_OUTCOME_UNKNOWN"
            error.provider_binding_id = "cb_account_a"
            error.provider_account_identity = "account_opaque_a"
            error.conversation_id = "conversation-1"
            error.parent_message_id = "message-1"
            error.request_message_id = "request-message-1"

            def handler(_payload):
                raise error

            service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler)
            service.submit_generation(
                OWNER,
                client_task_id="unknown-task",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                provider_binding_id="cb_account_a",
                provider_account_identity="account_opaque_a",
                client_conversation_id="workbench-conversation-1",
                retain_conversation=True,
            )
            wait_for_task(service, OWNER, "unknown-task", "error")

            class FakeBackend:
                poll_calls = []

                def __init__(self, access_token=None, proxy_url=None):
                    self.access_token = access_token

                def _poll_image_results(self, conversation_id, timeout, request_message_id="", initial_document=None):
                    self.poll = (conversation_id, timeout, request_message_id)
                    self.poll_calls.append(self.poll)
                    return ["file-1"], []

                def _get_conversation(self, _conversation_id):
                    return {"current_node": "assistant-1", "mapping": {"assistant-1": {"message": {"status": "in_progress"}}}}

                def resolve_conversation_image_urls(self, conversation_id, file_ids, sediment_ids, poll=False,
                                                    request_message_id=""):
                    return ["https://provider.example/image.png"]

                def download_image_bytes(self, _urls):
                    return [b"image-bytes"]

                def get_conversation_parent_message_id(self, _conversation_id):
                    return "message-2"

                def close(self):
                    return None

            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account_opaque_a"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="bound-token") as acquire,
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()) as binding_lock,
                mock.patch("services.account_service.account_service.release_image_slot") as release,
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", FakeBackend),
                mock.patch("services.protocol.conversation.format_image_result", return_value={"data": [{"url": "http://content-provider/images/result.png"}]}),
            ):
                resumed = service.resume_poll(OWNER, "unknown-task", 30, "http://content-provider")
                self.assertIn(resumed["status"], {"running", "success"}, resumed)
                task = wait_for_task(service, OWNER, "unknown-task", "success")

            acquire.assert_called_once_with("cb_account_a", model="auto")
            binding_lock.assert_called_once_with("cb_account_a", "workbench-conversation-1")
            release.assert_not_called()
            self.assertEqual(task["image_session_id"], "conversation-1")
            self.assertEqual(task["image_session_parent_id"], "message-2")
            self.assertEqual(task["data"], [{"url": "http://content-provider/images/result.png"}])
            self.assertEqual(FakeBackend.poll_calls, [("conversation-1", 30, "request-message-1")])
            recovered = service._tasks["owner-1:unknown-task"]
            self.assertGreater(recovered["_first_qualified_image_assets_observed_at"], 0)
            self.assertEqual(recovered["_first_qualified_image_asset_id_count"], 1)
            restarted = self.make_service(Path(tmp_dir) / "image_tasks.json")
            self.assertEqual(restarted._tasks["owner-1:unknown-task"]["_first_qualified_image_assets_observed_at"],
                             recovered["_first_qualified_image_assets_observed_at"])

    def test_unknown_resume_keeps_404_and_empty_final_without_images_unknown(self):
        scenarios = ("conversation-404", "empty-final-and-tasks")
        for scenario in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp_dir:
                error = RuntimeError("ChatGPT 生图超时")
                error.code = "CONVERSATION_OUTCOME_UNKNOWN"
                error.provider_binding_id = "cb_account_a"
                error.provider_account_identity = "account_opaque_a"
                error.conversation_id = "conversation-1"
                error.parent_message_id = "progress-leaf"
                error.request_message_id = "request-message-1"
                path = Path(tmp_dir) / "image_tasks.json"
                service = self.make_service(path, lambda _payload: (_ for _ in ()).throw(error))
                service.submit_generation(
                    OWNER,
                    client_task_id="unknown-task",
                    prompt="cat",
                    model="gpt-image-2",
                    size=None,
                    provider_binding_id="cb_account_a",
                    provider_account_identity="account_opaque_a",
                    client_conversation_id="workbench-conversation-1",
                    retain_conversation=True,
                )
                wait_for_task(service, OWNER, "unknown-task", "error")

                class FakeBackend:
                    poll_calls = []

                    def __init__(self, access_token=None, proxy_url=None):
                        self.access_token = access_token

                    def _get_conversation(self, _conversation_id):
                        if scenario == "conversation-404":
                            raise RuntimeError("/backend-api/conversation failed: status=404")
                        return {
                            "current_node": "assistant-1",
                            "mapping": {
                                "request-message-1": {
                                    "parent": "prior-turn",
                                    "message": {"author": {"role": "user"}},
                                },
                                "assistant-1": {
                                    "parent": "request-message-1",
                                    "message": {
                                        "author": {"role": "assistant"},
                                        "status": "finished_successfully",
                                        "end_turn": True,
                                        "content": {"content_type": "text", "parts": []},
                                    },
                                },
                            },
                        }

                    def _poll_image_results(self, conversation_id, timeout, request_message_id="", initial_document=None):
                        self.poll_calls.append((conversation_id, timeout, request_message_id))
                        # An empty /backend-api/tasks result and no branch image
                        # identifiers do not prove success or failure.
                        return [], []

                    def close(self):
                        return None

                with (
                    mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account_opaque_a"),
                    mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="bound-token"),
                    mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                    mock.patch("services.openai_backend_api.OpenAIBackendAPI", FakeBackend),
                ):
                    resumed = service.resume_poll(OWNER, "unknown-task", 30, "http://content-provider")
                    self.assertIn(resumed["status"], {"running", "error"}, resumed)
                    task = wait_for_task(service, OWNER, "unknown-task", "error")

                self.assertEqual(task["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
                self.assertEqual(task["binding_status"], "unknown")
                self.assertTrue(task["upstream_unfinished"])
                self.assertNotIn("recovery_status", task)
                if scenario == "conversation-404":
                    self.assertEqual(FakeBackend.poll_calls, [])
                    self.assertEqual(task["recovery_error_code"], "RECOVERY_READ_FAILED")
                    self.assertNotIn("status=404", task["error"])
                else:
                    self.assertEqual(
                        FakeBackend.poll_calls,
                        [("conversation-1", 30, "request-message-1")],
                    )

                restarted = self.make_service(path)
                persisted = restarted.list_tasks(OWNER, ["unknown-task"])["items"][0]
                self.assertEqual(persisted["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
                self.assertEqual(persisted["binding_status"], "unknown")

    def test_explicit_bounded_image_recovery_releases_capacity_after_three_empty_terminal_reads(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            error = RuntimeError("ChatGPT image result unknown")
            error.code = "CONVERSATION_OUTCOME_UNKNOWN"
            error.provider_binding_id = "cb_account_a"
            error.provider_account_identity = "account_opaque_a"
            error.conversation_id = "conversation-1"
            error.parent_message_id = "assistant-1"
            error.request_message_id = "request-message-1"
            service = self.make_service(
                Path(tmp_dir) / "image_tasks.json",
                lambda _payload: (_ for _ in ()).throw(error),
            )
            service.submit_generation(
                OWNER, client_task_id="unknown-task", prompt="cat", model="gpt-image-2", size=None,
                provider_binding_id="cb_account_a", provider_account_identity="account_opaque_a",
                client_conversation_id="workbench-conversation-1", retain_conversation=True,
            )
            wait_for_task(service, OWNER, "unknown-task", "error")
            service._update_task(
                "owner-1:unknown-task", created_ts=time.time() - 901, poll_failures=99,
            )

            class FakeBackend:
                reads = 0
                latest_read_running_once = True

                def __init__(self, access_token=None, proxy_url=None):
                    self.access_token = access_token

                def _get_conversation(self, _conversation_id):
                    type(self).reads += 1
                    if type(self).latest_read_running_once and type(self).reads == 2:
                        return {
                            "current_node": "assistant-running",
                            "mapping": {
                                "request-message-1": {
                                    "message": {"author": {"role": "user"}},
                                },
                                "assistant-running": {
                                    "parent": "request-message-1",
                                    "message": {
                                        "author": {"role": "assistant"},
                                        "status": "in_progress",
                                    },
                                },
                            },
                        }
                    return {
                        "current_node": "request-message-1",
                        "mapping": {
                            "request-message-1": {
                                "parent": "prior-turn",
                                "message": {"id": "request-message-1", "author": {"role": "user"}},
                            },
                        },
                    }

                def _poll_image_results(self, *_args, **_kwargs):
                    return [], []

                def _query_backend_tasks(self, **_kwargs):
                    if _kwargs.get("strict_schema") is not True:
                        raise AssertionError("recovery must require a strict tasks schema")
                    return []

                def close(self):
                    return None

            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account_opaque_a"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="bound-token"),
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", FakeBackend),
            ):
                service._update_task("owner-1:unknown-task", next_poll_at=0)
                service.resume_poll(
                    OWNER, "unknown-task", 30, "http://content-provider", True,
                )
                task = wait_for_task(service, OWNER, "unknown-task", "error")
                self.assertEqual(task.get("recovery_no_result_reads", 0), 0)
                FakeBackend.latest_read_running_once = False
                for expected in (1, 2, 3):
                    service._update_task("owner-1:unknown-task", next_poll_at=0)
                    service.resume_poll(
                        OWNER, "unknown-task", 30, "http://content-provider", True,
                    )
                    task = wait_for_task(service, OWNER, "unknown-task", "error")
                    self.assertEqual(task["recovery_no_result_reads"], expected)
                    if expected < 3:
                        expected_delay = 60 * (2 ** (expected - 1))
                        self.assertAlmostEqual(
                            task["next_poll_at"] - time.time(), expected_delay, delta=5,
                        )

            self.assertEqual(task["error_code"], "RESULT_UNRECOVERABLE")
            self.assertEqual(task["upstream_outcome"], "unknown")
            self.assertTrue(task["recovery_retryable"])
            self.assertFalse(task["recovery_requires_new_conversation"])
            self.assertFalse(task["upstream_unfinished"])
            self.assertEqual(
                service.resume_poll(OWNER, "unknown-task", 30, "http://content-provider", True),
                task,
            )

    def test_explicit_image_recovery_never_counts_a_running_branch(self):
        document = {
            "current_node": "assistant-1",
            "mapping": {
                "request-message-1": {
                    "message": {"author": {"role": "user"}},
                },
                "assistant-1": {
                    "parent": "request-message-1",
                    "message": {"author": {"role": "assistant"}, "status": "in_progress"},
                },
            },
        }
        from services.image_task_service import (
            _backend_tasks_may_be_active,
            _branch_read_state,
            _document_current_message_active,
        )
        self.assertEqual(_branch_read_state(document, "request-message-1"), "running")
        self.assertEqual(
            _branch_read_state({
                "current_node": "request-message-1",
                "mapping": {"request-message-1": {"message": {"author": {"role": "user"}}}},
            }, "request-message-1"),
            "no_result",
        )
        self.assertTrue(_backend_tasks_may_be_active([{"status": "running"}]))
        self.assertTrue(_backend_tasks_may_be_active([{"unexpected": "shape"}]))
        self.assertFalse(_backend_tasks_may_be_active([]))
        self.assertFalse(_backend_tasks_may_be_active([{"task_status": "finished"}]))
        self.assertTrue(_document_current_message_active({
            "current_node": "foreign-running",
            "mapping": {
                "foreign-running": {"message": {"status": "in_progress"}},
            },
        }))

    def test_legacy_missing_anchor_and_missing_chat_require_bounded_reads_without_attributing_images(self):
        for scenario, requires_new_conversation in (
            ("legacy-valid-chat", False),
            ("missing-chat", True),
            ("missing-then-valid-chat", False),
        ):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "image_tasks.json"
                path.write_text(json.dumps({"tasks": [{
                    "id": "legacy-task", "owner_id": "owner-1", "status": "error",
                    "mode": "generate", "model": "gpt-image-2",
                    "provider_binding_id": "cb_account_a",
                    "provider_account_identity": "account_opaque_a",
                    "client_conversation_id": "workbench-conversation-1",
                    "conversation_id": "conversation-1", "parent_message_id": "old-parent",
                    "binding_status": "unknown", "error_code": "CONVERSATION_OUTCOME_UNKNOWN",
                    "upstream_unfinished": True, "created_ts": time.time() - 901,
                    "created_at": "2026-09-14 00:00:00",
                    "updated_at": "2026-09-14 00:00:00",
                }]}), encoding="utf-8")
                service = self.make_service(path)

                class FakeBackend:
                    downloads = 0
                    reads = 0

                    def __init__(self, access_token=None, proxy_url=None):
                        self.access_token = access_token

                    def _get_conversation(self, _conversation_id):
                        type(self).reads += 1
                        if scenario == "missing-chat" or (
                            scenario == "missing-then-valid-chat" and type(self).reads == 1
                        ):
                            raise RuntimeError("/backend-api/conversation failed: status=404")
                        return {
                            "current_node": "finished-answer",
                            "mapping": {
                                "finished-answer": {
                                    "message": {
                                        "author": {"role": "assistant"},
                                        "status": "finished_successfully", "end_turn": True,
                                    },
                                },
                            },
                        }

                    def _query_backend_tasks(self, **_kwargs):
                        if _kwargs.get("strict_schema") is not True:
                            raise AssertionError("recovery must require a strict tasks schema")
                        return []

                    def download_image_bytes(self, _urls):
                        type(self).downloads += 1
                        return []

                    def close(self):
                        return None

                with (
                    mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account_opaque_a"),
                    mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="bound-token"),
                    mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                    mock.patch("services.openai_backend_api.OpenAIBackendAPI", FakeBackend),
                ):
                    for expected in (1, 2, 3):
                        service._update_task("owner-1:legacy-task", next_poll_at=0)
                        service.resume_poll(
                            OWNER, "legacy-task", 30, "http://content-provider", True,
                        )
                        task = wait_for_task(service, OWNER, "legacy-task", "error")
                        self.assertEqual(task["recovery_no_result_reads"], expected)

                self.assertEqual(task["error_code"], "RESULT_UNRECOVERABLE")
                self.assertEqual(
                    task["recovery_requires_new_conversation"], requires_new_conversation,
                )
                self.assertEqual(FakeBackend.downloads, 0)

    def test_unknown_resume_rejects_changed_bound_account_before_conversation_read(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            error = RuntimeError("ChatGPT 生图超时")
            error.code = "CONVERSATION_OUTCOME_UNKNOWN"
            error.provider_binding_id = "cb_account_a"
            error.provider_account_identity = "account_opaque_a"
            error.conversation_id = "conversation-1"
            error.parent_message_id = "progress-leaf"
            error.request_message_id = "request-message-1"
            service = self.make_service(
                Path(tmp_dir) / "image_tasks.json", lambda _payload: (_ for _ in ()).throw(error),
            )
            service.submit_generation(
                OWNER,
                client_task_id="changed-account-task",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                provider_binding_id="cb_account_a",
                provider_account_identity="account_opaque_a",
                client_conversation_id="workbench-conversation-1",
                retain_conversation=True,
            )
            wait_for_task(service, OWNER, "changed-account-task", "error")
            service._update_task("owner-1:changed-account-task", created_ts=time.time() - 901)

            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="different-account"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token") as acquire,
                mock.patch("services.openai_backend_api.OpenAIBackendAPI") as backend,
            ):
                service.resume_poll(OWNER, "changed-account-task", 30, "http://content-provider", True)
                task = wait_for_task(service, OWNER, "changed-account-task", "error")

            self.assertEqual(task["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
            self.assertEqual(task["binding_status"], "unknown")
            self.assertEqual(task["recovery_error_code"], "RECOVERY_AUTH_REQUIRED")
            self.assertIn("bound account connection", task["error"])
            self.assertEqual(task.get("recovery_no_result_reads", 0), 0)
            acquire.assert_not_called()
            backend.assert_not_called()

    def test_image_rate_limit_keeps_historical_poll_backoff(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            error = RuntimeError("ChatGPT image result unknown")
            error.code = "CONVERSATION_OUTCOME_UNKNOWN"
            error.provider_binding_id = "cb_account_a"
            error.provider_account_identity = "account_opaque_a"
            error.conversation_id = "conversation-1"
            error.parent_message_id = "assistant-1"
            error.request_message_id = "request-message-1"
            service = self.make_service(
                Path(tmp_dir) / "image_tasks.json",
                lambda _payload: (_ for _ in ()).throw(error),
            )
            service.submit_generation(
                OWNER, client_task_id="rate-limit-task", prompt="cat", model="gpt-image-2",
                size=None, provider_binding_id="cb_account_a",
                provider_account_identity="account_opaque_a",
                client_conversation_id="workbench-conversation-1", retain_conversation=True,
            )
            wait_for_task(service, OWNER, "rate-limit-task", "error")
            service._update_task(
                "owner-1:rate-limit-task",
                created_ts=time.time() - 901,
                poll_failures=99,
                next_poll_at=0,
            )

            class RateLimitError(RuntimeError):
                status_code = 429

            class RateLimitBackend:
                def __init__(self, access_token=None, proxy_url=None):
                    self.access_token = access_token

                def _get_conversation(self, _conversation_id):
                    raise RateLimitError("upstream status=429")

                def close(self):
                    return None

            with (
                mock.patch(
                    "services.account_service.account_service.get_bound_account_identity",
                    return_value="account_opaque_a",
                ),
                mock.patch(
                    "services.account_service.account_service.get_bound_text_access_token",
                    return_value="bound-token",
                ),
                mock.patch(
                    "services.account_service.account_service.conversation_binding_lock",
                    return_value=nullcontext(),
                ),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", RateLimitBackend),
            ):
                service.resume_poll(
                    OWNER, "rate-limit-task", 30, "http://content-provider", True,
                )
                task = wait_for_task(service, OWNER, "rate-limit-task", "error")

            stored = service._tasks["owner-1:rate-limit-task"]
            self.assertEqual(stored["poll_failures"], 100)
            self.assertEqual(task.get("recovery_no_result_reads", 0), 0)
            self.assertEqual(task["recovery_error_code"], "RECOVERY_RATE_LIMITED")
            self.assertNotIn("status=429", task["error"])
            self.assertAlmostEqual(task["next_poll_at"] - time.time(), 900, delta=5)

    def test_authoritative_finished_image_failure_is_terminal(self):
        document = {
            "current_node": "assistant-1",
            "mapping": {
                "request-1": {
                    "parent": "prior-turn",
                    "message": {"author": {"role": "user"}},
                },
                "assistant-1": {
                    "parent": "request-1",
                    "message": {
                        "author": {"role": "assistant"},
                        "status": "finished_successfully",
                        "end_turn": True,
                        "content": {
                            "content_type": "text",
                            "parts": ["Something went wrong while generating your image. Sorry about that."],
                        },
                    }
                }
            },
        }
        self.assertEqual(
            _authoritative_image_failure(document, "request-1"),
            "Something went wrong while generating your image. Sorry about that.",
        )
        document["mapping"]["assistant-1"]["message"]["status"] = "in_progress"
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "")

        document["mapping"]["assistant-1"]["message"]["status"] = "finished_successfully"
        document["mapping"]["assistant-1"]["message"]["content"]["parts"] = []
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "")

        document["mapping"]["assistant-1"]["message"]["content"]["parts"] = [
            "Something went wrong while generating your image. Sorry about that."
        ]
        document["mapping"]["later-user"] = {
            "parent": "assistant-1", "message": {"author": {"role": "user"}},
        }
        document["mapping"]["later-failure"] = {
            "parent": "later-user",
            "message": {
                "author": {"role": "assistant"},
                "status": "finished_successfully",
                "end_turn": True,
                "content": {
                    "content_type": "text",
                    "parts": ["Something went wrong while generating your image. Sorry about that."],
                },
            },
        }
        document["current_node"] = "later-failure"
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "")

        def known_generation_error_document():
            return {
                "current_node": "terminal-1",
                "mapping": {
                    "request-1": {
                        "parent": "prior-turn",
                        "message": {
                            "id": "request-1", "author": {"role": "user"},
                            # Input references belong to the user's request;
                            # only output assets after it disqualify this receipt.
                            "content": {
                                "content_type": "multimodal_text",
                                "parts": [{
                                    "content_type": "image_asset_pointer",
                                    "asset_pointer": "file-service://reference-image",
                                }],
                            },
                        },
                    },
                    "worker-1": {
                        "parent": "request-1",
                        "message": {
                            "id": "worker-1", "author": {"role": "assistant"},
                            "status": "finished_successfully", "end_turn": False,
                            "content": {"content_type": "code", "parts": []},
                        },
                    },
                    "tool-1": {
                        "parent": "worker-1",
                        "message": {
                            "id": "tool-1", "author": {"role": "tool"},
                            "status": "finished_successfully",
                            "metadata": {"is_error": True},
                            "content": {
                                "content_type": "text",
                                "parts": [
                                    "We experienced an error when generating images. Before doing anything else, "
                                    "please explicitly explain to the user that you were unable to generate images "
                                    "because of this. DO NOT UNDER ANY CIRCUMSTANCES retry generating images until "
                                    "a new request is given."
                                ],
                            },
                        },
                    },
                    "terminal-1": {
                        "parent": "tool-1",
                        "message": {
                            "id": "terminal-1", "author": {"role": "assistant"},
                            "status": "finished_successfully", "end_turn": True,
                            "content": {"content_type": "text", "parts": ["由于我这边发生了错误，我未能生成图片。"]},
                        },
                    },
                },
            }

        def localized_no_image_generated_document():
            # Provider receipt for product 341981723: original user request,
            # code node, reasoning recap, then this exact terminal assistant text.
            return {
                "current_node": "terminal-1",
                "mapping": {
                    "request-1": {
                        "parent": "prior-turn",
                        "message": {"id": "request-1", "author": {"role": "user"}},
                    },
                    "worker-1": {
                        "parent": "request-1",
                        "message": {
                            "id": "worker-1", "author": {"role": "assistant"},
                            "status": "finished_successfully", "end_turn": False,
                            "content": {"content_type": "code", "parts": []},
                        },
                    },
                    "recap-1": {
                        "parent": "worker-1",
                        "message": {
                            "id": "recap-1", "author": {"role": "assistant"},
                            "status": "finished_successfully", "end_turn": False,
                            "content": {"content_type": "reasoning_recap", "parts": []},
                        },
                    },
                    "terminal-1": {
                        "parent": "recap-1",
                        "message": {
                            "id": "terminal-1", "author": {"role": "assistant"},
                            "status": "finished_successfully", "end_turn": True,
                            "content": {
                                "content_type": "text",
                                "parts": [
                                    "无法生成图片：图片生成过程中发生了错误，因此这次未能完成生成。"
                                    "请重新发起一次新的图片生成请求后，我可以继续处理。"
                                ],
                            },
                        },
                    },
                },
            }

        document = known_generation_error_document()
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "由于我这边发生了错误，我未能生成图片。")

        document = known_generation_error_document()
        document["mapping"]["later-user"] = {
            "parent": "terminal-1", "message": {"id": "later-user", "author": {"role": "user"}},
        }
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "")

        document = known_generation_error_document()
        document["mapping"]["sibling"] = {
            "parent": "request-1", "message": {"id": "sibling", "author": {"role": "assistant"}},
        }
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "")

        document = known_generation_error_document()
        document["mapping"]["tool-1"]["message"]["metadata"] = {}
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "")

        document = known_generation_error_document()
        document["mapping"]["tool-1"]["message"]["content"]["parts"] = ["another tool error"]
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "")

        document = known_generation_error_document()
        document["mapping"]["worker-1"]["message"]["author"] = None
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "")

        document = known_generation_error_document()
        document["mapping"]["worker-1"]["message"] = None
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "")

        document = known_generation_error_document()
        document["mapping"]["worker-1"]["message"]["content"] = {
            "content_type": "image_asset_pointer", "asset_pointer": "file-service://generated-image",
        }
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "")

        localized_terminal = (
            "无法生成图片：图片生成过程中发生了错误，因此这次未能完成生成。"
            "请重新发起一次新的图片生成请求后，我可以继续处理。"
        )
        document = localized_no_image_generated_document()
        self.assertEqual(_authoritative_image_failure(document, "request-1"), localized_terminal)

        document = localized_no_image_generated_document()
        document["mapping"]["terminal-1"]["message"]["status"] = "in_progress"
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "")

        document = localized_no_image_generated_document()
        document["mapping"]["sibling"] = {
            "parent": "request-1", "message": {"id": "sibling", "author": {"role": "assistant"}},
        }
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "")

        document = localized_no_image_generated_document()
        document["mapping"]["later-user"] = {
            "parent": "terminal-1", "message": {"id": "later-user", "author": {"role": "user"}},
        }
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "")

        document = localized_no_image_generated_document()
        document["mapping"]["worker-1"]["message"]["content"] = {
            "content_type": "image_asset_pointer", "asset_pointer": "file-service://generated-image",
        }
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "")

        document = localized_no_image_generated_document()
        document["mapping"]["terminal-1"]["message"]["content"]["parts"] = ["无法生成图片：图片生成过程中发生了错误。"]
        self.assertEqual(_authoritative_image_failure(document, "request-1"), "")

    def test_legacy_resume_without_submitted_message_boundary_is_non_rotating_unknown(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            request_hash = "same-immutable-request-hash"
            path.write_text(
                json.dumps({"tasks": [
                    {
                        "id": "missing-boundary-task",
                        "owner_id": "owner-1",
                        "status": "error",
                        "mode": "generate",
                        "provider_binding_id": "cb_account_a",
                        "provider_account_identity": "account_opaque_a",
                        "client_conversation_id": "workbench-conversation-1",
                        "conversation_id": "conversation-1",
                        "parent_message_id": "progress-leaf",
                        "binding_status": "unknown",
                        "error_code": "CONVERSATION_OUTCOME_UNKNOWN",
                        "error": "/backend-api/conversation failed: status=404",
                        "request_hash": request_hash,
                        "upstream_unfinished": True,
                        "poll_failures": 7,
                        "next_poll_at": 4102444800,
                        "duration_ms": 123,
                        "updated_at": "2026-09-14 19:00:00",
                    },
                    {
                        # Even an exact request fingerprint and conversation on
                        # a sibling cannot prove this task's submitted node.
                        "id": "same-owner-same-request-sibling",
                        "owner_id": "owner-1",
                        "status": "error",
                        "mode": "generate",
                        "provider_binding_id": "cb_account_a",
                        "provider_account_identity": "account_opaque_a",
                        "client_conversation_id": "workbench-conversation-1",
                        "conversation_id": "conversation-1",
                        "parent_message_id": "older-branch-leaf",
                        "request_message_id": "sibling-request-message",
                        "binding_status": "unknown",
                        "error_code": "CONVERSATION_OUTCOME_UNKNOWN",
                        "request_hash": request_hash,
                        "upstream_unfinished": True,
                        "updated_at": "2026-09-14 18:59:00",
                    },
                    {
                        "id": "successful-sibling",
                        "owner_id": "owner-1",
                        "status": "success",
                        "mode": "generate",
                        "data": [{"url": "https://example.test/already-finished.png"}],
                        "updated_at": "2026-09-14 18:58:00",
                    },
                ]}),
                encoding="utf-8",
            )
            service = self.make_service(path)
            before = service.list_tasks(OWNER, ["missing-boundary-task"])["items"][0]
            with (
                mock.patch("services.image_task_service.threading.Thread") as thread,
                mock.patch("services.image_task_service.uuid.uuid4") as new_uuid,
            ):
                first = service.resume_poll(OWNER, "missing-boundary-task", 30, "http://content-provider")
                second = service.resume_poll(OWNER, "missing-boundary-task", 30, "http://content-provider")

            self.assertEqual(first, before)
            self.assertEqual(second, before)
            self.assertEqual(first["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
            self.assertEqual(first["binding_status"], "unknown")
            self.assertNotIn("upstream_submission_started", first)
            self.assertEqual(first["recovery_status"], "request_message_id_required")
            self.assertEqual(first["next_poll_at"], 4102444800)
            self.assertEqual(first["duration_ms"], 123)
            thread.assert_not_called()
            new_uuid.assert_not_called()

            stored = next(task for task in service._tasks.values() if task["id"] == "missing-boundary-task")
            self.assertEqual(stored["request_message_id"], "")
            self.assertEqual(stored["poll_failures"], 7)
            self.assertEqual(
                next(task for task in service._tasks.values() if task["id"] == "same-owner-same-request-sibling")
                ["request_message_id"],
                "sibling-request-message",
            )
            successful = service.list_tasks(OWNER, ["successful-sibling"])["items"][0]
            self.assertEqual(successful["status"], "success")
            self.assertEqual(successful["data"], [{"url": "https://example.test/already-finished.png"}])

            restarted = self.make_service(path)
            after_restart = restarted.list_tasks(OWNER, ["missing-boundary-task"])["items"][0]
            self.assertEqual(after_restart, before)
            self.assertEqual(after_restart["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
            self.assertEqual(after_restart["binding_status"], "unknown")

    def test_task_persists_request_and_observed_conversation_before_handler_returns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            observed = {}

            def handler(payload):
                callback = payload["progress_callback"]
                observed["request_message_id"] = callback.request_message_id
                callback.record_conversation_id("new-conversation-1")
                raise RuntimeError("stream interrupted after provider POST")

            service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler)
            service.submit_generation(
                OWNER,
                client_task_id="pre-post-boundary-task",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                provider_binding_id="cb_account_a",
                provider_account_identity="account_opaque_a",
                client_conversation_id="workbench-conversation-1",
                retain_conversation=True,
            )
            wait_for_task(service, OWNER, "pre-post-boundary-task", "error")
            stored = next(task for task in service._tasks.values() if task["id"] == "pre-post-boundary-task")

        self.assertTrue(observed["request_message_id"])
        self.assertEqual(stored["request_message_id"], observed["request_message_id"])
        self.assertEqual(stored["conversation_id"], "new-conversation-1")
        self.assertEqual(stored["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")

    def test_bound_policy_rejection_stays_terminal_and_keeps_request_id(self):
        class Backend:
            image_request_message_id = "request-message-1"

            def __init__(self, access_token=None):
                self.access_token = access_token

            def get_conversation_parent_message_id(self, _conversation_id):
                return "message-after-rejection"

            def close(self):
                return None

        request = ConversationRequest(
            model="gpt-image-2",
            prompt="cat",
            provider_binding_id="cb_account_a",
            provider_account_identity="account_opaque_a",
            client_conversation_id="workbench-conversation-1",
            conversation_id="conversation-1",
            parent_message_id="message-before-request",
            retain_conversation=True,
        )
        with (
            mock.patch("services.protocol.conversation.account_service.get_bound_account_identity", return_value="account_opaque_a"),
            mock.patch("services.protocol.conversation.account_service.acquire_bound_image_access_token", return_value="bound-token"),
            mock.patch("services.protocol.conversation.account_service.get_account", return_value={"email": "account@example.test"}),
            mock.patch("services.protocol.conversation.account_service.conversation_binding_lock", return_value=nullcontext()),
            mock.patch("services.protocol.conversation.account_service.mark_image_result"),
            mock.patch("services.protocol.conversation.account_service.release_image_slot"),
            mock.patch("services.protocol.conversation.OpenAIBackendAPI", Backend),
            mock.patch(
                "services.protocol.conversation.stream_image_outputs",
                side_effect=ImageContentPolicyError("This request violates our content policy.", "conversation-1"),
            ),
        ):
            with self.assertRaises(ImageGenerationError) as raised:
                _generate_bound_single_image(request, 1, 1)

        self.assertEqual(raised.exception.code, "content_policy_violation")
        self.assertEqual(raised.exception.request_message_id, "request-message-1")
        self.assertEqual(raised.exception.conversation_id, "conversation-1")

    def test_bound_image_slot_is_settled_once_when_a_waiter_acquires_after_release(self):
        class Backend:
            image_request_message_id = "request-message-1"

            def __init__(self, access_token=None):
                self.access_token = access_token

            def get_conversation_parent_message_id(self, _conversation_id):
                return "message-after-result"

            def close(self):
                return None

        scenarios = (
            ("success", None, [True], None),
            (
                "policy",
                ImageContentPolicyError("This request violates our content policy.", "conversation-1"),
                [False],
                "content_policy_violation",
            ),
            (
                "unknown",
                RuntimeError("stream interrupted after provider POST"),
                [False],
                "CONVERSATION_OUTCOME_UNKNOWN",
            ),
            ("initialization", None, [], None),
        )
        for name, stream_error, expected_marks, expected_code in scenarios:
            with self.subTest(name=name):
                slot = {"inflight": 1, "releases": 0}
                slot_lock = threading.Lock()
                first_release = threading.Event()
                waiter_acquired = threading.Event()

                def waiter():
                    if not first_release.wait(timeout=1):
                        return
                    with slot_lock:
                        slot["inflight"] += 1
                    waiter_acquired.set()

                waiting_thread = threading.Thread(target=waiter)
                waiting_thread.start()

                def release_image_slot(access_token):
                    self.assertEqual(access_token, "bound-token")
                    with slot_lock:
                        slot["releases"] += 1
                        slot["inflight"] -= 1
                        release_number = slot["releases"]
                    if release_number == 1:
                        first_release.set()
                        self.assertTrue(waiter_acquired.wait(timeout=1))

                marks = []

                def mark_image_result(access_token, success):
                    marks.append(success)
                    release_image_slot(access_token)

                request = ConversationRequest(
                    model="gpt-image-2",
                    prompt="cat",
                    provider_binding_id="cb_account_a",
                    provider_account_identity="account_opaque_a",
                    client_conversation_id="workbench-conversation-1",
                    conversation_id="conversation-1",
                    parent_message_id="message-before-request",
                    retain_conversation=True,
                )
                output = ImageOutput(
                    kind="result",
                    model="gpt-image-2",
                    index=1,
                    total=1,
                    data=[{"b64_json": "image"}],
                    conversation_id="conversation-1",
                )

                def get_account(_access_token):
                    if name == "initialization":
                        raise RuntimeError("account initialization failed")
                    return {"email": "account@example.test"}

                def stream_outputs(*_args, **_kwargs):
                    if stream_error is not None:
                        raise stream_error
                    return iter([output])

                with (
                    mock.patch("services.protocol.conversation.account_service.get_bound_account_identity", return_value="account_opaque_a"),
                    mock.patch("services.protocol.conversation.account_service.acquire_bound_image_access_token", return_value="bound-token"),
                    mock.patch("services.protocol.conversation.account_service.get_account", side_effect=get_account),
                    mock.patch("services.protocol.conversation.account_service.conversation_binding_lock", return_value=nullcontext()),
                    mock.patch("services.protocol.conversation.account_service.mark_image_result", side_effect=mark_image_result),
                    mock.patch("services.protocol.conversation.account_service.release_image_slot", side_effect=release_image_slot),
                    mock.patch("services.protocol.conversation.OpenAIBackendAPI", Backend),
                    mock.patch("services.protocol.conversation.stream_image_outputs", side_effect=stream_outputs),
                ):
                    if name == "success":
                        self.assertEqual(_generate_bound_single_image(request, 1, 1), [output])
                    else:
                        with self.assertRaises(Exception) as raised:
                            _generate_bound_single_image(request, 1, 1)
                        if expected_code:
                            self.assertIsInstance(raised.exception, ImageGenerationError)
                            self.assertEqual(raised.exception.code, expected_code)

                waiting_thread.join(timeout=1)
                self.assertFalse(waiting_thread.is_alive())
                self.assertEqual(marks, expected_marks)
                self.assertEqual(slot["releases"], 1)
                self.assertEqual(slot["inflight"], 1)

    def test_unknown_resume_maps_authoritative_finished_failure_to_terminal_code(self):
        def english_failure_document():
            return {
                "current_node": "assistant-1",
                "mapping": {
                    "request-1": {
                        "parent": "prior-turn",
                        "message": {"id": "request-1", "author": {"role": "user"}},
                    },
                    "assistant-1": {
                        "parent": "request-1",
                        "message": {
                            "id": "assistant-1", "author": {"role": "assistant"},
                            "status": "finished_successfully", "end_turn": True,
                            "content": {
                                "content_type": "text",
                                "parts": ["Something went wrong while generating your image. Sorry about that."],
                            },
                        },
                    },
                },
            }

        def known_generation_error_document():
            return {
                "current_node": "terminal-1",
                "mapping": {
                    "request-1": {
                        "parent": "prior-turn",
                        "message": {"id": "request-1", "author": {"role": "user"}},
                    },
                    "worker-1": {
                        "parent": "request-1",
                        "message": {
                            "id": "worker-1", "author": {"role": "assistant"},
                            "status": "finished_successfully", "end_turn": False,
                            "content": {"content_type": "code", "parts": []},
                        },
                    },
                    "tool-1": {
                        "parent": "worker-1",
                        "message": {
                            "id": "tool-1", "author": {"role": "tool"},
                            "status": "finished_successfully", "metadata": {"is_error": True},
                            "content": {
                                "content_type": "text",
                                "parts": [
                                    "We experienced an error when generating images. Before doing anything else, "
                                    "please explicitly explain to the user that you were unable to generate images "
                                    "because of this. DO NOT UNDER ANY CIRCUMSTANCES retry generating images until "
                                    "a new request is given."
                                ],
                            },
                        },
                    },
                    "terminal-1": {
                        "parent": "tool-1",
                        "message": {
                            "id": "terminal-1", "author": {"role": "assistant"},
                            "status": "finished_successfully", "end_turn": True,
                            "content": {
                                "content_type": "text",
                                "parts": ["由于我这边发生了错误，我未能生成图片。"],
                            },
                        },
                    },
                },
            }

        def localized_no_image_generated_document():
            return {
                "current_node": "terminal-1",
                "mapping": {
                    "request-1": {
                        "parent": "prior-turn",
                        "message": {"id": "request-1", "author": {"role": "user"}},
                    },
                    "worker-1": {
                        "parent": "request-1",
                        "message": {
                            "id": "worker-1", "author": {"role": "assistant"},
                            "status": "finished_successfully", "end_turn": False,
                            "content": {"content_type": "code", "parts": []},
                        },
                    },
                    "recap-1": {
                        "parent": "worker-1",
                        "message": {
                            "id": "recap-1", "author": {"role": "assistant"},
                            "status": "finished_successfully", "end_turn": False,
                            "content": {"content_type": "reasoning_recap", "parts": []},
                        },
                    },
                    "terminal-1": {
                        "parent": "recap-1",
                        "message": {
                            "id": "terminal-1", "author": {"role": "assistant"},
                            "status": "finished_successfully", "end_turn": True,
                            "content": {
                                "content_type": "text",
                                "parts": [
                                    "无法生成图片：图片生成过程中发生了错误，因此这次未能完成生成。"
                                    "请重新发起一次新的图片生成请求后，我可以继续处理。"
                                ],
                            },
                        },
                    },
                },
            }

        for name, document, refresh_old_unrecoverable in (
            ("english", english_failure_document(), False),
            ("known_tool", known_generation_error_document(), True),
            ("localized_no_tool", localized_no_image_generated_document(), True),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp_dir:
                error = RuntimeError("ChatGPT 生图超时")
                error.code = "CONVERSATION_OUTCOME_UNKNOWN"
                error.provider_binding_id = "cb_account_a"
                error.provider_account_identity = "account_opaque_a"
                error.conversation_id = "conversation-1"
                error.parent_message_id = "message-1"
                error.request_message_id = "request-1"

                service = self.make_service(Path(tmp_dir) / "image_tasks.json", lambda _payload: (_ for _ in ()).throw(error))
                service.submit_generation(
                    OWNER,
                    client_task_id="terminal-no-image-task",
                    prompt="cat",
                    model="gpt-image-2",
                    size=None,
                    provider_binding_id="cb_account_a",
                    provider_account_identity="account_opaque_a",
                    client_conversation_id="workbench-conversation-1",
                    retain_conversation=True,
                )
                wait_for_task(service, OWNER, "terminal-no-image-task", "error")
                if refresh_old_unrecoverable:
                    # This mirrors the original-only recovery receipt after it
                    # has reserved zero further requests: no edit-only state
                    # or finished-at marker is needed to read the original.
                    with service._transaction():
                        current = service._tasks["owner-1:terminal-no-image-task"]
                        self.assertEqual(current["request_message_id"], "request-1")
                        current.update(
                            error_code="RESULT_UNRECOVERABLE",
                            upstream_outcome="unknown",
                            recovery_no_result_reads=3,
                            _completion={
                                "state": "checking_original",
                                "allow_unconfirmed_retry": False,
                                "max_extra_requests": 0,
                            },
                        )
                        service._save_locked()

                class FakeBackend:
                    def __init__(self, access_token=None, proxy_url=None):
                        self.access_token = access_token

                    def _get_conversation(self, _conversation_id):
                        return document

                    def close(self):
                        return None

                with (
                    mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account_opaque_a"),
                    mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="bound-token"),
                    mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                    mock.patch("services.account_service.account_service.release_image_slot"),
                    mock.patch("services.openai_backend_api.OpenAIBackendAPI", FakeBackend),
                ):
                    resumed = service.resume_poll(
                        OWNER, "terminal-no-image-task", 30, "http://content-provider",
                        completion_recheck=refresh_old_unrecoverable,
                    )
                    self.assertIn(resumed["status"], {"running", "error"}, resumed)
                    task = wait_for_task(service, OWNER, "terminal-no-image-task", "error")

                self.assertEqual(task["error_code"], "NO_IMAGE_GENERATED", task)
                expected_terminal = "terminal-1" if refresh_old_unrecoverable else "assistant-1"
                self.assertEqual(task["image_session_parent_id"], expected_terminal)
                stored = service._tasks["owner-1:terminal-no-image-task"]
                if refresh_old_unrecoverable:
                    self.assertEqual(stored["recovery_no_result_reads"], 3)
                self.assertEqual(stored["upstream_outcome"], "failed")
                from services.generation_completion import unresolved
                self.assertFalse(unresolved(stored))
                self.assertFalse(stored["upstream_unfinished"])
                self.assertEqual(stored["next_poll_at"], 0)
                self.assertFalse(stored["recovery_retryable"])

    def test_different_owner_cannot_query_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = self.make_service(Path(tmp_dir) / "image_tasks.json")
            service.submit_generation(
                OWNER,
                client_task_id="private-task",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )

            wait_for_task(service, OWNER, "private-task", "success")
            result = service.list_tasks(OTHER_OWNER, ["private-task"])

            self.assertEqual(result["items"], [])
            self.assertEqual(result["missing_ids"], ["private-task"])

    def test_adopts_latest_completed_manual_image_after_an_edited_branch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path)
            generation_calls = []
            service = self.make_service(path, lambda payload: generation_calls.append(payload))
            AdoptionBackend.document = manual_image_document(divergent=True)
            AdoptionBackend.reads = AdoptionBackend.downloads = 0
            AdoptionBackend.resolved = []
            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="read-token"),
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", AdoptionBackend),
                mock.patch("services.protocol.conversation.format_image_result", return_value={"data": [{"url": "http://content/images/manual.png"}]}),
            ):
                result = service.adopt_latest_conversation_image(
                    OWNER, "policy-task", provider_binding_id="binding-1",
                    provider_account_identity="account-1", client_conversation_id="client-chat-1",
                    conversation_id="conversation-1", source_request_message_id="manual-latest",
                    source_image_message_id="latest-image", base_url="http://content",
                )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["adopted_source_request_message_id"], "manual-latest")
            self.assertEqual(result["adopted_source_image_message_id"], "latest-image")
            self.assertEqual(result["adopted_from_error_code"], "content_policy_violation")
            self.assertEqual(result["adopted_from_error"], "original policy failure")
            self.assertEqual(result["image_session_parent_id"], "latest-finished")
            self.assertEqual(AdoptionBackend.downloads, 1)
            self.assertEqual(generation_calls, [])
            self.assertEqual(AdoptionBackend.resolved[0][3]["request_message_id"], "manual-latest")

    def test_workbench_manual_recovery_uses_the_original_task_and_latest_verified_image(self):
        for error_code in ("content_policy_violation", "NO_IMAGE_GENERATED"):
            with self.subTest(error_code=error_code), tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "image_tasks.json"
                write_policy_task(path, error_code=error_code)
                generation_calls = []
                service = self.make_service(path, lambda payload: generation_calls.append(payload))
                AdoptionBackend.document = manual_image_document()
                AdoptionBackend.downloads = 0
                AdoptionBackend.resolved = []
                with (
                    mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                    mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="read-token"),
                    mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                    mock.patch("services.openai_backend_api.OpenAIBackendAPI", AdoptionBackend),
                    mock.patch("services.protocol.conversation.format_image_result", return_value={"data": [{"url": "http://content/images/manual.png"}]}),
                ):
                    first = service.recover_manual(
                        OWNER, "policy-task", provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="client-chat-1",
                        base_url="http://content",
                    )
                    second = service.recover_manual(
                        OWNER, "policy-task", provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="client-chat-1",
                        base_url="http://content",
                    )
                persisted = self.make_service(path).list_tasks(OWNER, ["policy-task"])["items"][0]
                self.assertEqual(first, second)
                self.assertEqual(first["id"], "policy-task")
                self.assertEqual(first["status"], "success")
                self.assertEqual(first["data"], [{"url": "http://content/images/manual.png"}])
                self.assertEqual(persisted["adopted_from_error_code"], error_code)
                self.assertEqual(persisted["adopted_source_request_message_id"], "manual-latest")
                self.assertEqual(persisted["adopted_source_image_message_id"], "latest-image")
                self.assertEqual(AdoptionBackend.downloads, 1)
                self.assertEqual(generation_calls, [])

    def test_workbench_manual_recovery_rejects_unproven_or_stopped_originals_before_upstream_read(self):
        cases = (
            (OTHER_OWNER, {}, "task not found"),
            (OWNER, {"request_message_id": ""}, "original request identity is unavailable"),
            (OWNER, {"conversation_id": ""}, "original conversation identity is unavailable"),
            (OWNER, {"error_code": "CONVERSATION_OUTCOME_UNKNOWN"}, "not eligible"),
            (OWNER, {"_recovery_suppressed": True}, "recovery is stopped"),
        )
        for identity, overrides, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "image_tasks.json"
                write_policy_task(path, **overrides)
                service = self.make_service(path)
                if overrides.get("_recovery_suppressed") is True:
                    # The production stop marker lives in the SQLite receipt;
                    # the legacy JSON import intentionally selects old fields.
                    with service._transaction():
                        service._tasks["owner-1:policy-task"]["_recovery_suppressed"] = True
                        service._save_locked()
                with mock.patch("services.openai_backend_api.OpenAIBackendAPI") as backend:
                    with self.assertRaisesRegex(ValueError, message):
                        service.recover_manual(
                            identity, "policy-task", provider_binding_id="binding-1",
                            provider_account_identity="account-1", client_conversation_id="client-chat-1",
                        )
                    backend.assert_not_called()
                original = service.list_tasks(OWNER, ["policy-task"])["items"][0]
                self.assertEqual(original["status"], "error")
                self.assertEqual(original.get("error_code"), overrides.get("error_code", "content_policy_violation"))

    def test_workbench_manual_recovery_rejects_foreign_authority_and_active_latest_turn(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path)
            service = self.make_service(path)
            with mock.patch("services.openai_backend_api.OpenAIBackendAPI") as backend:
                with self.assertRaisesRegex(ValueError, "authority does not match"):
                    service.recover_manual(
                        OWNER, "policy-task", provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="other-product",
                    )
                backend.assert_not_called()
            AdoptionBackend.document = manual_image_document(latest_active=True)
            AdoptionBackend.downloads = 0
            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="read-token"),
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", AdoptionBackend),
            ):
                with self.assertRaisesRegex(ValueError, "latest manual request is still active"):
                    service.recover_manual(
                        OWNER, "policy-task", provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="client-chat-1",
                    )
            self.assertEqual(AdoptionBackend.downloads, 0)
            self.assertEqual(service.list_tasks(OWNER, ["policy-task"])["items"][0]["status"], "error")

    def test_workbench_manual_recovery_never_adopts_a_later_sibling_branch_image(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path, error_code="NO_IMAGE_GENERATED")
            service = self.make_service(path)
            # The image is newer and shares an ancestor, but the original sent
            # request is not on the current branch. Time alone is insufficient.
            AdoptionBackend.document = manual_image_document(divergent=True)
            AdoptionBackend.downloads = 0
            AdoptionBackend.resolved = []
            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="read-token"),
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", AdoptionBackend),
            ):
                with self.assertRaisesRegex(ValueError, "original request is not on the current conversation branch"):
                    service.recover_manual(
                        OWNER, "policy-task", provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="client-chat-1",
                    )
            self.assertEqual(AdoptionBackend.downloads, 0)
            self.assertEqual(AdoptionBackend.resolved, [])
            result = service.list_tasks(OWNER, ["policy-task"])["items"][0]
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["error_code"], "NO_IMAGE_GENERATED")
            self.assertNotIn("adopted_source_request_message_id", result)

    def test_workbench_manual_recovery_preserves_upstream_read_rate_limit(self):
        class ReadRateLimited(Exception):
            status_code = 429
            retry_after = 12

        class LimitedBackend(AdoptionBackend):
            def _get_conversation(self, _conversation_id):
                raise ReadRateLimited("upstream read limited")

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path)
            service = self.make_service(path)
            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="read-token"),
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", LimitedBackend),
            ):
                with self.assertRaises(ImageThreadError) as raised:
                    service.recover_manual(
                        OWNER, "policy-task", provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="client-chat-1",
                    )
            self.assertEqual(raised.exception.code, "RECOVERY_RATE_LIMITED")
            self.assertEqual(raised.exception.status, 429)
            self.assertEqual(raised.exception.retry_after, 12)
            self.assertEqual(service.list_tasks(OWNER, ["policy-task"])["items"][0]["status"], "error")

    def test_adoption_never_falls_back_when_latest_manual_turn_is_active(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path)
            service = self.make_service(path)
            AdoptionBackend.document = manual_image_document(latest_active=True)
            AdoptionBackend.downloads = 0
            AdoptionBackend.resolved = []
            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="read-token"),
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", AdoptionBackend),
            ):
                with self.assertRaisesRegex(ValueError, "latest manual request is still active"):
                    service.adopt_latest_conversation_image(
                        OWNER, "policy-task", provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="client-chat-1",
                        conversation_id="conversation-1",
                    )
            self.assertEqual(AdoptionBackend.downloads, 0)
            self.assertEqual(AdoptionBackend.resolved, [])
            self.assertEqual(service.list_tasks(OWNER, ["policy-task"])["items"][0]["status"], "error")

    def test_adoption_requires_the_original_sent_request_or_verified_sibling_anchor(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path)
            service = self.make_service(path)
            document = manual_image_document(divergent=True)
            document["mapping"].pop("original-request")
            AdoptionBackend.document = document
            AdoptionBackend.downloads = 0
            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="read-token"),
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", AdoptionBackend),
                mock.patch("services.protocol.conversation.format_image_result", return_value={"data": [{"url": "http://content/images/manual.png"}]}),
            ):
                with self.assertRaisesRegex(ValueError, "original request is not a verified user message"):
                    service.adopt_latest_conversation_image(
                        OWNER, "policy-task", provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="client-chat-1",
                        conversation_id="conversation-1",
                    )
                with self.assertRaisesRegex(ValueError, "original request is not a verified user message"):
                    service.adopt_latest_conversation_image(
                        OWNER, "policy-task", provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="client-chat-1",
                        conversation_id="conversation-1", source_request_message_id="manual-latest",
                        source_image_message_id="latest-image",
                    )
            self.assertEqual(AdoptionBackend.downloads, 0)

    def test_adoption_fetches_an_image_that_the_original_task_never_downloaded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path, data=[])
            service = self.make_service(path)
            AdoptionBackend.document = manual_image_document()
            AdoptionBackend.downloads = 0
            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="read-token"),
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", AdoptionBackend),
                mock.patch("services.protocol.conversation.format_image_result", return_value={"data": [{"url": "http://content/images/manual.png"}]}),
            ):
                result = service.adopt_latest_conversation_image(
                    OWNER, "policy-task", provider_binding_id="binding-1",
                    provider_account_identity="account-1", client_conversation_id="client-chat-1",
                    conversation_id="conversation-1",
                )
            self.assertEqual(result["data"], [{"url": "http://content/images/manual.png"}])
            self.assertEqual(AdoptionBackend.downloads, 1)

    def test_adoption_rejects_owner_authority_and_bound_account_mismatches_before_download(self):
        cases = (
            (OTHER_OWNER, "binding-1", "account-1", "client-chat-1", "conversation-1", "task not found"),
            (OWNER, "wrong-binding", "account-1", "client-chat-1", "conversation-1", "authority does not match"),
            (OWNER, "binding-1", "wrong-account", "client-chat-1", "conversation-1", "authority does not match"),
            (OWNER, "binding-1", "account-1", "wrong-client", "conversation-1", "authority does not match"),
            (OWNER, "binding-1", "account-1", "client-chat-1", "wrong-conversation", "authority does not match"),
        )
        for identity, binding, account, client_chat, conversation, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "image_tasks.json"
                write_policy_task(path)
                service = self.make_service(path)
                with mock.patch("services.openai_backend_api.OpenAIBackendAPI") as backend:
                    with self.assertRaisesRegex(ValueError, message):
                        service.adopt_latest_conversation_image(
                            identity, "policy-task", provider_binding_id=binding,
                            provider_account_identity=account, client_conversation_id=client_chat,
                            conversation_id=conversation,
                        )
                backend.assert_not_called()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path)
            service = self.make_service(path)
            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="different-account"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token") as token,
                mock.patch("services.openai_backend_api.OpenAIBackendAPI") as backend,
            ):
                with self.assertRaisesRegex(ValueError, "provider account identity changed"):
                    service.adopt_latest_conversation_image(
                        OWNER, "policy-task", provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="client-chat-1",
                        conversation_id="conversation-1",
                    )
            token.assert_not_called()
            backend.assert_not_called()

    def test_adoption_rejects_broken_or_nonlatest_source_nodes(self):
        mismatched_original_id = manual_image_document()
        mismatched_original_id["mapping"]["original-request"]["message"]["id"] = "different-request"
        for document, source_request, source_image, message in (
            (manual_image_document(broken=True), "", "", "conversation branch is invalid"),
            (mismatched_original_id, "", "", "original request is not a verified user message"),
            (manual_image_document(), "manual-old", "old-image", "specified manual request is not the latest"),
            (manual_image_document(), "manual-latest", "old-image", "specified image node is not the latest"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "image_tasks.json"
                write_policy_task(path)
                service = self.make_service(path)
                AdoptionBackend.document = document
                AdoptionBackend.downloads = 0
                with (
                    mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                    mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="read-token"),
                    mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                    mock.patch("services.openai_backend_api.OpenAIBackendAPI", AdoptionBackend),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        service.adopt_latest_conversation_image(
                            OWNER, "policy-task", provider_binding_id="binding-1",
                            provider_account_identity="account-1", client_conversation_id="client-chat-1",
                            conversation_id="conversation-1", source_request_message_id=source_request,
                            source_image_message_id=source_image,
                        )
                self.assertEqual(AdoptionBackend.downloads, 0)

    def test_concurrent_adoption_cannot_return_a_different_source_image(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path)
            service = self.make_service(path)
            AdoptionBackend.document = manual_image_document()
            AdoptionBackend.reads = AdoptionBackend.downloads = 0

            def concurrent_adoption(*_args, **_kwargs):
                with service._lock:
                    task = service._tasks["owner-1:policy-task"]
                    task.update({
                        "status": "success",
                        "data": [{"url": "http://content/images/other.png"}],
                        "adopted_source_request_message_id": "other-manual-request",
                        "adopted_source_image_message_id": "other-image-node",
                        "error": "",
                        "error_code": "",
                    })
                    service._save_locked()
                return {"data": [{"url": "http://content/images/manual.png"}]}

            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="read-token"),
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", AdoptionBackend),
                mock.patch("services.protocol.conversation.format_image_result", side_effect=concurrent_adoption),
            ):
                with self.assertRaisesRegex(ValueError, "concurrently adopted from a different"):
                    service.adopt_latest_conversation_image(
                        OWNER, "policy-task", provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="client-chat-1",
                        conversation_id="conversation-1",
                    )

            stored = service.list_tasks(OWNER, ["policy-task"])["items"][0]
            self.assertEqual(stored["adopted_source_request_message_id"], "other-manual-request")
            self.assertEqual(stored["adopted_source_image_message_id"], "other-image-node")
            self.assertEqual(stored["data"], [{"url": "http://content/images/other.png"}])
            self.assertEqual(AdoptionBackend.reads, 2)
            self.assertEqual(AdoptionBackend.downloads, 1)

    def test_adoption_save_failure_restores_the_original_receipt_before_retry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path)
            service = self.make_service(path)
            AdoptionBackend.document = manual_image_document()
            AdoptionBackend.reads = AdoptionBackend.downloads = 0
            AdoptionBackend.resolved = []
            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="read-token"),
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", AdoptionBackend),
                mock.patch("services.protocol.conversation.format_image_result", return_value={"data": [{"url": "http://content/images/manual.png"}]}),
            ):
                with mock.patch.object(service, "_save_locked", side_effect=OSError("disk full")):
                    with self.assertRaisesRegex(ValueError, "could not be verified"):
                        service.adopt_latest_conversation_image(
                            OWNER, "policy-task", provider_binding_id="binding-1",
                            provider_account_identity="account-1", client_conversation_id="client-chat-1",
                            conversation_id="conversation-1",
                        )

                in_memory = service.list_tasks(OWNER, ["policy-task"])["items"][0]
                self.assertEqual(in_memory["status"], "error")
                self.assertEqual(in_memory["error_code"], "content_policy_violation")
                self.assertEqual(in_memory["error"], "original policy failure")
                self.assertNotIn("adopted_source_request_message_id", in_memory)

                reloaded = self.make_service(path)
                on_disk = reloaded.list_tasks(OWNER, ["policy-task"])["items"][0]
                self.assertEqual(on_disk["status"], "error")
                self.assertEqual(on_disk["error_code"], "content_policy_violation")
                self.assertEqual(on_disk["error"], "original policy failure")
                self.assertNotIn("adopted_source_request_message_id", on_disk)

                recovered = service.adopt_latest_conversation_image(
                    OWNER, "policy-task", provider_binding_id="binding-1",
                    provider_account_identity="account-1", client_conversation_id="client-chat-1",
                    conversation_id="conversation-1",
                )

            self.assertEqual(recovered["status"], "success")
            self.assertEqual(recovered["adopted_source_request_message_id"], "manual-latest")
            self.assertEqual(AdoptionBackend.reads, 4)
            self.assertEqual(AdoptionBackend.downloads, 2)
            self.assertEqual(len(AdoptionBackend.resolved), 2)

    def test_adoption_is_idempotent_and_a_restart_preserves_its_provenance(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            write_policy_task(path)
            service = self.make_service(path)
            AdoptionBackend.document = manual_image_document()
            AdoptionBackend.reads = AdoptionBackend.downloads = 0
            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account-1"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="read-token"),
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", AdoptionBackend),
                mock.patch("services.protocol.conversation.format_image_result", return_value={"data": [{"url": "http://content/images/manual.png"}]}),
            ):
                first = service.adopt_latest_conversation_image(
                    OWNER, "policy-task", provider_binding_id="binding-1",
                    provider_account_identity="account-1", client_conversation_id="client-chat-1",
                    conversation_id="conversation-1",
                )
                second = service.adopt_latest_conversation_image(
                    OWNER, "policy-task", provider_binding_id="binding-1",
                    provider_account_identity="account-1", client_conversation_id="client-chat-1",
                    conversation_id="conversation-1",
                )
                with self.assertRaisesRegex(ValueError, "manual request does not match"):
                    service.adopt_latest_conversation_image(
                        OWNER, "policy-task", provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="client-chat-1",
                        conversation_id="conversation-1", source_request_message_id="manual-old",
                        source_image_message_id="latest-image",
                    )
                with self.assertRaisesRegex(ValueError, "image node does not match"):
                    service.adopt_latest_conversation_image(
                        OWNER, "policy-task", provider_binding_id="binding-1",
                        provider_account_identity="account-1", client_conversation_id="client-chat-1",
                        conversation_id="conversation-1", source_request_message_id="manual-latest",
                        source_image_message_id="old-image",
                    )
            reloaded = self.make_service(path)
            persisted = reloaded.list_tasks(OWNER, ["policy-task"])["items"][0]
            self.assertEqual(first, second)
            self.assertEqual(persisted["adopted_source_request_message_id"], "manual-latest")
            self.assertEqual(persisted["adopted_source_image_message_id"], "latest-image")
            self.assertEqual(persisted["adopted_from_error_code"], "content_policy_violation")
            self.assertEqual(AdoptionBackend.reads, 2)
            self.assertEqual(AdoptionBackend.downloads, 1)

    def test_success_task_persists_to_new_service_instance(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            service = self.make_service(path)
            service.submit_generation(
                OWNER,
                client_task_id="persisted-task",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )
            wait_for_task(service, OWNER, "persisted-task", "success")

            reloaded = self.make_service(path)
            result = reloaded.list_tasks(OWNER, ["persisted-task"])

            self.assertEqual(result["missing_ids"], [])
            self.assertEqual(result["items"][0]["status"], "success")
            self.assertEqual(result["items"][0]["data"][0]["url"], "http://example.test/image.png")

    def test_startup_marks_unfinished_tasks_as_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "id": "queued-task",
                                "owner_id": "owner-1",
                                "status": "queued",
                                "mode": "generate",
                                "model": "gpt-image-2",
                                "created_at": "2099-01-01 00:00:00",
                                "updated_at": "2099-01-01 00:00:00",
                            },
                            {
                                "id": "running-task",
                                "owner_id": "owner-1",
                                "status": "running",
                                "mode": "generate",
                                "model": "gpt-image-2",
                                "created_at": "2099-01-01 00:00:00",
                                "updated_at": "2099-01-01 00:00:00",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            service = self.make_service(path)
            result = service.list_tasks(OWNER, ["queued-task", "running-task"])

            self.assertEqual([item["status"] for item in result["items"]], ["error", "error"])
            self.assertTrue(all("已中断" in item.get("error", "") for item in result["items"]))
            self.assertTrue(
                all(
                    item.get("error_code") == "CONVERSATION_OUTCOME_UNKNOWN"
                    for item in result["items"]
                )
            )


if __name__ == "__main__":
    unittest.main()
