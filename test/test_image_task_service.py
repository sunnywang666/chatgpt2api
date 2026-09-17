from __future__ import annotations

import json
import tempfile
import time
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

from services.image_task_service import ImageTaskService, _authoritative_image_failure
from services.openai_backend_api import ChatRequirements, ImageContentPolicyError, OpenAIBackendAPI
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
    def test_generation_post_records_submission_boundary_before_network_call(self):
        backend = object.__new__(OpenAIBackendAPI)
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
            def post(self, *_args, **_kwargs):
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

    def test_generation_post_is_not_called_when_submission_boundary_cannot_be_persisted(self):
        backend = object.__new__(OpenAIBackendAPI)
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
        session = mock.Mock()
        backend.session = session

        with self.assertRaisesRegex(OSError, "receipt save failed"):
            backend._start_image_generation(
                "cat", ChatRequirements(token="requirements"), "conduit", "gpt-image-2",
                conversation_id="conversation-1", parent_message_id="parent-1",
            )

        session.post.assert_not_called()
        self.assertFalse(backend.image_submission_started)

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

                def _poll_image_results(self, conversation_id, timeout, request_message_id=""):
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

                    def _poll_image_results(self, conversation_id, timeout, request_message_id=""):
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
                    self.assertIn("status=404", task["error"])
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
            self.assertIn("account identity changed", task["error"])
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
            self.assertIn("status=429", task["error"])
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
        with tempfile.TemporaryDirectory() as tmp_dir:
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

            class FakeBackend:
                def __init__(self, access_token=None, proxy_url=None):
                    self.access_token = access_token

                def _get_conversation(self, _conversation_id):
                    return {
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

                def close(self):
                    return None

            with (
                mock.patch("services.account_service.account_service.get_bound_account_identity", return_value="account_opaque_a"),
                mock.patch("services.account_service.account_service.get_bound_text_access_token", return_value="bound-token"),
                mock.patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
                mock.patch("services.account_service.account_service.release_image_slot"),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", FakeBackend),
            ):
                resumed = service.resume_poll(OWNER, "terminal-no-image-task", 30, "http://content-provider")
                self.assertIn(resumed["status"], {"running", "error"}, resumed)
                task = wait_for_task(service, OWNER, "terminal-no-image-task", "error")

            self.assertEqual(task["error_code"], "NO_IMAGE_GENERATED")

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
