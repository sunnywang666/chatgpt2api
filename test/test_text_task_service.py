import tempfile
import threading
import unittest
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from services.conversation_binding_service import ConversationBindingError
from services.text_task_service import TextTaskService, ContinuationExecutor


class QueuedExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        self.calls.append((function, args))

    def run(self):
        function, args = self.calls.pop(0)
        function(*args)


class ManualClock:
    def __init__(self, value=1000.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class TextTaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "tasks.sqlite3"
        self.queue = QueuedExecutor()
        self.body = {"client_request_id": "attempt-1", "client_conversation_id": "product-gallery",
                     "messages": [{"role": "user", "content": "private prompt and image bytes"}]}

    def test_receipt_precedes_generation_and_completed_result_survives_restart(self):
        def runner(body, on_cursor):
            on_cursor({"provider_binding_id": "binding", "provider_account_identity": "account", "conversation_id": "chat"})
            observed = service.read("owner", "attempt-1")
            self.assertEqual(observed["status"], "running")
            self.assertEqual(observed["conversation_id"], "chat")
            return {"content": "answer", "conversation_id": "chat", "parent_message_id": "answer-id"}
        service = TextTaskService(self.path, runner, self.queue)
        self.assertEqual(service.submit("owner", self.body)["status"], "queued")
        self.assertEqual(service.submit("owner", self.body)["status"], "queued")
        self.assertEqual(len(self.queue.calls), 1)
        self.assertNotIn(b"private prompt", self.path.read_bytes())
        self.queue.run()
        restarted = TextTaskService(self.path, runner, self.queue)
        self.assertEqual(restarted.read("owner", "attempt-1")["content"], "answer")
        self.assertEqual(restarted.submit("owner", self.body)["status"], "succeeded")
        self.assertEqual(len(self.queue.calls), 0)
        self.assertEqual(restarted.read("other-owner", "attempt-1")["status"], "not_found")

    def test_concurrent_same_request_is_scheduled_once_and_changed_input_conflicts(self):
        service = TextTaskService(self.path, executor=self.queue)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: service.submit("owner", self.body), range(20)))
        self.assertEqual(len(self.queue.calls), 1)
        with self.assertRaises(ConversationBindingError) as error:
            service.submit("owner", {**self.body, "model": "different"})
        self.assertEqual(error.exception.code, "CONVERSATION_REQUEST_CONFLICT")

    def test_restart_and_upstream_uncertainty_preserve_cursor_without_resubmitting(self):
        def runner(body, on_cursor):
            on_cursor({"provider_binding_id": "same-binding", "conversation_id": "same-chat"})
            raise ConversationBindingError("disconnected", code="CONVERSATION_OUTCOME_UNKNOWN")
        service = TextTaskService(self.path, runner, self.queue)
        service.submit("owner", self.body)
        self.queue.run()
        restarted = TextTaskService(self.path, runner, self.queue)
        self.assertEqual(restarted.read("owner", "attempt-1")["status"], "unknown")
        result = restarted.read("owner", "attempt-1")
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["conversation_id"], "same-chat")
        restarted.submit("owner", self.body)
        self.assertEqual(len(self.queue.calls), 0)

    def test_unknown_recovery_is_singleflight_across_service_instances(self):
        clock = ManualClock()
        calls = []
        started, release = threading.Event(), threading.Event()

        def reader(receipt):
            calls.append(receipt["recovery_attempt"])
            started.set()
            release.wait(2)
            return {"status": "running"}

        service = TextTaskService(self.path, executor=self.queue, clock=clock, recovery_reader=reader)
        service.submit("owner", self.body)
        service._update("owner", "attempt-1", status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                        provider_binding_id="binding", provider_account_identity="account",
                        conversation_id="chat")
        restarted = TextTaskService(self.path, executor=self.queue, clock=clock, recovery_reader=reader)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(service.read, "owner", "attempt-1")
                self.assertTrue(started.wait(1))
                second = pool.submit(restarted.read, "owner", "attempt-1")
                second_result = second.result(timeout=1)
                release.set()
                first_result = first.result(timeout=2)
        finally:
            release.set()
        self.assertEqual(calls, [1])
        self.assertEqual(second_result["status"], "unknown")
        self.assertEqual(first_result["status"], "unknown")
        self.assertEqual(first_result["recovery_attempt"], 1)
        self.assertGreater(first_result["recovery_next_at"], clock())
        self.assertEqual(restarted.read("other-owner", "attempt-1")["status"], "not_found")
        self.assertEqual(len(self.queue.calls), 1, "recovery must not resubmit the original text")

    def test_recovery_failure_is_safe_and_retries_after_persisted_cooldown(self):
        clock = ManualClock()
        calls = []

        def reader(receipt):
            calls.append(receipt["recovery_attempt"])
            if len(calls) == 1:
                raise RuntimeError("upstream token=secret should never be persisted")
            return {"status": "succeeded", "content": "answer", "conversation_id": "chat",
                    "parent_message_id": "answer-id", "binding_status": "bound"}

        service = TextTaskService(self.path, executor=self.queue, clock=clock, recovery_reader=reader)
        service.submit("owner", self.body)
        service._update("owner", "attempt-1", status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                        provider_binding_id="binding", provider_account_identity="account",
                        conversation_id="chat")
        failed = service.read("owner", "attempt-1")
        self.assertEqual(failed["status"], "unknown")
        self.assertEqual(failed["recovery_error_code"], "RECOVERY_READ_FAILED")
        self.assertEqual(failed["recovery_phase"], "read_text_request")
        self.assertEqual(failed["recovery_attempt"], 1)
        self.assertNotIn(b"upstream token=secret", self.path.read_bytes())

        restarted = TextTaskService(self.path, executor=self.queue, clock=clock, recovery_reader=reader)
        self.assertEqual(restarted.read("owner", "attempt-1")["status"], "unknown")
        self.assertEqual(calls, [1], "restart must honor the persisted recovery cooldown")
        clock.advance(TextTaskService.RECOVERY_BASE_BACKOFF_SECONDS + 1)
        recovered = restarted.read("owner", "attempt-1")
        self.assertEqual(recovered["status"], "succeeded")
        self.assertEqual(recovered["content"], "answer")
        self.assertIsNone(recovered["recovery_next_at"])
        self.assertEqual(calls, [1, 2])
        self.assertEqual(len(self.queue.calls), 1, "recovery must never schedule a second runner call")

    def test_recovery_success_requires_nonempty_content_and_parent_cursor(self):
        service = TextTaskService(self.path, executor=self.queue)
        invalid_results = [
            {"status": "succeeded", "content": "", "parent_message_id": "answer", "binding_status": "bound"},
            {"status": "succeeded", "content": "answer", "parent_message_id": "", "binding_status": "bound"},
            {"status": "succeeded", "content": "answer", "parent_message_id": "answer", "binding_status": "unknown"},
        ]
        for invalid in invalid_results:
            safe_result, error_code, phase = service._safe_recovery_result(invalid)
            self.assertIsNone(safe_result)
            self.assertEqual(error_code, "RECOVERY_INVALID_RESULT")
            self.assertEqual(phase, "read_text_result")
        safe_result, error_code, phase = service._safe_recovery_result(
            {"status": "succeeded", "content": "answer", "parent_message_id": "answer", "binding_status": "bound"}
        )
        self.assertEqual(safe_result, {"content": "answer", "parent_message_id": "answer", "binding_status": "bound"})
        self.assertIsNone(error_code)
        self.assertIsNone(phase)

    def test_late_runner_updates_cannot_regress_recovered_success(self):
        clock = ManualClock()

        def reader(receipt):
            return {"status": "succeeded", "content": "fresh", "parent_message_id": "fresh-answer",
                    "binding_status": "bound"}

        service = TextTaskService(self.path, executor=self.queue, clock=clock, recovery_reader=reader)
        service.submit("owner", self.body)
        service._update("owner", "attempt-1", status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                        provider_binding_id="binding", provider_account_identity="account",
                        conversation_id="chat")
        recovered = service.read("owner", "attempt-1")
        self.assertEqual(recovered["status"], "succeeded")
        self.assertIsNone(recovered["error_code"])

        def stale_update():
            service._update("owner", "attempt-1", status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                            parent_message_id="stale-answer")

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(stale_update) for _ in range(2)]
            for future in futures:
                future.result(timeout=2)
        final = service.read("owner", "attempt-1")
        self.assertEqual(final["status"], "succeeded")
        self.assertEqual(final["content"], "fresh")
        self.assertEqual(final["parent_message_id"], "fresh-answer")
        self.assertIsNone(final["error_code"])

    def test_expired_old_recovery_cannot_overwrite_newer_success(self):
        clock = ManualClock()
        calls = []
        first_started, release_first = threading.Event(), threading.Event()

        def reader(receipt):
            calls.append(receipt["recovery_attempt"])
            if len(calls) == 1:
                first_started.set()
                release_first.wait(2)
                return {"status": "running"}
            return {"status": "succeeded", "content": "fresh", "conversation_id": "chat",
                    "parent_message_id": "fresh-answer", "binding_status": "bound"}

        service = TextTaskService(self.path, executor=self.queue, clock=clock, recovery_reader=reader)
        service.submit("owner", self.body)
        service._update("owner", "attempt-1", status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                        provider_binding_id="binding", provider_account_identity="account",
                        conversation_id="chat")
        restarted = TextTaskService(self.path, executor=self.queue, clock=clock, recovery_reader=reader)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                old_read = pool.submit(service.read, "owner", "attempt-1")
                self.assertTrue(first_started.wait(1))
                clock.advance(TextTaskService.RECOVERY_LEASE_SECONDS + 1)
                fresh = restarted.read("owner", "attempt-1")
                self.assertEqual(fresh["status"], "succeeded")
                self.assertEqual(fresh["content"], "fresh")
                release_first.set()
                old_read.result(timeout=2)
        finally:
            release_first.set()
        final = restarted.read("owner", "attempt-1")
        self.assertEqual(final["status"], "succeeded")
        self.assertEqual(final["content"], "fresh")
        self.assertEqual(calls, [1, 2])
        self.assertEqual(len(self.queue.calls), 1, "recovery must not resubmit the original text")

    def test_restart_replays_only_unstarted_requests_with_same_input_and_message_identity(self):
        writes = []
        service = TextTaskService(self.path, lambda body, on_cursor: writes.append(body) or {"content": "ok"}, self.queue)
        first = service.submit("owner", self.body)
        restarted = TextTaskService(self.path, service.runner, self.queue)
        self.assertEqual(restarted.read("owner", "attempt-1")["status"], "not_started")
        with self.assertRaises(ConversationBindingError):
            restarted.submit("owner", {**self.body, "model": "changed"})
        restarted.submit("owner", self.body)
        self.queue.run()  # Old process cannot later run a transferred queued receipt.
        self.assertEqual(writes, [])
        self.queue.run()
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0]["_request_message_id"], first["request_message_id"])
        self.assertEqual(restarted.read("owner", "attempt-1")["status"], "succeeded")

    def test_running_request_after_restart_remains_unknown_and_never_resubmits(self):
        service = TextTaskService(self.path, executor=self.queue)
        service.submit("owner", self.body)
        service._update("owner", "attempt-1", status="running")
        restarted = TextTaskService(self.path, executor=self.queue)
        self.assertEqual(restarted.read("owner", "attempt-1")["status"], "unknown")
        restarted.submit("owner", self.body)
        self.assertEqual(len(self.queue.calls), 1)

    def test_legacy_account_rejection_before_any_chat_reuses_the_original_request(self):
        def rejected(body, on_cursor):
            on_cursor({"provider_binding_id": "wrong-plan", "provider_account_identity": "free"})
            raise ConversationBindingError("bound account cannot serve model")
        service = TextTaskService(self.path, rejected, self.queue)
        original = service.submit("owner", self.body)
        self.queue.run()
        self.assertEqual(service.read("owner", "attempt-1")["status"], "not_started")
        service.runner = lambda body, on_cursor: {"content": "done", "conversation_id": "new-product-chat"}
        service.submit("owner", self.body)
        self.queue.run()
        self.assertEqual(service.read("owner", "attempt-1")["request_message_id"], original["request_message_id"])
        self.assertEqual(service.read("owner", "attempt-1")["status"], "succeeded")
        # Simulate a legacy receipt written by an older owner. The normal
        # update path must reject this kind of late callback after success.
        with service._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT receipt FROM requests WHERE owner=? AND id=?", ("owner", "attempt-1")).fetchone()
            legacy_failed = {**json.loads(row[0]), "status": "failed", "error_code": "CONVERSATION_BINDING_UNAVAILABLE"}
            db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?",
                       (json.dumps(legacy_failed), "owner", "attempt-1"))
        self.assertEqual(service.read("owner", "attempt-1")["status"], "failed", "existing chats cannot be reclassified as never sent")

    def test_ready_continuation_precedes_queued_new_products(self):
        executor = ContinuationExecutor(max_workers=1)
        started, finish = threading.Event(), threading.Event()
        order = []
        def run(body):
            order.append(body["id"])
            if body["id"] == "first":
                started.set()
                finish.wait(3)
        try:
            first = executor.submit(run, {"id": "first"})
            self.assertTrue(started.wait(1))
            gallery = executor.submit(run, {"id": "new-gallery"})
            copy = executor.submit(run, {"id": "ready-copy", "conversation_id": "same-chat"})
            finish.set()
            for future in (first, gallery, copy):
                future.result(timeout=3)
            self.assertEqual(order, ["first", "ready-copy", "new-gallery"])
        finally:
            finish.set()
            executor.shutdown()

    def test_known_rejection_is_terminal_and_other_request_can_continue(self):
        def runner(body, on_cursor):
            if body["client_request_id"] == "attempt-1":
                raise ConversationBindingError("no account", code="CONVERSATION_BINDING_UNAVAILABLE")
            return {"content": "ok"}
        service = TextTaskService(self.path, runner, self.queue)
        service.submit("owner", self.body)
        service.submit("owner", {**self.body, "client_request_id": "attempt-2"})
        self.queue.run()
        self.assertEqual(service.read("owner", "attempt-1")["status"], "failed")
        self.queue.run()
        self.assertEqual(service.read("owner", "attempt-2")["status"], "succeeded")

    def test_slow_upstream_does_not_block_submission_or_readback(self):
        started, finish = threading.Event(), threading.Event()
        def runner(body, on_cursor):
            started.set()
            finish.wait(3)
            return {"content": "ok"}
        with ThreadPoolExecutor(max_workers=1) as executor:
            service = TextTaskService(self.path, runner, executor)
            try:
                service.submit("owner", self.body)
                self.assertTrue(started.wait(1))
                self.assertEqual(service.read("owner", "attempt-1")["status"], "running")
            finally:
                finish.set()

    def test_http_submission_returns_receipt_and_reads_are_owner_scoped(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.testclient import TestClient
        from unittest import mock
        from api.ai import create_router
        service = TextTaskService(self.path, executor=self.queue)
        app = FastAPI(); app.include_router(create_router())
        with mock.patch("api.ai.text_task_service", service), mock.patch("api.ai.filter_or_log", mock.AsyncMock()), mock.patch("api.ai.require_identity", side_effect=lambda token: {"id": token}):
            with TestClient(app) as client:
                response = client.post("/api/conversation-bindings/text", json=self.body, headers={"Authorization": "owner"})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["status"], "queued")
                self.assertEqual(client.get("/api/conversation-bindings/text-requests/attempt-1", headers={"Authorization": "owner"}).json()["status"], "queued")
                self.assertEqual(client.get("/api/conversation-bindings/text-requests/attempt-1", headers={"Authorization": "other"}).json()["status"], "not_found")
        self.assertEqual(len(self.queue.calls), 1)

    def test_last_user_message_has_the_saved_identity_for_text_and_gallery(self):
        from services.openai_backend_api import OpenAIBackendAPI
        backend = object.__new__(OpenAIBackendAPI)
        backend.text_request_message_id = "request-user-turn"
        for final in ["copy", [{"type": "text", "text": "gallery"}]]:
            rows = backend._api_messages_to_conversation_messages([
                {"role": "user", "content": "previous"}, {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": final}])
            self.assertEqual(rows[-1]["id"], "request-user-turn")
            self.assertEqual(len({row["id"] for row in rows}), 3)

    def test_recovery_requires_this_requests_user_turn_not_previous_answer(self):
        from services.conversation_binding_service import ConversationBindingService
        from unittest import mock
        from contextlib import nullcontext
        backend = mock.Mock()
        previous = {"id": "old-answer", "author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True,
                    "content": {"content_type": "text", "parts": ["old gallery result"]}}
        backend._get_conversation.return_value = {"conversation_id": "chat", "current_node": "old-answer", "mapping": {"old-answer": {"message": previous}}}
        receipt = {"provider_binding_id": "binding", "provider_account_identity": "account", "client_conversation_id": "copy",
                   "conversation_id": "chat", "request_message_id": "new-user"}
        with mock.patch("services.conversation_binding_service.account_service.get_bound_account_identity", return_value="account"), mock.patch("services.conversation_binding_service.account_service.get_bound_text_access_token", return_value="synthetic"), mock.patch("services.conversation_binding_service.account_service.conversation_binding_lock", return_value=nullcontext()), mock.patch("services.conversation_binding_service.OpenAIBackendAPI", return_value=backend):
            with self.assertRaises(ConversationBindingError):
                ConversationBindingService().read_text_request(receipt)
            backend._get_conversation.return_value = {"conversation_id": "chat", "current_node": "new-answer", "mapping": {
                "new-user": {"parent": "old-answer", "message": {"id": "new-user", "author": {"role": "user"}}},
                "new-answer": {"parent": "new-user", "message": {**previous, "id": "new-answer", "content": {"content_type": "text", "parts": ["new copy"]}}}}}
            result = ConversationBindingService().read_text_request(receipt)
            self.assertEqual(result["content"], "new copy")

    def test_bound_text_keeps_chat_history_while_unbound_behavior_is_unchanged(self):
        from services.openai_backend_api import OpenAIBackendAPI
        backend = object.__new__(OpenAIBackendAPI)
        args = ([{"role": "user", "content": "gallery"}], "gpt-5-6-instant", "UTC")
        self.assertTrue(backend._conversation_payload(*args)["history_and_training_disabled"])
        backend.retain_bound_conversation = True
        self.assertFalse(backend._conversation_payload(*args)["history_and_training_disabled"])


if __name__ == "__main__":
    unittest.main()
