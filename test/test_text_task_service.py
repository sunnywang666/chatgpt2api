import tempfile
import threading
import unittest
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from services.conversation_binding_service import (
    ConversationBindingError,
    ConversationBindingService,
    RECOVERY_CONVERSATION_COVERAGE_VERSION_FIELD,
    RECOVERY_CONVERSATION_SCAN_FIELD,
    TextRecoveryReason,
)
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

    def test_timeout_cursor_does_not_replace_original_request_parent(self):
        captured = []
        def runner(body, on_cursor):
            on_cursor({"provider_binding_id": "binding", "provider_account_identity": "account", "conversation_id": "chat"})
            raise ConversationBindingError("timeout", code="CONVERSATION_OUTCOME_UNKNOWN",
                                           parent_message_id="later-cursor")
        def reader(receipt):
            captured.append(receipt)
            return {"status": "running"}
        service = TextTaskService(self.path, runner, self.queue, recovery_reader=reader)
        service.submit("owner", {**self.body, "parent_message_id": "original-parent"})
        self.queue.run()
        result = service.read("owner", "attempt-1")
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(captured[0]["parent_message_id"], "later-cursor")
        self.assertEqual(captured[0]["request_parent_message_id"], "original-parent")
        self.assertEqual(len(self.queue.calls), 0)

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
            safe_result, error_code, phase, recovery_reason = service._safe_recovery_result(invalid)
            self.assertIsNone(safe_result)
            self.assertEqual(error_code, "RECOVERY_INVALID_RESULT")
            self.assertEqual(phase, "read_text_result")
            self.assertIsNone(recovery_reason)
        safe_result, error_code, phase, recovery_reason = service._safe_recovery_result(
            {"status": "succeeded", "content": "answer", "parent_message_id": "answer", "binding_status": "bound"}
        )
        self.assertEqual(safe_result, {"content": "answer", "parent_message_id": "answer", "binding_status": "bound"})
        self.assertIsNone(error_code)
        self.assertIsNone(phase)
        self.assertIsNone(recovery_reason)

    def test_safe_request_recovery_reason_is_persisted_without_exception_text(self):
        clock = ManualClock()

        def reader(receipt):
            return {"status": "running", "recovery_reason": "REQUEST_RESULT_INCOMPLETE"}

        service = TextTaskService(self.path, executor=self.queue, clock=clock, recovery_reader=reader)
        service.submit("owner", self.body)
        service._update("owner", "attempt-1", status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                        provider_binding_id="binding", provider_account_identity="account",
                        conversation_id="chat")
        result = service.read("owner", "attempt-1")
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["recovery_reason"], "REQUEST_RESULT_INCOMPLETE")
        self.assertEqual(result["recovery_error_code"], "UPSTREAM_OUTCOME_UNKNOWN")

    def test_explicit_bounded_recovery_authorizes_one_replacement_after_three_qualified_reads(self):
        clock = ManualClock()

        def reader(_receipt):
            raise ConversationBindingError(
                "original request is absent",
                code="CONVERSATION_BINDING_MISMATCH",
                recovery_reason="REQUEST_MESSAGE_NOT_FOUND",
            )

        service = TextTaskService(
            self.path, executor=self.queue, clock=clock, recovery_reader=reader,
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="binding", provider_account_identity="account",
            conversation_id="chat",
        )
        clock.advance(TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS + 1)

        first = service.recover("owner", "attempt-1", True)
        self.assertEqual(first["status"], "unknown")
        self.assertEqual(first["recovery_no_result_reads"], 1)
        clock.advance(TextTaskService.RECOVERY_BASE_BACKOFF_SECONDS + 1)
        second = service.recover("owner", "attempt-1", True)
        self.assertEqual(second["status"], "unknown")
        self.assertEqual(second["recovery_no_result_reads"], 2)
        clock.advance(TextTaskService.RECOVERY_BASE_BACKOFF_SECONDS * 2 + 1)
        final = service.recover("owner", "attempt-1", True)

        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["error_code"], "RESULT_UNRECOVERABLE")
        self.assertEqual(final["upstream_outcome"], "unknown")
        self.assertTrue(final["recovery_retryable"])
        self.assertFalse(final["recovery_requires_new_conversation"])
        self.assertEqual(service.recover("owner", "attempt-1", True), final)
        self.assertEqual(len(self.queue.calls), 1, "replacement is authorized, never sent here")

    def test_missing_conversation_receipt_uses_bounded_recovery_and_keeps_paid_binding(self):
        clock = ManualClock()
        calls = []

        def reader(receipt):
            calls.append(receipt)
            self.assertNotIn("conversation_id", receipt)
            raise ConversationBindingError(
                "original request conversation cannot be attributed uniquely",
                code="CONVERSATION_OUTCOME_UNKNOWN",
                recovery_reason="REQUEST_CONVERSATION_UNATTRIBUTABLE",
            )

        service = TextTaskService(
            self.path, executor=self.queue, clock=clock, recovery_reader=reader,
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="paid-binding",
            provider_account_identity="paid-account",
        )
        clock.advance(TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS + 1)

        for delay in (31, 61, 121):
            result = service.recover("owner", "attempt-1", True)
            clock.advance(delay)

        self.assertEqual(len(calls), 3)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "RESULT_UNRECOVERABLE")
        self.assertEqual(result["upstream_outcome"], "unknown")
        self.assertTrue(result["recovery_retryable"])
        self.assertTrue(result["recovery_requires_new_conversation"])
        self.assertEqual(result["provider_binding_id"], "paid-binding")
        self.assertEqual(result["provider_account_identity"], "paid-account")
        self.assertNotIn("conversation_id", result)
        self.assertNotIn("parent_message_id", result)
        self.assertEqual(len(self.queue.calls), 1, "recovery never resubmits upstream")

    def test_coverage_aware_empty_read_does_not_inherit_legacy_qualified_count(self):
        clock = ManualClock()

        def reader(_receipt):
            raise ConversationBindingError(
                "covered account history contains no matching request",
                code="CONVERSATION_OUTCOME_UNKNOWN",
                recovery_reason=TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value,
                recovery_scan={},
                recovery_coverage_version=1,
            )

        service = TextTaskService(
            self.path, executor=self.queue, clock=clock, recovery_reader=reader,
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="paid-binding",
            provider_account_identity="paid-account",
            recovery_no_result_reads=2,
        )
        clock.advance(TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS + 1)

        result = service.recover("owner", "attempt-1", True)

        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["recovery_no_result_reads"], 1)
        self.assertNotIn(RECOVERY_CONVERSATION_COVERAGE_VERSION_FIELD, result)
        with service._db() as db:
            stored = json.loads(db.execute(
                "SELECT receipt FROM requests WHERE owner=? AND id=?",
                ("owner", "attempt-1"),
            ).fetchone()[0])
        self.assertEqual(stored[RECOVERY_CONVERSATION_COVERAGE_VERSION_FIELD], 1)

    def test_exact_root_lookup_anchor_is_persisted_for_future_reads(self):
        clock = ManualClock()

        def reader(_receipt):
            return {
                "status": "unknown",
                "binding_status": "unknown",
                "conversation_id": "recovered-chat",
                "parent_message_id": "original-request-node",
                "request_parent_message_id": "",
                "recovery_reason": "REQUEST_RESULT_NOT_FOUND",
            }

        service = TextTaskService(
            self.path, executor=self.queue, clock=clock, recovery_reader=reader,
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="paid-binding",
            provider_account_identity="paid-account",
        )

        recovered = service.read("owner", "attempt-1")

        self.assertEqual(recovered["conversation_id"], "recovered-chat")
        self.assertEqual(recovered["parent_message_id"], "original-request-node")
        self.assertIn("request_parent_message_id", recovered)
        self.assertEqual(recovered["request_parent_message_id"], "")
        self.assertEqual(recovered["recovery_no_result_reads"], 0)

    def test_missing_conversation_scan_timeout_never_qualifies_as_no_result(self):
        clock = ManualClock()
        service = TextTaskService(
            self.path,
            executor=self.queue,
            clock=clock,
            recovery_reader=lambda _receipt: (_ for _ in ()).throw(
                TimeoutError("recent conversation recovery scan exceeded its time budget")
            ),
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="paid-binding",
            provider_account_identity="paid-account",
        )
        clock.advance(TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS + 1)

        for _ in range(3):
            result = service.recover("owner", "attempt-1", True)
            clock.advance(TextTaskService.RECOVERY_MAX_BACKOFF_SECONDS + 1)

        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["recovery_error_code"], "RECOVERY_READ_FAILED")
        self.assertEqual(result.get("recovery_no_result_reads", 0), 0)
        self.assertNotEqual(result.get("error_code"), "RESULT_UNRECOVERABLE")

    def test_oversized_recent_response_never_qualifies_as_no_result(self):
        clock = ManualClock()
        service = TextTaskService(
            self.path,
            executor=self.queue,
            clock=clock,
            recovery_reader=lambda _receipt: (_ for _ in ()).throw(
                RuntimeError("recent conversations response exceeds the requested limit")
            ),
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="paid-binding",
            provider_account_identity="paid-account",
        )
        clock.advance(TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS + 1)

        for _ in range(3):
            result = service.recover("owner", "attempt-1", True)
            clock.advance(TextTaskService.RECOVERY_MAX_BACKOFF_SECONDS + 1)

        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["recovery_error_code"], "RECOVERY_READ_FAILED")
        self.assertEqual(result.get("recovery_no_result_reads", 0), 0)
        self.assertNotEqual(result.get("error_code"), "RESULT_UNRECOVERABLE")

    def test_qualified_recovery_backoff_ignores_old_attempts_but_rate_limit_does_not(self):
        clock = ManualClock()

        def missing_reader(_receipt):
            raise ConversationBindingError(
                "original request is absent",
                code="CONVERSATION_BINDING_MISMATCH",
                recovery_reason="REQUEST_MESSAGE_NOT_FOUND",
            )

        service = TextTaskService(
            self.path, executor=self.queue, clock=clock, recovery_reader=missing_reader,
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="binding", provider_account_identity="account",
            conversation_id="chat", recovery_attempt=99,
        )
        clock.advance(TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS + 1)

        qualified = service.recover("owner", "attempt-1", True)

        self.assertEqual(qualified["recovery_attempt"], 100)
        self.assertEqual(qualified["recovery_no_result_reads"], 1)
        self.assertEqual(
            qualified["recovery_next_at"] - clock(),
            TextTaskService.RECOVERY_BASE_BACKOFF_SECONDS,
        )

        class RateLimitError(RuntimeError):
            status_code = 429

        rate_limit_path = self.path.with_name("rate-limit.sqlite3")
        rate_limit_body = {**self.body, "client_request_id": "rate-limit-attempt"}
        rate_limited = TextTaskService(
            rate_limit_path,
            executor=QueuedExecutor(),
            clock=clock,
            recovery_reader=lambda _receipt: (_ for _ in ()).throw(
                RateLimitError("upstream status=429")
            ),
        )
        rate_limited.submit("owner", rate_limit_body)
        rate_limited._update(
            "owner", "rate-limit-attempt", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="binding", provider_account_identity="account",
            conversation_id="chat", recovery_attempt=99,
        )

        limited = rate_limited.recover("owner", "rate-limit-attempt", True)

        self.assertEqual(limited["recovery_attempt"], 100)
        self.assertEqual(limited.get("recovery_no_result_reads", 0), 0)
        self.assertEqual(limited["recovery_error_code"], "RECOVERY_READ_FAILED")
        self.assertEqual(
            limited["recovery_next_at"] - clock(),
            TextTaskService.RECOVERY_MAX_BACKOFF_SECONDS,
        )

    def test_unrecoverable_opt_in_does_not_count_running_or_read_failures(self):
        clock = ManualClock()
        results = iter([
            {"status": "running", "recovery_reason": "REQUEST_RESULT_INCOMPLETE"},
            RuntimeError("network unavailable"),
            {"status": "running", "recovery_reason": "REQUEST_RESULT_INCOMPLETE"},
        ])

        def reader(_receipt):
            result = next(results)
            if isinstance(result, Exception):
                raise result
            return result

        service = TextTaskService(
            self.path, executor=self.queue, clock=clock, recovery_reader=reader,
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="binding", provider_account_identity="account",
            conversation_id="chat",
        )
        clock.advance(TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS + 1)
        for delay in (31, 61, 121):
            result = service.recover("owner", "attempt-1", True)
            self.assertEqual(result["status"], "unknown")
            self.assertEqual(result.get("recovery_no_result_reads", 0), 0)
            clock.advance(delay)

    def test_slow_missing_conversation_scan_persists_progress_across_restart(self):
        recovery_clock = ManualClock()

        class ScanClock:
            value = 0.0

            @classmethod
            def monotonic(cls):
                return cls.value

            @classmethod
            def advance(cls, seconds):
                cls.value += seconds

        class SlowBackend:
            def __init__(self):
                self.list_calls = 0
                self.detail_calls = []

            def _list_recent_conversations(self, *, offset, **_kwargs):
                self.list_calls += 1
                ScanClock.advance(1.5)
                update_time = 1100 if offset == 0 else 900
                return [
                    {
                        "id": f"conversation-{offset + index}",
                        "update_time": update_time - index,
                    }
                    for index in range(20)
                ]

            def _get_conversation(self, conversation_id, *, timeout_secs):
                self.detail_calls.append((conversation_id, timeout_secs))
                ScanClock.advance(9.5)
                return {"conversation_id": conversation_id, "mapping": {}}

        backend = SlowBackend()

        def reader(receipt):
            located, _document = ConversationBindingService._locate_text_request_conversation(
                backend, receipt,
            )
            return located

        service = TextTaskService(
            self.path,
            executor=self.queue,
            clock=recovery_clock,
            recovery_reader=reader,
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="binding", provider_account_identity="paid-account",
        )
        recovery_clock.advance(TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS + 1)

        with mock.patch(
            "services.conversation_binding_service.time.monotonic",
            side_effect=ScanClock.monotonic,
        ):
            first = service.recover("owner", "attempt-1", True)
            self.assertEqual(first["status"], "unknown")
            self.assertEqual(
                first["recovery_reason"],
                TextRecoveryReason.REQUEST_CONVERSATION_SCAN_INCOMPLETE.value,
            )
            self.assertEqual(first.get("recovery_no_result_reads", 0), 0)
            self.assertNotIn(RECOVERY_CONVERSATION_SCAN_FIELD, first)
            self.assertEqual(first["recovery_next_at"], recovery_clock() + 30)

            with service._db() as db:
                stored = json.loads(db.execute(
                    "SELECT receipt FROM requests WHERE owner=? AND id=?",
                    ("owner", "attempt-1"),
                ).fetchone()[0])
            self.assertEqual(stored[RECOVERY_CONVERSATION_SCAN_FIELD]["next_index"], 2)

            restarted = TextTaskService(
                self.path,
                executor=QueuedExecutor(),
                clock=recovery_clock,
                recovery_reader=reader,
            )
            result = first
            for _ in range(19):
                recovery_clock.advance(TextTaskService.RECOVERY_BASE_BACKOFF_SECONDS + 1)
                result = restarted.recover("owner", "attempt-1", True)

        self.assertEqual(backend.list_calls, 2)
        self.assertEqual(
            [conversation_id for conversation_id, _timeout in backend.detail_calls],
            [f"conversation-{index}" for index in range(40)],
        )
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(
            result["recovery_reason"],
            TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value,
        )
        self.assertEqual(result["recovery_no_result_reads"], 1)
        self.assertNotIn(RECOVERY_CONVERSATION_SCAN_FIELD, result)
        with restarted._db() as db:
            stored = json.loads(db.execute(
                "SELECT receipt FROM requests WHERE owner=? AND id=?",
                ("owner", "attempt-1"),
            ).fetchone()[0])
        self.assertNotIn(RECOVERY_CONVERSATION_SCAN_FIELD, stored)

    def test_scan_progress_uses_short_delay_but_network_failure_keeps_attempt_backoff(self):
        clock = ManualClock()
        state = {
            "identity": {
                "provider_binding_id": "binding",
                "provider_account_identity": "paid-account",
                "client_conversation_id": "product-gallery",
                "request_message_id": "request-message",
            },
            "conversation_ids": ["conversation-one"],
            "next_offset": 1,
            "coverage_complete": True,
            "time_order_valid": False,
            "last_update_time": None,
            "next_index": 0,
            "matches": [],
        }
        failures = iter([
            ConversationBindingError(
                "budget exhausted",
                code="CONVERSATION_OUTCOME_UNKNOWN",
                recovery_reason=TextRecoveryReason.REQUEST_CONVERSATION_SCAN_INCOMPLETE.value,
                recovery_scan=state,
            ),
            ConversationBindingError(
                "upstream read failed",
                code="RECOVERY_READ_FAILED",
                recovery_scan=state,
            ),
        ])

        def reader(_receipt):
            raise next(failures)

        service = TextTaskService(
            self.path, executor=self.queue, clock=clock, recovery_reader=reader,
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="binding", provider_account_identity="paid-account",
            request_message_id="request-message", recovery_attempt=21,
        )
        clock.advance(TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS + 1)

        progressed = service.recover("owner", "attempt-1", True)
        self.assertEqual(progressed["recovery_next_at"], clock() + 30)
        clock.advance(31)
        failed = service.recover("owner", "attempt-1", True)

        self.assertEqual(
            failed["recovery_next_at"],
            clock() + TextTaskService.RECOVERY_MAX_BACKOFF_SECONDS,
        )
        self.assertEqual(failed["recovery_error_code"], "RECOVERY_READ_FAILED")
        self.assertEqual(failed.get("recovery_no_result_reads", 0), 0)

    def test_missing_text_conversation_requires_replacement_chat(self):
        clock = ManualClock()

        def reader(_receipt):
            raise ConversationBindingError(
                "chat missing", code="CONVERSATION_OUTCOME_UNKNOWN",
                recovery_reason="CONVERSATION_NOT_FOUND",
            )

        service = TextTaskService(
            self.path, executor=self.queue, clock=clock, recovery_reader=reader,
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="binding", provider_account_identity="account",
            conversation_id="missing-chat", parent_message_id="old-parent",
        )
        clock.advance(TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS + 1)
        for delay in (31, 61, 121):
            result = service.recover("owner", "attempt-1", True)
            clock.advance(delay)

        self.assertEqual(result["error_code"], "RESULT_UNRECOVERABLE")
        self.assertTrue(result["recovery_requires_new_conversation"])
        self.assertEqual(result["provider_binding_id"], "binding")
        self.assertEqual(result["conversation_id"], "missing-chat")
        self.assertEqual(result["parent_message_id"], "old-parent")

    def test_terminal_empty_text_result_is_retryable_but_running_is_not(self):
        clock = ManualClock()
        reasons = iter([
            "REQUEST_RESULT_INCOMPLETE",
            "REQUEST_RESULT_TERMINAL_EMPTY",
            "REQUEST_RESULT_TERMINAL_EMPTY",
            "REQUEST_RESULT_TERMINAL_EMPTY",
        ])

        def reader(_receipt):
            return {"status": "running", "recovery_reason": next(reasons)}

        service = TextTaskService(
            self.path, executor=self.queue, clock=clock, recovery_reader=reader,
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="binding", provider_account_identity="account",
            conversation_id="chat",
        )
        clock.advance(TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS + 1)
        for delay in (31, 61, 121, 241):
            result = service.recover("owner", "attempt-1", True)
            clock.advance(delay)

        self.assertEqual(result["error_code"], "RESULT_UNRECOVERABLE")
        self.assertEqual(result["recovery_no_result_reads"], 3)
        self.assertFalse(result["recovery_requires_new_conversation"])

    def test_missing_result_qualifies_but_invalid_mapping_contract_does_not(self):
        clock = ManualClock()
        service = TextTaskService(
            self.path,
            executor=self.queue,
            clock=clock,
            recovery_reader=lambda _receipt: {
                "status": "unknown",
                "recovery_reason": "REQUEST_RESULT_NOT_FOUND",
            },
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="binding", provider_account_identity="account",
            conversation_id="chat",
        )
        clock.advance(TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS + 1)
        for delay in (31, 61, 121):
            result = service.recover("owner", "attempt-1", True)
            clock.advance(delay)

        self.assertEqual(result["error_code"], "RESULT_UNRECOVERABLE")
        self.assertEqual(result["recovery_no_result_reads"], 3)

        invalid_path = self.path.with_name("invalid-mapping.sqlite3")

        def invalid_reader(_receipt):
            raise ConversationBindingError(
                "conversation mapping is missing or invalid",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
            )

        invalid_service = TextTaskService(
            invalid_path,
            executor=QueuedExecutor(),
            clock=clock,
            recovery_reader=invalid_reader,
        )
        invalid_body = {**self.body, "client_request_id": "invalid-mapping"}
        invalid_service.submit("owner", invalid_body)
        invalid_service._update(
            "owner", "invalid-mapping", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="binding", provider_account_identity="account",
            conversation_id="chat",
        )
        for delay in (31, 61, 121):
            invalid_result = invalid_service.recover(
                "owner", "invalid-mapping", True,
            )
            clock.advance(delay)

        self.assertEqual(invalid_result["status"], "unknown")
        self.assertEqual(
            invalid_result["recovery_error_code"],
            "CONVERSATION_BINDING_CONTRACT_INVALID",
        )
        self.assertEqual(invalid_result.get("recovery_no_result_reads", 0), 0)
        self.assertNotEqual(invalid_result.get("error_code"), "RESULT_UNRECOVERABLE")

    def test_missing_anchor_waits_while_current_node_is_active_then_recovers(self):
        clock = ManualClock()
        document = {
            "conversation_id": "chat",
            "current_node": "assistant-running",
            "mapping": {
                "assistant-running": {
                    "message": {
                        "id": "assistant-running",
                        "author": {"role": "assistant"},
                        "status": "in_progress",
                    },
                },
            },
        }
        class Backend:
            def _get_conversation(self, _conversation_id):
                return document

        backend = Backend()

        def actual_reader(receipt):
            return ConversationBindingService._read_text_request_result(backend, receipt)

        service = TextTaskService(
            self.path,
            executor=self.queue,
            clock=clock,
            recovery_reader=actual_reader,
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="binding", provider_account_identity="account",
            conversation_id="chat",
        )
        clock.advance(TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS + 1)

        for _ in range(3):
            active = service.recover("owner", "attempt-1", True)
            self.assertEqual(active["status"], "unknown")
            self.assertEqual(active["recovery_reason"], "REQUEST_RESULT_INCOMPLETE")
            self.assertEqual(active.get("recovery_no_result_reads", 0), 0)
            clock.advance(TextTaskService.RECOVERY_MAX_BACKOFF_SECONDS + 1)

        document["mapping"]["assistant-running"]["message"].update({
            "status": "finished_successfully",
            "end_turn": True,
        })
        for _ in range(3):
            recovered = service.recover("owner", "attempt-1", True)
            clock.advance(TextTaskService.RECOVERY_MAX_BACKOFF_SECONDS + 1)

        self.assertEqual(recovered["status"], "failed")
        self.assertEqual(recovered["error_code"], "RESULT_UNRECOVERABLE")
        self.assertEqual(recovered["recovery_no_result_reads"], 3)

    def test_latest_valid_chat_evidence_clears_an_older_404_marker(self):
        clock = ManualClock()
        reasons = iter([
            "CONVERSATION_NOT_FOUND",
            "REQUEST_MESSAGE_NOT_FOUND",
            "REQUEST_MESSAGE_NOT_FOUND",
        ])

        def reader(_receipt):
            reason = next(reasons)
            raise ConversationBindingError(
                "bounded read has no result", code="CONVERSATION_OUTCOME_UNKNOWN",
                recovery_reason=reason,
            )

        service = TextTaskService(
            self.path, executor=self.queue, clock=clock, recovery_reader=reader,
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="binding", provider_account_identity="account",
            conversation_id="chat",
        )
        clock.advance(TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS + 1)
        first = service.recover("owner", "attempt-1", True)
        self.assertTrue(first["recovery_requires_new_conversation"])
        clock.advance(31)
        service.recover("owner", "attempt-1", True)
        clock.advance(61)
        final = service.recover("owner", "attempt-1", True)

        self.assertEqual(final["error_code"], "RESULT_UNRECOVERABLE")
        self.assertFalse(final["recovery_requires_new_conversation"])

    def test_default_recovery_does_not_authorize_replacement(self):
        clock = ManualClock()
        service = TextTaskService(
            self.path, executor=self.queue, clock=clock,
            recovery_reader=lambda _receipt: {
                "status": "running", "recovery_reason": "REQUEST_BRANCH_SUPERSEDED",
            },
        )
        service.submit("owner", self.body)
        service._update(
            "owner", "attempt-1", status="unknown",
            error_code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="binding", provider_account_identity="account",
            conversation_id="chat",
        )
        clock.advance(TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS + 1)
        for delay in (31, 61, 121):
            result = service.recover("owner", "attempt-1")
            self.assertEqual(result["status"], "unknown")
            self.assertEqual(result.get("recovery_no_result_reads", 0), 0)
            clock.advance(delay)

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
                recovered = client.post(
                    "/api/conversation-bindings/text-requests/attempt-1/recover",
                    json={"allow_unrecoverable_retry": True},
                    headers={"Authorization": "owner"},
                )
                self.assertEqual(recovered.status_code, 200, recovered.text)
                self.assertEqual(recovered.json()["status"], "queued")
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
