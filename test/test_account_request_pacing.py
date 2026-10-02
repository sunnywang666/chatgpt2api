import threading
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from services.account_request_pacing import AccountRequestClock
from services.config import config
from services.request_context import executing


class Response:
    status_code = 200
    headers = {"x-request-id": "fixture-request"}

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def iter_lines(self):
        yield b'data: {"type":"message"}'


class Context:
    owner = "owner"
    request_id = "request"

    def __init__(self, name):
        self.name = name
        self.released = 0
        self.stages = []

    def before_send(self):
        self.stages.append("before_send")

    def release_turn(self):
        self.released += 1

    def record_stage(self, stage, **fields):
        self.stages.append(stage)

    def log_fields(self):
        return {"model": "gpt-5 fixture", "operation": "text", "source": "key:test"}


class AccountRequestPacingTests(unittest.TestCase):
    def test_http_attempt_evidence_counts_preflight_and_send_without_secrets(self):
        with patch("services.account_request_pacing.logger.info") as log, \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_message_interval_secs", property(lambda _: 0)):
            clock = AccountRequestClock("account-hash")
            response = Response()
            response.headers = {"Set-Cookie": "secret-cookie"}
            clock.request(lambda *a, **kw: response, "POST", "https://provider/conversation?secret-query",
                          json={"messages": ["secret-prompt"]}, headers={"Authorization": "secret-token"},
                          _account_request_preflight=lambda read: read("GET", "https://provider/conversation/secret-id"))
            attempts = [c.args[0] for c in log.call_args_list if c.args[0].get("event") == "account_http_attempt"]
            self.assertEqual([(e["method"], e["phase"], e["status_code"]) for e in attempts],
                             [("GET", "conversation_preflight", 200), ("POST", "conversation", 200)])
            self.assertNotIn("secret-", repr(attempts))
            self.assertTrue(all(e["started_at"] > 0 and e["headers_elapsed_secs"] >= 0 for e in attempts))

    def test_http_attempt_evidence_records_transport_failure_but_not_unsent_deadline(self):
        from services.account_request_pacing import AccountRequestDeadlineExceeded
        with patch("services.account_request_pacing.logger.info") as log:
            clock = AccountRequestClock("account-hash")
            def fail(*args, **kwargs):
                raise OSError("secret-transport-detail")
            with self.assertRaises(OSError):
                clock.request(fail, "GET", "https://provider/conversation/secret-id")
            with self.assertRaises(AccountRequestDeadlineExceeded):
                clock.request(fail, "GET", "https://provider/conversation/secret-id",
                              _account_request_deadline_monotonic=time.monotonic() - 1)
            attempts = [c.args[0] for c in log.call_args_list if c.args[0].get("event") == "account_http_attempt"]
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["outcome"], "transport_error")
            self.assertIsNone(attempts[0]["status_code"])
            self.assertNotIn("secret-", repr(attempts))

    def test_actual_send_gap_includes_slow_fence_and_durable_clock_write(self):
        for method, message_interval, fail_first in (("GET", 0, False), ("POST", 5, False), ("GET", 0, True), ("POST", 5, True)):
            with self.subTest(method=method, fail_first=fail_first), tempfile.TemporaryDirectory() as tmp:
                now = [10000.0]
                def advance(seconds):
                    now[0] += seconds

                with patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
                     patch("services.account_request_pacing.time.time", side_effect=lambda: 1700000000 + now[0]), \
                     patch("services.account_request_pacing.time.sleep", side_effect=advance), \
                     patch.object(type(config), "account_request_interval_secs", property(lambda _: 2)), \
                     patch.object(type(config), "account_message_interval_secs", property(lambda _: message_interval)):
                    clock = AccountRequestClock("account", Path(tmp) / "clock.json")
                    save = clock._save
                    saves = []
                    def slow_first_save():
                        if not saves:
                            advance(0.25)
                        saves.append(now[0])
                        save()
                    clock._save = slow_first_save
                    sent = []
                    def send(*args, **kwargs):
                        sent.append(now[0])
                        if fail_first and len(sent) == 1:
                            raise OSError("transport outcome unknown")
                        return Response()
                    class SlowFence(Context):
                        def before_send(self):
                            super().before_send()
                            if not sent:
                                advance(0.25)
                    context = SlowFence("slow-durable-fence")
                    kwargs = {"_account_request_before_send": lambda: advance(0.25)}
                    with executing(context):
                        if fail_first:
                            with self.assertRaises(OSError):
                                clock.request(send, method, "https://provider/backend-api/conversation", **kwargs)
                            clock = AccountRequestClock("account", Path(tmp) / "clock.json")
                        else:
                            clock.request(send, method, "https://provider/backend-api/conversation", **kwargs)
                        clock.request(send, method, "https://provider/backend-api/conversation")
                    self.assertGreaterEqual(sent[1] - sent[0], max(2, message_interval))
                    restarted = AccountRequestClock("account", Path(tmp) / "clock.json")
                    self.assertGreaterEqual(restarted.next_request, sent[-1] + 2 - 0.001)
                    if method == "POST":
                        self.assertGreaterEqual(restarted.next_turn, sent[-1] + message_interval - 0.001)

    def test_preflight_get_obeys_actual_send_gap_after_slow_clock_write(self):
        now = [10000.0]
        def advance(seconds):
            now[0] += seconds
        with patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
             patch("services.account_request_pacing.time.sleep", side_effect=advance), \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 2)), \
             patch.object(type(config), "account_message_interval_secs", property(lambda _: 0)):
            clock = AccountRequestClock("account")
            writes = []
            def slow_first_save():
                if not writes:
                    advance(0.25)
                writes.append(now[0])
            clock._save = slow_first_save
            sent = []
            def send(method, url, **kwargs):
                sent.append((method, now[0]))
                return Response()
            clock.request(send, "POST", "https://provider/backend-api/conversation",
                          _account_request_preflight=lambda read: read("GET", "https://provider/backend-api/conversation/original"))
            self.assertEqual([x[0] for x in sent], ["GET", "POST"])
            self.assertGreaterEqual(sent[1][1] - sent[0][1], 2)

    def test_metadata_429_keeps_account_cooldown_and_retry_after(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = AccountRequestClock("account", Path(tmp) / "clock.json")

            class RateLimitedResponse(Response):
                status_code = 429
                headers = {"Retry-After": "123"}

            response = clock.request(
                lambda *_args, **_kwargs: RateLimitedResponse(),
                "GET",
                "https://chatgpt.com/backend-api/models",
            )

            self.assertEqual(response.status_code, 429)
            self.assertGreaterEqual(clock.cooldown_until - time.monotonic(), 122)
            restarted = AccountRequestClock("account", Path(tmp) / "clock.json")
            self.assertEqual(restarted.last_rate_limit_evidence["phase"], "account_read")
            self.assertEqual(restarted.last_rate_limit_evidence["retry_after_seconds"], 123)

    def test_archive_429_retains_phase_without_generation_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = AccountRequestClock("account", Path(tmp) / "clock.json")
            response = Response();response.status_code = 429
            clock.request(lambda *a, **kw: response, "PATCH",
                          "https://chatgpt.com/backend-api/conversation/private-conversation-id",
                          json={"is_archived": True})
            saved = AccountRequestClock("account", Path(tmp) / "clock.json").last_rate_limit_evidence
            self.assertEqual(saved["phase"], "conversation_archive")
            self.assertEqual(saved["upstream_request_id"], "fixture-request")
            self.assertNotIn("private-conversation-id", str(saved))

    def test_stream_lock_is_released_at_headers_not_stream_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = AccountRequestClock("account", Path(tmp) / "clock.json")
            responses = []
            contexts = [Context("a"), Context("b")]

            def send(method, url, **kwargs):
                response = Response()
                responses.append(response)
                return response

            with patch.object(type(config), "account_request_interval_secs", new_callable=lambda: property(lambda _: 0)), \
                 patch.object(type(config), "account_message_interval_secs", new_callable=lambda: property(lambda _: 0)):
                with executing(contexts[0]):
                    first = clock.request(send, "POST", "https://provider/backend-api/conversation",
                                          json={"model": "gpt-5 fixture"}, stream=True)

                second_done = threading.Event()

                def second_request():
                    with executing(contexts[1]):
                        clock.request(send, "POST", "https://provider/backend-api/conversation",
                                      json={"model": "gpt-5 fixture"}, stream=True)
                    second_done.set()

                worker = threading.Thread(target=second_request)
                worker.start()
                worker.join(1)
                self.assertTrue(second_done.is_set(), "second send remained serialized behind first response stream")
                self.assertEqual(len(responses), 2)
                self.assertEqual(contexts[0].released, 0)
                list(first.iter_lines())
                self.assertEqual(contexts[0].released, 1)
                worker.join(1)


if __name__ == "__main__":
    unittest.main()
