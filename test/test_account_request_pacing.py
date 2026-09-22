import threading
import tempfile
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
