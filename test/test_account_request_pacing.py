import threading
import json
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
    def test_zero_http_spacing_preserves_durable_message_and_429_waits(self):
        from services.config import ConfigStore
        import services.account_request_pacing as pacing
        for limited_path in ("/conversation/original", "/backend-api/me"):
            with self.subTest(limited_path=limited_path), tempfile.TemporaryDirectory() as tmp:
                settings_path = Path(tmp) / "config.json"
                settings_path.write_text(json.dumps({"auth-key": "test-only"}))
                settings = ConfigStore(settings_path)
                settings.update({"account_request_interval_secs": 0,
                                 "account_message_interval_secs": 5,
                                 "account_conversation_read_interval_secs": 0})
                path, now, sent = Path(tmp) / "clock.json", [10000.0], []
                def send(method, url, **kwargs):
                    sent.append(now[0])
                    response = Response()
                    if len(sent) == 3:
                        response.status_code = 429
                        response.headers = {"Retry-After": "120"}
                    return response
                def sleep(seconds):
                    now[0] += seconds
                with patch.object(pacing, "config", ConfigStore(settings_path)), \
                     patch.object(pacing.time, "monotonic", side_effect=lambda: now[0]), \
                     patch.object(pacing.time, "time", side_effect=lambda: 1700000000 + now[0]), \
                     patch.object(pacing.time, "sleep", side_effect=sleep):
                    for method, url in (("POST", "/conversation"), ("POST", "/conversation"),
                                        ("GET", limited_path), ("GET", limited_path)):
                        AccountRequestClock("account", path).request(send, method, "https://provider" + url)
                    self.assertEqual(sent, [10000, 10005, 10005, 10125])

    def test_persisted_zero_http_spacing_allows_reads_to_overlap(self):
        from services.config import ConfigStore
        import services.account_request_pacing as pacing
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "config.json"
            settings_path.write_text(json.dumps({"auth-key": "test-only"}))
            settings = ConfigStore(settings_path)
            settings.update({"account_request_interval_secs": 0,
                             "account_message_interval_secs": 5,
                             "account_conversation_read_interval_secs": 0})
            path = Path(tmp) / "clock.json"
            all_entered = threading.Barrier(4)
            errors, responses = [], []
            def send(*args, **kwargs):
                # All four transports must enter before any response returns.
                all_entered.wait(timeout=3)
                return Response()
            def read(index):
                try:
                    responses.append(AccountRequestClock("account", path).request(
                        send, "GET", f"https://provider/conversation/{index}"))
                except BaseException as exc:
                    errors.append(exc)
            with patch.object(pacing, "config", ConfigStore(settings_path)):
                workers = [threading.Thread(target=read, args=(i,)) for i in range(4)]
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(5)
                self.assertTrue(all(not worker.is_alive() for worker in workers))
                self.assertEqual(errors, [])
                self.assertEqual(len(responses), 4)
                self.assertEqual(AccountRequestClock("account", path).ordinary_read_queue, [])

    def test_account_metadata_does_not_block_original_result_delivery(self):
        init_body = {"gizmo_id": None, "requested_default_model": None,
                     "conversation_id": None, "timezone_offset_min": -480}
        for method, path, body in (
            ("GET", "/backend-api/me", None),
            ("POST", "/backend-api/conversation/init", init_body),
            ("GET", "/backend-api/accounts/check/v4-2023-04-27", None),
        ):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
                 patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 0)):
                state = Path(tmp) / "clock.json"
                entered, release, delivered = threading.Event(), threading.Event(), threading.Event()
                errors, sent = [], []
                def send(verb, url, **kwargs):
                    sent.append((verb, url))
                    if url.endswith(path):
                        entered.set()
                        if not release.wait(3):
                            raise TimeoutError("test metadata was not released")
                    return Response()
                def metadata():
                    try:
                        AccountRequestClock("account", state).request(send, method, "https://provider" + path, json=body)
                    except BaseException as exc:
                        errors.append(exc)
                def read():
                    try:
                        AccountRequestClock("account", state).request(send, "GET", "https://provider/conversation/original")
                        delivered.set()
                    except BaseException as exc:
                        errors.append(exc)
                worker, reader = threading.Thread(target=metadata), threading.Thread(target=read)
                worker.start()
                try:
                    self.assertTrue(entered.wait(1))
                    reader.start()
                    self.assertTrue(delivered.wait(1), "slow account metadata blocked original result delivery")
                    self.assertTrue(worker.is_alive())
                finally:
                    release.set()
                    worker.join(4)
                    if reader.ident is not None:
                        reader.join(4)
                self.assertFalse(worker.is_alive() or reader.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(len(sent), 2, "metadata or original result was replayed")

    def test_waiting_fifo_reader_rechecks_at_fractional_http_pace(self):
        for http_interval, expected_wait in ((.1, .1), (.5, .5), (5, 1)):
            with self.subTest(http_interval=http_interval):
                now = [10000.0]
                sent, sleeps = [], []
                clock = AccountRequestClock("account")
                clock.ordinary_read_queue = [{"owner": "earlier", "until": 10030}]
                def sleep(seconds):
                    sleeps.append(seconds)
                    self.assertEqual(sent, [], "later reader bypassed the queued owner")
                    self.assertEqual(clock.ordinary_read_queue[0]["owner"], "earlier")
                    # The earlier reader takes its turn while this caller waits.
                    with clock.lock:
                        clock._release_ordinary_read("earlier")
                    now[0] += seconds
                with patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
                     patch("services.account_request_pacing.time.sleep", side_effect=sleep), \
                     patch.object(type(config), "account_request_interval_secs", property(lambda _: http_interval)), \
                     patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 0)):
                    clock.request(lambda *a, **k: sent.append(now[0]) or Response(),
                                  "GET", "https://provider/conversation/original")
                self.assertEqual(sleeps, [expected_wait])
                self.assertEqual(sent, [10000 + expected_wait])
                self.assertEqual(clock.ordinary_read_queue, [])

    def test_metadata_preserves_read_queue_and_persisted_http_floor(self):
        now = [10000.0]
        sent = []
        with tempfile.TemporaryDirectory() as tmp, \
             patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
             patch("services.account_request_pacing.time.time", side_effect=lambda: 1700000000 + now[0]), \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 2)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 60)):
            path = Path(tmp) / "clock.json"
            clock = AccountRequestClock("account", path)
            clock.next_conversation_read = 10300
            clock.ordinary_read_queue = [{"owner": "waiting-original", "until": 10300}]
            clock.archive_read_owner, clock.archive_read_until = "archive", 10200
            clock._save()
            def sleep(seconds):
                # Another clock instance can inspect/admit work during pacing;
                # only the network-start reservation needs the process lock.
                other = AccountRequestClock("account", path)
                self.assertTrue(other.lock.acquire(blocking=False))
                other.lock.release()
                now[0] += seconds
            with patch("services.account_request_pacing.time.sleep", side_effect=sleep):
                for endpoint in ("me", "accounts/check/v4-2023-04-27"):
                    AccountRequestClock("account", path).request(
                        lambda *a, **kw: sent.append(now[0]) or Response(),
                        "GET", "https://provider/backend-api/" + endpoint)
            self.assertEqual(sent, [10000, 10002])
            restored = AccountRequestClock("account", path)
            self.assertAlmostEqual(restored.next_request, 10004, delta=1e-6)
            self.assertEqual(restored.next_conversation_read, 10300)
            self.assertEqual(restored.ordinary_read_queue, [{"owner": "waiting-original", "until": 10300}])
            self.assertEqual(restored.archive_read_owner, "archive")
            self.assertEqual(restored.archive_read_until, 10200)

    def test_late_metadata_or_result_does_not_erase_concurrent_429(self):
        for slow_path, limited_path, scope in (
            ("/backend-api/me", "/conversation/original", "conversation_read"),
            ("/conversation/original", "/backend-api/me", "account"),
            ("/backend-api/accounts/check/v4-2023-04-27", "/backend-api/me", "account"),
        ):
            with self.subTest(slow_path=slow_path, limited_path=limited_path), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
                 patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 0)):
                path = Path(tmp) / "clock.json"
                entered, release = threading.Event(), threading.Event()
                errors = []
                def send(method, url, **kwargs):
                    response = Response()
                    if url.endswith(slow_path):
                        entered.set()
                        if not release.wait(3):
                            raise TimeoutError("concurrent account request remained serialized")
                    else:
                        response.status_code = 429
                        response.headers = {}
                    return response
                def slow():
                    try:
                        AccountRequestClock("account", path).request(send, "GET", "https://provider" + slow_path)
                    except BaseException as exc:
                        errors.append(exc)
                worker = threading.Thread(target=slow)
                worker.start()
                try:
                    self.assertTrue(entered.wait(1))
                    AccountRequestClock("account", path).request(send, "GET", "https://provider" + limited_path)
                    before = json.loads(path.read_text())
                finally:
                    release.set()
                    worker.join(4)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                after = json.loads(path.read_text())
                self.assertEqual(after["last_rate_limit_evidence"]["scope"], scope)
                for field in ("rate_failures", "conversation_read_rate_failures", "last_rate_limit_evidence"):
                    self.assertEqual(after[field], before[field])
                for field in ("cooldown_until", "next_conversation_read"):
                    self.assertGreaterEqual(after[field], before[field] - .001)

    def test_non_metadata_init_or_write_retains_send_fence(self):
        from services.account_request_pacing import AccountRequestDeadlineExceeded
        for endpoint, body in (
            ("/backend-api/conversation/init", {"conversation_id": "existing"}),
            ("/backend-api/conversation/init", {"gizmo_id": None, "requested_default_model": None,
                                              "conversation_id": None, "timezone_offset_min": -480,
                                              "prompt": "an actual operation"}),
            ("/backend-api/files", {}),
        ):
            with self.subTest(endpoint=endpoint, body=body), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
                 patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 0)):
                path = Path(tmp) / "clock.json"
                entered, release = threading.Event(), threading.Event()
                errors, sent = [], []
                def send(method, url, **kwargs):
                    sent.append(method)
                    entered.set()
                    if not release.wait(3):
                        raise TimeoutError("test write was not released")
                    return Response()
                def write():
                    try:
                        AccountRequestClock("account", path).request(send, "POST", "https://provider" + endpoint, json=body)
                    except BaseException as exc:
                        errors.append(exc)
                read_errors, deadline_set = [], threading.Event()
                deadline = [0.0]
                def read():
                    try:
                        deadline[0] = time.monotonic() + .05
                        deadline_set.set()
                        AccountRequestClock("account", path).request(send, "GET", "https://provider/conversation/original",
                            _account_request_deadline_monotonic=deadline[0])
                    except BaseException as exc:
                        read_errors.append(exc)
                worker, reader = threading.Thread(target=write), threading.Thread(target=read)
                worker.start()
                try:
                    self.assertTrue(entered.wait(1))
                    reader.start()
                    self.assertTrue(deadline_set.wait(1))
                    # Deadline cleanup can wait for the write's lock. Keep the
                    # lock owner controlled here, rather than waiting for the
                    # reader to finish before releasing that same owner.
                    time.sleep(max(0, deadline[0] - time.monotonic()) + .05)
                    self.assertEqual(sent, ["POST"])
                finally:
                    release.set()
                    worker.join(4)
                    if reader.ident is not None:
                        reader.join(4)
                self.assertFalse(worker.is_alive() or reader.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(len(read_errors), 1)
                self.assertIsInstance(read_errors[0], AccountRequestDeadlineExceeded)
                self.assertEqual(sent, ["POST"])

    def test_fractional_http_floor_preserves_model_floor_and_retry_after(self):
        now = [10000.0]
        sent = []
        with tempfile.TemporaryDirectory() as tmp, \
             patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
             patch("services.account_request_pacing.time.time", side_effect=lambda: 1700000000 + now[0]), \
             patch("services.account_request_pacing.time.sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)), \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: .1)), \
             patch.object(type(config), "account_message_interval_secs", property(lambda _: 5)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 0)):
            path = Path(tmp) / "clock.json"

            def send(method, url, **kwargs):
                sent.append((method, now[0]))
                response = Response()
                if url.endswith("/limited"):
                    response.status_code = 429
                    response.headers = {"Retry-After": "30"}
                return response

            # A new clock instance must keep the fractional persisted floor.
            for name in ("first", "second"):
                AccountRequestClock("account", path).request(send, "GET", "https://provider/conversation/" + name)
            # Persisted wall-clock floats lose sub-microsecond precision.
            self.assertAlmostEqual(sent[1][1] - sent[0][1], .1, delta=1e-6)
            for _ in range(2):
                AccountRequestClock("account", path).request(send, "POST", "https://provider/conversation")
            self.assertGreaterEqual(sent[3][1] - sent[2][1], 5)
            AccountRequestClock("account", path).request(send, "GET", "https://provider/conversation/limited")
            AccountRequestClock("account", path).request(send, "GET", "https://provider/conversation/recovered")
            self.assertGreaterEqual(sent[-1][1] - sent[-2][1], 30)

    def test_attachment_lookup_keeps_http_pace_without_consuming_conversation_read_turn(self):
        now = [10000.0]
        sent = []
        with patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
             patch("services.account_request_pacing.time.sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)), \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 1)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 60)):
            clock = AccountRequestClock("account")
            clock.next_request = 10001
            clock.next_conversation_read = 10060
            def send(method, url, **kwargs):
                sent.append(now[0])
                return Response()
            url = "https://provider/backend-api/conversation/c/attachment/a/download"
            clock.request(send, "GET", url)
            self.assertEqual(sent, [10001])
            self.assertEqual(clock.next_conversation_read, 10060)
            clock.cooldown_until = 10010
            clock.request(send, "GET", url)
            self.assertEqual(sent, [10001, 10010])
            self.assertEqual(clock.next_conversation_read, 10060)

    def test_later_result_reader_cannot_overtake_waiting_reader(self):
        now = [10000.0]
        a_waiting, b_waiting, release_a, release_b = (threading.Event() for _ in range(4))
        b_decided, a_sent = threading.Event(), threading.Event()
        sent, errors = [], []
        sleeps = {"earlier": 0, "later": 0}
        with tempfile.TemporaryDirectory() as tmp, \
             patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
             patch("services.account_request_pacing.time.time", side_effect=lambda: 1700000000 + now[0]), \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 60)):
            path = Path(tmp) / "clock.json"
            earlier = AccountRequestClock("account", path)
            earlier.next_conversation_read = 10060
            earlier._save()
            later = AccountRequestClock("account", path)

            def send(method, url, **kwargs):
                name = url.rsplit("/", 1)[-1]
                sent.append((name, now[0]))
                (a_sent if name == "earlier" else b_decided).set()
                return Response()

            def sleep(seconds):
                name = threading.current_thread().name
                sleeps[name] += 1
                if sleeps[name] == 1:
                    (a_waiting if name == "earlier" else b_waiting).set()
                    if not (release_a if name == "earlier" else release_b).wait(2):
                        raise TimeoutError("reader was not released")
                elif name == "later":
                    b_decided.set()
                    if not a_sent.wait(2):
                        raise TimeoutError("earlier reader was starved")
                    now[0] = 10120
                else:
                    # The earlier caller may reacquire the file lock after its
                    # response while the later caller is persisting its turn.
                    # Yield for that OS-lock retry without moving the shared
                    # synthetic clock beyond the two explicitly released turns.
                    threading.Event().wait(0.001)

            def read(clock, name):
                try:
                    clock.request(send, "GET", "https://provider/conversation/" + name)
                except BaseException as exc:
                    errors.append(exc)

            with patch("services.account_request_pacing.time.sleep", side_effect=sleep):
                a = threading.Thread(target=read, args=(earlier, "earlier"), name="earlier")
                b = threading.Thread(target=read, args=(later, "later"), name="later")
                a.start()
                try:
                    self.assertTrue(a_waiting.wait(1))
                    b.start()
                    self.assertTrue(b_waiting.wait(1))
                    now[0] = 10060
                    release_b.set()  # The later caller wins the OS lock race.
                    self.assertTrue(b_decided.wait(1))
                    self.assertEqual(sent, [], "later reader stole the earlier reader's turn")
                finally:
                    release_a.set()
                    release_b.set()
                    a.join(3)
                    if b.ident is not None:
                        b.join(3)
            self.assertFalse(a.is_alive() or b.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(sent, [("earlier", 10060), ("later", 10120)])
            self.assertEqual(AccountRequestClock("account", path).ordinary_read_queue, [])

    def test_original_reads_overlap_without_holding_generation_send_lock(self):
        from services.request_context import current_request
        entered, release = threading.Event(), threading.Event()
        calls, errors = [], []
        context = Context("first-read")
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_message_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 0)):
            path = Path(tmp) / "clock.json"
            first = AccountRequestClock("account", path)
            second = AccountRequestClock("account", path)
            def send(method, url, **kwargs):
                calls.append((method, url.rsplit("/", 1)[-1], current_request.get()))
                if url.endswith("/first"):
                    entered.set()
                    if not release.wait(2):
                        raise TimeoutError("concurrent read/send remained blocked")
                return Response()
            def read():
                try:
                    with executing(context):
                        first.request(send, "GET", "https://provider/conversation/first")
                except BaseException as exc:
                    errors.append(exc)
            worker = threading.Thread(target=read)
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                second.request(send, "GET", "https://provider/conversation/second")
                second.request(send, "POST", "https://provider/conversation")
                self.assertTrue(worker.is_alive(), "first read must still be receiving")
            finally:
                release.set()
                worker.join(3)
            self.assertEqual(errors, [])
            self.assertEqual([(m, name) for m, name, _ in calls],
                             [("GET", "first"), ("GET", "second"), ("POST", "conversation")])
            self.assertIs(calls[0][2], context)
            self.assertEqual(AccountRequestClock("account", path).ordinary_read_queue, [])

    def test_late_original_read_success_preserves_other_read_cooldown(self):
        for retry_after in (None, "120"):
            with self.subTest(retry_after=retry_after), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
                 patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 0)):
                path = Path(tmp) / "clock.json"
                first, second = AccountRequestClock("account", path), AccountRequestClock("account", path)
                entered, release = threading.Event(), threading.Event()
                errors = []
                def send(method, url, **kwargs):
                    response = Response()
                    if url.endswith("/first"):
                        entered.set()
                        if not release.wait(2):
                            raise TimeoutError("second read remained serialized")
                    else:
                        response.status_code = 429
                        response.headers = {} if retry_after is None else {"Retry-After": retry_after}
                    return response
                def read():
                    try:
                        first.request(send, "GET", "https://provider/conversation/first")
                    except BaseException as exc:
                        errors.append(exc)
                worker = threading.Thread(target=read)
                worker.start()
                try:
                    self.assertTrue(entered.wait(1))
                    second.request(send, "GET", "https://provider/conversation/second")
                    before = json.loads(path.read_text())
                finally:
                    release.set()
                    worker.join(3)
                self.assertEqual(errors, [])
                after = json.loads(path.read_text())
                for field in ("rate_failures", "conversation_read_rate_failures", "last_rate_limit_evidence"):
                    self.assertEqual(after[field], before[field])
                self.assertGreaterEqual(after["cooldown_until"], before["cooldown_until"] - 0.001)
                self.assertGreaterEqual(after["next_conversation_read"], before["next_conversation_read"] - 0.001)
                self.assertEqual(after["last_rate_limit_evidence"]["scope"],
                                 "conversation_read" if retry_after is None else "account")

    def test_read_transport_error_is_joined_and_preserves_newer_reservation(self):
        entered, release = threading.Event(), threading.Event()
        errors = []
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 0)):
            path = Path(tmp) / "clock.json"
            first, second = AccountRequestClock("account", path), AccountRequestClock("account", path)
            def send(method, url, **kwargs):
                entered.set()
                if not release.wait(2):
                    raise AssertionError("other reader could not reserve")
                raise ConnectionError("controlled transport failure")
            def read():
                try:
                    first.request(send, "GET", "https://provider/conversation/first")
                except BaseException as exc:
                    errors.append(exc)
            worker = threading.Thread(target=read)
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                with second.lock:
                    second._ordinary_read_turn("later-reader", time.monotonic(), 60)
                    second.next_conversation_read = time.monotonic() + 60
                    second._save()
            finally:
                release.set()
                worker.join(3)
            self.assertEqual([type(exc) for exc in errors], [ConnectionError])
            reloaded = AccountRequestClock("account", path)
            self.assertGreater(reloaded.next_conversation_read, time.monotonic())
            self.assertEqual([entry["owner"] for entry in reloaded.ordinary_read_queue], ["later-reader"])

    def test_failed_read_floor_save_keeps_lock_and_reconciles_received_limit(self):
        for transport_error in (False, True):
            with self.subTest(transport_error=transport_error), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
                 patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 0)):
                path = Path(tmp) / "clock.json"
                clock, other = AccountRequestClock("account", path), AccountRequestClock("account", path)
                entered, release = threading.Event(), threading.Event()
                response = Response()
                response.status_code, response.headers = 429, {}
                saved, started = clock._save, threading.Thread.start
                failure_injected = False
                lock_was_held = []
                def send(*args, **kwargs):
                    entered.set()
                    if not release.wait(2):
                        raise TimeoutError("floor failure was not injected")
                    acquired = other.lock.acquire(blocking=False)
                    lock_was_held.append(not acquired)
                    if acquired:
                        other.lock.release()
                    if transport_error:
                        raise ConnectionError("controlled transport failure")
                    return response
                def start_and_wait(worker):
                    started(worker)
                    self.assertTrue(entered.wait(1))
                def save():
                    nonlocal failure_injected
                    if entered.is_set() and not failure_injected:
                        failure_injected = True
                        release.set()
                        raise KeyboardInterrupt("controlled caller interruption")
                    saved()
                with patch.object(threading.Thread, "start", start_and_wait), patch.object(clock, "_save", save):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        clock.request(send, "GET", "https://provider/conversation/original")
                self.assertEqual(lock_was_held, [True])
                self.assertFalse(clock.lock.locked())
                if transport_error:
                    self.assertIsInstance(caught.exception.__cause__, ConnectionError)
                else:
                    self.assertTrue(response.closed)
                    self.assertEqual(AccountRequestClock("account", path).conversation_read_rate_failures, 1)

    def test_read_reacquire_failure_closes_response_without_unlocked_save(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 0)):
            clock = AccountRequestClock("account", Path(tmp) / "clock.json")
            response = Response()
            original = clock.lock.acquire
            acquiring = 0
            def acquire(*args, **kwargs):
                nonlocal acquiring
                acquiring += 1
                if acquiring == 2:
                    raise OSError("controlled clock reload failure")
                return original(*args, **kwargs)
            with patch.object(clock.lock, "acquire", acquire):
                with self.assertRaisesRegex(OSError, "controlled clock reload failure"):
                    clock.request(lambda *a, **kw: response, "GET", "https://provider/conversation/original")
            self.assertTrue(response.closed)
            self.assertFalse(clock.lock.locked())

    def test_crashed_result_reader_expires_after_restart_without_blocking_post(self):
        now, sent = [10000.0], []
        with tempfile.TemporaryDirectory() as tmp, \
             patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
             patch("services.account_request_pacing.time.time", side_effect=lambda: 1700000000 + now[0]), \
             patch("services.account_request_pacing.time.sleep", side_effect=lambda delay: now.__setitem__(0, now[0] + delay)), \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_message_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 60)):
            path = Path(tmp) / "clock.json"
            original = AccountRequestClock("account", path)
            original.next_conversation_read = 10060
            with original.lock:
                self.assertTrue(original._ordinary_read_turn("crashed-worker", now[0], 60))
            restarted = AccountRequestClock("account", path)
            def send(method, *args, **kwargs):
                sent.append((method, now[0]))
                return Response()
            restarted.request(send, "POST", "https://provider/conversation")
            restarted.request(send, "GET", "https://provider/conversation/result")
            self.assertEqual([method for method, _ in sent], ["POST", "GET"])
            self.assertEqual(sent[0][1], 10000)
            self.assertAlmostEqual(sent[1][1], 10090, delta=1e-6)
            self.assertEqual(AccountRequestClock("account", path).ordinary_read_queue, [])

    def test_archive_read_booking_survives_restart_and_yields_to_waiting_result(self):
        from services.request_context import guarding_archive
        now = [10000.0]
        sent = []
        injected = [False]
        with tempfile.TemporaryDirectory() as tmp, \
             patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
             patch("services.account_request_pacing.time.time", side_effect=lambda: 1700000000 + now[0]), \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_message_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 60)):
            path = Path(tmp) / "clock.json"
            original = AccountRequestClock("account", path)
            original.next_conversation_read = now[0] + 60
            with original.lock:
                self.assertTrue(original._reserve_archive_read("old-work-v1", now[0]))
            clock = AccountRequestClock("account", path)
            def send(method, url, **kwargs):
                sent.append((method, url.rsplit("/", 1)[-1], now[0]))
                return Response()
            def advance(seconds):
                self.assertFalse(clock.lock.locked())
                if not injected[0]:
                    injected[0] = True
                    clock.request(send, "POST", "https://provider/conversation",
                                  _account_request_deadline_monotonic=now[0] + .5)
                    now[0] = 10060
                    archive = AccountRequestClock("account", path)
                    with guarding_archive(lambda: None, read_owner="old-work-v1"):
                        archive.request(send, "GET", "https://provider/conversation/archive-first")
                    with archive.lock:
                        self.assertFalse(archive._reserve_archive_read("old-work-v1", now[0]))
                    now[0] = 10065
                else:
                    now[0] += seconds
            with patch("services.account_request_pacing.time.sleep", side_effect=advance):
                clock.request(send, "GET", "https://provider/conversation/result")
                with guarding_archive(lambda: None, read_owner="old-work-v1"):
                    clock.request(send, "GET", "https://provider/conversation/archive-readback")
            self.assertEqual([(m, name) for m, name, _ in sent], [
                ("POST", "conversation"), ("GET", "archive-first"),
                ("GET", "result"), ("GET", "archive-readback")])
            self.assertEqual([at for _, _, at in sent], [10000, 10060, 10120, 10180])

    def test_abandoned_archive_read_booking_expires_without_blocking_post(self):
        now = [10000.0]
        sent = []
        with patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
             patch("services.account_request_pacing.time.sleep", side_effect=lambda delay: now.__setitem__(0, now[0] + delay)), \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_message_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 60)):
            clock = AccountRequestClock()
            clock.next_conversation_read = now[0] + 60
            with clock.lock:
                self.assertTrue(clock._reserve_archive_read("abandoned", now[0]))
                self.assertFalse(clock._reserve_archive_read("another-work", now[0]))
            def send(method, *args, **kwargs):
                sent.append((method, now[0]))
                return Response()
            clock.request(send, "POST", "https://provider/conversation")
            clock.request(send, "GET", "https://provider/conversation/result")
            self.assertEqual(sent, [("POST", 10000), ("GET", 10065)])
            clock.next_conversation_read = now[0] + 300
            with clock.lock:
                self.assertFalse(clock._reserve_archive_read("too-late", now[0]))

    def test_archive_attempt_consumes_booking_on_rate_limit_or_transport_error(self):
        from services.request_context import guarding_archive
        for outcome in ("rate_limit", "transport"):
            with self.subTest(outcome=outcome), \
                 patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
                 patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 60)):
                clock = AccountRequestClock()
                with clock.lock:
                    self.assertTrue(clock._reserve_archive_read("original", time.monotonic()))
                def send(*args, **kwargs):
                    if outcome == "transport":
                        raise OSError("transport interrupted")
                    response = Response()
                    response.status_code = 429
                    return response
                with guarding_archive(lambda: None, read_owner="original"):
                    if outcome == "transport":
                        with self.assertRaises(OSError):
                            clock.request(send, "GET", "https://provider/conversation/original")
                    else:
                        self.assertEqual(clock.request(send, "GET", "https://provider/conversation/original").status_code, 429)
                        self.assertGreater(clock.next_conversation_read, time.monotonic())
                        self.assertEqual(clock.rate_failures, 0)
                self.assertIsNone(clock.archive_read_owner)
                self.assertFalse(clock.lock.locked())

    def test_waiter_persistence_failure_releases_shared_clock(self):
        clock = AccountRequestClock()
        clock.next_conversation_read = time.monotonic() + 60
        with patch.object(clock, "_save", side_effect=OSError("fixture")), self.assertRaises(OSError):
            clock.request(lambda *a, **k: self.fail("unexpected send"), "GET", "https://provider/conversation/original")
        self.assertFalse(clock.lock.locked())

    def test_conversation_read_floor_survives_restart_and_transport_failure(self):
        now = [10000.0]
        sent = []
        def advance(seconds):
            now[0] += seconds
        def send(method, url, **kwargs):
            sent.append((method, now[0]))
            if len(sent) == 1:
                raise OSError("read outcome unknown")
            return Response()
        with tempfile.TemporaryDirectory() as tmp, \
             patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
             patch("services.account_request_pacing.time.time", side_effect=lambda: 1700000000 + now[0]), \
             patch("services.account_request_pacing.time.sleep", side_effect=advance), \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 1)), \
             patch.object(type(config), "account_message_interval_secs", property(lambda _: 5)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 15)):
            clock = AccountRequestClock("account", Path(tmp) / "clock.json")
            save = clock._save
            def slow_save():
                advance(.25)
                save()
            clock._save = slow_save
            with self.assertRaises(OSError):
                clock.request(send, "GET", "https://provider/conversation/original")
            clock = AccountRequestClock("account", Path(tmp) / "clock.json")
            self.assertEqual(clock.ordinary_read_queue, [])
            self.assertGreaterEqual(clock.next_conversation_read, sent[0][1] + 15)
            clock.request(send, "POST", "https://provider/conversation")
            self.assertLess(sent[1][1] - sent[0][1], 15)
            clock.request(send, "GET", "https://provider/conversation/other")
            self.assertGreaterEqual(sent[2][1] - sent[0][1], 15)
            self.assertEqual([s[0] for s in sent], ["GET", "POST", "GET"])

    def test_waiting_conversation_read_does_not_lock_generation_send(self):
        sleeping = threading.Event()
        finish_wait = threading.Event()
        errors = []
        sent = []
        with patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_message_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 15)):
            clock = AccountRequestClock("account")
            clock.next_conversation_read = time.monotonic() + 15
            def sleep_without_holding_lock(seconds):
                sleeping.set()
                if not finish_wait.wait(2):
                    raise TimeoutError("generation was blocked by waiting read")
            def send(method, *args, **kwargs):
                sent.append(method)
                return Response()
            def read():
                try:
                    clock.request(send, "GET", "https://provider/conversation/original")
                except BaseException as exc:
                    errors.append(exc)
            with patch("services.account_request_pacing.time.sleep", side_effect=sleep_without_holding_lock):
                worker = threading.Thread(target=read)
                worker.start()
                try:
                    self.assertTrue(sleeping.wait(1))
                    clock.request(send, "POST", "https://provider/conversation",
                                  _account_request_deadline_monotonic=time.monotonic() + .5)
                    self.assertEqual(sent, ["POST"])
                finally:
                    clock.next_conversation_read = 0
                    finish_wait.set()
                    worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(sent, ["POST", "GET"])

    def test_read_deadline_is_unsent_and_explicit_retry_after_still_cools_generation(self):
        from services.account_request_pacing import AccountRequestDeadlineExceeded
        now = [10000.0]
        sent = []
        def advance(seconds):
            now[0] += seconds
        def send(method, url, **kwargs):
            sent.append((method, now[0]))
            response = Response()
            if method == "GET":
                response.status_code = 429
                response.headers = {"Retry-After": "123"}
            return response
        with tempfile.TemporaryDirectory() as tmp, \
             patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
             patch("services.account_request_pacing.time.time", side_effect=lambda: 1700000000 + now[0]), \
             patch("services.account_request_pacing.time.sleep", side_effect=advance), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 15)):
            clock = AccountRequestClock("account", Path(tmp) / "clock.json")
            clock.next_conversation_read = now[0] + 15
            with self.assertRaises(AccountRequestDeadlineExceeded):
                clock.request(send, "GET", "https://provider/conversation/original",
                              _account_request_deadline_monotonic=now[0] + 5)
            self.assertEqual(sent, [])
            self.assertFalse(clock.lock.locked())
            self.assertEqual(clock.ordinary_read_queue, [])
            clock.request(send, "GET", "https://provider/conversation/original")
            clock = AccountRequestClock("account", Path(tmp) / "clock.json")
            self.assertEqual(clock.last_rate_limit_evidence["phase"], "conversation_read")
            clock.request(send, "POST", "https://provider/conversation")
            self.assertGreaterEqual(sent[1][1] - sent[0][1], 123)

    def test_legacy_clock_and_submission_preflight_keep_existing_semantics(self):
        import json
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_message_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 15)):
            path = Path(tmp) / "clock.json"
            clock = AccountRequestClock("account", path)
            clock._save()
            old = json.loads(path.read_text())
            del old["next_conversation_read"]
            path.write_text(json.dumps(old))
            clock = AccountRequestClock("account", path)
            self.assertLess(clock.next_conversation_read, time.monotonic())
            clock.next_conversation_read = time.monotonic() + 100
            clock._save()
            sent = []
            clock.request(lambda method, *a, **kw: sent.append(method) or Response(),
                          "POST", "https://provider/conversation",
                          _account_request_deadline_monotonic=time.monotonic() + 1,
                          _account_request_preflight=lambda read: read("GET", "https://provider/conversation/original"))
            self.assertEqual(sent, ["GET", "POST"])

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
            self.assertEqual(attempts[0]["transport_error_type"], "OSError")
            self.assertIsNone(attempts[0]["status_code"])
            self.assertNotIn("secret-", repr(attempts))

    def test_transport_diagnostics_keep_only_safe_category_code_and_send_timeout(self):
        from enum import IntEnum
        class Code(IntEnum):
            TIMEOUT = 28
        class PrivateError(TimeoutError):
            pass
        cases = [(PrivateError("secret-token-and-url"), Code.TIMEOUT, "TimeoutError", 28),
                 (RuntimeError("secret-token-and-url"), "secret-code", "other", None),
                 (RuntimeError("secret-token-and-url"), True, "other", None)]
        for error, code, category, expected in cases:
            error.code = code
            with self.subTest(category=category, code=expected), \
                 patch("services.account_request_pacing.logger.info") as log:
                def fail(*args, **kwargs):
                    raise error
                clock = AccountRequestClock("account-hash")
                with self.assertRaises(type(error)) as caught:
                    clock.request(fail, "GET", "https://provider/conversation/secret-id", timeout=12.5)
                self.assertIs(caught.exception, error)
                attempt = next(c.args[0] for c in log.call_args_list
                               if c.args[0].get("event") == "account_http_attempt")
                self.assertEqual(attempt["transport_error_type"], category)
                self.assertEqual(attempt["transport_error_code"], expected)
                self.assertEqual(attempt["request_timeout_secs"], 12.5)
                self.assertIsNone(attempt["status_code"])
                self.assertNotIn("secret-", repr(attempt))
                self.assertNotIn("PrivateError", repr(attempt))

    def test_diagnostic_extraction_failure_cannot_replace_transport_exception(self):
        class BadCode(OSError):
            @property
            def code(self):
                raise ValueError("secret-property")
            @property
            def response(self):
                raise ValueError("secret-response")
        error = BadCode("secret-original")
        with patch("services.account_request_pacing.logger.info") as log:
            def fail(*args, **kwargs):
                raise error
            with self.assertRaises(OSError) as caught:
                AccountRequestClock("account-hash").request(
                    fail, "GET", "https://provider/conversation/secret-id", timeout=10 ** 1000)
            self.assertIs(caught.exception, error)
            attempt = next(c.args[0] for c in log.call_args_list
                           if c.args[0].get("event") == "account_http_attempt")
            self.assertEqual(attempt["transport_error_type"], "OSError")
            self.assertIsNone(attempt["transport_error_code"])
            self.assertIsNone(attempt["request_timeout_secs"])
            self.assertNotIn("secret-", repr(attempt))

    def test_partial_transport_response_keeps_only_finite_allowlisted_metrics(self):
        from types import SimpleNamespace
        from curl_cffi import CurlInfo
        error = OSError("secret-original")
        error.response = SimpleNamespace(status_code=200, infos={
            CurlInfo.TOTAL_TIME: 12.5, CurlInfo.CONNECT_TIME: .2,
            CurlInfo.NAMELOOKUP_TIME: float("nan"), CurlInfo.APPCONNECT_TIME: True,
            CurlInfo.PRETRANSFER_TIME: -1, CurlInfo.STARTTRANSFER_TIME: "secret-time",
            CurlInfo.NUM_CONNECTS: 1, CurlInfo.OS_ERRNO: 0,
            CurlInfo.SIZE_DOWNLOAD_T: 5, CurlInfo.HTTP_CONNECTCODE: 200,
            CurlInfo.EFFECTIVE_URL: "secret-url", CurlInfo.PRIMARY_IP: "secret-address",
        })
        with patch("services.account_request_pacing.logger.info") as log:
            def fail(*args, **kwargs):
                raise error
            with self.assertRaises(OSError) as caught:
                AccountRequestClock("account-hash").request(fail, "GET", "https://provider/conversation/secret-id")
            self.assertIs(caught.exception, error)
            attempt = next(c.args[0] for c in log.call_args_list
                           if c.args[0].get("event") == "account_http_attempt")
            self.assertEqual(attempt["outcome"], "transport_error")
            self.assertIsNone(attempt["status_code"])
            self.assertEqual(attempt["transport_response_status"], 200)
            self.assertEqual(attempt["transport_details"], {
                "total_secs": 12.5, "connect_secs": .2, "num_connects": 1,
                "os_errno": 0, "downloaded_bytes": 5, "http_connect_code": 200})
            self.assertNotIn("secret-", repr(attempt))

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


class ReadRateLimitIsolationTests(unittest.TestCase):
    def setUp(self):
        self.now = 10000.0
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "clock.json"
        for target, kwargs in (
            ("services.account_request_pacing.time.monotonic", {"side_effect": lambda: self.now}),
            ("services.account_request_pacing.time.time", {"side_effect": lambda: 1700000000 + self.now}),
            ("services.account_request_pacing.time.sleep", {"side_effect": self.advance}),
        ):
            mock = patch(target, **kwargs)
            mock.start()
            self.addCleanup(mock.stop)
        for field, value in (("account_request_interval_secs", 1),
                             ("account_message_interval_secs", 5),
                             ("account_conversation_read_interval_secs", 30)):
            mock = patch.object(type(config), field, property(lambda _, value=value: value))
            mock.start()
            self.addCleanup(mock.stop)
        self.sent = []

    def advance(self, seconds):
        self.now += seconds

    def send(self, method, url, **kwargs):
        self.sent.append((method, self.now))
        response = Response()
        if method == "GET":
            response.status_code = 429  # No provider Retry-After.
        return response

    def test_parallel_read_limits_share_one_persisted_backoff_then_escalate_on_retry(self):
        all_sent = threading.Barrier(7)
        release = [threading.Event() for _ in range(6)]
        errors = []

        def send(method, url, **kwargs):
            index = int(url.rsplit("/", 1)[1])
            all_sent.wait(timeout=3)
            if not release[index].wait(5):
                raise TimeoutError("test response was not released")
            response = Response()
            response.status_code = 200 if index == 4 else 429
            response.headers = {}
            return response

        def read(index):
            try:
                AccountRequestClock("account", self.path).request(
                    send, "GET", f"https://provider/conversation/{index}")
            except BaseException as exc:
                errors.append(exc)

        def sleep(seconds):
            # Mutex/FIFO contention must yield to the real worker, not let six
            # threads race to advance the shared fake clock hundreds of seconds.
            if seconds <= 1:
                threading.Event().wait(0.001)
            else:
                self.advance(seconds)

        with patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 0)), \
             patch("services.account_request_pacing.time.sleep", side_effect=sleep):
            workers = [threading.Thread(target=read, args=(i,)) for i in range(6)]
            for worker in workers:
                worker.start()
            try:
                all_sent.wait(timeout=3)
                # A slow 200 and four more 429s cannot clear, multiply or extend
                # the first limit. Every reader uses a separately loaded clock.
                for index in (2, 4, 0, 5, 1, 3):
                    self.advance(1)
                    release[index].set()
                    workers[index].join(3)
                    self.assertFalse(workers[index].is_alive())
                    restored = AccountRequestClock("account", self.path)
                    self.assertEqual(restored.conversation_read_rate_failures, 1)
                    self.assertEqual(restored.last_conversation_read_rate_limit, 10001)
                    self.assertEqual(restored.next_conversation_read, 10061)
                    self.assertEqual(restored.rate_failures, 0)
            finally:
                for event in release:
                    event.set()
                for worker in workers:
                    worker.join(3)
            self.assertEqual(errors, [])
            self.assertTrue(restored.last_rate_limit_evidence["same_read_incident"])
            # A new original read after the existing cooldown is a new attempt.
            AccountRequestClock("account", self.path).request(
                self.send, "GET", "https://provider/conversation/original")
            self.assertEqual(self.sent, [("GET", 10061)])
            restored = AccountRequestClock("account", self.path)
            self.assertEqual(restored.conversation_read_rate_failures, 2)
            self.assertEqual(restored.next_conversation_read, 10181)
            self.assertFalse(restored.last_rate_limit_evidence["same_read_incident"])

    def test_explicit_retry_after_is_not_grouped_with_inflight_read_limit(self):
        clock = AccountRequestClock("account", self.path)
        with clock.lock:
            clock.limited(evidence={"phase": "conversation_read"}, read_sent_at=9999)
        restored = AccountRequestClock("account", self.path)
        with restored.lock:
            restored.limited(120, evidence={"phase": "conversation_read"},
                             retry_after_present=True, read_sent_at=9999)
        self.assertEqual(restored.conversation_read_rate_failures, 1)
        self.assertEqual(restored.rate_failures, 1)
        self.assertEqual(restored.cooldown_until, 10120)
        self.assertEqual(restored.last_rate_limit_evidence["scope"], "account")
        self.assertFalse(restored.last_rate_limit_evidence["same_read_incident"])

    def test_conversation_read_limits_do_not_delay_model_posts_after_restart(self):
        first = AccountRequestClock("account", self.path)
        other = AccountRequestClock("account", self.path)
        for clock in (first, other):
            clock.request(self.send, "GET", "https://provider/conversation/original")
            restarted = AccountRequestClock("account", self.path)
            restarted.request(self.send, "POST", "https://provider/conversation")
        self.assertEqual(self.sent, [("GET", 10000), ("POST", 10001),
                                     ("GET", 10060), ("POST", 10061)])
        self.assertEqual(restarted.rate_failures, 0)
        self.assertEqual(restarted.conversation_read_rate_failures, 2)
        self.assertEqual(restarted.next_turn, 10066)
        self.assertEqual(restarted.next_conversation_read, 10180)
        self.assertEqual(restarted.last_rate_limit_evidence["scope"], "conversation_read")
        from services.account_request_pacing import account_pacing_snapshot
        import hashlib
        identity = {"account_id": "fixture-account"}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "account_request_clocks" / (hashlib.sha256(b"fixture-account").hexdigest() + ".json")
            path.parent.mkdir()
            path.write_bytes(self.path.read_bytes())
            with patch("services.account_request_pacing.DATA_DIR", Path(root)):
                post = account_pacing_snapshot(identity)
                read = account_pacing_snapshot(identity, include_turn=False, include_conversation_read=True)
            self.assertEqual(post["next_at"], 1700010066)
            self.assertEqual(read["next_at"], 1700010180)
            self.assertEqual(read["cooldown_until"], 1700010180)

    def test_read_provider_backoff_is_not_credited_as_local_wait(self):
        from services.account_request_pacing import AccountRequestDeadlineExceeded
        clock = AccountRequestClock("account", self.path)
        clock.request(self.send, "GET", "https://provider/conversation/original")
        credited = []
        with self.assertRaises(AccountRequestDeadlineExceeded):
            clock.request(self.send, "GET", "https://provider/conversation/original",
                          _account_request_deadline_monotonic=self.now + 5,
                          _account_request_local_wait=credited.append)
        self.assertEqual(self.sent, [("GET", 10000)])
        self.assertEqual(credited, [])
        self.assertEqual(clock.ordinary_read_queue, [])

    def test_present_zero_or_invalid_retry_after_preserves_global_protection(self):
        for value in ("0", "invalid", ""):
            with self.subTest(retry_after=value):
                clock = AccountRequestClock()
                response = Response()
                response.status_code = 429
                response.headers = {"Retry-After": value}
                at = self.now
                clock.request(lambda *a, **k: response, "GET", "https://provider/conversation/original")
                clock.request(self.send, "POST", "https://provider/conversation")
                self.assertEqual(self.sent[-1], ("POST", at + 60))
                self.assertEqual(clock.rate_failures, 1)
                self.assertEqual(clock.conversation_read_rate_failures, 0)
                self.assertEqual(clock.last_rate_limit_evidence["scope"], "account")

    def test_read_and_global_limits_expire_independently(self):
        clock = AccountRequestClock("account", self.path)
        with clock.lock:
            clock.limited(evidence={"phase": "conversation_read"})
        self.now = 10200
        with clock.lock:
            clock.limited(evidence={"phase": "conversation"})
        self.now = 10901
        clock.request(lambda *a, **k: Response(), "GET", "https://provider/conversation/original")
        self.assertEqual(clock.conversation_read_rate_failures, 0)
        self.assertEqual(clock.rate_failures, 1)
        self.assertEqual(clock.next_conversation_read, 10961)  # Still globally backed off.
        with clock.lock:
            clock.limited(evidence={"phase": "conversation_read"})
        self.now = 11101
        clock.request(self.send, "POST", "https://provider/conversation")
        self.assertEqual(clock.rate_failures, 0)
        self.assertEqual(clock.conversation_read_rate_failures, 1)
        self.assertEqual(clock.next_turn, 11106)

    def test_old_mixed_history_remains_global_until_natural_expiry(self):
        clock = AccountRequestClock("account", self.path)
        with clock.lock:
            clock.limited(evidence={"phase": "conversation"})
            clock.limited(evidence={"phase": "conversation"})
        saved = json.loads(self.path.read_text())
        saved.pop("conversation_read_rate_failures", None)
        saved.pop("last_conversation_read_rate_limit", None)
        saved["last_rate_limit_evidence"]["phase"] = "conversation_read"
        self.path.write_text(json.dumps(saved))
        restored = AccountRequestClock("account", self.path)
        restored.request(self.send, "POST", "https://provider/conversation")
        self.assertEqual(self.sent, [("POST", 10120)])
        self.assertEqual(restored.rate_failures, 2)
        self.assertEqual(restored.conversation_read_rate_failures, 0)
        self.assertEqual(restored.next_turn, 10140)
        self.now = 10901
        restored.request(self.send, "POST", "https://provider/conversation")
        self.assertEqual(restored.rate_failures, 0)
        self.assertEqual(restored.next_turn, 10906)

    def test_unknown_metadata_prepare_post_and_archive_limits_remain_global(self):
        for method, url, body in (
            ("GET", "https://provider/models", None),
            ("POST", "https://provider/conversation/prepare", None),
            ("POST", "https://provider/conversation", None),
            ("PATCH", "https://provider/conversation/original", {"is_archived": True}),
        ):
            with self.subTest(method=method, url=url):
                clock = AccountRequestClock()
                response = Response()
                response.status_code = 429
                at = self.now
                clock.request(lambda *a, **k: response, method, url, json=body)
                self.assertEqual(clock.rate_failures, 1)
                self.assertEqual(clock.conversation_read_rate_failures, 0)
                self.assertEqual(clock.cooldown_until, at + 60)


def test_native_partial_response_timeout_keeps_stage_snapshot(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlsplit
    from curl_cffi import requests
    from curl_cffi.requests.exceptions import Timeout
    import services.account_request_pacing as pacing
    from services.config import ConfigStore

    release = threading.Event()
    received = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            received.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"hello")
            self.wfile.flush()
            release.wait(2)
            self.close_connection = True
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    session = requests.Session(trust_env=False)
    raw_send = session.request
    session.request = lambda method, url, **kw: raw_send(
        method, f"http://127.0.0.1:{server.server_port}" + urlsplit(url).path, **kw)
    (tmp_path / "config.json").write_text(json.dumps({"auth-key": "test-only"}))
    settings = ConfigStore(tmp_path / "config.json")
    settings.update({"account_request_interval_secs": 0, "account_conversation_read_interval_secs": 0})
    monkeypatch.setattr(pacing, "config", settings)
    monkeypatch.setattr(pacing, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pacing, "_clocks", {})
    pacing.pace_account_session(session, {"account_id": "fixture-only"}, "fixture-token")
    try:
        with patch("services.account_request_pacing.logger.info") as log:
            with __import__("pytest").raises(Timeout):
                session.get("https://chatgpt.com/backend-api/conversation/original", timeout=.2)
            attempt = next(c.args[0] for c in log.call_args_list
                           if c.args[0].get("event") == "account_http_attempt")
        assert attempt["outcome"] == "transport_error"
        assert attempt["transport_error_code"] == 28 and attempt["status_code"] is None
        assert attempt["transport_response_status"] == 200
        metrics = attempt["transport_details"]
        assert metrics["downloaded_bytes"] == 5
        assert metrics["num_connects"] == 1
        assert 0 <= metrics["connect_secs"] <= metrics["starttransfer_secs"] < metrics["total_secs"]
        assert .15 <= metrics["total_secs"] < 1
        assert len(received) == 1, "diagnostics must not retry"
    finally:
        release.set()
        session.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    unittest.main()
