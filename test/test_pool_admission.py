"""Independent data and fake time; these are not production capacity samples."""
import json
import hashlib
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from datetime import datetime, timezone

from services.account_service import AccountService
from services.model_service import ModelRoute
from services.storage.json_storage import JSONStorageBackend
from services.task_store import SendHoldRejected, TaskStore
from services.pool_admission import PoolAdmission
from services.request_context import current_request, AdmissionLost
from services.text_task_service import TextTaskService
from services.conversation_binding_service import ConversationBindingService
from services.image_task_service import ImageTaskService
from services import product_work, workflow_scheduling


class Clock:
    def __init__(self):
        self.now = 1000.0
    def __call__(self):
        return self.now


def build(root, clock=None):
    root = Path(root)
    accounts = AccountService(JSONStorageBackend(root / "accounts.json"))
    accounts.refresh_image_capability = lambda account_ref: None  # metadata is fixture-controlled
    store = TaskStore(root / "text_tasks.sqlite3")
    admission = PoolAdmission(store, accounts, clock=clock or (lambda: 1000.0),
                              settings=lambda: {"image_account_concurrency": 4, "codex_max_concurrency": 4},
                              model_types=lambda _: {"Plus"}, pacing=lambda a, now: {"next_at": now})
    return accounts, store, admission


def claim_worker(root, ready, start, result):
    _, _, admission = build(root)
    ready.put(True)
    start.wait(5)
    context = admission.claim_next()
    result.put(None if context is None else (context.request_id, context.claim))


def scoped_claim_worker(root, ready, start, result):
    from services.config import ConfigStore
    root = Path(root)
    accounts, store, admission = build(root)
    config = ConfigStore(root / "config.json")
    admission.settings = config.resource_settings
    ready.put(True)
    start.wait(5)
    context = admission.claim_next()
    result.put(None if context is None else (context.request_id, context.claim))


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = []
        self.write_accounts(1)
        self.clock = Clock()
        self.accounts, self.store, self.admission = build(self.root, self.clock)
        self.calls = []
        def runner(body, on_cursor):
            context = current_request.get()
            context.before_send()
            self.calls.append((context.owner, context.request_id, body))
            return {"content": "done"}
        self.text = TextTaskService(self.store.path, runner=runner, admission=self.admission, clock=self.clock)
        self.admission.register("text", lambda ctx, body: self.text._run(ctx.owner, ctx.request_id, body))
        self.images = ImageTaskService(self.root / "image_tasks.json", admission=self.admission, store=self.store)

    def write_accounts(self, count):
        self.rows = [{"access_token": "fixture-token-" + str(i), "account_id": "upstream-" + str(i),
                      "provider_account_identity": "account-" + str(i), "type": "Plus", "status": "正常",
                      "quota": 999, "source_type": "web", "conversation_binding_ids": ["binding-" + str(i)],
                      "limits_progress": [{"feature_name": "image_gen", "remaining": 999}],
                      "capacity_observed_at": datetime.now(timezone.utc).isoformat()}
                     for i in range(count)]
        (self.root / "accounts.json").write_text(json.dumps(self.rows))

    def submit(self, name, owner="happy", source=None, **values):
        body = {"client_request_id": name, "client_conversation_id": "session-" + name,
                "model": "fixture-text", "messages": [{"role": "user", "content": "private original input"}], **values}
        return self.text.submit(owner, body, source=source)

    def image(self, name, owner="happy"):
        return self.images.submit_generation({"id": owner, "role": "user", "external_image_client": True},
                                             client_task_id=name, prompt="private image input", model="gpt-image-2", size=None)

    def test_inactive_text_work_does_not_refresh_catalog_and_resume_keeps_id(self):
        self.submit("paused-catalog")
        with self.store.transaction() as db:
            receipt = self.store.read_receipt(db, "text", "happy", "paused-catalog")
            key = "work:paused-catalog"
            receipt["_work_key"] = key
            self.store.write_receipt(db, "text", "happy", "paused-catalog", receipt)
            self.store.set_runtime(db, key, {"key": key, "state": "paused", "slot_held": False})
        with patch.object(self.admission, "_types", return_value={"Plus"}) as route:
            self.assertIsNone(self.admission.claim_next())
            route.assert_not_called()
            with self.store.transaction() as db:
                work = self.store.runtime(db, key)
                work["state"] = "active"
                self.store.set_runtime(db, key, work)
            context = self.admission.claim_next()
            self.assertEqual(context.request_id, "paused-catalog")
            route.assert_called_once_with("fixture-text")
        self.assertEqual(self.calls, [])

    def test_recovery_loop_rescans_due_original_while_claim_pass_blocks(self):
        """A later due original is not held behind a synchronous claim pass."""
        self.submit("later-original")
        self.text._update(
            "happy", "later-original", status="unknown",
            provider_account_identity="account-0", provider_binding_id="binding-0",
            conversation_id="later-original", request_message_id="message-later-original",
            _submission_started=True, _turn_reserved=False, _executing=False,
            upstream_outcome="unknown", recovery_next_at=self.clock() + 1,
        )
        claim_entered, release_claim, claim_finished = threading.Event(), threading.Event(), threading.Event()
        first_scan, dispatched = threading.Event(), threading.Event()
        scan_count = [0]
        original_recover_one = self.admission.recover_one

        def slow_claim():
            claim_entered.set()
            release_claim.wait(5)
            claim_finished.set()
            return None

        def recover(*, background=False):
            self.assertTrue(background)
            scan_count[0] += 1
            try:
                return original_recover_one(background=background)
            finally:
                if scan_count[0] == 1:
                    first_scan.set()

        self.admission.claim_next = slow_claim
        self.admission.recover_one = recover
        self.admission.recoveries["text"] = lambda _owner, request_id: (
            dispatched.set() if request_id == "later-original" else None)
        worker = threading.Thread(target=self.admission._loop, name="fixture-admission-loop")
        worker.start()
        try:
            self.assertTrue(claim_entered.wait(1))
            self.assertTrue(first_scan.wait(1))
            self.clock.now += 1
            self.assertTrue(dispatched.wait(2))
            self.assertFalse(claim_finished.is_set())
        finally:
            self.admission._stop.set()
            release_claim.set()
            self.admission.wake()
            worker.join(2)
        recovery = self.admission._recovery_thread
        if recovery is not None:
            recovery.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(recovery is not None and recovery.is_alive())
        self.assertGreaterEqual(scan_count[0], 2)

    def test_daemon_saves_result_and_replenishes_without_consumer_polling(self):
        """The consumer can disconnect after acceptance; Provider owns progress."""
        first_started, release_first = threading.Event(), threading.Event()
        finished = {name: threading.Event() for name in ("first", "next")}
        sends = []

        def runner(body, on_cursor):
            context = current_request.get()
            context.before_send()
            sends.append(context.request_id)
            if context.request_id == "first":
                first_started.set()
                self.assertTrue(release_first.wait(5))
            else:
                # The next send observes the first result already committed.
                with self.store.connect() as db:
                    saved = self.store.read_receipt(db, "text", "happy", "first")
                self.assertEqual(saved["status"], "succeeded")
                self.assertEqual(saved["content"], "saved-first")
            return {"content": "saved-" + context.request_id}

        self.text.runner = runner
        execute = self.admission.execute
        def observed_execute(context):
            try:
                execute(context)
            finally:
                finished[context.request_id].set()
        self.admission.execute = observed_execute
        self.submit("first")
        self.submit("next")
        with patch.object(self.text, "read", side_effect=AssertionError("consumer read is not required")), \
             patch.object(self.admission, "_start_image_capability_refresh", return_value=None):
            self.admission.start()
            try:
                self.assertTrue(first_started.wait(3))
                self.assertEqual(sends, ["first"])
                with self.store.connect() as db:
                    self.assertEqual(self.store.read_receipt(db, "text", "happy", "next")["status"], "queued")
                release_first.set()
                self.assertTrue(finished["next"].wait(3))
                self.assertTrue(finished["first"].wait(3))
            finally:
                release_first.set()
                self.admission.stop()
                for event in finished.values():
                    event.wait(3)
                if self.admission._recovery_thread is not None:
                    self.admission._recovery_thread.join(2)
        self.assertEqual(sends, ["first", "next"])
        restarted = TextTaskService(self.store.path, runner=lambda *_a, **_k: self.fail("must not resend"))
        for name in finished:
            saved = restarted.read_snapshot("happy", name)
            self.assertEqual(saved["status"], "succeeded")
            self.assertEqual(saved["content"], "saved-" + name)

    def test_image_daemon_saves_result_and_replenishes_without_consumer_polling(self):
        first_started, release_first = threading.Event(), threading.Event()
        finished = {name: threading.Event() for name in ("image-first", "image-next")}
        sends = []
        self.admission.settings = lambda: {"image_account_concurrency": 1, "codex_max_concurrency": 1}

        def generate(payload):
            context = current_request.get()
            context.before_send()
            sends.append(context.request_id)
            if context.request_id == "image-first":
                first_started.set()
                self.assertTrue(release_first.wait(5))
            else:
                with self.store.connect() as db:
                    saved = self.store.read_receipt(db, "image", "happy", "image-first")
                self.assertEqual(saved["status"], "success")
                self.assertEqual(saved["data"], [{"b64_json": "aGVsbG8="}])
            return {"data": [{"b64_json": "aGVsbG8="}],
                    "_provider_binding_id": payload["provider_binding_id"],
                    "_provider_account_identity": payload["provider_account_identity"],
                    "_conversation_id": "chat-" + context.request_id,
                    "_parent_message_id": "final-" + context.request_id}

        self.images = ImageTaskService(self.root / "image_tasks.json", admission=self.admission,
                                       store=self.store, generation_handler=generate)
        self.admission.register("image", lambda ctx, body: self.images._run_task(
            ctx.owner + ":" + ctx.request_id, body["mode"], body["payload"], body["identity"], "gpt-image-2"))
        execute = self.admission.execute
        def observed_execute(context):
            try:
                execute(context)
            finally:
                finished[context.request_id].set()
        self.admission.execute = observed_execute
        self.image("image-first")
        self.image("image-next")
        with patch.object(self.images, "list_tasks", side_effect=AssertionError("consumer read is not required")), \
             patch.object(self.admission, "_start_image_capability_refresh", return_value=None):
            self.admission.start()
            try:
                self.assertTrue(first_started.wait(3))
                self.assertEqual(sends, ["image-first"])
                release_first.set()
                self.assertTrue(finished["image-next"].wait(3))
                self.assertTrue(finished["image-first"].wait(3))
            finally:
                release_first.set()
                self.admission.stop()
                for event in finished.values():
                    event.wait(3)
                if self.admission._recovery_thread is not None:
                    self.admission._recovery_thread.join(2)
        self.assertEqual(sends, ["image-first", "image-next"])
        with self.store.connect() as db:
            for name in finished:
                saved = self.store.read_receipt(db, "image", "happy", name)
                self.assertEqual(saved["status"], "success")
                self.assertEqual(saved["data"], [{"b64_json": "aGVsbG8="}])

    def test_restarted_daemon_recovers_original_and_replenishes_without_consumer(self):
        """Persisted UNKNOWN needs only the Provider's original-read worker."""
        from unittest.mock import Mock
        from test.test_unknown_turn_recovery import document

        self.submit("original")
        self.text._update(
            "happy", "original", status="unknown", _submission_started=True,
            _turn_reserved=True, _executing=False, upstream_outcome="unknown",
            provider_binding_id="binding-0", provider_account_identity="account-0",
            conversation_id="original-conversation", request_message_id="original-user",
            request_parent_message_id="parent", recovery_next_at=0,
        )
        self.submit("next-after-recovery")
        self.assertIsNone(self.admission.claim_next())
        original_reads, sends = [], []
        finished = threading.Event()

        def reader(receipt):
            original_reads.append(receipt["request_id"])
            upstream = document(receipt)
            upstream["mapping"][upstream["current_node"]]["message"]["content"]["parts"] = ["recovered-original"]
            return ConversationBindingService._read_text_request_result(Mock(), receipt, document=upstream)

        def runner(body, on_cursor):
            context = current_request.get()
            context.before_send()
            sends.append(context.request_id)
            return {"content": "next-result"}

        # Rebuild the service from disk before starting the ordinary daemon.
        self.text = TextTaskService(self.store.path, runner=runner, recovery_reader=reader,
                                    admission=self.admission, clock=self.clock)
        self.admission.recoveries["text"] = self.text.read
        execute = self.admission.execute
        def observed_execute(context):
            try:
                execute(context)
            finally:
                finished.set()
        self.admission.execute = observed_execute
        with patch.object(self.admission, "_start_image_capability_refresh", return_value=None):
            self.admission.start()
            try:
                self.assertTrue(finished.wait(4))
            finally:
                self.admission.stop()
                if self.admission._recovery_thread is not None:
                    self.admission._recovery_thread.join(2)
        self.assertEqual(original_reads, ["original"])
        self.assertEqual(sends, ["next-after-recovery"])
        with self.store.connect() as db:
            original = self.store.read_receipt(db, "text", "happy", "original")
            following = self.store.read_receipt(db, "text", "happy", "next-after-recovery")
        self.assertEqual(original["status"], "succeeded")
        self.assertEqual(original["content"], "recovered-original")
        self.assertEqual(original["request_message_id"], "original-user")
        self.assertFalse(original["_turn_reserved"])
        self.assertEqual(following["status"], "succeeded")
        self.assertEqual(following["content"], "next-result")

    def test_recovery_candidate_reads_preserve_terminal_payloads_and_late_completion(self):
        large_saved = [{"b64_json": "completed-payload-must-not-be-decoded" * 10000}]
        rows = {
            "saved": {"status": "success", "data": large_saved},
            "ready": {"status": "success", "data": large_saved, "_completion": {"state": "result_ready"}},
            "closed": {"status": "success", "data": large_saved, "_completion": {"state": "completed"}},
            "late": {"status": "success", "data": [{"b64_json": "original"}],
                     "_completion": {"state": "needs_attention", "next_at": None}},
            "ended-assets": {"status": "error", "error_code": "RESULT_UNRECOVERABLE",
                             "_attempt_finished_at": 900, "_pending_image_result_ids": {"file_ids": ["original-file"]}},
            "historical-unknown": {"status": "unknown"},
            "paused": {"status": "unknown", "_recovery_paused": True},
        }
        with self.store.transaction() as db:
            for rid, row in rows.items():
                self.store.write_receipt(db, "image", "happy", rid, {"id": rid, "owner_id": "happy", **row})
        loads = json.loads
        def candidate_decode(raw):
            self.assertNotIn("completed-payload-must-not-be-decoded", raw)
            return loads(raw)
        with self.store.connect() as db, patch("services.task_store.json.loads", side_effect=candidate_decode):
            recovery = list(self.store.receipts(db, statuses=("unknown", "failed", "error")))
            completion = list(self.store.receipts(db, statuses=("unknown", "failed", "error"),
                                                  include_pending_completion=True))
            pending = list(self.store.receipts(db, statuses=(), include_pending_completion=True))
        self.assertEqual({r[2] for r in recovery}, {"ended-assets", "historical-unknown", "paused"})
        self.assertEqual({r[2] for r in completion}, {"late", "ended-assets", "historical-unknown", "paused"})
        self.assertEqual({r[2] for r in pending}, {"late"})
        # Candidate selection must not erase history or change ordinary reads.
        with self.store.connect() as db:
            all_rows = {rid: row for _, _, rid, row in self.store.receipts(db)}
            self.assertEqual(set(all_rows), set(rows))
            for rid, row in rows.items():
                self.assertEqual(all_rows[rid], {"id": rid, "owner_id": "happy", **row})
                self.assertEqual(self.store.read_receipt(db, "image", "happy", rid), all_rows[rid])

    def test_due_original_read_reserves_the_account_read_fifo_before_dispatch(self):
        """A due recovery must not repeatedly lose a one-credit read edge."""
        from services import account_request_pacing as pacing
        from services.account_request_pacing import AccountRequestClock
        from services.config import config

        now, sent, started = [1000.0], [], threading.Event()
        finished = threading.Event()
        ordinary_waiting, release_ordinary = threading.Event(), threading.Event()
        ordinary_errors = []
        account = self.rows[0]
        identity = account["account_id"]
        key = hashlib.sha256(identity.encode()).hexdigest()
        path = self.root / "account_request_clocks" / f"{key}.json"
        receipt = {
            "provider_account_identity": account["provider_account_identity"],
            "provider_binding_id": "binding-0",
            "client_conversation_id": "due-conversation",
            "conversation_id": "due-conversation",
        }

        def send(method, url, **kwargs):
            sent.append((url.rsplit("/", 1)[-1], now[0]))
            return type("Response", (), {"status_code": 200, "headers": {}})()

        def recover(owner, request_id):
            AccountRequestClock(key[:12], path).request(
                send, "GET", "https://provider/conversation/due-conversation")
            started.set()

        def ordinary_read():
            try:
                AccountRequestClock(key[:12], path).request(
                    send, "GET", "https://provider/conversation/ordinary-reader")
            except BaseException as exc:
                ordinary_errors.append(exc)

        def wait(seconds):
            if threading.current_thread().name == "ordinary-reader" and not ordinary_waiting.is_set():
                ordinary_waiting.set()
                if not release_ordinary.wait(2):
                    raise TimeoutError("ordinary reader was not released")
                # This is a scheduling handoff to let the due FIFO owner
                # consume its already-due credit.  It is not an account-pacing
                # wait, so it must not advance the shared virtual clock.
                return
            now[0] += seconds

        with patch.object(pacing, "DATA_DIR", self.root), \
             patch.object(pacing, "_clocks", {}), \
             patch.object(self.admission, "wake", side_effect=finished.set), \
             patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
             patch("services.account_request_pacing.time.time", side_effect=lambda: 1700000000 + now[0]), \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 90)):
            clock = AccountRequestClock(key[:12], path)
            clock.next_conversation_read = now[0] + 90
            clock._save()
            self.admission.pacing = None

            # Before the edge, recovery has no worker slot but must take a FIFO
            # position.  This assertion is the pre-fix red edge: old dispatch
            # returned here without a ticket at all.
            self.assertFalse(self.admission._dispatch_original_read("text", "happy", "due", recover, receipt))
            self.assertEqual(len(AccountRequestClock(key[:12], path).ordinary_read_queue), 1)
            now[0] += 90
            with patch("services.account_request_pacing.time.sleep", side_effect=wait):
                ordinary = threading.Thread(target=ordinary_read, name="ordinary-reader")
                ordinary.start()
                try:
                    self.assertTrue(ordinary_waiting.wait(2))
                    self.assertTrue(self.admission._dispatch_original_read("text", "happy", "due", recover, receipt))
                    self.assertTrue(started.wait(2))
                    self.assertEqual(sent, [("due-conversation", 1090.0)])
                    # ``started`` is set by recover() before the dispatcher's
                    # finally block drops the FIFO ticket.  Wait for that
                    # cleanup before letting the ordinary reader race again.
                    self.assertTrue(finished.wait(2))
                finally:
                    release_ordinary.set()
                    ordinary.join(3)
            self.assertFalse(ordinary.is_alive())
            self.assertEqual(ordinary_errors, [])
            self.assertEqual(sent, [("due-conversation", 1090.0), ("ordinary-reader", 1180.0)])

    def test_second_admission_does_not_release_another_dispatchers_original_ticket(self):
        from services import account_request_pacing as pacing
        from services.account_request_pacing import AccountRequestClock
        from services.config import config

        now, entered, allow_return, released = [1000.0], threading.Event(), threading.Event(), threading.Event()
        account = self.rows[0]
        key = hashlib.sha256(account["account_id"].encode()).hexdigest()
        path = self.root / "account_request_clocks" / f"{key}.json"
        receipt = {"provider_account_identity": account["provider_account_identity"],
                   "provider_binding_id": "binding-0", "client_conversation_id": "same-original",
                   "conversation_id": "same-original"}
        peer = PoolAdmission(self.store, self.accounts, clock=self.clock,
                             settings=self.admission.settings, model_types=self.admission.model_types)

        def recover(owner, request_id):
            entered.set()
            self.assertTrue(allow_return.wait(2))

        with patch.object(pacing, "DATA_DIR", self.root), \
             patch.object(pacing, "_clocks", {}), \
             patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
             patch("services.account_request_pacing.time.time", side_effect=lambda: 1700000000 + now[0]), \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 90)):
            clock = AccountRequestClock(key[:12], path)
            clock.next_conversation_read = now[0] + 90
            clock._save()
            self.admission.pacing = None
            self.assertFalse(self.admission._dispatch_original_read("text", "happy", "same", recover, receipt))
            now[0] += 90
            from services.account_request_pacing import release_account_original_read as real_release
            def release(account_row, owner):
                result = real_release(account_row, owner)
                released.set()
                return result
            with patch("services.account_request_pacing.release_account_original_read", side_effect=release):
                self.assertTrue(self.admission._dispatch_original_read("text", "happy", "same", recover, receipt))
                self.assertTrue(entered.wait(2))
                before = list(AccountRequestClock(key[:12], path).ordinary_read_queue)
                self.assertFalse(peer._dispatch_original_read("text", "happy", "same", recover, receipt))
                self.assertEqual(AccountRequestClock(key[:12], path).ordinary_read_queue, before)
                allow_return.set()
                self.assertTrue(released.wait(2))
            self.assertEqual(AccountRequestClock(key[:12], path).ordinary_read_queue, [])

    def test_active_physical_read_does_not_leave_another_receipts_ticket(self):
        from services import account_request_pacing as pacing
        from services.account_request_pacing import AccountRequestClock

        entered, allow_return, finished = threading.Event(), threading.Event(), threading.Event()
        account = self.rows[0]
        key = hashlib.sha256(account["account_id"].encode()).hexdigest()
        path = self.root / "account_request_clocks" / f"{key}.json"
        receipt = {"provider_account_identity": "account-0", "conversation_id": "same-conversation"}
        def recover(owner, request_id):
            AccountRequestClock(key[:12], path).request(
                lambda *_args, **_kwargs: type("Response", (), {"status_code": 200, "headers": {}})(),
                "GET", "https://provider/conversation/same-conversation")
            entered.set()
            allow_return.wait(2)
        with patch.object(pacing, "DATA_DIR", self.root), \
             patch.object(pacing, "_clocks", {}), \
             patch.object(self.admission, "wake", side_effect=finished.set):
            self.admission.pacing = None
            self.assertTrue(self.admission._dispatch_original_read("text", "happy", "first", recover, receipt))
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(AccountRequestClock(key[:12], path).ordinary_read_queue, [])
                self.assertFalse(self.admission._dispatch_original_read("text", "happy", "second", recover, receipt))
                self.assertEqual(AccountRequestClock(key[:12], path).ordinary_read_queue, [])
            finally:
                allow_return.set()
                self.assertTrue(finished.wait(2))

    def test_async_image_recovery_keeps_existing_pacing_without_text_ticket(self):
        from services import account_request_pacing as pacing
        from services.account_request_pacing import AccountRequestClock
        from services.request_context import current_original_read_owner
        from services.config import config

        now, owners, finished = [1000.0], [], threading.Event()
        key = hashlib.sha256(self.rows[0]["account_id"].encode()).hexdigest()
        path = self.root / "account_request_clocks" / f"{key}.json"
        receipt = {"provider_account_identity": "account-0", "conversation_id": "image"}
        with patch.object(pacing, "DATA_DIR", self.root), \
             patch.object(pacing, "_clocks", {}), \
             patch("services.account_request_pacing.time.monotonic", side_effect=lambda: now[0]), \
             patch("services.account_request_pacing.time.time", side_effect=lambda: now[0]), \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 90)), \
             patch.object(self.admission, "wake", side_effect=finished.set):
            clock = AccountRequestClock(key[:12], path)
            clock.next_conversation_read = now[0] + 90
            clock._save()
            self.admission.pacing = None
            recover = lambda *_args: owners.append(current_original_read_owner.get())
            self.assertFalse(self.admission._dispatch_original_read("image", "happy", "image", recover, receipt))
            self.assertEqual(AccountRequestClock(key[:12], path).ordinary_read_queue, [])
            now[0] += 90
            self.clock.now = now[0]
            self.assertTrue(self.admission._dispatch_original_read("image", "happy", "image", recover, receipt))
            self.assertTrue(finished.wait(2))
            self.assertEqual(owners, [None])
            self.assertEqual(AccountRequestClock(key[:12], path).ordinary_read_queue, [])

    def test_original_read_invalid_clock_does_not_block_other_accounts(self):
        from services import account_request_pacing as pacing
        from services.account_request_pacing import AccountRequestClock

        self.write_accounts(2)
        self.admission.pacing = None
        key = hashlib.sha256(self.rows[0]["account_id"].encode()).hexdigest()
        path = self.root / "account_request_clocks" / f"{key}.json"
        AccountRequestClock(key[:12], path)._save()
        saved = json.loads(path.read_text())
        invalid_clocks = [
            {key: value for key, value in saved.items() if key != "next_request"},
            {**saved, "next_request": True},
            {key: value for key, value in saved.items() if key != "rate_failures"},
            {**saved, "rate_failures": float("inf")},
        ]
        calls, finished = [], threading.Event()
        def recover(owner, request_id):
            calls.append(request_id)
        with patch.object(pacing, "DATA_DIR", self.root), \
             patch.object(pacing, "_clocks", {}), \
             patch.object(self.admission, "wake", side_effect=finished.set):
            for invalid in invalid_clocks:
                with self.subTest(invalid=invalid):
                    before = json.dumps(invalid)
                    path.write_text(before)
                    self.assertFalse(self.admission._dispatch_original_read(
                        "text", "happy", "invalid", recover,
                        {"provider_account_identity": "account-0", "conversation_id": "invalid"}))
                    self.assertEqual(path.read_text(), before)
            self.assertTrue(self.admission._dispatch_original_read(
                "text", "happy", "healthy", recover,
                {"provider_account_identity": "account-1", "conversation_id": "healthy"}))
            self.assertTrue(finished.wait(2))
            self.assertEqual(calls, ["healthy"])

    def test_send_guard_does_not_decode_unrelated_saved_images(self):
        self.image("new-independent")
        context = self.admission.claim_next()
        self.admission.update_claim(context, _image_thread={"protocol": "image-thread-v1", "id": "new-thread"})
        marker = "unrelated-saved-image-must-stay-on-disk"
        saved = {"id": "old-saved", "owner_id": "happy", "status": "success",
                 "data": [{"b64_json": marker * 10000}], "upstream_unfinished": False}
        with self.store.transaction() as db:
            self.store.write_receipt(db, "image", "happy", "old-saved", saved)
        loads = json.loads
        def guarded_decode(raw):
            self.assertNotIn(marker, raw)
            return loads(raw)
        with patch("services.task_store.json.loads", side_effect=guarded_decode):
            context.before_send()
        self.assertTrue(self.read("image", "happy", "new-independent")["_submission_started"])
        self.assertEqual(self.read("image", "happy", "old-saved"), saved)

    def test_chat_send_capacity_guard_does_not_decode_unrelated_saved_images(self):
        self.submit("new-independent")
        context = self.admission.claim_next()
        marker = "unrelated-saved-image-must-stay-on-disk"
        saved = {"id": "old-saved", "owner_id": "happy", "status": "success",
                 "data": [{"b64_json": marker * 10000}], "upstream_unfinished": False}
        with self.store.transaction() as db:
            self.store.write_receipt(db, "image", "happy", "old-saved", saved)
        loads = json.loads
        def guarded_decode(raw):
            self.assertNotIn(marker, raw)
            return loads(raw)
        with patch("services.task_store.json.loads", side_effect=guarded_decode):
            context.before_send()
        self.assertTrue(self.read("text", "happy", "new-independent")["_submission_started"])
        self.assertEqual(self.read("image", "happy", "old-saved"), saved)

    def test_send_guard_preserves_legacy_unfinished_image_capacity(self):
        self.image("new-image")
        context = self.admission.claim_next()
        current = self.read("image", "happy", "new-image")
        self.admission.settings = lambda: {"image_account_concurrency": 1}
        legacy = {"id": "legacy", "owner_id": "happy", "status": "success",
                  "upstream_unfinished": True, "_account_resource": current["_account_resource"]}
        with self.store.transaction() as db:
            self.store.write_receipt(db, "image", "happy", "legacy", legacy)
        with self.assertRaisesRegex(AdmissionLost, "capacity decreased"):
            context.before_send()
        self.assertFalse(self.read("image", "happy", "new-image")["_submission_started"])
        self.assertEqual(self.read("image", "happy", "legacy"), legacy)

    def test_send_guard_preserves_cross_protocol_legacy_conversation_occupancy(self):
        self.submit("text-next")
        context = self.admission.claim_next()
        self.admission.update_claim(context, conversation_id="original-conversation")
        legacy = {"id": "legacy-image", "owner_id": "other-owner", "status": "success",
                  "upstream_unfinished": True, "provider_account_identity": "account-0",
                  "conversation_id": "original-conversation", "_sequence": 0}
        with self.store.transaction() as db:
            self.store.write_receipt(db, "image", "other-owner", "legacy-image", legacy)
        with self.assertRaisesRegex(AdmissionLost, "physical conversation is occupied"):
            context.before_send()
        self.assertFalse(self.read("text", "happy", "text-next")["_submission_started"])
        self.assertEqual(self.read("image", "other-owner", "legacy-image"), legacy)

    def test_independent_image_preparation_uses_capacity_before_next_send_clock(self):
        self.admission.settings = lambda: {"chat_account_concurrency": 2, "image_account_concurrency": 2}
        self.admission.pacing = lambda account, now: {"next_at": now + 30, "cooldown_until": 0}
        for name in ("first", "second", "third"):
            self.image(name)
        first = self.admission.claim_next()
        self.assertIsNotNone(first)
        second = self.admission.claim_next()
        self.assertIsNotNone(second)
        self.assertEqual({first.request_id, second.request_id}, {"first", "second"})
        self.assertIsNone(self.admission.claim_next())
        self.assertEqual(self.calls, [])  # Claiming prepares work; it never sends.

    def test_preparation_still_waits_for_cooldown_or_unreadable_account_clock(self):
        self.image("original")
        self.admission.pacing = lambda account, now: {"next_at": now + 30, "cooldown_until": now + 20}
        self.assertIsNone(self.admission.claim_next())
        self.admission.pacing = lambda account, now: {"next_at": None, "cooldown_until": None}
        self.assertIsNone(self.admission.claim_next())
        self.admission.pacing = lambda account, now: {"next_at": now + 30, "cooldown_until": now - 1}
        self.assertEqual(self.admission.claim_next().request_id, "original")

    def test_legacy_explicitly_unsent_retry_retains_real_input_for_restart(self):
        body = {"client_request_id": "legacy", "client_conversation_id": "legacy-session",
                "model": "fixture-text", "_public_route": "chat", "messages": [{"role": "user", "content": "original"}]}
        self.text.submit("happy", body)
        with self.store.transaction() as db:
            old = self.store.read_receipt(db, "text", "happy", "legacy")
            old.update(status="failed", error_code="TEXT_TASK_CAPACITY_EXCEEDED", upstream_outcome="not_sent", _input_ref=None)
            self.store.write_receipt(db, "text", "happy", "legacy", old)
        self.text.submit("happy", body)
        _, _, restarted = build(self.root, self.clock)
        restarted.register("text", lambda ctx, value: self.text._run(ctx.owner, ctx.request_id, value))
        restarted.execute(restarted.claim_next())
        self.assertEqual(self.text.read("happy", "legacy")["status"], "succeeded")
        self.assertEqual(len(self.calls), 1)

    def test_terminal_text_success_can_complete_work_before_admission_finally(self):
        from services.pool_admission import ExecutionContext
        from services.work_lifecycle import WorkLifecycleError, WorkLifecycleService

        lifecycle = WorkLifecycleService(self.text, self.images, clock=self.clock)
        self.submit("terminal-work", _public_route="chat", _public_session_ref="terminal-work")

        def runner(body, on_cursor):
            context = current_request.get()
            context.before_send()
            # The upstream call is still active here, so the work must remain
            # protected even though the fixture will return a result next.
            with self.assertRaisesRegex(WorkLifecycleError, "WORK_TURN_UNFINISHED"):
                lifecycle.update("text", {"id": "happy"}, "terminal-work", "completed", True)
            return {"content": "done"}

        saved = {}
        original_record_stage = ExecutionContext.record_stage

        def record_stage(context, stage, **extra):
            if stage == "artifact_saved":
                saved["receipt"] = context.receipt()
                saved["work"] = lifecycle.update(
                    "text", {"id": "happy"}, "terminal-work", "completed", True,
                )
            return original_record_stage(context, stage, **extra)

        self.text.runner = runner
        with patch.object(ExecutionContext, "record_stage", new=record_stage):
            context = self.admission.claim_next()
            self.assertIsNotNone(context)
            self.admission.execute(context)

        self.assertEqual(saved["receipt"]["status"], "succeeded")
        self.assertFalse(saved["receipt"]["_executing"])
        self.assertEqual(saved["work"]["state"], "completed")
        self.assertFalse(saved["work"]["slot_held"])

    def test_terminal_image_success_can_complete_work_before_admission_finally(self):
        from services.pool_admission import ExecutionContext
        from services.work_lifecycle import WorkLifecycleError, WorkLifecycleService

        lifecycle = WorkLifecycleService(self.text, self.images, clock=self.clock)

        def handler(payload):
            context = current_request.get()
            context.before_send()
            # The handler has not returned a result yet, so it remains an
            # active upstream turn and must still reject work completion.
            with self.assertRaisesRegex(WorkLifecycleError, "WORK_TURN_UNFINISHED"):
                lifecycle.update("image", {"id": "happy"}, "terminal-image", "completed", True)
            return {
                "data": [{"url": "https://provider.test/finished.png"}],
                "_provider_binding_id": payload["provider_binding_id"],
                "_provider_account_identity": payload["provider_account_identity"],
                "_conversation_id": "terminal-image-conversation",
                "_parent_message_id": "terminal-image-parent",
            }

        self.images.generation_handler = handler
        self.images.submit_generation(
            {"id": "happy", "role": "user", "external_image_client": True},
            client_task_id="terminal-image", prompt="private image input", model="gpt-image-2", size=None,
            scheduling={"workflow_id": "batch", "workflow_concurrency": 1},
        )
        self.admission.register(
            "image",
            lambda context, body: self.images._run_task(
                context.owner + ":" + context.request_id, body["mode"], body["payload"], body["identity"],
                str(body["payload"].get("model") or "gpt-image-2"),
            ),
        )

        saved = {}
        original_record_stage = ExecutionContext.record_stage

        def record_stage(context, stage, **extra):
            if stage == "artifact_saved":
                saved["receipt"] = context.receipt()
                saved["work"] = lifecycle.update(
                    "image", {"id": "happy"}, "terminal-image", "completed", True,
                )
            return original_record_stage(context, stage, **extra)

        with patch.object(ExecutionContext, "record_stage", new=record_stage):
            context = self.admission.claim_next()
            self.assertIsNotNone(context)
            self.assertEqual(context.kind, "image")
            self.admission.execute(context)

        self.assertEqual(saved["receipt"]["status"], "success")
        self.assertFalse(saved["receipt"]["_executing"])
        self.assertFalse(saved["receipt"]["upstream_unfinished"])
        self.assertEqual(saved["work"]["state"], "completed")
        self.assertFalse(saved["work"]["slot_held"])

    def read(self, kind, owner, name):
        with self.store.connect() as db:
            return self.store.read_receipt(db, kind, owner, name)

    def observe_successor_scan_wait(self, request_id, predicate):
        """Expose the real loop's scan and subsequent Event.wait without waking it."""
        observed, waiting = threading.Event(), threading.Event()
        real_event, real_claim = self.admission._event, self.admission.claim_next

        class ObservedEvent:
            def set(_self):
                real_event.set()

            def clear(_self):
                real_event.clear()

            def wait(_self, timeout=None):
                if observed.is_set():
                    waiting.set()
                return real_event.wait(timeout)

        def claim_next():
            claim = real_claim()
            receipt = self.read("text", "happy", request_id)
            if receipt is not None and predicate[0](receipt):
                observed.set()
            return claim

        self.admission._event = ObservedEvent()
        self.admission.claim_next = claim_next

        def restore():
            self.admission._event = real_event
            self.admission.claim_next = real_claim

        return observed, waiting, restore

    def test_pending_public_successor_claims_immediately_and_rechecks_original_before_send(self):
        session = {"client_conversation_id": "public-chain", "_public_route": "chat", "_public_session_ref": "public-chain"}
        self.submit("first", **session)
        self.submit("next", **session, _previous_request_id="first")
        with self.store.transaction() as db:
            original = self.store.read_receipt(db, "text", "happy", "first")
            original.update(status="succeeded", upstream_outcome="completed", provider_binding_id="binding-0",
                            provider_account_identity="account-0", conversation_id="chat-0", parent_message_id="answer-0")
            self.store.write_receipt(db, "text", "happy", "first", original)
        claim = self.admission.claim_next()
        self.assertIsNotNone(claim)
        self.assertEqual(claim.request_id, "next")
        self.assertIsNone(self.admission.claim_next())
        for key, changed in (("status", "unknown"), ("upstream_outcome", "unknown"),
                             ("parent_message_id", "drift"), ("provider_account_identity", "other")):
            with self.subTest(key=key):
                with self.store.transaction() as db:
                    self.store.write_receipt(db, "text", "happy", "first", {**original, key: changed})
                with self.assertRaises(AdmissionLost):
                    claim.before_send()
                self.assertEqual(self.calls, [])
                with self.store.transaction() as db:
                    self.store.write_receipt(db, "text", "happy", "first", original)
                self.clock.now += 2
                claim = self.admission.claim_next()
                self.assertIsNotNone(claim)
        self.admission.execute(claim)
        self.assertEqual(len(self.calls), 1)
        sent = self.calls[0][2]
        self.assertEqual(sent["parent_message_id"], "answer-0")
        self.assertEqual(sent["conversation_id"], "chat-0")

    def test_loop_wakes_and_sends_due_public_successor_after_predecessor_finishes(self):
        """A completed sequential turn releases its already-queued successor."""
        first_started, release_first, second_sent = threading.Event(), threading.Event(), threading.Event()
        finished = {request_id: threading.Event() for request_id in ("first", "second")}
        execute = self.admission.execute

        def tracked_execute(context):
            try:
                return execute(context)
            finally:
                finished[context.request_id].set()

        self.admission.execute = tracked_execute

        def runner(body, on_cursor):
            context = current_request.get()
            context.before_send()
            self.calls.append((context.owner, context.request_id, body))
            if context.request_id == "first":
                first_started.set()
                self.assertTrue(release_first.wait(2))
                return {"content": "first result", "provider_binding_id": "binding-0",
                        "provider_account_identity": "account-0", "conversation_id": "chat-0",
                        "parent_message_id": "first-parent", "upstream_outcome": "completed"}
            second_sent.set()
            return {"content": "second result", "provider_binding_id": "binding-0",
                    "provider_account_identity": "account-0", "conversation_id": "chat-0",
                    "parent_message_id": "second-parent", "upstream_outcome": "completed"}

        self.text.runner = runner
        session = {"client_conversation_id": "wake-chain", "_public_route": "chat",
                   "_public_session_ref": "wake-chain"}
        self.submit("first", **session)
        predicate = [lambda receipt: receipt.get("_public_previous_waiting") is True]
        successor_scanned, successor_waiting, restore = self.observe_successor_scan_wait("second", predicate)
        self.admission.start()
        try:
            self.assertTrue(first_started.wait(1))
            accepted = self.submit("second", **session, _previous_request_id="first")
            self.assertEqual(accepted["status"], "queued")
            self.assertTrue(successor_scanned.wait(1))
            self.assertTrue(successor_waiting.wait(1), "B submit wake must be consumed before A finishes")
            started_at = time.monotonic()
            release_first.set()
            self.assertTrue(second_sent.wait(.75), "completion wake must not wait for the one-second loop timeout")
            self.assertLess(time.monotonic() - started_at, .75)
            self.assertEqual([request_id for _, request_id, _ in self.calls], ["first", "second"])
            deadline = time.monotonic() + 1
            second = self.read("text", "happy", "second")
            while second["status"] != "succeeded" and time.monotonic() < deadline:
                time.sleep(.01)
                second = self.read("text", "happy", "second")
            self.assertEqual(second["status"], "succeeded")
            self.assertEqual(second["provider_account_identity"], "account-0")
            self.assertEqual(second["conversation_id"], "chat-0")
            self.assertEqual(second["_submission_parent_message_id"], "first-parent")
        finally:
            release_first.set()
            self.admission.stop()
            # A succeeded receipt precedes artifact_saved and execute cleanup.
            # Keep the fixture DB alive until both daemon executions finish.
            if first_started.is_set():
                self.assertTrue(finished["first"].wait(2))
            if second_sent.is_set():
                self.assertTrue(finished["second"].wait(2))
            if self.admission._recovery_thread is not None:
                self.admission._recovery_thread.join(2)
            restore()

    def test_loop_keeps_public_successor_queued_after_unknown_predecessor(self):
        """Wakeups must not turn an UNKNOWN original into a successor send."""
        from services.conversation_binding_service import ConversationBindingError

        first_started, release_first, second_sent = threading.Event(), threading.Event(), threading.Event()

        def runner(body, on_cursor):
            context = current_request.get()
            context.before_send()
            if context.request_id == "first":
                first_started.set()
                self.assertTrue(release_first.wait(2))
                raise ConversationBindingError("fixture unknown", code="CONVERSATION_OUTCOME_UNKNOWN")
            second_sent.set()
            return {"content": "must not send"}

        self.text.runner = runner
        session = {"client_conversation_id": "unknown-chain", "_public_route": "chat",
                   "_public_session_ref": "unknown-chain"}
        self.submit("first", **session)
        predicate = [lambda receipt: receipt.get("_public_previous_waiting") is True]
        successor_scanned, successor_waiting, restore = self.observe_successor_scan_wait("second", predicate)
        self.admission.start()
        try:
            self.assertTrue(first_started.wait(1))
            self.submit("second", **session, _previous_request_id="first")
            self.assertTrue(successor_scanned.wait(1))
            self.assertTrue(successor_waiting.wait(1))
            successor_scanned.clear()
            predicate[0] = lambda receipt: (
                receipt.get("_public_previous_waiting_reason") == "CHAT_PREVIOUS_REQUEST_UNKNOWN")
            release_first.set()
            self.assertTrue(successor_scanned.wait(.75), "A completion wake must rescan B")
            self.assertEqual(self.read("text", "happy", "first")["status"], "unknown")
            self.assertFalse(second_sent.is_set())
            second = self.read("text", "happy", "second")
            self.assertEqual(second["status"], "queued")
            self.assertEqual(second["_public_previous_waiting_reason"], "CHAT_PREVIOUS_REQUEST_UNKNOWN")
        finally:
            release_first.set()
            self.admission.stop()
            if self.admission._recovery_thread is not None:
                self.admission._recovery_thread.join(2)
            restore()

    def test_loop_keeps_public_successor_queued_until_account_pacing_is_due(self):
        """Completion wake reevaluates B but never bypasses its account clock."""
        first_started, release_first, second_sent = threading.Event(), threading.Event(), threading.Event()

        def runner(body, on_cursor):
            context = current_request.get()
            context.before_send()
            if context.request_id == "first":
                first_started.set()
                self.assertTrue(release_first.wait(2))
                return {"content": "first result", "provider_binding_id": "binding-0",
                        "provider_account_identity": "account-0", "conversation_id": "chat-0",
                        "parent_message_id": "first-parent", "upstream_outcome": "completed"}
            second_sent.set()
            return {"content": "must not send"}

        self.text.runner = runner
        session = {"client_conversation_id": "paced-chain", "_public_route": "chat",
                   "_public_session_ref": "paced-chain"}
        self.submit("first", **session)
        predicate = [lambda receipt: receipt.get("_public_previous_waiting") is True]
        successor_scanned, successor_waiting, restore = self.observe_successor_scan_wait("second", predicate)
        self.admission.start()
        try:
            self.assertTrue(first_started.wait(1))
            self.submit("second", **session, _previous_request_id="first")
            self.assertTrue(successor_scanned.wait(1))
            self.assertTrue(successor_waiting.wait(1))
            successor_scanned.clear()
            predicate[0] = lambda receipt: (
                receipt.get("status") == "queued" and not receipt.get("_public_previous_waiting"))
            self.admission.pacing = lambda _account, now: {"next_at": now + 60}
            release_first.set()
            self.assertTrue(successor_scanned.wait(.75), "A completion wake must re-evaluate paced B")
            self.assertEqual(self.read("text", "happy", "first")["status"], "succeeded")
            self.assertFalse(second_sent.is_set())
            second = self.read("text", "happy", "second")
            self.assertEqual(second["status"], "queued")
            self.assertNotIn("_public_previous_waiting", second)
        finally:
            release_first.set()
            self.admission.stop()
            if self.admission._recovery_thread is not None:
                self.admission._recovery_thread.join(2)
            restore()

    def test_completed_image_with_iso_creation_time_cannot_stop_text_recovery(self):
        self.image("saved-image")
        self.submit("original", provider_binding_id="binding-0", provider_account_identity="account-0")
        with self.store.transaction() as db:
            image = self.store.read_receipt(db, "image", "happy", "saved-image")
            self.assertIsInstance(image["created_at"], str)
            image.update(status="success", next_poll_at=0)
            self.store.write_receipt(db, "image", "happy", "saved-image", image)
            original = self.store.read_receipt(db, "text", "happy", "original")
            original.update(status="unknown", request_message_id="original-user-message", recovery_next_at=900)
            self.store.write_receipt(db, "text", "happy", "original", original)
        recovered = []
        self.admission.recoveries["text"] = lambda owner, request_id: recovered.append((owner, request_id))
        self.admission.recover_one()
        self.assertEqual(recovered, [("happy", "original")])
        self.assertEqual(self.read("image", "happy", "saved-image"), image)
        self.assertEqual(self.read("text", "happy", "original"), original)
        self.assertEqual(self.calls, [])

    def test_recovery_order_uses_retry_clocks_across_text_and_iso_dated_images(self):
        self.image("unknown-image")
        self.submit("unknown-text", provider_binding_id="binding-0", provider_account_identity="account-0")
        with self.store.transaction() as db:
            image = self.store.read_receipt(db, "image", "happy", "unknown-image")
            image.update(status="error", error_code="CONVERSATION_OUTCOME_UNKNOWN", conversation_id="original-conversation",
                         request_message_id="image-message", next_poll_at=0)
            self.store.write_receipt(db, "image", "happy", "unknown-image", image)
            original = self.store.read_receipt(db, "text", "happy", "unknown-text")
            original.update(status="unknown", request_message_id="text-message", recovery_next_at=950)
            self.store.write_receipt(db, "text", "happy", "unknown-text", original)
        recovered = []
        def read_image(owner, request_id):
            recovered.append(("image", request_id))
            with self.store.transaction() as db:
                row = self.store.read_receipt(db, "image", owner, request_id)
                row["next_poll_at"] = self.clock() + 60
                self.store.write_receipt(db, "image", owner, request_id, row)
        self.admission.recoveries["image"] = read_image
        self.admission.recoveries["text"] = lambda owner, request_id: recovered.append(("text", request_id))
        self.admission.recover_one()
        self.admission.recover_one()
        self.assertEqual(recovered, [("image", "unknown-image"), ("text", "unknown-text")])
        self.assertEqual(self.calls, [])

    def test_explicit_recovery_suppression_preserves_receipt_and_stops_recovery(self):
        self.submit("suppressed-text", provider_binding_id="binding-0", provider_account_identity="account-0")
        self.image("suppressed-image")
        with self.store.transaction() as db:
            text = self.store.read_receipt(db, "text", "happy", "suppressed-text")
            text.update(status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                        request_message_id="text-message", recovery_next_at=0,
                        _recovery_suppressed=True, _recovery_suppressed_reason="manual_reset")
            self.store.write_receipt(db, "text", "happy", "suppressed-text", text)
            image = self.store.read_receipt(db, "image", "happy", "suppressed-image")
            image.update(status="error", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                         conversation_id="image-conversation", request_message_id="image-message",
                         upstream_unfinished=True, next_poll_at=0,
                         _recovery_suppressed=True, _recovery_suppressed_reason="manual_reset")
            self.store.write_receipt(db, "image", "happy", "suppressed-image", image)
        recovered = []
        self.admission.recoveries["text"] = lambda owner, request_id: recovered.append(("text", request_id))
        self.admission.recoveries["image"] = lambda owner, request_id: recovered.append(("image", request_id))
        self.admission.recover_one()
        self.assertEqual(recovered, [])
        self.assertEqual(self.admission.resource_snapshot()["chat_turn"]["inflight"], 0)
        saved_text = self.read("text", "happy", "suppressed-text")
        self.assertEqual(saved_text["status"], "unknown")
        self.assertTrue(saved_text["_recovery_suppressed"])
        self.assertEqual(saved_text["request_message_id"], "text-message")

    def test_full_waiting_request_uses_new_account_without_resubmit(self):
        self.submit("a")
        first = self.admission.claim_next()
        first.before_send()
        self.submit("waiting", owner="wb")
        self.assertIsNone(self.admission.claim_next())
        self.assertEqual(self.read("text", "wb", "waiting")["status"], "queued")
        self.write_accounts(2)
        next_context = self.admission.claim_next()
        self.assertEqual(next_context.request_id, "waiting")
        self.admission.execute(next_context)
        self.assertEqual(self.text.read("wb", "waiting")["status"], "succeeded")
        self.assertEqual(self.calls[0][2]["provider_account_identity"], "account-1")
        self.assertEqual(self.read("text", "happy", "a")["_claim_id"], first.claim)

    def test_same_account_chat_capacity_can_overlap_without_changing_image_capacity(self):
        self.admission.settings = lambda: {
            "image_account_concurrency": 4,
            "chat_account_concurrency": 2,
            "codex_max_concurrency": 4,
        }
        self.submit("first", owner="happy", client_conversation_id="session-a")
        self.submit("second", owner="wb", client_conversation_id="session-b")
        first = self.admission.claim_next()
        first.before_send()
        second = self.admission.claim_next()
        self.assertIsNotNone(second)
        self.assertEqual(first.receipt()["provider_account_identity"], second.receipt()["provider_account_identity"])
        self.assertEqual(self.admission.resource_snapshot()["accounts"][0]["chat_turn"]["capacity"], 2)
        timeline = self.read("text", "happy", "first")["_execution_timeline"]
        self.assertEqual([item["stage"] for item in timeline], ["accepted", "execution_claimed", "send_guard_passed"])

    def test_scoped_second_chat_slot_uses_only_one_bound_original_and_never_opens_other_accounts(self):
        self.write_accounts(4)
        settings = {"image_account_concurrency": 4, "chat_account_concurrency": 1,
                    "codex_max_concurrency": 4}
        self.admission.settings = lambda: settings
        for index in range(4):
            self.submit("active-" + str(index), owner="owner-" + str(index),
                        provider_account_identity="account-" + str(index),
                        provider_binding_id="binding-" + str(index))
            active = self.admission.claim_next()
            self.assertEqual(active.request_id, "active-" + str(index))
            active.before_send()
        self.submit("ordinary-wait", owner="other", provider_account_identity="account-1",
                    provider_binding_id="binding-1")
        self.submit("later-same-account", owner="happy", provider_account_identity="account-0",
                    provider_binding_id="binding-0")
        self.submit("chosen-original", owner="wb", provider_account_identity="account-0",
                    provider_binding_id="binding-0")
        self.assertIsNone(self.admission.claim_next())
        settings["temporary_chat_second_slot"] = {
            "account_identity": "account-0", "request_id": "chosen-original", "expires_at": 1010}
        self.assertEqual(self.admission.resource_snapshot()["chat_turn"]["slots_total"], 5)
        extra = self.admission.claim_next()
        self.assertEqual(extra.request_id, "chosen-original")
        self.assertEqual(extra.receipt()["provider_account_identity"], "account-0")
        self.assertIsNone(self.admission.claim_next())
        self.assertEqual(self.read("text", "happy", "later-same-account")["status"], "queued")
        self.assertEqual(self.read("text", "other", "ordinary-wait")["status"], "queued")
        self.assertEqual(self.admission.resource_snapshot()["chat_turn"]["slots_total"], 4)
        self.assertEqual(self.admission.resource_snapshot()["chat_turn"]["inflight"], 5)
        extra.before_send()
        self.assertEqual(self.read("text", "wb", "chosen-original")["status"], "running")

    def test_scoped_second_slot_expires_without_restarting_or_resending(self):
        settings = {"image_account_concurrency": 4, "chat_account_concurrency": 1,
                    "codex_max_concurrency": 4,
                    "temporary_chat_second_slot": {"account_identity": "account-0",
                                                   "request_id": "chosen", "expires_at": 1001}}
        self.admission.settings = lambda: settings
        self.submit("active", provider_account_identity="account-0", provider_binding_id="binding-0")
        active = self.admission.claim_next()
        active.before_send()
        self.submit("chosen", owner="wb", provider_account_identity="account-0", provider_binding_id="binding-0")
        self.assertEqual(self.admission.resource_snapshot()["chat_turn"]["slots_total"], 2)
        self.clock.now = 1002
        self.assertEqual(self.admission.resource_snapshot()["chat_turn"]["slots_total"], 1)
        self.assertIsNone(self.admission.claim_next())
        self.assertEqual(self.read("text", "wb", "chosen")["status"], "queued")

    def test_scoped_second_slot_does_not_advance_same_conversation_or_unknown_turn(self):
        settings = {"image_account_concurrency": 4, "chat_account_concurrency": 1,
                    "codex_max_concurrency": 4,
                    "temporary_chat_second_slot": {"account_identity": "account-0",
                                                   "request_id": "chosen", "expires_at": 1010}}
        self.admission.settings = lambda: settings
        self.submit("active", client_conversation_id="same",
                    provider_account_identity="account-0", provider_binding_id="binding-0")
        active = self.admission.claim_next()
        active.before_send()
        self.submit("chosen", client_conversation_id="same",
                    provider_account_identity="account-0", provider_binding_id="binding-0")
        self.assertIsNone(self.admission.claim_next())
        with self.store.transaction() as db:
            receipt = self.store.read_receipt(db, "text", "happy", "active")
            receipt.update(status="unknown", _executing=False)
            self.store.write_receipt(db, "text", "happy", "active", receipt)
        self.assertEqual(self.admission.resource_snapshot()["chat_turn"]["slots_total"], 1)
        self.assertIsNone(self.admission.claim_next())

    def test_disable_restore_and_quota_restore_wake_original_image(self):
        self.rows[0]["managed_disabled"] = True
        (self.root / "accounts.json").write_text(json.dumps(self.rows))
        self.image("image")
        self.assertIsNone(self.admission.claim_next())
        self.rows[0].update(managed_disabled=False, quota=0, limits_progress=[{"feature_name": "image_gen", "remaining": 0}])
        (self.root / "accounts.json").write_text(json.dumps(self.rows))
        self.assertIsNone(self.admission.claim_next())
        self.rows[0]["quota"] = 999
        self.rows[0]["limits_progress"] = [{"feature_name": "image_gen", "remaining": 999}]
        (self.root / "accounts.json").write_text(json.dumps(self.rows))
        self.assertEqual(self.admission.claim_next().request_id, "image")

    def test_bound_account_never_moves_to_new_account(self):
        self.submit("bound", provider_binding_id="binding-0", provider_account_identity="account-0")
        self.write_accounts(2)
        self.rows[0]["managed_disabled"] = True
        (self.root / "accounts.json").write_text(json.dumps(self.rows))
        self.assertIsNone(self.admission.claim_next())
        self.assertEqual(self.read("text", "happy", "bound")["provider_account_identity"], "account-0")

    def test_four_to_five_accounts_adds_only_real_image_capacity(self):
        # Four is a fixture parameter, never a Production default change.
        self.write_accounts(4)
        for index in range(16):
            self.image("pending-" + str(index))
            ctx = self.admission.claim_next()
            self.assertIsNotNone(ctx)
            ctx.before_send()
            ctx.release_turn()  # Model stream ended, async image still pending.
        self.image("original-waiting")
        self.assertIsNone(self.admission.claim_next())
        self.assertEqual(self.admission.resource_snapshot()["image"]["slots_total"], 16)
        self.write_accounts(5)
        self.assertEqual(self.admission.resource_snapshot()["image"]["slots_total"], 20)
        next_context = self.admission.claim_next()
        self.assertEqual(next_context.request_id, "original-waiting")
        self.assertEqual(next_context.receipt()["provider_account_identity"], "account-4")

    def test_changed_quota_between_claim_and_send_is_fenced(self):
        self.image("reserved")
        ctx = self.admission.claim_next()
        self.rows[0]["quota"] = 0
        self.rows[0]["limits_progress"] = [{"feature_name": "image_gen", "remaining": 0}]
        (self.root / "accounts.json").write_text(json.dumps(self.rows))
        with self.assertRaises(AdmissionLost):
            ctx.before_send()
        self.assertFalse(ctx.receipt()["_submission_started"])

    def test_read_only_capacity_never_reserves_a_turn(self):
        self.submit("queued")
        before = self.store.path.read_bytes()
        for _ in range(3):
            snapshot = self.admission.resource_snapshot()
            self.assertEqual(snapshot["queue"]["queued"], 1)
            self.assertEqual(snapshot["accounts"][0]["chat_turn"]["occupied"], 0)
        self.assertEqual(before, self.store.path.read_bytes())
        self.assertEqual(self.admission.claim_next().request_id, "queued")

    def assert_sent_text_keeps_capacity_until_result(self, transport_error=False):
        from services.account_request_pacing import AccountRequestClock
        from services.config import config
        from services.request_context import executing

        self.write_accounts(2)  # Spare global workers must not hide an account overflow.
        self.admission.settings = lambda: {"image_account_concurrency": 4,
                                          "chat_account_concurrency": 1,
                                          "codex_max_concurrency": 4}
        for name in ("original", "next"):
            self.submit(name, provider_account_identity="account-0", provider_binding_id="binding-0")
        original = self.admission.claim_next()
        sends = []

        class Response:
            status_code, headers = 200, {}
            def close(self):
                pass
            def iter_lines(self):
                yield b'data: {"type":"message"}'

        def send(*args, **kwargs):
            sends.append(original.request_id)
            if transport_error:
                raise TimeoutError("fixture transport stopped after submission")
            return Response()

        clock = AccountRequestClock("fixture", self.root / "clock.json")
        with patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
             patch.object(type(config), "account_message_interval_secs", property(lambda _: 0)), \
             executing(original):
            if transport_error:
                with self.assertRaises(TimeoutError):
                    clock.request(send, "POST", "https://provider/backend-api/conversation", stream=True)
            else:
                response = clock.request(send, "POST", "https://provider/backend-api/conversation", stream=True)
                response.close()  # Actual pacing callback; result classification has not run yet.

        self.assertEqual(original.receipt()["status"], "running")
        self.assertFalse(original.receipt()["_turn_reserved"])
        self.assertEqual(self.admission.resource_snapshot()["accounts"][0]["chat_turn"]["occupied"], 1)
        self.assertIsNone(self.admission.claim_next())
        self.text._update("happy", "original", status="unknown", upstream_outcome="unknown", _executing=False)
        self.assertIsNone(self.admission.claim_next())
        self.assertEqual(self.admission.resource_snapshot()["accounts"][0]["chat_turn"]["occupied"], 1)
        self.text._update("happy", "original", status="succeeded", content="original saved answer")
        following = self.admission.claim_next()
        self.assertEqual(following.request_id, "next")
        following.before_send()
        self.assertEqual(self.admission.resource_snapshot()["accounts"][0]["chat_turn"]["occupied"], 1)
        self.assertEqual(sends, ["original"])

    def test_sent_text_stream_close_cannot_admit_an_extra_turn_before_result(self):
        self.assert_sent_text_keeps_capacity_until_result()

    def test_sent_text_transport_failure_cannot_admit_an_extra_turn_before_result(self):
        self.assert_sent_text_keeps_capacity_until_result(transport_error=True)

    def assert_capacity_read_does_not_deadlock_claim(self, read):
        from services.config import ConfigStore
        path = self.root / "config.json"
        path.write_text(json.dumps({"auth-key": "fixture-only"}))
        config = ConfigStore(path)
        config.update_resource_settings(0, 4, 4, 2)
        # Bound the same reentrant-lock wait so a regression fails instead of
        # hanging the test process. The account and SQLite locks remain real.
        settings_lock = threading.RLock()
        class BoundedSettingsLock:
            def __enter__(self):
                if not settings_lock.acquire(timeout=2):
                    raise TimeoutError("settings/account lock inversion")
            def __exit__(self, *args):
                settings_lock.release()
        config._update_lock = BoundedSettingsLock()
        self.admission.settings = config.resource_settings
        self.submit("original")
        before = self.read("text", "happy", "original")
        settings_held, reader_at_lock = threading.Event(), threading.Event()
        settings_guard = self.admission._settings_guard
        account_guard = self.admission._account_guard
        errors, claims, snapshots = [], [], []

        @contextmanager
        def observed_settings_guard():
            if threading.current_thread().name == "capacity-reader":
                reader_at_lock.set()
            with settings_guard():
                yield

        @contextmanager
        def observed_account_guard():
            with account_guard():
                if threading.current_thread().name == "capacity-reader":
                    reader_at_lock.set()
                yield

        self.admission._settings_guard = observed_settings_guard
        self.admission._account_guard = observed_account_guard

        def claim():
            try:
                with config.resource_settings_guard():
                    settings_held.set()
                    if not reader_at_lock.wait(3):
                        raise TimeoutError("reader did not reach a lock")
                    claims.append(self.admission.claim_next())
            except Exception as error:
                errors.append(error)

        def observe():
            try:
                if not settings_held.wait(3):
                    raise TimeoutError("claim did not hold settings")
                snapshots.append(read())
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=claim),
                   threading.Thread(target=observe, name="capacity-reader")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(6)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(len(snapshots), 1)
        self.assertEqual([context.request_id for context in claims], ["original"])
        after = self.read("text", "happy", "original")
        self.assertEqual(after["request_message_id"], before["request_message_id"])
        self.assertFalse(after["_submission_started"])
        self.assertEqual(self.calls, [])

    def test_resource_read_and_original_claim_do_not_deadlock(self):
        self.assert_capacity_read_does_not_deadlock_claim(self.admission.resource_snapshot)

    def test_model_resource_read_and_original_claim_do_not_deadlock(self):
        self.assert_capacity_read_does_not_deadlock_claim(lambda: self.admission.model_resources([]))

    def test_ordinary_company_and_internal_take_turns_on_one_occupancy(self):
        for i in range(5):
            self.submit("bulk" + str(i), source="company:happy")
        self.submit("listing-wb", owner="admin", source="internal:wb-to-ozon")
        self.submit("listing-ozon", owner="admin", source="internal:ozon-to-wb")
        self.submit("ordinary", owner="key-user", source="key:key-user")
        for _ in range(4):
            ctx = self.admission.claim_next()
            self.assertIsNotNone(ctx)
            self.assertIsNone(self.admission.claim_next())
            self.admission.execute(ctx)
        self.assertEqual({name for _, name, _ in self.calls}, {"bulk0", "listing-wb", "listing-ozon", "ordinary"})

    def test_same_session_blocks_later_turn_only_not_other_sources(self):
        self.write_accounts(2)
        self.submit("first", client_conversation_id="same")
        self.submit("second", client_conversation_id="same")
        first = self.admission.claim_next()
        first.before_send()
        self.submit("other", owner="wb")
        self.assertEqual(self.admission.claim_next().request_id, "other")
        self.assertIsNone(self.admission.claim_next())

    def test_durable_model_route_never_claims_same_plan_unobserved_account(self):
        self.accounts.add_account_items([{
            "access_token": "fixture-token-observed", "account_id": "upstream-observed",
            "provider_account_identity": "account-observed", "type": "Plus", "status": "正常",
            "quota": 999, "source_type": "web", "conversation_binding_ids": ["binding-observed"],
        }])
        self.admission.model_types = lambda _model: ModelRoute(
            account_types=frozenset({"Plus"}), account_identities=frozenset({"account-observed"}),
        )
        self.submit("observed-only")

        context = self.admission.claim_next()

        self.assertIsNotNone(context)
        receipt = self.read("text", "happy", "observed-only")
        self.assertEqual(receipt["provider_account_identity"], "account-observed")

    def terminal_empty_correction_pair(self):
        session = "same-public-session"
        # Public Chat stores the caller's session reference separately from
        # its owner-scoped hashed group used by the admission planner.
        group = "public-session-" + hashlib.sha256(("happy\0session\0" + session).encode()).hexdigest()
        self.assertNotEqual(group, session)
        self.submit("original", client_conversation_id=group,
                    _public_route="chat", _public_session_ref=session)
        original = self.read("text", "happy", "original")
        evidence = {"conversation_id": "original-conversation",
                    "request_message_id": original["request_message_id"],
                    "final_message_id": "empty-final", "observed_at": 100.0}
        with self.store.transaction() as db:
            original.update(status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                            recovery_reason="REQUEST_RESULT_TERMINAL_EMPTY",
                            _upstream_terminal=True, _turn_reserved=False,
                            provider_binding_id="binding-0",
                            provider_account_identity="account-0",
                            conversation_id="original-conversation",
                            _turn_end_evidence=evidence)
            self.store.write_receipt(db, "text", "happy", "original", original)
        self.submit("correction", client_conversation_id=group,
                    _public_route="chat", _public_session_ref=session,
                    _previous_request_id="original", _continue_after_terminal_empty=True)
        return evidence

    def test_verified_terminal_empty_correction_claims_once_without_changing_original(self):
        evidence = self.terminal_empty_correction_pair()
        original = self.read("text", "happy", "original")
        correction = self.read("text", "happy", "correction")
        self.assertEqual(correction["_terminal_empty_correction_of"], "original")
        self.assertTrue(self.store.load_input(correction["_input_ref"]))
        self.text.recovery_reader = lambda receipt: {
            "status": "unknown", "recovery_reason": "REQUEST_RESULT_TERMINAL_EMPTY",
            "provider_binding_id": receipt["provider_binding_id"],
            "provider_account_identity": receipt["provider_account_identity"],
            "client_conversation_id": receipt["client_conversation_id"],
            "conversation_id": receipt["conversation_id"],
            "_turn_end_evidence": evidence,
        }
        claim = self.admission.claim_next()
        self.assertIsNotNone(claim)
        self.assertEqual(claim.request_id, "correction")
        self.assertIsNone(self.admission.claim_next())
        self.admission.execute(claim)
        self.assertEqual([(owner, request_id) for owner, request_id, _ in self.calls],
                         [("happy", "correction")])
        self.assertEqual(self.read("text", "happy", "original"), original)
        self.assertEqual(self.read("text", "happy", "correction")["status"], "succeeded")

    def test_invalid_terminal_empty_link_never_clears_original_order_barrier(self):
        self.terminal_empty_correction_pair()
        original = self.read("text", "happy", "original")
        correction = self.read("text", "happy", "correction")
        cases = (
            ("correction", "_terminal_empty_correction_of", "other-request"),
            ("correction", "_previous_request_id", "other-request"),
            ("correction", "_public_session_ref", "other-session"),
            ("correction", "parent_message_id", "wrong-final"),
            ("correction", "provider_binding_id", "wrong-binding"),
            ("correction", "provider_account_identity", "wrong-account"),
            ("correction", "_input_ref", None),
            ("original", "_upstream_terminal", False),
            ("original", "recovery_reason", "REQUEST_RESULT_INCOMPLETE"),
        )
        for name, field, value in cases:
            with self.subTest(name=name, field=field):
                with self.store.transaction() as db:
                    row = dict(original if name == "original" else correction)
                    row[field] = value
                    self.store.write_receipt(db, "text", "happy", name, row)
                self.assertIsNone(self.admission.claim_next())
                waiting = self.read("text", "happy", "correction")["waiting"]
                self.assertIn("earlier_group_request_unfinished", waiting["reasons"])
                with self.store.transaction() as db:
                    self.store.write_receipt(db, "text", "happy", name,
                                             original if name == "original" else correction)

    def completed_terminal_empty_correction(self):
        evidence = self.terminal_empty_correction_pair()
        self.text.recovery_reader = lambda receipt: {
            "status": "unknown", "recovery_reason": "REQUEST_RESULT_TERMINAL_EMPTY",
            **{key: receipt[key] for key in (
                "provider_binding_id", "provider_account_identity",
                "client_conversation_id", "conversation_id")},
            "_turn_end_evidence": evidence,
        }
        runner = self.text.runner
        def completed_runner(body, on_cursor):
            result = runner(body, on_cursor)
            on_cursor({"parent_message_id": "correction-final"})
            return {**result, "parent_message_id": "correction-final"}
        self.text.runner = completed_runner
        self.admission.execute(self.admission.claim_next())
        correction = self.read("text", "happy", "correction")
        self.assertEqual(correction["status"], "succeeded")
        self.assertEqual(correction["_submission_parent_message_id"], evidence["final_message_id"])
        self.assertEqual(correction["parent_message_id"], "correction-final")
        self.submit("successor", client_conversation_id=correction["client_conversation_id"],
                    _public_route="chat", _public_session_ref=correction["_public_session_ref"],
                    _previous_request_id="correction")

    def test_completed_correction_allows_successor_after_cursor_advance_and_restart(self):
        self.completed_terminal_empty_correction()
        original = self.read("text", "happy", "original")
        self.assertEqual(original["status"], "unknown")
        self.assertEqual(self.read("text", "happy", "successor")["parent_message_id"], "correction-final")
        _, _, restarted = build(self.root, self.clock)
        claim = restarted.claim_next()
        self.assertIsNotNone(claim)
        self.assertEqual(claim.request_id, "successor")
        self.assertIsNone(self.admission.claim_next())
        self.assertEqual(self.read("text", "happy", "original"), original)

    def test_completed_correction_still_requires_original_submission_anchor_and_identity(self):
        self.completed_terminal_empty_correction()
        correction = self.read("text", "happy", "correction")
        original = self.read("text", "happy", "original")
        for field, value in (
            ("_submission_parent_message_id", None),
            ("_submission_parent_message_id", "wrong-final"),
            ("provider_account_identity", "wrong-account"),
            ("_previous_request_id", "wrong-request"),
            ("_public_session_ref", "wrong-session"),
            ("status", "unknown"),
        ):
            with self.subTest(field=field, value=value):
                with self.store.transaction() as db:
                    self.store.write_receipt(db, "text", "happy", "correction", {**correction, field: value})
                self.assertIsNone(self.admission.claim_next())
                self.assertIn("earlier_group_request_unfinished",
                              self.read("text", "happy", "successor")["waiting"]["reasons"])
                self.assertEqual(self.read("text", "happy", "original"), original)
        with self.store.transaction() as db:
            self.store.write_receipt(db, "text", "happy", "correction", correction)

    def test_terminal_empty_correction_competing_workers_claim_only_once(self):
        self.terminal_empty_correction_pair()
        workers = [build(self.root, self.clock)[2] for _ in range(2)]
        start = threading.Barrier(3)
        def claim(worker):
            start.wait()
            return worker.claim_next()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(claim, worker) for worker in workers]
            start.wait()
            claims = [future.result() for future in futures]
        self.assertEqual([ctx.request_id for ctx in claims if ctx is not None], ["correction"])
        self.assertIsNone(self.admission.claim_next())

    def test_ordinary_unknown_still_blocks_same_session_successor(self):
        self.submit("original", client_conversation_id="same")
        self.submit("successor", client_conversation_id="same")
        with self.store.transaction() as db:
            original = self.store.read_receipt(db, "text", "happy", "original")
            original.update(status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN")
            self.store.write_receipt(db, "text", "happy", "original", original)
        self.assertIsNone(self.admission.claim_next())
        self.assertIn("earlier_group_request_unfinished",
                      self.read("text", "happy", "successor")["waiting"]["reasons"])

    def test_succeeded_same_session_original_releases_queued_successor(self):
        self.submit("original", client_conversation_id="same")
        self.submit("successor", client_conversation_id="same")

        self.admission.execute(self.admission.claim_next())

        original = self.read("text", "happy", "original")
        self.assertEqual(original["status"], "succeeded")
        successor = self.admission.claim_next()
        self.assertIsNotNone(successor)
        self.assertEqual(successor.request_id, "successor")

    def test_unknown_turn_uses_remaining_capacity_without_overclaiming_same_variant(self):
        self.admission.settings = lambda: {
            "chat_account_concurrency": 3,
            "image_account_concurrency": 4,
            "codex_max_concurrency": 4,
        }
        self.submit("sent")
        sent = self.admission.claim_next()
        sent.before_send()
        self.text._update("happy", "sent", status="unknown", upstream_outcome="unknown", _executing=False)
        for request_id in ("queued-1", "queued-2", "queued-3"):
            self.submit(request_id)

        claimed = [self.admission.claim_next(), self.admission.claim_next()]

        self.assertEqual({context.request_id for context in claimed}, {"queued-1", "queued-2"})
        self.assertIsNone(self.admission.claim_next())
        self.assertEqual(self.read("text", "happy", "queued-3")["status"], "queued")
        account = self.admission.resource_snapshot()["accounts"][0]["chat_turn"]
        self.assertEqual((account["capacity"], account["occupied"]), (3, 3))

    def test_paused_unknown_still_blocks_same_session_successor(self):
        self.submit("original", client_conversation_id="same")
        self.submit("successor", client_conversation_id="same")
        with self.store.transaction() as db:
            original = self.store.read_receipt(db, "text", "happy", "original")
            original.update(status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                            _recovery_paused=True, _attempt_finished_at=self.clock())
            self.store.write_receipt(db, "text", "happy", "original", original)
        self.assertIsNone(self.admission.claim_next())
        self.assertIn("earlier_group_request_unfinished",
                      self.read("text", "happy", "successor")["waiting"]["reasons"])

    def test_snapshot_skips_only_planner_terminal_records(self):
        """Terminal history remains in the input but is not a planner request."""
        for request_id in ("succeeded", "failed-known", "paused-unknown", "failed-unknown", "queued"):
            self.submit(request_id, client_conversation_id="projection-" + request_id)
        with self.store.transaction() as db:
            succeeded = self.store.read_receipt(db, "text", "happy", "succeeded")
            succeeded.update(status="succeeded", content="saved", upstream_outcome="completed")
            self.store.write_receipt(db, "text", "happy", "succeeded", succeeded)
            failed_known = self.store.read_receipt(db, "text", "happy", "failed-known")
            failed_known.update(status="failed", error_code="INPUT_REJECTED", upstream_outcome="not_sent")
            self.store.write_receipt(db, "text", "happy", "failed-known", failed_known)
            paused = self.store.read_receipt(db, "text", "happy", "paused-unknown")
            paused.update(status="unknown", _recovery_paused=True, _attempt_finished_at=self.clock())
            self.store.write_receipt(db, "text", "happy", "paused-unknown", paused)
            failed_unknown = self.store.read_receipt(db, "text", "happy", "failed-unknown")
            failed_unknown.update(status="failed", error_code="RESULT_UNRECOVERABLE", upstream_outcome="unknown")
            self.store.write_receipt(db, "text", "happy", "failed-unknown", failed_unknown)
            receipts = list(self.store.receipts(db))
            snapshot = self.admission._snapshot(
                self.rows, receipts, self.admission.settings(), {"fixture-text": {"Plus"}}, self.clock(),
                self.store.runtime(db, "fairness", {}),
                workflow_scheduling.state_snapshot(self.store, db), product_work.state_snapshot(self.store, db), db=db,
            )
        request_ids = {request.ref.request_id for request in snapshot.requests}
        self.assertNotIn("succeeded", request_ids)
        self.assertNotIn("failed-known", request_ids)
        self.assertTrue({"paused-unknown", "failed-unknown", "queued"} <= request_ids)

    def test_capacity_only_snapshot_matches_full_chat_turns(self):
        self.write_accounts(2)
        self.admission.settings = lambda: {
            "chat_account_concurrency": 2,
            "image_account_concurrency": 4,
            "codex_max_concurrency": 4,
        }
        for request_id, account in (("running", "account-0"), ("paused", "account-0"),
                                    ("failed-unknown", "account-1")):
            self.submit(request_id, provider_account_identity=account,
                        provider_binding_id="binding-" + account.rsplit("-", 1)[1])
        with self.store.transaction() as db:
            running = self.store.read_receipt(db, "text", "happy", "running")
            running.update(status="running", _executing=True, _claim_id="live", _claim_until=self.clock() + 30,
                           _submission_started=True, _turn_reserved=True)
            self.store.write_receipt(db, "text", "happy", "running", running)
            paused = self.store.read_receipt(db, "text", "happy", "paused")
            paused.update(status="unknown", _recovery_paused=True, _attempt_finished_at=self.clock())
            self.store.write_receipt(db, "text", "happy", "paused", paused)
            failed = self.store.read_receipt(db, "text", "happy", "failed-unknown")
            failed.update(status="failed", error_code="RESULT_UNRECOVERABLE", upstream_outcome="unknown")
            self.store.write_receipt(db, "text", "happy", "failed-unknown", failed)
        self.terminal_empty_correction_pair()
        with self.store.connect() as db:
            receipts = list(self.store.receipts(db))
            rows, settings = self.admission._rows(), self.admission.settings()
            full = self.admission._snapshot(
                rows, receipts, settings, {"fixture-text": {"Plus"}}, self.clock(),
                self.store.runtime(db, "fairness", {}),
                workflow_scheduling.state_snapshot(self.store, db), product_work.state_snapshot(self.store, db), db=db,
            )
            capacity = self.admission._snapshot(rows, receipts, settings, {}, self.clock(), {}, db=db,
                                                capacity_only=True)
        turns = lambda snapshot: sorted((resource.key, resource.capacity, resource.occupied, resource.next_at)
                                        for resource in snapshot.resources if resource.key.startswith("chat_turn:"))
        self.assertEqual(turns(capacity), turns(full))

    def test_final_guard_capacity_projection_serializes_with_next_claim(self):
        self.write_accounts(2)
        self.admission.settings = lambda: {
            "chat_account_concurrency": 1,
            "image_account_concurrency": 4,
            "codex_max_concurrency": 4,
        }
        self.submit("first", owner="first")
        self.submit("second", owner="second")
        first = self.admission.claim_next()
        entered, release, guard_done = threading.Event(), threading.Event(), threading.Event()
        result, errors = {}, []
        original_snapshot = self.admission._snapshot

        def hold_capacity_snapshot(*args, **kwargs):
            if kwargs.get("capacity_only"):
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test did not release final guard")
            return original_snapshot(*args, **kwargs)

        with patch.object(self.admission, "_snapshot", side_effect=hold_capacity_snapshot):
            guard = threading.Thread(target=lambda: (first.before_send(), guard_done.set()))
            guard.start()
            self.assertTrue(entered.wait(2))

            def claim():
                try:
                    result["context"] = self.admission.claim_next()
                except BaseException as exc:
                    errors.append(exc)

            claimer = threading.Thread(target=claim)
            claimer.start()
            time.sleep(.05)
            self.assertTrue(claimer.is_alive())
            release.set()
            guard.join(2)
            claimer.join(2)
        self.assertTrue(guard_done.is_set())
        self.assertEqual(errors, [])
        self.assertEqual(result["context"].request_id, "second")

    def test_restart_loads_actual_multimodal_input_and_claims_once(self):
        self.submit("original", messages=[{"role": "user", "content": [(b"original image", "image.png", "image/png")]}])
        _, _, restarted = build(self.root, self.clock)
        restarted.register("text", self.admission.handlers["text"])
        context = restarted.claim_next()
        restarted.execute(context)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][2]["messages"][0]["content"][0][0], b"original image")
        self.assertIsNone(self.admission.claim_next())
        receipt = self.text.read("happy", "original")
        self.assertNotIn("_input_ref", receipt)
        self.assertNotIn(b"original image", self.store.path.read_bytes())
        self.assertEqual(self.store.input_dir.stat().st_mode & 0o777, 0o700)
        self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in self.store.input_dir.iterdir()))

    def test_expired_unsent_claim_is_fenced_and_sent_claim_is_unknown(self):
        self.submit("original")
        old = self.admission.claim_next()
        self.clock.now += 31
        next_context = self.admission.claim_next()
        self.assertEqual(next_context.request_id, old.request_id)
        self.assertNotEqual(next_context.claim, old.claim)
        with self.assertRaises(AdmissionLost):
            old.before_send()
        next_context.before_send()
        self.clock.now += 31
        self.assertIsNone(self.admission.claim_next())
        self.assertEqual(self.read("text", "happy", "original")["status"], "unknown")
        self.write_accounts(2)
        self.assertIsNone(self.admission.claim_next())

    def test_catalog_prefilter_retains_live_text_but_planner_keeps_saved_image_data(self):
        expected_models = set()
        for status in ("queued", "running", "not_started"):
            self.submit("catalog-" + status)
            with self.store.transaction() as db:
                receipt = self.store.read_receipt(db, "text", "happy", "catalog-" + status)
                receipt.update(status=status, model="catalog-" + status)
                self.store.write_receipt(db, "text", "happy", "catalog-" + status, receipt)
            expected_models.add("catalog-" + status)
        self.image("saved-catalog-image")
        data = [{"b64_json": "saved-source-preserved" * 1000}]
        with self.store.transaction() as db:
            receipt = self.store.read_receipt(db, "image", "happy", "saved-catalog-image")
            receipt.update(status="success", data=data, upstream_unfinished=False)
            self.store.write_receipt(db, "image", "happy", "saved-catalog-image", receipt)
        catalog_calls, reads = [], []
        self.admission.model_types = lambda model: catalog_calls.append(model) or {"Plus"}
        original = self.store.receipts
        def observed(db, **kwargs):
            rows = list(original(db, **kwargs))
            reads.append((kwargs, rows))
            return iter(rows)
        with patch.object(self.store, "receipts", side_effect=observed):
            self.admission.claim_next()
        self.assertTrue(expected_models <= set(catalog_calls))
        filtered = [rows for args, rows in reads if args.get("statuses") == ("queued", "running", "not_started")]
        self.assertEqual(len(filtered), 1)
        self.assertNotIn("saved-catalog-image", [rid for _, _, rid, _ in filtered[0]])
        complete = [r for args, rows in reads if not args for _, _, rid, r in rows if rid == "saved-catalog-image"]
        self.assertTrue(complete)
        self.assertTrue(all(r["data"] == data for r in complete))

    def test_unknown_originals_do_not_probe_catalog_but_keep_order_fence(self):
        from services.pool_admission import unfinished
        catalog_calls = []
        self.admission.model_types = lambda model: catalog_calls.append(model) or {"Plus"}
        for name, values in [
            ("unknown", {"status": "unknown"}),
            ("failed-unknown", {"status": "failed", "error_code": "RESULT_UNRECOVERABLE",
                                "upstream_outcome": "unknown"}),
            ("completed-original", {"status": "failed", "error_code": "RESULT_UNRECOVERABLE",
                                    "upstream_outcome": "unknown", "_attempt_finished_at": 999,
                                    "_completion": {"state": "completed", "selected_id": "saved-child"}}),
        ]:
            self.submit(name, provider_binding_id="binding-0", provider_account_identity="account-0")
            with self.store.transaction() as db:
                receipt = self.store.read_receipt(db, "text", "happy", name)
                receipt.update(values)
                self.store.write_receipt(db, "text", "happy", name, receipt)
            self.assertIsNone(self.admission.claim_next())
            saved = self.read("text", "happy", name)
            self.assertTrue(unfinished("text", saved))
            for key, value in values.items():
                self.assertEqual(saved[key], value)
        self.assertEqual(catalog_calls, [])

    def test_not_started_blocks_its_chain_without_consuming_an_independent_claim(self):
        session = "not-started-chain"
        self.submit("not-started", source="shared", client_conversation_id=session)
        with self.store.transaction() as db:
            receipt = self.store.read_receipt(db, "text", "happy", "not-started")
            receipt["status"] = "not_started"
            self.store.write_receipt(db, "text", "happy", "not-started", receipt)
        self.submit("same-chain", source="shared", client_conversation_id=session)
        self.submit("independent", owner="other", source="shared")

        context = self.admission.claim_next()

        self.assertIsNotNone(context)
        self.assertEqual(context.request_id, "independent")
        original = self.read("text", "happy", "not-started")
        self.assertEqual(original["status"], "not_started")
        self.assertFalse(original["_submission_started"])
        self.assertEqual(self.calls, [])
        self.assertIn("earlier_group_request_unfinished",
                      self.read("text", "happy", "same-chain")["waiting"]["reasons"])

    def test_expired_unsent_worker_releases_execution_marker_for_work_pause(self):
        from services.work_lifecycle import WorkLifecycleService
        self.submit("stopped-unsent", _public_session_ref="session-stopped-unsent")
        old = self.admission.claim_next()
        self.clock.now += 31
        with self.store.transaction() as db:
            self.admission._recover_claims(db, list(self.store.receipts(db)), self.clock())
        receipt = self.read("text", "happy", "stopped-unsent")
        self.assertEqual(receipt["status"], "queued")
        self.assertFalse(receipt["_executing"])
        self.assertIsNone(receipt["_claim_until"])
        with self.assertRaises(AdmissionLost):
            old.before_send()
        lifecycle = WorkLifecycleService(self.text, self.images, clock=self.clock)
        work = lifecycle.update("text", {"id": "happy"}, "stopped-unsent", "paused")
        self.assertEqual(work["state"], "paused")
        self.assertFalse(work["slot_held"])
        self.assertIsNone(self.admission.claim_next())
        self.assertEqual(self.calls, [])

    def test_new_queued_text_still_discovers_models_for_admission(self):
        catalog_calls = []
        self.admission.model_types = lambda model: catalog_calls.append(model) or {"Plus"}
        self.submit("new-work")
        self.assertEqual(self.admission.claim_next().request_id, "new-work")
        self.assertEqual(catalog_calls, ["fixture-text"])

    def test_legacy_fenced_unsent_marker_is_cleared_without_changing_original_input(self):
        self.submit("legacy-unsent")
        with self.store.transaction() as db:
            row = self.store.read_receipt(db, "text", "happy", "legacy-unsent")
            row.update(_executing=True, _claim_id=None, _claim_until=999, _submission_started=False)
            self.store.write_receipt(db, "text", "happy", "legacy-unsent", row)
            self.admission._recover_claims(db, list(self.store.receipts(db)), self.clock())
        saved = self.read("text", "happy", "legacy-unsent")
        self.assertEqual(saved, {**row, "_executing": False, "_claim_until": None})
        self.assertEqual(self.calls, [])

    def test_legacy_marker_cleanup_does_not_change_sent_or_owned_work(self):
        for index, change in enumerate((
                {"status": "unknown"}, {"_claim_id": "live", "_claim_until": 2000},
                {"_submission_started": True}, {"_executing": False})):
            with self.subTest(change=change):
                rid = "preserved-" + str(index)
                self.submit(rid)
                with self.store.transaction() as db:
                    row = self.store.read_receipt(db, "text", "happy", rid)
                    row.update(_executing=True, _claim_id=None, _claim_until=999,
                               _submission_started=False)
                    row.update(change)
                    self.store.write_receipt(db, "text", "happy", rid, row)
                    self.admission._recover_claims(db, [("text", "happy", rid, row.copy())], self.clock())
                self.assertEqual(self.read("text", "happy", rid), row)
        self.assertEqual(self.calls, [])

    def test_expired_captured_image_claim_preserves_real_failure_or_uses_neutral_state(self):
        for prior_code in (None, "RECOVERY_TIMED_OUT"):
            with self.subTest(prior_code=prior_code):
                rid = "captured-" + str(prior_code)
                self.image(rid)
                context = self.admission.claim_next()
                context.before_send()
                context.record_stage("send_call_started")
                detail = {"phase": "collect_image_result", "type": "TimeoutError", "at": 1000}
                self.admission.update_claim(context, result_file_ids=["original-file"],
                    recovery_error_code=prior_code, last_recovery_failure=detail)
                self.clock.now += 31
                self.assertIsNone(self.admission.claim_next())
                record = self.read("image", "happy", rid)
                self.assertEqual(record["recovery_error_code"], prior_code or "RECOVERY_RESULT_INCOMPLETE")
                self.assertEqual(record["last_recovery_failure"], detail)
                self.assertEqual(record["recovery_phase"], "download_image_result")
                self.assertEqual(record["result_file_ids"], ["original-file"])
                self.assertFalse(record["upstream_unfinished"])
                self.assertEqual(sum(e["stage"] == "send_call_started" for e in record["_execution_timeline"]), 1)

    def test_account_disabled_between_claim_and_send_is_not_sent(self):
        self.submit("original")
        context = self.admission.claim_next()
        self.rows[0]["managed_disabled"] = True
        (self.root / "accounts.json").write_text(json.dumps(self.rows))
        with self.assertRaises(AdmissionLost):
            context.before_send()
        self.assertFalse(self.read("text", "happy", "original")["_submission_started"])

    def test_two_worker_processes_compete_for_one_original_slot(self):
        self.submit("one")
        self.submit("two", owner="wb")
        mp = multiprocessing.get_context("spawn")
        ready, results, start = mp.Queue(), mp.Queue(), mp.Event()
        children = [mp.Process(target=claim_worker, args=(str(self.root), ready, start, results)) for _ in range(2)]
        for child in children:
            child.start()
        for _ in children:
            self.assertTrue(ready.get(timeout=15))
        start.set()
        claimed = [results.get(timeout=15) for _ in children]
        for child in children:
            child.join(timeout=10)
            self.assertEqual(child.exitcode, 0)
        self.assertEqual(sum(value is not None for value in claimed), 1)

    def test_two_workers_compete_for_only_the_named_extra_chat_claim(self):
        from services.config import ConfigStore
        (self.root / "config.json").write_text(json.dumps({"auth-key": "fixture-admin"}))
        config = ConfigStore(self.root / "config.json")
        self.submit("active", provider_account_identity="account-0", provider_binding_id="binding-0")
        active = self.admission.claim_next()
        active.before_send()
        self.submit("other", owner="happy", provider_account_identity="account-0", provider_binding_id="binding-0")
        self.submit("chosen", owner="wb", provider_account_identity="account-0", provider_binding_id="binding-0")
        config.update_temporary_chat_second_slot(0, {
            "account_identity": "account-0", "request_id": "chosen", "expires_at": 1100})
        mp = multiprocessing.get_context("spawn")
        ready, results, start = mp.Queue(), mp.Queue(), mp.Event()
        children = [mp.Process(target=scoped_claim_worker,
                               args=(str(self.root), ready, start, results)) for _ in range(2)]
        for child in children:
            child.start()
        for _ in children:
            self.assertTrue(ready.get(timeout=15))
        start.set()
        claimed = [results.get(timeout=15) for _ in children]
        for child in children:
            child.join(timeout=10)
            self.assertEqual(child.exitcode, 0)
        self.assertEqual([value[0] for value in claimed if value is not None], ["chosen"])
        self.assertEqual(self.read("text", "happy", "other")["status"], "queued")

    def test_stopped_scoped_slot_cannot_claim_or_send_original(self):
        from services.config import ConfigStore
        (self.root / "config.json").write_text(json.dumps({"auth-key": "fixture-admin"}))
        config = ConfigStore(self.root / "config.json")
        self.admission.settings = config.resource_settings
        self.submit("active", provider_account_identity="account-0", provider_binding_id="binding-0")
        self.admission.claim_next().before_send()
        self.submit("chosen", owner="wb", provider_account_identity="account-0", provider_binding_id="binding-0")
        config.update_temporary_chat_second_slot(0, {
            "account_identity": "account-0", "request_id": "chosen", "expires_at": 1100})
        config.update_temporary_chat_second_slot(1, None)
        self.assertIsNone(self.admission.claim_next())
        self.assertEqual(self.read("text", "wb", "chosen")["status"], "queued")

        config.update_temporary_chat_second_slot(2, {
            "account_identity": "account-0", "request_id": "chosen", "expires_at": 1100})
        chosen = self.admission.claim_next()
        self.assertEqual(chosen.request_id, "chosen")
        config.update_temporary_chat_second_slot(3, None)
        with self.assertRaises(AdmissionLost):
            chosen.before_send()
        self.assertFalse(self.read("text", "wb", "chosen")["_submission_started"])
        self.admission.execute(chosen)
        self.assertEqual(self.read("text", "wb", "chosen")["status"], "queued")
        self.assertEqual(self.calls, [])

    def test_stop_interleaved_with_scoped_claim_cannot_send_after_stop(self):
        from services.config import ConfigStore
        (self.root / "config.json").write_text(json.dumps({"auth-key": "fixture-admin"}))
        config = ConfigStore(self.root / "config.json")
        self.admission.settings = config.resource_settings
        self.submit("active", provider_account_identity="account-0", provider_binding_id="binding-0")
        self.admission.claim_next().before_send()
        self.submit("chosen", owner="wb", provider_account_identity="account-0", provider_binding_id="binding-0")
        config.update_temporary_chat_second_slot(0, {
            "account_identity": "account-0", "request_id": "chosen", "expires_at": 1100})

        snapshot_ready, release_claim = threading.Event(), threading.Event()
        original_snapshot = self.admission._snapshot
        def paused_snapshot(*args, **kwargs):
            result = original_snapshot(*args, **kwargs)
            snapshot_ready.set()
            if not release_claim.wait(5):
                raise AssertionError("claim was not released")
            return result
        self.admission._snapshot = paused_snapshot
        claim_result, stop_result = {}, {}
        def claim():
            try:
                claim_result["context"] = self.admission.claim_next()
            except Exception as error:
                claim_result["error"] = error
        def stop():
            try:
                stop_result["settings"] = config.update_temporary_chat_second_slot(1, None)
            except Exception as error:
                stop_result["error"] = error
        claimant = threading.Thread(target=claim)
        stopper = threading.Thread(target=stop)
        claimant.start()
        try:
            self.assertTrue(snapshot_ready.wait(5))
            stopper.start()
        finally:
            release_claim.set()
        claimant.join(10)
        stopper.join(10)
        self.assertFalse(claimant.is_alive())
        self.assertFalse(stopper.is_alive())
        self.assertNotIn("error", claim_result)
        self.assertNotIn("error", stop_result)
        self.assertEqual(stop_result["settings"]["revision"], 2)
        chosen = claim_result["context"]
        self.assertEqual(chosen.request_id, "chosen")
        with self.assertRaises(AdmissionLost):
            chosen.before_send()
        receipt = self.read("text", "wb", "chosen")
        self.assertFalse(receipt["_submission_started"])
        self.assertNotIn("send_guard_passed", [item["stage"] for item in receipt["_execution_timeline"]])

    def test_scoped_claim_expired_before_send_does_not_submit(self):
        from services.config import ConfigStore
        (self.root / "config.json").write_text(json.dumps({"auth-key": "fixture-admin"}))
        config = ConfigStore(self.root / "config.json")
        self.admission.settings = config.resource_settings
        self.submit("active", provider_account_identity="account-0", provider_binding_id="binding-0")
        self.admission.claim_next().before_send()
        self.submit("chosen", owner="wb", provider_account_identity="account-0", provider_binding_id="binding-0")
        config.update_temporary_chat_second_slot(0, {
            "account_identity": "account-0", "request_id": "chosen", "expires_at": 1001})
        chosen = self.admission.claim_next()
        self.assertEqual(chosen.request_id, "chosen")
        self.clock.now = 1002
        with self.assertRaises(AdmissionLost):
            chosen.before_send()
        self.assertFalse(self.read("text", "wb", "chosen")["_submission_started"])

    def test_duplicate_upstream_account_does_not_double_turn_capacity(self):
        self.write_accounts(2)
        self.rows[1]["account_id"] = self.rows[0]["account_id"]
        (self.root / "accounts.json").write_text(json.dumps(self.rows))
        self.submit("one")
        self.submit("two", owner="wb")
        self.assertIsNotNone(self.admission.claim_next())
        self.assertIsNone(self.admission.claim_next())

    def test_private_input_drift_never_runs(self):
        self.submit("original")
        receipt = self.read("text", "happy", "original")
        path = self.store.input_dir / receipt["_input_ref"]
        path.write_text(path.read_text().replace("private original input", "tampered original input"))
        self.admission.execute(self.admission.claim_next())
        self.assertEqual(self.calls, [])
        self.assertEqual(self.text.read("happy", "original")["error_code"], "TASK_INPUT_UNAVAILABLE")

    def test_workers_scale_with_usable_physical_accounts_beyond_four(self):
        self.write_accounts(3)
        self.admission.settings = lambda: {"chat_account_concurrency": 2, "image_account_concurrency": 4, "codex_max_concurrency": 4}
        for i in range(7):
            self.submit("many-" + str(i))
        claims = [self.admission.claim_next() for _ in range(6)]
        self.assertTrue(all(claims))
        self.assertIsNone(self.admission.claim_next())
        self.assertEqual(self.admission.resource_snapshot()["execution"]["chat_workers_limit"], 6)
        self.assertEqual(len({c.receipt()["provider_account_identity"] for c in claims}), 3)
        self.write_accounts(4)
        self.assertEqual(self.admission.claim_next().request_id, "many-6")

    def test_generated_image_save_releases_image_capacity_but_preserves_receipt_and_memory(self):
        self.admission.settings = lambda: {"chat_account_concurrency": 2, "image_account_concurrency": 1, "codex_max_concurrency": 4}
        self.image("saving")
        first = self.admission.claim_next()
        first.before_send()
        self.image("next")
        self.assertIsNone(self.admission.claim_next())
        self.admission.update_claim(first, upstream_unfinished=False, result_file_ids=["original-file"], request_message_id="original-message")
        before = first.receipt()
        snapshot = self.admission.resource_snapshot()
        self.assertEqual(snapshot["image"]["inflight"], 0)
        self.assertEqual(snapshot["queue"]["saving_images"], 1)
        self.assertGreater(snapshot["execution"]["active_input_bytes"], 0)
        next_context = self.admission.claim_next()
        self.assertEqual(next_context.request_id, "next")
        next_context.before_send()
        for context in (first, next_context):
            event = next(e for e in context.receipt()["_execution_timeline"] if e["stage"] == "send_guard_passed")
            self.assertEqual(event["image_capacity_at_send"], 1)
            self.assertEqual(event["image_occupied_at_send_including_current"], 1)
            self.assertNotIn("original-file", json.dumps(event))
        self.assertEqual(first.receipt(), before)

    def test_unknown_or_multi_send_image_keeps_generation_capacity(self):
        from services.pool_admission import image_generation_active
        self.assertTrue(image_generation_active("image", {"status": "error", "upstream_unfinished": True}, True))
        self.assertTrue(image_generation_active("image", {"upstream_unfinished": False}, True))
        self.assertTrue(image_generation_active("text", {"_operation": "image", "_upstream_terminal": True, "_expected_sends": 2}, True))
        self.assertFalse(image_generation_active("text", {"_operation": "image", "_upstream_terminal": True, "_expected_sends": 1}, True))

    def test_same_source_people_rotate_and_cursor_survives_restart(self):
        for i in range(8):
            self.submit("bulk-" + str(i), owner="person-a", source="internal:happy")
        self.submit("small", owner="person-b", source="internal:happy")
        first = self.admission.claim_next()
        self.assertEqual(first.owner, "person-a")
        self.admission.execute(first)
        _, _, restarted = build(self.root, self.clock)
        self.assertEqual(restarted.claim_next().owner, "person-b")

    def test_product_image_lane_rotates_after_text_without_changing_text_order(self):
        """A ready main image gets the next released bound turn, not a new priority."""
        source = "internal:listing"
        for request_id in ("older-copy", "later-copy"):
            self.submit(request_id, source=source, product_work={
                "product_ref": request_id, "step_ref": "copy",
            })
        self.images.submit_generation(
            {"id": "happy", "role": "admin", "external_image_client": True,
             "_trusted_source": source},
            client_task_id="main-image", prompt="private image input", model="gpt-image-2", size=None,
            # The image is a later AI step of the claimed product, so it must
            # retain that product admission marker rather than consume a new
            # product slot.
            product_work={"product_ref": "older-copy", "step_ref": "image"},
        )

        first = self.admission.claim_next()
        self.assertEqual((first.kind, first.request_id), ("text", "older-copy"))
        # Simulate the completed original turning over its account slot.  The
        # successful claim already persisted the existing fairness cursor.
        with self.store.transaction() as db:
            receipt = self.store.read_receipt(db, "text", "happy", "older-copy")
            receipt.update(status="succeeded", upstream_outcome="completed", _attempt_finished_at=self.clock.now,
                           _executing=False, _turn_reserved=False, _claim_id=None, _claim_until=None)
            self.store.write_receipt(db, "text", "happy", "older-copy", receipt)

        second = self.admission.claim_next()
        self.assertEqual((second.kind, second.request_id), ("image", "main-image"))

    def test_non_product_requests_keep_their_existing_sequence_lane(self):
        source = "internal:listing"
        self.submit("older-copy", source=source)
        self.submit("later-copy", source=source)
        self.images.submit_generation(
            {"id": "happy", "role": "user", "external_image_client": True,
             "_trusted_source": source},
            client_task_id="later-image", prompt="private image input", model="gpt-image-2", size=None,
        )
        first = self.admission.claim_next()
        self.assertEqual((first.kind, first.request_id), ("text", "older-copy"))
        with self.store.transaction() as db:
            receipt = self.store.read_receipt(db, "text", "happy", "older-copy")
            receipt.update(status="succeeded", upstream_outcome="completed", _attempt_finished_at=self.clock.now,
                           _executing=False, _turn_reserved=False, _claim_id=None, _claim_until=None)
            self.store.write_receipt(db, "text", "happy", "older-copy", receipt)
        second = self.admission.claim_next()
        self.assertEqual((second.kind, second.request_id), ("text", "later-copy"))

    def test_product_image_lane_keeps_conversation_and_capacity_guards(self):
        source = "internal:listing"
        self.submit("same-conversation-copy", source=source, product_work={
            "product_ref": "copy-product", "step_ref": "copy",
        })
        self.images.submit_generation(
            {"id": "happy", "role": "admin", "external_image_client": True,
             "_trusted_source": source},
            client_task_id="same-conversation-image", prompt="private image input", model="gpt-image-2", size=None,
            product_work={"product_ref": "image-product", "step_ref": "image"},
        )
        with self.store.transaction() as db:
            image = self.store.read_receipt(db, "image", "happy", "same-conversation-image")
            image["client_conversation_id"] = "session-same-conversation-copy"
            self.store.write_receipt(db, "image", "happy", "same-conversation-image", image)

        first = self.admission.claim_next()
        self.assertEqual((first.kind, first.request_id), ("text", "same-conversation-copy"))
        self.assertIsNone(self.admission.claim_next())
        image = self.read("image", "happy", "same-conversation-image")
        self.assertEqual(image["waiting"]["reasons"], ["earlier_group_request_unfinished"])

        # A distinct product image still needs both the account Chat turn and
        # image-generation resource; a lane does not bypass either limit.
        with self.store.transaction() as db:
            copy = self.store.read_receipt(db, "text", "happy", "same-conversation-copy")
            copy.update(status="succeeded", upstream_outcome="completed", _attempt_finished_at=self.clock.now,
                        _executing=False, _turn_reserved=False, _claim_id=None, _claim_until=None)
            self.store.write_receipt(db, "text", "happy", "same-conversation-copy", copy)
        self.submit("occupied-turn", owner="other", source="internal:other")
        with self.store.transaction() as db:
            holder = self.store.read_receipt(db, "text", "other", "occupied-turn")
            holder.update(status="running", upstream_outcome="unknown", _submission_started=True,
                          _executing=True, _turn_reserved=True, _claim_id="fixture-claim",
                          _claim_until=self.clock.now + 60, provider_binding_id="binding-0",
                          provider_account_identity="account-0", conversation_id="occupied-conversation")
            self.store.write_receipt(db, "text", "other", "occupied-turn", holder)
        self.assertIsNone(self.admission.claim_next())
        image = self.read("image", "happy", "same-conversation-image")
        self.assertEqual(image["waiting"]["reasons"], ["resource_full"])

    def test_chat_text_and_image_send_edges_follow_completed_read_capacity_marker(self):
        self.admission.settings = lambda: {
            "chat_account_concurrency": 1, "image_account_concurrency": 1, "codex_max_concurrency": 4,
        }
        self.submit("unreadable-original")
        original_context = self.admission.claim_next()
        self.assertEqual(original_context.request_id, "unreadable-original")
        with self.store.transaction() as db:
            original = self.store.read_receipt(db, "text", "happy", "unreadable-original")
            original.update(
                status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN", upstream_outcome="unknown",
                _submission_started=True, _turn_reserved=False, _executing=False,
                _claim_id=None, _claim_until=None, conversation_id="original-conversation",
                recovery_attempt=9,
            )
            self.store.write_receipt(db, "text", "happy", "unreadable-original", original)

        def unavailable(_receipt):
            error = ConnectionError("fixture original GET unavailable")
            error._original_result_get_attempted = True
            raise error

        self.text.recovery_reader = unavailable
        self.assertEqual(self.text.read("happy", "unreadable-original")["status"], "unknown")
        original = self.read("text", "happy", "unreadable-original")
        self.assertEqual(original["recovery_attempt"], 10)
        self.assertEqual(original["recovery_next_at"], self.clock.now + 300)
        self.assertEqual(original["_result_read_capacity_released_at"], self.clock.now)
        self.assertEqual(self.admission.resource_snapshot()["chat_turn"]["inflight"], 0)

        # The first failed exact GET makes unrelated work sendable immediately;
        # it does not wait for the next original result read.
        self.submit("text-pass")
        text_pass = self.admission.claim_next()
        self.assertEqual(text_pass.request_id, "text-pass")
        text_pass.before_send()
        with self.store.transaction() as db:
            receipt = self.store.read_receipt(db, "text", "happy", "text-pass")
            receipt.update(status="succeeded", upstream_outcome="completed", _attempt_finished_at=self.clock.now,
                           _executing=False, _turn_reserved=False, _claim_id=None, _claim_until=None)
            self.store.write_receipt(db, "text", "happy", "text-pass", receipt)

        self.image("image-pass")
        image_pass = self.admission.claim_next()
        self.assertEqual((image_pass.kind, image_pass.request_id), ("image", "image-pass"))
        image_pass.before_send()
        with self.store.transaction() as db:
            receipt = self.store.read_receipt(db, "image", "happy", "image-pass")
            receipt.update(status="success", upstream_unfinished=False, _executing=False,
                           _claim_id=None, _claim_until=None)
            self.store.write_receipt(db, "image", "happy", "image-pass", receipt)

        def progress_reader(content):
            def read(receipt):
                request = receipt["request_message_id"]
                final = "progress-" + content
                doc = {"conversation_id": receipt["conversation_id"], "current_node": final, "mapping": {
                    request: {"parent": "parent", "message": {"id": request, "author": {"role": "user"}}},
                    final: {"parent": request, "message": {"id": final, "author": {"role": "assistant"},
                        "status": "in_progress", "end_turn": None, "channel": "final",
                        "content": {"content_type": "text", "parts": [content]}}},
                }}
                return ConversationBindingService._read_text_request_result(None, receipt, document=doc)
            return read

        # Claim an image while the original marker releases its account. A real
        # identity-safe original node update before POST must fence the image.
        self.clock.now = original["recovery_next_at"] + 1
        self.image("image-race")
        image_race = self.admission.claim_next()
        self.assertEqual((image_race.kind, image_race.request_id), ("image", "image-race"))
        self.text.recovery_reader = progress_reader("verified-one")
        self.assertEqual(self.text.read("happy", "unreadable-original")["status"], "unknown")
        original = self.read("text", "happy", "unreadable-original")
        self.assertNotIn("_result_read_capacity_released_at", original)
        with self.assertRaisesRegex(AdmissionLost, "Chat account capacity changed"):
            image_race.before_send()
        image = self.read("image", "happy", "image-race")
        self.assertFalse(image["_submission_started"])
        self.assertEqual(image["waiting"]["reasons"], ["chat_capacity_changed"])
        # Retire this deliberately fenced image before exercising the text edge.
        with self.store.transaction() as db:
            image = self.store.read_receipt(db, "image", "happy", "image-race")
            image.update(status="error", upstream_unfinished=False, _attempt_finished_at=self.clock.now,
                         _executing=False, _claim_id=None, _claim_until=None)
            self.store.write_receipt(db, "image", "happy", "image-race", image)

        # The next scheduled read is still bounded to five minutes. Its completed
        # exact GET failure releases local capacity again; a later
        # verified node update fences a claimed text request at the same edge.
        self.clock.now = original["recovery_next_at"] + 1
        self.text.recovery_reader = unavailable
        self.assertEqual(self.text.read("happy", "unreadable-original")["status"], "unknown")
        original = self.read("text", "happy", "unreadable-original")
        self.assertIn("_result_read_capacity_released_at", original)
        self.clock.now = original["recovery_next_at"] + 1
        self.submit("text-race")
        text_race = self.admission.claim_next()
        self.assertEqual(text_race.request_id, "text-race")
        self.text.recovery_reader = progress_reader("verified-two")
        self.assertEqual(self.text.read("happy", "unreadable-original")["status"], "unknown")
        with self.assertRaisesRegex(AdmissionLost, "Chat account capacity changed"):
            text_race.before_send()
        text = self.read("text", "happy", "text-race")
        self.assertFalse(text["_submission_started"])
        self.assertEqual(text["waiting"]["reasons"], ["chat_capacity_changed"])

    def test_image_wire_request_does_not_consume_text_worker_budget(self):
        self.admission.text_workers = 1
        self.admission.settings = lambda: {"chat_account_concurrency": 2, "image_account_concurrency": 4, "codex_max_concurrency": 4}
        self.submit("image-wire")
        with self.store.transaction() as db:
            r = self.store.read_receipt(db, "text", "happy", "image-wire")
            r.update(_operation="image", model="gpt-image-2")
            self.store.write_receipt(db, "text", "happy", "image-wire", r)
        first = self.admission.claim_next()
        self.assertEqual(first.request_id, "image-wire")
        self.submit("recognition")
        self.assertEqual(self.admission.claim_next().request_id, "recognition")
        self.assertEqual(self.admission.resource_snapshot()["execution"]["chat_workers_active"], 1)

    def test_bad_identity_row_cannot_break_dynamic_capacity_or_leak_input(self):
        self.rows.append({"access_token": "unverified", "type": "Plus", "status": "正常", "quota": 999})
        (self.root / "accounts.json").write_text(json.dumps(self.rows))
        self.submit("blocked", provider_binding_id="binding-missing", provider_account_identity="missing")
        self.assertIsNone(self.admission.claim_next())
        snapshot = self.admission.resource_snapshot()
        self.assertEqual(snapshot["queue"]["by_reason"], {"bound_account_unavailable": 1})
        self.assertNotIn("private original input", json.dumps(snapshot))
        self.assertNotIn("fixture-token", json.dumps(snapshot))

    def test_waiting_diagnostics_redact_account_eligibility_failures(self):
        cases = (
            ("disabled", lambda: self.rows[0].update(managed_disabled=True), "account_disabled"),
            ("source", lambda: self.rows[0].update(source_type="codex"), "source_not_authorized"),
            ("plan", lambda: self.rows[0].update(type="Free"), "paid_plan_required"),
            ("identity", lambda: self.rows.append({**self.rows[0], "account_id": "duplicate-upstream",
                                                     "access_token": "fixture-token-duplicate"}), "identity_not_unique"),
            ("model-plan", lambda: setattr(self.admission, "model_types", lambda _model: ModelRoute(
                account_types=frozenset({"Pro"}), account_identities=frozenset({"account-0"}))), "model_plan_not_allowed"),
            ("model-account", lambda: setattr(self.admission, "model_types", lambda _model: ModelRoute(
                account_types=frozenset({"Plus"}), account_identities=frozenset({"other-account"}))), "model_account_not_allowed"),
            ("limit", lambda: self.rows[0].update(limits_progress=[{"feature_name": "fixture-text", "remaining": 0}]), "model_limit_zero"),
        )
        for name, configure, expected in cases:
            with self.subTest(name=name):
                self.write_accounts(1)
                self.admission.model_types = lambda _: {"Plus"}
                configure()
                (self.root / "accounts.json").write_text(json.dumps(self.rows))
                self.submit("diagnostic-" + name, provider_account_identity="account-0", provider_binding_id="binding-0")

                self.assertIsNone(self.admission.claim_next())

                waiting = self.read("text", "happy", "diagnostic-" + name)["waiting"]
                details = waiting["admission_diagnostics"]
                checks = {check for item in details["account_eligibility"] for check in item["failed_checks"]}
                self.assertIn(expected, checks)
                serialized = json.dumps(details)
                self.assertNotIn("account-0", serialized)
                self.assertNotIn("fixture-token", serialized)
                self.assertNotIn("private original input", serialized)
                self.assertIn("snapshot_revision", details)

    def test_waiting_diagnostics_preserve_missing_bound_offer_fact(self):
        self.submit("missing-offer", provider_account_identity="absent-account",
                    provider_binding_id="absent-binding")

        self.assertIsNone(self.admission.claim_next())

        waiting = self.read("text", "happy", "missing-offer")["waiting"]
        self.assertEqual(waiting["reasons"], ["bound_account_unavailable"])
        details = waiting["admission_diagnostics"]
        self.assertEqual(details["account_eligibility"][0]["failed_checks"], ["matching_offer_missing"])
        self.assertNotIn("absent-account", json.dumps(details))

    def test_waiting_diagnostics_report_redacted_resource_limit(self):
        self.submit("running", provider_account_identity="account-0", provider_binding_id="binding-0")
        running = self.admission.claim_next()
        running.before_send()
        self.submit("waiting-resource", provider_account_identity="account-0", provider_binding_id="binding-0")

        self.assertIsNone(self.admission.claim_next())

        waiting = self.read("text", "happy", "waiting-resource")["waiting"]
        details = waiting["admission_diagnostics"]
        turn = next(item for item in details["resource_limits"] if item["family"] == "chat_turn")
        self.assertEqual((turn["capacity"], turn["occupied"], turn["required_units"]), (1, 1, 1))
        self.assertEqual(turn["blockers"], ["resource_full"])
        self.assertTrue(turn["occupied_known"])
        serialized = json.dumps(details)
        self.assertNotIn("account-0", serialized)
        self.assertNotIn("fixture-token", serialized)
        self.assertNotIn("private original input", serialized)

    def test_unchanged_deferred_waiting_diagnostics_do_not_rewrite_receipts(self):
        self.accounts.update_account("fixture-token-0", {"managed_disabled": True}, quiet=True)
        self.submit("blocked-diagnostic")
        self.assertIsNone(self.admission.claim_next())
        initial = self.read("text", "happy", "blocked-diagnostic")["waiting"]
        self.assertIn("admission_diagnostics", initial)
        # Snapshot UUID/time are diagnostic stamps only. An unchanged blocked
        # queue must not serialize its receipt again on every admission tick.
        with patch.object(self.store, "write_receipt", wraps=self.store.write_receipt) as writes:
            self.assertIsNone(self.admission.claim_next())
        writes.assert_not_called()
        self.assertEqual(self.read("text", "happy", "blocked-diagnostic")["waiting"], initial)

    def test_large_paused_queue_builds_one_offer_per_account_capability(self):
        """Paused history remains visible, but cannot multiply identical offers."""
        self.write_accounts(4)
        self.admission.settings = lambda: {
            "chat_account_concurrency": 6,
            "image_account_concurrency": 4,
            "codex_max_concurrency": 4,
        }
        self.submit("dispatchable")
        with self.store.transaction() as db:
            for index in range(1244):
                request_id = "paused-" + str(index)
                work_key = "work:paused-" + str(index)
                receipt = {
                    "id": request_id,
                    "request_id": request_id,
                    "status": "queued",
                    "_route": "chat",
                    "_operation": "text",
                    "model": "fixture-text",
                    "_source": "fixture",
                    "_work_key": work_key,
                    "_sequence": index + 2,
                    "client_conversation_id": "paused-session-" + str(index),
                    "_input_bytes": 1,
                    "created_at": self.clock(),
                    "_submission_started": False,
                }
                db.execute("INSERT INTO requests(owner,id,request_hash,receipt) VALUES(?,?,?,?)",
                           ("paused-owner", request_id, "paused-hash-" + str(index), json.dumps(receipt)))
                self.store.set_runtime(db, work_key, {"key": work_key, "state": "paused", "slot_held": False})
            for index in range(1015):
                key = "product_work:paused-" + str(index)
                self.store.set_runtime(db, key, {"key": key, "state": "paused", "owner": "paused-owner"})

        from services.admission_planner import Offer
        with patch("services.pool_admission.Offer", wraps=Offer) as offers:
            context = self.admission.claim_next()

        self.assertEqual(context.request_id, "dispatchable")
        # Four accounts × the one queued (chat, fixture-text, text) capability.
        # The 1,244 paused rows still receive their existing work-state deferral.
        self.assertEqual(offers.call_count, 4)
        self.assertIn("work_not_active", self.read("text", "paused-owner", "paused-0")["waiting"]["reasons"])

    def test_waiting_diagnostic_unknown_resource_family_is_redacted(self):
        from services.admission_planner import OfferDiagnostic, ResourceBlock
        from services.pool_admission import waiting_admission_diagnostics

        details = waiting_admission_diagnostics(1000, "fixture-snapshot", (
            OfferDiagnostic("private-account", resource_blocks=(ResourceBlock(
                "private-resource:fixture-secret", 1, None, 1, None,
                ("occupancy_unknown", "unapproved-blocker")),)),
            OfferDiagnostic("private-account", ineligible_reasons=(
                "account_disabled", "unapproved-check")),
        ))

        resource = details["resource_limits"][0]
        self.assertEqual(resource["family"], "other")
        self.assertEqual(resource["blockers"], ["occupancy_unknown"])
        self.assertEqual(details["account_eligibility"][0]["failed_checks"], ["account_disabled"])
        serialized = json.dumps(details)
        self.assertNotIn("private-account", serialized)
        self.assertNotIn("private-resource", serialized)
        self.assertNotIn("fixture-secret", serialized)

    def test_waiting_diagnostic_projection_is_bounded_after_persistence(self):
        from services.pool_admission import (WAITING_DIAGNOSTIC_MAX_ACCOUNT_ENTRIES,
                                             WAITING_DIAGNOSTIC_MAX_JSON_BYTES,
                                             WAITING_DIAGNOSTIC_MAX_RESOURCE_ENTRIES,
                                             waiting_admission_diagnostics)
        from services.admission_planner import OfferDiagnostic, ResourceBlock

        self.write_accounts(1000)
        for row in self.rows:
            row["managed_disabled"] = True
        (self.root / "accounts.json").write_text(json.dumps(self.rows))
        self.submit("bounded-diagnostics")

        self.assertIsNone(self.admission.claim_next())

        stored = self.read("text", "happy", "bounded-diagnostics")["waiting"]["admission_diagnostics"]
        self.assertEqual(len(stored["account_eligibility"]), WAITING_DIAGNOSTIC_MAX_ACCOUNT_ENTRIES)
        self.assertEqual(stored["omitted_account_eligibility"], 1000 - WAITING_DIAGNOSTIC_MAX_ACCOUNT_ENTRIES)
        self.assertTrue(stored["truncated"])
        self.assertLessEqual(len(json.dumps(stored, ensure_ascii=False, separators=(",", ":")).encode()),
                             WAITING_DIAGNOSTIC_MAX_JSON_BYTES)

        resource_details = waiting_admission_diagnostics(1000, "a" * 32, tuple(
            OfferDiagnostic("account-" + str(index), resource_blocks=(ResourceBlock(
                "chat_turn:resource-" + str(index), 1, 1, 1, 1000,
                ("resource_full",)),))
            for index in range(1000)
        ))
        self.assertEqual(len(resource_details["resource_limits"]), WAITING_DIAGNOSTIC_MAX_RESOURCE_ENTRIES)
        self.assertEqual(resource_details["omitted_resource_limits"], 1000 - WAITING_DIAGNOSTIC_MAX_RESOURCE_ENTRIES)
        self.assertLessEqual(len(json.dumps(resource_details, ensure_ascii=False, separators=(",", ":")).encode()),
                             WAITING_DIAGNOSTIC_MAX_JSON_BYTES)

    def test_slow_image_saves_have_bounded_local_workers_without_holding_generation(self):
        self.admission.settings = lambda: {"chat_account_concurrency": 2, "image_account_concurrency": 1, "codex_max_concurrency": 4}
        for name in ("save-a", "save-b"):
            self.image(name)
            context = self.admission.claim_next()
            context.before_send()
            self.admission.update_claim(context, upstream_unfinished=False, result_file_ids=[name],
                                        request_message_id=name, _upstream_terminal=True)
        self.image("third")
        self.assertIsNone(self.admission.claim_next())
        self.submit("text-still-runs")
        self.assertEqual(self.admission.claim_next().request_id, "text-still-runs")
        resource = self.admission.resource_snapshot()
        self.assertEqual(resource["image"]["inflight"], 0)
        self.assertEqual(resource["image"]["dispatchable_now"], 0)
        self.assertEqual(resource["execution"]["image_workers_active"], 2)
        self.assertEqual(resource["execution"]["image_workers_limit"], 2)
        with self.store.transaction() as db:
            saved = self.store.read_receipt(db, "image", "happy", "save-a")
            saved.update(status="success", _executing=False)
            self.store.write_receipt(db, "image", "happy", "save-a", saved)
        self.assertEqual(self.admission.claim_next().request_id, "third")

    def test_finished_outcome_records_original_once_without_replay(self):
        self.submit("metrics")
        context = self.admission.claim_next()
        self.admission.execute(context)
        context.record_outcome()
        finished = [item for item in context.receipt()["_execution_timeline"] if item["stage"] == "task_finished"]
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["task_status"], "succeeded")
        self.assertEqual(finished[0]["queue_wait_seconds"], 0)
        self.assertEqual(len(self.calls), 1)

    def test_image_artifact_stage_follows_commit_and_does_not_repeat_on_read(self):
        from services.request_context import executing
        self.image("saved-image")
        context = self.admission.claim_next()
        original_record = context.record_stage
        observed = []
        def record(stage, **extra):
            saved = context.receipt()
            self.assertEqual(saved["status"], "success")
            self.assertEqual(saved["data"], [{"b64_json": "cG5n"}])
            observed.append(stage)
            return original_record(stage, **extra)
        context.record_stage = record
        with executing(context):
            self.images._update_task("happy:saved-image", status="success", data=[{"b64_json": "cG5n"}])
            self.images._update_task("happy:saved-image", status="success", data=[{"b64_json": "cG5n"}])
        self.assertEqual(observed, ["artifact_saved"])
        saved = context.receipt()
        event = next(e for e in saved["_execution_timeline"] if e["stage"] == "artifact_saved")
        self.assertEqual(event["image_count"], 1)
        self.assertFalse(any(e["stage"] == "send_call_started" for e in saved["_execution_timeline"]))

    def test_image_artifact_observation_failure_cannot_undo_saved_result(self):
        from services.request_context import executing
        self.image("saved-image")
        context = self.admission.claim_next()
        with patch.object(context, "record_stage", side_effect=RuntimeError("observation unavailable")), executing(context):
            self.images._update_task("happy:saved-image", status="success", data=[{"b64_json": "cG5n"}])
        self.assertEqual(context.receipt()["status"], "success")

    def test_image_artifact_is_not_announced_when_persistence_fails(self):
        from services.request_context import executing
        self.image("saved-image")
        context = self.admission.claim_next()
        with patch.object(context, "record_stage") as record, \
                patch.object(self.images, "_save_locked", side_effect=RuntimeError("storage unavailable")), \
                executing(context), self.assertRaisesRegex(RuntimeError, "storage unavailable"):
            self.images._update_task("happy:saved-image", status="success", data=[{"b64_json": "cG5n"}])
        record.assert_not_called()
        self.assertNotEqual(context.receipt()["status"], "success")

    def test_unsent_capacity_wait_is_not_logged_as_terminal_failure(self):
        self.image("unsent")
        context = self.admission.claim_next()
        def unavailable(ctx, body):
            self.admission.update_claim(ctx, status="error", error_code="IMAGE_RESOURCE_UNAVAILABLE", upstream_unfinished=False)
        self.admission.register("image", unavailable)
        self.admission.execute(context)
        receipt = self.read("image", "happy", "unsent")
        self.assertEqual(receipt["status"], "queued")
        self.assertFalse(any(item["stage"] == "task_finished" for item in receipt["_execution_timeline"]))
        failure = receipt["_execution_timeline"][-1]
        self.assertEqual(failure["stage"], "pre_submit_failure")
        self.assertEqual(failure["error_code"], "IMAGE_RESOURCE_UNAVAILABLE")

    def test_unsent_binding_reason_is_preserved_in_timeline_on_requeue(self):
        self.image("binding-unavailable")
        context = self.admission.claim_next()
        def unavailable(ctx, body):
            self.admission.update_claim(ctx, status="error", error_code="CONVERSATION_BINDING_UNAVAILABLE",
                                        last_recovery_failure={"binding_reason": "image_capacity_stale"})
        self.admission.register("image", unavailable)
        self.admission.execute(context)
        receipt = self.read("image", "happy", "binding-unavailable")
        self.assertEqual(receipt["status"], "queued")
        self.assertFalse(receipt["_submission_started"])
        self.assertEqual(receipt["_execution_timeline"][-1]["binding_reason"], "image_capacity_stale")

    def test_unsent_exception_retains_reason_and_retries_only_original_once(self):
        self.image("prepare-failed")
        context = self.admission.claim_next()
        def fail_before_send(ctx, body):
            self.admission.update_claim(ctx, progress="starting_generation")
            raise RuntimeError("private request body and credential must not be logged")
        self.admission.register("image", fail_before_send)
        with self.assertLogs("chatgpt2api", level="WARNING") as captured:
            self.admission.execute(context)
        receipt = self.read("image", "happy", "prepare-failed")
        self.assertEqual(receipt["status"], "queued")
        self.assertFalse(receipt["_submission_started"])
        failure = receipt["_execution_timeline"][-1]
        self.assertEqual(failure["stage"], "pre_submit_failure")
        self.assertEqual(failure["phase"], "before_submit")
        self.assertEqual(failure["error_type"], "RuntimeError")
        self.assertNotIn("private request body", json.dumps(receipt))
        self.assertNotIn("private request body", "\n".join(captured.output))
        self.assertFalse(any(item["stage"] == "send_call_started" for item in receipt["_execution_timeline"]))
        self.clock.now += 2
        next_context = self.admission.claim_next()
        self.assertEqual(next_context.request_id, "prepare-failed")
        sends = []
        def succeed(ctx, body):
            ctx.before_send()
            ctx.record_stage("send_call_started")
            sends.append(ctx.request_id)
            self.admission.update_claim(ctx, status="success", upstream_unfinished=False)
        self.admission.register("image", succeed)
        self.admission.execute(next_context)
        final = self.read("image", "happy", "prepare-failed")
        self.assertEqual(final["status"], "success")
        self.assertEqual(sends, ["prepare-failed"])
        self.assertEqual(sum(item["stage"] == "pre_submit_failure" for item in final["_execution_timeline"]), 1)
        self.assertEqual(sum(item["stage"] == "send_call_started" for item in final["_execution_timeline"]), 1)

    def test_guard_rejection_after_preparation_records_reason_and_reclaims_original_once(self):
        self.image("prepared-expired")
        context = self.admission.claim_next()
        sends = []
        def expired_after_prepare(ctx, body):
            self.clock.now += self.admission.CLAIM_SECONDS + 1
            ctx.before_send()
            sends.append(ctx.request_id)
        self.admission.register("image", expired_after_prepare)
        with self.assertLogs("chatgpt2api", level="WARNING") as captured:
            self.admission.execute(context)
        receipt = self.read("image", "happy", "prepared-expired")
        self.assertEqual(receipt["status"], "queued")
        self.assertFalse(receipt["_submission_started"])
        self.assertEqual(sends, [])
        failures = [e for e in receipt["_execution_timeline"] if e["stage"] == "pre_submit_failure"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["reason"], "claim_expired")
        self.assertEqual(failures[0]["error_type"], "AdmissionLost")
        self.assertNotIn("private image input", "\n".join(captured.output))
        self.clock.now += 2
        renewed = self.admission.claim_next()
        self.assertEqual(renewed.request_id, context.request_id)
        def succeed(ctx, body):
            ctx.before_send()
            ctx.record_stage("send_call_started")
            sends.append(ctx.request_id)
            self.admission.update_claim(ctx, status="success", upstream_unfinished=False)
        self.admission.register("image", succeed)
        self.admission.execute(renewed)
        final = self.read("image", "happy", "prepared-expired")
        self.assertEqual(sends, ["prepared-expired"])
        self.assertEqual(final["status"], "success")
        self.assertEqual(sum(e["stage"] == "send_call_started" for e in final["_execution_timeline"]), 1)
        self.assertEqual(sum(e["stage"] == "pre_submit_failure" for e in final["_execution_timeline"]), 1)

    def test_guard_distinguishes_account_capability_and_binding_rejections(self):
        cases = (
            ({"managed_disabled": True}, "account_unavailable"),
            ({"capacity_used_since_observation": True}, "image_capability_unavailable"),
            ({"conversation_binding_ids": []}, "image_binding_changed"),
        )
        original = dict(self.rows[0])
        for index, (changes, reason) in enumerate(cases):
            with self.subTest(reason=reason):
                self.rows[0] = dict(original)
                (self.root / "accounts.json").write_text(json.dumps(self.rows))
                request_id = "guard-" + str(index)
                self.image(request_id)
                context = self.admission.claim_next()
                self.assertEqual(context.request_id, request_id)
                self.rows[0].update(changes)
                (self.root / "accounts.json").write_text(json.dumps(self.rows))
                with self.assertLogs("chatgpt2api", level="WARNING") as captured:
                    with self.assertRaises(AdmissionLost):
                        context.before_send()
                receipt = self.read("image", "happy", request_id)
                self.assertEqual(receipt["status"], "queued")
                self.assertFalse(receipt["_submission_started"])
                failures = [e for e in receipt["_execution_timeline"] if e["stage"] == "pre_submit_failure"]
                self.assertEqual(failures[-1]["reason"], reason)
                self.assertNotIn("fixture-token", "\n".join(captured.output))
                self.assertNotIn("private image input", "\n".join(captured.output))

    def test_claim_renewal_uses_writer_acquisition_time_after_sqlite_contention(self):
        self.submit("heartbeat-contention")
        context = self.admission.claim_next()
        with self.store.connect() as db:
            before = self.store.read_receipt(db, "text", "happy", "heartbeat-contention")
        original = {key: before.get(key) for key in (
            "request_hash", "_input_ref", "provider_binding_id", "provider_account_identity",
            "client_conversation_id", "request_message_id",
        )}
        holder = self.store._connect()
        holder.execute("BEGIN IMMEDIATE")
        started, finished, errors = threading.Event(), threading.Event(), []

        def renew():
            started.set()
            try:
                self.admission.renew_claim(context)
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        worker = threading.Thread(target=renew)
        worker.start()
        try:
            self.assertTrue(started.wait(1))
            time.sleep(.05)
            self.assertFalse(finished.is_set(), "renewal must wait for the real SQLite writer")
            self.clock.now += self.admission.CLAIM_SECONDS + 1
            holder.commit()
        finally:
            if holder.in_transaction:
                holder.rollback()
            holder.close()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        with self.store.connect() as db:
            renewed = self.store.read_receipt(db, "text", "happy", "heartbeat-contention")
        self.assertEqual(renewed["_claim_id"], context.claim)
        self.assertEqual(renewed["_claim_until"], self.clock() + self.admission.CLAIM_SECONDS)
        self.assertEqual({key: renewed.get(key) for key in original}, original)
        # The strict send guard sees the fresh writer-time lease. No handler or
        # upstream POST runs as part of a heartbeat renewal.
        context.before_send()
        self.assertEqual(self.calls, [])
        with self.store.connect() as db:
            guarded = self.store.read_receipt(db, "text", "happy", "heartbeat-contention")
        self.assertTrue(guarded["_submission_started"])
        self.assertEqual({key: guarded.get(key) for key in original}, original)

    def test_busy_heartbeat_retries_real_sqlite_timeout_and_allows_exactly_one_send(self):
        self.submit("heartbeat-busy")
        context = self.admission.claim_next()
        holders = []
        busy_seen, renewed = threading.Event(), threading.Event()
        renew = self.admission.renew_claim

        def observed_renew(ctx):
            result = renew(ctx)
            renewed.set()
            return result

        def runner(body, on_cursor):
            # The normal handler has reached its external-call boundary. Hold
            # the writer only now, so this is the real heartbeat contention,
            # rather than blocking initial handler bookkeeping.
            holder = self.store._connect()
            holders.append(holder)
            holder.execute("BEGIN IMMEDIATE")
            try:
                self.assertTrue(busy_seen.wait(2), "real SQLite writer timeout was not observed")
                holder.commit()
                self.assertTrue(renewed.wait(2), "heartbeat must retry after the short writer timeout")
            finally:
                if holder.in_transaction:
                    holder.rollback()
                holder.close()
            current_request.get().before_send()
            self.calls.append(("happy", "heartbeat-busy", body))
            return {"content": "done"}

        def capture(entry):
            if entry.get("stage") == "claim_heartbeat_failed":
                busy_seen.set()

        self.text.runner = runner
        with patch.object(self.admission, "CLAIM_SECONDS", .06), \
                patch.object(self.admission, "CLAIM_HEARTBEAT_WRITER_TIMEOUT_SECS", .01), \
                patch.object(self.admission, "CLAIM_HEARTBEAT_RETRY_SECS", .01), \
                patch.object(self.admission, "renew_claim", side_effect=observed_renew), \
                patch("services.pool_admission.logger.warning", side_effect=capture):
            self.admission.execute(context)
        self.assertTrue(busy_seen.is_set())
        self.assertTrue(renewed.is_set())
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.read("text", "happy", "heartbeat-busy")["status"], "succeeded")

    def test_recovery_heartbeat_retries_after_real_sqlite_timeout_without_sending(self):
        self.image("recovery-heartbeat")
        context = self.admission.claim_next()
        holders = []
        busy_seen, renewed = threading.Event(), threading.Event()
        renew = self.admission.renew_claim

        def observed_renew(ctx):
            result = renew(ctx)
            renewed.set()
            return result

        def recover():
            holder = self.store._connect()
            holders.append(holder)
            holder.execute("BEGIN IMMEDIATE")
            try:
                self.assertTrue(busy_seen.wait(2), "recovery heartbeat did not hit the held writer")
                holder.commit()
                self.assertTrue(renewed.wait(2), "recovery heartbeat did not retry")
            finally:
                if holder.in_transaction:
                    holder.rollback()
                holder.close()

        def capture(entry):
            if entry.get("stage") == "claim_heartbeat_failed":
                busy_seen.set()

        with patch.object(self.admission, "CLAIM_SECONDS", .06), \
                patch.object(self.admission, "CLAIM_HEARTBEAT_WRITER_TIMEOUT_SECS", .01), \
                patch.object(self.admission, "CLAIM_HEARTBEAT_RETRY_SECS", .01), \
                patch.object(self.admission, "renew_claim", side_effect=observed_renew), \
                patch("services.pool_admission.logger.warning", side_effect=capture):
            self.admission.run_recovery(context, recover, ())
        current = self.read("image", "happy", "recovery-heartbeat")
        self.assertTrue(busy_seen.is_set())
        self.assertTrue(renewed.is_set())
        self.assertFalse(current["_executing"])
        self.assertEqual(self.calls, [])

    def test_recovery_replaced_claim_is_fenced_without_sending(self):
        self.image("recovery-fenced")
        context = self.admission.claim_next()
        with self.store.transaction() as db:
            current = self.store.read_receipt(db, "image", "happy", "recovery-fenced")
            original = {key: current.get(key) for key in ("request_hash", "_input_ref", "provider_binding_id")}
            current["_claim_id"] = "other-recovery"
            self.store.write_receipt(db, "image", "happy", "recovery-fenced", current)

        def would_send():
            context.before_send()
            self.calls.append(("happy", "recovery-fenced"))

        with self.assertRaises(AdmissionLost):
            self.admission.run_recovery(context, would_send, ())
        current = self.read("image", "happy", "recovery-fenced")
        self.assertEqual(current["_claim_id"], "other-recovery")
        self.assertFalse(current["_submission_started"])
        self.assertEqual({key: current.get(key) for key in original}, original)
        self.assertEqual(self.calls, [])

    def test_renew_claim_fences_replaced_claim_without_sending(self):
        self.submit("heartbeat-fenced")
        context = self.admission.claim_next()
        with self.store.transaction() as db:
            current = self.store.read_receipt(db, "text", "happy", "heartbeat-fenced")
            original = {key: current.get(key) for key in ("request_hash", "_input_ref", "provider_binding_id")}
            current["_claim_id"] = "other-executor"
            self.store.write_receipt(db, "text", "happy", "heartbeat-fenced", current)
        with self.assertRaises(AdmissionLost):
            self.admission.renew_claim(context)
        with self.assertRaises(AdmissionLost):
            context.before_send()
        with self.store.connect() as db:
            current = self.store.read_receipt(db, "text", "happy", "heartbeat-fenced")
        self.assertEqual(current["_claim_id"], "other-executor")
        self.assertFalse(current["_submission_started"])
        self.assertEqual({key: current.get(key) for key in original}, original)
        self.assertEqual(self.calls, [])

    def test_expired_claim_without_heartbeat_is_reclaimed_without_replaying_input(self):
        self.submit("heartbeat-dead")
        old = self.admission.claim_next()
        with self.store.connect() as db:
            before = self.store.read_receipt(db, "text", "happy", "heartbeat-dead")
        original = {key: before.get(key) for key in ("request_hash", "_input_ref", "provider_binding_id")}
        self.clock.now += self.admission.CLAIM_SECONDS + 1
        renewed = self.admission.claim_next()
        self.assertIsNotNone(renewed)
        self.assertEqual(renewed.request_id, old.request_id)
        self.assertNotEqual(renewed.claim, old.claim)
        with self.assertRaises(AdmissionLost):
            old.before_send()
        with self.store.connect() as db:
            current = self.store.read_receipt(db, "text", "happy", "heartbeat-dead")
        self.assertEqual(current["_claim_id"], renewed.claim)
        self.assertFalse(current["_submission_started"])
        self.assertEqual({key: current.get(key) for key in original}, original)
        self.assertEqual(self.calls, [])

    def test_guard_reason_and_non_transient_heartbeat_diagnostics_do_not_log_exception_text(self):
        from services.pool_admission import record_unsent_failure
        import sqlite3

        self.image("heartbeat")
        context = self.admission.claim_next()
        with patch("services.pool_admission.logger.warning") as logged:
            receipt = {}
            record_unsent_failure(context, receipt, self.clock(), "AdmissionLost", reason="private credential")
            self.assertNotIn("private credential", str(logged.call_args_list))
            self.assertNotIn("reason", receipt["_execution_timeline"][0])
        error = sqlite3.OperationalError("private credential and database path")
        error.sqlite_errorcode = sqlite3.SQLITE_CORRUPT
        failed = threading.Event()
        def reject_renewal(ctx):
            failed.set()
            raise error
        def capture(entry):
            if entry.get("stage") == "claim_heartbeat_failed":
                failed.set()
        def succeed(ctx, body):
            self.assertTrue(failed.wait(2), "heartbeat failure was not observed")
            ctx.before_send()
            ctx.record_stage("send_call_started")
            self.admission.update_claim(ctx, status="success", upstream_unfinished=False)
        self.admission.register("image", succeed)
        with patch.object(self.admission, "CLAIM_SECONDS", .03), \
                patch.object(self.admission, "renew_claim", side_effect=reject_renewal), \
                patch("services.pool_admission.logger.warning", side_effect=capture) as logged:
            self.admission.execute(context)
        entries = [call.args[0] for call in logged.call_args_list
                   if call.args[0].get("stage") == "claim_heartbeat_failed"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["error_type"], "OperationalError")
        self.assertEqual(entries[0]["request_ref"], hashlib.sha256(b"happy:heartbeat").hexdigest()[:24])
        self.assertNotIn("private credential", str(logged.call_args_list))
        self.assertEqual(self.read("image", "happy", "heartbeat")["status"], "success")

    def test_unsent_diagnostic_does_not_attribute_old_attempt_progress(self):
        self.image("stale-progress")
        with self.store.transaction() as db:
            receipt = self.store.read_receipt(db, "image", "happy", "stale-progress")
            receipt["progress"] = "starting_generation"
            self.store.write_receipt(db, "image", "happy", "stale-progress", receipt)
        context = self.admission.claim_next()
        self.admission.register("image", lambda *args: self.fail("handler must not start"))
        with patch.object(self.store, "load_input", side_effect=RuntimeError("before handler")):
            self.admission.execute(context)
        receipt = self.read("image", "happy", "stale-progress")
        self.assertEqual(receipt["status"], "queued")
        self.assertEqual(receipt["_execution_timeline"][-1]["phase"], "before_submit")
        self.assertFalse(receipt["_submission_started"])

    def test_send_hold_skips_exact_original_then_releases_same_input_and_id(self):
        self.submit("held")
        self.submit("independent", owner="other")
        with self.store.connect() as db:
            before = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", ("happy", "held")).fetchone()[0]
        hold = self.store.set_text_send_hold("happy", "held", True, expected_version=0)
        self.assertEqual(hold["state"], "held")
        independent = self.admission.claim_next()
        self.assertEqual(independent.request_id, "independent")
        self.admission.execute(independent)
        self.assertEqual(self.store.set_text_send_hold("happy", "held", False, expected_version=hold["version"])["state"], "active")
        context = self.admission.claim_next()
        self.assertEqual((context.owner, context.request_id), ("happy", "held"))
        with self.store.connect() as db:
            after = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", ("happy", "held")).fetchone()[0]
        self.assertEqual(after, before)
        self.assertFalse(self.read("text", "happy", "held")["_submission_started"])

    def test_send_hold_preserves_same_conversation_order_but_not_independent_work(self):
        self.submit("held")
        self.submit("following")
        self.submit("independent", owner="other")
        with self.store.transaction() as db:
            for request_id, sequence in (("held", 1), ("following", 2)):
                receipt = self.store.read_receipt(db, "text", "happy", request_id)
                receipt.update(provider_account_identity="account-0", provider_binding_id="binding-0",
                               conversation_id="same-conversation", _sequence=sequence)
                self.store.write_receipt(db, "text", "happy", request_id, receipt)
        self.store.set_text_send_hold("happy", "held", True, expected_version=0)
        self.assertEqual(self.admission.claim_next().request_id, "independent")
        self.assertIsNone(self.admission.claim_next())

    def test_send_hold_and_claim_have_deterministic_writer_ordering(self):
        self.submit("hold-first")
        original_write = self.store.write_receipt
        hold_writer, release_hold, claim_done = threading.Event(), threading.Event(), threading.Event()
        hold_result, claim_result = {}, {}

        def hold_first_write(db, kind, owner, request_id, receipt):
            if request_id == "hold-first" and receipt.get("_send_hold") is True:
                hold_writer.set()
                self.assertTrue(release_hold.wait(2))
            return original_write(db, kind, owner, request_id, receipt)

        with patch.object(self.store, "write_receipt", side_effect=hold_first_write):
            holder = threading.Thread(target=lambda: hold_result.setdefault(
                "value", self.store.set_text_send_hold("happy", "hold-first", True, expected_version=0)))
            holder.start()
            self.assertTrue(hold_writer.wait(2))
            claimer = threading.Thread(target=lambda: (claim_result.setdefault("value", self.admission.claim_next()), claim_done.set()))
            claimer.start()
            self.assertFalse(claim_done.wait(.05))
            release_hold.set()
            holder.join(2)
            claimer.join(2)
        self.assertEqual(hold_result["value"]["state"], "held")
        self.assertIsNone(claim_result["value"])

        self.submit("claim-first")
        original_snapshot = self.admission._snapshot
        claim_writer, release_claim, hold_done = threading.Event(), threading.Event(), threading.Event()
        claim_result, hold_result = {}, {}

        def claim_first_snapshot(*args, **kwargs):
            claim_writer.set()
            self.assertTrue(release_claim.wait(2))
            return original_snapshot(*args, **kwargs)

        with patch.object(self.admission, "_snapshot", side_effect=claim_first_snapshot):
            claimer = threading.Thread(target=lambda: claim_result.setdefault("value", self.admission.claim_next()))
            claimer.start()
            self.assertTrue(claim_writer.wait(2))

            def hold_after_claim_writer():
                try:
                    hold_result["value"] = self.store.set_text_send_hold("happy", "claim-first", True, expected_version=0)
                except SendHoldRejected as exc:
                    hold_result["value"] = exc.code
                finally:
                    hold_done.set()

            holder = threading.Thread(target=hold_after_claim_writer)
            holder.start()
            self.assertFalse(hold_done.wait(.05))
            release_claim.set()
            claimer.join(2)
            holder.join(2)
        self.assertIsNotNone(claim_result["value"])
        self.assertEqual(hold_result["value"], "SEND_HOLD_IN_FLIGHT")

    def test_send_hold_final_guard_rejects_a_stale_held_receipt(self):
        self.submit("guard")
        context = self.admission.claim_next()
        with self.store.transaction() as db:
            receipt = self.store.read_receipt(db, "text", "happy", "guard")
            receipt["_send_hold"] = True
            self.store.write_receipt(db, "text", "happy", "guard", receipt)
        with self.assertRaisesRegex(AdmissionLost, "send hold"):
            context.before_send()
        self.assertFalse(self.read("text", "happy", "guard")["_submission_started"])

    def test_send_hold_deadline_expires_without_revival(self):
        self.submit("deadline")
        with self.store.transaction() as db:
            receipt = self.store.read_receipt(db, "text", "happy", "deadline")
            receipt["_scheduling"] = {"wait_deadline": datetime.fromtimestamp(
                self.clock.now + 1, timezone.utc).isoformat()}
            self.store.write_receipt(db, "text", "happy", "deadline", receipt)
        with patch("services.task_store.time.time", return_value=self.clock.now):
            self.store.set_text_send_hold("happy", "deadline", True, expected_version=0)
        self.clock.now += 2
        self.assertIsNone(self.admission.claim_next())
        with patch("services.task_store.time.time", return_value=self.clock.now), \
                self.assertRaisesRegex(SendHoldRejected, "SEND_HOLD_WAIT_DEADLINE_EXCEEDED"):
            self.store.set_text_send_hold("happy", "deadline", False, expected_version=1)
        receipt = self.read("text", "happy", "deadline")
        self.assertEqual((receipt["status"], receipt["error_code"]), ("failed", "WAIT_DEADLINE_EXCEEDED"))


if __name__ == "__main__":
    unittest.main()
