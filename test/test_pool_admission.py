"""Independent data and fake time; these are not production capacity samples."""
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from services.account_service import AccountService
from services.storage.json_storage import JSONStorageBackend
from services.task_store import TaskStore
from services.pool_admission import PoolAdmission
from services.request_context import current_request, AdmissionLost
from services.text_task_service import TextTaskService
from services.image_task_service import ImageTaskService


class Clock:
    def __init__(self):
        self.now = 1000.0
    def __call__(self):
        return self.now


def build(root, clock=None):
    root = Path(root)
    accounts = AccountService(JSONStorageBackend(root / "accounts.json"))
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
                      "quota": 999, "source_type": "web", "conversation_binding_ids": ["binding-" + str(i)]}
                     for i in range(count)]
        (self.root / "accounts.json").write_text(json.dumps(self.rows))

    def submit(self, name, owner="happy", source=None, **values):
        body = {"client_request_id": name, "client_conversation_id": "session-" + name,
                "model": "fixture-text", "messages": [{"role": "user", "content": "private original input"}], **values}
        return self.text.submit(owner, body, source=source)

    def image(self, name, owner="happy"):
        return self.images.submit_generation({"id": owner, "role": "user", "external_image_client": True},
                                             client_task_id=name, prompt="private image input", model="gpt-image-2", size=None)

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

    def read(self, kind, owner, name):
        with self.store.connect() as db:
            return self.store.read_receipt(db, kind, owner, name)

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

    def test_disable_restore_and_quota_restore_wake_original_image(self):
        self.rows[0]["managed_disabled"] = True
        (self.root / "accounts.json").write_text(json.dumps(self.rows))
        self.image("image")
        self.assertIsNone(self.admission.claim_next())
        self.rows[0].update(managed_disabled=False, quota=0)
        (self.root / "accounts.json").write_text(json.dumps(self.rows))
        self.assertIsNone(self.admission.claim_next())
        self.rows[0]["quota"] = 999
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


if __name__ == "__main__":
    unittest.main()
