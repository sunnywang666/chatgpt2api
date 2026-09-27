"""Independent data and fake time; these are not production capacity samples."""
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
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

    def terminal_empty_correction_pair(self):
        session = "same-public-session"
        self.submit("original", client_conversation_id=session,
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
        self.submit("correction", client_conversation_id=session,
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
        self.assertEqual(self.admission.claim_next().request_id, "next")
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


if __name__ == "__main__":
    unittest.main()
