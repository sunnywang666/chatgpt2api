"""First-send compaction is an execution copy, never a retained-input rewrite."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

from services.account_service import AccountService
from services.category_directory_derivation import compact_json
from services.combined_copy_derivation import LEGACY_BOUNDARY, LEGACY_SUFFIX, MODEL, EFFORT
from services.account_request_pacing import AccountRequestClock
from services.config import config
from services.model_service import ModelCatalogService, ModelRoute
from services.pool_admission import PoolAdmission
from services.request_context import AdmissionLost, current_request
from services.storage.json_storage import JSONStorageBackend
from services.task_store import TaskStore
from services.text_task_service import TextTaskService
from test.test_combined_copy_derivation import WBOZ_CURRENT_CASES


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def branches(count=10):
    evidence = {
        "source_attributes": [{"source_attribute_index": 0, "name": "size", "values": ["16"]}],
        "source_images": [{"image_ref": "source-a", "url": "asset://a"}],
        "source_title": "Pearl ring",
        "source_description": "Fixture source",
    }
    return [{
        "preparation_index": index,
        "target_platform": "ozon",
        "category": {"name": "Ring" if index % 2 == 0 else "Brooch", "variant": str(index)},
        "schema": [{"attribute_id": index + 1, "name": "Color", "required": True}],
        **evidence,
    } for index in range(count)]


def queued_copy_body(request_id="queued-copy", *, rows=None):
    rows = rows or branches()
    prompt = "Keep the original copy response contract." + LEGACY_SUFFIX + LEGACY_BOUNDARY + compact_json(rows)
    return {
        "client_request_id": request_id,
        "client_conversation_id": "batch-837-copy",
        "model": MODEL,
        "thinking_effort": EFFORT,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "text", "text": '{"image_ref":"source-a"}'},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDRA=="}},
        ]}],
    }


def queued_json_copy_body(request_id="queued-json-copy", *, rows=None, frozen=None):
    rows = rows or branches()
    frozen = compact_json(rows) if frozen is None else compact_json(frozen)
    envelope = {
        "copy_contract": "fixture-original-contract",
        "category_attribute_instructions": LEGACY_SUFFIX + LEGACY_BOUNDARY + frozen,
    }
    return {
        "client_request_id": request_id,
        "client_conversation_id": "batch-837-json-copy",
        "model": MODEL,
        "thinking_effort": EFFORT,
        "messages": [{"role": "user", "content": [{"type": "text", "text": compact_json(envelope)}]}],
    }


class QueuedCombinedCopyCompactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock = Clock()
        account = {"access_token": "fixture-token", "account_id": "physical", "provider_account_identity": "account-0",
                   "type": "Plus", "status": "正常", "quota": 99, "source_type": "web",
                   "conversation_binding_ids": ["binding-0"],
                   "capacity_observed_at": datetime.now(timezone.utc).isoformat()}
        (self.root / "accounts.json").write_text(json.dumps([account]))
        self.store = TaskStore(self.root / "text_tasks.sqlite3")
        self.accounts = AccountService(JSONStorageBackend(self.root / "accounts.json"))
        self.admission = PoolAdmission(self.store, self.accounts, clock=self.clock,
                                       settings=lambda: {"image_account_concurrency": 4, "codex_max_concurrency": 4},
                                       model_types=lambda _: {"Plus"}, pacing=lambda account, now: {"next_at": now})
        self.sent = []

        def runner(body, on_cursor):
            current_request.get().before_send()
            self.sent.append(deepcopy(body))
            return {"content": "fixture answer", "parent_message_id": "answer"}

        self.text = TextTaskService(self.store.path, runner=runner, admission=self.admission, clock=self.clock)
        self.admission.register("text", lambda context, body: self.text._run(context.owner, context.request_id, body))

    def receipt(self, request_id="queued-copy"):
        with self.store.connect() as db:
            return self.store.read_receipt(db, "text", "admin", request_id)

    def stored_hash(self, request_id="queued-copy"):
        with self.store.connect() as db:
            return db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", ("admin", request_id)).fetchone()[0]

    def submit(self, body=None):
        body = body or queued_copy_body()
        self.text.submit("admin", body, source="internal:listing")
        return body

    def claim(self):
        context = self.admission.claim_next()
        self.assertIsNotNone(context)
        return context

    @staticmethod
    def trial_route():
        return ModelRoute(account_types=frozenset({"Plus"}), account_identities=frozenset({"account-0"}))

    def test_first_send_uses_compact_copy_without_rewriting_original_or_repeating_after_restart(self):
        body = self.submit()
        before = deepcopy(body)
        hash_before = self.stored_hash()
        receipt_before = self.receipt()
        durable_before = self.store.load_input(receipt_before["_input_ref"])

        self.admission.execute(self.claim())

        self.assertEqual(len(self.sent), 1)
        sent = self.sent[0]
        self.assertNotEqual(sent["messages"], durable_before["messages"])
        compact_prompt = sent["messages"][0]["content"][0]["text"]
        evidence = json.loads(compact_prompt.split("\nFrozen evidence sets: ", 1)[1].split("\nFrozen attribute preparations: ", 1)[0])
        preparations = json.loads(compact_prompt.rsplit("\nFrozen attribute preparations: ", 1)[1])
        self.assertEqual(len(evidence), 1)
        self.assertEqual([row["preparation_index"] for row in preparations], list(range(10)))
        self.assertEqual(sent["messages"][0]["content"][1:], durable_before["messages"][0]["content"][1:])

        after = self.receipt()
        self.assertEqual(self.store.load_input(after["_input_ref"]), durable_before)
        self.assertEqual(after["_input_ref"], receipt_before["_input_ref"])
        self.assertEqual(self.stored_hash(), hash_before)
        self.assertEqual(after["request_id"], "queued-copy")
        self.assertEqual(after["provider_account_identity"], "account-0")
        self.assertTrue(any(item.get("stage") == "retained_input_compacted" for item in after["_execution_timeline"]))

        restarted = PoolAdmission(self.store, self.accounts, clock=self.clock,
                                  settings=self.admission.settings, model_types=self.admission.model_types,
                                  pacing=self.admission.pacing)
        restarted.register("text", self.admission.handlers["text"])
        self.assertIsNone(restarted.claim_next())
        self.assertEqual(len(self.sent), 1)

    def test_first_send_compacts_the_observed_json_envelope_without_rewriting_it(self):
        body = queued_json_copy_body(rows=branches(50))
        self.submit(body)
        before = self.store.load_input(self.receipt("queued-json-copy")["_input_ref"])
        before_hash = self.stored_hash("queued-json-copy")

        self.admission.execute(self.claim())

        sent = self.sent[0]
        self.assertNotEqual(sent["messages"], before["messages"])
        envelope = json.loads(sent["messages"][0]["content"][0]["text"])
        compact = envelope["category_attribute_instructions"]
        evidence = json.loads(compact.split("\nFrozen evidence sets: ", 1)[1].split("\nFrozen attribute preparations: ", 1)[0])
        preparations = json.loads(compact.rsplit("\nFrozen attribute preparations: ", 1)[1])
        self.assertEqual(envelope["copy_contract"], "fixture-original-contract")
        self.assertEqual(len(evidence), 1)
        self.assertEqual([row["preparation_index"] for row in preparations], list(range(50)))
        after = self.receipt("queued-json-copy")
        self.assertEqual(self.store.load_input(after["_input_ref"]), before)
        self.assertEqual(self.stored_hash("queued-json-copy"), before_hash)
        self.assertTrue(any(item.get("stage") == "retained_input_compacted" for item in after["_execution_timeline"]))

    def test_nonlegacy_or_noncopy_requests_keep_their_original_execution_body(self):
        cases = (
            ("ordinary", {"model": "fixture-text", "thinking_effort": "standard"}),
            ("wrong-source", {"model": MODEL, "thinking_effort": EFFORT}),
        )
        for request_id, updates in cases:
            with self.subTest(request_id=request_id):
                body = queued_copy_body(request_id)
                body.update(updates)
                source = "key:admin" if request_id == "wrong-source" else "internal:listing"
                self.text.submit("admin", body, source=source)
                self.admission.execute(self.claim())
                sent = self.sent.pop()
                self.assertEqual(sent["messages"], body["messages"])
                self.assertFalse(any(item.get("stage") == "retained_input_compacted"
                                     for item in self.receipt(request_id)["_execution_timeline"]))

    def test_new_json_compact_envelope_passes_through_unchanged(self):
        body = queued_json_copy_body("new-json", frozen={
            "preparations": [{"preparation_index": 0, "branch_set_index": 0}],
            "branch_sets": [{"branch_set_index": 0}],
            "evidence_sets": [{"evidence_set_index": 0}],
        })
        self.submit(body)
        self.admission.execute(self.claim())

        self.assertEqual(self.sent[0]["messages"], body["messages"])
        self.assertFalse(any(item.get("stage") == "retained_input_compacted"
                             for item in self.receipt("new-json")["_execution_timeline"]))

    def test_current_or_future_json_compact_input_first_sends_byte_for_byte(self):
        cases = list(WBOZ_CURRENT_CASES)
        future = deepcopy(WBOZ_CURRENT_CASES[0])
        future["name"] = "future-prompt-wording"
        payload = json.loads(future["input"]["messages"][0]["content"][0]["text"])
        payload["category_attribute_instructions"] = payload["category_attribute_instructions"].replace(
            "Return no row", "Return no category row", 1)
        future["input"]["messages"][0]["content"][0]["text"] = compact_json(payload)
        cases.append(future)
        for case in cases:
            with self.subTest(case=case["name"]):
                request_id = "current-json-" + case["name"]
                body = queued_copy_body(request_id)
                body.update(deepcopy(case["input"]), client_request_id=request_id)
                self.submit(body)
                before = self.store.load_input(self.receipt(request_id)["_input_ref"])
                hash_before = self.stored_hash(request_id)

                self.admission.execute(self.claim())

                after = self.receipt(request_id)
                self.assertEqual(self.sent[-1]["messages"], before["messages"])
                self.assertEqual(self.store.load_input(after["_input_ref"]), before)
                self.assertEqual(self.stored_hash(request_id), hash_before)
                self.assertFalse(any(item.get("stage") == "retained_input_compacted"
                                     for item in after["_execution_timeline"]))

    def test_corrupt_current_json_reference_is_not_an_unsolicited_first_send_transform(self):
        body = queued_copy_body("corrupt-current-json")
        payload = json.loads(WBOZ_CURRENT_CASES[0]["input"]["messages"][0]["content"][0]["text"])
        frozen = json.loads(payload["category_attribute_instructions"].split("\nFrozen attribute preparations: ", 1)[1])
        frozen["branch_sets"][0]["reply_preparation_index"] = 1
        payload["category_attribute_instructions"] = payload["category_attribute_instructions"].split(
            "\nFrozen attribute preparations: ", 1)[0] + "\nFrozen attribute preparations: " + compact_json(frozen)
        body["messages"][0]["content"][0]["text"] = compact_json(payload)
        self.submit(body)

        self.admission.execute(self.claim())

        sent = self.receipt("corrupt-current-json")
        self.assertEqual(self.sent[0]["messages"], body["messages"])
        self.assertNotEqual(sent["status"], "failed")
        self.assertFalse(any(item.get("stage") == "retained_input_compacted"
                             for item in sent["_execution_timeline"]))

    def test_current_compact_input_is_excluded_from_temporary_firstsend_trial(self):
        body = queued_copy_body("current-trial")
        body.update(deepcopy(WBOZ_CURRENT_CASES[0]["input"]), client_request_id="current-trial",
                    product_work={"product_ref": "product-current", "step_ref": "copy"})
        self.submit(body)
        self.admission.settings = lambda: {
            "revision": 7, "image_account_concurrency": 4, "codex_max_concurrency": 4,
            "temporary_firstsend_model_trial": {
                "account_identity": "account-0", "request_id": "current-trial", "expires_at": 1100,
            },
        }

        with patch.object(self.admission, "_fresh_bound_firstsend_model_capability") as fresh:
            self.admission.execute(self.claim())

        fresh.assert_not_called()
        self.assertEqual((self.sent[0]["model"], self.sent[0]["thinking_effort"]), (MODEL, EFFORT))
        self.assertNotIn("effective_model", self.receipt("current-trial").get("_temporary_firstsend_model_trial") or {})

    def test_existing_send_marker_and_lost_claim_do_not_receive_compaction(self):
        self.submit()
        context = self.claim()
        with self.store.transaction() as db:
            row = self.store.read_receipt(db, "text", "admin", "queued-copy")
            row["_execution_timeline"].append({"stage": "send_call_started", "at": self.clock()})
            self.store.write_receipt(db, "text", "admin", "queued-copy", row)
        # The normal send guard remains authoritative for the old marker.  The
        # compactor adds no record and a conflicting queued state becomes an
        # exact-ID UNKNOWN rather than another model POST.
        self.admission.execute(context)
        marked = self.receipt()
        self.assertEqual(marked["status"], "unknown")
        self.assertEqual(marked["upstream_outcome"], "unknown")
        self.assertFalse(any(item.get("stage") == "retained_input_compacted" for item in marked["_execution_timeline"]))
        self.assertEqual(self.sent, [])

    def test_lost_claim_does_not_receive_compaction(self):
        self.submit(queued_copy_body("lost-claim"))
        context = self.claim()
        with self.store.transaction() as db:
            row = self.store.read_receipt(db, "text", "admin", "lost-claim")
            row.update(status="failed", _claim_id=None, _claim_until=0, _executing=False)
            self.store.write_receipt(db, "text", "admin", "lost-claim", row)
        self.admission.execute(context)
        lost = self.receipt("lost-claim")
        self.assertFalse(any(item.get("stage") == "retained_input_compacted" for item in lost["_execution_timeline"]))

    def test_recognized_malformed_legacy_input_fails_once_without_a_model_send(self):
        body = queued_copy_body()
        body["messages"][0]["content"][0]["text"] = body["messages"][0]["content"][0]["text"].replace(
            '"preparation_index":1', '"preparation_index":9', 1)
        self.submit(body)
        self.admission.execute(self.claim())

        failed = self.receipt()
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["upstream_outcome"], "not_sent")
        self.assertFalse(failed["_submission_started"])
        self.assertEqual(self.sent, [])
        self.assertIsNone(self.admission.claim_next())

    def test_overflowing_json_number_stops_before_send_without_requeue(self):
        body = queued_json_copy_body()
        text = body["messages"][0]["content"][0]["text"]
        body["messages"][0]["content"][0]["text"] = '{"opaque_number":1e999,' + text[1:]
        self.submit(body)
        self.admission.execute(self.claim())

        failed = self.receipt("queued-json-copy")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["upstream_outcome"], "not_sent")
        self.assertEqual(failed["error_code"], "CHAT_DERIVED_INPUT_UNSUPPORTED")
        self.assertEqual(self.sent, [])
        self.assertIsNone(self.admission.claim_next())

    def test_invalid_unicode_envelope_stops_before_send_without_requeue(self):
        body = queued_json_copy_body()
        text = body["messages"][0]["content"][0]["text"]
        body["messages"][0]["content"][0]["text"] = '{"opaque_text":"\\ud800",' + text[1:]
        self.submit(body)
        self.admission.execute(self.claim())

        failed = self.receipt("queued-json-copy")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["upstream_outcome"], "not_sent")
        self.assertEqual(failed["error_code"], "CHAT_DERIVED_INPUT_UNSUPPORTED")
        self.assertEqual(self.sent, [])
        self.assertIsNone(self.admission.claim_next())

    def test_helper_size_limit_keeps_the_valid_original_first_send(self):
        body = queued_copy_body()
        prompt = body["messages"][0]["content"][0]["text"]
        body["messages"][0]["content"][0]["text"] = "x" * 70_000 + prompt
        self.submit(body)
        self.admission.execute(self.claim())

        self.assertEqual(self.sent[0]["messages"], body["messages"])
        self.assertFalse(any(item.get("stage") == "retained_input_compacted"
                             for item in self.receipt()["_execution_timeline"]))

    def test_equal_or_larger_compactor_output_keeps_the_original_first_send(self):
        body = self.submit()
        larger = deepcopy(body["messages"])
        larger[0]["content"].append({"type": "text", "text": "unneeded expanded execution copy"})
        proof = {"kind": "combined_copy_compact_v1", "preparation_count": 10,
                 "evidence_set_count": 1, "image_count": 1}
        with patch("services.combined_copy_derivation.compact_retained_messages", return_value=(larger, proof)):
            self.admission.execute(self.claim())

        self.assertEqual(self.sent[0]["messages"], body["messages"])
        self.assertFalse(any(item.get("stage") == "retained_input_compacted"
                             for item in self.receipt()["_execution_timeline"]))

    def test_exact_unsent_product_original_uses_in_memory_instant_trial_after_fresh_capability_check(self):
        body = queued_copy_body()
        body["product_work"] = {"product_ref": "product-837", "step_ref": "copy"}
        self.submit(body)
        before = self.store.load_input(self.receipt()["_input_ref"])
        self.admission.settings = lambda: {
            "revision": 7, "image_account_concurrency": 4, "codex_max_concurrency": 4,
            "temporary_firstsend_model_trial": {
                "account_identity": "account-0", "request_id": "queued-copy", "expires_at": 1100,
            },
        }

        with patch.object(self.admission, "_fresh_bound_firstsend_model_capability") as fresh, \
             patch("services.model_service.model_catalog_service.prepared_bound_model_route_at_send",
                   return_value=self.trial_route()):
            self.admission.execute(self.claim())

        fresh.assert_called_once()
        self.assertEqual(self.sent[0]["model"], "gpt-5-6-instant")
        self.assertEqual(self.sent[0]["thinking_effort"], "none")
        after = self.receipt()
        self.assertEqual(after["model"], MODEL)
        self.assertEqual(self.store.load_input(after["_input_ref"]), before)
        self.assertEqual(after["_temporary_firstsend_model_trial"], {
            "account_identity": "account-0", "request_id": "queued-copy", "expires_at": 1100,
            "revision": 7, "original_model": MODEL, "original_thinking_effort": EFFORT,
            "effective_model": "gpt-5-6-instant", "effective_thinking_effort": "none",
            "sent_sequence": 0, "sent_at": 1000.0,
        })
        audit = [entry for entry in after["_execution_timeline"]
                 if entry.get("stage") == "first_send_model_trial_selected"]
        self.assertEqual(audit, [{"stage": "first_send_model_trial_selected", "at": 1000.0,
                                  "original_model": MODEL, "original_thinking_effort": EFFORT,
                                  "effective_model": "gpt-5-6-instant", "effective_thinking_effort": "none"}])

    def test_exact_batch_trial_selects_each_target_without_a_per_account_limit(self):
        bodies = []
        for request_id in ("batch-a", "batch-b", "ordinary-c"):
            body = queued_copy_body(request_id)
            body["client_conversation_id"] = "conversation-" + request_id
            body["product_work"] = {"product_ref": "product-" + request_id, "step_ref": "copy"}
            bodies.append(self.submit(body))
        self.admission.settings = lambda: {
            # Three independent product parents are intentional here.  This
            # verifies one account may select each exact first-send trial
            # target; it must not accidentally exercise the separate product
            # admission cap with the fixture's default one Chat turn.
            "revision": 7, "image_account_concurrency": 4, "codex_max_concurrency": 4,
            "chat_account_concurrency": 3,
            "temporary_firstsend_model_trial": {"targets": [
                {"account_identity": "account-0", "request_id": "batch-a"},
                {"account_identity": "account-0", "request_id": "batch-b"},
            ], "expires_at": 1100},
        }

        with patch.object(self.admission, "_fresh_bound_firstsend_model_capability") as fresh, \
             patch("services.model_service.model_catalog_service.prepared_bound_model_route_at_send",
                   return_value=self.trial_route()):
            self.admission.execute(self.claim())
            self.admission.execute(self.claim())
            self.admission.execute(self.claim())

        self.assertEqual(fresh.call_count, 2)
        self.assertEqual([(body["client_request_id"], body["model"], body["thinking_effort"]) for body in self.sent], [
            ("batch-a", "gpt-5-6-instant", "none"),
            ("batch-b", "gpt-5-6-instant", "none"),
            ("ordinary-c", MODEL, EFFORT),
        ])
        for request_id, body in zip(("batch-a", "batch-b"), bodies[:2]):
            after = self.receipt(request_id)
            self.assertEqual(after["model"], MODEL)
            self.assertEqual(self.store.load_input(after["_input_ref"]), body)
            self.assertEqual(after["_temporary_firstsend_model_trial"]["request_id"], request_id)
        self.assertNotIn("effective_model", self.receipt("ordinary-c").get("_temporary_firstsend_model_trial") or {})

    def test_expired_batch_trial_leaves_an_unsent_original_on_its_normal_model_path(self):
        body = queued_copy_body()
        body["product_work"] = {"product_ref": "product-837", "step_ref": "copy"}
        self.submit(body)
        self.admission.settings = lambda: {
            "revision": 7, "image_account_concurrency": 4, "codex_max_concurrency": 4,
            "temporary_firstsend_model_trial": {"targets": [
                {"account_identity": "account-0", "request_id": "queued-copy"},
            ], "expires_at": 1000},
        }

        with patch.object(self.admission, "_fresh_bound_firstsend_model_capability") as fresh:
            self.admission.execute(self.claim())

        fresh.assert_not_called()
        self.assertEqual(self.sent[0]["model"], MODEL)
        self.assertNotIn("effective_model", self.receipt().get("_temporary_firstsend_model_trial") or {})

    def test_missing_fresh_candidate_capability_requeues_without_changing_original(self):
        body = queued_copy_body()
        body["product_work"] = {"product_ref": "product-837", "step_ref": "copy"}
        self.submit(body)
        before = self.store.load_input(self.receipt()["_input_ref"])
        self.admission.settings = lambda: {
            "revision": 7, "image_account_concurrency": 4, "codex_max_concurrency": 4,
            "temporary_firstsend_model_trial": {
                "account_identity": "account-0", "request_id": "queued-copy", "expires_at": 1100,
            },
        }

        with patch.object(self.admission, "_fresh_bound_firstsend_model_capability",
                          side_effect=AdmissionLost("first-send model is unavailable before send")):
            self.admission.execute(self.claim())

        after = self.receipt()
        self.assertEqual(self.sent, [])
        self.assertEqual(after["status"], "queued")
        self.assertEqual(after["model"], MODEL)
        self.assertEqual(self.store.load_input(after["_input_ref"]), before)
        self.assertFalse(any(entry.get("stage") == "first_send_model_trial_selected"
                             for entry in after["_execution_timeline"]))

    def test_revoked_or_expired_trial_leaves_the_unsent_original_on_its_normal_model_path(self):
        body = queued_copy_body()
        body["product_work"] = {"product_ref": "product-837", "step_ref": "copy"}
        self.submit(body)
        active = {
            "revision": 7, "image_account_concurrency": 4, "codex_max_concurrency": 4,
            "temporary_firstsend_model_trial": {
                "account_identity": "account-0", "request_id": "queued-copy", "expires_at": 1100,
            },
        }
        self.admission.settings = lambda: active
        context = self.claim()
        # The control is cleared after claim but before execution.  No model
        # substitution survives its exact-ID/revision check.
        self.admission.settings = lambda: {"revision": 8, "image_account_concurrency": 4,
                                           "codex_max_concurrency": 4}
        self.admission.execute(context)

        self.assertEqual(self.sent[0]["model"], MODEL)
        self.assertEqual(self.sent[0]["thinking_effort"], EFFORT)
        self.assertFalse(any(entry.get("stage") == "first_send_model_trial_selected"
                             for entry in self.receipt()["_execution_timeline"]))

    def test_trial_revocation_after_capability_read_requeues_without_a_post(self):
        body = queued_copy_body()
        body["product_work"] = {"product_ref": "product-837", "step_ref": "copy"}
        self.submit(body)
        active = {
            "revision": 7, "image_account_concurrency": 4, "codex_max_concurrency": 4,
            "temporary_firstsend_model_trial": {
                "account_identity": "account-0", "request_id": "queued-copy", "expires_at": 1100,
            },
        }
        self.admission.settings = lambda: active
        context = self.claim()

        def revoke_after_read(*_):
            active.clear()
            active.update({"revision": 8, "image_account_concurrency": 4,
                           "codex_max_concurrency": 4})

        with patch.object(self.admission, "_fresh_bound_firstsend_model_capability",
                          side_effect=revoke_after_read), \
             patch("services.model_service.model_catalog_service.prepared_bound_model_route_at_send",
                   return_value=self.trial_route()):
            self.admission.execute(context)

        after = self.receipt()
        self.assertEqual(self.sent, [])
        self.assertEqual(after["status"], "queued")
        self.assertFalse(after["_submission_started"])
        self.assertFalse(any(item.get("stage") == "first_send_model_trial_selected"
                             for item in after["_execution_timeline"]))

    def test_trial_catalog_prepare_can_use_the_same_http_clock_before_the_post(self):
        """The catalog GET must run before, never inside, the POST clock lock."""
        body = queued_copy_body()
        body["product_work"] = {"product_ref": "product-837", "step_ref": "copy"}
        self.submit(body)
        self.admission.settings = lambda: {
            "revision": 7, "image_account_concurrency": 4, "codex_max_concurrency": 4,
            "temporary_firstsend_model_trial": {
                "account_identity": "account-0", "request_id": "queued-copy", "expires_at": 1100,
            },
        }
        pace = AccountRequestClock("account-0", self.root / "trial-clock.json")
        calls = []

        def send(method, _url, **_kwargs):
            calls.append(method)
            return SimpleNamespace(status_code=200, headers={}, close=lambda: None)

        class PacingCatalogBackend:
            def __init__(self, *, access_token=""):
                self.access_token = access_token

            def list_models(self):
                # A regression that calls the real catalog path from
                # ``before_send`` holds this same clock and cannot reach POST.
                pace.request(send, "GET", "https://fixture/backend-api/models",
                             _account_request_deadline_monotonic=time.monotonic() + .2)
                return {"object": "list", "data": [{"id": "gpt-5-6-instant"}]}

            def close(self):
                pass

        catalog = ModelCatalogService(
            self.accounts, backend_factory=PacingCatalogBackend,
            cache_ttl_seconds=300,
        )

        def runner(body, on_cursor):
            pace.request(send, "POST", "https://fixture/backend-api/conversation", json=body,
                         _account_request_deadline_monotonic=time.monotonic() + 1)
            # The pacing response is deliberately minimal.  TextTaskService
            # still needs its normal terminal runner mapping after the POST.
            return {"content": "fixture answer", "parent_message_id": "answer"}

        self.text.runner = runner
        with patch.object(type(config), "account_request_interval_secs", new_callable=PropertyMock, return_value=0), \
             patch.object(type(config), "account_message_interval_secs", new_callable=PropertyMock, return_value=0), \
             patch.object(self.accounts, "refresh_access_token", side_effect=lambda token, **_kwargs: token), \
             patch("services.model_service.model_catalog_service", catalog):
            self.admission.execute(self.claim())

        self.assertEqual(calls, ["GET", "POST"])
        self.assertTrue(self.receipt()["_submission_started"])

    def test_trial_with_expired_prepared_catalog_proof_requeues_without_a_post(self):
        body = queued_copy_body()
        body["product_work"] = {"product_ref": "product-837", "step_ref": "copy"}
        self.submit(body)
        self.admission.settings = lambda: {
            "revision": 7, "image_account_concurrency": 4, "codex_max_concurrency": 4,
            "temporary_firstsend_model_trial": {
                "account_identity": "account-0", "request_id": "queued-copy", "expires_at": 1100,
            },
        }
        with patch.object(self.admission, "_fresh_bound_firstsend_model_capability"), \
             patch("services.model_service.model_catalog_service.prepared_bound_model_route_at_send",
                   side_effect=[self.trial_route(), None]):
            self.admission.execute(self.claim())

        after = self.receipt()
        self.assertEqual(self.sent, [])
        self.assertEqual(after["status"], "queued")
        self.assertFalse(after["_submission_started"])

    def test_trial_control_expiring_after_catalog_prepare_requeues_without_a_post(self):
        """A clock wait may outlive an armed control, so final guard rechecks it."""
        body = queued_copy_body()
        body["product_work"] = {"product_ref": "product-837", "step_ref": "copy"}
        self.submit(body)
        self.admission.settings = lambda: {
            "revision": 7, "image_account_concurrency": 4, "codex_max_concurrency": 4,
            "temporary_firstsend_model_trial": {
                "account_identity": "account-0", "request_id": "queued-copy", "expires_at": 1010,
            },
        }

        def expire_after_prepare(*_):
            self.clock.now = 1010

        context = self.claim()
        with patch.object(self.admission, "_fresh_bound_firstsend_model_capability",
                          side_effect=expire_after_prepare), \
             patch("services.model_service.model_catalog_service.prepared_bound_model_route_at_send",
                   return_value=self.trial_route()):
            self.admission.execute(context)

        self.assertIsNotNone(context._prepared_firstsend_model_route)
        after = self.receipt()
        self.assertEqual(self.sent, [])
        self.assertEqual(after["status"], "queued")
        self.assertFalse(after["_submission_started"])

    def test_trial_credential_rotation_after_catalog_prepare_requeues_without_a_post(self):
        """A new bound credential invalidates the prepared exact-account proof."""
        body = queued_copy_body()
        body["product_work"] = {"product_ref": "product-837", "step_ref": "copy"}
        self.submit(body)
        self.admission.settings = lambda: {
            "revision": 7, "image_account_concurrency": 4, "codex_max_concurrency": 4,
            "temporary_firstsend_model_trial": {
                "account_identity": "account-0", "request_id": "queued-copy", "expires_at": 1100,
            },
        }

        class CatalogBackend:
            def __init__(self, *, access_token=""):
                self.access_token = access_token

            def list_models(self):
                return {"object": "list", "data": [{"id": "gpt-5-6-instant"}]}

            def close(self):
                pass

        catalog = ModelCatalogService(self.accounts, backend_factory=CatalogBackend,
                                      cache_ttl_seconds=300, clock=lambda: self.clock.now)
        prepared = catalog.prepared_bound_model_route_at_send
        calls = 0

        def rotate_after_prepare(identity, model):
            nonlocal calls
            calls += 1
            route = prepared(identity, model)
            if calls == 1:
                self.accounts._apply_refreshed_tokens(
                    "fixture-token", {"access_token": "rotated-fixture-token"}, "test_rotation")
            return route

        with patch.object(self.accounts, "refresh_access_token", side_effect=lambda token, **_kwargs: token), \
             patch("services.model_service.model_catalog_service", catalog), \
             patch.object(catalog, "prepared_bound_model_route_at_send", side_effect=rotate_after_prepare):
            self.admission.execute(self.claim())

        after = self.receipt()
        self.assertEqual(calls, 2)
        self.assertEqual(self.sent, [])
        self.assertEqual(after["status"], "queued")
        self.assertFalse(after["_submission_started"])

    def test_durable_trial_audit_survives_timeline_truncation(self):
        body = queued_copy_body()
        body["product_work"] = {"product_ref": "product-837", "step_ref": "copy"}
        self.submit(body)
        with self.store.transaction() as db:
            receipt = self.store.read_receipt(db, "text", "admin", "queued-copy")
            receipt["_execution_timeline"] = [{"stage": "prior", "at": float(i)} for i in range(32)]
            self.store.write_receipt(db, "text", "admin", "queued-copy", receipt)
        self.admission.settings = lambda: {
            "revision": 7, "image_account_concurrency": 4, "codex_max_concurrency": 4,
            "temporary_firstsend_model_trial": {
                "account_identity": "account-0", "request_id": "queued-copy", "expires_at": 1100,
            },
        }
        with patch.object(self.admission, "_fresh_bound_firstsend_model_capability"), \
             patch("services.model_service.model_catalog_service.prepared_bound_model_route_at_send",
                   return_value=self.trial_route()):
            self.admission.execute(self.claim())
        with self.store.transaction() as db:
            after = self.store.read_receipt(db, "text", "admin", "queued-copy")
            after["_execution_timeline"] = [{"stage": "later", "at": float(i)} for i in range(40)][-32:]
            self.store.write_receipt(db, "text", "admin", "queued-copy", after)
        after = self.receipt()
        self.assertNotIn("first_send_model_trial_selected", [item.get("stage") for item in after["_execution_timeline"]])
        self.assertEqual(after["_temporary_firstsend_model_trial"]["effective_model"], "gpt-5-6-instant")
        self.assertEqual(after["_temporary_firstsend_model_trial"]["sent_sequence"], 0)

    def test_trial_configuration_cannot_claim_sent_unknown_or_succeeded_original(self):
        for status, outcome in (("queued", "unknown"), ("running", "unknown"), ("unknown", "unknown"), ("succeeded", "completed")):
            with self.subTest(status=status):
                body = queued_copy_body()
                body["client_request_id"] = "queued-copy-" + status
                body["product_work"] = {"product_ref": "product-837", "step_ref": status}
                self.submit(body)
                with self.store.transaction() as db:
                    receipt = self.store.read_receipt(db, "text", "admin", body["client_request_id"])
                    receipt.update(status=status, upstream_outcome=outcome, _submission_started=True,
                                   _last_sent_sequence=0, _executing=False, _claim_id=None, _claim_until=0)
                    self.store.write_receipt(db, "text", "admin", body["client_request_id"], receipt)
                self.admission.settings = lambda: {
                    "revision": 7, "image_account_concurrency": 4, "codex_max_concurrency": 4,
                    "temporary_firstsend_model_trial": {
                        "account_identity": "account-0", "request_id": body["client_request_id"], "expires_at": 1100,
                    },
                }
                self.assertIsNone(self.admission.claim_next())
                after = self.receipt(body["client_request_id"])
                if status == "queued":
                    self.assertEqual(after["status"], "unknown")
                    self.assertEqual(after["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
                self.assertNotIn("effective_model", after.get("_temporary_firstsend_model_trial") or {})
        self.assertEqual(self.sent, [])

    def test_nonlegacy_queued_text_send_fence_keeps_existing_claim_behavior(self):
        body = {"client_request_id": "nonlegacy-sent", "client_conversation_id": "other",
                "model": "fixture-text", "messages": [{"role": "user", "content": "original"}]}
        self.text.submit("admin", body, source="ordinary")
        with self.store.transaction() as db:
            receipt = self.store.read_receipt(db, "text", "admin", "nonlegacy-sent")
            receipt.update(status="queued", _submission_started=True, _last_sent_sequence=0,
                           upstream_outcome="unknown", _executing=False, _claim_id=None, _claim_until=0)
            self.store.write_receipt(db, "text", "admin", "nonlegacy-sent", receipt)
        context = self.admission.claim_next()
        self.assertIsNotNone(context)
        self.assertEqual(context.request_id, "nonlegacy-sent")
        self.assertEqual(self.receipt("nonlegacy-sent")["status"], "running")
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main()
