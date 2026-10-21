"""First-send compaction is an execution copy, never a retained-input rewrite."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from services.account_service import AccountService
from services.category_directory_derivation import compact_json
from services.combined_copy_derivation import LEGACY_BOUNDARY, LEGACY_SUFFIX, MODEL, EFFORT
from services.pool_admission import PoolAdmission
from services.request_context import current_request
from services.storage.json_storage import JSONStorageBackend
from services.task_store import TaskStore
from services.text_task_service import TextTaskService


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

    def test_nonlegacy_or_noncopy_requests_keep_their_original_execution_body(self):
        cases = (
            ("ordinary", {"model": "fixture-text", "thinking_effort": "standard"}),
            ("wrong-source", {"model": MODEL, "thinking_effort": EFFORT}),
            ("already-compact", {"model": MODEL, "thinking_effort": EFFORT}),
        )
        for request_id, updates in cases:
            with self.subTest(request_id=request_id):
                body = queued_copy_body(request_id)
                body.update(updates)
                if request_id == "already-compact":
                    original = body["messages"][0]["content"][0]["text"]
                    body["messages"][0]["content"][0]["text"] = original.replace(
                        "Each preparation_index identifies one frozen store/variant/category schema below.",
                        "Each preparation_index points to one branch_set_index.",
                        1,
                    )
                source = "key:admin" if request_id == "wrong-source" else "internal:listing"
                self.text.submit("admin", body, source=source)
                self.admission.execute(self.claim())
                sent = self.sent.pop()
                self.assertEqual(sent["messages"], body["messages"])
                self.assertFalse(any(item.get("stage") == "retained_input_compacted"
                                     for item in self.receipt(request_id)["_execution_timeline"]))

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


if __name__ == "__main__":
    unittest.main()
