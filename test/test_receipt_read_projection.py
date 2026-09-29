"""Synthetic receipt-only regressions; no provider calls or production data."""
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from services.image_task_service import ImageTaskService, _public_task
from services.image_thread import PROTOCOL, bind_waiting_threads, source_fingerprint
from test.test_pool_admission import build


class ReceiptReadProjectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "accounts.json").write_text("[]")
        self.accounts, self.store, self.admission = build(self.root)
        self.images = ImageTaskService(self.root / "images.json", store=self.store,
                                       admission=self.admission, retention_days_getter=lambda: 30)
        self.large = "UNREQUESTED_OUTPUT_SENTINEL" + "A" * (1024 * 1024)

    def receipt(self, name, **values):
        return {"id": name, "owner_id": "owner", "status": "success", "updated_at": "2099-01-01 00:00:00",
                "retain_receipt": True, "model": "gpt-image-2", **values}

    def put(self, task, kind="image"):
        with self.store.transaction() as db:
            if kind == "text":
                db.execute("INSERT OR IGNORE INTO requests VALUES(?,?,?,?)",
                           (task["owner_id"], task["id"], "fixture", "{}"))
            self.store.write_receipt(db, kind, task["owner_id"], task["id"], task)

    def read(self, name, kind="image"):
        with self.store.connect() as db:
            return self.store.read_receipt(db, kind, "owner", name)

    @contextmanager
    def no_large_sql_results(self):
        """Catch payloads as they cross SQLite's boundary, before JSON decoding."""
        connect = self.store._connect
        forbidden = "UNREQUESTED_OUTPUT_SENTINEL"
        class Cursor:
            def __init__(self, cursor): self.cursor = cursor
            def check(self, row):
                if row is not None:
                    for value in row:
                        if isinstance(value, str) and forbidden in value:
                            raise AssertionError("unrequested output crossed SQLite/Python boundary")
                return row
            def __iter__(self):
                return (self.check(row) for row in self.cursor)
            def fetchone(self): return self.check(self.cursor.fetchone())
            def __getattr__(self, name): return getattr(self.cursor, name)
        class Connection:
            def __init__(self, db): self.db = db
            def execute(self, *args): return Cursor(self.db.execute(*args))
            def __enter__(self): self.db.__enter__(); return self
            def __exit__(self, *args): return self.db.__exit__(*args)
            def __getattr__(self, name): return getattr(self.db, name)
        with patch.object(self.store, "_connect", side_effect=lambda: Connection(connect())):
            yield

    def test_projection_preserves_absence_null_booleans_numbers_and_nested_types(self):
        for number, fields in enumerate(({}, {"upstream_unfinished": None}, {"upstream_unfinished": False},
                                         {"upstream_unfinished": True}, {"upstream_unfinished": 0},
                                         {"upstream_unfinished": 1})):
            self.put(self.receipt(str(number), **fields, data=[{"b64_json": self.large}],
                _image_thread={"nested": [True, False, None, 0, 1, 2.5, "quoted \"text\""]}))
        with self.no_large_sql_results(), self.store.connect() as db:
            projected = {request_id: task for _, _, request_id, task in self.store.scheduling_receipts(db)}
        self.assertNotIn("upstream_unfinished", projected["0"])
        for number, value in enumerate((None, False, True, 0, 1), start=1):
            self.assertIs(type(projected[str(number)]["upstream_unfinished"]), type(value))
            self.assertEqual(projected[str(number)]["upstream_unfinished"], value)
        for task in projected.values():
            self.assertNotIn("_turn_reserved", task)
            self.assertNotIn("data", task)
            self.assertEqual(task["_image_thread"]["nested"], [True, False, None, 0, 1, 2.5, 'quoted "text"'])

    def test_status_by_id_owner_order_duplicates_and_full_list(self):
        wanted = self.receipt("wanted", data=[{"url": "https://fixture.invalid/image.png"}])
        self.put(wanted)
        self.put(self.receipt("hidden", owner_id="other", data=[{"b64_json": self.large}]))
        with self.no_large_sql_results():
            result = self.images.list_tasks({"id": "owner"}, [" wanted ", "hidden", "missing", "wanted", "missing", " "])
            self.assertEqual(result, {"items": [_public_task(wanted), _public_task(wanted)],
                                      "missing_ids": ["hidden", "missing", "missing"]})
            self.assertEqual(self.images.list_tasks({"id": "owner"}, []),
                             {"items": [_public_task(wanted)], "missing_ids": []})
        self.assertEqual(self.images._tasks, {})

    def test_idle_status_capacity_and_scheduler_never_read_unrequested_output(self):
        self.put(self.receipt("legacy", data=[{"b64_json": self.large}]))
        self.put(self.receipt("text", status="succeeded", content=self.large, result={"body": self.large}), "text")
        with self.no_large_sql_results(), patch.object(self.store, "receipts", side_effect=AssertionError("full scan")):
            self.assertEqual(self.images.list_tasks({"id": "owner"}, ["missing"])["items"], [])
            self.assertEqual(self.images.resource_occupancy(), {"by_account": {}, "unattributed": 0})
            self.admission.resource_snapshot()
            self.admission.model_resources(["fixture-text"])
            self.assertIsNone(self.admission.claim_next())
            self.admission.recover_one()
        self.assertEqual(self.images._tasks, {})

    def test_metadata_cleanup_matches_full_predicates_without_rewriting_survivors(self):
        variants = [{}, {"upstream_unfinished": None}, {"upstream_unfinished": False},
                    {"upstream_unfinished": True}, {"upstream_unfinished": 0},
                    {"upstream_unfinished": 1}, {"retain_receipt": True},
                    {"error_code": "CONVERSATION_OUTCOME_UNKNOWN"}, {"status": "running"},
                    {"status": "unknown"}]
        tasks = {"owner:" + str(i): self.receipt(str(i), retain_receipt=False, updated_at="2000-01-01 00:00:00",
                 data=[{"b64_json": self.large}], **{k: v for k, v in changes.items() if k != "retain_receipt"})
                 for i, changes in enumerate(variants)}
        tasks["owner:6"]["retain_receipt"] = True
        for task in tasks.values(): self.put(task)
        expected_removed = self.images._expired_task_keys(tasks)
        with self.store.connect() as db:
            originals = dict(db.execute("SELECT task_key,receipt FROM image_requests"))
        with self.no_large_sql_results(), patch.object(self.images, "_save_locked", side_effect=AssertionError("full save")):
            self.images.list_tasks({"id": "owner"}, ["missing"])
        with self.store.connect() as db:
            remaining = dict(db.execute("SELECT task_key,receipt FROM image_requests"))
        self.assertEqual(remaining, {key: raw for key, raw in originals.items() if key not in expected_removed})
        self.assertIn("owner:7", remaining)  # UNKNOWN never expires.
        self.assertIn("owner:3", remaining)  # True differs from integer 1.
        self.assertNotIn("owner:5", remaining)

    def test_expired_claim_recovery_preserves_arbitrary_full_payload(self):
        for kind, submitted, result in (("text", False, False), ("text", True, False),
                                         ("image", False, False), ("image", True, False), ("image", True, True)):
            name = kind + str(submitted) + str(result)
            self.put(self.receipt(name, status="running", _claim_id="expired", _claim_until=1,
                _submission_started=submitted, result_file_ids=["file-original"] if result else [],
                data=[{"b64_json": self.large}], arbitrary={"output": self.large}, content=self.large), kind)
        with self.store.transaction() as db:
            rows = list(self.store.scheduling_receipts(db))
            self.admission._recover_claims(db, rows, 1000)
            for kind, owner, name, projected in rows:
                saved = self.store.read_receipt(db, kind, owner, name)
                self.assertEqual(saved["data"], [{"b64_json": self.large}])
                self.assertEqual(saved["arbitrary"], {"output": self.large})
                self.assertEqual(saved["content"], self.large)
                self.assertEqual(saved["status"], projected["status"])
                if saved["_submission_started"]:
                    self.assertEqual(saved["status"], "unknown" if kind == "text" else "error")
                    self.assertEqual(saved["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
                else:
                    self.assertEqual(saved["status"], "queued")

    def test_codex_binding_scan_preserves_full_target(self):
        self.admission.codex = SimpleNamespace(_eligible_account=lambda *args, **kwargs: None)
        self.admission._next_codex_probe = float("inf")
        queued = self.receipt("codex", status="queued", _route="codex", _input_ref="fixture",
            client_conversation_id="session", arbitrary={"body": self.large}, content=self.large)
        self.put(queued, "text")
        account = {"provider_account_identity": "account", "account_id": "fixture-upstream",
                   "quota": 0, "codex_affinities": {"session": {"state": "ready"}}}
        with patch.object(self.admission, "_rows", return_value=[account]):
            self.assertIsNone(self.admission.claim_next())
        saved = self.read("codex", "text")
        self.assertEqual(saved["provider_account_identity"], "account")
        self.assertEqual(saved["arbitrary"], queued["arbitrary"])
        self.assertEqual(saved["content"], queued["content"])

    def test_terminal_empty_and_unknown_planning_matches_full_history(self):
        original = self.receipt("original", status="unknown", route="chat", _public_session_ref="session",
            _sequence=1, client_conversation_id="session", provider_binding_id="binding",
            provider_account_identity="account", conversation_id="conversation", request_message_id="request",
            recovery_reason="REQUEST_RESULT_TERMINAL_EMPTY", _upstream_terminal=True,
            _turn_end_evidence={"conversation_id": "conversation", "request_message_id": "request",
                                "final_message_id": "final", "observed_at": 100.0}, content=self.large)
        correction = {**original, "id": "correction", "status": "succeeded", "_sequence": 2,
            "_input_ref": "fixture", "_previous_request_id": "original", "_terminal_empty_correction_of": "original",
            "_submission_parent_message_id": "final", "parent_message_id": "correction-final",
            "request_message_id": "correction-request"}
        self.put(original, "text"); self.put(correction, "text")
        self.put(self.receipt("unknown-image", status="error", upstream_unfinished=True,
                             error_code="CONVERSATION_OUTCOME_UNKNOWN", data=[{"b64_json": self.large}]))
        for terminal in (True, 1, False, None):
            original["_upstream_terminal"] = terminal
            self.put(original, "text")
            with self.store.connect() as db:
                full = list(self.store.receipts(db))
                with self.no_large_sql_results():
                    projected = list(self.store.scheduling_receipts(db))
                a = self.admission._snapshot([], full, self.admission._settings(), {}, 1000, {})
                b = self.admission._snapshot([], projected, self.admission._settings(), {}, 1000, {})
                self.assertEqual(a, replace(b, revision=a.revision))
                previous = next(r for r in b.requests if r.ref.request_id == "original")
                self.assertEqual(previous.state, "unknown")
                self.assertEqual(previous.order_group, None if terminal is True else "session")

    def test_thread_binding_and_planner_match_full_receipts_and_reject_changed_source(self):
        source = self.receipt("source", provider_binding_id="binding", provider_account_identity="account",
            client_conversation_id="session", conversation_id="conversation", parent_message_id="parent",
            request_message_id="request", data=[{"b64_json": self.large}])
        thread = {"protocol": PROTOCOL, "id": "thread", "previous_task_id": "source",
                  "edit_source_task_id": "source", "origin_task_id": "source",
                  "edit_source_fingerprint": source_fingerprint(source)}
        child = self.receipt("child", status="queued", _image_thread=thread, client_conversation_id="session",
                             arbitrary={"output": self.large})
        self.put(source); self.put(child)
        with self.store.transaction() as db:
            full = list(self.store.receipts(db)); projected = list(self.store.scheduling_receipts(db))
            args = ([], self.admission._settings(), {}, 1000, {})
            a = self.admission._snapshot(args[0], full, *args[1:])
            reads = []
            def source_reader(owner, task_id):
                reads.append((owner, task_id))
                return self.store.read_receipt(db, "image", owner, task_id)
            b = self.admission._snapshot(args[0], projected, *args[1:], source_reader=source_reader)
            self.assertEqual(a, replace(b, revision=a.revision))
            self.assertEqual(reads, [("owner", "source")])
            bind_waiting_threads(self.store, db, projected)
            saved = self.store.read_receipt(db, "image", "owner", "child")
            self.assertEqual(saved["arbitrary"], child["arbitrary"])
            self.assertEqual(saved["provider_account_identity"], "account")
            self.assertEqual(projected[1][3]["provider_account_identity"], "account")
            source["data"] = [{"b64_json": "changed-original-output"}]
            self.store.write_receipt(db, "image", "owner", "source", source)
            bind_waiting_threads(self.store, db, projected)
            saved = self.store.read_receipt(db, "image", "owner", "child")
            self.assertEqual(saved["_image_thread_waiting_reason"], "IMAGE_THREAD_SOURCE_CHANGED")
            self.assertEqual(saved["arbitrary"], child["arbitrary"])


if __name__ == "__main__":
    unittest.main()
