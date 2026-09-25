from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from unittest import mock

from scripts.reconcile_incident_image_receipts import ReconciliationError, reconcile


def receipt(task_id: str, *, status: str = "success", owner: str = "owner-1") -> dict:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "id": task_id, "owner_id": owner, "request_hash": "hash-" + task_id,
        "status": status, "mode": "generate", "model": "gpt-image-2",
        "created_at": now, "updated_at": now, "created_ts": 1, "updated_ts": 2,
        "binding_status": "bound", "conversation_id": "original-conversation",
        "provider_binding_id": "original-binding", "provider_account_identity": "original-account",
        "client_conversation_id": "original-client", "parent_message_id": "original-parent",
        "error_code": "CONTENT_POLICY_VIOLATION" if status == "error" else "",
        "data": [] if status == "error" else [{"url": "https://example.test/images/original.png"}],
    }


class IncidentImageReceiptsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.json_path = self.root / "image_tasks.json"
        self.db_path = self.root / "text_tasks.sqlite3"
        self.ids = ["original-a", "original-b", "original-c", "original-error"]
        self.items = [receipt(task_id, status="error" if task_id.endswith("error") else "success")
                      for task_id in self.ids]
        self.items.append(receipt("existing"))
        self._write_json()
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("CREATE TABLE task_runtime(name TEXT PRIMARY KEY,value TEXT NOT NULL)")
            db.execute("INSERT INTO task_runtime VALUES('image_json_imported','true')")
            db.execute("CREATE TABLE image_requests(task_key TEXT PRIMARY KEY,receipt TEXT NOT NULL)")
            db.execute("CREATE TABLE requests(owner TEXT,id TEXT,receipt TEXT,PRIMARY KEY(owner,id))")
            db.execute("INSERT INTO image_requests VALUES(?,?)",
                       ("owner-1:existing", json.dumps(self.items[-1])))
            # The stopped historical recovery set must be untouched by this repair.
            for index in range(55):
                saved = {**receipt(f"stopped-{index}", status="error"), "_recovery_suppressed": True}
                db.execute("INSERT INTO image_requests VALUES(?,?)",
                           (f"owner-1:stopped-{index}", json.dumps(saved)))
            db.execute("INSERT INTO requests VALUES('owner-1','text-unknown','{}')")

    def _write_json(self) -> None:
        self.json_path.write_text(json.dumps({"tasks": self.items}), encoding="utf-8")

    def _rows(self) -> dict[str, dict]:
        with closing(sqlite3.connect(self.db_path)) as db, db:
            return {key: json.loads(raw) for key, raw in db.execute("SELECT task_key,receipt FROM image_requests")}

    def test_dry_run_then_atomic_original_id_insert_preserves_existing_ledgers(self) -> None:
        initial_rows = self._rows()
        preview = reconcile(self.root, self.ids)
        self.assertFalse(preview["applied"])
        self.assertEqual(preview["count"], 4)
        self.assertEqual(self._rows(), initial_rows)
        self.assertNotIn("original-a", json.dumps(preview["selected"]))

        applied = reconcile(self.root, self.ids, apply=True,
                            expected_json_sha256=preview["json_sha256"])
        self.assertTrue(applied["applied"])
        rows = self._rows()
        self.assertEqual(len(rows), len(initial_rows) + 4)
        for key, value in initial_rows.items():
            self.assertEqual(rows[key], value)
        for task_id in self.ids:
            self.assertEqual(rows[f"owner-1:{task_id}"]["id"], task_id)
        self.assertEqual(rows["owner-1:original-error"]["error_code"],
                         "CONTENT_POLICY_VIOLATION")
        self.assertNotIn("_recovery_suppressed", rows["owner-1:original-error"])
        with closing(sqlite3.connect(self.db_path)) as db, db:
            self.assertEqual(db.execute("SELECT count(*) FROM requests").fetchone()[0], 1)
        repeated = reconcile(self.root, self.ids, apply=True,
                             expected_json_sha256=preview["json_sha256"])
        self.assertEqual(repeated["state"], "already_present")
        self.assertEqual(self._rows(), rows)

    def test_changed_json_refuses_without_sqlite_write(self) -> None:
        preview = reconcile(self.root, self.ids)
        self.items[0]["updated_ts"] = 3
        self._write_json()
        with self.assertRaisesRegex(ReconciliationError, "digest changed"):
            reconcile(self.root, self.ids, apply=True,
                      expected_json_sha256=preview["json_sha256"])
        self.assertEqual(len(self._rows()), 56)

    def test_unselected_json_only_refuses(self) -> None:
        self.items.append(receipt("unexpected-fifth"))
        self._write_json()
        with self.assertRaisesRegex(ReconciliationError, "differ from the explicit selection"):
            reconcile(self.root, self.ids)

    def test_changed_existing_receipt_refuses(self) -> None:
        self.items[-1]["status"] = "error"
        self.items[-1]["data"] = []
        self._write_json()
        with self.assertRaisesRegex(ReconciliationError, "differs between ledgers"):
            reconcile(self.root, self.ids)

    def test_changed_existing_original_identity_refuses(self) -> None:
        existing = self.items[-1]
        for field in (
            "request_hash", "provider_binding_id", "provider_account_identity",
            "client_conversation_id", "conversation_id", "parent_message_id",
        ):
            with self.subTest(field=field):
                original = existing[field]
                existing[field] = "different-original-identity"
                self._write_json()
                with self.assertRaisesRegex(ReconciliationError, "differs between ledgers"):
                    reconcile(self.root, self.ids)
                existing[field] = original
        self._write_json()

    def test_nonterminal_or_resultless_success_refuses(self) -> None:
        for status, data, message in (("running", [], "not terminal"),
                                      ("success", [], "no saved image result")):
            with self.subTest(status=status, data=data):
                self.items[0]["status"] = status
                self.items[0]["data"] = data
                self._write_json()
                with self.assertRaisesRegex(ReconciliationError, message):
                    reconcile(self.root, self.ids)

    def test_unknown_error_refuses_without_restarting_recovery(self) -> None:
        self.items[3]["error_code"] = "CONVERSATION_OUTCOME_UNKNOWN"
        self._write_json()
        with self.assertRaisesRegex(ReconciliationError, "outcome is UNKNOWN"):
            reconcile(self.root, self.ids)
        self.assertEqual(len(self._rows()), 56)

        original_json = self.json_path.read_bytes()
        preview = reconcile(self.root, self.ids,
                            quarantine_unknown_ids={"original-error"})
        applied = reconcile(self.root, self.ids, apply=True,
                            expected_json_sha256=preview["json_sha256"],
                            quarantine_unknown_ids={"original-error"})
        self.assertTrue(applied["applied"])
        self.assertEqual(applied["quarantined_count"], 1)
        self.assertEqual(self.json_path.read_bytes(), original_json)
        stored = self._rows()["owner-1:original-error"]
        self.assertEqual(stored["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
        self.assertTrue(stored["_recovery_suppressed"])
        self.assertEqual(stored["data"], [])

        from services.image_task_service import ImageTaskService
        from services.task_store import TaskStore
        service = ImageTaskService(self.json_path, store=TaskStore(self.db_path),
                                   retention_days_getter=lambda: 30)
        with mock.patch("services.image_task_service.threading.Thread",
                        side_effect=AssertionError("recovery thread must not start")):
            result = service.resume_poll({"id": "owner-1"}, "original-error",
                                         allow_unrecoverable_retry=True)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
        self.assertTrue(self._rows()["owner-1:original-error"]["_recovery_suppressed"])
        repeated = reconcile(self.root, self.ids, apply=True,
                             expected_json_sha256=preview["json_sha256"],
                             quarantine_unknown_ids={"original-error"})
        self.assertEqual(repeated["state"], "already_present")

    def test_quarantine_requires_an_exact_unknown_selected_id(self) -> None:
        with self.assertRaisesRegex(ReconciliationError, "quarantine target is not UNKNOWN"):
            reconcile(self.root, self.ids, quarantine_unknown_ids={"original-a"})
        with self.assertRaisesRegex(ReconciliationError, "must be explicitly selected"):
            reconcile(self.root, self.ids, quarantine_unknown_ids={"another-id"})

    def test_missing_import_marker_refuses(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("DELETE FROM task_runtime")
        with self.assertRaisesRegex(ReconciliationError, "not confirmed"):
            reconcile(self.root, self.ids)

    def test_selected_id_under_another_owner_refuses(self) -> None:
        other = receipt("original-a", owner="owner-2")
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("INSERT INTO image_requests VALUES(?,?)",
                       ("owner-2:original-a", json.dumps(other)))
        with self.assertRaisesRegex(ReconciliationError, "another SQLite owner"):
            reconcile(self.root, self.ids)


if __name__ == "__main__":
    unittest.main()
