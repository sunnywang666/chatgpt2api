"""Read-time repair for one fully evidenced native-Codex rejection."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services.text_task_service import TextTaskService


class QueuedExecutor:
    def submit(self, *_args):
        return None


class HistoricalCodexRejectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "tasks.sqlite3"
        self.recovery_reader = mock.Mock(side_effect=AssertionError("explicit rejection must not be read upstream"))
        self.service = TextTaskService(self.path, executor=QueuedExecutor(), clock=lambda: 100.0,
                                       recovery_reader=self.recovery_reader)

    def _receipt(self, request_id, *, wire=b"", output_ref=None, **changes):
        body = {
            "client_request_id": request_id,
            "client_conversation_id": "synthetic-codex-session-" + request_id,
            "model": "gpt-5.6-codex",
            "_route": "codex",
            "_operation": "text",
            "_forward": {"protocol": "codex", "payload": {"model": "gpt-5.6-codex", "input": []},
                         "headers": {}, "identity": {"id": "synthetic-owner"}, "compact": False},
        }
        self.service.submit("owner", body)
        output = output_ref or self.service.store.create_output()
        if wire:
            with self.service.store.output_file(output, append=True) as handle:
                handle.write(wire)
        timeline = [
            {"stage": "accepted", "at": 1.0},
            {"stage": "execution_claimed", "at": 2.0},
            {"stage": "send_guard_passed", "at": 3.0},
            {"stage": "upstream_terminal", "at": 4.0},
            {"stage": "task_finished", "at": 5.0},
        ]
        receipt = {
            "status": "unknown", "error_code": "codex_request_rejected", "upstream_outcome": "unknown",
            "_submission_started": True, "_upstream_terminal": True, "_turn_reserved": True, "_executing": False,
            "_claim_id": "expired-synthetic-claim", "_claim_until": 99.0,
            "_wire_output": output, "_wire_size": 0, "_wire_error_status": 400,
            "_wire_error_payload": {"error": {"code": "codex_request_rejected", "type": "invalid_request_error",
                                                  "message": "synthetic request rejected"}},
            "original_upstream_response_ids": {"request_id": "req_synthetic"},
            "_execution_timeline": timeline,
        }
        self.service._update("owner", request_id, **{**receipt, **changes})
        with self.service.store.connect() as db:
            return self.service.store.read_receipt(db, "text", "owner", request_id)

    def test_exact_historical_codex_rejection_is_corrected_on_owner_read_without_recovery(self):
        before = self._receipt("synthetic-explicit-rejection")

        public = self.service.read("owner", "synthetic-explicit-rejection")

        self.assertEqual(public["status"], "failed")
        self.assertEqual(public["error_code"], "codex_request_rejected")
        self.assertEqual(public["upstream_outcome"], "rejected")
        self.assertNotIn("_wire_output", public)
        self.recovery_reader.assert_not_called()
        with self.service.store.connect() as db:
            stored = self.service.store.read_receipt(db, "text", "owner", "synthetic-explicit-rejection")
        self.assertIs(stored["_upstream_terminal"], True)
        self.assertIs(stored["_turn_reserved"], False)
        self.assertIs(stored["_executing"], False)
        self.assertIsNone(stored["_claim_id"])
        self.assertIsNone(stored["_claim_until"])
        self.assertEqual(stored["_wire_output"], before["_wire_output"])
        self.assertEqual(stored["_wire_size"], 0)
        self.assertEqual(stored["_wire_error_payload"], before["_wire_error_payload"])
        self.assertEqual(stored["original_upstream_response_ids"], before["original_upstream_response_ids"])
        self.assertEqual(stored["_execution_timeline"], before["_execution_timeline"])
        self.assertTrue(stored["_input_ref"])

    def test_historical_codex_rejection_leaves_partial_missing_or_live_evidence_unknown(self):
        missing_output = "f" * 32 + ".bin"
        cases = {
            "partial-wire": {"wire": b"partial native response", "_wire_size": 0},
            "missing-wire": {"output_ref": missing_output},
            "live-claim": {"_claim_until": 101.0},
            "paused": {"_recovery_paused": True},
            "timeout": {"_wire_error_status": 408},
            "saved-response": {"_codex_response_id": "resp_synthetic"},
        }
        for name, changes in cases.items():
            with self.subTest(case=name):
                before = self._receipt("synthetic-guard-" + name, **changes)
                public = self.service.read("owner", "synthetic-guard-" + name)
                self.assertEqual(public["status"], "unknown")
                self.assertEqual(public["upstream_outcome"], "unknown")
                with self.service.store.connect() as db:
                    stored = self.service.store.read_receipt(db, "text", "owner", "synthetic-guard-" + name)
                self.assertEqual(stored["status"], before["status"])
                self.assertEqual(stored["_turn_reserved"], before["_turn_reserved"])
                self.assertEqual(stored["_claim_id"], before["_claim_id"])
                self.assertEqual(stored["_wire_output"], before["_wire_output"])
        self.recovery_reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
