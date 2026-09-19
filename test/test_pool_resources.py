import json
from datetime import datetime, timezone
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from services.config import ConfigStore
from services.codex_service import CodexService, CodexServiceError
from services.pool_resources import resource_snapshot
from services.image_task_service import ImageTaskService
from services.storage.base import AccountCommitUncertain
from test.test_codex_service import FakeAccounts, account, observation


class PoolResourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "config.json"
        self.path.write_text(json.dumps({"auth-key": "test-pool-management-secret"}))
        self.config = ConfigStore(self.path)

    def test_settings_persist_compare_revision_and_do_not_overwrite_other_config(self):
        self.config.update_resource_settings(0, 2, 3)
        reopened = ConfigStore(self.path)
        self.assertEqual(reopened.resource_settings(), {
            "revision": 1, "image_account_concurrency": 2, "codex_max_concurrency": 3})
        self.assertEqual(reopened.auth_key, self.config.auth_key)
        with self.assertRaises(ValueError):
            self.config.update_resource_settings(0, 1, 1)
        self.assertEqual(ConfigStore(self.path).resource_settings(), reopened.resource_settings())
        original_config = self.path.read_bytes()
        with patch("services.storage.json_storage.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(AccountCommitUncertain):
                self.config.update_resource_settings(1, 1, 1)
        self.assertEqual(self.config.resource_settings(), reopened.resource_settings())
        self.assertEqual(self.path.read_bytes(), original_config)
        self.assertEqual(ConfigStore(self.path).resource_settings(), reopened.resource_settings())

    def test_live_codex_limit_change_keeps_admitted_requests_and_recovers_capacity(self):
        with patch("services.codex_service.config", self.config):
            service = CodexService(FakeAccounts([]))
            service._acquire_capacity()
            service._acquire_capacity()
            self.config.update_resource_settings(0, 3, 1)
            with self.assertRaises(CodexServiceError):
                service._acquire_capacity()
            service._capacity.release()
            with self.assertRaises(CodexServiceError):
                service._acquire_capacity()
            service._capacity.release()
            service._acquire_capacity()
            service._capacity.release()
            self.assertEqual(service.resource_snapshot()["inflight"], 0)

    def test_model_specific_exhaustion_blocks_only_that_model_without_probe(self):
        record = account()
        record["codex_observation"] = observation(models=[
            {"id": "model-a", "label": "A"}, {"id": "model-b", "label": "B"}],
            limits=[{"id": " MODEL-A ", "label": "A", "windows": [{"used_percent": 100}]}])
        service = CodexService(FakeAccounts([record]), session_factory=Mock(side_effect=AssertionError("no probe")))
        self.assertIsNone(service._eligible_account(record, "model-a", allow_probe=False))
        self.assertIsNotNone(service._eligible_account(record, "model-b", allow_probe=False))
        record["codex_observation"]["limits"][0]["id"] = "unmapped-bucket"
        self.assertIsNone(service._eligible_account(record, "model-b", allow_probe=False))

    def test_resource_totals_unknown_zero_and_overlap_bounds_preserve_original_receipt(self):
        tasks = ImageTaskService(Path(self.tmp.name) / "tasks.json")
        tasks._tasks = {"original": {"provider_account_identity": "a", "upstream_unfinished": True,
                                      "status": "error", "error_code": "CONVERSATION_OUTCOME_UNKNOWN"}}
        rows = [
            {"provider_account_identity": "a", "quota": 9, "image_inflight": 1,
             "limits_progress": [{"feature_name": "image_gen", "remaining": 9}]},
            {"provider_account_identity": "b", "quota": 0,
             "limits_progress": [{"feature_name": "image_gen", "remaining": 0}]},
            {"provider_account_identity": "c", "quota": 0},
        ]
        accounts = Mock()
        for row in rows:
            row.update(source_type="web", access_token="synthetic-chat")
        accounts.list_accounts.return_value = rows
        accounts._is_image_account_available.side_effect = lambda row: row["quota"] > 0
        codex = Mock()
        codex.resource_snapshot.return_value = {"eligible_accounts": 0, "inflight": 0, "slots_total": 0, "slots_free": 0}
        with patch("services.pool_resources.config", self.config):
            image = resource_snapshot(accounts, tasks, codex)["image"]
        self.assertIsNone(image["remaining"])
        self.assertEqual(image["known_remaining"], 9)
        self.assertEqual(image["observed_accounts"], 0)
        self.assertEqual(image["total_accounts"], 3)
        self.assertIsNone(image["slots_free"])
        self.assertEqual((image["slots_free_min"], image["slots_free_max"]), (1, 2))
        self.assertEqual(tasks._tasks["original"]["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
        rows[0]["image_inflight"] = 0
        with patch("services.pool_resources.config", self.config):
            self.assertEqual(resource_snapshot(accounts, tasks, codex)["image"]["slots_free"], 2)

    def test_mixed_pool_image_totals_exclude_codex_only_and_include_exhausted_chat(self):
        rows = [
            {"source_type": "web", "access_token": "chat", "quota": 9,
             "capacity_observed_at": datetime.now(timezone.utc).isoformat(),
             "limits_progress": [{"feature_name": "image_gen", "remaining": 9}]},
            {"source_type": "web", "access_token": "exhausted", "quota": 0,
             "capacity_observed_at": datetime.now(timezone.utc).isoformat(),
             "limits_progress": [{"feature_name": "image_gen", "remaining": 0}]},
            {"source_type": "codex", "access_token": "codex", "quota": 0},
        ]
        accounts = Mock()
        accounts.list_accounts.return_value = rows
        accounts._is_image_account_available.side_effect = lambda row: row["quota"] > 0
        with patch("services.pool_resources.config", self.config):
            result = resource_snapshot(accounts, ImageTaskService(Path(self.tmp.name) / "tasks.json"), Mock())["image"]
        self.assertEqual(result["total_accounts"], 2)
        self.assertEqual(result["observed_accounts"], 2)
        self.assertEqual(result["remaining"], 9)

    def test_settings_read_one_snapshot_and_legacy_form_cannot_partially_commit(self):
        snapshot = {"revision": 1, "image_account_concurrency": 2, "codex_max_concurrency": 3}
        with patch.object(self.config, "_resource_data", return_value=snapshot) as read:
            self.assertEqual(self.config.resource_settings(), snapshot)
            read.assert_called_once()
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "resource-settings"):
            self.config.update({"image_account_concurrency": 1, "auth-key": "replacement"})
        self.assertFalse(self.config._resource_path.exists())
        self.assertEqual(self.path.read_bytes(), before)

    def test_codex_unobserved_capacity_is_unknown_not_zero(self):
        record = account(source_type="codex")
        record["codex_observation"] = observation(state="unknown")
        result = CodexService(FakeAccounts([record])).resource_snapshot()
        self.assertEqual(result["state"], "unknown")
        self.assertIsNone(result["slots_total"])
        self.assertIsNone(result["slots_free"])


if __name__ == "__main__":
    unittest.main()
