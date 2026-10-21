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
            "revision": 1, "image_account_concurrency": 2, "codex_max_concurrency": 3,
            "chat_account_concurrency": 1})
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
        with tasks._transaction():
            tasks._tasks = {"owner:original": {"id": "original", "owner_id": "owner", "provider_account_identity": "a", "upstream_unfinished": True,
                                              "status": "error", "error_code": "CONVERSATION_OUTCOME_UNKNOWN"}}
            tasks._save_locked()
        rows = [
            {"provider_account_identity": "a", "quota": 9, "image_inflight": 1,
             "limits_progress": [{"feature_name": "image_gen", "remaining": 9}]},
            {"provider_account_identity": "b", "quota": 0,
             "limits_progress": [{"feature_name": "image_gen", "remaining": 0}]},
            {"provider_account_identity": "c", "quota": 0},
        ]
        accounts = Mock()
        for row in rows:
            row.update(source_type="web", access_token="synthetic-chat", type="Plus", status="正常")
        rows[0]["capacity_observed_at"] = datetime.now(timezone.utc).isoformat()
        accounts.list_accounts.return_value = rows
        accounts._is_image_account_available.side_effect = lambda row: row["quota"] > 0
        codex = Mock()
        codex.resource_snapshot.return_value = {"eligible_accounts": 0, "inflight": 0, "slots_total": 0, "slots_free": 0}
        with patch("services.pool_resources.config", self.config):
            image = resource_snapshot(accounts, tasks, codex)["image"]
        self.assertIsNone(image["remaining"])
        self.assertEqual(image["known_remaining"], 9)
        self.assertEqual(image["observed_accounts"], 1)
        self.assertEqual(image["total_accounts"], 3)
        self.assertIsNone(image["slots_free"])
        self.assertEqual((image["slots_free_min"], image["slots_free_max"]), (1, 2))
        self.assertEqual(tasks._tasks["owner:original"]["error_code"], "CONVERSATION_OUTCOME_UNKNOWN")
        rows[0]["image_inflight"] = 0
        with patch("services.pool_resources.config", self.config):
            self.assertEqual(resource_snapshot(accounts, tasks, codex)["image"]["slots_free"], 2)

    def test_admission_projection_skips_legacy_durable_scan_without_changing_merged_payload(self):
        rows = [{"source_type": "web", "access_token": "synthetic-chat", "type": "Plus", "status": "正常",
                 "provider_account_identity": "a", "image_inflight": 1,
                 "capacity_observed_at": datetime.now(timezone.utc).isoformat(),
                 "limits_progress": [{"feature_name": "image_gen", "remaining": 3}]}]
        accounts = Mock()
        accounts.list_accounts.return_value = rows
        accounts._is_image_account_available.return_value = True
        tasks = Mock()
        tasks.resource_occupancy.side_effect = AssertionError("admission payload already owns durable occupancy")
        shared = {
            "accounts": [{"image": {"capacity": 2}}],
            "settings": {"revision": 4, "image_account_concurrency": 2, "codex_max_concurrency": 3},
            "image": {"slots_total": 2, "inflight": 1, "slots_free": 1, "dispatchable_now": 0},
            "codex": {"slots_total": 0, "slots_free": 0},
            "queue": {"mode": "durable_original_receipts", "queued": 843},
        }
        tasks.admission.resource_snapshot.return_value = shared
        codex = Mock()
        codex.resource_snapshot.return_value = {"legacy": "discarded"}

        with patch("services.pool_resources.config", self.config):
            result = resource_snapshot(accounts, tasks, codex)

        observed_at = result.pop("observed_at")
        self.assertIsNotNone(datetime.fromisoformat(observed_at))
        self.assertEqual(result, {
            **shared,
            "image": {"remaining": 3, "known_remaining": 3, "observed_accounts": 1, "total_accounts": 1,
                      "eligible_accounts": 1, "slots_total": 2, "slots_free": 1, "slots_free_min": 1,
                      "slots_free_max": 1, "process_inflight": 1, "durable_inflight": 1,
                      "occupancy_uncertain": False, "inflight": 1, "dispatchable_now": 0},
        })
        tasks.resource_occupancy.assert_not_called()

    def test_legacy_projection_still_reads_durable_occupancy(self):
        accounts = Mock()
        accounts.list_accounts.return_value = []
        tasks = Mock()
        tasks.admission = None
        tasks.resource_occupancy.return_value = {"by_account": {}, "unattributed": 0}

        with patch("services.pool_resources.config", self.config):
            resource_snapshot(accounts, tasks, Mock())

        tasks.resource_occupancy.assert_called_once_with()

    def test_image_slots_use_fresh_image_evidence_not_generic_quota(self):
        rows = [{"source_type": "web", "access_token": "synthetic", "type": "Plus",
                 "status": "正常", "quota": 999, "provider_account_identity": "a",
                 "capacity_observed_at": datetime.now(timezone.utc).isoformat(),
                 "limits_progress": [{"feature_name": "image_gen", "remaining": 2}]}]
        accounts = Mock()
        accounts.list_accounts.return_value = rows
        accounts._is_image_account_available.return_value = True
        tasks = ImageTaskService(Path(self.tmp.name) / "tasks.json")
        with patch("services.pool_resources.config", self.config):
            image = resource_snapshot(accounts, tasks, Mock())["image"]
            self.assertEqual(image["slots_total"], 2)
            rows[0]["capacity_read_failed_at"] = datetime.now(timezone.utc).isoformat()
            image = resource_snapshot(accounts, tasks, Mock())["image"]
            self.assertEqual(image["slots_total"], 0)
            self.assertEqual(image["eligible_accounts"], 0)

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
            self.assertEqual(self.config.resource_settings(), {**snapshot, "chat_account_concurrency": 1})
            read.assert_called_once()
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "resource-settings"):
            self.config.update({"image_account_concurrency": 1, "auth-key": "replacement"})
        self.assertFalse(self.config._resource_path.exists())
        self.assertEqual(self.path.read_bytes(), before)

    def test_chat_account_activity_limit_is_persisted_separately_from_image_limit(self):
        result = self.config.update_resource_settings(0, 4, 3, 2)
        self.assertEqual(result["chat_account_concurrency"], 2)
        self.assertEqual(ConfigStore(self.path).chat_account_concurrency, 2)
        # An old caller that omits the new optional field preserves the value.
        result = self.config.update_resource_settings(1, 4, 3)
        self.assertEqual(result["chat_account_concurrency"], 2)

    def test_one_request_second_slot_persists_and_requires_global_one(self):
        trial = {"account_identity": "account_fixture", "request_id": "original-request", "expires_at": 1100}
        result = self.config.update_temporary_chat_second_slot(0, trial)
        self.assertEqual(result["revision"], 1)
        self.assertEqual(ConfigStore(self.path).resource_settings()["temporary_chat_second_slot"], trial)
        with self.assertRaisesRegex(ValueError, "global_one"):
            self.config.update_resource_settings(1, 4, 4, 2)
        self.assertEqual(self.config.resource_settings()["revision"], 1)
        self.assertEqual(self.config.update_temporary_chat_second_slot(1, None)["revision"], 2)
        self.assertNotIn("temporary_chat_second_slot", ConfigStore(self.path).resource_settings())
        self.assertEqual(self.config.update_resource_settings(2, 4, 4, 2)["chat_account_concurrency"], 2)

    def test_one_exact_firstsend_model_trial_persists_and_can_be_revoked(self):
        trial = {"account_identity": "account_fixture", "request_id": "original-request", "expires_at": 1100}
        result = self.config.update_temporary_firstsend_model_trial(0, trial)
        self.assertEqual(result["revision"], 1)
        self.assertEqual(ConfigStore(self.path).resource_settings()["temporary_firstsend_model_trial"], trial)
        # Existing capacity settings remain untouched by the model-trial arm.
        self.assertEqual(result["chat_account_concurrency"], 1)
        stopped = self.config.update_temporary_firstsend_model_trial(1, None)
        self.assertEqual(stopped["revision"], 2)
        self.assertNotIn("temporary_firstsend_model_trial", stopped)

    def test_bounded_firstsend_model_trial_targets_are_exact_and_unique(self):
        trial = {"targets": [
            {"account_identity": "account_fixture", "request_id": "original-request-a"},
            {"account_identity": "account_fixture", "request_id": "original-request-b"},
        ], "expires_at": 4600}
        result = self.config.update_temporary_firstsend_model_trial(0, trial)
        self.assertEqual(result["temporary_firstsend_model_trial"], trial)
        duplicate = {"targets": [
            {"account_identity": "account_fixture", "request_id": "original-request-a"},
            {"account_identity": "other", "request_id": "original-request-a"},
        ], "expires_at": 4600}
        with self.assertRaisesRegex(ValueError, "resource_settings_invalid"):
            self.config.update_temporary_firstsend_model_trial(1, duplicate)

    def test_codex_unobserved_capacity_is_unknown_not_zero(self):
        record = account(source_type="codex")
        record["codex_observation"] = observation(state="unknown")
        result = CodexService(FakeAccounts([record])).resource_snapshot()
        self.assertEqual(result["state"], "unknown")
        self.assertIsNone(result["slots_total"])
        self.assertIsNone(result["slots_free"])


if __name__ == "__main__":
    unittest.main()
