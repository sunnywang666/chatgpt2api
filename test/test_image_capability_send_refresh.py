"""Local original-receipt tests; no upstream requests or account credentials."""
import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, PropertyMock, Mock
from types import SimpleNamespace

from services.account_service import AccountService
from services.openai_backend_api import OpenAIBackendAPI

from services.account_request_pacing import AccountRequestClock, AccountRequestDeadlineExceeded
from services.config import config
from services.request_context import AdmissionLost, executing
from services.image_task_service import ImageTaskService
from test.test_pool_admission import build, Clock
from test.test_account_request_pacing import Response


class ImageCapabilitySendRefreshTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.row = {"access_token": "fixture-only", "account_id": "upstream-test",
            "provider_account_identity": "account-test", "type": "Plus", "status": "正常",
            "source_type": "web", "conversation_binding_ids": ["binding-test"],
            "limits_progress": [{"feature_name": "image_gen", "remaining": 1000}],
            "capacity_observed_at": datetime.now(timezone.utc).isoformat()}
        self.save()
        self.clock = Clock()
        self.accounts, self.store, self.admission = build(self.root, self.clock)
        self.images = ImageTaskService(self.root / "images.json", admission=self.admission, store=self.store)
        self.images.submit_generation({"id": "owner", "role": "user", "external_image_client": True},
            client_task_id="original-image", prompt="fixture", model="gpt-image-2", size=None)
        self.context = self.admission.claim_next()
        self.assertIsNotNone(self.context)
        self.row["capacity_observed_at"] = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        self.save()
        self.pace = AccountRequestClock("fixture", self.root / "clock.json")
        self.calls = []
        self.addCleanup(patch.stopall)
        patch.object(type(config), "account_request_interval_secs", new_callable=PropertyMock, return_value=0).start()
        patch.object(type(config), "account_message_interval_secs", new_callable=PropertyMock, return_value=0).start()

    def save(self):
        (self.root / "accounts.json").write_text(json.dumps([self.row]))

    def send(self, method, url, **kwargs):
        self.calls.append(method)
        return Response()

    def generation(self, **kwargs):
        with executing(self.context):
            return self.pace.request(self.send, "POST", "https://fixture/backend-api/conversation", **kwargs)

    def refresh(self, account_ref, *, deadline, before_read):
        self.assertEqual(account_ref, self.accounts.pool_account_ref(self.row))
        self.assertGreater(deadline, time.monotonic())
        before_read()
        # The three normal metadata calls use worker threads. This thread must
        # acquire the very same HTTP clock; a held clock would deadlock here.
        completed = threading.Event()
        errors = []
        def read():
            try:
                self.pace.request(self.send, "GET", "https://fixture/backend-api/me",
                    _account_request_deadline_monotonic=deadline)
            except Exception as exc:
                errors.append(exc)
            finally:
                completed.set()
        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        self.assertTrue(completed.wait(2), "metadata blocked behind the generation HTTP lock")
        thread.join()
        self.assertEqual(errors, [])
        self.row["capacity_observed_at"] = datetime.now(timezone.utc).isoformat()
        self.row["capacity_used_since_observation"] = False
        self.save()

    def test_expired_prepared_image_refreshes_and_sends_original_once(self):
        self.accounts.refresh_image_capability = self.refresh
        original = self.context.receipt()
        self.generation()
        self.assertEqual(self.calls, ["GET", "POST"])
        saved = self.context.receipt()
        for key in ("id", "_claim_id", "provider_binding_id", "provider_account_identity", "_input_ref"):
            self.assertEqual(saved[key], original[key])
        self.assertTrue(saved["_submission_started"])
        with self.assertRaises(AdmissionLost):
            self.generation()
        self.assertEqual(self.calls, ["GET", "POST"], "never refresh/replay an already submitted request")

    def test_zero_failed_read_disabled_and_unknown_capacity_do_not_refresh_or_send(self):
        for change in ({"limits_progress": [{"feature_name": "image_gen", "remaining": 0}]},
                       {"capacity_read_failed_at": datetime.now(timezone.utc).isoformat()},
                       {"managed_disabled": True}, {"limits_progress": []}):
            original = dict(self.row)
            self.row.update(change)
            self.save()
            self.accounts.refresh_image_capability = lambda *a, **kw: self.fail("unsafe metadata refresh")
            self.assertIsNone(self.context.image_capability_refresh())
            # Check the guard directly so this case leaves no requeue mutation
            # that could weaken the next independent negative assertion.
            with self.assertRaises(AdmissionLost):
                self.admission._before_send_guard(self.context)
            self.assertEqual(self.calls, [])
            self.row = original

    def test_new_zero_or_binding_change_during_refresh_prevents_post(self):
        def zero(ref, **kwargs):
            self.refresh(ref, **kwargs)
            self.row["limits_progress"] = [{"feature_name": "image_gen", "remaining": 0}]
            self.save()
        self.accounts.refresh_image_capability = zero
        with self.assertRaises(AdmissionLost):
            self.generation()
        self.assertEqual(self.calls, ["GET"])
        self.assertFalse(self.context.receipt()["_submission_started"])

    def test_claim_change_during_refresh_prevents_post(self):
        def changed(ref, **kwargs):
            self.refresh(ref, **kwargs)
            self.admission.update_claim(self.context, _claim_id="other-claim")
        self.accounts.refresh_image_capability = changed
        with self.assertRaises(AdmissionLost):
            self.generation()
        self.assertEqual(self.calls, ["GET"])

    def test_binding_change_during_refresh_prevents_post(self):
        def changed(ref, **kwargs):
            self.refresh(ref, **kwargs)
            self.row["conversation_binding_ids"] = []
            self.save()
        self.accounts.refresh_image_capability = changed
        with self.assertRaises((AdmissionLost, RuntimeError)):
            self.generation()
        self.assertEqual(self.calls, ["GET"])

    def test_expired_budget_prevents_metadata_and_generation(self):
        self.accounts.refresh_image_capability = lambda *a, **kw: self.fail("expired read")
        with self.assertRaises(AccountRequestDeadlineExceeded):
            self.generation(_account_request_deadline_monotonic=time.monotonic() - 1)
        self.assertEqual(self.calls, [])

    def test_refresh_reapplies_updated_account_pace_before_generation(self):
        ready = []
        def refreshed(ref, **kwargs):
            self.refresh(ref, **kwargs)
            with self.pace.lock:
                self.pace.next_request = time.monotonic() + 0.025
                ready.append(self.pace.next_request)
                self.pace._save()
        self.accounts.refresh_image_capability = refreshed
        original_send = self.send
        def send(method, url, **kwargs):
            if method == "POST":
                self.assertGreaterEqual(time.monotonic(), ready[0])
            return original_send(method, url, **kwargs)
        self.send = send
        self.generation()
        self.assertEqual(self.calls, ["GET", "POST"])

    def test_limits_observation_is_stamped_after_pacing_at_actual_send(self):
        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://fixture"
        backend._headers = Mock(return_value={})
        began = datetime.now(timezone.utc)
        with self.pace.lock:
            self.pace.next_request = time.monotonic() + 0.025
            self.pace._save()
        def post(url, **kwargs):
            self.assertFalse(hasattr(backend, "_capacity_observed_at"))
            def send(method, url, **kwargs):
                sampled = datetime.fromisoformat(backend._capacity_observed_at)
                self.assertGreaterEqual((sampled - began).total_seconds(), 0.02)
                self.assertLessEqual(sampled, datetime.now(timezone.utc))
                return SimpleNamespace(status_code=200, headers={}, json=lambda: {"limits_progress": []})
            return self.pace.request(send, "POST", url, **kwargs)
        backend.session = SimpleNamespace(post=post)
        backend._get_conversation_init()
        self.assertTrue(backend._capacity_observed_at)

    def test_metadata_cancellation_prevents_transport_and_does_not_invent_observation(self):
        backend = object.__new__(OpenAIBackendAPI)
        backend.metadata_before_send = Mock(side_effect=AdmissionLost("cancelled"))
        backend.metadata_deadline = time.monotonic() + 1
        with self.assertRaises(AdmissionLost):
            self.pace.request(self.send, "GET", "https://fixture/backend-api/me",
                              **backend._metadata_request_options(capacity=True))
        self.assertEqual(self.calls, [])
        self.assertFalse(hasattr(backend, "_capacity_observed_at"))

    def test_bounded_real_refresh_keeps_sample_stamp_and_rejects_concurrent_consumption(self):
        # Use the real update/CAS path, with only upstream metadata replaced.
        workspace = "12345678-1234-5678-9234-567812345678"
        self.row["account_id"] = workspace
        self.row["user_id"] = "fixture-user"
        self.save()
        sampled = (datetime.now(timezone.utc) - timedelta(seconds=50)).isoformat()
        info = {"capacity_observed_at": sampled, "quota": 1000,
                "limits_progress": [{"feature_name": "image_gen", "remaining": 1000}]}
        self.accounts.refresh_image_capability = lambda ref, **kw: AccountService.refresh_image_capability(self.accounts, ref, **kw)
        with patch.object(self.accounts, "refresh_access_token", side_effect=AssertionError("send-edge auth refresh")), \
             patch.object(self.accounts, "_verified_chat_info", return_value=(("fixture-user", workspace), info)):
            self.context.image_capability_refresh()(time.monotonic() + 10)
        saved = self.accounts.get_account("fixture-only")
        self.assertEqual(saved["capacity_observed_at"], sampled)
        # Consume while a subsequent capacity response is in flight. Even a
        # newer positive response must not erase this persisted consumption.
        self.accounts.mark_image_result("fixture-only", False)
        def metadata(token, **kwargs):
            kwargs["before_read"]()
            self.accounts.mark_image_result(token, False)
            return (("fixture-user", workspace), {**info, "capacity_observed_at": datetime.now(timezone.utc).isoformat()})
        with patch.object(self.accounts, "_verified_chat_info", side_effect=metadata):
            with self.assertRaises(AdmissionLost):
                self.generation()
        saved = self.accounts.get_account("fixture-only")
        self.assertTrue(saved["capacity_used_since_observation"])
        self.assertEqual(saved["fail"], 2)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.context.receipt()["_submission_started"])

    def test_stopped_work_during_refresh_cancels_metadata_before_transport(self):
        self.admission.update_claim(self.context, _work_key="original-work")
        with self.store.transaction() as db:
            self.store.set_runtime(db, "original-work", {"state": "active", "slot_held": True})
        def cancelled(ref, **kwargs):
            with self.store.transaction() as db:
                self.store.set_runtime(db, "original-work", {"state": "paused", "slot_held": False})
            self.refresh(ref, **kwargs)
        self.accounts.refresh_image_capability = cancelled
        with self.assertRaises(AdmissionLost):
            self.generation()
        self.assertEqual(self.calls, [])
        self.assertFalse(self.context.receipt()["_submission_started"])
