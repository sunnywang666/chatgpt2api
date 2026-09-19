"""Focused account-center safety and state regressions."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

# The Provider module constructs service singletons during import. Isolate them
# before importing any service so this harness cannot read or write local data.
_GLOBAL_DATA_DIR = tempfile.mkdtemp(prefix="provider-ai-account-center-")
os.environ.setdefault("PROVIDER_DATA_DIR", _GLOBAL_DATA_DIR)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import owned_accounts
from api.errors import install_exception_handlers
from services.account_service import AccountService, CodexAuthorizationAttachError
from services.auth_service import AuthService
from services.codex_service import CodexService
from services.model_service import ModelCatalogService
from services.storage.json_storage import JSONStorageBackend


ACCOUNT_ID = "12345678-1234-5678-9234-567812345678"
OTHER_ACCOUNT_ID = "87654321-4321-6789-9234-567812345678"
SUBJECT = "account-center-subject"


def jwt(marker: str, *, subject: str = SUBJECT, account_id: str = ACCOUNT_ID) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({
        "sub": subject,
        "exp": int(time.time()) + 7 * 86400,
        "marker": marker,
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    }).encode()).decode().rstrip("=")
    return f"{header}.{payload}.signature"


def credentials(marker: str, *, subject: str = SUBJECT, account_id: str = ACCOUNT_ID) -> dict[str, str]:
    return {
        "access_token": jwt(marker + "-access", subject=subject, account_id=account_id),
        "refresh_token": marker + "-refresh-secret",
        "id_token": jwt(marker + "-id", subject=subject, account_id=account_id),
        "account_id": account_id,
    }


class AccountCenterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.storage = JSONStorageBackend(Path(self.tmp.name) / "accounts.json")
        self.accounts = AccountService(self.storage)

    def add_dual_account(self) -> tuple[str, dict[str, str]]:
        codex = credentials("codex")
        self.accounts.add_account_items([{
            **codex,
            "source_type": "codex",
            "email": "secret.person@example.test",
            "managed_owner": "workbench:org:original",
            "managed_account_id": "original-row",
            "quota": 7,
            "codex_credentials": codex,
            "codex_affinities": {"keep": {"state": "bound"}},
            "task_receipts": {"keep": "receipt"},
            "codex_observation": {
                "state": "observed",
                "observed_at": "2026-09-19T12:00:00+00:00",
                "models": [{"id": "gpt-codex", "label": "Codex", "reasoning_efforts": ["high"]}],
                "limits": [],
            },
        }])
        ref = self.accounts.list_pool_accounts()[0]["account_ref"]
        return ref, codex

    def test_projection_mutations_use_stable_ref_and_never_expose_secrets(self) -> None:
        ref, codex = self.add_dual_account()
        first = self.accounts.list_pool_accounts()[0]
        self.assertEqual(first["id"], ref)
        self.assertEqual(first["account_ref"], ref)
        self.assertEqual(first["authorization_ref"], ref)
        self.assertEqual(first["identity_label"], "s***@example.test")
        self.assertEqual(first["chat"]["authorization_status"], "missing")
        self.assertEqual(first["codex"]["authorization_status"], "saved")
        self.assertNotIn(codex["access_token"], json.dumps(first))
        self.assertNotIn("secret.person", json.dumps(first))

        labeled = self.accounts.set_pool_account_label(ref, "  Main AI  ")
        self.assertEqual(labeled["label"], "Main AI")
        disabled = self.accounts.set_pool_account_enabled(ref, False)
        self.assertFalse(disabled["enabled"])
        self.assertEqual(self.accounts.list_accounts()[0]["quota"], 7)
        enabled = self.accounts.set_pool_account_enabled(ref, True)
        self.assertTrue(enabled["enabled"])
        self.assertEqual(enabled["connection_status"], "unavailable")
        self.assertEqual(self.accounts.list_accounts()[0]["quota"], 7)
        saved = self.accounts.list_accounts()[0]
        self.assertEqual(AccountService.pool_account_ref(saved), ref)
        self.assertEqual(saved["codex_affinities"], {"keep": {"state": "bound"}})
        self.assertEqual(saved["task_receipts"], {"keep": "receipt"})
        self.assertEqual(saved["managed_owner"], "workbench:org:original")

    def test_repeated_enable_is_idempotent_and_reenable_waits_for_refresh(self) -> None:
        ref, _codex = self.add_dual_account()
        before = self.storage.file_path.read_bytes()
        unchanged = self.accounts.set_pool_account_enabled(ref, True)
        self.assertEqual(unchanged["connection_status"], "connected")
        self.assertEqual(self.storage.file_path.read_bytes(), before)
        self.assertEqual(self.accounts.list_accounts()[0]["quota"], 7)

        self.accounts.set_pool_account_enabled(ref, False)
        enabled = self.accounts.set_pool_account_enabled(ref, True)
        self.assertEqual(enabled["connection_status"], "unavailable")
        self.assertEqual(self.accounts.list_accounts()[0]["status"], "禁用")
        self.assertEqual(self.accounts.list_accounts()[0]["quota"], 7)

        def observe(token):
            account = self.accounts.get_account(token)
            self.accounts.update_account(token, {
                "codex_observation": {
                    **account["codex_observation"],
                    "state": "observed",
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                },
            }, quiet=True)
            return {"state": "observed"}

        with patch("services.codex_service.codex_service.refresh_account", side_effect=observe):
            refreshed = self.accounts.refresh_pool_account(ref, ["codex"], False)
        self.assertEqual(refreshed["connection_status"], "connected")
        self.assertEqual(self.accounts.list_accounts()[0]["status"], "正常")
        self.assertEqual(self.accounts.list_accounts()[0]["quota"], 7)

    def test_identityless_legacy_pool_ref_does_not_authorize_codex_merge(self) -> None:
        self.accounts.add_account_items([{
            "access_token": "opaque-legacy-chat-token",
            "source_type": "web",
            "status": "正常",
            "quota": 5,
        }])
        before = self.accounts.list_pool_accounts()[0]
        ref = before["account_ref"]
        self.assertEqual(before["authorization_ref"], ref)
        self.assertIsNone(AccountService.codex_authorization_ref(self.accounts.list_accounts()[0]))

        codex = credentials("legacy-attach")
        codex_identity_ref = AccountService._authorization_ref(
            *AccountService._codex_identity(codex)
        )
        self.assertNotEqual(ref, codex_identity_ref)
        persisted_before = self.storage.file_path.read_bytes()
        with self.assertRaisesRegex(CodexAuthorizationAttachError, "account_conflict"):
            self.accounts.attach_codex_authorization(codex, ref)

        after = self.accounts.list_pool_accounts()[0]
        self.assertEqual(after["account_ref"], ref)
        self.assertEqual(after["authorization_ref"], ref)
        self.assertEqual(after["codex"]["authorization_status"], "missing")
        self.assertEqual(self.accounts.list_accounts()[0]["managed_pool_account_ref"], ref)
        self.assertEqual(self.storage.file_path.read_bytes(), persisted_before)

    def test_verified_matching_attach_preserves_distinct_persisted_pool_ref(self) -> None:
        chat = credentials("verified-chat")
        persisted_ref = "car_" + "A" * 43
        self.accounts.add_account_items([{
            **chat,
            "source_type": "web",
            "managed_pool_account_ref": persisted_ref,
        }])
        self.assertNotEqual(
            persisted_ref,
            AccountService._authorization_ref(*AccountService._codex_identity(chat)),
        )
        codex = credentials("verified-codex")
        self.accounts.attach_codex_authorization(codex, persisted_ref)

        after = self.accounts.list_pool_accounts()[0]
        self.assertEqual(after["account_ref"], persisted_ref)
        self.assertEqual(after["authorization_ref"], persisted_ref)
        self.assertEqual(after["codex"]["authorization_status"], "saved")

    def test_codex_only_refresh_never_calls_chat_with_codex_token(self) -> None:
        ref, _codex = self.add_dual_account()
        with patch.object(self.accounts, "_verified_chat_info", side_effect=AssertionError("Codex token reached Chat")), \
                patch("services.codex_service.codex_service.refresh_account", return_value={"state": "observed"}) as refresh:
            result = self.accounts.refresh_pool_account(ref, ["chat", "codex"], False)
        self.assertEqual(result["chat"]["authorization_status"], "missing")
        refresh.assert_called_once()

    def test_same_stale_refresh_is_coalesced_per_account(self) -> None:
        ref, _codex = self.add_dual_account()
        started = threading.Event()
        release = threading.Event()
        calls = 0

        def once(account_ref, routes, stale_only):
            nonlocal calls
            calls += 1
            started.set()
            self.assertTrue(release.wait(3))
            return {"account_ref": account_ref, "routes": routes, "stale_only": stale_only}

        with patch.object(self.accounts, "_refresh_pool_account_once", side_effect=once):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(self.accounts.refresh_pool_account, ref, ["chat", "codex"], True)
                self.assertTrue(started.wait(2))
                second = executor.submit(self.accounts.refresh_pool_account, ref, ["chat", "codex"], True)
                release.set()
                self.assertEqual(first.result(), second.result())
        self.assertEqual(calls, 1)

    def test_chat_refresh_cas_discards_observation_after_token_rotation(self) -> None:
        original = credentials("chat-cas")
        self.accounts.add_account_items([{
            **original,
            "source_type": "web",
            "user_id": SUBJECT,
            "capacity_observed_at": "2026-09-18T00:00:00+00:00",
            "limits_progress": [{"feature_name": "image_gen", "remaining": 2}],
        }])
        ref = self.accounts.list_pool_accounts()[0]["account_ref"]
        started = threading.Event()
        release = threading.Event()
        fresh_info = {
            "user_id": SUBJECT,
            "account_id": ACCOUNT_ID,
            "quota": 9,
            "limits_progress": [{"feature_name": "image_gen", "remaining": 9}],
        }

        def blocked(_token):
            started.set()
            self.assertTrue(release.wait(3))
            return (SUBJECT, ACCOUNT_ID), fresh_info

        with patch.object(self.accounts, "_verified_chat_info", side_effect=blocked):
            thread = threading.Thread(target=self.accounts._refresh_pool_chat, args=(ref,), daemon=True)
            thread.start()
            self.assertTrue(started.wait(2))
            rotated = jwt("chat-cas-rotated")
            self.accounts._apply_refreshed_tokens(original["access_token"], {"access_token": rotated}, "test")
            release.set()
            thread.join(3)
            self.assertFalse(thread.is_alive())
        current = self.accounts.get_account(rotated)
        self.assertEqual(current["limits_progress"][0]["remaining"], 2)
        self.assertEqual(current["capacity_observed_at"], "2026-09-18T00:00:00+00:00")

    def test_targeted_chat_import_preserves_codex_and_rejects_identity_mismatch(self) -> None:
        ref, codex = self.add_dual_account()
        chat = credentials("chat")
        info = {"user_id": SUBJECT, "account_id": ACCOUNT_ID, "quota": 4, "limits_progress": []}
        with patch.object(self.accounts, "_request_access_token_refresh", return_value=chat), \
                patch.object(self.accounts, "_verified_chat_info", return_value=((SUBJECT, ACCOUNT_ID), info)) as verify:
            receipt = self.accounts.import_owned_account(
                "workbench:org:boss",
                {**chat, "source_type": "web", "account_ref": ref},
            )
        self.assertEqual(receipt["import_status"], "updated")
        self.assertEqual(receipt["authorization_ref"], ref)
        verify.assert_called_once_with(chat["access_token"])
        saved = self.accounts.list_accounts()[0]
        self.assertEqual(saved["source_type"], "web")
        self.assertEqual(saved["codex_credentials"], codex)
        self.assertEqual(saved["managed_owner"], "workbench:org:original")
        self.assertEqual(saved["codex_affinities"], {"keep": {"state": "bound"}})
        self.assertEqual(AccountService.pool_account_ref(saved), ref)

        before = self.storage.file_path.read_bytes()
        mismatched = credentials("wrong", subject="other-subject", account_id=OTHER_ACCOUNT_ID)
        with self.assertRaisesRegex(CodexAuthorizationAttachError, "account_conflict"):
            self.accounts.import_owned_account(
                "workbench:org:boss",
                {**mismatched, "source_type": "web", "account_ref": ref},
            )
        self.assertEqual(self.storage.file_path.read_bytes(), before)

    def test_internal_routes_require_admin_and_trusted_owner(self) -> None:
        ref, _codex = self.add_dual_account()
        auth = AuthService(self.storage)
        _user_item, user = auth.create_key(role="user", name="user", owner_subject="workbench:org:a")
        _admin_item, admin = auth.create_key(role="admin", name="admin")
        with patch("api.support.auth_service", auth), patch("api.owned_accounts.account_service", self.accounts):
            app = FastAPI()
            install_exception_handlers(app)
            app.include_router(owned_accounts.create_router())
            client = TestClient(app)
            route = f"/api/workbench/ai/pool/accounts/{ref}/label"
            owner = {"X-Workbench-Account-Owner": "workbench:org:boss"}
            self.assertEqual(client.post(route, headers={**owner, "Authorization": "Bearer " + user}, json={"label": "x"}).status_code, 403)
            self.assertEqual(client.post(route, headers={"Authorization": "Bearer " + admin}, json={"label": "x"}).status_code, 400)
            updated = client.post(route, headers={**owner, "Authorization": "Bearer " + admin}, json={"label": "Boss label"})
            self.assertEqual(updated.status_code, 200, updated.text)
            self.assertEqual(updated.json()["item"]["label"], "Boss label")
            invalid = client.post(route, headers={**owner, "Authorization": "Bearer " + admin}, json={"label": "x" * 81})
            self.assertEqual(invalid.status_code, 422)


class _CatalogBackend:
    def __init__(self, access_token: str) -> None:
        self.access_token = access_token

    def list_models(self) -> dict:
        model = "anonymous" if not self.access_token else "gpt-chat"
        return {"data": [{"id": model}]}

    def close(self) -> None:
        return None


class ManagementModelTests(unittest.TestCase):
    def test_type_catalog_keeps_account_observation_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            accounts = AccountService(JSONStorageBackend(Path(directory) / "accounts.json"))
            accounts.add_account_items([{"access_token": "chat-token", "source_type": "web", "type": "Plus", "status": "正常"}])
            accounts.refresh_access_token = lambda token, **_kwargs: token
            catalog = ModelCatalogService(accounts, backend_factory=lambda access_token="": _CatalogBackend(access_token))
            item = next(item for item in catalog.management_models() if item["id"] == "gpt-chat")
            self.assertEqual(item["supported_accounts"], 1)
            self.assertIsNone(item["available_accounts"])
            self.assertEqual(item["pending_accounts"], 1)
            self.assertEqual(item["accounts"][0]["state"], "unknown")
            self.assertEqual(item["accounts"][0]["reason"], "account_type_catalog_only")

    def test_chat_catalog_never_uses_codex_only_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            accounts = AccountService(JSONStorageBackend(Path(directory) / "accounts.json"))
            codex = credentials("catalog-codex")
            accounts.add_account_items([{
                **codex,
                "source_type": "codex",
                "type": "Plus",
                "status": "正常",
                "codex_credentials": codex,
            }])
            calls = []

            def backend(access_token=""):
                calls.append(access_token)
                self.assertNotEqual(access_token, codex["access_token"])
                return _CatalogBackend(access_token)

            catalog = ModelCatalogService(accounts, backend_factory=backend)
            catalog.management_models()
            self.assertEqual(calls, [""])


if __name__ == "__main__":
    unittest.main()
