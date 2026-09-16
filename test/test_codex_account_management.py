"""Account-center integration: routes remain independent and secrets stay private."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from services.account_service import AccountService
from services.config import config
from services.storage.json_storage import JSONStorageBackend
from services.owned_accounts import public_owned_account


class CodexAccountManagementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.accounts = AccountService(JSONStorageBackend(Path(self.tmp.name) / "accounts.json"))

    def test_import_and_projection_keep_upstream_account_identity_private(self):
        result = self.accounts.import_owned_account("workbench:o:u", {
            "access_token": "private-access", "refresh_token": "private-refresh",
            "account_id": "private-account-id", "source_type": "codex",
        })
        self.assertEqual(self.accounts.get_account("private-access")["account_id"], "private-account-id")
        self.assertEqual(result["codex"]["state"], "unknown")
        self.assertIsNone(result["capacity"]["remaining"])
        self.assertNotIn("private-", str(result))
        self.assertEqual(self.accounts.list_owned_accounts("workbench:o:other"), [])

    def test_codex_oauth_uses_its_authorization_client_and_never_redirects(self):
        response = Mock(status_code=200, text='{}')
        response.json.return_value = {"access_token": "rotated", "refresh_token": "rotated-refresh"}
        session = Mock()
        session.post.return_value = response
        with patch("curl_cffi.requests.Session", return_value=session):
            self.accounts._request_access_token_refresh("secret", {"source_type": "codex"})
        kwargs = session.post.call_args.kwargs
        self.assertEqual(kwargs["data"]["client_id"], "app_EMoamEEZ73f0CkXaXp7hrann")
        self.assertFalse(kwargs["allow_redirects"])
        session.close.assert_called_once()

    def test_codex_oauth_error_never_returns_raw_upstream_secrets(self):
        response = Mock(status_code=400, text='private-error')
        response.json.return_value = {"error_description": "private-token-echo"}
        session = Mock()
        session.post.return_value = response
        with patch("curl_cffi.requests.Session", return_value=session):
            with self.assertRaisesRegex(RuntimeError, '^codex_oauth_refresh_http_400$'):
                self.accounts._request_access_token_refresh("secret", {"source_type": "codex"})

    def test_metadata_observation_does_not_delete_image_limited_account(self):
        self.accounts.add_account_items([{"access_token": "token", "status": "正常"}])
        with patch.object(type(config), "auto_remove_rate_limited_accounts", new_callable=lambda: property(lambda _: False)):
            self.accounts.update_account("token", {"status": "限流"})
        with patch.object(type(config), "auto_remove_rate_limited_accounts", new_callable=lambda: property(lambda _: True)):
            self.accounts.update_account("token", {"codex_observation": {"state": "observed"}}, quiet=True)
        self.assertIsNotNone(self.accounts.get_account("token"))

    def test_codex_background_refresh_does_not_probe_image_route(self):
        self.accounts.import_owned_account("workbench:o:u", {"access_token": "token", "source_type": "codex"})
        with patch.object(self.accounts, "refresh_access_token", return_value="token"), patch("services.codex_service.codex_service.refresh_account", return_value={"state": "observed"}) as probe, patch("services.openai_backend_api.OpenAIBackendAPI") as web:
            result = self.accounts.fetch_remote_info("token")
        probe.assert_called_once_with("token")
        web.assert_not_called()
        self.assertEqual(result["status"], "正常")

    def test_codex_projection_has_no_internal_binding_or_token(self):
        result = public_owned_account({"managed_account_id": "managed", "codex_observation": {"state": "observed", "access_token": "private", "models": [], "limits": []}, "codex_affinities": {"private-key": "private-session"}})
        self.assertNotIn("private", str(result))

    def test_background_refresh_lists_exclude_both_forms_of_disabled_account(self):
        self.accounts.add_account_items([
            {"access_token": "active", "refresh_token": "refresh-active", "status": "正常"},
            {"access_token": "disabled-status", "refresh_token": "refresh-status", "status": "禁用"},
            {"access_token": "disabled-managed", "refresh_token": "refresh-managed", "managed_disabled": True},
        ])
        # Also cover legacy/inconsistent status metadata: the explicit disable
        # flag remains authoritative even if status has not caught up.
        self.accounts._accounts["disabled-managed"]["status"] = "正常"
        with patch.object(self.accounts, "_token_needs_refresh", return_value=True), \
                patch.object(self.accounts, "_refresh_token_keepalive_anchor", return_value=None):
            self.assertEqual(self.accounts.list_expiring_access_tokens(), ["active"])
            self.assertEqual(self.accounts.list_refresh_token_keepalive_tokens(), ["active"])

    def test_two_owners_cannot_read_refresh_or_disable_foreign_accounts(self):
        owned = {}
        for owner in ("workbench:o:a", "workbench:o:b"):
            owned[owner] = self.accounts.import_owned_account(owner, {
                "access_token": f"private-{owner}", "source_type": "codex",
            })
        for owner, foreign in (("workbench:o:a", "workbench:o:b"), ("workbench:o:b", "workbench:o:a")):
            items = self.accounts.list_owned_accounts(owner)
            self.assertEqual([item["id"] for item in items], [owned[owner]["id"]])
            self.assertNotIn("private-", str(items))
            with patch.object(self.accounts, "refresh_access_token") as refresh:
                with self.assertRaises(KeyError):
                    self.accounts.refresh_owned_account(owner, owned[foreign]["id"])
                refresh.assert_not_called()
            with self.assertRaises(KeyError):
                self.accounts.set_owned_account_enabled(owner, owned[foreign]["id"], False)


if __name__ == '__main__':
    unittest.main()
