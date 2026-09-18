"""Authorization attachment remains behind the existing internal admin bridge."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import owned_accounts
from api.errors import install_exception_handlers
from api.external_images import external_image_boundary
from services.auth_service import AuthService
from services.storage.json_storage import JSONStorageBackend


class CodexAuthorizationApiTests(unittest.TestCase):
    route = "/api/workbench/ai/pool/codex-authorization"
    body = {"access_token": "private-access", "refresh_token": "private-refresh",
            "id_token": "private-id", "account_id": "12345678-1234-5678-9234-567812345678"}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        auth = AuthService(JSONStorageBackend(Path(temporary.name) / "accounts.json"))
        _, self.admin = auth.create_key(role="admin", name="management")
        _, self.user = auth.create_key(role="user", name="program")
        self.accounts = Mock()
        self.logins = Mock()
        for target, value in (("api.support.auth_service", auth),
                              ("api.owned_accounts.account_service", self.accounts),
                              ("api.owned_accounts.codex_login_service", self.logins)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        app = FastAPI()
        app.middleware("http")(external_image_boundary)
        install_exception_handlers(app)
        app.include_router(owned_accounts.create_router())
        self.client = TestClient(app)
        self.trusted = {"Authorization": "Bearer " + self.admin,
                        "X-Workbench-Account-Owner": "workbench:org:boss"}

    def test_only_admin_bridge_can_attach_and_response_contains_no_material(self):
        self.accounts.attach_codex_authorization.return_value = {"attached": True, **self.body}
        response = self.client.post(self.route, headers=self.trusted, json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"attached": True})
        self.accounts.attach_codex_authorization.assert_called_once_with(self.body, None)

    def test_selected_pool_and_owned_file_attachment_forward_only_safe_target(self):
        account_ref = "car_" + "A" * 43
        selected = self.client.post(
            self.route,
            headers=self.trusted,
            json={**self.body, "account_ref": account_ref},
        )
        self.assertEqual(selected.status_code, 200)
        self.accounts.attach_codex_authorization.assert_called_once_with(self.body, account_ref)

        self.accounts.reset_mock()
        owned = self.client.post(
            "/api/workbench/ai/accounts/owned-account/codex-authorization",
            headers=self.trusted,
            json=self.body,
        )
        self.assertEqual(owned.status_code, 200)
        self.accounts.attach_owned_codex_authorization.assert_called_once_with(
            "workbench:org:boss", "owned-account", self.body
        )

    def test_login_routes_preserve_owner_scope_and_never_add_internal_material(self):
        request_id = "12345678-1234-5678-9234-567812345678"
        public = {
            "id": "opaque-session",
            "state": "pending",
            "verification_url": "https://auth.openai.com/codex/device",
            "user_code": "ABCD-EFGH",
            "expires_at": "2026-09-18T00:15:00+00:00",
            "poll_after_seconds": 5,
        }
        self.logins.start.return_value = public
        response = self.client.post(
            "/api/workbench/ai/codex-login",
            headers=self.trusted,
            json={"client_request_id": request_id, "mode": "import"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), public)
        self.logins.start.assert_called_once_with(
            "workbench:org:boss", "owned", "import", request_id, None
        )
        self.assertNotIn("token", response.text.lower())
        self.assertNotIn("verifier", response.text.lower())

        self.logins.get.return_value = {**public, "state": "succeeded"}
        read = self.client.get(
            "/api/workbench/ai/codex-login/opaque-session",
            headers={**self.trusted, "X-Workbench-Account-Owner": "workbench:org:other"},
        )
        self.assertEqual(read.status_code, 200)
        self.logins.get.assert_called_once_with(
            "workbench:org:other", "owned", "opaque-session"
        )

        self.logins.cancel.return_value = {**public, "state": "cancelled"}
        cancelled = self.client.delete(
            "/api/workbench/ai/pool/codex-login/opaque-session",
            headers=self.trusted,
        )
        self.assertEqual(cancelled.status_code, 200)
        self.logins.cancel.assert_called_once_with(
            "workbench:org:boss", "pool", "opaque-session"
        )

    def test_normal_key_public_ingress_and_missing_owner_cannot_attach(self):
        for headers, status in (
            ({}, 401),
            ({**self.trusted, "Authorization": "Bearer " + self.user}, 403),
            ({"Authorization": "Bearer " + self.admin}, 400),
            ({**self.trusted, "X-Workbench-Image-Client": "1"}, 404),
        ):
            with self.subTest(status=status):
                response = self.client.post(self.route, headers=headers, json=self.body)
                self.assertEqual(response.status_code, status)
        self.accounts.attach_codex_authorization.assert_not_called()

    def test_conflict_and_validation_never_echo_credentials(self):
        self.accounts.attach_codex_authorization.side_effect = ValueError(str(self.body))
        response = self.client.post(self.route, headers=self.trusted, json=self.body)
        self.assertEqual(response.status_code, 409)
        self.assertNotIn("private-", response.text)
        response = self.client.post(self.route, headers=self.trusted,
                                    json={**self.body, "managed_owner": "private-owner"})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("private-", response.text)


if __name__ == "__main__":
    unittest.main()
