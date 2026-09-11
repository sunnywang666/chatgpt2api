from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "test-auth")

from services.account_service import AccountService
from services.auth_service import AuthService
from services.config import config
from services.openai_backend_api import InvalidAccessTokenError
from services.storage.json_storage import JSONStorageBackend
from utils.helper import anonymize_token, split_image_model


class AccountCapabilityTests(unittest.TestCase):
    def test_product_binding_requires_one_account_capable_of_text_and_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items([
                {"access_token": "free", "type": "Free", "status": "正常", "quota": 100},
                {"access_token": "pro", "type": "Pro", "status": "正常", "quota": 100},
            ])
            service.fetch_remote_info = lambda token, event="": service.get_account(token)
            with patch("services.model_service.model_catalog_service.route_for_model", return_value=SimpleNamespace(account_types=frozenset({"Pro"}))):
                for _ in range(4):
                    _, _, token = service.create_conversation_binding(image_model="gpt-image-2", text_model="gpt-5-6-instant")
                    self.assertEqual(token, "pro")
                    service.release_image_slot(token)
            with patch("services.model_service.model_catalog_service.route_for_model", return_value=SimpleNamespace(account_types=frozenset())):
                with self.assertRaisesRegex(RuntimeError, "supports both"):
                    service.create_conversation_binding(image_model="gpt-image-2", text_model="unavailable")

    def test_product_messages_never_use_free_even_when_model_allows_it(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items([
                {"access_token": "free", "type": "Free", "status": "正常", "quota": 100},
                {"access_token": "pro", "type": "Pro", "status": "正常", "quota": 100},
            ])
            service.fetch_remote_info = lambda token, event="": service.get_account(token)
            service.refresh_access_token = lambda token, event="": token
            for model in ("auto", "free-compatible"):
                with patch("services.model_service.model_catalog_service.route_for_model", return_value=SimpleNamespace(account_types=frozenset({"free", "Pro"}))):
                    binding, _, token = service.create_conversation_binding(image_model="gpt-image-2", text_model=model)
                    self.assertEqual(token, "pro")
                    service.release_image_slot(token)
            # A subscription downgrade / legacy Free binding cannot send a
            # continuation or image, but original answers remain readable.
            service.update_account("pro", {"type": "Free"})
            self.assertEqual(service.get_bound_text_access_token(binding, model="auto"), "pro")
            with self.assertRaisesRegex(RuntimeError, "paid account required"):
                service.get_bound_text_access_token(binding, model="auto", for_message=True)
            with self.assertRaisesRegex(RuntimeError, "cannot generate images"):
                service.acquire_bound_image_access_token(binding, image_model="gpt-image-2")
            with self.assertRaises(RuntimeError):
                service.create_conversation_binding(image_model="gpt-image-2")
            self.assertFalse(any(service._image_inflight.values()))

    def test_free_account_transport_rejects_messages_but_allows_result_queries(self):
        from services.account_request_pacing import pace_account_session, AccountRequestClock
        from unittest.mock import Mock
        send = Mock(return_value="original-result")
        session = SimpleNamespace(request=send)
        with patch.object(AccountRequestClock, "request", side_effect=lambda fn, method, url, **kw: fn(method, url, **kw)):
            pace_account_session(session, {"type": "Free"}, "free-test-token")
            for path in ("/backend-api/f/conversation", "/backend-api/conversation", "/backend-api/f/conversation/prepare", "/backend-api/codex/responses"):
                with self.assertRaisesRegex(RuntimeError, "Free account messages are disabled"):
                    session.request("POST", "https://chatgpt.com" + path)
            send.assert_not_called()
            self.assertEqual(session.request("GET", "https://chatgpt.com/backend-api/conversation/original"), "original-result")
            self.assertEqual(send.call_count, 1)

    def test_conversation_binding_pins_one_account_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items(
                [
                    {"access_token": "token-a", "type": "Pro", "status": "正常", "quota": 3},
                    {"access_token": "token-b", "type": "Pro", "status": "正常", "quota": 3},
                ]
            )
            service.fetch_remote_info = (
                lambda access_token, event="fetch_remote_info": service.get_account(access_token)
            )

            binding_id, account_identity, first_token = service.create_conversation_binding(
                image_model="gpt-image-2"
            )
            service.release_image_slot(first_token)
            bound_token = service.acquire_bound_image_access_token(
                binding_id,
                image_model="gpt-image-2",
            )
            service.release_image_slot(bound_token)

            self.assertTrue(binding_id.startswith("cb_"))
            self.assertEqual(bound_token, first_token)
            self.assertEqual(service.get_bound_account_identity(binding_id), account_identity)
            service.update_account(first_token, {"status": "异常", "quota": 0})
            with self.assertRaisesRegex(RuntimeError, "conversation binding unavailable"):
                service.acquire_bound_image_access_token(
                    binding_id,
                    image_model="gpt-image-2",
                )

    def test_separate_conversations_get_distinct_bindings_on_the_same_account(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items(
                [{"access_token": "token-a", "type": "Pro", "status": "正常", "quota": 3}]
            )
            service.fetch_remote_info = (
                lambda access_token, event="fetch_remote_info": service.get_account(access_token)
            )

            first_binding, _, first_token = service.create_conversation_binding(
                image_model="gpt-image-2"
            )
            service.release_image_slot(first_token)
            second_binding, _, second_token = service.create_conversation_binding(
                image_model="gpt-image-2"
            )
            service.release_image_slot(second_token)

            self.assertNotEqual(first_binding, second_binding)
            self.assertEqual(first_token, second_token)
            self.assertIsNot(
                service.conversation_binding_lock(first_binding),
                service.conversation_binding_lock(second_binding),
            )

    def test_image_accounts_require_positive_quota(self) -> None:
        self.assertFalse(
            AccountService._is_image_account_available(
                {"status": "限流", "quota": 1}
            )
        )
        self.assertFalse(
            AccountService._is_image_account_available(
                {"status": "正常", "quota": 0}
            )
        )
        self.assertTrue(AccountService._is_image_account_available({"status": "正常", "quota": 1}))

    def test_prolite_variants_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            self.assertEqual(service._normalize_account_type("prolite"), "ProLite")
            self.assertEqual(service._normalize_account_type("pro_lite"), "ProLite")

    def test_search_account_type_ignores_unrelated_scalar_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            self.assertIsNone(
                service._search_account_type(
                    {
                        "amr": ["pwd", "otp", "mfa"],
                        "chatgpt_compute_residency": "no_constraint",
                        "chatgpt_data_residency": "no_constraint",
                        "user_id": "user-I52GFfLGFM0dokFk2dBiKEBn",
                    }
                )
            )

    def test_mark_image_result_consumes_quota(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_accounts(["token-1"])
            service.update_account(
                "token-1",
                {
                    "status": "正常",
                    "quota": 1,
                },
            )

            updated = service.mark_image_result("token-1", success=True)

            self.assertIsNotNone(updated)
            self.assertEqual(updated["quota"], 0)
            self.assertEqual(updated["status"], "限流")

    def test_split_image_model_supports_plan_type_prefix(self) -> None:
        self.assertEqual(split_image_model("gpt-image-2"), (None, "gpt-image-2"))
        self.assertEqual(split_image_model("plus-codex-gpt-image-2"), ("plus", "codex-gpt-image-2"))
        self.assertEqual(split_image_model("team-codex-gpt-image-2"), ("team", "codex-gpt-image-2"))
        self.assertEqual(split_image_model("pro-codex-gpt-image-2"), ("pro", "codex-gpt-image-2"))
        self.assertEqual(split_image_model("plus-gpt-image-2"), (None, None))
        self.assertEqual(split_image_model("unknown-image-model"), (None, None))

    def test_get_available_access_token_filters_by_plan_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items(
                [
                    {"access_token": "token-plus", "type": "Plus", "status": "正常", "quota": 3},
                    {"access_token": "token-pro", "type": "Pro", "status": "正常", "quota": 3},
                ]
            )

            service.fetch_remote_info = lambda access_token, event="fetch_remote_info": service.get_account(access_token)

            plus_token = service.get_available_access_token(plan_type="plus")
            pro_token = service.get_available_access_token(plan_type="pro")
            service.release_image_slot(plus_token)
            service.release_image_slot(pro_token)

            self.assertEqual(plus_token, "token-plus")
            self.assertEqual(pro_token, "token-pro")

    def test_refresh_accounts_can_remove_invalid_token_without_confirmation_delay(self) -> None:
        original_value = config.data.get("auto_remove_invalid_accounts")
        config.data["auto_remove_invalid_accounts"] = True
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                service.add_account_items([{"access_token": "invalid-token", "status": "正常"}])

                with patch(
                    "services.openai_backend_api.OpenAIBackendAPI.get_user_info",
                    side_effect=InvalidAccessTokenError("token invalidated (/backend-api/me)"),
                ):
                    result = service.refresh_accounts(["invalid-token"], defer_invalid_removal=False)

                self.assertEqual(result["refreshed"], 0)
                self.assertEqual(len(result["errors"]), 1)
                self.assertEqual(result["items"], [])
                self.assertIsNone(service.get_account("invalid-token"))
        finally:
            if original_value is None:
                config.data.pop("auto_remove_invalid_accounts", None)
            else:
                config.data["auto_remove_invalid_accounts"] = original_value

    def test_refresh_accounts_defers_invalid_token_removal_by_default(self) -> None:
        original_value = config.data.get("auto_remove_invalid_accounts")
        config.data["auto_remove_invalid_accounts"] = True
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                service.add_account_items([{"access_token": "invalid-token", "status": "正常"}])

                with patch(
                    "services.openai_backend_api.OpenAIBackendAPI.get_user_info",
                    side_effect=InvalidAccessTokenError("token invalidated (/backend-api/me)"),
                ):
                    result = service.refresh_accounts(["invalid-token"])

                account = service.get_account("invalid-token")
                self.assertEqual(result["refreshed"], 0)
                self.assertEqual(len(result["errors"]), 1)
                self.assertIsNotNone(account)
                self.assertEqual(account["invalid_count"], 1)
        finally:
            if original_value is None:
                config.data.pop("auto_remove_invalid_accounts", None)
            else:
                config.data["auto_remove_invalid_accounts"] = original_value


class TokenLogTests(unittest.TestCase):
    def test_anonymize_token_hides_raw_value(self) -> None:
        token = "super-secret-token"
        token_ref = anonymize_token(token)

        self.assertTrue(token_ref.startswith("token:"))
        self.assertNotIn(token, token_ref)


class AuthServiceTests(unittest.TestCase):
    def test_create_authenticate_disable_and_delete_user_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))

            item, raw_key = service.create_key(role="user", name="Alice")

            self.assertEqual(item["role"], "user")
            self.assertEqual(item["name"], "Alice")
            self.assertTrue(item["enabled"])
            self.assertTrue(raw_key.startswith("sk-"))

            authed = service.authenticate(raw_key)
            self.assertIsNotNone(authed)
            self.assertEqual(authed["id"], item["id"])
            self.assertEqual(authed["role"], "user")
            self.assertIsNotNone(authed["last_used_at"])

            updated = service.update_key(item["id"], {"enabled": False}, role="user")
            self.assertIsNotNone(updated)
            self.assertFalse(updated["enabled"])
            self.assertIsNone(service.authenticate(raw_key))

            self.assertTrue(service.delete_key(item["id"], role="user"))
            self.assertFalse(service.delete_key(item["id"], role="user"))
            self.assertEqual(service.list_keys(role="user"), [])

    def test_authenticate_ignores_last_used_save_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            item, raw_key = service.create_key(role="user", name="Alice")

            def fail_save() -> None:
                raise OSError("disk unavailable")

            service._save = fail_save

            authed = service.authenticate(raw_key)

            self.assertIsNotNone(authed)
            self.assertEqual(authed["id"], item["id"])
            self.assertIsNotNone(authed["last_used_at"])

    def test_update_user_key_replaces_raw_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            item, raw_key = service.create_key(role="user", name="Alice")

            updated = service.update_key(item["id"], {"key": "sk-user-custom-key"}, role="user")

            self.assertIsNotNone(updated)
            self.assertIsNone(service.authenticate(raw_key))

            authed = service.authenticate("sk-user-custom-key")
            self.assertIsNotNone(authed)
            self.assertEqual(authed["id"], item["id"])

    def test_user_key_name_must_be_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            first, _ = service.create_key(role="user", name="Alice")
            second, _ = service.create_key(role="user", name="Bob")

            with self.assertRaisesRegex(ValueError, "这个名称已经在使用中了"):
                service.create_key(role="user", name="Alice")

            with self.assertRaisesRegex(ValueError, "这个名称已经在使用中了"):
                service.update_key(second["id"], {"name": "Alice"}, role="user")

            updated = service.update_key(first["id"], {"name": "Alice"}, role="user")
            self.assertIsNotNone(updated)
            self.assertEqual(updated["name"], "Alice")


if __name__ == "__main__":
    unittest.main()
