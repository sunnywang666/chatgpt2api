"""Read-only identity display must not rebind accounts or expose credentials."""
import base64
from copy import deepcopy
import json
import os
import tempfile
import unittest

os.environ.setdefault("PROVIDER_DATA_DIR", tempfile.mkdtemp(prefix="account-display-test-"))

from services.owned_accounts import masked_identity, public_owned_account, public_pool_account


WORKSPACE = "12345678-1234-5678-9234-567812345678"
OTHER_WORKSPACE = "87654321-4321-6789-9234-567812345678"


def jwt(**payload):
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{body}.signature"


def account(*, subject="subject-one", workspace=WORKSPACE, name="icecream"):
    claims = {"sub": subject, "exp": 1, "https://api.openai.com/auth": {
        "chatgpt_account_id": workspace, "chatgpt_user_id": "user-one",
    }}
    return {
        "source_type": "codex", "account_id": workspace, "user_id": "user-one",
        "managed_account_id": "managed-one", "managed_owner": "original-owner",
        "access_token": jwt(**claims, **{"https://api.openai.com/profile": {"email": "hello@example.test"}}),
        "id_token": jwt(**claims, name=name, email="hello@example.test"),
        "refresh_token": "private-refresh", "password": "private-password",
    }


class AccountDisplayTests(unittest.TestCase):
    def test_codex_only_name_and_masked_email_survive_both_projections_without_writes(self):
        row = account()
        before = deepcopy(row)
        for project in (public_pool_account, public_owned_account):
            with self.subTest(project=project.__name__):
                result = project(row)
                self.assertEqual(result["label"], "icecream")
                self.assertEqual(result["identity_label"], "h***@example.test")
                self.assertEqual(result["chat"]["authorization_status"], "missing")
                self.assertEqual(result["codex"]["authorization_status"], "saved")
                encoded = json.dumps(result)
                for private in (row["access_token"], row["id_token"], "private-refresh", "private-password",
                                "hello@example.test", WORKSPACE, "subject-one", "original-owner"):
                    self.assertNotIn(private, encoded)
                self.assertEqual(row, before)

    def test_managed_label_and_stored_email_remain_authoritative(self):
        row = {**account(), "managed_label": "公司研发", "email": "verified@example.test"}
        result = public_pool_account(row)
        self.assertEqual(result["label"], "公司研发")
        self.assertEqual(result["identity_label"], "v***@example.test")

    def test_matching_attached_codex_can_fill_display_only(self):
        stored = account()
        row = {**stored, "source_type": "web", "codex_credentials": stored,
               "access_token": jwt(sub="subject-one", **{"https://api.openai.com/auth": {"chatgpt_account_id": WORKSPACE}}),
               "id_token": ""}
        self.assertEqual(public_pool_account(row)["label"], "icecream")
        self.assertEqual(masked_identity(row)["email"], "h***@example.test")
        row["codex_credentials"] = account(subject="someone-else", name="wrong-person")
        self.assertIsNone(masked_identity(row)["name"])
        self.assertIsNone(masked_identity(row)["email"])

    def test_incoherent_subject_workspace_or_user_never_contributes_profile(self):
        for change in (
            {"id_token": jwt(sub="someone-else", name="wrong-person")},
            {"account_id": OTHER_WORKSPACE},
            {"user_id": "different-chat-user"},
            {"access_token": "malformed", "id_token": jwt(sub="subject-one", name="wrong-person")},
        ):
            with self.subTest(change=list(change)):
                result = masked_identity({**account(), **change})
                self.assertIsNone(result["name"])
                self.assertIsNone(result["email"])

    def test_id_claims_without_subject_are_not_a_profile(self):
        row = account()
        row["id_token"] = jwt(name="wrong-person", email="wrong@example.test")
        result = masked_identity(row)
        self.assertIsNone(result["name"])
        self.assertEqual(result["email"], "h***@example.test")

    def test_missing_profile_keeps_legacy_fallback(self):
        result = public_pool_account({"access_token": "opaque", "source_type": "web", "account_id": WORKSPACE})
        self.assertEqual(result["label"], "1234…5678")
        self.assertEqual(result["identity_label"], "1234…5678")

    def test_email_as_name_is_masked_and_non_text_names_are_ignored(self):
        self.assertEqual(public_pool_account(account(name="hello@example.test"))["label"], "h***@example.test")
        for name in ({"secret": "value"}, ["value"], "bad\nname", "x" * 161):
            with self.subTest(name_type=type(name).__name__):
                self.assertIsNone(masked_identity(account(name=name))["name"])

    def test_same_profile_name_does_not_merge_different_accounts(self):
        first = public_pool_account(account())
        second = public_pool_account(account(subject="subject-two", workspace=OTHER_WORKSPACE))
        self.assertEqual(first["label"], second["label"])
        self.assertNotEqual(first["authorization_ref"], second["authorization_ref"])


if __name__ == "__main__":
    unittest.main()
