from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from services.account_service import AccountService, CodexAuthorizationAttachError
from services.codex_login_service import CodexLoginError, CodexLoginService
from services.codex_service import CODEX_MODELS_URL, CODEX_USAGE_URL, CodexService
from services.storage.json_storage import JSONStorageBackend


ACCOUNT_ID = "12345678-1234-5678-9234-567812345678"
OTHER_ACCOUNT_ID = "87654321-4321-6789-9234-567812345678"
SUBJECT = "login-flow-subject"


def jwt(subject: str = SUBJECT, account_id: str = ACCOUNT_ID, marker: str = "") -> str:
    payload = {
        "sub": subject,
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
        "jti": marker,
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def credentials(label: str, subject: str = SUBJECT, account_id: str = ACCOUNT_ID) -> dict[str, str]:
    return {
        "access_token": jwt(subject, account_id, f"access-{label}"),
        "refresh_token": f"refresh-{label}",
        "id_token": jwt(subject, account_id, f"id-{label}"),
        "account_id": account_id,
    }


class FakeResponse:
    def __init__(self, status: int, payload: dict | None = None, headers: dict | None = None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeHttp:
    def __init__(self, outcomes: list[object]):
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict]] = []
        self.closed = 0

    def factory(self):
        parent = self

        class Session:
            def post(self, url, **kwargs):
                parent.calls.append((url, kwargs))
                outcome = parent.outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

            def close(self):
                parent.closed += 1

        return Session()


class MutableClock:
    def __init__(self, now: float = 1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def device_start(interval: str = "1") -> FakeResponse:
    return FakeResponse(200, {
        "device_auth_id": "private-device-authorization",
        "user_code": "ABCD-EFGH",
        "interval": interval,
    })


def device_complete() -> FakeResponse:
    verifier = "v" * 64
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).decode().rstrip("=")
    return FakeResponse(200, {
        "authorization_code": "private-authorization-code",
        "code_verifier": verifier,
        "code_challenge": challenge,
    })


def token_exchange(value: dict[str, str]) -> FakeResponse:
    return FakeResponse(200, value)


class CodexLoginFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.accounts = AccountService(JSONStorageBackend(root / "accounts.json"))
        self.sessions_path = root / "codex_login_sessions.json"
        self.clock = MutableClock()
        self.probe = patch(
            "services.codex_service.codex_service.observe_import_authorization",
            return_value={"state": "observed", "observed_at": "2026-09-18T00:00:00+00:00",
                          "failed_at": None, "models": [], "limits": [], "error_code": None},
        )
        self.probe.start()
        self.addCleanup(self.probe.stop)
        # Fixture OAuth exchange: real refresh tokens are never sent from
        # tests, and each result is bound to the submitted fixture token.
        def fixture_refresh(refresh_token, _account=None, **_kwargs):
            self.assertTrue(refresh_token.startswith("refresh-"))
            return credentials(refresh_token.removeprefix("refresh-"))

        self.refresh_exchange = patch.object(
            self.accounts, "_request_access_token_refresh", side_effect=fixture_refresh
        )
        self.refresh_exchange.start()
        self.addCleanup(self.refresh_exchange.stop)

    def service(self, http: FakeHttp) -> CodexLoginService:
        return CodexLoginService(
            self.sessions_path,
            self.accounts,
            http.factory,
            clock=self.clock,
            sleeper=lambda _seconds: None,
            auto_start_workers=False,
        )

    def add_primary(self, owner: str = "workbench:org:one", label: str = "primary") -> dict:
        value = credentials(label)
        self.accounts.import_owned_account(owner, {**value, "source_type": "web"})
        return self.accounts.list_owned_accounts(owner)[0]

    def test_successful_import_is_durable_redacted_and_same_request_is_reused(self):
        imported = credentials("first")
        http = FakeHttp([device_start("2"), FakeResponse(403), device_complete(), token_exchange(imported)])
        service = self.service(http)
        request_id = str(uuid.uuid4())

        pending = service.start("workbench:org:one", "owned", "import", request_id)
        self.assertEqual(pending["state"], "pending")
        self.assertEqual(pending["verification_url"], CodexLoginService.VERIFICATION_URL)
        self.assertEqual(pending["user_code"], "ABCD-EFGH")
        self.assertNotIn("device_auth_id", pending)
        service._run(pending["id"])

        succeeded = service.get("workbench:org:one", "owned", pending["id"])
        self.assertEqual(succeeded["state"], "succeeded")
        self.assertTrue(succeeded["account_ref"].startswith("car_"))
        self.assertNotIn("user_code", succeeded)
        stored_text = self.sessions_path.read_text()
        for secret in imported.values():
            self.assertNotIn(secret, stored_text)
        for secret in ("private-device", "private-authorization-code", "v" * 64):
            self.assertNotIn(secret, stored_text)
        self.assertEqual(os.stat(self.sessions_path).st_mode & 0o777, 0o600)
        self.assertEqual(len(self.accounts.list_owned_accounts("workbench:org:one")), 1)

        call_count = len(http.calls)
        self.assertEqual(
            service.start("workbench:org:one", "owned", "import", request_id),
            succeeded,
        )
        self.assertEqual(len(http.calls), call_count)
        with self.assertRaisesRegex(CodexLoginError, "idempotency_conflict"):
            service.start(
                "workbench:org:one",
                "owned",
                "attach",
                request_id,
                succeeded["account_ref"],
            )

        self.assertEqual(http.calls[-1][0], CodexLoginService.OAUTH_TOKEN_URL)
        exchange_kwargs = http.calls[-1][1]
        self.assertIn("data", exchange_kwargs)
        self.assertNotIn("json", exchange_kwargs)
        self.assertFalse(exchange_kwargs["allow_redirects"])

    def test_same_identity_deduplicates_across_submitters_and_preserves_owner(self):
        first = credentials("first")
        item = self.accounts.import_owned_account(
            "workbench:org:one", {**first, "source_type": "codex"}
        )
        rotated = credentials("rotated")
        same = self.accounts.import_owned_account(
            "workbench:org:one", {**rotated, "source_type": "codex"}
        )
        self.assertEqual(same["authorization_ref"], item["authorization_ref"])
        self.assertEqual(same["import_status"], "updated")
        self.assertEqual(len(self.accounts.list_accounts()), 1)
        stored = self.accounts.list_accounts()[0]
        self.assertEqual(stored["access_token"], first["access_token"])
        self.assertEqual(stored["codex_credentials"], rotated)

        cross = self.accounts.import_owned_account(
            "workbench:org:two", {**credentials("other-owner"), "source_type": "codex"}
        )
        self.assertEqual(cross["import_status"], "updated")
        self.assertEqual(self.accounts.list_owned_accounts("workbench:org:two"), [])
        self.assertEqual(self.accounts.list_accounts()[0]["managed_owner"], "workbench:org:one")

        shared_accounts = AccountService(JSONStorageBackend(Path(self.tmp.name) / "shared.json"))
        shared_accounts.add_account_items([{**credentials("shared"), "source_type": "codex"}])
        with patch.object(shared_accounts, "_request_access_token_refresh", return_value=credentials("claim-shared")):
            shared = shared_accounts.import_owned_account(
                "workbench:org:one", {**credentials("claim-shared"), "source_type": "codex"}
            )
        self.assertEqual(shared["import_status"], "updated")
        self.assertIsNone(shared_accounts.list_accounts()[0].get("managed_owner"))

    def test_verified_cross_owner_import_keeps_chat_root_and_only_returns_submitted_observation(self):
        original = credentials("original")
        self.accounts.add_account_items([{
            **original, "source_type": "web", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "quota": 17,
            "codex_affinities": {"task": {"state": "bound"}},
            "codex_response_ids": {"response": {"owner": "original"}},
        }])
        self.accounts.storage.save_auth_keys([{"id": "original-key", "key_hash": "fixture-hash"}])
        key_bytes = self.accounts.storage.auth_keys_path.read_bytes()
        before = self.accounts.storage.load_accounts()[0]

        receipt = self.accounts.import_owned_account(
            "workbench:org:two", {**credentials("new"), "source_type": "codex"}
        )
        self.assertEqual(set(receipt), {"authorization_ref", "import_status", "codex"})
        self.assertEqual(receipt["import_status"], "updated")
        self.assertEqual(receipt["codex"]["state"], "observed")
        self.assertEqual(self.accounts.list_owned_accounts("workbench:org:two"), [])
        with self.assertRaises(KeyError):
            self.accounts.set_owned_account_enabled("workbench:org:two", "original-row", False)
        after = self.accounts.storage.load_accounts()[0]
        for field in ("access_token", "refresh_token", "id_token", "managed_owner", "managed_account_id", "quota", "codex_affinities", "codex_response_ids"):
            self.assertEqual(after[field], before[field])
        self.assertEqual(after["codex_credentials"], credentials("new"))
        self.assertEqual(self.accounts.storage.auth_keys_path.read_bytes(), key_bytes)
        self.assertEqual(self.accounts.refresh_submitted_codex_observation(
            "workbench:org:two", receipt["authorization_ref"]
        )["state"], "observed")
        with self.assertRaisesRegex(CodexAuthorizationAttachError, "account_not_found"):
            self.accounts.refresh_submitted_codex_observation("workbench:org:three", receipt["authorization_ref"])
        refreshed = credentials("trusted-refresh")
        with patch.object(self.accounts, "_request_access_token_refresh", return_value=refreshed):
            self.accounts.refresh_codex_access_token(original["access_token"], force=True)
        self.assertEqual(self.accounts.refresh_submitted_codex_observation(
            "workbench:org:two", receipt["authorization_ref"]
        )["state"], "observed")
        self.accounts.import_owned_codex_authorization("workbench:org:one", credentials("later-owner-rotation"))
        with self.assertRaisesRegex(CodexAuthorizationAttachError, "account_not_found"):
            self.accounts.refresh_submitted_codex_observation("workbench:org:two", receipt["authorization_ref"])

    def test_forged_claims_without_upstream_proof_cannot_replace_foreign_credentials(self):
        original = credentials("original")
        self.accounts.add_account_items([{**original, "managed_owner": "workbench:org:one", "managed_account_id": "row"}])
        before = self.accounts.storage.file_path.read_bytes()
        failed = {"state": "read_failed", "observed_at": None, "failed_at": "now", "models": [], "limits": [], "error_code": "usage_unverified"}
        with patch("services.codex_service.codex_service.observe_import_authorization", return_value=failed), \
                patch.object(self.accounts, "_request_access_token_refresh", side_effect=RuntimeError("oauth rejected")):
            with self.assertRaisesRegex(CodexAuthorizationAttachError, "refresh_unverified"):
                self.accounts.import_owned_account("workbench:org:two", {**credentials("forged"), "source_type": "codex"})
        self.assertEqual(self.accounts.storage.file_path.read_bytes(), before)

    def test_foreign_manual_codex_refresh_must_be_exchanged_before_replacement(self):
        old = credentials("owner")
        self.accounts.add_account_items([{
            **old, "source_type": "codex", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "codex_credentials": old,
        }])
        before = self.accounts.storage.file_path.read_bytes()
        incoming = credentials("submitted")
        with patch.object(self.accounts, "_request_access_token_refresh", side_effect=RuntimeError("codex_oauth_refresh_http_400")) as exchange:
            with self.assertRaisesRegex(CodexAuthorizationAttachError, "refresh_unverified"):
                self.accounts.import_owned_codex_authorization("workbench:org:two", incoming)
        exchange.assert_called_once_with(incoming["refresh_token"], {"source_type": "codex"}, timeout=20)
        self.assertEqual(self.accounts.storage.file_path.read_bytes(), before)

        with patch.object(self.accounts, "_request_access_token_refresh", side_effect=TimeoutError("unknown outcome")) as exchange:
            with self.assertRaisesRegex(CodexAuthorizationAttachError, "refresh_unverified"):
                self.accounts.import_owned_codex_authorization("workbench:org:two", incoming)
        exchange.assert_called_once()
        self.assertEqual(self.accounts.storage.file_path.read_bytes(), before)

        with patch.object(self.accounts, "_request_access_token_refresh", return_value=credentials("wrong-subject", subject="other")):
            with self.assertRaisesRegex(CodexAuthorizationAttachError, "account_conflict"):
                self.accounts.import_owned_codex_authorization("workbench:org:two", incoming)
        self.assertEqual(self.accounts.storage.file_path.read_bytes(), before)

        issued = credentials("oauth-returned")
        with patch.object(self.accounts, "_request_access_token_refresh", return_value=issued):
            receipt = self.accounts.import_owned_codex_authorization("workbench:org:two", incoming)
        self.assertEqual(receipt["import_status"], "updated")
        self.assertEqual(self.accounts.storage.load_accounts()[0]["codex_credentials"], issued)
        self.assertEqual(self.accounts.storage.load_accounts()[0]["managed_owner"], "workbench:org:one")

    def test_fresh_official_exchange_401_observation_is_unknown_until_later_read(self):
        old = credentials("owner")
        self.accounts.add_account_items([{
            **old, "source_type": "codex", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "codex_credentials": old,
        }])
        observation = {
            "state": "auth_required", "verified": False, "observed_at": None,
            "failed_at": "now", "models": [], "limits": [], "error_code": "codex_http_401",
        }
        with patch("services.codex_service.codex_service.observe_import_authorization", return_value=observation), \
                patch.object(self.accounts, "_request_access_token_refresh", return_value=credentials("issued")):
            receipt = self.accounts.import_owned_codex_authorization("workbench:org:two", credentials("submitted"))
        self.assertEqual(receipt["codex"]["state"], "read_failed")
        self.assertEqual(receipt["codex"]["error_code"], "usage_unverified")
        self.assertEqual(self.accounts.storage.load_accounts()[0]["managed_owner"], "workbench:org:one")

    def test_same_codex_refresh_token_never_exchanges_and_requires_bearer_proof(self):
        old = credentials("stored")
        self.accounts.add_account_items([{
            **old, "source_type": "codex", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "codex_credentials": old,
        }])
        before = self.accounts.storage.file_path.read_bytes()
        incoming = {**credentials("fresh-access"), "refresh_token": old["refresh_token"]}
        with patch.object(self.accounts, "_request_access_token_refresh", side_effect=AssertionError("must not consume stored refresh")):
            receipt = self.accounts.import_owned_codex_authorization("workbench:org:two", incoming)
        self.assertEqual(receipt["import_status"], "updated")
        saved = self.accounts.storage.load_accounts()[0]
        self.assertEqual(saved["codex_credentials"]["access_token"], incoming["access_token"])
        self.assertEqual(saved["codex_credentials"]["refresh_token"], old["refresh_token"])
        self.assertEqual(saved["codex_credentials"]["id_token"], old["id_token"])
        self.assertEqual(saved["managed_owner"], "workbench:org:one")

        failed = {"state": "read_failed", "verified": False, "observed_at": None,
                  "failed_at": "now", "models": [], "limits": [], "error_code": "usage_unverified"}
        newer = {**credentials("failed-access"), "refresh_token": old["refresh_token"]}
        with patch("services.codex_service.codex_service.observe_import_authorization", return_value=failed), \
                patch.object(self.accounts, "_request_access_token_refresh", side_effect=AssertionError("must not consume stored refresh")):
            with self.assertRaisesRegex(CodexAuthorizationAttachError, "upstream_unverified"):
                self.accounts.import_owned_codex_authorization("workbench:org:three", newer)
        self.assertEqual(self.accounts.storage.load_accounts()[0], saved)
        self.assertNotEqual(self.accounts.storage.file_path.read_bytes(), before)

    def test_chat_primary_refresh_cannot_be_imported_as_separate_codex_refresh(self):
        primary = credentials("chat-primary")
        attached = credentials("separate-codex")
        self.accounts.add_account_items([{
            **primary, "source_type": "web", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "codex_credentials": attached,
        }])
        before = self.accounts.storage.file_path.read_bytes()
        incoming = {**credentials("new-codex"), "refresh_token": primary["refresh_token"]}
        with patch.object(self.accounts, "_request_access_token_refresh", side_effect=AssertionError("must not consume Chat refresh")), \
                patch("services.codex_service.codex_service.observe_import_authorization", side_effect=AssertionError("must reject before Codex probe")):
            with self.assertRaisesRegex(CodexAuthorizationAttachError, "account_conflict"):
                self.accounts.import_owned_codex_authorization("workbench:org:two", incoming)
        self.assertEqual(self.accounts.storage.file_path.read_bytes(), before)
        self.assertEqual(self.accounts.storage.load_accounts()[0]["codex_credentials"], attached)

    def test_same_codex_refresh_stale_and_save_failure_do_not_consume_it(self):
        old = credentials("stored")
        self.accounts.add_account_items([{
            **old, "source_type": "codex", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "codex_credentials": old,
        }])
        incoming = {**credentials("fresh"), "refresh_token": old["refresh_token"]}
        before = self.accounts.storage.file_path.read_bytes()
        with patch.object(self.accounts, "_request_access_token_refresh", side_effect=AssertionError("must not consume stored refresh")), \
                patch.object(self.accounts, "_save_accounts", side_effect=OSError("pre-replace failure")):
            with self.assertRaisesRegex(OSError, "pre-replace failure"):
                self.accounts.import_owned_codex_authorization("workbench:org:two", incoming)
        self.assertEqual(self.accounts.storage.file_path.read_bytes(), before)

        started = threading.Event()
        release = threading.Event()
        normal = {"state": "observed", "verified": True, "observed_at": "now", "failed_at": None,
                  "models": [], "limits": [], "error_code": None}

        def blocked_probe(_credentials):
            started.set()
            self.assertTrue(release.wait(5))
            return normal

        with patch("services.codex_service.codex_service.observe_import_authorization", side_effect=blocked_probe), \
                patch.object(self.accounts, "_request_access_token_refresh", side_effect=AssertionError("must not consume stored refresh")):
            with ThreadPoolExecutor(max_workers=2) as executor:
                waiting = executor.submit(self.accounts.import_owned_codex_authorization, "workbench:org:two", incoming)
                self.assertTrue(started.wait(5))
                self.accounts.attach_codex_authorization(credentials("concurrent"))
                release.set()
                with self.assertRaisesRegex(CodexAuthorizationAttachError, "stale_target"):
                    waiting.result()
        self.assertEqual(self.accounts.storage.load_accounts()[0]["codex_credentials"], credentials("concurrent"))

    def test_manual_codex_uncertain_save_reads_original_issued_bundle(self):
        old = credentials("stored")
        self.accounts.add_account_items([{
            **old, "source_type": "codex", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "codex_credentials": old,
        }])
        submitted = credentials("submitted")
        issued = credentials("issued")
        sync_calls = 0
        real_sync = JSONStorageBackend._sync_directory

        def uncertain_twice(directory):
            nonlocal sync_calls
            sync_calls += 1
            if sync_calls <= 2:
                raise OSError("durability pending")
            return real_sync(directory)

        with patch.object(self.accounts, "_request_access_token_refresh", return_value=issued) as exchange, \
                patch.object(JSONStorageBackend, "_sync_directory", side_effect=uncertain_twice):
            receipt = self.accounts.import_owned_codex_authorization("workbench:org:two", submitted)
        exchange.assert_called_once()
        self.assertEqual(receipt["import_status"], "updated")
        self.assertEqual(self.accounts.storage.load_accounts()[0]["codex_credentials"], issued)

    def test_same_codex_refresh_uncertain_save_recovers_without_exchange(self):
        old = credentials("stored")
        self.accounts.add_account_items([{
            **old, "source_type": "codex", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "codex_credentials": old,
        }])
        incoming = {**credentials("fresh"), "refresh_token": old["refresh_token"]}
        sync_calls = 0
        real_sync = JSONStorageBackend._sync_directory

        def uncertain_twice(directory):
            nonlocal sync_calls
            sync_calls += 1
            if sync_calls <= 2:
                raise OSError("durability pending")
            return real_sync(directory)

        with patch.object(self.accounts, "_request_access_token_refresh", side_effect=AssertionError("must not consume stored refresh")), \
                patch.object(JSONStorageBackend, "_sync_directory", side_effect=uncertain_twice):
            receipt = self.accounts.import_owned_codex_authorization("workbench:org:two", incoming)
        self.assertEqual(receipt["import_status"], "updated")
        self.assertEqual(self.accounts.storage.load_accounts()[0]["codex_credentials"]["refresh_token"], old["refresh_token"])

    def test_manual_codex_unconfirmed_save_reports_unknown_without_reexchange(self):
        old = credentials("stored")
        self.accounts.add_account_items([{
            **old, "source_type": "codex", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "codex_credentials": old,
        }])
        submitted = credentials("submitted")
        issued = credentials("issued")
        with patch.object(self.accounts, "_request_access_token_refresh", return_value=issued) as exchange, \
                patch.object(JSONStorageBackend, "_sync_directory", side_effect=OSError("durability pending")):
            with self.assertRaisesRegex(CodexAuthorizationAttachError, "codex_authorization_import_unknown"):
                self.accounts.import_owned_codex_authorization("workbench:org:two", submitted)
        exchange.assert_called_once()
        restarted = AccountService(JSONStorageBackend(self.accounts.storage.file_path))
        self.assertEqual(restarted.list_accounts()[0]["codex_credentials"], issued)
        self.assertEqual(restarted.list_accounts()[0]["managed_owner"], "workbench:org:one")

    def test_same_chat_refresh_never_exchanges_and_preserves_stored_id(self):
        old = credentials("stored-chat")
        self.accounts.add_account_items([{
            **old, "source_type": "web", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "user_id": SUBJECT,
            "capacity_observed_at": "2026-09-17T00:00:00+00:00",
        }])
        incoming = {**credentials("fresh-chat"), "refresh_token": old["refresh_token"]}
        info = {"user_id": SUBJECT, "account_id": ACCOUNT_ID, "quota": 7,
                "limits_progress": [{"feature_name": "image_gen", "remaining": 7}]}
        with patch.object(self.accounts, "_request_access_token_refresh", side_effect=AssertionError("must not consume stored refresh")), \
                patch.object(self.accounts, "_verified_chat_info", return_value=((SUBJECT, ACCOUNT_ID), info)):
            receipt = self.accounts.import_owned_account("workbench:org:two", {**incoming, "source_type": "web"})
        self.assertEqual(receipt["import_status"], "updated")
        saved = self.accounts.storage.load_accounts()[0]
        self.assertEqual(saved["access_token"], incoming["access_token"])
        self.assertEqual(saved["refresh_token"], old["refresh_token"])
        self.assertEqual(saved["id_token"], old["id_token"])
        self.assertEqual(saved["managed_owner"], "workbench:org:one")

    def test_same_chat_refresh_read_timeout_stale_and_save_failure_leave_pool_usable(self):
        old = credentials("stored-chat")
        self.accounts.add_account_items([{
            **old, "source_type": "web", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "user_id": SUBJECT,
            "capacity_observed_at": "2026-09-17T00:00:00+00:00",
        }])
        incoming = {**credentials("fresh-chat"), "refresh_token": old["refresh_token"]}
        before = self.accounts.storage.file_path.read_bytes()
        info = {"user_id": SUBJECT, "account_id": ACCOUNT_ID, "quota": 7,
                "limits_progress": [{"feature_name": "image_gen", "remaining": 7}]}
        with patch.object(self.accounts, "_request_access_token_refresh", side_effect=AssertionError("must not consume stored refresh")), \
                patch("services.openai_backend_api.OpenAIBackendAPI") as backend_type:
            backend_type.return_value.get_user_info.side_effect = TimeoutError("read timeout")
            with self.assertRaisesRegex(CodexAuthorizationAttachError, "upstream_unverified"):
                self.accounts.import_owned_account("workbench:org:two", {**incoming, "source_type": "web"})
        self.assertEqual(self.accounts.storage.file_path.read_bytes(), before)

        with patch.object(self.accounts, "_request_access_token_refresh", side_effect=AssertionError("must not consume stored refresh")), \
                patch.object(self.accounts, "_verified_chat_info", return_value=((SUBJECT, ACCOUNT_ID), info)), \
                patch.object(self.accounts, "_save_accounts", side_effect=OSError("pre-replace failure")):
            with self.assertRaisesRegex(OSError, "pre-replace failure"):
                self.accounts.import_owned_account("workbench:org:two", {**incoming, "source_type": "web"})
        self.assertEqual(self.accounts.storage.file_path.read_bytes(), before)
        self.assertEqual(self.accounts.resolve_access_token(old["access_token"]), old["access_token"])

        started = threading.Event()
        release = threading.Event()

        def blocked_read(_token):
            started.set()
            self.assertTrue(release.wait(5))
            return (SUBJECT, ACCOUNT_ID), info

        with patch.object(self.accounts, "_request_access_token_refresh", side_effect=AssertionError("must not consume stored refresh")), \
                patch.object(self.accounts, "_verified_chat_info", side_effect=blocked_read):
            with ThreadPoolExecutor(max_workers=2) as executor:
                waiting = executor.submit(self.accounts.import_owned_account, "workbench:org:two", {**incoming, "source_type": "web"})
                self.assertTrue(started.wait(5))
                self.accounts._apply_refreshed_tokens(old["access_token"], {"access_token": jwt(marker="concurrent-chat")}, "test")
                release.set()
                with self.assertRaisesRegex(CodexAuthorizationAttachError, "stale_target"):
                    waiting.result()
        self.assertEqual(self.accounts.storage.load_accounts()[0]["managed_owner"], "workbench:org:one")

    def test_same_chat_refresh_uncertain_save_returns_original_receipt_after_readback(self):
        old = credentials("stored-chat")
        self.accounts.add_account_items([{
            **old, "source_type": "web", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "user_id": SUBJECT,
            "capacity_observed_at": "2026-09-17T00:00:00+00:00",
        }])
        incoming = {**credentials("fresh-chat"), "refresh_token": old["refresh_token"]}
        info = {"user_id": SUBJECT, "account_id": ACCOUNT_ID, "quota": 7,
                "limits_progress": [{"feature_name": "image_gen", "remaining": 7}]}
        sync_calls = 0
        real_sync = JSONStorageBackend._sync_directory

        def uncertain_twice(directory):
            nonlocal sync_calls
            sync_calls += 1
            if sync_calls <= 2:
                raise OSError("durability pending")
            return real_sync(directory)

        with patch.object(self.accounts, "_request_access_token_refresh", side_effect=AssertionError("must not consume stored refresh")), \
                patch.object(self.accounts, "_verified_chat_info", return_value=((SUBJECT, ACCOUNT_ID), info)), \
                patch.object(JSONStorageBackend, "_sync_directory", side_effect=uncertain_twice):
            receipt = self.accounts.import_owned_account("workbench:org:two", {**incoming, "source_type": "web"})
        self.assertEqual(receipt["import_status"], "updated")
        self.assertEqual(receipt["capacity"]["remaining"], 7)
        self.assertEqual(self.accounts.resolve_access_token(old["access_token"]), incoming["access_token"])
        self.assertEqual(self.accounts.storage.load_accounts()[0]["managed_owner"], "workbench:org:one")

    def test_foreign_chat_refresh_must_be_exchanged_before_rekey(self):
        old = credentials("owner-chat")
        self.accounts.add_account_items([{
            **old, "source_type": "oauth_login", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "user_id": SUBJECT,
            "capacity_observed_at": "2026-09-17T00:00:00+00:00",
        }])
        before = self.accounts.storage.file_path.read_bytes()
        incoming = credentials("submitted-chat")
        with patch.object(self.accounts, "_request_access_token_refresh", side_effect=RuntimeError("oauth_refresh_http_400")) as exchange:
            with self.assertRaisesRegex(CodexAuthorizationAttachError, "refresh_unverified"):
                self.accounts.import_owned_account("workbench:org:two", {**incoming, "source_type": "oauth_login"})
        exchange.assert_called_once_with(incoming["refresh_token"], {"source_type": "oauth_login"}, timeout=20)
        self.assertEqual(self.accounts.storage.file_path.read_bytes(), before)

        issued = credentials("chat-oauth-returned")
        info = {"user_id": SUBJECT, "account_id": ACCOUNT_ID, "quota": 6, "limits_progress": []}
        with patch.object(self.accounts, "_request_access_token_refresh", return_value=issued), \
                patch.object(self.accounts, "_verified_chat_info", return_value=((SUBJECT, ACCOUNT_ID), info)) as read:
            receipt = self.accounts.import_owned_account("workbench:org:two", {**incoming, "source_type": "oauth_login"})
        read.assert_called_once_with(issued["access_token"])
        self.assertEqual(receipt["import_status"], "updated")
        self.assertEqual(receipt["route"], "chat")
        account = self.accounts.storage.load_accounts()[0]
        self.assertEqual(account["access_token"], issued["access_token"])
        self.assertEqual(account["refresh_token"], issued["refresh_token"])
        self.assertEqual(account["source_type"], "oauth_login")
        self.assertEqual(account["managed_owner"], "workbench:org:one")

    def test_manual_proof_requires_protected_upstream_usage_response(self):
        class UsageResponse:
            def __init__(self, status, payload):
                self.status_code = status
                self.content = json.dumps(payload).encode()
                self.payload = payload

            def json(self):
                return self.payload

        class UsageSession:
            def __init__(self, response):
                self.response = response
                self.calls = []
                self.closed = False

            def get(self, url, **kwargs):
                self.calls.append((url, kwargs))
                return self.response.pop(0) if isinstance(self.response, list) else self.response

            def close(self):
                self.closed = True

        incoming = credentials("submitted")
        for status, payload, expected in (
            (401, {"rate_limit": {"primary_window": {"used_percent": 5}}}, "auth_required"),
            (200, {"unrelated": True}, "read_failed"),
            (200, {"rate_limit": {"primary_window": {"used_percent": 5}}}, "observed"),
        ):
            with self.subTest(status=status, payload=payload):
                session = UsageSession(UsageResponse(status, payload))
                factory_kwargs = []

                def factory(**kwargs):
                    factory_kwargs.append(kwargs)
                    return session

                result = CodexService(self.accounts, factory).observe_import_authorization(incoming)
                self.assertEqual(result["state"], expected)
                self.assertEqual(session.calls[0][0], CODEX_USAGE_URL)
                self.assertEqual(session.calls[0][1]["headers"]["authorization"], "Bearer " + incoming["access_token"])
                self.assertEqual(session.calls[0][1]["headers"]["chatgpt-account-id"], ACCOUNT_ID)
                self.assertFalse(session.calls[0][1]["allow_redirects"])
                self.assertTrue(factory_kwargs[0]["verify"])
                self.assertTrue(session.closed)
        session = UsageSession([
            UsageResponse(429, {}),
            UsageResponse(200, {"models": [{"slug": "gpt-5.6-codex"}]}),
        ])
        fallback = CodexService(self.accounts, lambda **_: session).observe_import_authorization(incoming)
        self.assertTrue(fallback["verified"])
        self.assertEqual(fallback["state"], "read_failed")
        self.assertEqual(session.calls[1][0], CODEX_MODELS_URL)
        mixed = UsageSession([
            UsageResponse(401, {}),
            UsageResponse(200, {"models": [{"slug": "gpt-5.6-codex"}]}),
        ])
        mixed_result = CodexService(self.accounts, lambda **_: mixed).observe_import_authorization(incoming)
        self.assertEqual(mixed_result["state"], "read_failed")
        self.assertTrue(mixed_result["verified"])

    def test_simultaneous_same_identity_import_deduplicates_and_stale_probe_cannot_overwrite(self):
        same = credentials("same")
        with ThreadPoolExecutor(max_workers=2) as executor:
            receipts = list(executor.map(
                lambda owner: self.accounts.import_owned_codex_authorization(owner, same),
                ("workbench:org:one", "workbench:org:two"),
            ))
        self.assertEqual({item["import_status"] for item in receipts}, {"created", "unchanged"})
        self.assertEqual(len(self.accounts.list_accounts()), 1)
        self.assertEqual(receipts[0]["authorization_ref"], receipts[1]["authorization_ref"])

        started = threading.Event()
        release = threading.Event()
        normal_probe = {"state": "observed", "observed_at": "now", "failed_at": None, "models": [], "limits": [], "error_code": None}

        def probe(value):
            if value["refresh_token"] == "refresh-slow":
                started.set()
                self.assertTrue(release.wait(5))
            return normal_probe

        with patch("services.codex_service.codex_service.observe_import_authorization", side_effect=probe):
            with ThreadPoolExecutor(max_workers=2) as executor:
                slow = executor.submit(self.accounts.import_owned_codex_authorization, "workbench:org:three", credentials("slow"))
                self.assertTrue(started.wait(5))
                fast = self.accounts.import_owned_codex_authorization("workbench:org:two", credentials("fast"))
                release.set()
                with self.assertRaisesRegex(CodexAuthorizationAttachError, "stale_target"):
                    slow.result()
        self.assertEqual(fast["import_status"], "updated")
        self.assertEqual(self.accounts.list_accounts()[0]["codex_credentials"], credentials("fast"))

    def test_exact_foreign_chat_token_returns_only_capacity_and_different_token_fails_closed(self):
        original = credentials("chat")
        self.accounts.add_account_items([{
            **original, "source_type": "web", "managed_owner": "workbench:org:one",
            "managed_account_id": "chat-row", "quota": 11,
            "limits_progress": [{"feature_name": "image_gen", "remaining": 4}],
        }])
        before = self.accounts.storage.file_path.read_bytes()
        receipt = self.accounts.import_owned_account("workbench:org:two", {"access_token": original["access_token"]})
        self.assertEqual(set(receipt), {"import_status", "route", "capacity"})
        self.assertEqual(receipt["route"], "chat")
        self.assertEqual(receipt["capacity"]["remaining"], 4)
        self.assertEqual(self.accounts.storage.file_path.read_bytes(), before)
        with patch("services.codex_service.codex_service.observe_import_authorization", side_effect=AssertionError("Chat must not probe Codex")):
            complete = self.accounts.import_owned_account("workbench:org:two", {**original, "source_type": "web"})
        self.assertEqual(complete, receipt)
        self.assertEqual(self.accounts.storage.file_path.read_bytes(), before)
        with patch.object(self.accounts, "_verified_chat_info", side_effect=CodexAuthorizationAttachError("chat_authorization_upstream_unverified")):
            with self.assertRaisesRegex(CodexAuthorizationAttachError, "upstream_unverified"):
                self.accounts.import_owned_account("workbench:org:two", {"access_token": jwt(marker="new-chat-token")})
        self.assertEqual(self.accounts.storage.file_path.read_bytes(), before)

    def test_verified_new_chat_token_rotates_same_foreign_record_and_preserves_bindings(self):
        old = credentials("old-chat")
        attached = credentials("separate-codex")
        self.accounts.add_account_items([{
            **old, "source_type": "web", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "user_id": SUBJECT,
            "capacity_observed_at": "2026-09-17T00:00:00+00:00",
            "conversation_binding_ids": ["cb_original"],
            "provider_account_identity": "account_original",
            "codex_credentials": attached,
            "codex_response_ids": {"receipt": {"state": "finished"}},
            "quota": 4,
        }])
        self.accounts.storage.save_auth_keys([{"id": "original-key", "key_hash": "fixture-hash"}])
        keys_before = self.accounts.storage.auth_keys_path.read_bytes()
        before = self.accounts.storage.load_accounts()[0]
        incoming = credentials("fresh-chat")
        info = {
            "user_id": SUBJECT, "account_id": ACCOUNT_ID, "quota": 9,
            "limits_progress": [{"feature_name": "image_gen", "remaining": 9}],
        }
        with patch.object(self.accounts, "_verified_chat_info", return_value=((SUBJECT, ACCOUNT_ID), info)) as proof, \
                patch("services.codex_service.codex_service.observe_import_authorization", side_effect=AssertionError("Chat must not probe Codex")):
            receipt = self.accounts.import_owned_account("workbench:org:two", {**incoming, "source_type": "web"})
        proof.assert_called_once_with(incoming["access_token"])
        self.assertEqual(receipt["import_status"], "updated")
        self.assertEqual(receipt["route"], "chat")
        self.assertEqual(receipt["capacity"]["remaining"], 9)
        self.assertEqual(set(receipt), {"import_status", "route", "capacity"})
        self.assertEqual(self.accounts.list_owned_accounts("workbench:org:two"), [])
        with self.assertRaises(KeyError):
            self.accounts.set_owned_account_enabled("workbench:org:two", "original-row", False)
        after = self.accounts.storage.load_accounts()[0]
        self.assertEqual(after["access_token"], incoming["access_token"])
        self.assertEqual(after["refresh_token"], incoming["refresh_token"])
        for field in ("managed_owner", "managed_account_id", "source_type", "codex_credentials", "codex_response_ids", "conversation_binding_ids", "provider_account_identity"):
            self.assertEqual(after[field], before[field])
        self.assertEqual(self.accounts.storage.auth_keys_path.read_bytes(), keys_before)
        self.assertEqual(self.accounts.resolve_access_token(old["access_token"]), incoming["access_token"])
        self.assertEqual(self.accounts._bound_token_locked("cb_original"), incoming["access_token"])
        restarted = AccountService(JSONStorageBackend(self.accounts.storage.file_path))
        self.assertEqual(restarted._bound_token_locked("cb_original"), incoming["access_token"])
        self.assertEqual(len(restarted.list_accounts()), 1)

    def test_chat_identity_proof_uses_protected_read_and_closes_client(self):
        info = {"user_id": SUBJECT, "account_id": ACCOUNT_ID, "quota": 3}
        with patch("services.openai_backend_api.OpenAIBackendAPI") as backend_type:
            backend_type.return_value.get_user_info.return_value = info
            identity, observed = self.accounts._verified_chat_info("submitted-bearer")
        self.assertEqual(identity, (SUBJECT, ACCOUNT_ID))
        self.assertEqual(observed, info)
        backend_type.assert_called_once_with("submitted-bearer")
        backend_type.return_value.get_user_info.assert_called_once_with()
        backend_type.return_value.close.assert_called_once_with()
        with patch("services.openai_backend_api.OpenAIBackendAPI") as backend_type:
            backend_type.return_value.get_user_info.return_value = {"user_id": SUBJECT}
            with self.assertRaisesRegex(CodexAuthorizationAttachError, "upstream_unverified"):
                self.accounts._verified_chat_info("submitted-bearer")
            backend_type.return_value.close.assert_called_once_with()

    def test_explicit_chat_source_does_not_become_codex_when_all_fields_present(self):
        with patch("services.codex_service.codex_service.observe_import_authorization", side_effect=AssertionError("Chat must not probe Codex")):
            row = self.accounts.import_owned_account(
                "workbench:org:one", {**credentials("oauth-chat"), "source_type": "oauth_login"}
            )
        account = self.accounts.storage.load_accounts()[0]
        self.assertEqual(row["id"], account["managed_account_id"])
        self.assertEqual(account["source_type"], "oauth_login")
        self.assertNotIn("codex_credentials", account)

    def test_chat_rotation_needs_verified_matching_principal_and_cas(self):
        old = credentials("old-chat")
        self.accounts.add_account_items([{
            **old, "source_type": "web", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "user_id": SUBJECT,
            "capacity_observed_at": "2026-09-17T00:00:00+00:00",
        }])
        before = self.accounts.storage.file_path.read_bytes()
        incoming = credentials("new-chat")
        wrong = {"user_id": "other", "account_id": ACCOUNT_ID, "quota": 3, "limits_progress": []}
        with patch.object(self.accounts, "_verified_chat_info", return_value=(("other", ACCOUNT_ID), wrong)):
            with self.assertRaisesRegex(CodexAuthorizationAttachError, "account_conflict"):
                self.accounts.import_owned_account("workbench:org:two", {**incoming, "source_type": "web"})
        self.assertEqual(self.accounts.storage.file_path.read_bytes(), before)

        started = threading.Event()
        release = threading.Event()
        valid = {"user_id": SUBJECT, "account_id": ACCOUNT_ID, "quota": 3, "limits_progress": []}

        def blocked_proof(_token):
            started.set()
            self.assertTrue(release.wait(5))
            return (SUBJECT, ACCOUNT_ID), valid

        with patch.object(self.accounts, "_verified_chat_info", side_effect=blocked_proof):
            with ThreadPoolExecutor(max_workers=2) as executor:
                waiting = executor.submit(self.accounts.import_owned_account, "workbench:org:two", {**incoming, "source_type": "web"})
                self.assertTrue(started.wait(5))
                self.accounts._apply_refreshed_tokens(old["access_token"], {"access_token": jwt(marker="other-refresh")}, "test")
                release.set()
                with self.assertRaisesRegex(CodexAuthorizationAttachError, "stale_target"):
                    waiting.result()
        self.assertEqual(self.accounts.list_accounts()[0]["access_token"], jwt(marker="other-refresh"))

    def test_chat_rotation_unknown_commit_is_read_back_before_retry(self):
        old = credentials("old-chat")
        self.accounts.add_account_items([{
            **old, "source_type": "web", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "user_id": SUBJECT,
            "capacity_observed_at": "2026-09-17T00:00:00+00:00",
        }])
        incoming = credentials("uncertain-chat")
        info = {"user_id": SUBJECT, "account_id": ACCOUNT_ID, "quota": 2, "limits_progress": []}
        with patch.object(self.accounts, "_verified_chat_info", return_value=((SUBJECT, ACCOUNT_ID), info)):
            with patch.object(JSONStorageBackend, "_sync_directory", side_effect=OSError("durability pending")):
                with self.assertRaisesRegex(CodexAuthorizationAttachError, "chat_authorization_import_unknown"):
                    self.accounts.import_owned_account("workbench:org:two", {**incoming, "source_type": "web"})
        # The original operation may have committed. Retry first settles the
        # exact persisted pool and never repeats credential rotation.
        receipt = self.accounts.import_owned_account("workbench:org:two", {"access_token": incoming["access_token"]})
        self.assertEqual(receipt["import_status"], "unchanged")
        self.assertEqual(self.accounts.storage.load_accounts()[0]["managed_owner"], "workbench:org:one")
        self.assertEqual(len(self.accounts.storage.load_accounts()), 1)

    def test_new_import_save_failure_rolls_back_pool_memory(self):
        with patch.object(self.accounts, "_save_accounts", side_effect=OSError("storage unavailable")):
            with self.assertRaisesRegex(OSError, "storage unavailable"):
                self.accounts.import_owned_account(
                    "workbench:org:one",
                    {**credentials("will-rollback"), "source_type": "codex"},
                )
        self.assertEqual(self.accounts.list_accounts(), [])

    def test_complete_chat_file_identity_is_not_cloned_but_unknown_identity_keeps_legacy_fallback(self):
        original = self.add_primary()
        info = {"user_id": SUBJECT, "account_id": ACCOUNT_ID, "quota": 3, "limits_progress": []}
        with patch.object(self.accounts, "_verified_chat_info", return_value=((SUBJECT, ACCOUNT_ID), info)):
            same = self.accounts.import_owned_account(
                "workbench:org:one", {**credentials("rotated-chat"), "source_type": "web"}
            )
        self.assertEqual(same["import_status"], "updated")
        self.assertEqual(same["route"], "chat")
        self.assertEqual(self.accounts.list_owned_accounts("workbench:org:one")[0]["authorization_ref"], original["authorization_ref"])
        self.assertEqual(len(self.accounts.list_accounts()), 1)

        opaque = {
            "access_token": "opaque-access-token",
            "refresh_token": "opaque-refresh-token",
            "source_type": "web",
        }
        fallback = self.accounts.import_owned_account("workbench:org:one", opaque)
        self.assertNotEqual(fallback["id"], original["id"])
        self.assertEqual(len(self.accounts.list_accounts()), 2)

    def test_cancel_expiry_restart_and_owner_scope_erase_device_material(self):
        http = FakeHttp([device_start(), device_start(), device_start()])
        service = self.service(http)
        cancelled = service.start("workbench:org:one", "owned", "import", str(uuid.uuid4()))
        with self.assertRaisesRegex(CodexLoginError, "not_found"):
            service.get("workbench:org:two", "owned", cancelled["id"])
        with self.assertRaisesRegex(CodexLoginError, "not_found"):
            service.get("workbench:org:one", "pool", cancelled["id"])
        result = service.cancel("workbench:org:one", "owned", cancelled["id"])
        self.assertEqual(result["state"], "cancelled")
        self.assertNotIn("user_code", result)

        expiring = service.start("workbench:org:one", "owned", "import", str(uuid.uuid4()))
        self.clock.now += CodexLoginService.SESSION_TTL_SECONDS + 1
        service._run(expiring["id"])
        self.assertEqual(service.get("workbench:org:one", "owned", expiring["id"])["state"], "expired")

        pending = service.start("workbench:org:one", "owned", "import", str(uuid.uuid4()))
        restarted = CodexLoginService(
            self.sessions_path,
            self.accounts,
            FakeHttp([]).factory,
            clock=self.clock,
            sleeper=lambda _seconds: None,
            auto_start_workers=False,
        )
        interrupted = restarted.get("workbench:org:one", "owned", pending["id"])
        self.assertEqual(interrupted["state"], "interrupted")
        self.assertEqual(interrupted["error_code"], "codex_login_interrupted")
        text = self.sessions_path.read_text()
        self.assertNotIn("private-device", text)
        self.assertNotIn("ABCD-EFGH", text)

    def test_capacity_is_owner_and_global_bounded_and_completing_cannot_cancel(self):
        http = FakeHttp([device_start(), device_start()])
        service = self.service(http)
        service.MAX_ACTIVE_PER_OWNER = 1
        first = service.start("workbench:org:one", "owned", "import", str(uuid.uuid4()))
        with self.assertRaisesRegex(CodexLoginError, "capacity"):
            service.start("workbench:org:one", "owned", "import", str(uuid.uuid4()))
        second = service.start("workbench:org:two", "owned", "import", str(uuid.uuid4()))
        service.MAX_ACTIVE_GLOBAL = 2
        with self.assertRaisesRegex(CodexLoginError, "capacity"):
            service.start("workbench:org:three", "owned", "import", str(uuid.uuid4()))
        with service._lock:
            service._sessions[first["id"]]["state"] = "completing"
        with self.assertRaisesRegex(CodexLoginError, "completion_in_progress"):
            service.cancel("workbench:org:one", "owned", first["id"])
        self.assertEqual(service.cancel("workbench:org:two", "owned", second["id"])["state"], "cancelled")

    def test_poll_429_backoff_and_unknown_exchange_never_retry(self):
        http = FakeHttp([
            device_start("120"),
            FakeResponse(429, headers={"Retry-After": "180"}),
            device_complete(),
            RuntimeError("private upstream response"),
        ])
        service = self.service(http)
        pending = service.start("workbench:org:one", "owned", "import", str(uuid.uuid4()))
        service._run(pending["id"])
        failed = service.get("workbench:org:one", "owned", pending["id"])
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["error_code"], "codex_login_exchange_outcome_unknown")
        self.assertEqual(failed["poll_after_seconds"], 180)
        self.assertEqual(len(http.calls), 4)
        self.assertNotIn("private upstream", self.sessions_path.read_text())
        restarted = CodexLoginService(
            self.sessions_path,
            self.accounts,
            FakeHttp([]).factory,
            auto_start_workers=False,
        )
        self.assertEqual(restarted.get("workbench:org:one", "owned", pending["id"]), failed)

    def test_session_save_failure_after_account_commit_recovers_by_readback(self):
        imported = credentials("durable-account")
        http = FakeHttp([device_start(), device_complete(), token_exchange(imported)])
        service = self.service(http)
        pending = service.start("workbench:org:one", "owned", "import", str(uuid.uuid4()))
        original_save = service._save_locked
        saves = 0

        def fail_final_save():
            nonlocal saves
            saves += 1
            if saves == 3:
                raise OSError("simulated session fsync failure")
            original_save()

        with patch.object(service, "_save_locked", side_effect=fail_final_save):
            service._run(pending["id"])
        self.assertEqual(service.get("workbench:org:one", "owned", pending["id"])["state"], "succeeded")
        self.assertEqual(len(http.calls), 3)

        restarted = CodexLoginService(
            self.sessions_path,
            self.accounts,
            FakeHttp([]).factory,
            auto_start_workers=False,
        )
        recovered = restarted.get("workbench:org:one", "owned", pending["id"])
        self.assertEqual(recovered["state"], "succeeded")
        self.assertEqual(len(self.accounts.list_owned_accounts("workbench:org:one")), 1)
        self.assertNotIn("completion_authorization_ref", self.sessions_path.read_text())

    def test_device_import_of_foreign_identity_recovers_completion_without_owner_transfer(self):
        original = credentials("existing")
        self.accounts.add_account_items([{
            **original, "managed_owner": "workbench:org:one", "managed_account_id": "original-row",
            "codex_affinities": {"task": {"state": "bound"}},
        }])
        incoming = credentials("device-new")
        http = FakeHttp([device_start(), device_complete(), token_exchange(incoming)])
        service = self.service(http)
        pending = service.start("workbench:org:two", "owned", "import", str(uuid.uuid4()))
        original_save = service._save_locked
        saves = 0

        def fail_final_save():
            nonlocal saves
            saves += 1
            if saves == 3:
                raise OSError("simulated session fsync failure")
            original_save()

        with patch.object(service, "_save_locked", side_effect=fail_final_save):
            service._run(pending["id"])
        restarted = CodexLoginService(
            self.sessions_path, AccountService(JSONStorageBackend(self.accounts.storage.file_path)),
            FakeHttp([]).factory, auto_start_workers=False,
        )
        recovered = restarted.get("workbench:org:two", "owned", pending["id"])
        self.assertEqual(recovered["state"], "succeeded")
        self.assertEqual(recovered["import_status"], "updated")
        self.assertEqual(recovered["codex"]["state"], "observed")
        self.assertEqual(recovered["account_ref"], AccountService._authorization_ref(SUBJECT, ACCOUNT_ID))
        self.assertEqual(len(http.calls), 3)
        saved = self.accounts.storage.load_accounts()[0]
        self.assertEqual(saved["managed_owner"], "workbench:org:one")
        self.assertEqual(saved["managed_account_id"], "original-row")
        self.assertEqual(saved["codex_affinities"], {"task": {"state": "bound"}})
        self.assertEqual(saved["codex_credentials"], incoming)
        self.assertEqual(self.accounts.list_owned_accounts("workbench:org:two"), [])

    def test_foreign_import_uncertain_account_commit_recovers_from_original_receipt(self):
        self.accounts.add_account_items([{
            **credentials("existing"), "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row",
        }])
        incoming = credentials("uncertain-new")
        http = FakeHttp([device_start(), device_complete(), token_exchange(incoming)])
        service = self.service(http)
        pending = service.start("workbench:org:two", "owned", "import", str(uuid.uuid4()))
        with patch.object(JSONStorageBackend, "_sync_directory", side_effect=OSError("durability pending")):
            service._run(pending["id"])
            uncertain = service.get("workbench:org:two", "owned", pending["id"])
            self.assertEqual(uncertain["state"], "interrupted")
            self.assertEqual(uncertain["error_code"], "codex_login_save_failed")
        restarted_accounts = AccountService(JSONStorageBackend(self.accounts.storage.file_path))
        restarted = CodexLoginService(
            self.sessions_path, restarted_accounts, FakeHttp([]).factory,
            auto_start_workers=False,
        )
        recovered = restarted.get("workbench:org:two", "owned", pending["id"])
        self.assertEqual(recovered["state"], "succeeded")
        self.assertEqual(recovered["import_status"], "updated")
        self.assertEqual(recovered["codex"]["state"], "observed")
        self.assertEqual(len(http.calls), 3)
        self.assertEqual(restarted_accounts.list_accounts()[0]["managed_owner"], "workbench:org:one")

    def test_device_exchange_same_refresh_keeps_new_id_and_recovers_original_digest(self):
        primary = credentials("chat-primary")
        attached = credentials("old-codex")
        self.accounts.add_account_items([{
            **primary, "source_type": "web", "managed_owner": "workbench:org:one",
            "managed_account_id": "original-row", "codex_credentials": attached,
        }])
        incoming = {**credentials("device-new"), "refresh_token": attached["refresh_token"]}
        self.assertNotEqual(incoming["id_token"], attached["id_token"])
        http = FakeHttp([device_start(), device_complete(), token_exchange(incoming)])
        service = self.service(http)
        pending = service.start("workbench:org:two", "owned", "import", str(uuid.uuid4()))
        with patch.object(self.accounts, "_request_access_token_refresh", side_effect=AssertionError("device exchange is proof")), \
                patch.object(JSONStorageBackend, "_sync_directory", side_effect=OSError("durability pending")):
            service._run(pending["id"])
            uncertain = service.get("workbench:org:two", "owned", pending["id"])
            self.assertEqual(uncertain["state"], "interrupted")
            self.assertEqual(uncertain["error_code"], "codex_login_save_failed")
        saved = self.accounts.storage.load_accounts()[0]
        self.assertEqual(saved["codex_credentials"], incoming)
        restarted_accounts = AccountService(JSONStorageBackend(self.accounts.storage.file_path))
        restarted = CodexLoginService(
            self.sessions_path, restarted_accounts, FakeHttp([]).factory,
            auto_start_workers=False,
        )
        recovered = restarted.get("workbench:org:two", "owned", pending["id"])
        self.assertEqual(recovered["state"], "succeeded")
        self.assertEqual(recovered["import_status"], "updated")
        self.assertEqual(len(http.calls), 3)
        self.assertEqual(restarted_accounts.list_accounts()[0]["managed_owner"], "workbench:org:one")

    def test_replace_then_directory_failure_recovers_exact_original_login_and_preserves_pool(self):
        from services.storage.base import AccountCommitUncertain
        original = self.add_primary()
        primary = credentials("primary")["access_token"]
        self.accounts._accounts[primary]["codex_affinities"] = {"original": {"state": "bound"}}
        self.accounts._accounts[primary]["codex_response_ids"] = {"receipt": {"owner": "caller"}}
        self.accounts._save_accounts()
        before = self.accounts.storage.load_accounts()[0]
        self.accounts.storage.save_auth_keys([{"id": "ordinary-key", "key_hash": "fixture-hash"}])
        key_bytes = self.accounts.storage.auth_keys_path.read_bytes()
        incoming = credentials("directory-fault")
        http = FakeHttp([device_start(), device_complete(), token_exchange(incoming)])
        service = self.service(http)
        pending = service.start("workbench:org:one", "owned", "attach", str(uuid.uuid4()), original["authorization_ref"])
        with patch.object(JSONStorageBackend, "_sync_directory", side_effect=OSError("durability pending")):
            service._run(pending["id"])
            uncertain = service.get("workbench:org:one", "owned", pending["id"])
            self.assertEqual(uncertain["state"], "interrupted")
            self.assertEqual(uncertain["error_code"], "codex_login_save_failed")
            with self.assertRaises(AccountCommitUncertain):
                self.accounts._save_accounts()
            self.assertIn("completion_credential_digest", self.sessions_path.read_text())
        restarted_accounts = AccountService(JSONStorageBackend(self.accounts.storage.file_path))
        restarted = CodexLoginService(self.sessions_path, restarted_accounts, FakeHttp([]).factory, auto_start_workers=False)
        self.assertEqual(restarted.get("workbench:org:one", "owned", pending["id"])["state"], "succeeded")
        # The same process can also settle its cache through original GET only.
        self.assertEqual(service.get("workbench:org:one", "owned", pending["id"])["state"], "succeeded")
        self.assertEqual(len(http.calls), 3)
        after = self.accounts.storage.load_accounts()[0]
        for field in ("access_token", "refresh_token", "id_token", "managed_owner", "managed_account_id", "codex_affinities", "codex_response_ids"):
            self.assertEqual(after[field], before[field])
        self.assertEqual(after["codex_credentials"], incoming)
        self.assertEqual(self.accounts.storage.auth_keys_path.read_bytes(), key_bytes)
        self.assertNotIn("completion_credential_digest", self.sessions_path.read_text())

    def test_pre_replace_failure_reads_original_pool_and_never_exchanges_again(self):
        self.add_primary()
        before = self.accounts.storage.file_path.read_bytes()
        incoming = credentials("new-account", subject="other", account_id=OTHER_ACCOUNT_ID)
        http = FakeHttp([device_start(), device_complete(), token_exchange(incoming)])
        service = self.service(http)
        pending = service.start("workbench:org:one", "owned", "import", str(uuid.uuid4()))
        with patch.object(JSONStorageBackend, "save_accounts", side_effect=OSError("before replace")):
            service._run(pending["id"])
        result = service.get("workbench:org:one", "owned", pending["id"])
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error_code"], "codex_login_save_not_applied")
        self.assertEqual(self.accounts.storage.file_path.read_bytes(), before)
        self.assertEqual(len(http.calls), 3)

    def test_recovery_does_not_mistake_old_same_identity_credentials_for_new_save(self):
        old = credentials("old")
        incoming = credentials("new")
        item = self.accounts.import_owned_codex_authorization("workbench:org:one", old)
        result = self.accounts.codex_login_completion_readback(
            "workbench:org:one", "owned", "attach", item["authorization_ref"],
            item["authorization_ref"], AccountService.codex_credential_digest(incoming))
        self.assertFalse(result["applied"])

    def test_attach_checks_selected_ref_and_stale_credentials_before_write(self):
        item = self.add_primary()
        account_ref = item["authorization_ref"]
        login_credentials = credentials("login")
        http = FakeHttp([device_start(), device_complete(), token_exchange(login_credentials)])
        service = self.service(http)
        pending = service.start(
            "workbench:org:one",
            "owned",
            "attach",
            str(uuid.uuid4()),
            account_ref,
        )
        newer = credentials("newer")
        self.accounts.attach_owned_codex_authorization(
            "workbench:org:one", item["id"], newer
        )
        service._run(pending["id"])
        failed = service.get("workbench:org:one", "owned", pending["id"])
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["error_code"], "codex_login_stale_target")
        self.assertEqual(self.accounts.list_accounts()[0]["codex_credentials"], newer)

        wrong_identity = credentials("wrong", subject="other", account_id=OTHER_ACCOUNT_ID)
        with self.assertRaisesRegex(CodexAuthorizationAttachError, "account_conflict"):
            self.accounts.attach_codex_authorization(wrong_identity, account_ref)

    def test_pool_requires_attach_and_ambiguous_ref_fails_before_network(self):
        item = self.add_primary()
        http = FakeHttp([])
        service = self.service(http)
        with self.assertRaisesRegex(CodexLoginError, "invalid_mode"):
            service.start("workbench:boss", "pool", "import", str(uuid.uuid4()))

        duplicate = credentials("duplicate")
        self.accounts.add_account_items([{
            **duplicate,
            "access_token": duplicate["access_token"] + "-different-storage-key",
        }])
        with self.assertRaisesRegex(CodexAuthorizationAttachError, "account_ambiguous"):
            service.start(
                "workbench:boss",
                "pool",
                "attach",
                str(uuid.uuid4()),
                item["authorization_ref"],
            )
        self.assertEqual(http.calls, [])


if __name__ == "__main__":
    unittest.main()
