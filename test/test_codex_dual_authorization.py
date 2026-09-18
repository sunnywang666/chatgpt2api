from __future__ import annotations

import base64
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from services.account_service import AccountService, CodexAuthorizationAttachError
from services.codex_service import CodexService, CodexServiceError
from services.storage.json_storage import JSONStorageBackend
from test.test_codex_service import FakeResponse, FakeSession, SessionFactory, observation


ACCOUNT_ID = "12345678-1234-5678-9234-567812345678"
OTHER_ACCOUNT_ID = "87654321-4321-6789-9234-567812345678"
SUBJECT = "user-subject-1"


def jwt(
    subject: str,
    *,
    account_id: str = ACCOUNT_ID,
    exp: int | None = None,
    marker: str = "",
) -> str:
    payload: dict[str, object] = {"sub": subject}
    if account_id:
        payload["https://api.openai.com/auth"] = {"chatgpt_account_id": account_id}
    if exp is not None:
        payload["exp"] = exp
    if marker:
        payload["jti"] = marker
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def credentials(
    label: str,
    *,
    subject: str = SUBJECT,
    account_id: str = ACCOUNT_ID,
    exp: int | None = None,
) -> dict[str, str]:
    return {
        "access_token": jwt(subject, account_id=account_id, exp=exp, marker=f"access-{label}"),
        "refresh_token": f"codex-refresh-{label}",
        "id_token": jwt(subject, account_id=account_id, marker=f"id-{label}"),
        "account_id": account_id,
    }


class CodexDualAuthorizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "accounts.json"
        self.accounts = AccountService(JSONStorageBackend(self.path))

    def add_primary(self, token_label: str = "primary", **updates) -> str:
        token = jwt(SUBJECT, account_id=ACCOUNT_ID, exp=int(time.time()) + 7 * 86400)
        item = {
            "access_token": token,
            "refresh_token": f"chat-refresh-{token_label}",
            "id_token": jwt(SUBJECT, account_id=ACCOUNT_ID),
            "account_id": ACCOUNT_ID,
            "source_type": "web",
            "status": "正常",
            "managed_owner": "workbench:o:boss",
            "managed_account_id": "managed-original",
            "quota": 7,
            "codex_observation": observation(),
            "codex_affinities": {"affinity": {"state": "bound"}},
            "codex_response_ids": {"receipt": {"owner": "caller"}},
        }
        item.update(updates)
        self.accounts.add_account_items([item])
        return token

    def test_attach_persists_on_same_record_and_safe_projections_hide_nested_secrets(self):
        primary = self.add_primary()
        before = self.accounts.get_account(primary)
        result = self.accounts.attach_codex_authorization(credentials("one"))

        self.assertEqual(result, {"attached": True})
        current = self.accounts.get_account(primary)
        self.assertEqual(current["access_token"], before["access_token"])
        self.assertEqual(current["refresh_token"], before["refresh_token"])
        self.assertEqual(current["id_token"], before["id_token"])
        self.assertEqual(current["managed_owner"], before["managed_owner"])
        self.assertEqual(current["managed_account_id"], before["managed_account_id"])
        self.assertEqual(current["quota"], before["quota"])
        self.assertEqual(current["codex_affinities"], before["codex_affinities"])
        self.assertEqual(current["codex_response_ids"], before["codex_response_ids"])
        self.assertEqual(current["codex_credentials"], credentials("one"))
        self.assertEqual(current["codex_observation"]["state"], "unknown")
        exported = self.accounts.build_export_items([primary])
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]["access_token"], before["access_token"])
        self.assertEqual(exported[0]["refresh_token"], before["refresh_token"])
        self.assertEqual(exported[0]["id_token"], before["id_token"])
        self.assertNotIn("codex-refresh-one", json.dumps(exported))

        restarted = AccountService(JSONStorageBackend(self.path))
        self.assertEqual(restarted.get_account(primary)["codex_credentials"], credentials("one"))
        detached_copy = restarted.get_account(primary)
        detached_copy["codex_credentials"]["access_token"] = "mutated-copy"
        self.assertEqual(restarted.get_account(primary)["codex_credentials"], credentials("one"))
        safe = json.dumps({
            "owned": restarted.list_owned_accounts("workbench:o:boss"),
            "pool": restarted.list_pool_accounts(),
        })
        self.assertNotIn("codex-refresh-one", safe)
        self.assertNotIn(credentials("one")["access_token"], safe)
        self.assertNotIn(credentials("one")["id_token"], safe)

    def test_attach_is_idempotent_and_preserves_rejection_evidence(self):
        old = credentials("old")
        primary = self.add_primary(codex_credentials=old)
        rejected = {
            "credential_digest": CodexService._credential_digest(self.accounts.get_account(primary)),
            "failed_at": "2026-09-18T00:00:00+00:00",
        }
        self.accounts.update_account(primary, {"codex_auth_rejection": rejected}, quiet=True)
        with patch.object(self.accounts, "_save_accounts", wraps=self.accounts._save_accounts) as save:
            self.assertEqual(self.accounts.attach_codex_authorization(old), {"attached": True})
            save.assert_not_called()

        self.accounts.attach_codex_authorization(credentials("new"))
        current = self.accounts.get_account(primary)
        self.assertEqual(current["codex_auth_rejection"], rejected)
        self.assertEqual(CodexService.account_projection(current)["state"], "unknown")
        self.assertNotEqual(CodexService._credential_digest(current), rejected["credential_digest"])

    def test_attach_save_failure_rolls_back_memory_and_allows_durable_retry(self):
        primary = self.add_primary()
        payload = credentials("retry")
        with patch.object(
            self.accounts,
            "_save_accounts",
            side_effect=RuntimeError("simulated storage failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated storage failure"):
                self.accounts.attach_codex_authorization(payload)

        self.assertNotIn("codex_credentials", self.accounts.get_account(primary))
        restarted_after_failure = AccountService(JSONStorageBackend(self.path))
        self.assertNotIn("codex_credentials", restarted_after_failure.get_account(primary))

        self.assertEqual(
            restarted_after_failure.attach_codex_authorization(payload),
            {"attached": True},
        )
        restarted_after_retry = AccountService(JSONStorageBackend(self.path))
        self.assertEqual(
            restarted_after_retry.get_account(primary)["codex_credentials"],
            payload,
        )

    def test_attach_rejects_missing_mismatched_conflicting_and_ambiguous_material(self):
        self.add_primary()
        cases = []
        missing = credentials("missing")
        missing.pop("refresh_token")
        cases.append(missing)
        mismatched_tokens = credentials("mismatch")
        mismatched_tokens["id_token"] = jwt("other-subject")
        cases.append(mismatched_tokens)
        cases.append(credentials("wrong-account", account_id=OTHER_ACCOUNT_ID))
        cases.append(credentials("wrong-subject", subject="other-subject"))
        for payload in cases:
            with self.subTest(payload=list(payload)):
                with self.assertRaises(CodexAuthorizationAttachError):
                    self.accounts.attach_codex_authorization(payload)
        self.assertNotIn("codex_credentials", self.accounts.list_accounts()[0])

        duplicate_primary = jwt(SUBJECT, account_id=ACCOUNT_ID, exp=int(time.time()) + 86400)
        self.accounts.add_account_items([{
            "access_token": duplicate_primary + "-duplicate",
            "id_token": jwt(SUBJECT),
            "account_id": ACCOUNT_ID,
        }])
        with self.assertRaisesRegex(CodexAuthorizationAttachError, "account_ambiguous"):
            self.accounts.attach_codex_authorization(credentials("ambiguous"))

    def test_unique_exact_pair_ignores_other_valid_half_matches(self):
        primary = self.add_primary()
        same_subject = jwt(SUBJECT, account_id=OTHER_ACCOUNT_ID, marker="same-subject")
        same_workspace = jwt("other-subject", account_id=ACCOUNT_ID, marker="same-workspace")
        self.accounts.add_account_items([{
            "access_token": same_subject,
            "id_token": jwt(SUBJECT, account_id=OTHER_ACCOUNT_ID),
            "account_id": OTHER_ACCOUNT_ID,
        }, {
            "access_token": same_workspace,
            "id_token": jwt("other-subject", account_id=ACCOUNT_ID),
            "account_id": ACCOUNT_ID,
        }])
        before_half_matches = {
            token: self.accounts.get_account(token)
            for token in (same_subject, same_workspace)
        }

        self.assertEqual(
            self.accounts.attach_codex_authorization(credentials("unique-exact")),
            {"attached": True},
        )

        self.assertEqual(
            self.accounts.get_account(primary)["codex_credentials"],
            credentials("unique-exact"),
        )
        for token, before in before_half_matches.items():
            self.assertEqual(self.accounts.get_account(token), before)

    def test_half_matches_without_exact_pair_still_conflict(self):
        accounts = AccountService(JSONStorageBackend(Path(self.tmp.name) / "half-match.json"))
        accounts.add_account_items([{
            "access_token": jwt(SUBJECT, account_id=OTHER_ACCOUNT_ID, marker="same-subject"),
            "id_token": jwt(SUBJECT, account_id=OTHER_ACCOUNT_ID),
            "account_id": OTHER_ACCOUNT_ID,
        }, {
            "access_token": jwt("other-subject", account_id=ACCOUNT_ID, marker="same-workspace"),
            "id_token": jwt("other-subject", account_id=ACCOUNT_ID),
            "account_id": ACCOUNT_ID,
        }])
        with self.assertRaisesRegex(CodexAuthorizationAttachError, "account_conflict"):
            accounts.attach_codex_authorization(credentials("no-exact"))
        self.assertTrue(all("codex_credentials" not in item for item in accounts.list_accounts()))

    def test_exact_primary_with_conflicting_original_id_subject_is_rejected(self):
        accounts = AccountService(JSONStorageBackend(Path(self.tmp.name) / "original-id-conflict.json"))
        accounts.add_account_items([{
            "access_token": jwt(SUBJECT, account_id=ACCOUNT_ID, marker="primary"),
            "id_token": jwt("other-subject", account_id=ACCOUNT_ID),
            "account_id": ACCOUNT_ID,
        }])
        with self.assertRaisesRegex(CodexAuthorizationAttachError, "account_conflict"):
            accounts.attach_codex_authorization(credentials("conflicting-original-id"))
        self.assertNotIn("codex_credentials", accounts.list_accounts()[0])

    def test_codex_requests_use_nested_authorization_and_primary_key_for_affinity(self):
        primary = self.add_primary(codex_credentials=credentials("request"))
        session = FakeSession(post_response=FakeResponse(payload={"id": "response"}))
        service = CodexService(self.accounts, SessionFactory([session]))

        result = service.submit(
            {"id": "caller", "role": "user"},
            {"model": "gpt-5.6-codex", "input": []},
            {"session-id": "dual-auth-session"},
        )

        self.assertEqual(result.status_code, 200)
        sent = session.calls[0][2]["headers"]
        self.assertEqual(sent["authorization"], f"Bearer {credentials('request')['access_token']}")
        self.assertEqual(sent["chatgpt-account-id"], ACCOUNT_ID)
        current = self.accounts.get_account(primary)
        self.assertIn("affinity", current["codex_affinities"])
        self.assertEqual(current["access_token"], primary)
        self.assertEqual(current["refresh_token"], "chat-refresh-primary")

    def test_refresh_rotates_only_nested_codex_credentials(self):
        expired = credentials("expired", exp=int(time.time()) - 10)
        primary = self.add_primary(codex_credentials=expired)
        before = self.accounts.get_account(primary)
        rotated = credentials("rotated", exp=int(time.time()) + 86400)

        with patch.object(
            self.accounts,
            "_request_access_token_refresh",
            return_value={
                "access_token": rotated["access_token"],
                "refresh_token": rotated["refresh_token"],
                "id_token": rotated["id_token"],
            },
        ) as request:
            self.assertEqual(self.accounts.refresh_codex_access_token(primary), primary)

        request.assert_called_once()
        self.assertEqual(request.call_args.args[0], expired["refresh_token"])
        self.assertEqual(request.call_args.args[1]["source_type"], "codex")
        current = self.accounts.get_account(primary)
        self.assertEqual(current["codex_credentials"], rotated)
        for field in ("access_token", "refresh_token", "id_token", "account_id"):
            self.assertEqual(current[field], before[field])
        self.assertIsNotNone(current["last_codex_token_refresh_at"])
        self.assertIsNone(current["last_token_refresh_at"])

    def test_concurrent_attachment_wins_over_stale_expired_refresh(self):
        expired = credentials("expired", exp=int(time.time()) - 10)
        primary = self.add_primary(codex_credentials=expired)
        newer = credentials("newer", exp=int(time.time()) + 86400)
        stale = credentials("stale", exp=int(time.time()) + 86400)

        def rotate_while_refreshing(*_args, **_kwargs):
            self.accounts.attach_codex_authorization(newer)
            return {
                "access_token": stale["access_token"],
                "refresh_token": stale["refresh_token"],
                "id_token": stale["id_token"],
            }

        with patch.object(self.accounts, "_request_access_token_refresh", side_effect=rotate_while_refreshing):
            self.assertEqual(self.accounts.refresh_codex_access_token(primary), primary)
        self.assertEqual(self.accounts.get_account(primary)["codex_credentials"], newer)

    def test_stale_refresh_failure_does_not_backoff_concurrently_attached_authorization(self):
        expired = credentials("expired", exp=int(time.time()) - 10)
        primary = self.add_primary(codex_credentials=expired)
        newer = credentials("newer", exp=int(time.time()) + 86400)

        def rotate_then_fail(*_args, **_kwargs):
            self.accounts.attach_codex_authorization(newer)
            raise RuntimeError("private upstream material")

        with patch.object(self.accounts, "_request_access_token_refresh", side_effect=rotate_then_fail):
            self.assertEqual(self.accounts.refresh_codex_access_token(primary), primary)
        current = self.accounts.get_account(primary)
        self.assertEqual(current["codex_credentials"], newer)
        self.assertIsNone(current["last_codex_token_refresh_error"])
        self.assertIsNone(current["last_codex_token_refresh_error_at"])

    def test_codex_refresh_failure_is_safe_and_bounded_without_touching_chat_state(self):
        expired = credentials("expired", exp=int(time.time()) - 10)
        primary = self.add_primary(codex_credentials=expired)
        before = self.accounts.get_account(primary)
        with patch.object(
            self.accounts,
            "_request_access_token_refresh",
            side_effect=RuntimeError("private upstream token echo"),
        ) as request:
            self.assertEqual(self.accounts.refresh_codex_access_token(primary), primary)
            self.assertEqual(self.accounts.refresh_codex_access_token(primary), primary)
        request.assert_called_once()
        current = self.accounts.get_account(primary)
        self.assertEqual(current["last_codex_token_refresh_error"], "codex_oauth_refresh_failed")
        self.assertNotIn("private", json.dumps(current))
        self.assertEqual(current["codex_credentials"], expired)
        for field in ("access_token", "refresh_token", "id_token", "account_id"):
            self.assertEqual(current[field], before[field])
        self.assertIsNone(current["last_token_refresh_error"])

    def test_inflight_rejection_cannot_attach_to_concurrently_rotated_codex_authorization(self):
        primary = self.add_primary(codex_credentials=credentials("sent"))
        session = FakeSession(post_response=FakeResponse(status=401, payload={}))
        service = CodexService(self.accounts, SessionFactory([session]))
        original_post = session.post

        def rotate_then_reject(*args, **kwargs):
            self.assertEqual(
                kwargs["headers"]["authorization"],
                f"Bearer {credentials('sent')['access_token']}",
            )
            self.accounts.attach_codex_authorization(credentials("replacement"))
            return original_post(*args, **kwargs)

        session.post = rotate_then_reject
        with self.assertRaises(CodexServiceError) as rejected:
            service.submit(
                {"id": "caller", "role": "user"},
                {"model": "gpt-5.6-codex", "input": []},
                {"session-id": "concurrent-rotation"},
            )
        self.assertEqual(rejected.exception.code, "codex_auth_required")
        current = self.accounts.get_account(primary)
        self.assertEqual(current["codex_credentials"], credentials("replacement"))
        self.assertNotIn("codex_auth_rejection", current)
        self.assertEqual(CodexService.account_projection(current)["state"], "unknown")
        binding = next(
            value for key, value in current["codex_affinities"].items() if key != "affinity"
        )
        self.assertEqual(binding["state"], "bound")


if __name__ == "__main__":
    unittest.main()
