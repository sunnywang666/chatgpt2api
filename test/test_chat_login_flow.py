from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit
from unittest.mock import patch

import pytest

from services.account_service import CodexAuthorizationAttachError
from services.chat_login_service import ChatLoginError, ChatLoginService


ACCOUNT_REF = "car_" + "A" * 43
OWNER = "workbench:org:one"


class FakeResponse:
    def __init__(self, status: int, payload: dict | None = None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeHttp:
    def __init__(self, outcomes: list[object]):
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict]] = []
        self.lock = threading.Lock()

    def factory(self):
        parent = self

        class Session:
            def post(self, url, **kwargs):
                with parent.lock:
                    parent.calls.append((url, kwargs))
                    outcome = parent.outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

            def close(self):
                pass

        return Session()


class FakeAccounts:
    def __init__(self):
        self.revision = "revision-one"
        self.imports: list[tuple[str, dict]] = []
        self.fail_imports = 0

    def codex_login_target(self, owner, account_ref, *, pool=False):
        if account_ref != ACCOUNT_REF:
            raise CodexAuthorizationAttachError("codex_authorization_account_not_found")
        return {"account_ref": account_ref, "revision": self.revision}

    def chat_login_committed_receipt(self, credentials, account_ref=None):
        return None

    def import_owned_account(self, owner, payload, *, verified_oauth=False):
        self.imports.append((owner, dict(payload)))
        if self.fail_imports:
            self.fail_imports -= 1
            raise RuntimeError("simulated process interruption")
        return {
            "authorization_ref": ACCOUNT_REF,
            "import_status": "updated" if payload.get("account_ref") else "created",
            "route": "chat",
            "capacity": {
                "route": "chatgpt_image_gen",
                "source": "limits_progress.image_gen.remaining",
                "unit": "upstream_image_gen",
                "state": "unknown",
                "remaining": None,
                "observed_at": None,
                "failed_at": None,
                "reset_after": None,
                "codex_capacity": None,
            },
        }


def callback_for(started: dict, code: str = "private-code") -> str:
    state = parse_qs(urlsplit(started["authorize_url"]).query)["state"][0]
    return "https://platform.openai.com/auth/callback?" + urlencode({
        "code": code, "state": state, "scope": "openid profile email offline_access",
    })


@pytest.fixture
def root():
    with tempfile.TemporaryDirectory() as value:
        yield Path(value)


def make_service(root: Path, accounts: FakeAccounts, http: FakeHttp) -> ChatLoginService:
    return ChatLoginService(root / "chat_login_sessions.json", accounts, http.factory)


@pytest.mark.parametrize("include_scope", [True, False])
def test_success_is_manual_callback_durable_redacted_and_idempotent(root, include_scope):
    accounts = FakeAccounts()
    tokens = {
        "access_token": "private-access",
        "refresh_token": "private-refresh",
        "id_token": "private-id",
    }
    http = FakeHttp([FakeResponse(200, tokens)])
    service = make_service(root, accounts, http)
    request_id = str(uuid.uuid4())

    started = service.start(OWNER, "owned", "import", request_id)
    assert started["state"] == "pending_callback"
    assert started["return_mode"] == "manual_callback"
    assert started["authorize_url"].startswith(ChatLoginService.AUTHORIZE_URL + "?")
    callback = callback_for(started)
    if not include_scope:
        callback = callback.split("&scope=", 1)[0]
    result = service.submit_callback(OWNER, "owned", started["id"], callback)

    assert result["state"] == "succeeded"
    assert result["account_ref"] == ACCOUNT_REF
    assert result["import_status"] == "created"
    assert result["route"] == "chat"
    assert "authorize_url" not in result
    assert service.start(OWNER, "owned", "import", request_id) == result
    assert len(http.calls) == 1
    assert http.calls[0][0] == ChatLoginService.TOKEN_URL
    assert http.calls[0][1]["allow_redirects"] is False
    assert http.calls[0][1]["json"]["client_id"] == ChatLoginService.CLIENT_ID
    assert http.calls[0][1]["json"]["code"] == "private-code"
    assert "scope" not in http.calls[0][1]["json"]
    assert accounts.imports[0][1]["source_type"] == "oauth_login"
    persisted = (root / "chat_login_sessions.json").read_text()
    assert os.stat(root / "chat_login_sessions.json").st_mode & 0o777 == 0o600
    for secret in (*tokens.values(), "private-code"):
        assert secret not in persisted


def test_callback_requires_exact_url_and_state_before_any_exchange(root):
    accounts = FakeAccounts()
    http = FakeHttp([])
    service = make_service(root, accounts, http)
    started = service.start(OWNER, "owned", "import", str(uuid.uuid4()))

    invalid = [
        "raw-code",
        "http://platform.openai.com/auth/callback?code=x&state=y",
        "https://evil.example/auth/callback?code=x&state=y",
        "https://platform.openai.com/other?code=x&state=y",
        "https://platform.openai.com/auth/callback?code=x&state=y#fragment",
        callback_for(started) + "&scope=duplicate",
        callback_for(started) + "&state=duplicate",
        callback_for(started) + "&code=duplicate",
        callback_for(started) + "&error=denied",
        callback_for(started) + "&redirect_uri=https://evil.example/",
    ]
    for value in invalid:
        with pytest.raises(ChatLoginError, match="invalid_callback"):
            service.submit_callback(OWNER, "owned", started["id"], value)
    wrong_state = "https://platform.openai.com/auth/callback?code=x&state=wrong&scope=openid+profile"
    with pytest.raises(ChatLoginError, match="state_mismatch"):
        service.submit_callback(OWNER, "owned", started["id"], wrong_state)
    assert service.get(OWNER, "owned", started["id"])["state"] == "pending_callback"
    assert http.calls == []


def test_unknown_exchange_and_duplicate_callback_never_exchange_twice(root):
    accounts = FakeAccounts()
    http = FakeHttp([RuntimeError("private upstream failure")])
    service = make_service(root, accounts, http)
    started = service.start(OWNER, "owned", "import", str(uuid.uuid4()))
    callback = callback_for(started)

    first = service.submit_callback(OWNER, "owned", started["id"], callback)
    second = service.submit_callback(OWNER, "owned", started["id"], callback)

    assert first == second
    assert first["state"] == "interrupted"
    assert first["error_code"] == "chat_login_exchange_outcome_unknown"
    assert len(http.calls) == 1
    assert "private upstream" not in (root / "chat_login_sessions.json").read_text()


def test_tokens_persist_before_import_and_restart_resumes_without_exchange(root):
    accounts = FakeAccounts()
    accounts.fail_imports = 1
    tokens = {"access_token": "access-save", "refresh_token": "refresh-save", "id_token": "id-save"}
    http = FakeHttp([FakeResponse(200, tokens)])
    service = make_service(root, accounts, http)
    started = service.start(OWNER, "owned", "attach", str(uuid.uuid4()), ACCOUNT_REF)

    saving = service.submit_callback(OWNER, "owned", started["id"], callback_for(started))
    assert saving["state"] == "saving"
    stored = (root / "chat_login_sessions.json").read_text()
    assert "access-save" in stored and "refresh-save" in stored

    restarted = make_service(root, accounts, FakeHttp([]))
    recovered = restarted.get(OWNER, "owned", started["id"])
    assert recovered["state"] == "succeeded"
    assert recovered["account_ref"] == ACCOUNT_REF
    assert len(http.calls) == 1
    assert len(accounts.imports) == 2
    assert "access-save" not in (root / "chat_login_sessions.json").read_text()


def test_restart_during_exchange_is_unknown_and_cannot_reuse_code(root):
    accounts = FakeAccounts()
    service = make_service(root, accounts, FakeHttp([]))
    started = service.start(OWNER, "owned", "import", str(uuid.uuid4()))
    with service._lock:
        session = service._sessions[started["id"]]
        session["state"] = "exchanging"
        session["callback_digest"] = "digest"
        session.pop("authorize_url", None)
        service._save_locked()

    restarted = make_service(root, accounts, FakeHttp([]))
    result = restarted.get(OWNER, "owned", started["id"])
    assert result["state"] == "interrupted"
    assert result["error_code"] == "chat_login_exchange_outcome_unknown"
    persisted = (root / "chat_login_sessions.json").read_text()
    assert "code_verifier" not in persisted


def test_concurrent_duplicate_callback_has_one_exchange(root):
    accounts = FakeAccounts()
    entered = threading.Event()
    release = threading.Event()

    class BlockingHttp(FakeHttp):
        def factory(self):
            parent = self

            class Session:
                def post(self, url, **kwargs):
                    parent.calls.append((url, kwargs))
                    entered.set()
                    release.wait(2)
                    return FakeResponse(200, {
                        "access_token": "concurrent-access",
                        "refresh_token": "concurrent-refresh",
                        "id_token": "concurrent-id",
                    })

                def close(self):
                    pass

            return Session()

    http = BlockingHttp([])
    service = make_service(root, accounts, http)
    started = service.start(OWNER, "owned", "import", str(uuid.uuid4()))
    callback = callback_for(started)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.submit_callback, OWNER, "owned", started["id"], callback)
        assert entered.wait(1)
        second = executor.submit(service.submit_callback, OWNER, "owned", started["id"], callback)
        assert second.result(timeout=1)["state"] == "exchanging"
        release.set()
        assert first.result(timeout=2)["state"] == "succeeded"
    assert len(http.calls) == 1


def test_cancel_only_pending_and_pool_requires_attach(root):
    accounts = FakeAccounts()
    service = make_service(root, accounts, FakeHttp([]))
    with pytest.raises(ChatLoginError, match="invalid_mode"):
        service.start(OWNER, "pool", "import", str(uuid.uuid4()))
    started = service.start(OWNER, "owned", "import", str(uuid.uuid4()))
    assert service.cancel(OWNER, "owned", started["id"])["state"] == "cancelled"
    assert service.cancel(OWNER, "owned", started["id"])["state"] == "cancelled"
    with pytest.raises(ChatLoginError, match="not_found"):
        service.get("workbench:org:other", "owned", started["id"])


def test_http_factory_forces_verified_tls(root):
    with patch("services.chat_login_service.proxy_settings.build_session_kwargs", return_value={"verify": False}) as build, \
            patch("services.chat_login_service.requests.Session", return_value=object()) as create:
        ChatLoginService._new_http_session()
    build.assert_called_once_with(upstream=True, impersonate="chrome")
    create.assert_called_once_with(verify=True)


def test_unknown_save_reads_committed_result_after_restart_without_exchange(root):
    class CommitThenLoseReceipt(FakeAccounts):
        committed = False
        def import_owned_account(self, owner, payload, *, verified_oauth=False):
            assert verified_oauth is True
            self.imports.append((owner, payload))
            self.committed = True
            raise CodexAuthorizationAttachError("chat_authorization_import_unknown")

        def chat_login_committed_receipt(self, credentials, account_ref=None):
            return {"authorization_ref": ACCOUNT_REF, "import_status": "unchanged"} if self.committed else None

    accounts = CommitThenLoseReceipt()
    http = FakeHttp([FakeResponse(200, {"access_token": "save-once", "refresh_token": "refresh-once"})])
    service = make_service(root, accounts, http)
    started = service.start(OWNER, "owned", "import", str(uuid.uuid4()))
    assert service.submit_callback(OWNER, "owned", started["id"], callback_for(started))["state"] == "saving"
    restored = make_service(root, accounts, FakeHttp([]))
    assert restored.get(OWNER, "owned", started["id"])["state"] == "succeeded"
    assert len(http.calls) == len(accounts.imports) == 1
    assert "save-once" not in (root / "chat_login_sessions.json").read_text()


def test_verified_oauth_import_does_not_refresh_again_and_readback_checks_target(root):
    from services.account_service import AccountService
    from services.storage.json_storage import JSONStorageBackend
    from test.test_codex_dual_authorization import jwt, SUBJECT, ACCOUNT_ID

    accounts = AccountService(JSONStorageBackend(root / "accounts.json"))
    material = {"access_token": jwt(SUBJECT, marker="chat-access"), "refresh_token": "new-refresh",
                "id_token": jwt(SUBJECT, marker="chat-id"), "source_type": "oauth_login"}
    info = {"user_id": SUBJECT, "account_id": ACCOUNT_ID, "quota": 5}
    with patch.object(accounts, "_verified_chat_info", return_value=((SUBJECT, ACCOUNT_ID), info)), \
         patch.object(accounts, "_request_access_token_refresh", side_effect=AssertionError("must not rotate again")):
        accounts.import_owned_account(OWNER, material, verified_oauth=True)
    restored = AccountService(JSONStorageBackend(root / "accounts.json"))
    receipt = restored.chat_login_committed_receipt(material)
    assert receipt and receipt["import_status"] == "unchanged"
    assert restored.chat_login_committed_receipt({**material, "refresh_token": "wrong"}) is None
    with pytest.raises(CodexAuthorizationAttachError, match="account_conflict"):
        restored.chat_login_committed_receipt(material, ACCOUNT_REF)
