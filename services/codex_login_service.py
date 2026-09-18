"""Durable, owner-scoped official Codex device authorization sessions."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from curl_cffi import requests

from services.account_service import (
    AccountService,
    CodexAuthorizationAttachError,
    account_service,
)
from services.config import DATA_DIR
from services.proxy_service import proxy_settings


class CodexLoginError(ValueError):
    def __init__(self, status_code: int, code: str):
        super().__init__(code)
        self.status_code = status_code
        self.code = code


class CodexLoginService:
    DEVICE_USER_CODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
    DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
    OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
    VERIFICATION_URL = "https://auth.openai.com/codex/device"
    REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"
    CLIENT_ID = AccountService._CODEX_OAUTH_CLIENT_ID

    SESSION_TTL_SECONDS = 15 * 60
    MAX_ACTIVE_GLOBAL = 32
    MAX_ACTIVE_PER_OWNER = 4
    MAX_RETAINED = 256
    MAX_POLL_INTERVAL_SECONDS = SESSION_TTL_SECONDS
    TERMINAL_STATES = {"succeeded", "cancelled", "expired", "failed", "interrupted"}
    PUBLIC_FIELDS = (
        "id",
        "state",
        "verification_url",
        "user_code",
        "expires_at",
        "poll_after_seconds",
        "account_ref",
        "import_status",
        "codex",
        "error_code",
    )
    SENSITIVE_FIELDS = ("device_auth_id", "user_code", "verification_url")

    def __init__(
        self,
        path: Path | None = None,
        accounts: AccountService | None = None,
        session_factory: Callable[[], object] | None = None,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
        auto_start_workers: bool = True,
    ) -> None:
        self.path = path or DATA_DIR / "codex_login_sessions.json"
        self.accounts = accounts or account_service
        self._session_factory = session_factory or self._new_http_session
        self._clock = clock or time.time
        self._sleeper = sleeper or time.sleep
        self._auto_start_workers = auto_start_workers
        self._lock = threading.RLock()
        self._sessions = self._load_sessions()
        changed = False
        for session in self._sessions.values():
            if session.get("state") in {"completing", "interrupted"} and session.get("completion_authorization_ref"):
                self._reconcile_completion_locked(session)
                changed = True
            elif session.get("state") in {"pending", "completing"}:
                session["state"] = "interrupted"
                session["error_code"] = "codex_login_interrupted"
                self._erase_terminal(session)
                changed = True
        if changed:
            self._save_locked()

    @staticmethod
    def _iso(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    @staticmethod
    def _request_hash(scope: str, mode: str, account_ref: str) -> str:
        value = json.dumps(
            {"scope": scope, "mode": mode, "account_ref": account_ref},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    def _load_sessions(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError("codex login session storage is unreadable") from exc
        items = payload.get("sessions") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise RuntimeError("codex login session storage is invalid")
        result: dict[str, dict] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            session_id = str(item.get("id") or "").strip()
            if session_id:
                result[session_id] = dict(item)
        return result

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".codex-login-", dir=self.path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                json.dump(
                    {"version": 1, "sessions": list(self._sessions.values())},
                    target,
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temp_name, self.path)
            os.chmod(self.path, 0o600)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    @classmethod
    def _erase_sensitive(cls, session: dict) -> None:
        for key in cls.SENSITIVE_FIELDS:
            session.pop(key, None)

    @classmethod
    def _erase_terminal(cls, session: dict) -> None:
        cls._erase_sensitive(session)
        session.pop("target_revision", None)
        session.pop("completion_authorization_ref", None)
        session.pop("completion_credential_digest", None)

    @classmethod
    def _public(cls, session: dict) -> dict:
        return {
            key: session[key]
            for key in cls.PUBLIC_FIELDS
            if session.get(key) is not None
        }

    def _prune_locked(self) -> None:
        if len(self._sessions) <= self.MAX_RETAINED:
            return
        terminal = sorted(
            (
                (str(item.get("created_at") or ""), session_id)
                for session_id, item in self._sessions.items()
                if item.get("state") in self.TERMINAL_STATES
            )
        )
        for _created_at, session_id in terminal[: len(self._sessions) - self.MAX_RETAINED]:
            self._sessions.pop(session_id, None)

    def _new_http_session(self):
        kwargs = proxy_settings.build_session_kwargs(upstream=True, impersonate="chrome")
        # OAuth credentials must never cross an unverified TLS connection even
        # when the general proxy runtime is configured for diagnostic skipping.
        kwargs["verify"] = True
        return requests.Session(**kwargs)

    @staticmethod
    def _json_object(response: object) -> dict:
        try:
            value = response.json()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _status(response: object) -> int:
        try:
            return int(getattr(response, "status_code", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _active_counts_locked(self, owner: str) -> tuple[int, int]:
        active = [item for item in self._sessions.values() if item.get("state") in {"pending", "completing"}]
        return len(active), sum(item.get("owner") == owner for item in active)

    def _lookup_locked(self, owner: str, scope: str, session_id: str) -> dict:
        session = self._sessions.get(str(session_id or "").strip())
        if not session or session.get("owner") != owner or session.get("scope") != scope:
            raise CodexLoginError(404, "codex_login_not_found")
        return session

    def start(
        self,
        owner: str,
        scope: str,
        mode: str,
        client_request_id: str,
        account_ref: str | None = None,
    ) -> dict:
        owner = str(owner or "").strip()
        scope = str(scope or "").strip()
        mode = str(mode or "").strip()
        account_ref = str(account_ref or "").strip()
        try:
            request_id = str(uuid.UUID(str(client_request_id or "")))
        except (TypeError, ValueError, AttributeError):
            raise CodexLoginError(422, "codex_login_invalid_request_id") from None
        if scope not in {"owned", "pool"}:
            raise CodexLoginError(404, "codex_login_not_found")
        if mode not in {"import", "attach"} or (scope == "pool" and mode != "attach"):
            raise CodexLoginError(422, "codex_login_invalid_mode")
        if (mode == "attach") != bool(account_ref):
            raise CodexLoginError(422, "codex_login_account_ref_required" if mode == "attach" else "codex_login_account_ref_not_allowed")

        request_hash = self._request_hash(scope, mode, account_ref)
        with self._lock:
            for item in self._sessions.values():
                if item.get("owner") == owner and item.get("client_request_id") == request_id:
                    if item.get("request_hash") != request_hash:
                        raise CodexLoginError(409, "codex_login_idempotency_conflict")
                    return self._public(item)
            target_revision = None
            if mode == "attach":
                target = self.accounts.codex_login_target(owner, account_ref, pool=scope == "pool")
                target_revision = target["revision"]
            global_count, owner_count = self._active_counts_locked(owner)
            if global_count >= self.MAX_ACTIVE_GLOBAL or owner_count >= self.MAX_ACTIVE_PER_OWNER:
                raise CodexLoginError(429, "codex_login_capacity")
            now = self._clock()
            session_id = uuid.uuid4().hex
            session = {
                "id": session_id,
                "owner": owner,
                "scope": scope,
                "mode": mode,
                "client_request_id": request_id,
                "request_hash": request_hash,
                "state": "completing",
                "created_at": self._iso(now),
                "expires_at": self._iso(now + self.SESSION_TTL_SECONDS),
                "expires_epoch": now + self.SESSION_TTL_SECONDS,
                "poll_after_seconds": 5,
                "account_ref": account_ref or None,
                "target_revision": target_revision,
                "error_code": None,
            }
            self._sessions[session_id] = session
            self._prune_locked()
            self._save_locked()

        try:
            client = self._session_factory()
            try:
                response = client.post(
                    self.DEVICE_USER_CODE_URL,
                    headers={"Accept": "application/json", "Content-Type": "application/json"},
                    json={"client_id": self.CLIENT_ID},
                    timeout=30,
                    allow_redirects=False,
                )
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        except Exception:
            self._fail(session_id, "codex_login_device_authorization_unavailable")
            return self.get(owner, scope, session_id)

        data = self._json_object(response)
        device_auth_id = str(data.get("device_auth_id") or "").strip()
        user_code = str(data.get("user_code") or data.get("usercode") or "").strip()
        try:
            interval = int(str(data.get("interval") or "5").strip())
        except (TypeError, ValueError):
            interval = 5
        interval = max(1, min(interval, self.MAX_POLL_INTERVAL_SECONDS))
        if self._status(response) != 200:
            self._fail(session_id, "codex_login_device_authorization_unavailable")
            return self.get(owner, scope, session_id)
        if not device_auth_id or len(device_auth_id) > 1000 or not user_code or len(user_code) > 200:
            self._fail(session_id, "codex_login_device_authorization_invalid")
            return self.get(owner, scope, session_id)

        with self._lock:
            current = self._sessions[session_id]
            current.update(
                state="pending",
                verification_url=self.VERIFICATION_URL,
                user_code=user_code,
                device_auth_id=device_auth_id,
                poll_after_seconds=interval,
            )
            self._save_locked()
            result = self._public(current)
        if self._auto_start_workers:
            threading.Thread(
                target=self._run,
                args=(session_id,),
                name=f"codex-login-{session_id[:8]}",
                daemon=True,
            ).start()
        return result

    def _reconcile_completion_locked(self, session: dict) -> None:
        try:
            readback = self.accounts.codex_login_completion_readback(
                str(session.get("owner") or ""), str(session.get("scope") or ""),
                str(session.get("mode") or ""), str(session.get("account_ref") or ""),
                str(session.get("completion_authorization_ref") or ""),
                str(session.get("completion_credential_digest") or ""),
            )
        except Exception:
            session["state"] = "interrupted"
            session["error_code"] = "codex_login_save_failed"
            self._erase_sensitive(session)
            # Keep the original operation's identity/digest for later GET or
            # restart readback. No token material or repeat exchange is needed.
            return
        if readback.get("applied"):
            session["state"] = "succeeded"
            session["account_ref"] = readback.get("account_ref") or session.get("account_ref")
            if session.get("mode") == "import":
                session["import_status"] = readback.get("import_status")
                session["codex"] = readback.get("codex")
            session["error_code"] = None
        else:
            session["state"] = "failed"
            session["error_code"] = "codex_login_save_not_applied"
        self._erase_terminal(session)

    def get(self, owner: str, scope: str, session_id: str) -> dict:
        with self._lock:
            session = self._lookup_locked(owner, scope, session_id)
            if session.get("state") == "interrupted" and session.get("completion_authorization_ref"):
                self._reconcile_completion_locked(session)
                self._save_locked()
            return self._public(session)

    def cancel(self, owner: str, scope: str, session_id: str) -> dict:
        with self._lock:
            session = self._lookup_locked(owner, scope, session_id)
            state = session.get("state")
            if state == "pending":
                session["state"] = "cancelled"
                session["error_code"] = None
                self._erase_terminal(session)
                self._save_locked()
            elif state == "cancelled":
                pass
            elif state == "completing":
                raise CodexLoginError(409, "codex_login_completion_in_progress")
            else:
                raise CodexLoginError(409, "codex_login_not_cancellable")
            return self._public(session)

    def _fail(self, session_id: str, error_code: str, *, state: str = "failed") -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.get("state") in self.TERMINAL_STATES:
                return
            session["state"] = state
            session["error_code"] = error_code
            self._erase_terminal(session)
            self._save_locked()

    def _expire_if_needed(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.get("state") != "pending":
                return True
            if self._clock() < float(session.get("expires_epoch") or 0):
                return False
            session["state"] = "expired"
            session["error_code"] = "codex_login_expired"
            self._erase_terminal(session)
            self._save_locked()
            return True

    def _wait_pending(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.get("state") != "pending":
                return False
            remaining = max(0.0, float(session.get("expires_epoch") or 0) - self._clock())
            if remaining <= 0:
                interval = 0.0
            else:
                interval = min(
                    remaining,
                    max(1, min(int(session.get("poll_after_seconds") or 5), self.MAX_POLL_INTERVAL_SECONDS)),
                )
        if interval:
            self._sleeper(interval)
        return not self._expire_if_needed(session_id)

    def _run(self, session_id: str) -> None:
        while True:
            if not self._wait_pending(session_id):
                return
            with self._lock:
                session = self._sessions.get(session_id)
                if not session or session.get("state") != "pending":
                    return
                device_auth_id = str(session.get("device_auth_id") or "")
                user_code = str(session.get("user_code") or "")
            try:
                client = self._session_factory()
                try:
                    response = client.post(
                        self.DEVICE_TOKEN_URL,
                        headers={"Accept": "application/json", "Content-Type": "application/json"},
                        json={"device_auth_id": device_auth_id, "user_code": user_code},
                        timeout=30,
                        allow_redirects=False,
                    )
                finally:
                    close = getattr(client, "close", None)
                    if callable(close):
                        close()
            except Exception:
                self._fail(session_id, "codex_login_poll_unavailable")
                return

            status = self._status(response)
            if status in {403, 404}:
                continue
            if status == 429:
                try:
                    retry_after = int(str((getattr(response, "headers", {}) or {}).get("Retry-After") or ""))
                except (TypeError, ValueError):
                    retry_after = 0
                with self._lock:
                    session = self._sessions.get(session_id)
                    if not session or session.get("state") != "pending":
                        return
                    current = int(session.get("poll_after_seconds") or 5)
                    session["poll_after_seconds"] = max(
                        current,
                        min(max(1, retry_after or current * 2), self.MAX_POLL_INTERVAL_SECONDS),
                    )
                    self._save_locked()
                continue
            if status != 200:
                self._fail(session_id, "codex_login_poll_rejected")
                return

            data = self._json_object(response)
            authorization_code = str(data.get("authorization_code") or "").strip()
            code_verifier = str(data.get("code_verifier") or "").strip()
            code_challenge = str(data.get("code_challenge") or "").strip()
            try:
                verifier_bytes = code_verifier.encode("ascii")
            except UnicodeEncodeError:
                verifier_bytes = b""
            expected_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(verifier_bytes).digest()
            ).decode("ascii").rstrip("=")
            if (
                not authorization_code
                or len(authorization_code) > 20_000
                or not 43 <= len(code_verifier) <= 128
                or not verifier_bytes
                or not code_challenge
                or not hmac.compare_digest(code_challenge, expected_challenge)
            ):
                self._fail(session_id, "codex_login_poll_invalid")
                return
            with self._lock:
                session = self._sessions.get(session_id)
                if not session or session.get("state") != "pending":
                    return
                session["state"] = "completing"
                self._erase_sensitive(session)
                self._save_locked()
            self._exchange_and_complete(session_id, authorization_code, code_verifier)
            return

    def _exchange_and_complete(self, session_id: str, authorization_code: str, code_verifier: str) -> None:
        try:
            client = self._session_factory()
            try:
                response = client.post(
                    self.OAUTH_TOKEN_URL,
                    headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
                    data={
                        "grant_type": "authorization_code",
                        "code": authorization_code,
                        "redirect_uri": self.REDIRECT_URI,
                        "client_id": self.CLIENT_ID,
                        "code_verifier": code_verifier,
                    },
                    timeout=30,
                    allow_redirects=False,
                )
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        except Exception:
            self._fail(session_id, "codex_login_exchange_outcome_unknown")
            return
        status = self._status(response)
        if status != 200:
            code = "codex_login_exchange_outcome_unknown" if status == 0 or status == 408 or status == 429 or status >= 500 else "codex_login_exchange_rejected"
            self._fail(session_id, code)
            return
        data = self._json_object(response)
        access_token = str(data.get("access_token") or "").strip()
        refresh_token = str(data.get("refresh_token") or "").strip()
        id_token = str(data.get("id_token") or "").strip()
        try:
            workspace_ids = AccountService._jwt_workspace_ids(access_token, id_token)
            response_workspace = AccountService._validated_workspace_id(data.get("account_id"))
            if response_workspace:
                workspace_ids.add(response_workspace)
            if len(workspace_ids) != 1:
                raise CodexAuthorizationAttachError("codex_authorization_invalid_material")
            credentials = AccountService._validated_codex_credentials({
                "access_token": access_token,
                "refresh_token": refresh_token,
                "id_token": id_token,
                "account_id": next(iter(workspace_ids)),
            })
        except CodexAuthorizationAttachError:
            self._fail(session_id, "codex_login_invalid_authorization")
            return

        try:
            with self._lock:
                session = self._sessions.get(session_id)
                if not session or session.get("state") != "completing":
                    return
                authorization_ref = AccountService._authorization_ref(
                    *AccountService._codex_identity(credentials)
                )
                session["completion_authorization_ref"] = authorization_ref
                session["completion_credential_digest"] = AccountService.codex_credential_digest(credentials)
                self._save_locked()
                if session["mode"] == "import":
                    item = self.accounts.import_owned_codex_authorization(
                        session["owner"], credentials, verified_exchange=True
                    )
                    account_ref = str(item.get("authorization_ref") or "") or None
                    session["import_status"] = item["import_status"]
                    session["codex"] = item["codex"]
                elif session["scope"] == "pool":
                    self.accounts.attach_codex_authorization(
                        credentials,
                        session.get("account_ref"),
                        session.get("target_revision"),
                    )
                    account_ref = session.get("account_ref")
                else:
                    target = self.accounts.codex_login_target(
                        session["owner"], session.get("account_ref") or "", pool=False
                    )
                    # The path ID is intentionally absent from the device-login
                    # contract; the opaque reference resolves the owner target.
                    if target["revision"] != session.get("target_revision"):
                        raise CodexAuthorizationAttachError("codex_authorization_stale_target")
                    self.accounts.attach_codex_authorization(
                        credentials,
                        session.get("account_ref"),
                        session.get("target_revision"),
                    )
                    account_ref = session.get("account_ref")
                session["state"] = "succeeded"
                session["account_ref"] = account_ref
                session["error_code"] = None
                self._erase_terminal(session)
                try:
                    self._save_locked()
                except Exception:
                    # The account mutation already completed. Keep the current
                    # process truthful; the previously persisted completing
                    # record is reconciled by safe account readback on restart.
                    pass
        except CodexAuthorizationAttachError as exc:
            code = "codex_login_stale_target" if exc.code == "codex_authorization_stale_target" else exc.code
            self._fail(session_id, code)
        except Exception:
            with self._lock:
                session = self._sessions.get(session_id)
                if session and session.get("completion_authorization_ref"):
                    self._reconcile_completion_locked(session)
                    try:
                        self._save_locked()
                    except Exception:
                        # Previously persisted completing record stays the
                        # recovery authority when the session store also fails.
                        pass
                else:
                    self._fail(session_id, "codex_login_save_failed")


codex_login_service = CodexLoginService()
