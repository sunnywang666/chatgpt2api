"""Durable Workbench Chat OAuth sessions using a manual callback handoff."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlencode, urlsplit

from curl_cffi import requests

from services.account_service import (
    AccountService,
    CodexAuthorizationAttachError,
    account_service,
)
from services.config import DATA_DIR
from services.openai_oauth import (
    auth_base,
    common_headers,
    platform_auth0_client,
    platform_base,
    platform_oauth_audience,
    platform_oauth_client_id,
    platform_oauth_redirect_uri,
    sec_ch_ua,
    user_agent,
)
from services.proxy_service import proxy_settings
from utils.pkce import generate_pkce


class ChatLoginError(ValueError):
    def __init__(self, status_code: int, code: str):
        super().__init__(code)
        self.status_code = status_code
        self.code = code


class ChatLoginService:
    AUTHORIZE_URL = f"{auth_base}/api/accounts/authorize"
    TOKEN_URL = f"{auth_base}/api/accounts/oauth/token"
    REDIRECT_URI = platform_oauth_redirect_uri
    CLIENT_ID = platform_oauth_client_id
    RETURN_MODE = "manual_callback"

    SESSION_TTL_SECONDS = 10 * 60
    MAX_ACTIVE_GLOBAL = 32
    MAX_ACTIVE_PER_OWNER = 4
    MAX_RETAINED = 256
    TERMINAL_STATES = {"succeeded", "cancelled", "expired", "failed", "interrupted"}
    ACTIVE_STATES = {"pending_callback", "exchanging", "saving"}
    PUBLIC_FIELDS = (
        "id",
        "state",
        "return_mode",
        "authorize_url",
        "expires_at",
        "account_ref",
        "import_status",
        "route",
        "capacity",
        "error_code",
    )
    SENSITIVE_FIELDS = (
        "oauth_state",
        "code_verifier",
        "pending_credentials",
        "target_revision",
    )

    def __init__(
        self,
        path: Path | None = None,
        accounts: AccountService | None = None,
        session_factory: Callable[[], object] | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.path = path or DATA_DIR / "chat_login_sessions.json"
        self.accounts = accounts or account_service
        self._session_factory = session_factory or self._new_http_session
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._saving_sessions: set[str] = set()
        self._sessions = self._load_sessions()
        changed = False
        for session in self._sessions.values():
            if session.get("state") == "exchanging":
                session["state"] = "interrupted"
                session["error_code"] = "chat_login_exchange_outcome_unknown"
                self._erase_sensitive(session)
                changed = True
            elif session.get("state") == "saving" and not isinstance(
                session.get("pending_credentials"), dict
            ):
                session["state"] = "interrupted"
                session["error_code"] = "chat_login_save_failed"
                self._erase_sensitive(session)
                changed = True
        if changed:
            self._save_locked()

    @staticmethod
    def _iso(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    @staticmethod
    def _request_hash(scope: str, mode: str, account_ref: str) -> str:
        material = json.dumps(
            {"scope": scope, "mode": mode, "account_ref": account_ref},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(material).hexdigest()

    def _load_sessions(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError("chat login session storage is unreadable") from exc
        items = payload.get("sessions") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise RuntimeError("chat login session storage is invalid")
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
        fd, temp_name = tempfile.mkstemp(prefix=".chat-login-", dir=self.path.parent)
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
        session.pop("authorize_url", None)

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
            (str(item.get("created_at") or ""), session_id)
            for session_id, item in self._sessions.items()
            if item.get("state") in self.TERMINAL_STATES
        )
        for _created, session_id in terminal[: len(self._sessions) - self.MAX_RETAINED]:
            self._sessions.pop(session_id, None)

    @staticmethod
    def _new_http_session():
        kwargs = proxy_settings.build_session_kwargs(upstream=True, impersonate="chrome")
        kwargs["verify"] = True
        return requests.Session(**kwargs)

    @staticmethod
    def _status(response: object) -> int:
        try:
            return int(getattr(response, "status_code", 0) or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _json_object(response: object) -> dict:
        try:
            value = response.json()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def _lookup_locked(self, owner: str, scope: str, session_id: str) -> dict:
        session = self._sessions.get(str(session_id or "").strip())
        if not session or session.get("owner") != owner or session.get("scope") != scope:
            raise ChatLoginError(404, "chat_login_not_found")
        return session

    def _active_counts_locked(self, owner: str) -> tuple[int, int]:
        active = [item for item in self._sessions.values() if item.get("state") in self.ACTIVE_STATES]
        return len(active), sum(item.get("owner") == owner for item in active)

    def _expire_locked(self, session: dict) -> bool:
        if session.get("state") != "pending_callback":
            return False
        if self._clock() < float(session.get("expires_epoch") or 0):
            return False
        session["state"] = "expired"
        session["error_code"] = "chat_login_expired"
        self._erase_sensitive(session)
        self._save_locked()
        return True

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
            raise ChatLoginError(422, "chat_login_invalid_request_id") from None
        if scope not in {"owned", "pool"}:
            raise ChatLoginError(404, "chat_login_not_found")
        if mode not in {"import", "attach"} or (scope == "pool" and mode != "attach"):
            raise ChatLoginError(422, "chat_login_invalid_mode")
        if (mode == "attach") != bool(account_ref):
            code = "chat_login_account_ref_required" if mode == "attach" else "chat_login_account_ref_not_allowed"
            raise ChatLoginError(422, code)

        request_hash = self._request_hash(scope, mode, account_ref)
        with self._lock:
            for item in self._sessions.values():
                if item.get("owner") == owner and item.get("client_request_id") == request_id:
                    if item.get("request_hash") != request_hash:
                        raise ChatLoginError(409, "chat_login_idempotency_conflict")
                    self._expire_locked(item)
                    return self._public(item)
            for previous in list(self._sessions.values()):
                self._expire_locked(previous)
            target_revision = None
            if mode == "attach":
                target = self.accounts.codex_login_target(owner, account_ref, pool=scope == "pool")
                target_revision = target["revision"]
            global_count, owner_count = self._active_counts_locked(owner)
            if global_count >= self.MAX_ACTIVE_GLOBAL or owner_count >= self.MAX_ACTIVE_PER_OWNER:
                raise ChatLoginError(429, "chat_login_capacity")

            verifier, challenge = generate_pkce()
            session_id = uuid.uuid4().hex
            oauth_state = f"{session_id}.{secrets.token_urlsafe(16)}"
            now = self._clock()
            params = {
                "issuer": auth_base,
                "client_id": self.CLIENT_ID,
                "audience": platform_oauth_audience,
                "redirect_uri": self.REDIRECT_URI,
                "device_id": str(uuid.uuid4()),
                "screen_hint": "login_or_signup",
                "max_age": "0",
                "scope": "openid profile email offline_access",
                "response_type": "code",
                "response_mode": "query",
                "state": oauth_state,
                "nonce": secrets.token_urlsafe(32),
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "auth0Client": platform_auth0_client,
            }
            session = {
                "id": session_id,
                "owner": owner,
                "scope": scope,
                "mode": mode,
                "client_request_id": request_id,
                "request_hash": request_hash,
                "state": "pending_callback",
                "return_mode": self.RETURN_MODE,
                "authorize_url": f"{self.AUTHORIZE_URL}?{urlencode(params)}",
                "oauth_state": oauth_state,
                "code_verifier": verifier,
                "created_at": self._iso(now),
                "expires_at": self._iso(now + self.SESSION_TTL_SECONDS),
                "expires_epoch": now + self.SESSION_TTL_SECONDS,
                "account_ref": account_ref or None,
                "target_revision": target_revision,
                "error_code": None,
            }
            self._sessions[session_id] = session
            self._prune_locked()
            self._save_locked()
            return self._public(session)

    @classmethod
    def _parse_callback(cls, callback_url: str) -> tuple[str, str, str | None]:
        raw = str(callback_url or "").strip()
        if not raw or len(raw) > 30_000:
            raise ChatLoginError(422, "chat_login_invalid_callback")
        try:
            parsed = urlsplit(raw)
        except ValueError:
            raise ChatLoginError(422, "chat_login_invalid_callback") from None
        if (
            parsed.scheme != "https"
            or parsed.netloc != "platform.openai.com"
            or parsed.path != "/auth/callback"
            or parsed.fragment
        ):
            raise ChatLoginError(422, "chat_login_invalid_callback")
        try:
            query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
        except ValueError:
            raise ChatLoginError(422, "chat_login_invalid_callback") from None
        # OpenAI includes scope in the redirect. It is metadata, not a claim of granted access.
        allowed = {"code", "state", "scope", "error", "error_description"}
        if set(query) - allowed or any(len(values) != 1 for values in query.values()):
            raise ChatLoginError(422, "chat_login_invalid_callback")
        state = str((query.get("state") or [""])[0]).strip()
        code = str((query.get("code") or [""])[0]).strip()
        error = str((query.get("error") or [""])[0]).strip() or None
        if not state or len(state) > 1000 or len(code) > 20_000:
            raise ChatLoginError(422, "chat_login_invalid_callback")
        if (not code) == (error is None):
            raise ChatLoginError(422, "chat_login_invalid_callback")
        return code, state, error

    def submit_callback(self, owner: str, scope: str, session_id: str, callback_url: str) -> dict:
        code, state, callback_error = self._parse_callback(callback_url)
        callback_digest = hashlib.sha256(str(callback_url).strip().encode()).hexdigest()
        with self._lock:
            session = self._lookup_locked(owner, scope, session_id)
            self._expire_locked(session)
            previous_digest = str(session.get("callback_digest") or "")
            if previous_digest:
                if not hmac.compare_digest(previous_digest, callback_digest):
                    raise ChatLoginError(409, "chat_login_callback_conflict")
                return self._public(session)
            if session.get("state") != "pending_callback":
                raise ChatLoginError(409, "chat_login_not_submittable")
            if not hmac.compare_digest(str(session.get("oauth_state") or ""), state):
                raise ChatLoginError(409, "chat_login_state_mismatch")
            session["callback_digest"] = callback_digest
            if callback_error is not None:
                session["state"] = "failed"
                session["error_code"] = "chat_login_authorization_rejected"
                self._erase_sensitive(session)
                self._save_locked()
                return self._public(session)
            verifier = str(session.get("code_verifier") or "")
            session["state"] = "exchanging"
            session["error_code"] = None
            session.pop("authorize_url", None)
            session.pop("oauth_state", None)
            session.pop("code_verifier", None)
            self._save_locked()

        try:
            client = self._session_factory()
            try:
                response = client.post(
                    self.TOKEN_URL,
                    headers={
                        **common_headers,
                        "referer": f"{platform_base}/",
                        "origin": platform_base,
                        "auth0-client": platform_auth0_client,
                        "sec-ch-ua": sec_ch_ua,
                        "user-agent": user_agent,
                    },
                    json={
                        "client_id": self.CLIENT_ID,
                        "code_verifier": verifier,
                        "grant_type": "authorization_code",
                        "code": code,
                        "redirect_uri": self.REDIRECT_URI,
                    },
                    timeout=60,
                    allow_redirects=False,
                )
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        except Exception:
            return self._finish_exchange_failure(session_id, "chat_login_exchange_outcome_unknown", interrupted=True)

        status = self._status(response)
        if status != 200:
            unknown = status in {0, 408, 429} or status >= 500
            code_name = "chat_login_exchange_outcome_unknown" if unknown else "chat_login_exchange_rejected"
            return self._finish_exchange_failure(session_id, code_name, interrupted=unknown)
        data = self._json_object(response)
        credentials = {
            "access_token": str(data.get("access_token") or "").strip(),
            "refresh_token": str(data.get("refresh_token") or "").strip(),
            "id_token": str(data.get("id_token") or "").strip(),
        }
        if not credentials["access_token"] or not credentials["refresh_token"]:
            return self._finish_exchange_failure(session_id, "chat_login_invalid_authorization")
        if any(len(value) > 20_000 for value in credentials.values()):
            return self._finish_exchange_failure(session_id, "chat_login_invalid_authorization")

        with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.get("state") != "exchanging":
                raise ChatLoginError(409, "chat_login_not_submittable")
            session["pending_credentials"] = credentials
            session["state"] = "saving"
            session["error_code"] = None
            self._save_locked()
        return self._resume_save(owner, scope, session_id)

    def _finish_exchange_failure(self, session_id: str, error_code: str, *, interrupted: bool = False) -> dict:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ChatLoginError(404, "chat_login_not_found")
            session["state"] = "interrupted" if interrupted else "failed"
            session["error_code"] = error_code
            self._erase_sensitive(session)
            self._save_locked()
            return self._public(session)

    def _resume_save(self, owner: str, scope: str, session_id: str) -> dict:
        with self._lock:
            session = self._lookup_locked(owner, scope, session_id)
            if session.get("state") != "saving":
                return self._public(session)
            if session_id in self._saving_sessions:
                return self._public(session)
            credentials = session.get("pending_credentials")
            if not isinstance(credentials, dict):
                session["state"] = "interrupted"
                session["error_code"] = "chat_login_save_failed"
                self._erase_sensitive(session)
                self._save_locked()
                return self._public(session)
            self._saving_sessions.add(session_id)
            session_snapshot = dict(session)

        try:
            committed = self.accounts.chat_login_committed_receipt(credentials, session_snapshot.get("account_ref"))
            if session_snapshot.get("mode") == "attach" and committed is None:
                target = self.accounts.codex_login_target(
                    owner,
                    str(session_snapshot.get("account_ref") or ""),
                    pool=scope == "pool",
                )
                if target.get("revision") != session_snapshot.get("target_revision"):
                    raise CodexAuthorizationAttachError("chat_authorization_stale_target")
            payload = {**credentials, "source_type": "oauth_login"}
            if session_snapshot.get("mode") == "attach":
                payload["account_ref"] = session_snapshot.get("account_ref")
            item = committed or self.accounts.import_owned_account(owner, payload, verified_oauth=True)
        except CodexAuthorizationAttachError as exc:
            with self._lock:
                current = self._sessions.get(session_id)
                if current and current.get("state") == "saving":
                    if exc.code == "chat_authorization_import_unknown":
                        # Keep the already exchanged material for authoritative
                        # account-storage readback. Never exchange the code again.
                        current["error_code"] = "chat_login_save_interrupted"
                    else:
                        current["state"] = "failed"
                        current["error_code"] = exc.code
                        self._erase_sensitive(current)
                    self._save_locked()
                return self._public(current or session_snapshot)
        except Exception:
            # Tokens were durably saved before import began. Keep the original
            # saving state so GET after a process restart can retry the
            # idempotent identity merge without exchanging the code again.
            with self._lock:
                current = self._sessions.get(session_id)
                if current and current.get("state") == "saving":
                    current["error_code"] = "chat_login_save_interrupted"
                    self._save_locked()
                return self._public(current or session_snapshot)
        finally:
            with self._lock:
                self._saving_sessions.discard(session_id)

        account_ref = str(item.get("authorization_ref") or item.get("id") or "").strip() or None
        import_status = str(item.get("import_status") or "created").strip()
        if import_status not in {"created", "updated", "unchanged"}:
            import_status = "created"
        with self._lock:
            current = self._sessions.get(session_id)
            if not current or current.get("state") != "saving":
                raise ChatLoginError(409, "chat_login_not_submittable")
            current.update(
                state="succeeded",
                account_ref=account_ref or current.get("account_ref"),
                import_status=import_status,
                route="chat",
                capacity=item.get("capacity"),
                error_code=None,
            )
            self._erase_sensitive(current)
            self._save_locked()
            return self._public(current)

    def get(self, owner: str, scope: str, session_id: str) -> dict:
        with self._lock:
            session = self._lookup_locked(owner, scope, session_id)
            self._expire_locked(session)
            should_resume = session.get("state") == "saving"
            result = self._public(session)
        return self._resume_save(owner, scope, session_id) if should_resume else result

    def cancel(self, owner: str, scope: str, session_id: str) -> dict:
        with self._lock:
            session = self._lookup_locked(owner, scope, session_id)
            self._expire_locked(session)
            state = session.get("state")
            if state == "pending_callback":
                session["state"] = "cancelled"
                session["error_code"] = None
                self._erase_sensitive(session)
                self._save_locked()
            elif state == "cancelled":
                pass
            elif state in {"exchanging", "saving"}:
                raise ChatLoginError(409, "chat_login_completion_in_progress")
            else:
                raise ChatLoginError(409, "chat_login_not_cancellable")
            return self._public(session)


chat_login_service = ChatLoginService()
