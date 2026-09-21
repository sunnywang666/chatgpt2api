"""Native Codex HTTP bridge over the existing ChatGPT account pool.

The bridge deliberately has no OpenAI-compatible translation layer.  Request
bodies and successful upstream payloads are relayed byte-for-byte; only
credentials, account selection and bounded safety metadata are owned here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import hashlib
import json
import math
import threading
import time
from typing import Callable, Iterable, Iterator, Mapping

from curl_cffi import requests

from services.account_service import account_service
from services.config import config
from services.proxy_service import proxy_settings


CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_MODELS_URL = f"{CODEX_BASE_URL}/models?client_version=0.149.1"
CODEX_RESPONSES_URL = f"{CODEX_BASE_URL}/responses"
CODEX_COMPACT_URL = f"{CODEX_RESPONSES_URL}/compact"
CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"

MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_NONSTREAM_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_STREAM_BYTES = 128 * 1024 * 1024
MAX_STREAM_SECONDS = 20 * 60
OBSERVATION_MAX_AGE_SECONDS = 5 * 60
MAX_OPAQUE_BINDINGS_PER_ACCOUNT = 256
MAX_SELECTION_PROBES = 3

_FORWARDED_HEADERS = {
    "session-id",
    "thread-id",
    "x-client-request-id",
    "x-codex-window-id",
    "x-codex-turn-metadata",
    "x-codex-beta-features",
    "originator",
    "user-agent",
}
_RETURNED_HEADERS = {
    "content-type",
    "cache-control",
    "x-request-id",
    "openai-request-id",
    "x-openai-request-id",
}


class CodexServiceError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.rate_limit = ({"layer": "provider_capacity", "origin": "local_guard", "phase": "admission",
                            "retry_after_seconds": None, "cooldown_until": None}
                           if status_code == 429 and code == "codex_busy" else None)


@dataclass
class CodexHTTPResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes | None = None
    stream: Iterable[bytes] | None = None


class ManagedCodexStream:
    """Closeable lazy iterator whose abort callback also works before iteration."""

    def __init__(self, iterator_factory: Callable[[], Iterator[bytes]], abort: Callable[[], None]):
        self._iterator_factory = iterator_factory
        self._abort = abort
        self._iterator: Iterator[bytes] | None = None
        self._closed = False
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        with self._lock:
            if self._closed:
                raise StopIteration
            if self._iterator is None:
                self._iterator = self._iterator_factory()
            iterator = self._iterator
        return next(iterator)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            iterator = self._iterator
        self._abort()
        if iterator is not None:
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except (RuntimeError, ValueError):
                    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _jwt_account_id(token: str) -> str:
    try:
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(part).decode("utf-8"))
        auth = payload.get("https://api.openai.com/auth") or {}
        return str(auth.get("chatgpt_account_id") or "").strip()
    except Exception:
        return ""


def _safe_headers(headers: Mapping[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    total = 0
    for name, value in headers.items():
        key = str(name).lower()
        if key not in _FORWARDED_HEADERS:
            continue
        text = str(value)
        total += len(key) + len(text)
        if len(text) > 8192 or total > 16384 or "\r" in text or "\n" in text:
            raise CodexServiceError(400, "invalid_codex_headers", "Codex request headers are invalid")
        result[key] = text
    return result


def _response_headers(headers: Mapping[str, object]) -> dict[str, str]:
    return {
        str(name).lower(): str(value)
        for name, value in headers.items()
        if str(name).lower() in _RETURNED_HEADERS
    }


def _safe_model(item: object) -> dict | None:
    if not isinstance(item, dict):
        return None
    model_id = str(item.get("id") or item.get("slug") or item.get("model") or "").strip()
    if not model_id or len(model_id) > 160:
        return None
    label = str(item.get("label") or item.get("display_name") or item.get("name") or model_id).strip()[:200]
    raw_efforts = (
        item.get("reasoning_efforts")
        or item.get("supported_reasoning_efforts")
        or item.get("supported_reasoning_levels")
        or []
    )
    efforts: list[str] = []
    if isinstance(raw_efforts, list):
        for effort in raw_efforts:
            value = (effort.get("reasoning_effort") or effort.get("effort")) if isinstance(effort, dict) else effort
            text = str(value or "").strip()
            if text and len(text) <= 40 and text not in efforts:
                efforts.append(text)
    return {"id": model_id, "label": label or model_id, "reasoning_efforts": efforts}


def _project_models(payload: object) -> list[dict]:
    raw = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(raw, list) and isinstance(payload, list):
        raw = payload
    if not isinstance(raw, list):
        raise ValueError("models payload is invalid")
    models: list[dict] = []
    for item in raw:
        model = _safe_model(item)
        if model is None:
            continue
        models.append(model)
        if len(models) == 256:
            break
    return models


def _safe_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        if not math.isfinite(number) or number < 0 or number > 100:
            return None
        return number
    except (TypeError, ValueError):
        return None


def _usage_windows(item: dict) -> list[dict]:
    windows: list[dict] = []
    for key in ("primary", "secondary", "primary_window", "secondary_window"):
        window = item.get(key)
        if not isinstance(window, dict):
            continue
        used = _safe_number(window.get("used_percent"))
        if used is None:
            continue
        try:
            seconds = max(0, int(
                window.get("window_seconds")
                or window.get("limit_window_seconds")
                or int(window.get("window_minutes") or 0) * 60
            ))
        except (TypeError, ValueError):
            seconds = 0
        try:
            resets_at = int(window.get("resets_at") or window.get("reset_at") or 0) or None
        except (TypeError, ValueError):
            resets_at = None
        windows.append({"used_percent": used, "window_seconds": seconds, "resets_at": resets_at})
    return windows


def _project_limits(payload: object) -> tuple[list[dict], bool]:
    if not isinstance(payload, dict):
        raise ValueError("usage payload is invalid")
    candidates: list[dict] = []
    raw_many = payload.get("rate_limits")
    if isinstance(raw_many, list):
        candidates.extend(item for item in raw_many if isinstance(item, dict))
    raw_by_id = payload.get("rate_limits_by_limit_id")
    if isinstance(raw_by_id, dict):
        for limit_id, item in raw_by_id.items():
            if isinstance(item, dict):
                candidates.append({"limit_id": limit_id, **item})
    raw_one = payload.get("rate_limit")
    if isinstance(raw_one, dict):
        candidates.append({"limit_id": "codex", "limit_name": "Codex", **raw_one})
    additional = payload.get("additional_rate_limits")
    if isinstance(additional, list):
        for index, outer in enumerate(additional):
            if not isinstance(outer, dict) or not isinstance(outer.get("rate_limit"), dict):
                continue
            name = str(outer.get("limit_name") or f"additional-{index + 1}")
            candidates.append({
                "limit_id": str(outer.get("limit_id") or name),
                "limit_name": name,
                **outer["rate_limit"],
            })
    if not candidates and any(key in payload for key in ("primary", "secondary", "limit_id")):
        candidates.append(payload)
    if not candidates:
        raise ValueError("usage payload has no rate limits")

    result: list[dict] = []
    # A named model limit reaching zero does not exhaust the whole Codex route.
    # Only the top-level flag or the general ``codex`` bucket can do that.
    limited = bool(payload.get("limit_reached") or payload.get("rate_limit_reached_type"))
    seen: set[str] = set()
    for index, item in enumerate(candidates[:64]):
        limit_id = str(item.get("limit_id") or item.get("id") or f"limit-{index + 1}").strip()[:120]
        if limit_id in seen:
            continue
        seen.add(limit_id)
        windows = _usage_windows(item)
        has_explicit_state = isinstance(item.get("allowed"), bool) or isinstance(item.get("limit_reached"), bool)
        has_reached_type = bool(str(item.get("rate_limit_reached_type") or "").strip())
        if not windows and not has_explicit_state and not has_reached_type:
            continue
        if limit_id.lower() == "codex":
            limited = limited or bool(item.get("limit_reached") or item.get("rate_limit_reached_type"))
            limited = limited or item.get("allowed") is False
            limited = limited or any(window["used_percent"] >= 100 for window in windows)
        result.append({
            "id": limit_id,
            "label": str(item.get("limit_name") or item.get("label") or limit_id).strip()[:160],
            "windows": windows,
        })
    if not result:
        raise ValueError("usage payload has no valid rate limits")
    return result, limited


class _RuntimeCapacity:
    """A live limit never cancels or reassigns already admitted requests."""
    def __init__(self, limit):
        self._limit = limit
        self._active = 0
        self._lock = threading.Lock()

    def acquire(self, blocking=False):
        if blocking:
            raise ValueError("Codex admission is non-blocking")
        with self._lock:
            if self._active >= self._limit():
                return False
            self._active += 1
            return True

    def release(self):
        with self._lock:
            if self._active <= 0:
                raise ValueError("capacity released without admission")
            self._active -= 1

    def snapshot(self):
        with self._lock:
            return {"limit": self._limit(), "active": self._active}


class CodexService:
    def __init__(
        self,
        accounts=account_service,
        session_factory: Callable[..., object] = requests.Session,
        *,
        max_concurrency: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        chat_catalog: object | None = None,
    ) -> None:
        self.accounts = accounts
        self.session_factory = session_factory
        self._capacity = _RuntimeCapacity(
            (lambda: config.codex_max_concurrency) if max_concurrency is None
            else (lambda: max(1, max_concurrency))
        )
        self._lock = threading.RLock()
        self._inflight: set[str] = set()
        self._affinity_locks = tuple(threading.Lock() for _ in range(64))
        self._probe_index = 0
        self._clock = clock
        self._chat_catalog = chat_catalog

    def resource_snapshot(self) -> dict:
        # Never refresh credentials or probe the upstream merely for a dashboard.
        rows = [item for item in self.accounts.list_accounts() if not item.get("managed_disabled")
                and self.account_projection(item)["authorization_status"] == "saved"]
        observed = 0
        for item in rows:
            projection = self.account_projection(item)
            known_ids = {model["id"].strip().casefold() for model in projection["models"]}
            exhausted = {limit["id"].strip().casefold() for limit in projection["limits"]
                         if any(window["used_percent"] >= 100 for window in limit["windows"])}
            if (self._observation_fresh(projection) and not exhausted - known_ids - {"codex"}
                    and projection["state"] in {"observed", "limited", "auth_required"}):
                observed += 1
        complete = observed == len(rows)
        eligible = [item for item in rows
                    if self._eligible_account(item, allow_probe=False) is not None]
        with self._lock:
            available = sum(str(item.get("access_token") or "") not in self._inflight for item in eligible)
            capacity = self._capacity.snapshot()
        return {"state": "observed" if complete else "partial" if observed else "unknown",
                "total_accounts": len(rows), "observed_accounts": observed,
                "eligible_accounts": len(eligible), "inflight": capacity["active"],
                "slots_total": min(capacity["limit"], len(eligible)) if complete else None,
                "slots_free": min(max(0, capacity["limit"] - capacity["active"]), available) if complete else None}

    @classmethod
    def account_projection(cls, account: dict) -> dict:
        raw = account.get("codex_observation") if isinstance(account, dict) else None
        if not isinstance(raw, dict):
            raw = {}
        if isinstance(account, dict) and "codex_credentials" in account:
            try:
                cls._account_headers(account)
            except CodexServiceError:
                raw = {
                    **raw,
                    "state": "unknown",
                    "observed_at": None,
                    "failed_at": None,
                    "error_code": None,
                }
        rejection = account.get("codex_auth_rejection") if isinstance(account, dict) else None
        try:
            rejected_current = (
                isinstance(rejection, dict)
                and bool(account.get("access_token"))
                and rejection.get("credential_digest") == cls._credential_digest(account)
            )
        except CodexServiceError:
            rejected_current = False
        if rejected_current:
            # A concurrent catalog refresh must not hide a newer rejection.
            raw = {**raw, "state": "auth_required", "error_code": "codex_http_401",
                   "failed_at": rejection.get("failed_at")}
        state = str(raw.get("state") or "unknown")
        if state not in {"observed", "unknown", "read_failed", "auth_required", "limited"}:
            state = "unknown"
        models = [model for item in raw.get("models", []) if (model := _safe_model(item)) is not None]
        limits: list[dict] = []
        for item in raw.get("limits", []):
            if not isinstance(item, dict):
                continue
            windows = []
            for window in item.get("windows", []):
                if not isinstance(window, dict) or _safe_number(window.get("used_percent")) is None:
                    continue
                windows.append({
                    "used_percent": _safe_number(window.get("used_percent")),
                    "window_seconds": max(0, int(window.get("window_seconds") or 0)),
                    "resets_at": window.get("resets_at") if isinstance(window.get("resets_at"), int) else None,
                })
            limits.append({
                "id": str(item.get("id") or "")[:120],
                "label": str(item.get("label") or item.get("id") or "")[:160],
                "windows": windows,
            })
        # Credential presence is independent of quota observations and execution
        # success. A Chat-only primary token is not a saved Codex authorization.
        credentials = account.get("codex_credentials")
        explicit = credentials if isinstance(credentials, dict) else account if account.get("source_type") == "codex" else {}
        authorization_status = "saved" if (
            isinstance(explicit.get("access_token"), str) and explicit["access_token"].strip()
            and isinstance(explicit.get("account_id"), str) and explicit["account_id"].strip()
        ) else "missing"
        return {
            "authorization_status": authorization_status,
            "state": state,
            "observed_at": raw.get("observed_at") if isinstance(raw.get("observed_at"), str) else None,
            "failed_at": raw.get("failed_at") if isinstance(raw.get("failed_at"), str) else None,
            "models": models,
            "limits": limits,
            "error_code": str(raw.get("error_code") or "")[:80] or None,
        }

    def management_models(self) -> dict:
        by_id: dict[str, dict] = {}
        newest: str | None = None
        for account in self.accounts.list_accounts():
            projection = self.account_projection(account)
            active = not account.get("managed_disabled") and account.get("status") not in {"禁用", "异常"}
            fresh = self._observation_fresh(projection)
            exhausted_model_limits = {
                limit["id"].strip().casefold()
                for limit in projection["limits"]
                if any(window["used_percent"] >= 100 for window in limit["windows"])
            }
            # WHAM can use a display-cased model ID as limit_name, which the
            # usage projection preserves as its fallback ID. Only case/space
            # normalization is justified; do not guess aliases from labels.
            known_model_ids = {model["id"].strip().casefold() for model in projection["models"]}
            unmapped_exhausted_limit = bool(exhausted_model_limits - known_model_ids - {"codex"})
            for model in projection["models"]:
                current = by_id.setdefault(model["id"], {
                    "id": model["id"],
                    "label": model["label"],
                    "route": "codex",
                    "capabilities": ["responses"],
                    "state": "unknown",
                    "available_accounts": None,
                    "supported_accounts": 0,
                    "pending_accounts": 0,
                    "unavailable_accounts": 0,
                    "accounts": [],
                    "reasoning_efforts": [],
                })
                from services.owned_accounts import public_pool_account
                safe = public_pool_account(account)
                current["supported_accounts"] += 1
                if (active and fresh and projection["state"] == "observed"
                        and model["id"].strip().casefold() in exhausted_model_limits):
                    current["unavailable_accounts"] += 1
                    account_state, reason = "unavailable", "model_limited"
                elif active and fresh and projection["state"] == "observed" and unmapped_exhausted_limit:
                    # An exhausted bucket of unproven scope cannot establish
                    # that this model is either available or exhausted.
                    current["pending_accounts"] += 1
                    account_state, reason = "unknown", "unknown"
                elif active and fresh and projection["state"] == "observed":
                    current["available_accounts"] = int(current["available_accounts"] or 0) + 1
                    current["state"] = "available"
                    account_state, reason = "available", "observed"
                elif not active:
                    current["unavailable_accounts"] += 1
                    account_state, reason = "unavailable", "disabled" if account.get("managed_disabled") or account.get("status") == "禁用" else "account_unavailable"
                elif fresh and projection["state"] in {"limited", "auth_required"}:
                    current["unavailable_accounts"] += 1
                    account_state, reason = "unavailable", "limited" if projection["state"] == "limited" else "auth_required"
                else:
                    current["pending_accounts"] += 1
                    account_state = "unknown"
                    reason = (
                        projection["state"]
                        if projection["state"] in {"unknown", "read_failed"}
                        else "stale"
                    )
                current["accounts"].append({
                    "account_ref": safe["account_ref"],
                    "label": safe["label"],
                    "state": account_state,
                    "reason": reason,
                })
                for effort in model["reasoning_efforts"]:
                    if effort not in current["reasoning_efforts"]:
                        current["reasoning_efforts"].append(effort)
            observed = projection["observed_at"]
            if observed and (newest is None or observed > newest):
                newest = observed
        for item in by_id.values():
            if item["state"] != "available" and item["pending_accounts"] == 0 and item["unavailable_accounts"]:
                item["state"] = "unavailable"
                item["available_accounts"] = 0
        items = list(by_id.values())
        if self._chat_catalog is not None:
            items.extend(self._chat_catalog.management_models())
        return {"items": sorted(items, key=lambda item: (item["route"], item["id"])), "observed_at": newest}

    def _session(self, account: dict):
        kwargs = proxy_settings.build_session_kwargs(account=account, upstream=True, impersonate="chrome", verify=True)
        kwargs["verify"] = True
        return self.session_factory(**kwargs)

    @staticmethod
    def _account_headers(account: dict, forwarded: Mapping[str, object] | None = None) -> dict[str, str]:
        credentials = account.get("codex_credentials")
        if isinstance(credentials, dict):
            token = str(credentials.get("access_token") or "").strip()
            account_id = str(credentials.get("account_id") or "").strip()
        else:
            token = str(account.get("access_token") or "").strip()
            account_id = str(account.get("account_id") or "").strip() or _jwt_account_id(token)
        if not token:
            raise CodexServiceError(503, "codex_account_unavailable", "No Codex account is available")
        headers = _safe_headers(forwarded or {})
        headers["authorization"] = f"Bearer {token}"
        if account_id:
            headers["chatgpt-account-id"] = account_id
        headers["accept"] = "text/event-stream, application/json"
        return headers

    @classmethod
    def _credential_digest(cls, account: dict) -> str:
        # Bind a rejection to the actual authorization, without storing another
        # token copy or preventing recovery after token/account rotation.
        headers = cls._account_headers(account)
        value = [headers["authorization"], headers.get("chatgpt-account-id", "")]
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()

    def observe_import_authorization(self, credentials: dict[str, str]) -> dict:
        """Prove a supplied bearer token with a protected upstream usage read.

        JWT claims alone are only consistency hints. This read neither selects
        nor mutates a stored account, so a failed proof cannot affect a peer's
        authorization or observation.
        """
        account = {"codex_credentials": credentials, "source_type": "codex"}
        observed_at = _utc_now()
        session = None
        usage_status = None
        models_status = None
        try:
            session = self._session(account)
            response = session.get(
                CODEX_USAGE_URL,
                headers=self._account_headers(account, {"user-agent": "codex-cli/0.149.1"}),
                timeout=(5, 20),
                allow_redirects=False,
            )
            usage_status = response.status_code
            if response.status_code != 200 or len(response.content) > 4 * 1024 * 1024:
                raise ValueError("upstream usage read did not verify authorization")
            limits, limited = _project_limits(response.json())
            return {
                "state": "limited" if limited else "observed",
                "verified": True,
                "observed_at": observed_at,
                "failed_at": None,
                "models": [],
                "limits": limits,
                "error_code": None,
            }
        except Exception:
            # Usage can be temporarily unavailable while the protected Codex
            # catalog still proves the bearer token and selected workspace.
            try:
                if session is None:
                    raise ValueError("upstream session unavailable")
                response = session.get(
                    CODEX_MODELS_URL,
                    headers=self._account_headers(account, {"user-agent": "codex-cli/0.149.1"}),
                    timeout=(5, 20),
                    allow_redirects=False,
                )
                models_status = response.status_code
                if response.status_code != 200 or len(response.content) > 4 * 1024 * 1024:
                    raise ValueError("upstream catalog did not verify authorization")
                models = _project_models(response.json())
                verified = True
            except Exception:
                models = []
                verified = False
            auth_required = usage_status == 401 and models_status == 401
            return {
                "state": "auth_required" if auth_required else "read_failed",
                "verified": verified,
                "observed_at": None,
                "failed_at": observed_at,
                "models": models,
                "limits": [],
                "error_code": "codex_http_401" if auth_required else "usage_unverified",
            }
        finally:
            try:
                if session is not None:
                    session.close()
            except Exception:
                pass

    def refresh_account(self, access_token: str) -> dict:
        before = self.accounts.get_account(access_token)
        current_token = self._refresh_authorization(access_token, event="codex_observation")
        account = self.accounts.get_account(current_token)
        if account is None:
            raise KeyError("account not found")
        previous = self.account_projection(account)
        rejection = account.get("codex_auth_rejection")
        if not isinstance(rejection, dict):
            rejection = {}
        # Legacy 401s did not record the rejected credential. Do not invent a
        # fingerprint from today's token. Only a witnessed rotation or a token
        # issued after that failure establishes that this is new authorization.
        legacy_rejection = not rejection and previous["error_code"] == "codex_http_401"
        if legacy_rejection:
            changed = before and self._credential_digest(before) != self._credential_digest(account)
            failed_at = _parse_iso(previous["failed_at"])
            try:
                authorization_token = self._credential_fields(account)[0]
                part = authorization_token.split(".")[1]
                issued_at = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))).get("iat")
                changed = changed or (failed_at is not None and isinstance(issued_at, (int, float))
                                      and not isinstance(issued_at, bool) and issued_at > failed_at.timestamp())
            except (IndexError, ValueError, TypeError, AttributeError):
                pass
            legacy_rejection = not changed
        observed_at = _utc_now()
        models: list[dict] | None = None
        limits: list[dict] | None = None
        limited = False
        error_code: str | None = None
        auth_required = False
        session = self._session(account)
        headers = self._account_headers(account, {"user-agent": "codex-cli/0.149.1"})
        try:
            for kind, url in (("models", CODEX_MODELS_URL), ("usage", CODEX_USAGE_URL)):
                try:
                    response = session.get(url, headers=headers, timeout=(10, 30), allow_redirects=False)
                except Exception:
                    error_code = f"{kind}_transport_error"
                    break
                if response.status_code == 401:
                    auth_required = True
                    error_code = f"{kind}_auth_required"
                    break
                if response.status_code == 403:
                    error_code = f"{kind}_access_denied"
                    break
                if response.status_code != 200:
                    error_code = f"{kind}_http_{response.status_code}"
                    break
                try:
                    if len(response.content) > 4 * 1024 * 1024:
                        raise ValueError("response too large")
                    payload = response.json()
                    if kind == "models":
                        models = _project_models(payload)
                    else:
                        limits, limited = _project_limits(payload)
                except Exception:
                    error_code = f"{kind}_invalid_response"
                    break
        finally:
            try:
                session.close()
            except Exception:
                pass

        if error_code:
            observation = {
                **previous,
                "state": "auth_required" if auth_required else "read_failed",
                "failed_at": observed_at,
                "error_code": error_code,
            }
        else:
            observation = {
                "state": "limited" if limited else "observed",
                "observed_at": observed_at,
                "failed_at": None,
                "models": models or [],
                "limits": limits or [],
                "error_code": None,
            }
        if rejection.get("credential_digest") == self._credential_digest(account):
            # Catalog/usage reads do not prove permission to execute Responses.
            observation.update(state="auth_required", error_code="codex_http_401",
                               failed_at=rejection.get("failed_at"))
        elif legacy_rejection:
            observation.update(state="auth_required", error_code="codex_http_401",
                               failed_at=previous["failed_at"])
        self.accounts.update_account(current_token, {"codex_observation": observation}, quiet=True,
                                     expected_codex_credentials=self._credential_fields(account))
        return self.account_projection(self.accounts.get_account(current_token) or {"codex_observation": observation})

    def _observation_fresh(self, projection: dict) -> bool:
        observed_at = _parse_iso(projection.get("observed_at"))
        if observed_at is None:
            return False
        return (datetime.now(timezone.utc) - observed_at).total_seconds() <= OBSERVATION_MAX_AGE_SECONDS

    def _record_model_catalog(self, token: str, request_account: dict, models: list[dict]) -> None:
        account = self.accounts.get_account(token)
        if account is None:
            return
        observation = account.get("codex_observation")
        observation = dict(observation) if isinstance(observation, dict) else {}
        observation["models"] = models
        self.accounts.update_account(
            token,
            {"codex_observation": observation},
            quiet=True,
            expected_codex_credentials=self._credential_fields(request_account),
        )

    @staticmethod
    def _affinity_key(identity: dict, forwarded: Mapping[str, object], payload: dict | None = None) -> str:
        lowered = {str(key).lower(): str(value) for key, value in forwarded.items()}
        session = lowered.get("session-id") or lowered.get("thread-id")
        if not session and payload:
            session = str(payload.get("prompt_cache_key") or "")
        material = f"{identity.get('id', '')}\0{session or 'default'}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _identity_digest(identity: dict) -> str:
        return hashlib.sha256(str(identity.get("id") or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _metadata(account: dict, key: str) -> dict:
        value = account.get(key)
        return dict(value) if isinstance(value, dict) else {}

    def _persist_mapping(self, token: str, key: str, mapping_key: str, value: object, *, required: bool = False) -> bool:
        from contextlib import nullcontext
        with getattr(self.accounts, "admission_transaction", nullcontext)(), self._lock:
            account = self.accounts.get_account(token)
            if not account:
                if required:
                    raise CodexServiceError(503, "codex_account_unavailable", "The bound Codex account is unavailable")
                return False
            mapping = self._metadata(account, key)
            if mapping_key not in mapping and len(mapping) >= MAX_OPAQUE_BINDINGS_PER_ACCOUNT:
                if required:
                    raise CodexServiceError(429, "codex_binding_capacity", "The Codex binding ledger is full")
                return False
            mapping[mapping_key] = value
            self.accounts.update_account(token, {key: mapping}, quiet=True)
            return True

    def _response_owner(self, response_id: str, identity: dict) -> str | None:
        digest = hashlib.sha256(response_id.encode("utf-8")).hexdigest()
        owner = self._identity_digest(identity)
        seen = False
        for account in self.accounts.list_accounts():
            value = self._metadata(account, "codex_response_ids").get(digest)
            if value is None:
                continue
            seen = True
            if isinstance(value, dict) and value.get("owner") == owner:
                return str(account.get("access_token") or "") or None
        if seen:
            raise CodexServiceError(409, "codex_response_owner_mismatch", "The previous response belongs to another caller")
        return None

    def _set_affinity_state(self, token: str, affinity: str, state: str) -> None:
        account = self.accounts.get_account(token)
        if not account:
            return
        bindings = self._metadata(account, "codex_affinities")
        current = bindings.get(affinity)
        if not isinstance(current, dict):
            current = {"bound_at": _utc_now()}
        self._persist_mapping(token, "codex_affinities", affinity, {
            **current, "state": state, "updated_at": _utc_now(),
        }, required=True)

    def _eligible_account(
        self,
        account: dict | None,
        requested_model: str = "",
        *,
        allow_probe: bool = True,
    ) -> dict | None:
        if not account:
            return None
        token = str(account.get("access_token") or "")
        if not token or account.get("managed_disabled") or account.get("status") in {"禁用", "异常"}:
            return None
        if float((account.get("codex_rate_limit") or {}).get("cooldown_until") or 0) > time.time():
            return None
        try:
            self._account_headers(account)
        except CodexServiceError:
            return None
        projection = self.account_projection(account)
        if projection["state"] == "unknown" or not self._observation_fresh(projection):
            if not allow_probe:
                return None
            projection = self.refresh_account(token)
            account = self.accounts.get_account(token) or account
        model_ids = {item["id"] for item in projection["models"]}
        if projection["state"] != "observed" or (requested_model and requested_model not in model_ids):
            return None
        exhausted = {limit["id"].strip().casefold() for limit in projection["limits"]
                     if any(window["used_percent"] >= 100 for window in limit["windows"])}
        known_ids = {model.casefold() for model in model_ids}
        if "codex" in exhausted or exhausted - known_ids - {"codex"}:
            return None
        if requested_model and requested_model.casefold() in exhausted:
            return None
        if not requested_model and not known_ids - exhausted:
            return None
        return account

    def _eligible_accounts(self, requested_model: str = "") -> list[dict]:
        accounts = self.accounts.list_accounts()
        fresh = [
            eligible
            for account in accounts
            if (eligible := self._eligible_account(
                account, requested_model, allow_probe=False
            )) is not None
        ]
        if fresh:
            return fresh

        probe_candidates: list[dict] = []
        for account in accounts:
            token = str(account.get("access_token") or "")
            if not token or account.get("managed_disabled") or account.get("status") in {"禁用", "异常"}:
                continue
            projection = self.account_projection(account)
            if projection["state"] == "unknown" or not self._observation_fresh(projection):
                probe_candidates.append(account)
        if not probe_candidates:
            return []

        with self._lock:
            start = self._probe_index % len(accounts)
            selected: list[dict] = []
            scanned = 0
            candidate_tokens = {str(item.get("access_token") or "") for item in probe_candidates}
            while scanned < len(accounts) and len(selected) < MAX_SELECTION_PROBES:
                account = accounts[(start + scanned) % len(accounts)]
                if str(account.get("access_token") or "") in candidate_tokens:
                    selected.append(account)
                scanned += 1
            self._probe_index = (start + scanned) % len(accounts)
        for account in selected:
            eligible = self._eligible_account(account, requested_model, allow_probe=True)
            if eligible is not None:
                return [eligible]
        return []

    def _select_account(
        self,
        identity: dict,
        forwarded: Mapping[str, object],
        payload: dict | None = None,
        *,
        bind_session: bool = True,
    ) -> tuple[dict, str, str]:
        affinity = self._affinity_key(identity, forwarded, payload)
        stripe = self._affinity_locks[int(affinity[:8], 16) % len(self._affinity_locks)]
        with stripe:
            return self._select_account_locked(
                identity,
                forwarded,
                payload,
                affinity=affinity,
                bind_session=bind_session,
            )

    def _select_account_locked(
        self,
        identity: dict,
        forwarded: Mapping[str, object],
        payload: dict | None,
        *,
        affinity: str,
        bind_session: bool,
    ) -> tuple[dict, str, str]:
        previous_id = str((payload or {}).get("previous_response_id") or "").strip()
        forced_token = self._response_owner(previous_id, identity) if previous_id else None
        if previous_id and forced_token is None:
            raise CodexServiceError(409, "codex_response_owner_unknown", "The previous response cannot be resumed safely")
        requested_model = str((payload or {}).get("model") or "").strip()

        from services.request_context import current_request, AdmissionLost
        context = current_request.get()
        if context is not None:
            selected = context.selected_account()
            if selected is None:
                raise AdmissionLost("original Codex account is unavailable")
            admitted_token = str(selected["access_token"])
            if forced_token and forced_token != admitted_token:
                raise CodexServiceError(409, "codex_binding_conflict", "The original account differs from the previous response")
            forced_token = admitted_token

        # Resolve the immutable session binding against the full pool before
        # considering availability.  A disabled, limited or stale bound
        # account is an explicit failure, never permission to change owners.
        all_accounts = self.accounts.list_accounts()
        bound = [
            account for account in all_accounts
            if bind_session and affinity in self._metadata(account, "codex_affinities")
        ]
        if len(bound) > 1:
            raise CodexServiceError(409, "codex_binding_conflict", "The Codex session binding is ambiguous")
        bound_account = bound[0] if bound else None
        if bound_account:
            binding = self._metadata(bound_account, "codex_affinities").get(affinity)
            if not isinstance(binding, dict) or binding.get("state") in {"pending", "unknown"}:
                raise CodexServiceError(409, "codex_session_outcome_unknown", "The previous Codex request outcome is unknown")
            bound_token = str(bound_account.get("access_token") or "")
            if forced_token and forced_token != bound_token:
                raise CodexServiceError(409, "codex_binding_conflict", "The response and session bindings do not match")
            forced_token = bound_token

        if forced_token:
            selected = self._eligible_account(self.accounts.get_account(forced_token), requested_model)
            if selected is None:
                raise CodexServiceError(503, "codex_bound_account_unavailable", "The bound Codex account is unavailable")
            accounts = [selected]
        else:
            accounts = self._eligible_accounts(requested_model)
        with self._lock:
            account = next(
                (item for item in accounts if str(item.get("access_token") or "") not in self._inflight),
                None,
            )
            if account is None:
                raise CodexServiceError(429, "codex_busy", "All eligible Codex accounts are busy")
            token = str(account["access_token"])
        current_token = self._refresh_authorization(token, event="codex_request")
        account = self.accounts.get_account(current_token)
        if account is None or account.get("managed_disabled") or account.get("status") in {"禁用", "异常"}:
            raise CodexServiceError(503, "codex_account_unavailable", "No Codex account is available")
        token = current_token
        with self._lock:
            if token in self._inflight:
                raise CodexServiceError(429, "codex_busy", "The selected Codex account is busy")
            self._inflight.add(token)
        if bind_session and not bound_account:
            try:
                self._persist_mapping(token, "codex_affinities", affinity, {
                    "state": "bound", "bound_at": _utc_now(), "updated_at": _utc_now(),
                }, required=True)
            except Exception:
                with self._lock:
                    self._inflight.discard(token)
                raise
        return account, token, affinity if bind_session else ""

    def _release_account(self, token: str, affinity: str, *, known_terminal: bool) -> None:
        from services.request_context import current_request
        context = current_request.get()
        if context is not None:
            context.terminal(known_terminal)
        with self._lock:
            self._inflight.discard(token)
        if not affinity:
            return
        try:
            self._set_affinity_state(token, affinity, "bound" if known_terminal else "unknown")
        except Exception:
            # A persisted ``pending`` state is the safe fallback if terminal
            # state storage fails.  Capacity must still be released.
            pass

    def _acquire_capacity(self) -> None:
        if not self._capacity.acquire(blocking=False):
            raise CodexServiceError(429, "codex_busy", "Codex request capacity is busy")

    @staticmethod
    def _credential_fields(account: dict) -> tuple[str, str]:
        credentials = account.get("codex_credentials")
        if isinstance(credentials, dict):
            return str(credentials.get("access_token") or ""), str(credentials.get("account_id") or "")
        return str(account.get("access_token") or ""), str(account.get("account_id") or "")

    def _refresh_authorization(self, token: str, *, event: str) -> str:
        refresh = getattr(self.accounts, "refresh_codex_access_token", None)
        if callable(refresh):
            return refresh(token, event=event) or token
        return self.accounts.refresh_access_token(token, event=event) or token

    def _mark_observation_state(self, token: str, state: str, error_code: str, request_account: dict, request_digest: str) -> None:
        account = self.accounts.get_account(token)
        if not account:
            return
        projection = self.account_projection(account)
        projection.update(state=state, failed_at=_utc_now(), error_code=error_code)
        updates = {"codex_observation": projection}
        if error_code == "codex_http_401":
            updates["codex_auth_rejection"] = {
                "credential_digest": request_digest,
                "failed_at": projection["failed_at"],
            }
        self.accounts.update_account(token, updates, quiet=True,
                                     expected_codex_credentials=self._credential_fields(request_account))

    def _safe_upstream_error(
        self,
        status: int,
        token: str,
        affinity: str,
        *,
        post: bool,
        request_account: dict,
        request_digest: str,
    ) -> CodexServiceError:
        try:
            if status == 429:
                self._mark_observation_state(token, "limited", "codex_http_429", request_account, request_digest)
            elif status == 401:
                self._mark_observation_state(token, "auth_required", f"codex_http_{status}", request_account, request_digest)
            elif status == 403:
                # A permission/policy rejection does not establish that the
                # credential expired. Keep the account unavailable without
                # recommending token rotation or replaying the request.
                self._mark_observation_state(token, "read_failed", "codex_http_403", request_account, request_digest)
        except Exception:
            pass
        # A timeout response does not prove that the non-idempotent POST was
        # rejected before execution. Keep its session quarantined like a 5xx.
        unknown_outcome = post and (status == 408 or status >= 500)
        known_terminal = not unknown_outcome
        self._release_account(token, affinity, known_terminal=known_terminal)
        self._capacity.release()
        if unknown_outcome:
            return CodexServiceError(502, "codex_upstream_outcome_unknown", "The Codex request outcome is unknown")
        if status == 429:
            return CodexServiceError(429, "codex_limited", "The selected Codex account is temporarily limited")
        if status == 401:
            return CodexServiceError(503, "codex_auth_required", "The upstream rejected the selected Codex account credential")
        if status == 403:
            return CodexServiceError(503, "codex_access_denied", "The upstream denied access for the selected Codex account")
        if 400 <= status < 500:
            return CodexServiceError(400, "codex_request_rejected", "The Codex request was rejected")
        return CodexServiceError(502, "codex_upstream_error", "The Codex service is temporarily unavailable")

    def list_native_models(self, identity: dict, forwarded_headers: Mapping[str, object]) -> CodexHTTPResponse:
        self._acquire_capacity()
        token = ""
        affinity = ""
        session = None
        try:
            account, token, affinity = self._select_account(
                identity, forwarded_headers, bind_session=False
            )
            request_digest = self._credential_digest(account)
            session = self._session(account)
            response = session.get(
                CODEX_MODELS_URL,
                headers=self._account_headers(account, forwarded_headers),
                timeout=(10, 30),
                allow_redirects=False,
            )
            if response.status_code != 200:
                raise self._safe_upstream_error(response.status_code, token, affinity, post=False,
                                                request_account=account, request_digest=request_digest)
            body = bytes(response.content)
            if len(body) > 4 * 1024 * 1024:
                raise CodexServiceError(502, "codex_response_too_large", "The Codex model catalog is too large")
            try:
                models = _project_models(json.loads(body))
            except Exception:
                models = None
            if models is not None:
                self._record_model_catalog(token, account, models)
            self._release_account(token, affinity, known_terminal=True)
            self._capacity.release()
            return CodexHTTPResponse(200, _response_headers(response.headers), body=body)
        except CodexServiceError:
            if token and token in self._inflight:
                self._release_account(token, affinity, known_terminal=True)
                self._capacity.release()
            elif not token:
                self._capacity.release()
            raise
        except Exception as exc:
            if token:
                self._release_account(token, affinity, known_terminal=True)
            self._capacity.release()
            raise CodexServiceError(502, "codex_read_failed", "The Codex model catalog could not be read") from exc
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass

    @staticmethod
    def _terminal_from_line(line: bytes) -> tuple[bool, str | None]:
        if not line.startswith(b"data:"):
            return False, None
        raw = line[5:].strip()
        if not raw or raw == b"[DONE]":
            return raw == b"[DONE]", None
        try:
            event = json.loads(raw)
        except Exception:
            return False, None
        event_type = str(event.get("type") or "") if isinstance(event, dict) else ""
        if event_type not in {"response.completed", "response.failed"}:
            return False, None
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        response_id = str(response.get("id") or event.get("response_id") or "").strip() or None
        return True, response_id

    def _stream_bytes(
        self,
        response,
        token: str,
        owner: str,
        finish: Callable[[bool], None],
        terminal_seen: threading.Event,
    ) -> Iterator[bytes]:
        started = self._clock()
        total = 0
        terminal = False
        buffer = b""
        try:
            for chunk in response.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                chunk = bytes(chunk)
                total += len(chunk)
                if total > MAX_STREAM_BYTES or self._clock() - started > MAX_STREAM_SECONDS:
                    raise CodexServiceError(502, "codex_stream_limit", "The Codex response exceeded its safe stream limit")
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    self._observe_stream_limit(line, token, response)
                    is_terminal, response_id = self._terminal_from_line(line.rstrip(b"\r"))
                    terminal = terminal or is_terminal
                    if is_terminal:
                        terminal_seen.set()
                    if response_id:
                        digest = hashlib.sha256(response_id.encode("utf-8")).hexdigest()
                        self._persist_mapping(token, "codex_response_ids", digest, {
                            "owner": owner, "bound_at": _utc_now(),
                        })
                yield chunk
            if buffer:
                self._observe_stream_limit(buffer, token, response)
                is_terminal, response_id = self._terminal_from_line(buffer.rstrip(b"\r"))
                terminal = terminal or is_terminal
                if is_terminal:
                    terminal_seen.set()
                if response_id:
                    digest = hashlib.sha256(response_id.encode("utf-8")).hexdigest()
                    self._persist_mapping(token, "codex_response_ids", digest, {
                        "owner": owner, "bound_at": _utc_now(),
                    })
        finally:
            finish(terminal)

    def _observe_stream_limit(self, line, token, response):
        try:
            if not line.startswith(b"data:"):
                return
            event = json.loads(line[5:].strip())
            from services.request_context import current_request
            context = current_request.get()
            if context is not None and isinstance(event, dict):
                original_response = event.get("response")
                response_id = (original_response.get("id") if isinstance(original_response, dict) else None) or event.get("response_id")
                if isinstance(response_id, str) and response_id.strip():
                    receipt = context.receipt()
                    if receipt.get("upstream_response_id") and receipt["upstream_response_id"] != response_id:
                        raise CodexServiceError(502, "codex_response_identity_mismatch", "Original response identity changed")
                    if not receipt.get("upstream_response_id"):
                        context.admission.update_claim(context, upstream_response_id=response_id,
                            upstream_request_id=_response_headers(response.headers).get("x-request-id"))
            if event.get("type") == "response.failed" and context is not None:
                context.admission.update_claim(context, _upstream_failed=True)
            error = event.get("error") or ((event.get("response") or {}).get("error") if event.get("type") == "response.failed" else None)
            if not isinstance(error, dict) or error.get("code", error.get("type")) not in {"rate_limit_exceeded", "rate_limit_error", "too_many_requests"}:
                return
            account = self.accounts.get_account(token)
            if account:
                self._mark_observation_state(token, "limited", "codex_sse_rate_limit", account, self._credential_digest(account))
            self._record_limit(response, "sse_rate_limit", error.get("retry_after_seconds", error.get("retry_after")), token=token)
        except (ValueError, TypeError, AttributeError):
            return

    def _record_limit(self, response, origin, retry=None, *, token=""):
        from services.request_context import current_request
        from services.account_request_pacing import retry_after_seconds
        context = current_request.get()
        headers = response.headers
        value = str(retry if retry is not None else headers.get("retry-after", headers.get("Retry-After", "")))
        request_id = headers.get("x-request-id") or headers.get("openai-request-id")
        safe_id = request_id if isinstance(request_id, str) and len(request_id) <= 160 and request_id.isascii() and not any(c.isspace() for c in request_id) else None
        evidence = {"layer": "upstream_codex", "origin": origin, "phase": "responses_stream" if origin == "sse_rate_limit" else "responses",
                    "upstream_request_id": safe_id, "retry_after_seconds": retry_after_seconds(value),
                    "retry_after_raw": value[:160] if value.isascii() else None,
                    "observed_at": time.time(), "cooldown_until": time.time() + max(OBSERVATION_MAX_AGE_SECONDS, retry_after_seconds(value))}
        if token:
            self.accounts.update_account(token, {"codex_rate_limit": evidence}, quiet=True)
        if context is not None:
            context.record_limit({**evidence, **context.log_fields()})
        from utils.log import logger
        logger.warning({"event": "codex_rate_limited", **evidence})
        return evidence

    def _managed_stream(self, response, token: str, affinity: str, owner: str, session) -> ManagedCodexStream:
        cleanup_lock = threading.Lock()
        cleaned = False
        terminal_seen = threading.Event()

        def finish(known_terminal: bool) -> None:
            nonlocal cleaned
            with cleanup_lock:
                if cleaned:
                    return
                cleaned = True
            try:
                response.close()
            except Exception:
                pass
            try:
                session.close()
            except Exception:
                pass
            # close() can run while the generator is suspended after yielding
            # its terminal chunk. Preserve the evidence already parsed, even
            # when abort cleanup wins over the generator's finally block.
            self._release_account(token, affinity, known_terminal=known_terminal or terminal_seen.is_set())
            self._capacity.release()

        return ManagedCodexStream(
            lambda: self._stream_bytes(response, token, owner, finish, terminal_seen),
            lambda: finish(False),
        )

    @staticmethod
    def _read_nonstream(response) -> bytes:
        body = bytearray()
        for chunk in response.iter_content(chunk_size=16384):
            if not chunk:
                continue
            body.extend(chunk)
            if len(body) > MAX_NONSTREAM_RESPONSE_BYTES:
                raise CodexServiceError(502, "codex_response_too_large", "The Codex response is too large")
        return bytes(body)

    def submit(
        self,
        identity: dict,
        payload: dict,
        forwarded_headers: Mapping[str, object],
        *,
        compact: bool = False,
    ) -> CodexHTTPResponse:
        if not isinstance(payload, dict):
            raise CodexServiceError(400, "invalid_codex_request", "A JSON object is required")
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(raw) > MAX_REQUEST_BYTES:
            raise CodexServiceError(413, "codex_request_too_large", "The Codex request is too large")
        self._acquire_capacity()
        token = ""
        affinity = ""
        session = None
        attempted = False
        try:
            account, token, affinity = self._select_account(identity, forwarded_headers, payload)
            headers = self._account_headers(account, forwarded_headers)
            request_digest = self._credential_digest(account)
            headers["content-type"] = "application/json"
            session = self._session(account)
            # From this point a transport error is an unknown POST outcome.  It
            # must never be retried or moved to another account.
            self._set_affinity_state(token, affinity, "pending")
            from services.request_context import current_request
            context = current_request.get()
            if context is not None:
                context.before_send()
            attempted = True
            response = session.post(
                CODEX_COMPACT_URL if compact else CODEX_RESPONSES_URL,
                # curl-cffi 0.15.0 in uv.lock accepts raw bytes via data=;
                # content= is a newer API and fails before transport on 0.15.
                data=raw,
                headers=headers,
                timeout=(10, MAX_STREAM_SECONDS),
                allow_redirects=False,
                stream=True,
            )
            if response.status_code < 200 or response.status_code >= 300:
                evidence = self._record_limit(response, "http_429", token=token) if response.status_code == 429 else None
                try:
                    response.close()
                except Exception:
                    pass
                error = self._safe_upstream_error(response.status_code, token, affinity, post=True,
                                                  request_account=account, request_digest=request_digest)
                if evidence:
                    error.rate_limit = evidence
                raise error
            headers_out = _response_headers(response.headers)
            content_type = headers_out.get("content-type", "").lower()
            if "text/event-stream" in content_type:
                return CodexHTTPResponse(
                    response.status_code,
                    headers_out,
                    stream=self._managed_stream(
                        response, token, affinity, self._identity_digest(identity), session
                    ),
                )

            body = self._read_nonstream(response)
            try:
                parsed = json.loads(body)
                response_id = str(parsed.get("id") or "") if isinstance(parsed, dict) else ""
                if response_id:
                    digest = hashlib.sha256(response_id.encode("utf-8")).hexdigest()
                    self._persist_mapping(token, "codex_response_ids", digest, {
                        "owner": self._identity_digest(identity), "bound_at": _utc_now(),
                    })
            except Exception:
                pass
            self._release_account(token, affinity, known_terminal=True)
            self._capacity.release()
            try:
                session.close()
            except Exception:
                pass
            return CodexHTTPResponse(response.status_code, headers_out, body=body)
        except CodexServiceError:
            if token and token in self._inflight:
                self._release_account(token, affinity, known_terminal=not attempted)
                self._capacity.release()
            elif not token:
                self._capacity.release()
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
            raise
        except Exception as exc:
            if token:
                self._release_account(token, affinity, known_terminal=not attempted)
            self._capacity.release()
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
            code = "codex_upstream_outcome_unknown" if attempted else "codex_transport_failed"
            raise CodexServiceError(502, code, "The Codex request outcome is unknown" if attempted else "The Codex request could not be sent") from exc


from services.model_service import model_catalog_service

codex_service = CodexService(chat_catalog=model_catalog_service)
