from __future__ import annotations

import math
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import RLock
from typing import Any

from services.account_service import AccountService, account_service
from services.openai_backend_api import OpenAIBackendAPI
from utils.helper import UpstreamHTTPError
from utils.log import logger


@dataclass(frozen=True)
class ModelRoute:
    account_types: frozenset[str]
    allow_anonymous: bool = False
    # `None` preserves the narrow type-only compatibility seam used by older
    # in-process callers.  Catalog-produced routes always contain an explicit
    # set, including an empty one, and must therefore select only observed
    # physical accounts.
    account_identities: frozenset[str] | None = None


class ModelUnavailableError(RuntimeError):
    pass


class ModelCatalogService:
    """Caches independently observed Chat model catalogs for active accounts."""

    # A cold read probes one endpoint per physical account.  Keep that burst
    # bounded while allowing the ordinary pool size to be observed in parallel
    # instead of serial account-timeouts.
    MAX_DISCOVERY_WORKERS = 8
    DISCOVERY_BUDGET_SECONDS = 10.0
    DISCOVERY_RETRY_SECONDS = 1.0

    def __init__(
        self,
        accounts: AccountService,
        *,
        backend_factory: Callable[..., Any] = OpenAIBackendAPI,
        cache_ttl_seconds: float = 300,
        clock: Callable[[], float] = time.monotonic,
        observed_clock: Callable[[], float] = time.time,
    ) -> None:
        self._accounts = accounts
        self._backend_factory = backend_factory
        self._cache_ttl_seconds = max(1.0, float(cache_ttl_seconds))
        self._clock = clock
        self._observed_clock = observed_clock
        self._lock = RLock()
        self._expires_at = 0.0
        self._account_signature: tuple[tuple[str, str, str, str], ...] = ()
        self._anonymous_models: dict[str, dict[str, Any]] = {}
        self._anonymous_observation: dict[str, Any] = {}
        # A successful catalog read remains capability evidence after a later
        # read failure or rate limit.  It is deliberately separate from the
        # executable map below, which is rebuilt only from this refresh.
        self._model_observations: dict[str, dict[str, Any]] = {}
        self._catalog_unknown = False
        self._models_by_account: dict[str, dict[str, dict[str, Any]]] = {}
        self._account_types: dict[str, str] = {}
        self._executor = ThreadPoolExecutor(max_workers=self.MAX_DISCOVERY_WORKERS)
        self._inflight_accounts: dict[str, tuple[Future, str, str]] = {}
        self._anonymous_inflight: Future | None = None

    @staticmethod
    def _model_map(result: object) -> dict[str, dict[str, Any]]:
        if not isinstance(result, dict) or not isinstance(result.get("data"), list):
            raise TypeError("upstream model response has no data list")
        models: dict[str, dict[str, Any]] = {}
        for item in result["data"]:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if model_id and model_id not in models:
                models[model_id] = dict(item)
        return models

    @staticmethod
    def _credential_fingerprint(access_token: str) -> str:
        # Private cache coherence only; never project a credential or its raw
        # identifier into a model row.
        import hashlib
        return hashlib.sha256(access_token.encode()).hexdigest()

    def _active_accounts(self) -> dict[str, tuple[str, str, str, str]]:
        """Return exactly one stable physical account and its dispatch state."""
        candidates: dict[str, list[tuple[str, str, str, str]]] = {}
        for account in self._accounts.list_accounts():
            if not isinstance(account, dict):
                continue
            if str(account.get("source_type") or "").strip().lower() not in {"web", "oauth_login", "password"}:
                continue
            access_token = str(account.get("access_token") or "").strip()
            account_type = self._accounts._normalize_account_type(account.get("type"))
            identity = self._accounts._stable_account_identity(account)
            if access_token and account_type and identity:
                status = str(account.get("status") or "").strip()
                state = (
                    "disabled" if account.get("managed_disabled") or status == "禁用"
                    else "invalid_credential" if status == "异常"
                    else "limited" if status == "限流"
                    else "active"
                )
                candidates.setdefault(identity, []).append((
                    access_token, account_type, state, self._credential_fingerprint(access_token),
                ))
        active: dict[str, tuple[str, str, str, str]] = {}
        for identity, rows in candidates.items():
            if len(rows) == 1:
                active[identity] = rows[0]
            else:
                logger.warning({"event": "model_catalog_account_identity_ambiguous"})
        return active

    @staticmethod
    def _signature(accounts: dict[str, tuple[str, str, str, str]]) -> tuple[tuple[str, str, str, str], ...]:
        return tuple(sorted(
            (identity, account_type, state, fingerprint)
            for identity, (_token, account_type, state, fingerprint) in accounts.items()
        ))

    def _fetch_models(self, access_token: str = "") -> tuple[dict[str, dict[str, Any]], float, float]:
        backend = self._backend_factory(access_token=access_token)
        try:
            models = self._model_map(backend.list_models())
            return models, self._clock(), self._observed_clock()
        finally:
            backend.close()

    def _fetch_account_models(self, access_token: str) -> tuple[dict[str, dict[str, Any]], float, float]:
        resolved_token = self._accounts.refresh_access_token(access_token, event="list_models") or access_token
        return self._fetch_models(resolved_token)

    @staticmethod
    def _observation_reason(state: str) -> str:
        return {
            "observed": "model_catalog_observed",
            "stale": "stale",
            "read_failed": "read_failed",
            "limited": "limited",
            "disabled": "disabled",
            "invalid_credential": "auth_required",
        }.get(state, "read_failed")

    def _observation_state_after_failure(self, observation: dict[str, Any], now: float) -> str:
        observed_at = observation.get("observed_at")
        if type(observed_at) in {int, float} and now - float(observed_at) <= self._cache_ttl_seconds:
            return "read_failed"
        return "stale"

    @staticmethod
    def _at(value: object) -> float:
        return float(value) if type(value) in {int, float} else 0.0

    def _failure_retry_at(self, exc: BaseException | None, now: float) -> float:
        """Keep an anonymous upstream 429's requested retry window intact."""
        retry_after = getattr(exc, "retry_after", None)
        if (
            isinstance(exc, UpstreamHTTPError)
            and exc.status_code == 429
            and type(retry_after) in {int, float}
            and math.isfinite(float(retry_after))
            and retry_after >= 0
        ):
            return now + max(self.DISCOVERY_RETRY_SECONDS, float(retry_after))
        return now + self.DISCOVERY_RETRY_SECONDS

    def _reusable_observation(
        self,
        observation: dict[str, Any] | None,
        account_type: str,
        state: str,
        credential_fingerprint: str,
        now: float,
    ) -> bool:
        return bool(
            state == "active"
            and isinstance(observation, dict)
            and isinstance(observation.get("models"), dict)
            and observation.get("state") == "unknown"
            and observation.get("observation_state") == "observed"
            and observation.get("account_type") == account_type
            and observation.get("credential_fingerprint") == credential_fingerprint
            and observation.get("observed_account_state") == "active"
            and observation.get("last_seen_account_state") == "active"
            and self._at(observation.get("refresh_after")) > now
        )

    def _account_due(
        self,
        identity: str,
        account_type: str,
        state: str,
        credential_fingerprint: str,
        observation: dict[str, Any] | None,
        now: float,
    ) -> bool:
        if state != "active":
            return False
        probe = self._inflight_accounts.get(identity)
        if probe is not None:
            return True
        if self._reusable_observation(observation, account_type, state, credential_fingerprint, now):
            return False
        return self._at((observation or {}).get("retry_after")) <= now

    def _account_next_refresh(
        self,
        identity: str,
        account_type: str,
        state: str,
        credential_fingerprint: str,
        observation: dict[str, Any] | None,
        now: float,
    ) -> float | None:
        if state != "active":
            return None
        probe = self._inflight_accounts.get(identity)
        if probe is not None:
            return now if probe[0].done() else now + self.DISCOVERY_RETRY_SECONDS
        if self._reusable_observation(observation, account_type, state, credential_fingerprint, now):
            return self._at(observation.get("refresh_after"))
        retry_after = self._at((observation or {}).get("retry_after"))
        return retry_after if retry_after > now else now

    def _anonymous_due(self, now: float) -> bool:
        observation = self._anonymous_observation
        if self._anonymous_inflight is not None:
            return True
        return self._at(observation.get("refresh_after")) <= now and self._at(observation.get("retry_after")) <= now

    def _anonymous_next_refresh(self, now: float) -> float:
        if self._anonymous_inflight is not None:
            return now if self._anonymous_inflight.done() else now + self.DISCOVERY_RETRY_SECONDS
        observation = self._anonymous_observation
        refresh_after = self._at(observation.get("refresh_after"))
        retry_after = self._at(observation.get("retry_after"))
        if refresh_after > now:
            return refresh_after
        if retry_after > now:
            return retry_after
        return now

    def _set_failed_observation(
        self,
        observation: dict[str, Any] | None,
        *,
        account_type: str,
        credential_fingerprint: str,
        now: float,
        observed_now: float,
    ) -> dict[str, Any]:
        if observation is None:
            observation = {"models": {}, "credential_fingerprint": credential_fingerprint,
                           "observed_account_state": "active"}
        observation_state = self._observation_state_after_failure(observation, observed_now)
        observation.update(
            account_type=account_type,
            state="unavailable",
            reason="read_failed" if observation_state == "read_failed" else "stale",
            observation_state=observation_state,
            last_seen_account_state="active",
            last_read_failed_at=observed_now,
            retry_after=now + self.DISCOVERY_RETRY_SECONDS,
        )
        return observation

    def _refresh(self, accounts: dict[str, tuple[str, str, str, str]], signature: tuple[tuple[str, str, str, str], ...]) -> None:
        now, observed_now = self._clock(), self._observed_clock()
        observations = {
            identity: {**observation, "models": {key: dict(value) for key, value in observation.get("models", {}).items()}}
            for identity, observation in self._model_observations.items()
            if identity in accounts
        }
        anonymous_models = dict(self._anonymous_models)

        for identity, (future, probe_type, probe_fingerprint) in list(self._inflight_accounts.items()):
            if (identity not in accounts or accounts[identity][2] != "active"
                    or accounts[identity][1] != probe_type or accounts[identity][3] != probe_fingerprint):
                future.cancel()
                self._inflight_accounts.pop(identity, None)

        anonymous_future = None
        anonymous_is_new = False
        if self._anonymous_due(now):
            if self._anonymous_inflight is None:
                self._anonymous_inflight = self._executor.submit(self._fetch_models)
                anonymous_is_new = True
            anonymous_future = self._anonymous_inflight

        account_futures: dict[str, tuple[object, bool]] = {}
        new_futures = [anonymous_future] if anonymous_is_new and anonymous_future is not None else []
        for identity, (access_token, account_type, state, fingerprint) in accounts.items():
            observation = observations.get(identity)
            if not self._account_due(identity, account_type, state, fingerprint, observation, now):
                continue
            probe = self._inflight_accounts.get(identity)
            future = probe[0] if probe is not None else None
            is_new = False
            if future is None:
                future = self._executor.submit(self._fetch_account_models, access_token)
                self._inflight_accounts[identity] = (future, account_type, fingerprint)
                is_new = True
                new_futures.append(future)
            account_futures[identity] = (future, is_new)

        if new_futures:
            wait(new_futures, timeout=self.DISCOVERY_BUDGET_SECONDS)
        # Credentials or account state can change during the bounded wait.
        # A completed read must still belong to the current physical account.
        accounts = self._active_accounts()
        signature = self._signature(accounts)
        observations = {identity: row for identity, row in observations.items() if identity in accounts}
        for identity, (future, probe_type, probe_fingerprint) in list(self._inflight_accounts.items()):
            current = accounts.get(identity)
            if (current is None or current[2] != "active"
                    or current[1] != probe_type or current[3] != probe_fingerprint):
                future.cancel()
                self._inflight_accounts.pop(identity, None)
                account_futures.pop(identity, None)
        # Waiting is bounded, but it can still take long enough for a peer's
        # cache record to expire.  Snapshot eligibility using the completion
        # time rather than the time the refresh began.
        now, observed_now = self._clock(), self._observed_clock()
        if anonymous_future is not None:
            if anonymous_future.done():
                self._anonymous_inflight = None
                try:
                    models, read_at, read_observed_at = anonymous_future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning({"event": "model_catalog_anonymous_failed", "error_type": type(exc).__name__})
                    self._anonymous_observation["retry_after"] = self._failure_retry_at(exc, now)
                else:
                    if read_at + self._cache_ttl_seconds > now:
                        anonymous_models = models
                    self._anonymous_observation = {
                        "refresh_after": read_at + self._cache_ttl_seconds,
                        "observed_at": read_observed_at,
                    }
            elif anonymous_is_new:
                logger.warning({"event": "model_catalog_anonymous_timeout"})
                self._anonymous_observation["retry_after"] = now + self.DISCOVERY_RETRY_SECONDS

        models_by_account: dict[str, dict[str, dict[str, Any]]] = {}
        for identity, (_access_token, account_type, state, fingerprint) in accounts.items():
            observation = observations.get(identity)
            if state != "active":
                if observation is not None:
                    observation_state = observation.get("observation_state", "unknown")
                    if state == "limited":
                        observation_state = "observed" if self._observation_state_after_failure(observation, observed_now) == "read_failed" else "stale"
                    observation.update(
                        account_type=account_type,
                        state=state,
                        reason=self._observation_reason(state),
                        observation_state=observation_state,
                        last_seen_account_state=state,
                        last_attempt_at=observed_now,
                    )
                continue
            future_info = account_futures.get(identity)
            if future_info is not None:
                future, is_new = future_info
                if future.done():
                    self._inflight_accounts.pop(identity, None)
                    try:
                        models, read_at, read_observed_at = future.result()
                    except Exception as exc:  # noqa: BLE001
                        observations[identity] = observation = self._set_failed_observation(
                            observation, account_type=account_type, credential_fingerprint=fingerprint,
                            now=now, observed_now=observed_now,
                        )
                        logger.warning({"event": "model_catalog_account_failed", "error_type": type(exc).__name__})
                    else:
                        observations[identity] = observation = {
                            "models": models,
                            "account_type": account_type,
                            "state": "unknown",
                            "reason": self._observation_reason("observed") if read_at + self._cache_ttl_seconds > now else "stale",
                            "observation_state": "observed" if read_at + self._cache_ttl_seconds > now else "stale",
                            "observed_at": read_observed_at,
                            "last_attempt_at": observed_now,
                            "refresh_after": read_at + self._cache_ttl_seconds,
                            "credential_fingerprint": fingerprint,
                            "observed_account_state": "active",
                            "last_seen_account_state": "active",
                        }
                elif is_new:
                    observations[identity] = observation = self._set_failed_observation(
                        observation, account_type=account_type, credential_fingerprint=fingerprint,
                        now=now, observed_now=observed_now,
                    )
                    logger.warning({"event": "model_catalog_account_timeout"})
                elif observation is not None:
                    # An unfinished read is not executable evidence. Retain
                    # that same future so a later caller can collect it.
                    observations[identity] = observation = self._set_failed_observation(
                        observation, account_type=account_type, credential_fingerprint=fingerprint,
                        now=now, observed_now=observed_now,
                    )
            if self._reusable_observation(observations.get(identity), account_type, state, fingerprint, now):
                models_by_account[identity] = observations[identity]["models"]

        self._anonymous_models = anonymous_models
        self._model_observations = observations
        self._models_by_account = models_by_account
        self._account_types = {
            identity: account_type
            for identity, (_token, account_type, state, _fingerprint) in accounts.items()
            if state == "active" and identity in models_by_account
        }
        self._catalog_unknown = bool(
            any(account_type in {"Plus", "Pro", "ProLite", "Team", "Enterprise"}
                and state in {"active", "limited"} and identity not in models_by_account
                for identity, (_token, account_type, state, _fingerprint) in accounts.items())
        )
        next_refreshes = [self._anonymous_next_refresh(now)]
        next_refreshes.extend(
            next_at for identity, (_token, account_type, state, fingerprint) in accounts.items()
            if (next_at := self._account_next_refresh(identity, account_type, state, fingerprint, observations.get(identity), now)) is not None
        )
        self._expires_at = min(next_refreshes, default=now + self._cache_ttl_seconds)
        self._account_signature = signature

    def _ensure_catalog(self) -> None:
        accounts = self._active_accounts()
        signature = self._signature(accounts)
        with self._lock:
            if signature == self._account_signature and self._clock() < self._expires_at:
                return
            self._refresh(accounts, signature)

    def list_models(self) -> dict[str, Any]:
        self._ensure_catalog()
        with self._lock:
            union: dict[str, dict[str, Any]] = {
                model_id: dict(item)
                for model_id, item in self._anonymous_models.items()
            }
            for identity in sorted(self._model_observations):
                for model_id, item in self._model_observations[identity].get("models", {}).items():
                    union.setdefault(model_id, dict(item))
            result = {
                "object": "list",
                "data": [union[model_id] for model_id in sorted(union)],
            }
            # Keep readiness with this exact snapshot. Missing from a partial
            # catalog does not prove that a configured model is unsupported.
            if self._catalog_unknown:
                result["model_catalog"] = {"state": "partial"}
            return result

    def management_models(self) -> list[dict[str, Any]]:
        """Project observed account-level Chat capability for management reads."""
        self._ensure_catalog()
        accounts = self._accounts.list_accounts()
        with self._lock:
            anonymous = {key: dict(value) for key, value in self._anonymous_models.items()}
            observations = {
                identity: {**observation, "models": {key: dict(value) for key, value in observation.get("models", {}).items()}}
                for identity, observation in self._model_observations.items()
            }
        model_payloads: dict[str, dict[str, Any]] = dict(anonymous)
        for observation in observations.values():
            for model_id, payload in observation["models"].items():
                model_payloads.setdefault(model_id, payload)
        # gpt-image-2 is a provider alias.  Its evidence comes from the
        # persisted Chat image_gen counter, never from an upstream model row.
        model_payloads.pop("gpt-image-2", None)

        from services.owned_accounts import public_pool_account
        from services.public_chat_service import public_reasoning_efforts

        result: list[dict[str, Any]] = []
        for model_id in sorted(model_payloads):
            capabilities = (
                ["image_generation", "image_edit"]
                if model_id == "gpt-image-2" else ["text", "image_input"]
            )
            projected_accounts = []
            for account in accounts:
                if str(account.get("source_type") or "").strip().lower() not in {"web", "oauth_login", "password"}:
                    continue
                identity = self._accounts._stable_account_identity(account)
                observation = observations.get(identity)
                if observation is None or model_id not in observation["models"]:
                    continue
                safe = public_pool_account(account)
                observation_state = str(observation.get("observation_state") or "unknown")
                state = "unknown" if observation_state == "observed" and observation.get("state") == "unknown" else "unavailable"
                projected_accounts.append({
                    "account_ref": safe["account_ref"],
                    "label": safe["label"],
                    "state": state,
                    "reason": observation["reason"],
                    "capabilities": capabilities,
                    "observed_at": observation.get("observed_at"),
                    "observation_state": observation.get("observation_state", "unknown"),
                })
            supported = len(projected_accounts)
            result.append({
                "id": model_id,
                "label": str(model_payloads[model_id].get("label") or model_id)[:160],
                "route": "chat",
                "capabilities": capabilities,
                "state": "unknown",
                "supported_accounts": supported,
                "available_accounts": None,
                "pending_accounts": supported,
                "unavailable_accounts": 0,
                "accounts": projected_accounts,
                "reasoning_efforts": public_reasoning_efforts(model_id),
            })
        image_accounts = self.image_capability_rows(management=True)
        if image_accounts:
            pending = sum(row["state"] == "unknown" for row in image_accounts)
            unavailable = sum(row["state"] == "unavailable" for row in image_accounts)
            result.append({
                "id": "gpt-image-2",
                "label": "gpt-image-2",
                "route": "chat",
                "capabilities": ["image_generation", "image_edit"],
                "state": "unknown" if pending else "unavailable",
                # Capability observation is deliberately separate from a
                # dispatchable slot, so no aggregate free-capacity claim.
                "supported_accounts": len(image_accounts),
                "available_accounts": None,
                "pending_accounts": pending,
                "unavailable_accounts": unavailable,
                "accounts": image_accounts,
                "reasoning_efforts": [],
            })
        return result

    def image_capability_rows(self, *, management: bool = False) -> list[dict[str, Any]]:
        """Expose stored image_gen capability evidence for the provider alias."""
        from services.owned_accounts import image_capability_projection, public_pool_account

        rows: list[dict[str, Any]] = []
        for account in self._accounts.list_accounts():
            if str(account.get("source_type") or "").strip().lower() not in {"web", "oauth_login", "password"}:
                continue
            if self._accounts._normalize_account_type(account.get("type")) not in {"Plus", "Pro", "ProLite", "Team", "Enterprise"}:
                continue
            evidence = image_capability_projection(account)
            # With no valid upstream image_gen observation there is no model
            # capability to advertise. A failed later read retains its last
            # valid evidence and is included as read_failed/stale instead.
            if not evidence["capable"]:
                continue
            row = {
                "account_ref": self._accounts.pool_account_ref(account),
                "state": evidence["state"],
                "reason": evidence["reason"],
                "capabilities": ["image_generation", "image_edit"],
                "observed_at": evidence["observed_at"],
                "observation_state": evidence["observation_state"],
            }
            if management:
                row["label"] = public_pool_account(account)["label"]
            rows.append(row)
        return sorted(rows, key=lambda item: item["account_ref"])

    def public_accounts_for_model(self, model: str, capabilities: list[str]) -> list[dict[str, Any]]:
        """Return capability-only account rows for the ordinary model directory.

        The opaque pool reference is sufficient for a caller to distinguish
        compatible company accounts.  Labels, identities, ownership and all
        credentials remain management-only.
        """
        if str(model or "").strip() == "gpt-image-2":
            return self.image_capability_rows()
        self._ensure_catalog()
        with self._lock:
            observed = {
                identity: observation
                for identity, observation in self._model_observations.items()
                if model in observation.get("models", {})
            }
        rows = []
        for account in self._accounts.list_accounts():
            observation = observed.get(self._accounts._stable_account_identity(account))
            if observation is None:
                continue
            rows.append({
                "account_ref": self._accounts.pool_account_ref(account),
                # Catalog capability is known; dispatch capacity remains a
                # separate admission decision and is deliberately not claimed.
                "state": "unknown" if (observation.get("observation_state") == "observed"
                                             and observation.get("state") == "unknown") else "unavailable",
                "reason": observation["reason"],
                "capabilities": list(capabilities),
                "observed_at": observation.get("observed_at"),
                "observation_state": observation.get("observation_state", "unknown"),
            })
        return sorted(rows, key=lambda item: item["account_ref"])

    def route_for_model(self, model: str) -> ModelRoute:
        model = str(model or "").strip()
        self._ensure_catalog()
        with self._lock:
            account_types = frozenset(
                self._account_types[identity]
                for identity, models in self._models_by_account.items()
                if model in models
            )
            identities = frozenset(
                identity for identity, models in self._models_by_account.items()
                if model in models
            )
            return ModelRoute(
                account_types=account_types,
                allow_anonymous=(model in self._anonymous_models
                                 and self._at(self._anonymous_observation.get("refresh_after")) > self._clock()),
                account_identities=identities,
            )

    def known_account_types_for_model(self, model: str) -> frozenset[str]:
        """Return capability evidence, which is intentionally broader than routing."""
        model = str(model or "").strip()
        self._ensure_catalog()
        with self._lock:
            return frozenset(
                str(observation.get("account_type") or "")
                for observation in self._model_observations.values()
                if model in observation.get("models", {}) and observation.get("account_type")
            )

    def catalog_is_unknown(self) -> bool:
        self._ensure_catalog()
        with self._lock:
            return self._catalog_unknown


model_catalog_service = ModelCatalogService(account_service)
