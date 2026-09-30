from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import RLock
from typing import Any

from services.account_service import AccountService, account_service
from services.openai_backend_api import OpenAIBackendAPI
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
        self._account_signature: tuple[tuple[str, str, str], ...] = ()
        self._anonymous_models: dict[str, dict[str, Any]] = {}
        # A successful catalog read remains capability evidence after a later
        # read failure or rate limit.  It is deliberately separate from the
        # executable map below, which is rebuilt only from this refresh.
        self._model_observations: dict[str, dict[str, Any]] = {}
        self._catalog_unknown = False
        self._models_by_account: dict[str, dict[str, dict[str, Any]]] = {}
        self._account_types: dict[str, str] = {}
        self._executor = ThreadPoolExecutor(max_workers=self.MAX_DISCOVERY_WORKERS)
        self._inflight_accounts: dict[str, object] = {}
        self._anonymous_inflight: object | None = None

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

    def _active_accounts(self) -> dict[str, tuple[str, str, str]]:
        """Return exactly one stable physical account and its dispatch state.

        Model access belongs to a concrete account, not its subscription type.
        Duplicate stable identities are excluded because choosing either record
        would lose the original-account authority used by durable receipts.
        """
        candidates: dict[str, list[tuple[str, str, str]]] = {}
        for account in self._accounts.list_accounts():
            if not isinstance(account, dict):
                continue
            if str(account.get("source_type") or "").strip().lower() not in {"web", "oauth_login", "password"}:
                # Codex authorization is a separate bearer and must never be
                # sent to Chat's model-catalog endpoint.
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
                candidates.setdefault(identity, []).append((access_token, account_type, state))
        active: dict[str, tuple[str, str, str]] = {}
        for identity, rows in candidates.items():
            if len(rows) == 1:
                active[identity] = rows[0]
            else:
                logger.warning({"event": "model_catalog_account_identity_ambiguous"})
        return active

    @staticmethod
    def _signature(accounts: dict[str, tuple[str, str, str]]) -> tuple[tuple[str, str, str], ...]:
        return tuple(sorted(
            (identity, account_type, state)
            for identity, (_token, account_type, state) in accounts.items()
        ))

    def _fetch_models(self, access_token: str = "") -> dict[str, dict[str, Any]]:
        backend = self._backend_factory(access_token=access_token)
        try:
            return self._model_map(backend.list_models())
        finally:
            backend.close()

    def _fetch_account_models(self, access_token: str) -> dict[str, dict[str, Any]]:
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

    def _refresh(self, accounts: dict[str, tuple[str, str, str]], signature: tuple[tuple[str, str, str], ...]) -> None:
        models_by_account: dict[str, dict[str, dict[str, Any]]] = {}
        now = self._observed_clock()
        observations = {
            identity: {
                **observation,
                "models": {key: dict(value) for key, value in observation.get("models", {}).items()},
            }
            for identity, observation in self._model_observations.items()
            if identity in accounts
        }
        anonymous_models = self._anonymous_models
        # A timed-out or failed observation is never executable.  It also
        # shortens the cache so a completed slow read is followed by a fresh
        # observation rather than freezing failure evidence for the full TTL.
        retry_soon = False
        # One executor is retained for this catalog instance.  Slow requests
        # therefore occupy at most this bounded worker set across refreshes;
        # late futures are discarded rather than republished into a newer view.
        previous_anonymous = self._anonymous_inflight
        if previous_anonymous is not None and previous_anonymous.done():
            self._anonymous_inflight = None
        anonymous_future = self._anonymous_inflight
        anonymous_is_current = False
        if anonymous_future is None:
            anonymous_future = self._executor.submit(self._fetch_models)
            self._anonymous_inflight = anonymous_future
            anonymous_is_current = True
        # Removed or newly non-dispatchable accounts must not retain an
        # in-flight slot indefinitely.  A running call cannot be forcefully
        # stopped, but the shared bounded executor prevents it from spawning
        # another worker and any queued call is cancelled.
        for identity, future in list(self._inflight_accounts.items()):
            if identity not in accounts or accounts[identity][2] != "active":
                future.cancel()
                self._inflight_accounts.pop(identity, None)

        account_futures = {}
        current_futures = []
        for identity, (access_token, _account_type, state) in accounts.items():
            if state != "active":
                continue
            previous = self._inflight_accounts.get(identity)
            if previous is not None and previous.done():
                self._inflight_accounts.pop(identity, None)
                previous = None
            if previous is None:
                future = self._executor.submit(self._fetch_account_models, access_token)
                self._inflight_accounts[identity] = future
                account_futures[identity] = (future, True)
                current_futures.append(future)
            else:
                account_futures[identity] = (previous, False)
        done, _pending = wait(
            [anonymous_future, *current_futures],
            timeout=self.DISCOVERY_BUDGET_SECONDS,
        )
        if anonymous_is_current and anonymous_future in done:
            try:
                anonymous_models = anonymous_future.result()
            except Exception as exc:  # noqa: BLE001 - retain cached models on upstream failure
                logger.warning({
                    "event": "model_catalog_anonymous_failed",
                    "error_type": type(exc).__name__,
                })
                retry_soon = True
        else:
            logger.warning({"event": "model_catalog_anonymous_timeout"})
            retry_soon = True

        for identity, (_access_token, account_type, state) in accounts.items():
            if state != "active":
                previous = observations.get(identity)
                if previous is not None:
                    observation_state = previous.get("observation_state", "unknown")
                    if state == "limited":
                        observation_state = ("observed" if self._observation_state_after_failure(previous, now) == "read_failed"
                                             else "stale")
                    previous.update(
                        account_type=account_type,
                        state=state,
                        reason=self._observation_reason(state),
                        observation_state=observation_state,
                        last_attempt_at=now,
                    )
                continue
            future, is_current = account_futures[identity]
            if not is_current or future not in done:
                previous = observations.get(identity)
                if previous is not None:
                    observation_state = self._observation_state_after_failure(previous, now)
                    previous.update(
                        account_type=account_type,
                        state="unavailable",
                        reason="read_failed" if observation_state == "read_failed" else "stale",
                        observation_state=observation_state,
                        last_read_failed_at=now,
                    )
                logger.warning({"event": "model_catalog_account_timeout"})
                retry_soon = True
                continue
            try:
                models = future.result()
            except Exception as exc:  # noqa: BLE001 - failed observations must not route work
                previous = observations.get(identity)
                if previous is not None:
                    observation_state = self._observation_state_after_failure(previous, now)
                    previous.update(
                        account_type=account_type,
                        state="unavailable",
                        reason="read_failed" if observation_state == "read_failed" else "stale",
                        observation_state=observation_state,
                        last_read_failed_at=now,
                    )
                logger.warning({"event": "model_catalog_account_failed", "error_type": type(exc).__name__})
                retry_soon = True
                continue
            models_by_account[identity] = models
            previous = observations.get(identity, {})
            observations[identity] = {
                "models": models,
                "account_type": account_type,
                "state": "unknown",
                "reason": self._observation_reason("observed"),
                "observation_state": "observed",
                "observed_at": now,
                "last_attempt_at": now,
            }
        self._anonymous_models = anonymous_models
        self._model_observations = observations
        # Historical capability proves that a known model may be queued, but
        # it cannot prove that an arbitrary newly requested model is absent.
        # Any paid account without a fresh observation therefore makes the
        # directory incomplete and unknown-model admission retryable.
        self._catalog_unknown = bool(
            any(account_type in {"Plus", "Pro", "ProLite", "Team", "Enterprise"}
                and state in {"active", "limited"} and identity not in models_by_account
                for identity, (_token, account_type, state) in accounts.items())
        )
        self._models_by_account = models_by_account
        self._account_types = {
            identity: account_type
            for identity, (_token, account_type, state) in accounts.items()
            if state == "active" and identity in models_by_account
        }
        self._account_signature = signature
        # `wait()` does not include a retained prior future.  Treat both it
        # and any transient failure as a short retry, otherwise a late result
        # discarded for snapshot safety could leave recovery stalled for five
        # minutes.
        retry_soon = retry_soon or any(
            not is_current or future not in done
            for future, is_current in account_futures.values()
        ) or (not anonymous_is_current or anonymous_future not in done)
        self._expires_at = self._clock() + (1.0 if retry_soon else self._cache_ttl_seconds)

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
        return {
            "object": "list",
            "data": [union[model_id] for model_id in sorted(union)],
        }

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
        return result

    def public_accounts_for_model(self, model: str, capabilities: list[str]) -> list[dict[str, Any]]:
        """Return capability-only account rows for the ordinary model directory.

        The opaque pool reference is sufficient for a caller to distinguish
        compatible company accounts.  Labels, identities, ownership and all
        credentials remain management-only.
        """
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
                allow_anonymous=model in self._anonymous_models,
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
