from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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

    def __init__(
        self,
        accounts: AccountService,
        *,
        backend_factory: Callable[..., Any] = OpenAIBackendAPI,
        cache_ttl_seconds: float = 300,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._accounts = accounts
        self._backend_factory = backend_factory
        self._cache_ttl_seconds = max(1.0, float(cache_ttl_seconds))
        self._clock = clock
        self._lock = RLock()
        self._expires_at = 0.0
        self._account_signature: tuple[tuple[str, str], ...] = ()
        self._anonymous_models: dict[str, dict[str, Any]] = {}
        self._models_by_account: dict[str, dict[str, dict[str, Any]]] = {}
        self._account_types: dict[str, str] = {}

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

    def _active_accounts(self) -> dict[str, tuple[str, str]]:
        """Return exactly one usable token for each stable physical account.

        Model access belongs to a concrete account, not its subscription type.
        Duplicate stable identities are excluded because choosing either record
        would lose the original-account authority used by durable receipts.
        """
        candidates: dict[str, list[tuple[str, str]]] = {}
        for account in self._accounts.list_accounts():
            if (not isinstance(account, dict) or account.get("managed_disabled")
                    or account.get("status") in {"禁用", "异常", "限流"}):
                continue
            if str(account.get("source_type") or "").strip().lower() not in {"web", "oauth_login", "password"}:
                # Codex authorization is a separate bearer and must never be
                # sent to Chat's model-catalog endpoint.
                continue
            access_token = str(account.get("access_token") or "").strip()
            account_type = self._accounts._normalize_account_type(account.get("type"))
            identity = self._accounts._stable_account_identity(account)
            if access_token and account_type and identity:
                candidates.setdefault(identity, []).append((access_token, account_type))
        active: dict[str, tuple[str, str]] = {}
        for identity, rows in candidates.items():
            if len(rows) == 1:
                active[identity] = rows[0]
            else:
                logger.warning({"event": "model_catalog_account_identity_ambiguous"})
        return active

    @staticmethod
    def _signature(accounts: dict[str, tuple[str, str]]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((identity, account_type) for identity, (_token, account_type) in accounts.items()))

    def _fetch_models(self, access_token: str = "") -> dict[str, dict[str, Any]]:
        backend = self._backend_factory(access_token=access_token)
        try:
            return self._model_map(backend.list_models())
        finally:
            backend.close()

    def _fetch_account_models(self, access_token: str) -> dict[str, dict[str, Any]]:
        resolved_token = self._accounts.refresh_access_token(access_token, event="list_models") or access_token
        return self._fetch_models(resolved_token)

    def _refresh(self, accounts: dict[str, tuple[str, str]], signature: tuple[tuple[str, str], ...]) -> None:
        models_by_account: dict[str, dict[str, dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=min(self.MAX_DISCOVERY_WORKERS, len(accounts) + 1)) as executor:
            anonymous_future = executor.submit(self._fetch_models)
            account_futures = {
                identity: executor.submit(self._fetch_account_models, access_token)
                for identity, (access_token, _account_type) in accounts.items()
            }
            try:
                anonymous_models = anonymous_future.result()
            except Exception as exc:  # noqa: BLE001 - retain cached models on upstream failure
                logger.warning({
                    "event": "model_catalog_anonymous_failed",
                    "error_type": type(exc).__name__,
                })
                anonymous_models = self._anonymous_models

            for identity, future in account_futures.items():
                try:
                    models_by_account[identity] = future.result()
                except Exception as exc:  # noqa: BLE001 - failed observations must not route work
                    logger.warning({
                        "event": "model_catalog_account_failed",
                        "error_type": type(exc).__name__,
                    })

        self._anonymous_models = anonymous_models
        self._models_by_account = models_by_account
        self._account_types = {identity: account_type for identity, (_token, account_type) in accounts.items()}
        self._account_signature = signature
        self._expires_at = self._clock() + self._cache_ttl_seconds

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
            for identity in sorted(self._models_by_account):
                for model_id, item in self._models_by_account[identity].items():
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
            by_account = {
                identity: {key: dict(value) for key, value in models.items()}
                for identity, models in self._models_by_account.items()
            }
        model_payloads: dict[str, dict[str, Any]] = dict(anonymous)
        for models in by_account.values():
            for model_id, payload in models.items():
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
                if model_id not in by_account.get(identity, {}):
                    continue
                safe = public_pool_account(account)
                # A model-catalog read proves capability, not free capacity.
                state, reason = "unknown", "model_catalog_observed"
                projected_accounts.append({
                    "account_ref": safe["account_ref"],
                    "label": safe["label"],
                    "state": state,
                    "reason": reason,
                    "capabilities": capabilities,
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
                identity for identity, models in self._models_by_account.items()
                if model in models
            }
        rows = []
        for account in self._accounts.list_accounts():
            if self._accounts._stable_account_identity(account) not in observed:
                continue
            rows.append({
                "account_ref": self._accounts.pool_account_ref(account),
                # Catalog capability is known; dispatch capacity remains a
                # separate admission decision and is deliberately not claimed.
                "state": "unknown",
                "reason": "model_catalog_observed",
                "capabilities": list(capabilities),
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


model_catalog_service = ModelCatalogService(account_service)
