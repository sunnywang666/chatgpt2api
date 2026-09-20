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


class ModelUnavailableError(RuntimeError):
    pass


class ModelCatalogService:
    """Caches the model catalogs advertised to each active account type."""

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
        self._account_signature: tuple[tuple[str, int], ...] = ()
        self._anonymous_models: dict[str, dict[str, Any]] = {}
        self._models_by_account_type: dict[str, dict[str, dict[str, Any]]] = {}

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

    def _active_accounts_by_type(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for account in self._accounts.list_accounts():
            if not isinstance(account, dict) or account.get("status") in {"禁用", "异常"}:
                continue
            if str(account.get("source_type") or "").strip().lower() not in {"web", "oauth_login", "password"}:
                # Codex authorization is a separate bearer and must never be
                # sent to Chat's model-catalog endpoint.
                continue
            access_token = str(account.get("access_token") or "").strip()
            account_type = self._accounts._normalize_account_type(account.get("type"))
            if access_token and account_type:
                groups.setdefault(account_type, []).append(access_token)
        return groups

    @staticmethod
    def _signature(groups: dict[str, list[str]]) -> tuple[tuple[str, int], ...]:
        return tuple(
            (account_type, len(tokens))
            for account_type, tokens in sorted(groups.items())
        )

    def _fetch_models(self, access_token: str = "") -> dict[str, dict[str, Any]]:
        backend = self._backend_factory(access_token=access_token)
        try:
            return self._model_map(backend.list_models())
        finally:
            backend.close()

    def _fetch_account_type_models(
        self,
        account_type: str,
        access_tokens: list[str],
    ) -> dict[str, dict[str, Any]] | None:
        attempted_tokens: set[str] = set()
        last_error: Exception | None = None
        for access_token in access_tokens:
            try:
                resolved_token = self._accounts.refresh_access_token(
                    access_token,
                    event="list_models",
                ) or access_token
                if resolved_token in attempted_tokens:
                    continue
                attempted_tokens.add(resolved_token)
                return self._fetch_models(resolved_token)
            except Exception as exc:  # noqa: BLE001 - try the next account for any upstream failure
                last_error = exc
        if last_error is not None:
            logger.warning({
                "event": "model_catalog_account_type_failed",
                "account_type": account_type,
                "error_type": type(last_error).__name__,
            })
        return None

    def _refresh(self, groups: dict[str, list[str]], signature: tuple[tuple[str, int], ...]) -> None:
        models_by_account_type: dict[str, dict[str, dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(groups) + 1)) as executor:
            anonymous_future = executor.submit(self._fetch_models)
            account_futures = {
                account_type: executor.submit(
                    self._fetch_account_type_models,
                    account_type,
                    access_tokens,
                )
                for account_type, access_tokens in groups.items()
            }
            try:
                anonymous_models = anonymous_future.result()
            except Exception as exc:  # noqa: BLE001 - retain cached models on upstream failure
                logger.warning({
                    "event": "model_catalog_anonymous_failed",
                    "error_type": type(exc).__name__,
                })
                anonymous_models = self._anonymous_models

            for account_type, future in account_futures.items():
                models = future.result()
                if models is not None:
                    models_by_account_type[account_type] = models
                elif account_type in self._models_by_account_type:
                    models_by_account_type[account_type] = self._models_by_account_type[account_type]

        self._anonymous_models = anonymous_models
        self._models_by_account_type = models_by_account_type
        self._account_signature = signature
        self._expires_at = self._clock() + self._cache_ttl_seconds

    def _ensure_catalog(self) -> None:
        groups = self._active_accounts_by_type()
        signature = self._signature(groups)
        with self._lock:
            if signature == self._account_signature and self._clock() < self._expires_at:
                return
            self._refresh(groups, signature)

    def list_models(self) -> dict[str, Any]:
        self._ensure_catalog()
        with self._lock:
            union: dict[str, dict[str, Any]] = {
                model_id: dict(item)
                for model_id, item in self._anonymous_models.items()
            }
            for account_type in sorted(self._models_by_account_type):
                for model_id, item in self._models_by_account_type[account_type].items():
                    union.setdefault(model_id, dict(item))
        return {
            "object": "list",
            "data": [union[model_id] for model_id in sorted(union)],
        }

    def management_models(self) -> list[dict[str, Any]]:
        """Project the real Chat catalog without inventing per-account probes."""
        self._ensure_catalog()
        accounts = self._accounts.list_accounts()
        with self._lock:
            anonymous = {key: dict(value) for key, value in self._anonymous_models.items()}
            by_type = {
                account_type: {key: dict(value) for key, value in models.items()}
                for account_type, models in self._models_by_account_type.items()
            }
        model_types: dict[str, set[str]] = {}
        model_payloads: dict[str, dict[str, Any]] = dict(anonymous)
        for account_type, models in by_type.items():
            for model_id, payload in models.items():
                model_types.setdefault(model_id, set()).add(account_type)
                model_payloads.setdefault(model_id, payload)

        from services.owned_accounts import public_pool_account
        from services.public_chat_service import public_reasoning_efforts

        result: list[dict[str, Any]] = []
        for model_id in sorted(model_payloads):
            supported_types = model_types.get(model_id, set())
            projected_accounts = []
            unavailable = 0
            pending = 0
            for account in accounts:
                if str(account.get("source_type") or "").strip().lower() not in {"web", "oauth_login", "password"}:
                    continue
                account_type = self._accounts._normalize_account_type(account.get("type"))
                if account_type not in supported_types:
                    continue
                safe = public_pool_account(account)
                if account.get("managed_disabled") or account.get("status") == "禁用":
                    state, reason = "unavailable", "disabled"
                    unavailable += 1
                elif account.get("status") == "异常":
                    state, reason = "unavailable", "account_unavailable"
                    unavailable += 1
                else:
                    # The cache proves support for this account type only. It
                    # does not prove that every peer account was observed.
                    state, reason = "unknown", "account_type_catalog_only"
                    pending += 1
                projected_accounts.append({
                    "account_ref": safe["account_ref"],
                    "label": safe["label"],
                    "state": state,
                    "reason": reason,
                })
            capabilities = (
                ["image_generation", "image_edit"]
                if model_id == "gpt-image-2" else ["text", "image_input"]
            )
            supported = len(projected_accounts)
            state = "unavailable" if supported and unavailable == supported else "unknown"
            result.append({
                "id": model_id,
                "label": str(model_payloads[model_id].get("label") or model_id)[:160],
                "route": "chat",
                "capabilities": capabilities,
                "state": state,
                "supported_accounts": supported,
                "available_accounts": 0 if state == "unavailable" else None,
                "pending_accounts": pending,
                "unavailable_accounts": unavailable,
                "accounts": projected_accounts,
                "reasoning_efforts": public_reasoning_efforts(model_id),
            })
        return result

    def route_for_model(self, model: str) -> ModelRoute:
        model = str(model or "").strip()
        self._ensure_catalog()
        with self._lock:
            account_types = frozenset(
                account_type
                for account_type, models in self._models_by_account_type.items()
                if model in models
            )
            return ModelRoute(
                account_types=account_types,
                allow_anonymous=model in self._anonymous_models,
            )


model_catalog_service = ModelCatalogService(account_service)
