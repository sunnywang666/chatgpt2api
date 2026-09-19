"""Safe Workbench projections over the existing account pool, not another pool."""
from __future__ import annotations

from datetime import datetime, timezone
import math


OBSERVATION_MAX_AGE_SECONDS = 5 * 60


def _parse_observed_at(value: object) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def observation_is_fresh(value: object) -> bool:
    observed_at = _parse_observed_at(value)
    return bool(
        observed_at
        and (datetime.now(timezone.utc) - observed_at).total_seconds() <= OBSERVATION_MAX_AGE_SECONDS
    )


def _mask_email(value: object) -> str | None:
    email = str(value or "").strip()
    local, separator, domain = email.partition("@")
    if not separator or not local or not domain:
        return None
    return f"{local[:1]}***@{domain}"


def _mask_identifier(value: object) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if len(raw) <= 8:
        return "***"
    return f"{raw[:4]}…{raw[-4:]}"


def masked_identity(account: dict) -> dict[str, str | None]:
    email = _mask_email(account.get("email"))
    account_id = _mask_identifier(account.get("account_id"))
    user_id = _mask_identifier(account.get("user_id"))
    return {
        "email": email,
        "account_id": account_id,
        "user_id": user_id,
    }


def identity_label(account: dict) -> str | None:
    identity = masked_identity(account)
    return identity["email"] or identity["account_id"] or identity["user_id"]


def chat_projection(account: dict) -> dict:
    source_type = str(account.get("source_type") or "").strip().lower()
    token_present = bool(str(account.get("access_token") or "").strip())
    if source_type in {"web", "oauth_login", "password"}:
        authorization_status = "saved" if token_present else "missing"
    elif source_type == "codex":
        # A Codex bearer is not Chat authorization. This distinction prevents
        # account-center refreshes from sending it to Chat metadata endpoints.
        authorization_status = "missing"
    else:
        authorization_status = "unknown" if token_present else "missing"

    failed_at = account.get("capacity_read_failed_at")
    observed_at = account.get("capacity_observed_at")
    status = str(account.get("status") or "")
    if authorization_status != "saved":
        state = "unknown"
        error_code = None
    elif status == "异常" and account.get("last_invalid_at"):
        state = "auth_required"
        error_code = "chat_authorization_required"
    elif failed_at:
        state = "read_failed"
        error_code = "chat_metadata_read_failed"
    elif observed_at:
        state = "observed"
        error_code = None
    else:
        state = "unknown"
        error_code = None
    result = {
        "authorization_status": authorization_status,
        "state": state,
        "observed_at": observed_at if isinstance(observed_at, str) else None,
    }
    if error_code:
        result["error_code"] = error_code
    return result


def observed_capacity(account: dict) -> dict:
    limits = account.get("limits_progress")
    image_limit = next((value for value in limits if isinstance(value, dict) and value.get("feature_name") == "image_gen"), None) if isinstance(limits, list) else None
    remaining = image_limit.get("remaining") if image_limit else None
    valid = (type(remaining) is int and remaining >= 0) or (
        isinstance(remaining, float) and math.isfinite(remaining)
        and remaining >= 0 and remaining.is_integer()
    )
    return {
        "route": "chatgpt_image_gen",
        "source": "limits_progress.image_gen.remaining",
        "unit": "upstream_image_gen",
        "state": "read_failed" if account.get("capacity_read_failed_at") else "stale" if valid and account.get("capacity_used_since_observation") else "observed" if valid else "unknown",
        "remaining": int(remaining) if valid else None,
        "observed_at": account.get("capacity_observed_at"),
        "failed_at": account.get("capacity_read_failed_at"),
        "reset_after": image_limit.get("reset_after") if image_limit else None,
        "codex_capacity": None,
    }


def public_owned_account(account: dict) -> dict:
    from services.codex_service import codex_service
    capacity = observed_capacity(account)
    codex = codex_service.account_projection(account)
    chat = chat_projection(account)
    masked = masked_identity(account)
    fallback_label = identity_label(account)
    status = str(account.get("status") or "")
    managed_disabled = bool(account.get("managed_disabled"))
    from services.account_service import AccountService
    result = {
        "id": account["managed_account_id"],
        "label": str(account.get("managed_label") or "").strip() or fallback_label or "已接入账号",
        "identity": masked,
        "identity_label": fallback_label,
        "source_type": account.get("source_type", "web"),
        "enabled": not managed_disabled,
        "connection_status": "disabled" if managed_disabled else "unavailable" if status in {"异常", "禁用"} else "connected" if codex["state"] in {"observed", "limited"} or capacity["observed_at"] else "unverified",
        "capacity": capacity,
        "chat": chat,
        "codex": codex,
        "updated_at": account.get("managed_updated_at"),
    }
    result["authorization_ref"] = AccountService.pool_account_ref(account)
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def public_pool_account(account: dict) -> dict:
    from services.codex_service import codex_service
    capacity = observed_capacity(account)
    codex = codex_service.account_projection(account)
    chat = chat_projection(account)
    masked = masked_identity(account)
    fallback_label = identity_label(account)
    status = str(account.get("status") or "")
    managed_disabled = bool(account.get("managed_disabled"))
    source = account.get("source_type")
    from services.account_service import AccountService
    account_ref = AccountService.pool_account_ref(account)
    result = {
        "id": account_ref,
        "account_ref": account_ref,
        "label": str(account.get("managed_label") or "").strip() or fallback_label or "AI 账号",
        "identity": masked,
        "identity_label": fallback_label,
        "source_type": source if source in {"web", "codex", "oauth_login", "password"} else "unknown",
        "enabled": not managed_disabled,
        "connection_status": "disabled" if managed_disabled else "unavailable" if status in {"异常", "禁用"} else "connected" if codex["state"] in {"observed", "limited"} or capacity["observed_at"] else "unverified",
        "capacity": capacity,
        "chat": chat,
        "codex": codex,
        "updated_at": account.get("managed_updated_at") or capacity["observed_at"] or codex.get("observed_at"),
    }
    # The management bridge retains only authorization_ref. Keep that value
    # bound to the persisted pool record even when an authorization adds or
    # changes the account's verified identity later.
    result["authorization_ref"] = account_ref
    return result
