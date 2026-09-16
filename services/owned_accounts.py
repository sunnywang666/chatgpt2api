"""Safe Workbench projections over the existing account pool, not another pool."""
from __future__ import annotations

from datetime import datetime, timezone
import math


def observed_capacity(account: dict) -> dict:
    limits = account.get("limits_progress")
    image_limit = next((value for value in limits if isinstance(value, dict) and value.get("feature_name") == "image_gen"), None) if isinstance(limits, list) else None
    remaining = image_limit.get("remaining") if image_limit else None
    valid = isinstance(remaining, (int, float)) and not isinstance(remaining, bool) and math.isfinite(remaining) and remaining >= 0 and remaining == int(remaining)
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
    email = str(account.get("email") or "")
    local, sep, domain = email.partition("@")
    capacity = observed_capacity(account)
    status = str(account.get("status") or "")
    return {
        "id": account["managed_account_id"],
        "label": f"{local[:1]}***@{domain}" if sep else "已接入账号",
        "source_type": account.get("source_type", "web"),
        "enabled": not bool(account.get("managed_disabled")),
        "connection_status": "disabled" if account.get("managed_disabled") else "unavailable" if status in {"异常", "禁用"} else "connected" if capacity["observed_at"] else "unverified",
        "capacity": capacity,
        "updated_at": account.get("managed_updated_at"),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
