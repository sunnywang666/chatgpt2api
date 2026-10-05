from __future__ import annotations

from typing import Any

from services.account_service import account_service
from services.model_service import model_catalog_service
from utils.helper import CODEX_IMAGE_MODEL


def list_models() -> dict[str, Any]:
    result = model_catalog_service.list_models()
    data = result.get("data")
    if not isinstance(data, list):
        return result
    seen = {str(item.get("id") or "").strip() for item in data if isinstance(item, dict)}
    dynamic_models: set[str] = set()
    accounts = account_service.list_accounts()
    # gpt-image-2 is a provider alias. It is listed only when persisted
    # image_gen evidence exists, rather than treating account presence as a
    # claim that the upstream account can generate images.
    observed_image_accounts = model_catalog_service.image_capability_rows()
    if not observed_image_accounts:
        data[:] = [
            item for item in data
            if not isinstance(item, dict) or str(item.get("id") or "").strip() != "gpt-image-2"
        ]
        seen = {str(item.get("id") or "").strip() for item in data if isinstance(item, dict)}
    codex_types = {
        normalized
        for account in accounts
        if isinstance(account, dict)
           and account_service._normalize_source_type(account.get("source_type")) == "codex"
           and (normalized := account_service._normalize_account_type(account.get("type")))
    }

    if observed_image_accounts:
        dynamic_models.add("gpt-image-2")
    if codex_types & {"Plus", "Team", "Pro"}:
        dynamic_models.add(CODEX_IMAGE_MODEL)
    if "Plus" in codex_types:
        dynamic_models.add(f"plus-{CODEX_IMAGE_MODEL}")
    if "Team" in codex_types:
        dynamic_models.add(f"team-{CODEX_IMAGE_MODEL}")
    if "Pro" in codex_types:
        dynamic_models.add(f"pro-{CODEX_IMAGE_MODEL}")

    for model in sorted(dynamic_models):
        if model not in seen:
            data.append({
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": "chatgpt2api",
                "permission": [],
                "root": model,
                "parent": None,
            })
    return result
