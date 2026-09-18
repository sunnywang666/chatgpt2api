"""Safe projections and admission checks for the ordinary-key Chat API."""
from __future__ import annotations

from typing import Any

from services.model_service import model_catalog_service
from utils.helper import is_supported_image_model


PAID_ACCOUNT_TYPES = frozenset({"Plus", "Pro", "ProLite", "Team", "Enterprise"})
CHAT_IMAGE_MIME_TYPES = ["image/png", "image/jpeg", "image/webp"]
CHAT_INPUT_LIMITS = {
    "max_messages": 100,
    "max_text_bytes": 1024 * 1024,
    "max_json_body_bytes": 140 * 1024 * 1024,
    "max_images": 16,
    "max_image_bytes": 50 * 1024 * 1024,
    "max_total_image_bytes": 100 * 1024 * 1024,
    "max_image_pixels": 50_000_000,
    "image_mime_types": CHAT_IMAGE_MIME_TYPES,
}
IMAGE_OUTPUT_LIMITS = {
    "max_outputs_per_task": 1,
    "max_input_images": 16,
    "max_image_bytes": 50 * 1024 * 1024,
    "max_total_image_bytes": 100 * 1024 * 1024,
    "image_mime_types": CHAT_IMAGE_MIME_TYPES,
}


class PublicChatContractError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def is_public_text_model(model: object) -> bool:
    model_id = str(model or "").strip()
    if not model_id or model_id == "auto" or is_supported_image_model(model_id):
        return False
    route = model_catalog_service.route_for_model(model_id)
    return bool(route.account_types & PAID_ACCOUNT_TYPES)


def require_public_text_model(model: object) -> str:
    model_id = str(model or "").strip()
    if not model_id or model_id == "auto":
        raise PublicChatContractError(
            "CHAT_MODEL_REQUIRED",
            "model must be an advertised Chat text model",
        )
    if is_supported_image_model(model_id):
        raise PublicChatContractError(
            "CHAT_MODEL_UNSUPPORTED",
            "image generation models are not supported by chat-requests",
        )
    if not is_public_text_model(model_id):
        raise PublicChatContractError(
            "CHAT_MODEL_UNSUPPORTED",
            "model is not available to the ordinary Chat service",
        )
    return model_id


def project_public_models(result: object) -> dict[str, Any]:
    if not isinstance(result, dict) or not isinstance(result.get("data"), list):
        raise PublicChatContractError("MODEL_DISCOVERY_UNAVAILABLE", "model discovery unavailable")
    data: list[dict[str, Any]] = []
    for raw in result["data"]:
        if not isinstance(raw, dict):
            continue
        model_id = str(raw.get("id") or "").strip()
        if model_id == "gpt-image-2":
            data.append({
                **raw,
                "capabilities": ["image_generation", "image_edit"],
                "input_limits": dict(IMAGE_OUTPUT_LIMITS),
            })
        elif is_public_text_model(model_id):
            data.append({
                **raw,
                "capabilities": ["text", "image_input"],
                "input_limits": dict(CHAT_INPUT_LIMITS),
            })
    return {"object": "list", "data": data}


def project_public_chat_receipt(receipt: object) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise PublicChatContractError("CHAT_RECEIPT_INVALID", "chat request receipt is invalid")
    result: dict[str, Any] = {
        "request_id": str(receipt.get("request_id") or ""),
        "route": "chat",
        "model": str(receipt.get("model") or ""),
        "status": str(receipt.get("status") or "unknown"),
    }
    content = receipt.get("content")
    if isinstance(content, str) and content:
        result["content"] = content
    error_code = receipt.get("error_code")
    if isinstance(error_code, str) and error_code:
        result["error_code"] = error_code
    for field in ("created_at", "updated_at", "started_at", "finished_at"):
        value = receipt.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[field] = value

    recovery_mapping = {
        "recovery_attempt": "attempt",
        "recovery_next_at": "next_at",
        "recovery_error_code": "error_code",
        "recovery_phase": "phase",
        "recovery_reason": "reason",
        "recovery_no_result_reads": "no_result_reads",
        "recovery_retryable": "retryable",
        "recovery_requires_new_conversation": "requires_new_conversation",
        "upstream_outcome": "upstream_outcome",
    }
    recovery = {
        public_name: receipt[internal_name]
        for internal_name, public_name in recovery_mapping.items()
        if receipt.get(internal_name) is not None
    }
    if recovery:
        result["recovery"] = recovery
    return result
