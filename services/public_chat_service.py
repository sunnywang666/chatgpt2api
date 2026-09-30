"""Safe projections and admission checks for the ordinary-key Chat API."""
from __future__ import annotations

import math
from typing import Any

from services.model_service import model_catalog_service
from utils.helper import is_supported_image_model


PAID_ACCOUNT_TYPES = frozenset({"Plus", "Pro", "ProLite", "Team", "Enterprise"})
CHAT_IMAGE_MIME_TYPES = ["image/png", "image/jpeg", "image/webp"]
# Only the model whose native thinking transport supports this public setting.
PUBLIC_CHAT_REASONING_EFFORTS = {"gpt-5-6-thinking": ("high",)}
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
    known_types = getattr(model_catalog_service, "known_account_types_for_model", None)
    if callable(known_types):
        result = known_types(model_id)
        if isinstance(result, frozenset) and result & PAID_ACCOUNT_TYPES:
            return True
    # Compatibility doubles used by local callers before observation history
    # existed still model only the current executable route.
    route = model_catalog_service.route_for_model(model_id)
    return bool(route.account_types & PAID_ACCOUNT_TYPES)


def public_reasoning_efforts(model: object) -> list[str]:
    return list(PUBLIC_CHAT_REASONING_EFFORTS.get(str(model or "").strip(), ()))


def require_public_reasoning_effort(model: object, effort: object) -> None:
    if effort is not None and effort not in public_reasoning_efforts(model):
        raise PublicChatContractError(
            "CHAT_REASONING_UNSUPPORTED",
            "reasoning_effort=high is not supported for this model",
        )


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
        unknown = getattr(model_catalog_service, "catalog_is_unknown", None)
        if callable(unknown) and unknown() is True:
            raise PublicChatContractError(
                "MODEL_DISCOVERY_UNAVAILABLE",
                "model capability discovery is temporarily unavailable",
            )
        raise PublicChatContractError(
            "CHAT_MODEL_UNSUPPORTED",
            "model is not available to the ordinary Chat service",
        )
    # A known capability with no fresh executable account is accepted into the
    # existing durable admission path.  It stays queued under its original
    # request ID until a compatible account is observed again; do not relabel
    # this temporary condition as permanent unsupported.
    return model_id


def project_public_models(result: object) -> dict[str, Any]:
    if not isinstance(result, dict) or not isinstance(result.get("data"), list):
        raise PublicChatContractError("MODEL_DISCOVERY_UNAVAILABLE", "model discovery unavailable")
    unknown = getattr(model_catalog_service, "catalog_is_unknown", None)
    catalog_unknown = callable(unknown) and unknown() is True
    data: list[dict[str, Any]] = []
    has_text_model = False

    def public_model_fields(raw: dict[str, Any]) -> dict[str, Any]:
        # Upstream catalog extensions are not a public-account contract.
        # Keep OpenAI model metadata only; provider/account material must not
        # cross ordinary ingress merely because it appeared in a model row.
        return {
            field: raw[field]
            for field in ("id", "object", "created", "owned_by", "root", "parent")
            if field in raw and (type(raw[field]) in {str, int, float} or raw[field] is None)
        }

    def safe_accounts(model_id: str, capabilities: list[str]) -> list[dict[str, Any]]:
        route = model_catalog_service.route_for_model(model_id)
        # Older in-process compatibility doubles expose type support only. Do
        # not manufacture individual-account visibility from that weaker fact.
        if getattr(route, "account_identities", None) is None:
            return []
        return model_catalog_service.public_accounts_for_model(model_id, capabilities)

    for raw in result["data"]:
        if not isinstance(raw, dict):
            continue
        raw_model_id = raw.get("id")
        if not isinstance(raw_model_id, str):
            continue
        model_id = raw_model_id.strip()
        if model_id == "gpt-image-2":
            capabilities = ["image_generation", "image_edit"]
            data.append({
                **public_model_fields(raw),
                "capabilities": capabilities,
                "input_limits": dict(IMAGE_OUTPUT_LIMITS),
                "accounts": safe_accounts(model_id, capabilities),
            })
        elif is_public_text_model(model_id):
            has_text_model = True
            capabilities = ["text", "image_input"]
            item = {
                **public_model_fields(raw),
                "capabilities": capabilities,
                "input_limits": dict(CHAT_INPUT_LIMITS),
                "accounts": safe_accounts(model_id, capabilities),
            }
            efforts = public_reasoning_efforts(model_id)
            if efforts:
                item["reasoning_efforts"] = efforts
            data.append(item)
    # A partial paid catalog still has useful, per-account historical and
    # healthy-model rows.  Do not turn those models into an empty directory;
    # only fail when no known text model can be truthfully advertised.
    if catalog_unknown and not has_text_model:
        raise PublicChatContractError(
            "MODEL_DISCOVERY_UNAVAILABLE",
            "model capability discovery is temporarily unavailable",
        )
    result: dict[str, Any] = {"object": "list", "data": data}
    if catalog_unknown:
        result["model_catalog"] = {"state": "partial"}
    return result


def project_public_chat_receipt(receipt: object) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise PublicChatContractError("CHAT_RECEIPT_INVALID", "chat request receipt is invalid")
    result: dict[str, Any] = {
        "request_id": str(receipt.get("request_id") or ""),
        "route": "chat",
        "model": str(receipt.get("model") or ""),
        "status": str(receipt.get("status") or "unknown"),
    }
    conversation = receipt.get("conversation")
    if isinstance(conversation, dict) and conversation.get("protocol") == "sequential-v1":
        result["conversation"] = {key: conversation.get(key) for key in (
            "client_conversation_id", "previous_request_id", "protocol",
        )}
    content = receipt.get("content")
    if isinstance(content, str) and content:
        result["content"] = content
    error_code = receipt.get("error_code")
    if isinstance(error_code, str) and error_code:
        result["error_code"] = error_code
    for field in ("waiting", "rate_limit"):
        if isinstance(receipt.get(field), dict):
            result[field] = receipt[field]
    terminal_empty = receipt.get("terminal_empty")
    if (result["status"] == "unknown" and isinstance(terminal_empty, dict)
            and terminal_empty.get("verified") is True
            and terminal_empty.get("original_request_id") == result["request_id"]
            and terminal_empty.get("same_conversation_continuation") is True
            and type(terminal_empty.get("observed_at")) in {int, float}
            and math.isfinite(terminal_empty["observed_at"]) and terminal_empty["observed_at"] > 0):
        result["terminal_empty"] = {
            "verified": True, "original_request_id": result["request_id"],
            "observed_at": terminal_empty["observed_at"],
            "same_conversation_continuation": True,
        }
    if isinstance(receipt.get("correction_of_request_id"), str):
        result["correction_of_request_id"] = receipt["correction_of_request_id"]
    non_text_result = receipt.get("result")
    if (result["status"] == "failed" and error_code == "CHAT_RESPONSE_NOT_TEXT"
            and isinstance(non_text_result, dict) and non_text_result.get("type") == "non_text"
            and non_text_result.get("artifact_type") == "image"
            and type(non_text_result.get("artifact_count")) is int and non_text_result["artifact_count"] > 0):
        # Original account/conversation/tool/asset references remain private
        # in the owner-scoped durable receipt, never exposed as download URLs.
        result["result"] = {
            "type": "non_text", "artifact_type": "image", "artifact_count": non_text_result["artifact_count"],
        }
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
        "recovery_scan_progress": "scan",
        "recovery_last_read_error": "last_read_error",
    }
    recovery = {
        public_name: receipt[internal_name]
        for internal_name, public_name in recovery_mapping.items()
        if receipt.get(internal_name) is not None
    }
    if recovery:
        result["recovery"] = recovery
    return result
