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
    has_known_model = False

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
            accounts = safe_accounts(model_id, capabilities)
            has_known_model = has_known_model or bool(accounts)
            data.append({
                **public_model_fields(raw),
                "capabilities": capabilities,
                "input_limits": dict(IMAGE_OUTPUT_LIMITS),
                "accounts": accounts,
            })
        elif is_public_text_model(model_id):
            has_known_model = True
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
    # only fail when no observed image or known text model can be advertised.
    if catalog_unknown and not has_known_model:
        raise PublicChatContractError(
            "MODEL_DISCOVERY_UNAVAILABLE",
            "model capability discovery is temporarily unavailable",
        )
    result: dict[str, Any] = {"object": "list", "data": data}
    if catalog_unknown:
        result["model_catalog"] = {"state": "partial"}
    return result


EXECUTION_ENUMS = {
    "phase": {"queued", "preparing", "sending", "receiving", "recovering", "stalled", "completed", "failed", "unknown"},
    "send_state": {"not_sent", "attempted", "response_received", "unknown"},
    "local_state": {"not_started", "active", "ended", "unknown"},
    "wait_state": {"waiting", "ended", "completed", "unknown"},
    "upstream_outcome": {"not_sent", "unknown", "completed", "rejected"},
    "upstream_status": {"in_progress", "running", "pending", "queued", "finished_successfully"},
    "stream_end": {"done", "eof", "hard_timeout", "transport_error", "consumer_closed"},
}
EXECUTION_TIMES = ("accepted_at", "sent_at", "response_received_at", "first_stream_event_at",
                   "local_finished_at", "observation_started_at", "last_checked_at", "last_progress_at", "wait_ended_at", "upstream_updated_at", "stream_ended_at")
EXECUTION_RESOURCES = {
    "local_worker": {"active", "idle", "unknown"},
    "account_turn": {"held", "released", "unattributed", "unknown"},
    "conversation": {"protected", "ready", "unknown"},
}


def safe_public_execution(value: object) -> dict[str, Any]:
    """No raw timelines, private account/message IDs or upstream payloads."""
    if not isinstance(value, dict):
        return {}
    result = {key: value[key] for key, allowed in EXECUTION_ENUMS.items()
              if isinstance(value.get(key), str) and value[key] in allowed}
    for key in EXECUTION_TIMES:
        number = value.get(key)
        if type(number) in (int, float) and math.isfinite(number) and number > 0:
            result[key] = number
    for key in ("unchanged_reads", "sse_data_count", "sse_parse_errors"):
        count = value.get(key)
        if type(count) is int and 0 <= count <= 2147483647:
            result[key] = count
    if type(value.get("sse_error_event")) is bool:
        result["sse_error_event"] = value["sse_error_event"]
    resources = value.get("resources")
    if isinstance(resources, dict):
        result["resources"] = {key: resources[key] for key, allowed in EXECUTION_RESOURCES.items()
                               if isinstance(resources.get(key), str) and resources[key] in allowed}
    return result


def project_text_execution(receipt: dict[str, Any]) -> dict[str, Any]:
    from services.pool_admission import original_turn_ended, unknown_text_result

    status = receipt.get("status")
    stages = {}
    timeline = receipt.get("_execution_timeline")
    for item in timeline if isinstance(timeline, list) else []:
        if isinstance(item, dict) and isinstance(item.get("stage"), str):
            at = item.get("at")
            if type(at) in (int, float) and math.isfinite(at) and at > 0:
                stages[item["stage"]] = at
    sent, response = stages.get("send_call_started"), stages.get("response_headers_received")
    send_state = ("response_received" if response else "attempted" if sent or receipt.get("_submission_started") is True
                  else "not_sent" if receipt.get("upstream_outcome") == "not_sent"
                  or receipt.get("_submission_started") is False and (
                      status in {"queued", "not_started"} or status == "running" and receipt.get("_claim_id"))
                  else "unknown")
    local_end = stages.get("task_finished") or receipt.get("finished_at")
    local = ("active" if receipt.get("_executing") is True and status == "running"
             else "ended" if local_end else "not_started" if status in {"queued", "not_started"} else "unknown")
    unresolved = status == "unknown" or (status == "failed" and receipt.get("upstream_outcome") == "unknown")
    released_wait = (status == "failed" and receipt.get("error_code") == "RESULT_UNRECOVERABLE"
                     and receipt.get("upstream_outcome") == "unknown"
                     and type(receipt.get("_execution_wait_ended_at")) in (int, float)
                     and math.isfinite(receipt["_execution_wait_ended_at"]))
    wait_end = receipt.get("_result_wait_ended_at") or receipt.get("_execution_wait_ended_at")
    progress = receipt.get("_result_last_progress_at")
    stalled = (type(wait_end) in (int, float) and math.isfinite(wait_end)
               and (type(progress) not in (int, float) or progress <= wait_end))
    outcome = ("completed" if status == "succeeded" or receipt.get("upstream_outcome") == "completed"
               else "rejected" if original_turn_ended("text", receipt) and not receipt.get("_upstream_terminal")
               else "not_sent" if send_state == "not_sent" and not unresolved else "unknown")
    phase = ("completed" if status == "succeeded" else "stalled" if unresolved and stalled
             else "recovering" if unresolved else "failed" if status == "failed"
             else "queued" if status in {"queued", "not_started"}
             else "receiving" if response else "sending" if send_state == "attempted"
             else "preparing" if status == "running" else "unknown")
    turn_active = (status == "running" or unknown_text_result(receipt)) and not original_turn_ended("text", receipt) and not released_wait
    accounted = turn_active and (unknown_text_result(receipt) or receipt.get("_turn_reserved", True))
    resources = {
        "local_worker": "active" if local == "active" else "idle" if local in {"ended", "not_started"} else "unknown",
        "account_turn": ("held" if receipt.get("provider_account_identity") or receipt.get("_account_resource")
                         else "unattributed") if accounted else "released",
        "conversation": "protected" if unresolved or status == "running" else "ready" if status == "succeeded" else "unknown",
    }
    observation = receipt.get("_original_result_observation")
    observation = observation if isinstance(observation, dict) else {}
    nodes = observation.get("nodes")
    latest = nodes[-1] if isinstance(nodes, list) and nodes and isinstance(nodes[-1], dict) else {}
    stream = next((item for item in reversed(receipt.get("_execution_timeline") or [])
                   if isinstance(item, dict) and item.get("stage") == "stream_finished"), {})
    return safe_public_execution({
        "phase": phase, "send_state": send_state, "local_state": local,
        "wait_state": "completed" if status == "succeeded" else "ended" if wait_end or status == "failed"
                      else "waiting" if status in {"queued", "running", "unknown"} else "unknown",
        "upstream_outcome": outcome, "upstream_status": latest.get("status"),
        "accepted_at": receipt.get("created_at"), "sent_at": sent, "response_received_at": response,
        # The first SSE line is transport evidence, never a reasoning/token signal.
        "first_stream_event_at": stages.get("first_output"), "local_finished_at": local_end,
        "observation_started_at": receipt.get("_result_observation_started_at"),
        "last_checked_at": receipt.get("_result_last_checked_at"), "last_progress_at": progress,
        "wait_ended_at": wait_end, "upstream_updated_at": observation.get("upstream_updated_at"),
        "unchanged_reads": receipt.get("_result_no_progress_reads"), "resources": resources,
        "stream_ended_at": stream.get("at"),
        **{key: stream[key] for key in ("stream_end", "sse_data_count", "sse_parse_errors", "sse_error_event") if key in stream},
    })


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
    if isinstance(receipt.get("scheduling"), dict):
        try:
            from services.workflow_scheduling import normalize_scheduling
            result["scheduling"] = normalize_scheduling(receipt["scheduling"])
        except ValueError:
            # Corrupt legacy receipt metadata must not become a public contract.
            pass
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
    execution = safe_public_execution(receipt.get("execution"))
    if execution:
        result["execution"] = execution
    return result
