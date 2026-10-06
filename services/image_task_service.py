from __future__ import annotations

from services.image_thread import (PROTOCOL as IMAGE_THREAD_PROTOCOL, ImageThreadError, input_fields, accept_thread, public_thread, finished_parent, saved_image_bytes, source_fingerprint)

import json
import math
import hashlib
import os
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from services.config import DATA_DIR, config
from services.task_store import TaskStore, recovery_control, pending_image_result_ids
from services.pool_admission import image_result_stage
from contextlib import contextmanager
from services.request_context import current_request, AdmissionLost
from utils.log import logger
from services.content_filter import request_text
from services.log_service import LOG_TYPE_CALL, log_service
from services.protocol import openai_v1_image_edit, openai_v1_image_generations
from utils.helper import is_codex_image_model

TASK_STATUS_QUEUED = "queued"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_SUCCESS = "success"
TASK_STATUS_ERROR = "error"
TERMINAL_STATUSES = {TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}
UNFINISHED_STATUSES = {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING}
UNRECOVERABLE_MIN_AGE_SECONDS = 15.0 * 60.0
UNRECOVERABLE_QUALIFIED_READS = 3
IMAGE_ACTIVE_BUDGET_SECONDS = 300.0


class UnrecoverableRead(RuntimeError):
    def __init__(self, message: str, *, requires_new_conversation: bool) -> None:
        super().__init__(message)
        self.requires_new_conversation = requires_new_conversation


class ConversationImageAdoptionError(ValueError):
    """A caller-safe refusal to adopt a manually completed conversation image."""


def _holds_upstream_slot(task: dict[str, Any]) -> bool:
    if task.get("_attempt_finished_at") and not task.get("_executing"):
        return False
    if "upstream_unfinished" in task:
        return task["upstream_unfinished"] is True
    # Old receipts did not distinguish local execution from upstream work.
    return bool(task.get("provider_account_identity")) and (
        task.get("status") == TASK_STATUS_RUNNING
        or task.get("error_code") == "CONVERSATION_OUTCOME_UNKNOWN"
    )


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _timestamp(value: object) -> float:
    if not isinstance(value, str) or not value.strip():
        return 0.0
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:26], fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _clean(value: object, default: str = "") -> str:
    return str(value or default).strip()


def _owner_id(identity: dict[str, object]) -> str:
    return _clean(identity.get("id")) or "anonymous"


def _task_key(owner_id: str, task_id: str) -> str:
    return f"{owner_id}:{task_id}"


def _collect_image_urls(data: list[Any]) -> list[str]:
    urls: list[str] = []
    for item in data:
        if isinstance(item, dict):
            url = item.get("url")
            if isinstance(url, str) and url:
                urls.append(url)
    return urls


def _task_age_seconds(task: dict[str, Any], now: float | None = None) -> float:
    current = time.time() if now is None else now
    created_ts = task.get("created_ts")
    if isinstance(created_ts, (int, float)) and created_ts > 0:
        return max(0.0, current - float(created_ts))
    created_at = _timestamp(task.get("created_at"))
    return max(0.0, current - created_at) if created_at > 0 else 0.0


def _upstream_status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    message = str(exc)
    return 404 if "status=404" in message else None


def _retry_after_seconds(exc: BaseException) -> int | None:
    value = getattr(exc, "retry_after", None)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return int(value)
    return None


def _recovery_failure_code(exc: BaseException, phase: str, *, result_captured: bool = False) -> str:
    status = _upstream_status_code(exc)
    if status == 429:
        return "RECOVERY_RATE_LIMITED"
    if status in {401, 403}:
        return "RECOVERY_AUTH_REQUIRED"
    if isinstance(exc, ImageThreadError):
        return "RECOVERY_THREAD_UNCONFIRMED"
    from services.openai_backend_api import ImageActiveDeadlineExceeded, ImagePollTimeoutError
    if isinstance(exc, (TimeoutError, ImageActiveDeadlineExceeded, ImagePollTimeoutError)):
        return "RECOVERY_TIMED_OUT"
    exc_type = type(exc)
    type_name = f"{exc_type.__module__}.{exc_type.__name__}".lower()
    if (
        getattr(exc, "recovery_transport_failure", False) is True
        or
        (isinstance(exc, (ConnectionError, OSError)) and not isinstance(exc, TimeoutError))
        or any(marker in type_name for marker in ("curl_cffi", "connection", "network"))
    ):
        return "RECOVERY_TRANSPORT_FAILED"
    if phase == "download_image_result":
        return "RECOVERY_DOWNLOAD_FAILED"
    if result_captured:
        return "RECOVERY_RESULT_INCOMPLETE"
    return "RECOVERY_READ_FAILED"


_IMAGE_THREAD_FAILURE_CODES = frozenset({
    "IMAGE_THREAD_UPSTREAM_CHANGED", "IMAGE_THREAD_TURN_UNCONFIRMED",
})
_BINDING_FAILURE_REASONS = frozenset({
    "binding_id_required", "bound_account_missing", "bound_image_capability_unavailable",
    "image_capacity_disabled", "image_capacity_auth_required", "image_capacity_limited",
    "image_capacity_read_failed", "image_capacity_stale",
})


def _failure_details(exc: BaseException, phase: str) -> dict[str, Any]:
    # Persist diagnosis without exception text, response bodies, URLs or headers.
    details = {
        "phase": phase,
        "type": type(exc).__name__[:80],
        "status_code": _upstream_status_code(exc),
        "at": time.time(),
    }
    if isinstance(exc, ImageThreadError) and isinstance(exc.code, str) and exc.code in _IMAGE_THREAD_FAILURE_CODES:
        details["code"] = exc.code
    reason = getattr(exc, "binding_reason", None)
    if (getattr(exc, "code", None) == "CONVERSATION_BINDING_UNAVAILABLE"
            and getattr(exc, "upstream_submitted", None) is False
            and isinstance(reason, str) and reason in _BINDING_FAILURE_REASONS):
        details["binding_reason"] = reason
    return details


def _public_failure_details(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    phases = {
        "collect_image_result", "validate_image_result", "read_image_request",
        "resolve_image_result", "download_image_result", "confirm_image_turn", "save_image_result",
        "handler_operation", "select_image_account", "upload_image_reference", "bootstrap",
        "chat_requirements", "prepare_conversation", "start_image_generation",
        "stream_image_generation", "receive_image_result",
    }
    phase = value.get("phase")
    name = value.get("type")
    status = value.get("status_code")
    at = value.get("at")
    details = {
        "phase": phase if isinstance(phase, str) and phase in phases else "unknown",
        "type": name if isinstance(name, str) and len(name) <= 80 and name.isidentifier() else "Error",
        "status_code": status if type(status) is int and 100 <= status <= 599 else None,
        "at": at if type(at) in (int, float) and math.isfinite(at) and at >= 0 else None,
    }
    if name == "ImageThreadError" and isinstance(value.get("code"), str) and value["code"] in _IMAGE_THREAD_FAILURE_CODES:
        details["code"] = value["code"]
    reason = value.get("binding_reason")
    if isinstance(reason, str) and reason in _BINDING_FAILURE_REASONS:
        details["binding_reason"] = reason
    return details


def _safe_recovery_error(code: str, phase: str) -> str:
    if phase == "download_image_result":
        reason = {
            "RECOVERY_RATE_LIMITED": "the provider is rate limited",
            "RECOVERY_AUTH_REQUIRED": "the bound account connection must be restored",
            "RECOVERY_TRANSPORT_FAILED": "the provider could not be reached",
            "RECOVERY_READ_FAILED": "the provider response could not be read",
            "RECOVERY_THREAD_UNCONFIRMED": "the original image turn could not be confirmed",
            "RECOVERY_TIMED_OUT": "result retrieval or confirmation timed out",
            "RECOVERY_RESULT_INCOMPLETE": "result retrieval or confirmation did not complete",
        }.get(code, "the result download did not complete")
        return (
            f"Generated image is preserved; {reason}. "
            "Result download will resume without generating again."
        )
    return {
        "RECOVERY_RATE_LIMITED": "Image result lookup is rate limited; retry after the provider cooldown.",
        "RECOVERY_AUTH_REQUIRED": "Image result lookup requires the bound account connection to be restored.",
        "RECOVERY_TRANSPORT_FAILED": "Image result lookup could not reach the provider; retry when connectivity returns.",
        "RECOVERY_READ_FAILED": "Image result lookup returned an unreadable response; retry the same request lookup.",
        "RECOVERY_TIMED_OUT": "Image result lookup timed out; retry the same request lookup.",
    }.get(code, "Image result lookup did not complete; retry the same request lookup.")


def _branch_read_state(document: object, request_message_id: str) -> str:
    """Classify an exact request branch without treating active work as absent."""
    if not isinstance(document, dict) or not isinstance(document.get("mapping"), dict):
        return "unknown"
    mapping = document["mapping"]
    if request_message_id not in mapping:
        return "unattributable"
    children: dict[str, list[str]] = {}
    for node_id, node in mapping.items():
        if isinstance(node, dict) and node.get("parent"):
            children.setdefault(str(node["parent"]), []).append(str(node_id))
    pending = list(children.get(request_message_id, []))
    visited: set[str] = set()
    terminal = False
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        node = mapping.get(node_id) or {}
        message = node.get("message") if isinstance(node, dict) else {}
        author = message.get("author") if isinstance(message, dict) else {}
        role = _clean(author.get("role")).lower() if isinstance(author, dict) else ""
        if role == "user":
            continue
        status = _clean(message.get("status")).lower() if isinstance(message, dict) else ""
        if status in {"in_progress", "running", "pending", "queued"}:
            return "running"
        if role == "assistant" and status == "finished_successfully" and message.get("end_turn") is True:
            terminal = True
        pending.extend(children.get(node_id, []))
    return "terminal_without_result" if terminal else "no_result"


def _document_current_message_active(document: object) -> bool:
    mapping = document.get("mapping") if isinstance(document, dict) else None
    if not isinstance(mapping, dict):
        return True
    current_node = _clean(document.get("current_node"))
    node = mapping.get(current_node) if current_node else None
    message = node.get("message") if isinstance(node, dict) else None
    status = _clean(message.get("status")).lower() if isinstance(message, dict) else ""
    return status in {"in_progress", "running", "pending", "queued"}


def _backend_tasks_may_be_active(tasks: object) -> bool:
    if not isinstance(tasks, list):
        return True
    for task in tasks:
        if not isinstance(task, dict):
            return True
        status = _clean(
            task.get("status") or task.get("state") or task.get("task_status")
        ).lower()
        if status not in {
            "completed", "complete", "finished", "done", "succeeded", "success",
            "failed", "cancelled", "canceled",
        }:
            return True
    return False


def _latest_completed_manual_image_turn(
    document: object,
    original_request_message_id: str,
    original_task_created_ts: float,
    extract_records: Callable[[dict[str, Any], str], list[dict[str, Any]]],
    *,
    require_original_on_current_branch: bool = False,
) -> tuple[str, dict[str, Any], str]:
    """Select the latest completed manual image turn on the authoritative branch.

    The original request must either be on the current-node parent chain or
    remain in the full mapping with its own parent on the current branch. The
    task receipt's parent may point at a later failure response and is not used
    as a pre-send anchor. A newer user turn always wins; if that turn is active
    or has no completed image, an older image is never substituted.
    """
    if not isinstance(document, dict) or not isinstance(document.get("mapping"), dict):
        raise ConversationImageAdoptionError("conversation branch is unavailable")
    mapping = document["mapping"]
    current_node = _clean(document.get("current_node"))
    if not current_node or not original_request_message_id:
        raise ConversationImageAdoptionError("conversation branch is unavailable")

    reversed_path: list[str] = []
    node_id = current_node
    while node_id:
        if node_id in reversed_path:
            raise ConversationImageAdoptionError("conversation branch is invalid")
        node = mapping.get(node_id)
        if not isinstance(node, dict):
            raise ConversationImageAdoptionError("conversation branch is invalid")
        reversed_path.append(node_id)
        node_id = _clean(node.get("parent"))
    path = list(reversed(reversed_path))
    original_node = mapping.get(original_request_message_id) or {}
    original_message = original_node.get("message") if isinstance(original_node, dict) else None
    original_author = original_message.get("author") if isinstance(original_message, dict) else None
    original_request_verified = (
        isinstance(original_message, dict)
        and _clean(original_message.get("id")) == original_request_message_id
        and isinstance(original_author, dict)
        and _clean(original_author.get("role")).lower() == "user"
    )
    if not original_request_verified:
        raise ConversationImageAdoptionError("original request is not a verified user message")
    if require_original_on_current_branch and original_request_message_id not in path:
        raise ConversationImageAdoptionError("original request is not on the current conversation branch")
    if original_request_message_id in path:
        original_index = path.index(original_request_message_id)
    else:
        # A user may edit a later prompt in ChatGPT, moving current_node onto a
        # sibling branch. Read the verified original user node's own parent;
        # the receipt parent may already have advanced to a terminal response.
        branch_anchor = _clean(original_node.get("parent"))
        if (
            not branch_anchor
            or branch_anchor not in path
            or original_task_created_ts <= 0
        ):
            raise ConversationImageAdoptionError("original request is not linked to the current conversation branch")
        original_index = path.index(branch_anchor)

    manual_requests: list[tuple[int, str]] = []
    for index, candidate_id in enumerate(path[original_index + 1:], start=original_index + 1):
        node = mapping.get(candidate_id) or {}
        message = node.get("message") if isinstance(node, dict) else None
        author = message.get("author") if isinstance(message, dict) else None
        if isinstance(author, dict) and _clean(author.get("role")).lower() == "user":
            manual_requests.append((index, candidate_id))
    if not manual_requests:
        raise ConversationImageAdoptionError("no later manual request exists on the current conversation branch")

    request_index, request_message_id = manual_requests[-1]
    if original_request_message_id not in path:
        request_node = mapping.get(request_message_id) or {}
        request_message = request_node.get("message") if isinstance(request_node, dict) else None
        created = request_message.get("create_time") if isinstance(request_message, dict) else None
        if not isinstance(created, (int, float)) or float(created) <= original_task_created_ts:
            raise ConversationImageAdoptionError("latest manual request does not postdate the original task")
    segment = path[request_index + 1:]
    completed = False
    for candidate_id in segment:
        node = mapping.get(candidate_id) or {}
        message = node.get("message") if isinstance(node, dict) else None
        if not isinstance(message, dict):
            continue
        status = _clean(message.get("status")).lower()
        if status in {"in_progress", "running", "pending", "queued"}:
            raise ConversationImageAdoptionError("latest manual request is still active")
        author = message.get("author")
        if (
            isinstance(author, dict)
            and _clean(author.get("role")).lower() == "assistant"
            and status == "finished_successfully"
            and message.get("end_turn") is True
        ):
            completed = True
    if not completed:
        raise ConversationImageAdoptionError("latest manual request is not complete")

    records = extract_records(document, request_message_id)
    segment_ids = set(segment)
    records = [
        record for record in records
        if (
            isinstance(record, dict)
            and _clean(record.get("message_id")) in segment_ids
            and (record.get("file_ids") or record.get("sediment_ids"))
        )
    ]
    if not records:
        raise ConversationImageAdoptionError("latest manual request has no completed image")
    records.sort(key=lambda record: path.index(_clean(record.get("message_id"))))
    return request_message_id, records[-1], current_node


def _request_hash(mode: str, payload: dict[str, Any]) -> str:
    """Hash the immutable task request without persisting prompts or image bytes."""
    image_hashes = []
    for item in payload.get("images") or []:
        if isinstance(item, tuple) and item and isinstance(item[0], bytes):
            image_hashes.append(hashlib.sha256(item[0]).hexdigest())
    mask_hashes = []
    for item in payload.get("mask") or []:
        if isinstance(item, tuple) and item and isinstance(item[0], bytes):
            mask_hashes.append(hashlib.sha256(item[0]).hexdigest())
    contract = {
        "mode": mode,
        "prompt_sha256": hashlib.sha256(_clean(payload.get("prompt")).encode("utf-8")).hexdigest(),
        "model": _clean(payload.get("model"), "gpt-image-2"),
        "size": _clean(payload.get("size")),
        "quality": _clean(payload.get("quality"), "auto"),
        "provider_binding_id": _clean(payload.get("provider_binding_id")),
        "provider_account_identity": _clean(payload.get("provider_account_identity")),
        "client_conversation_id": _clean(payload.get("client_conversation_id")),
        "conversation_id": _clean(payload.get("conversation_id")),
        "parent_message_id": _clean(payload.get("parent_message_id")),
        "retain_conversation": bool(payload.get("retain_conversation")),
        "image_sha256": image_hashes,
        "mask_sha256": mask_hashes,
    }
    # Keep old task fingerprints unchanged when the caller has no override.
    if _clean(payload.get("upstream_model")):
        contract["upstream_model"] = _clean(payload.get("upstream_model"))
    if "_requested_account_ref" in payload:
        contract["_requested_account_ref"] = payload["_requested_account_ref"]
    if "_scheduling" in payload:
        contract["_scheduling"] = payload["_scheduling"]
    contract.update(input_fields(payload))
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class AuthoritativeImageTaskFailure(RuntimeError):
    code = "NO_IMAGE_GENERATED"

    def __init__(self, message: str, *, terminal_parent_message_id: str = ""):
        super().__init__(message)
        self.terminal_parent_message_id = terminal_parent_message_id


_KNOWN_IMAGE_GENERATION_ERROR = (
    "We experienced an error when generating images. Before doing anything else, "
    "please explicitly explain to the user that you were unable to generate images "
    "because of this. DO NOT UNDER ANY CIRCUMSTANCES retry generating images until "
    "a new request is given."
)

_KNOWN_LOCALIZED_NO_IMAGE_GENERATED = (
    "无法生成图片：图片生成过程中发生了错误，因此这次未能完成生成。"
    "请重新发起一次新的图片生成请求后，我可以继续处理。"
)


def _normalized_text(parts: object) -> str:
    text = "\n".join(part for part in parts if isinstance(part, str)).strip() if isinstance(parts, list) else ""
    return " ".join(text.lower().split())


def _authoritative_image_failure(document: object, request_message_id: str) -> str:
    if not isinstance(document, dict):
        return ""
    current_node = _clean(document.get("current_node"))
    mapping = document.get("mapping")
    if not current_node or not request_message_id or not isinstance(mapping, dict):
        return ""
    reversed_path: list[str] = []
    message_id = current_node
    while message_id and message_id not in reversed_path:
        node = mapping.get(message_id)
        if not isinstance(node, dict):
            return ""
        reversed_path.append(message_id)
        if message_id == request_message_id:
            break
        message_id = _clean(node.get("parent"))
    if request_message_id not in reversed_path:
        return ""
    path = list(reversed(reversed_path))
    request_index = path.index(request_message_id)
    for message_id in path[request_index + 1:]:
        node = mapping.get(message_id) or {}
        message = node.get("message") if isinstance(node, dict) else {}
        author = message.get("author") if isinstance(message, dict) else {}
        if not isinstance(author, dict):
            author = {}
        if _clean(author.get("role")).lower() == "user":
            return ""
    node = mapping.get(current_node) if current_node and isinstance(mapping, dict) else None
    message = node.get("message") if isinstance(node, dict) else None
    if not isinstance(message, dict):
        return ""
    author = message.get("author")
    content = message.get("content")
    if (
        not isinstance(author, dict)
        or _clean(author.get("role")).lower() != "assistant"
        or _clean(message.get("status")) != "finished_successfully"
        or message.get("end_turn") is not True
        or not isinstance(content, dict)
        or _clean(content.get("content_type")) != "text"
    ):
        return ""
    parts = content.get("parts")
    text = "\n".join(part for part in parts if isinstance(part, str)).strip() if isinstance(parts, list) else ""
    normalized = _normalized_text(parts)
    explicit_failures = {
        "something went wrong while generating your image. sorry about that.",
        "something went wrong while generating your image.",
    }
    # A terminal failure is valid only for an unambiguous original branch
    # with no generated asset, regardless of the assistant's language.
    from services.openai_backend_api import OpenAIBackendAPI
    request = mapping.get(request_message_id)
    request_message = request.get("message") if isinstance(request, dict) else None
    request_author = request_message.get("author") if isinstance(request_message, dict) else None
    if (not isinstance(request_author, dict)
            or _clean(request_author.get("role")).lower() != "user"
            or (request_message.get("id") and request_message.get("id") != request_message_id)):
        return ""
    children: dict[str, list[str]] = {}
    for node_id, candidate in mapping.items():
        if isinstance(candidate, dict):
            parent_id = _clean(candidate.get("parent"))
            if parent_id:
                children.setdefault(parent_id, []).append(str(node_id))
    request_path = path[request_index:]
    for index, message_id in enumerate(request_path):
        branch_node = mapping.get(message_id) or {}
        branch_message = branch_node.get("message") if isinstance(branch_node, dict) else {}
        branch_author = branch_message.get("author") if isinstance(branch_message, dict) else {}
        if not isinstance(branch_author, dict):
            branch_author = {}
        if index and _clean(branch_author.get("role")).lower() not in {"assistant", "tool"}:
            return ""
        if index:
            output = {
                "content": branch_message.get("content"), "metadata": branch_message.get("metadata"),
            }
            files, sediments = OpenAIBackendAPI._extract_image_reference_ids(output)
            if files or sediments or OpenAIBackendAPI._has_image_asset_pointer(output):
                return ""
        following = children.get(message_id, [])
        if index == len(request_path) - 1:
            if following:
                return ""
        elif len(following) != 1 or following[0] != request_path[index + 1]:
            return ""
    if normalized in explicit_failures:
        return text
    # This exact localized receipt was observed after an unambiguous original
    # request branch that ended without a generation tool node or image asset.
    # Keep this an exact sentence match: generic localized error text remains
    # unknown unless the provider supplies the known tool error receipt below.
    if text == _KNOWN_LOCALIZED_NO_IMAGE_GENERATED:
        return text
    parent = mapping.get(_clean(node.get("parent"))) if isinstance(node, dict) else None
    tool_message = parent.get("message") if isinstance(parent, dict) else None
    tool_author = tool_message.get("author") if isinstance(tool_message, dict) else None
    tool_content = tool_message.get("content") if isinstance(tool_message, dict) else None
    tool_metadata = tool_message.get("metadata") if isinstance(tool_message, dict) else None
    if (
        isinstance(tool_author, dict)
        and _clean(tool_author.get("role")).lower() == "tool"
        and _clean(tool_message.get("status")) == "finished_successfully"
        and isinstance(tool_metadata, dict)
        and tool_metadata.get("is_error") is True
        and isinstance(tool_content, dict)
        and _clean(tool_content.get("content_type")) == "text"
        and _normalized_text(tool_content.get("parts")) == _normalized_text([_KNOWN_IMAGE_GENERATION_ERROR])
    ):
        return text
    return ""


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    item = {
        "recovery_control": recovery_control(task),
        "attempt_state": "succeeded" if task.get("status") == TASK_STATUS_SUCCESS else "ended" if task.get("_attempt_finished_at") else "active",
        "id": task.get("id"),
        "status": task.get("status"),
        "result_stage": image_result_stage(task),
        "mode": task.get("mode"),
        "model": task.get("model"),
        "size": task.get("size"),
        "quality": task.get("quality"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
    }
    if task.get("_scheduling") is not None:
        item["scheduling"] = task["_scheduling"]
    if isinstance(task.get("_completion"), dict):
        item["completion"] = {k: task["_completion"][k] for k in (
            "state", "reason", "replacement_id", "selected_id", "conversation_mode", "max_extra_requests")
            if k in task["_completion"]}
    if task.get("_requested_account_ref"):
        item["account_ref"] = task["_requested_account_ref"]
    if public_thread(task):
        item["image_thread"] = public_thread(task)
    if task.get("conversation_id"):
        item["image_session_id"] = task.get("conversation_id")
    if task.get("parent_message_id"):
        item["image_session_parent_id"] = task.get("parent_message_id")
    for field in (
        "provider_binding_id",
        "provider_account_identity",
        "client_conversation_id",
        "binding_status",
        "error_code",
        "waiting",
        "rate_limit",
        "upstream_model",
        "next_poll_at",
        "active_attempt_started_at",
        "active_attempt_deadline_at",
        "upstream_outcome",
        "recovery_retryable",
        "recovery_error_code",
        "recovery_phase",
        "last_recovery_failure",
        "adopted_source_request_message_id",
        "adopted_source_image_message_id",
        "adopted_from_error_code",
        "adopted_at",
    ):
        if task.get(field):
            item[field] = (_public_failure_details(task[field])
                           if field == "last_recovery_failure" else task.get(field))
    retry_after = task.get("recovery_retry_after_seconds")
    if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool) and retry_after >= 0:
        item["recovery_retry_after_seconds"] = retry_after
    if isinstance(task.get("upstream_submission_started"), bool):
        item["upstream_submission_started"] = task["upstream_submission_started"]
    if isinstance(task.get("upstream_unfinished"), bool):
        item["upstream_unfinished"] = task["upstream_unfinished"]
    if task.get("recovery_no_result_reads"):
        item["recovery_no_result_reads"] = task.get("recovery_no_result_reads")
    if _clean(task.get("error_code")) == "RESULT_UNRECOVERABLE":
        item["upstream_unfinished"] = False
        item["recovery_requires_new_conversation"] = bool(
            task.get("recovery_requires_new_conversation")
        )
    if (
        task.get("status") == TASK_STATUS_ERROR
        and _clean(task.get("error_code")) == "CONVERSATION_OUTCOME_UNKNOWN"
        and not _clean(task.get("request_message_id"))
    ):
        # Legacy receipts may predate persistence of the submitted user-message
        # boundary. Keep the upstream outcome UNKNOWN, but tell callers that
        # automated polling cannot safely continue until that exact boundary is
        # recovered from independent evidence.
        item["recovery_status"] = "request_message_id_required"
    if task.get("data") is not None:
        item["data"] = task.get("data")
    if task.get("usage") is not None:
        item["usage"] = task.get("usage")
    if task.get("error"):
        item["error"] = task.get("error")
    if task.get("adopted_from_error"):
        item["adopted_from_error"] = task.get("adopted_from_error")
    if task.get("progress"):
        item["progress"] = task.get("progress")
    deadline = task.get("active_attempt_deadline_at")
    if isinstance(deadline, (int, float)) and not isinstance(deadline, bool) and deadline > 0:
        item["active_budget_remaining_secs"] = round(max(0.0, float(deadline) - time.time()), 1)
    if task.get("duration_ms") is not None:
        item["duration_ms"] = task.get("duration_ms")
    if task.get("status") in (TASK_STATUS_RUNNING, TASK_STATUS_QUEUED):
        if task.get("status") == TASK_STATUS_RUNNING:
            # RUNNING 状态仅在 started_ts 被设置后（image_stream_resolve_start）才计时
            base_ts = task.get("started_ts")
        else:
            # QUEUED 状态从 created_ts 开始计时（排队等待中）
            base_ts = task.get("created_ts") or task.get("updated_ts")
        if base_ts:
            item["elapsed_secs"] = round(time.time() - base_ts, 1)
    return item


class ImageTaskService:
    def __init__(
        self,
        path: Path,
        *,
        generation_handler: Callable[[dict[str, Any]], dict[str, Any]] = openai_v1_image_generations.handle,
        edit_handler: Callable[[dict[str, Any]], dict[str, Any]] = openai_v1_image_edit.handle,
        retention_days_getter: Callable[[], int] | None = None,
        admission=None,
        store: TaskStore | None = None,
    ):
        self.path = path
        self.store = store or TaskStore(path.parent / "text_tasks.sqlite3")
        self.admission = admission
        self._transaction_local = threading.local()
        self.generation_handler = generation_handler
        self.edit_handler = edit_handler
        self.retention_days_getter = retention_days_getter or (lambda: config.image_retention_days)
        self._lock = threading.RLock()
        self._slot_condition = threading.Condition(self._lock)
        self._tasks: dict[str, dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction():
            # Import once into the existing receipt database. The legacy file
            # stays in place for rollback; it is no longer a second writer.
            imported = self.store.runtime(self._transaction_local.db, "image_json_imported", False)
            if not imported:
                self._tasks = self._load_locked()
                self._save_locked()
                self.store.set_runtime(self._transaction_local.db, "image_json_imported", True)
            changed = self._recover_unfinished_locked()
            changed = self._cleanup_locked() or changed
            if changed:
                self._save_locked()

    @contextmanager
    def _transaction(self, *, task_key: str | None = None):
        with self._lock:
            if getattr(self._transaction_local, "db", None) is not None:
                yield self._transaction_local.db
                return
            with self.store.transaction() as db:
                self._transaction_local.db = db
                self._transaction_local.task_key = task_key
                try:
                    if task_key is None:
                        self._tasks = {key: json.loads(raw) for key, raw in db.execute("SELECT task_key,receipt FROM image_requests")}
                    else:
                        # A progress update needs only its current durable
                        # receipt, not every historical image's base64 bytes.
                        row = db.execute("SELECT receipt FROM image_requests WHERE task_key=?", (task_key,)).fetchone()
                        if row is None:
                            self._tasks.pop(task_key, None)
                        else:
                            self._tasks[task_key] = json.loads(row[0])
                    yield db
                finally:
                    self._transaction_local.db = None
                    self._transaction_local.task_key = None

    def resource_occupancy(self) -> dict:
        """Internal aggregate only: keep unfinished original receipts after restart."""
        fields = ("_attempt_finished_at", "_executing", "upstream_unfinished",
                  "provider_account_identity", "status", "error_code")
        expression = "json_object(" + ",".join(
            "'" + key + "',json_extract(receipt,'$." + key + "')" for key in fields) + ")"
        held, unattributed = {}, 0
        with self.store.connect() as db:
            for raw, unfinished_type in db.execute(
                    "SELECT " + expression + ",json_type(receipt,'$.upstream_unfinished') FROM image_requests"):
                task = json.loads(raw)
                # JSON booleans are typed evidence; numeric 1 is not true.
                if unfinished_type is None:
                    task.pop("upstream_unfinished", None)
                else:
                    task["upstream_unfinished"] = unfinished_type == "true"
                if not _holds_upstream_slot(task):
                    continue
                identity = str(task.get("provider_account_identity") or "")
                if identity:
                    held[identity] = held.get(identity, 0) + 1
                else:
                    unattributed += 1
        return {"by_account": held, "unattributed": unattributed}

    def submit_generation(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        quality: str = "auto",
        base_url: str = "",
        provider_binding_id: str = "",
        provider_account_identity: str = "",
        client_conversation_id: str = "",
        conversation_id: str = "",
        parent_message_id: str = "",
        retain_conversation: bool = False,
        upstream_model: str = "",
        image_thread_id: str = "",
        edit_source_task_id: str = "",
        edit_source_index: int = 0,
        account_ref: str | None = None,
        scheduling: dict | None = None,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            **({"_requested_account_ref": account_ref} if account_ref is not None else {}),
            **({"_scheduling": scheduling} if scheduling is not None else {}),
            "model": model,
            "n": 1,
            "size": size,
            "quality": quality,
            "response_format": "url",
            "base_url": base_url,
            "provider_binding_id": provider_binding_id,
            "provider_account_identity": provider_account_identity,
            "client_conversation_id": client_conversation_id,
            "conversation_id": conversation_id,
            "parent_message_id": parent_message_id,
            "retain_conversation": retain_conversation,
            "upstream_model": upstream_model,
            **({"image_thread_id": image_thread_id, "edit_source_task_id": edit_source_task_id,
                "edit_source_index": edit_source_index} if image_thread_id or edit_source_task_id or edit_source_index else {}),
        }
        return self._submit(identity, client_task_id=client_task_id, mode="generate", payload=payload)

    def submit_edit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        quality: str = "auto",
        base_url: str = "",
        images: list[tuple[bytes, str, str]] | None = None,
        masks: list[tuple[bytes, str, str]] | None = None,
        provider_binding_id: str = "",
        provider_account_identity: str = "",
        client_conversation_id: str = "",
        conversation_id: str = "",
        parent_message_id: str = "",
        retain_conversation: bool = False,
        upstream_model: str = "",
        image_thread_id: str = "",
        edit_source_task_id: str = "",
        edit_source_index: int = 0,
        account_ref: str | None = None,
        scheduling: dict | None = None,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            **({"_requested_account_ref": account_ref} if account_ref is not None else {}),
            **({"_scheduling": scheduling} if scheduling is not None else {}),
            "images": images or [],
            "mask": masks or [],
            "model": model,
            "n": 1,
            "size": size,
            "quality": quality,
            "response_format": "url",
            "base_url": base_url,
            "provider_binding_id": provider_binding_id,
            "provider_account_identity": provider_account_identity,
            "client_conversation_id": client_conversation_id,
            "conversation_id": conversation_id,
            "parent_message_id": parent_message_id,
            "retain_conversation": retain_conversation,
            "upstream_model": upstream_model,
            **({"image_thread_id": image_thread_id, "edit_source_task_id": edit_source_task_id,
                "edit_source_index": edit_source_index} if image_thread_id or edit_source_task_id or edit_source_index else {}),
        }
        return self._submit(identity, client_task_id=client_task_id, mode="edit", payload=payload)

    def list_tasks(self, identity: dict[str, object], task_ids: list[str]) -> dict[str, Any]:
        owner = _owner_id(identity)
        requested_ids = [_clean(task_id) for task_id in task_ids if _clean(task_id)]
        if requested_ids and getattr(self._transaction_local, "db", None) is None:
            # Original-result and work-policy reads need only these receipts.
            # Do not acquire the writer lock or decode every saved image merely
            # to authorize/poll one ID. Expired receipts stay absent publicly;
            # startup and the existing full-list/write paths still clean them.
            cutoff = self._retention_cutoff()
            items, missing_ids = [], []
            with self.store.connect() as db:
                db.execute("BEGIN")
                for task_id in requested_ids:
                    task = self.store.read_receipt(db, "image", owner, task_id)
                    if (task is None or task.get("owner_id") != owner
                            or self._receipt_expired(task, cutoff)):
                        missing_ids.append(task_id)
                    else:
                        items.append(_public_task(task))
            return {"items": items, "missing_ids": missing_ids}
        with self._transaction():
            if self._cleanup_locked():
                self._save_locked()
            items = []
            missing_ids = []
            for task_id in requested_ids:
                task = self._tasks.get(_task_key(owner, task_id))
                if task is None:
                    missing_ids.append(task_id)
                else:
                    items.append(_public_task(task))
            if not requested_ids:
                items = [
                    _public_task(task)
                    for task in self._tasks.values()
                    if task.get("owner_id") == owner
                ]
                items.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
                missing_ids = []
            return {"items": items, "missing_ids": missing_ids}

    def failure_continuation(self, identity: dict[str, object], task_id: str) -> dict[str, str] | None:
        """Return one fresh, exact failure cursor without changing its receipt.

        This is deliberately narrower than recovery.  It reads only the named
        terminal policy failure, keeps the original receipt untouched, and
        refuses any branch with an asset, active turn, later user turn, or
        branch fork.
        """
        owner = _owner_id(identity)
        key = _task_key(owner, _clean(task_id))
        with self._transaction():
            task = self._tasks.get(key)
            if not task:
                return None
            live_claim = (
                bool(task.get("_claim_id"))
                and float(task.get("_claim_until") or 0) > time.time()
            )
            if (
                task.get("status") != TASK_STATUS_ERROR
                or _clean(task.get("error_code")).lower() not in {
                    "content_policy_violation", "no_image_generated",
                }
                or task.get("upstream_unfinished") is not False
                # A send initially records UNKNOWN. A later terminal policy
                # refusal can leave that historical marker behind; only the
                # exact fresh conversation read below may prove it ended.
                or _clean(task.get("upstream_outcome")).lower() == "generated"
                or _clean(task.get("binding_status")).lower() != "bound"
                or task.get("_recovery_paused") is True
                or task.get("_recovery_suppressed") is True
                or task.get("_executing") is True
                or task.get("recovery_claim_id")
                or live_claim
                or task.get("_turn_reserved") is True
                or task.get("waiting")
                or any(task.get(field) for field in (
                    "data", "result_file_ids", "result_sediment_ids",
                    "_pending_image_result_ids", "_pending_image_output",
                    "adopted_source_request_message_id", "adopted_source_image_message_id",
                ))
            ):
                return None
            from services.generation_completion import read_only_original_completion
            if task.get("_completion") and not read_only_original_completion(task["_completion"]):
                return None
            fields = (
                "provider_binding_id", "provider_account_identity", "client_conversation_id",
                "conversation_id", "request_message_id",
            )
            snapshot = {field: _clean(task.get(field)) for field in fields}
            for field in ("request_parent_message_id", "_image_thread_request_parent"):
                if _clean(task.get(field)):
                    snapshot[field] = _clean(task.get(field))
            if not all(snapshot.values()):
                return None

        from services.account_service import account_service
        from services.generation_completion import retry_cursor
        from services.openai_backend_api import OpenAIBackendAPI

        try:
            if account_service.get_bound_account_identity(snapshot["provider_binding_id"]) != snapshot["provider_account_identity"]:
                return None
            access_token = account_service.get_bound_text_access_token(
                snapshot["provider_binding_id"], model="auto"
            )
        except Exception as exc:
            status = _upstream_status_code(exc)
            if status == 429:
                error = ImageThreadError("RECOVERY_RATE_LIMITED", status=429)
                error.retry_after = _retry_after_seconds(exc)
                raise error from exc
            raise ImageThreadError(
                "RECOVERY_AUTH_REQUIRED" if status in {401, 403} else "RECOVERY_READ_FAILED",
                status=503,
            ) from exc

        backend = None
        try:
            with account_service.conversation_binding_lock(
                snapshot["provider_binding_id"], snapshot["client_conversation_id"]
            ):
                # Recheck authority after taking the same conversation lock as
                # submission; the account may have rotated while we waited.
                if account_service.get_bound_account_identity(snapshot["provider_binding_id"]) != snapshot["provider_account_identity"]:
                    return None
                backend = OpenAIBackendAPI(access_token=access_token)
                document = backend._get_conversation(snapshot["conversation_id"])
                if (
                    not isinstance(document, dict)
                    or _clean(document.get("conversation_id")) not in {"", snapshot["conversation_id"]}
                ):
                    return None
                proof = retry_cursor(document, snapshot, kind="image")
                if not proof:
                    return None
                mapping = document.get("mapping")
                head = _clean(document.get("current_node"))
                node = mapping.get(head) if isinstance(mapping, dict) else None
                message = node.get("message") if isinstance(node, dict) else None
                author = message.get("author") if isinstance(message, dict) else None
                if (
                    not isinstance(author, dict)
                    or _clean(author.get("role")).lower() != "assistant"
                    or _clean(message.get("status")) != "finished_successfully"
                    or message.get("end_turn") is not True
                    or _clean(message.get("id")) != head
                ):
                    return None
                return {
                    "source_task_id": _clean(task_id),
                    "source_request_message_id": snapshot["request_message_id"],
                    "provider_binding_id": snapshot["provider_binding_id"],
                    "provider_account_identity": snapshot["provider_account_identity"],
                    "client_conversation_id": snapshot["client_conversation_id"],
                    "conversation_id": snapshot["conversation_id"],
                    "parent_message_id": proof["retry_parent_message_id"],
                }
        except ImageThreadError:
            raise
        except Exception as exc:
            status = _upstream_status_code(exc)
            if status == 429:
                error = ImageThreadError("RECOVERY_RATE_LIMITED", status=429)
                error.retry_after = _retry_after_seconds(exc)
                raise error from exc
            raise ImageThreadError(
                "RECOVERY_AUTH_REQUIRED" if status in {401, 403} else "RECOVERY_READ_FAILED",
                status=503,
            ) from exc
        finally:
            if backend is not None:
                backend.close()

    def set_thread_archived(self, identity: dict[str, object], task_id: str, archived: bool) -> dict[str, Any]:
        """Change only the owner's latest, fully recovered image conversation."""
        from services.account_service import account_service
        from services.openai_backend_api import OpenAIBackendAPI

        owner = _owner_id(identity)
        key = _task_key(owner, _clean(task_id))
        with self._transaction(task_key=key):
            task = self._tasks.get(key)
            if not task or not task.get("_image_thread"):
                raise ImageThreadError("IMAGE_THREAD_NOT_FOUND", status=404)
            binding = _clean(task.get("provider_binding_id"))
            account = _clean(task.get("provider_account_identity"))
            client = _clean(task.get("client_conversation_id"))
        if not binding or not account or not client:
            raise ImageThreadError("IMAGE_THREAD_BINDING_UNAVAILABLE")
        if account_service.get_bound_account_identity(binding) != account:
            raise ImageThreadError("IMAGE_THREAD_BINDING_CHANGED")
        token = account_service.get_bound_text_access_token(binding, model="auto")
        with account_service.conversation_binding_lock(binding, client):
            # Generation on this conversation uses the same lock. Recheck after
            # obtaining it so an in-flight result or newer revision cannot be
            # hidden by an old approval event.
            with self._transaction(task_key=key) as db:
                task = self._tasks.get(key)
                thread = (task or {}).get("_image_thread") or {}
                thread_id = thread.get("id")
                # Keep every member, including UNKNOWN and selected retry
                # history, but do not decode unrelated saved image payloads.
                # This fresh read remains inside the conversation binding lock.
                members = [task if member_key == key else json.loads(raw)
                    for member_key, raw in db.execute(
                        "SELECT task_key,receipt FROM image_requests "
                        "WHERE json_extract(receipt,'$.owner_id')=? "
                        "AND json_extract(receipt,'$._image_thread.id')=?",
                        (owner, thread_id))] if isinstance(thread_id, str) and thread_id else []
                from services.image_thread import selected_thread_result
                owned = {item["id"]: item for item in members}
                # The original failed attempt stays in history. An explicitly
                # selected terminal result resolves only that attempt, not any
                # newer task, active generation, or unrelated unknown branch.
                resolved = [selected_thread_result(item, owned) for item in members]
                task = selected_thread_result(task, owned)
                if (not thread or not members or max(members, key=lambda item: item.get("_sequence", 0)) is not task
                        or any(item.get("status") != TASK_STATUS_SUCCESS or not item.get("_image_thread_terminal") for item in resolved)):
                    raise ImageThreadError("IMAGE_THREAD_NOT_TERMINAL")
                conversation_id = _clean(task.get("conversation_id"))
                parent_id = _clean(task.get("parent_message_id"))
                request_id = _clean(task.get("adopted_source_request_message_id") or task.get("request_message_id"))
                if not conversation_id or not parent_id or not request_id:
                    raise ImageThreadError("IMAGE_THREAD_TURN_UNCONFIRMED")
            backend = OpenAIBackendAPI(access_token=token)
            try:
                result_ids = (task.get("result_file_ids") or []) + (task.get("result_sediment_ids") or [])
                def validate_terminal(document):
                    from services.image_thread import archive_parent
                    return archive_parent(document, conversation_id, request_id, parent_id, result_ids,
                        expected_parent=task.get("_image_thread_request_parent"),
                        predecessor_request_message_id=task.get("_image_thread_predecessor_message"),
                        predecessor_result_ids=task.get("_image_thread_predecessor_result_ids"),
                        expected_file_ids=task.get("result_file_ids") or [],
                        expected_sediment_ids=task.get("result_sediment_ids") or [])
                backend.set_conversation_archived(conversation_id, parent_id, archived,
                                                 validate_document=validate_terminal)
            finally:
                backend.close()
        return {"image_thread": public_thread(task), "archived": archived, "task_id": task_id}

    def archive_thread(self, identity: dict[str, object], task_id: str) -> dict[str, Any]:
        return self.set_thread_archived(identity, task_id, True)

    def restore_thread(self, identity: dict[str, object], task_id: str) -> dict[str, Any]:
        return self.set_thread_archived(identity, task_id, False)

    def _submit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        mode: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        from services.workflow_scheduling import normalize_scheduling
        if "_scheduling" in payload:
            payload = {**payload, "_scheduling": normalize_scheduling(payload["_scheduling"])}
        if payload.get("_scheduling") is not None and self.admission is None:
            raise ValueError("SCHEDULING_UNAVAILABLE")
        task_id = _clean(client_task_id)
        if not task_id:
            raise ValueError("client_task_id is required")
        owner = _owner_id(identity)
        key = _task_key(owner, task_id)
        now = _now_iso()
        should_start = False
        thread_fields = input_fields(payload)
        if thread_fields and self.admission is None:
            raise ImageThreadError("IMAGE_THREAD_SCHEDULER_UNAVAILABLE", status=503)
        # An accepted ID is a read: don't fsync another copy of its input.
        with self.store.connect() as db:
            existing = self.store.read_receipt(db, "image", owner, task_id)
        if existing is not None and not self._receipt_expired(existing, self._retention_cutoff()):
            if existing.get("request_hash") and existing["request_hash"] != _request_hash(mode, payload):
                raise ValueError("client_task_id already exists with a different immutable request")
            return _public_task(existing)
        source_snapshot = None
        source_bytes = None
        if thread_fields.get("edit_source_task_id"):
            # Existing IDs recover without rereading a potentially offline source.
            # Any remote-backed stored-image read is outside the SQLite write lock.
            with self.store.connect() as db:
                duplicate = self.store.read_receipt(db, "image", owner, task_id)
                source_snapshot = self.store.read_receipt(db, "image", owner, thread_fields["edit_source_task_id"])
                if source_snapshot and (source_snapshot.get("_completion") or {}).get("selected_id"):
                    from services.image_thread import selected_thread_result
                    selected_id = source_snapshot["_completion"]["selected_id"]
                    selected = self.store.read_receipt(db, "image", owner, selected_id)
                    source_snapshot = selected_thread_result(source_snapshot, {selected_id: selected})
            if duplicate is None:
                if source_snapshot is None:
                    raise ImageThreadError("IMAGE_THREAD_SOURCE_UNAVAILABLE")
                source_bytes = saved_image_bytes(source_snapshot)
        def read_source(source):
            if source_bytes is None or source_fingerprint(source) != source_fingerprint(source_snapshot):
                raise ImageThreadError("IMAGE_THREAD_SOURCE_CHANGED")
            return source_bytes
        # Persist/fsync potentially large edit input before taking the shared
        # receipt writer lock. Keep a referenced file even if commit reporting
        # fails; only a never-linked, freshly created input is disposable.
        input_ref = self.store.save_input({"payload": payload, "identity": {
            k: identity[k] for k in ("id", "name", "role", "external_image_client", "_trusted_source")
            if k in identity}, "mode": mode})
        input_linked = False
        try:
            with self._transaction(task_key=key) as db:
                cutoff = self._retention_cutoff()
                task = self._tasks.get(key)
                if task is not None and self._receipt_expired(task, cutoff):
                    db.execute("DELETE FROM image_requests WHERE task_key=?", (key,))
                    self._tasks.pop(key, None)
                    task = None
                if task is not None:
                    if task.get("request_hash") and task.get("request_hash") != _request_hash(mode, payload):
                        raise ValueError("client_task_id already exists with a different immutable request")
                    return _public_task(task)
                task = {
                    "id": task_id,
                    "owner_id": owner,
                    "retain_receipt": bool(identity.get("external_image_client")),
                    "status": TASK_STATUS_QUEUED,
                    "mode": mode,
                    "model": _clean(payload.get("model"), "gpt-image-2"),
                    "upstream_model": _clean(payload.get("upstream_model")),
                    "size": _clean(payload.get("size")),
                    "quality": _clean(payload.get("quality"), "auto"),
                    "base_url": _clean(payload.get("base_url")),
                    "created_at": now,
                    "updated_at": now,
                    "created_ts": time.time(),
                    "provider_binding_id": _clean(payload.get("provider_binding_id")),
                    "provider_account_identity": _clean(payload.get("provider_account_identity")),
                    "client_conversation_id": _clean(payload.get("client_conversation_id")),
                    "conversation_id": _clean(payload.get("conversation_id")),
                    "parent_message_id": _clean(payload.get("parent_message_id")),
                    "binding_status": "bound" if payload.get("provider_binding_id") else "unbound",
                    "request_hash": _request_hash(mode, payload),
                    "upstream_unfinished": False,
                    "admission_recorded": True,
                    "_input_ref": None,
                    "_sequence": self.store.next_sequence(self._transaction_local.db),
                    "_source": str(identity.get("_trusted_source") or "key:" + owner),
                    "_input_bytes": __import__("services.text_task_service", fromlist=["_retained_size"])._retained_size(payload),
                    "_submission_started": False,
                    "_turn_reserved": False,
                    "_execution_timeline": [{"stage": "accepted", "at": time.time()}],
                }
                if is_codex_image_model(task["model"]):
                    task["_route"] = "codex"
                    if any(payload.get(k) for k in ("provider_binding_id", "conversation_id", "parent_message_id", "retain_conversation")):
                        raise ImageThreadError("IMAGE_ACCOUNT_SELECTION_CONFLICT")
                if "_requested_account_ref" in payload:
                    from services.account_service import account_service
                    accounts = self.admission.accounts if self.admission is not None else account_service
                    requested_identity = accounts.resolve_image_account(payload["_requested_account_ref"])
                    if (task.get("provider_account_identity") and task["provider_account_identity"] != requested_identity):
                        raise ImageThreadError("IMAGE_ACCOUNT_SELECTION_CONFLICT")
                    if task.get("provider_binding_id"):
                        try:
                            bound_identity = accounts.get_bound_account_identity(task["provider_binding_id"])
                        except RuntimeError:
                            raise ImageThreadError("IMAGE_ACCOUNT_SELECTION_CONFLICT") from None
                        if bound_identity != requested_identity:
                            raise ImageThreadError("IMAGE_ACCOUNT_SELECTION_CONFLICT")
                    task.update(_requested_account_ref=payload["_requested_account_ref"],
                                _requested_account_identity=requested_identity)
                if thread_fields:
                    # Ordering requires this owner's thread history, not saved image
                    # bytes from every account. Only the edit source needs full data.
                    owned = {row["id"]: row for (raw,) in db.execute(
                        "SELECT json_remove(receipt,'$.data') FROM image_requests "
                        "WHERE json_extract(receipt,'$.owner_id')=?", (owner,))
                        if not self._receipt_expired(row := json.loads(raw), cutoff)}
                    source_id = thread_fields.get("edit_source_task_id")
                    if source_id and source_id in owned:
                        source = self.store.read_receipt(db, "image", owner, source_id)
                        owned[source_id] = source
                        selected_id = (source.get("_completion") or {}).get("selected_id")
                        if selected_id and selected_id in owned:
                            owned[selected_id] = self.store.read_receipt(db, "image", owner, selected_id)
                    accept_thread(task, owned.values(), payload, mode, output_reader=read_source)
                from services.generation_completion import attach_replacement
                attach_replacement(self.store, self._transaction_local.db, "image", owner, task_id, payload, task)
                if task.get("_same_session_retry_of"):
                    root_id = task["_same_session_retry_of"]
                    self._tasks[owner + ":" + root_id] = self.store.read_receipt(
                        self._transaction_local.db, "image", owner, root_id)
                if task.get("retain_receipt") and not task.get("_completion_of") and task.get("_route", "chat") == "chat":
                    task["_automatic_generation_recovery"] = True
                from services.workflow_scheduling import prepare_receipt
                from services.work_lifecycle import ensure_work
                prepare_receipt(task, payload.get("_scheduling"), task["_source"])
                ensure_work(self.store, self._transaction_local.db, "image", owner, task_id, task,
                            source=task["_source"], scheduling=task.get("_scheduling"))
                task["_input_ref"] = input_ref
                input_linked = True
                self._tasks[key] = task
                self._save_locked(task_key=key)
                should_start = True

        finally:
            if not input_linked:
                try:
                    (self.store.input_dir / input_ref).unlink(missing_ok=True)
                except OSError:
                    logger.warning({"event": "unused_image_input_cleanup_failed"})

        if should_start and self.admission is not None:
            self.admission.wake()
        elif should_start:
            thread = threading.Thread(
                target=self._run_task,
                args=(key, mode, payload, dict(identity), _clean(payload.get("model"), "gpt-image-2")),
                name=f"image-task-{task_id[:16]}",
                daemon=True,
            )
            thread.start()
        return _public_task(task)

    def _run_task(
        self,
        key: str,
        mode: str,
        payload: dict[str, Any],
        identity: dict[str, object],
        model: str,
    ) -> None:
        with self._transaction(task_key=key):
            original = self._tasks.get(key) or {}
            payload = {**payload, **{k: original[k] for k in ("_requested_account_ref", "_requested_account_identity") if original.get(k)}}
        if (identity.get("external_image_client") or payload.get("_requested_account_ref")) and not payload.get("_admission_claim"):
            # Allocate once for this newly persisted task, before any generation.
            # Duplicate submissions never enter this thread. Account selection
            # only performs readiness reads; a failure here is NOT submitted.
            from services.account_service import account_service
            token = ""
            try:
                capacities = {str(item.get("provider_account_identity") or ""): min(
                    max(1, int(config.image_account_concurrency)), account_service.image_account_capacity(item, model))
                    for item in account_service.list_accounts()}
                with self._transaction():
                    held = {}
                    for other in self._tasks.values():
                        identity_id = str(other.get("provider_account_identity") or "")
                        if identity_id and _holds_upstream_slot(other):
                            held[identity_id] = held.get(identity_id, 0) + 1
                    unavailable = {identity_id for identity_id, count in held.items()
                                   if count >= capacities.get(identity_id, max(1, int(config.image_account_concurrency)))}
                binding, account_identity, token = account_service.create_conversation_binding(
                    image_model=model, excluded_account_identities=unavailable,
                    **({"requested_account_identity": payload["_requested_account_identity"]} if payload.get("_requested_account_identity") else {}))
                payload = {**payload, "provider_binding_id": binding,
                           "provider_account_identity": account_identity,
                           "client_conversation_id": "image-task-" + uuid.uuid4().hex,
                           "retain_conversation": True}
                # Recheck after selection: another task can finish selection
                # while this selector waits for an account slot. Persist our
                # durable occupancy before releasing the temporary slot.
                selected = account_service.get_account(token) or {}
                capacity = min(max(1, int(config.image_account_concurrency)),
                               account_service.image_account_capacity(selected, model))
                with self._transaction():
                    occupied = sum(1 for other_key, other in self._tasks.items()
                                   if other_key != key and other.get("provider_account_identity") == account_identity
                                   and _holds_upstream_slot(other))
                    if occupied >= capacity:
                        raise RuntimeError("resource became occupied before submission")
                    self._update_task(key, provider_binding_id=binding,
                                      provider_account_identity=account_identity,
                                      client_conversation_id=payload["client_conversation_id"],
                                      binding_status="bound", upstream_unfinished=True)
            except Exception:
                self._update_task(key, status=TASK_STATUS_ERROR, error_code="IMAGE_RESOURCE_UNAVAILABLE",
                                  error="No available resource for the selected durable image route",
                                  upstream_unfinished=False)
                return
            finally:
                if token:
                    account_service.release_image_slot(token)

        # Persist account admission before calling the handler. A query timeout
        # or process restart must not make another upstream generation fit.
        account = _clean(payload.get("provider_account_identity"))
        if account and not identity.get("external_image_client") and not payload.get("_admission_claim"):
            while True:
                with self._transaction():
                    occupied = sum(1 for other_key, task in self._tasks.items()
                                   if other_key != key and task.get("provider_account_identity") == account
                                   and _holds_upstream_slot(task))
                    if occupied < max(1, int(config.image_account_concurrency)):
                        self._update_task(key, upstream_unfinished=True)
                        break
                # Legacy injected executors also wait without holding SQLite.
                with self._slot_condition:
                    self._slot_condition.wait(timeout=0.1)
        started = time.time()
        with self._transaction(task_key=key):
            current = self._tasks.get(key) or {}
            active_started_at = current.get("active_attempt_started_at")
            active_deadline_at = current.get("active_attempt_deadline_at")
        # Shared external routes can swap handlers and therefore do not infer
        # coverage from their normalized payload. Their receipt changes only
        # when the active bound-Chat attempt propagates an explicit marker.
        submission_boundary_covered = (
            not bool(identity.get("external_image_client"))
            and bool(_clean(payload.get("provider_binding_id")))
            and bool(_clean(payload.get("provider_account_identity")))
            and bool(_clean(payload.get("client_conversation_id")))
            and bool(payload.get("retain_conversation"))
            and not is_codex_image_model(model)
        )
        if submission_boundary_covered:
            self._update_task(key, upstream_submission_started=False)
        self._update_task(key, status=TASK_STATUS_RUNNING, error="")
        with self._transaction(task_key=key):
            task = self._tasks.get(key) or {}
            request_message_id = _clean(task.get("request_message_id"))
        if not request_message_id:
            request_message_id = str(uuid.uuid4())
            self._update_task(key, request_message_id=request_message_id)
        # Keep the last reported operation separate from the recovery action.
        # Capturing a file ID alone does not prove that downloading started.
        handler_failure_phase = "handler_operation"
        # 创建进度回调，每个步骤完成后更新任务状态
        def progress_callback(step: str) -> None:
            nonlocal handler_failure_phase
            handler_failure_phase = {
                "getting_account": "select_image_account",
                "uploading": "upload_image_reference",
                "bootstrapping": "bootstrap",
                "getting_token": "chat_requirements",
                "preparing_conversation": "prepare_conversation",
                "starting_generation": "start_image_generation",
                "generating": "stream_image_generation",
                "image_stream_resolve_start": "resolve_image_result",
                "receiving_image": "receive_image_result",
            }.get(step, "handler_operation")
            self._update_task(key, progress=step)
        progress_callback.request_message_id = request_message_id
        progress_callback.active_deadline_at = (
            float(active_deadline_at)
            if isinstance(active_deadline_at, (int, float))
            and not isinstance(active_deadline_at, bool)
            and active_deadline_at > 0
            else None
        )

        def start_active_attempt() -> float:
            nonlocal active_started_at, active_deadline_at
            with self._transaction(task_key=key):
                current = self._tasks.get(key) or {}
                saved_start = current.get("active_attempt_started_at")
                saved_deadline = current.get("active_attempt_deadline_at")
            if (
                not isinstance(saved_start, (int, float))
                or isinstance(saved_start, bool)
                or saved_start <= 0
            ):
                saved_start = time.time()
            if (
                not isinstance(saved_deadline, (int, float))
                or isinstance(saved_deadline, bool)
                or saved_deadline <= 0
            ):
                saved_deadline = float(saved_start) + IMAGE_ACTIVE_BUDGET_SECONDS
            active_started_at = float(saved_start)
            active_deadline_at = float(saved_deadline)
            progress_callback.active_deadline_at = active_deadline_at
            self._update_task(
                key,
                active_attempt_started_at=active_started_at,
                active_attempt_deadline_at=active_deadline_at,
                started_ts=active_started_at,
            )
            return active_deadline_at

        progress_callback.start_active_attempt = start_active_attempt

        def record_local_pacing_wait(seconds: float) -> None:
            nonlocal active_deadline_at
            if seconds <= 0 or active_deadline_at is None:
                return
            with self._transaction(task_key=key):
                current = self._tasks.get(key) or {}
                deadline = current.get("active_attempt_deadline_at")
                if not isinstance(deadline, (int, float)) or deadline <= 0:
                    return
                active_deadline_at = float(deadline) + seconds
                self._update_task(key, active_attempt_deadline_at=active_deadline_at,
                                  active_local_pacing_wait_secs=float(current.get("active_local_pacing_wait_secs") or 0) + seconds)
            progress_callback.active_deadline_at = active_deadline_at
            progress_callback.local_pacing_wait_secs = float(getattr(progress_callback, "local_pacing_wait_secs", 0)) + seconds

        progress_callback.record_local_pacing_wait = record_local_pacing_wait

        def record_conversation_id(conversation_id: str) -> None:
            conversation_id = _clean(conversation_id)
            if conversation_id:
                self._update_task(key, conversation_id=conversation_id)
        progress_callback.record_conversation_id = record_conversation_id

        def record_submission_started() -> None:
            # A previously deferred request may still carry a proven-unsent
            # outcome. Once the POST begins, only a later result can settle it.
            self._update_task(
                key, upstream_submission_started=True, upstream_outcome="unknown",
                error_code="", waiting={}, recovery_retryable=False,
                recovery_requires_new_conversation=False,
            )
        progress_callback.record_submission_started = record_submission_started

        def record_result_ids(file_ids: list[str], sediment_ids: list[str]) -> None:
            result_ids = list(dict.fromkeys(str(item) for item in file_ids + sediment_ids if item))
            self._update_task(
                key,
                result_file_ids=list(dict.fromkeys(str(item) for item in file_ids if item)),
                result_sediment_ids=list(dict.fromkeys(str(item) for item in sediment_ids if item)),
                _pending_image_result_ids=None,
                _pending_image_output=None,
                progress="receiving_image",
                upstream_outcome="generated",
                upstream_unfinished=False,
            )
            progress_callback.image_thread_result_ids = result_ids
        progress_callback.record_result_ids = record_result_ids
        progress_callback.record_downloaded_image_items = lambda coverage, items: self._store_pending_image_output(
            key, coverage, items,
        )
        progress_callback.record_pending_result_ids = lambda files, sediments: self._update_task(
            key, _pending_image_result_ids={"file_ids": files, "sediment_ids": sediments},
        )
        progress_callback.image_thread = payload.get("_image_thread")
        progress_callback.failed_retry_original = payload.get("_failed_retry_original")
        progress_callback.failed_retry_predecessor = payload.get("_failed_retry_predecessor")
        progress_callback.failed_retry_required = bool(payload.get("_continue_after_failed_attempt"))
        progress_callback.image_thread_predecessor_message = payload.get("_image_thread_predecessor_message")
        progress_callback.image_thread_predecessor_result_ids = payload.get("_image_thread_predecessor_result_ids")
        prior_id = (payload.get("_image_thread") or {}).get("previous_task_id")
        if prior_id:
            with self.store.connect() as db:
                from services.image_thread import selected_thread_result
                prior = self.store.read_receipt(db, "image", _owner_id(identity), prior_id)
                owned = {prior_id: prior} if prior else {}
                selected_id = ((prior or {}).get("_completion") or {}).get("selected_id")
                if selected_id:
                    selected = self.store.read_receipt(db, "image", _owner_id(identity), selected_id)
                    if selected:
                        owned[selected_id] = selected
                prior = selected_thread_result(owned.get(prior_id), owned) or {}
                # Call-local proof only; do not rewrite a successful source receipt
                # or invalidate an already accepted edit's source fingerprint.
                progress_callback.image_thread_predecessor_cursor_proof = {
                    k: prior.get(k) for k in ("_image_thread_request_parent", "_image_thread_predecessor_message",
                        "_image_thread_predecessor_result_ids", "result_file_ids", "result_sediment_ids")}
        # 将进度回调添加到 payload 中（handler 会提取并传递给 ConversationRequest）
        payload_with_progress = {**payload, "progress_callback": progress_callback}
        failure_phase = "handler_operation"
        try:
            handler = self.edit_handler if mode == "edit" else self.generation_handler
            result = handler(payload_with_progress)
            failure_phase = "validate_image_result"
            if not isinstance(result, dict):
                raise RuntimeError("image task returned streaming result unexpectedly")
            data = result.get("data")
            account_email = _clean(result.get("_account_email") or result.get("account_email"))
            provider_binding_id = _clean(result.get("_provider_binding_id"))
            provider_account_identity = _clean(result.get("_provider_account_identity"))
            conversation_id = _clean(result.get("_conversation_id"))
            parent_message_id = _clean(result.get("_parent_message_id"))
            if not isinstance(data, list) or not data:
                upstream = _clean(result.get("message"))
                if upstream:
                    message = upstream
                else:
                    message = "号池中没有可用账号或所有账号均被限流，请检查号池状态（账号额度、是否被封禁、是否到达生图上限）"
                error = RuntimeError(message)
                if account_email:
                    setattr(error, "account_email", account_email)
                raise error
            usage = result.get("usage")
            duration_ms = int((time.time() - started) * 1000)
            expected_binding_id = _clean(payload.get("provider_binding_id"))
            expected_account_identity = _clean(payload.get("provider_account_identity"))
            if expected_binding_id and provider_binding_id != expected_binding_id:
                raise RuntimeError("bound image result changed provider binding identity")
            if expected_account_identity and provider_account_identity != expected_account_identity:
                raise RuntimeError("bound image result changed provider account identity")
            if payload.get("_image_thread") and (len(data) != 1 or not result.get("_image_thread_terminal") or (payload.get("conversation_id") and payload["conversation_id"] != conversation_id)):
                raise ImageThreadError("IMAGE_THREAD_TURN_UNCONFIRMED", submitted=True)
            if (
                bool(provider_binding_id) != bool(provider_account_identity)
                or bool(provider_binding_id) != bool(conversation_id)
                or bool(provider_binding_id) != bool(parent_message_id)
            ):
                raise RuntimeError("bound image result is missing authoritative conversation state")
            self._update_task(
                key,
                status=TASK_STATUS_SUCCESS,
                _pending_image_result_ids=None,
                _pending_image_output=None,
                data=data,
                usage=usage,
                error="",
                error_code="",
                waiting={},
                _ready_at=0,
                duration_ms=duration_ms,
                provider_binding_id=provider_binding_id,
                provider_account_identity=provider_account_identity,
                conversation_id=conversation_id,
                parent_message_id=parent_message_id,
                binding_status="bound" if provider_binding_id else "unbound",
                **({"_image_thread_terminal": True} if payload.get("_image_thread") else {}),
                upstream_unfinished=False,
                upstream_outcome="generated",
                recovery_error_code="",
                recovery_phase="",
                recovery_retry_after_seconds=None,
                next_poll_at=0,
            )
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用完成",
                request_preview=request_text(payload.get("prompt")),
                urls=_collect_image_urls(data),
                account_email=account_email,
            )
        except Exception as exc:
            if failure_phase == "handler_operation":
                failure_phase = handler_failure_phase
            error_message = str(exc) or "image task failed"
            with self._transaction(task_key=key):
                current = dict(self._tasks.get(key) or {})
            admission_rejection = (exc if isinstance(exc, AdmissionLost)
                                   else exc.__cause__ if getattr(exc, "code", None) == "IMAGE_GENERATION_NOT_SUBMITTED"
                                   and getattr(exc, "upstream_submitted", None) is False
                                   and isinstance(exc.__cause__, AdmissionLost) else None)
            if (admission_rejection is not None and self.admission is not None
                    and payload.get("_admission_claim")
                    and current.get("_claim_id") == payload["_admission_claim"]
                    and current.get("_submission_started") is False):
                # A local capacity/claim change is still admission, not a
                # failed generation. Let the scheduler requeue this original
                # without consuming its bounded upstream-failure retry.
                raise admission_rejection
            account_email = _clean(getattr(exc, "account_email", ""))
            conversation_id = _clean(
                getattr(exc, "conversation_id", "") or current.get("conversation_id")
            )
            provider_binding_id = _clean(
                getattr(exc, "provider_binding_id", "")
                or current.get("provider_binding_id")
                or payload.get("provider_binding_id")
            )
            provider_account_identity = _clean(
                getattr(exc, "provider_account_identity", "")
                or current.get("provider_account_identity")
                or payload.get("provider_account_identity")
            )
            parent_message_id = _clean(
                getattr(exc, "parent_message_id", "") or current.get("parent_message_id")
            )
            request_message_id = _clean(
                getattr(exc, "request_message_id", "") or current.get("request_message_id")
            )
            result_captured = bool(
                current.get("result_file_ids") or current.get("result_sediment_ids")
            )
            error_code = _clean(getattr(exc, "code", ""))
            upstream_submitted = getattr(exc, "upstream_submitted", None)
            known_not_submitted = upstream_submitted is False
            if (known_not_submitted and error_code == "IMAGE_THREAD_PREVIOUS_UNCONFIRMED"
                    and self.admission is not None and payload.get("_admission_claim")
                    and current.get("_submission_started") is False):
                # A failed read or unarchive happened before the image POST.
                # Keep the same durable request and thread cursor in admission;
                # the scheduler retries after a bounded delay without another
                # client submission or a second image-task identity.
                attempts = int(current.get("_archive_restore_attempts") or 0) + 1
                delay = min(60, 2 ** min(attempts, 6))
                self._update_task(
                    key, status=TASK_STATUS_QUEUED, error="原生图会话恢复待重试",
                    error_code=error_code, upstream_unfinished=False,
                    upstream_submission_started=False, upstream_outcome="not_submitted",
                    _archive_restore_attempts=attempts,
                    _ready_at=self.admission.clock() + delay,
                    _claim_id=None, _executing=False, _turn_reserved=False,
                    waiting={"reason": "archive_restore", "next_check_at": self.admission.clock() + delay},
                    active_attempt_started_at=None, active_attempt_deadline_at=None,
                )
                return
            retryable_not_submitted = (
                known_not_submitted and error_code == "IMAGE_GENERATION_NOT_SUBMITTED"
            )
            # Only explicit terminal rejections prove there is no generation
            # left upstream. An unclassified transport exception does not.
            terminal = not result_captured and error_code.lower() in {
                "no_image_generated", "content_policy_violation",
                "conversation_binding_contract_invalid",
            }
            if known_not_submitted:
                terminal = True
            if retryable_not_submitted:
                error_code = "RESULT_UNRECOVERABLE"
            if account and not terminal:
                error_code = "CONVERSATION_OUTCOME_UNKNOWN"
            recovery_phase = (
                "download_image_result" if result_captured else "read_image_request"
            )
            recovery_error_code = _recovery_failure_code(exc, failure_phase, result_captured=result_captured)
            retry_after = _retry_after_seconds(exc)
            if error_code == "CONVERSATION_OUTCOME_UNKNOWN":
                error_message = _safe_recovery_error(recovery_error_code, recovery_phase)
            duration_ms = int((time.time() - started) * 1000)
            self._update_task(key, status=TASK_STATUS_ERROR, error=error_message, data=[],
                              last_recovery_failure=_failure_details(exc, failure_phase),
                              duration_ms=duration_ms,
                              upstream_unfinished=bool(account) and not terminal and not result_captured,
                              **(
                                  {
                                      "recovery_error_code": recovery_error_code,
                                      "recovery_phase": recovery_phase,
                                      "recovery_retry_after_seconds": retry_after,
                                      "next_poll_at": time.time() + (
                                          retry_after if retry_after is not None else 0
                                      ),
                                      **({"upstream_outcome": "generated"} if result_captured else {}),
                                  }
                                  if error_code == "CONVERSATION_OUTCOME_UNKNOWN"
                                  else {
                                      "recovery_error_code": "",
                                      "recovery_phase": "",
                                      "recovery_retry_after_seconds": None,
                                      **({
                                          "upstream_outcome": (
                                              "rejected" if error_code.lower() == "content_policy_violation" else "failed"
                                          ),
                                          "next_poll_at": 0,
                                      } if terminal and not known_not_submitted
                                         and error_code.lower() in {"content_policy_violation", "no_image_generated"} else {}),
                                  }
                              ),
                              **(
                                  {
                                      "upstream_submission_started": False,
                                      "_submission_started": False,
                                      "upstream_outcome": "not_submitted",
                                      **(
                                          {
                                              "recovery_retryable": True,
                                              "recovery_requires_new_conversation": False,
                                          }
                                          if retryable_not_submitted else {}
                                      ),
                                  }
                                  if known_not_submitted else {}
                              ),
                              **({"provider_binding_id": provider_binding_id} if provider_binding_id else {}),
                              **({"provider_account_identity": provider_account_identity} if provider_account_identity else {}),
                              **({"conversation_id": conversation_id} if conversation_id else {}),
                              **({"parent_message_id": parent_message_id} if parent_message_id else {}),
                              **({"request_message_id": request_message_id} if request_message_id else {}),
                              **(
                                  {
                                      "binding_status": (
                                          "bound"
                                          if conversation_id and parent_message_id
                                          else (
                                              "unknown"
                                              if error_code == "CONVERSATION_OUTCOME_UNKNOWN"
                                              else "unavailable"
                                          )
                                      )
                                  }
                                  if provider_binding_id
                                  else {}
                              ),
                              **({"error_code": error_code} if error_code else {}))
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用失败",
                request_preview=request_text(payload.get("prompt")),
                status="failed",
                error=error_message,
                account_email=account_email,
            )

    def _log_call(
        self,
        identity: dict[str, object],
        mode: str,
        model: str,
        started: float,
        suffix: str,
        *,
        request_preview: str = "",
        status: str = "success",
        error: str = "",
        urls: list[str] | None = None,
        account_email: str = "",
    ) -> None:
        endpoint = "/v1/images/edits" if mode == "edit" else "/v1/images/generations"
        summary_prefix = "图生图" if mode == "edit" else "文生图"
        detail = {
            "key_id": identity.get("id"),
            "key_name": identity.get("name"),
            "role": identity.get("role"),
            "endpoint": endpoint,
            "model": model,
            "started_at": datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": _now_iso(),
            "duration_ms": int((time.time() - started) * 1000),
            "status": status,
        }
        if request_preview:
            detail["request_text"] = request_preview
        if error:
            detail["error"] = error
        if account_email:
            detail["account_email"] = account_email
        if urls:
            detail["urls"] = list(dict.fromkeys(urls))
        try:
            log_service.add(LOG_TYPE_CALL, f"{summary_prefix}{suffix}", detail)
        except Exception:
            pass

    def _update_task(self, key: str, **updates: Any) -> None:
        saved_count = None
        nested = getattr(self._transaction_local, "db", None) is not None
        with self._transaction(task_key=key):
            task = self._tasks.get(key)
            if task is None:
                return
            context = current_request.get()
            if context is not None and context.kind == "image" and key == _task_key(context.owner, context.request_id) and task.get("_claim_id") != context.claim:
                raise AdmissionLost("original image claim changed")
            if task.get("status") == TASK_STATUS_SUCCESS and updates.get("status") not in (None, TASK_STATUS_SUCCESS):
                return
            if (task.get("status") != TASK_STATUS_SUCCESS and updates.get("status") == TASK_STATUS_SUCCESS
                    and isinstance(updates.get("data"), list) and updates["data"]):
                saved_count = len(updates["data"])
            task.update(updates)
            # Record the first newly observed qualified IDs atomically. An old
            # receipt or an unrelated status update must not invent this time.
            observed_ids = any(updates.get(field) for field in ("result_file_ids", "result_sediment_ids"))
            first_observation = task.get("_first_qualified_image_assets_observed_at")
            if (observed_ids and task.get("upstream_unfinished") is False and task.get("request_message_id")
                    and not (type(first_observation) in (int, float)
                             and math.isfinite(first_observation) and first_observation > 0)):
                asset_ids = {item for field in ("result_file_ids", "result_sediment_ids")
                             for item in (task.get(field) if isinstance(task.get(field), list) else [])
                             if isinstance(item, str) and item}
                if asset_ids:
                    observed_at = time.time()
                    task["_first_qualified_image_assets_observed_at"] = observed_at
                    task["_first_qualified_image_asset_id_count"] = len(asset_ids)
                    task["_execution_timeline"] = [*(task.get("_execution_timeline") or []),
                        {"stage": "qualified_image_assets_observed", "at": observed_at,
                         "image_count": len(asset_ids)}][-32:]
            task["updated_at"] = _now_iso()
            task["updated_ts"] = time.time()
            # Aggregate callers may have changed other receipts. A scoped
            # caller, however, must not rewrite unrelated cached history (which
            # another worker may already have updated in durable storage).
            aggregate = nested and getattr(self._transaction_local, "task_key", None) != key
            self._save_locked(task_key=None if aggregate else key)
            self._slot_condition.notify_all()
        # Normal dispatch and original-result recovery both persist here. Emit
        # after commit, never on merely receiving an ID or downloading bytes.
        if (saved_count is not None and getattr(self._transaction_local, "db", None) is None
                and context is not None and context.kind == "image"
                and key == _task_key(context.owner, context.request_id)):
            try:
                context.record_stage("artifact_saved", image_count=saved_count)
            except Exception:
                # Observability must not turn a committed result into a retry.
                logger.warning({"event": "pool_artifact_observation_unavailable", "layer": "provider"})

    def _load_locked(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw_items = raw.get("tasks") if isinstance(raw, dict) else raw
        if not isinstance(raw_items, list):
            return {}
        tasks: dict[str, dict[str, Any]] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            task_id = _clean(item.get("id"))
            owner = _clean(item.get("owner_id"))
            if not task_id or not owner:
                continue
            status = _clean(item.get("status"))
            if status not in {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING, TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}:
                status = TASK_STATUS_ERROR
            task = {
                "id": task_id,
                "owner_id": owner,
                "status": status,
                "mode": "edit" if item.get("mode") == "edit" else "generate",
                "model": _clean(item.get("model"), "gpt-image-2"),
                "upstream_model": _clean(item.get("upstream_model")),
                "size": _clean(item.get("size")),
                "quality": _clean(item.get("quality"), "auto"),
                "created_at": _clean(item.get("created_at"), _now_iso()),
                "updated_at": _clean(item.get("updated_at"), _clean(item.get("created_at"), _now_iso())),
                "created_ts": item.get("created_ts"),
                "updated_ts": item.get("updated_ts"),
                "started_ts": item.get("started_ts"),
                "active_attempt_started_at": item.get("active_attempt_started_at"),
                "active_attempt_deadline_at": item.get("active_attempt_deadline_at"),
                "progress": item.get("progress"),
                "duration_ms": item.get("duration_ms"),
                "provider_binding_id": _clean(item.get("provider_binding_id")),
                "provider_account_identity": _clean(item.get("provider_account_identity")),
                "client_conversation_id": _clean(item.get("client_conversation_id")),
                "conversation_id": _clean(item.get("conversation_id")),
                "parent_message_id": _clean(item.get("parent_message_id")),
                "request_message_id": _clean(item.get("request_message_id")),
                "binding_status": _clean(item.get("binding_status"), "unbound"),
                "error_code": _clean(item.get("error_code")),
                "request_hash": _clean(item.get("request_hash")),
                "upstream_unfinished": _holds_upstream_slot(item),
                "upstream_submission_started": (
                    item.get("upstream_submission_started")
                    if isinstance(item.get("upstream_submission_started"), bool)
                    else None
                ),
                "admission_recorded": item.get("admission_recorded") is True,
                "retain_receipt": item.get("retain_receipt") is True,
                "next_poll_at": item.get("next_poll_at", 0),
                "poll_failures": item.get("poll_failures", 0),
                "recovery_no_result_reads": item.get("recovery_no_result_reads", 0),
                "upstream_outcome": _clean(item.get("upstream_outcome")),
                "recovery_retryable": item.get("recovery_retryable") is True,
                "recovery_requires_new_conversation": item.get("recovery_requires_new_conversation") is True,
                "recovery_error_code": _clean(item.get("recovery_error_code")),
                "recovery_phase": _clean(item.get("recovery_phase")),
                "last_recovery_failure": _public_failure_details(item.get("last_recovery_failure")),
                "_pending_image_result_ids": pending_image_result_ids(item) or None,
                "recovery_retry_after_seconds": item.get("recovery_retry_after_seconds"),
                "deadline_recovery_started": item.get("deadline_recovery_started") is True,
                "result_file_ids": [
                    _clean(value) for value in item.get("result_file_ids", [])
                    if _clean(value)
                ] if isinstance(item.get("result_file_ids"), list) else [],
                "result_sediment_ids": [
                    _clean(value) for value in item.get("result_sediment_ids", [])
                    if _clean(value)
                ] if isinstance(item.get("result_sediment_ids"), list) else [],
                "adopted_source_request_message_id": _clean(item.get("adopted_source_request_message_id")),
                "adopted_source_image_message_id": _clean(item.get("adopted_source_image_message_id")),
                "adopted_from_error_code": _clean(item.get("adopted_from_error_code")),
                "adopted_from_error": _clean(item.get("adopted_from_error")),
                "adopted_at": _clean(item.get("adopted_at")),
            }
            data = item.get("data")
            if isinstance(data, list):
                task["data"] = data
            usage = item.get("usage")
            if isinstance(usage, dict):
                task["usage"] = usage
            error = _clean(item.get("error"))
            if error:
                task["error"] = error
            if (
                task.get("binding_status") == "unknown"
                and task.get("error_code") == "CONVERSATION_OUTCOME_UNKNOWN"
                and any(
                    marker in error
                    for marker in (
                        "/backend-api/f/conversation failed: status=403",
                        "/backend-api/conversation failed: status=403",
                    )
                )
            ):
                task["binding_status"] = "unavailable"
                task["error_code"] = "CONVERSATION_BINDING_UNAVAILABLE"
            tasks[_task_key(owner, task_id)] = task
        return tasks

    def _save_locked(self, *, task_key: str | None = None) -> None:
        tasks = self._tasks.values() if task_key is None else (self._tasks[task_key],)
        db = getattr(self._transaction_local, "db", None)
        if db is None:
            # Compatibility for existing maintenance/tests using the local lock.
            with self.store.transaction() as db:
                for task in tasks:
                    self.store.write_receipt(db, "image", task["owner_id"], task["id"], task)
            return
        for task in tasks:
            self.store.write_receipt(db, "image", task["owner_id"], task["id"], task)

    def _recover_unfinished_locked(self) -> bool:
        changed = False
        for task in self._tasks.values():
            if task.get("_recovery_suppressed") is True:
                continue
            if task.get("_input_ref") and (task.get("status") == TASK_STATUS_QUEUED or task.get("_claim_id")):
                # The shared scheduler fences expired claims. Starting a second
                # worker must never interrupt another worker's active request.
                continue
            if task.get("status") in UNFINISHED_STATUSES:
                result_captured = bool(
                    task.get("result_file_ids") or task.get("result_sediment_ids")
                )
                known_not_submitted = task.get("upstream_submission_started") is False
                not_started = (task.get("status") == TASK_STATUS_QUEUED
                               and task.get("admission_recorded") is True
                               and task.get("upstream_unfinished") is False)
                previous_poll_at = task.get("next_poll_at")
                task["status"] = TASK_STATUS_ERROR
                task["error"] = "服务已重启，未完成的图片任务已中断"
                task["error_code"] = (
                    "RESULT_UNRECOVERABLE" if known_not_submitted
                    else "IMAGE_TASK_NOT_STARTED" if not_started
                    else "CONVERSATION_OUTCOME_UNKNOWN"
                )
                if known_not_submitted:
                    task["upstream_unfinished"] = False
                    task["upstream_outcome"] = "not_submitted"
                    task["recovery_retryable"] = True
                    task["recovery_requires_new_conversation"] = False
                elif result_captured:
                    task["upstream_unfinished"] = False
                    task["upstream_outcome"] = "generated"
                    task["recovery_error_code"] = task.get("recovery_error_code") or "RECOVERY_RESULT_INCOMPLETE"
                    task["recovery_phase"] = "download_image_result"
                    task["next_poll_at"] = 0
                elif task["error_code"] == "CONVERSATION_OUTCOME_UNKNOWN":
                    task["recovery_error_code"] = "RECOVERY_READ_FAILED"
                    task["recovery_phase"] = "read_image_request"
                    deadline = task.get("active_attempt_deadline_at")
                    if (
                        isinstance(deadline, (int, float))
                        and not isinstance(deadline, bool)
                        and deadline > 0
                        and time.time() >= float(deadline)
                    ):
                        # A process exit during the first post-deadline read
                        # must not strand the receipt behind its old schedule.
                        task["deadline_recovery_started"] = False
                        task["next_poll_at"] = 0
                if task.get("provider_binding_id"):
                    task["binding_status"] = (
                        "bound"
                        if known_not_submitted and task.get("conversation_id") and task.get("parent_message_id")
                        else "unavailable" if not_started or known_not_submitted else "unknown"
                    )
                if task.get("_recovery_paused") is True:
                    task["next_poll_at"] = previous_poll_at
                task["updated_at"] = _now_iso()
                changed = True
        return changed

    def _retention_cutoff(self) -> float:
        try:
            retention_days = max(1, int(self.retention_days_getter()))
        except Exception:
            retention_days = 30
        return time.time() - retention_days * 86400

    @staticmethod
    def _receipt_expired(task: dict, cutoff: float) -> bool:
        return (task.get("status") in TERMINAL_STATUSES and not _holds_upstream_slot(task)
                and not task.get("retain_receipt") and not task.get("_recovery_paused")
                and task.get("error_code") != "CONVERSATION_OUTCOME_UNKNOWN"
                and _timestamp(task.get("updated_at")) < cutoff)

    def _cleanup_locked(self) -> bool:
        cutoff = self._retention_cutoff()
        removed_keys = [
            key
            for key, task in self._tasks.items()
            if self._receipt_expired(task, cutoff)
        ]
        for key in removed_keys:
            self._tasks.pop(key, None)
            self._transaction_local.db.execute("DELETE FROM image_requests WHERE task_key=?", (key,))
        return bool(removed_keys)

    @staticmethod
    def can_locate_original_cursor(task):
        return (all(_clean(task.get(k)) for k in (
            "provider_binding_id", "provider_account_identity", "client_conversation_id", "request_message_id"))
            and task.get("_image_cursor_lookup_error") not in {
                "REQUEST_CONVERSATION_UNATTRIBUTABLE", "CONVERSATION_BINDING_MISMATCH",
                "CONVERSATION_BINDING_CONTRACT_INVALID"})

    def resume_poll(
        self,
        identity: dict[str, object],
        task_id: str,
        extra_timeout_secs: float = 30.0,
        base_url: str = "",
        allow_unrecoverable_retry: bool = False,
        completion_recheck: bool = False,
        wait_for_completion: bool = False,
    ) -> dict[str, Any]:
        """恢复对已超时任务的轮询，额外等待 extra_timeout_secs 秒。"""
        owner = _owner_id(identity)
        key = _task_key(owner, _clean(task_id))
        with self._transaction(task_key=key):
            task = self._tasks.get(key)
            if task is None:
                raise ValueError("task not found")
            saved_image = bool(task.get("result_file_ids") or task.get("result_sediment_ids"))
            # Pending evidence permits another original read, never direct download.
            saved_image = saved_image or bool(pending_image_result_ids(task))
            from services.generation_completion import ended_image_edit_recheck
            explicit_ended_recheck = completion_recheck and ended_image_edit_recheck(task)
            if task.get("_recovery_suppressed") is True or task.get("_recovery_paused") is True or task.get("_attempt_finished_at") and not saved_image and not explicit_ended_recheck:
                # Keep this endpoint observational while an operator has
                # explicitly stopped the old recovery path.
                return _public_task(task)
            if task.get("status") in {TASK_STATUS_RUNNING, TASK_STATUS_SUCCESS}:
                return _public_task(task)
            if task.get("status") != TASK_STATUS_ERROR:
                raise ValueError("task is not in error state")
            recheck = completion_recheck and bool(task.get("_completion")) and task.get("upstream_outcome") == "unknown"
            if _clean(task.get("error_code")) == "RESULT_UNRECOVERABLE" and not recheck and not saved_image:
                return _public_task(task)
            if (_clean(task.get("error_code")) != "CONVERSATION_OUTCOME_UNKNOWN" and not recheck
                    and not (saved_image and task.get("error_code") == "RESULT_UNRECOVERABLE")):
                raise ValueError("task outcome is not unknown")
            conversation_id = _clean(task.get("conversation_id"))
            if not conversation_id and not self.can_locate_original_cursor(task):
                raise ValueError("task has no conversation_id")
            if not _clean(task.get("request_message_id")):
                # Do not rotate this legacy UNKNOWN through a zero-duration
                # RUNNING/error cycle. There is no safe message boundary to
                # query, and selecting an ancestor, latest node, matching
                # prompt, or another task's request would risk attribution to a
                # different generation. The task remains recoverable when an
                # independently proven request_message_id is repaired in its
                # persisted receipt.
                if not allow_unrecoverable_retry:
                    return _public_task(task)
            deadline = task.get("active_attempt_deadline_at")
            deadline_expired = (
                isinstance(deadline, (int, float))
                and not isinstance(deadline, bool)
                and deadline > 0
                and time.time() >= float(deadline)
            )
            now = time.time()
            retry_after = task.get("recovery_retry_after_seconds")
            provider_cooldown_active = (
                _clean(task.get("recovery_error_code")) == "RECOVERY_RATE_LIMITED"
                and isinstance(retry_after, (int, float))
                and not isinstance(retry_after, bool)
                and retry_after >= 0
                and now < float(task.get("next_poll_at") or 0)
            )
            # The first post-deadline read may bypass a locally synthesized
            # backoff, but never a provider-supplied 429 Retry-After window.
            deadline_recovery_start = (
                deadline_expired
                and not task.get("deadline_recovery_started")
                and not provider_cooldown_active
            )
            if now < float(task.get("next_poll_at") or 0) and not deadline_recovery_start:
                return _public_task(task)
            mode = task.get("mode", "generate")
            model = task.get("model", "gpt-image-2")
            recovery_context = None
            if self.admission is not None:
                from services.pool_admission import ExecutionContext
                claim = uuid.uuid4().hex
                task.update(_claim_id=claim, _claim_until=self.admission.clock() + self.admission.CLAIM_SECONDS,
                            _submission_started=True, _executing=True, _turn_reserved=False)
                recovery_context = ExecutionContext(self.admission, "image", owner, _clean(task_id), claim)
            # 将任务状态重置为 running
            self._update_task(
                key,
                status=TASK_STATUS_RUNNING,
                error="",
                **({"deadline_recovery_started": True} if deadline_recovery_start else {}),
            )

        arguments = (key, conversation_id, extra_timeout_secs, base_url, dict(identity), mode, model,
                     bool(allow_unrecoverable_retry), bool(deadline_expired))
        target = self.admission.run_recovery if recovery_context is not None else self._run_resume_poll
        worker_args = (recovery_context, self._run_resume_poll, arguments) if recovery_context is not None else arguments
        if wait_for_completion:
            # The dispatcher already owns a bounded worker and conversation
            # permit. Keep both until the actual original-read I/O finishes.
            target(*worker_args)
            return self.list_tasks(identity, task_ids=[task_id])["items"][0]
        thread = threading.Thread(target=target, args=worker_args,
            name=f"image-resume-{_clean(task_id)[:16]}", daemon=True)
        thread.start()
        return _public_task(task)

    def recover_manual(
        self,
        identity: dict[str, object],
        task_id: str,
        *,
        provider_binding_id: str,
        provider_account_identity: str,
        client_conversation_id: str,
        base_url: str = "",
    ) -> dict[str, Any]:
        """Resolve a later manual image on the original task's bound branch.

        The caller cannot choose a Provider conversation or an image node.  The
        existing adoption path verifies the original sent message and the
        latest completed manual turn before atomically saving the result.
        """
        owner = _owner_id(identity)
        key = _task_key(owner, _clean(task_id))
        supplied_identity = tuple(map(_clean, (
            provider_binding_id, provider_account_identity, client_conversation_id,
        )))
        if not all(supplied_identity):
            raise ConversationImageAdoptionError("complete conversation authority is required")
        with self._transaction(task_key=key):
            task = self._tasks.get(key)
            if task is None:
                raise ConversationImageAdoptionError("task not found")
            if tuple(_clean(task.get(field)) for field in (
                "provider_binding_id", "provider_account_identity", "client_conversation_id",
            )) != supplied_identity:
                raise ConversationImageAdoptionError("conversation authority does not match the original task")
            if task.get("_recovery_suppressed") is True:
                raise ConversationImageAdoptionError("original task recovery is stopped")
            if task.get("status") == TASK_STATUS_SUCCESS:
                return _public_task(task)
            if task.get("status") != TASK_STATUS_ERROR or _clean(task.get("error_code")).lower() not in {
                "content_policy_violation", "no_image_generated",
            }:
                raise ConversationImageAdoptionError("task is not eligible for manual image adoption")
            if not _clean(task.get("request_message_id")):
                raise ConversationImageAdoptionError("original request identity is unavailable")
            conversation_id = _clean(task.get("conversation_id"))
            if not conversation_id:
                raise ConversationImageAdoptionError("original conversation identity is unavailable")

        try:
            return self.adopt_latest_conversation_image(
                identity,
                task_id,
                provider_binding_id=provider_binding_id,
                provider_account_identity=provider_account_identity,
                client_conversation_id=client_conversation_id,
                conversation_id=conversation_id,
                base_url=base_url,
                allow_no_image_generated=True,
                require_original_on_current_branch=True,
            )
        except ConversationImageAdoptionError as exc:
            # A read-only upstream 429 is not a task outcome or an invalid
            # manual image. Preserve its layer and Retry-After for the caller.
            upstream = exc.__cause__
            if upstream is not None and _upstream_status_code(upstream) == 429:
                error = ImageThreadError("RECOVERY_RATE_LIMITED", status=429)
                error.retry_after = _retry_after_seconds(upstream)
                raise error from exc
            raise

    def adopt_latest_conversation_image(
        self,
        identity: dict[str, object],
        task_id: str,
        *,
        provider_binding_id: str,
        provider_account_identity: str,
        client_conversation_id: str,
        conversation_id: str,
        source_request_message_id: str = "",
        source_image_message_id: str = "",
        base_url: str = "",
        allow_no_image_generated: bool = False,
        require_original_on_current_branch: bool = False,
    ) -> dict[str, Any]:
        """Adopt the latest completed manual image turn without generating again."""
        owner = _owner_id(identity)
        key = _task_key(owner, _clean(task_id))
        supplied_identity = tuple(map(_clean, (
            provider_binding_id,
            provider_account_identity,
            client_conversation_id,
            conversation_id,
        )))
        if not all(supplied_identity):
            raise ConversationImageAdoptionError("complete conversation authority is required")
        expected_source_request = _clean(source_request_message_id)
        expected_source_image = _clean(source_image_message_id)
        eligible_errors = {"content_policy_violation"}
        if allow_no_image_generated:
            eligible_errors.add("no_image_generated")

        with self._transaction(task_key=key):
            task = self._tasks.get(key)
            if task is None:
                raise ConversationImageAdoptionError("task not found")
            stored_identity = tuple(_clean(task.get(field)) for field in (
                "provider_binding_id",
                "provider_account_identity",
                "client_conversation_id",
                "conversation_id",
            ))
            if stored_identity != supplied_identity:
                raise ConversationImageAdoptionError("conversation authority does not match the original task")
            if task.get("_recovery_suppressed") is True:
                raise ConversationImageAdoptionError("original task recovery is stopped")
            if task.get("status") == TASK_STATUS_SUCCESS:
                if task.get("adopted_source_request_message_id"):
                    if (
                        expected_source_request
                        and expected_source_request
                        != _clean(task.get("adopted_source_request_message_id"))
                    ):
                        raise ConversationImageAdoptionError(
                            "specified manual request does not match the adopted image"
                        )
                    if (
                        expected_source_image
                        and expected_source_image
                        != _clean(task.get("adopted_source_image_message_id"))
                    ):
                        raise ConversationImageAdoptionError(
                            "specified image node does not match the adopted image"
                        )
                    return _public_task(task)
                raise ConversationImageAdoptionError("task already completed without manual image adoption")
            if task.get("status") != TASK_STATUS_ERROR:
                raise ConversationImageAdoptionError("task is not in a terminal error state")
            if _clean(task.get("error_code")).lower() not in eligible_errors:
                raise ConversationImageAdoptionError("task is not eligible for manual image adoption")
            original_request_message_id = _clean(task.get("request_message_id"))
            if not original_request_message_id:
                raise ConversationImageAdoptionError("original request identity is unavailable")
            original_error_code = _clean(task.get("error_code"))
            original_error = _clean(task.get("error"))
            original_task_created_ts = task.get("created_ts")
            if not isinstance(original_task_created_ts, (int, float)):
                original_task_created_ts = _timestamp(task.get("created_at"))

        backend = None
        try:
            from services.account_service import account_service
            from services.openai_backend_api import OpenAIBackendAPI
            from services.protocol.conversation import format_image_result

            if account_service.get_bound_account_identity(provider_binding_id) != provider_account_identity:
                raise ConversationImageAdoptionError("provider account identity changed")
            access_token = account_service.get_bound_text_access_token(provider_binding_id, model="auto")
            with account_service.conversation_binding_lock(provider_binding_id, client_conversation_id):
                backend = OpenAIBackendAPI(access_token=access_token)
                document = backend._get_conversation(conversation_id)
                document_id = _clean(document.get("conversation_id") if isinstance(document, dict) else "")
                if document_id and document_id != conversation_id:
                    raise ConversationImageAdoptionError("conversation identity changed")
                source_request_id, image_record, current_node = _latest_completed_manual_image_turn(
                    document,
                    original_request_message_id,
                    float(original_task_created_ts or 0),
                    backend._extract_image_tool_records,
                    require_original_on_current_branch=require_original_on_current_branch,
                )
                if expected_source_request and source_request_id != expected_source_request:
                    raise ConversationImageAdoptionError("specified manual request is not the latest completed image turn")
                if expected_source_image and _clean(image_record.get("message_id")) != expected_source_image:
                    raise ConversationImageAdoptionError("specified image node is not the latest completed image")
                tasks = backend._query_backend_tasks(
                    conversation_id=conversation_id,
                    timeout_secs=5.0,
                    strict_schema=True,
                )
                if _backend_tasks_may_be_active(tasks) or _document_current_message_active(document):
                    raise ConversationImageAdoptionError("latest manual request is still active")
                image_urls = backend.resolve_conversation_image_urls(
                    conversation_id,
                    list(image_record.get("file_ids") or []),
                    list(image_record.get("sediment_ids") or []),
                    poll=False,
                    request_message_id=source_request_id,
                )
                if not image_urls:
                    raise ConversationImageAdoptionError("latest manual image could not be resolved")
                image_bytes = backend.download_image_bytes(image_urls)
                if not image_bytes:
                    raise ConversationImageAdoptionError("latest manual image could not be downloaded")
                latest_document = backend._get_conversation(conversation_id)
                latest_document_id = _clean(
                    latest_document.get("conversation_id")
                    if isinstance(latest_document, dict) else ""
                )
                if latest_document_id and latest_document_id != conversation_id:
                    raise ConversationImageAdoptionError("conversation identity changed")
                latest_source_request, latest_image_record, latest_current_node = (
                    _latest_completed_manual_image_turn(
                        latest_document,
                        original_request_message_id,
                        float(original_task_created_ts or 0),
                        backend._extract_image_tool_records,
                        require_original_on_current_branch=require_original_on_current_branch,
                    )
                )
                latest_tasks = backend._query_backend_tasks(
                    conversation_id=conversation_id,
                    timeout_secs=5.0,
                    strict_schema=True,
                )
                if (
                    _backend_tasks_may_be_active(latest_tasks)
                    or _document_current_message_active(latest_document)
                    or latest_source_request != source_request_id
                    or _clean(latest_image_record.get("message_id")) != _clean(image_record.get("message_id"))
                    or latest_current_node != current_node
                ):
                    raise ConversationImageAdoptionError("conversation changed during image adoption")
            image_items = [
                {"b64_json": __import__("base64").b64encode(item).decode("ascii")}
                for item in image_bytes
            ]
            data = format_image_result(
                image_items,
                "",
                "b64_json",
                base_url,
                int(time.time()),
            )["data"]
            if not data:
                raise ConversationImageAdoptionError("latest manual image could not be stored")

            with self._transaction(task_key=key):
                current = self._tasks.get(key)
                if current is None:
                    raise ConversationImageAdoptionError("task not found")
                if current.get("status") == TASK_STATUS_SUCCESS and current.get("adopted_source_request_message_id"):
                    if (
                        _clean(current.get("adopted_source_request_message_id"))
                        == source_request_id
                        and _clean(current.get("adopted_source_image_message_id"))
                        == _clean(image_record.get("message_id"))
                    ):
                        return _public_task(current)
                    raise ConversationImageAdoptionError(
                        "task was concurrently adopted from a different conversation image"
                    )
                current_identity = tuple(_clean(current.get(field)) for field in (
                    "provider_binding_id",
                    "provider_account_identity",
                    "client_conversation_id",
                    "conversation_id",
                ))
                if (
                    current.get("status") != TASK_STATUS_ERROR
                    or _clean(current.get("error_code")).lower() not in eligible_errors
                    or current.get("_recovery_suppressed") is True
                    or _clean(current.get("request_message_id")) != original_request_message_id
                    or current_identity != supplied_identity
                ):
                    raise ConversationImageAdoptionError("original task changed during image adoption")
                snapshot = dict(current)
                try:
                    current.update({
                        "status": TASK_STATUS_SUCCESS,
                        "_pending_image_result_ids": None,
                        "data": data,
                        "error": "",
                        "error_code": "",
                        "binding_status": "bound",
                        "parent_message_id": current_node,
                        "upstream_unfinished": False,
                        "next_poll_at": 0,
                        "adopted_source_request_message_id": source_request_id,
                        "adopted_source_image_message_id": _clean(image_record.get("message_id")),
                        "adopted_from_error_code": original_error_code,
                        "adopted_from_error": original_error,
                        "adopted_at": _now_iso(),
                        "updated_at": _now_iso(),
                        "updated_ts": time.time(),
                    })
                    self._save_locked(task_key=key)
                except Exception:
                    current.clear()
                    current.update(snapshot)
                    raise
                self._slot_condition.notify_all()
                return _public_task(current)
        except ConversationImageAdoptionError:
            raise
        except Exception as exc:
            raise ConversationImageAdoptionError(
                "conversation image adoption could not be verified"
            ) from exc
        finally:
            if backend is not None:
                backend.close()

    def _store_pending_image_output(self, key, coverage, items):
        """Keep downloaded originals private until exact-turn confirmation."""
        import base64
        from services.durable_forward import MAX_OUTPUT_BYTES
        if not isinstance(items, list) or not items:
            raise ValueError("private image output empty")
        for item in items:
            if not isinstance(item, dict) or set(item) != {"b64_json"} or not base64.b64decode(item["b64_json"], validate=True):
                raise ValueError("invalid private image output")
        encoded = json.dumps(items).encode()
        if len(encoded) > MAX_OUTPUT_BYTES:
            raise ValueError("private image output limit")
        ref = self.store.create_output()
        with self.store.output_file(ref, append=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        with self._transaction(task_key=key):
            current = self._tasks.get(key) or {}
            actual = {"conversation_id": _clean(current.get("conversation_id")),
                      "request_message_id": _clean(current.get("request_message_id")),
                      "file_ids": list(current.get("result_file_ids") or []),
                      "sediment_ids": list(current.get("result_sediment_ids") or [])}
            if actual != coverage:
                raise AdmissionLost("original image output coverage changed")
            self._update_task(key, _pending_image_output={"output_ref": ref, "coverage": coverage})

    def _run_resume_poll(
        self,
        key: str,
        conversation_id: str,
        extra_timeout_secs: float,
        base_url: str,
        identity: dict[str, object],
        mode: str,
        model: str,
        allow_unrecoverable_retry: bool,
        deadline_expired: bool,
    ) -> None:
        """后台线程：继续轮询已有 conversation_id 的图片结果。"""
        started = time.time()
        backend = None
        account_service = None
        access_token = ""
        conversation_available = False
        failure_phase = "read_image_request"
        try:
            from services.account_service import account_service
            from services.openai_backend_api import ImageContentPolicyError, ImagePollTimeoutError, OpenAIBackendAPI
            from services.protocol.conversation import format_image_result, _validate_image_download_count

            with self._transaction(task_key=key) as db:
                task = self._tasks.get(key)
                binding_id = _clean(task.get("provider_binding_id")) if task else ""
                account_identity = _clean(task.get("provider_account_identity")) if task else ""
                client_conversation_id = _clean(task.get("client_conversation_id")) if task else ""
                request_message_id = _clean(task.get("request_message_id")) if task else ""
                persisted_file_ids = list(task.get("result_file_ids") or []) if task else []
                persisted_sediment_ids = list(task.get("result_sediment_ids") or []) if task else []
                pending_ids = pending_image_result_ids(task) if task else {}
                image_thread = (task or {}).get("_image_thread")
                expected_parent = (task or {}).get("_image_thread_request_parent")
                predecessor_message = (task or {}).get("_image_thread_predecessor_message")
                predecessor_ids = (task or {}).get("_image_thread_predecessor_result_ids")
                retry_predecessor = self.store.read_receipt(db, "image", _owner_id(identity),
                    (image_thread or {}).get("previous_task_id") or "")
            def recovered_parent(backend, result_file_ids, result_sediment_ids):
                if image_thread:
                    return finished_parent(backend._get_conversation(conversation_id), conversation_id,
                        request_message_id, expected_parent=expected_parent,
                        expected_result_ids=result_file_ids + result_sediment_ids,
                        predecessor_request_message_id=predecessor_message,
                        predecessor_result_ids=predecessor_ids)
                return backend.get_conversation_parent_message_id(conversation_id)
            def downloaded_items(backend, files, sediments):
                # A final conversation read may fail after download. Keep bytes
                # private until that read succeeds, and reuse them on recovery.
                import base64
                from services.durable_forward import MAX_OUTPUT_BYTES
                nonlocal failure_phase
                coverage = {"conversation_id": conversation_id, "request_message_id": request_message_id,
                            "file_ids": list(files), "sediment_ids": list(sediments)}
                cached = (task or {}).get("_pending_image_output") or {}
                if isinstance(cached, dict) and cached.get("coverage") == coverage:
                    try:
                        with self.store.output_file(cached["output_ref"]) as handle:
                            encoded = handle.read(MAX_OUTPUT_BYTES + 1)
                        if len(encoded) > MAX_OUTPUT_BYTES:
                            raise ValueError("private image output limit")
                        items = json.loads(encoded)
                        if not isinstance(items, list) or not items:
                            raise ValueError("private image output empty")
                        for item in items:
                            if not isinstance(item, dict) or set(item) != {"b64_json"} or not base64.b64decode(item["b64_json"], validate=True):
                                raise ValueError("invalid private image output")
                        _validate_image_download_count(items, files, sediments)
                        context = current_request.get()
                        if context is not None:
                            try:
                                context.record_stage("attachment_download_cache_reused", image_count=len(items))
                            except Exception:
                                pass  # Losing timing evidence must not force a download.
                        return items
                    except (OSError, ValueError, KeyError, TypeError):
                        pass  # Missing/corrupt private cache: fetch the same assets.
                failure_phase = "resolve_image_result"
                image_urls = backend.resolve_conversation_image_urls(
                    conversation_id, files, sediments, poll=False, request_message_id=request_message_id)
                if not image_urls:
                    raise RuntimeError("generated image URL could not be resolved")
                failure_phase = "download_image_result"
                downloaded = backend.download_image_bytes(image_urls)
                if not downloaded:
                    raise RuntimeError("generated image could not be downloaded")
                items = [{"b64_json": base64.b64encode(value).decode("ascii")} for value in downloaded]
                encoded = json.dumps(items).encode()
                if len(encoded) > MAX_OUTPUT_BYTES:
                    raise ValueError("private image output limit")
                self._store_pending_image_output(key, coverage, items)
                _validate_image_download_count(items, files, sediments)
                return items
            if not binding_id or not account_identity or not client_conversation_id:
                error = RuntimeError("conversation binding unavailable: task authority missing")
                error.status_code = 403
                raise error
            authoritative_identity = account_service.get_bound_account_identity(binding_id)
            if authoritative_identity != account_identity:
                error = RuntimeError("conversation binding unavailable: account identity changed")
                error.status_code = 403
                raise error
            # Reading the original task consumes no new generation admission.
            access_token = account_service.get_bound_text_access_token(binding_id, model="auto")
            with account_service.conversation_binding_lock(binding_id, client_conversation_id):
                backend = OpenAIBackendAPI(access_token=access_token)
                located_document = None
                if not conversation_id:
                    from services.conversation_binding_service import (
                        ConversationBindingService, ConversationBindingError, RECOVERY_CONVERSATION_SCAN_FIELD,
                    )
                    # Reuse the bounded exact-message scan already used by text
                    # and durable image forwarding. Never match titles/prompts.
                    try:
                        located, located_document = ConversationBindingService._locate_text_request_conversation(
                            backend, {**task, "started_at": task.get("started_ts"),
                                      "created_at": task.get("created_ts"),
                                      "request_parent_message_id": expected_parent or ""})
                    except ConversationBindingError as exc:
                        self._update_task(key, **{
                            RECOVERY_CONVERSATION_SCAN_FIELD: exc.recovery_scan,
                            "_image_cursor_lookup_error": exc.recovery_reason or exc.code,
                        })
                        # The locator wraps read failures to retain scan progress.
                        # Preserve their real account cooldown/auth classification.
                        read_error = exc.recovery_read_error or {}
                        exc.status_code = read_error.get("http_status")
                        if read_error.get("retry_after_seconds") is not None:
                            exc.retry_after = max(read_error["retry_after_seconds"], getattr(exc, "retry_after", 0))
                        raise
                    conversation_id = located["conversation_id"]
                    expected_parent = expected_parent or located["request_parent_message_id"]
                    self._update_task(key, conversation_id=conversation_id,
                                      _image_thread_request_parent=expected_parent,
                                      _image_cursor_lookup_error=None,
                                      **{RECOVERY_CONVERSATION_SCAN_FIELD: {}})
                    task = {**task, "conversation_id": conversation_id,
                            "_image_thread_request_parent": expected_parent}
                def record_pending_ids(files, sediments):
                    pending_ids.update(file_ids=files, sediment_ids=sediments)
                    self._update_task(key, _pending_image_result_ids=dict(pending_ids))
                backend.progress_callback = lambda _step: None
                backend.progress_callback.record_pending_result_ids = record_pending_ids
                if persisted_file_ids or persisted_sediment_ids:
                    confirmed_parent = None
                    extract_records = getattr(backend, "_extract_image_tool_records", None)
                    if image_thread and callable(extract_records):
                        failure_phase = "confirm_image_turn"
                        document = backend._get_conversation(conversation_id)
                        records = extract_records(document, request_message_id)
                        files = list(dict.fromkeys(x for record in records for x in record["file_ids"]))
                        sediments = list(dict.fromkeys(x for record in records for x in record["sediment_ids"]))
                        original_ids = set(persisted_file_ids + persisted_sediment_ids)
                        if (not set(persisted_file_ids) <= set(files)
                                or not set(persisted_sediment_ids) <= set(sediments)):
                            raise ImageThreadError("IMAGE_THREAD_UPSTREAM_CHANGED")
                        if (set(files) > set(persisted_file_ids)
                                or set(sediments) > set(persisted_sediment_ids)):
                            # A stream can expose one image before the same
                            # original turn produces the rest. Expand only from
                            # its complete, unbranched final snapshot; cached
                            # partial bytes must not trap recovery forever.
                            confirmed_parent = finished_parent(document, conversation_id, request_message_id,
                                expected_parent=expected_parent, expected_result_ids=files + sediments,
                                predecessor_request_message_id=predecessor_message,
                                predecessor_result_ids=predecessor_ids, require_final=True)
                            persisted_file_ids, persisted_sediment_ids = files, sediments
                            self._update_task(key, result_file_ids=files, result_sediment_ids=sediments,
                                              _pending_image_result_ids=None)
                        else:
                            # No permission to replace vanished assets or
                            # adopt another branch, even if it has a final.
                            finished_parent(document, conversation_id, request_message_id,
                                expected_parent=expected_parent, expected_result_ids=list(original_ids),
                                predecessor_request_message_id=predecessor_message,
                                predecessor_result_ids=predecessor_ids)
                            try:
                                confirmed_parent = finished_parent(document, conversation_id, request_message_id,
                                    expected_parent=expected_parent, expected_result_ids=list(original_ids),
                                    predecessor_request_message_id=predecessor_message,
                                    predecessor_result_ids=predecessor_ids, require_final=True)
                            except ImageThreadError:
                                pass  # Tool leaves still need a later final read.
                    self._update_task(key, progress="receiving_image", recovery_phase="download_image_result")
                    image_items = downloaded_items(backend, persisted_file_ids, persisted_sediment_ids)
                    failure_phase = "confirm_image_turn"
                    parent_message_id = confirmed_parent or recovered_parent(backend, persisted_file_ids, persisted_sediment_ids)
                    failure_phase = "save_image_result"
                    data = format_image_result(
                        image_items,
                        "",
                        "b64_json",
                        base_url,
                        int(time.time()),
                    )["data"]
                    self._update_task(
                        key,
                        status=TASK_STATUS_SUCCESS,
                        _pending_image_result_ids=None,
                        _pending_image_output=None,
                        data=data,
                        error="",
                        error_code="",
                        binding_status="bound",
                        parent_message_id=parent_message_id,
                        **({"_image_thread_terminal": True} if image_thread else {}),
                        upstream_unfinished=False,
                        upstream_outcome="generated",
                        next_poll_at=0,
                        poll_failures=0,
                        recovery_no_result_reads=0,
                        recovery_requires_new_conversation=False,
                        recovery_error_code="",
                        recovery_phase="",
                        recovery_retry_after_seconds=None,
                        duration_ms=int((time.time() - started) * 1000),
                    )
                    self._log_call(
                        identity,
                        mode,
                        model,
                        started,
                        "调用完成（恢复下载）",
                        status="success",
                        urls=_collect_image_urls(data),
                    )
                    return
                try:
                    document = located_document or backend._get_conversation(conversation_id)
                    conversation_available = True
                    from services.generation_completion import retry_cursor
                    self._update_task(key, _retry_cursor=retry_cursor(document, task, kind="image"))
                except Exception as exc:
                    if _upstream_status_code(exc) != 404:
                        raise
                    if not (allow_unrecoverable_retry or deadline_expired):
                        raise
                    tasks = backend._query_backend_tasks(
                        conversation_id=conversation_id, timeout_secs=5.0, strict_schema=True,
                    )
                    if _backend_tasks_may_be_active(tasks):
                        raise RuntimeError("original conversation is missing while an upstream task may still be active") from exc
                    raise UnrecoverableRead(
                        "original conversation is missing after a bounded result read",
                        requires_new_conversation=True,
                    ) from exc
                if not request_message_id:
                    tasks = backend._query_backend_tasks(
                        conversation_id=conversation_id, timeout_secs=5.0, strict_schema=True,
                    )
                    if _document_current_message_active(document) or _backend_tasks_may_be_active(tasks):
                        raise RuntimeError("legacy image request may still be running upstream")
                    raise UnrecoverableRead(
                        "original legacy image result cannot be safely attributed after a bounded read",
                        requires_new_conversation=False,
                    )
                branch_state = _branch_read_state(document, request_message_id)
                if branch_state in {"running", "terminal_without_result", "no_result"}:
                    self._update_task(key, _completion_read_at=time.time())
                authoritative_failure = _authoritative_image_failure(document, request_message_id)
                if authoritative_failure:
                    raise AuthoritativeImageTaskFailure(
                        authoritative_failure,
                        terminal_parent_message_id=_clean(document.get("current_node")),
                    )
                no_active_task = False
                if allow_unrecoverable_retry or deadline_expired:
                    tasks = backend._query_backend_tasks(
                        conversation_id=conversation_id, timeout_secs=5.0, strict_schema=True,
                    )
                    no_active_task = (
                        not _backend_tasks_may_be_active(tasks)
                        and not _document_current_message_active(document)
                    )

                def latest_unrecoverable_requirement() -> bool | None:
                    try:
                        latest_document = backend._get_conversation(conversation_id)
                    except Exception as latest_exc:
                        if _upstream_status_code(latest_exc) != 404:
                            raise
                        latest_tasks = backend._query_backend_tasks(
                            conversation_id=conversation_id,
                            timeout_secs=5.0,
                            strict_schema=True,
                        )
                        return True if not _backend_tasks_may_be_active(latest_tasks) else None
                    latest_tasks = backend._query_backend_tasks(
                        conversation_id=conversation_id,
                        timeout_secs=5.0,
                        strict_schema=True,
                    )
                    latest_state = _branch_read_state(latest_document, request_message_id)
                    if (
                        _document_current_message_active(latest_document)
                        or _backend_tasks_may_be_active(latest_tasks)
                        or latest_state not in {
                            "terminal_without_result", "unattributable", "no_result",
                        }
                    ):
                        return None
                    extract_records = getattr(backend, "_extract_image_tool_records", None)
                    if callable(extract_records):
                        for record in extract_records(latest_document, request_message_id):
                            if record.get("file_ids") or record.get("sediment_ids"):
                                return None
                    # Only a qualified inactive snapshot can authorize an
                    # absent edit continuation. A failed query cannot do so.
                    proof = retry_cursor(latest_document, task, kind="image", predecessor=retry_predecessor)
                    qualified = (not proof or proof.get("source") != "absent_image_thread_request"
                                 or int(task.get("recovery_no_result_reads") or 0) + 1 >= UNRECOVERABLE_QUALIFIED_READS)
                    self._update_task(key, _retry_cursor=proof,
                        **({"_completion_read_at": time.time()} if proof and qualified else {}))
                    return False

                try:
                    strict_ids = None
                    confirmed_final_parent = None
                    extract_records = getattr(backend, "_extract_image_tool_records", None)
                    if image_thread and callable(extract_records):
                        records = extract_records(document, request_message_id)
                        files = list(dict.fromkeys(x for record in records for x in record["file_ids"]))
                        sediments = list(dict.fromkeys(x for record in records for x in record["sediment_ids"]))
                        if ((files or sediments)
                                and set(pending_ids.get("file_ids", [])) <= set(files)
                                and set(pending_ids.get("sediment_ids", [])) <= set(sediments)):
                            try:
                                # Only this recovery's just-read original may
                                # bypass another asset settle cycle.
                                finished_parent(document, conversation_id, request_message_id,
                                    expected_parent=expected_parent,
                                    expected_result_ids=files + sediments,
                                    predecessor_request_message_id=predecessor_message,
                                    predecessor_result_ids=predecessor_ids)
                            except ImageThreadError:
                                pass
                            else:
                                from services.protocol.conversation import _observe_image_terminal
                                _observe_image_terminal()
                                strict_ids = (files, sediments)
                                try:
                                    confirmed_final_parent = finished_parent(document, conversation_id, request_message_id,
                                        expected_parent=expected_parent, expected_result_ids=files + sediments,
                                        predecessor_request_message_id=predecessor_message,
                                        predecessor_result_ids=predecessor_ids, require_final=True)
                                except ImageThreadError:
                                    pass  # A tool leaf may still acquire later assets.
                    if strict_ids is None and pending_ids and config.image_settle_enabled:
                        # Query before sleeping so a strictly finished original
                        # can proceed immediately. A fallback settle still needs
                        # a NEW observation after the wait, never the pre-wait doc.
                        settle_wait = min(max(0.0, float(config.image_settle_secs)),
                                          max(0.0, float(extra_timeout_secs)))
                        if settle_wait:
                            time.sleep(settle_wait)
                        extra_timeout_secs = max(0.0, float(extra_timeout_secs) - settle_wait)
                        if extra_timeout_secs <= 0:
                            raise ImagePollTimeoutError("原会话结果仍待稳定确认，保留原请求继续读取。", conversation_id)
                        document = backend._get_conversation(conversation_id)
                        self._update_task(key, _retry_cursor=retry_cursor(document, task, kind="image"))
                        authoritative_failure = _authoritative_image_failure(document, request_message_id)
                        if authoritative_failure:
                            raise AuthoritativeImageTaskFailure(
                                authoritative_failure,
                                terminal_parent_message_id=_clean(document.get("current_node")),
                            )
                    pending_options = (
                        {"initial_file_ids": pending_ids.get("file_ids", []),
                         "initial_sediment_ids": pending_ids.get("sediment_ids", []),
                         "require_fresh_result_ids": True}
                        if pending_ids else {}
                    )
                    file_ids, sediment_ids = strict_ids if strict_ids is not None else backend._poll_image_results(
                        conversation_id,
                        extra_timeout_secs,
                        request_message_id=request_message_id,
                        initial_document=document,
                        **pending_options,
                    )
                except Exception as exc:
                    if (
                        not pending_ids and no_active_task
                        and branch_state in {"terminal_without_result", "unattributable", "no_result"}
                        and exc.__class__.__name__ == "ImagePollTimeoutError"
                    ):
                        latest_requirement = latest_unrecoverable_requirement()
                        if latest_requirement is not None:
                            raise UnrecoverableRead(
                                "original image branch has no attributable result after a bounded read",
                                requires_new_conversation=latest_requirement,
                            ) from exc
                    raise
                if not file_ids and not sediment_ids:
                    if (
                        not pending_ids and no_active_task
                        and branch_state in {"terminal_without_result", "unattributable", "no_result"}
                    ):
                        latest_requirement = latest_unrecoverable_requirement()
                        if latest_requirement is not None:
                            raise UnrecoverableRead(
                                "original image branch has no attributable result after a bounded read",
                                requires_new_conversation=latest_requirement,
                            )
                    raise RuntimeError(
                        f"继续等待 {extra_timeout_secs} 秒后仍未找到图片结果。"
                    )

                self._update_task(
                    key,
                    result_file_ids=list(dict.fromkeys(file_ids)),
                    result_sediment_ids=list(dict.fromkeys(sediment_ids)),
                    _pending_image_result_ids=None,
                    progress="receiving_image",
                    upstream_outcome="generated",
                    upstream_unfinished=False,
                    recovery_phase="download_image_result",
                )

                image_items = downloaded_items(backend, file_ids, sediment_ids)
                failure_phase = "confirm_image_turn"
                # A later external branch cannot invalidate a complete original
                # result. Continue/archive still recheck the current cursor.
                # Persisted IDs or cached bytes alone never supply this proof.
                parent_message_id = (confirmed_final_parent
                    if confirmed_final_parent and len(image_items) == len(set(file_ids + sediment_ids))
                    else recovered_parent(backend, file_ids, sediment_ids))
                if confirmed_final_parent and len(image_items) == len(set(file_ids + sediment_ids)):
                    logger.info({"event": "image_complete_final_reused", "conversation_id": conversation_id,
                                 "image_count": len(image_items)})
            failure_phase = "save_image_result"
            data = format_image_result(
                image_items,
                "",  # prompt 已不重要，结果已经拿到了
                "b64_json",
                base_url,
                int(time.time()),
            )["data"]
            self._update_task(
                key,
                status=TASK_STATUS_SUCCESS,
                _pending_image_result_ids=None,
                _pending_image_output=None,
                data=data,
                error="",
                error_code="",
                binding_status="bound",
                parent_message_id=parent_message_id,
                **({"_image_thread_terminal": True} if image_thread else {}),
                upstream_unfinished=False,
                next_poll_at=0,
                poll_failures=0,
                recovery_no_result_reads=0,
                recovery_requires_new_conversation=False,
                recovery_error_code="",
                recovery_phase="",
                recovery_retry_after_seconds=None,
                upstream_outcome="generated",
                duration_ms=int((time.time() - started) * 1000),
            )
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用完成（续轮询）",
                status="success",
                urls=_collect_image_urls(data),
            )
        except Exception as exc:
            error_message = str(exc) or "resume poll failed"
            duration_ms = int((time.time() - started) * 1000)
            error_code = _clean(getattr(exc, "code", ""))
            with self._transaction(task_key=key):
                current = self._tasks.get(key, {})
                failures = int(current.get("poll_failures") or 0) + 1
                qualified_reads = int(current.get("recovery_no_result_reads") or 0)
                requires_new_conversation = bool(current.get("recovery_requires_new_conversation"))
                result_captured = bool(
                    current.get("result_file_ids") or current.get("result_sediment_ids")
                )
                qualified_read_recorded = False
                if conversation_available:
                    requires_new_conversation = False
                if (allow_unrecoverable_retry or deadline_expired) and isinstance(exc, UnrecoverableRead):
                    requires_new_conversation = exc.requires_new_conversation
                    if deadline_expired or _task_age_seconds(current) >= UNRECOVERABLE_MIN_AGE_SECONDS:
                        qualified_reads += 1
                        qualified_read_recorded = True
            if isinstance(exc, ImageContentPolicyError):
                error_code = "content_policy_violation"
            terminal = not result_captured and error_code in {"NO_IMAGE_GENERATED", "content_policy_violation"}
            # A captured image is a download obligation, even if earlier reads
            # exhausted the no-result budget. Keep its retry/cooldown evidence.
            unrecoverable = qualified_reads >= UNRECOVERABLE_QUALIFIED_READS and not result_captured and not terminal
            recovery_phase = (
                "download_image_result" if result_captured else "read_image_request"
            )
            recovery_error_code = _recovery_failure_code(exc, failure_phase, result_captured=result_captured)
            retry_after = _retry_after_seconds(exc)
            final_error_code = (
                error_code if terminal
                else "RESULT_UNRECOVERABLE" if unrecoverable
                else "CONVERSATION_OUTCOME_UNKNOWN"
            )
            if final_error_code == "CONVERSATION_OUTCOME_UNKNOWN":
                error_message = _safe_recovery_error(recovery_error_code, recovery_phase)
            terminal_retry_cursor = None
            if isinstance(exc, AuthoritativeImageTaskFailure) and not result_captured:
                # Reuse the existing cursor proof from this exact authoritative
                # document. A failed error code alone cannot end an UNKNOWN.
                from services.generation_completion import retry_cursor
                terminal_retry_cursor = retry_cursor(document, current, kind="image")
            self._update_task(
                key,
                status=TASK_STATUS_ERROR,
                error=error_message,
                error_code=final_error_code,
                last_recovery_failure=_failure_details(exc, failure_phase),
                binding_status=("bound" if terminal
                                else "unavailable" if unrecoverable and requires_new_conversation
                                else "bound" if unrecoverable else "unknown"),
                data=[],
                duration_ms=duration_ms,
                upstream_unfinished=not (terminal or unrecoverable or result_captured),
                **({"_retry_cursor": terminal_retry_cursor} if terminal_retry_cursor else {}),
                poll_failures=failures,
                recovery_no_result_reads=qualified_reads,
                recovery_requires_new_conversation=requires_new_conversation,
                recovery_error_code=(
                    recovery_error_code
                    if final_error_code == "CONVERSATION_OUTCOME_UNKNOWN" else ""
                ),
                recovery_phase=(
                    recovery_phase
                    if final_error_code == "CONVERSATION_OUTCOME_UNKNOWN" else ""
                ),
                recovery_retry_after_seconds=(
                    retry_after
                    if final_error_code == "CONVERSATION_OUTCOME_UNKNOWN" else None
                ),
                **(
                    {"parent_message_id": _clean(getattr(exc, "terminal_parent_message_id", ""))}
                    if terminal and _clean(getattr(exc, "terminal_parent_message_id", ""))
                    else {}
                ),
                upstream_outcome=("generated" if result_captured else
                                  "rejected" if terminal and error_code == "content_policy_violation" else
                                  "failed" if terminal else "unknown"),
                **({"recovery_retryable": False} if terminal else
                   {"recovery_retryable": True} if unrecoverable else {}),
                next_poll_at=(
                    0
                    if terminal or unrecoverable
                    else time.time() + (
                        retry_after
                        if retry_after is not None
                        else 5
                        if deadline_expired and qualified_read_recorded
                        else 30
                        if deadline_expired
                        else min(
                            900,
                            60 * 2 ** min(
                                (qualified_reads if qualified_read_recorded else failures) - 1,
                                4,
                            ),
                        )
                    )
                ),
            )
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用失败（续轮询）",
                status="failed",
                error=error_message,
            )
        finally:
            if backend is not None:
                backend.close()


image_task_service = ImageTaskService(DATA_DIR / "image_tasks.json")
