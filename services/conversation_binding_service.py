from __future__ import annotations

import hashlib
import math
import re
import time
from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import Any
from curl_cffi import CurlInfo
from curl_cffi.requests.exceptions import RequestException

from services.account_service import account_service
from services.account_request_pacing import AccountReadRetryBudgetInsufficient
from services.openai_backend_api import ConversationArchiveCursorMismatch, OpenAIBackendAPI
from services.protocol.conversation import conversation_events
from utils.helper import UpstreamHTTPError


class TextRecoveryReason(str, Enum):
    ACCOUNT_IDENTITY_MISMATCH = "ACCOUNT_IDENTITY_MISMATCH"
    CONVERSATION_ID_MISMATCH = "CONVERSATION_ID_MISMATCH"
    REQUEST_MESSAGE_NOT_FOUND = "REQUEST_MESSAGE_NOT_FOUND"
    REQUEST_PARENT_MISMATCH = "REQUEST_PARENT_MISMATCH"
    REQUEST_BRANCH_AMBIGUOUS = "REQUEST_BRANCH_AMBIGUOUS"
    REQUEST_BRANCH_SUPERSEDED = "REQUEST_BRANCH_SUPERSEDED"
    REQUEST_RESULT_INCOMPLETE = "REQUEST_RESULT_INCOMPLETE"
    REQUEST_CONVERSATION_ADVANCED = "REQUEST_CONVERSATION_ADVANCED"
    REQUEST_RESULT_NOT_FOUND = "REQUEST_RESULT_NOT_FOUND"
    REQUEST_RESULT_TERMINAL_EMPTY = "REQUEST_RESULT_TERMINAL_EMPTY"
    REQUEST_RESULT_NON_TEXT = "REQUEST_RESULT_NON_TEXT"
    REQUEST_CONVERSATION_UNATTRIBUTABLE = "REQUEST_CONVERSATION_UNATTRIBUTABLE"
    REQUEST_CONVERSATION_SCAN_INCOMPLETE = "REQUEST_CONVERSATION_SCAN_INCOMPLETE"
    CONVERSATION_NOT_FOUND = "CONVERSATION_NOT_FOUND"


_ACTIVE_TEXT_RESULT_STATUSES = frozenset({"in_progress", "running", "pending", "queued"})
NON_TEXT_RESULT_FIELD = "_non_text_result"
TURN_END_EVIDENCE_FIELD = "_turn_end_evidence"
RESULT_OBSERVATION_FIELD = "_original_result_observation"
OBSERVED_CONTENT_TYPES = frozenset({"text", "code", "execution_output", "thoughts", "reasoning_recap"})


def _content_observation(content):
    """Measure known text/tool envelopes without retaining their contents."""
    if not isinstance(content, dict):
        return None
    kind = content.get("content_type")
    if not isinstance(kind, str):
        return None
    items = finished = 0
    if kind == "text":
        parts = content.get("parts")
        if not isinstance(parts, list) or any(not isinstance(part, str) for part in parts):
            return None
        chars = sum(len(part) for part in parts)
    elif kind in {"code", "execution_output", "reasoning_recap"}:
        text = content.get("content" if kind == "reasoning_recap" else "text")
        if not isinstance(text, str):
            return None
        chars = len(text)
    elif kind == "thoughts":
        thoughts = content.get("thoughts")
        if not isinstance(thoughts, list) or len(thoughts) > 128:
            return None
        chars, items = 0, len(thoughts)
        for thought in thoughts:
            if (not isinstance(thought, dict)
                    or not isinstance(thought.get("summary"), str)
                    or not isinstance(thought.get("content"), str)
                    or type(thought.get("finished")) is not bool
                    or not isinstance(thought.get("chunks"), list)
                    or len(thought["chunks"]) > 512
                    or any(not isinstance(chunk, str) for chunk in thought["chunks"])):
                return None
            chars += len(thought["summary"]) + len(thought["content"]) + sum(map(len, thought["chunks"]))
            items += len(thought["chunks"])
            finished += int(thought["finished"])
    else:
        return None  # Media or unknown envelopes cannot qualify as empty text.
    return {"content_type": kind, "text_chars": chars, "content_items": items, "finished_items": finished}


def _result_observation(mapping, children, request_message_id, conversation_id):
    """Observe only this request's unambiguous branch, never message contents.

    An upstream heartbeat timestamp alone is not generation progress. Keep a
    bounded structural/text-length observation to distinguish a changing turn
    from a stale in_progress label. Missing or branching nodes cannot qualify.
    """
    node_id, visited, nodes, updated_at = request_message_id, {request_message_id}, [], None
    while children.get(node_id):
        successors = children[node_id]
        if len(successors) != 1 or len(nodes) >= 128:
            return None
        node_id = successors[0]
        if node_id in visited:
            return None
        visited.add(node_id)
        message = (mapping.get(node_id) or {}).get("message")
        if not isinstance(message, dict) or message.get("id") != node_id:
            return None
        author = message.get("author")
        role = author.get("role") if isinstance(author, dict) else None
        if role == "user":
            return None  # A later user turn cannot establish our progress.
        status = message.get("status")
        if role not in {"assistant", "tool"} or status not in _ACTIVE_TEXT_RESULT_STATUSES | {"finished_successfully"}:
            return None
        content = _content_observation(message.get("content"))
        if content is None:
            return None
        nodes.append({"id": node_id, "role": role, "status": status,
                      "end_turn": message.get("end_turn") is True, **content})
        value = message.get("update_time")
        if type(value) in (int, float) and math.isfinite(value) and value > 0:
            updated_at = max(updated_at or value, value)
    if not nodes:
        return None
    return {"conversation_id": conversation_id, "request_message_id": request_message_id,
            "nodes": nodes, "upstream_updated_at": updated_at}


def _request_parent_matches_receipt(
    receipt: dict[str, Any], observed_parent: str, mapping: dict[str, Any],
) -> bool:
    expected_parent = str(receipt.get(
        "request_parent_message_id", receipt.get("parent_message_id"),
    ) or "").strip()
    if not expected_parent or observed_parent == expected_parent:
        return True

    # A batched model POST can omit an intermediate context node from the GET
    # mapping. Accept only the exact outbound root saved before that POST,
    # and only when the predicted intermediate node was not persisted at all.
    # If it exists, a different parent is a real branch mismatch.
    submission_root = str(receipt.get("_submission_parent_message_id") or "").strip()
    return bool(
        submission_root
        and observed_parent == submission_root
        and submission_root in mapping
        and expected_parent not in mapping
    )


def _completed_request_turn(mapping, children, request_message_id, conversation_id, *, allow_completed_tool_call=False,
                            allow_completed_commentary=False):
    """Positive terminal evidence for the exact original branch, even without a usable answer."""
    node_id, visited = request_message_id, {request_message_id}
    while True:
        successors = children.get(node_id, [])
        if len(successors) != 1:
            return None
        node_id = successors[0]
        if node_id in visited:
            return None
        visited.add(node_id)
        node = mapping.get(node_id)
        message = node.get("message") if isinstance(node, dict) else None
        if not isinstance(message, dict) or message.get("id") != node_id:
            return None
        author = message.get("author")
        role = author.get("role") if isinstance(author, dict) else None
        content = message.get("content")
        # A completed final can follow a stale in_progress reasoning snapshot.
        # Only known assistant ancestors qualify; active tools and the final
        # itself still cannot prove that this exact turn ended.
        stale_reasoning = (role == "assistant" and isinstance(content, dict)
                           and content.get("content_type") in {"thoughts", "reasoning_recap"}
                           and message.get("status") == "in_progress"
                           and message.get("end_turn") is not True)
        stale_tool_call = False
        if (allow_completed_tool_call and role == "assistant" and isinstance(content, dict)
                and content.get("content_type") == "code" and isinstance(content.get("text"), str)
                and message.get("status") == "in_progress" and message.get("end_turn") is not True
                and message.get("channel") in (None, "analysis")
                and isinstance(message.get("recipient"), str) and message["recipient"].strip()
                and message["recipient"].strip() != "all"):
            following = children.get(node_id, [])
            tool_node = mapping.get(following[0]) if len(following) == 1 else None
            tool = tool_node.get("message") if isinstance(tool_node, dict) else None
            # The persisted view can place one assistant thought snapshot
            # between a call and its output. It is transparent only on the
            # same unique chain; another call, branch or terminal is not.
            if (isinstance(tool, dict) and tool.get("id") == following[0]
                    and isinstance(tool.get("author"), dict)
                    and tool["author"].get("role") == "assistant"
                    and isinstance(tool.get("content"), dict)
                    and tool["content"].get("content_type") == "thoughts"
                    and tool.get("status") in {"in_progress", "finished_successfully"}
                    and tool.get("end_turn") is not True
                    and tool.get("channel") in (None, "analysis")
                    and tool.get("recipient") in (None, "all")):
                following = children.get(following[0], [])
                tool_node = mapping.get(following[0]) if len(following) == 1 else None
                tool = tool_node.get("message") if isinstance(tool_node, dict) else None
            # Some completed tool calls retain an active assistant code node.
            # Require its exact, completed tool output before considering the
            # later nonempty final. This exception never establishes empty
            # reply evidence or permits another generation attempt.
            stale_tool_call = bool(
                isinstance(tool, dict) and tool.get("id") == following[0]
                and isinstance(tool.get("author"), dict)
                and tool["author"].get("role") == "tool"
                and tool["author"].get("name") == message["recipient"]
                and tool.get("status") == "finished_successfully"
                and isinstance(tool.get("content"), dict)
                and tool["content"].get("content_type") == "execution_output"
                and isinstance(tool["content"].get("text"), str)
                and isinstance(tool.get("metadata"), dict)
                and tool["metadata"].get("is_complete") is True
            )
        if (role not in {"assistant", "tool"}
                or message.get("status") != "finished_successfully" and not (stale_reasoning or stale_tool_call)):
            return None
        if role == "assistant" and message.get("end_turn") is True:
            completed_commentary = (allow_completed_commentary and message.get("channel") == "commentary"
                and isinstance(content, dict) and content.get("content_type") == "text"
                and isinstance(content.get("parts"), list) and all(isinstance(p, str) for p in content["parts"])
                and any(p.strip() for p in content["parts"]))
            if message.get("channel") not in {None, "final"} and not completed_commentary:
                return None
            for successor in children.get(node_id, []):
                following = mapping.get(successor, {}).get("message")
                if (not isinstance(following, dict) or following.get("id") != successor
                        or not isinstance(following.get("author"), dict)
                        or following["author"].get("role") != "user"):
                    return None
            return {"conversation_id": conversation_id, "request_message_id": request_message_id,
                    "final_message_id": node_id, "observed_at": time.time()}


def _completed_external_successor(mapping, children, request_message_id, current_node, conversation_id):
    """A later user turn completed on the same unambiguous current chain.

    This is diagnostic evidence only: it neither ends the original turn nor
    makes the later answer an answer to the original request.
    """
    if not isinstance(current_node, str):
        return False
    node_id, seen, later_user = current_node, set(), None
    while node_id != request_message_id:
        if not node_id or node_id in seen or len(seen) >= 128:
            return False
        seen.add(node_id)
        node = mapping.get(node_id)
        message = node.get("message") if isinstance(node, dict) else None
        if not isinstance(message, dict) or message.get("id") != node_id:
            return False
        author = message.get("author")
        role = author.get("role") if isinstance(author, dict) else None
        if role not in {"user", "assistant", "tool"}:
            return False
        if role == "user" and later_user is None:
            later_user = node_id
        parent = node.get("parent")
        if not isinstance(parent, str) or children.get(parent) != [node_id]:
            return False
        node_id = parent
    if not later_user:
        return False
    ended = _completed_request_turn(mapping, children, later_user, conversation_id)
    return bool(ended and ended["final_message_id"] == current_node)


def _empty_reply_evidence(mapping, children, request_message_id, conversation_id, current_node):
    """Observe an empty original final and a safe current cursor, not remote cancellation.

    Some completed HTTP responses leave an empty final marked in_progress. The
    task service separately requires durable local response-end evidence before
    allowing the user's one same-conversation retry of this observation.
    """
    node_id, seen = request_message_id, {request_message_id}
    while len(seen) <= 128:
        successors = children.get(node_id, [])
        if len(successors) != 1:
            return None
        node_id = successors[0]
        if node_id in seen:
            return None
        seen.add(node_id)
        node = mapping.get(node_id)
        message = node.get("message") if isinstance(node, dict) else None
        if not isinstance(message, dict) or message.get("id") != node_id:
            return None
        author = message.get("author")
        role = author.get("role") if isinstance(author, dict) else None
        if role not in {"assistant", "tool"}:
            return None
        following = children.get(node_id, [])
        final = (role == "assistant" and message.get("channel") in {None, "final"}
                 and (not following or all(
                     isinstance(mapping.get(child), dict)
                     and isinstance(mapping[child].get("message"), dict)
                     and mapping[child]["message"].get("id") == child
                     and isinstance(mapping[child]["message"].get("author"), dict)
                     and mapping[child]["message"]["author"].get("role") == "user"
                     for child in following)))
        content = message.get("content")
        if final:
            parts = content.get("parts") if isinstance(content, dict) else None
            if (not isinstance(content, dict) or content.get("content_type") != "text"
                    or not isinstance(parts, list) or not all(isinstance(p, str) for p in parts)
                    or "".join(parts).strip()):
                return None
            if not isinstance(current_node, str):
                return None
            if current_node != node_id and not _completed_external_successor(
                    mapping, children, node_id, current_node, conversation_id):
                return None
            return {"conversation_id": conversation_id, "request_message_id": request_message_id,
                    "final_message_id": node_id, "retry_parent_message_id": current_node,
                    "observed_at": time.time()}
        # Reasoning snapshots can retain in_progress even after a later final
        # exists. Only known assistant reasoning envelopes get that allowance;
        # tools/code must still be complete. The caller also proves local close.
        stale_reasoning = (role == "assistant" and isinstance(content, dict)
                           and content.get("content_type") in {"thoughts", "reasoning_recap"}
                           and message.get("status") == "in_progress")
        if (message.get("status") != "finished_successfully" and not stale_reasoning or not isinstance(content, dict)
                or content.get("content_type") not in OBSERVED_CONTENT_TYPES
                or role == "tool" and content.get("content_type") != "execution_output"):
            return None
    return None


def is_recovery_image_pointer(value: object) -> bool:
    # Retain an upstream reference, never a download URL or tool arguments.
    return isinstance(value, str) and bool(re.fullmatch(
        r"(?:file-service|sediment)://[A-Za-z0-9_-]{1,200}", value,
    ))


def _completed_image_result(mapping, children, request_message_id, conversation_id):
    """Require one completed lineage from this user through an image to an empty final.

    A generic empty answer, a tool invocation, a sibling result, or an input
    attachment is not proof of this outcome. Later user turns are not ours.
    """
    node_id = request_message_id
    visited = {node_id}
    artifacts = []
    while True:
        successors = children.get(node_id, [])
        if len(successors) != 1:
            return None
        node_id = successors[0]
        if node_id in visited:
            return None
        visited.add(node_id)
        node = mapping.get(node_id)
        message = node.get("message") if isinstance(node, dict) else None
        if not isinstance(message, dict) or message.get("id") != node_id:
            return None
        author = message.get("author")
        role = author.get("role") if isinstance(author, dict) else None
        if role not in {"assistant", "tool"} or message.get("status") != "finished_successfully":
            return None
        content = message.get("content")
        if not isinstance(content, dict):
            return None
        parts = content.get("parts")
        if role == "tool" and content.get("content_type") == "multimodal_text" and isinstance(parts, list):
            for part in parts:
                if (isinstance(part, dict) and part.get("content_type") == "image_asset_pointer"
                        and is_recovery_image_pointer(part.get("asset_pointer"))):
                    artifact = {"tool_message_id": node_id, "asset_pointer": part["asset_pointer"]}
                    if artifact not in artifacts:
                        artifacts.append(artifact)
        if role == "assistant" and message.get("end_turn") is True:
            if (message.get("channel") != "final" or content.get("content_type") != "text"
                    or not isinstance(parts, list) or not all(isinstance(part, str) for part in parts)
                    or "".join(parts).strip() or not artifacts):
                return None
            # Nothing in the original result may still branch or continue
            # after this final. Subsequent user turns remain separate.
            for successor in children.get(node_id, []):
                following = mapping.get(successor, {}).get("message")
                if (not isinstance(following, dict) or following.get("id") != successor
                        or not isinstance(following.get("author"), dict)
                        or following["author"].get("role") != "user"):
                    return None
            return {
                "conversation_id": conversation_id,
                "request_message_id": request_message_id,
                "final_message_id": node_id,
                "artifacts": artifacts,
            }


RECOVERY_CONVERSATION_SCAN_FIELD = "_recovery_conversation_scan"
RECOVERY_CONVERSATION_COVERAGE_VERSION_FIELD = "_recovery_conversation_coverage_version"
_TEXT_FAILURE_PHASES = frozenset({"stream_open", "stream_event", "result_check", "cursor_read", "runner"})
_TEXT_FAILURE_CATEGORIES = frozenset({
    "http", "timeout", "transport", "parse", "empty_result", "provider_error", "other",
})
_TEXT_HTTP_REQUEST_STAGES = {
    "bootstrap": "bootstrap",
    "chat_requirements_prepare": "chat_requirements_prepare",
    "chat_requirements_finalize": "chat_requirements_finalize",
    "/backend-api/conversation": "conversation",
    "/backend-anon/conversation": "conversation",
}
_TEXT_HTTP_REQUEST_STAGE_VALUES = frozenset({*_TEXT_HTTP_REQUEST_STAGES.values(), "unknown"})
_TEXT_HTTP_ERROR_FORMS = frozenset({
    "error", "error_text", "detail_error", "detail_error_text", "detail",
    "detail_text", "validation", "message_text", "object", "list", "text",
    "empty", "opaque",
})
_TEXT_422_FIELDS = frozenset({
    "action", "messages", "model", "parent_message_id", "conversation_id",
    "conversation_mode", "thinking_effort", "history_and_training_disabled",
    "force_use_sse", "supported_encodings", "system_hints", "timezone",
    "websocket_request_id",
})


def _text_422_field(value: object) -> str:
    if not isinstance(value, str) or len(value) > 96:
        return ""
    top_level = value.split(".", 1)[0].split("[", 1)[0]
    return top_level if top_level in _TEXT_422_FIELDS else ""


def _text_http_diagnostic(exc: UpstreamHTTPError) -> dict[str, str]:
    """Keep response shape; only 422 may name a validated field. Never keep text."""
    body = exc.body
    candidates: list[object] = []
    form = "opaque"
    if isinstance(body, dict):
        form = "object"
        candidates.append(body.get("param"))
        error = body.get("error")
        detail = body.get("detail")
        if isinstance(error, dict):
            form = "error"
            candidates.append(error.get("param"))
        elif isinstance(detail, dict):
            nested = detail.get("error")
            form = (
                "detail_error" if isinstance(nested, dict)
                else "detail_error_text" if isinstance(nested, str) else "detail"
            )
            candidates.append(detail.get("param"))
            if isinstance(nested, dict):
                candidates.append(nested.get("param"))
        elif isinstance(detail, list):
            form = "validation"
            for item in detail[:20]:
                if not isinstance(item, dict):
                    continue
                location = item.get("loc")
                if isinstance(location, list) and len(location) >= 2 and location[0] == "body":
                    candidates.append(location[1])
        elif isinstance(detail, str):
            form = "detail_text"
        elif isinstance(error, str):
            form = "error_text"
        elif isinstance(body.get("message"), str):
            form = "message_text"
    elif isinstance(body, list):
        form = "list"
    elif isinstance(body, str):
        form = "text" if body else "empty"
    elif body is None:
        form = "empty"
    fields = {_text_422_field(value) for value in candidates} - {""} if exc.status_code == 422 else set()
    return {
        "original_upstream_error_form": form,
        **({"original_upstream_rejected_field": next(iter(fields))} if len(fields) == 1 else {}),
    }


def _text_failure_category(exc: Exception) -> str:
    if isinstance(exc, UpstreamHTTPError):
        return "http"
    name = type(exc).__name__
    if isinstance(exc, TimeoutError) or name in {"Timeout", "ReadTimeout", "ConnectTimeout"}:
        return "timeout"
    if isinstance(exc, ConnectionError) or name in {"ProxyError", "DNSError", "ConnectionError", "SSLError"}:
        return "transport"
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return "parse"
    return "other"


# Only typed, bounded evidence is persisted. Exception text, response bodies,
# URLs, tokens and conversation titles must never enter a recovery diagnostic.
_READ_PHASES = frozenset({"conversation_list", "conversation_detail", "matched_conversation"})
_READ_CATEGORIES = frozenset({"http", "timeout", "transport", "parse", "other"})
_READ_CODES = frozenset({"RECOVERY_READ_FAILED", "CONVERSATION_BINDING_CONTRACT_INVALID",
                         "CONVERSATION_BINDING_MISMATCH"})


def safe_recovery_read_error(value):
    if not isinstance(value, dict) or set(value) != {
        "phase", "category", "http_status", "retry_after_seconds", "candidate_ref",
    }:
        return None
    status, delay = value["http_status"], value["retry_after_seconds"]
    ref = value["candidate_ref"]
    if (not isinstance(value["phase"], str) or value["phase"] not in _READ_PHASES
            or not isinstance(value["category"], str) or value["category"] not in _READ_CATEGORIES
            or status is not None and (type(status) is not int or not 100 <= status <= 599)
            or delay is not None and (type(delay) is not int or not 0 <= delay <= 2147483647)
            or ref is not None and (not isinstance(ref, str) or not re.fullmatch(r"[0-9a-f]{16}", ref))):
        return None
    return dict(value)


def safe_scan_failures(value, uninspected_ids):
    if not isinstance(value, dict) or len(value) > 100 or set(value) - set(uninspected_ids):
        return None
    result = {}
    for candidate, row in value.items():
        if not isinstance(row, dict) or set(row) != {"code", "error", "attempts", "next_at"}:
            return None
        error = safe_recovery_read_error(row["error"])
        if (not isinstance(row["code"], str) or row["code"] not in _READ_CODES or error is None
                or error["phase"] != "conversation_detail"
                or error["candidate_ref"] != hashlib.sha256(candidate.encode()).hexdigest()[:16]
                or type(row["attempts"]) is not int or not 1 <= row["attempts"] <= 2147483647
                or type(row["next_at"]) not in {int, float}
                or not math.isfinite(row["next_at"]) or row["next_at"] < 0):
            return None
        result[candidate] = {**row, "error": error}
    return result


class ConversationBindingError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "CONVERSATION_BINDING_UNAVAILABLE",
        provider_binding_id: str = "",
        provider_account_identity: str = "",
        conversation_id: str = "",
        parent_message_id: str = "",
        recovery_reason: str = "",
        recovery_scan: dict[str, Any] | None = None,
        recovery_coverage_version: int | None = None,
        recovery_read_error: dict[str, Any] | None = None,
        original_failure_phase: str = "",
        original_http_status: int | None = None,
        original_exception_category: str = "",
        original_upstream_request_stage: str = "",
        original_upstream_error_form: str = "",
        original_upstream_rejected_field: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.provider_binding_id = provider_binding_id
        self.provider_account_identity = provider_account_identity
        self.conversation_id = conversation_id
        self.parent_message_id = parent_message_id
        self.recovery_reason = recovery_reason
        self.recovery_scan = recovery_scan
        self.recovery_read_error = safe_recovery_read_error(recovery_read_error)
        self.recovery_coverage_version = recovery_coverage_version
        self.original_failure_phase = (
            original_failure_phase if original_failure_phase in _TEXT_FAILURE_PHASES else ""
        )
        self.original_http_status = (
            original_http_status
            if type(original_http_status) is int and 100 <= original_http_status <= 599 else None
        )
        self.original_exception_category = (
            original_exception_category if original_exception_category in _TEXT_FAILURE_CATEGORIES else ""
        )
        self.original_upstream_request_stage = (
            original_upstream_request_stage
            if self.original_http_status is not None
            and original_upstream_request_stage in _TEXT_HTTP_REQUEST_STAGE_VALUES else ""
        )
        is_stream_open_http = (
            self.original_failure_phase == "stream_open"
            and self.original_exception_category == "http"
            and self.original_http_status is not None
        )
        self.original_upstream_error_form = (
            original_upstream_error_form
            if is_stream_open_http and original_upstream_error_form in _TEXT_HTTP_ERROR_FORMS else ""
        )
        self.original_upstream_rejected_field = (
            original_upstream_rejected_field
            if is_stream_open_http and self.original_http_status == 422
            and original_upstream_rejected_field in _TEXT_422_FIELDS else ""
        )


class ConversationBindingService:
    RECOVERY_RECENT_CONVERSATION_LIMIT = 20
    RECOVERY_MAX_CONVERSATION_IDS = 100
    RECOVERY_SCAN_TIMEOUT_SECONDS = 20.0
    RECOVERY_LIST_TIMEOUT_SECONDS = 10.0
    RECOVERY_DETAIL_MIN_TIMEOUT_SECONDS = 5.0
    RECOVERY_DISPATCH_CLOCK_SKEW_SECONDS = 30.0
    RECOVERY_COVERAGE_VERSION = 1

    def set_archived(self, body: dict[str, Any], archived: bool) -> dict[str, Any]:
        binding = body["provider_binding_id"]
        if account_service.get_bound_account_identity(binding) != body["provider_account_identity"]:
            raise ConversationBindingError("provider account identity changed", code="CONVERSATION_BINDING_MISMATCH")
        token = account_service.get_bound_text_access_token(binding, model="auto")
        with account_service.conversation_binding_lock(binding, body["client_conversation_id"]):
            backend = OpenAIBackendAPI(access_token=token)
            try:
                # The backend checks the original turn and current cursor before
                # PATCH, then verifies its readback. A second outer GET adds
                # account read pressure without another mutation boundary.
                try:
                    result = backend.set_conversation_archived(body["conversation_id"], body["parent_message_id"], archived)
                except ConversationArchiveCursorMismatch:
                    if body.get("_public_session_ref"):
                        raise ConversationBindingError("original product conversation changed", code="CONVERSATION_BINDING_MISMATCH") from None
                    raise
                return {**body, **result}
            finally:
                backend.close()

    def archive(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.set_archived(body, True)

    def read_text_request(self, receipt: dict[str, Any]) -> dict[str, Any]:
        """Recover only the answer descending from this request's own user turn."""
        binding = receipt["provider_binding_id"]
        if account_service.get_bound_account_identity(binding) != receipt["provider_account_identity"]:
            raise ConversationBindingError(
                "provider account identity changed",
                code="CONVERSATION_BINDING_MISMATCH",
                recovery_reason=TextRecoveryReason.ACCOUNT_IDENTITY_MISMATCH.value,
            )
        token = account_service.get_bound_text_access_token(binding, model="auto")
        with account_service.conversation_binding_lock(binding, receipt["client_conversation_id"]):
            backend = OpenAIBackendAPI(access_token=token)
            try:
                if not str(receipt.get("conversation_id") or "").strip():
                    located_receipt, located_document = self._locate_text_request_conversation(
                        backend, receipt,
                    )
                    try:
                        recovered = self._read_text_request_result(
                            backend, located_receipt, document=located_document,
                        )
                    except ConversationBindingError as exc:
                        exc.recovery_scan = {}
                        raise
                    return {**recovered, RECOVERY_CONVERSATION_SCAN_FIELD: {}}
                try:
                    return self._read_text_request_result(backend, receipt)
                except UpstreamHTTPError as exc:
                    if exc.status_code != 404:
                        raise
                    raise ConversationBindingError(
                        "original conversation is missing",
                        code="CONVERSATION_OUTCOME_UNKNOWN",
                        conversation_id=str(receipt.get("conversation_id") or ""),
                        recovery_reason=TextRecoveryReason.CONVERSATION_NOT_FOUND.value,
                    ) from exc
            finally:
                backend.close()

    @classmethod
    def _locate_text_request_conversation(
        cls,
        backend: OpenAIBackendAPI,
        receipt: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Find one exact request user node within a bounded recent-account scan."""
        request_message_id = str(receipt.get("request_message_id") or "").strip()
        if not request_message_id:
            raise ConversationBindingError(
                "original request message identity is required",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
            )
        deadline = time.monotonic() + cls.RECOVERY_SCAN_TIMEOUT_SECONDS
        scan_identity = cls._recovery_scan_identity(receipt)
        scan: dict[str, Any] | None = None

        def remaining_timeout() -> float:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise cls._scan_incomplete(scan)
            return remaining

        scan = cls._validated_recovery_scan(receipt, scan_identity)
        if scan is None:
            scan = {
                "identity": scan_identity,
                "conversation_ids": [],
                "next_offset": 0,
                "coverage_complete": False,
                "time_order_valid": True,
                "last_update_time": None,
                "next_index": 0,
                "matches": [],
            }

        while not scan["coverage_complete"]:
            # Bound each persisted candidate window, then inspect it before
            # continuing from the accumulated upstream offset in another
            # recovery round. This keeps the receipt small without turning the
            # cap into a permanent stop on older account histories.
            if (
                scan["conversation_ids"]
                and len(scan["conversation_ids"])
                    + cls.RECOVERY_RECENT_CONVERSATION_LIMIT
                    > cls.RECOVERY_MAX_CONVERSATION_IDS
            ):
                break
            list_timeout = remaining_timeout()
            if list_timeout < cls.RECOVERY_DETAIL_MIN_TIMEOUT_SECONDS:
                raise cls._scan_incomplete(scan)
            try:
                recent = backend._list_recent_conversations(
                    limit=cls.RECOVERY_RECENT_CONVERSATION_LIMIT,
                    offset=scan["next_offset"],
                    timeout_secs=min(
                        cls.RECOVERY_LIST_TIMEOUT_SECONDS,
                        list_timeout,
                    ),
                    strict_schema=True,
                )
            except Exception as exc:
                raise cls._scan_read_failed(scan, exc, phase="conversation_list") from exc
            raw_page_ids = [
                str(item.get("id") or item.get("conversation_id") or "").strip()
                for item in recent
            ]
            page_ids = list(dict.fromkeys(raw_page_ids))
            page_ids = [item for item in page_ids if item not in scan["conversation_ids"]]

            page_times = [cls._conversation_update_time(item) for item in recent]
            if scan["time_order_valid"] and all(value is not None for value in page_times):
                numeric_times = [float(value) for value in page_times]
                ordered = not (
                    any(left < right for left, right in zip(numeric_times, numeric_times[1:]))
                    or (numeric_times and scan["last_update_time"] is not None
                        and numeric_times[0] > scan["last_update_time"])
                )
                if ordered and numeric_times:
                    scan["last_update_time"] = numeric_times[-1]
                elif not ordered:
                    # Offset pagination can drift while newer conversations
                    # arrive. Once order is not trustworthy, only a short page
                    # may prove complete coverage.
                    scan["time_order_valid"] = False
                    scan["last_update_time"] = None
            else:
                scan["time_order_valid"] = False
                scan["last_update_time"] = None

            scan["next_offset"] += len(raw_page_ids)
            dispatch_at = cls._receipt_dispatch_time(receipt)
            covered_by_time = (
                scan["time_order_valid"]
                and scan["last_update_time"] is not None
                and dispatch_at is not None
                and scan["last_update_time"]
                    < dispatch_at - cls.RECOVERY_DISPATCH_CLOCK_SKEW_SECONDS
            )
            if covered_by_time:
                # The same trusted time boundary that excludes later pages
                # also excludes this page's old tail. Pair before deduplication
                # so a repeated ID with a recent timestamp remains a candidate.
                cutoff = dispatch_at - cls.RECOVERY_DISPATCH_CLOCK_SKEW_SECONDS
                recent_ids = {
                    conversation_id
                    for conversation_id, updated_at in zip(raw_page_ids, page_times)
                    if updated_at >= cutoff
                }
                page_ids = [item for item in page_ids if item in recent_ids]
            scan["conversation_ids"].extend(page_ids)
            scan["coverage_complete"] = (
                len(recent) < cls.RECOVERY_RECENT_CONVERSATION_LIMIT
                or covered_by_time
            )

        conversation_ids = scan["conversation_ids"]
        matches = scan["matches"]
        failed_reads = scan.get("failed_reads", {})
        visited_this_round = set()
        last_failure = None
        while scan["next_index"] < len(conversation_ids):
            index = scan["next_index"]
            conversation_id = conversation_ids[index]
            # Failed/unread candidates remain in the tail. Inspect each at most
            # once per recovery window; a bad candidate cannot starve later ones.
            if conversation_id in visited_this_round:
                break
            prior_failure = failed_reads.get(conversation_id)
            if prior_failure and time.time() < prior_failure["next_at"]:
                visited_this_round.add(conversation_id)
                conversation_ids.append(conversation_ids.pop(index))
                continue
            detail_timeout = remaining_timeout()
            if detail_timeout < cls.RECOVERY_DETAIL_MIN_TIMEOUT_SECONDS:
                raise cls._scan_incomplete(scan)
            visited_this_round.add(conversation_id)
            try:
                document = backend._get_conversation(conversation_id, timeout_secs=detail_timeout)
                if not isinstance(document, dict):
                    raise ConversationBindingError("conversation document is invalid",
                                                   code="CONVERSATION_BINDING_CONTRACT_INVALID")
                request_parent_message_id = cls._request_anchor_from_document(
                    document, conversation_id, request_message_id, recovery_scan=scan,
                )
            except Exception as exc:
                failure = cls._scan_read_failed(scan, exc, candidate=conversation_id)
                # Auth and rate limiting affect the account, not only one
                # document. Stop immediately; outer recovery honors Retry-After.
                if failure.recovery_read_error["http_status"] in {401, 403, 429}:
                    raise failure from exc
                attempts = min(2147483647, int((prior_failure or {}).get("attempts", 0)) + 1)
                delay = max(min(900, 30 * 2 ** min(attempts - 1, 5)),
                            failure.recovery_read_error["retry_after_seconds"] or 0)
                failed_reads[conversation_id] = {
                    "code": failure.code, "error": failure.recovery_read_error,
                    "attempts": attempts, "next_at": time.time() + delay,
                }
                scan["failed_reads"] = failed_reads
                conversation_ids.append(conversation_ids.pop(index))
                last_failure = failure
                continue
            failed_reads.pop(conversation_id, None)
            if not failed_reads:
                scan.pop("failed_reads", None)
            scan["next_index"] += 1
            if request_parent_message_id is not None:
                matches.append({"conversation_id": conversation_id,
                                "request_parent_message_id": request_parent_message_id})
                if len(matches) > 1:
                    break

        if len(matches) > 1:
            raise ConversationBindingError(
                "original request message appears in multiple conversations",
                code="CONVERSATION_BINDING_MISMATCH",
                recovery_scan={},
            )
        if not scan["coverage_complete"]:
            # All candidates in this bounded window were inspected. Retain a
            # found match for uniqueness checking, discard completed unrelated
            # ids, and continue from next_offset on the next short-delay read.
            retained_matches = [str(match["conversation_id"]) for match in matches]
            scan["conversation_ids"] = retained_matches + conversation_ids[scan["next_index"]:]
            scan["next_index"] = len(retained_matches)
            raise cls._scan_incomplete(scan)
        if failed_reads:
            # A match behind a failed read is provisional, not proven unique.
            # Never turn unread evidence into absence, success or freed capacity.
            first = min(failed_reads.values(), key=lambda row: row["next_at"])
            failure = last_failure or ConversationBindingError(
                "candidate reads remain unresolved", code=first["code"],
                recovery_scan=scan, recovery_read_error=first["error"],
            )
            failure.retry_after = max(1, math.ceil(first["next_at"] - time.time()))
            raise failure
        if not matches:
            raise ConversationBindingError(
                "original request conversation cannot be attributed uniquely",
                code="CONVERSATION_OUTCOME_UNKNOWN",
                recovery_reason=(
                    TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value
                ),
                recovery_scan={},
                recovery_coverage_version=cls.RECOVERY_COVERAGE_VERSION,
            )
        if scan["next_index"] < len(conversation_ids):
            raise cls._scan_incomplete(scan)

        # Use a fresh bounded window for the exact matched conversation. A full
        # account scan may already have consumed nearly all of the 20-second
        # budget, while the durable recovery claim lasts 60 seconds.
        if receipt.get(RECOVERY_CONVERSATION_SCAN_FIELD) != scan:
            raise cls._scan_incomplete(scan)

        conversation_id = matches[0]["conversation_id"]
        request_parent_message_id = matches[0]["request_parent_message_id"]
        try:
            document = backend._get_conversation(
                conversation_id,
                timeout_secs=remaining_timeout(),
            )
        except UpstreamHTTPError as exc:
            if exc.status_code == 404:
                raise ConversationBindingError(
                    "located request conversation is missing",
                    code="CONVERSATION_OUTCOME_UNKNOWN",
                    conversation_id=conversation_id,
                    recovery_reason=TextRecoveryReason.CONVERSATION_NOT_FOUND.value,
                    recovery_scan={},
                ) from exc
            raise cls._scan_read_failed(scan, exc, phase="matched_conversation", candidate=conversation_id) from exc
        except ConversationBindingError:
            raise
        except Exception as exc:
            raise cls._scan_read_failed(scan, exc, phase="matched_conversation", candidate=conversation_id) from exc
        fresh_request_parent_message_id = cls._request_anchor_from_document(
            document,
            conversation_id,
            request_message_id,
            recovery_scan=scan,
        )
        if fresh_request_parent_message_id is None:
            raise ConversationBindingError(
                "request message anchor is missing",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
                recovery_scan=scan,
            )
        if fresh_request_parent_message_id != request_parent_message_id:
            raise ConversationBindingError(
                "request parent changed after conversation recovery scan",
                code="CONVERSATION_BINDING_MISMATCH",
                conversation_id=conversation_id,
                parent_message_id=fresh_request_parent_message_id,
                recovery_reason=TextRecoveryReason.REQUEST_PARENT_MISMATCH.value,
                recovery_scan={},
            )
        if not _request_parent_matches_receipt(
            receipt, request_parent_message_id, document["mapping"],
        ):
            raise ConversationBindingError(
                "original request parent changed",
                code="CONVERSATION_BINDING_MISMATCH",
                conversation_id=conversation_id,
                parent_message_id=request_parent_message_id,
                recovery_reason=TextRecoveryReason.REQUEST_PARENT_MISMATCH.value,
            )
        return (
            {
                **receipt,
                "conversation_id": conversation_id,
                # Both anchors are exact nodes from the recovered mapping. The
                # request parent validates branch identity; the request node is a
                # usable continuation cursor until a completed answer replaces it.
                "parent_message_id": request_message_id,
                # An empty value is authoritative for a root request. Keeping
                # the key prevents the branch reader from falling back to the
                # request node itself as its expected parent.
                "request_parent_message_id": request_parent_message_id,
            },
            document,
        )

    @staticmethod
    def _recovery_scan_identity(receipt: dict[str, Any]) -> dict[str, str]:
        return {
            key: str(receipt.get(key) or "").strip()
            for key in (
                "provider_binding_id",
                "provider_account_identity",
                "client_conversation_id",
                "request_message_id",
            )
        }

    @staticmethod
    def _conversation_update_time(item: dict[str, Any]) -> float | None:
        value = item.get("update_time") or item.get("updated_at")
        return ConversationBindingService._timestamp(value)

    @staticmethod
    def _timestamp(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            if not isinstance(value, str):
                return None
            normalized = value.strip()
            if normalized.endswith(("Z", "z")):
                normalized = normalized[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return None
            timestamp = parsed.timestamp()
        return timestamp if math.isfinite(timestamp) and timestamp > 0 else None

    @staticmethod
    def _receipt_dispatch_time(receipt: dict[str, Any]) -> float | None:
        value = receipt.get("started_at") or receipt.get("created_at")
        return ConversationBindingService._timestamp(value)

    @classmethod
    def _validated_recovery_scan(
        cls,
        receipt: dict[str, Any],
        expected_identity: dict[str, str],
    ) -> dict[str, Any] | None:
        value = receipt.get(RECOVERY_CONVERSATION_SCAN_FIELD)
        if not isinstance(value, dict) or value.get("identity") != expected_identity:
            return None
        conversation_ids = value.get("conversation_ids")
        next_offset = value.get("next_offset")
        coverage_complete = value.get("coverage_complete")
        time_order_valid = value.get("time_order_valid")
        last_update_time = value.get("last_update_time")
        next_index = value.get("next_index")
        matches = value.get("matches")
        if (
            set(value) - {"failed_reads"} != {
                "identity", "conversation_ids", "next_offset", "coverage_complete",
                "time_order_valid", "last_update_time", "next_index", "matches",
            }
            or not isinstance(conversation_ids, list)
            or len(conversation_ids) > cls.RECOVERY_MAX_CONVERSATION_IDS
            or any(not isinstance(item, str) or not item or len(item) > 200 for item in conversation_ids)
            or len(set(conversation_ids)) != len(conversation_ids)
            or not isinstance(next_offset, int) or isinstance(next_offset, bool)
            or next_offset < len(conversation_ids)
            or not isinstance(coverage_complete, bool)
            or not isinstance(time_order_valid, bool)
            or (last_update_time is not None and (
                isinstance(last_update_time, bool)
                or not isinstance(last_update_time, (int, float))
                or not math.isfinite(last_update_time)
                or last_update_time <= 0
            ))
            or (not time_order_valid and last_update_time is not None)
            or not isinstance(next_index, int) or isinstance(next_index, bool)
            or next_index < 0 or next_index > len(conversation_ids)
            or not isinstance(matches, list) or len(matches) > 2
        ):
            return None
        failures = safe_scan_failures(value.get("failed_reads", {}), conversation_ids[next_index:])
        if failures is None:
            return None
        safe_matches = []
        for match in matches:
            if not isinstance(match, dict) or set(match) != {"conversation_id", "request_parent_message_id"}:
                return None
            conversation_id = match.get("conversation_id")
            parent_message_id = match.get("request_parent_message_id")
            if (
                not isinstance(conversation_id, str)
                or conversation_id not in conversation_ids
                or not isinstance(parent_message_id, str)
                or len(parent_message_id) > 200
            ):
                return None
            safe_matches.append({
                "conversation_id": conversation_id,
                "request_parent_message_id": parent_message_id,
            })
        return {
            "identity": expected_identity,
            "conversation_ids": list(conversation_ids),
            "next_offset": next_offset,
            "coverage_complete": coverage_complete,
            "time_order_valid": time_order_valid,
            "last_update_time": last_update_time,
            "next_index": next_index,
            "matches": safe_matches,
            **({"failed_reads": failures} if failures else {}),
        }

    @staticmethod
    def _scan_incomplete(scan: dict[str, Any] | None = None) -> ConversationBindingError:
        return ConversationBindingError(
            "bounded recent conversation scan is incomplete",
            code="CONVERSATION_OUTCOME_UNKNOWN",
            recovery_reason=TextRecoveryReason.REQUEST_CONVERSATION_SCAN_INCOMPLETE.value,
            recovery_scan=scan,
        )

    @staticmethod
    def _scan_read_failed(scan=None, cause=None, *, phase="conversation_detail", candidate=None):
        status = getattr(cause, "status_code", None)
        status = status if type(status) is int and 100 <= status <= 599 else None
        delay = getattr(cause, "retry_after", None)
        delay = (math.ceil(delay) if type(delay) in {int, float} and math.isfinite(delay)
                 and 0 <= delay <= 2147483647 else None)
        category = _text_failure_category(cause) if cause is not None else "other"
        if isinstance(cause, ConversationBindingError):
            category = "parse"
        code = getattr(cause, "code", "RECOVERY_READ_FAILED")
        return ConversationBindingError(
            "bounded recent conversation read failed",
            code=code if code in _READ_CODES else "RECOVERY_READ_FAILED",
            recovery_scan=scan,
            recovery_read_error={"phase": phase, "category": category,
                "http_status": status, "retry_after_seconds": delay,
                "candidate_ref": hashlib.sha256(candidate.encode()).hexdigest()[:16] if candidate else None},
        )

    @staticmethod
    def _request_anchor_from_document(
        document: dict[str, Any],
        conversation_id: str,
        request_message_id: str,
        *,
        recovery_scan: dict[str, Any],
    ) -> str | None:
        returned_conversation_id = str(document.get("conversation_id") or conversation_id).strip()
        mapping = document.get("mapping")
        request_node = mapping.get(request_message_id) if isinstance(mapping, dict) else None
        request_message = request_node.get("message") if isinstance(request_node, dict) else None
        request_author = request_message.get("author") if isinstance(request_message, dict) else None
        if returned_conversation_id != conversation_id:
            raise ConversationBindingError(
                "conversation identity changed during request recovery",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
                recovery_scan=recovery_scan,
            )
        if not isinstance(mapping, dict):
            raise ConversationBindingError(
                "conversation mapping is missing or invalid",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
                recovery_scan=recovery_scan,
            )
        if request_node is None:
            return None
        if (
            not isinstance(request_node, dict)
            or not isinstance(request_message, dict)
            or request_message.get("id") != request_message_id
            or not isinstance(request_author, dict)
            or request_author.get("role") != "user"
        ):
            raise ConversationBindingError(
                "request message anchor is invalid",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
                recovery_scan=recovery_scan,
            )
        return str(request_node.get("parent") or "").strip()

    def read_text(self, body: dict[str, Any]) -> dict[str, Any]:
        """Read an already-issued cursor on its bound account; never send a message."""
        # This direct, non-durable GET has the same 60-second budget for
        # queueing and HTTP. Do not leave a read queued after its caller gives
        # up; durable request recovery uses its own persisted scheduling.
        deadline = time.monotonic() + 60.0
        keys = ("provider_binding_id", "provider_account_identity", "client_conversation_id",
                "conversation_id", "parent_message_id")
        if any(not isinstance(body.get(key), str) or not body[key].strip() for key in keys):
            raise ConversationBindingError("original text cursor is required", code="CONVERSATION_BINDING_CONTRACT_INVALID")
        binding_id = body["provider_binding_id"]
        if account_service.get_bound_account_identity(binding_id) != body["provider_account_identity"]:
            raise ConversationBindingError("provider account identity changed", code="CONVERSATION_BINDING_MISMATCH")
        token = account_service.get_bound_text_access_token(binding_id, model="auto")
        with account_service.conversation_binding_lock(binding_id, body["client_conversation_id"]):
            backend = OpenAIBackendAPI(access_token=token)
            try:
                first_error = None
                for attempt in range(2):
                    try:
                        retry_options = ({"reuse_read_credit": True}
                                         if attempt and self._unsent_direct_read_connection(first_error) else {})
                        return self._read_text_result(backend, body, deadline_monotonic=deadline,
                                                      connect_timeout_secs=10.0,
                                                      minimum_budget_secs=10.0, **retry_options)
                    except AccountReadRetryBudgetInsufficient:
                        if first_error is not None:
                            raise first_error from None
                        raise
                    except Exception as exc:
                        if (attempt or time.monotonic() >= deadline
                                or not self._retryable_direct_read_connection(exc)):
                            raise
                        first_error = exc
            finally:
                backend.close()

    @staticmethod
    def _retryable_direct_read_connection(exc):
        """Only one new connection for a failed, read-only original cursor GET.

        This does not apply to durable task recovery, polling, archive or sends.
        A timeout after HTTP/partial data and certificate/auth/rate errors are
        not retryable here. A reused TLS connection has num_connects=0 and
        cannot be mistaken for a new handshake that hit its connection cap.
        These observations do not prove that the server received no request;
        retry safety comes from reading the same cursor without any mutation.
        """
        try:
            if not isinstance(exc, RequestException) or exc.code not in (28, 35):
                return False
            response = exc.response
            if response is not None and response.status_code:
                return False
            infos = getattr(response, "infos", {}) or {}
            if infos.get(CurlInfo.HTTP_CONNECTCODE) in (401, 403, 407, 429):
                return False
            return (type(infos.get(CurlInfo.NUM_CONNECTS)) is int
                    and infos[CurlInfo.NUM_CONNECTS] > 0
                    and type(infos.get(CurlInfo.APPCONNECT_TIME)) in (int, float)
                    and infos.get(CurlInfo.APPCONNECT_TIME) == 0
                    and type(infos.get(CurlInfo.SIZE_DOWNLOAD_T)) is int
                    and infos.get(CurlInfo.SIZE_DOWNLOAD_T) == 0)
        except Exception:
            return False

    @staticmethod
    def _unsent_direct_read_connection(exc):
        """A bounded reconnect may use the credit already paid by this GET.

        Missing native counters keep the ordinary paced retry. This never
        refunds a shared bucket or changes a task's submission/UNKNOWN state.
        """
        if not ConversationBindingService._retryable_direct_read_connection(exc):
            return False
        try:
            infos = exc.response.infos
            zero = (CurlInfo.REQUEST_SIZE, CurlInfo.SIZE_UPLOAD_T, CurlInfo.EARLYDATA_SENT_T,
                    CurlInfo.REDIRECT_COUNT, CurlInfo.RESPONSE_CODE, CurlInfo.HTTP_VERSION)
            return (infos.get(CurlInfo.NUM_CONNECTS) == 1
                    and type(infos.get(CurlInfo.HTTP_CONNECTCODE)) is int
                    and infos[CurlInfo.HTTP_CONNECTCODE] in (0, 200)
                    and all(type(infos.get(key)) is int and infos[key] == 0 for key in zero))
        except Exception:
            return False

    @staticmethod
    def _read_text_result(
        backend: OpenAIBackendAPI, cursor: dict[str, Any], *, deadline_monotonic: float | None = None,
        connect_timeout_secs: float | None = None,
        minimum_budget_secs: float | None = None,
        reuse_read_credit: bool = False,
    ) -> dict[str, Any]:
        options = {} if deadline_monotonic is None else {"deadline_monotonic": deadline_monotonic}
        if connect_timeout_secs is not None:
            options["connect_timeout_secs"] = connect_timeout_secs
        if minimum_budget_secs is not None:
            options["minimum_budget_secs"] = minimum_budget_secs
        if reuse_read_credit:
            options["reuse_read_credit"] = True
        document = backend._get_conversation(cursor["conversation_id"], **options)
        if document.get("conversation_id", cursor["conversation_id"]) != cursor["conversation_id"]:
            raise ConversationBindingError("conversation identity changed", code="CONVERSATION_BINDING_MISMATCH")
        mapping = document.get("mapping") or {}
        current = str(document.get("current_node") or "")
        node_id = current
        visited: set[str] = set()
        while node_id and node_id not in visited:
            visited.add(node_id)
            node = mapping.get(node_id) or {}
            if node_id == cursor["parent_message_id"]:
                break
            # A later user turn is not the result of the original request.
            if (node.get("message") or {}).get("author", {}).get("role") == "user":
                raise ConversationBindingError("original turn was superseded", code="CONVERSATION_BINDING_MISMATCH")
            node_id = str(node.get("parent") or "")
        if node_id != cursor["parent_message_id"]:
            raise ConversationBindingError("original turn is not on the active branch", code="CONVERSATION_BINDING_MISMATCH")
        message = (mapping.get(current) or {}).get("message") or {}
        result = {key: cursor[key] for key in ("provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id", "parent_message_id")}
        if (message.get("id") != current or message.get("author", {}).get("role") != "assistant"
                or message.get("status") != "finished_successfully" or message.get("end_turn") is not True
                or message.get("channel") not in (None, "final")):
            return {**result, "binding_status": "unknown", "status": "running"}
        content = message.get("content") or {}
        parts = content.get("parts")
        if content.get("content_type") != "text" or not isinstance(parts, list) or not parts or not all(isinstance(part, str) for part in parts):
            return {**result, "binding_status": "unknown", "status": "running"}
        text = "".join(parts).strip()
        if not text:
            return {**result, "binding_status": "unknown", "status": "running"}
        return {**result, "binding_status": "bound", "status": "succeeded", "parent_message_id": current, "content": text}

    @staticmethod
    def _read_text_request_result(
        backend: OpenAIBackendAPI,
        receipt: dict[str, Any],
        *,
        document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Read the exact request branch without requiring it to be current.

        A later user turn can make the original request no longer reachable
        from ``current_node`` even though its completed answer is still in the
        conversation mapping. This reader starts at the persisted request user
        node, walks only assistant/tool descendants, and stops at later user
        nodes. It never changes the receipt's binding or cursor.
        """
        conversation_id = str(receipt.get("conversation_id") or "").strip()
        request_message_id = str(receipt.get("request_message_id") or "").strip()
        document = document if document is not None else backend._get_conversation(conversation_id)
        returned_conversation_id = str(document.get("conversation_id") or conversation_id).strip()
        if returned_conversation_id != conversation_id:
            raise ConversationBindingError(
                "conversation identity changed",
                code="CONVERSATION_BINDING_MISMATCH",
                conversation_id=returned_conversation_id,
                recovery_reason=TextRecoveryReason.CONVERSATION_ID_MISMATCH.value,
            )

        if "mapping" not in document or not isinstance(document.get("mapping"), dict):
            raise ConversationBindingError(
                "conversation mapping is missing or invalid",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
                conversation_id=conversation_id,
            )
        mapping = document["mapping"]
        request_node = mapping.get(request_message_id)
        result = {key: receipt[key] for key in (
            "provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id",
            "parent_message_id", "request_parent_message_id",
        ) if key in receipt and (
            key != "parent_message_id"
            or str(receipt[key] or "").strip()
        )}
        if request_message_id and request_node is None:
            current_node_id = str(document.get("current_node") or "").strip()
            current_node = mapping.get(current_node_id)
            current_message = current_node.get("message") if isinstance(current_node, dict) else None
            current_author = current_message.get("author") if isinstance(current_message, dict) else None
            current_role = str(current_author.get("role") or "").strip().lower() \
                if isinstance(current_author, dict) else ""
            current_status = str(current_message.get("status") or "").strip().lower() \
                if isinstance(current_message, dict) else ""
            if current_role in {"assistant", "tool"} and current_status in _ACTIVE_TEXT_RESULT_STATUSES:
                return {
                    **result,
                    "binding_status": "unknown",
                    "status": "running",
                    "recovery_reason": TextRecoveryReason.REQUEST_RESULT_INCOMPLETE.value,
                }
        request_message = request_node.get("message") if isinstance(request_node, dict) else None
        request_author = request_message.get("author") if isinstance(request_message, dict) else None
        request_role = request_author.get("role") if isinstance(request_author, dict) else None
        if (not request_message_id or not isinstance(request_node, dict)
                or not isinstance(request_message, dict)
                or request_message.get("id") != request_message_id
                or request_role != "user"):
            raise ConversationBindingError(
                "original request user turn is missing",
                code="CONVERSATION_BINDING_MISMATCH",
                conversation_id=conversation_id,
                recovery_reason=TextRecoveryReason.REQUEST_MESSAGE_NOT_FOUND.value,
            )

        request_parent = str(request_node.get("parent") or "").strip()
        if not _request_parent_matches_receipt(receipt, request_parent, mapping):
            raise ConversationBindingError(
                "original request parent changed",
                code="CONVERSATION_BINDING_MISMATCH",
                conversation_id=conversation_id,
                parent_message_id=request_parent,
                recovery_reason=TextRecoveryReason.REQUEST_PARENT_MISMATCH.value,
            )

        children: dict[str, list[str]] = {}
        for raw_node_id, raw_node in mapping.items():
            if not isinstance(raw_node, dict):
                continue
            node_id = str(raw_node_id)
            parent = str(raw_node.get("parent") or "").strip()
            if parent:
                children.setdefault(parent, []).append(node_id)

        candidates: list[tuple[str, str]] = []
        visited: set[str] = set()
        later_user_seen = False
        active_result_seen = False
        terminal_empty_seen = False
        pending = list(children.get(request_message_id, []))
        while pending:
            node_id = pending.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            node = mapping.get(node_id)
            if not isinstance(node, dict):
                continue
            message = node.get("message") or {}
            if not isinstance(message, dict):
                message = {}
            author = message.get("author") or {}
            role = str(author.get("role") or "").strip().lower() if isinstance(author, dict) else ""
            if role == "user":
                later_user_seen = True
                continue
            if role and role not in {"assistant", "tool"}:
                continue
            status = str(message.get("status") or "").strip().lower()
            if status in _ACTIVE_TEXT_RESULT_STATUSES:
                active_result_seen = True
            if role == "assistant":
                content = message.get("content") or {}
                parts = content.get("parts") if isinstance(content, dict) else None
                terminal = status == "finished_successfully" and message.get("end_turn") is True
                text = "".join(parts).strip() if (
                    message.get("id") == node_id
                    and terminal
                    and message.get("channel") in (None, "final")
                    and isinstance(parts, list)
                    and parts
                    and all(isinstance(part, str) for part in parts)
                    and content.get("content_type") == "text"
                ) else ""
                if text:
                    candidates.append((node_id, text))
                elif terminal:
                    terminal_empty_seen = True
            pending.extend(children.get(node_id, []))

        if len(candidates) > 1:
            raise ConversationBindingError(
                "multiple completed answers descend from the original request",
                code="CONVERSATION_BINDING_MISMATCH",
                conversation_id=conversation_id,
                recovery_reason=TextRecoveryReason.REQUEST_BRANCH_AMBIGUOUS.value,
            )
        if not candidates:
            from services.generation_completion import retry_cursor
            continuation = retry_cursor(document, receipt)
            non_text_result = None if active_result_seen else _completed_image_result(
                mapping, children, request_message_id, conversation_id,
            )
            if non_text_result:
                return {
                    **result,
                    "binding_status": "bound",
                    "status": "failed",
                    "error_code": "CHAT_RESPONSE_NOT_TEXT",
                    "recovery_reason": TextRecoveryReason.REQUEST_RESULT_NON_TEXT.value,
                    "parent_message_id": non_text_result["final_message_id"],
                    NON_TEXT_RESULT_FIELD: non_text_result,
                }
            advanced = active_result_seen and later_user_seen and _completed_external_successor(
                mapping, children, request_message_id, document.get("current_node"), conversation_id)
            if advanced:
                recovery_reason = TextRecoveryReason.REQUEST_CONVERSATION_ADVANCED.value
            elif active_result_seen:
                recovery_reason = TextRecoveryReason.REQUEST_RESULT_INCOMPLETE.value
            elif terminal_empty_seen:
                recovery_reason = TextRecoveryReason.REQUEST_RESULT_TERMINAL_EMPTY.value
            elif later_user_seen:
                recovery_reason = TextRecoveryReason.REQUEST_BRANCH_SUPERSEDED.value
            else:
                recovery_reason = TextRecoveryReason.REQUEST_RESULT_NOT_FOUND.value
            return {
                **result,
                "binding_status": "unknown",
                "status": "running" if active_result_seen and not advanced else "unknown",
                "recovery_reason": recovery_reason,
                "_retry_cursor": continuation,
                **({RESULT_OBSERVATION_FIELD: observation} if active_result_seen
                   and (observation := _result_observation(mapping, children, request_message_id, conversation_id)) else {}),
                **({TURN_END_EVIDENCE_FIELD: ended} if not active_result_seen and terminal_empty_seen
                   and (ended := _completed_request_turn(mapping, children, request_message_id, conversation_id)) else {}),
                **({"_empty_reply_evidence": empty} if (empty := _empty_reply_evidence(
                    mapping, children, request_message_id, conversation_id, document.get("current_node"))) else {}),
            }
        parent_message_id, text = candidates[0]
        ended = _completed_request_turn(mapping, children, request_message_id, conversation_id,
                                        allow_completed_tool_call=True)
        if not ended or ended["final_message_id"] != parent_message_id:
            # One completed text message is insufficient when another branch
            # remains active or ambiguous. Use the same positive evidence as
            # empty-result recovery before declaring this request complete.
            return {
                **result, "binding_status": "unknown",
                "status": "running" if active_result_seen else "unknown",
                "recovery_reason": (TextRecoveryReason.REQUEST_RESULT_INCOMPLETE.value if active_result_seen
                                    else TextRecoveryReason.REQUEST_BRANCH_AMBIGUOUS.value),
            }
        search_result = None
        if (receipt.get("_chat_recovery") or {}).get("kind") == "search":
            search_result = backend._search_result_from_message(conversation_id, mapping[parent_message_id]["message"])
            text = search_result["answer"]
        return {
            **result,
            "binding_status": "bound",
            "status": "succeeded",
            "parent_message_id": parent_message_id,
            "content": text,
            **({"_search_result": search_result} if search_result is not None else {}),
        }

    @staticmethod
    def _completed_image_tool_parent(mapping, node_id):
        """Recognize the complete user -> image call -> completed image branch.

        A generic tool response or a null end_turn is never terminal evidence.
        Match both declared and reverse edges, including the tool recipient.
        """
        def message_at(key):
            node = mapping.get(key)
            message = node.get("message") if isinstance(node, dict) else None
            return message if (isinstance(message, dict) and message.get("id") == key
                and isinstance(message.get("author"), dict)
                and message.get("status") == "finished_successfully") else None

        def only_child(parent_id, child_id):
            return (mapping[parent_id].get("children") == [child_id]
                and [key for key, node in mapping.items() if node.get("parent") == parent_id] == [child_id])

        tool = message_at(node_id)
        if (not tool or tool["author"].get("role") != "tool" or tool.get("recipient") != "all"
                or mapping[node_id].get("children") != []
                or tool.get("end_turn") is not None and tool.get("end_turn") is not True):
            return False
        content = tool.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(content, dict) or content.get("content_type") != "multimodal_text" or not isinstance(parts, list):
            return False
        pointers = [part for part in parts if isinstance(part, dict)]
        if (not pointers or any(not isinstance(part, (str, dict)) for part in parts)
                or any(part.get("content_type") != "image_asset_pointer"
                    or not is_recovery_image_pointer(part.get("asset_pointer")) for part in pointers)):
            return False
        call_id = mapping[node_id].get("parent")
        call = message_at(call_id) if isinstance(call_id, str) else None
        if (not call or call_id == node_id or call["author"].get("role") != "assistant"
                or call.get("end_turn") is not False or not isinstance(call.get("content"), dict)
                or call["content"].get("content_type") != "code"
                or not isinstance(tool["author"].get("name"), str) or not tool["author"]["name"].strip()
                or call.get("recipient") != tool["author"]["name"] or not only_child(call_id, node_id)):
            return False
        user_id = mapping[call_id].get("parent")
        user = message_at(user_id) if isinstance(user_id, str) else None
        return bool(user and user_id not in {node_id, call_id}
            and user["author"].get("role") == "user" and only_child(user_id, call_id))

    @staticmethod
    def _check_superseded_original(document, conversation_id, parent_message_id, original_user):
        def reject(code):
            raise ConversationBindingError("original request cannot be superseded before send", code=code)
        if (not isinstance(document, dict) or document.get("conversation_id") != conversation_id
                or not isinstance(document.get("mapping"), dict) or not document["mapping"]
                or type(document.get("is_archived")) is not bool):
            reject("CHAT_SUPERSEDE_READ_UNAVAILABLE")
        mapping = document["mapping"]
        if any(not isinstance(node, dict) for node in mapping.values()):
            reject("CHAT_SUPERSEDE_READ_UNAVAILABLE")
        if original_user in mapping or any(isinstance(node.get("message"), dict)
                and node["message"].get("id") == original_user for node in mapping.values()):
            reject("CHAT_SUPERSEDE_ORIGINAL_FOUND")
        parent = mapping.get(parent_message_id) or {}
        message = parent.get("message") or {}
        if (not isinstance(message, dict) or not isinstance(message.get("author"), dict)):
            reject("CHAT_SUPERSEDE_READ_UNAVAILABLE")
        terminal = ((message.get("author") or {}).get("role") == "assistant"
            and message.get("status") == "finished_successfully" and message.get("end_turn") is True)
        if (document.get("current_node") != parent_message_id or message.get("id") != parent_message_id
                or not (terminal or ConversationBindingService._completed_image_tool_parent(mapping, parent_message_id))
                or parent.get("children") or any(node.get("parent") == parent_message_id for node in mapping.values())):
            reject("CHAT_SUPERSEDE_CURSOR_CHANGED")

    @staticmethod
    def _no_final_successor_tail(document, original):
        """Completed code/tool steps are context, not a final text answer.

        This permits one explicitly requested continuation; it does not prove
        the previous upstream turn was cancelled or erase its UNKNOWN receipt.
        """
        def reject():
            raise ConversationBindingError("original no-final branch cannot be continued",
                                           code="CHAT_SUPERSEDE_CURSOR_CHANGED")
        if (not isinstance(document, dict) or document.get("conversation_id") != original.get("conversation_id")
                or document.get("is_archived") is not False or not isinstance(document.get("mapping"), dict)):
            reject()
        mapping = document["mapping"]
        user = original.get("request_message_id")
        expected = original.get("request_parent_message_id") or original.get("_submission_parent_message_id")
        if not user or not expected or user not in mapping or any(not isinstance(x, dict) for x in mapping.values()):
            reject()
        children = {}
        for key, node in mapping.items():
            parent = node.get("parent")
            if parent is not None and not isinstance(parent, str):
                reject()
            children.setdefault(parent, []).append(key)
        node_id, seen, previous_role, context = user, set(), None, []
        while True:
            if node_id in seen:
                reject()
            seen.add(node_id)
            node = mapping[node_id]
            message = node.get("message")
            if (not isinstance(message, dict) or message.get("id") != node_id
                    or not isinstance(message.get("author"), dict)
                    or message.get("status") != "finished_successfully"):
                reject()
            role = message["author"].get("role")
            if node_id == user:
                if role != "user" or node.get("parent") != expected:
                    reject()
            elif role == "assistant":
                content = message.get("content")
                if (previous_role not in {"user", "tool"} or message.get("end_turn") is not False
                        or message.get("channel") not in {None, "analysis"}
                        or not isinstance(content, dict) or content.get("content_type") != "code"):
                    reject()
            elif role != "tool" or previous_role != "assistant" or message.get("end_turn") is True:
                reject()
            context.append(deepcopy({k: message.get(k) for k in
                ("id", "author", "recipient", "channel", "content", "status", "end_turn")}))
            next_nodes = children.get(node_id, [])
            if len(next_nodes) > 1 or (node.get("children") or []) != next_nodes:
                reject()
            if not next_nodes:
                if role != "tool" or document.get("current_node") != node_id:
                    reject()
                return node_id, context
            previous_role, node_id = role, next_nodes[0]

    def complete_text(self, body: dict[str, Any], *, on_cursor=None) -> dict[str, Any]:
        binding_id = str(body.get("provider_binding_id") or "").strip()
        account_identity = str(body.get("provider_account_identity") or "").strip()
        client_conversation_id = str(body.get("client_conversation_id") or "").strip()
        conversation_id = str(body.get("conversation_id") or "").strip()
        parent_message_id = str(body.get("parent_message_id") or "").strip()
        requested_identity = body.get("_requested_account_identity")
        if body.get("_requested_account_ref") and not requested_identity:
            raise ConversationBindingError("account selection has no durable identity", code="CHAT_ACCOUNT_SELECTION_CONFLICT")
        if requested_identity and account_identity and requested_identity != account_identity:
            raise ConversationBindingError("account selection conflicts with binding", code="CHAT_ACCOUNT_SELECTION_CONFLICT")
        model = str(body.get("model") or "auto").strip() or "auto"
        image_model = str(body.get("image_model") or "gpt-image-2").strip() or "gpt-image-2"
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ConversationBindingError(
                "conversation messages are required",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
            )
        if not client_conversation_id:
            raise ConversationBindingError(
                "client_conversation_id is required",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
            )
        if binding_id:
            if not account_identity:
                raise ConversationBindingError(
                    "provider account identity is required for a bound account",
                    code="CONVERSATION_BINDING_CONTRACT_INVALID",
                )
            if bool(conversation_id) != bool(parent_message_id):
                raise ConversationBindingError(
                    "conversation continuation requires conversation_id and parent_message_id",
                    code="CONVERSATION_BINDING_CONTRACT_INVALID",
                )
            try:
                authoritative_identity = account_service.get_bound_account_identity(binding_id)
            except RuntimeError as exc:
                raise ConversationBindingError(str(exc)) from exc
            if authoritative_identity != account_identity:
                raise ConversationBindingError(
                    "provider account identity changed",
                    code="CONVERSATION_BINDING_MISMATCH",
                )
        elif conversation_id or parent_message_id:
            raise ConversationBindingError(
                "upstream cursor requires provider_binding_id",
                code="CONVERSATION_BINDING_CONTRACT_INVALID",
            )
        else:
            try:
                if body.get("_text_only_binding") is True:
                    binding_id, account_identity = account_service.create_text_conversation_binding(
                        text_model=model,
                        **({"requested_account_identity": requested_identity} if requested_identity else {}),
                    )
                else:
                    if requested_identity:
                        raise ConversationBindingError("account selection requires ordinary text binding", code="CHAT_ACCOUNT_SELECTION_CONFLICT")
                    binding_id, account_identity, image_token = account_service.create_conversation_binding(
                        image_model=image_model, text_model=model
                    )
                    account_service.release_image_slot(image_token)
            except RuntimeError as exc:
                raise ConversationBindingError(str(exc)) from exc

        if requested_identity and account_identity != requested_identity:
            raise ConversationBindingError("selected account changed", code="CHAT_ACCOUNT_SELECTION_CONFLICT")
        if on_cursor:
            on_cursor({"provider_binding_id": binding_id, "provider_account_identity": account_identity,
                       "client_conversation_id": client_conversation_id,
                       **({"conversation_id": conversation_id, "parent_message_id": parent_message_id} if conversation_id else {})})
        try:
            access_token = account_service.get_bound_text_access_token(
                binding_id,
                model=model,
                for_message=True,
                **({"requested_account_identity": requested_identity} if requested_identity else {}),
            )
        except RuntimeError as exc:
            raise ConversationBindingError(str(exc)) from exc

        with account_service.conversation_binding_lock(binding_id, client_conversation_id):
            backend = OpenAIBackendAPI(access_token=access_token)
            backend.retain_bound_conversation = True
            backend.text_request_message_id = str(body.get("_request_message_id") or "")
            if body.get("_public_session_ref") or body.get("_supersedes_request_message_id"):
                backend.text_cursor_callback = on_cursor
            if body.get("_supersedes_request_message_id"):
                def check_original(send):
                    try:
                        fresh = backend._get_conversation(conversation_id, _send=send)
                    except Exception as exc:
                        raise ConversationBindingError("original conversation read is temporarily unavailable",
                                                       code="CHAT_SUPERSEDE_READ_UNAVAILABLE") from exc
                    if body.get("_supersedes_no_final_original"):
                        if self._no_final_successor_tail(fresh, body["_supersedes_no_final_original"]) != no_final_context:
                            raise ConversationBindingError("original no-final cursor changed", code="CHAT_SUPERSEDE_CURSOR_CHANGED")
                    else:
                        self._check_superseded_original(fresh, conversation_id, parent_message_id,
                                                        body["_supersedes_request_message_id"])
                    if fresh["is_archived"]:
                        raise ConversationBindingError("original conversation restore is unconfirmed",
                                                       code="CHAT_SUPERSEDE_READ_UNAVAILABLE")
                backend.text_pre_send_check = check_original
            elif body.get("_failed_retry_original"):
                def check_failed_retry(send):
                    from services.generation_completion import retry_cursor, retry_evidence
                    original = body["_failed_retry_original"]
                    try:
                        fresh = backend._get_conversation(conversation_id, _send=send)
                        current, saved = retry_cursor(fresh, original), retry_evidence(original)
                    except Exception as exc:
                        raise ConversationBindingError("original retry read unavailable",
                                                       code="COMPLETION_ORIGINAL_READ_UNAVAILABLE") from exc
                    if not current or not saved or any(current[k] != saved[k] for k in (
                            "conversation_id", "request_message_id", "retry_parent_message_id")):
                        raise ConversationBindingError("original retry branch changed",
                                                       code="COMPLETION_ORIGINAL_CURSOR_UNCONFIRMED")
                backend.text_pre_send_check = check_failed_retry
            elif body.get("_empty_retry_original"):
                def check_empty_retry(send):
                    from services.text_task_service import TextTaskService
                    original = body["_empty_retry_original"]
                    try:
                        fresh_document = backend._get_conversation(conversation_id, _send=send)
                        fresh = self._read_text_request_result(backend, original, document=fresh_document)
                    except Exception as exc:
                        raise ConversationBindingError("empty-response original read unavailable",
                                                       code="CHAT_TERMINAL_EMPTY_READ_UNAVAILABLE") from exc
                    if (fresh_document.get("is_archived") is True or not TextTaskService._fresh_terminal_empty_continuation(
                            original, body["_empty_retry_receipt"], fresh)):
                        raise ConversationBindingError("empty-response retry cursor changed",
                                                       code="CHAT_TERMINAL_EMPTY_UNVERIFIED")
                backend.text_pre_send_check = check_empty_retry
            failure_phase = "stream_open"
            try:
                # All bound text consumers can resume an archived Chat. The
                # public-session flag describes a client protocol, not whether
                # the original upstream conversation needs restoring.
                if conversation_id:
                    try:
                        document = backend._get_conversation(conversation_id)
                    except Exception as exc:
                        raise ConversationBindingError("original conversation read is temporarily unavailable",
                                                       code=("CHAT_SUPERSEDE_READ_UNAVAILABLE" if body.get("_supersedes_request_message_id")
                                                             else "CHAT_ARCHIVE_RESTORE_UNCONFIRMED")) from exc
                    if body.get("_supersedes_request_message_id"):
                        if body.get("_supersedes_no_final_original"):
                            no_final_context = self._no_final_successor_tail(document, body["_supersedes_no_final_original"])
                            parent_message_id = no_final_context[0]
                            if on_cursor:
                                on_cursor({"request_parent_message_id": parent_message_id})
                        else:
                            self._check_superseded_original(document, conversation_id, parent_message_id,
                                                            body["_supersedes_request_message_id"])
                    if not isinstance(document, dict) or type(document.get("is_archived")) is not bool:
                        raise ConversationBindingError("original conversation visibility is unconfirmed",
                                                       code="CHAT_ARCHIVE_RESTORE_UNCONFIRMED")
                    if str(document.get("conversation_id") or conversation_id) != conversation_id:
                        raise ConversationBindingError("original conversation identity changed",
                                                       code="CONVERSATION_BINDING_MISMATCH")
                    if document.get("current_node") != parent_message_id:
                        raise ConversationBindingError("original product conversation changed", code="CONVERSATION_BINDING_MISMATCH")
                    if document.get("is_archived") is True:
                        try:
                            backend.set_conversation_archived(conversation_id, parent_message_id, False)
                        except Exception as exc:
                            raise ConversationBindingError("original conversation restore is temporarily unavailable",
                                                           code="CHAT_ARCHIVE_RESTORE_UNCONFIRMED") from exc
                parts: list[str] = []
                returned_conversation_id = ""
                for event in conversation_events(
                    backend,
                    messages=messages,
                    model=model,
                    thinking_effort=str(body.get("thinking_effort") or ""),
                    conversation_id=conversation_id,
                    parent_message_id=parent_message_id,
                ):
                    failure_phase = "stream_event"
                    old_conversation_id = returned_conversation_id
                    returned_conversation_id = str(
                        event.get("conversation_id") or returned_conversation_id
                    )
                    if on_cursor and returned_conversation_id and returned_conversation_id != old_conversation_id:
                        on_cursor({"conversation_id": returned_conversation_id})
                    if event.get("type") == "conversation.delta":
                        delta = str(event.get("delta") or "")
                        if delta:
                            parts.append(delta)
                failure_phase = "result_check"
                if not returned_conversation_id:
                    raise ConversationBindingError(
                        "upstream response has no conversation_id",
                        code="CONVERSATION_OUTCOME_UNKNOWN",
                        provider_binding_id=binding_id,
                        provider_account_identity=account_identity,
                        original_failure_phase=failure_phase,
                        original_exception_category="empty_result",
                    )
                if conversation_id and returned_conversation_id != conversation_id:
                    raise ConversationBindingError(
                        "upstream conversation identity changed",
                        code="CONVERSATION_BINDING_MISMATCH",
                        provider_binding_id=binding_id,
                        provider_account_identity=account_identity,
                        conversation_id=returned_conversation_id,
                    )
                if body.get("_public_session_ref") or body.get("_supersedes_request_message_id"):
                    # Neither a socket EOF nor the conversation's latest answer
                    # proves our turn finished. Use this request's unique branch.
                    failure_phase = "cursor_read"
                    try:
                        recovered = self._read_text_request_result(backend, {
                            "provider_binding_id": binding_id, "provider_account_identity": account_identity,
                            "client_conversation_id": client_conversation_id, "conversation_id": returned_conversation_id,
                            "request_message_id": backend.text_request_message_id,
                            "request_parent_message_id": getattr(backend, "text_request_parent_message_id", ""),
                        })
                    except ConversationBindingError as exc:
                        # The model turn was sent. An unproven GET result must
                        # retain occupancy and original-request recovery, even
                        # when the returned document has a foreign cursor.
                        raise ConversationBindingError(
                            "original sequential response could not be verified",
                            code="CONVERSATION_OUTCOME_UNKNOWN",
                            provider_binding_id=binding_id, provider_account_identity=account_identity,
                            conversation_id=returned_conversation_id,
                            original_failure_phase=failure_phase, original_exception_category="provider_error",
                        ) from exc
                    if recovered.get("status") != "succeeded":
                        raise ConversationBindingError(
                            "original sequential response is not complete", code="CONVERSATION_OUTCOME_UNKNOWN",
                            provider_binding_id=binding_id, provider_account_identity=account_identity,
                            conversation_id=returned_conversation_id,
                            original_failure_phase=failure_phase, original_exception_category="empty_result",
                        )
                    account_service.mark_text_used(access_token)
                    return {**recovered, "_upstream_terminal": True}
                content = "".join(parts).strip()
                if not content:
                    raise ConversationBindingError(
                        "upstream response was empty",
                        code="CONVERSATION_OUTCOME_UNKNOWN",
                        provider_binding_id=binding_id,
                        provider_account_identity=account_identity,
                        conversation_id=returned_conversation_id,
                        original_failure_phase=failure_phase,
                        original_exception_category="empty_result",
                    )
                failure_phase = "cursor_read"
                next_parent_message_id = backend.get_conversation_parent_message_id(
                    returned_conversation_id
                )
                account_service.mark_text_used(access_token)
                return {
                    "content": content,
                    "provider_binding_id": binding_id,
                    "provider_account_identity": account_identity,
                    "conversation_id": returned_conversation_id,
                    "parent_message_id": next_parent_message_id,
                    "binding_status": "bound",
                }
            except ConversationBindingError as exc:
                if exc.code == "CONVERSATION_OUTCOME_UNKNOWN" and not exc.original_failure_phase:
                    exc.original_failure_phase = failure_phase
                    exc.original_exception_category = "provider_error"
                raise
            except Exception as exc:
                if body.get("_supersedes_request_message_id"):
                    from services.request_context import current_request
                    context = current_request.get()
                    if context is not None and not context.receipt().get("_submission_started"):
                        raise ConversationBindingError("successor pre-send check did not complete",
                                                       code="CHAT_SUPERSEDE_READ_UNAVAILABLE") from exc
                original_category = _text_failure_category(exc)
                original_status = exc.status_code if isinstance(exc, UpstreamHTTPError) else None
                recovered_parent = ""
                if returned_conversation_id:
                    try:
                        recovered_parent = backend.get_conversation_parent_message_id(
                            returned_conversation_id
                        )
                        # A stream timeout may happen after the answer was saved.
                        # Read that turn once, never regenerate it or accept an old continuation answer.
                        if recovered_parent and not conversation_id and not body.get("_public_session_ref"):
                            recovered = self._read_text_result(backend, {
                                "provider_binding_id": binding_id,
                                "provider_account_identity": account_identity,
                                "client_conversation_id": client_conversation_id,
                                "conversation_id": returned_conversation_id,
                                "parent_message_id": recovered_parent,
                            })
                            if recovered.get("status") == "succeeded":
                                return recovered
                    except Exception:
                        pass
                raise ConversationBindingError(
                    str(exc) or "upstream conversation outcome is unknown",
                    code="CONVERSATION_OUTCOME_UNKNOWN",
                    provider_binding_id=binding_id,
                    provider_account_identity=account_identity,
                    conversation_id=returned_conversation_id,
                    parent_message_id=recovered_parent,
                    original_failure_phase=failure_phase,
                    original_http_status=original_status,
                    original_exception_category=original_category,
                    original_upstream_request_stage=(
                        _TEXT_HTTP_REQUEST_STAGES.get(exc.context, "unknown")
                        if isinstance(exc, UpstreamHTTPError) else ""
                    ),
                    **(_text_http_diagnostic(exc)
                       if isinstance(exc, UpstreamHTTPError) and failure_phase == "stream_open" else {}),
                ) from exc
            finally:
                backend.close()


conversation_binding_service = ConversationBindingService()
