from __future__ import annotations

from services.image_thread import finished_parent, ImageThreadError

import base64
import json
import math
import re
import threading
import time
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Iterator

import tiktoken

from services.account_service import account_service
from services.config import config
from services.image_storage_service import image_storage_service
from services.openai_backend_api import (
    ImageActiveDeadlineExceeded,
    ImageContentPolicyError,
    ImagePollTimeoutError,
    OpenAIBackendAPI,
    _is_content_policy_error,
)
from utils.helper import (
    IMAGE_MODELS,
    extract_image_from_message_content,
    is_codex_image_model,
    is_supported_image_model,
    split_image_model,
)
from utils.image_tokens import count_image_content_tokens
from utils.log import logger


class ImageGenerationError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        error_type: str = "server_error",
        code: str | None = "upstream_error",
        param: str | None = None,
        account_email: str = "",
        conversation_id: str = "",
        provider_binding_id: str = "",
        provider_account_identity: str = "",
        parent_message_id: str = "",
        request_message_id: str = "",
        upstream_submitted: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.param = param
        self.account_email = account_email
        self.conversation_id = conversation_id
        self.provider_binding_id = provider_binding_id
        self.provider_account_identity = provider_account_identity
        self.parent_message_id = parent_message_id
        self.request_message_id = request_message_id
        self.upstream_submitted = upstream_submitted

    def to_openai_error(self) -> dict[str, Any]:
        error_dict = {
            "error": {
                "message": public_image_error_message(str(self)),
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }
        if self.account_email:
            error_dict["error"]["account_email"] = self.account_email
        return error_dict


def public_image_error_message(message: str) -> str:
    text = str(message or "").strip()
    lower = text.lower()
    if any(item in lower for item in ("backend-api/", "status=", "body=", "chatgpt.com", "upstreamhttperror")):
        return "The image generation request failed. Please try again later."
    return text or "The image generation request failed. Please try again later."


def is_token_invalid_error(message: str) -> bool:
    text = str(message or "").lower()
    return (
        "token_invalidated" in text
        or "token_revoked" in text
        or "authentication token has been invalidated" in text
        or "invalidated oauth token" in text
    )


def is_tls_connection_error(message: str) -> bool:
    """检测 TLS/SSL 连接错误，这类错误通常可以通过重试解决。"""
    text = str(message or "").lower()
    return (
        "curl: (35)" in text
        or "tls connect error" in text
        or "openssl_internal" in text
        or "ssl: wrong_version_number" in text
        or "ssl: certificate_verify_failed" in text
        or "connection aborted" in text
        or "remote disconnected" in text
        or "connection reset by peer" in text
    )


def is_connection_timeout_error(message: str) -> bool:
    """检测连接超时错误（如 curl 28），这类错误可通过同账号短等待重试解决。"""
    text = str(message or "").lower()
    return (
        "curl: (28)" in text
        or "operation timed out" in text
        or "connection timed out" in text
        or "read timed out" in text
        or "connect timeout" in text
    )


def image_stream_error_message(message: str) -> str:
    text = str(message or "")
    if is_token_invalid_error(text):
        return "image generation failed"
    if is_tls_connection_error(text):
        return "upstream image connection failed, please retry later"
    if is_connection_timeout_error(text):
        return "upstream connection timed out, please retry later"
    return text or "image generation failed"


REFERENCED_IMAGE_IDS_RE = re.compile(r'"referenced_image_ids"\s*:\s*\[([^\]]+)\]')
# 检测模型返回的部分工具调用 JSON（如 {"size":"1920x1088","n":1}）
# 这些 JSON 包含图片生成工具的参数，但没有实际生成图片
TOOL_PARAMS_JSON_RE = re.compile(
    r'\{\s*"size"\s*:\s*"\d+x\d+"\s*,\s*"n"\s*:\s*\d+\s*\}'
)


def is_model_text_reply_instead_of_image(message: str) -> bool:
    """检测模型是否返回了文本回复（包含工具调用 JSON）而非实际生成图片。

    当上游 ChatGPT 未能触发图片生成工具时，会返回一段描述性文本，
    其中可能包含 JSON 参数（如 prompt、referenced_image_ids、size/n 等）。
    这种情况应被视为「上游未生成图片」而非「内容策略违规」。

    检测两种模式：
    1. 完整的工具调用 JSON（含 referenced_image_ids）
    2. 部分的工具参数 JSON（如 {"size":"1920x1088","n":1}）
    """
    if not message:
        return False
    if REFERENCED_IMAGE_IDS_RE.search(message):
        return True
    # 检测部分工具参数 JSON（模型返回了工具参数但未触发工具）
    if TOOL_PARAMS_JSON_RE.search(message):
        return True
    return False


def encode_images(images: Iterable[tuple[bytes, str, str]]) -> list[str]:
    return [base64.b64encode(data).decode("ascii") for data, _, _ in images if data]


def save_image_bytes(image_data: bytes, base_url: str | None = None) -> str:
    return image_storage_service.save(image_data, base_url).url


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and str(item.get("type") or "") in {"text", "input_text", "output_text"}:
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


def normalize_messages(messages: object, system: Any = None) -> list[dict[str, Any]]:
    normalized = []
    if config.global_system_prompt:
        normalized.append({"role": "system", "content": config.global_system_prompt})
    system_text = message_text(system)
    if system_text:
        normalized.append({"role": "system", "content": system_text})
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "user")
            content = message.get("content", "")
            text = message_text(content)
            images: list[tuple[bytes, str]] = []
            if role == "user":
                images.extend(extract_image_from_message_content(content))
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict) or part.get("type") != "image":
                            continue
                        data = part.get("data")
                        if isinstance(data, (bytes, bytearray)) and all(existing[0] != bytes(data) for existing in images):
                            images.append((bytes(data), str(part.get("mime") or "image/png")))
            if images:
                parts: list[Any] = []
                if text:
                    parts.append({"type": "text", "text": text})
                for data, mime in images:
                    parts.append({"type": "image", "data": data, "mime": mime})
                normalized.append({"role": role, "content": parts})
            else:
                normalized.append({"role": role, "content": text})
    return normalized


def prompt_with_global_system(prompt: str) -> str:
    return f"{config.global_system_prompt}\n\n{prompt}" if config.global_system_prompt else prompt


def assistant_history_text(messages: list[dict[str, Any]]) -> str:
    return "".join(str(item.get("content") or "") for item in messages if item.get("role") == "assistant")


def assistant_history_messages(messages: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("content") or "") for item in messages if item.get("role") == "assistant" and item.get("content")]


def build_image_prompt(prompt: str, size: str | None, quality: str = "auto") -> str:
    hints = []
    if size:
        hints.append(f"输出图片尺寸为 {size}。")
    if quality:
        hints.append(f"输出图片质量为 {quality}。")
    return f"{prompt.strip()}\n\n{''.join(hints)}" if hints else prompt


def encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        try:
            return tiktoken.get_encoding("o200k_base")
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")


def count_message_image_tokens(messages: list[dict[str, Any]], model: str) -> int:
    return sum(count_image_content_tokens(message.get("content"), model) for message in messages)


def count_message_text_tokens(messages: list[dict[str, Any]], model: str) -> int:
    encoding = encoding_for_model(model)
    total = 0
    for message in messages:
        total += 3
        for key, value in message.items():
            if key == "content" and isinstance(value, list):
                total += len(encoding.encode(message_text(value)))
            elif isinstance(value, str):
                total += len(encoding.encode(value))
            else:
                continue
            if key == "name":
                total += 1
    return total + 3


def count_message_tokens(messages: list[dict[str, Any]], model: str) -> int:
    return count_message_text_tokens(messages, model) + count_message_image_tokens(messages, model)


def count_text_tokens(text: str, model: str) -> int:
    return len(encoding_for_model(model).encode(text))


def format_image_result(
    items: list[dict[str, Any]],
    prompt: str,
    response_format: str,
    base_url: str | None = None,
    created: int | None = None,
    message: str = "",
) -> dict[str, Any]:
    data: list[dict[str, Any]] = []
    for item in items:
        b64_json = str(item.get("b64_json") or "").strip()
        if not b64_json:
            continue
        revised_prompt = str(item.get("revised_prompt") or prompt).strip() or prompt
        if response_format == "b64_json":
            data.append({
                "b64_json": b64_json,
                "url": save_image_bytes(base64.b64decode(b64_json), base_url),
                "revised_prompt": revised_prompt,
            })
        else:
            data.append({
                "url": save_image_bytes(base64.b64decode(b64_json), base_url),
                "revised_prompt": revised_prompt,
            })
    result: dict[str, Any] = {"created": created or int(time.time()), "data": data}
    if message and not data:
        result["message"] = message
    return result


@dataclass
class ConversationRequest:
    model: str = "auto"
    prompt: str = ""
    messages: list[dict[str, Any]] | None = None
    thinking_effort: str = ""
    images: list[str] | None = None
    n: int = 1
    size: str | None = None
    quality: str = "auto"
    response_format: str = "b64_json"
    base_url: str | None = None
    message_as_error: bool = False
    progress_callback: Any = None  # Callable[[str], None] | None
    provider_binding_id: str = ""
    provider_account_identity: str = ""
    client_conversation_id: str = ""
    conversation_id: str = ""
    parent_message_id: str = ""
    retain_conversation: bool = False
    upstream_model: str = ""
    _defer_image_publication: bool = field(default=False, init=False, repr=False)


@dataclass
class ConversationState:
    text: str = ""
    raw_text: str = ""
    conversation_id: str = ""
    file_ids: list[str] = field(default_factory=list)
    sediment_ids: list[str] = field(default_factory=list)
    blocked: bool = False
    tool_invoked: bool | None = None
    turn_use_case: str = ""


@dataclass
class ImageOutput:
    kind: str
    model: str
    index: int
    total: int
    created: int = field(default_factory=lambda: int(time.time()))
    text: str = ""
    upstream_event_type: str = ""
    data: list[dict[str, Any]] = field(default_factory=list)
    account_email: str = ""
    conversation_id: str = ""
    provider_binding_id: str = ""
    provider_account_identity: str = ""
    parent_message_id: str = ""
    image_thread_terminal: bool = False
    _pending_image_items: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def to_chunk(self) -> dict[str, Any]:
        chunk: dict[str, Any] = {
            "object": "image.generation.chunk",
            "created": self.created,
            "model": self.model,
            "index": self.index,
            "total": self.total,
            "progress_text": self.text,
            "upstream_event_type": self.upstream_event_type,
            "data": [],
        }
        if self.account_email:
            chunk["_account_email"] = self.account_email
        if self.conversation_id:
            chunk["_conversation_id"] = self.conversation_id
        if self.provider_binding_id:
            chunk["_provider_binding_id"] = self.provider_binding_id
        if self.provider_account_identity:
            chunk["_provider_account_identity"] = self.provider_account_identity
        if self.parent_message_id:
            chunk["_parent_message_id"] = self.parent_message_id
        if self.image_thread_terminal:
            chunk["_image_thread_terminal"] = True
        if self.kind == "message":
            chunk.update({
                "object": "image.generation.message",
                "message": self.text,
            })
            chunk.pop("progress_text", None)
            chunk.pop("upstream_event_type", None)
        elif self.kind == "result":
            chunk.update({
                "object": "image.generation.result",
                "data": self.data,
            })
            chunk.pop("progress_text", None)
            chunk.pop("upstream_event_type", None)
        return chunk


def _message_output_error(output: ImageOutput, *, account_email: str = "") -> ImageGenerationError:
    """Turn a terminal text-only image response into its truthful category."""
    message = output.text or "Upstream completed without generating an image."
    is_policy = _is_content_policy_error(message)
    lower_message = message.lower()
    still_processing = (
        "may still be processing" in lower_message
        or "started upstream but the response was incomplete" in lower_message
    )
    return ImageGenerationError(
        message,
        status_code=400,
        error_type="invalid_request_error",
        code=(
            "content_policy_violation" if is_policy
            else "CONVERSATION_OUTCOME_UNKNOWN" if still_processing
            else "NO_IMAGE_GENERATED"
        ),
        account_email=account_email,
        conversation_id=output.conversation_id,
    )


def _active_budget_remaining(request: ConversationRequest) -> float | None:
    deadline = getattr(request.progress_callback, "active_deadline_at", None)
    if not isinstance(deadline, (int, float)) or isinstance(deadline, bool) or deadline <= 0:
        return None
    return max(0.0, float(deadline) - time.time())


def _active_poll_timeout(request: ConversationRequest, desired: float) -> float:
    remaining = _active_budget_remaining(request)
    if remaining is None:
        return float(desired)
    if remaining <= 0:
        raise ImagePollTimeoutError("image active generation deadline exhausted")
    return max(0.001, min(float(desired), remaining))


def _record_result_ids(
    request: ConversationRequest,
    file_ids: list[str],
    sediment_ids: list[str],
) -> None:
    consume = getattr(request, "_record_image_capacity_consumed", None)
    if callable(consume) and (file_ids or sediment_ids):
        consume()
    callback = getattr(request.progress_callback, "record_result_ids", None)
    if callable(callback) and (file_ids or sediment_ids):
        callback(list(dict.fromkeys(file_ids)), list(dict.fromkeys(sediment_ids)))


def _validate_image_download_count(items, file_ids, sediment_ids):
    # Distinct IDs in one namespace denote distinct immutable assets. Mixed
    # namespaces may offer alternate addresses for the same image; do not infer
    # their image count by adding the two lists.
    if bool(file_ids) != bool(sediment_ids) and len(items) != len(set(file_ids or sediment_ids)):
        raise ValueError("original image downloads are incomplete")


def _downloaded_image_output(request, items, *, conversation_id, request_message_id,
                             file_ids, sediment_ids, index, total):
    output = ImageOutput(kind="result", model=request.model, index=index, total=total,
                         conversation_id=conversation_id)
    if request._defer_image_publication:
        callback = getattr(request.progress_callback, "record_downloaded_image_items", None)
        if callable(callback):
            callback({"conversation_id": conversation_id, "request_message_id": request_message_id,
                      "file_ids": list(dict.fromkeys(file_ids)),
                      "sediment_ids": list(dict.fromkeys(sediment_ids))}, items)
        _validate_image_download_count(items, file_ids, sediment_ids)
        # Bound generation collects this internally and verifies the exact turn
        # before formatting it. Neither to_chunk nor collection exposes this field.
        output._pending_image_items = items
    else:
        _validate_image_download_count(items, file_ids, sediment_ids)
        output.data = format_image_result(items, request.prompt, request.response_format,
                                          request.base_url, output.created)["data"]
    return output


def _start_active_attempt(request: ConversationRequest) -> None:
    callback = getattr(request.progress_callback, "start_active_attempt", None)
    if callable(callback):
        callback()


def assistant_message_text(message: dict[str, Any]) -> str:
    content = message.get("content") or {}
    parts = content.get("parts") or []
    if isinstance(parts, list) and parts:
        text = "".join(part for part in parts if isinstance(part, str))
        if text:
            return text
    # Fallback: content_type "code" stores text in the "text" field instead of "parts"
    text_field = str(content.get("text") or "")
    if text_field:
        return text_field
    return ""


def is_visible_assistant_message(message: dict[str, Any]) -> bool:
    """Return whether an upstream assistant message is intended for the user."""
    author = message.get("author")
    if not isinstance(author, dict):
        return False
    role = str(author.get("role") or "").strip().lower()
    if role != "assistant":
        return False

    metadata = message.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("is_visually_hidden_from_conversation") is True:
        return False

    # Tool commands are emitted as assistant messages addressed to recipients
    # such as "web". Only messages addressed to everyone are user-visible.
    recipient = str(message.get("recipient") or "").strip().lower()
    if recipient and recipient != "all":
        return False

    # Reasoning and other internal channels must not leak into API text output.
    channel = str(message.get("channel") or "").strip().lower()
    if channel and channel != "final":
        return False

    return True


def strip_history(text: str, history_text: str = "") -> str:
    text = str(text or "")
    history_text = str(history_text or "")
    while history_text and text.startswith(history_text):
        text = text[len(history_text):]
    return text


def sanitize_output_text(text: str) -> str:
    text = str(text or "")

    def is_internal_annotation_part(part: str) -> bool:
        value = part.strip()
        if not value:
            return True
        lower = value.lower()
        return bool(
            re.fullmatch(r"turn\d+[a-z]*\d*", lower)
            or re.fullmatch(r"turn\d+\w*", lower)
            or lower.startswith(("turn", "source", "sources"))
        )

    def readable_annotation_part(parts: list[str]) -> str:
        for part in parts:
            value = part.strip()
            if value and not is_internal_annotation_part(value):
                return value
        return ""

    def annotation_text(payload: str) -> str:
        parts = [part.strip() for part in payload.split("\ue202")]
        kind = (parts[0] if parts else "").lower()
        data = parts[1:]
        if kind == "url":
            label = data[0] if data else ""
            url = data[1] if len(data) > 1 else ""
            if label and url.startswith(("http://", "https://")):
                return f"{label} ({url})"
            return label or url
        if kind == "cite":
            return readable_annotation_part(data)
        return readable_annotation_part(data)

    def replace_annotation(match: re.Match[str]) -> str:
        return annotation_text(match.group(1))

    def replace_annotation_before_punctuation(match: re.Match[str]) -> str:
        leading_space = match.group(1)
        replacement = annotation_text(match.group(2))
        return f"{leading_space}{replacement}" if replacement else ""

    # ChatGPT web sometimes returns rich annotation markers using private-use
    # characters. API clients cannot render those. Preserve readable labels
    # from entity/link annotations, while removing internal citation pointers.
    text = re.sub(r"(\s*)\ue200([^\ue201]*)\ue201(?=[.,;:!?])", replace_annotation_before_punctuation, text)
    text = re.sub(r"\ue200([^\ue201]*)\ue201", replace_annotation, text)
    text = re.sub(r"\ue200[^\ue201]*$", "", text)
    return text


def assistant_raw_text(event: dict[str, Any], current_text: str = "", history_text: str = "") -> str:
    for candidate in (event, event.get("v")):
        if not isinstance(candidate, dict):
            continue
        message = candidate.get("message")
        if not isinstance(message, dict) or not is_visible_assistant_message(message):
            continue
        text = assistant_message_text(message)
        if text:
            return strip_history(text, history_text)
    return apply_text_patch(event, current_text, history_text)


def assistant_text(event: dict[str, Any], current_text: str = "", history_text: str = "") -> str:
    return sanitize_output_text(assistant_raw_text(event, current_text, history_text))


def event_assistant_text(event: dict[str, Any], history_text: str = "") -> str:
    for candidate in (event, event.get("v")):
        if not isinstance(candidate, dict):
            continue
        message = candidate.get("message")
        if isinstance(message, dict) and is_visible_assistant_message(message):
            return strip_history(assistant_message_text(message), history_text)
    return ""


def apply_text_patch(event: dict[str, Any], current_text: str = "", history_text: str = "") -> str:
    if event.get("p") == "/message/content/parts/0":
        return apply_patch_op(event, current_text, history_text)

    operations = event.get("v")
    if isinstance(operations, str) and current_text and not event.get("p") and not event.get("o"):
        return current_text + operations

    if event.get("o") == "patch" and isinstance(operations, list):
        text = current_text
        for item in operations:
            if isinstance(item, dict):
                text = apply_text_patch(item, text, history_text)
        return text

    if not isinstance(operations, list):
        return current_text

    text = current_text
    for item in operations:
        if isinstance(item, dict):
            text = apply_text_patch(item, text, history_text)
    return text


def apply_patch_op(operation: dict[str, Any], current_text: str, history_text: str = "") -> str:
    op = operation.get("o")
    value = str(operation.get("v") or "")
    if op == "append":
        return current_text + value
    if op == "replace":
        return strip_history(value, history_text)
    return current_text


def add_unique(values: list[str], candidates: list[str]) -> None:
    for candidate in candidates:
        if candidate and candidate not in values:
            values.append(candidate)


FILE_SERVICE_ID_RE = re.compile(r"file-service://([A-Za-z0-9_-]+)")
FILE_ID_RE = re.compile(r"\b(file[-_](?!service\b)[A-Za-z0-9_-]+)\b")
# 真正的图片文件 ID 格式：file_00000000 + 24位十六进制字符（共32字符）
# 用于过滤非图片文件 ID（如 file_upload_business_upsell）
REAL_IMAGE_FILE_ID_RE = re.compile(r"\bfile_00000000[a-f0-9]{24}\b")
SEDIMENT_ID_RE = re.compile(r"sediment://([A-Za-z0-9_-]+)")


def extract_conversation_ids(payload: str) -> tuple[str, list[str], list[str]]:
    conversation_match = re.search(r'"conversation_id"\s*:\s*"([^"]+)"', payload)
    conversation_id = conversation_match.group(1) if conversation_match else ""
    file_ids: list[str] = []
    # Negative lookahead excludes "file-service" (URI prefix, not a real id).
    add_unique(file_ids, FILE_SERVICE_ID_RE.findall(payload))
    # 只提取真正的图片文件 ID（file_00000000... 格式），过滤非图片文件 ID（如 file_upload_business_upsell）
    add_unique(file_ids, REAL_IMAGE_FILE_ID_RE.findall(payload))
    sediment_ids = SEDIMENT_ID_RE.findall(payload)
    return conversation_id, file_ids, sediment_ids


def is_image_tool_event(event: dict[str, Any]) -> bool:
    value = event.get("v")
    message = event.get("message") or (value.get("message") if isinstance(value, dict) else None)
    if not isinstance(message, dict):
        return False
    metadata = message.get("metadata") or {}
    author = message.get("author") or {}
    content = message.get("content") or {}
    if author.get("role") != "tool":
        return False
    if metadata.get("async_task_type") == "image_gen":
        return True
    if content.get("content_type") != "multimodal_text":
        return False
    return any(
        isinstance(part, dict) and (
                part.get("content_type") == "image_asset_pointer"
                or str(part.get("asset_pointer") or "").startswith(("file-service://", "sediment://"))
        )
        for part in content.get("parts") or []
    )


def update_conversation_state(state: ConversationState, payload: str, event: dict[str, Any] | None = None) -> None:
    conversation_id, file_ids, sediment_ids = extract_conversation_ids(payload)
    if conversation_id and not state.conversation_id:
        state.conversation_id = conversation_id
    # Accept generated IDs only from a complete image tool result. A prior
    # ``tool_invoked`` marker or an unscoped patch is insufficient: either can
    # repeat uploaded input IDs, tool arguments, or referenced_image_ids.
    image_context = isinstance(event, dict) and is_image_tool_event(event)
    if image_context:
        add_unique(state.file_ids, file_ids)
        add_unique(state.sediment_ids, sediment_ids)
    if not isinstance(event, dict):
        return
    state.conversation_id = str(event.get("conversation_id") or state.conversation_id)
    value = event.get("v")
    if isinstance(value, dict):
        state.conversation_id = str(value.get("conversation_id") or state.conversation_id)
    if event.get("type") == "moderation":
        moderation = event.get("moderation_response")
        if isinstance(moderation, dict) and moderation.get("blocked") is True:
            state.blocked = True
    if event.get("type") == "server_ste_metadata":
        metadata = event.get("metadata")
        if isinstance(metadata, dict):
            if isinstance(metadata.get("tool_invoked"), bool):
                state.tool_invoked = metadata["tool_invoked"]
            state.turn_use_case = str(metadata.get("turn_use_case") or state.turn_use_case)


def conversation_base_event(event_type: str, state: ConversationState, **extra: Any) -> dict[str, Any]:
    return {
        "type": event_type,
        "text": state.text,
        "conversation_id": state.conversation_id,
        "file_ids": list(state.file_ids),
        "sediment_ids": list(state.sediment_ids),
        "blocked": state.blocked,
        "tool_invoked": state.tool_invoked,
        "turn_use_case": state.turn_use_case,
        **extra,
    }


def iter_conversation_payloads(payloads: Iterator[str], history_text: str = "",
                               history_messages: list[str] | None = None) -> Iterator[dict[str, Any]]:
    state = ConversationState()
    history_messages = history_messages or []
    history_index = 0
    for payload in payloads:
        # print(f"[upstream_sse] {payload}", flush=True)
        if not payload:
            continue
        if payload == "[DONE]":
            yield conversation_base_event("conversation.done", state, done=True)
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            update_conversation_state(state, payload)
            yield conversation_base_event("conversation.raw", state, payload=payload)
            continue
        if not isinstance(event, dict):
            yield conversation_base_event("conversation.event", state, raw=event)
            continue
        update_conversation_state(state, payload, event)
        if history_index < len(history_messages) and event_assistant_text(event, history_text) == history_messages[history_index]:
            history_index += 1
            state.raw_text = ""
            state.text = ""
            continue
        next_raw_text = assistant_raw_text(event, state.raw_text, history_text)
        next_text = sanitize_output_text(next_raw_text)
        state.raw_text = next_raw_text
        if next_text != state.text:
            delta = next_text[len(state.text):] if next_text.startswith(state.text) else next_text
            state.text = next_text
            yield conversation_base_event("conversation.delta", state, raw=event, delta=delta)
            continue
        yield conversation_base_event("conversation.event", state, raw=event)


def conversation_events(
    backend: OpenAIBackendAPI,
    messages: list[dict[str, Any]] | None = None,
    model: str = "auto",
    prompt: str = "",
    images: list[str] | None = None,
    size: str | None = None,
    quality: str = "auto",
    thinking_effort: str = "",
    conversation_id: str = "",
    parent_message_id: str = "",
) -> Iterator[dict[str, Any]]:
    normalized = normalize_messages(messages or ([{"role": "user", "content": prompt}] if prompt else []))
    image_model = is_supported_image_model(model)
    history_text = "" if image_model else assistant_history_text(normalized)
    history_messages = [] if image_model else assistant_history_messages(normalized)
    final_prompt = prompt_with_global_system(build_image_prompt(prompt, size, quality)) if image_model else prompt
    payloads = backend.stream_conversation(
        messages=normalized,
        model=model,
        prompt=final_prompt,
        images=images if image_model else None,
        system_hints=["picture_v2"] if image_model else None,
        thinking_effort=thinking_effort if not image_model else "",
        conversation_id=conversation_id,
        parent_message_id=parent_message_id,
    )
    try:
        yield from iter_conversation_payloads(payloads, history_text, history_messages)
    finally:
        close = getattr(payloads, "close", None)
        if callable(close):
            close()


def text_backend(model: str = "auto") -> OpenAIBackendAPI:
    return OpenAIBackendAPI(access_token=account_service.get_text_access_token(model=model))


def stream_text_deltas(backend: OpenAIBackendAPI, request: ConversationRequest) -> Iterator[str]:
    from services.request_context import current_request
    from services.durable_forward import prepare_chat_recovery
    context = current_request.get()
    recoverable = context is not None and prepare_chat_recovery(context, request.model, request.messages)
    attempted_tokens: set[str] = set()
    token = getattr(backend, "access_token", "")
    emitted = False
    emitted_text = ""
    while True:
        if token and token in attempted_tokens:
            raise RuntimeError("no available text account")
        if token:
            attempted_tokens.add(token)
        active_backend = None
        try:
            active_backend = OpenAIBackendAPI(access_token=token)
            if recoverable:
                active_backend.retain_bound_conversation = True
            for event in conversation_events(
                active_backend,
                messages=request.messages,
                model=request.model,
                prompt=request.prompt,
                thinking_effort=request.thinking_effort,
            ):
                if recoverable:
                    receipt = context.receipt()
                    conversation_id = str(event.get("conversation_id") or "")
                    if conversation_id and conversation_id != receipt.get("conversation_id"):
                        if receipt.get("conversation_id"):
                            raise RuntimeError("original conversation identity changed")
                        context.admission.update_claim(context, conversation_id=conversation_id)
                if event.get("type") != "conversation.delta":
                    continue
                delta = str(event.get("delta") or "")
                if delta:
                    emitted = True
                    emitted_text += delta
                    yield delta
            if recoverable:
                # [DONE] and EOF describe the transport, not our model turn.
                # Verify the exact persisted user branch before emitting a
                # successful wire terminator or releasing its account turn.
                from services.conversation_binding_service import ConversationBindingError, ConversationBindingService
                receipt = context.receipt()
                try:
                    if not receipt.get("conversation_id") or not receipt.get("request_message_id"):
                        raise ValueError("original cursor is unavailable")
                    recovered = ConversationBindingService._read_text_request_result(active_backend, receipt)
                except Exception as exc:
                    raise ConversationBindingError("original stream result could not be verified",
                                                   code="CONVERSATION_OUTCOME_UNKNOWN") from exc
                if recovered.get("status") != "succeeded":
                    raise ConversationBindingError("original stream result is not complete",
                                                   code="CONVERSATION_OUTCOME_UNKNOWN")
                context.terminal(True)
                final_text = sanitize_output_text(recovered["content"])
                # The saved-result reader strips outer whitespace. Preserve
                # whitespace already delivered without treating it as drift.
                delivered = emitted_text.lstrip()
                if final_text == delivered.rstrip():
                    remaining = ""
                elif final_text.startswith(delivered):
                    remaining = final_text[len(delivered):]
                else:
                    raise ConversationBindingError("original stream prefix does not match its saved result",
                                                   code="CONVERSATION_OUTCOME_UNKNOWN")
                if remaining:
                    yield remaining
            account_service.mark_text_used(token)
            return
        except Exception as exc:
            error_message = str(exc)
            submitted = context is not None and context.receipt().get("_submission_started")
            if token and not emitted and not submitted and is_token_invalid_error(error_message):
                refreshed_token = account_service.refresh_access_token(token, force=True, event="text_stream")
                if refreshed_token and refreshed_token != token and refreshed_token not in attempted_tokens:
                    token = refreshed_token
                else:
                    account_service.remove_invalid_token(token, "text_stream")
                    token = account_service.get_text_access_token(
                        excluded_tokens=set(attempted_tokens),
                        model=request.model,
                    )
                if token:
                    continue
            raise
        finally:
            if active_backend is not None:
                active_backend.close()


def collect_text(backend: OpenAIBackendAPI, request: ConversationRequest) -> str:
    return "".join(stream_text_deltas(backend, request))


def _get_detailed_error_from_tasks(
    backend: OpenAIBackendAPI,
    conversation_id: str,
    timeout_secs: float = 10.0,
    wait_secs: float = 2.0,
) -> str:
    """从 /backend-api/tasks/ 接口获取结构化错误信息。

    当 SSE 流检测到 moderation 拦截时，轮询 tasks 接口获取详细错误文本。
    使用结构化字段（metadata.is_error, author.role, content.content_type）判断，
    而非依赖易变的文本匹配。

    参数：
    - `backend`：OpenAIBackendAPI 实例。
    - `conversation_id`：会话 ID。
    - `timeout_secs`：请求超时秒数。
    - `wait_secs`：等待任务创建的秒数。设为 0 可跳过等待。

    返回：
    - 详细错误信息文本，如果未找到则返回空字符串。
    """
    import time as _time
    try:
        if wait_secs > 0:
            _time.sleep(wait_secs)
        tasks = backend._query_backend_tasks(conversation_id=conversation_id, timeout_secs=timeout_secs)
        if not tasks:
            return ""

        for task in tasks:
            is_error, error_msg, metadata = backend.check_task_error(task)
            if is_error and error_msg:
                logger.info({
                    "event": "image_task_structured_error",
                    "conversation_id": conversation_id,
                    "error_msg": error_msg,
                    "metadata": metadata,
                })
                return error_msg
        return ""
    except Exception as exc:
        logger.warning({
            "event": "image_task_error_query_failed",
            "conversation_id": conversation_id,
            "error": str(exc),
        })
        return ""


def _remove_image_conversation_later(
        backend: OpenAIBackendAPI,
        conversation_id: str,
        *,
        success: bool,
        retain_conversation: bool = False,
) -> None:
    if retain_conversation:
        return
    if not conversation_id:
        return
    if not (config.image_remove_conversation_always or (success and config.image_remove_conversation_after_result)):
        return

    def _run() -> None:
        try:
            backend.delete_conversation(conversation_id)
            logger.info({"event": "image_conversation_removed", "conversation_id": conversation_id})
        except Exception as exc:
            logger.warning({
                "event": "image_conversation_remove_failed",
                "conversation_id": conversation_id,
                "error": str(exc),
            })

    threading.Thread(target=_run, name=f"remove-image-conversation-{conversation_id}", daemon=True).start()


def _observe_image_terminal():
    """Record a verified single-send turn without releasing its work or slot."""
    from services.request_context import current_request
    context = current_request.get()
    if context is None:
        return
    try:
        receipt = context.receipt()
        # A first image/turn must not terminate a multi-send task in telemetry.
        # Retried generations retain their history; don't present its first
        # turn's timestamp as the completion of a later send either.
        if (int(receipt.get("_expected_sends") or 1) != 1
                or int(receipt.get("_send_sequence") or 1) != 1):
            return
        if any(e.get("stage") == "upstream_terminal" and e.get("known") is True
               for e in receipt.get("_execution_timeline", [])):
            return
        # terminal(True) changes admission occupancy. This is timing evidence
        # only: downloading, publishing and work completion retain their gates.
        context.record_stage("upstream_terminal", known=True)
    except Exception:
        pass  # Missing diagnostics must not retry or discard the original.


def _stream_image_terminal_result(backend, event, request_message_id, file_ids, sediment_ids, expected_parent=""):
    """One signal-triggered read; a tool leaf cannot close an open image stream."""
    from services.request_context import AdmissionLost
    from services.account_request_pacing import AccountRequestDeadlineExceeded
    check = getattr(backend, "image_poll_terminal_check", None)
    conversation_id = str(event.get("conversation_id") or "")
    if not conversation_id or not request_message_id or not (file_ids or sediment_ids):
        return False
    try:
        document = backend._get_conversation(conversation_id)
        current = (document.get("mapping", {}).get(document.get("current_node")) or {}).get("message") or {}
        if (current.get("author", {}).get("role") != "assistant"
                or current.get("status") != "finished_successfully"
                or current.get("end_turn") is not True or current.get("channel") not in (None, "final")):
            return False
        records = backend._extract_image_tool_records(document, request_message_id)
        if ({x for record in records for x in record["file_ids"]} != set(file_ids)
                or {x for record in records for x in record["sediment_ids"]} != set(sediment_ids)):
            return False
        if callable(check):
            if check(document, conversation_id, request_message_id, file_ids, sediment_ids) is not True:
                return False
        else:
            finished_parent(document, conversation_id, request_message_id,
                            expected_parent=expected_parent or None, expected_result_ids=file_ids+sediment_ids)
        _observe_image_terminal()
        return True
    except (AdmissionLost, AccountRequestDeadlineExceeded, ImageActiveDeadlineExceeded):
        raise
    except Exception:
        # The paced transport has already persisted any 429 cooldown. Continue
        # this original stream/recovery; never submit another generation here.
        logger.info({"event": "image_stream_result_check_deferred", "conversation_id": conversation_id})
        return False


def stream_image_outputs(
        backend: OpenAIBackendAPI,
        request: ConversationRequest,
        index: int = 1,
        total: int = 1,
) -> Iterator[ImageOutput]:
    request_message_id = str(
        getattr(request.progress_callback, "request_message_id", "") or "",
    )
    if request_message_id:
        backend.image_request_message_id = request_message_id
    record_conversation_id = getattr(request.progress_callback, "record_conversation_id", None)
    last: dict[str, Any] = {}
    recorded_result_ids = ([], [])
    checked_signals = set()
    stream_result_confirmed = False
    events = conversation_events(
            backend,
            prompt=request.prompt,
            model=request.model,
            images=request.images or [],
            size=request.size,
            quality=request.quality,
            conversation_id=request.conversation_id,
            parent_message_id=request.parent_message_id,
    )
    with closing(events):
        for event in events:
            last = event
            event_conversation_id = str(event.get("conversation_id") or "")
            if event_conversation_id and callable(record_conversation_id):
                record_conversation_id(event_conversation_id)
            result_ids = (list(event.get("file_ids") or []), list(event.get("sediment_ids") or []))
            if result_ids != recorded_result_ids:
                # These IDs have already passed the image-tool output filter.
                # Persist before another stream read or URL lookup can fail/wait.
                _record_result_ids(request, *result_ids)
                recorded_result_ids = result_ids
            raw = event.get("raw") or {}
            value = raw.get("v") if isinstance(raw, dict) else None
            message = (raw.get("message") or (value.get("message") if isinstance(value, dict) else None)) if isinstance(raw, dict) else None
            terminal_signal = (isinstance(message, dict) and message.get("end_turn") is True
                               and message.get("status") == "finished_successfully"
                               and (message.get("author") or {}).get("role") == "assistant")
            signal = (tuple(result_ids[0]), tuple(result_ids[1]),
                      str(message.get("id") or "final") if terminal_signal else "assets")
            if (result_ids[0] or result_ids[1]) and signal not in checked_signals:
                checked_signals.add(signal)
                original_message_id = str(getattr(backend, "image_request_message_id", "") or request_message_id)
                if _stream_image_terminal_result(backend, event, original_message_id, *result_ids,
                                                 expected_parent=request.parent_message_id):
                    stream_result_confirmed = True
                    logger.info({"event": "image_stream_result_confirmed", "conversation_id": event_conversation_id})
                    break
            if event.get("type") == "conversation.delta":
                yield ImageOutput(
                    kind="progress",
                    model=request.model,
                    index=index,
                    total=total,
                    text=str(event.get("delta") or ""),
                    upstream_event_type="conversation.delta",
                    conversation_id=str(event.get("conversation_id") or ""),
                )
                continue
            if event.get("type") == "conversation.event":
                raw = event.get("raw")
                raw_type = str(raw.get("type") or "") if isinstance(raw, dict) else ""
                yield ImageOutput(
                    kind="progress",
                    model=request.model,
                    index=index,
                    total=total,
                    upstream_event_type=raw_type,
                    conversation_id=str(event.get("conversation_id") or ""),
                )
    submitted_request_message_id = str(
        getattr(backend, "image_request_message_id", "") or "",
    ).strip()
    if submitted_request_message_id:
        request_message_id = submitted_request_message_id
    conversation_id = str(last.get("conversation_id") or "")
    if conversation_id and callable(record_conversation_id):
        record_conversation_id(conversation_id)
    file_ids = [str(item) for item in last.get("file_ids") or []]
    sediment_ids = [str(item) for item in last.get("sediment_ids") or []]
    message = str(last.get("text") or "").strip()
    logger.info({
        "event": "image_stream_resolve_start",
        "conversation_id": conversation_id,
        "file_ids": file_ids,
        "sediment_ids": sediment_ids,
        "tool_invoked": last.get("tool_invoked"),
        "turn_use_case": last.get("turn_use_case"),
    })
    if request.progress_callback:
        request.progress_callback("image_stream_resolve_start")
    if message and not file_ids and not sediment_ids and last.get("blocked"):
        # 尝试从 /backend-api/tasks/ 获取详细错误信息
        detailed_error = _get_detailed_error_from_tasks(backend, conversation_id, wait_secs=0)
        error_text = detailed_error or message or "Image generation was rejected by upstream policy."
        raise ImageContentPolicyError(error_text, conversation_id)
    should_poll_for_image = bool(request.images) or last.get("turn_use_case") == "image gen"
    if message and not file_ids and not sediment_ids and not should_poll_for_image:
        yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=message, conversation_id=conversation_id)
        return

    # 检测模型是否返回了文本描述（含 referenced_image_ids）而非实际生成图片
    # 这说明模型已发起图片生成工具调用，但 SSE 在工具完成前断开，
    # 图片可能正在异步生成中。需要使用更积极的轮询策略来获取结果。
    is_text_reply = bool(message and is_model_text_reply_instead_of_image(message))
    if is_text_reply:
        logger.info({
            "event": "image_detected_text_reply_with_ids",
            "conversation_id": conversation_id,
            "message_preview": message[:200],
        })

    # Read the original conversation first. The poller diagnoses task errors
    # only when that authoritative read has no result assets.

    # 当检测到文本回复（含 referenced_image_ids）时，使用更长的超时来轮询图片结果。
    # 因为上游可能将图片生成作为异步任务执行，SSE 流在工具完成前就断开了，
    # 导致对话文档中尚未写入图片工具的响应记录。
    active_budget_configured = _active_budget_remaining(request) is not None
    poll_timeout = _active_poll_timeout(request, config.image_poll_timeout_secs)
    if is_text_reply and conversation_id:
        # Use the remainder of the one persisted active attempt. A text reply
        # must not create a fresh 300-second window.
        poll_timeout = _active_poll_timeout(request, max(config.image_poll_timeout_secs, 300))
        logger.info({
            "event": "image_text_reply_extended_poll",
            "conversation_id": conversation_id,
            "poll_timeout_secs": poll_timeout,
        })

    try:
        image_urls = backend.resolve_conversation_image_urls(
            conversation_id, file_ids, sediment_ids, poll_timeout_secs=poll_timeout,
            request_message_id=request_message_id,
            **({"poll": False} if stream_result_confirmed else {}),
        )
    except (ImageContentPolicyError, ImagePollTimeoutError) as exc:
        if stream_result_confirmed:
            raise
        # 当检测到文本回复时，task error 不应直接判定为内容策略违规，
        # 因为图片可能仍在后台异步生成中
        if is_text_reply and isinstance(exc, ImageContentPolicyError):
            logger.warning({
                "event": "image_text_reply_task_error_ignored",
                "conversation_id": conversation_id,
                "error": str(exc),
            })
            image_urls = []
        else:
            raise
    except Exception as exc:
        if stream_result_confirmed:
            raise
        # 当检测到文本回复时，首次轮询的临时网络错误不应直接中断，
        # 因为图片可能仍在后台异步生成中，后续 retry poll 会继续尝试。
        if is_text_reply and conversation_id:
            logger.warning({
                "event": "image_text_reply_first_poll_error_ignored",
                "conversation_id": conversation_id,
                "error": repr(exc)[:300],
            })
            image_urls = []
        else:
            raise

    if image_urls:
        _record_result_ids(request, file_ids, sediment_ids)
        if request.progress_callback:
            request.progress_callback("receiving_image")
        image_items = [
            {"b64_json": base64.b64encode(image_data).decode("ascii")}
            for image_data in backend.download_image_bytes(image_urls)
        ]
        output = _downloaded_image_output(request, image_items,
            conversation_id=conversation_id, request_message_id=request_message_id,
            file_ids=file_ids, sediment_ids=sediment_ids, index=index, total=total)
        if output.data or output._pending_image_items:
            yield output
        return

    if stream_result_confirmed:
        # The exact original turn already ended with these durable asset IDs.
        # Missing download URLs are a retrieval failure, not another generation
        # wait or a text-only answer. The task resumes its original assets.
        raise RuntimeError("generated image download URLs are not available")

    if message:
        # 检测模型是否返回了文本描述（含 referenced_image_ids）而非实际生成图片
        # 这说明模型已发起图片生成工具调用，但 SSE 在工具完成前断开。
        # 此时应再尝试轮询图片结果，而不是直接把文本当作最终输出。
        if is_text_reply and conversation_id:
            logger.info({
                "event": "image_model_text_reply_retry_poll",
                "conversation_id": conversation_id,
                "message_preview": message[:200],
            })
            retry_poll_timeout = _active_poll_timeout(
                request, max(config.image_poll_timeout_secs, 300),
            )
            max_poll_retries = 1 if active_budget_configured else 3
            for poll_attempt in range(1, max_poll_retries + 1):
                try:
                    polled_file_ids, polled_sediment_ids = backend._poll_image_results(
                        conversation_id,
                        retry_poll_timeout,
                        file_ids,
                        sediment_ids,
                        request_message_id=request_message_id,
                    )
                    file_ids.extend(item for item in polled_file_ids if item and item not in file_ids)
                    sediment_ids.extend(item for item in polled_sediment_ids if item and item not in sediment_ids)
                    break
                except Exception as exc:
                    logger.warning({
                        "event": "image_model_text_reply_poll_failed",
                        "conversation_id": conversation_id,
                        "poll_attempt": poll_attempt,
                        "error": repr(exc)[:300],
                    })
                    if (
                        poll_attempt < max_poll_retries
                        and not isinstance(exc, (ImagePollTimeoutError, ImageContentPolicyError))
                    ):
                        time.sleep(30.0 * poll_attempt)
                        continue
                    if active_budget_configured:
                        raise
                    break
            _record_result_ids(request, file_ids, sediment_ids)

            if file_ids or sediment_ids:
                image_urls = backend.resolve_conversation_image_urls(
                    conversation_id, file_ids, sediment_ids, poll=False,
                    request_message_id=request_message_id,
                )
                if image_urls:
                    if request.progress_callback:
                        request.progress_callback("receiving_image")
                    image_items = [
                        {"b64_json": base64.b64encode(image_data).decode("ascii")}
                        for image_data in backend.download_image_bytes(image_urls)
                    ]
                    output = _downloaded_image_output(request, image_items,
                        conversation_id=conversation_id, request_message_id=request_message_id,
                        file_ids=file_ids, sediment_ids=sediment_ids, index=index, total=total)
                    if output.data or output._pending_image_items:
                        yield output
                        return
            if active_budget_configured:
                raise ImagePollTimeoutError(
                    "image request may still be processing after the active generation deadline",
                    conversation_id,
                )
        elif is_text_reply:
            logger.warning({
                "event": "image_model_text_reply_no_image",
                "conversation_id": conversation_id,
                "message_preview": message[:200],
            })
        yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=message, conversation_id=conversation_id)
        return

    # 兜底：当 message 为空且图片 URL 解析失败时，先尝试一次短延迟重试轮询
    # 然后抛出明确错误而非让调用方得到 "upstream completed without generating images" 这种模糊报错
    logger.warning({
        "event": "image_stream_no_result_fallback",
        "conversation_id": conversation_id,
        "file_ids": file_ids,
        "sediment_ids": sediment_ids,
        "should_poll_for_image": should_poll_for_image,
    })
    if should_poll_for_image and conversation_id:
        retry_poll_timeout = _active_poll_timeout(
            request, max(config.image_poll_timeout_secs, 300),
        )
        max_fallback_retries = 1 if active_budget_configured else 3
        for poll_attempt in range(1, max_fallback_retries + 1):
            retry_wait_secs = min(
                30.0 * poll_attempt,
                float(config.image_poll_initial_wait_secs) * poll_attempt,
            )
            if active_budget_configured:
                retry_wait_secs = min(
                    retry_wait_secs,
                    _active_poll_timeout(request, retry_wait_secs),
                )
            if retry_wait_secs > 0:
                time.sleep(retry_wait_secs)
            try:
                polled_file_ids, polled_sediment_ids = backend._poll_image_results(
                    conversation_id,
                    _active_poll_timeout(request, retry_poll_timeout),
                    file_ids,
                    sediment_ids,
                    request_message_id=request_message_id,
                )
                file_ids.extend(item for item in polled_file_ids if item and item not in file_ids)
                sediment_ids.extend(item for item in polled_sediment_ids if item and item not in sediment_ids)
                break
            except Exception as exc:
                logger.warning({
                    "event": "image_stream_retry_poll_failed",
                    "conversation_id": conversation_id,
                    "poll_attempt": poll_attempt,
                    "error": repr(exc)[:300],
                })
                if (
                    poll_attempt < max_fallback_retries
                    and not isinstance(exc, (ImagePollTimeoutError, ImageContentPolicyError))
                ):
                    time.sleep(30.0 * poll_attempt)
                    continue
                if active_budget_configured:
                    raise
                break
        _record_result_ids(request, file_ids, sediment_ids)
        
        if file_ids or sediment_ids:
            image_urls = backend.resolve_conversation_image_urls(
                conversation_id, file_ids, sediment_ids, poll=False,
                request_message_id=request_message_id,
            )
            if image_urls:
                if request.progress_callback:
                    request.progress_callback("receiving_image")
                image_items = [
                    {"b64_json": base64.b64encode(image_data).decode("ascii")}
                    for image_data in backend.download_image_bytes(image_urls)
                ]
                output = _downloaded_image_output(request, image_items,
                    conversation_id=conversation_id, request_message_id=request_message_id,
                    file_ids=file_ids, sediment_ids=sediment_ids, index=index, total=total)
                if output.data or output._pending_image_items:
                    yield output
                    return
        
        if active_budget_configured:
            raise ImagePollTimeoutError(
                "image request may still be processing after the active generation deadline",
                conversation_id,
            )
        yield ImageOutput(
            kind="message",
            model=request.model,
            index=index,
            total=total,
            text=(
                "Image generation completed upstream but the result could not be retrieved. "
                "The image may still be processing. Please try again in a moment."
            ),
            conversation_id=conversation_id,
        )
    elif message:
        yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=message, conversation_id=conversation_id)
    else:
        # conversation_id 也为空时（SSE 流极短、未捕获到会话 ID），
        # 仍然 yield 一条消息，避免 stream_image_outputs_with_pool 产生
        # "upstream completed without generating images" 模糊报错
        yield ImageOutput(kind="message", model=request.model, index=index, total=total,
                          text="Image generation started upstream but the response was incomplete. "
                               "Please try again.",
                          conversation_id=conversation_id)


def _codex_response_images(value: Any) -> list[str]:
    if isinstance(value, dict):
        if value.get("type") == "image_generation_call" and isinstance(value.get("result"), str):
            result = value["result"].strip()
            if result:
                return [result.split(",", 1)[1] if result.startswith("data:image/") else result]
        images: list[str] = []
        for item in value.values():
            images.extend(_codex_response_images(item))
        return images
    if isinstance(value, list):
        images: list[str] = []
        for item in value:
            images.extend(_codex_response_images(item))
        return images
    return []


def stream_codex_image_outputs(
        backend: OpenAIBackendAPI,
        request: ConversationRequest,
        index: int = 1,
        total: int = 1,
) -> Iterator[ImageOutput]:
    images = _codex_response_images(list(backend.iter_codex_image_response_events(
        prompt=request.prompt,
        images=request.images or [],
        size=request.size,
        quality=request.quality,
    )))
    if not images:
        raise ImageGenerationError("No image result found in response")
    data = format_image_result(
        [{"b64_json": item, "revised_prompt": request.prompt} for item in images],
        request.prompt,
        request.response_format,
        request.base_url,
        int(time.time()),
    )["data"]
    if data:
        yield ImageOutput(kind="result", model=request.model, index=index, total=total, data=data)
        return
    raise ImageGenerationError("No image result found in response")


def _generate_bound_single_image(
        request: ConversationRequest,
        index: int,
        total: int,
) -> list[ImageOutput]:
    """Run one durable conversation step without account or conversation failover."""
    if is_codex_image_model(request.model):
        raise ImageGenerationError(
            "conversation binding is not supported for the selected image model",
            code="CONVERSATION_BINDING_UNSUPPORTED",
            provider_binding_id=request.provider_binding_id,
            conversation_id=request.conversation_id,
            parent_message_id=request.parent_message_id,
        )
    if bool(request.conversation_id) != bool(request.parent_message_id):
        raise ImageGenerationError(
            "conversation continuation requires conversation_id and parent_message_id",
            code="CONVERSATION_BINDING_CONTRACT_INVALID",
            provider_binding_id=request.provider_binding_id,
            conversation_id=request.conversation_id,
            parent_message_id=request.parent_message_id,
        )
    if not request.client_conversation_id:
        raise ImageGenerationError(
            "client_conversation_id is required for retained conversation binding",
            code="CONVERSATION_BINDING_CONTRACT_INVALID",
            provider_binding_id=request.provider_binding_id,
            provider_account_identity=request.provider_account_identity,
            conversation_id=request.conversation_id,
            parent_message_id=request.parent_message_id,
        )

    def check_current_parent(send=None) -> None:
        """Refuse a retained image continuation after its saved reply ceased to be the tip.

        A normal direct-listing continuation has no Provider image-thread
        envelope, so it cannot rely on the thread-only predecessor proof below.
        The same check runs once before streaming and again at the paced POST
        edge.  It accepts a completed final assistant reply or the existing
        completed tool/image endpoint; malformed, active, branched, or later
        user state is never a valid image parent.
        """
        try:
            document = backend._get_conversation(
                request.conversation_id,
                **({"_send": send} if send else {}),
            )
        except Exception as exc:
            raise ImageGenerationError(
                "bound image conversation could not be read before submission",
                code="CONVERSATION_OUTCOME_UNKNOWN",
                provider_binding_id=binding_id,
                provider_account_identity=account_identity,
                conversation_id=request.conversation_id,
                parent_message_id=request.parent_message_id,
                upstream_submitted=False,
            ) from exc
        mapping = document.get("mapping") if isinstance(document, dict) else None
        mapping_valid = (
            isinstance(mapping, dict)
            and bool(mapping)
            and all(isinstance(node, dict) for node in mapping.values())
        )
        parent = mapping.get(request.parent_message_id) if mapping_valid else None
        message = parent.get("message") if isinstance(parent, dict) else None
        author = message.get("author") if isinstance(message, dict) else None
        children = (
            mapping_valid
            and isinstance(parent, dict)
            and isinstance(parent.get("children"), list)
            and not parent["children"]
            and not any(isinstance(node, dict) and node.get("parent") == request.parent_message_id
                        for node in mapping.values())
        )
        assistant_final = (
            isinstance(message, dict)
            and message.get("id") == request.parent_message_id
            and isinstance(author, dict)
            and author.get("role") == "assistant"
            and message.get("status") == "finished_successfully"
            and message.get("end_turn") is True
            and message.get("channel") in {None, "final"}
            and children
        )
        image_tool_final = False
        if mapping_valid:
            from services.conversation_binding_service import ConversationBindingService
            image_tool_final = ConversationBindingService._completed_image_tool_parent(
                mapping, request.parent_message_id,
            )
        if (
            not isinstance(document, dict)
            or not mapping_valid
            or str(document.get("conversation_id") or "").strip() != request.conversation_id
            or str(document.get("current_node") or "").strip() != request.parent_message_id
            or not (assistant_final or image_tool_final)
        ):
            raise ImageGenerationError(
                "bound image conversation cursor changed before submission",
                code="CONVERSATION_BINDING_MISMATCH",
                provider_binding_id=binding_id,
                provider_account_identity=account_identity,
                conversation_id=request.conversation_id,
                parent_message_id=request.parent_message_id,
                upstream_submitted=False,
            )

    token = ""
    binding_id = request.provider_binding_id
    account_identity = request.provider_account_identity
    from services.request_context import current_request
    pool_managed = bool(binding_id) and current_request.get() is not None
    try:
        if binding_id:
            authoritative_identity = account_service.get_bound_account_identity(binding_id)
            if not account_identity or authoritative_identity != account_identity:
                raise ImageGenerationError(
                    "provider account identity changed",
                    code="CONVERSATION_BINDING_MISMATCH",
                    provider_binding_id=binding_id,
                    provider_account_identity=account_identity,
                    conversation_id=request.conversation_id,
                    parent_message_id=request.parent_message_id,
                )
            token = account_service.acquire_bound_image_access_token(
                binding_id,
                image_model=request.model,
                **({"reserve_slot": False} if pool_managed else {}),
            )
        else:
            binding_id, account_identity, token = account_service.create_conversation_binding(
                image_model=request.model,
            )
            request.provider_binding_id = binding_id
            request.provider_account_identity = account_identity
    except RuntimeError as exc:
        error = ImageGenerationError(
            str(exc) or "conversation binding unavailable",
            code="CONVERSATION_BINDING_UNAVAILABLE",
            provider_binding_id=binding_id,
            provider_account_identity=account_identity,
            conversation_id=request.conversation_id,
            parent_message_id=request.parent_message_id,
        )
        # This block ends before any backend/stream is created. Preserve that
        # positive evidence; the same error code elsewhere can be uncertain.
        error.upstream_submitted = False
        error.binding_reason = getattr(exc, "binding_reason", None) or {
            "conversation binding unavailable: binding id is required": "binding_id_required",
            "conversation binding unavailable: bound account missing": "bound_account_missing",
            "conversation binding unavailable: bound account cannot generate images": "bound_image_capability_unavailable",
        }.get(str(exc))
        raise error from exc

    slot_acquired = bool(token) and not pool_managed
    image_result_marked = False
    capacity_consumed = False

    def record_capacity_consumed():
        nonlocal capacity_consumed
        if not capacity_consumed:
            account_service.mark_image_capacity_consumed(token)
            capacity_consumed = True

    account_email = ""
    backend: OpenAIBackendAPI | None = None
    outputs: list[ImageOutput] = []
    last_conversation_id = request.conversation_id
    try:
        account = account_service.get_account(token) or {}
        account_email = str(account.get("email") or "").strip()
        with account_service.conversation_binding_lock(binding_id, request.client_conversation_id):
            # Account capacity/cooldown and the per-Chat serialization wait are
            # outside the active generation budget. Preparation starts here.
            _start_active_attempt(request)
            backend = OpenAIBackendAPI(access_token=token)
            # Request-owned model selection; never mutate the shared pool
            # default used by Content or other callers.
            backend.retain_bound_conversation = True
            backend.image_upstream_model = request.upstream_model
            if request.progress_callback:
                backend.progress_callback = request.progress_callback
            thread = getattr(request.progress_callback, "image_thread", None)
            failed_original = getattr(request.progress_callback, "failed_retry_original", None)
            try:
                if failed_original or getattr(request.progress_callback, "failed_retry_required", False):
                    def check_retry(send=None):
                        from services.generation_completion import retry_cursor, retry_evidence
                        if (not isinstance(failed_original, dict) or failed_original.get("result_file_ids")
                                or failed_original.get("result_sediment_ids") or failed_original.get("data")
                                or failed_original.get("recovery_phase") == "download_image_result"):
                            raise ImageGenerationError("original retry evidence unavailable",
                                code="COMPLETION_ORIGINAL_CURSOR_UNCONFIRMED", upstream_submitted=False)
                        try:
                            document = backend._get_conversation(request.conversation_id, **({"_send": send} if send else {}))
                        except Exception as exc:
                            raise ImageGenerationError("original retry branch read unavailable",
                                code="COMPLETION_ORIGINAL_READ_UNAVAILABLE", upstream_submitted=False) from exc
                        current = retry_cursor(document, failed_original, kind="image",
                            predecessor=getattr(request.progress_callback, "failed_retry_predecessor", None))
                        saved = retry_evidence(failed_original)
                        if not current or not saved or any(current.get(k) != saved.get(k) for k in (
                                "conversation_id", "request_message_id", "retry_parent_message_id", "source")):
                            raise ImageGenerationError("original retry branch changed",
                                code="COMPLETION_ORIGINAL_CURSOR_UNCONFIRMED", upstream_submitted=False)
                    check_retry()
                    backend.image_pre_send_check = check_retry
                elif thread and request.conversation_id:
                    try:
                        prior_message = getattr(request.progress_callback, "image_thread_predecessor_message", None)
                        proof = getattr(request.progress_callback, "image_thread_predecessor_cursor_proof", None) or {}
                        document = backend._get_conversation(request.conversation_id)
                        parent = finished_parent(document, request.conversation_id, prior_message,
                            expected_result_ids=getattr(request.progress_callback,
                                "image_thread_predecessor_result_ids", None),
                            expected_parent=proof.get("_image_thread_request_parent"),
                            predecessor_request_message_id=proof.get("_image_thread_predecessor_message"),
                            predecessor_result_ids=proof.get("_image_thread_predecessor_result_ids"))
                        if parent != request.parent_message_id:
                            from services.image_thread import archive_parent
                            if not proof or proof.get("result_file_ids") is None or proof.get("result_sediment_ids") is None:
                                raise ImageThreadError("IMAGE_THREAD_UPSTREAM_CHANGED")
                            parent = archive_parent(document, request.conversation_id, prior_message,
                                request.parent_message_id, getattr(request.progress_callback,
                                    "image_thread_predecessor_result_ids", None) or [],
                                expected_parent=proof.get("_image_thread_request_parent"),
                                predecessor_request_message_id=proof.get("_image_thread_predecessor_message"),
                                predecessor_result_ids=proof.get("_image_thread_predecessor_result_ids"),
                                expected_file_ids=proof["result_file_ids"],
                                expected_sediment_ids=proof["result_sediment_ids"])
                            request = replace(request, parent_message_id=parent)
                        # A later review reversal continues this exact conversation.
                        # Restore it before any image POST; failure leaves the original
                        # request unsubmitted and available for exact-ID recovery.
                        if document.get("is_archived") is True:
                            backend.set_conversation_archived(request.conversation_id, parent, False)
                    except Exception as exc:
                        raise ImageGenerationError("Original image conversation cannot yet be continued",
                            code=getattr(exc, "code", "IMAGE_THREAD_PREVIOUS_UNCONFIRMED"), upstream_submitted=False) from exc
                elif request.conversation_id:
                    # Non-thread bound edits, including the original listing
                    # revision route, still must not fork from a completed
                    # reply after a later user turn. Keep the Provider's
                    # current-parent check at the actual POST boundary too.
                    check_current_parent()
                    backend.image_pre_send_check = check_current_parent
                request = replace(request)
                request._defer_image_publication = True
                if pool_managed:
                    request._record_image_capacity_consumed = record_capacity_consumed
                confirmed_result = None
                if thread:
                    def poll_terminal(document, conversation_id, request_message_id, files, sediments):
                        nonlocal confirmed_result
                        confirmed_result = None
                        try:
                            checks = dict(expected_parent=request.parent_message_id or None,
                                expected_result_ids=files + sediments,
                                predecessor_request_message_id=getattr(request.progress_callback,
                                    "image_thread_predecessor_message", None),
                                predecessor_result_ids=getattr(request.progress_callback,
                                    "image_thread_predecessor_result_ids", None))
                            try:
                                parent = finished_parent(document, conversation_id, request_message_id,
                                    require_final=True, **checks)
                            except ImageThreadError:
                                finished_parent(document, conversation_id, request_message_id, **checks)
                            else:
                                # Call-local proof for these immutable original
                                # assets; never restore it from telemetry/cache.
                                confirmed_result = (conversation_id, request_message_id,
                                                    frozenset(files + sediments), parent)
                            _observe_image_terminal()
                            return True
                        except ImageThreadError:
                            return False
                    # Tool leaves still need the post-download fence. A complete
                    # assistant final can already prove this original result.
                    backend.image_poll_terminal_check = poll_terminal
                for output in stream_image_outputs(backend, request, index, total):
                    last_conversation_id = output.conversation_id or last_conversation_id
                    output.account_email = output.account_email or account_email
                    output.provider_binding_id = binding_id
                    output.provider_account_identity = account_identity
                    if output.kind == "message" and request.message_as_error:
                        error = _message_output_error(output, account_email=account_email)
                        error.provider_binding_id = binding_id
                        error.provider_account_identity = account_identity
                        error.conversation_id = last_conversation_id
                        raise error
                    outputs.append(output)
                if not last_conversation_id:
                    raise ImageGenerationError(
                        "upstream response has no conversation_id",
                        code="CONVERSATION_OUTCOME_UNKNOWN",
                        account_email=account_email,
                        provider_binding_id=binding_id,
                    )
                if request.conversation_id and last_conversation_id != request.conversation_id:
                    raise ImageGenerationError(
                        "upstream conversation identity changed",
                        code="CONVERSATION_BINDING_MISMATCH",
                        account_email=account_email,
                        provider_binding_id=binding_id,
                        conversation_id=last_conversation_id,
                    )
                if not any(output.kind == "result" for output in outputs):
                    raise ImageGenerationError(
                        "upstream completed without generating images",
                        code="no_image_generated",
                        account_email=account_email,
                        provider_binding_id=binding_id,
                        conversation_id=last_conversation_id,
                    )
                if thread:
                    original_message = str(getattr(backend, "image_request_message_id", ""))
                    result_ids = frozenset(getattr(request.progress_callback, "image_thread_result_ids", None) or [])
                    downloaded_count = sum(len(output._pending_image_items) for output in outputs if output.kind == "result")
                    if (confirmed_result and confirmed_result[:3] == (last_conversation_id, original_message, result_ids)
                            and downloaded_count == len(result_ids)):
                        next_parent_message_id = confirmed_result[3]
                        logger.info({"event": "image_complete_final_reused", "conversation_id": last_conversation_id,
                                     "image_count": downloaded_count})
                    else:
                        next_parent_message_id = finished_parent(backend._get_conversation(last_conversation_id),
                            last_conversation_id, original_message,
                            expected_parent=request.parent_message_id or None,
                            expected_result_ids=result_ids,
                            predecessor_request_message_id=getattr(request.progress_callback, "image_thread_predecessor_message", None),
                            predecessor_result_ids=getattr(request.progress_callback, "image_thread_predecessor_result_ids", None))
                    _observe_image_terminal()
                else:
                    next_parent_message_id = backend.get_conversation_parent_message_id(last_conversation_id)
                for output in outputs:
                    if output._pending_image_items:
                        output.data = format_image_result(output._pending_image_items, request.prompt,
                            request.response_format, request.base_url, output.created)["data"]
                        output._pending_image_items = []
                    output.provider_binding_id = binding_id
                    output.provider_account_identity = account_identity
                    output.conversation_id = last_conversation_id
                    output.parent_message_id = next_parent_message_id
                    output.image_thread_terminal = bool(thread)
                image_result_marked = True
                account_service.mark_image_result(token, True, **({"release_slot": False} if pool_managed else {}),
                    **({"capacity_consumed": True} if capacity_consumed else {}))
                return outputs
            except Exception as exc:
                from services.request_context import AdmissionLost
                upstream_submitted = getattr(backend, "image_submission_started", None)
                # A local guard rejection before the generation POST consumes
                # no image capacity. Do not invalidate a fresh observation or
                # count it as an upstream image failure. Unknown/submitted
                # attempts retain the existing accounting and recovery path.
                unsent_guard_rejection = isinstance(exc, AdmissionLost) and upstream_submitted is False
                if not image_result_marked and not unsent_guard_rejection:
                    image_result_marked = True
                    account_service.mark_image_result(token, False, **({"release_slot": False} if pool_managed else {}),
                        **({"capacity_consumed": True} if capacity_consumed else {}))
                conversation_id = str(getattr(exc, "conversation_id", "") or last_conversation_id)
                parent_message_id = str(getattr(exc, "parent_message_id", "") or "")
                request_message_id = str(getattr(backend, "image_request_message_id", "") or "")
                if not thread and conversation_id and backend is not None and not parent_message_id:
                    try:
                        parent_message_id = backend.get_conversation_parent_message_id(conversation_id)
                    except Exception:
                        pass
                if isinstance(exc, ImageGenerationError):
                    exc.provider_binding_id = binding_id
                    exc.provider_account_identity = account_identity
                    exc.conversation_id = conversation_id
                    exc.parent_message_id = parent_message_id
                    exc.request_message_id = request_message_id
                    if exc.upstream_submitted is None and isinstance(upstream_submitted, bool):
                        exc.upstream_submitted = upstream_submitted
                    raise
                if isinstance(exc, ImageContentPolicyError):
                    raise ImageGenerationError(
                        str(exc),
                        status_code=400,
                        error_type="invalid_request_error",
                        code="content_policy_violation",
                        account_email=account_email,
                        provider_binding_id=binding_id,
                        provider_account_identity=account_identity,
                        conversation_id=conversation_id,
                        parent_message_id=parent_message_id,
                        request_message_id=request_message_id,
                        upstream_submitted=(
                            upstream_submitted if isinstance(upstream_submitted, bool) else None
                        ),
                    ) from exc
                error = ImageGenerationError(
                    image_stream_error_message(str(exc)),
                    code=(
                        "IMAGE_GENERATION_NOT_SUBMITTED"
                        if upstream_submitted is False
                        else "CONVERSATION_OUTCOME_UNKNOWN"
                        if conversation_id
                        else "CONVERSATION_BINDING_UNAVAILABLE"
                    ),
                    account_email=account_email,
                    provider_binding_id=binding_id,
                    provider_account_identity=account_identity,
                    conversation_id=conversation_id,
                    parent_message_id=parent_message_id,
                    request_message_id=request_message_id,
                    upstream_submitted=(
                        upstream_submitted if isinstance(upstream_submitted, bool) else None
                    ),
                )
                upstream_status = getattr(exc, "status_code", None)
                if type(upstream_status) is int and 400 <= upstream_status <= 599:
                    error.status_code = upstream_status
                retry_after = getattr(exc, "retry_after", None)
                if (type(retry_after) in (int, float) and math.isfinite(retry_after)
                        and retry_after >= 0):
                    error.retry_after = retry_after
                raise error from exc
    finally:
        try:
            if backend is not None:
                backend.close()
        finally:
            if slot_acquired and not image_result_marked:
                account_service.release_image_slot(token)


def _generate_single_image(
        request: ConversationRequest,
        index: int,
        total: int,
) -> list[ImageOutput]:
    """为单张图片执行生成逻辑（含重试），返回结果列表。

    该函数在独立线程中运行，每个线程使用不同的账号，
    实现并行生图，避免串行超时阻塞。
    """
    if request.retain_conversation:
        return _generate_bound_single_image(request, index, total)

    # 模型返回文本而非图片的最大重试次数
    MAX_TEXT_REPLY_RETRIES = 3
    # TLS 连接错误最大重试次数
    MAX_TLS_RETRIES = 3
    # 连接超时错误最大重试次数（同账号短等待重试）
    MAX_CONN_TIMEOUT_RETRIES = 3
    # 轮询超时错误最大重试次数（换账号重试）
    MAX_POLL_TIMEOUT_RETRIES = 4

    text_reply_retry_count = 0
    tls_retry_count = 0
    conn_timeout_retry_count = 0
    poll_timeout_retry_count = 0
    account_email = ""

    while True:
        try:
            if request.progress_callback:
                request.progress_callback("getting_account")
            plan_type, _ = split_image_model(request.model)
            codex_model = is_codex_image_model(request.model)
            token = account_service.get_available_access_token(
                plan_type=plan_type,
                source_type="codex" if codex_model else None,
                plan_types=("plus", "team", "pro") if codex_model and not plan_type else None,
            )
        except RuntimeError as exc:
            raise ImageGenerationError(str(exc) or "image generation failed", account_email=account_email) from exc

        emitted_for_token = False
        returned_message = False
        returned_result = False
        account = account_service.get_account(token) or {}
        account_email = str(account.get("email") or "").strip()
        logger.debug({
            "event": "image_account_lookup",
            "token_prefix": token[:12] + "..." if len(token) > 12 else token,
            "account_email": account_email,
            "account_found": bool(account),
            "index": index,
        })
        backend = None
        try:
            # Unbound pool selection is also admission waiting. Start the
            # active budget only after the account has been selected.
            _start_active_attempt(request)
            backend = OpenAIBackendAPI(access_token=token)
            if request.progress_callback:
                backend.progress_callback = request.progress_callback
            stream_fn = stream_codex_image_outputs if is_codex_image_model(request.model) else stream_image_outputs
            backend.retain_bound_conversation = True
            backend.image_upstream_model = request.upstream_model
            outputs: list[ImageOutput] = []
            last_conversation_id = ""
            try:
                for output in stream_fn(backend, request, index, total):
                    last_conversation_id = output.conversation_id or last_conversation_id
                    if account_email and not output.account_email:
                        output.account_email = account_email
                    if output.kind == "message" and request.message_as_error:
                        raise _message_output_error(output, account_email=account_email)
                    emitted_for_token = True
                    returned_message = output.kind == "message"
                    returned_result = returned_result or output.kind == "result"
                    outputs.append(output)
            except Exception as exc:
                # 异常路径（内容政策拒绝、轮询超时等）会话 ID 只挂在异常上
                last_conversation_id = last_conversation_id or str(getattr(exc, "conversation_id", "") or "")
                raise
            finally:
                _remove_image_conversation_later(backend, last_conversation_id, success=returned_result)
            if returned_message:
                account_service.mark_image_result(token, False)
                return outputs
            if not returned_result:
                account_service.mark_image_result(token, False)
                if emitted_for_token:
                    conv_id = outputs[-1].conversation_id if outputs else ""
                    raise ImageGenerationError(
                        "upstream completed without generating images",
                        status_code=400,
                        error_type="invalid_request_error",
                        code="no_image_generated",
                        account_email=account_email,
                        conversation_id=conv_id,
                    )
                return outputs
            account_service.mark_image_result(token, True)
            return outputs
        except ImagePollTimeoutError as exc:
            account_service.mark_image_result(token, False)
            if account_email:
                setattr(exc, "account_email", account_email)
            # 轮询超时：换账号重试
            if not emitted_for_token and getattr(backend, "image_submission_started", None) is not True:
                poll_timeout_retry_count += 1
                if poll_timeout_retry_count <= MAX_POLL_TIMEOUT_RETRIES:
                    logger.warning({
                        "event": "image_poll_timeout_retry",
                        "request_token": token,
                        "account_email": account_email,
                        "retry_count": poll_timeout_retry_count,
                        "index": index,
                        "error": str(exc)[:200],
                    })
                    continue
                logger.warning({
                    "event": "image_poll_timeout_exhausted_retries",
                    "request_token": token,
                    "account_email": account_email,
                    "retry_count": poll_timeout_retry_count,
                    "index": index,
                })
                raise
            raise
        except ImageContentPolicyError as exc:
            account_service.mark_image_result(token, False)
            logger.warning({
                "event": "image_stream_content_policy_error",
                "request_token": token,
                "account_email": account_email,
                "error": str(exc),
                "index": index,
            })
            raise ImageGenerationError(
                str(exc) or "Image generation was rejected by upstream policy.",
                status_code=400,
                error_type="invalid_request_error",
                code="content_policy_violation",
                account_email=account_email,
                conversation_id=getattr(exc, "conversation_id", ""),
            ) from exc
        except ImageGenerationError as exc:
            account_service.mark_image_result(token, False)
            if account_email and not getattr(exc, "account_email", ""):
                exc.account_email = account_email
            error_text = str(exc)
            # 如果是模型返回文本而非图片，尝试换账号重试
            if (is_model_text_reply_instead_of_image(error_text) and not emitted_for_token
                    and getattr(backend, "image_submission_started", None) is not True):
                text_reply_retry_count += 1
                if text_reply_retry_count <= MAX_TEXT_REPLY_RETRIES:
                    logger.warning({
                        "event": "image_model_text_reply_retry",
                        "request_token": token,
                        "account_email": account_email,
                        "retry_count": text_reply_retry_count,
                        "index": index,
                        "error": error_text[:200],
                    })
                    continue
                logger.warning({
                    "event": "image_model_text_reply_exhausted_retries",
                    "request_token": token,
                    "account_email": account_email,
                    "retry_count": text_reply_retry_count,
                    "index": index,
                })
                raise ImageGenerationError(
                    "Image generation failed: the upstream model returned a text description "
                    "instead of generating an image. Please try again later.",
                    status_code=502,
                    error_type="server_error",
                    code="upstream_text_reply",
                    account_email=account_email,
                    conversation_id=getattr(exc, "conversation_id", ""),
                ) from exc
            logger.warning({
                "event": "image_stream_generation_error",
                "request_token": token,
                "account_email": account_email,
                "error": error_text,
                "index": index,
            })
            raise
        except Exception as exc:
            account_service.mark_image_result(token, False)
            last_error = str(exc)
            logger.warning({
                "event": "image_stream_fail",
                "request_token": token,
                "account_email": account_email,
                "error": last_error,
                "index": index,
            })
            if getattr(backend, "image_submission_started", None) is True:
                # No yielded chunk is not proof that no generation was sent.
                # In particular, a confirmed asset's URL failure must never
                # re-enter generation or switch the selected account.
                raise ImageGenerationError(image_stream_error_message(last_error),
                    account_email=account_email, conversation_id=last_conversation_id,
                    upstream_submitted=True) from exc
            if not emitted_for_token and is_token_invalid_error(last_error):
                refreshed_token = account_service.refresh_access_token(token, force=True, event="image_stream")
                if refreshed_token and refreshed_token != token:
                    token = refreshed_token
                    continue
                account_service.remove_invalid_token(token, "image_stream")
                continue
            # TLS/SSL 连接错误：自动重试
            if not emitted_for_token and is_tls_connection_error(last_error):
                tls_retry_count += 1
                if tls_retry_count <= MAX_TLS_RETRIES:
                    logger.warning({
                        "event": "image_stream_tls_retry",
                        "request_token": token,
                        "account_email": account_email,
                        "retry_count": tls_retry_count,
                        "index": index,
                        "error": last_error[:200],
                    })
                    time.sleep(min(2.0 * tls_retry_count, 10.0))
                    continue
            # 连接超时错误（curl 28）：同账号短等待重试，不切换账号
            if not emitted_for_token and is_connection_timeout_error(last_error):
                conn_timeout_retry_count += 1
                if conn_timeout_retry_count <= MAX_CONN_TIMEOUT_RETRIES:
                    wait_secs = min(3.0 * conn_timeout_retry_count, 9.0)
                    logger.warning({
                        "event": "image_stream_conn_timeout_retry",
                        "request_token": token,
                        "account_email": account_email,
                        "retry_count": conn_timeout_retry_count,
                        "index": index,
                        "wait_secs": wait_secs,
                        "error": last_error[:200],
                    })
                    time.sleep(wait_secs)
                    continue
            raise ImageGenerationError(image_stream_error_message(last_error), account_email=account_email, conversation_id="") from exc
        finally:
            if backend is not None:
                backend.close()


def stream_image_outputs_with_pool(request: ConversationRequest) -> Iterator[ImageOutput]:
    """并行生成多张图片，每张图片使用独立线程和账号，互不阻塞。"""
    if not is_supported_image_model(request.model):
        raise ImageGenerationError("unsupported image model,supported models: " + ", ".join(sorted(IMAGE_MODELS)))

    from services.request_context import current_request
    context = current_request.get()
    if context is not None:
        # The admitted legacy multi-output call retains one account reservation
        # and executes its bounded slots in order. It cannot escape the original
        # claim through context-less ThreadPoolExecutor workers or retry a slot.
        from services import durable_image_forward
        recoverable = durable_image_forward.prepare(context, request)
        for index in range(request.n):
            if recoverable and (saved := durable_image_forward.cached_slot(context, index)) is not None:
                yield from saved
                continue
            context.image_slot(index)
            slot_request = durable_image_forward.slot_request(context, request, index) if recoverable else request
            outputs = _generate_single_image(slot_request, index + 1, request.n)
            if recoverable:
                durable_image_forward.save_slot(context, index, outputs)
            else:
                context.image_slot_complete(index)
            yield from outputs
        return

    if request.n <= 1:
        # 单张图片，直接执行（无需线程池开销）
        outputs = _generate_single_image(request, 1, 1)
        for output in outputs:
            yield output
        return

    # 多张图片：根据配置选择并行或串行执行
    if not config.image_parallel_generation:
        logger.info({
            "event": "image_serial_generation_start",
            "n": request.n,
            "model": request.model,
        })
        for index in range(1, request.n + 1):
            outputs = _generate_single_image(request, index, request.n)
            for output in outputs:
                yield output
        return

    logger.info({
        "event": "image_parallel_generation_start",
        "n": request.n,
        "model": request.model,
    })
    # 每张图片一个线程，同时启动
    futures = {}
    results: dict[int, list[ImageOutput]] = {}
    errors: dict[int, Exception] = {}
    with ThreadPoolExecutor(max_workers=request.n) as executor:
        for index in range(1, request.n + 1):
            future = executor.submit(_generate_single_image, request, index, request.n)
            futures[future] = index

        # 按完成顺序收集结果
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                errors[index] = exc
                logger.warning({
                    "event": "image_parallel_generation_error",
                    "index": index,
                    "error": str(exc)[:300],
                })

    # yield 结果：跳过索引顺序限制，不再让低索引失败阻塞高索引成功结果
    emitted = False
    last_error = ""
    # 先 yield 所有成功的结果
    for index in range(1, request.n + 1):
        if index in results:
            for output in results[index]:
                emitted = True
                yield output
        elif index in errors:
            last_error = str(errors[index])
            if not emitted:
                logger.warning({
                    "event": "image_parallel_failure_before_success",
                    "failed_index": index,
                    "error": last_error[:200],
                })

    # 如果有失败但也有成功，记录警告
    if emitted:
        for index in range(1, request.n + 1):
            if index in errors:
                logger.warning({
                    "event": "image_parallel_partial_failure",
                    "failed_index": index,
                    "error": str(errors[index])[:200],
                })

    if not emitted:
        if not last_error:
            last_error = "no account in the pool could generate images — check account quota and rate-limit status"
        raise ImageGenerationError(image_stream_error_message(last_error), conversation_id="")


def stream_image_chunks(outputs: Iterable[ImageOutput]) -> Iterator[dict[str, Any]]:
    for output in outputs:
        yield output.to_chunk()


def collect_image_outputs(outputs: Iterable[ImageOutput]) -> dict[str, Any]:
    created = None
    data: list[dict[str, Any]] = []
    message = ""
    progress_parts: list[str] = []
    account_email = ""
    provider_binding_id = ""
    provider_account_identity = ""
    conversation_id = ""
    parent_message_id = ""
    image_thread_terminal = False
    for output in outputs:
        created = created or output.created
        if output.account_email and not account_email:
            account_email = output.account_email
        provider_binding_id = output.provider_binding_id or provider_binding_id
        provider_account_identity = output.provider_account_identity or provider_account_identity
        conversation_id = output.conversation_id or conversation_id
        parent_message_id = output.parent_message_id or parent_message_id
        image_thread_terminal = output.image_thread_terminal or image_thread_terminal
        if output.kind == "progress" and output.text:
            progress_parts.append(output.text)
        elif output.kind == "message":
            message = output.text
        elif output.kind == "result":
            data.extend(output.data)

    result: dict[str, Any] = {"created": created or int(time.time()), "data": data}
    if not data:
        text = message or "".join(progress_parts).strip()
        if text:
            result["message"] = text
    if account_email:
        result["_account_email"] = account_email
    if provider_binding_id:
        result["_provider_binding_id"] = provider_binding_id
    if provider_account_identity:
        result["_provider_account_identity"] = provider_account_identity
    if conversation_id:
        result["_conversation_id"] = conversation_id
    if parent_message_id:
        result["_parent_message_id"] = parent_message_id
    if image_thread_terminal:
        result["_image_thread_terminal"] = True
    return result
