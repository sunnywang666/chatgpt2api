import base64
import hashlib
import json
import mimetypes
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from curl_cffi import requests
from fastapi import HTTPException
from services.proxy_service import proxy_settings
from utils.log import logger
from utils.upstream_error import (
    safe_upstream_error,
    safe_upstream_response_ids,
    sse_error_detail,
    upstream_response_ids_from_headers,
)

BASE_IMAGE_MODELS = {"gpt-image-2", "codex-gpt-image-2"}
IMAGE_MODEL_PLAN_TYPES = ("plus", "team", "pro")
CODEX_IMAGE_MODEL = "codex-gpt-image-2"
PREFIXED_CODEX_IMAGE_MODELS = {
    f"{plan_type}-{CODEX_IMAGE_MODEL}"
    for plan_type in IMAGE_MODEL_PLAN_TYPES
}
IMAGE_MODELS = BASE_IMAGE_MODELS | PREFIXED_CODEX_IMAGE_MODELS
PUBLIC_IMAGE_MODELS = BASE_IMAGE_MODELS | PREFIXED_CODEX_IMAGE_MODELS
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SUPPORTED_JSON_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
MAX_JSON_IMAGE_BYTES = 10 * 1024 * 1024
MAX_JSON_EDIT_IMAGES = 10
DATA_URL_IMAGE_RE = re.compile(r"^data:(?P<mime>[-+./\w]+);base64,(?P<data>.*)$", re.DOTALL)
REMOTE_IMAGE_TIMEOUT_SECONDS = 20


def _image_extension(mime_type: str) -> str:
    image_type = mime_type.split("/", 1)[1].split(";", 1)[0].lower() if "/" in mime_type else "png"
    return "jpg" if image_type == "jpeg" else image_type or "png"


def _decode_json_image_string(value: str, index: int, filename: str | None = None, mime_type: str | None = None) -> tuple[bytes, str, str]:
    text = value.strip()
    if not text:
        raise HTTPException(status_code=400, detail={"error": "image file is empty"})
    match = DATA_URL_IMAGE_RE.match(text)
    if match:
        resolved_mime = (match.group("mime") or "image/png").lower()
        encoded = match.group("data")
    else:
        if text.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail={"error": "remote image URLs are not supported"})
        resolved_mime = (mime_type or "image/png").lower()
        encoded = text
    if resolved_mime == "image/jpg":
        resolved_mime = "image/jpeg"
    if resolved_mime not in SUPPORTED_JSON_IMAGE_MIME_TYPES:
        raise HTTPException(status_code=400, detail={"error": "unsupported image mime type"})
    try:
        image_data = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"error": "invalid base64 image data"}) from exc
    if not image_data:
        raise HTTPException(status_code=400, detail={"error": "image file is empty"})
    if len(image_data) > MAX_JSON_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail={"error": "image file is too large"})
    return image_data, filename or f"image_{index}.{_image_extension(resolved_mime)}", resolved_mime


def _extract_json_image_value(item: object) -> tuple[str, str | None, str | None]:
    if isinstance(item, str):
        return item, None, None
    if not isinstance(item, dict):
        raise HTTPException(status_code=400, detail={"error": "image entry must be a base64 string or object"})
    filename = str(item.get("filename") or item.get("file_name") or "").strip() or None
    mime_type = str(item.get("mime_type") or item.get("mimeType") or "").strip() or None
    value = item.get("b64_json") or item.get("base64")
    if not value:
        image_url = item.get("image_url") or item.get("url")
        if isinstance(image_url, dict):
            filename = filename or str(image_url.get("filename") or image_url.get("file_name") or "").strip() or None
            mime_type = mime_type or str(image_url.get("mime_type") or image_url.get("mimeType") or "").strip() or None
            value = image_url.get("url") or image_url.get("image_url")
        else:
            value = image_url
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=400, detail={"error": "image entry must include image data"})
    return value, filename, mime_type


def normalize_json_edit_images(image: object = None, images: object = None) -> list[tuple[bytes, str, str]]:
    raw_images = images if images is not None else image
    if raw_images is None:
        raise HTTPException(status_code=400, detail={"error": "image file is required"})
    entries = raw_images if isinstance(raw_images, list) else [raw_images]
    if not entries:
        raise HTTPException(status_code=400, detail={"error": "image file is required"})
    if len(entries) > MAX_JSON_EDIT_IMAGES:
        raise HTTPException(status_code=400, detail={"error": f"images supports up to {MAX_JSON_EDIT_IMAGES} items"})
    normalized = []
    for index, item in enumerate(entries, start=1):
        value, filename, mime_type = _extract_json_image_value(item)
        normalized.append(_decode_json_image_string(value, index, filename, mime_type))
    return normalized


def new_uuid() -> str:
    return str(uuid.uuid4())


def split_image_model(model: object) -> tuple[str | None, str | None]:
    normalized = str(model or "").strip().lower()
    if not normalized:
        return None, None
    if normalized in BASE_IMAGE_MODELS:
        return None, normalized
    for plan_type in IMAGE_MODEL_PLAN_TYPES:
        prefix = f"{plan_type}-"
        if normalized.startswith(prefix):
            base_model = normalized[len(prefix):]
            if base_model == CODEX_IMAGE_MODEL:
                return plan_type, base_model
    return None, None


def is_supported_image_model(model: object) -> bool:
    _, base_model = split_image_model(model)
    return base_model is not None


def is_codex_image_model(model: object) -> bool:
    _, base_model = split_image_model(model)
    return base_model == CODEX_IMAGE_MODEL


def is_image_chat_request(body: dict[str, object]) -> bool:
    model = str(body.get("model") or "").strip()
    modalities = body.get("modalities")
    if is_supported_image_model(model):
        return True
    return isinstance(modalities, list) and "image" in {str(item or "").strip().lower() for item in modalities}


_UPSTREAM_STREAM_ERROR_BODY_LIMIT = 4096
_UPSTREAM_STREAM_ERROR_READ_SECONDS = 1.0


class UpstreamHTTPError(RuntimeError):
    """Raised when an upstream HTTP call returns a non-2xx status.

    Carries structured fields (status_code, body, retry_after) so callers can
    branch on status code instead of string-matching on str(exc). The full
    body is preserved on the instance for explicit, safe diagnostic
    extraction; it is deliberately excluded from the exception text because
    generic callers may log that text.
    """

    def __init__(
        self,
        context: str,
        status_code: int,
        body: Any,
        retry_after: int | None = None,
        response_ids: dict[str, str] | None = None,
    ) -> None:
        self.context = context
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after
        self.response_ids = safe_upstream_response_ids(response_ids)
        safe_context = str(context or "upstream").split("?", 1)[0].split("#", 1)[0] or "upstream"
        super().__init__(f"{safe_context} failed: status={status_code}")


def _bounded_stream_error_text(response: requests.Response) -> str:
    """Consume only a small, time-bounded non-2xx streaming body.

    curl_cffi leaves ``response.text`` empty for a streamed error until the
    caller drains ``iter_content``.  This is deliberately used only after an
    HTTP failure: successful SSE responses remain entirely stream-owned by
    their normal consumer.
    """
    queue = getattr(response, "queue", None)
    # curl_cffi exposes ``queue`` only for stream=True responses.  Do not turn
    # a normal empty non-stream response into another transport read.
    if queue is None or not callable(getattr(queue, "get", None)):
        return ""
    chunks: list[bytes] = []
    remaining = _UPSTREAM_STREAM_ERROR_BODY_LIMIT
    deadline = time.monotonic() + _UPSTREAM_STREAM_ERROR_READ_SECONDS
    terminal = False
    try:
        while remaining > 0:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                chunk = queue.get(timeout=timeout)
            except Exception:
                break
            if not isinstance(chunk, (bytes, bytearray)):
                terminal = True
                break
            piece = bytes(chunk[:remaining])
            chunks.append(piece)
            remaining -= len(piece)
    finally:
        if terminal:
            # The producer has explicitly ended, so curl_cffi's close does not
            # wait on an active transfer.
            close = getattr(response, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        else:
            # Never call Response.close while curl_cffi is still blocked in a
            # stream task: its implementation waits without a timeout.  Signal
            # its existing cancellation event and let the request's configured
            # transport timeout finish cleanup.  Its own completion callback
            # then closes the curl handle on the completed task, never in
            # parallel with an active curl transfer.
            quit_now = getattr(response, "quit_now", None)
            if callable(getattr(quit_now, "set", None)):
                quit_now.set()
            stream_task = getattr(response, "stream_task", None)
            add_done_callback = getattr(stream_task, "add_done_callback", None)
            close = getattr(response, "close", None)
            if callable(add_done_callback) and callable(close):
                def close_after_stream(_future):
                    try:
                        close()
                    except Exception:
                        pass
                add_done_callback(close_after_stream)
    return b"".join(chunks).decode("utf-8", errors="replace")


def ensure_ok(response: requests.Response, context: str) -> None:
    if 200 <= response.status_code < 300:
        return
    text = response.text
    streamed = not text and getattr(response, "queue", None) is not None
    if streamed:
        text = _bounded_stream_error_text(response)
    body: Any = text
    try:
        # curl_cffi's response.json() observes an empty pre-drain stream;
        # parse the bounded bytes ourselves in that case.
        body = json.loads(text) if streamed else response.json()
    except Exception:
        pass
    retry_after_header = response.headers.get("Retry-After") if hasattr(response, "headers") else None
    retry_after: int | None = None
    if retry_after_header is not None:
        ra_str = str(retry_after_header).strip()
        if ra_str.isdigit():
            retry_after = int(ra_str)
    response_ids = upstream_response_ids_from_headers(getattr(response, "headers", None))
    raise UpstreamHTTPError(context, response.status_code, body, retry_after=retry_after, response_ids=response_ids)


def _stream_exception_detail(exc: Exception) -> dict[str, str]:
    detail = {}
    formatter = getattr(exc, "to_openai_error", None)
    if callable(formatter):
        try:
            event = formatter()
            if isinstance(event, dict):
                detail = sse_error_detail(event)
        except Exception:
            pass
    if not detail:
        detail = safe_upstream_error({"message": str(exc), "type": exc.__class__.__name__})
    detail.setdefault("type", "upstream_error")
    detail.setdefault("message", "Stream failed")
    return detail


def sse_json_stream(items) -> Iterator[str]:
    yield ": stream-open\n\n"
    try:
        for item in items:
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
    except Exception as exc:
        logger.warning({
            "event": "sse_stream_error",
            "error_type": exc.__class__.__name__,
        })
        error = {"error": _stream_exception_detail(exc)}
        yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def anthropic_sse_stream(items) -> Iterator[str]:
    try:
        for item in items:
            event = str(item.get("type") or "message_delta") if isinstance(item, dict) else "message_delta"
            yield f"event: {event}\n"
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
    except Exception as exc:
        logger.warning({
            "event": "anthropic_sse_stream_error",
            "error_type": exc.__class__.__name__,
        })
        error = {"type": "error", "error": _stream_exception_detail(exc)}
        yield "event: error\n"
        yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"


def _sse_error_category(event: dict) -> str:
    # Categories remain fixed; a separate bounded, redacted diagnostic retains
    # the upstream code and reason without retaining the response body.
    error = event.get("error")
    error = error if isinstance(error, dict) else event
    codes = {value for key in ("code", "type") if isinstance(value := error.get(key), str)}
    for category, allowed in (
        ("quota", {"insufficient_quota", "quota_exceeded", "usage_limit_reached"}),
        ("rate_limit", {"rate_limit_exceeded", "rate_limit_error", "too_many_requests"}),
        ("auth", {"invalid_api_key", "authentication_error", "unauthorized", "invalid_token"}),
        ("upstream", {"server_error", "internal_error", "internal_server_error", "overloaded_error"}),
    ):
        if codes & allowed:
            return category
    return "unknown"


_SAFE_SSE_EVENT_TYPES = ("resume_conversation_token", "server_ste_metadata", "stream_handoff")
_SAFE_RESUME_TOKEN_KINDS = frozenset({"topic"})
_SAFE_STREAM_HANDOFF_OPTION_TYPES = ("resume_sse_endpoint", "subscribe_ws_topic")


def _observe_sse_event_metadata(event: dict, observation: dict) -> None:
    """Keep only fixed, content-free markers needed to diagnose stream closure."""
    event_type = event.get("type")
    if not isinstance(event_type, str) or event_type not in _SAFE_SSE_EVENT_TYPES:
        return
    counts = observation.setdefault("sse_event_type_counts", {
        name: 0 for name in _SAFE_SSE_EVENT_TYPES
    })
    counts[event_type] = min(counts.get(event_type, 0) + 1, 2147483647)
    if event_type == "server_ste_metadata":
        observation["server_ste_metadata_seen"] = True
        return
    if event_type == "stream_handoff":
        observation["stream_handoff_seen"] = True
        option_counts = observation.setdefault("stream_handoff_option_counts", {
            name: 0 for name in _SAFE_STREAM_HANDOFF_OPTION_TYPES
        })
        options = event.get("options")
        if isinstance(options, list):
            for option in options:
                option_type = option.get("type") if isinstance(option, dict) else None
                if option_type not in _SAFE_STREAM_HANDOFF_OPTION_TYPES:
                    continue
                option_counts[option_type] = min(option_counts.get(option_type, 0) + 1, 2147483647)
                observation[f"{option_type}_seen"] = True
        return

    observation["resume_token_seen"] = True
    raw_kind = event.get("kind")
    kind = str(raw_kind).strip().lower() if isinstance(raw_kind, str) else ""
    safe_kind = kind if kind in _SAFE_RESUME_TOKEN_KINDS else ("other" if kind else "unknown")
    previous = observation.get("resume_token_kind")
    if previous is None:
        observation["resume_token_kind"] = safe_kind
    elif previous != safe_kind:
        observation["resume_token_kind"] = "multiple"


def _observe_sse_message(event: dict, observation: dict, request_message_id: str) -> None:
    """Count complete message snapshots, never infer completion from patches.

    These are event counts, not unique messages or authority to skip the
    original-branch GET. No message IDs, text, URLs or field values are saved.
    """
    def count(key):
        observation[key] = min(observation.get(key, 0) + 1, 2147483647)

    value = event.get("v")
    frame = value if isinstance(value, dict) and isinstance(value.get("message"), dict) else event
    message = frame.get("message")
    if not isinstance(message, dict):
        return
    count("sse_message_snapshot_events")
    author = message.get("author")
    if not isinstance(author, dict) or author.get("role") != "assistant":
        return
    if message.get("status") != "finished_successfully" or message.get("end_turn") is not True:
        return
    count("sse_terminal_assistant_events")
    has_id = isinstance(message.get("id"), str) and bool(message["id"])
    if has_id:
        count("sse_terminal_id_events")
    metadata = message.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if message.get("channel") == "final" or metadata.get("channel") == "final":
        count("sse_terminal_final_channel_events")
    parents = {parent for container in (frame, message, metadata)
               for key in ("parent", "parent_id")
               if isinstance(parent := container.get(key), str) and parent}
    if parents:
        count("sse_terminal_parent_events")
    if has_id and request_message_id and parents == {request_message_id}:
        count("sse_terminal_direct_parent_match_events")


def iter_sse_payloads(response: requests.Response, *, observation: dict | None = None,
                      request_message_id: str = "") -> Iterator[str]:
    def observe_error(event: dict, *, named: bool = False):
        observation["sse_error_event"] = True
        detail = sse_error_detail(event, named_error=named)
        if observation.get("sse_error_category", "unknown") == "unknown":
            observation["sse_error_category"] = _sse_error_category(detail)
        if detail:
            # Keep the first reason, but allow a later diagnostic to fill fields
            # absent from a code-only error event.
            retained = observation.setdefault("upstream_error", {})
            for key, value in detail.items():
                retained.setdefault(key, value)

    if observation is not None:
        for key in ("sse_message_snapshot_events", "sse_terminal_assistant_events", "sse_terminal_id_events",
                    "sse_terminal_final_channel_events", "sse_terminal_parent_events",
                    "sse_terminal_direct_parent_match_events"):
            observation.setdefault(key, 0)
        observation.setdefault("resume_token_seen", False)
        observation.setdefault("server_ste_metadata_seen", False)
        observation.setdefault("stream_handoff_seen", False)
        for option_type in _SAFE_STREAM_HANDOFF_OPTION_TYPES:
            observation.setdefault(f"{option_type}_seen", False)
        observation.setdefault("sse_event_type_counts", {name: 0 for name in _SAFE_SSE_EVENT_TYPES})
        observation.setdefault("stream_handoff_option_counts", {
            name: 0 for name in _SAFE_STREAM_HANDOFF_OPTION_TYPES
        })
    error_event = False
    for raw_line in response.iter_lines():
        if not raw_line:
            error_event = False
            continue
        line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else str(raw_line)
        if line.startswith("event:"):
            error_event = line[6:].strip() == "error"
        if observation is not None and error_event:
            observation["sse_error_event"] = True
            observation.setdefault("sse_error_category", "unknown")
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload:
            if observation is not None:
                observation["sse_data_count"] = min(observation.get("sse_data_count", 0) + 1, 2147483647)
                if payload == "[DONE]":
                    observation["stream_end"] = "done"
                else:
                    try:
                        event = json.loads(payload)
                    except (ValueError, TypeError):
                        observation["sse_parse_errors"] = min(observation.get("sse_parse_errors", 0) + 1, 2147483647)
                        if error_event:
                            observe_error({"message": payload}, named=True)
                    else:
                        if isinstance(event, dict):
                            _observe_sse_event_metadata(event, observation)
                            _observe_sse_message(event, observation, request_message_id)
                        if isinstance(event, dict) and (error_event or event.get("type") in ("error", "response.failed") or event.get("error")):
                            observe_error(event, named=error_event)
                        elif error_event and isinstance(event, str):
                            observe_error({"message": event}, named=True)
            yield payload


def save_images_from_text(text: str, prefix: str) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matches = re.findall(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", text or "")
    saved_paths: list[Path] = []
    timestamp = int(time.time() * 1000)
    for index, data_url in enumerate(matches, start=1):
        header, encoded = data_url.split(",", 1)
        image_type = header.split(";")[0].removeprefix("data:image/").strip() or "png"
        extension = "jpg" if image_type == "jpeg" else image_type
        output_path = OUTPUT_DIR / f"{prefix}_{timestamp}_{index}.{extension}"
        output_path.write_bytes(base64.b64decode(encoded))
        saved_paths.append(output_path)
    return saved_paths


def anonymize_token(token: object) -> str:
    value = str(token or "").strip()
    if not value:
        return "token:empty"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"token:{digest}"


def extract_response_prompt(input_value: object) -> str:
    if isinstance(input_value, str):
        return input_value.strip()
    if isinstance(input_value, dict):
        role = str(input_value.get("role") or "").strip().lower()
        if role and role != "user":
            return ""
        return extract_prompt_from_message_content(input_value.get("content"))
    if not isinstance(input_value, list):
        return ""
    prompt_parts: list[str] = []
    for item in input_value:
        if isinstance(item, dict) and str(item.get("type") or "").strip() == "input_text":
            text = str(item.get("text") or "").strip()
            if text:
                prompt_parts.append(text)
            continue
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role and role != "user":
            continue
        prompt = extract_prompt_from_message_content(item.get("content"))
        if prompt:
            prompt_parts.append(prompt)
    return "\n".join(prompt_parts).strip()


def has_response_image_generation_tool(body: dict[str, object]) -> bool:
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict) and str(tool.get("type") or "").strip() == "image_generation":
                return True
    tool_choice = body.get("tool_choice")
    return isinstance(tool_choice, dict) and str(tool_choice.get("type") or "").strip() == "image_generation"


def extract_prompt_from_message_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        if item_type == "text":
            text = str(item.get("text") or "").strip()
            if text:
                parts.append(text)
        elif item_type == "input_text":
            text = str(item.get("text") or item.get("input_text") or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _message_image_url(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("url") or value.get("image_url") or "").strip()
    return str(value or "").strip()


def _decode_message_image_url(value: object) -> tuple[bytes, str] | None:
    source = _message_image_url(value)
    if source.startswith("data:"):
        header, _, data = source.partition(",")
        mime = header.split(";")[0].removeprefix("data:") or "image/png"
        return base64.b64decode(data), mime
    if not source.startswith(("http://", "https://")):
        return None
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    try:
        response = requests.get(
            source,
            headers={"Accept": "image/*,*/*;q=0.8", "User-Agent": "chatgpt2api vision fetcher"},
            timeout=REMOTE_IMAGE_TIMEOUT_SECONDS,
            allow_redirects=True,
            **proxy_settings.build_session_kwargs(),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"error": f"image_url fetch failed: {exc}"}) from exc
    if not 200 <= response.status_code < 300:
        raise HTTPException(status_code=400, detail={"error": f"image_url fetch failed: HTTP {response.status_code}"})
    content_length = str(response.headers.get("content-length") or "").strip()
    if content_length.isdigit() and int(content_length) > MAX_JSON_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail={"error": "image_url exceeds 10MB limit"})
    image_data = response.content
    if not image_data:
        raise HTTPException(status_code=400, detail={"error": "image_url returned empty content"})
    if len(image_data) > MAX_JSON_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail={"error": "image_url exceeds 10MB limit"})
    mime = str(response.headers.get("content-type") or "image/png").split(";", 1)[0].lower()
    guessed_mime = mimetypes.guess_type(parsed.path)[0] or ""
    if mime and not mime.startswith("image/") and mime not in {"application/octet-stream", "binary/octet-stream"}:
        raise HTTPException(status_code=400, detail={"error": "image_url must point to an image"})
    if not mime.startswith("image/") and guessed_mime.startswith("image/"):
        mime = guessed_mime
    if not mime.startswith("image/"):
        mime = "image/png"
    return image_data, mime


def _decode_message_image_object(item: dict[str, object]) -> tuple[bytes, str] | None:
    data = item.get("data")
    if isinstance(data, (bytes, bytearray)):
        return bytes(data), str(item.get("mime") or item.get("mime_type") or "image/png")
    for key in ("image_url", "url"):
        image = _decode_message_image_url(item.get(key))
        if image:
            return image
    value = item.get("b64_json") or item.get("base64")
    if isinstance(value, str) and value.strip():
        image_data, _, mime = _decode_json_image_string(
            value,
            1,
            mime_type=str(item.get("mime") or item.get("mime_type") or item.get("mimeType") or "image/png"),
        )
        return image_data, mime
    source = item.get("source")
    if isinstance(source, dict) and str(source.get("type") or "") == "base64":
        encoded = str(source.get("data") or "")
        mime = str(source.get("media_type") or source.get("mime_type") or "image/png")
        image_data, _, resolved_mime = _decode_json_image_string(encoded, 1, mime_type=mime)
        return image_data, resolved_mime
    return None


def extract_image_from_message_content(content: object) -> list[tuple[bytes, str]]:
    if not isinstance(content, list):
        return []
    images = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        if item_type == "image_url":
            image = _decode_message_image_url(item.get("image_url") or item.get("url") or item)
            if image:
                images.append(image)
        elif item_type in {"input_image", "image"}:
            image = _decode_message_image_object(item)
            if image:
                images.append(image)
    return images


def extract_chat_image(body: dict[str, object]) -> list[tuple[bytes, str]]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        images = extract_image_from_message_content(message.get("content"))
        if images:
            return images
    return []


def extract_chat_prompt(body: dict[str, object]) -> str:
    direct_prompt = str(body.get("prompt") or "").strip()
    if direct_prompt:
        return direct_prompt
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    prompt_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        prompt = extract_prompt_from_message_content(message.get("content"))
        if prompt:
            prompt_parts.append(prompt)
    return "\n".join(prompt_parts).strip()


def parse_image_count(raw_value: object) -> int:
    try:
        value = int(raw_value or 1)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"error": "n must be an integer"}) from exc
    if value < 1 or value > 4:
        raise HTTPException(status_code=400, detail={"error": "n must be between 1 and 4"})
    return value


def build_chat_image_markdown_content(image_result: dict[str, object]) -> str:
    image_items = image_result.get("data") if isinstance(image_result.get("data"), list) else []
    markdown_images: list[str] = []
    for index, item in enumerate(image_items, start=1):
        if not isinstance(item, dict):
            continue
        b64_json = str(item.get("b64_json") or "").strip()
        if b64_json:
            markdown_images.append(f"![image_{index}](data:image/png;base64,{b64_json})")
    return "\n\n".join(markdown_images) if markdown_images else "Image generation completed."
