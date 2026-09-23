from __future__ import annotations

import base64
import binascii
import io
import ipaddress
import json
import mimetypes
import re
import socket
import warnings
from pathlib import PurePosixPath
from typing import Any, TypeGuard
from urllib.parse import unquote, unquote_to_bytes, urljoin, urlparse

from curl_cffi import CurlOpt, requests
from curl_cffi.curl import ffi, lib
from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from starlette.datastructures import UploadFile

from services.proxy_service import proxy_settings

ImageInput = tuple[bytes, str, str]
ImageSource = str | UploadFile | ImageInput

MAX_IMAGE_REFERENCE_BYTES = 50 * 1024 * 1024
MAX_IMAGE_INPUT_COUNT = 16
MAX_IMAGE_INPUT_BYTES = MAX_IMAGE_REFERENCE_BYTES * 2
MAX_IMAGE_REDIRECTS = 5
IMAGE_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MAX_BASE64_INPUT_CHARS = 4 * ((MAX_IMAGE_REFERENCE_BYTES + 2) // 3)
MAX_CHAT_TEXT_BYTES = 1024 * 1024
IMAGE_REFERENCE_FIELDS = {"image", "image[]", "images", "images[]", "image_url", "image_url[]"}
MASK_REFERENCE_FIELDS = {"mask", "mask[]"}


def _input_error(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail={"error": message})


def _normalize_mime_type(mime_type: object) -> str:
    value = _clean(mime_type).split(";", 1)[0].strip().lower()
    return "image/jpeg" if value == "image/jpg" else value


def _validated_image_input(data: bytes, filename: str, mime_type: str) -> ImageInput:
    """Validate bytes as a bounded raster and return its canonical MIME type."""
    if not data:
        raise _input_error("image file is empty")
    if len(data) > MAX_IMAGE_REFERENCE_BYTES:
        raise _input_error("image URL exceeds 50MB limit")

    declared_mime = _normalize_mime_type(mime_type)
    if declared_mime in {"application/octet-stream", "binary/octet-stream"}:
        declared_mime = ""
    if declared_mime and not declared_mime.startswith("image/"):
        raise _input_error("image MIME type is not a raster image")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                if width < 1 or height < 1 or width * height > 50_000_000:
                    raise _input_error("image dimensions exceed the safe limit")
                image.load()
                image_format = str(image.format or "").upper()
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, OSError, SyntaxError,
            UnidentifiedImageError, ValueError) as exc:
        raise _input_error("image bytes are not a valid raster image") from exc

    actual_mime = _normalize_mime_type(Image.MIME.get(image_format, ""))
    if not actual_mime or not actual_mime.startswith("image/"):
        raise _input_error("image format is not a supported raster image")
    if declared_mime and declared_mime != actual_mime:
        raise _input_error("image MIME type does not match the image bytes")
    return data, _safe_filename(filename, actual_mime, "image"), actual_mime


def _clean(value: object, default: str = "") -> str:
    """清理字符串：转换为字符串并去掉首尾空白。"""
    text = str(value if value is not None else default).strip()
    return text or default


def _is_upload(value: object) -> TypeGuard[UploadFile]:
    """识别上传文件：兼容 Starlette 表单返回的 UploadFile。"""
    return isinstance(value, UploadFile)


def _parse_bool(value: object) -> bool | None:
    """解析布尔字段：兼容 JSON 布尔值和表单字符串。"""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = _clean(value).lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise HTTPException(status_code=400, detail={"error": "stream must be a boolean"})


def _parse_count(value: object) -> int:
    """解析生成数量：保持图片接口的 1 到 4 限制。"""
    try:
        count = int(value or 1)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"error": "n must be an integer"}) from exc
    if count < 1 or count > 4:
        raise HTTPException(status_code=400, detail={"error": "n must be between 1 and 4"})
    return count


def _payload_from_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """构造图片编辑载荷：从表单或 JSON 字段提取通用参数。"""
    prompt = _clean(fields.get("prompt"))
    if not prompt:
        raise HTTPException(status_code=400, detail={"error": "prompt is required"})
    payload = {
        "prompt": prompt,
        "model": _clean(fields.get("model"), "gpt-image-2"),
        "n": _parse_count(fields.get("n")),
        "size": _clean(fields.get("size")) or None,
        "quality": _clean(fields.get("quality"), "auto"),
        "response_format": _clean(fields.get("response_format"), "b64_json"),
        "stream": _parse_bool(fields.get("stream")),
    }
    if "client_task_id" in fields:
        payload["client_task_id"] = _clean(fields.get("client_task_id"))
    for field in (
        "provider_binding_id",
        "provider_account_identity",
        "client_conversation_id",
        "conversation_id",
        "parent_message_id",
        "upstream_model",
    ):
        if field in fields:
            payload[field] = _clean(fields.get(field))
    for field in ("image_thread_id", "edit_source_task_id"):
        if field in fields:
            if not isinstance(fields[field], str):
                raise HTTPException(400, detail={"code": "IMAGE_THREAD_INPUT_INVALID"})
            payload[field] = fields[field]
    if "edit_source_index" in fields:
        value = fields["edit_source_index"]
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if type(value) is not int or not 0 <= value < 16:
            raise HTTPException(400, detail={"code": "IMAGE_THREAD_INPUT_INVALID"})
        payload["edit_source_index"] = value
    if "retain_conversation" in fields:
        payload["retain_conversation"] = bool(_parse_bool(fields.get("retain_conversation")))
    return payload


def _json_reference_value(value: object) -> object:
    """解析表单图片引用：支持把 images 字段写成 JSON 字符串。"""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _decode_base64_image(value: object, filename: str, mime_type: str) -> ImageInput:
    encoded = str(value).strip()
    if len(encoded) > MAX_BASE64_INPUT_CHARS:
        raise _input_error("image URL exceeds 50MB limit")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"error": "invalid base64 image data"}) from exc
    return _validated_image_input(data, filename, mime_type)


def _source_from_object(value: dict[str, Any]) -> list[ImageSource]:
    """提取图片引用对象：支持 image_url 或 url，明确拒绝 file_id。"""
    has_url = "image_url" in value or "url" in value
    if value.get("file_id"):
        raise HTTPException(
            status_code=400,
            detail={"error": "file_id image references are not supported; use image_url instead"},
        )
    inline = value.get("b64_json") or value.get("base64")
    if inline:
        filename = _clean(value.get("filename") or value.get("file_name"), "image")
        mime_type = _clean(value.get("mime_type") or value.get("mimeType"))
        return [_decode_base64_image(inline, filename, mime_type)]
    if not has_url:
        raise HTTPException(status_code=400, detail={"error": "image reference must include image_url"})
    image_url = value.get("image_url", value.get("url"))
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    return _sources_from_value(image_url)


def _sources_from_value(value: object) -> list[ImageSource]:
    """展开图片引用：把字符串、数组和对象统一成图片来源列表。"""
    value = _json_reference_value(value)
    if _is_upload(value):
        return [value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.lower().startswith(("data:", "http://", "https://")):
            return [text]
        return [_decode_base64_image(text, "image", "")]
    if isinstance(value, list):
        if len(value) > MAX_IMAGE_INPUT_COUNT:
            raise _input_error(f"at most {MAX_IMAGE_INPUT_COUNT} images are allowed")
        sources: list[ImageSource] = []
        for item in value:
            sources.extend(_sources_from_value(item))
            if len(sources) > MAX_IMAGE_INPUT_COUNT:
                raise _input_error(f"at most {MAX_IMAGE_INPUT_COUNT} images are allowed")
        return sources
    if isinstance(value, dict):
        return _source_from_object(value)
    if value is None:
        return []
    raise HTTPException(status_code=400, detail={"error": "invalid image reference"})


def _json_image_sources(body: dict[str, Any]) -> list[ImageSource]:
    """读取 JSON 图片引用：优先支持官方 images 数组字段。"""
    sources: list[ImageSource] = []
    for key in ("images", "image", "image_url"):
        if key in body:
            sources.extend(_sources_from_value(body.get(key)))
            if len(sources) > MAX_IMAGE_INPUT_COUNT:
                raise _input_error(f"at most {MAX_IMAGE_INPUT_COUNT} images are allowed")
    return sources


def _json_mask_sources(body: dict[str, Any]) -> list[ImageSource]:
    """读取 JSON mask 引用。"""
    mask = body.get("mask")
    if mask is not None:
        return _sources_from_value(mask)
    return []


async def parse_image_edit_request(request: Request) -> tuple[dict[str, Any], list[ImageSource], list[ImageSource]]:
    """解析图片编辑请求：同时支持 multipart 上传和官方 JSON 图片 URL。
    
    返回 (payload, image_sources, mask_sources)
    """
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type == "application/json":
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail={"error": "invalid JSON body"}) from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail={"error": "JSON body must be an object"})
        return _payload_from_fields(body), _json_image_sources(body), _json_mask_sources(body)

    form = await request.form()
    fields: dict[str, Any] = {}
    for key in ("client_task_id", "prompt", "model", "n", "size", "quality", "response_format", "stream",
                "image_thread_id", "edit_source_task_id", "edit_source_index", "provider_binding_id",
                "provider_account_identity", "client_conversation_id", "conversation_id", "parent_message_id",
                "retain_conversation", "upstream_model"):
        value = form.get(key)
        if isinstance(value, str):
            fields[key] = value
    sources: list[ImageSource] = []
    mask_sources: list[ImageSource] = []
    for key, value in form.multi_items():
        if key in IMAGE_REFERENCE_FIELDS:
            sources.extend(_sources_from_value(value))
            if len(sources) > MAX_IMAGE_INPUT_COUNT:
                raise _input_error(f"at most {MAX_IMAGE_INPUT_COUNT} images are allowed")
        elif key in MASK_REFERENCE_FIELDS:
            mask_sources.extend(_sources_from_value(value))
            if len(mask_sources) > MAX_IMAGE_INPUT_COUNT:
                raise _input_error(f"at most {MAX_IMAGE_INPUT_COUNT} images are allowed")
    return _payload_from_fields(fields), sources, mask_sources


def _extension_from_mime(mime_type: str) -> str:
    """推导图片扩展名：把 MIME 类型转换为常见文件后缀。"""
    subtype = mime_type.split("/", 1)[1].split("+", 1)[0] if "/" in mime_type else "png"
    if subtype == "jpeg":
        return "jpg"
    return re.sub(r"[^a-z0-9]+", "", subtype.lower()) or "png"


def _safe_filename(name: str, mime_type: str, fallback: str) -> str:
    """生成安全文件名：清理 URL 文件名并补齐扩展名。"""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not cleaned:
        cleaned = fallback
    if "." not in cleaned:
        cleaned = f"{cleaned}.{_extension_from_mime(mime_type)}"
    return cleaned


def _decode_data_url(url: str) -> ImageInput:
    """解码 data URL：把内联图片转成标准图片输入元组。"""
    header, separator, payload = url.partition(",")
    if not separator:
        raise HTTPException(status_code=400, detail={"error": "invalid data image URL"})
    mime_header = header.split(";", 1)[0]
    mime_type = mime_header[5:] if mime_header.lower().startswith("data:") else mime_header
    if mime_type and not mime_type.startswith("image/"):
        raise HTTPException(status_code=400, detail={"error": "image_url must point to an image"})
    if len(payload) > MAX_BASE64_INPUT_CHARS * 3:
        raise _input_error("image URL exceeds 50MB limit")
    try:
        data = base64.b64decode(payload, validate=True) if ";base64" in header else unquote_to_bytes(payload)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"error": "invalid data image URL"}) from exc
    if not data:
        raise HTTPException(status_code=400, detail={"error": "image URL is empty"})
    if len(data) > MAX_IMAGE_REFERENCE_BYTES:
        raise HTTPException(status_code=400, detail={"error": "image URL exceeds 50MB limit"})
    filename = f"image_url.{_extension_from_mime(mime_type)}" if mime_type else "image_url"
    return _validated_image_input(data, filename, mime_type)


def normalize_inline_chat_messages(messages: object) -> list[dict[str, Any]]:
    """Validate public Chat messages and materialize only inline image bytes.

    The ordinary Chat boundary intentionally does not fetch remote URLs. Every
    image passes through the same bounded raster validation used by image edit
    requests before the durable text request can be accepted.
    """
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, detail={"code": "CHAT_MESSAGES_INVALID", "error": "messages must be a non-empty array"})
    if len(messages) > 100:
        raise HTTPException(400, detail={"code": "CHAT_MESSAGES_INVALID", "error": "at most 100 messages are allowed"})

    normalized: list[dict[str, Any]] = []
    image_count = 0
    image_bytes = 0
    text_bytes = 0
    has_user = False

    def invalid(message: str, code: str = "CHAT_MESSAGES_INVALID") -> HTTPException:
        return HTTPException(400, detail={"code": code, "error": message})

    def image_part(part: dict[str, Any]) -> dict[str, Any]:
        nonlocal image_count, image_bytes
        kind = str(part.get("type") or "").strip()
        if kind in {"image_url", "input_image"}:
            allowed = {"type", "image_url"}
            if set(part) - allowed:
                raise invalid("image parts do not accept options", "CHAT_OPTION_UNSUPPORTED")
            source = part.get("image_url")
            if isinstance(source, dict):
                if set(source) != {"url"}:
                    raise invalid("image_url accepts only an inline url", "CHAT_OPTION_UNSUPPORTED")
                source = source.get("url")
            if not isinstance(source, str) or not source.strip():
                raise invalid("image_url must be an inline data URL")
            source = source.strip()
            if source.lower().startswith(("http://", "https://")):
                raise invalid("remote image URLs are not supported", "REMOTE_IMAGE_URL_NOT_SUPPORTED")
            if not source.lower().startswith("data:image/"):
                raise invalid("image_url must be an image data URL")
            header, separator, encoded = source.partition(",")
            if not separator:
                raise invalid("invalid data image URL")
            if ";base64" in header.lower() and len(encoded) > MAX_BASE64_INPUT_CHARS:
                raise invalid("image data exceeds 50MB limit")
            if ";base64" not in header.lower() and len(encoded) > MAX_IMAGE_REFERENCE_BYTES * 3:
                raise invalid("image data exceeds 50MB limit")
            data, _, mime = _decode_data_url(source)
        else:
            allowed = {"type", "data", "mime", "mime_type"}
            if set(part) - allowed:
                raise invalid("image parts do not accept options", "CHAT_OPTION_UNSUPPORTED")
            source = part.get("data")
            declared_mime = str(part.get("mime") or part.get("mime_type") or "image/png")
            if isinstance(source, str):
                if source.lower().startswith(("http://", "https://")):
                    raise invalid("remote image URLs are not supported", "REMOTE_IMAGE_URL_NOT_SUPPORTED")
                if not source.lower().startswith("data:image/"):
                    raise invalid("string image data must be an image data URL")
                header, separator, encoded = source.partition(",")
                if not separator:
                    raise invalid("invalid data image URL")
                if ";base64" in header.lower() and len(encoded) > MAX_BASE64_INPUT_CHARS:
                    raise invalid("image data exceeds 50MB limit")
                if ";base64" not in header.lower() and len(encoded) > MAX_IMAGE_REFERENCE_BYTES * 3:
                    raise invalid("image data exceeds 50MB limit")
                data, _, mime = _decode_data_url(source)
            elif isinstance(source, (bytes, bytearray)):
                data, _, mime = _validated_image_input(bytes(source), "chat-image", declared_mime)
            else:
                raise invalid("image data must be bytes or an image data URL")
        if mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise invalid("image format must be PNG, JPEG, or WebP", "CHAT_IMAGE_FORMAT_UNSUPPORTED")
        image_count += 1
        image_bytes += len(data)
        if image_count > MAX_IMAGE_INPUT_COUNT:
            raise invalid(f"at most {MAX_IMAGE_INPUT_COUNT} images are allowed")
        if image_bytes > MAX_IMAGE_INPUT_BYTES:
            raise invalid("combined image inputs exceed 100MB limit")
        return {"type": "image", "data": data, "mime": mime}

    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise invalid("each message must contain only role and content")
        role = str(message.get("role") or "").strip().lower()
        if role not in {"system", "user", "assistant"}:
            raise invalid("message role must be system, user, or assistant")
        has_user = has_user or role == "user"
        content = message.get("content")
        if isinstance(content, str):
            text_bytes += len(content.encode("utf-8"))
            if text_bytes > MAX_CHAT_TEXT_BYTES:
                raise invalid("combined message text exceeds 1MiB limit")
            normalized.append({"role": role, "content": content})
            continue
        if not isinstance(content, list) or not content:
            raise invalid("message content must be text or a non-empty parts array")
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise invalid("message parts must be objects")
            kind = str(part.get("type") or "").strip()
            if kind in {"text", "input_text"}:
                if set(part) != {"type", "text"} or not isinstance(part.get("text"), str):
                    raise invalid("text parts accept only type and text")
                text_bytes += len(part["text"].encode("utf-8"))
                if text_bytes > MAX_CHAT_TEXT_BYTES:
                    raise invalid("combined message text exceeds 1MiB limit")
                parts.append({"type": "text", "text": part["text"]})
            elif kind in {"image_url", "input_image", "image"}:
                if role != "user":
                    raise invalid("only user messages may contain images")
                parts.append(image_part(part))
            else:
                raise invalid("message part type is not supported", "CHAT_OPTION_UNSUPPORTED")
        normalized.append({"role": role, "content": parts})
    if not has_user:
        raise invalid("messages must include a user message")
    return normalized


def _response_mime_type(response: requests.Response, parsed_path: str) -> str:
    """识别下载图片类型：优先响应头，必要时按 URL 后缀推断。"""
    header_type = str(response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    guessed_type = mimetypes.guess_type(parsed_path)[0] or ""
    if header_type.startswith("image/"):
        return header_type
    if header_type and header_type not in {"application/octet-stream", "binary/octet-stream"}:
        raise HTTPException(status_code=400, detail={"error": "image_url must point to an image"})
    if guessed_type.startswith("image/"):
        return guessed_type
    if not header_type or header_type in {"application/octet-stream", "binary/octet-stream"}:
        return ""
    raise HTTPException(status_code=400, detail={"error": "image_url must point to an image"})


def _filename_from_url(parsed_path: str, mime_type: str) -> str:
    """生成 URL 图片文件名：从链接路径提取名称并做安全化。"""
    raw_name = PurePosixPath(unquote(parsed_path)).name
    return _safe_filename(raw_name, mime_type, "image_url")


def _public_destination(url: str) -> tuple[str, int, str]:
    """Resolve one URL to a public IP that can be pinned for the request."""
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise _input_error("image_url must be a public http or https URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise _input_error("image_url must be a public http or https URL")
    try:
        hostname = parsed.hostname
    except ValueError as exc:
        raise _input_error("image_url must be a public http or https URL") from exc
    if not hostname:
        raise _input_error("image_url must be a public http or https URL")
    if hostname.endswith("."):
        raise _input_error("image_url hostname is invalid")
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise _input_error("image_url hostname is invalid") from exc
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise _input_error("image_url port is invalid") from exc
    try:
        literal = ipaddress.ip_address(ascii_hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if not literal.is_global:
            raise _input_error("image_url destination must resolve to a public IP")
        return ascii_hostname, port, str(literal)

    try:
        addresses = {
            str(item[4][0])
            for item in socket.getaddrinfo(ascii_hostname, port, type=socket.SOCK_STREAM)
            if item[4] and item[4][0]
        }
    except (OSError, socket.gaierror) as exc:
        raise _input_error("image_url hostname could not be resolved") from exc
    if not addresses:
        raise _input_error("image_url hostname could not be resolved")
    try:
        parsed_addresses = [ipaddress.ip_address(address) for address in addresses]
    except ValueError as exc:
        raise _input_error("image_url hostname resolved to an invalid IP") from exc
    if any(not address.is_global for address in parsed_addresses):
        raise _input_error("image_url destination must resolve only to public IPs")
    # Pin a single public address. Re-resolving and re-pinning on every manual
    # redirect prevents a later DNS answer from changing the destination to a
    # private address during this download.
    return ascii_hostname, port, sorted(addresses)[0]


def _read_response_bytes(response: requests.Response) -> bytes:
    """Read a streamed response while enforcing the per-image byte limit."""
    content_length = _clean(response.headers.get("content-length"))
    if content_length and content_length.isdigit() and int(content_length) > MAX_IMAGE_REFERENCE_BYTES:
        response.close()
        raise _input_error("image_url exceeds 50MB limit")
    chunks: list[bytes] = []
    size = 0
    try:
        for chunk in response.iter_content():
            if not chunk:
                continue
            size += len(chunk)
            if size > MAX_IMAGE_REFERENCE_BYTES:
                raise _input_error("image_url exceeds 50MB limit")
            chunks.append(bytes(chunk))
    finally:
        response.close()
    data = b"".join(chunks)
    if not data:
        raise _input_error("image_url returned empty content")
    return data


def _curl_slist(values: list[str]):
    """Build a libcurl string list for options not wrapped by curl_cffi."""
    values_ptr = ffi.NULL
    for value in values:
        values_ptr = lib.curl_slist_append(values_ptr, value.encode("ascii"))
    if values and values_ptr == ffi.NULL:
        raise _input_error("image_url destination pinning could not be initialized")
    return values_ptr


def _download_image_url(url: str) -> ImageInput:
    """下载远程图片：把 http/https 图片链接转成标准图片输入元组。"""
    source = _clean(url)
    if source.lower().startswith("data:"):
        return _decode_data_url(source)
    current = source
    proxy_kwargs = proxy_settings.build_session_kwargs(upstream=True)
    proxy_url = _clean(proxy_kwargs.get("proxy"))
    if proxy_kwargs.get("proxies"):
        raise _input_error("remote image fetch requires a pin-capable configured egress proxy")
    try:
        proxy_scheme = urlparse(proxy_url).scheme.lower() if proxy_url else ""
    except ValueError as exc:
        raise _input_error("remote image fetch requires a valid egress proxy for destination pinning") from exc
    if proxy_url and proxy_scheme not in {"http", "https"}:
        raise _input_error("remote image fetch requires an HTTP or HTTPS egress proxy for destination pinning")
    for redirect_count in range(MAX_IMAGE_REDIRECTS + 1):
        try:
            parsed = urlparse(current)
        except ValueError as exc:
            raise _input_error("image_url must be a public http or https URL") from exc
        ascii_hostname, port, resolved_ip = _public_destination(current)
        mapped_hostname = f"[{ascii_hostname}]" if ":" in ascii_hostname else ascii_hostname
        resolve_address = f"[{resolved_ip}]" if ":" in resolved_ip else resolved_ip
        if proxy_url:
            connect_to = f"{mapped_hostname}:{port}:{resolve_address}:{port}"
            try:
                connect_to_list = _curl_slist([connect_to])
            except HTTPException:
                raise
            except Exception as exc:
                raise _input_error("image_url destination pinning could not be initialized") from exc
            curl_options = {CurlOpt.CONNECT_TO: connect_to_list}
        else:
            connect_to_list = ffi.NULL
            resolve_host = f"{mapped_hostname}:{port}:{resolve_address}"
            curl_options = {CurlOpt.RESOLVE: [resolve_host]}
        session = None
        try:
            session = requests.Session(trust_env=False, curl_options=curl_options)
            response = session.get(
                current,
                headers={"Accept": "image/*,*/*;q=0.8", "User-Agent": "chatgpt2api image fetcher"},
                timeout=60,
                allow_redirects=False,
                stream=True,
                **proxy_kwargs,
            )
        except HTTPException:
            if session is not None:
                session.close()
            if connect_to_list != ffi.NULL:
                lib.curl_slist_free_all(connect_to_list)
            raise
        except Exception as exc:
            if session is not None:
                session.close()
            if connect_to_list != ffi.NULL:
                lib.curl_slist_free_all(connect_to_list)
            raise _input_error("image_url fetch failed") from exc
        if 300 <= response.status_code < 400:
            location = _clean(response.headers.get("location"))
            response.close()
            if session is not None:
                session.close()
            if connect_to_list != ffi.NULL:
                lib.curl_slist_free_all(connect_to_list)
            if not location:
                raise _input_error("image_url redirect did not provide a destination")
            if redirect_count >= MAX_IMAGE_REDIRECTS:
                raise _input_error("image_url has too many redirects")
            current = urljoin(current, location)
            continue
        if not 200 <= response.status_code < 300:
            response.close()
            if session is not None:
                session.close()
            if connect_to_list != ffi.NULL:
                lib.curl_slist_free_all(connect_to_list)
            raise _input_error("image_url fetch failed")
        try:
            mime_type = _response_mime_type(response, parsed.path)
        except Exception:
            response.close()
            if session is not None:
                session.close()
            if connect_to_list != ffi.NULL:
                lib.curl_slist_free_all(connect_to_list)
            raise
        try:
            data = _read_response_bytes(response)
        except HTTPException:
            raise
        except Exception as exc:
            raise _input_error("image_url fetch failed") from exc
        finally:
            if session is not None:
                session.close()
            if connect_to_list != ffi.NULL:
                lib.curl_slist_free_all(connect_to_list)
        actual_mime = _validated_image_input(data, _filename_from_url(parsed.path, mime_type), mime_type)[2]
        return data, _filename_from_url(parsed.path, actual_mime), actual_mime
    raise _input_error("image_url has too many redirects")


async def read_image_sources(sources: list[ImageSource]) -> list[ImageInput]:
    """读取图片来源：上传文件直接读取，URL 下载后统一返回图片元组。"""
    if len(sources) > MAX_IMAGE_INPUT_COUNT:
        raise _input_error(f"at most {MAX_IMAGE_INPUT_COUNT} images are allowed")
    images: list[ImageInput] = []
    total_bytes = 0
    for source in sources:
        if isinstance(source, tuple):
            image = _validated_image_input(*source)
            total_bytes += len(image[0])
            if total_bytes > MAX_IMAGE_INPUT_BYTES:
                raise _input_error("combined image inputs exceed 100MB limit")
            images.append(image)
            continue
        if _is_upload(source):
            try:
                chunks: list[bytes] = []
                size = 0
                while True:
                    chunk = await source.read(IMAGE_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_IMAGE_REFERENCE_BYTES:
                        raise _input_error("image file exceeds 50MB limit")
                    chunks.append(bytes(chunk))
                image_data = b"".join(chunks)
            finally:
                await source.close()
            image = _validated_image_input(image_data, source.filename or "image.png", source.content_type or "")
            total_bytes += len(image[0])
            if total_bytes > MAX_IMAGE_INPUT_BYTES:
                raise _input_error("combined image inputs exceed 100MB limit")
            images.append(image)
            continue
        image = await run_in_threadpool(_download_image_url, source)
        total_bytes += len(image[0])
        if total_bytes > MAX_IMAGE_INPUT_BYTES:
            raise _input_error("combined image inputs exceed 100MB limit")
        images.append(image)
    if not images:
        raise HTTPException(status_code=400, detail={"error": "image file or image_url is required"})
    return images
