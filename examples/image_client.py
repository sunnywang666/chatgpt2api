#!/usr/bin/env python3
"""Small durable client for chatgpt2api's persistent image-task API."""

from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import tempfile
import uuid
from typing import Any, BinaryIO, Iterator
from urllib import error, parse, request


STATE_SCHEMA_VERSION = 1
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_INPUT_IMAGE_BYTES = 50 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9_.:-]{1,200}")


class ClientError(RuntimeError):
    """A safe user-facing error that never contains credentials."""


class HttpFailure(ClientError):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"HTTP {status}: {detail}")


class UnsafeRedirect(ClientError):
    pass


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _emit(value: object, *, stream: Any | None = None) -> None:
    if stream is None:
        stream = sys.stdout
    json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")


def _load_env_file(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ClientError(f"cannot read env file {path}: {exc.strerror or exc}") from exc
    for number, original in enumerate(lines, start=1):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key or not key.replace("_", "A").isalnum() or key[0].isdigit():
            raise ClientError(f"invalid env assignment at {path}:{number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _origin(value: str) -> tuple[str, str, int]:
    parsed = parse.urlsplit(value)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if scheme not in {"http", "https"} or not host:
        raise ClientError("SERVER_ROOT must be an absolute http or https URL")
    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise ClientError("SERVER_ROOT has an invalid port") from exc
    return scheme, host, port


def _normalize_server_root(value: str) -> str:
    text = str(value or "").strip().rstrip("/")
    if not text:
        raise ClientError("SERVER_ROOT is required")
    parsed = parse.urlsplit(text)
    _origin(text)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ClientError("SERVER_ROOT must not contain credentials, a query, or a fragment")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1") or path == "/v1":
        raise ClientError("SERVER_ROOT is the service root, not OPENAI_BASE_URL ending in /v1")
    return parse.urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


class _SameOriginRedirectHandler(request.HTTPRedirectHandler):
    def __init__(self, allowed_origin: tuple[str, str, int]):
        super().__init__()
        self.allowed_origin = allowed_origin

    def redirect_request(self, req: request.Request, fp: BinaryIO, code: int, msg: str, headers: Any, newurl: str):
        absolute = parse.urljoin(req.full_url, newurl)
        if _origin(absolute) != self.allowed_origin:
            raise UnsafeRedirect("refused to send credentials across an origin-changing redirect")
        return super().redirect_request(req, fp, code, msg, headers, absolute)


class ApiClient:
    def __init__(self, server_root: str, bearer_token: str, timeout: float):
        self.server_root = _normalize_server_root(server_root)
        self.bearer_token = str(bearer_token or "").strip()
        if not self.bearer_token:
            raise ClientError("CHATGPT2API_BEARER_TOKEN is required")
        if timeout <= 0:
            raise ClientError("timeout must be greater than zero")
        self.timeout = timeout
        self.origin = _origin(self.server_root)
        self.opener = request.build_opener(_SameOriginRedirectHandler(self.origin))

    def url(self, endpoint: str) -> str:
        if not endpoint.startswith("/") or "://" in endpoint:
            raise ClientError("API endpoint must be a root-relative path")
        result = f"{self.server_root}{endpoint}"
        if _origin(result) != self.origin:
            raise ClientError("API endpoint resolved outside SERVER_ROOT")
        return result

    def open(self, method: str, endpoint: str, *, body: bytes | None = None, content_type: str = ""):
        headers = {
            "Accept": "application/json" if endpoint != "" else "*/*",
            "Authorization": f"Bearer {self.bearer_token}",
            "User-Agent": "chatgpt2api-persistent-image-client/1",
            "X-Workbench-Image-Client": "1",
        }
        if content_type:
            headers["Content-Type"] = content_type
        req = request.Request(self.url(endpoint), data=body, headers=headers, method=method)
        try:
            return self.opener.open(req, timeout=self.timeout)
        except UnsafeRedirect:
            raise
        except error.HTTPError as exc:
            detail = _http_error_detail(exc)
            raise HttpFailure(exc.code, detail) from exc
        except (TimeoutError, error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ClientError(f"request result is unknown: {reason}") from exc

    def json(self, method: str, endpoint: str, *, payload: object | None = None, body: bytes | None = None, content_type: str = "") -> dict[str, Any]:
        if payload is not None:
            body = _json_bytes(payload)
            content_type = "application/json"
        with self.open(method, endpoint, body=body, content_type=content_type) as response:
            raw = response.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise ClientError("server JSON response exceeds 4 MiB")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ClientError("server returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ClientError("server JSON response must be an object")
        return value


def _http_error_detail(exc: error.HTTPError) -> str:
    try:
        raw = exc.read(64 * 1024)
        value = json.loads(raw.decode("utf-8"))
    except Exception:
        return str(exc.reason or "request failed")
    detail = value.get("detail") if isinstance(value, dict) else value
    if isinstance(detail, dict):
        detail = detail.get("error") or detail.get("code") or detail
    if isinstance(detail, (dict, list)):
        return json.dumps(detail, ensure_ascii=False, separators=(",", ":"))
    return str(detail or exc.reason or "request failed")


def _read_input_image(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ClientError(f"cannot read image {path}: {exc.strerror or exc}") from exc
    if not resolved.is_file():
        raise ClientError(f"image is not a regular file: {path}")
    size = resolved.stat().st_size
    if size <= 0:
        raise ClientError(f"image is empty: {path}")
    if size > MAX_INPUT_IMAGE_BYTES:
        raise ClientError(f"image exceeds 50 MiB: {path}")
    data = resolved.read_bytes()
    if len(data) > MAX_INPUT_IMAGE_BYTES:
        raise ClientError(f"image exceeds 50 MiB: {path}")
    return {
        "data": data,
        "name": resolved.name,
        "content_type": mimetypes.guess_type(resolved.name)[0] or "application/octet-stream",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _request_contract(args: argparse.Namespace, images: list[dict[str, Any]]) -> dict[str, Any]:
    mode = "edit" if images else "generate"
    return {
        "schema": "chatgpt2api.persistent-image-input.v1",
        "mode": mode,
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "model": args.model,
        "size": args.size or "",
        "quality": args.quality,
        "images": [
            {"sha256": item["sha256"], "bytes": item["bytes"]}
            for item in images
        ],
    }


def _fingerprint(contract: dict[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(_json_bytes(contract)).hexdigest()}"


def _state_path(args: argparse.Namespace) -> Path:
    value = args.state or os.environ.get("IMAGE_CLIENT_STATE") or ".image-client-task.json"
    return Path(value).expanduser()


def _validate_task_id(value: object) -> str:
    if not isinstance(value, str) or TASK_ID_PATTERN.fullmatch(value) is None or value in {".", ".."}:
        raise ClientError("client task ID must be 1..200 characters from A-Z, a-z, 0-9, underscore, period, colon, or hyphen, and cannot be . or ..")
    return value


def _load_state(path: Path, *, required: bool = False) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise ClientError(f"state file does not exist: {path}")
        return None
    if not path.is_file():
        raise ClientError(f"state path is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClientError(f"state file is unreadable or invalid: {path}") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != STATE_SCHEMA_VERSION
        or not isinstance(value.get("client_task_id"), str)
        or not isinstance(value.get("input_fingerprint"), str)
    ):
        raise ClientError(f"state file has an unsupported shape: {path}")
    value["client_task_id"] = _validate_task_id(value["client_task_id"])
    return value


def _atomic_write_state(path: Path, state: dict[str, Any]) -> None:
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        parent = parent.resolve(strict=True)
    except OSError as exc:
        raise ClientError(f"cannot create state directory {path.parent}: {exc.strerror or exc}") from exc
    if not parent.is_dir():
        raise ClientError(f"state parent is not a directory: {path.parent}")
    target = parent / path.name
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
        dir_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)


@contextlib.contextmanager
def _state_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _multipart_body(fields: dict[str, str], images: list[dict[str, Any]]) -> tuple[bytes, str]:
    boundary = f"----chatgpt2api-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode("utf-8"),
            b"\r\n",
        ])
    for item in images:
        filename = str(item["name"]).replace('"', "_").replace("\r", "_").replace("\n", "_")
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode(),
            f'Content-Type: {item["content_type"]}\r\n\r\n'.encode(),
            item["data"],
            b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _task_id(args: argparse.Namespace, state: dict[str, Any] | None) -> str:
    explicit_value = getattr(args, "task_id", None)
    explicit = _validate_task_id(explicit_value) if explicit_value is not None else ""
    if state is not None:
        saved = state["client_task_id"]
        if explicit and explicit != saved:
            raise ClientError("task ID does not match the durable state file")
        return saved
    if not explicit:
        raise ClientError("provide --task-id or an existing --state file")
    return explicit


def _lookup_task(api: ApiClient, task_id: str) -> dict[str, Any]:
    query = parse.urlencode({"ids": task_id})
    envelope = api.json("GET", f"/api/image-tasks?{query}")
    items = envelope.get("items")
    missing = envelope.get("missing_ids")
    if not isinstance(items, list) or not isinstance(missing, list):
        raise ClientError("task-list response is missing items or missing_ids")
    matches = [item for item in items if isinstance(item, dict) and item.get("id") == task_id]
    unexpected = [item.get("id") for item in items if isinstance(item, dict) and item.get("id") != task_id]
    if unexpected or len(matches) > 1:
        raise ClientError("task-list response did not preserve the requested client task identity")
    if matches:
        if task_id in missing:
            raise ClientError("task-list response reports the same task as present and missing")
        return matches[0]
    if task_id not in missing:
        raise ClientError("task-list response did not account for the requested client task ID")
    raise ClientError(f"server reports client task ID {task_id!r} as missing")


def _command_models(api: ApiClient, _args: argparse.Namespace) -> int:
    _emit(api.json("GET", "/v1/models"))
    return 0


def _command_submit(api: ApiClient, args: argparse.Namespace) -> int:
    explicit_task_id = _validate_task_id(args.client_task_id) if args.client_task_id is not None else ""
    images = [_read_input_image(value) for value in args.image]
    contract = _request_contract(args, images)
    fingerprint = _fingerprint(contract)
    state_path = _state_path(args)
    with _state_lock(state_path):
        state = _load_state(state_path)
        if state is not None:
            if explicit_task_id and explicit_task_id != state["client_task_id"]:
                raise ClientError("--client-task-id does not match the durable state file")
            if fingerprint != state["input_fingerprint"]:
                raise ClientError("durable client task ID already belongs to different immutable input")
            task = _lookup_task(api, state["client_task_id"])
            _emit(task)
            return 0

        task_id = explicit_task_id or _validate_task_id(f"image-{uuid.uuid4()}")
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "client_task_id": task_id,
            "input_fingerprint": fingerprint,
            "input": contract,
            "phase": "prepared",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        _atomic_write_state(state_path, state)

        try:
            if images:
                fields = {
                    "client_task_id": task_id,
                    "prompt": args.prompt,
                    "model": args.model,
                    "quality": args.quality,
                }
                if args.size:
                    fields["size"] = args.size
                body, content_type = _multipart_body(fields, images)
                result = api.json(
                    "POST",
                    "/api/image-tasks/edits",
                    body=body,
                    content_type=content_type,
                )
            else:
                payload: dict[str, object] = {
                    "client_task_id": task_id,
                    "prompt": args.prompt,
                    "model": args.model,
                    "quality": args.quality,
                }
                if args.size:
                    payload["size"] = args.size
                result = api.json("POST", "/api/image-tasks/generations", payload=payload)
            if result.get("id") != task_id:
                raise ClientError("submit response changed the client task identity")
        except HttpFailure as exc:
            state.update(phase="http_error", http_status=exc.status, updated_at=_utc_now())
            _atomic_write_state(state_path, state)
            raise
        except ClientError:
            state.update(phase="unknown", updated_at=_utc_now())
            _atomic_write_state(state_path, state)
            raise
        state.update(
            phase="accepted",
            last_status=str(result.get("status") or ""),
            updated_at=_utc_now(),
        )
        _atomic_write_state(state_path, state)
    _emit(result)
    return 0


def _command_status(api: ApiClient, args: argparse.Namespace) -> int:
    state = _load_state(_state_path(args))
    _emit(_lookup_task(api, _task_id(args, state)))
    return 0


def _command_resume(api: ApiClient, args: argparse.Namespace) -> int:
    if not 5 <= args.extra_timeout_secs <= 120:
        raise ClientError("--extra-timeout-secs must be between 5 and 120")
    state = _load_state(_state_path(args))
    task_id = _task_id(args, state)
    encoded = parse.quote(task_id, safe="")
    result = api.json(
        "POST",
        f"/api/image-tasks/{encoded}/resume-poll",
        payload={"extra_timeout_secs": args.extra_timeout_secs},
    )
    if result.get("id") != task_id:
        raise ClientError("resume response changed the client task identity")
    _emit(result)
    return 0


def _safe_output_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.name or path.name in {".", ".."}:
        raise ClientError("output must name a file")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise ClientError(f"output directory does not exist: {path.parent}") from exc
    if not parent.is_dir():
        raise ClientError(f"output parent is not a directory: {path.parent}")
    target = parent / path.name
    if os.path.lexists(target):
        raise ClientError(f"refusing to overwrite existing output: {target}")
    return target


def _command_download(api: ApiClient, args: argparse.Namespace) -> int:
    state = _load_state(_state_path(args))
    task_id = _task_id(args, state)
    task = _lookup_task(api, task_id)
    if task.get("status") != "success":
        raise ClientError(f"task is not successful; current status is {task.get('status')!r}")
    data = task.get("data")
    if not isinstance(data, list) or args.index < 0 or args.index >= len(data):
        raise ClientError(f"task result index {args.index} does not exist")
    target = _safe_output_path(args.output)
    encoded = parse.quote(task_id, safe="")
    endpoint = f"/api/image-tasks/{encoded}/images/{args.index}"
    with api.open("GET", endpoint) as response:
        content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type and not (content_type.startswith("image/") or content_type == "application/octet-stream"):
            raise ClientError(f"download response is not an image: {content_type}")
        length = str(response.headers.get("Content-Length") or "")
        if length.isdigit() and int(length) > MAX_DOWNLOAD_BYTES:
            raise ClientError("download exceeds 100 MiB")
        fd = -1
        total = 0
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ClientError("download exceeds 100 MiB")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            if fd >= 0:
                os.close(fd)
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
            raise
    _emit({"client_task_id": task_id, "index": args.index, "output": str(target), "bytes": total})
    return 0


def _chat_state_path(args: argparse.Namespace) -> Path:
    return Path(args.state or ".chat-client-request.json").expanduser()


def _load_chat_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClientError("cannot read Chat state file") from exc
    if not isinstance(value, dict) or value.get("schema") != "chatgpt2api.chat-request.v1":
        raise ClientError("not a Chat request state file")
    _validate_task_id(value.get("request_id"))
    return value


def _chat_request_id(args: argparse.Namespace, state: dict[str, Any] | None) -> str:
    explicit = getattr(args, "request_id", None)
    if state:
        if explicit and explicit != state["request_id"]:
            raise ClientError("request ID does not match the Chat state file")
        return state["request_id"]
    return _validate_task_id(explicit)


def _chat_receipt(api: ApiClient, request_id: str, *, recover: bool = False) -> dict[str, Any]:
    path = f"/api/chat-requests/{parse.quote(request_id, safe='')}"
    result = api.json("POST" if recover else "GET", path + ("/recover" if recover else ""),
                      payload={} if recover else None)
    if result.get("request_id") != request_id:
        raise ClientError("Chat response changed the original request ID")
    return result


def _command_chat_submit(api: ApiClient, args: argparse.Namespace) -> int:
    parts: list[dict[str, Any]] = [{"type": "text", "text": args.prompt}]
    for path in args.image:
        item = _read_input_image(path)
        if item["content_type"] not in {"image/png", "image/jpeg", "image/webp"}:
            raise ClientError("Chat images must be PNG, JPEG or WebP")
        parts.append({"type": "image_url", "image_url": {"url":
            f"data:{item['content_type']};base64," + base64.b64encode(item["data"]).decode("ascii")}})
    body = {"model": args.model, "messages": [{"role": "user", "content": parts}]}
    fingerprint = _fingerprint(body)
    state_path = _chat_state_path(args)
    with _state_lock(state_path):
        state = _load_chat_state(state_path)
        if state:
            request_id = _chat_request_id(args, state)
            if fingerprint != state.get("input_fingerprint"):
                raise ClientError("Chat request already belongs to different immutable input")
            _emit(_chat_receipt(api, request_id))
            return 0
        request_id = _validate_task_id(args.request_id or f"chat-{uuid.uuid4()}")
        state = {"schema": "chatgpt2api.chat-request.v1", "request_id": request_id,
                 "input_fingerprint": fingerprint, "phase": "prepared", "created_at": _utc_now()}
        _atomic_write_state(state_path, state)
        try:
            result = api.json("POST", "/api/chat-requests", payload={"client_request_id": request_id, **body})
            if result.get("request_id") != request_id:
                raise ClientError("Chat response changed the original request ID")
        except (ClientError, KeyboardInterrupt):
            state.update(phase="unknown", updated_at=_utc_now())
            _atomic_write_state(state_path, state)
            raise
        state.update(phase="accepted", last_status=result.get("status"), updated_at=_utc_now())
        _atomic_write_state(state_path, state)
    _emit(result)
    return 0


def _command_chat_status(api: ApiClient, args: argparse.Namespace) -> int:
    request_id = _chat_request_id(args, _load_chat_state(_chat_state_path(args)))
    _emit(_chat_receipt(api, request_id, recover=args.command == "chat-recover"))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", help="optional KEY=VALUE file; existing environment values win")
    parser.add_argument("--server-root", help="service root override; normally use SERVER_ROOT")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds (default: 30)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("models", help="read the server's current model catalog")

    submit = subparsers.add_parser("submit", help="submit once, or query an existing durable task")
    submit.add_argument("--state", help="durable state file (default: IMAGE_CLIENT_STATE or .image-client-task.json)")
    submit.add_argument("--client-task-id", help="stable 1..200 character URL-safe caller ID; generated when omitted")
    submit.add_argument("--prompt", required=True)
    submit.add_argument("--model", required=True, help="choose an ID returned by the models command")
    submit.add_argument("--size")
    submit.add_argument("--quality", default="auto")
    submit.add_argument("--image", action="append", default=[], metavar="PATH", help="repeat for an edit task")

    status = subparsers.add_parser("status", help="query the original task; performs no upstream resume poll")
    status.add_argument("--state")
    status.add_argument("--task-id")

    resume = subparsers.add_parser("resume", help="continue polling the original unknown-outcome receipt")
    resume.add_argument("--state")
    resume.add_argument("--task-id")
    resume.add_argument("--extra-timeout-secs", type=float, default=30.0)

    download = subparsers.add_parser("download", help="download one authenticated result by receipt and index")
    download.add_argument("--state")
    download.add_argument("--task-id")
    download.add_argument("--index", type=int, default=0)
    download.add_argument("--output", required=True)
    chat = subparsers.add_parser("chat-submit", help="submit one durable Chat text/image request, or read its original result")
    chat.add_argument("--state")
    chat.add_argument("--request-id")
    chat.add_argument("--model", required=True)
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--image", action="append", default=[])
    for command in ("chat-status", "chat-recover"):
        operation = subparsers.add_parser(command, help="read the original Chat receipt; recover only reads the upstream result")
        operation.add_argument("--state")
        operation.add_argument("--request-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.env_file:
            _load_env_file(Path(args.env_file).expanduser())
        api = ApiClient(
            args.server_root or os.environ.get("SERVER_ROOT", ""),
            os.environ.get("CHATGPT2API_BEARER_TOKEN", ""),
            args.timeout,
        )
        commands = {
            "models": _command_models,
            "chat-submit": _command_chat_submit,
            "chat-status": _command_chat_status,
            "chat-recover": _command_chat_status,
            "submit": _command_submit,
            "status": _command_status,
            "resume": _command_resume,
            "download": _command_download,
        }
        return commands[args.command](api, args)
    except (ClientError, ValueError) as exc:
        _emit({"error": str(exc)}, stream=sys.stderr)
        return 1
    except KeyboardInterrupt:
        _emit(
            {"error": "interrupted; submission outcome may be unknown, so reuse the durable state and query status"},
            stream=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
