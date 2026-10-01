#!/usr/bin/env python3
"""Small durable client for chatgpt2api's persistent image and Chat APIs."""

from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import fcntl
import hashlib
from http.client import HTTPException
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
        try:
            with self.open(method, endpoint, body=body, content_type=content_type) as response:
                raw = response.read(MAX_JSON_BYTES + 1)
        except (OSError, HTTPException) as exc:
            raise ClientError("response interrupted; query the original durable request") from exc
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
    contract = {
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
    account_ref = _selected_account(args)
    if account_ref is not None:
        contract["account_ref"] = account_ref
    contract.update(_image_work_fields(args, images))
    scheduling = _scheduling(args)
    if scheduling is not None:
        contract["scheduling"] = scheduling
    return contract


def _selected_account(args: argparse.Namespace) -> str | None:
    account_ref = getattr(args, "account_ref", None)
    if account_ref is not None and not re.fullmatch(r"car_[A-Za-z0-9_-]{43}", account_ref):
        raise ClientError("--account-ref must be an opaque car_ reference from the model directory")
    return account_ref


def _scheduling(args: argparse.Namespace) -> dict | None:
    fields = ("workflow_id", "workflow_concurrency", "min_send_interval_seconds", "not_before", "wait_deadline")
    result = {key: getattr(args, key) for key in fields if getattr(args, key, None) is not None}
    if not result:
        return None
    if "workflow_concurrency" in result and (not result.get("workflow_id") or not 1 <= result["workflow_concurrency"] <= 64):
        raise ClientError("--workflow-concurrency requires --workflow-id and a value from 1 to 64")
    if "min_send_interval_seconds" in result and not 0 <= result["min_send_interval_seconds"] <= 86400:
        raise ClientError("--min-send-interval-seconds must be from 0 to 86400")
    dates = {}
    for key in ("not_before", "wait_deadline"):
        if key in result:
            try:
                dates[key] = dt.datetime.fromisoformat(result[key].replace("Z", "+00:00"))
                if dates[key].tzinfo is None or dates[key].utcoffset() != dt.timedelta(0):
                    raise ValueError()
            except ValueError:
                raise ClientError(f"--{key.replace('_', '-')} requires UTC ISO8601") from None
    if len(dates) == 2 and dates["wait_deadline"] <= dates["not_before"]:
        raise ClientError("wait deadline must follow not-before")
    return result


def _image_work_fields(args: argparse.Namespace, images: list[dict[str, Any]]) -> dict[str, Any]:
    thread = getattr(args, "thread_id", None)
    source = getattr(args, "source_task_id", None)
    index = getattr(args, "source_index", 0)
    if thread is None:
        if source is not None or index:
            raise ClientError("--source-task-id and --source-index require --thread-id")
        return {}
    fields: dict[str, Any] = {"image_thread_id": _validate_task_id(thread)}
    if args.model != "gpt-image-2":
        raise ClientError("image threads require the advertised gpt-image-2 route")
    if source is not None:
        if not images or not 0 <= index < len(images):
            raise ClientError("--source-index must identify a supplied original result image")
        fields.update(edit_source_task_id=_validate_task_id(source), edit_source_index=index)
    elif index:
        raise ClientError("--source-index requires --source-task-id")
    return fields


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
                if "account_ref" in contract:
                    fields["account_ref"] = contract["account_ref"]
                if "scheduling" in contract:
                    fields["scheduling"] = json.dumps(contract["scheduling"], separators=(",", ":"))
                for key in ("image_thread_id", "edit_source_task_id", "edit_source_index"):
                    if key in contract:
                        fields[key] = str(contract[key])
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
                if "account_ref" in contract:
                    payload["account_ref"] = contract["account_ref"]
                if "scheduling" in contract:
                    payload["scheduling"] = contract["scheduling"]
                for key in ("image_thread_id", "edit_source_task_id", "edit_source_index"):
                    if key in contract:
                        payload[key] = contract[key]
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
    return _download_task(api, args, task_id)


def _download_task(api: ApiClient, args: argparse.Namespace, task_id: str, *, emit=True) -> int:
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
        created = False
        total = 0
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
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
                if total == 0 or (length.isdigit() and total != int(length)):
                    raise ClientError("download incomplete; retry downloading the original result")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException as exc:
            if fd >= 0:
                os.close(fd)
            if created:
                with contextlib.suppress(FileNotFoundError):
                    target.unlink()
            if isinstance(exc, (OSError, HTTPException)):
                raise ClientError("download or save failed; retry downloading the original result") from exc
            raise
    if emit:
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


def _verify_chat_receipt(result: dict[str, Any], request_id: str,
                         conversation: dict[str, Any] | None = None) -> None:
    if result.get("request_id") != request_id or result.get("route") != "chat":
        raise ClientError("Chat response changed the original request ID or route")
    if conversation is not None:
        actual = result.get("conversation")
        if not isinstance(actual, dict) or any(key not in actual or actual[key] != value for key, value in conversation.items()):
            raise ClientError("Chat response does not confirm the original sequential-v1 session; query the original ID, do not resubmit")


def _chat_receipt(api: ApiClient, request_id: str, *, recover: bool = False,
                  conversation: dict[str, Any] | None = None) -> dict[str, Any]:
    path = f"/api/chat-requests/{parse.quote(request_id, safe='')}"
    result = api.json("POST" if recover else "GET", path + ("/recover" if recover else ""),
                      payload={} if recover else None)
    _verify_chat_receipt(result, request_id, conversation)
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
    scheduling = _scheduling(args)
    if scheduling is not None:
        body["scheduling"] = scheduling
    account_ref = _selected_account(args)
    if account_ref is not None:
        body["account_ref"] = account_ref
    conversation = None
    if args.previous_request_id is not None and args.session_id is None:
        raise ClientError("--previous-request-id requires --session-id")
    if args.session_id is not None:
        body["client_conversation_id"] = _validate_task_id(args.session_id)
        if args.previous_request_id is not None:
            body["previous_request_id"] = _validate_task_id(args.previous_request_id)
        conversation = {"protocol": "sequential-v1", "client_conversation_id": args.session_id,
                        "previous_request_id": args.previous_request_id}
    fingerprint = _fingerprint(body)
    state_path = _chat_state_path(args)
    with _state_lock(state_path):
        state = _load_chat_state(state_path)
        if state:
            request_id = _chat_request_id(args, state)
            if fingerprint != state.get("input_fingerprint"):
                raise ClientError("Chat request already belongs to different immutable input")
            _emit(_chat_receipt(api, request_id, conversation=conversation))
            return 0
        request_id = _validate_task_id(args.request_id or f"chat-{uuid.uuid4()}")
        if args.previous_request_id:
            if request_id == args.previous_request_id:
                raise ClientError("a new turn must have a different request ID from its predecessor")
            previous = _chat_receipt(api, args.previous_request_id)
            prior_session = previous.get("conversation")
            if not isinstance(prior_session, dict) or prior_session.get("protocol") != "sequential-v1" or prior_session.get("client_conversation_id") != args.session_id:
                raise ClientError("previous Chat request does not belong to this sequential-v1 session")
            if previous.get("status") not in {"succeeded", "queued", "running"}:
                raise ClientError("previous Chat request cannot accept a dependent turn; recover its original result first")
            # Admission persists this dependency and sends only after the
            # previous answer finishes. Client acceptance is not a model send.
        state = {"schema": "chatgpt2api.chat-request.v1", "request_id": request_id,
                 "input_fingerprint": fingerprint, "phase": "prepared", "created_at": _utc_now()}
        if account_ref is not None:
            state["account_ref"] = account_ref
        if conversation is not None:
            state["conversation"] = conversation
        _atomic_write_state(state_path, state)
        try:
            result = api.json("POST", "/api/chat-requests", payload={"client_request_id": request_id, **body})
            _verify_chat_receipt(result, request_id, conversation)
        except (ClientError, KeyboardInterrupt):
            state.update(phase="unknown", updated_at=_utc_now())
            _atomic_write_state(state_path, state)
            raise
        state.update(phase="accepted", last_status=result.get("status"), updated_at=_utc_now())
        _atomic_write_state(state_path, state)
    _emit(result)
    return 0


def _command_chat_status(api: ApiClient, args: argparse.Namespace) -> int:
    state = _load_chat_state(_chat_state_path(args))
    request_id = _chat_request_id(args, state)
    _emit(_chat_receipt(api, request_id, recover=args.command == "chat-recover",
                        conversation=state.get("conversation") if state else None))
    return 0


def _command_chat_save(api: ApiClient, args: argparse.Namespace) -> int:
    state = _load_chat_state(_chat_state_path(args))
    request_id = _chat_request_id(args, state)
    result = _chat_receipt(api, request_id, conversation=state.get("conversation") if state else None)
    return _save_chat_result(result, request_id, args.output)


def _save_chat_result(result, request_id, output, *, emit=True):
    if result.get("status") != "succeeded" or not isinstance(result.get("content"), str):
        raise ClientError("original Chat result is not ready to save")
    target = _safe_output_path(output)
    created = False
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(fd, "wb") as handle:
            handle.write(_json_bytes(result))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if created:
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
        raise
    if emit:
        _emit({"request_id": request_id, "output": str(target), "bytes": target.stat().st_size})
    return 0


def _command_completion(api: ApiClient, args: argparse.Namespace) -> int:
    """Keep the original state and record the one server-created selected result."""
    chat = args.command.startswith("chat-")
    path = _chat_state_path(args) if chat else _state_path(args)
    action = args.command.rsplit("-", 1)[-1]
    with _state_lock(path):
        state = _load_chat_state(path) if chat else _load_state(path)
        if not state:
            raise ClientError("completion requires the original durable state file")
        original_id = state["request_id" if chat else "client_task_id"]
        endpoint = f"/api/{'chat-requests' if chat else 'image-tasks'}/{parse.quote(original_id, safe='')}/completion"
        payload = None
        if action == "recover":
            payload = {"action": "recover", "allow_unconfirmed_retry": args.allow_unconfirmed_retry}
        current = api.json("POST" if payload else "GET", endpoint, payload=payload)
        if (current.get("protocol") != "generation-completion-v1" or current.get("original_id") != original_id
                or current.get("kind") != ("text" if chat else "image")):
            raise ClientError("completion response changed original identity")
        prior = state.get("completion") or {}
        for key in ("replacement_id", "selected_id"):
            if prior.get(key) and prior[key] != current.get(key):
                raise ClientError("completion response changed its durable result selection")
        state["completion"] = current
        _atomic_write_state(path, state)
        selected = current.get("selected_id")
        if action in {"save", "complete", "rework"} and not selected:
            raise ClientError("no verified result selected; preserve the original task and inspect completion reason")
        if action == "save":
            if chat:
                result = current.get("result") or {}
                _verify_chat_receipt(result, selected)
                _save_chat_result(result, selected, args.output, emit=False)
            else:
                _download_task(api, argparse.Namespace(output=args.output, index=0), selected, emit=False)
            output = Path(args.output).expanduser().resolve(strict=True)
            saved = output.read_bytes()
            state["completion_output"] = {"selected_id": selected, "path": str(output),
                                          "bytes": len(saved), "sha256": hashlib.sha256(saved).hexdigest()}
            _atomic_write_state(path, state)
        elif action in {"complete", "rework"}:
            if action == "complete":
                saved = state.get("completion_output") or {}
                if saved.get("selected_id") != selected or not args.reviewed:
                    raise ClientError("save and review the selected actual result before completing the task")
                data = Path(saved["path"]).read_bytes()
                if not data or len(data) != saved["bytes"] or hashlib.sha256(data).hexdigest() != saved["sha256"]:
                    raise ClientError("saved result no longer matches; task remains incomplete")
            payload = {"action": action, "selected_id": selected}
            if action == "complete":
                payload.update(results_saved=True, reviewed=True)
            current = api.json("POST", endpoint, payload=payload)
            if current.get("original_id") != original_id or current.get("selected_id") != selected:
                raise ClientError("completion acknowledgement changed result identity")
            state["completion"] = current
            _atomic_write_state(path, state)
    _emit(current)
    return 0


def _command_work_status(api: ApiClient, args: argparse.Namespace) -> int:
    chat = args.command.startswith("chat-")
    path = _chat_state_path(args) if chat else _state_path(args)
    with _state_lock(path):
        state = _load_chat_state(path) if chat else _load_state(path)
        if not state:
            raise ClientError("work status requires the original state file")
        request_id = state["request_id" if chat else "client_task_id"]
        ref = (state.get("conversation") or {}).get("client_conversation_id") if chat else state.get("input", {}).get("image_thread_id")
        endpoint = f"/api/{'chat-requests' if chat else 'image-tasks'}/{parse.quote(request_id, safe='')}/work"
        result = api.json("GET", endpoint)
        if (result.get("protocol") != "work-v1" or result.get("request_id") != request_id
                or result.get("kind") != ("text" if chat else "image") or ref and result.get("work_ref") != ref):
            raise ClientError("work response changed original identity")
        state["work"] = result
        _atomic_write_state(path, state)
    _emit(result)
    return 0


def _command_work_lifecycle(api: ApiClient, args: argparse.Namespace) -> int:
    """An explicit work-complete event archives; rework restores that same work."""
    chat = args.command.startswith("chat-")
    archived = args.command in {"complete", "chat-complete"}
    path = _chat_state_path(args) if chat else _state_path(args)
    with _state_lock(path):
        state = _load_chat_state(path) if chat else _load_state(path)
        if state is None:
            raise ClientError("work lifecycle requires the original durable state file")
        if chat:
            task_id = _chat_request_id(args, state)
            expected = state.get("conversation")
            if not isinstance(expected, dict) or not expected.get("client_conversation_id"):
                raise ClientError("work lifecycle requires an original sequential-v1 session")
            current = _chat_receipt(api, task_id, conversation=expected)
            if current.get("status") != "succeeded":
                raise ClientError("work is not complete; query the original request before archiving or rework")
            endpoint = f"/api/chat-requests/{parse.quote(task_id, safe='')}/{'archive' if archived else 'restore'}-conversation"
        else:
            task_id = state["client_task_id"]
            thread_id = state.get("input", {}).get("image_thread_id")
            current = _lookup_task(api, task_id)
            expected = current.get("image_thread")
            if (not thread_id or not isinstance(expected, dict) or expected.get("protocol") != "image-thread-v1"
                    or expected.get("id") != thread_id):
                raise ClientError("work lifecycle requires the original image-thread-v1 receipt")
            if current.get("status") != "success":
                raise ClientError("work is not complete; query the original image before archiving or rework")
            endpoint = f"/api/image-tasks/{parse.quote(task_id, safe='')}/{'archive' if archived else 'restore'}-thread"
        state["lifecycle"] = {"operation": "archive" if archived else "restore", "status": "prepared", "updated_at": _utc_now()}
        _atomic_write_state(path, state)
        try:
            result = api.json("POST", endpoint, payload={})
            scope = result.get("conversation") if chat else result.get("image_thread")
            id_key, scope_key = ("request_id", "client_conversation_id") if chat else ("task_id", "id")
            if (result.get(id_key) != task_id or result.get("archived") is not archived
                    or not isinstance(scope, dict) or scope.get("protocol") != expected.get("protocol")
                    or scope.get(scope_key) != expected.get(scope_key)):
                raise ClientError("lifecycle response did not confirm the original work and archive state")
        except ClientError:
            state["lifecycle"].update(status="unknown", updated_at=_utc_now())
            _atomic_write_state(path, state)
            raise
        state["lifecycle"].update(status="confirmed", archived=archived, updated_at=_utc_now())
        _atomic_write_state(path, state)
    _emit(result)
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
    submit.add_argument("--account-ref", help="advanced: require this opaque company account; omission keeps automatic allocation")
    submit.add_argument("--thread-id", help="application work reference for an image-thread-v1 conversation")
    submit.add_argument("--source-task-id", help="original successful task whose saved image is supplied for rework")
    submit.add_argument("--source-index", type=int, default=0, help="index of the supplied original result among --image inputs")
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
    chat.add_argument("--session-id", help="application work session; requires Provider sequential-v1")
    chat.add_argument("--previous-request-id", help="previous succeeded request in the same session; checked before a new submit")
    chat.add_argument("--account-ref", help="advanced: require this opaque company account; omission keeps automatic allocation")
    chat.add_argument("--model", required=True)
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--image", action="append", default=[])
    for command in ("chat-status", "chat-recover"):
        operation = subparsers.add_parser(command, help="read the original Chat receipt; recover only reads the upstream result")
        operation.add_argument("--state")
        operation.add_argument("--request-id")
    save = subparsers.add_parser("chat-save", help="save the original successful Chat result, never rerun the model")
    save.add_argument("--state", required=True)
    save.add_argument("--output", required=True)
    for command in ("work-status", "chat-work-status"):
        work = subparsers.add_parser(command, help="read original work and automatic archive/restore progress")
        work.add_argument("--state", required=True)
    for operation in (submit, chat):
        operation.add_argument("--workflow-id")
        operation.add_argument("--workflow-concurrency", type=int)
        operation.add_argument("--min-send-interval-seconds", type=float)
        operation.add_argument("--not-before", help="UTC ISO8601 earliest actual send")
        operation.add_argument("--wait-deadline", help="UTC ISO8601 deadline for work that has not yet been sent")
    for command in ("complete", "rework", "chat-complete", "chat-rework"):
        lifecycle = subparsers.add_parser(command, help="archive completed work or restore its original conversation for rework")
        lifecycle.add_argument("--state", required=True)
    for prefix in ("chat-", ""):
        for action in ("recover", "status", "save", "complete", "rework"):
            operation = subparsers.add_parser(prefix + "completion-" + action,
                help="bounded pure-generation recovery with original identity and selected saved result")
            operation.add_argument("--state", required=True)
            if action == "recover":
                operation.add_argument("--allow-unconfirmed-retry", action="store_true",
                    help="explicitly permit at most one additional generation while original stop/outcome remains unknown")
            if action == "save":
                operation.add_argument("--output", required=True)
            if action == "complete":
                operation.add_argument("--reviewed", action="store_true")
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
            "chat-save": _command_chat_save,
            "work-status": _command_work_status,
            "chat-work-status": _command_work_status,
            "chat-complete": _command_work_lifecycle,
            "chat-rework": _command_work_lifecycle,
            "complete": _command_work_lifecycle,
            "rework": _command_work_lifecycle,
            "submit": _command_submit,
            "status": _command_status,
            "resume": _command_resume,
            "download": _command_download,
        }
        commands.update({prefix + "completion-" + action: _command_completion
                         for prefix in ("chat-", "") for action in ("recover", "status", "save", "complete", "rework")})
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
