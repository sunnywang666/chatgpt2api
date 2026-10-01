#!/usr/bin/env python3
"""One-at-a-time durable acceptance driver for the isolated Provider runtime.

This intentionally imports the candidate's existing ``examples/image_client.py``
for its same-origin HTTP client, safe redirect handling, atomic state writer,
and image multipart builder.  It adds only the narrow C17 receipt journal:
write a request record before its one POST, then refuse all new sends after a
HTTP failure, timeout/unknown outcome, or terminal failed/unknown receipt.

It is not a load scheduler.  The owner invokes each accepted request separately.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time
import uuid
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

SCHEMA = "provider-isolated-single-request-v2"
MAX_TEXT = 36
MAX_IMAGE = 12
TEXT_TERMINAL = {"succeeded", "failed", "unknown"}
IMAGE_TERMINAL = {"success", "error"}
SAFE_FIELDS = {"status", "id", "kind", "account_ref", "http_status", "result_bytes", "archive_status"}


def _die(code: str) -> None:
    raise SystemExit(code)


def _emit(**event: object) -> None:
    print(json.dumps({key: value for key, value in event.items() if key in SAFE_FIELDS},
                     separators=(",", ":")), flush=True)


def _require_0600(path: Path, label: str) -> None:
    info = path.stat()
    if not path.is_file() or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        _die(f"client_error={label}_must_be_current_user_0600")


def _read_key(path: Path) -> str:
    _require_0600(path, "test_key_file")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        _die("client_error=test_key_missing")
    return value


def _read_ready(path: Path) -> str:
    _require_0600(path, "ready_file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _die("client_error=invalid_local_readiness")
    host, port = str(value.get("host") or ""), value.get("port")
    if host not in {"127.0.0.1", "localhost", "::1"} or type(port) is not int or not 1 <= port <= 65535:
        _die("client_error=invalid_local_readiness")
    return f"http://{host}:{port}"


def _load_shared(source: Path):
    path = source.resolve() / "examples" / "image_client.py"
    if not path.is_file():
        _die("client_error=provider_examples_image_client_missing")
    spec = importlib.util.spec_from_file_location("provider_existing_image_client", path)
    if spec is None or spec.loader is None:
        _die("client_error=provider_existing_image_client_unloadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": SCHEMA, "requests": {}, "halted": False}
    _require_0600(path, "state_file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _die("client_error=invalid_state_file")
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA
            or not isinstance(value.get("requests"), dict) or type(value.get("halted")) is not bool):
        _die("client_error=invalid_state_file")
    return value


def _save(shared: Any, path: Path, state: dict[str, Any]) -> None:
    # Existing client helper fsyncs both replaced file and its parent directory.
    shared._atomic_write_state(path, state)
    os.chmod(path, 0o600)


def _new_id(prefix: str) -> str:
    return f"acceptance-{prefix}-{uuid.uuid4().hex}"


def _validate_id(shared: Any, value: str) -> str:
    try:
        return shared._validate_task_id(value)
    except Exception:
        _die("client_error=invalid_request_id")


def _safe_account(payload: object) -> str | None:
    if isinstance(payload, dict) and isinstance(payload.get("account_ref"), str):
        return payload["account_ref"]
    return None


def _halt(state: dict[str, Any], record: dict[str, Any], outcome: str, *, http_status: int | None = None) -> None:
    record["outcome"] = outcome
    if http_status is not None:
        record["http_status"] = http_status
    record["updated_at"] = int(time.time())
    state["halted"] = True
    state["halt_reason"] = outcome


def _chat_receipt(shared: Any, api: Any, request_id: str) -> dict[str, Any]:
    value = api.json("GET", f"/api/chat-requests/{quote(request_id, safe='')}")
    if value.get("request_id") != request_id or value.get("route") != "chat":
        raise shared.ClientError("original Chat receipt identity mismatch")
    return value


def _image_receipt(shared: Any, api: Any, task_id: str) -> dict[str, Any]:
    value = api.json("GET", "/api/image-tasks?" + urlencode({"ids": task_id}))
    items, missing = value.get("items"), value.get("missing_ids")
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict) or items[0].get("id") != task_id:
        if isinstance(missing, list) and task_id in missing:
            raise shared.ClientError("original image receipt is missing")
        raise shared.ClientError("original image receipt identity mismatch")
    return items[0]


def _save_text_content(record: dict[str, Any], receipt: dict[str, Any], result_dir: Path) -> int:
    content = receipt.get("content")
    if receipt.get("status") != "succeeded" or not isinstance(content, str):
        _die("client_error=text_result_not_ready_to_save")
    result_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(result_dir, 0o700)
    target = result_dir / f"{record['id']}.txt"
    if target.exists():
        _die("client_error=refuse_overwrite_result")
    raw = content.encode("utf-8")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    # Readback proves the saved text is exactly the public terminal result.
    saved = target.read_bytes()
    if saved != raw:
        _die("client_error=text_result_readback_mismatch")
    record["result_file"] = target.name
    record["result_bytes"] = len(saved)
    record["result_sha256"] = hashlib.sha256(saved).hexdigest()
    return len(saved)


def _submit(shared: Any, api: Any, state_path: Path, state: dict[str, Any], *,
            request_id: str, kind: str, endpoint: str, body: dict[str, Any],
            content_type: str = "", raw_body: bytes | None = None, session_id: str = "") -> int:
    if state["halted"]:
        _die("client_error=halted_state_allows_original_id_read_only")
    if any(item.get("submission") in {"prepared_not_sent", "sending_once"}
           for item in state["requests"].values()):
        _die("client_error=unconfirmed_submission_requires_original_readback")
    if request_id in state["requests"]:
        _die("client_error=existing_id_allows_original_id_read_only")
    used_text = sum(int(isinstance(item, dict) and item.get("budget_reserved", {}).get("text") == 1)
                    for item in state["requests"].values())
    used_image = sum(int(isinstance(item, dict) and item.get("budget_reserved", {}).get("image") == 1)
                     for item in state["requests"].values())
    if used_text + int(kind == "text") > MAX_TEXT or used_image + int(kind == "image") > MAX_IMAGE:
        _die("client_error=approved_budget_exceeded")
    record: dict[str, Any] = {
        "id": request_id, "kind": kind, "endpoint": endpoint, "body": body,
        "session_id": session_id, "budget_reserved": {"text": int(kind == "text"), "image": int(kind == "image")},
        "submission": "prepared_not_sent", "outcome": "pending", "created_at": int(time.time()),
    }
    # The immutable body/ID/budget are durable before the first byte can leave.
    state["requests"][request_id] = record
    _save(shared, state_path, state)
    record["submission"] = "sending_once"
    _save(shared, state_path, state)
    try:
        result = api.json("POST", endpoint, payload=None if raw_body is not None else body,
                          body=raw_body, content_type=content_type)
    except shared.HttpFailure as exc:
        _halt(state, record, "http_failed", http_status=int(exc.status))
        _save(shared, state_path, state)
        _emit(status="submission_failed", id=request_id, kind=kind, http_status=int(exc.status))
        return 2
    except (shared.ClientError, OSError, TimeoutError):
        _halt(state, record, "submission_unknown")
        _save(shared, state_path, state)
        _emit(status="submission_unknown", id=request_id, kind=kind)
        return 2
    returned_id = result.get("request_id") if kind == "text" else result.get("id")
    if returned_id != request_id:
        _halt(state, record, "submission_identity_unknown")
        _save(shared, state_path, state)
        _emit(status="submission_unknown", id=request_id, kind=kind)
        return 2
    record.update(submission="accepted", outcome="pending", accepted_status=str(result.get("status") or ""), updated_at=int(time.time()))
    _save(shared, state_path, state)
    _emit(status="submitted", id=request_id, kind=kind, account_ref=_safe_account(result))
    return 0


def _read_original(shared: Any, api: Any, state_path: Path, state: dict[str, Any], request_id: str) -> int:
    record = state["requests"].get(request_id)
    if not isinstance(record, dict):
        _die("client_error=read_requires_previously_journaled_original_id")
    kind = record.get("kind")
    try:
        receipt = _chat_receipt(shared, api, request_id) if kind == "text" else _image_receipt(shared, api, request_id)
    except shared.HttpFailure as exc:
        _halt(state, record, "read_http_failed", http_status=int(exc.status))
        _save(shared, state_path, state)
        _emit(status="read_failed", id=request_id, kind=str(kind), http_status=int(exc.status))
        return 2
    except shared.ClientError:
        _halt(state, record, "read_unknown")
        _save(shared, state_path, state)
        _emit(status="read_unknown", id=request_id, kind=str(kind))
        return 2
    status = str(receipt.get("status") or "").lower()
    record["submission"] = "accepted"
    record["last_status"] = status
    record["account_ref"] = _safe_account(receipt)
    record["updated_at"] = int(time.time())
    terminal = status in (TEXT_TERMINAL if kind == "text" else IMAGE_TERMINAL)
    if terminal and status not in {"succeeded", "success"}:
        _halt(state, record, "terminal_" + status)
    elif terminal:
        record["outcome"] = "succeeded"
    _save(shared, state_path, state)
    _emit(status=status or "unclassified", id=request_id, kind=str(kind), account_ref=_safe_account(receipt))
    return 0 if record.get("outcome") == "succeeded" else 2


def _work(shared: Any, api: Any, state_path: Path, state: dict[str, Any], request_id: str, active: bool) -> int:
    record = state["requests"].get(request_id)
    if not isinstance(record, dict) or record.get("kind") != "text":
        _die("client_error=text_work_requires_journaled_text_id")
    if active:
        last_work = record.get("work") if isinstance(record.get("work"), dict) else {}
        if last_work.get("last_archive_status") != "confirmed" or last_work.get("archived") is not True:
            _die("client_error=restore_requires_confirmed_completed_work")
        desired, body = False, {"state": "active", "results_saved": False}
    else:
        if record.get("outcome") != "succeeded" or not record.get("result_file"):
            _die("client_error=complete_requires_saved_successful_text")
        desired, body = True, {"state": "completed", "results_saved": True}
    try:
        updated = api.json("POST", f"/api/chat-requests/{quote(request_id, safe='')}/work", payload=body)
    except shared.HttpFailure as exc:
        _halt(state, record, "work_http_failed", http_status=int(exc.status))
        _save(shared, state_path, state)
        _emit(status="work_failed", id=request_id, kind="text", http_status=int(exc.status))
        return 2
    except shared.ClientError:
        _halt(state, record, "work_unknown")
        _save(shared, state_path, state)
        _emit(status="work_unknown", id=request_id, kind="text")
        return 2
    archive = updated.get("archive") if isinstance(updated, dict) else None
    archive_status = archive.get("status") if isinstance(archive, dict) else None
    record["work"] = {"desired_archived": desired, "last_archive_status": archive_status}
    record["updated_at"] = int(time.time())
    _save(shared, state_path, state)
    _emit(status="work_updated", id=request_id, kind="text", archive_status=archive_status)
    return 0


def _work_read(shared: Any, api: Any, state_path: Path, state: dict[str, Any], request_id: str) -> int:
    record = state["requests"].get(request_id)
    if not isinstance(record, dict) or record.get("kind") != "text":
        _die("client_error=text_work_requires_journaled_text_id")
    try:
        value = api.json("GET", f"/api/chat-requests/{quote(request_id, safe='')}/work")
    except shared.HttpFailure as exc:
        _halt(state, record, "work_read_http_failed", http_status=int(exc.status))
        _save(shared, state_path, state)
        _emit(status="work_read_failed", id=request_id, kind="text", http_status=int(exc.status))
        return 2
    except shared.ClientError:
        _halt(state, record, "work_read_unknown")
        _save(shared, state_path, state)
        _emit(status="work_read_unknown", id=request_id, kind="text")
        return 2
    archive = value.get("archive") if isinstance(value, dict) else None
    if not isinstance(archive, dict):
        _halt(state, record, "work_archive_unknown")
        _save(shared, state_path, state)
        _emit(status="work_read_unknown", id=request_id, kind="text")
        return 2
    record["work"] = {"desired_archived": archive.get("desired"), "last_archive_status": archive.get("status"),
                      "archived": archive.get("archived")}
    if archive.get("status") == "unknown":
        _halt(state, record, "work_archive_unknown")
    _save(shared, state_path, state)
    _emit(status="work_read", id=request_id, kind="text", archive_status=archive.get("status"))
    return 0 if archive.get("status") == "confirmed" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-source", type=Path, required=True,
                        help="bootstrap-created candidate source; imports its examples/image_client.py")
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--test-key-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    actions = parser.add_subparsers(dest="action", required=True)
    text = actions.add_parser("text-submit", help="journal and submit exactly one Chat turn")
    text.add_argument("--request-id")
    text.add_argument("--model", required=True)
    text.add_argument("--prompt", required=True)
    text.add_argument("--session-id", required=True)
    text.add_argument("--previous-request-id")
    image = actions.add_parser("image-submit", help="journal and submit exactly one generation or edit")
    image.add_argument("--task-id")
    image.add_argument("--model", required=True)
    image.add_argument("--prompt", required=True)
    image.add_argument("--image", type=Path, help="one PNG/JPEG/WebP source makes this an edit")
    image.add_argument("--size")
    image.add_argument("--quality", default="auto")
    read = actions.add_parser("read", help="read only an already journaled original receipt")
    read.add_argument("--id", required=True)
    save = actions.add_parser("text-save", help="save and read back an original successful text result")
    save.add_argument("--id", required=True)
    complete = actions.add_parser("text-complete", help="set completed/results_saved through the real work lifecycle")
    complete.add_argument("--id", required=True)
    restore = actions.add_parser("text-restore", help="restore the same completed text work for a later same-session turn")
    restore.add_argument("--id", required=True)
    work = actions.add_parser("text-work-read", help="read lifecycle; confirmed is the only archive/restore success")
    work.add_argument("--id", required=True)
    args = parser.parse_args()

    runtime = args.provider_source.resolve().parent
    for path in (args.ready_file, args.test_key_file, args.state_file, args.result_dir):
        if not path.resolve().is_relative_to(runtime):
            _die("client_error=all_paths_must_be_inside_isolated_runtime")
    candidate_sha = (runtime / "candidate.sha").read_text().strip()
    if json.loads(args.ready_file.read_text()).get("candidate_sha") != candidate_sha:
        _die("client_error=ready_candidate_mismatch")
    state_path = args.state_file.expanduser()
    state_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open(str(state_path) + ".lock", "a") as lock:
        os.chmod(lock.name, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _die("client_error=another_driver_owns_journal")
        return _run(args)


def _run(args) -> int:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                 "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        os.environ.pop(name, None)
    shared = _load_shared(args.provider_source)
    api = shared.ApiClient(_read_ready(args.ready_file), _read_key(args.test_key_file), 30.0)
    state_path = args.state_file.expanduser()
    state = _state(state_path)

    if args.action == "text-submit":
        request_id = _validate_id(shared, args.request_id or _new_id("text"))
        session_id = _validate_id(shared, args.session_id)
        body: dict[str, Any] = {"client_request_id": request_id, "model": args.model,
                                "client_conversation_id": session_id,
                                "messages": [{"role": "user", "content": args.prompt}]}
        if args.previous_request_id:
            previous_id = _validate_id(shared, args.previous_request_id)
            previous = state["requests"].get(previous_id)
            if (not isinstance(previous, dict) or previous.get("kind") != "text"
                    or previous.get("outcome") != "succeeded" or previous.get("session_id") != session_id):
                _die("client_error=previous_request_must_be_journaled_success_in_same_session")
            previous_work = previous.get("work") if isinstance(previous.get("work"), dict) else {}
            if previous_work.get("last_archive_status") != "confirmed" or previous_work.get("archived") is not False:
                _die("client_error=previous_request_must_have_confirmed_restore")
            body["previous_request_id"] = previous_id
        return _submit(shared, api, state_path, state, request_id=request_id, kind="text",
                       endpoint="/api/chat-requests", body=body, session_id=session_id)
    if args.action == "image-submit":
        task_id = _validate_id(shared, args.task_id or _new_id("image"))
        body: dict[str, Any] = {"client_task_id": task_id, "prompt": args.prompt, "model": args.model,
                                "quality": args.quality}
        if args.size:
            body["size"] = args.size
        if args.image is None:
            return _submit(shared, api, state_path, state, request_id=task_id, kind="image",
                           endpoint="/api/image-tasks/generations", body=body)
        image = shared._read_input_image(str(args.image))
        if image["content_type"] not in {"image/png", "image/jpeg", "image/webp"}:
            _die("client_error=edit_input_must_be_png_jpeg_or_webp")
        fields = {key: str(value) for key, value in body.items()}
        raw, content_type = shared._multipart_body(fields, [image])
        # Preserve the exact multipart input privately so evidence survives a crash.
        journal_body = {"content_type": content_type, "wire_body_b64": base64.b64encode(raw).decode("ascii")}
        return _submit(shared, api, state_path, state, request_id=task_id, kind="image",
                       endpoint="/api/image-tasks/edits", body=journal_body,
                       raw_body=raw, content_type=content_type)
    request_id = _validate_id(shared, args.id)
    if args.action == "read":
        return _read_original(shared, api, state_path, state, request_id)
    if args.action == "text-save":
        record = state["requests"].get(request_id)
        if not isinstance(record, dict) or record.get("kind") != "text":
            _die("client_error=text_save_requires_journaled_text_id")
        if record.get("result_file"):
            _die("client_error=text_result_already_saved")
        status = _read_original(shared, api, state_path, state, request_id)
        if status:
            return status
        receipt = _chat_receipt(shared, api, request_id)
        size = _save_text_content(record, receipt, args.result_dir.expanduser())
        _save(shared, state_path, state)
        _emit(status="result_saved", id=request_id, kind="text", result_bytes=size)
        return 0
    if args.action == "text-complete":
        return _work(shared, api, state_path, state, request_id, active=False)
    if args.action == "text-restore":
        return _work(shared, api, state_path, state, request_id, active=True)
    return _work_read(shared, api, state_path, state, request_id)


if __name__ == "__main__":
    raise SystemExit(main())
