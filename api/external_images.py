"""The public image ingress exposes ordinary-key operations only.

The existing private Provider routes and receipts remain the source of truth.
The reverse proxy sets this marker on every request on its public AI prefix.
"""
from __future__ import annotations

import re
import base64
import asyncio
import time
from urllib.parse import quote, unquote, urlsplit

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

from api.support import require_identity
from services.image_storage_service import image_storage_service


def is_external(request: Request) -> bool:
    return request.headers.get("x-workbench-image-client") == "1"


async def external_image_boundary(request: Request, call_next):
    if not is_external(request):
        return await call_next(request)
    if request.method == "OPTIONS":
        return await call_next(request)
    path = request.url.path
    allowed = (
        request.method == "GET" and path in {"/v1/models", "/api/image-tasks"}
        or request.method == "POST" and path in {
            "/v1/images/generations", "/v1/images/edits",
            "/api/image-tasks/generations", "/api/image-tasks/edits",
        }
        or request.method == "POST" and re.fullmatch(r"/api/image-tasks/[^/]+/resume-poll", path)
        or request.method == "GET" and re.fullmatch(r"/api/image-tasks/[^/]+/images/[0-9]+", path)
    )
    if not allowed:
        return JSONResponse({"detail": {"error": "route not available on image service"}}, status_code=404)
    try:
        identity = require_identity(request.headers.get("authorization"))
        if identity.get("role") != "user":
            raise HTTPException(403, detail={"error": "use an ordinary service key"})
    except HTTPException as exc:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    response = await call_next(request)
    response.headers["Cache-Control"] = "private, no-store"
    return response


def validate_external_input(request: Request, payload: dict, *, synchronous: bool = False) -> None:
    if not is_external(request):
        return
    if str(payload.get("model") or "gpt-image-2") != "gpt-image-2":
        raise HTTPException(400, detail={"error": "this durable image service supports gpt-image-2; Codex route recovery is not supported"})
    if int(payload.get("n") or 1) != 1:
        raise HTTPException(400, detail={"error": "this durable image service accepts one output per task"})
    task_id = payload.get("client_task_id")
    if not task_id:
        raise HTTPException(400, detail={"error": "client_task_id is required; persist it before submitting and reuse it after a timeout"})
    if task_id is not None and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", str(task_id)):
        raise HTTPException(400, detail={"error": "client_task_id must be 1..200 URL-safe characters"})
    if any(payload.get(field) for field in (
        "provider_binding_id", "provider_account_identity", "client_conversation_id",
        "conversation_id", "parent_message_id", "retain_conversation", "upstream_model",
    )):
        raise HTTPException(400, detail={"error": "use the original task ID to resume a task"})
    if synchronous and (payload.get("response_format", "b64_json") != "b64_json" or payload.get("stream")):
        raise HTTPException(400, detail={"error": "synchronous image service uses b64_json; use image-tasks for downloadable results"})


def client_task(task: dict, request: Request) -> dict:
    if not is_external(request):
        return task
    result = {key: value for key, value in task.items() if key not in {
        "provider_binding_id", "provider_account_identity", "client_conversation_id",
        "image_session_id", "image_session_parent_id", "upstream_model",
    }}
    if isinstance(result.get("data"), list):
        prefix = request.headers.get("x-forwarded-prefix", "").rstrip("/")
        if prefix and (not re.fullmatch(r"/[A-Za-z0-9/_-]+", prefix) or prefix.startswith("//")):
            prefix = ""
        result["data"] = [
            {"url": f"{prefix}/api/image-tasks/{quote(str(task['id']), safe='')}/images/{index}"}
            for index, _ in enumerate(result["data"])
        ]
    if result.get("error"):
        result["error"] = "Image task did not complete; inspect error_code and resume only the original task."
    result.pop("progress", None)
    return result


def task_image_bytes(task: dict, index: int) -> bytes:
    data = task.get("data")
    if task.get("status") != "success" or not isinstance(data, list) or index < 0 or index >= len(data):
        raise HTTPException(404, detail={"error": "image not found"})
    item = data[index]
    if isinstance(item, dict) and item.get("b64_json"):
        try:
            return base64.b64decode(item["b64_json"], validate=True)
        except (ValueError, TypeError):
            raise HTTPException(502, detail={"error": "stored image is invalid"}) from None
    url = str(item.get("url") or "") if isinstance(item, dict) else ""
    path = urlsplit(url).path
    # Resolve only a Provider-owned image recorded by this exact task. Never
    # fetch a receipt URL over the network or accept a caller-supplied path.
    marker = "/images/"
    public_base = str(image_storage_service.settings().get("public_base_url") or "").rstrip("/")
    if public_base and url.startswith(public_base + "/"):
        relative = unquote(urlsplit(url[len(public_base) + 1:]).path)
    elif marker in path:
        relative = unquote(path.split(marker, 1)[1])
    else:
        raise HTTPException(404, detail={"error": "image not found"})
    return image_storage_service.get_bytes(relative)


def client_sync_result(result, request: Request):
    if not is_external(request) or not isinstance(result, dict):
        return result
    return {key: value for key, value in result.items() if key in {"created", "data", "usage"}}


async def synchronous_external_task(identity: dict, payload: dict, *, edit: bool):
    from services.image_task_service import image_task_service
    task_id = payload["client_task_id"]
    arguments = {key: payload.get(key) for key in ("prompt", "model", "size", "quality", "base_url")}
    arguments["client_task_id"] = task_id
    if edit:
        arguments.update(images=payload["images"], masks=payload.get("mask"))
    try:
        task = await run_in_threadpool(
            image_task_service.submit_edit if edit else image_task_service.submit_generation,
            {**identity, "external_image_client": True}, **arguments)
    except ValueError:
        raise HTTPException(409, detail={"error": "client_task_id conflicts with original input"}) from None
    deadline = time.monotonic() + 30
    while task.get("status") in {"queued", "running"} and time.monotonic() < deadline:
        await asyncio.sleep(.25)
        listing = await run_in_threadpool(image_task_service.list_tasks, identity, [task_id])
        if not listing["items"]:
            break
        task = listing["items"][0]
    if task.get("status") == "success":
        data = []
        for index in range(len(task.get("data") or [])):
            content = await run_in_threadpool(task_image_bytes, task, index)
            data.append({"b64_json": base64.b64encode(content).decode("ascii")})
        return {"created": int(time.time()), "data": data, "task_id": task_id}
    if task.get("status") == "error" and task.get("error_code") != "CONVERSATION_OUTCOME_UNKNOWN":
        return JSONResponse({"error": {"code": task.get("error_code") or "image_generation_failed", "message": "Read the original task for its result; do not automatically resubmit"}, "task_id": task_id}, status_code=429 if task.get("error_code") == "IMAGE_RESOURCE_UNAVAILABLE" else 502)
    return JSONResponse({"task_id": task_id, "status": task.get("status"), "message": "Submission retained; query the original image-task ID. Do not submit a new task."}, status_code=202)
