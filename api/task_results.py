"""Owner-scoped, durable-only result receipts for Provider tasks.

This boundary intentionally reads one TaskStore receipt. It must remain
separate from the legacy task read APIs: those APIs may resume observation or
derive work state, whereas this endpoint is a pure post-send result read.
"""
from __future__ import annotations

import re
import base64
from typing import Literal
from urllib.parse import unquote, urlsplit

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from api.external_images import client_task, is_external
from api.key_policy import require_chat_text_policy, require_codex_endpoint, require_image_policy
from api.support import require_identity
from services.image_task_service import _public_task, image_task_service
from services.image_storage_service import image_storage_service
from services.public_chat_service import project_public_chat_receipt
from services.text_task_service import TextTaskService, text_task_service

_REQUEST_ID = re.compile(r"(?!\.{1,2}\Z)[A-Za-z0-9_.:-]{1,200}\Z")


def _owner(identity: dict[str, object]) -> str:
    owner = str(identity.get("id") or "").strip()
    if not owner:
        raise HTTPException(401, detail={"code": "KEY_IDENTITY_INVALID"})
    return owner


def _receipt(kind: str, owner: str, request_id: str) -> dict | None:
    # Deliberately bypass service.read/list_tasks/recovery paths. A SQLite read
    # is the only task operation this endpoint is permitted to perform.
    service = text_task_service if kind == "text" else image_task_service
    with service.store.connect() as db:
        return service.store.read_receipt(db, kind, owner, request_id)


def _status(kind: str, receipt: dict) -> str:
    value = receipt.get("status")
    status = value if isinstance(value, str) and value else "unknown"
    # RESULT_UNRECOVERABLE deliberately ends automatic reads, but its durable
    # upstream fact remains unknown. It is not evidence of a failed original.
    if receipt.get("upstream_outcome") == "unknown" and (
        (kind == "text" and status == "failed")
        or (kind == "image" and status == "error")
    ):
        return "unknown"
    if kind == "image":
        # Existing image callers retain an original-result error as UNKNOWN,
        # rather than as a terminal failure, until that conversation outcome
        # is independently resolved. A recovery pause changes neither fact.
        if status == "error" and receipt.get("error_code") == "CONVERSATION_OUTCOME_UNKNOWN":
            return "unknown"
        status = {"success": "succeeded", "error": "failed"}.get(status, status)
    return status if status in {"not_started", "queued", "running", "succeeded", "failed", "unknown"} else "unknown"


def _text_projection(receipt: dict) -> tuple[dict, object, bool]:
    # Reuse the existing public receipt projection; never expose the raw
    # durable SQLite row as a compatibility value.
    public = project_public_chat_receipt(TextTaskService._public(receipt))
    # _public intentionally removes private underscore fields before the
    # public contract sees them, so retain only the allowlisted route fact.
    if receipt.get("_route") == "codex":
        public["route"] = "codex"
    content = public.get("content")
    ready = receipt.get("status") == "succeeded" and isinstance(content, str) and bool(content.strip())
    return public, content if ready else None, ready


def _image_projection(receipt: dict, request: Request, *, private_bound: bool) -> tuple[dict, object, bool]:
    # client_task supplies the established external download-URL projection.
    public = client_task(_public_task(receipt), request)
    if not private_bound:
        # Existing public/company reads must never acquire the trusted bound
        # cursor fields retained by private Content/admin snapshots.
        for field in (
            "provider_binding_id", "provider_account_identity", "client_conversation_id",
            "image_session_id", "image_session_parent_id", "upstream_model", "adopted_from_error",
        ):
            public.pop(field, None)
    data = public.get("data")
    persisted_data = receipt.get("data")
    # A durable success marker alone is not enough: every persisted item must
    # be a readable image reference before this boundary certifies a result.
    complete_data = isinstance(persisted_data, list) and bool(persisted_data) and all(
        _locally_consumable_image_item(item) for item in persisted_data
    )
    ready = receipt.get("status") == "success" and complete_data and isinstance(data, list) and bool(data)
    return public, data if ready else None, ready


def _locally_consumable_image_item(item: object) -> bool:
    """Mirror task_image_bytes validation without reading an image or calling HTTP."""
    if not isinstance(item, dict):
        return False
    encoded = item.get("b64_json")
    if encoded:
        if not isinstance(encoded, str):
            return False
        try:
            return bool(base64.b64decode(encoded, validate=True))
        except (ValueError, TypeError):
            return False
    url = item.get("url")
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        path = urlsplit(url).path
        marker = "/images/"
        public_base = str(image_storage_service.settings().get("public_base_url") or "").rstrip("/")
        if public_base and url.startswith(public_base + "/"):
            relative = unquote(urlsplit(url[len(public_base) + 1:]).path)
        elif marker in path:
            relative = unquote(path.split(marker, 1)[1])
        else:
            return False
        # Metadata/path existence only: no image bytes, WebDAV, or upstream
        # request is opened while classifying a receipt as result-ready.
        return bool(relative) and image_storage_service.exists(relative)
    except (HTTPException, ValueError, TypeError):
        return False


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/task-results/{kind}/{request_id}")
    async def read_task_result(
        kind: Literal["text", "image"],
        request_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        if not _REQUEST_ID.fullmatch(request_id):
            raise HTTPException(400, detail={"code": "TASK_RESULT_ID_INVALID"})
        identity = require_identity(authorization, request=request)
        private_bound = identity.get("role") == "admin" and not is_external(request)
        if identity.get("role") != "user" and not private_bound:
            raise HTTPException(403, detail={"code": "ORDINARY_KEY_REQUIRED"})
        owner = _owner(identity)
        receipt = await run_in_threadpool(_receipt, kind, owner, request_id)
        if receipt is None:
            raise HTTPException(404, detail={"code": "TASK_RESULT_NOT_FOUND"})

        # Authorize after the owner-scoped lookup so an unprivileged key cannot
        # learn whether a different owner's ID exists. These are pure policy
        # checks and do not contact an upstream or change durable state.
        if kind == "text":
            if receipt.get("_route") == "codex":
                require_codex_endpoint(identity)
            else:
                require_chat_text_policy(identity, endpoint="task-results", model=receipt.get("model"))
            compatibility, result, result_ready = _text_projection(receipt)
            if private_bound:
                # Match the established trusted snapshot projection rather than
                # treating its bound conversation cursor as a public receipt.
                compatibility = TextTaskService._public(receipt)
        else:
            require_image_policy(identity, receipt.get("model"))
            compatibility, result, result_ready = _image_projection(receipt, request, private_bound=private_bound)

        return JSONResponse({
            "protocol": "task-result-v1",
            "kind": kind,
            "request_id": request_id,
            "status": _status(kind, receipt),
            "result_ready": result_ready,
            "result": {"content": result} if kind == "text" and result_ready else (
                {"data": result} if kind == "image" and result_ready else None
            ),
            # Temporary explicit compatibility bridge for existing consumers.
            # It is an existing public projection, never the raw receipt.
            "receipt": compatibility,
        }, headers={"Cache-Control": "private, no-store"})

    return router
