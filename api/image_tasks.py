from __future__ import annotations

from services.request_context import trusted_source
from services.image_thread import ImageThreadError

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from api.external_images import client_task, task_image_bytes, validate_external_input, is_external
from pydantic import BaseModel, ConfigDict, Field

from api.image_inputs import parse_image_edit_request, read_image_sources
from api.support import require_identity, resolve_image_base_url
from api.key_policy import require_image_policy
from services.content_filter import check_request
from services.image_task_service import image_task_service
from services.log_service import LoggedCall


class ImageGenerationTaskRequest(BaseModel):
    client_task_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    model: str = "gpt-image-2"
    n: int = Field(default=1, ge=1, le=4)
    size: str | None = None
    quality: str = "auto"
    provider_binding_id: str = ""
    provider_account_identity: str = ""
    client_conversation_id: str = ""
    conversation_id: str = ""
    parent_message_id: str = ""
    retain_conversation: bool = False
    upstream_model: str = ""
    image_thread_id: str = ""
    edit_source_task_id: str = ""
    edit_source_index: int = Field(default=0, ge=0, le=15)


class ResumePollRequest(BaseModel):
    extra_timeout_secs: float = Field(default=30.0, ge=5.0, le=120.0)
    allow_unrecoverable_retry: bool = False


class ArchiveImageThreadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdoptLatestConversationImageRequest(BaseModel):
    provider_binding_id: str = Field(..., min_length=1, max_length=300)
    provider_account_identity: str = Field(..., min_length=1, max_length=300)
    client_conversation_id: str = Field(..., min_length=1, max_length=300)
    conversation_id: str = Field(..., min_length=1, max_length=300)
    source_request_message_id: str = Field(default="", max_length=300)
    source_image_message_id: str = Field(default="", max_length=300)


class ManualRecoveryRequest(BaseModel):
    """Workbench's original-task authority; the Provider owns the conversation ID."""

    model_config = ConfigDict(extra="forbid")

    provider_binding_id: str = Field(..., min_length=1, max_length=300)
    provider_account_identity: str = Field(..., min_length=1, max_length=300)
    client_conversation_id: str = Field(..., min_length=1, max_length=300)


def _parse_task_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


async def filter_or_log(call: LoggedCall, text: str) -> None:
    try:
        await run_in_threadpool(check_request, text)
    except HTTPException as exc:
        call.log("调用失败", status="failed", error=str(exc.detail))
        raise


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/image-tasks")
    async def list_image_tasks(
        request: Request,
        ids: str = Query(default=""),
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization, request=request)
        task_ids = _parse_task_ids(ids)
        if getattr(request.state, "company_identity", None) and not task_ids:
            raise HTTPException(400, detail={"code": "ORIGINAL_TASK_IDS_REQUIRED"})
        result = await run_in_threadpool(image_task_service.list_tasks, identity, task_ids)
        if getattr(request.state, "company_identity", None) and result.get("missing_ids"):
            raise HTTPException(404, detail={"code": "IMAGE_TASK_NOT_FOUND"})
        return {**result, "items": [client_task(item, request) for item in result["items"]]}

    @router.post("/api/image-tasks/generations")
    async def create_generation_task(
        body: ImageGenerationTaskRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization, request=request)
        validate_external_input(request, body.model_dump())
        require_image_policy(identity, body.model)
        await filter_or_log(LoggedCall(identity, "/api/image-tasks/generations", body.model, "文生图任务", request_text=body.prompt), body.prompt)
        try:
            result = await run_in_threadpool(
                image_task_service.submit_generation,
                {**identity, "external_image_client": is_external(request), "_trusted_source": trusted_source(identity, request)},
                client_task_id=body.client_task_id,
                prompt=body.prompt,
                model=body.model,
                size=body.size,
                quality=body.quality,
                base_url=resolve_image_base_url(request),
                provider_binding_id=body.provider_binding_id,
                provider_account_identity=body.provider_account_identity,
                client_conversation_id=body.client_conversation_id,
                conversation_id=body.conversation_id,
                parent_message_id=body.parent_message_id,
                retain_conversation=body.retain_conversation,
                upstream_model=body.upstream_model,
                image_thread_id=body.image_thread_id,
                edit_source_task_id=body.edit_source_task_id,
                edit_source_index=body.edit_source_index,
            )
            return client_task(result, request)
        except ImageThreadError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None
        except ValueError as exc:
            status = 409 if "different immutable request" in str(exc) else 400
            raise HTTPException(status_code=status, detail={"error": str(exc)}) from exc

    @router.post("/api/image-tasks/edits")
    async def create_edit_task(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization, request=request)
        payload, image_sources, mask_sources = await parse_image_edit_request(request)
        validate_external_input(request, payload)
        require_image_policy(identity, payload.get("model"))
        client_task_id = str(payload.get("client_task_id") or "").strip()
        if not client_task_id:
            raise HTTPException(status_code=400, detail={"error": "client_task_id is required"})
        prompt = str(payload["prompt"])
        model = str(payload["model"])
        await filter_or_log(LoggedCall(identity, "/api/image-tasks/edits", model, "图生图任务", request_text=prompt), prompt)
        images = await read_image_sources(image_sources)
        masks = await read_image_sources(mask_sources) if mask_sources else None
        try:
            result = await run_in_threadpool(
                image_task_service.submit_edit,
                {**identity, "external_image_client": is_external(request), "_trusted_source": trusted_source(identity, request)},
                client_task_id=client_task_id,
                prompt=prompt,
                model=model,
                size=payload["size"],
                quality=payload["quality"],
                base_url=resolve_image_base_url(request),
                images=images,
                masks=masks,
                provider_binding_id=str(payload.get("provider_binding_id") or ""),
                provider_account_identity=str(payload.get("provider_account_identity") or ""),
                client_conversation_id=str(payload.get("client_conversation_id") or ""),
                conversation_id=str(payload.get("conversation_id") or ""),
                parent_message_id=str(payload.get("parent_message_id") or ""),
                retain_conversation=bool(payload.get("retain_conversation")),
                upstream_model=str(payload.get("upstream_model") or ""),
                image_thread_id=payload.get("image_thread_id", ""),
                edit_source_task_id=payload.get("edit_source_task_id", ""),
                edit_source_index=payload.get("edit_source_index", 0),
            )
            return client_task(result, request)
        except ImageThreadError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None
        except ValueError as exc:
            status = 409 if "different immutable request" in str(exc) else 400
            raise HTTPException(status_code=status, detail={"error": str(exc)}) from exc

    @router.post("/api/image-tasks/{task_id}/resume-poll")
    async def resume_image_poll(
        task_id: str,
        body: ResumePollRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization, request=request)
        if getattr(request.state, "company_identity", None) and body.allow_unrecoverable_retry:
            raise HTTPException(400, detail={"code": "ORIGINAL_TASK_RECOVERY_ONLY"})
        try:
            result = await run_in_threadpool(
                image_task_service.resume_poll,
                identity,
                task_id,
                body.extra_timeout_secs,
                resolve_image_base_url(request),
                body.allow_unrecoverable_retry,
            )
            return client_task(result, request)
        except ImageThreadError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None
        except ValueError as exc:
            if getattr(request.state, "company_identity", None) and str(exc) == "task not found":
                raise HTTPException(404, detail={"code": "IMAGE_TASK_NOT_FOUND"}) from None
            status = 409 if "different immutable request" in str(exc) else 400
            raise HTTPException(status_code=status, detail={"error": str(exc)}) from exc

    @router.post("/api/image-tasks/{task_id}/archive-thread")
    async def archive_image_thread(task_id: str, body: ArchiveImageThreadRequest, request: Request,
                                   authorization: str | None = Header(default=None)):
        identity = require_identity(authorization, request=request)
        try:
            return await run_in_threadpool(image_task_service.archive_thread, identity, task_id)
        except ImageThreadError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None
        except Exception as exc:
            raise HTTPException(503, detail={"code": "IMAGE_THREAD_ARCHIVE_UNCONFIRMED"}) from exc

    @router.post("/api/image-tasks/{task_id}/restore-thread")
    async def restore_image_thread(task_id: str, body: ArchiveImageThreadRequest, request: Request,
                                   authorization: str | None = Header(default=None)):
        identity = require_identity(authorization, request=request)
        try:
            return await run_in_threadpool(image_task_service.restore_thread, identity, task_id)
        except ImageThreadError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None
        except Exception as exc:
            raise HTTPException(503, detail={"code": "IMAGE_THREAD_RESTORE_UNCONFIRMED"}) from exc

    @router.post("/api/image-tasks/{task_id}/adopt-latest-conversation-image")
    async def adopt_latest_conversation_image(
        task_id: str,
        body: AdoptLatestConversationImageRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization, request=request)
        try:
            result = await run_in_threadpool(
                image_task_service.adopt_latest_conversation_image,
                identity,
                task_id,
                provider_binding_id=body.provider_binding_id,
                provider_account_identity=body.provider_account_identity,
                client_conversation_id=body.client_conversation_id,
                conversation_id=body.conversation_id,
                source_request_message_id=body.source_request_message_id,
                source_image_message_id=body.source_image_message_id,
                base_url=resolve_image_base_url(request),
            )
            return client_task(result, request)
        except ImageThreadError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"error": str(exc)}) from exc

    @router.post("/api/image-tasks/{task_id}/manual-recover")
    async def recover_manual_image(
        task_id: str,
        body: ManualRecoveryRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization, request=request)
        try:
            result = await run_in_threadpool(
                image_task_service.recover_manual,
                identity,
                task_id,
                provider_binding_id=body.provider_binding_id,
                provider_account_identity=body.provider_account_identity,
                client_conversation_id=body.client_conversation_id,
                base_url=resolve_image_base_url(request),
            )
            return client_task(result, request)
        except ImageThreadError as exc:
            if exc.code == "RECOVERY_RATE_LIMITED":
                retry_after = getattr(exc, "retry_after", None)
                raise HTTPException(
                    429,
                    detail={
                        "code": exc.code,
                        "rate_limit": {
                            "layer": "chatgpt_upstream",
                            "phase": "manual_recovery_read",
                            "retry_after_seconds": retry_after,
                        },
                    },
                    headers={"Retry-After": str(retry_after)} if retry_after is not None else None,
                ) from None
            raise HTTPException(exc.status, detail={"code": exc.code}) from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"error": str(exc)}) from exc

    @router.get("/api/image-tasks/{task_id}/images/{index}")
    async def download_task_image(task_id: str, index: int, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization, request=request)
        result = await run_in_threadpool(image_task_service.list_tasks, identity, [task_id])
        if not result["items"]:
            raise HTTPException(status_code=404, detail={"error": "image not found"})
        content = await run_in_threadpool(task_image_bytes, result["items"][0], index)
        return Response(content, media_type="image/png", headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})

    return router
