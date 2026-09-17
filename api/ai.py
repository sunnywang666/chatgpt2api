from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from api.external_images import client_sync_result, validate_external_input, is_external, synchronous_external_task
from api.image_inputs import parse_image_edit_request, read_image_sources
from api.support import require_identity, resolve_image_base_url
from services.content_filter import check_request, request_shape, request_text
from services.conversation_binding_service import (
    ConversationBindingError,
    conversation_binding_service,
)
from services.editable_file_task_service import editable_file_task_service
from services.text_task_service import text_task_service
from services.log_service import LoggedCall
from services.protocol import (
    anthropic_v1_messages,
    openai_v1_chat_complete,
    openai_v1_image_edit,
    openai_v1_image_generations,
    openai_v1_models,
    openai_v1_response,
    openai_search,
)


class ImageGenerationRequest(BaseModel):
    client_task_id: str | None = None
    prompt: str = Field(..., min_length=1)
    model: str = "gpt-image-2"
    n: int = Field(default=1, ge=1, le=4)
    size: str | None = None
    quality: str = "auto"
    response_format: str = "b64_json"
    history_disabled: bool = True
    stream: bool | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    prompt: str | None = None
    n: int | None = None
    stream: bool | None = None
    modalities: list[str] | None = None
    messages: list[dict[str, object]] | None = None


class ResponseCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    input: object | None = None
    tools: list[dict[str, object]] | None = None
    tool_choice: object | None = None
    stream: bool | None = None


class ConversationBindingTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "auto"
    image_model: str = "gpt-image-2"
    messages: list[dict[str, object]]
    thinking_effort: str = "standard"
    provider_binding_id: str | None = None
    provider_account_identity: str | None = None
    client_conversation_id: str
    client_request_id: str | None = Field(default=None, min_length=1, max_length=200)
    conversation_id: str | None = None
    parent_message_id: str | None = None


class TextRequestRecoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allow_unrecoverable_retry: bool = False


class ConversationArchiveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_binding_id: str = Field(min_length=1)
    provider_account_identity: str = Field(min_length=1)
    client_conversation_id: str = Field(min_length=1)
    conversation_id: str = Field(pattern=r"^[a-zA-Z0-9-]+$")
    parent_message_id: str = Field(min_length=1)


class AnthropicMessageRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[dict[str, object]] | None = None
    system: object | None = None
    stream: bool | None = None


class SearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class EditableFileTaskRequest(BaseModel):
    prompt: str = ""
    base64_images: list[str] = Field(default_factory=list)
    client_task_id: str | None = None


async def filter_or_log(call: LoggedCall, text: str) -> None:
    try:
        await run_in_threadpool(check_request, text)
    except HTTPException as exc:
        call.log("调用失败", status="failed", error=str(exc.detail))
        raise


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    async def list_models(request: Request, authorization: str | None = Header(default=None)):
        require_identity(authorization)
        try:
            result = await run_in_threadpool(openai_v1_models.list_models)
            if is_external(request):
                return {**result, "data": [item for item in result.get("data", []) if item.get("id") == "gpt-image-2"]}
            return result
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error": "model discovery unavailable" if is_external(request) else str(exc)}) from exc

    @router.post("/v1/images/generations")
    async def generate_images(
            body: ImageGenerationRequest,
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        validate_external_input(request, payload, synchronous=True)
        payload["base_url"] = resolve_image_base_url(request)
        call = LoggedCall(identity, "/v1/images/generations", body.model, "文生图", request_text=body.prompt)
        await filter_or_log(call, body.prompt)
        if is_external(request):
            return await synchronous_external_task(identity, payload, edit=False)
        return client_sync_result(await call.run(openai_v1_image_generations.handle, payload), request)

    @router.post("/v1/images/edits")
    async def edit_images(
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload, image_sources, mask_sources = await parse_image_edit_request(request)
        validate_external_input(request, payload, synchronous=True)
        prompt = str(payload["prompt"])
        model = str(payload["model"])
        call = LoggedCall(identity, "/v1/images/edits", model, "图生图", request_text=prompt)
        await filter_or_log(call, prompt)
        payload["images"] = await read_image_sources(image_sources)
        if mask_sources:
            payload["mask"] = await read_image_sources(mask_sources)
        payload["base_url"] = resolve_image_base_url(request)
        if is_external(request):
            return await synchronous_external_task(identity, payload, edit=True)
        return client_sync_result(await call.run(openai_v1_image_edit.handle, payload), request)

    @router.post("/v1/chat/completions")
    async def create_chat_completion(body: ChatCompletionRequest, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("prompt"), payload.get("messages"))
        call = LoggedCall(
            identity,
            "/v1/chat/completions",
            model,
            "文本生成",
            request_text=request_preview,
            request_shape=request_shape(payload.get("messages")),
        )
        await filter_or_log(call, request_preview)
        return await call.run(openai_v1_chat_complete.handle, payload)

    @router.post("/v1/responses")
    async def create_response(body: ResponseCreateRequest, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("input"), payload.get("instructions"))
        call = LoggedCall(
            identity,
            "/v1/responses",
            model,
            "Responses",
            request_text=request_preview,
            request_shape=request_shape(payload.get("input")),
        )
        await filter_or_log(call, request_preview)
        return await call.run(openai_v1_response.handle, payload)

    @router.post("/api/conversation-bindings/archive")
    async def archive_bound_conversation(body: ConversationArchiveRequest,
            authorization: str | None = Header(default=None)):
        require_identity(authorization)
        try:
            return await run_in_threadpool(conversation_binding_service.archive, body.model_dump())
        except ConversationBindingError as exc:
            raise HTTPException(status_code=409, detail={"code": exc.code}) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"code": "CONVERSATION_ARCHIVE_UNCONFIRMED"}) from exc

    @router.get("/api/conversation-bindings/text")
    async def read_bound_text(
            provider_binding_id: str, provider_account_identity: str,
            client_conversation_id: str, conversation_id: str, parent_message_id: str,
            authorization: str | None = Header(default=None),
    ):
        require_identity(authorization)
        try:
            return await run_in_threadpool(conversation_binding_service.read_text, {
                "provider_binding_id": provider_binding_id,
                "provider_account_identity": provider_account_identity,
                "client_conversation_id": client_conversation_id,
                "conversation_id": conversation_id,
                "parent_message_id": parent_message_id,
            })
        except ConversationBindingError as exc:
            raise HTTPException(status_code=409, detail={"code": exc.code}) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"code": "CONVERSATION_READ_UNAVAILABLE"}) from exc

    @router.get("/api/conversation-bindings/text-requests/{request_id}")
    async def read_bound_text_request(request_id: str, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        return await run_in_threadpool(text_task_service.read, str(identity.get("id") or "anonymous"), request_id)

    @router.post("/api/conversation-bindings/text-requests/{request_id}/recover")
    async def recover_bound_text_request(
        request_id: str,
        body: TextRequestRecoveryRequest,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        return await run_in_threadpool(
            text_task_service.recover,
            str(identity.get("id") or "anonymous"),
            request_id,
            body.allow_unrecoverable_retry,
        )

    @router.post("/api/conversation-bindings/text")
    async def continue_bound_text(
            body: ConversationBindingTextRequest,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        owner = str(identity.get("id") or "anonymous")
        payload = body.model_dump(mode="python")
        try:
            if body.client_request_id:
                # Reject a durable id/body conflict before the optional AI
                # review can make an external request. submit repeats the same
                # check transactionally after review to close the race.
                existing = await run_in_threadpool(
                    text_task_service.validate_submission, owner, payload,
                )
                if existing is not None:
                    if existing.get("status") == "not_started":
                        return await run_in_threadpool(
                            text_task_service.submit, owner, payload,
                        )
                    return existing
            request_preview = request_text(payload.get("messages"))
            await filter_or_log(
                LoggedCall(
                    identity,
                    "/api/conversation-bindings/text",
                    body.model,
                    "绑定会话文本",
                    request_text=request_preview,
                ),
                request_preview,
            )
            if body.client_request_id:
                return await run_in_threadpool(text_task_service.submit, owner, payload)
            return await run_in_threadpool(conversation_binding_service.complete_text, payload)
        except ConversationBindingError as exc:
            detail = {"code": exc.code, "error": str(exc)}
            for key in (
                "provider_binding_id",
                "provider_account_identity",
                "conversation_id",
                "parent_message_id",
            ):
                value = str(getattr(exc, key, "") or "").strip()
                if value:
                    detail[key] = value
            detail["binding_status"] = (
                "unknown" if exc.code == "CONVERSATION_OUTCOME_UNKNOWN" else "unavailable"
            )
            raise HTTPException(
                status_code=409,
                detail=detail,
            ) from exc

    @router.post("/v1/messages")
    async def create_message(
            body: AnthropicMessageRequest,
            authorization: str | None = Header(default=None),
            x_api_key: str | None = Header(default=None, alias="x-api-key"),
            anthropic_version: str | None = Header(default=None, alias="anthropic-version"),
    ):
        identity = require_identity(authorization or (f"Bearer {x_api_key}" if x_api_key else None))
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("system"), payload.get("messages"), payload.get("tools"))
        call = LoggedCall(identity, "/v1/messages", model, "Messages", request_text=request_preview)
        await filter_or_log(call, request_preview)
        return await call.run(anthropic_v1_messages.handle, payload, sse="anthropic")

    @router.post("/v1/search")
    async def search(body: SearchRequest, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        call = LoggedCall(identity, "/v1/search", openai_search.MODEL, "搜索", request_text=body.prompt)
        await filter_or_log(call, body.prompt)
        return await call.run(openai_search.handle, body.model_dump(mode="python"))

    @router.get("/v1/editable-file-tasks")
    async def list_editable_file_tasks(ids: str = "", authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        task_ids = [item.strip() for item in ids.split(",") if item.strip()]
        return await run_in_threadpool(editable_file_task_service.list_tasks, identity, task_ids)

    @router.get("/files/{file_path:path}")
    async def download_editable_file(file_path: str):
        try:
            path = await run_in_threadpool(editable_file_task_service.public_file_path, file_path)
        except Exception as exc:
            raise HTTPException(status_code=404, detail={"error": "file not found"}) from exc
        return FileResponse(path, filename=path.name)

    @router.post("/v1/ppt/generations")
    async def create_ppt_task(body: EditableFileTaskRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        await filter_or_log(LoggedCall(identity, "/v1/ppt/generations", "gpt-5-5-thinking", "PPT生成任务", request_text=body.prompt), body.prompt)
        return await run_in_threadpool(
            editable_file_task_service.submit_ppt,
            identity,
            client_task_id=body.client_task_id or "",
            prompt=body.prompt,
            base64_images=body.base64_images,
            base_url=resolve_image_base_url(request),
        )

    @router.post("/v1/psd/generations")
    async def create_psd_task(body: EditableFileTaskRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        await filter_or_log(LoggedCall(identity, "/v1/psd/generations", "gpt-5-5-thinking", "PSD生成任务", request_text=body.prompt), body.prompt)
        return await run_in_threadpool(
            editable_file_task_service.submit_psd,
            identity,
            client_task_id=body.client_task_id or "",
            prompt=body.prompt,
            base64_images=body.base64_images,
            base_url=resolve_image_base_url(request),
        )

    return router
