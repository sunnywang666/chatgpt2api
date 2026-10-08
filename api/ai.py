from __future__ import annotations

from api.recovery_control import RecoveryControlRequest, update_recovery_control

from services.request_context import trusted_source

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from api.external_images import client_sync_result, validate_external_input, is_external, synchronous_external_task
from api.image_inputs import parse_image_edit_request, read_image_sources
from api.support import require_identity, resolve_image_base_url
from api.key_policy import require_image_policy, require_chat_text_policy
from utils.helper import UpstreamHTTPError, is_image_chat_request, has_response_image_generation_tool
from services.content_filter import check_request, request_shape, request_text
from services.account_request_pacing import AccountReadRetryBudgetInsufficient
from services.conversation_binding_service import (
    ConversationBindingError,
    conversation_binding_service,
)
from services.editable_file_task_service import editable_file_task_service
from services.text_task_service import TextTaskService, text_task_service
from services.log_service import LoggedCall
from services.public_chat_service import PublicChatContractError, project_public_models
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
    account_ref: str | None = Field(default=None, strict=True, pattern=r"^car_[A-Za-z0-9_-]{43}$")
    scheduling: object | None = None

    @field_validator("account_ref", mode="before")
    @classmethod
    def reject_null_account_ref(cls, value):
        if value is None:
            raise ValueError("account_ref must be omitted or an advertised account reference")
        return value


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    prompt: str | None = None
    n: int | None = None
    stream: bool | None = None
    modalities: list[str] | None = None
    messages: list[dict[str, object]] | None = None
    account_ref: str | None = Field(default=None, strict=True, pattern=r"^car_[A-Za-z0-9_-]{43}$")

    @field_validator("account_ref", mode="before")
    @classmethod
    def reject_null_account_ref(cls, value):
        if value is None:
            raise ValueError("account_ref must be omitted or an advertised account reference")
        return value


class ResponseCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    input: object | None = None
    tools: list[dict[str, object]] | None = None
    tool_choice: object | None = None
    stream: bool | None = None
    account_ref: str | None = Field(default=None, strict=True, pattern=r"^car_[A-Za-z0-9_-]{43}$")

    @field_validator("account_ref", mode="before")
    @classmethod
    def reject_null_account_ref(cls, value):
        if value is None:
            raise ValueError("account_ref must be omitted or an advertised account reference")
        return value


class ConversationBindingTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "auto"
    image_model: str = "gpt-image-2"
    messages: list[dict[str, object]] | None = None
    thinking_effort: str = "standard"
    provider_binding_id: str | None = None
    provider_account_identity: str | None = None
    client_conversation_id: str
    client_request_id: str | None = Field(default=None, min_length=1, max_length=200)
    supersedes_request_id: str | None = Field(default=None, min_length=1, max_length=200)
    derived_input: dict[str, object] | None = Field(default=None, exclude_if=lambda value: value is None)
    continue_after_no_final: bool | None = Field(default=None, strict=True, exclude_if=lambda value: value is None)
    conversation_id: str | None = None
    parent_message_id: str | None = None


    @model_validator(mode="after")
    def original_or_successor(self):
        no_final = "continue_after_no_final" in self.model_fields_set
        attributes_upgrade = self.derived_input == {"kind": "attributes_required_only_v4"}
        if attributes_upgrade and not no_final:
            raise ValueError("attributes upgrade requires explicit no-final continuation")
        if no_final and (self.continue_after_no_final is not True or not self.supersedes_request_id
                         or self.derived_input is not None and not attributes_upgrade):
            raise ValueError("no-final continuation requires an explicit original-request successor")
        if self.supersedes_request_id is None:
            if self.derived_input is not None:
                raise ValueError("derived input requires an original request")
            if self.messages is None:
                raise ValueError("messages are required")
        elif self.model_fields_set != (TextTaskService.SUPERSEDE_FIELDS | ({"derived_input"} if self.derived_input is not None else set()) | ({"continue_after_no_final"} if no_final else set())):
            raise ValueError("explicit successor requires only original binding and request identities")
        return self


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
    account_ref: str | None = Field(default=None, strict=True, pattern=r"^car_[A-Za-z0-9_-]{43}$")

    @field_validator("account_ref", mode="before")
    @classmethod
    def reject_null_account_ref(cls, value):
        if value is None:
            raise ValueError("account_ref must be omitted or an advertised account reference")
        return value


class SearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    account_ref: str | None = Field(default=None, strict=True, pattern=r"^car_[A-Za-z0-9_-]{43}$")
    scheduling: object | None = None

    @field_validator("account_ref", mode="before")
    @classmethod
    def reject_null_account_ref(cls, value):
        if value is None:
            raise ValueError("account_ref must be omitted or an advertised account reference")
        return value


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


def _compatibility_payload(body: BaseModel) -> dict:
    """Keep caller-only directives in the durable envelope, never upstream."""
    payload = body.model_dump(mode="python")
    # These names are private receipt fields. A permissive compatibility model
    # must never let a caller inject them directly.
    payload.pop("_scheduling", None)
    if payload.get("account_ref") is None:
        payload.pop("account_ref", None)
    if "scheduling" in body.model_fields_set:
        try:
            if payload.get("scheduling") is None:
                raise ValueError("SCHEDULING_INVALID: scheduling must be an object")
            from services.workflow_scheduling import normalize_scheduling
            scheduling = normalize_scheduling(payload.pop("scheduling", None))
        except ValueError as exc:
            if str(exc).startswith("SCHEDULING_INVALID"):
                raise HTTPException(400, detail={"code": "SCHEDULING_INVALID"}) from None
            raise
        if scheduling is not None:
            payload["_scheduling"] = scheduling
    else:
        payload.pop("scheduling", None)
    return payload


def _require_selected_compatibility_admission(request: Request, payload: dict) -> None:
    """Caller directives are meaningful only on a stable durable receipt."""
    selected = "account_ref" in payload
    scheduled = "_scheduling" in payload
    if not selected and not scheduled:
        return
    if not str(request.headers.get("x-client-request-id") or "").strip():
        raise HTTPException(400, detail={"code": "ACCOUNT_SELECTION_REQUEST_ID_REQUIRED" if selected else "SCHEDULING_REQUEST_ID_REQUIRED"})
    if text_task_service.admission is None:
        raise HTTPException(503, detail={"code": "ACCOUNT_SELECTION_REQUIRES_DURABLE_ADMISSION" if selected else "SCHEDULING_REQUIRES_DURABLE_ADMISSION"})


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    async def list_models(request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization, request=request)
        require_chat_text_policy(identity)
        catalog_pending = False

        def pending_response():
            return JSONResponse(status_code=503, headers={"Retry-After": "1"},
                                content={"detail": {"code": "MODEL_CATALOG_PENDING"}})

        try:
            result = await run_in_threadpool(openai_v1_models.list_models)
            catalog_pending = isinstance(result, dict) and result.get("model_catalog") == {"state": "partial"}
            if catalog_pending and result.get("data") == []:
                return pending_response()
            if is_external(request):
                result = await run_in_threadpool(project_public_models, result)
                from services.image_thread import PROTOCOL
                from services.image_task_service import image_task_service
                return {**result, "service_capabilities": {"image_thread": PROTOCOL if image_task_service.admission is not None else None}}
            # Preserve the private model-discovery contract. Public/company
            # callers receive this capability on the marked ingress only.
            return result
        except PublicChatContractError as exc:
            if catalog_pending and exc.code == "MODEL_DISCOVERY_UNAVAILABLE":
                return pending_response()
            raise HTTPException(status_code=502, detail={"code": exc.code}) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error": "model discovery unavailable" if is_external(request) else str(exc)}) from exc

    @router.post("/v1/images/generations")
    async def generate_images(
            body: ImageGenerationRequest,
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload = _compatibility_payload(body)
        raw = await request.json()
        # This compatibility endpoint has no thread contract. Reject rather
        # than silently ignore thread fields; keep ordinary legacy hashes.
        thread_fields = {k: raw[k] for k in ("image_thread_id", "edit_source_task_id", "edit_source_index") if k in raw}
        validate_external_input(request, {**payload, **thread_fields}, synchronous=True)
        _require_selected_compatibility_admission(request, payload)
        require_image_policy(identity, payload.get("model"))
        payload["base_url"] = resolve_image_base_url(request)
        call = LoggedCall(identity, "/v1/images/generations", body.model, "文生图", request_text=body.prompt)
        await filter_or_log(call, body.prompt)
        if is_external(request):
            return await synchronous_external_task({**identity, "_trusted_source": trusted_source(identity, request)}, payload, edit=False)
        if text_task_service.admission is not None:
            from services.durable_forward import respond
            return await respond(identity, payload, request, "openai_v1_image_generations", operation="image")
        return client_sync_result(await call.run(openai_v1_image_generations.handle, payload), request)

    @router.post("/v1/images/edits")
    async def edit_images(
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload, image_sources, mask_sources = await parse_image_edit_request(request)
        validate_external_input(request, payload, synchronous=True)
        _require_selected_compatibility_admission(request, payload)
        require_image_policy(identity, payload.get("model"))
        prompt = str(payload["prompt"])
        model = str(payload["model"])
        call = LoggedCall(identity, "/v1/images/edits", model, "图生图", request_text=prompt)
        await filter_or_log(call, prompt)
        payload["images"] = await read_image_sources(image_sources)
        if mask_sources:
            payload["mask"] = await read_image_sources(mask_sources)
        payload["base_url"] = resolve_image_base_url(request)
        if is_external(request):
            return await synchronous_external_task({**identity, "_trusted_source": trusted_source(identity, request)}, payload, edit=True)
        if text_task_service.admission is not None:
            from services.durable_forward import respond
            return await respond(identity, payload, request, "openai_v1_image_edit", operation="image")
        return client_sync_result(await call.run(openai_v1_image_edit.handle, payload), request)

    @router.post("/v1/chat/completions")
    async def create_chat_completion(body: ChatCompletionRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        payload = _compatibility_payload(body)
        _require_selected_compatibility_admission(request, payload)
        if is_image_chat_request(payload):
            require_image_policy(identity, payload.get("model"))
        else:
            require_chat_text_policy(identity, endpoint="/v1/chat/completions", model=payload.get("model"))
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
        if text_task_service.admission is not None:
            from services.durable_forward import respond
            return await respond(identity, payload, request, "openai_v1_chat_complete", operation="image" if is_image_chat_request(payload) else "text")
        return await call.run(openai_v1_chat_complete.handle, payload)

    @router.post("/v1/responses")
    async def create_response(body: ResponseCreateRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        payload = _compatibility_payload(body)
        _require_selected_compatibility_admission(request, payload)
        if has_response_image_generation_tool(payload):
            require_image_policy(identity, payload.get("model"))
        else:
            require_chat_text_policy(identity, endpoint="/v1/responses", model=payload.get("model"))
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
        if text_task_service.admission is not None:
            from services.durable_forward import respond
            return await respond(identity, payload, request, "openai_v1_response", operation="image" if has_response_image_generation_tool(payload) else "text")
        return await call.run(openai_v1_response.handle, payload)

    @router.post("/api/conversation-bindings/archive")
    async def archive_bound_conversation(body: ConversationArchiveRequest,
            authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        if identity.get("role") != "admin":
            raise HTTPException(404, detail={"code": "TASK_NOT_FOUND"})
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
        identity = require_identity(authorization)
        # Legacy direct cursors have no persisted caller ownership. Only the
        # existing trusted Content/admin path may use them; ordinary callers
        # read their owner-scoped durable text-request receipt below.
        if identity.get("role") != "admin":
            raise HTTPException(status_code=404, detail={"code": "TASK_NOT_FOUND"})
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
        except AccountReadRetryBudgetInsufficient as exc:
            # read_text preserves the original connection error after a failed
            # first attempt; this exception therefore means no GET was sent.
            raise HTTPException(status_code=503, detail={
                "code": "CONVERSATION_READ_DEFERRED", "reason": "read_budget_insufficient",
                "read_sent": False, "retryable": True,
            }) from exc
        except UpstreamHTTPError as exc:
            # Preserve the upstream recovery signal for trusted legacy callers;
            # neither the upstream body nor a locally invented cooldown belongs
            # in this response. A real upstream error is not a not-sent receipt.
            if exc.status_code == 429:
                headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
                raise HTTPException(status_code=429, detail={
                    "code": "CONVERSATION_READ_RATE_LIMITED",
                }, headers=headers) from exc
            if exc.status_code in {401, 403}:
                raise HTTPException(status_code=exc.status_code, detail={
                    "code": "CONVERSATION_READ_AUTH_REQUIRED",
                }) from exc
            raise HTTPException(status_code=503, detail={"code": "CONVERSATION_READ_UNAVAILABLE"}) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"code": "CONVERSATION_READ_UNAVAILABLE"}) from exc

    @router.get("/api/conversation-bindings/text-requests/{request_id}")
    async def read_bound_text_request(request_id: str, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        return await run_in_threadpool(text_task_service.read, str(identity.get("id") or "anonymous"), request_id)

    @router.post("/api/conversation-bindings/text-requests/{request_id}/recovery-control")
    async def control_bound_text_recovery(request_id: str, body: RecoveryControlRequest, response: Response,
                                          authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        response.headers["Cache-Control"] = "private, no-store"
        return await update_recovery_control(text_task_service, "text", str(identity.get("id") or "anonymous"),
                                             request_id, body)

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
            explicit_ended_recheck=True,
        )

    @router.post("/api/conversation-bindings/text-requests/{request_id}/resume-unsent-successor")
    async def resume_unsent_bound_successor(request_id: str, body: ConversationBindingTextRequest,
                                            authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        if identity.get("role") != "admin":
            raise HTTPException(501, detail={"code": "SERVICE_OPERATION_UNAVAILABLE"})
        require_chat_text_policy(identity)
        try:
            owner = str(identity.get("id") or "anonymous")
            payload = body.model_dump(mode="python", exclude_unset=True)
            if payload.get("derived_input") == {"kind": "attributes_required_only_v4"}:
                if payload.get("client_request_id") != request_id or text_task_service.admission is None:
                    raise ConversationBindingError("original successor envelope required", code="CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE")
                existing = await run_in_threadpool(text_task_service.validate_submission, owner, payload)
                if existing is None:
                    review_payload = await run_in_threadpool(text_task_service.submission_input, owner, payload)
                    preview = request_text(review_payload.get("messages"))
                    await filter_or_log(LoggedCall(identity,
                        "/api/conversation-bindings/text-requests/resume-unsent-successor",
                        str(review_payload.get("model") or "auto"), "绑定会话文本", request_text=preview), preview)
            return await run_in_threadpool(text_task_service.resume_unsent_successor,
                owner, request_id, payload)
        except ConversationBindingError as exc:
            raise HTTPException(409, detail={"code": exc.code, "error": str(exc)}) from exc

    @router.post("/api/conversation-bindings/text")
    async def continue_bound_text(
            body: ConversationBindingTextRequest,
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        if identity.get("role") != "admin":
            raise HTTPException(501, detail={"code": "SERVICE_OPERATION_UNAVAILABLE",
                "error": "ordinary-client bound conversation submission is not supported; use the supported Chat API"})
        require_chat_text_policy(identity)
        owner = str(identity.get("id") or "anonymous")
        payload = body.model_dump(mode="python", exclude_unset=body.supersedes_request_id is not None)
        # Preserve hashes of every pre-existing ordinary submission.
        if body.supersedes_request_id is None:
            payload.pop("supersedes_request_id", None)
        try:
            if body.supersedes_request_id is not None and not body.client_request_id:
                raise ConversationBindingError("explicit successor request identity is required",
                                               code="CHAT_SUPERSEDE_INVALID")
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
                            text_task_service.submit, owner, payload, source=trusted_source(identity, request),
                        )
                    return existing
            review_payload = (await run_in_threadpool(text_task_service.submission_input, owner, payload)
                              if body.supersedes_request_id is not None else payload)
            request_preview = request_text(review_payload.get("messages"))
            await filter_or_log(
                LoggedCall(
                    identity,
                    "/api/conversation-bindings/text",
                    str(review_payload.get("model") or "auto"),
                    "绑定会话文本",
                    request_text=request_preview,
                ),
                request_preview,
            )
            if body.client_request_id:
                return await run_in_threadpool(text_task_service.submit, owner, payload, source=trusted_source(identity, request))
            if text_task_service.admission is not None:
                import uuid
                payload["client_request_id"] = request.headers.get("x-client-request-id") or uuid.uuid4().hex
                return await run_in_threadpool(text_task_service.submit, owner, payload, source=trusted_source(identity, request))
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
            request: Request,
            authorization: str | None = Header(default=None),
            x_api_key: str | None = Header(default=None, alias="x-api-key"),
            anthropic_version: str | None = Header(default=None, alias="anthropic-version"),
    ):
        identity = require_identity(authorization or (f"Bearer {x_api_key}" if x_api_key else None))
        require_chat_text_policy(identity)
        payload = _compatibility_payload(body)
        _require_selected_compatibility_admission(request, payload)
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("system"), payload.get("messages"), payload.get("tools"))
        call = LoggedCall(identity, "/v1/messages", model, "Messages", request_text=request_preview)
        await filter_or_log(call, request_preview)
        if text_task_service.admission is not None:
            from services.durable_forward import respond
            return await respond(identity, payload, request, "anthropic_v1_messages", operation="text")
        return await call.run(anthropic_v1_messages.handle, payload, sse="anthropic")

    @router.post("/v1/search")
    async def search(body: SearchRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        require_chat_text_policy(identity)
        payload = _compatibility_payload(body)
        _require_selected_compatibility_admission(request, payload)
        call = LoggedCall(identity, "/v1/search", openai_search.MODEL, "搜索", request_text=body.prompt)
        await filter_or_log(call, body.prompt)
        if text_task_service.admission is not None:
            from services.durable_forward import respond
            return await respond(identity, {**payload, "model": openai_search.MODEL}, request, "openai_search")
        return await call.run(openai_search.handle, payload)

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

    async def submit_file_task(handler, identity, request, body):
        try:
            return await run_in_threadpool(
                handler, {**identity, "_trusted_source": trusted_source(identity, request)},
                client_task_id=body.client_task_id or "", prompt=body.prompt,
                base64_images=body.base64_images, base_url=resolve_image_base_url(request),
            )
        except ConversationBindingError as exc:
            raise HTTPException(409, detail={"code": exc.code, "task_id": body.client_task_id}) from None

    @router.post("/v1/ppt/generations")
    async def create_ppt_task(body: EditableFileTaskRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        require_chat_text_policy(identity)
        await filter_or_log(LoggedCall(identity, "/v1/ppt/generations", "gpt-5-5-thinking", "PPT生成任务", request_text=body.prompt), body.prompt)
        return await submit_file_task(editable_file_task_service.submit_ppt, identity, request, body)

    @router.post("/v1/psd/generations")
    async def create_psd_task(body: EditableFileTaskRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        require_chat_text_policy(identity)
        await filter_or_log(LoggedCall(identity, "/v1/psd/generations", "gpt-5-5-thinking", "PSD生成任务", request_text=body.prompt), body.prompt)
        return await submit_file_task(editable_file_task_service.submit_psd, identity, request, body)

    return router
