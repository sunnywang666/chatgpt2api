from __future__ import annotations

from services.request_context import trusted_source

import hashlib
import re
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from api.image_inputs import normalize_inline_chat_messages
from api.key_policy import require_chat_text_policy
from api.support import require_identity
from services.content_filter import check_request, request_shape, request_text
from services.conversation_binding_service import ConversationBindingError
from services.log_service import LoggedCall
from services.public_chat_service import (
    PublicChatContractError,
    project_public_chat_receipt,
    require_public_reasoning_effort,
    require_public_text_model,
)
from services.text_task_service import text_task_service


class PublicChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_request_id: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )
    model: str = Field(min_length=1, max_length=200)
    messages: list[dict[str, object]] = Field(min_length=1, max_length=100)
    reasoning_effort: Literal["high"] | None = None
    # An application-owned work session, NOT an upstream conversation cursor.
    client_conversation_id: str | None = Field(default=None, min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.:-]+$")
    previous_request_id: str | None = Field(default=None, min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.:-]+$")

    @field_validator("client_conversation_id", "previous_request_id", mode="before")
    @classmethod
    def valid_continuation_reference(cls, value):
        if not isinstance(value, str) or value in {".", ".."}:
            raise ValueError("continuation references must be omitted or valid identifiers")
        return value

    @model_validator(mode="after")
    def continuation_contract(self):
        if self.previous_request_id and (not self.client_conversation_id or self.previous_request_id == self.client_request_id):
            raise ValueError("a different predecessor and a work session are required")
        if self.client_conversation_id and self.messages[-1].get("role") != "user":
            raise ValueError("a sequential turn must end with its new user input")
        return self

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def reject_explicit_null_reasoning_effort(cls, value: object) -> object:
        if value is None:
            raise ValueError("reasoning_effort must be omitted or 'high'")
        return value

    @field_validator("client_request_id")
    @classmethod
    def addressable_request_id(cls, value: str) -> str:
        if value in {".", ".."}:
            raise ValueError("client_request_id must not be a dot segment")
        return value


class OriginalRecoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _ordinary_identity(authorization: str | None, request: Request) -> dict[str, object]:
    identity = require_identity(authorization, request=request)
    if identity.get("role") != "user":
        raise HTTPException(403, detail={"code": "ORDINARY_KEY_REQUIRED"})
    return identity


def _owner(identity: dict[str, object]) -> str:
    value = str(identity.get("id") or "").strip()
    if not value:
        raise HTTPException(401, detail={"code": "KEY_IDENTITY_INVALID"})
    return value


def _server_conversation_id(owner: str, request_id: str) -> str:
    digest = hashlib.sha256(f"{owner}\0{request_id}".encode()).hexdigest()
    return f"public-chat-{digest}"


def _payload(owner: str, body: PublicChatRequest, messages: list[dict]) -> dict:
    payload = {
        "client_request_id": body.client_request_id,
        "model": body.model.strip(),
        "messages": messages,
        "client_conversation_id": _server_conversation_id(owner, body.client_request_id),
        "_text_only_binding": True,
        "_public_route": "chat",
    }
    if body.client_conversation_id is not None:
        digest = hashlib.sha256(f"{owner}\0session\0{body.client_conversation_id}".encode()).hexdigest()
        payload["client_conversation_id"] = f"public-session-{digest}"
        payload["_public_session_ref"] = body.client_conversation_id
        if body.previous_request_id is not None:
            payload["_previous_request_id"] = body.previous_request_id
    # Absence retains the exact legacy identity; explicit high is persisted
    # through the existing internal field and participates in conflict checks.
    if body.reasoning_effort is not None:
        payload["thinking_effort"] = body.reasoning_effort
    return payload


def _not_found(request_id: str) -> HTTPException:
    return HTTPException(404, detail={"code": "CHAT_REQUEST_NOT_FOUND", "request_id": request_id})


def _validated_request_id(request_id: str) -> str:
    value = str(request_id or "").strip()
    if value in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", value):
        raise HTTPException(400, detail={"code": "CHAT_REQUEST_ID_INVALID"})
    return value


def _projection(receipt: dict, request_id: str) -> dict:
    if receipt.get("status") == "not_found":
        raise _not_found(request_id)
    try:
        return project_public_chat_receipt(receipt)
    except PublicChatContractError as exc:
        raise HTTPException(503, detail={"code": exc.code}) from None


def create_router() -> APIRouter:
    router = APIRouter()

    @router.post("/api/chat-requests")
    async def create_chat_request(
        body: PublicChatRequest,
        response: Response,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = _ordinary_identity(authorization, request)
        owner = _owner(identity)
        messages = await run_in_threadpool(normalize_inline_chat_messages, body.messages)
        payload = _payload(owner, body, messages)
        try:
            # Check immutable identity before current model/capacity admission.
            # A prior accepted request remains readable when the catalog later
            # changes, and UNKNOWN is never turned into another generation.
            existing = await run_in_threadpool(
                text_task_service.validate_submission,
                owner,
                payload,
            )
        except ConversationBindingError as exc:
            raise HTTPException(
                409,
                detail={"code": "CHAT_REQUEST_CONFLICT", "request_id": body.client_request_id},
            ) from exc
        safe_capacity_retry = bool(
            existing
            and existing.get("route") == "chat"
            and existing.get("status") == "failed"
            and existing.get("error_code") == "TEXT_TASK_CAPACITY_EXCEEDED"
            and existing.get("upstream_outcome") == "not_sent"
        )
        if existing is not None and not safe_capacity_retry:
            # Public callers can never prove that a different process died
            # before dispatch. Even a legacy ``not_started`` classification is
            # therefore query-only here; accepting it again could race an
            # original process that has not claimed its queued receipt yet.
            result = _projection(existing, body.client_request_id)
            response.status_code = 200 if result["status"] in {"succeeded", "failed"} else 202
            response.headers["Cache-Control"] = "private, no-store"
            return result

        require_chat_text_policy(identity, endpoint="/api/chat-requests", model=body.model)
        try:
            require_public_text_model(body.model)
        except PublicChatContractError as exc:
            raise HTTPException(400, detail={"code": exc.code, "error": str(exc)}) from None
        except Exception:
            raise HTTPException(503, detail={"code": "MODEL_DISCOVERY_UNAVAILABLE"}) from None

        try:
            require_public_reasoning_effort(body.model, body.reasoning_effort)
        except PublicChatContractError as exc:
            raise HTTPException(400, detail={"code": exc.code, "error": str(exc)}) from None

        preview = request_text(messages)
        call = LoggedCall(
            identity,
            "/api/chat-requests",
            body.model,
            "公共 Chat 请求",
            request_text=preview,
            request_shape=request_shape(messages),
        )
        try:
            await run_in_threadpool(check_request, preview)
            receipt = await run_in_threadpool(text_task_service.submit, owner, payload, source=trusted_source(identity, request))
        except ConversationBindingError as exc:
            call.log("提交失败", status="failed", error=exc.code)
            raise HTTPException(
                409,
                detail={"code": exc.code if exc.code.startswith("CHAT_") else "CHAT_REQUEST_CONFLICT",
                        "request_id": body.client_request_id,
                        **({"previous_request_id": body.previous_request_id} if body.previous_request_id else {})},
            ) from exc
        except HTTPException as exc:
            call.log("提交失败", status="failed", error=str(exc.detail))
            raise
        result = _projection(receipt, body.client_request_id)
        if result.get("error_code") == "TEXT_TASK_CAPACITY_EXCEEDED":
            call.log("容量拒绝", result, status=result["status"])
            response.status_code = 429
        else:
            call.log("已提交", result, status=result["status"])
            response.status_code = 200 if result["status"] in {"succeeded", "failed"} else 202
        response.headers["Cache-Control"] = "private, no-store"
        return result

    @router.get("/api/chat-requests/{request_id}")
    async def read_chat_request(
        request_id: str,
        response: Response,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = _ordinary_identity(authorization, request)
        request_id = _validated_request_id(request_id)
        receipt = await run_in_threadpool(text_task_service.read, _owner(identity), request_id)
        response.headers["Cache-Control"] = "private, no-store"
        return _projection(receipt, request_id)

    @router.post("/api/chat-requests/{request_id}/recover")
    async def recover_chat_request(
        request_id: str,
        response: Response,
        request: Request,
        body: OriginalRecoveryRequest | None = None,
        authorization: str | None = Header(default=None),
    ):
        del body
        identity = _ordinary_identity(authorization, request)
        request_id = _validated_request_id(request_id)
        receipt = await run_in_threadpool(
            text_task_service.recover,
            _owner(identity),
            request_id,
            False,
        )
        response.headers["Cache-Control"] = "private, no-store"
        return _projection(receipt, request_id)

    return router
