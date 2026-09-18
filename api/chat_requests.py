from __future__ import annotations

import hashlib
import re

from fastapi import APIRouter, Header, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.image_inputs import normalize_inline_chat_messages
from api.key_policy import require_chat_text_policy
from api.support import require_identity
from services.content_filter import check_request, request_shape, request_text
from services.conversation_binding_service import ConversationBindingError
from services.log_service import LoggedCall
from services.public_chat_service import (
    PublicChatContractError,
    project_public_chat_receipt,
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

    @field_validator("client_request_id")
    @classmethod
    def addressable_request_id(cls, value: str) -> str:
        if value in {".", ".."}:
            raise ValueError("client_request_id must not be a dot segment")
        return value


class OriginalRecoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _ordinary_identity(authorization: str | None) -> dict[str, object]:
    identity = require_identity(authorization)
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
    return {
        "client_request_id": body.client_request_id,
        "model": body.model.strip(),
        "messages": messages,
        "client_conversation_id": _server_conversation_id(owner, body.client_request_id),
        "_text_only_binding": True,
        "_public_route": "chat",
    }


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
        authorization: str | None = Header(default=None),
    ):
        identity = _ordinary_identity(authorization)
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
            receipt = await run_in_threadpool(text_task_service.submit, owner, payload)
        except ConversationBindingError as exc:
            call.log("提交失败", status="failed", error=exc.code)
            raise HTTPException(
                409,
                detail={"code": "CHAT_REQUEST_CONFLICT", "request_id": body.client_request_id},
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
        authorization: str | None = Header(default=None),
    ):
        identity = _ordinary_identity(authorization)
        request_id = _validated_request_id(request_id)
        receipt = await run_in_threadpool(text_task_service.read, _owner(identity), request_id)
        response.headers["Cache-Control"] = "private, no-store"
        return _projection(receipt, request_id)

    @router.post("/api/chat-requests/{request_id}/recover")
    async def recover_chat_request(
        request_id: str,
        response: Response,
        body: OriginalRecoveryRequest | None = None,
        authorization: str | None = Header(default=None),
    ):
        del body
        identity = _ordinary_identity(authorization)
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
