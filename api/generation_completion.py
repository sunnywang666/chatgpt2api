"""Explicit ordinary-client authorization, separate from original-result reads."""
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.support import require_identity
from api.key_policy import require_chat_text_policy, require_image_policy
from services.generation_completion import CompletionError, get_generation_completion_service
from services.work_lifecycle import WorkLifecycleError


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["recover", "complete", "rework"] = "recover"
    allow_unconfirmed_retry: bool = Field(default=False, strict=True)
    retry_not_sent_failure_at: float | None = Field(default=None, gt=0, allow_inf_nan=False, strict=True)
    selected_id: str | None = Field(default=None, min_length=1, max_length=200)
    results_saved: bool = Field(default=False, strict=True)
    reviewed: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def validate_action(self):
        if self.retry_not_sent_failure_at is not None and (self.action != "recover" or self.allow_unconfirmed_retry):
            raise ValueError("confirmed not-sent repair is a separate recovery action")
        if self.action == "complete":
            if not self.selected_id or not self.results_saved or not self.reviewed or self.allow_unconfirmed_retry:
                raise ValueError("completion requires the selected result, saved and reviewed")
        elif self.action == "rework":
            if not self.selected_id or self.results_saved or self.reviewed or self.allow_unconfirmed_retry:
                raise ValueError("rework requires only the selected result")
        elif self.selected_id or self.results_saved or self.reviewed:
            raise ValueError("recovery cannot acknowledge a result")
        return self


def create_router(kind):
    router = APIRouter()
    path = "/api/" + ("chat-requests" if kind == "text" else "image-tasks") + "/{request_id}/completion"

    def authorize(request, authorization, request_id, *, enforce_policy=False):
        identity = require_identity(authorization, request=request)
        if identity.get("role") != "user":
            raise HTTPException(403, detail={"code": "ORDINARY_KEY_REQUIRED"})
        service = get_generation_completion_service()
        try:
            with service.store.connect() as db:
                row = service._root(db, kind, str(identity["id"]), request_id)
            if enforce_policy and kind == "text":
                require_chat_text_policy(identity, endpoint="/api/chat-requests", model=row.get("model"))
            elif enforce_policy:
                require_image_policy(identity, row.get("model"))
        except WorkLifecycleError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None
        return identity, service

    @router.get(path)
    async def read_completion(request_id: str, request: Request, authorization: str | None = Header(default=None)):
        identity, service = authorize(request, authorization, request_id)
        try:
            return await run_in_threadpool(service.read, kind, identity, request_id)
        except WorkLifecycleError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None

    @router.post(path)
    async def update_completion(request_id: str, body: CompletionRequest, request: Request,
                                authorization: str | None = Header(default=None)):
        identity, service = authorize(request, authorization, request_id, enforce_policy=True)
        try:
            if body.action == "complete":
                return await run_in_threadpool(service.complete, kind, identity, request_id, body.selected_id)
            if body.action == "rework":
                return await run_in_threadpool(service.rework, kind, identity, request_id, body.selected_id)
            return await run_in_threadpool(service.start, kind, identity, request_id,
                                           allow_unconfirmed_retry=body.allow_unconfirmed_retry,
                                           retry_not_sent_failure_at=body.retry_not_sent_failure_at)
        except WorkLifecycleError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None
    return router
