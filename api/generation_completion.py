"""Explicit ordinary-client authorization, separate from original-result reads."""
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.support import require_identity
from api.external_images import is_external
from api.key_policy import require_chat_text_policy, require_codex_endpoint, require_image_policy
from api.task_results import _status as task_result_status
from services.conversation_binding_service import ConversationBindingError
from services.generation_completion import CompletionError, get_generation_completion_service
from services.work_lifecycle import WorkLifecycleError


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["recover", "retry", "complete", "rework"] = "recover"
    allow_unconfirmed_retry: bool = Field(default=False, strict=True)
    retry_not_sent_failure_at: float | None = Field(default=None, gt=0, allow_inf_nan=False, strict=True)
    selected_id: str | None = Field(default=None, min_length=1, max_length=200)
    results_saved: bool = Field(default=False, strict=True)
    reviewed: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def validate_action(self):
        if self.retry_not_sent_failure_at is not None and (self.action != "recover" or self.allow_unconfirmed_retry):
            raise ValueError("confirmed not-sent repair is a separate recovery action")
        if self.action == "retry":
            if (self.selected_id or self.results_saved or self.reviewed or self.allow_unconfirmed_retry
                    or self.retry_not_sent_failure_at is not None):
                raise ValueError("bound retry has no caller-supplied retry authority")
        elif self.action == "complete":
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

    def authorize(request, authorization, request_id, *, enforce_policy=False,
                  admin_bound_action: str | None = None):
        identity = require_identity(authorization, request=request)
        if (identity.get("role") != "user"
                and not (admin_bound_action in {"recover", "retry"} and identity.get("role") == "admin")):
            raise HTTPException(403, detail={"code": "ORDINARY_KEY_REQUIRED"})
        service = get_generation_completion_service()
        try:
            with service.store.connect() as db:
                row = service._root(
                    db, kind, str(identity["id"]), request_id,
                    read_only_original=(admin_bound_action == "recover" and identity.get("role") == "admin"),
                    trusted_bound_retry=(admin_bound_action == "retry" and identity.get("role") == "admin"),
                )
            if enforce_policy and kind == "text":
                if row.get("_route") == "codex":
                    require_codex_endpoint(identity)
                else:
                    require_chat_text_policy(identity, endpoint="/api/chat-requests", model=row.get("model"))
            elif enforce_policy:
                require_image_policy(identity, row.get("model"))
        except WorkLifecycleError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None
        return identity, service, row

    @router.get(path)
    async def read_completion(request_id: str, request: Request, authorization: str | None = Header(default=None),
                              action: Literal["retry"] | None = Query(default=None)):
        internal_retry = action == "retry"
        identity, service, _row = authorize(request, authorization, request_id,
                                            enforce_policy=internal_retry,
                                            admin_bound_action="retry" if internal_retry else None)
        if internal_retry:
            if (getattr(request.state, "company_identity", None) is not None
                    or identity.get("role") != "admin" or is_external(request)):
                raise HTTPException(403, detail={"code": "ORDINARY_KEY_REQUIRED"})
            try:
                attempt = await run_in_threadpool(service.read_bound_attempt, kind, identity, request_id)
                attempt_id = attempt.get("request_id") or attempt.get("id")
                if not isinstance(attempt_id, str) or not attempt_id:
                    raise HTTPException(409, detail={"code": "TASK_ATTEMPT_CONFLICT"})
                return JSONResponse({
                    "protocol": "task-attempt-v1", "kind": kind, "original_id": request_id,
                    "attempt_id": attempt_id, "status": task_result_status(kind, attempt),
                    "result_ready": bool(
                        attempt.get("status") == ("succeeded" if kind == "text" else "success")
                        and (isinstance(attempt.get("content"), str) and attempt["content"].strip()
                             if kind == "text" else attempt.get("data"))
                    ),
                    "existing": True, "created": False,
                }, headers={"Cache-Control": "private, no-store"})
            except WorkLifecycleError as exc:
                raise HTTPException(exc.status, detail={"code": exc.code}) from None
        try:
            return await run_in_threadpool(service.read, kind, identity, request_id)
        except WorkLifecycleError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None

    @router.post(path)
    async def update_completion(request_id: str, body: CompletionRequest, request: Request,
                                authorization: str | None = Header(default=None)):
        requested_admin_action = body.action if body.action in {"recover", "retry"} else None
        identity, service, before = authorize(request, authorization, request_id, enforce_policy=True,
                                               admin_bound_action=requested_admin_action)
        company_original_only = getattr(request.state, "company_identity", None) is not None
        admin_original_only = identity.get("role") == "admin"
        if body.action == "retry" and not admin_original_only:
            raise HTTPException(403, detail={"code": "ORDINARY_KEY_REQUIRED"})
        if admin_original_only:
            allowed = {"recover": {"action", "allow_unconfirmed_retry"}, "retry": {"action"}}
            if body.action not in allowed or body.allow_unconfirmed_retry or body.model_fields_set - allowed[body.action]:
                # Recovery remains original-only. Retry is a named command
                # with no caller-provided retry authority or input.
                raise HTTPException(403, detail={"code": "ORDINARY_KEY_REQUIRED"})
        if company_original_only:
            # Company image clients already own work/archive/restore actions.
            # Acknowledge their selected saved result through the same contract,
            # without expanding original-only recovery or its retry authority.
            fields = {
                "recover": {"action", "allow_unconfirmed_retry", "retry_not_sent_failure_at"},
                "complete": {"action", "selected_id", "results_saved", "reviewed"},
                "rework": {"action", "selected_id"},
                "retry": set(),
            }[body.action]
            if kind != "image" or body.allow_unconfirmed_retry or body.model_fields_set - fields:
                raise HTTPException(403, detail={"code": "COMPANY_ORIGINAL_RECOVERY_REQUIRED"})
        try:
            if body.action == "complete":
                return await run_in_threadpool(service.complete, kind, identity, request_id, body.selected_id)
            if body.action == "rework":
                return await run_in_threadpool(service.rework, kind, identity, request_id, body.selected_id)
            if admin_original_only and body.action == "recover" and kind == "text":
                # Bound text recovery already has a durable exact-original
                # reader. Keep this command on that path rather than creating
                # a completion state or making a recovery GET authorize send.
                return await run_in_threadpool(service.text.recover, str(identity["id"]), request_id,
                                               explicit_ended_recheck=True)
            if admin_original_only and body.action == "retry" and kind == "text":
                outcome = await run_in_threadpool(service.text.retry_bound_attempt,
                                                   str(identity["id"]), request_id)
                attempt = outcome["attempt"]
                return {
                    "protocol": "task-attempt-v1", "kind": "text", "original_id": request_id,
                    "attempt_id": attempt["request_id"], "status": task_result_status("text", attempt),
                    "result_ready": outcome["result_ready"], "existing": outcome["existing"],
                    "created": not outcome["result_ready"] and not outcome["existing"],
                }
            result = await run_in_threadpool(service.start, kind, identity, request_id,
                                             allow_unconfirmed_retry=body.allow_unconfirmed_retry,
                                             retry_not_sent_failure_at=body.retry_not_sent_failure_at,
                                             original_only=company_original_only or (admin_original_only and body.action == "recover"),
                                             read_only_original=admin_original_only and body.action == "recover",
                                             trusted_bound_retry=admin_original_only and body.action == "retry")
            if admin_original_only and body.action == "retry":
                attempt_id = result.get("selected_id") or result.get("replacement_id") or request_id
                before_state = before.get("_completion") or {}
                existing = (before_state.get("bound_retry") is True and attempt_id == request_id
                            or before_state.get("replacement_id") == attempt_id
                            or attempt_id == request_id and before.get("status") in {"queued", "running", "success", "succeeded"})
                ready = result.get("state") in {"result_ready", "completed"} and result.get("selected_id") == attempt_id
                status = (result.get("replacement_status") if attempt_id != request_id
                          else result.get("original_status"))
                if kind == "image":
                    with service.store.connect() as db:
                        current = service.store.read_receipt(db, kind, str(identity["id"]), attempt_id) or {}
                    status = task_result_status(kind, current)
                return {
                    "protocol": "task-attempt-v1", "kind": kind, "original_id": request_id,
                    "attempt_id": attempt_id, "status": status,
                    "result_ready": ready, "existing": existing, "created": not ready and not existing,
                }
            return result
        except ConversationBindingError as exc:
            raise HTTPException(409, detail={"code": exc.code}) from None
        except WorkLifecycleError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None
    return router
