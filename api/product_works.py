"""Opt-in Workbench product admission over the existing durable receipts."""
from typing import Literal
import re
import time

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.support import require_identity
from services.request_context import trusted_source
from services.work_lifecycle import WorkLifecycleError
from utils.log import logger


class ProductFairnessInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    organization_id: str = Field(strict=True, min_length=1, max_length=200)
    subject_id: str = Field(strict=True, min_length=1, max_length=200)
    root_workflow_id: str = Field(strict=True, min_length=1, max_length=200)

    @field_validator("organization_id", "subject_id", "root_workflow_id")
    @classmethod
    def stable_identity(cls, value):
        if any(ord(c) < 33 or ord(c) == 127 for c in value):
            raise ValueError("invalid fairness identity")
        return value


class ProductWorkInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_ref: str = Field(strict=True, min_length=1, max_length=200)
    step_ref: str = Field(strict=True, min_length=1, max_length=200)
    fairness: ProductFairnessInput | None = Field(default=None, exclude_if=lambda value: value is None)

    @field_validator("fairness", mode="before")
    @classmethod
    def supplied_fairness(cls, value):
        if value is None:
            raise ValueError("omit fairness when unavailable")
        return value

    @field_validator("product_ref", "step_ref")
    @classmethod
    def stable_ref(cls, value):
        if value != value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("invalid product reference")
        return value


def product_identity(identity, request):
    from services.product_work import require_product_source
    scoped = {**identity, "_trusted_source": trusted_source(identity, request)}
    require_product_source(scoped)
    return scoped


def get_product_work_service():
    from services.product_work import ProductWorkService
    from services.text_task_service import text_task_service
    from services.image_task_service import image_task_service
    admission = text_task_service.admission
    if admission is None or image_task_service.admission is not admission:
        raise WorkLifecycleError("PRODUCT_WORK_UNAVAILABLE", 503)
    service = getattr(admission, "product_work", None)
    if service is None:
        service = ProductWorkService(text_task_service, image_task_service)
        admission.product_work = service
    return service


def _capability_log(entry):
    # Diagnostics must not change a capability response or expose request data.
    try:
        logger.info({"event": "product_work_capability_probe", **entry})
    except Exception:
        pass


def capability_payload(request, authorization, observation=None):
    """Build the capability response outside the event-loop thread.

    The account/settings snapshots behind product_capacity_limit share the
    admission locks with active work.  This endpoint is read-only, but its
    synchronous identity, service lookup, and capacity read can wait on those
    locks, so keep the entire synchronous chain together in the thread pool.
    """
    started = time.monotonic()
    timing = {}
    status = 200
    error_type = None
    try:
        if observation is not None:
            timing["threadpool_wait_ms"] = max(0.0, (started - observation["started"]) * 1000)
        identity_started = time.monotonic()
        try:
            product_identity(require_identity(authorization, request=request), request)
        finally:
            timing["identity_ms"] = max(0.0, (time.monotonic() - identity_started) * 1000)
        service_started = time.monotonic()
        try:
            service = get_product_work_service()
        finally:
            timing["service_lookup_ms"] = max(0.0, (time.monotonic() - service_started) * 1000)
        observed_read = getattr(service, "observe_product_capacity_limit", None)
        capacity = observed_read(timing) if callable(observed_read) else service.product_capacity_limit()
        return {"protocol": "product-work-v1", "shared_capacity_scope": "provider",
            "capacity_source": "eligible_chat_turn_total", "cross_kind": True,
            "admission_scope": "product",
            "product_count_limit": capacity,
            "retained_products_consume_capacity": True,
            "close_after_archive": True, "adoption": True, "resume": True,
            "cancel_unsent_text_step": True, "admission_control": True}
    except Exception as exc:
        status = exc.status if isinstance(exc, WorkLifecycleError) else (
            exc.status_code if isinstance(exc, HTTPException) else 500)
        error_type = (type(exc).__name__ if isinstance(exc, (WorkLifecycleError, HTTPException))
                      else "CapabilityReadError")
        raise
    finally:
        if observation is not None:
            # This executes in the worker even when the caller has timed out.
            _capability_log({"stage": "completed", "probe_id": observation["probe_id"],
                             "status": status, "error_type": error_type,
                             "duration_ms": max(0.0, (time.monotonic() - observation["started"]) * 1000),
                             **timing})


class CloseProductWork(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_ref: str = Field(strict=True, min_length=1, max_length=200)
    results_saved: Literal[True]
    outcome: Literal["completed", "failed", "cancelled"] = "completed"
    expected_version: int | None = Field(default=None, strict=True, ge=1)

    @field_validator("results_saved", mode="before")
    @classmethod
    def saved_is_boolean(cls, value):
        if value is not True:
            raise ValueError("results_saved=true required")
        return value


class ResumeProductWork(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_ref: str = Field(strict=True, min_length=1, max_length=200)
    expected_version: int = Field(strict=True, ge=1)


class AdmissionControlProductWork(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_ref: str = Field(strict=True, min_length=1, max_length=200)
    paused: bool = Field(strict=True)
    expected_version: int = Field(strict=True, ge=0)


class CancelUnsentTextStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_ref: str = Field(strict=True, min_length=1, max_length=200)
    request_id: str = Field(strict=True, min_length=1, max_length=200)
    expected_version: int = Field(strict=True, ge=1)

    @field_validator("product_ref", "request_id")
    @classmethod
    def stable_ref(cls, value):
        if value != value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("invalid product reference")
        return value


class AdoptMember(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["text", "image"]
    request_id: str = Field(strict=True, min_length=1, max_length=200)
    step_ref: str = Field(strict=True, min_length=1, max_length=200)


class AdoptProductWork(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_ref: str = Field(strict=True, min_length=1, max_length=200)
    members: list[AdoptMember] = Field(min_length=1, max_length=200)


def create_router():
    router = APIRouter()

    async def invoke(request, authorization, response, method, *args, **kwargs):
        response.headers["Cache-Control"] = "private, no-store"
        try:
            identity = product_identity(require_identity(authorization, request=request), request)
            service = get_product_work_service()
            return await run_in_threadpool(getattr(service, method), identity, *args, **kwargs)
        except WorkLifecycleError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None

    @router.get("/api/product-works/capabilities")
    async def capabilities(request: Request, response: Response, authorization: str | None = Header(default=None)):
        response.headers["Cache-Control"] = "private, no-store"
        raw_probe = request.headers.get("x-workbench-capability-probe", "")
        probe = raw_probe.lower() if re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", raw_probe,
        ) else None
        observation = {"started": time.monotonic(), "probe_id": probe}
        _capability_log({"stage": "received", "probe_id": probe})
        try:
            return await run_in_threadpool(capability_payload, request, authorization, observation)
        except WorkLifecycleError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None

    @router.get("/api/product-works")
    async def status(request: Request, response: Response,
                     product_ref: str = Query(min_length=1, max_length=200),
                     authorization: str | None = Header(default=None)):
        return await invoke(request, authorization, response, "get", product_ref)

    @router.post("/api/product-works/close")
    async def close(body: CloseProductWork, request: Request, response: Response,
                    authorization: str | None = Header(default=None)):
        return await invoke(request, authorization, response, "close", body.product_ref,
                            results_saved=body.results_saved, outcome=body.outcome,
                            expected_version=body.expected_version)

    @router.post("/api/product-works/cancel-step")
    async def cancel_step(body: CancelUnsentTextStep, request: Request, response: Response,
                          authorization: str | None = Header(default=None)):
        return await invoke(request, authorization, response, "cancel_step", body.product_ref,
                            body.request_id, body.expected_version)

    @router.post("/api/product-works/resume")
    async def resume(body: ResumeProductWork, request: Request, response: Response,
                     authorization: str | None = Header(default=None)):
        return await invoke(request, authorization, response, "resume", body.product_ref,
                            expected_version=body.expected_version)

    @router.post("/api/product-works/admission-control")
    async def admission_control(body: AdmissionControlProductWork, request: Request, response: Response,
                                authorization: str | None = Header(default=None)):
        return await invoke(request, authorization, response, "admission_control", body.product_ref,
                            body.paused, body.expected_version)

    @router.post("/api/product-works/adopt")
    async def adopt(body: AdoptProductWork, request: Request, response: Response,
                    authorization: str | None = Header(default=None)):
        return await invoke(request, authorization, response, "adopt", body.product_ref,
                            [m.model_dump() for m in body.members])

    return router
