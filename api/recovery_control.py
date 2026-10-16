"""Owner-scoped control of existing original-result reads, never generation."""
from typing import Literal

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from services.task_store import RecoveryControlConflict


class RecoveryControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["active", "paused"]
    expected_version: StrictInt = Field(default=None, ge=0)

    @field_validator("expected_version", mode="before")
    @classmethod
    def reject_explicit_null(cls, value):
        if value is None:
            raise ValueError("expected_version must be omitted or a non-negative integer")
        return value


async def update_recovery_control(service, kind, owner, request_id, body):
    try:
        control = await run_in_threadpool(service.store.set_recovery_paused,
                                         kind, owner, request_id, body.state == "paused",
                                         expected_version=body.expected_version)
    except RecoveryControlConflict as exc:
        raise HTTPException(409, detail={"code": "RECOVERY_CONTROL_CONFLICT",
                                        "recovery_control": exc.control}) from exc
    if control is None:
        raise HTTPException(404, detail={"code": "REQUEST_NOT_FOUND"})
    if body.state == "active" and service.admission is not None:
        service.admission.wake()
    return {"request_id": request_id, "recovery_control": control}
