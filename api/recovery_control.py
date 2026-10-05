"""Owner-scoped control of existing original-result reads, never generation."""
from typing import Literal

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict


class RecoveryControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["active", "paused"]


async def update_recovery_control(service, kind, owner, request_id, body):
    control = await run_in_threadpool(service.store.set_recovery_paused,
                                     kind, owner, request_id, body.state == "paused")
    if control is None:
        raise HTTPException(404, detail={"code": "REQUEST_NOT_FOUND"})
    if body.state == "active" and service.admission is not None:
        service.admission.wake()
    return {"request_id": request_id, "recovery_control": control}
