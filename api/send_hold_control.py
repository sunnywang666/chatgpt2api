"""Owner-scoped, pre-send-only hold for an existing text original."""
from typing import Literal

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from services.task_store import SendHoldConflict, SendHoldRejected


class SendHoldRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["active", "held"]
    expected_version: StrictInt = Field(ge=0)


async def update_send_hold(service, owner, request_id, body):
    if service.admission is None:
        raise HTTPException(503, detail={"code": "SEND_HOLD_ADMISSION_REQUIRED"})
    try:
        control = await run_in_threadpool(
            service.store.set_text_send_hold, owner, request_id, body.state == "held",
            expected_version=body.expected_version,
        )
    except SendHoldConflict as exc:
        raise HTTPException(409, detail={"code": "SEND_HOLD_CONFLICT", "send_hold": exc.control}) from exc
    except SendHoldRejected as exc:
        raise HTTPException(409, detail={"code": exc.code, "send_hold": exc.control}) from exc
    if control is None:
        raise HTTPException(404, detail={"code": "REQUEST_NOT_FOUND"})
    service.admission.wake()
    return {"request_id": request_id, "send_hold": control}
