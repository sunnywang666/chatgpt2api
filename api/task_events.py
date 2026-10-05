"""Bounded result notifications backed only by the owner's durable receipt.

No upstream reads, new submissions, event store or callback destinations. A
reconnection replays current state; it never promises exactly-once delivery.
"""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from api.key_policy import require_chat_text_policy, require_codex_endpoint, require_image_policy
from api.support import require_identity
from services.pool_admission import image_original_recovery_pending, image_result_stage
from services.task_store import pending_image_result_ids

STREAM_SECONDS = 30
CHECK_SECONDS = 1


def _automatic_unsent_image_retry(service, receipt, work_active):
    # Completion schedules a proven-unsent original in a later transaction.
    # Both the no-claim/no-completion gap and checking_original are pending;
    # neither may terminate a subscription before that same-ID retry runs.
    if (getattr(getattr(service, "admission", None), "generation_completion", None) is None
            or not work_active or receipt.get("status") != "error"
            or receipt.get("error_code") != "RESULT_UNRECOVERABLE"
            or receipt.get("_automatic_generation_recovery") is not True
            or receipt.get("recovery_retryable") is not True
            or receipt.get("_submission_started") is not False
            or receipt.get("upstream_submission_started") is not False
            or receipt.get("upstream_unfinished") is not False
            or receipt.get("upstream_outcome") not in {"not_sent", "not_submitted"}
            or receipt.get("_recovery_paused") is True or receipt.get("_recovery_suppressed") is True
            or receipt.get("_attempt_finished_at") or receipt.get("_completion_of")
            or receipt.get("data") or receipt.get("result_file_ids") or receipt.get("result_sediment_ids")
            or pending_image_result_ids(receipt) or receipt.get("_pending_image_output")):
        return False
    state = receipt.get("_completion")
    return not state or bool(isinstance(state, dict)
        and state.get("state") == "checking_original" and state.get("automatic_failure_retry") is True
        and not state.get("same_request_retry") and not state.get("replacement_id") and not state.get("selected_id")
        and isinstance(state.get("next_at"), (int, float)))


def _retrying_unsent(receipt):
    # The handler can persist an error before PoolAdmission.execute's finally
    # returns this same unsent request to the queue. Only that live claim's
    # narrow retry window is nonterminal; no model send or recovery starts here.
    claim_until = receipt.get("_claim_until")
    return bool(receipt.get("status") in {"error", "failed", "unknown"}
                and receipt.get("error_code") in {"CONVERSATION_OUTCOME_UNKNOWN",
                    "CONVERSATION_BINDING_UNAVAILABLE", "IMAGE_RESOURCE_UNAVAILABLE"}
                and receipt.get("_submission_started") is False
                and receipt.get("_executing") is True and receipt.get("_claim_id")
                and isinstance(claim_until, (int, float)) and claim_until > time.time()
                and not receipt.get("_attempt_finished_at")
                and receipt.get("_recovery_paused") is not True
                and receipt.get("_recovery_suppressed") is not True)


def _snapshot(kind, service, identity, request_id):
    with service.store.connect() as db:
        receipt = service.store.read_receipt(db, kind, str(identity["id"]), request_id)
        work = (service.store.runtime(db, receipt["_work_key"])
                if kind == "image" and receipt and receipt.get("status") == "error"
                and receipt.get("error_code") == "RESULT_UNRECOVERABLE" and receipt.get("_work_key") else None)
    if receipt is None:
        raise HTTPException(404, detail={"code": "ORIGINAL_REQUEST_NOT_FOUND"})
    if kind == "image":
        require_image_policy(identity, receipt.get("model"))
        ready = receipt.get("status") == "success" and isinstance(receipt.get("data"), list) and bool(receipt["data"])
        result_count = len(receipt["data"]) if ready else 0
    else:
        if receipt.get("_route") == "codex":
            require_codex_endpoint(identity)
        else:
            require_chat_text_policy(identity, model=receipt.get("model"))
        ready = receipt.get("status") == "succeeded" and isinstance(receipt.get("content"), str) and bool(receipt["content"].strip())
        result_count = 1 if ready else 0
    status = receipt.get("status")
    if status not in {"queued", "not_started", "running", "success", "succeeded", "failed", "error", "unknown"}:
        status = "unknown"
    retrying_unsent = (_retrying_unsent(receipt) or kind == "image" and _automatic_unsent_image_retry(
        service, receipt, not receipt.get("_work_key") or bool(work and work.get("state") == "active")))
    return {"protocol": "task-notification-v1", "kind": kind, "request_id": request_id,
            "status": status, "result_ready": ready, "result_count": result_count,
            **({"result_stage": "preparing" if retrying_unsent else image_result_stage(receipt)} if kind == "image" else {}),
            "retrying_unsent": retrying_unsent,
            "recovering_original": kind == "image" and image_original_recovery_pending(receipt)}


def _event(name, data):
    return "event: " + name + "\ndata: " + json.dumps(data, ensure_ascii=True, separators=(",", ":")) + "\n\n"


def create_router(kind, get_service):
    router = APIRouter()
    prefix = "/api/image-tasks" if kind == "image" else "/api/chat-requests"

    @router.get(prefix + "/{request_id}/events")
    async def events(request_id: str, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization, request=request)
        initial = await run_in_threadpool(_snapshot, kind, get_service(), identity, request_id)

        async def stream():
            deadline = time.monotonic() + STREAM_SECONDS
            state, previous = initial, None
            while True:
                if await request.is_disconnected():
                    return
                if state != previous:
                    name = "result_ready" if state["result_ready"] else "state"
                    yield _event(name, state)
                    previous = state
                if state["result_ready"]:
                    return
                if (state["status"] in {"failed", "error", "unknown", "success", "succeeded"}
                        and not state["recovering_original"] and not state["retrying_unsent"]):
                    # Includes a malformed/empty success: never announce ready.
                    yield _event("needs_attention", state)
                    return
                if time.monotonic() >= deadline:
                    yield _event("reconnect", {"request_id": request_id})
                    return
                yield ": waiting for persisted result\n\n"
                await asyncio.sleep(CHECK_SECONDS)
                try:
                    # Revoked keys stop receiving updates even on an open stream.
                    current = require_identity(authorization, request=request)
                    if current.get("id") != identity.get("id"):
                        raise HTTPException(403)
                    state = await run_in_threadpool(_snapshot, kind, get_service(), current, request_id)
                except HTTPException:
                    yield _event("access_lost", {"request_id": request_id})
                    return

        return StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "private, no-store", "X-Accel-Buffering": "no",
        })

    return router
