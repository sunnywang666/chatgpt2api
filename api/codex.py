"""Authenticated native Codex HTTP routes."""

from __future__ import annotations

import json

from fastapi import APIRouter, Header, HTTPException, Request
from starlette.background import BackgroundTask
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse

from api.support import require_identity
from api.key_policy import require_codex_policy, require_codex_endpoint
from services.codex_service import MAX_REQUEST_BYTES, CodexHTTPResponse, CodexServiceError, codex_service


def _ordinary_identity(authorization: str | None) -> dict:
    identity = require_identity(authorization)
    if identity.get("role") != "user":
        raise HTTPException(status_code=403, detail={"error": "This Codex API requires a normal user key"})
    return identity


def _error(exc: CodexServiceError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"error": {"code": exc.code, "message": str(exc)}},
    )


def _as_response(result: CodexHTTPResponse):
    allowed = {"content-type", "cache-control", "x-request-id", "openai-request-id", "x-openai-request-id"}
    headers = {
        key: value for key, value in result.headers.items()
        if key.lower() in allowed
    }
    headers.update({"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})
    media_type = None if "content-type" in result.headers else "application/json"
    if result.stream is not None:
        close = getattr(result.stream, "close", None)
        background = BackgroundTask(close) if callable(close) else None
        return StreamingResponse(
            result.stream,
            status_code=result.status_code,
            headers=headers,
            media_type=media_type,
            background=background,
        )
    return Response(result.body or b"", status_code=result.status_code, headers=headers, media_type=media_type)


async def _payload(request: Request) -> dict:
    if request.headers.get("upgrade", "").lower() == "websocket":
        raise HTTPException(
            status_code=426,
            detail={"error": {"code": "codex_websocket_unsupported", "message": "Use Codex HTTP transport"}},
        )
    try:
        declared = int(request.headers.get("content-length") or 0)
    except ValueError:
        declared = MAX_REQUEST_BYTES + 1
    if declared > MAX_REQUEST_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": {"code": "codex_request_too_large", "message": "The Codex request is too large"}},
        )
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_REQUEST_BYTES:
            raise HTTPException(
                status_code=413,
                detail={"error": {"code": "codex_request_too_large", "message": "The Codex request is too large"}},
            )
    try:
        payload = json.loads(bytes(body))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "invalid_codex_request", "message": "A valid JSON object is required"}},
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "invalid_codex_request", "message": "A JSON object is required"}},
        )
    return payload


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/codex/v1/models")
    async def models(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = _ordinary_identity(authorization)
        require_codex_endpoint(identity)
        if request.headers.get("upgrade", "").lower() == "websocket":
            raise HTTPException(
                status_code=426,
                detail={"error": {"code": "codex_websocket_unsupported", "message": "Use Codex HTTP transport"}},
            )
        try:
            result = await run_in_threadpool(codex_service.list_native_models, identity, request.headers)
        except CodexServiceError as exc:
            raise _error(exc) from exc
        return _as_response(result)

    async def submit(request: Request, authorization: str | None, *, compact: bool):
        identity = _ordinary_identity(authorization)
        payload = await _payload(request)
        require_codex_policy(identity, payload)
        try:
            result = await run_in_threadpool(
                codex_service.submit,
                identity,
                payload,
                request.headers,
                compact=compact,
            )
        except CodexServiceError as exc:
            raise _error(exc) from exc
        return _as_response(result)

    @router.post("/codex/v1/responses")
    async def responses(request: Request, authorization: str | None = Header(default=None)):
        return await submit(request, authorization, compact=False)

    @router.post("/codex/v1/responses/compact")
    async def compact(request: Request, authorization: str | None = Header(default=None)):
        return await submit(request, authorization, compact=True)

    return router
