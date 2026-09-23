"""Private company ingress over the existing public task handlers and stores.

The Workbench BFF authenticates its company session and supplies a server-only
credential. No pool key is issued, copied to a client, or used as task ownership.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
from uuid import UUID

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from api.support import require_admin
from services.program_key_policy import make_policy

PREFIX = "/api/workbench/company"
PUBLIC_PREFIX = "/api/v1/content/company-ai"
_ID = r"[A-Za-z0-9_:-][A-Za-z0-9_.:-]{0,199}"


def allowed_route(method: str, path: str) -> bool:
    if method == "GET":
        return path in {"/session", "/v1/models", "/api/image-tasks"} or bool(
            re.fullmatch(rf"/api/chat-requests/{_ID}", path)
            or re.fullmatch(rf"/api/image-tasks/{_ID}/images/[0-9]+", path)
        )
    return method == "POST" and (
        path in {"/api/chat-requests", "/api/image-tasks/generations", "/api/image-tasks/edits"}
        or bool(re.fullmatch(rf"/api/chat-requests/{_ID}/recover", path))
        or bool(re.fullmatch(rf"/api/chat-requests/{_ID}/archive-conversation", path))
        or bool(re.fullmatch(rf"/api/chat-requests/{_ID}/restore-conversation", path))
        or bool(re.fullmatch(rf"/api/image-tasks/{_ID}/resume-poll", path))
        or bool(re.fullmatch(rf"/api/image-tasks/{_ID}/archive-thread", path))
        or bool(re.fullmatch(rf"/api/image-tasks/{_ID}/restore-thread", path))
    )


def company_identity(org: str, user: str, connector: str) -> dict:
    if any(not value or len(value) > 200 or any(ord(c) < 33 or ord(c) == 127 for c in value)
           for value in (org, user)):
        raise HTTPException(400, detail={"code": "COMPANY_IDENTITY_INVALID"})
    try:
        normalized = str(UUID(connector))
    except (ValueError, AttributeError):
        raise HTTPException(400, detail={"code": "CONNECTOR_ID_INVALID"}) from None
    if connector != normalized:
        raise HTTPException(400, detail={"code": "CONNECTOR_ID_INVALID"})
    subject = json.dumps([org, user, connector], separators=(",", ":"), ensure_ascii=True)
    owner = "company_" + hashlib.sha256(subject.encode()).hexdigest()
    return {"id": owner, "name": "Company application", "role": "user", "enabled": True,
            "policy": make_policy(["chat"], revision=1).to_record()}


async def company_request_boundary(request: Request, call_next):
    path = request.scope["path"]
    if path != PREFIX and not path.startswith(PREFIX + "/"):
        return await call_next(request)
    target = path[len(PREFIX):]
    if request.headers.get("x-workbench-image-client") == "1":
        return JSONResponse({"detail": {"code": "COMPANY_PRIVATE_INGRESS_REQUIRED"}}, status_code=403)
    # Exact route/method allowlist prevents this server credential from becoming
    # an arbitrary admin proxy. Authenticate before touching the request body.
    if not allowed_route(request.method, target):
        return JSONResponse({"detail": {"code": "COMPANY_ROUTE_NOT_FOUND"}}, status_code=404)
    try:
        require_admin(request.headers.get("authorization"))
        org = request.headers.get("x-workbench-company-org", "")
        user = request.headers.get("x-workbench-company-user", "")
        connector = request.headers.get("x-workbench-company-connector", "")
        expected = request.headers.get("x-workbench-expected-user", "")
        if not expected or not hmac.compare_digest(expected.encode(), user.encode()):
            raise HTTPException(409, detail={"code": "COMPANY_USER_CHANGED"})
        identity = company_identity(org, user, connector)
    except HTTPException as exc:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code,
                            headers={"Cache-Control": "private, no-store"})
    if target == "/session":
        return JSONResponse({"contract_version": 1, "org_id": org, "user_id": user,
                             "connector_id": connector}, headers={"Cache-Control": "private, no-store"})
    request.state.company_identity = identity
    request.scope["path"] = target
    request.scope["raw_path"] = target.encode()
    headers = [(k, v) for k, v in request.scope["headers"]
               if k.lower() not in {b"authorization", b"x-workbench-image-client", b"x-forwarded-prefix",
                                    b"x-workbench-company-org", b"x-workbench-company-user",
                                    b"x-workbench-company-connector", b"x-workbench-expected-user"}]
    headers.extend([(b"x-workbench-image-client", b"1"),
                    (b"x-forwarded-prefix", PUBLIC_PREFIX.encode())])
    request.scope["headers"] = headers
    response = await call_next(request)
    response.headers["Cache-Control"] = "private, no-store"
    return response
