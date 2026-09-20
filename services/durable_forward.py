"""Compatibility transports backed by the original text-request receipt.

Disconnecting a subscriber never cancels the accepted executor. Native wire
bytes are private output files; repeating an original ID only subscribes again.
No access token, cookie or arbitrary forwarded header is persisted.
"""
import asyncio
import json
import os
import uuid

from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse

from services.request_context import trusted_source


FORWARDED_HEADERS = {"session-id", "thread-id", "x-client-request-id", "originator", "version", "user-agent", "openai-beta",
                     "x-codex-window-id", "x-codex-turn-metadata", "x-codex-beta-features"}
OUTPUT_HEADERS = {"content-type", "cache-control", "x-request-id", "openai-request-id", "x-openai-request-id", "retry-after"}
MAX_OUTPUT_BYTES = 128 * 1024 * 1024


def envelope(identity, payload, request, protocol, *, operation="text", compact=False):
    headers = {k.lower(): v for k, v in request.headers.items() if k.lower() in FORWARDED_HEADERS}
    request_id = str(headers.get("x-client-request-id") or payload.get("client_request_id") or uuid.uuid4().hex)
    route = "codex" if protocol == "codex" else "chat"
    model = str(payload.get("model") or "auto")
    from utils.helper import is_codex_image_model
    if operation == "image" and is_codex_image_model(model):
        from services.openai_backend_api import CODEX_RESPONSES_MODEL
        route, model = "codex", CODEX_RESPONSES_MODEL
    body = {"client_request_id": request_id, "client_conversation_id": str(payload.get("client_conversation_id") or headers.get("session-id") or headers.get("thread-id") or ""),
            "model": model, "_route": route, "_operation": operation,
            "_expected_sends": max(1, min(4, int(payload.get("n") or 1))) if operation == "image" else 1,
            "_forward": {"protocol": protocol, "payload": payload, "headers": headers,
                         "identity": {k: identity[k] for k in ("id", "name", "role") if k in identity}, "compact": compact}}
    if route == "codex":
        from services.codex_service import codex_service
        body["client_conversation_id"] = codex_service._affinity_key(identity, headers, payload)
    return body


def raw_receipt(service, owner, request_id):
    with service.store.connect() as db:
        return service.store.read_receipt(db, "text", owner, request_id)


async def respond(identity, payload, request, protocol, *, operation="text", compact=False, service=None):
    if service is None:
        from services.text_task_service import text_task_service as service
    body = envelope(identity, payload, request, protocol, operation=operation, compact=compact)
    owner, request_id = str(identity["id"]), body["client_request_id"]
    from fastapi.concurrency import run_in_threadpool
    from services.conversation_binding_service import ConversationBindingError
    try:
        await run_in_threadpool(service.submit, owner, body, source=trusted_source(identity, request))
    except ConversationBindingError:
        raise HTTPException(409, detail={"code": "REQUEST_ID_CONFLICT", "request_id": request_id}) from None
    # Async waiting consumes no executor worker. HTTP cancellation affects only
    # this subscriber, not the original receipt or its independently held claim.
    while True:
        r = await run_in_threadpool(raw_receipt, service, owner, request_id)
        head = r.get("_wire_head")
        if head:
            break
        if r["status"] in {"unknown", "failed"}:
            return Response(json.dumps({"error": {"code": r.get("error_code"), "request_id": request_id}, "status": r["status"], "rate_limit": r.get("rate_limit")}),
                            status_code=409 if r["status"] == "unknown" else int(r.get("_wire_error_status") or 503),
                            headers={"X-Request-ID": request_id, "Cache-Control": "private, no-store"}, media_type="application/json")
        await asyncio.sleep(.1)
    headers = {**head["headers"], "X-Request-ID": request_id, "Cache-Control": "private, no-store"}
    if not head["stream"]:
        while r["status"] in {"queued", "running"}:
            await asyncio.sleep(.1)
            r = await run_in_threadpool(raw_receipt, service, owner, request_id)
        with service.store.output_file(r["_wire_output"]) as handle:
            data = handle.read(r.get("_wire_size", 0))
        return Response(data, status_code=head["status"], headers=headers)
    async def chunks():
        offset = 0
        while True:
            r = await run_in_threadpool(raw_receipt, service, owner, request_id)
            size = r.get("_wire_size", 0)
            if size > offset:
                with service.store.output_file(r["_wire_output"]) as handle:
                    handle.seek(offset)
                    data = handle.read(min(65536, size - offset))
                if data:
                    offset += len(data)
                    yield data
                    continue
            if r["status"] not in {"queued", "running"}:
                # An incomplete original stream ends incomplete, never with a
                # fabricated success marker or an automatic upstream replay.
                return
            await asyncio.sleep(.1)
    return StreamingResponse(chunks(), status_code=head["status"], headers=headers)


def run(service, owner, request_id, body):
    spec = body["_forward"]
    output = service.store.create_output()
    service._update(owner, request_id, _wire_output=output, _wire_size=0)
    protocol = spec["protocol"]
    try:
        if protocol == "codex":
            from services.codex_service import codex_service
            result = codex_service.submit(spec["identity"], spec["payload"], spec["headers"], compact=spec["compact"])
            status, headers = result.status_code, result.headers
            stream = result.stream is not None
            parts = result.stream if stream else [result.body or b""]
        else:
            from importlib import import_module
            handler = import_module("services.protocol." + protocol).handle
            result = handler(spec["payload"])
            stream = not isinstance(result, (dict, list, str, bytes, bytearray))
            status = 200
            headers = {"content-type": "text/event-stream" if stream else "application/json"}
            from services.log_service import _strip_internal_response_fields
            def wire_events():
                anthropic = protocol == "anthropic_v1_messages"
                if not anthropic:
                    yield ": stream-open\n\n"
                for item in result:
                    public = _strip_internal_response_fields(item)
                    if anthropic:
                        yield "event: " + str(public.get("type") or "message_delta") + "\n"
                    yield "data: " + json.dumps(public, ensure_ascii=False) + "\n\n"
                if not anthropic:
                    yield "data: [DONE]\n\n"
            parts = wire_events() if stream else [json.dumps(_strip_internal_response_fields(result), ensure_ascii=False).encode()]
        head = {"status": status, "headers": {k.lower(): str(v) for k, v in headers.items() if k.lower() in OUTPUT_HEADERS}, "stream": stream}
        # For non-stream output, publish headers only after all bytes persist.
        if stream:
            service._update(owner, request_id, _wire_head=head)
        size = 0
        try:
            with service.store.output_file(output, append=True) as handle:
                for part in parts:
                    data = part.encode() if isinstance(part, str) else bytes(part)
                    size += len(data)
                    if size > MAX_OUTPUT_BYTES:
                        raise RuntimeError("private output limit")
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                    service._update(owner, request_id, _wire_size=size)
        finally:
            close = getattr(parts, "close", None)
            if callable(close):
                close()
        current = raw_receipt(service, owner, request_id)
        terminal = protocol != "codex" or current.get("_upstream_terminal") is True
        failed = bool(current.get("_upstream_failed"))
        service._update(owner, request_id, status="failed" if failed and terminal else "succeeded" if terminal else "unknown",
                        error_code="codex_response_failed" if failed and terminal else None if terminal else "codex_upstream_outcome_unknown",
                        _wire_head=head, _turn_reserved=not terminal, finished_at=service._now())
    except Exception as exc:
        current = raw_receipt(service, owner, request_id)
        code = getattr(exc, "code", "CONVERSATION_OUTCOME_UNKNOWN")
        not_sent = not current.get("_submission_started")
        if not_sent and code in {"codex_busy", "codex_bound_account_unavailable", "codex_transport_failed"}:
            raise
        known_rejection = code in {"codex_limited", "codex_auth_required", "codex_permission_denied", "invalid_codex_request"}
        service._update(owner, request_id, status="failed" if not_sent or known_rejection else "unknown",
                        error_code=code, _turn_reserved=not (not_sent or known_rejection),
                        _wire_error_status=getattr(exc, "status_code", 503),
                        upstream_outcome="not_sent" if not_sent else "rejected" if known_rejection else "unknown")
