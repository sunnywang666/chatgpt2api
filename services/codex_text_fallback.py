"""Server-owned Chat-to-Codex envelopes preserving verified original context."""
from __future__ import annotations

import hashlib
import json

from services.conversation_binding_service import ConversationBindingError


def _reject(code):
    raise ConversationBindingError("Codex fallback cannot use this original request", code=code)


class CodexFallbackResponseError(ValueError):
    """A terminal native response cannot safely become a text-task success."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _stable(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip() and len(value) <= 200


def _responses_input(messages):
    if not isinstance(messages, list) or not messages:
        _reject("CODEX_FALLBACK_CONTEXT_UNAVAILABLE")
    converted = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            _reject("CODEX_FALLBACK_CONTEXT_UNAVAILABLE")
        content = message.get("content")
        parts = [{"type": "input_text", "text": content}] if isinstance(content, str) else []
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    _reject("CODEX_FALLBACK_CONTEXT_UNAVAILABLE")
                kind = item.get("type")
                if kind in {"text", "input_text"} and isinstance(item.get("text"), str):
                    parts.append({"type": "input_text", "text": item["text"]})
                elif kind in {"image_url", "input_image"}:
                    source = item.get("image_url")
                    image = source.get("url") if isinstance(source, dict) else source
                    if not isinstance(image, str) or not image:
                        _reject("CODEX_FALLBACK_CONTEXT_UNAVAILABLE")
                    converted_image = {"type": "input_image", "image_url": image}
                    detail = item.get("detail")
                    if detail is None and isinstance(source, dict):
                        detail = source.get("detail")
                    if detail in {"low", "high", "auto"}:
                        converted_image["detail"] = detail
                    parts.append(converted_image)
                else:
                    _reject("CODEX_FALLBACK_CONTEXT_UNAVAILABLE")
        if not parts:
            _reject("CODEX_FALLBACK_CONTEXT_UNAVAILABLE")
        converted.append({"role": message["role"], "content": parts})
    return converted


def _native_body(owner, predecessor_id, client_request_id, model, account_ref, payload, *, marker, origin_id, parent_hash,
                 session_id=None):
    session_id = session_id or "codex-fallback:" + hashlib.sha256(
        (owner + "\0" + predecessor_id + "\0" + client_request_id).encode()
    ).hexdigest()
    headers = {"x-client-request-id": client_request_id, "session-id": session_id}
    from services.codex_service import CodexService
    affinity = CodexService._affinity_key({"id": owner, "role": "admin"}, headers, payload)
    return {
        "client_request_id": client_request_id,
        "client_conversation_id": affinity,
        "model": model,
        "_route": "codex", "_operation": "text", "_expected_sends": 1,
        "_requested_account_ref": account_ref,
        "_forward": {"protocol": "codex", "payload": payload, "headers": headers,
                     "identity": {"id": owner, "role": "admin"}, "compact": False},
        marker: predecessor_id,
        "_codex_origin_request_id": origin_id,
        "_codex_fallback_input_hash": parent_hash,
        "_codex_fallback_protocol": "codex-responses-v1",
    }


def _original(store, db, owner, original_id):
    row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", (owner, original_id)).fetchone()
    if not row:
        _reject("CODEX_FALLBACK_ORIGINAL_NOT_FOUND")
    request_hash, raw = row
    receipt = json.loads(raw)
    if (receipt.get("status") != "failed" or receipt.get("error_code") != "MODEL_NOT_FOUND"
            or receipt.get("upstream_outcome") != "rejected" or receipt.get("_submission_started") is not True
            or receipt.get("_route", "chat") != "chat" or receipt.get("_operation", "text") != "text"
            or receipt.get("_forward_protocol") or receipt.get("_codex_fallback_of")):
        _reject("CODEX_FALLBACK_ORIGINAL_INELIGIBLE")
    try:
        original_body = store.load_input(receipt["_input_ref"])
        from services.text_task_service import TextTaskService
        if TextTaskService._submission_identity(owner, original_body) != (original_id, request_hash):
            _reject("CODEX_FALLBACK_ORIGINAL_INELIGIBLE")
    except (OSError, ValueError, TypeError, KeyError):
        _reject("CODEX_FALLBACK_CONTEXT_UNAVAILABLE")
    return request_hash, original_body


def _original_messages(store, db, owner, original_id, original_body):
    # A submitted physical cursor means the body may contain only the last
    # turn. Rebuild its complete retained history without modifying that body.
    if original_body.get("conversation_id") or original_body.get("parent_message_id"):
        from services.codex_chat_history import load_bound_chat_history
        return load_bound_chat_history(store, db, owner, original_id)
    return original_body.get("messages")


def build_body(store, db, owner, original_id, client_request_id, model, account_ref, reasoning_effort=None):
    if not all(_stable(value) for value in (original_id, client_request_id, model, account_ref)):
        _reject("CODEX_FALLBACK_INVALID")
    if reasoning_effort is not None and not _stable(reasoning_effort):
        _reject("CODEX_FALLBACK_INVALID")
    original_hash, original_body = _original(store, db, owner, original_id)
    messages = _original_messages(store, db, owner, original_id, original_body)
    payload = {"model": model, "input": _responses_input(messages),
               "stream": True, "store": False, "include": ["reasoning.encrypted_content"]}
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    body = _native_body(owner, original_id, client_request_id, model, account_ref, payload,
                        marker="_codex_fallback_of", origin_id=original_id, parent_hash=original_hash)
    if original_body.get("product_work") is not None:
        body["product_work"] = original_body["product_work"]
    return body


def validate_prepared(store, db, owner, body):
    """Rebuild in the insert transaction so callers cannot alter retained input."""
    original_id = body.get("_codex_fallback_of")
    payload = ((body.get("_forward") or {}).get("payload") or {})
    reasoning = payload.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    expected = build_body(store, db, owner, original_id, body.get("client_request_id"), body.get("model"),
                          body.get("_requested_account_ref"), effort)
    if body != expected:
        _reject("CODEX_FALLBACK_INVALID")
    other = db.execute(
        "SELECT id FROM requests WHERE owner=? AND json_extract(receipt,'$._codex_fallback_of')=? AND id<>? LIMIT 1",
        (owner, original_id, body["client_request_id"]),
    ).fetchone()
    if other:
        _reject("CODEX_FALLBACK_CONFLICT")


def _completed_response(service, receipt):
    if receipt.get("_upstream_terminal") is not True:
        return {}
    try:
        with service.store.output_file(receipt["_wire_output"]) as handle:
            wire = handle.read(receipt.get("_wire_size", 0))
        response = json.loads(wire)
    except (OSError, TypeError, KeyError):
        return {}
    except ValueError:
        response = None
        try:
            for line in wire.decode("utf-8").splitlines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                if (isinstance(event, dict) and event.get("type") == "response.completed"
                        and isinstance(event.get("response"), dict)):
                    response = event["response"]
        except (UnicodeDecodeError, ValueError):
            return {}
    if not isinstance(response, dict) or response.get("status") != "completed":
        return {}
    if receipt.get("upstream_response_id") and response.get("id") != receipt["upstream_response_id"]:
        raise CodexFallbackResponseError("CODEX_FALLBACK_RESPONSE_INVALID")
    return response


def _final_text(response):
    output = response.get("output") if isinstance(response, dict) else None
    if not isinstance(output, list):
        return None
    text = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                text.append(part["text"])
    result = "".join(text)
    return result if result.strip() else None


def build_continue_body(store, db, owner, parent_id, client_request_id, messages, step_ref=None):
    if not all(_stable(value) for value in (parent_id, client_request_id)):
        _reject("CODEX_FALLBACK_INVALID")
    row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", (owner, parent_id)).fetchone()
    if not row:
        _reject("CODEX_FALLBACK_PARENT_UNAVAILABLE")
    parent_hash, raw = row
    parent = json.loads(raw)
    if (parent.get("status") != "succeeded" or parent.get("_route") != "codex"
            or parent.get("_forward_protocol") != "codex" or not parent.get("_codex_origin_request_id")):
        _reject("CODEX_FALLBACK_PARENT_UNAVAILABLE")
    try:
        saved = store.load_input(parent["_input_ref"])
        from services.text_task_service import TextTaskService
        if TextTaskService._submission_identity(owner, saved) != (parent_id, parent_hash):
            _reject("CODEX_FALLBACK_PARENT_UNAVAILABLE")
        previous_payload = saved["_forward"]["payload"]
        response = _completed_response(type("Service", (), {"store": store})(), parent)
    except (OSError, ValueError, TypeError, KeyError):
        _reject("CODEX_FALLBACK_PARENT_UNAVAILABLE")
    if (not isinstance(previous_payload, dict) or not isinstance(previous_payload.get("input"), list)
            or not isinstance(response.get("output"), list) or not isinstance(response.get("id"), str)
            or response["id"] != parent.get("_codex_response_id") or _final_text(response) is None):
        _reject("CODEX_FALLBACK_PARENT_UNAVAILABLE")
    account_ref = parent.get("_requested_account_ref")
    model = previous_payload.get("model")
    if not _stable(account_ref) or not _stable(model):
        _reject("CODEX_FALLBACK_PARENT_UNAVAILABLE")
    payload = {"model": model, "input": [*previous_payload["input"], *response["output"], *_responses_input(messages)],
               "stream": True, "store": False, "include": ["reasoning.encrypted_content"]}
    reasoning = previous_payload.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, dict):
            _reject("CODEX_FALLBACK_PARENT_UNAVAILABLE")
        payload["reasoning"] = reasoning
    session_id = ((saved.get("_forward") or {}).get("headers") or {}).get("session-id")
    if not _stable(session_id):
        _reject("CODEX_FALLBACK_PARENT_UNAVAILABLE")
    parent_work = saved.get("product_work")
    if parent_work is not None:
        try:
            from services.product_work import normalize_product_work
            product_work = normalize_product_work(parent_work)
            if step_ref is not None:
                product_work = normalize_product_work({
                    "product_ref": product_work["product_ref"], "step_ref": step_ref,
                })
        except (TypeError, ValueError, KeyError):
            _reject("CODEX_FALLBACK_PARENT_UNAVAILABLE")
    elif step_ref is not None:
        _reject("CODEX_FALLBACK_PRODUCT_WORK_STEP_INVALID")
    body = _native_body(owner, parent_id, client_request_id, model, account_ref, payload,
                        marker="_codex_continue_of", origin_id=parent["_codex_origin_request_id"], parent_hash=parent_hash,
                        session_id=session_id)
    if parent_work is not None:
        body["product_work"] = product_work
    body["_codex_continue_messages"] = messages
    return body


def validate_prepared_continue(store, db, owner, body):
    parent_id = body.get("_codex_continue_of")
    expected = build_continue_body(store, db, owner, parent_id, body.get("client_request_id"),
                                   body.get("_codex_continue_messages"),
                                   (body.get("product_work") or {}).get("step_ref"))
    if body != expected:
        _reject("CODEX_FALLBACK_INVALID")
    other = db.execute(
        "SELECT id FROM requests WHERE owner=? AND json_extract(receipt,'$._codex_continue_of')=? AND id<>? LIMIT 1",
        (owner, parent_id, body["client_request_id"]),
    ).fetchone()
    if other:
        _reject("CODEX_FALLBACK_CONFLICT")


def project_completed_wire(service, receipt):
    """Project only a fully saved native terminal response; never SSE deltas."""
    if not receipt.get("_codex_origin_request_id"):
        return {}
    response = _completed_response(service, receipt)
    response_id = response.get("id")
    output = response.get("output")
    if not isinstance(response_id, str) or not response_id or not isinstance(output, list):
        raise CodexFallbackResponseError("CODEX_FALLBACK_RESPONSE_INVALID")
    text = _final_text(response)
    if text is None:
        raise CodexFallbackResponseError("CODEX_FALLBACK_EMPTY_RESPONSE")
    return {"_codex_response_id": response_id, "content": text}
