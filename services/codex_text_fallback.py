"""Server-owned Chat-to-Codex envelopes preserving verified original context."""
from __future__ import annotations

import hashlib
import json
import time

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
        if TextTaskService._no_final_model_rejection_retry(store, db, owner, receipt, time.time()):
            _reject("CODEX_FALLBACK_ORIGINAL_INELIGIBLE")
        # C is already the one consumed, bound retry after a missing-node
        # chain.  The frozen R/P/C inputs and roots stay decisive even if an
        # ancestor later changes state; C cannot start a fallback D.
        if TextTaskService._missing_node_model_rejection_retry_consumed(store, db, owner, receipt):
            _reject("CODEX_FALLBACK_ORIGINAL_INELIGIBLE")
        if receipt.get(TextTaskService.MISSING_NODE_RETRY_FIELD):
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


def fallback_options_context(store, db, owner, original_id):
    """Validate only the retained original input before exposing cached choices."""
    _original_hash, original_body = _original(store, db, owner, original_id)
    _responses_input(_original_messages(store, db, owner, original_id, original_body))
    existing = db.execute(
        "SELECT id FROM requests WHERE owner=? AND (json_extract(receipt,'$._codex_fallback_of')=? OR json_extract(receipt,'$._supersedes_request_id')=?) LIMIT 1",
        (owner, original_id, original_id),
    ).fetchone()
    return existing[0] if existing else None


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
        "SELECT id FROM requests WHERE owner=? AND id<>? AND (json_extract(receipt,'$._codex_fallback_of')=? OR json_extract(receipt,'$._supersedes_request_id')=?) LIMIT 1",
        (owner, body["client_request_id"], original_id, original_id),
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


def _rejected_retry_parent(store, db, owner, parent_id):
    """Return the one fully proved rejected native fallback eligible for one new send."""
    if not _stable(parent_id):
        _reject("CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE")
    row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", (owner, parent_id)).fetchone()
    if not row:
        _reject("CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE")
    parent_hash, raw = row
    parent = json.loads(raw)
    wire_error = parent.get("_wire_error_payload")
    error = wire_error.get("error") if isinstance(wire_error, dict) else None
    from services.product_work import _claims
    if (
        parent.get("status") != "failed"
        or parent.get("error_code") != "codex_request_rejected"
        or parent.get("upstream_outcome") != "rejected"
        or parent.get("_route") != "codex"
        or parent.get("_operation") != "text"
        or parent.get("_forward_protocol") != "codex"
        or parent.get("_submission_started") is not True
        or parent.get("_upstream_terminal") is not True
        or parent.get("_turn_reserved") is not False
        or parent.get("_executing") is not False
        or _claims(parent)
        or any(parent.get(key) for key in (
            "recovery_claim_id", "_recovery_claim_id", "_recovery_paused", "_recovery_suppressed",
            "upstream_unfinished", "_completion", "_completion_of", "_codex_continue_of",
            "_codex_rejected_retry_of", "_codex_response_id",
        ))
        or any(parent.get(key) is not None for key in (
            "recovery_claimed_at", "recovery_lease_until", "_recovery_claim_until",
        ))
        or parent.get("_wire_head") is not None
        or type(parent.get("_wire_size")) is not int
        or parent["_wire_size"] != 0
        or not isinstance(parent.get("_wire_output"), str)
        or any(field in parent for field in ("content", "result", "data", "result_file_ids", "result_sediment_ids"))
        or parent.get("_wire_error_status") != 400
        or not isinstance(error, dict)
        or error.get("code") != "codex_request_rejected"
        or error.get("type") != "invalid_request_error"
        or not _stable(parent.get("_codex_fallback_of"))
        or not _stable(parent.get("_codex_origin_request_id"))
        or parent.get("_codex_fallback_protocol") != "codex-responses-v1"
    ):
        _reject("CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE")
    try:
        with store.output_file(parent["_wire_output"]) as handle:
            if handle.read(1) != b"":
                _reject("CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE")
        saved = store.load_input(parent["_input_ref"])
        from services.text_task_service import TextTaskService
        if TextTaskService._submission_identity(owner, saved) != (parent_id, parent_hash):
            _reject("CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE")
        payload = ((saved.get("_forward") or {}).get("payload") or {})
        reasoning = payload.get("reasoning") if isinstance(payload, dict) else None
        effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
        expected_parent = build_body(store, db, owner, parent["_codex_fallback_of"], parent_id,
                                     saved.get("model"), saved.get("_requested_account_ref"), effort)
    except (OSError, ValueError, TypeError, KeyError):
        _reject("CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE")
    if saved != expected_parent:
        _reject("CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE")
    original_id = parent["_codex_fallback_of"]
    original = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", (owner, original_id)).fetchone()
    if not original:
        _reject("CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE")
    original_hash, original_raw = original
    original_receipt = json.loads(original_raw)
    forward = saved.get("_forward") or {}
    headers = forward.get("headers") or {}
    if (parent.get("_codex_origin_request_id") != original_id
            or parent.get("_codex_fallback_input_hash") != original_hash
            or parent.get("model") != saved.get("model")
            or parent.get("client_conversation_id") != saved.get("client_conversation_id")
            or parent.get("_requested_account_ref") != saved.get("_requested_account_ref")
            or headers.get("x-client-request-id") != parent_id
            or not _stable(headers.get("session-id"))
            or any(key in original_receipt for key in (
                "content", "result", "data", "result_file_ids", "result_sediment_ids"))
            or _claims(original_receipt)
            or any(original_receipt.get(key) for key in (
                "_recovery_paused", "_recovery_suppressed", "_completion", "_completion_of",
                "upstream_unfinished"))
            or any(original_receipt.get(key) is not None for key in (
                "recovery_claimed_at", "recovery_lease_until", "_recovery_claim_until"))):
        _reject("CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE")
    siblings = db.execute(
        "SELECT id FROM requests WHERE owner=? AND "
        "(json_extract(receipt,'$._codex_fallback_of')=? OR json_extract(receipt,'$._supersedes_request_id')=?)",
        (owner, parent["_codex_fallback_of"], parent["_codex_fallback_of"]),
    ).fetchall()
    if len(siblings) != 1 or siblings[0][0] != parent_id:
        _reject("CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE")
    return parent_hash, parent, saved


def build_rejected_retry_body(store, db, owner, parent_id, client_request_id):
    """Register one explicit retry of a proved empty-wire Codex rejection."""
    if not all(_stable(value) for value in (parent_id, client_request_id)):
        _reject("CODEX_REJECTED_RETRY_INVALID")
    parent_hash, parent, saved = _rejected_retry_parent(store, db, owner, parent_id)
    forward = saved.get("_forward") or {}
    payload, headers = forward.get("payload"), forward.get("headers")
    session_id = headers.get("session-id") if isinstance(headers, dict) else None
    account_identity = parent.get("_requested_account_identity")
    if (not isinstance(payload, dict) or not _stable(saved.get("model"))
            or not _stable(saved.get("_requested_account_ref")) or not _stable(session_id)
            or not _stable(account_identity)):
        _reject("CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE")
    body = _native_body(owner, parent_id, client_request_id, saved["model"], saved["_requested_account_ref"], payload,
                        marker="_codex_rejected_retry_of", origin_id=parent["_codex_origin_request_id"],
                        parent_hash=parent_hash, session_id=session_id)
    # TextTaskService compares this fixed identity to the current pool-ref
    # resolution before durable registration.  A remapped account therefore
    # cannot turn D -> E into a different Codex account attempt.
    body["provider_account_identity"] = account_identity
    if saved.get("product_work") is not None:
        body["product_work"] = saved["product_work"]
    return body


def validate_prepared_rejected_retry(store, db, owner, body, *, receipt=None):
    if receipt is not None:
        # The runner selects its account and lineage from the durable receipt.
        # Proving the retained body alone must not authorize a changed receipt.
        expected_receipt = {key: body.get(key) for key in (
            "_codex_rejected_retry_of", "_codex_origin_request_id", "_codex_fallback_input_hash",
            "_codex_fallback_protocol", "_codex_fallback_of", "_codex_continue_of",
            "model", "client_conversation_id", "_requested_account_ref", "_route", "_operation",
        )}
        expected_receipt.update(
            request_id=body.get("client_request_id"),
            _requested_account_identity=body.get("provider_account_identity"),
            provider_account_identity=body.get("provider_account_identity"),
            _forward_protocol=(body.get("_forward") or {}).get("protocol"),
        )
        if any(receipt.get(key) != value for key, value in expected_receipt.items()):
            _reject("CODEX_REJECTED_RETRY_INVALID")
    parent_id = body.get("_codex_rejected_retry_of")
    expected = build_rejected_retry_body(store, db, owner, parent_id, body.get("client_request_id"))
    if body != expected:
        _reject("CODEX_REJECTED_RETRY_INVALID")
    other = db.execute(
        "SELECT id FROM requests WHERE owner=? AND id<>? AND ("
        "json_extract(receipt,'$._codex_rejected_retry_of')=? OR "
        "json_extract(receipt,'$._codex_continue_of')=? OR "
        "json_extract(receipt,'$._codex_fallback_of')=? OR "
        "json_extract(receipt,'$._supersedes_request_id')=?) LIMIT 1",
        (owner, body["client_request_id"], parent_id, parent_id, parent_id, parent_id),
    ).fetchone()
    if other:
        _reject("CODEX_REJECTED_RETRY_CONFLICT")


def validate_rejected_retry_before_send(store, db, owner, receipt):
    """Rebuild E at the network edge so a changed D/P cannot be resent."""
    request_id = receipt.get("request_id")
    parent_id = receipt.get("_codex_rejected_retry_of")
    if not _stable(request_id) or not _stable(parent_id):
        _reject("CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE")
    row = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
    if not row:
        _reject("CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE")
    try:
        saved = store.load_input(receipt["_input_ref"])
        from services.text_task_service import TextTaskService
        if TextTaskService._submission_identity(owner, saved) != (request_id, row[0]):
            _reject("CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE")
        validate_prepared_rejected_retry(store, db, owner, saved, receipt=receipt)
    except (OSError, ValueError, TypeError, KeyError):
        _reject("CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE")


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
