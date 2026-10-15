"""Rebuild a bound Chat request's retained local context for Codex fallback."""
from __future__ import annotations

from copy import deepcopy
import json

from services.conversation_binding_service import ConversationBindingError

MAX_HISTORY_STEPS = 12
MAX_PARENT_CANDIDATES = 17
_CONTEXT_CODE = "CODEX_FALLBACK_CONTEXT_UNAVAILABLE"
_SCOPE_FIELDS = (
    "provider_binding_id",
    "provider_account_identity",
    "client_conversation_id",
    "conversation_id",
)
_ALLOWED_ROLES = frozenset({"system", "developer", "user", "assistant"})


def _reject() -> None:
    raise ConversationBindingError("Codex fallback context unavailable", code=_CONTEXT_CODE)


def _field(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip()
    return normalized if normalized and normalized == value and len(normalized) <= 300 else ""


def _scope(body: object) -> dict[str, str] | None:
    if not isinstance(body, dict):
        return None
    result = {field: _field(body.get(field)) for field in _SCOPE_FIELDS}
    return result if all(result.values()) else None


def _same_scope(receipt: object, scope: dict[str, str]) -> bool:
    return isinstance(receipt, dict) and all(
        _field(receipt.get(field)) == scope[field] for field in _SCOPE_FIELDS
    )


def _load_verified_input(store, owner: str, request_id: str, request_hash: object, receipt: object) -> dict:
    if not isinstance(receipt, dict) or not isinstance(request_hash, str):
        _reject()
    try:
        body = store.load_input(receipt["_input_ref"])
        from services.text_task_service import TextTaskService
        if not isinstance(body, dict) or TextTaskService._submission_identity(owner, body) != (request_id, request_hash):
            _reject()
        return body
    except ConversationBindingError:
        raise
    except (OSError, ValueError, TypeError, KeyError):
        _reject()


def _messages(messages: object, *, predecessor: bool) -> list[dict]:
    if not isinstance(messages, list) or not messages or len(messages) > 64:
        _reject()
    result = []
    for message in messages:
        if not isinstance(message, dict) or set(message) - {"role", "content", "name"}:
            _reject()
        if message.get("role") not in _ALLOWED_ROLES:
            _reject()
        content = message.get("content")
        if isinstance(content, str):
            if not content:
                _reject()
        elif isinstance(content, list) and content:
            for part in content:
                if not isinstance(part, dict):
                    _reject()
                kind = part.get("type")
                if kind in {"text", "input_text"} and isinstance(part.get("text"), str):
                    continue
                if kind in {"image_url", "input_image"}:
                    image = part.get("image_url")
                    image = image.get("url") if isinstance(image, dict) else image
                    if isinstance(image, str) and image:
                        continue
                _reject()
        else:
            _reject()
        result.append(deepcopy(message))
    return result


def _is_completed_text(receipt: object) -> bool:
    return (
        isinstance(receipt, dict)
        and receipt.get("status") == "succeeded"
        and receipt.get("upstream_outcome") == "completed"
        and isinstance(receipt.get("content"), str)
        and bool(receipt["content"].strip())
        and receipt.get("_route", "chat") == "chat"
        and receipt.get("_operation", "text") == "text"
        and not receipt.get("_forward_protocol")
    )


def _predecessors(db, owner: str, parent_message_id: str) -> list[tuple[str, str, dict]]:
    rows = db.execute(
        "SELECT id,request_hash,receipt FROM requests "
        "WHERE owner=? AND json_extract(receipt,'$.parent_message_id')=? LIMIT ?",
        (owner, parent_message_id, MAX_PARENT_CANDIDATES),
    ).fetchall()
    if len(rows) >= MAX_PARENT_CANDIDATES:
        _reject()
    result = []
    for request_id, request_hash, raw_receipt in rows:
        try:
            receipt = json.loads(raw_receipt)
        except (TypeError, ValueError):
            _reject()
        if _is_completed_text(receipt):
            result.append((request_id, request_hash, receipt))
    return result


def load_bound_chat_history(store, db, owner: str, original_id: str) -> list[dict]:
    """Return retained root-to-original messages, or the existing safe context error.

    The caller supplies its existing SQLite connection.  This function performs
    only SELECTs and private input reads; it creates no durable state and makes
    no upstream request.
    """
    if not _field(owner) or not _field(original_id):
        _reject()
    row = db.execute(
        "SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
        (owner, original_id),
    ).fetchone()
    if row is None:
        _reject()
    request_hash, raw_receipt = row
    try:
        original_receipt = json.loads(raw_receipt)
    except (TypeError, ValueError):
        _reject()
    original = _load_verified_input(store, owner, original_id, request_hash, original_receipt)
    scope = _scope(original)
    parent = _field(original.get("parent_message_id"))
    if scope is None or not parent or not _same_scope(original_receipt, scope):
        _reject()
    original_messages = _messages(original.get("messages"), predecessor=False)

    newest_first: list[list[dict]] = []
    seen_parents: set[str] = set()
    for _ in range(MAX_HISTORY_STEPS):
        if parent in seen_parents:
            _reject()
        seen_parents.add(parent)
        candidates = _predecessors(db, owner, parent)
        matching = [item for item in candidates if _same_scope(item[2], scope)]
        if len(matching) != 1:
            _reject()
        predecessor_id, predecessor_hash, predecessor_receipt = matching[0]
        body = _load_verified_input(store, owner, predecessor_id, predecessor_hash, predecessor_receipt)
        root = (
            _field(body.get("client_conversation_id")) == scope["client_conversation_id"]
            and not _field(body.get("provider_binding_id"))
            and not _field(body.get("provider_account_identity"))
            and not _field(body.get("conversation_id"))
            and not _field(body.get("parent_message_id"))
        )
        if not root and _scope(body) != scope:
            _reject()
        # Historical product-step metadata is not execution input. The exact
        # physical branch above determines context; only the target request's
        # product membership is inherited by the new fallback.
        if body.get("_forward"):
            _reject()
        predecessor_messages = _messages(body.get("messages"), predecessor=True)
        newest_first.append([
            *predecessor_messages,
            {"role": "assistant", "content": predecessor_receipt["content"]},
        ])
        if root:
            context: list[dict] = []
            for turn in reversed(newest_first):
                context.extend(turn)
            return [*context, *original_messages]
        parent = _field(body.get("parent_message_id"))
        if not parent:
            _reject()
    _reject()
