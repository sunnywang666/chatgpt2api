"""Owner-scoped image conversation linkage in the existing image receipts.

No second scheduler or task database. Public callers name their own work and an
exact prior image task, never an upstream account/conversation/message cursor.
All acceptance and dependency updates are made under TaskStore's transaction.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from urllib.parse import unquote, urlsplit

from services.openai_backend_api import OpenAIBackendAPI

PROTOCOL = "image-thread-v1"
_FIELDS = ("provider_binding_id", "provider_account_identity", "client_conversation_id",
           "conversation_id", "parent_message_id")


class ImageThreadError(ValueError):
    def __init__(self, code: str, *, status: int = 409, submitted: bool = False):
        super().__init__(code)
        self.code, self.status = code, status
        self.upstream_submitted = submitted


def _id(value):
    return isinstance(value, str) and value not in {".", ".."} and bool(re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", value))


def input_fields(payload):
    """Return only explicit new fields; absence preserves old immutable hashes."""
    thread = payload.get("image_thread_id")
    source = payload.get("edit_source_task_id")
    index = payload.get("edit_source_index", 0)
    if not thread and not source and index in (0, None):
        return {}
    if not _id(thread) or (source and not _id(source)):
        raise ImageThreadError("IMAGE_THREAD_INPUT_INVALID", status=400)
    if type(index) is not int or index < 0 or index >= 16 or (not source and index != 0):
        raise ImageThreadError("IMAGE_THREAD_INPUT_INVALID", status=400)
    if any(payload.get(key) for key in (*_FIELDS, "retain_conversation", "upstream_model")):
        raise ImageThreadError("IMAGE_THREAD_CURSOR_FORBIDDEN", status=400)
    if payload.get("model", "gpt-image-2") != "gpt-image-2" or int(payload.get("n") or 1) != 1:
        raise ImageThreadError("IMAGE_THREAD_MODEL_UNSUPPORTED", status=400)
    return {"image_thread_id": thread, **({"edit_source_task_id": source, "edit_source_index": index} if source else {})}


def saved_image_bytes(task, index=0):
    """Read only an output recorded on this receipt, never fetch its URL."""
    from services.image_storage_service import image_storage_service
    data = task.get("data")
    if task.get("status") != "success" or not isinstance(data, list) or not 0 <= index < len(data):
        raise ImageThreadError("IMAGE_THREAD_SOURCE_UNAVAILABLE")
    item = data[index]
    if not isinstance(item, dict):
        raise ImageThreadError("IMAGE_THREAD_SOURCE_UNAVAILABLE")
    if item.get("b64_json"):
        try:
            result = base64.b64decode(item["b64_json"], validate=True)
        except (ValueError, TypeError):
            raise ImageThreadError("IMAGE_THREAD_SOURCE_UNAVAILABLE") from None
    else:
        url = str(item.get("url") or "")
        public_base = str(image_storage_service.settings().get("public_base_url") or "").rstrip("/")
        if public_base and url.startswith(public_base + "/"):
            relative = unquote(urlsplit(url[len(public_base) + 1:]).path)
        elif "/images/" in urlsplit(url).path:
            relative = unquote(urlsplit(url).path.split("/images/", 1)[1])
        else:
            raise ImageThreadError("IMAGE_THREAD_SOURCE_UNAVAILABLE")
        try:
            result = image_storage_service.get_bytes(relative)
        except Exception:
            raise ImageThreadError("IMAGE_THREAD_SOURCE_UNAVAILABLE") from None
    if not result:
        raise ImageThreadError("IMAGE_THREAD_SOURCE_UNAVAILABLE")
    return result


def public_thread(task):
    thread = task.get("_image_thread")
    if not isinstance(thread, dict):
        return None
    return {"protocol": PROTOCOL, "id": thread["id"],
            "previous_task_id": thread.get("previous_task_id"),
            "edit_source_task_id": thread.get("edit_source_task_id")}


def source_fingerprint(task):
    if not isinstance(task, dict):
        return None
    return hashlib.sha256(json.dumps({k: task.get(k) for k in (*_FIELDS, "id", "owner_id", "status", "data", "request_message_id", "adopted_source_request_message_id", "_image_thread", "_recovery_suppressed")}, sort_keys=True).encode()).hexdigest()

def _source_usable(task):
    return (task and task.get("status") == "success" and not task.get("_recovery_suppressed")
            and all(isinstance(task.get(k), str) and task[k] for k in _FIELDS)
            and bool(task.get("request_message_id") or task.get("adopted_source_request_message_id")))


def accept_thread(task, tasks, payload, mode, *, output_reader=saved_image_bytes):
    fields = input_fields(payload)
    if not fields:
        return
    owner, thread_id = task["owner_id"], fields["image_thread_id"]
    owned = {r["id"]: r for r in tasks if r.get("owner_id") == owner}
    members = [r for r in owned.values() if (r.get("_image_thread") or {}).get("id") == thread_id]
    members.sort(key=lambda r: r.get("_sequence", 0))
    if members and (not members[-1].get("_sequence") or len({r.get("_sequence") for r in members}) != len(members)):
        raise ImageThreadError("IMAGE_THREAD_HISTORY_INVALID")
    previous = members[-1] if members else None
    source_id = fields.get("edit_source_task_id")
    source = owned.get(source_id) if source_id else None
    origin = (previous.get("_image_thread") or {}).get("origin_task_id") if previous else None
    if source_id:
        if mode != "edit" or not _source_usable(source):
            raise ImageThreadError("IMAGE_THREAD_SOURCE_UNAVAILABLE")
        source_thread = source.get("_image_thread")
        if source_thread:
            if source_thread.get("id") != thread_id or not members:
                raise ImageThreadError("IMAGE_THREAD_SOURCE_MISMATCH")
        elif previous:
            if source_id != origin:
                raise ImageThreadError("IMAGE_THREAD_SOURCE_MISMATCH")
        else:
            # Proven legacy output may be the root of a new link record, but
            # the original receipt/hash/cursor is never relabelled or rewritten.
            if any((r.get("_image_thread") or {}).get("origin_task_id") == source_id for r in owned.values()):
                raise ImageThreadError("IMAGE_THREAD_LEGACY_ALREADY_LINKED")
            previous, origin = source, source_id
        images = payload.get("images") or []
        index = fields["edit_source_index"]
        if len(source.get("data") or []) != 1 or index >= len(images):
            raise ImageThreadError("IMAGE_THREAD_SOURCE_MISMATCH")
        original = output_reader(source)
        supplied = images[index][0]
        if not isinstance(supplied, bytes) or hashlib.sha256(original).digest() != hashlib.sha256(supplied).digest():
            raise ImageThreadError("IMAGE_THREAD_SOURCE_MISMATCH")
    task["_image_thread"] = {"protocol": PROTOCOL, "id": thread_id,
                             "previous_task_id": previous["id"] if previous else None,
                             "edit_source_task_id": source_id, "origin_task_id": origin,
                             **({"edit_source_fingerprint": source_fingerprint(source),
                                 "edit_source_sha256": hashlib.sha256(original).hexdigest()} if source_id else {})}
    task["retain_receipt"] = True
    task["client_conversation_id"] = (previous.get("client_conversation_id") if previous else None) or (
        "image-thread-" + hashlib.sha256((owner + "\0" + thread_id).encode()).hexdigest())


def predecessor_state(task, owned):
    """Read-only prerequisite result; used by preview, claim, and send guard."""
    thread = task.get("_image_thread")
    if not thread:
        return {}, None
    if thread.get("protocol") != PROTOCOL or not _id(thread.get("id")):
        return {}, "IMAGE_THREAD_HISTORY_INVALID"
    source_id = thread.get("edit_source_task_id")
    if source_id and source_fingerprint(owned.get(source_id)) != thread.get("edit_source_fingerprint"):
        return {}, "IMAGE_THREAD_SOURCE_CHANGED"
    previous_id = thread.get("previous_task_id")
    if previous_id is None:
        return {}, None
    previous = owned.get(previous_id)
    if not _source_usable(previous):
        return {}, "IMAGE_THREAD_PREVIOUS_PENDING"
    if (previous.get("_image_thread") or {}).get("id") != thread["id"] and previous_id != thread.get("origin_task_id"):
        return {}, "IMAGE_THREAD_HISTORY_INVALID"
    if previous.get("_image_thread") and not previous.get("_image_thread_terminal"):
        return {}, "IMAGE_THREAD_PREVIOUS_UNCONFIRMED"
    if previous.get("client_conversation_id") != task.get("client_conversation_id"):
        return {}, "IMAGE_THREAD_HISTORY_INVALID"
    binding = {k: previous[k] for k in _FIELDS}
    for key in ("provider_binding_id", "provider_account_identity"):
        if task.get(key) and task[key] != binding[key]:
            return {}, "IMAGE_THREAD_BINDING_CHANGED"
    binding["_image_thread_predecessor_message"] = previous.get("adopted_source_request_message_id") or previous["request_message_id"]
    binding["_image_thread_predecessor_result_ids"] = list(dict.fromkeys(
        (previous.get("result_file_ids") or []) + (previous.get("result_sediment_ids") or [])))
    return binding, None


def bind_waiting_threads(store, db, receipts):
    by_owner = {}
    for kind, owner, task_id, task in receipts:
        if kind == "image":
            by_owner.setdefault(owner, {})[task_id] = task
    for kind, owner, task_id, task in receipts:
        if kind == "image" and task.get("_image_thread") and task.get("status") == "queued":
            binding, reason = predecessor_state(task, by_owner[owner])
            changes = {} if reason else {k: v for k, v in binding.items() if task.get(k) != v}
            if task.get("_image_thread_waiting_reason") != reason:
                changes["_image_thread_waiting_reason"] = reason
            if changes:
                task.update(changes)
                store.write_receipt(db, kind, owner, task_id, task)


def _image_result_ids(message):
    payload = {"content": message.get("content"), "metadata": message.get("metadata")}
    if not OpenAIBackendAPI._has_image_asset_pointer(payload):
        return set()
    file_ids, sediment_ids = OpenAIBackendAPI._extract_image_reference_ids(payload)
    return set(file_ids) | set(sediment_ids)


def finished_parent(document, conversation_id, request_message_id, *, expected_parent=None,
                    expected_result_ids=None):
    """Prove one exact completed turn, not the newest arbitrary current_node.

    User/manual successors, siblings, missing nodes and unfinished tools are
    rejected. This is used before continuing and after each image generation.
    """
    if not isinstance(document, dict) or document.get("conversation_id", conversation_id) != conversation_id:
        raise ImageThreadError("IMAGE_THREAD_UPSTREAM_CHANGED")
    mapping = document.get("mapping")
    if not isinstance(mapping, dict) or request_message_id not in mapping:
        raise ImageThreadError("IMAGE_THREAD_TURN_UNCONFIRMED")
    root = mapping[request_message_id]
    message = root.get("message") if isinstance(root, dict) else None
    if not isinstance(message, dict) or message.get("id") != request_message_id or (message.get("author") or {}).get("role") != "user":
        raise ImageThreadError("IMAGE_THREAD_TURN_UNCONFIRMED")
    if expected_parent is not None and root.get("parent") != expected_parent:
        raise ImageThreadError("IMAGE_THREAD_UPSTREAM_CHANGED")
    children = {}
    for key, node in mapping.items():
        if isinstance(node, dict):
            children.setdefault(node.get("parent"), []).append(key)
    if expected_parent is not None and children.get(expected_parent) != [request_message_id]:
        raise ImageThreadError("IMAGE_THREAD_UPSTREAM_CHANGED")
    seen, current = {request_message_id}, request_message_id
    saw_assistant = False
    expected = {item for item in (expected_result_ids or ()) if isinstance(item, str) and item}
    observed = set()
    for _ in range(len(mapping)):
        next_ids = children.get(current, [])
        if len(next_ids) != 1:
            raise ImageThreadError("IMAGE_THREAD_TURN_UNCONFIRMED")
        current = next_ids[0]
        if current in seen:
            raise ImageThreadError("IMAGE_THREAD_TURN_UNCONFIRMED")
        seen.add(current)
        node = mapping[current]
        msg = node.get("message")
        if (not isinstance(msg, dict) or msg.get("id") != current
                or (msg.get("author") or {}).get("role") not in {"assistant", "tool"}
                or msg.get("status") != "finished_successfully"):
            raise ImageThreadError("IMAGE_THREAD_TURN_UNCONFIRMED")
        role = (msg.get("author") or {}).get("role")
        if role == "assistant":
            saw_assistant = True
        observed.update(_image_result_ids(msg))
        if expected and not observed <= expected:
            raise ImageThreadError("IMAGE_THREAD_UPSTREAM_CHANGED")
        if role == "assistant" and msg.get("end_turn") is True:
            if msg.get("channel") not in {None, "final"} or document.get("current_node") != current or children.get(current):
                raise ImageThreadError("IMAGE_THREAD_UPSTREAM_CHANGED")
            return current
        if (role == "tool" and saw_assistant and expected
                and document.get("current_node") == current and not children.get(current)
                and _image_result_ids(msg) and observed == expected):
            return current
    raise ImageThreadError("IMAGE_THREAD_TURN_UNCONFIRMED")
