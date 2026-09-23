from __future__ import annotations

from services.image_thread import (PROTOCOL as IMAGE_THREAD_PROTOCOL, ImageThreadError, input_fields, accept_thread, public_thread, finished_parent, saved_image_bytes, source_fingerprint)

import json
import hashlib
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from services.config import DATA_DIR, config
from services.task_store import TaskStore
from contextlib import contextmanager
from services.request_context import current_request, AdmissionLost
from services.content_filter import request_text
from services.log_service import LOG_TYPE_CALL, log_service
from services.protocol import openai_v1_image_edit, openai_v1_image_generations
from utils.helper import is_codex_image_model

TASK_STATUS_QUEUED = "queued"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_SUCCESS = "success"
TASK_STATUS_ERROR = "error"
TERMINAL_STATUSES = {TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}
UNFINISHED_STATUSES = {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING}
UNRECOVERABLE_MIN_AGE_SECONDS = 15.0 * 60.0
UNRECOVERABLE_QUALIFIED_READS = 3
IMAGE_ACTIVE_BUDGET_SECONDS = 300.0


class UnrecoverableRead(RuntimeError):
    def __init__(self, message: str, *, requires_new_conversation: bool) -> None:
        super().__init__(message)
        self.requires_new_conversation = requires_new_conversation


class ConversationImageAdoptionError(ValueError):
    """A caller-safe refusal to adopt a manually completed conversation image."""


def _holds_upstream_slot(task: dict[str, Any]) -> bool:
    if "upstream_unfinished" in task:
        return task["upstream_unfinished"] is True
    # Old receipts did not distinguish local execution from upstream work.
    return bool(task.get("provider_account_identity")) and (
        task.get("status") == TASK_STATUS_RUNNING
        or task.get("error_code") == "CONVERSATION_OUTCOME_UNKNOWN"
    )


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _timestamp(value: object) -> float:
    if not isinstance(value, str) or not value.strip():
        return 0.0
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:26], fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _clean(value: object, default: str = "") -> str:
    return str(value or default).strip()


def _owner_id(identity: dict[str, object]) -> str:
    return _clean(identity.get("id")) or "anonymous"


def _task_key(owner_id: str, task_id: str) -> str:
    return f"{owner_id}:{task_id}"


def _collect_image_urls(data: list[Any]) -> list[str]:
    urls: list[str] = []
    for item in data:
        if isinstance(item, dict):
            url = item.get("url")
            if isinstance(url, str) and url:
                urls.append(url)
    return urls


def _task_age_seconds(task: dict[str, Any], now: float | None = None) -> float:
    current = time.time() if now is None else now
    created_ts = task.get("created_ts")
    if isinstance(created_ts, (int, float)) and created_ts > 0:
        return max(0.0, current - float(created_ts))
    created_at = _timestamp(task.get("created_at"))
    return max(0.0, current - created_at) if created_at > 0 else 0.0


def _upstream_status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    message = str(exc)
    return 404 if "status=404" in message else None


def _retry_after_seconds(exc: BaseException) -> int | None:
    value = getattr(exc, "retry_after", None)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return int(value)
    return None


def _recovery_failure_code(exc: BaseException, phase: str) -> str:
    status = _upstream_status_code(exc)
    if status == 429:
        return "RECOVERY_RATE_LIMITED"
    if status in {401, 403}:
        return "RECOVERY_AUTH_REQUIRED"
    if phase == "download_image_result":
        return "RECOVERY_DOWNLOAD_FAILED"
    exc_type = type(exc)
    type_name = f"{exc_type.__module__}.{exc_type.__name__}".lower()
    if (
        getattr(exc, "recovery_transport_failure", False) is True
        or
        (isinstance(exc, (ConnectionError, OSError)) and not isinstance(exc, TimeoutError))
        or any(marker in type_name for marker in ("curl_cffi", "connection", "network"))
    ):
        return "RECOVERY_TRANSPORT_FAILED"
    return "RECOVERY_READ_FAILED"


def _safe_recovery_error(code: str, phase: str) -> str:
    if phase == "download_image_result":
        reason = {
            "RECOVERY_RATE_LIMITED": "the provider is rate limited",
            "RECOVERY_AUTH_REQUIRED": "the bound account connection must be restored",
            "RECOVERY_TRANSPORT_FAILED": "the provider could not be reached",
            "RECOVERY_READ_FAILED": "the provider response could not be read",
        }.get(code, "the result download did not complete")
        return (
            f"Generated image is preserved; {reason}. "
            "Result download will resume without generating again."
        )
    return {
        "RECOVERY_RATE_LIMITED": "Image result lookup is rate limited; retry after the provider cooldown.",
        "RECOVERY_AUTH_REQUIRED": "Image result lookup requires the bound account connection to be restored.",
        "RECOVERY_TRANSPORT_FAILED": "Image result lookup could not reach the provider; retry when connectivity returns.",
        "RECOVERY_READ_FAILED": "Image result lookup returned an unreadable response; retry the same request lookup.",
    }.get(code, "Image result lookup did not complete; retry the same request lookup.")


def _branch_read_state(document: object, request_message_id: str) -> str:
    """Classify an exact request branch without treating active work as absent."""
    if not isinstance(document, dict) or not isinstance(document.get("mapping"), dict):
        return "unknown"
    mapping = document["mapping"]
    if request_message_id not in mapping:
        return "unattributable"
    children: dict[str, list[str]] = {}
    for node_id, node in mapping.items():
        if isinstance(node, dict) and node.get("parent"):
            children.setdefault(str(node["parent"]), []).append(str(node_id))
    pending = list(children.get(request_message_id, []))
    visited: set[str] = set()
    terminal = False
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        node = mapping.get(node_id) or {}
        message = node.get("message") if isinstance(node, dict) else {}
        author = message.get("author") if isinstance(message, dict) else {}
        role = _clean(author.get("role")).lower() if isinstance(author, dict) else ""
        if role == "user":
            continue
        status = _clean(message.get("status")).lower() if isinstance(message, dict) else ""
        if status in {"in_progress", "running", "pending", "queued"}:
            return "running"
        if role == "assistant" and status == "finished_successfully" and message.get("end_turn") is True:
            terminal = True
        pending.extend(children.get(node_id, []))
    return "terminal_without_result" if terminal else "no_result"


def _document_current_message_active(document: object) -> bool:
    mapping = document.get("mapping") if isinstance(document, dict) else None
    if not isinstance(mapping, dict):
        return True
    current_node = _clean(document.get("current_node"))
    node = mapping.get(current_node) if current_node else None
    message = node.get("message") if isinstance(node, dict) else None
    status = _clean(message.get("status")).lower() if isinstance(message, dict) else ""
    return status in {"in_progress", "running", "pending", "queued"}


def _backend_tasks_may_be_active(tasks: object) -> bool:
    if not isinstance(tasks, list):
        return True
    for task in tasks:
        if not isinstance(task, dict):
            return True
        status = _clean(
            task.get("status") or task.get("state") or task.get("task_status")
        ).lower()
        if status not in {
            "completed", "complete", "finished", "done", "succeeded", "success",
            "failed", "cancelled", "canceled",
        }:
            return True
    return False


def _latest_completed_manual_image_turn(
    document: object,
    original_request_message_id: str,
    original_task_created_ts: float,
    extract_records: Callable[[dict[str, Any], str], list[dict[str, Any]]],
) -> tuple[str, dict[str, Any], str]:
    """Select the latest completed manual image turn on the authoritative branch.

    The original request must either be on the current-node parent chain or
    remain in the full mapping with its own parent on the current branch. The
    task receipt's parent may point at a later failure response and is not used
    as a pre-send anchor. A newer user turn always wins; if that turn is active
    or has no completed image, an older image is never substituted.
    """
    if not isinstance(document, dict) or not isinstance(document.get("mapping"), dict):
        raise ConversationImageAdoptionError("conversation branch is unavailable")
    mapping = document["mapping"]
    current_node = _clean(document.get("current_node"))
    if not current_node or not original_request_message_id:
        raise ConversationImageAdoptionError("conversation branch is unavailable")

    reversed_path: list[str] = []
    node_id = current_node
    while node_id:
        if node_id in reversed_path:
            raise ConversationImageAdoptionError("conversation branch is invalid")
        node = mapping.get(node_id)
        if not isinstance(node, dict):
            raise ConversationImageAdoptionError("conversation branch is invalid")
        reversed_path.append(node_id)
        node_id = _clean(node.get("parent"))
    path = list(reversed(reversed_path))
    original_node = mapping.get(original_request_message_id) or {}
    original_message = original_node.get("message") if isinstance(original_node, dict) else None
    original_author = original_message.get("author") if isinstance(original_message, dict) else None
    original_request_verified = (
        isinstance(original_message, dict)
        and _clean(original_message.get("id")) == original_request_message_id
        and isinstance(original_author, dict)
        and _clean(original_author.get("role")).lower() == "user"
    )
    if not original_request_verified:
        raise ConversationImageAdoptionError("original request is not a verified user message")
    if original_request_message_id in path:
        original_index = path.index(original_request_message_id)
    else:
        # A user may edit a later prompt in ChatGPT, moving current_node onto a
        # sibling branch. Read the verified original user node's own parent;
        # the receipt parent may already have advanced to a terminal response.
        branch_anchor = _clean(original_node.get("parent"))
        if (
            not branch_anchor
            or branch_anchor not in path
            or original_task_created_ts <= 0
        ):
            raise ConversationImageAdoptionError("original request is not linked to the current conversation branch")
        original_index = path.index(branch_anchor)

    manual_requests: list[tuple[int, str]] = []
    for index, candidate_id in enumerate(path[original_index + 1:], start=original_index + 1):
        node = mapping.get(candidate_id) or {}
        message = node.get("message") if isinstance(node, dict) else None
        author = message.get("author") if isinstance(message, dict) else None
        if isinstance(author, dict) and _clean(author.get("role")).lower() == "user":
            manual_requests.append((index, candidate_id))
    if not manual_requests:
        raise ConversationImageAdoptionError("no later manual request exists on the current conversation branch")

    request_index, request_message_id = manual_requests[-1]
    if original_request_message_id not in path:
        request_node = mapping.get(request_message_id) or {}
        request_message = request_node.get("message") if isinstance(request_node, dict) else None
        created = request_message.get("create_time") if isinstance(request_message, dict) else None
        if not isinstance(created, (int, float)) or float(created) <= original_task_created_ts:
            raise ConversationImageAdoptionError("latest manual request does not postdate the original task")
    segment = path[request_index + 1:]
    completed = False
    for candidate_id in segment:
        node = mapping.get(candidate_id) or {}
        message = node.get("message") if isinstance(node, dict) else None
        if not isinstance(message, dict):
            continue
        status = _clean(message.get("status")).lower()
        if status in {"in_progress", "running", "pending", "queued"}:
            raise ConversationImageAdoptionError("latest manual request is still active")
        author = message.get("author")
        if (
            isinstance(author, dict)
            and _clean(author.get("role")).lower() == "assistant"
            and status == "finished_successfully"
            and message.get("end_turn") is True
        ):
            completed = True
    if not completed:
        raise ConversationImageAdoptionError("latest manual request is not complete")

    records = extract_records(document, request_message_id)
    segment_ids = set(segment)
    records = [
        record for record in records
        if (
            isinstance(record, dict)
            and _clean(record.get("message_id")) in segment_ids
            and (record.get("file_ids") or record.get("sediment_ids"))
        )
    ]
    if not records:
        raise ConversationImageAdoptionError("latest manual request has no completed image")
    records.sort(key=lambda record: path.index(_clean(record.get("message_id"))))
    return request_message_id, records[-1], current_node


def _request_hash(mode: str, payload: dict[str, Any]) -> str:
    """Hash the immutable task request without persisting prompts or image bytes."""
    image_hashes = []
    for item in payload.get("images") or []:
        if isinstance(item, tuple) and item and isinstance(item[0], bytes):
            image_hashes.append(hashlib.sha256(item[0]).hexdigest())
    mask_hashes = []
    for item in payload.get("mask") or []:
        if isinstance(item, tuple) and item and isinstance(item[0], bytes):
            mask_hashes.append(hashlib.sha256(item[0]).hexdigest())
    contract = {
        "mode": mode,
        "prompt_sha256": hashlib.sha256(_clean(payload.get("prompt")).encode("utf-8")).hexdigest(),
        "model": _clean(payload.get("model"), "gpt-image-2"),
        "size": _clean(payload.get("size")),
        "quality": _clean(payload.get("quality"), "auto"),
        "provider_binding_id": _clean(payload.get("provider_binding_id")),
        "provider_account_identity": _clean(payload.get("provider_account_identity")),
        "client_conversation_id": _clean(payload.get("client_conversation_id")),
        "conversation_id": _clean(payload.get("conversation_id")),
        "parent_message_id": _clean(payload.get("parent_message_id")),
        "retain_conversation": bool(payload.get("retain_conversation")),
        "image_sha256": image_hashes,
        "mask_sha256": mask_hashes,
    }
    # Keep old task fingerprints unchanged when the caller has no override.
    if _clean(payload.get("upstream_model")):
        contract["upstream_model"] = _clean(payload.get("upstream_model"))
    contract.update(input_fields(payload))
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class AuthoritativeImageTaskFailure(RuntimeError):
    code = "NO_IMAGE_GENERATED"


def _authoritative_image_failure(document: object, request_message_id: str) -> str:
    if not isinstance(document, dict):
        return ""
    current_node = _clean(document.get("current_node"))
    mapping = document.get("mapping")
    if not current_node or not request_message_id or not isinstance(mapping, dict):
        return ""
    reversed_path: list[str] = []
    message_id = current_node
    while message_id and message_id not in reversed_path:
        node = mapping.get(message_id)
        if not isinstance(node, dict):
            return ""
        reversed_path.append(message_id)
        if message_id == request_message_id:
            break
        message_id = _clean(node.get("parent"))
    if request_message_id not in reversed_path:
        return ""
    path = list(reversed(reversed_path))
    request_index = path.index(request_message_id)
    for message_id in path[request_index + 1:]:
        node = mapping.get(message_id) or {}
        message = node.get("message") if isinstance(node, dict) else {}
        author = message.get("author") if isinstance(message, dict) else {}
        if _clean(author.get("role")).lower() == "user":
            return ""
    node = mapping.get(current_node) if current_node and isinstance(mapping, dict) else None
    message = node.get("message") if isinstance(node, dict) else None
    if not isinstance(message, dict):
        return ""
    author = message.get("author")
    content = message.get("content")
    if (
        not isinstance(author, dict)
        or _clean(author.get("role")).lower() != "assistant"
        or _clean(message.get("status")) != "finished_successfully"
        or message.get("end_turn") is not True
        or not isinstance(content, dict)
        or _clean(content.get("content_type")) != "text"
    ):
        return ""
    parts = content.get("parts")
    text = "\n".join(part for part in parts if isinstance(part, str)).strip() if isinstance(parts, list) else ""
    normalized = " ".join(text.lower().split())
    explicit_failures = {
        "something went wrong while generating your image. sorry about that.",
        "something went wrong while generating your image.",
    }
    return text if normalized in explicit_failures else ""


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    item = {
        "id": task.get("id"),
        "status": task.get("status"),
        "mode": task.get("mode"),
        "model": task.get("model"),
        "size": task.get("size"),
        "quality": task.get("quality"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
    }
    if public_thread(task):
        item["image_thread"] = public_thread(task)
    if task.get("conversation_id"):
        item["image_session_id"] = task.get("conversation_id")
    if task.get("parent_message_id"):
        item["image_session_parent_id"] = task.get("parent_message_id")
    for field in (
        "provider_binding_id",
        "provider_account_identity",
        "client_conversation_id",
        "binding_status",
        "error_code",
        "waiting",
        "rate_limit",
        "upstream_model",
        "next_poll_at",
        "active_attempt_started_at",
        "active_attempt_deadline_at",
        "upstream_outcome",
        "recovery_retryable",
        "recovery_error_code",
        "recovery_phase",
        "adopted_source_request_message_id",
        "adopted_source_image_message_id",
        "adopted_from_error_code",
        "adopted_at",
    ):
        if task.get(field):
            item[field] = task.get(field)
    retry_after = task.get("recovery_retry_after_seconds")
    if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool) and retry_after >= 0:
        item["recovery_retry_after_seconds"] = retry_after
    if isinstance(task.get("upstream_submission_started"), bool):
        item["upstream_submission_started"] = task["upstream_submission_started"]
    if isinstance(task.get("upstream_unfinished"), bool):
        item["upstream_unfinished"] = task["upstream_unfinished"]
    if task.get("recovery_no_result_reads"):
        item["recovery_no_result_reads"] = task.get("recovery_no_result_reads")
    if _clean(task.get("error_code")) == "RESULT_UNRECOVERABLE":
        item["upstream_unfinished"] = False
        item["recovery_requires_new_conversation"] = bool(
            task.get("recovery_requires_new_conversation")
        )
    if (
        task.get("status") == TASK_STATUS_ERROR
        and _clean(task.get("error_code")) == "CONVERSATION_OUTCOME_UNKNOWN"
        and not _clean(task.get("request_message_id"))
    ):
        # Legacy receipts may predate persistence of the submitted user-message
        # boundary. Keep the upstream outcome UNKNOWN, but tell callers that
        # automated polling cannot safely continue until that exact boundary is
        # recovered from independent evidence.
        item["recovery_status"] = "request_message_id_required"
    if task.get("data") is not None:
        item["data"] = task.get("data")
    if task.get("usage") is not None:
        item["usage"] = task.get("usage")
    if task.get("error"):
        item["error"] = task.get("error")
    if task.get("adopted_from_error"):
        item["adopted_from_error"] = task.get("adopted_from_error")
    if task.get("progress"):
        item["progress"] = task.get("progress")
    deadline = task.get("active_attempt_deadline_at")
    if isinstance(deadline, (int, float)) and not isinstance(deadline, bool) and deadline > 0:
        item["active_budget_remaining_secs"] = round(max(0.0, float(deadline) - time.time()), 1)
    if task.get("duration_ms") is not None:
        item["duration_ms"] = task.get("duration_ms")
    if task.get("status") in (TASK_STATUS_RUNNING, TASK_STATUS_QUEUED):
        if task.get("status") == TASK_STATUS_RUNNING:
            # RUNNING 状态仅在 started_ts 被设置后（image_stream_resolve_start）才计时
            base_ts = task.get("started_ts")
        else:
            # QUEUED 状态从 created_ts 开始计时（排队等待中）
            base_ts = task.get("created_ts") or task.get("updated_ts")
        if base_ts:
            item["elapsed_secs"] = round(time.time() - base_ts, 1)
    return item


class ImageTaskService:
    def __init__(
        self,
        path: Path,
        *,
        generation_handler: Callable[[dict[str, Any]], dict[str, Any]] = openai_v1_image_generations.handle,
        edit_handler: Callable[[dict[str, Any]], dict[str, Any]] = openai_v1_image_edit.handle,
        retention_days_getter: Callable[[], int] | None = None,
        admission=None,
        store: TaskStore | None = None,
    ):
        self.path = path
        self.store = store or TaskStore(path.parent / "text_tasks.sqlite3")
        self.admission = admission
        self._transaction_local = threading.local()
        self.generation_handler = generation_handler
        self.edit_handler = edit_handler
        self.retention_days_getter = retention_days_getter or (lambda: config.image_retention_days)
        self._lock = threading.RLock()
        self._slot_condition = threading.Condition(self._lock)
        self._tasks: dict[str, dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction():
            # Import once into the existing receipt database. The legacy file
            # stays in place for rollback; it is no longer a second writer.
            imported = self.store.runtime(self._transaction_local.db, "image_json_imported", False)
            if not imported:
                self._tasks = self._load_locked()
                self._save_locked()
                self.store.set_runtime(self._transaction_local.db, "image_json_imported", True)
            changed = self._recover_unfinished_locked()
            changed = self._cleanup_locked() or changed
            if changed:
                self._save_locked()

    @contextmanager
    def _transaction(self):
        with self._lock:
            if getattr(self._transaction_local, "db", None) is not None:
                yield self._transaction_local.db
                return
            with self.store.transaction() as db:
                self._transaction_local.db = db
                try:
                    self._tasks = {key: json.loads(raw) for key, raw in db.execute("SELECT task_key,receipt FROM image_requests")}
                    yield db
                finally:
                    self._transaction_local.db = None

    def resource_occupancy(self) -> dict:
        """Internal aggregate only: keep unfinished original receipts after restart."""
        with self._transaction():
            held = {}
            unattributed = 0
            for task in self._tasks.values():
                if not _holds_upstream_slot(task):
                    continue
                identity = str(task.get("provider_account_identity") or "")
                if identity:
                    held[identity] = held.get(identity, 0) + 1
                else:
                    unattributed += 1
            return {"by_account": held, "unattributed": unattributed}

    def submit_generation(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        quality: str = "auto",
        base_url: str = "",
        provider_binding_id: str = "",
        provider_account_identity: str = "",
        client_conversation_id: str = "",
        conversation_id: str = "",
        parent_message_id: str = "",
        retain_conversation: bool = False,
        upstream_model: str = "",
        image_thread_id: str = "",
        edit_source_task_id: str = "",
        edit_source_index: int = 0,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "model": model,
            "n": 1,
            "size": size,
            "quality": quality,
            "response_format": "url",
            "base_url": base_url,
            "provider_binding_id": provider_binding_id,
            "provider_account_identity": provider_account_identity,
            "client_conversation_id": client_conversation_id,
            "conversation_id": conversation_id,
            "parent_message_id": parent_message_id,
            "retain_conversation": retain_conversation,
            "upstream_model": upstream_model,
            **({"image_thread_id": image_thread_id, "edit_source_task_id": edit_source_task_id,
                "edit_source_index": edit_source_index} if image_thread_id or edit_source_task_id or edit_source_index else {}),
        }
        return self._submit(identity, client_task_id=client_task_id, mode="generate", payload=payload)

    def submit_edit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        quality: str = "auto",
        base_url: str = "",
        images: list[tuple[bytes, str, str]] | None = None,
        masks: list[tuple[bytes, str, str]] | None = None,
        provider_binding_id: str = "",
        provider_account_identity: str = "",
        client_conversation_id: str = "",
        conversation_id: str = "",
        parent_message_id: str = "",
        retain_conversation: bool = False,
        upstream_model: str = "",
        image_thread_id: str = "",
        edit_source_task_id: str = "",
        edit_source_index: int = 0,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "images": images or [],
            "mask": masks or [],
            "model": model,
            "n": 1,
            "size": size,
            "quality": quality,
            "response_format": "url",
            "base_url": base_url,
            "provider_binding_id": provider_binding_id,
            "provider_account_identity": provider_account_identity,
            "client_conversation_id": client_conversation_id,
            "conversation_id": conversation_id,
            "parent_message_id": parent_message_id,
            "retain_conversation": retain_conversation,
            "upstream_model": upstream_model,
            **({"image_thread_id": image_thread_id, "edit_source_task_id": edit_source_task_id,
                "edit_source_index": edit_source_index} if image_thread_id or edit_source_task_id or edit_source_index else {}),
        }
        return self._submit(identity, client_task_id=client_task_id, mode="edit", payload=payload)

    def list_tasks(self, identity: dict[str, object], task_ids: list[str]) -> dict[str, Any]:
        owner = _owner_id(identity)
        requested_ids = [_clean(task_id) for task_id in task_ids if _clean(task_id)]
        with self._transaction():
            if self._cleanup_locked():
                self._save_locked()
            items = []
            missing_ids = []
            for task_id in requested_ids:
                task = self._tasks.get(_task_key(owner, task_id))
                if task is None:
                    missing_ids.append(task_id)
                else:
                    items.append(_public_task(task))
            if not requested_ids:
                items = [
                    _public_task(task)
                    for task in self._tasks.values()
                    if task.get("owner_id") == owner
                ]
                items.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
                missing_ids = []
            return {"items": items, "missing_ids": missing_ids}

    def _submit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        mode: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        task_id = _clean(client_task_id)
        if not task_id:
            raise ValueError("client_task_id is required")
        owner = _owner_id(identity)
        key = _task_key(owner, task_id)
        now = _now_iso()
        should_start = False
        thread_fields = input_fields(payload)
        if thread_fields and self.admission is None:
            raise ImageThreadError("IMAGE_THREAD_SCHEDULER_UNAVAILABLE", status=503)
        source_snapshot = None
        source_bytes = None
        if thread_fields.get("edit_source_task_id"):
            # Existing IDs recover without rereading a potentially offline source.
            # Any remote-backed stored-image read is outside the SQLite write lock.
            with self.store.connect() as db:
                duplicate = self.store.read_receipt(db, "image", owner, task_id)
                source_snapshot = self.store.read_receipt(db, "image", owner, thread_fields["edit_source_task_id"])
            if duplicate is None:
                if source_snapshot is None:
                    raise ImageThreadError("IMAGE_THREAD_SOURCE_UNAVAILABLE")
                source_bytes = saved_image_bytes(source_snapshot)
        def read_source(source):
            if source_bytes is None or source_fingerprint(source) != source_fingerprint(source_snapshot):
                raise ImageThreadError("IMAGE_THREAD_SOURCE_CHANGED")
            return source_bytes
        with self._transaction():
            cleaned = self._cleanup_locked()
            task = self._tasks.get(key)
            if task is not None:
                if task.get("request_hash") and task.get("request_hash") != _request_hash(mode, payload):
                    raise ValueError("client_task_id already exists with a different immutable request")
                if cleaned:
                    self._save_locked()
                return _public_task(task)
            task = {
                "id": task_id,
                "owner_id": owner,
                "retain_receipt": bool(identity.get("external_image_client")),
                "status": TASK_STATUS_QUEUED,
                "mode": mode,
                "model": _clean(payload.get("model"), "gpt-image-2"),
                "upstream_model": _clean(payload.get("upstream_model")),
                "size": _clean(payload.get("size")),
                "quality": _clean(payload.get("quality"), "auto"),
                "base_url": _clean(payload.get("base_url")),
                "created_at": now,
                "updated_at": now,
                "created_ts": time.time(),
                "provider_binding_id": _clean(payload.get("provider_binding_id")),
                "provider_account_identity": _clean(payload.get("provider_account_identity")),
                "client_conversation_id": _clean(payload.get("client_conversation_id")),
                "conversation_id": _clean(payload.get("conversation_id")),
                "parent_message_id": _clean(payload.get("parent_message_id")),
                "binding_status": "bound" if payload.get("provider_binding_id") else "unbound",
                "request_hash": _request_hash(mode, payload),
                "upstream_unfinished": False,
                "admission_recorded": True,
                "_input_ref": None,
                "_sequence": self.store.next_sequence(self._transaction_local.db),
                "_source": str(identity.get("_trusted_source") or "key:" + owner),
                "_input_bytes": __import__("services.text_task_service", fromlist=["_retained_size"])._retained_size(payload),
                "_submission_started": False,
                "_turn_reserved": False,
                "_execution_timeline": [{"stage": "accepted", "at": time.time()}],
            }
            accept_thread(task, self._tasks.values(), payload, mode, output_reader=read_source)
            task["_input_ref"] = self.store.save_input({"payload": payload, "identity": {k: identity[k] for k in ("id", "name", "role", "external_image_client", "_trusted_source") if k in identity}, "mode": mode})
            self._tasks[key] = task
            self._save_locked()
            should_start = True

        if should_start and self.admission is not None:
            self.admission.wake()
        elif should_start:
            thread = threading.Thread(
                target=self._run_task,
                args=(key, mode, payload, dict(identity), _clean(payload.get("model"), "gpt-image-2")),
                name=f"image-task-{task_id[:16]}",
                daemon=True,
            )
            thread.start()
        return _public_task(task)

    def _run_task(
        self,
        key: str,
        mode: str,
        payload: dict[str, Any],
        identity: dict[str, object],
        model: str,
    ) -> None:
        if identity.get("external_image_client") and not payload.get("_admission_claim"):
            # Allocate once for this newly persisted task, before any generation.
            # Duplicate submissions never enter this thread. Account selection
            # only performs readiness reads; a failure here is NOT submitted.
            from services.account_service import account_service
            token = ""
            try:
                capacities = {str(item.get("provider_account_identity") or ""): min(
                    max(1, int(config.image_account_concurrency)), max(0, int(item.get("quota") or 0)))
                    for item in account_service.list_accounts()}
                with self._transaction():
                    held = {}
                    for other in self._tasks.values():
                        identity_id = str(other.get("provider_account_identity") or "")
                        if identity_id and _holds_upstream_slot(other):
                            held[identity_id] = held.get(identity_id, 0) + 1
                    unavailable = {identity_id for identity_id, count in held.items()
                                   if count >= capacities.get(identity_id, max(1, int(config.image_account_concurrency)))}
                binding, account_identity, token = account_service.create_conversation_binding(
                    image_model=model, excluded_account_identities=unavailable)
                payload = {**payload, "provider_binding_id": binding,
                           "provider_account_identity": account_identity,
                           "client_conversation_id": "image-task-" + uuid.uuid4().hex,
                           "retain_conversation": True}
                # Recheck after selection: another task can finish selection
                # while this selector waits for an account slot. Persist our
                # durable occupancy before releasing the temporary slot.
                selected = account_service.get_account(token) or {}
                capacity = min(max(1, int(config.image_account_concurrency)),
                               max(0, int(selected.get("quota") or 0)))
                with self._transaction():
                    occupied = sum(1 for other_key, other in self._tasks.items()
                                   if other_key != key and other.get("provider_account_identity") == account_identity
                                   and _holds_upstream_slot(other))
                    if occupied >= capacity:
                        raise RuntimeError("resource became occupied before submission")
                    self._update_task(key, provider_binding_id=binding,
                                      provider_account_identity=account_identity,
                                      client_conversation_id=payload["client_conversation_id"],
                                      binding_status="bound", upstream_unfinished=True)
            except Exception:
                self._update_task(key, status=TASK_STATUS_ERROR, error_code="IMAGE_RESOURCE_UNAVAILABLE",
                                  error="No available resource for the selected durable image route",
                                  upstream_unfinished=False)
                return
            finally:
                if token:
                    account_service.release_image_slot(token)

        # Persist account admission before calling the handler. A query timeout
        # or process restart must not make another upstream generation fit.
        account = _clean(payload.get("provider_account_identity"))
        if account and not identity.get("external_image_client") and not payload.get("_admission_claim"):
            while True:
                with self._transaction():
                    occupied = sum(1 for other_key, task in self._tasks.items()
                                   if other_key != key and task.get("provider_account_identity") == account
                                   and _holds_upstream_slot(task))
                    if occupied < max(1, int(config.image_account_concurrency)):
                        self._update_task(key, upstream_unfinished=True)
                        break
                # Legacy injected executors also wait without holding SQLite.
                with self._slot_condition:
                    self._slot_condition.wait(timeout=0.1)
        started = time.time()
        with self._transaction():
            current = self._tasks.get(key) or {}
            active_started_at = current.get("active_attempt_started_at")
            active_deadline_at = current.get("active_attempt_deadline_at")
        # Shared external routes can swap handlers and therefore do not infer
        # coverage from their normalized payload. Their receipt changes only
        # when the active bound-Chat attempt propagates an explicit marker.
        submission_boundary_covered = (
            not bool(identity.get("external_image_client"))
            and bool(_clean(payload.get("provider_binding_id")))
            and bool(_clean(payload.get("provider_account_identity")))
            and bool(_clean(payload.get("client_conversation_id")))
            and bool(payload.get("retain_conversation"))
            and not is_codex_image_model(model)
        )
        if submission_boundary_covered:
            self._update_task(key, upstream_submission_started=False)
        self._update_task(key, status=TASK_STATUS_RUNNING, error="")
        with self._transaction():
            task = self._tasks.get(key) or {}
            request_message_id = _clean(task.get("request_message_id"))
        if not request_message_id:
            request_message_id = str(uuid.uuid4())
            self._update_task(key, request_message_id=request_message_id)
        # 创建进度回调，每个步骤完成后更新任务状态
        def progress_callback(step: str) -> None:
            self._update_task(key, progress=step)
        progress_callback.request_message_id = request_message_id
        progress_callback.active_deadline_at = (
            float(active_deadline_at)
            if isinstance(active_deadline_at, (int, float))
            and not isinstance(active_deadline_at, bool)
            and active_deadline_at > 0
            else None
        )

        def start_active_attempt() -> float:
            nonlocal active_started_at, active_deadline_at
            with self._transaction():
                current = self._tasks.get(key) or {}
                saved_start = current.get("active_attempt_started_at")
                saved_deadline = current.get("active_attempt_deadline_at")
            if (
                not isinstance(saved_start, (int, float))
                or isinstance(saved_start, bool)
                or saved_start <= 0
            ):
                saved_start = time.time()
            if (
                not isinstance(saved_deadline, (int, float))
                or isinstance(saved_deadline, bool)
                or saved_deadline <= 0
            ):
                saved_deadline = float(saved_start) + IMAGE_ACTIVE_BUDGET_SECONDS
            active_started_at = float(saved_start)
            active_deadline_at = float(saved_deadline)
            progress_callback.active_deadline_at = active_deadline_at
            self._update_task(
                key,
                active_attempt_started_at=active_started_at,
                active_attempt_deadline_at=active_deadline_at,
                started_ts=active_started_at,
            )
            return active_deadline_at

        progress_callback.start_active_attempt = start_active_attempt

        def record_conversation_id(conversation_id: str) -> None:
            conversation_id = _clean(conversation_id)
            if conversation_id:
                self._update_task(key, conversation_id=conversation_id)
        progress_callback.record_conversation_id = record_conversation_id

        def record_submission_started() -> None:
            self._update_task(key, upstream_submission_started=True)
        progress_callback.record_submission_started = record_submission_started

        def record_result_ids(file_ids: list[str], sediment_ids: list[str]) -> None:
            self._update_task(
                key,
                result_file_ids=list(dict.fromkeys(str(item) for item in file_ids if item)),
                result_sediment_ids=list(dict.fromkeys(str(item) for item in sediment_ids if item)),
                progress="receiving_image",
                upstream_outcome="generated",
                upstream_unfinished=False,
            )
        progress_callback.record_result_ids = record_result_ids
        progress_callback.image_thread = payload.get("_image_thread")
        progress_callback.image_thread_predecessor_message = payload.get("_image_thread_predecessor_message")
        # 将进度回调添加到 payload 中（handler 会提取并传递给 ConversationRequest）
        payload_with_progress = {**payload, "progress_callback": progress_callback}
        try:
            handler = self.edit_handler if mode == "edit" else self.generation_handler
            result = handler(payload_with_progress)
            if not isinstance(result, dict):
                raise RuntimeError("image task returned streaming result unexpectedly")
            data = result.get("data")
            account_email = _clean(result.get("_account_email") or result.get("account_email"))
            provider_binding_id = _clean(result.get("_provider_binding_id"))
            provider_account_identity = _clean(result.get("_provider_account_identity"))
            conversation_id = _clean(result.get("_conversation_id"))
            parent_message_id = _clean(result.get("_parent_message_id"))
            if not isinstance(data, list) or not data:
                upstream = _clean(result.get("message"))
                if upstream:
                    message = upstream
                else:
                    message = "号池中没有可用账号或所有账号均被限流，请检查号池状态（账号额度、是否被封禁、是否到达生图上限）"
                error = RuntimeError(message)
                if account_email:
                    setattr(error, "account_email", account_email)
                raise error
            usage = result.get("usage")
            duration_ms = int((time.time() - started) * 1000)
            expected_binding_id = _clean(payload.get("provider_binding_id"))
            expected_account_identity = _clean(payload.get("provider_account_identity"))
            if expected_binding_id and provider_binding_id != expected_binding_id:
                raise RuntimeError("bound image result changed provider binding identity")
            if expected_account_identity and provider_account_identity != expected_account_identity:
                raise RuntimeError("bound image result changed provider account identity")
            if payload.get("_image_thread") and (len(data) != 1 or not result.get("_image_thread_terminal") or (payload.get("conversation_id") and payload["conversation_id"] != conversation_id)):
                raise ImageThreadError("IMAGE_THREAD_TURN_UNCONFIRMED", submitted=True)
            if (
                bool(provider_binding_id) != bool(provider_account_identity)
                or bool(provider_binding_id) != bool(conversation_id)
                or bool(provider_binding_id) != bool(parent_message_id)
            ):
                raise RuntimeError("bound image result is missing authoritative conversation state")
            self._update_task(
                key,
                status=TASK_STATUS_SUCCESS,
                data=data,
                usage=usage,
                error="",
                duration_ms=duration_ms,
                provider_binding_id=provider_binding_id,
                provider_account_identity=provider_account_identity,
                conversation_id=conversation_id,
                parent_message_id=parent_message_id,
                binding_status="bound" if provider_binding_id else "unbound",
                **({"_image_thread_terminal": True} if payload.get("_image_thread") else {}),
                upstream_unfinished=False,
                upstream_outcome="generated",
                recovery_error_code="",
                recovery_phase="",
                recovery_retry_after_seconds=None,
                next_poll_at=0,
            )
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用完成",
                request_preview=request_text(payload.get("prompt")),
                urls=_collect_image_urls(data),
                account_email=account_email,
            )
        except Exception as exc:
            error_message = str(exc) or "image task failed"
            with self._transaction():
                current = dict(self._tasks.get(key) or {})
            account_email = _clean(getattr(exc, "account_email", ""))
            conversation_id = _clean(
                getattr(exc, "conversation_id", "") or current.get("conversation_id")
            )
            provider_binding_id = _clean(
                getattr(exc, "provider_binding_id", "")
                or current.get("provider_binding_id")
                or payload.get("provider_binding_id")
            )
            provider_account_identity = _clean(
                getattr(exc, "provider_account_identity", "")
                or current.get("provider_account_identity")
                or payload.get("provider_account_identity")
            )
            parent_message_id = _clean(
                getattr(exc, "parent_message_id", "") or current.get("parent_message_id")
            )
            request_message_id = _clean(
                getattr(exc, "request_message_id", "") or current.get("request_message_id")
            )
            result_captured = bool(
                current.get("result_file_ids") or current.get("result_sediment_ids")
            )
            error_code = _clean(getattr(exc, "code", ""))
            upstream_submitted = getattr(exc, "upstream_submitted", None)
            known_not_submitted = upstream_submitted is False
            retryable_not_submitted = (
                known_not_submitted and error_code == "IMAGE_GENERATION_NOT_SUBMITTED"
            )
            # Only explicit terminal rejections prove there is no generation
            # left upstream. An unclassified transport exception does not.
            terminal = error_code.lower() in {
                "no_image_generated", "content_policy_violation",
                "conversation_binding_contract_invalid",
            }
            if known_not_submitted:
                terminal = True
            if retryable_not_submitted:
                error_code = "RESULT_UNRECOVERABLE"
            if account and not terminal:
                error_code = "CONVERSATION_OUTCOME_UNKNOWN"
            recovery_phase = (
                "download_image_result" if result_captured else "read_image_request"
            )
            recovery_error_code = _recovery_failure_code(exc, recovery_phase)
            retry_after = _retry_after_seconds(exc)
            if error_code == "CONVERSATION_OUTCOME_UNKNOWN":
                error_message = _safe_recovery_error(recovery_error_code, recovery_phase)
            duration_ms = int((time.time() - started) * 1000)
            self._update_task(key, status=TASK_STATUS_ERROR, error=error_message, data=[],
                              duration_ms=duration_ms,
                              upstream_unfinished=bool(account) and not terminal and not result_captured,
                              **(
                                  {
                                      "recovery_error_code": recovery_error_code,
                                      "recovery_phase": recovery_phase,
                                      "recovery_retry_after_seconds": retry_after,
                                      "next_poll_at": time.time() + (
                                          retry_after if retry_after is not None else 0
                                      ),
                                      **({"upstream_outcome": "generated"} if result_captured else {}),
                                  }
                                  if error_code == "CONVERSATION_OUTCOME_UNKNOWN"
                                  else {
                                      "recovery_error_code": "",
                                      "recovery_phase": "",
                                      "recovery_retry_after_seconds": None,
                                  }
                              ),
                              **(
                                  {
                                      "upstream_submission_started": False,
                                      "upstream_outcome": "not_submitted",
                                      **(
                                          {
                                              "recovery_retryable": True,
                                              "recovery_requires_new_conversation": False,
                                          }
                                          if retryable_not_submitted else {}
                                      ),
                                  }
                                  if known_not_submitted else {}
                              ),
                              **({"provider_binding_id": provider_binding_id} if provider_binding_id else {}),
                              **({"provider_account_identity": provider_account_identity} if provider_account_identity else {}),
                              **({"conversation_id": conversation_id} if conversation_id else {}),
                              **({"parent_message_id": parent_message_id} if parent_message_id else {}),
                              **({"request_message_id": request_message_id} if request_message_id else {}),
                              **(
                                  {
                                      "binding_status": (
                                          "bound"
                                          if conversation_id and parent_message_id
                                          else (
                                              "unknown"
                                              if error_code == "CONVERSATION_OUTCOME_UNKNOWN"
                                              else "unavailable"
                                          )
                                      )
                                  }
                                  if provider_binding_id
                                  else {}
                              ),
                              **({"error_code": error_code} if error_code else {}))
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用失败",
                request_preview=request_text(payload.get("prompt")),
                status="failed",
                error=error_message,
                account_email=account_email,
            )

    def _log_call(
        self,
        identity: dict[str, object],
        mode: str,
        model: str,
        started: float,
        suffix: str,
        *,
        request_preview: str = "",
        status: str = "success",
        error: str = "",
        urls: list[str] | None = None,
        account_email: str = "",
    ) -> None:
        endpoint = "/v1/images/edits" if mode == "edit" else "/v1/images/generations"
        summary_prefix = "图生图" if mode == "edit" else "文生图"
        detail = {
            "key_id": identity.get("id"),
            "key_name": identity.get("name"),
            "role": identity.get("role"),
            "endpoint": endpoint,
            "model": model,
            "started_at": datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": _now_iso(),
            "duration_ms": int((time.time() - started) * 1000),
            "status": status,
        }
        if request_preview:
            detail["request_text"] = request_preview
        if error:
            detail["error"] = error
        if account_email:
            detail["account_email"] = account_email
        if urls:
            detail["urls"] = list(dict.fromkeys(urls))
        try:
            log_service.add(LOG_TYPE_CALL, f"{summary_prefix}{suffix}", detail)
        except Exception:
            pass

    def _update_task(self, key: str, **updates: Any) -> None:
        with self._transaction():
            task = self._tasks.get(key)
            if task is None:
                return
            context = current_request.get()
            if context is not None and context.kind == "image" and key == _task_key(context.owner, context.request_id) and task.get("_claim_id") != context.claim:
                raise AdmissionLost("original image claim changed")
            if task.get("status") == TASK_STATUS_SUCCESS and updates.get("status") not in (None, TASK_STATUS_SUCCESS):
                return
            task.update(updates)
            task["updated_at"] = _now_iso()
            task["updated_ts"] = time.time()
            self._save_locked()
            self._slot_condition.notify_all()

    def _load_locked(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw_items = raw.get("tasks") if isinstance(raw, dict) else raw
        if not isinstance(raw_items, list):
            return {}
        tasks: dict[str, dict[str, Any]] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            task_id = _clean(item.get("id"))
            owner = _clean(item.get("owner_id"))
            if not task_id or not owner:
                continue
            status = _clean(item.get("status"))
            if status not in {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING, TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}:
                status = TASK_STATUS_ERROR
            task = {
                "id": task_id,
                "owner_id": owner,
                "status": status,
                "mode": "edit" if item.get("mode") == "edit" else "generate",
                "model": _clean(item.get("model"), "gpt-image-2"),
                "upstream_model": _clean(item.get("upstream_model")),
                "size": _clean(item.get("size")),
                "quality": _clean(item.get("quality"), "auto"),
                "created_at": _clean(item.get("created_at"), _now_iso()),
                "updated_at": _clean(item.get("updated_at"), _clean(item.get("created_at"), _now_iso())),
                "created_ts": item.get("created_ts"),
                "updated_ts": item.get("updated_ts"),
                "started_ts": item.get("started_ts"),
                "active_attempt_started_at": item.get("active_attempt_started_at"),
                "active_attempt_deadline_at": item.get("active_attempt_deadline_at"),
                "progress": item.get("progress"),
                "duration_ms": item.get("duration_ms"),
                "provider_binding_id": _clean(item.get("provider_binding_id")),
                "provider_account_identity": _clean(item.get("provider_account_identity")),
                "client_conversation_id": _clean(item.get("client_conversation_id")),
                "conversation_id": _clean(item.get("conversation_id")),
                "parent_message_id": _clean(item.get("parent_message_id")),
                "request_message_id": _clean(item.get("request_message_id")),
                "binding_status": _clean(item.get("binding_status"), "unbound"),
                "error_code": _clean(item.get("error_code")),
                "request_hash": _clean(item.get("request_hash")),
                "upstream_unfinished": _holds_upstream_slot(item),
                "upstream_submission_started": (
                    item.get("upstream_submission_started")
                    if isinstance(item.get("upstream_submission_started"), bool)
                    else None
                ),
                "admission_recorded": item.get("admission_recorded") is True,
                "retain_receipt": item.get("retain_receipt") is True,
                "next_poll_at": item.get("next_poll_at", 0),
                "poll_failures": item.get("poll_failures", 0),
                "recovery_no_result_reads": item.get("recovery_no_result_reads", 0),
                "upstream_outcome": _clean(item.get("upstream_outcome")),
                "recovery_retryable": item.get("recovery_retryable") is True,
                "recovery_requires_new_conversation": item.get("recovery_requires_new_conversation") is True,
                "recovery_error_code": _clean(item.get("recovery_error_code")),
                "recovery_phase": _clean(item.get("recovery_phase")),
                "recovery_retry_after_seconds": item.get("recovery_retry_after_seconds"),
                "deadline_recovery_started": item.get("deadline_recovery_started") is True,
                "result_file_ids": [
                    _clean(value) for value in item.get("result_file_ids", [])
                    if _clean(value)
                ] if isinstance(item.get("result_file_ids"), list) else [],
                "result_sediment_ids": [
                    _clean(value) for value in item.get("result_sediment_ids", [])
                    if _clean(value)
                ] if isinstance(item.get("result_sediment_ids"), list) else [],
                "adopted_source_request_message_id": _clean(item.get("adopted_source_request_message_id")),
                "adopted_source_image_message_id": _clean(item.get("adopted_source_image_message_id")),
                "adopted_from_error_code": _clean(item.get("adopted_from_error_code")),
                "adopted_from_error": _clean(item.get("adopted_from_error")),
                "adopted_at": _clean(item.get("adopted_at")),
            }
            data = item.get("data")
            if isinstance(data, list):
                task["data"] = data
            usage = item.get("usage")
            if isinstance(usage, dict):
                task["usage"] = usage
            error = _clean(item.get("error"))
            if error:
                task["error"] = error
            if (
                task.get("binding_status") == "unknown"
                and task.get("error_code") == "CONVERSATION_OUTCOME_UNKNOWN"
                and any(
                    marker in error
                    for marker in (
                        "/backend-api/f/conversation failed: status=403",
                        "/backend-api/conversation failed: status=403",
                    )
                )
            ):
                task["binding_status"] = "unavailable"
                task["error_code"] = "CONVERSATION_BINDING_UNAVAILABLE"
            tasks[_task_key(owner, task_id)] = task
        return tasks

    def _save_locked(self) -> None:
        db = getattr(self._transaction_local, "db", None)
        if db is None:
            # Compatibility for existing maintenance/tests using the local lock.
            with self.store.transaction() as db:
                for task in self._tasks.values():
                    self.store.write_receipt(db, "image", task["owner_id"], task["id"], task)
            return
        for task in self._tasks.values():
            self.store.write_receipt(db, "image", task["owner_id"], task["id"], task)

    def _recover_unfinished_locked(self) -> bool:
        changed = False
        for task in self._tasks.values():
            if task.get("_recovery_suppressed") is True:
                continue
            if task.get("_input_ref") and (task.get("status") == TASK_STATUS_QUEUED or task.get("_claim_id")):
                # The shared scheduler fences expired claims. Starting a second
                # worker must never interrupt another worker's active request.
                continue
            if task.get("status") in UNFINISHED_STATUSES:
                result_captured = bool(
                    task.get("result_file_ids") or task.get("result_sediment_ids")
                )
                known_not_submitted = task.get("upstream_submission_started") is False
                not_started = (task.get("status") == TASK_STATUS_QUEUED
                               and task.get("admission_recorded") is True
                               and task.get("upstream_unfinished") is False)
                task["status"] = TASK_STATUS_ERROR
                task["error"] = "服务已重启，未完成的图片任务已中断"
                task["error_code"] = (
                    "RESULT_UNRECOVERABLE" if known_not_submitted
                    else "IMAGE_TASK_NOT_STARTED" if not_started
                    else "CONVERSATION_OUTCOME_UNKNOWN"
                )
                if known_not_submitted:
                    task["upstream_unfinished"] = False
                    task["upstream_outcome"] = "not_submitted"
                    task["recovery_retryable"] = True
                    task["recovery_requires_new_conversation"] = False
                elif result_captured:
                    task["upstream_unfinished"] = False
                    task["upstream_outcome"] = "generated"
                    task["recovery_error_code"] = "RECOVERY_DOWNLOAD_FAILED"
                    task["recovery_phase"] = "download_image_result"
                    task["next_poll_at"] = 0
                elif task["error_code"] == "CONVERSATION_OUTCOME_UNKNOWN":
                    task["recovery_error_code"] = "RECOVERY_READ_FAILED"
                    task["recovery_phase"] = "read_image_request"
                    deadline = task.get("active_attempt_deadline_at")
                    if (
                        isinstance(deadline, (int, float))
                        and not isinstance(deadline, bool)
                        and deadline > 0
                        and time.time() >= float(deadline)
                    ):
                        # A process exit during the first post-deadline read
                        # must not strand the receipt behind its old schedule.
                        task["deadline_recovery_started"] = False
                        task["next_poll_at"] = 0
                if task.get("provider_binding_id"):
                    task["binding_status"] = (
                        "bound"
                        if known_not_submitted and task.get("conversation_id") and task.get("parent_message_id")
                        else "unavailable" if not_started or known_not_submitted else "unknown"
                    )
                task["updated_at"] = _now_iso()
                changed = True
        return changed

    def _cleanup_locked(self) -> bool:
        try:
            retention_days = max(1, int(self.retention_days_getter()))
        except Exception:
            retention_days = 30
        cutoff = time.time() - retention_days * 86400
        removed_keys = [
            key
            for key, task in self._tasks.items()
            if task.get("status") in TERMINAL_STATUSES and not _holds_upstream_slot(task)
            and not task.get("retain_receipt")
            and task.get("error_code") != "CONVERSATION_OUTCOME_UNKNOWN"
            and _timestamp(task.get("updated_at")) < cutoff
        ]
        for key in removed_keys:
            self._tasks.pop(key, None)
            self._transaction_local.db.execute("DELETE FROM image_requests WHERE task_key=?", (key,))
        return bool(removed_keys)

    def resume_poll(
        self,
        identity: dict[str, object],
        task_id: str,
        extra_timeout_secs: float = 30.0,
        base_url: str = "",
        allow_unrecoverable_retry: bool = False,
    ) -> dict[str, Any]:
        """恢复对已超时任务的轮询，额外等待 extra_timeout_secs 秒。"""
        owner = _owner_id(identity)
        key = _task_key(owner, _clean(task_id))
        with self._transaction():
            task = self._tasks.get(key)
            if task is None:
                raise ValueError("task not found")
            if task.get("_recovery_suppressed") is True:
                # Keep this endpoint observational while an operator has
                # explicitly stopped the old recovery path.
                return _public_task(task)
            if task.get("status") in {TASK_STATUS_RUNNING, TASK_STATUS_SUCCESS}:
                return _public_task(task)
            if task.get("status") != TASK_STATUS_ERROR:
                raise ValueError("task is not in error state")
            if _clean(task.get("error_code")) == "RESULT_UNRECOVERABLE":
                return _public_task(task)
            if _clean(task.get("error_code")) != "CONVERSATION_OUTCOME_UNKNOWN":
                raise ValueError("task outcome is not unknown")
            conversation_id = _clean(task.get("conversation_id"))
            if not conversation_id:
                raise ValueError("task has no conversation_id")
            if not _clean(task.get("request_message_id")):
                # Do not rotate this legacy UNKNOWN through a zero-duration
                # RUNNING/error cycle. There is no safe message boundary to
                # query, and selecting an ancestor, latest node, matching
                # prompt, or another task's request would risk attribution to a
                # different generation. The task remains recoverable when an
                # independently proven request_message_id is repaired in its
                # persisted receipt.
                if not allow_unrecoverable_retry:
                    return _public_task(task)
            deadline = task.get("active_attempt_deadline_at")
            deadline_expired = (
                isinstance(deadline, (int, float))
                and not isinstance(deadline, bool)
                and deadline > 0
                and time.time() >= float(deadline)
            )
            now = time.time()
            retry_after = task.get("recovery_retry_after_seconds")
            provider_cooldown_active = (
                _clean(task.get("recovery_error_code")) == "RECOVERY_RATE_LIMITED"
                and isinstance(retry_after, (int, float))
                and not isinstance(retry_after, bool)
                and retry_after >= 0
                and now < float(task.get("next_poll_at") or 0)
            )
            # The first post-deadline read may bypass a locally synthesized
            # backoff, but never a provider-supplied 429 Retry-After window.
            deadline_recovery_start = (
                deadline_expired
                and not task.get("deadline_recovery_started")
                and not provider_cooldown_active
            )
            if now < float(task.get("next_poll_at") or 0) and not deadline_recovery_start:
                return _public_task(task)
            mode = task.get("mode", "generate")
            model = task.get("model", "gpt-image-2")
            recovery_context = None
            if self.admission is not None:
                from services.pool_admission import ExecutionContext
                claim = uuid.uuid4().hex
                task.update(_claim_id=claim, _claim_until=self.admission.clock() + self.admission.CLAIM_SECONDS,
                            _submission_started=True, _executing=True, _turn_reserved=False)
                recovery_context = ExecutionContext(self.admission, "image", owner, _clean(task_id), claim)
            # 将任务状态重置为 running
            self._update_task(
                key,
                status=TASK_STATUS_RUNNING,
                error="",
                **({"deadline_recovery_started": True} if deadline_recovery_start else {}),
            )

        # 启动新线程继续轮询
        arguments = (key, conversation_id, extra_timeout_secs, base_url, dict(identity), mode, model,
                     bool(allow_unrecoverable_retry), bool(deadline_expired))
        thread = threading.Thread(
            target=self.admission.run_recovery if recovery_context is not None else self._run_resume_poll,
            args=(recovery_context, self._run_resume_poll, arguments) if recovery_context is not None else arguments,
            name=f"image-resume-{_clean(task_id)[:16]}",
            daemon=True,
        )
        thread.start()
        return _public_task(task)

    def adopt_latest_conversation_image(
        self,
        identity: dict[str, object],
        task_id: str,
        *,
        provider_binding_id: str,
        provider_account_identity: str,
        client_conversation_id: str,
        conversation_id: str,
        source_request_message_id: str = "",
        source_image_message_id: str = "",
        base_url: str = "",
    ) -> dict[str, Any]:
        """Adopt the latest completed manual image turn without generating again."""
        owner = _owner_id(identity)
        key = _task_key(owner, _clean(task_id))
        supplied_identity = tuple(map(_clean, (
            provider_binding_id,
            provider_account_identity,
            client_conversation_id,
            conversation_id,
        )))
        if not all(supplied_identity):
            raise ConversationImageAdoptionError("complete conversation authority is required")
        expected_source_request = _clean(source_request_message_id)
        expected_source_image = _clean(source_image_message_id)

        with self._transaction():
            task = self._tasks.get(key)
            if task is None:
                raise ConversationImageAdoptionError("task not found")
            stored_identity = tuple(_clean(task.get(field)) for field in (
                "provider_binding_id",
                "provider_account_identity",
                "client_conversation_id",
                "conversation_id",
            ))
            if stored_identity != supplied_identity:
                raise ConversationImageAdoptionError("conversation authority does not match the original task")
            if task.get("status") == TASK_STATUS_SUCCESS:
                if task.get("adopted_source_request_message_id"):
                    if (
                        expected_source_request
                        and expected_source_request
                        != _clean(task.get("adopted_source_request_message_id"))
                    ):
                        raise ConversationImageAdoptionError(
                            "specified manual request does not match the adopted image"
                        )
                    if (
                        expected_source_image
                        and expected_source_image
                        != _clean(task.get("adopted_source_image_message_id"))
                    ):
                        raise ConversationImageAdoptionError(
                            "specified image node does not match the adopted image"
                        )
                    return _public_task(task)
                raise ConversationImageAdoptionError("task already completed without manual image adoption")
            if task.get("status") != TASK_STATUS_ERROR:
                raise ConversationImageAdoptionError("task is not in a terminal error state")
            if _clean(task.get("error_code")).lower() != "content_policy_violation":
                raise ConversationImageAdoptionError("task is not eligible for manual image adoption")
            original_request_message_id = _clean(task.get("request_message_id"))
            if not original_request_message_id:
                raise ConversationImageAdoptionError("original request identity is unavailable")
            original_error_code = _clean(task.get("error_code"))
            original_error = _clean(task.get("error"))
            original_task_created_ts = task.get("created_ts")
            if not isinstance(original_task_created_ts, (int, float)):
                original_task_created_ts = _timestamp(task.get("created_at"))

        backend = None
        try:
            from services.account_service import account_service
            from services.openai_backend_api import OpenAIBackendAPI
            from services.protocol.conversation import format_image_result

            if account_service.get_bound_account_identity(provider_binding_id) != provider_account_identity:
                raise ConversationImageAdoptionError("provider account identity changed")
            access_token = account_service.get_bound_text_access_token(provider_binding_id, model="auto")
            with account_service.conversation_binding_lock(provider_binding_id, client_conversation_id):
                backend = OpenAIBackendAPI(access_token=access_token)
                document = backend._get_conversation(conversation_id)
                document_id = _clean(document.get("conversation_id") if isinstance(document, dict) else "")
                if document_id and document_id != conversation_id:
                    raise ConversationImageAdoptionError("conversation identity changed")
                source_request_id, image_record, current_node = _latest_completed_manual_image_turn(
                    document,
                    original_request_message_id,
                    float(original_task_created_ts or 0),
                    backend._extract_image_tool_records,
                )
                if expected_source_request and source_request_id != expected_source_request:
                    raise ConversationImageAdoptionError("specified manual request is not the latest completed image turn")
                if expected_source_image and _clean(image_record.get("message_id")) != expected_source_image:
                    raise ConversationImageAdoptionError("specified image node is not the latest completed image")
                tasks = backend._query_backend_tasks(
                    conversation_id=conversation_id,
                    timeout_secs=5.0,
                    strict_schema=True,
                )
                if _backend_tasks_may_be_active(tasks) or _document_current_message_active(document):
                    raise ConversationImageAdoptionError("latest manual request is still active")
                image_urls = backend.resolve_conversation_image_urls(
                    conversation_id,
                    list(image_record.get("file_ids") or []),
                    list(image_record.get("sediment_ids") or []),
                    poll=False,
                    request_message_id=source_request_id,
                )
                if not image_urls:
                    raise ConversationImageAdoptionError("latest manual image could not be resolved")
                image_bytes = backend.download_image_bytes(image_urls)
                if not image_bytes:
                    raise ConversationImageAdoptionError("latest manual image could not be downloaded")
                latest_document = backend._get_conversation(conversation_id)
                latest_document_id = _clean(
                    latest_document.get("conversation_id")
                    if isinstance(latest_document, dict) else ""
                )
                if latest_document_id and latest_document_id != conversation_id:
                    raise ConversationImageAdoptionError("conversation identity changed")
                latest_source_request, latest_image_record, latest_current_node = (
                    _latest_completed_manual_image_turn(
                        latest_document,
                        original_request_message_id,
                        float(original_task_created_ts or 0),
                        backend._extract_image_tool_records,
                    )
                )
                latest_tasks = backend._query_backend_tasks(
                    conversation_id=conversation_id,
                    timeout_secs=5.0,
                    strict_schema=True,
                )
                if (
                    _backend_tasks_may_be_active(latest_tasks)
                    or _document_current_message_active(latest_document)
                    or latest_source_request != source_request_id
                    or _clean(latest_image_record.get("message_id")) != _clean(image_record.get("message_id"))
                    or latest_current_node != current_node
                ):
                    raise ConversationImageAdoptionError("conversation changed during image adoption")
            image_items = [
                {"b64_json": __import__("base64").b64encode(item).decode("ascii")}
                for item in image_bytes
            ]
            data = format_image_result(
                image_items,
                "",
                "b64_json",
                base_url,
                int(time.time()),
            )["data"]
            if not data:
                raise ConversationImageAdoptionError("latest manual image could not be stored")

            with self._transaction():
                current = self._tasks.get(key)
                if current is None:
                    raise ConversationImageAdoptionError("task not found")
                if current.get("status") == TASK_STATUS_SUCCESS and current.get("adopted_source_request_message_id"):
                    if (
                        _clean(current.get("adopted_source_request_message_id"))
                        == source_request_id
                        and _clean(current.get("adopted_source_image_message_id"))
                        == _clean(image_record.get("message_id"))
                    ):
                        return _public_task(current)
                    raise ConversationImageAdoptionError(
                        "task was concurrently adopted from a different conversation image"
                    )
                current_identity = tuple(_clean(current.get(field)) for field in (
                    "provider_binding_id",
                    "provider_account_identity",
                    "client_conversation_id",
                    "conversation_id",
                ))
                if (
                    current.get("status") != TASK_STATUS_ERROR
                    or _clean(current.get("error_code")).lower() != "content_policy_violation"
                    or _clean(current.get("request_message_id")) != original_request_message_id
                    or current_identity != supplied_identity
                ):
                    raise ConversationImageAdoptionError("original task changed during image adoption")
                snapshot = dict(current)
                try:
                    current.update({
                        "status": TASK_STATUS_SUCCESS,
                        "data": data,
                        "error": "",
                        "error_code": "",
                        "binding_status": "bound",
                        "parent_message_id": current_node,
                        "upstream_unfinished": False,
                        "next_poll_at": 0,
                        "adopted_source_request_message_id": source_request_id,
                        "adopted_source_image_message_id": _clean(image_record.get("message_id")),
                        "adopted_from_error_code": original_error_code,
                        "adopted_from_error": original_error,
                        "adopted_at": _now_iso(),
                        "updated_at": _now_iso(),
                        "updated_ts": time.time(),
                    })
                    self._save_locked()
                except Exception:
                    current.clear()
                    current.update(snapshot)
                    raise
                self._slot_condition.notify_all()
                return _public_task(current)
        except ConversationImageAdoptionError:
            raise
        except Exception as exc:
            raise ConversationImageAdoptionError(
                "conversation image adoption could not be verified"
            ) from exc
        finally:
            if backend is not None:
                backend.close()

    def _run_resume_poll(
        self,
        key: str,
        conversation_id: str,
        extra_timeout_secs: float,
        base_url: str,
        identity: dict[str, object],
        mode: str,
        model: str,
        allow_unrecoverable_retry: bool,
        deadline_expired: bool,
    ) -> None:
        """后台线程：继续轮询已有 conversation_id 的图片结果。"""
        started = time.time()
        backend = None
        account_service = None
        access_token = ""
        conversation_available = False
        try:
            from services.account_service import account_service
            from services.openai_backend_api import ImageContentPolicyError, OpenAIBackendAPI
            from services.protocol.conversation import format_image_result

            with self._transaction():
                task = self._tasks.get(key)
                binding_id = _clean(task.get("provider_binding_id")) if task else ""
                account_identity = _clean(task.get("provider_account_identity")) if task else ""
                client_conversation_id = _clean(task.get("client_conversation_id")) if task else ""
                request_message_id = _clean(task.get("request_message_id")) if task else ""
                persisted_file_ids = list(task.get("result_file_ids") or []) if task else []
                persisted_sediment_ids = list(task.get("result_sediment_ids") or []) if task else []
                image_thread = (task or {}).get("_image_thread")
                expected_parent = (task or {}).get("_image_thread_request_parent")
            def recovered_parent(backend):
                if image_thread:
                    return finished_parent(backend._get_conversation(conversation_id), conversation_id, request_message_id, expected_parent=expected_parent)
                return backend.get_conversation_parent_message_id(conversation_id)
            if not binding_id or not account_identity or not client_conversation_id:
                error = RuntimeError("conversation binding unavailable: task authority missing")
                error.status_code = 403
                raise error
            authoritative_identity = account_service.get_bound_account_identity(binding_id)
            if authoritative_identity != account_identity:
                error = RuntimeError("conversation binding unavailable: account identity changed")
                error.status_code = 403
                raise error
            # Reading the original task consumes no new generation admission.
            access_token = account_service.get_bound_text_access_token(binding_id, model="auto")
            with account_service.conversation_binding_lock(binding_id, client_conversation_id):
                backend = OpenAIBackendAPI(access_token=access_token)
                if persisted_file_ids or persisted_sediment_ids:
                    self._update_task(key, progress="receiving_image", recovery_phase="download_image_result")
                    image_urls = backend.resolve_conversation_image_urls(
                        conversation_id,
                        persisted_file_ids,
                        persisted_sediment_ids,
                        poll=False,
                        request_message_id=request_message_id,
                    )
                    if not image_urls:
                        raise RuntimeError("generated image URL could not be resolved")
                    downloaded = backend.download_image_bytes(image_urls)
                    if not downloaded:
                        raise RuntimeError("generated image could not be downloaded")
                    image_items = [
                        {"b64_json": __import__("base64").b64encode(image_data).decode("ascii")}
                        for image_data in downloaded
                    ]
                    parent_message_id = recovered_parent(backend)
                    data = format_image_result(
                        image_items,
                        "",
                        "b64_json",
                        base_url,
                        int(time.time()),
                    )["data"]
                    self._update_task(
                        key,
                        status=TASK_STATUS_SUCCESS,
                        data=data,
                        error="",
                        error_code="",
                        binding_status="bound",
                        parent_message_id=parent_message_id,
                        **({"_image_thread_terminal": True} if image_thread else {}),
                        upstream_unfinished=False,
                        upstream_outcome="generated",
                        next_poll_at=0,
                        poll_failures=0,
                        recovery_no_result_reads=0,
                        recovery_requires_new_conversation=False,
                        recovery_error_code="",
                        recovery_phase="",
                        recovery_retry_after_seconds=None,
                        duration_ms=int((time.time() - started) * 1000),
                    )
                    self._log_call(
                        identity,
                        mode,
                        model,
                        started,
                        "调用完成（恢复下载）",
                        status="success",
                        urls=_collect_image_urls(data),
                    )
                    return
                try:
                    document = backend._get_conversation(conversation_id)
                    conversation_available = True
                except Exception as exc:
                    if _upstream_status_code(exc) != 404:
                        raise
                    if not (allow_unrecoverable_retry or deadline_expired):
                        raise
                    tasks = backend._query_backend_tasks(
                        conversation_id=conversation_id, timeout_secs=5.0, strict_schema=True,
                    )
                    if _backend_tasks_may_be_active(tasks):
                        raise RuntimeError("original conversation is missing while an upstream task may still be active") from exc
                    raise UnrecoverableRead(
                        "original conversation is missing after a bounded result read",
                        requires_new_conversation=True,
                    ) from exc
                if not request_message_id:
                    tasks = backend._query_backend_tasks(
                        conversation_id=conversation_id, timeout_secs=5.0, strict_schema=True,
                    )
                    if _document_current_message_active(document) or _backend_tasks_may_be_active(tasks):
                        raise RuntimeError("legacy image request may still be running upstream")
                    raise UnrecoverableRead(
                        "original legacy image result cannot be safely attributed after a bounded read",
                        requires_new_conversation=False,
                    )
                branch_state = _branch_read_state(document, request_message_id)
                authoritative_failure = _authoritative_image_failure(document, request_message_id)
                if authoritative_failure:
                    raise AuthoritativeImageTaskFailure(authoritative_failure)
                no_active_task = False
                if allow_unrecoverable_retry or deadline_expired:
                    tasks = backend._query_backend_tasks(
                        conversation_id=conversation_id, timeout_secs=5.0, strict_schema=True,
                    )
                    no_active_task = (
                        not _backend_tasks_may_be_active(tasks)
                        and not _document_current_message_active(document)
                    )

                def latest_unrecoverable_requirement() -> bool | None:
                    try:
                        latest_document = backend._get_conversation(conversation_id)
                    except Exception as latest_exc:
                        if _upstream_status_code(latest_exc) != 404:
                            raise
                        latest_tasks = backend._query_backend_tasks(
                            conversation_id=conversation_id,
                            timeout_secs=5.0,
                            strict_schema=True,
                        )
                        return True if not _backend_tasks_may_be_active(latest_tasks) else None
                    latest_tasks = backend._query_backend_tasks(
                        conversation_id=conversation_id,
                        timeout_secs=5.0,
                        strict_schema=True,
                    )
                    latest_state = _branch_read_state(latest_document, request_message_id)
                    if (
                        _document_current_message_active(latest_document)
                        or _backend_tasks_may_be_active(latest_tasks)
                        or latest_state not in {
                            "terminal_without_result", "unattributable", "no_result",
                        }
                    ):
                        return None
                    extract_records = getattr(backend, "_extract_image_tool_records", None)
                    if callable(extract_records):
                        for record in extract_records(latest_document, request_message_id):
                            if record.get("file_ids") or record.get("sediment_ids"):
                                return None
                    return False

                try:
                    file_ids, sediment_ids = backend._poll_image_results(
                        conversation_id,
                        extra_timeout_secs,
                        request_message_id=request_message_id,
                    )
                except Exception as exc:
                    if (
                        no_active_task
                        and branch_state in {"terminal_without_result", "unattributable", "no_result"}
                        and exc.__class__.__name__ == "ImagePollTimeoutError"
                    ):
                        latest_requirement = latest_unrecoverable_requirement()
                        if latest_requirement is not None:
                            raise UnrecoverableRead(
                                "original image branch has no attributable result after a bounded read",
                                requires_new_conversation=latest_requirement,
                            ) from exc
                    raise
                if not file_ids and not sediment_ids:
                    if (
                        no_active_task
                        and branch_state in {"terminal_without_result", "unattributable", "no_result"}
                    ):
                        latest_requirement = latest_unrecoverable_requirement()
                        if latest_requirement is not None:
                            raise UnrecoverableRead(
                                "original image branch has no attributable result after a bounded read",
                                requires_new_conversation=latest_requirement,
                            )
                    raise RuntimeError(
                        f"继续等待 {extra_timeout_secs} 秒后仍未找到图片结果。"
                    )

                self._update_task(
                    key,
                    result_file_ids=list(dict.fromkeys(file_ids)),
                    result_sediment_ids=list(dict.fromkeys(sediment_ids)),
                    progress="receiving_image",
                    upstream_outcome="generated",
                    upstream_unfinished=False,
                    recovery_phase="download_image_result",
                )

                image_urls = backend.resolve_conversation_image_urls(
                    conversation_id, file_ids, sediment_ids, poll=False,
                    request_message_id=request_message_id,
                )
                if not image_urls:
                    raise RuntimeError("图片 URL 解析失败")

                image_items = [
                    {"b64_json": __import__("base64").b64encode(image_data).decode("ascii")}
                    for image_data in backend.download_image_bytes(image_urls)
                ]
                parent_message_id = recovered_parent(backend)
            data = format_image_result(
                image_items,
                "",  # prompt 已不重要，结果已经拿到了
                "b64_json",
                base_url,
                int(time.time()),
            )["data"]
            self._update_task(
                key,
                status=TASK_STATUS_SUCCESS,
                data=data,
                error="",
                error_code="",
                binding_status="bound",
                parent_message_id=parent_message_id,
                **({"_image_thread_terminal": True} if image_thread else {}),
                upstream_unfinished=False,
                next_poll_at=0,
                poll_failures=0,
                recovery_no_result_reads=0,
                recovery_requires_new_conversation=False,
                recovery_error_code="",
                recovery_phase="",
                recovery_retry_after_seconds=None,
                upstream_outcome="generated",
                duration_ms=int((time.time() - started) * 1000),
            )
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用完成（续轮询）",
                status="success",
                urls=_collect_image_urls(data),
            )
        except Exception as exc:
            error_message = str(exc) or "resume poll failed"
            duration_ms = int((time.time() - started) * 1000)
            error_code = _clean(getattr(exc, "code", ""))
            with self._transaction():
                current = self._tasks.get(key, {})
                failures = int(current.get("poll_failures") or 0) + 1
                qualified_reads = int(current.get("recovery_no_result_reads") or 0)
                requires_new_conversation = bool(current.get("recovery_requires_new_conversation"))
                result_captured = bool(
                    current.get("result_file_ids") or current.get("result_sediment_ids")
                )
                qualified_read_recorded = False
                if conversation_available:
                    requires_new_conversation = False
                if (allow_unrecoverable_retry or deadline_expired) and isinstance(exc, UnrecoverableRead):
                    requires_new_conversation = exc.requires_new_conversation
                    if deadline_expired or _task_age_seconds(current) >= UNRECOVERABLE_MIN_AGE_SECONDS:
                        qualified_reads += 1
                        qualified_read_recorded = True
            if isinstance(exc, ImageContentPolicyError):
                error_code = "content_policy_violation"
            terminal = error_code in {"NO_IMAGE_GENERATED", "content_policy_violation"}
            unrecoverable = qualified_reads >= UNRECOVERABLE_QUALIFIED_READS
            recovery_phase = (
                "download_image_result" if result_captured else "read_image_request"
            )
            recovery_error_code = _recovery_failure_code(exc, recovery_phase)
            retry_after = _retry_after_seconds(exc)
            final_error_code = (
                "RESULT_UNRECOVERABLE" if unrecoverable
                else error_code if terminal else "CONVERSATION_OUTCOME_UNKNOWN"
            )
            if final_error_code == "CONVERSATION_OUTCOME_UNKNOWN":
                error_message = _safe_recovery_error(recovery_error_code, recovery_phase)
            self._update_task(
                key,
                status=TASK_STATUS_ERROR,
                error=error_message,
                error_code=final_error_code,
                binding_status=("unavailable" if unrecoverable and requires_new_conversation
                                else "bound" if terminal or unrecoverable else "unknown"),
                data=[],
                duration_ms=duration_ms,
                upstream_unfinished=not (terminal or unrecoverable or result_captured),
                poll_failures=failures,
                recovery_no_result_reads=qualified_reads,
                recovery_requires_new_conversation=requires_new_conversation,
                recovery_error_code=(
                    recovery_error_code
                    if final_error_code == "CONVERSATION_OUTCOME_UNKNOWN" else ""
                ),
                recovery_phase=(
                    recovery_phase
                    if final_error_code == "CONVERSATION_OUTCOME_UNKNOWN" else ""
                ),
                recovery_retry_after_seconds=(
                    retry_after
                    if final_error_code == "CONVERSATION_OUTCOME_UNKNOWN" else None
                ),
                **({"upstream_outcome": "unknown", "recovery_retryable": True}
                   if unrecoverable else {}),
                next_poll_at=(
                    0
                    if unrecoverable
                    else time.time() + (
                        retry_after
                        if retry_after is not None
                        else 5
                        if deadline_expired and qualified_read_recorded
                        else 30
                        if deadline_expired
                        else min(
                            900,
                            60 * 2 ** min(
                                (qualified_reads if qualified_read_recorded else failures) - 1,
                                4,
                            ),
                        )
                    )
                ),
            )
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用失败（续轮询）",
                status="failed",
                error=error_message,
            )
        finally:
            if backend is not None:
                backend.close()


image_task_service = ImageTaskService(DATA_DIR / "image_tasks.json")
