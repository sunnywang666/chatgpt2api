from __future__ import annotations

import json
import hashlib
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from services.account_service import account_service
from services.config import DATA_DIR
from services.content_filter import request_text
from services.log_service import LOG_TYPE_CALL, log_service
from services.openai_backend_api import EDITABLE_FILE_MODEL, OpenAIBackendAPI
from utils.helper import new_uuid

TASK_STATUS_QUEUED = "queued"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_SUCCESS = "success"
TASK_STATUS_ERROR = "error"
UNFINISHED_STATUSES = {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING}
EDITABLE_FILE_PLAN_TYPES = ("Plus", "Team", "Pro", "Enterprise")
EDITABLE_FILE_ROOT = DATA_DIR / "files"
EDITABLE_FILE_TASKS_PATH = DATA_DIR / "editable_file_tasks.json"


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _clean(value: object, default: str = "") -> str:
    return str(value or default).strip()


def _owner_id(identity: dict[str, object]) -> str:
    return _clean(identity.get("id")) or "anonymous"


def _task_key(owner_id: str, task_id: str) -> str:
    return f"{owner_id}:{task_id}"


def _elapsed_seconds(task: dict[str, Any]) -> int:
    start = float(task.get("started_ts") or task.get("created_ts") or 0)
    end = float(task.get("ended_ts") or time.time())
    return max(0, int(end - start)) if start else 0


def _file_url(path: Path, base_url: str) -> str:
    rel = path.resolve().relative_to(EDITABLE_FILE_ROOT.resolve()).as_posix()
    prefix = str(base_url or "").strip().rstrip("/")
    return f"{prefix}/files/{quote(rel, safe='/')}" if prefix else f"/files/{quote(rel, safe='/')}"


def _editable_access_token() -> str:
    from services.request_context import current_request, AdmissionLost
    context = current_request.get()
    if context is not None:
        account = context.selected_account()
        if account is None:
            raise AdmissionLost("original editable-file account is unavailable")
        token = _clean(account.get("access_token"))
        return account_service.refresh_access_token(token, event="editable_file_task") or token
    accounts = [
        item for item in account_service.list_accounts()
        if _clean(item.get("access_token"))
           and item.get("status") not in {"禁用", "异常"}
           and account_service._account_matches_any_plan_type(item, EDITABLE_FILE_PLAN_TYPES)
    ]
    if not accounts:
        raise RuntimeError("no available plus/team/pro account")
    accounts.sort(key=lambda item: _clean(item.get("last_used_at")))
    token = _clean(accounts[0].get("access_token"))
    return account_service.refresh_access_token(token, event="editable_file_task") or token


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    item = {
        "id": task.get("id"),
        "taskId": task.get("id"),
        "status": task.get("status"),
        "kind": task.get("kind"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
        "elapsed_seconds": _elapsed_seconds(task),
    }
    for key in ("result", "error"):
        if task.get(key):
            item[key] = task[key]
    return item


class EditableFileTaskService:
    def __init__(self, path: Path = EDITABLE_FILE_TASKS_PATH, *, text_tasks=None) -> None:
        self.path = path
        self.text_tasks = text_tasks
        self._lock = threading.RLock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._tasks = self._load_locked()
            if self._recover_unfinished_locked():
                self._save_locked()

    def _durable_tasks(self):
        if self.text_tasks is not None:
            return self.text_tasks if self.text_tasks.admission is not None else None
        from services.text_task_service import text_task_service
        return text_task_service if text_task_service.admission is not None else None

    @staticmethod
    def _durable_public(receipt):
        task = {**receipt, "id": receipt.get("_editable_task_id"), "kind": receipt.get("_editable_kind"),
                "status": {"succeeded": "success", "failed": "error", "unknown": "error"}.get(receipt["status"], receipt["status"])}
        result = _public_task(task)
        if receipt.get("error_code"):
            result["error_code"] = receipt["error_code"]
        for field in ("waiting", "rate_limit"):
            if receipt.get(field):
                result[field] = receipt[field]
        return result

    def submit_ppt(self, identity: dict[str, object], *, client_task_id: str = "", prompt: str = "", base64_images: list[str] | None = None, base_url: str = "") -> dict[str, Any]:
        return self._submit(identity, client_task_id=client_task_id, kind="ppt", prompt=prompt, base64_images=base64_images or [], base_url=base_url)

    def submit_psd(self, identity: dict[str, object], *, client_task_id: str = "", prompt: str = "", base64_images: list[str] | None = None, base_url: str = "") -> dict[str, Any]:
        return self._submit(identity, client_task_id=client_task_id, kind="psd", prompt=prompt, base64_images=base64_images or [], base_url=base_url)

    def list_tasks(self, identity: dict[str, object], task_ids: list[str]) -> dict[str, Any]:
        owner = _owner_id(identity)
        requested = [_clean(item) for item in task_ids if _clean(item)]
        durable = self._durable_tasks()
        if durable is not None:
            with self._lock:
                all_items = {task["id"]: _public_task(task) for task in self._tasks.values() if task.get("owner_id") == owner}
            with durable.store.connect() as db:
                for kind, row_owner, _, receipt in durable.store.receipts(db):
                    if kind == "text" and row_owner == owner and receipt.get("_editable_task_id"):
                        all_items[receipt["_editable_task_id"]] = self._durable_public(receipt)
            return {"items": [all_items[key] for key in (requested or list(all_items)) if key in all_items],
                    "missing_ids": [key for key in requested if key not in all_items]}
        with self._lock:
            if requested:
                items = [task for task_id in requested if (task := self._tasks.get(_task_key(owner, task_id)))]
                return {"items": [_public_task(item) for item in items], "missing_ids": [task_id for task_id in requested if _task_key(owner, task_id) not in self._tasks]}
            items = [task for task in self._tasks.values() if task.get("owner_id") == owner]
        items.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return {"items": [_public_task(item) for item in items], "missing_ids": []}

    def _submit(self, identity: dict[str, object], *, client_task_id: str, kind: str, prompt: str, base64_images: list[str], base_url: str) -> dict[str, Any]:
        task_id = _clean(client_task_id) or new_uuid()
        owner = _owner_id(identity)
        key = _task_key(owner, task_id)
        now = _now_iso()
        durable = self._durable_tasks()
        if durable is not None:
            with self._lock:
                if key in self._tasks:
                    return _public_task(self._tasks[key])  # Never recreate a legacy job.
            request_id = "editable-file-" + hashlib.sha256(task_id.encode()).hexdigest()
            body = {"client_request_id": request_id, "client_conversation_id": request_id,
                    "model": EDITABLE_FILE_MODEL,
                    "_editable": {"task_id": task_id, "kind": kind, "prompt": prompt, "base64_images": base64_images,
                                  "base_url": base_url, "identity": {k: identity[k] for k in ("id", "name", "role") if k in identity}}}
            durable.submit(owner, body, source=identity.get("_trusted_source"))
            return self.list_tasks(identity, [task_id])["items"][0]
        with self._lock:
            if key in self._tasks:
                return _public_task(self._tasks[key])
            ts = time.time()
            self._tasks[key] = {"id": task_id, "owner_id": owner, "status": TASK_STATUS_QUEUED, "kind": kind, "model": EDITABLE_FILE_MODEL, "created_at": now, "updated_at": now, "created_ts": ts, "updated_ts": ts}
            task = dict(self._tasks[key])
            self._save_locked()
        threading.Thread(target=self._run_task, args=(key, kind, prompt, base64_images, dict(identity), base_url), name=f"{kind}-file-task-{task_id[:16]}", daemon=True).start()
        return _public_task(task)

    def run_admitted(self, body):
        """One existing file export under the original shared text-task claim."""
        spec = body["_editable"]
        if spec["kind"] == "psd" and not spec["base64_images"]:
            from services.conversation_binding_service import ConversationBindingError
            raise ConversationBindingError("base64_images is empty", code="EDITABLE_FILE_INPUT_INVALID")
        started = time.time()
        token = _editable_access_token()
        backend = OpenAIBackendAPI(token)
        try:
            owner = _owner_id(spec["identity"])
            directory = hashlib.sha256((owner + ":" + spec["task_id"]).encode()).hexdigest()
            output_dir = EDITABLE_FILE_ROOT / spec["kind"] / directory
            export = backend.export_psd_zip if spec["kind"] == "psd" else backend.export_ppt_zip
            result = export(spec["base64_images"], spec["prompt"], output_dir)
            account_service.mark_text_used(token)
            data = {"conversation_id": result.conversation_id,
                    "primary_url": _file_url(result.primary_path, spec["base_url"]),
                    "zip_url": _file_url(result.zip_path, spec["base_url"])}
            self._log_call(spec["identity"], spec["kind"], started, request_text(spec["prompt"]), result=data)
            return {"result": data}
        except Exception:
            self._log_call(spec["identity"], spec["kind"], started, request_text(spec["prompt"]),
                           status="failed", error="original editable-file request did not finish")
            raise
        finally:
            backend.close()

    def _run_task(self, key: str, kind: str, prompt: str, base64_images: list[str], identity: dict[str, object], base_url: str) -> None:
        started = time.time()
        token = ""
        account_email = ""
        self._update_task(key, status=TASK_STATUS_RUNNING, error="", started_ts=started)
        backend = None
        try:
            if kind == "psd" and not base64_images:
                raise ValueError("base64_images is empty")
            token = _editable_access_token()
            account = account_service.get_account(token) or {}
            account_email = _clean(account.get("email"))
            backend = OpenAIBackendAPI(token)
            output_dir = EDITABLE_FILE_ROOT / kind / key.rsplit(":", 1)[-1]
            result = backend.export_psd_zip(base64_images, prompt, output_dir) if kind == "psd" else backend.export_ppt_zip(base64_images, prompt, output_dir)
            account_service.mark_text_used(token)
            data = {"conversation_id": result.conversation_id, "primary_url": _file_url(result.primary_path, base_url), "zip_url": _file_url(result.zip_path, base_url)}
            self._update_task(key, status=TASK_STATUS_SUCCESS, result=data, account_email=account_email, error="", ended_ts=time.time())
            self._log_call(identity, kind, started, request_text(prompt), account_email=account_email, result=data)
        except Exception as exc:
            error = str(exc) or "editable file task failed"
            self._update_task(key, status=TASK_STATUS_ERROR, error=error, account_email=account_email, ended_ts=time.time())
            self._log_call(identity, kind, started, request_text(prompt), status="failed", error=error, account_email=account_email)
        finally:
            if backend is not None:
                backend.close()

    def public_file_path(self, relative_path: str) -> Path:
        raw = str(relative_path or "").replace("\\", "/").lstrip("/")
        path = (EDITABLE_FILE_ROOT / raw).resolve()
        path.relative_to(EDITABLE_FILE_ROOT.resolve())
        if not path.is_file():
            raise FileNotFoundError(raw)
        return path

    def _update_task(self, key: str, **updates: Any) -> None:
        with self._lock:
            task = self._tasks.get(key)
            if task is None:
                return
            task.update(updates)
            task["updated_at"] = _now_iso()
            task["updated_ts"] = time.time()
            self._save_locked()

    def _load_locked(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        tasks: dict[str, dict[str, Any]] = {}
        for item in (raw.get("tasks") if isinstance(raw, dict) else raw) or []:
            if not isinstance(item, dict):
                continue
            task_id = _clean(item.get("id"))
            owner = _clean(item.get("owner_id"))
            if not task_id or not owner:
                continue
            task = {
                "id": task_id,
                "owner_id": owner,
                "status": _clean(item.get("status"), TASK_STATUS_ERROR),
                "kind": "psd" if item.get("kind") == "psd" else "ppt",
                "created_at": _clean(item.get("created_at"), _now_iso()),
                "updated_at": _clean(item.get("updated_at"), _clean(item.get("created_at"), _now_iso())),
                "created_ts": float(item.get("created_ts") or 0),
                "updated_ts": float(item.get("updated_ts") or 0),
            }
            for field in ("result", "error", "started_ts", "ended_ts"):
                if item.get(field):
                    task[field] = item[field]
            tasks[_task_key(owner, task_id)] = task
        return tasks

    def _save_locked(self) -> None:
        items = sorted(self._tasks.values(), key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps({"tasks": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self.path)

    def _recover_unfinished_locked(self) -> bool:
        changed = False
        for task in self._tasks.values():
            if task.get("status") in UNFINISHED_STATUSES:
                task["status"] = TASK_STATUS_ERROR
                task["error"] = "服务已重启，未完成的任务已中断"
                task["ended_ts"] = time.time()
                task["updated_at"] = _now_iso()
                task["updated_ts"] = time.time()
                changed = True
        return changed

    def _log_call(
            self,
            identity: dict[str, object],
            kind: str,
            started: float,
            request_preview: str,
            *,
            status: str = "success",
            error: str = "",
            account_email: str = "",
            result: dict[str, str] | None = None,
    ) -> None:
        detail = {
            "key_id": identity.get("id"),
            "key_name": identity.get("name"),
            "role": identity.get("role"),
            "endpoint": f"/v1/{kind}/generations",
            "model": EDITABLE_FILE_MODEL,
            "started_at": datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": _now_iso(),
            "duration_ms": int((time.time() - started) * 1000),
            "status": status,
        }
        if request_preview:
            detail["request_text"] = request_preview
        if account_email:
            detail["account_email"] = account_email
        if error:
            detail["error"] = error
        if result:
            detail["result"] = result
        try:
            log_service.add(LOG_TYPE_CALL, f"{kind.upper()}生成任务{'失败' if status == 'failed' else '完成'}", detail)
        except Exception:
            pass


editable_file_task_service = EditableFileTaskService()
