#!/usr/bin/env python3
"""Run one bounded C17 round through the existing isolated-client journal.

No network target is accepted except the runtime's 0600 loopback ready file.
The script never creates a second journal, retries an ID, or schedules another
round.  Each POST is delegated to real_candidate_client._submit, which fsyncs
the identical runtime/client-state.json before sending.
"""
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import fcntl
import importlib.util
import json
import hashlib
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any
from urllib.parse import quote

ROUND_SECONDS = 600
POLL_SECONDS = 2
ACCOUNT_REF = re.compile(r"car_[A-Za-z0-9_-]{43}$")
SAFE = {"id", "status", "kind", "account_ref", "timing_ms", "result_bytes", "file", "width", "height", "archive_status", "error_code", "http_status"}


def emit(**value: object) -> None:
    print(json.dumps({key: item for key, item in value.items() if key in SAFE}, separators=(",", ":")), flush=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("MODULE_LOAD_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prompt(kind: str) -> str:
    targets = {"short": 96, "4k": 4096, "16k": 16384}
    prefix = "Reply only with OK.\nInput follows:\n"
    return prefix + "x" * max(0, targets[kind] - len(prefix.encode()))


def planned(round_no: int, refs: list[str], source_image: Path, text_model: str, image_model: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    # Independent sessions rotate across all three test principals: two text
    # observations per ordinary key, while the single state budget remains global.
    key_files = ("ordinary-test.key", "same-owner-test.key", "other-owner-test.key")
    for index, size in enumerate(("short", "4k", "16k", "short", "4k", "16k"), start=1):
            account_ref = refs[(index - 1) % len(refs)]
            key_file = key_files[(index - 1) % len(key_files)]
            request_id = f"acceptance-c17-r{round_no}-text-{index}-{size}"
            result.append({"id": request_id, "kind": "text", "account_ref": account_ref, "key_file": key_file,
                           "session_id": f"acceptance-c17-r{round_no}-session-{index}",
                           "endpoint": "/api/chat-requests",
                           "body": {"client_request_id": request_id, "model": text_model,
                                    "account_ref": account_ref,
                                    "client_conversation_id": f"acceptance-c17-r{round_no}-session-{index}",
                                    "messages": [{"role": "user", "content": prompt(size)}]}})
    image_ref = refs[0]
    generation_id = f"acceptance-c17-r{round_no}-image-generation"
    generation_thread = f"acceptance-c17-r{round_no}-image-generation-thread"
    result.append({"id": generation_id, "kind": "image", "account_ref": image_ref,
                   "key_file": "ordinary-test.key",
                   "endpoint": "/api/image-tasks/generations",
                   "body": {"client_task_id": generation_id, "model": image_model, "account_ref": image_ref,
                            "image_thread_id": generation_thread,
                            "n": 1, "quality": "auto",
                            "prompt": "Acceptance C17: create a plain blue geometric square, no text."}})
    raw = source_image.read_bytes()
    edit_ref = refs[-1]
    edit_id = f"acceptance-c17-r{round_no}-image-edit"
    # Keep the exact multipart wire body in the shared state journal, as the
    # base client does; it is mode 0600 and never written to stdout.
    result.append({"id": edit_id, "kind": "image", "account_ref": edit_ref,
                   "key_file": "ordinary-test.key",
                   "endpoint": "/api/image-tasks/edits", "edit_source": source_image,
                   "image_thread_id": f"acceptance-c17-r{round_no}-image-edit-thread",
                   "edit_meta": {"bytes": len(raw)}})
    return result


def make_edit(entry: dict[str, Any], shared: Any, image_model: str) -> tuple[dict[str, Any], bytes, str]:
    image = shared._read_input_image(str(entry["edit_source"]))
    task_id = entry["id"]
    fields = {"client_task_id": task_id, "model": image_model, "account_ref": entry["account_ref"],
              "image_thread_id": entry["image_thread_id"],
              "n": "1", "quality": "auto",
              "prompt": "Acceptance C17 edit: keep the geometric subject and make its background blue, no text."}
    raw, content_type = shared._multipart_body(fields, [image])
    return {"content_type": content_type, "wire_body_b64": base64.b64encode(raw).decode("ascii")}, raw, content_type


def halt(driver: Any, shared: Any, state_path: Path, state: dict[str, Any], record: dict[str, Any], code: str) -> None:
    driver._halt(state, record, code)
    driver._save(shared, state_path, state)
    emit(status="stopped", id=record.get("id"), kind=record.get("kind"), account_ref=record.get("account_ref"), error_code=code)


def save_image(api: Any, shared: Any, driver: Any, state_path: Path, state: dict[str, Any], record: dict[str, Any], results: Path) -> bool:
    receipt = driver._image_receipt(shared, api, record["id"])
    if not isinstance(receipt.get("data"), list) or not receipt["data"]:
        halt(driver, shared, state_path, state, record, "IMAGE_RESULT_MISSING")
        return False
    target = results / f"{record['id']}.png"
    try:
        with api.open("GET", f"/api/image-tasks/{quote(record['id'], safe='')}/images/0") as response:
            raw = response.read(shared.MAX_DOWNLOAD_BYTES + 1)
        if not raw or len(raw) > shared.MAX_DOWNLOAD_BYTES:
            raise ValueError("IMAGE_RESULT_BYTES_INVALID")
        from PIL import Image
        with Image.open(BytesIO(raw)) as image:
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
        if not target.exists():
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as output:
                output.write(raw); output.flush(); os.fsync(output.fileno())
        if target.read_bytes() != raw:
            raise ValueError("IMAGE_FILE_READBACK_FAILED")
    except Exception:
        halt(driver, shared, state_path, state, record, "IMAGE_RESULT_VALIDATION_FAILED")
        return False
    record["result_file"] = target.name
    record["result_bytes"] = len(raw)
    record["dimensions"] = [width, height]
    driver._save(shared, state_path, state)
    emit(status="result_saved", id=record["id"], kind="image", account_ref=record.get("account_ref"),
         result_bytes=len(raw), file=target.name, width=width, height=height)
    return True


def save_text(driver, record, receipt, results):
    target = results / f"{record['id']}.txt"
    if not target.exists():
        return driver._save_text_content(record, receipt, results)
    content = receipt.get("content")
    if receipt.get("status") != "succeeded" or not isinstance(content, str) or target.read_bytes() != content.encode():
        raise RuntimeError("TEXT_FILE_READBACK_MISMATCH")
    raw = target.read_bytes()
    record.update(result_file=target.name, result_bytes=len(raw), result_sha256=hashlib.sha256(raw).hexdigest())
    return len(raw)


def image_complete(api: Any, shared: Any, driver: Any, state_path: Path, state: dict[str, Any], record: dict[str, Any]) -> bool:
    try:
        work = api.json("POST", f"/api/image-tasks/{quote(record['id'], safe='')}/work",
                        payload={"state": "completed", "results_saved": True})
    except shared.HttpFailure as exc:
        driver._halt(state, record, "IMAGE_WORK_HTTP_FAILED", http_status=int(exc.status)); driver._save(shared, state_path, state)
        emit(status="work_failed", id=record["id"], kind="image", http_status=int(exc.status), error_code="IMAGE_WORK_HTTP_FAILED")
        return False
    except shared.ClientError:
        halt(driver, shared, state_path, state, record, "IMAGE_WORK_UNKNOWN")
        return False
    archive = work.get("archive") if isinstance(work, dict) else None
    record["work"] = {"desired_archived": True, "last_archive_status": archive.get("status") if isinstance(archive, dict) else None}
    driver._save(shared, state_path, state)
    return True


def poll_work(api: Any, shared: Any, driver: Any, state_path: Path, state: dict[str, Any], record: dict[str, Any]) -> bool:
    try:
        if record["kind"] == "text":
            return driver._work_read(shared, api, state_path, state, record["id"]) == 0
        work = api.json("GET", f"/api/image-tasks/{quote(record['id'], safe='')}/work")
    except shared.HttpFailure as exc:
        driver._halt(state, record, "IMAGE_WORK_READ_HTTP_FAILED", http_status=int(exc.status)); driver._save(shared, state_path, state)
        emit(status="work_read_failed", id=record["id"], kind="image", http_status=int(exc.status), error_code="IMAGE_WORK_READ_HTTP_FAILED")
        return False
    except shared.ClientError:
        halt(driver, shared, state_path, state, record, "IMAGE_WORK_READ_UNKNOWN")
        return False
    archive = work.get("archive") if isinstance(work, dict) else None
    if not isinstance(archive, dict):
        halt(driver, shared, state_path, state, record, "IMAGE_ARCHIVE_UNCLASSIFIED")
        return False
    record["work"] = {"desired_archived": archive.get("desired"), "last_archive_status": archive.get("status"), "archived": archive.get("archived")}
    if archive.get("status") == "unknown":
        halt(driver, shared, state_path, state, record, "IMAGE_ARCHIVE_UNKNOWN")
        return False
    driver._save(shared, state_path, state)
    emit(status="work_read", id=record["id"], kind="image", account_ref=record.get("account_ref"), archive_status=archive.get("status"))
    return archive.get("status") == "confirmed"


def run(args: argparse.Namespace) -> int:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                 "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        os.environ.pop(name, None)
    runtime = args.runtime.resolve()
    provider = runtime / "provider"
    driver = load_module("isolated_real_client", args.client_script.resolve())
    key_paths = {name: runtime / name for name in ("ordinary-test.key", "same-owner-test.key", "other-owner-test.key")}
    for path in (provider, runtime / "ready.json", runtime / "client-state.json", runtime / "results", *key_paths.values()):
        if not path.resolve().is_relative_to(runtime):
            raise RuntimeError("RUNTIME_PATH_INVALID")
    shared = driver._load_shared(provider)
    ready = runtime / "ready.json"; key = runtime / "ordinary-test.key"; state_path = runtime / "client-state.json"; results = runtime / "results"
    if json.loads(ready.read_text()).get("candidate_sha") != (runtime / "candidate.sha").read_text().strip():
        raise RuntimeError("READY_CANDIDATE_MISMATCH")
    base = driver._read_ready(ready)
    apis = {name: shared.ApiClient(base, driver._read_key(path), 30.0) for name, path in key_paths.items()}
    results.mkdir(mode=0o700, exist_ok=True); os.chmod(results, 0o700)
    deadline = time.monotonic() + ROUND_SECONDS
    active: list[str] = []
    with open(str(state_path) + ".lock", "a") as lock:
        os.chmod(lock.name, 0o600)
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("JOURNAL_LOCKED")
        state = driver._state(state_path)
        key_by_id = state.setdefault("c17_key_files", {})
        if not isinstance(key_by_id, dict):
            raise RuntimeError("KEY_PROVENANCE_INVALID")
        entries = planned(args.round, args.account_ref, results / "acceptance-func-image-01.png", args.text_model, args.image_model)

        def submit_entry(entry: dict[str, Any]) -> bool:
            if time.monotonic() >= deadline or state.get("halted"):
                return False
            request_id = entry["id"]
            if request_id in state["requests"]:
                if key_by_id.get(request_id) not in apis:
                    raise RuntimeError("KEY_PROVENANCE_MISSING")
                active.append(request_id); emit(status="original_id_read_only", id=request_id, kind=entry["kind"], account_ref=entry.get("account_ref")); return True
            body, raw, content_type = entry.get("body"), None, ""
            if entry.get("edit_source"):
                body, raw, content_type = make_edit(entry, shared, args.image_model)
            # Key provenance is fsynced in the same state file before _submit
            # persists its ID/body/budget reservation and can issue the POST.
            key_by_id[request_id] = entry["key_file"]
            driver._save(shared, state_path, state)
            outcome = driver._submit(shared, apis[entry["key_file"]], state_path, state, request_id=request_id, kind=entry["kind"], endpoint=entry["endpoint"], body=body, raw_body=raw, content_type=content_type, session_id=entry.get("session_id", ""))
            active.append(request_id)
            return outcome == 0 and not state.get("halted")

        # The first two independent text sessions are the low tier.  They must
        # be read, saved and marked completed before this round receives the
        # remaining mixed workload; archive confirmation stays asynchronous.
        low_entries = entries[:2]
        for entry in low_entries:
            if not submit_entry(entry):
                break
        low_pending = {entry["id"] for entry in low_entries if entry["id"] in active}
        while low_pending and time.monotonic() < deadline and not state.get("halted"):
            ready_count = 0
            for request_id in tuple(low_pending):
                record = state["requests"].get(request_id); key_name = key_by_id.get(request_id)
                if not isinstance(record, dict) or key_name not in apis:
                    if isinstance(record, dict): halt(driver, shared, state_path, state, record, "KEY_PROVENANCE_MISSING")
                    else: raise RuntimeError("KEY_PROVENANCE_MISSING")
                    break
                api = apis[key_name]
                status = driver._read_original(shared, api, state_path, state, request_id)
                if status != 0 or state.get("halted"):
                    continue
                if not record.get("result_file"):
                    receipt = driver._chat_receipt(shared, api, request_id)
                    size = save_text(driver, record, receipt, results); driver._save(shared, state_path, state)
                    emit(status="result_saved", id=request_id, kind="text", account_ref=record.get("account_ref"), result_bytes=size, file=record["result_file"])
                if not record.get("work"):
                    driver._work(shared, api, state_path, state, request_id, active=False)
                if record.get("work") and not state.get("halted"):
                    low_pending.remove(request_id); ready_count += 1
            if not ready_count and low_pending and not state.get("halted"):
                time.sleep(min(POLL_SECONDS, max(0, deadline-time.monotonic())))
        if not low_pending and not state.get("halted"):
            for entry in entries[2:]:
                if not submit_entry(entry):
                    break
        pending = set(active)
        archived_confirmed = 0
        while pending and time.monotonic() < deadline and not state.get("halted"):
            progressed = False
            for request_id in tuple(pending):
                record = state["requests"].get(request_id)
                if not isinstance(record, dict):
                    continue
                key_name = key_by_id.get(request_id)
                if key_name not in apis:
                    halt(driver, shared, state_path, state, record, "KEY_PROVENANCE_MISSING"); break
                api = apis[key_name]
                before = time.monotonic()
                status = driver._read_original(shared, api, state_path, state, request_id)
                emit(status="read", id=request_id, kind=record.get("kind"), account_ref=record.get("account_ref"), timing_ms=int((time.monotonic()-before)*1000))
                if state.get("halted"): break
                if status != 0: continue
                if record["kind"] == "text":
                    if not record.get("result_file"):
                        receipt = driver._chat_receipt(shared, api, request_id)
                        size = save_text(driver, record, receipt, results); driver._save(shared, state_path, state)
                        emit(status="result_saved", id=request_id, kind="text", account_ref=record.get("account_ref"), result_bytes=size, file=record["result_file"])
                    if not record.get("work") and driver._work(shared, api, state_path, state, request_id, active=False) != 0: break
                else:
                    if not record.get("result_file") and not save_image(api, shared, driver, state_path, state, record, results): break
                    if not record.get("work") and not image_complete(api, shared, driver, state_path, state, record): break
                if record.get("work") and poll_work(api, shared, driver, state_path, state, record):
                    archived_confirmed += 1; pending.remove(request_id); progressed = True
                if state.get("halted"): break
            if not progressed and pending and not state.get("halted"):
                time.sleep(min(POLL_SECONDS, max(0, deadline-time.monotonic())))
        if pending or len(active) != len(entries):
            state.update(halted=True, halt_reason="ROUND_INCOMPLETE_OR_DEADLINE")
            driver._save(shared, state_path, state)
        emit(status="round_finished" if not state.get("halted") else "round_incomplete", timing_ms=int((ROUND_SECONDS - max(0, deadline-time.monotonic()))*1000), archive_status=f"confirmed:{archived_confirmed}")
    return 0 if not state.get("halted") else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--account-ref", action="append", required=True)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--image-model", default="gpt-image-2")
    parser.add_argument("--client-script", type=Path, default=Path(__file__).with_name("real_candidate_client.py"))
    args = parser.parse_args()
    if not 1 <= len(args.account_ref) <= 2 or len(set(args.account_ref)) != len(args.account_ref) or any(ACCOUNT_REF.fullmatch(item) is None for item in args.account_ref):
        emit(status="stopped", error_code="ACCOUNT_REF_INVALID"); return 2
    def deadline(_signum, _frame):
        raise TimeoutError("ROUND_DEADLINE")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(ROUND_SECONDS)
    try:
        return run(args)
    except TimeoutError:
        driver = load_module("deadline_client", args.client_script.resolve())
        shared = driver._load_shared(args.runtime.resolve() / "provider")
        state_path = args.runtime.resolve() / "client-state.json"
        with open(str(state_path) + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = driver._state(state_path)
            state.update(halted=True, halt_reason="ROUND_DEADLINE_OR_IO_TIMEOUT")
            driver._save(shared, state_path, state)
        emit(status="round_incomplete", error_code="ROUND_DEADLINE_OR_IO_TIMEOUT")
        return 2
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError):
        emit(status="stopped", error_code="ROUND_PRECONDITION_OR_LOCAL_IO_FAILED")
        return 2
    finally:
        signal.alarm(0)

if __name__ == "__main__":
    raise SystemExit(main())
