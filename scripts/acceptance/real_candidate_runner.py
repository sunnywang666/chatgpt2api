#!/usr/bin/env python3
"""Serve the exact Provider candidate with only admission lifecycle enabled.

This is an artifact runner, not product code.  It deliberately omits the
production app lifespan's account watcher, backup scheduler, and image cleanup
scheduler.  It never reads the input mounts and rejects argparse-provided
protected source/data paths.
"""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import signal
import sys
from typing import Any

def _die(message: str) -> None:
    raise SystemExit(message)


def _safe_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _candidate_marker(source: Path) -> str:
    """Bootstrap verifies Git before stripping .git from the isolated copy."""
    try:
        return (source.parent / "candidate.sha").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        _die(f"runner_error=candidate_sha_marker_unavailable:{type(exc).__name__}")


def _write_ready(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="bootstrap-created provider source root")
    parser.add_argument("--expected-sha", required=True,
                        help="public exact candidate Git SHA verified by bootstrap")
    parser.add_argument("--data", type=Path, required=True, help="bootstrap-created isolated data root")
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--protected-path", action="append", type=Path, required=True,
                        help="production source/data path; runner source/data must not be beneath it")
    args = parser.parse_args()

    source, data, ready_file = _safe_path(args.source), _safe_path(args.data), _safe_path(args.ready_file)
    protected = [_safe_path(value) for value in args.protected_path]
    if any(_inside(source, root) or _inside(data, root) for root in protected):
        _die("runner_error=source_or_data_is_under_protected_path")
    runtime_root = source.parent
    if not _inside(ready_file, runtime_root) or any(_inside(ready_file, root) for root in protected):
        _die("runner_error=ready_file_must_be_inside_isolated_runtime")
    if not (source / "main.py").is_file() or not (source / "config.json").is_file():
        _die("runner_error=isolated_source_incomplete")
    if not (data / "accounts.json").is_file() or not (data / "auth_keys.json").is_file():
        _die("runner_error=isolated_data_incomplete")
    expected_sha = str(args.expected_sha or "").strip().lower()
    if len(expected_sha) != 40 or any(char not in "0123456789abcdef" for char in expected_sha):
        _die("runner_error=expected_sha_must_be_40_lowercase_hex")
    sha = _candidate_marker(source)
    if sha != expected_sha:
        _die("runner_error=candidate_sha_mismatch")

    # Do not inherit a production backend or credentials from container env.
    for name in ("DATABASE_URL", "GIT_REPO_URL", "GIT_TOKEN", "GIT_BRANCH", "GIT_FILE_PATH", "GIT_AUTH_KEYS_FILE_PATH"):
        os.environ.pop(name, None)
    # Requests libraries honor these by default.  The bootstrap whitelist in
    # the isolated config is the sole transport source for this process.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                 "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        os.environ.pop(name, None)
    os.environ["STORAGE_BACKEND"] = "json"
    os.environ["PROVIDER_DATA_DIR"] = str(data)
    os.environ.pop("CHATGPT2API_AUTH_KEY", None)  # isolated config owns its random bootstrap key
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.chdir(source)
    sys.path[:] = [str(source), *[entry for entry in sys.path if entry != str(source)]]

    # Imports happen only after all path/backend isolation has been established.
    from fastapi import FastAPI
    import uvicorn
    from api import ai, chat_requests, image_tasks
    from api.errors import install_exception_handlers
    from services.config import config
    from services.pool_admission import configure_original_task_admission
    from services.text_task_service import text_task_service
    from services.image_task_service import image_task_service

    @asynccontextmanager
    async def acceptance_lifespan(_: FastAPI):
        admission = configure_original_task_admission()
        admission.start()
        try:
            yield
        finally:
            admission.stop()
            if text_task_service.admission is admission:
                text_task_service.admission = None
            if image_task_service.admission is admission:
                image_task_service.admission = None

    app = FastAPI(title="isolated-provider-acceptance", lifespan=acceptance_lifespan)
    install_exception_handlers(app)
    # These are the product's real API routers; no fake model/router seam exists.
    app.include_router(ai.create_router())
    app.include_router(chat_requests.create_router())
    app.include_router(image_tasks.create_router())

    config_server = uvicorn.Config(app, host="127.0.0.1", port=0, access_log=False, log_level="warning")
    server = uvicorn.Server(config_server)

    async def publish_ready() -> None:
        # Uvicorn has bound by startup.  No user token or provider credential is
        # included in this local readiness record.
        sockets = [sock for item in (server.servers or []) for sock in item.sockets or []]
        if not sockets:
            raise RuntimeError("listener missing after startup")
        port = int(sockets[0].getsockname()[1])
        config.data["base_url"] = f"http://127.0.0.1:{port}"
        _write_ready(ready_file, {"status": "ready", "host": "127.0.0.1", "port": port, "candidate_sha": sha})
        print(json.dumps({"status": "ready", "port": port, "candidate_sha": sha}, separators=(",", ":")), flush=True)

    original_started = server.startup

    async def startup_with_ready(*args, **kwargs):
        await original_started(*args, **kwargs)
        await publish_ready()

    server.startup = startup_with_ready

    def stop_handler(_signum, _frame):
        # Operators must first use the client readback and confirm every receipt
        # is terminal.  This handler never retries or rewrites a receipt.
        server.should_exit = True

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    try:
        server.run()
    finally:
        ready_file.unlink(missing_ok=True)
        print(json.dumps({"status": "stopped"}, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
