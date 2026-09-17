#!/usr/bin/env python3
"""Engineering-only Mac CLI -> candidate router/service -> local mock QA.

No production URL, account, or credential is used. The only URL remapping is
inside this harness's session factory; production CodexService constants stay
unchanged. A compact report without keys is written under ignored data/.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import uvicorn
from curl_cffi import requests
from fastapi import FastAPI
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
CLIENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CLIENT_DIR))

from api import codex as codex_api  # noqa: E402
import api.support as api_support  # noqa: E402
from mock_provider import MockHandler, TOOL_CYCLE  # noqa: E402
from services.account_service import AccountService  # noqa: E402
from services.auth_service import AuthService  # noqa: E402
from services.codex_service import (  # noqa: E402
    CODEX_MODELS_URL,
    CODEX_RESPONSES_URL,
    CODEX_USAGE_URL,
    CodexService,
)
from services.storage.json_storage import JSONStorageBackend  # noqa: E402


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class MappedSession:
    """Map only the candidate's fixed upstream endpoints to the local mock."""

    def __init__(self, mock_base_url: str, **_ignored: object) -> None:
        self._mock_base_url = mock_base_url.rstrip("/")
        self._session = requests.Session()

    def _map(self, url: str) -> str:
        routes = {
            CODEX_MODELS_URL: f"{self._mock_base_url}/v1/models",
            CODEX_USAGE_URL: f"{self._mock_base_url}/v1/usage",
            CODEX_RESPONSES_URL: f"{self._mock_base_url}/v1/responses",
        }
        try:
            return routes[url]
        except KeyError as error:
            raise AssertionError(f"unexpected candidate upstream URL: {url}") from error

    def get(self, url: str, **kwargs: object):
        return self._session.get(self._map(url), **kwargs)

    def post(self, url: str, **kwargs: object):
        return self._session.post(self._map(url), **kwargs)

    def close(self) -> None:
        self._session.close()


def http_status(url: str, key: str, *, method: str = "GET", body: bytes | None = None) -> int:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310: loopback harness URL
            response.read()
            return response.status
    except urllib.error.HTTPError as error:
        error.read()
        return error.code


def run_cli(base_url: str, ordinary_key: str, state_root: Path, fixture_root: Path, *, model: str, resume: str = "") -> subprocess.CompletedProcess[str]:
    command = [
        "bash",
        str(CLIENT_DIR / "run-codex.sh"),
        "--base-url",
        base_url,
        "--model",
        model,
        "--state-root",
        str(state_root),
        "--fixture-root",
        str(fixture_root),
    ]
    if resume:
        command.extend(["--resume", resume])
    environment = dict(os.environ)
    environment["CODEX_PROVIDER_API_KEY"] = ordinary_key
    return subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, timeout=90, check=False)


def thread_id(stdout: str) -> str:
    for line in stdout.splitlines():
        with suppress(json.JSONDecodeError):
            event = json.loads(line)
            if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
                return event["thread_id"]
    raise AssertionError("CLI did not emit a thread.started event")


def main() -> int:
    report_root = ROOT / "data" / "codex-client-qa"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"candidate-chain-{uuid.uuid4().hex}.json"
    mock_port = free_port()
    api_port = free_port()
    mock_base_url = f"http://127.0.0.1:{mock_port}"
    service_base_url = f"http://127.0.0.1:{api_port}/ai/codex/v1"
    MockHandler.candidate_service_mode = True
    MockHandler.cycle_stage = 0
    MockHandler.observations = []
    mock_server = ThreadingHTTPServer(("127.0.0.1", mock_port), MockHandler)
    mock_thread = threading.Thread(target=mock_server.serve_forever, daemon=True)
    mock_thread.start()

    original_service = codex_api.codex_service
    original_auth = api_support.auth_service
    server: uvicorn.Server | None = None
    server_thread: threading.Thread | None = None
    try:
        with TemporaryDirectory(prefix="codex-client-qa-") as temporary:
            temporary_root = Path(temporary)
            storage = JSONStorageBackend(temporary_root / "accounts.json", temporary_root / "auth_keys.json")
            accounts = AccountService(storage)
            auth = AuthService(storage)
            user_item, ordinary_key = auth.create_key(role="user", name="qa ordinary user", capabilities=["codex_coding"])
            _admin_item, admin_key = auth.create_key(role="admin", name="qa admin")
            upstream_token = "qa-fake-upstream-account-token"
            accounts.import_owned_account(
                "qa-owner",
                {"access_token": upstream_token, "account_id": "qa-fake-account", "source_type": "codex"},
            )
            accounts.update_account(upstream_token, {"codex_observation": {"state": "unknown"}}, quiet=True)
            service = CodexService(accounts, lambda **kwargs: MappedSession(mock_base_url, **kwargs))
            codex_api.codex_service = service
            api_support.auth_service = auth
            app = FastAPI()
            app.include_router(codex_api.create_router(), prefix="/ai")
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=api_port, log_level="warning"))
            server_thread = threading.Thread(target=server.run, daemon=True)
            server_thread.start()
            deadline = time.monotonic() + 10
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.05)
            if not server.started:
                raise AssertionError("candidate API server did not start")

            first = run_cli(service_base_url, ordinary_key, temporary_root / "state", temporary_root / "fixture", model="gpt-5.1-codex-mini")
            if first.returncode != 0:
                raise AssertionError(f"first Codex CLI command failed with exit {first.returncode}")
            first_thread_id = thread_id(first.stdout)
            test_result = subprocess.run(
                [sys.executable, "-m", "unittest", "-q"], cwd=temporary_root / "fixture", text=True, capture_output=True, timeout=30, check=False
            )
            if test_result.returncode != 0:
                raise AssertionError("mock-directed CLI edit did not make the fixture test pass")
            resumed = run_cli(
                service_base_url,
                ordinary_key,
                temporary_root / "state",
                temporary_root / "fixture",
                model="gpt-5.1-codex-mini",
                resume=first_thread_id,
            )
            if resumed.returncode != 0:
                raise AssertionError(f"resumed Codex CLI command failed with exit {resumed.returncode}")
            if thread_id(resumed.stdout) != first_thread_id:
                raise AssertionError("resume did not retain the original isolated thread ID")

            post_calls = [item for item in MockHandler.observations if item["method"] == "POST"]
            expected_history = [[], ["call_read_fixture"], ["call_read_fixture", "call_repair_fixture"], [call_id for call_id, _ in TOOL_CYCLE]]
            actual_history = [item["function_call_output_ids"] for item in post_calls]
            if actual_history[:4] != expected_history:
                raise AssertionError("candidate relay did not preserve function-call output history")
            account = accounts.get_account(upstream_token) or {}
            if not account.get("codex_response_ids") or not account.get("codex_affinities"):
                raise AssertionError("candidate service did not persist response/session ownership bindings")
            observed_models = ((account.get("codex_observation") or {}).get("models") or [])
            if not any(item.get("id") == "gpt-5.1-codex-mini" for item in observed_models if isinstance(item, dict)):
                raise AssertionError("candidate service did not parse the mock native model catalog")

            if auth.update_key(str(user_item["id"]), {"enabled": False}, role="user") is None:
                raise AssertionError("ordinary QA key could not be revoked")
            revoked_get = http_status(f"{service_base_url}/models", ordinary_key)
            revoked_post = http_status(f"{service_base_url}/responses", ordinary_key, method="POST", body=b'{"model":"gpt-5.1-codex-mini","input":[]}')
            admin_status = http_status(f"{service_base_url}/models", admin_key)
            if (revoked_get, revoked_post, admin_status) != (401, 401, 403):
                raise AssertionError("candidate authorization boundary did not return expected revoked/admin statuses")

            report = {
                "result": "passed",
                "acceptance": "engineering_mock_only",
                "model": "gpt-5.1-codex-mini",
                "tool_output_history": actual_history,
                "fixture_test": "passed",
                "resume": "same_thread_id",
                "revoked_get_status": revoked_get,
                "revoked_post_status": revoked_post,
                "admin_models_status": admin_status,
                "key_material_logged": False,
            }
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"candidate engineering QA passed; report: {report_path}")
            return 0
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=5)
        codex_api.codex_service = original_service
        api_support.auth_service = original_auth
        mock_server.shutdown()
        mock_server.server_close()
        mock_thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
