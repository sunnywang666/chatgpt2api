"""Real loopback HTTP/process restarts; only model execution is synthetic.

No upstream account, model HTTP, or production data is used. This does not
simulate killing a process after a real upstream send with an UNKNOWN result.
"""
from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def _wait(check, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = check()
        if result:
            return result
        time.sleep(.02)
    raise AssertionError("isolated HTTP fixture did not reach the expected state")


def _http(port, key, method, path, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request(method, path, body=json.dumps(body) if body is not None else None,
                           headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


@contextmanager
def _server(root, mode):
    # The child binds port zero itself, avoiding a free-port discovery race.
    ready = root / "ready.json"
    ready.unlink(missing_ok=True)
    env = {**os.environ, "PROVIDER_DATA_DIR": str(root), "CHATGPT2API_AUTH_KEY": "fixture-admin-not-used",
           "PYTHONDONTWRITEBYTECODE": "1"}
    process = subprocess.Popen([sys.executable, "-m", "test.test_public_durable_http_recovery",
                                str(root), mode], cwd=Path(__file__).resolve().parents[1],
                               env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        def started():
            if process.poll() is not None:
                raise AssertionError(process.stderr.read().decode()[-3000:])
            if not ready.exists():
                return None
            port = json.loads(ready.read_text())["port"]
            try:
                return port if _http(port, "", "GET", "/fixture-ready")[0] == 200 else None
            except (OSError, http.client.HTTPException):
                return None
        port = _wait(started)
        yield port, (root / "fixture-key").read_text(), process
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        process.stderr.close()


def _input(root, request_id):
    body = {"client_request_id": request_id, "model": "fixture-text",
            "messages": [{"role": "user", "content": "synthetic input"}]}
    # The caller persists identity and immutable input before opening HTTP.
    path = root / "caller-state.json"
    path.write_text(json.dumps(body))
    return json.loads(path.read_text())


def _success(port, key, request_id):
    def read():
        status, receipt = _http(port, key, "GET", "/api/chat-requests/" + request_id)
        assert status == 200
        return receipt if receipt["status"] == "succeeded" else None
    return _wait(read)


def test_queued_http_receipt_survives_process_restart_without_resubmission(tmp_path):
    body = _input(tmp_path, "queued-original")
    with _server(tmp_path, "queued") as (port, key, first):
        status, receipt = _http(port, key, "POST", "/api/chat-requests", body)
        assert status == 202 and receipt["status"] == "queued"
        assert receipt["request_id"] == body["client_request_id"]
        assert not (tmp_path / "sends.jsonl").exists()
    # Uvicorn may re-raise SIGTERM after completing its shutdown handler.
    assert first.returncode in (0, -signal.SIGTERM)
    with _server(tmp_path, "execute") as (port, key, second):
        assert second.pid != first.pid
        receipt = _success(port, key, body["client_request_id"])
        assert receipt["content"] == "synthetic saved answer"
        status, repeated = _http(port, key, "POST", "/api/chat-requests", body)
        assert status == 200 and repeated == receipt
        assert _http(port, key, "POST", "/api/chat-requests", {**body, "messages": [{"role": "user", "content": "changed"}]})[0] == 409
    assert (tmp_path / "sends.jsonl").read_text().splitlines() == ['"queued-original"']


def test_real_socket_disconnect_keeps_original_execution_and_result(tmp_path):
    body = _input(tmp_path, "disconnected-original")
    with _server(tmp_path, "blocked-send") as (port, key, _process):
        encoded = json.dumps(body).encode()
        wire = (f"POST /api/chat-requests HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                f"Authorization: Bearer {key}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(encoded)}\r\n\r\n").encode() + encoded
        with socket.create_connection(("127.0.0.1", port), timeout=2) as subscriber:
            subscriber.sendall(wire)
            _wait(lambda: (tmp_path / "sends.jsonl").exists())
            # Disconnect the actual HTTP socket without receiving its response.
            subscriber.shutdown(socket.SHUT_RDWR)
        status, running = _http(port, key, "GET", "/api/chat-requests/" + body["client_request_id"])
        assert status == 200 and running["status"] == "running"
        (tmp_path / "release-send").touch()
        receipt = _success(port, key, body["client_request_id"])
        assert receipt["content"] == "synthetic saved answer"
        assert _http(port, key, "POST", "/api/chat-requests", body) == (200, receipt)
    assert (tmp_path / "sends.jsonl").read_text().splitlines() == ['"disconnected-original"']


def _serve(root, mode):
    # Independent of the caller's optional sitecustomize offline guard: this
    # process must never initiate any outbound socket, including model probes.
    def forbidden(*_args, **_kwargs):
        raise AssertionError("HTTP recovery fixture forbids outbound network")
    socket.socket.connect = forbidden
    socket.socket.connect_ex = forbidden
    socket.create_connection = forbidden
    import curl_cffi.requests
    import requests.sessions
    curl_cffi.requests.Session.request = forbidden
    requests.sessions.Session.request = forbidden

    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    import uvicorn
    from fastapi import FastAPI
    from api import chat_requests, support
    from services.account_service import AccountService
    from services.auth_service import AuthService
    from services.pool_admission import PoolAdmission
    from services import public_chat_service
    from services.request_context import current_request
    from services.storage.json_storage import JSONStorageBackend
    from services.task_store import TaskStore
    from services.text_task_service import TextTaskService

    accounts = AccountService(JSONStorageBackend(root / "accounts.json"))
    if not accounts.list_accounts():
        accounts.add_account_items([{"access_token": "fixture-upstream-never-sent", "source_type": "web",
                                    "type": "Plus", "status": "正常"}])
    token = accounts.list_accounts()[0]["access_token"]
    accounts.update_account(token, {"managed_disabled": mode == "queued", "status": "正常"})
    auth = AuthService(JSONStorageBackend(root / "accounts.json", root / "auth_keys.json"))
    key_file = root / "fixture-key"
    if not key_file.exists():
        _, key = auth.create_key(role="user", routes=["chat"])
        key_file.write_text(key)
        key_file.chmod(0o600)
    support.auth_service = auth
    public_chat_service.model_catalog_service = SimpleNamespace(
        known_account_types_for_model=lambda _model: frozenset({"Plus"}),
        catalog_is_unknown=lambda: False)
    chat_requests.check_request = lambda *_args: None
    store = TaskStore(root / "tasks.sqlite3")
    admission = PoolAdmission(store, accounts, settings=lambda: {
        "chat_account_concurrency": 1, "image_account_concurrency": 1, "codex_max_concurrency": 1},
        model_types=lambda _model: {"Plus"}, pacing=lambda _account, now: {"next_at": now})

    def synthetic_model(body, on_cursor):
        current_request.get().before_send()
        with (root / "sends.jsonl").open("a") as log:
            log.write(json.dumps(body["client_request_id"]) + "\n")
            log.flush()
            os.fsync(log.fileno())
        if mode == "blocked-send":
            _wait(lambda: (root / "release-send").exists(), timeout=10)
        return {"content": "synthetic saved answer"}

    tasks = TextTaskService(store.path, runner=synthetic_model, admission=admission)
    admission.register("text", lambda context, body: tasks._run(context.owner, context.request_id, body))
    chat_requests.text_task_service = tasks

    @asynccontextmanager
    async def lifespan(_app):
        admission.start()
        try:
            yield
        finally:
            admission.stop()

    app = FastAPI(lifespan=lifespan)
    app.include_router(chat_requests.create_router())

    @app.get("/fixture-ready")
    async def ready():
        return {"ready": True}

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(64)
        listener.setblocking(False)
        (root / "ready.json").write_text(json.dumps({"port": listener.getsockname()[1]}))
        uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False)).run(sockets=[listener])


if __name__ == "__main__":
    _serve(Path(sys.argv[1]), sys.argv[2])
