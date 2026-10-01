"""Real loopback HTTP/process restarts with isolated dependency fixtures.

Auth, admission, durable storage and lifecycle handlers are real. Model and
archive execution, catalog observations and credential refresh are synthetic;
AI content review is disabled. No upstream account or production data is used.
This does not simulate killing a process after a real upstream send with an
UNKNOWN result, or verify the deployed company's ingress/authentication.
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

import pytest


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


def test_independent_client_saves_multiturn_work_reuses_slot_and_restores_original(tmp_path):
    """Exercise the shipped CLI over sockets, not a replacement client/receipt."""
    from examples import image_client

    def cli(port, key, *args):
        env = {**os.environ, "SERVER_ROOT": f"http://127.0.0.1:{port}", "CHATGPT2API_BEARER_TOKEN": key}
        result = subprocess.run([sys.executable, str(Path(image_client.__file__)), *map(str, args)],
                                env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    def submit(port, key, name, session, previous=None):
        args = ["chat-submit", "--state", tmp_path / (name + ".json"), "--request-id", name,
                "--session-id", session, "--model", "fixture-text", "--prompt", "controlled input",
                "--workflow-id", "batch", "--workflow-concurrency", "1"]
        if previous:
            args += ["--previous-request-id", previous]
        return cli(port, key, *args)

    with _server(tmp_path, "lifecycle") as (port, key, _):
        submit(port, key, "A-1", "A")
        _success(port, key, "A-1")
        first = cli(port, key, "chat-work-status", "--state", tmp_path / "A-1.json")
        assert first["state"] == "active" and first["slot_held"] is True
        assert first["archive"]["status"] == "not_requested"
        submit(port, key, "B-1", "B")
        waiting = _http(port, key, "GET", "/api/chat-requests/B-1")[1]
        assert waiting["status"] == "queued"
        assert "workflow_concurrency" in waiting["waiting"]["reasons"]
        submit(port, key, "A-2", "A", "A-1")
        _success(port, key, "A-2")
        cli(port, key, "chat-save", "--state", tmp_path / "A-2.json", "--output", tmp_path / "answer.json")
        saved = json.loads((tmp_path / "answer.json").read_text())
        assert saved["request_id"] == "A-2" and saved["content"] == "synthetic saved answer"
        assert not (tmp_path / "archives.jsonl").exists()
        cli(port, key, "chat-complete", "--state", tmp_path / "A-2.json")
        _success(port, key, "B-1")
        work = cli(port, key, "chat-work-status", "--state", tmp_path / "A-2.json")
        assert work["state"] == "completed" and not work["slot_held"]
        assert work["archive"]["archived"] is True
        assert _http(port, key, "POST", "/api/chat-requests/A-1/work",
                     {"state": "completed", "results_saved": True})[0] == 409
        cli(port, key, "chat-save", "--state", tmp_path / "B-1.json", "--output", tmp_path / "B-answer.json")
        cli(port, key, "chat-complete", "--state", tmp_path / "B-1.json")
    with _server(tmp_path, "lifecycle") as (port, key, _):
        assert cli(port, key, "chat-status", "--state", tmp_path / "A-2.json") == saved
        cli(port, key, "chat-rework", "--state", tmp_path / "A-2.json")
        submit(port, key, "A-3", "A", "A-2")
        _success(port, key, "A-3")
        current = cli(port, key, "chat-work-status", "--state", tmp_path / "A-3.json")
        assert current["work_ref"] == "A" and current["state"] == "active"
    assert [json.loads(x) for x in (tmp_path / "sends.jsonl").read_text().splitlines()] == ["A-1", "A-2", "B-1", "A-3"]
    archives = [json.loads(x) for x in (tmp_path / "archives.jsonl").read_text().splitlines()]
    assert [(x["request_id"], x["archived"]) for x in archives] == [("A-2", True), ("B-1", True), ("A-2", False)]
    assert archives[0]["conversation_id"] == archives[2]["conversation_id"]
    assert archives[1]["conversation_id"] != archives[0]["conversation_id"]


def test_archive_failure_http_restart_never_changes_successor_work(tmp_path):
    body = {**_input(tmp_path, "A-1"), "client_conversation_id": "A",
            "scheduling": {"workflow_id": "batch", "workflow_concurrency": 1}}
    with _server(tmp_path, "lifecycle") as (port, key, _):
        assert _http(port, key, "POST", "/api/chat-requests", body)[0] == 202
        _success(port, key, "A-1")
        (tmp_path / "fail-archive").touch()
        assert _http(port, key, "POST", "/api/chat-requests/A-1/work",
                     {"state": "completed", "results_saved": True})[0] == 200
        _wait(lambda: _http(port, key, "GET", "/api/chat-requests/A-1/work")[1]["archive"]["status"] == "unknown")
        assert _http(port, key, "POST", "/api/chat-requests/A-1/work", {"state": "active"})[0] == 409
        assert _http(port, key, "POST", "/api/chat-requests", {
            **body, "client_request_id": "B-1", "client_conversation_id": "B"})[0] == 202
        _success(port, key, "B-1")
    (tmp_path / "fail-archive").unlink()
    with _server(tmp_path, "lifecycle") as (port, key, _):
        _wait(lambda: _http(port, key, "GET", "/api/chat-requests/A-1/work")[1]["archive"]["status"] == "confirmed")
        b = _http(port, key, "GET", "/api/chat-requests/B-1/work")[1]
        assert b["state"] == "active" and b["slot_held"] is True
        assert b["archive"]["status"] == "not_requested"
    actions = [json.loads(x) for x in (tmp_path / "archives.jsonl").read_text().splitlines()]
    assert len(actions) >= 2 and {x["request_id"] for x in actions} == {"A-1"}
    assert [json.loads(x) for x in (tmp_path / "sends.jsonl").read_text().splitlines()] == ["A-1", "B-1"]


def test_image_http_selected_zero_quota_refresh_executes_original_and_saves_bytes(tmp_path):
    from examples import image_client
    body = {"client_task_id": "selected-original", "model": "gpt-image-2", "prompt": "controlled image",
            "account_ref": "car_" + "B" * 43}
    with _server(tmp_path, "image-recovery") as (port, key, _):
        response = _http(port, key, "POST", "/api/image-tasks/generations", body)
        assert response[0] == 200 and response[1]["id"] == "selected-original"
        def item():
            return _http(port, key, "GET", "/api/image-tasks?ids=selected-original")[1]["items"][0]
        _wait(lambda: item()["status"] == "queued" and (tmp_path / "metadata-reads.jsonl").exists())
        assert not (tmp_path / "image-sends.jsonl").exists()
        assert _http(port, key, "POST", "/api/image-tasks/generations", {**body, "account_ref": "car_" + "A" * 43})[0] == 409
        # Change only the controlled upstream observation, not the stored
        # account, managed flag, caller identity, original request or queue.
        (tmp_path / "quota-recovered").touch()
        # Keep the real 30-second metadata probe interval: a shorter test
        # timeout would mistake normal cooldown for a stalled durable queue.
        _wait(lambda: item()["status"] == "success", timeout=40)
        result = subprocess.run([sys.executable, str(Path(image_client.__file__)), "download",
                                 "--task-id", "selected-original", "--output", str(tmp_path / "actual.png")],
                                env={**os.environ, "SERVER_ROOT": f"http://127.0.0.1:{port}", "CHATGPT2API_BEARER_TOKEN": key},
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        import base64
        assert (tmp_path / "actual.png").read_bytes() == base64.b64decode(_FIXTURE_PNG)
    sends = [json.loads(x) for x in (tmp_path / "image-sends.jsonl").read_text().splitlines()]
    assert sends == [{"request_id": "selected-original", "account": "account-B",
                      "model": body["model"], "prompt": body["prompt"]}]
    probes = [json.loads(x) for x in (tmp_path / "metadata-reads.jsonl").read_text().splitlines()]
    assert len(probes) >= 2 and probes[0]["recovered"] is False and probes[1]["recovered"] is True
    assert probes[1]["at"] - probes[0]["at"] >= 29.5


def test_http_scheduling_survives_restart_and_deadline_never_sends(tmp_path):
    from datetime import datetime, timezone
    def at(seconds):
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat()
    options = {"not_before": at(time.time() + 4), "wait_deadline": at(time.time() + 20)}
    body = {**_input(tmp_path, "scheduled-original"), "scheduling": options}
    with _server(tmp_path, "execute") as (port, key, _):
        assert _http(port, key, "POST", "/api/chat-requests", body)[0] == 202
        def waiting():
            receipt = _http(port, key, "GET", "/api/chat-requests/scheduled-original")[1]
            return receipt if "not_before" in receipt.get("waiting", {}).get("reasons", []) else None
        receipt = _wait(waiting)
        assert receipt["scheduling"] == options
        assert not (tmp_path / "sends.jsonl").exists()
    with _server(tmp_path, "execute") as (port, key, _):
        assert _http(port, key, "POST", "/api/chat-requests", {**body, "scheduling": {}})[0] == 409
        receipt = _success(port, key, "scheduled-original")
        assert receipt["scheduling"] == options
        assert receipt["started_at"] >= datetime.fromisoformat(options["not_before"]).timestamp()
    # Capacity remains unavailable until after this request's waiting budget.
    with _server(tmp_path, "queued") as (port, key, _):
        expired = {**_input(tmp_path, "deadline-original"), "scheduling": {"wait_deadline": at(time.time() + 1)}}
        assert _http(port, key, "POST", "/api/chat-requests", expired)[0] == 202
        _wait(lambda: _http(port, key, "GET", "/api/chat-requests/deadline-original")[1].get("error_code") == "WAIT_DEADLINE_EXCEEDED")
    with _server(tmp_path, "execute") as (port, key, _):
        receipt = _http(port, key, "GET", "/api/chat-requests/deadline-original")[1]
        assert receipt["status"] == "failed" and receipt["error_code"] == "WAIT_DEADLINE_EXCEEDED"
    assert (tmp_path / "sends.jsonl").read_text().splitlines() == ['"scheduled-original"']


def test_http_one_account_parallel_sessions_never_overlap_turns(tmp_path):
    def turn(name, session, previous=None):
        return {**_input(tmp_path, name), "client_conversation_id": session,
                **({"previous_request_id": previous} if previous else {})}
    with _server(tmp_path, "parallel") as (port, key, _):
        for name in ("A-1", "B-1"):
            assert _http(port, key, "POST", "/api/chat-requests", turn(name, name[0]))[0] == 202
        def sent():
            path = tmp_path / "sends.jsonl"
            return path.exists() and len(path.read_text().splitlines()) == 2
        _wait(sent)
        for name in ("A-1", "B-1"):
            assert _http(port, key, "GET", "/api/chat-requests/" + name)[1]["status"] == "running"
        for name, previous in (("A-2", "A-1"), ("A-3", "A-2")):
            body = turn(name, "A", previous)
            assert _http(port, key, "POST", "/api/chat-requests", body)[0] == 202
            assert _http(port, key, "POST", "/api/chat-requests", body)[0] == 202
            assert _http(port, key, "POST", "/api/chat-requests", {**body, "messages": [{"role": "user", "content": "drift"}]})[0] == 409
        assert _http(port, key, "POST", "/api/chat-requests", turn("A-fork", "A", "A-1"))[0] == 409
        assert len((tmp_path / "sends.jsonl").read_text().splitlines()) == 2
        for previous, successor, count in (("A-1", "A-2", 3), ("A-2", "A-3", 4)):
            (tmp_path / ("release-" + previous)).touch()
            _success(port, key, previous)
            _wait(lambda: len((tmp_path / "sends.jsonl").read_text().splitlines()) == count)
            assert _http(port, key, "GET", "/api/chat-requests/B-1")[1]["status"] == "running"
            assert _http(port, key, "GET", "/api/chat-requests/" + successor)[1]["status"] == "running"
        for name in ("B-1", "A-3"):
            (tmp_path / ("release-" + name)).touch()
            _success(port, key, name)
    sends = [json.loads(x) for x in (tmp_path / "sends.jsonl").read_text().splitlines()]
    assert sorted(sends[:2]) == ["A-1", "B-1"] and sends[2:] == ["A-2", "A-3"]


def test_queued_public_successors_survive_restart_and_unknown_predecessor(tmp_path):
    bodies = [{**_input(tmp_path, f"queued-{i}"), "client_conversation_id": "persistent-chain",
               **({"previous_request_id": f"queued-{i-1}"} if i > 1 else {})} for i in range(1, 4)]
    with _server(tmp_path, "queued") as (port, key, _):
        for body in bodies:
            assert _http(port, key, "POST", "/api/chat-requests", body)[0] == 202
        assert not (tmp_path / "sends.jsonl").exists()
    with _server(tmp_path, "lifecycle") as (port, key, _):
        receipts = [_success(port, key, body["client_request_id"]) for body in bodies]
        from services.task_store import TaskStore
        with TaskStore(tmp_path / "tasks.sqlite3").transaction() as db:
            private_receipts = [r for _, _, _, r in TaskStore(tmp_path / "tasks.sqlite3").receipts(db)]
        assert len({r["conversation_id"] for r in private_receipts}) == 1
        assert len({r["provider_account_identity"] for r in private_receipts}) == 1
    assert [json.loads(x) for x in (tmp_path / "sends.jsonl").read_text().splitlines()] == [b["client_request_id"] for b in bodies]

    # A controlled retained UNKNOWN is not permission to send its accepted child.
    root = tmp_path / "unknown"
    root.mkdir()
    with _server(root, "queued") as (port, key, _):
        for body in bodies[:2]:
            assert _http(port, key, "POST", "/api/chat-requests", body)[0] == 202
    from services.task_store import TaskStore
    store = TaskStore(root / "tasks.sqlite3")
    with store.transaction() as db:
        for kind, owner, request_id, receipt in store.receipts(db):
            if request_id == "queued-1":
                receipt.update(status="unknown", upstream_outcome="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN")
                store.write_receipt(db, kind, owner, request_id, receipt)
    with _server(root, "lifecycle") as (port, key, _):
        def blocked():
            receipt = _http(port, key, "GET", "/api/chat-requests/queued-2")[1]
            return receipt if "CHAT_PREVIOUS_REQUEST_UNKNOWN" in (receipt.get("waiting") or {}).get("reasons", []) else None
        receipt = _wait(blocked)
        assert receipt["status"] == "queued"
        assert _http(port, key, "GET", "/api/chat-requests/queued-1")[1]["status"] == "unknown"
        assert not (root / "sends.jsonl").exists()


_FIXTURE_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


def test_http_stalled_original_cli_restart_and_late_save_without_another_send(tmp_path):
    from examples import image_client
    body = {**_input(tmp_path, "stalled-original"), "client_conversation_id": "stalled-session"}
    (tmp_path / "read-state.json").write_text(json.dumps({
        "schema": "chatgpt2api.chat-request.v1", "request_id": body["client_request_id"], "request": body,
    }))
    def cli(port, key, *args):
        result = subprocess.run([sys.executable, str(Path(image_client.__file__)), *map(str, args)],
                                env={**os.environ, "SERVER_ROOT": f"http://127.0.0.1:{port}", "CHATGPT2API_BEARER_TOKEN": key},
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    with _server(tmp_path, "stalled") as (port, key, _):
        assert _http(port, key, "POST", "/api/chat-requests", body)[0] == 202
        def ended():
            receipt = _http(port, key, "GET", "/api/chat-requests/stalled-original")[1]
            return receipt if receipt.get("execution", {}).get("phase") == "stalled" else None
        receipt = _wait(ended)
        execution = receipt["execution"]
        assert receipt["status"] == "unknown" and execution["wait_state"] == "ended"
        assert execution["send_state"] == "response_received"
        assert execution["resources"]["account_turn"] == "held"
        assert receipt["recovery"]["reason"] == "REQUEST_RESULT_INCOMPLETE"
        assert _http(port, key, "POST", "/api/chat-requests", body)[1]["status"] == "unknown"
    with _server(tmp_path, "stalled") as (port, key, _):
        observed = cli(port, key, "chat-status", "--request-id", body["client_request_id"], "--state", tmp_path / "read-state.json")
        assert observed["execution"]["wait_ended_at"] == execution["wait_ended_at"]
        assert observed["execution"]["resources"]["conversation"] == "protected"
        assert observed["execution"]["send_state"] == "response_received"
        (tmp_path / "original-completed").touch()
        _success(port, key, body["client_request_id"])
        cli(port, key, "chat-save", "--state", tmp_path / "read-state.json",
            "--output", tmp_path / "original-result.json")
        saved = json.loads((tmp_path / "original-result.json").read_text())
        assert saved["content"] == "late original HTTP result"
        assert saved["request_id"] == body["client_request_id"]
    assert (tmp_path / "sends.jsonl").read_text().splitlines() == ['"stalled-original"']


def test_http_stalled_completion_late_original_closes_full_workflow_after_restart(tmp_path):
    from examples import image_client
    state, output = tmp_path / "stalled-state.json", tmp_path / "late-result.json"
    request_id = "stalled-completion-original"
    endpoint = "/api/chat-requests/" + request_id + "/completion"
    def cli(port, key, command, *extra):
        result = subprocess.run([sys.executable, str(Path(image_client.__file__)), command,
                                 "--state", str(state), *map(str, extra)],
            env={**os.environ, "SERVER_ROOT": f"http://127.0.0.1:{port}", "CHATGPT2API_BEARER_TOKEN": key},
            capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    with _server(tmp_path, "stalled-completion") as (port, key, _):
        cli(port, key, "chat-submit", "--request-id", request_id, "--session-id", "stalled-work",
            "--model", "fixture-text", "--prompt", "controlled retained objective")
        _wait(lambda: _http(port, key, "GET", "/api/chat-requests/" + request_id)[1].get(
            "execution", {}).get("phase") == "stalled")
        stopped = cli(port, key, "chat-completion-recover")
        assert stopped["state"] == "needs_attention" and stopped["reason"] == "COMPLETION_ORIGINAL_END_UNCONFIRMED"
        assert not stopped.get("replacement_id")
    with _server(tmp_path, "stalled-completion") as (port, key, _):
        (tmp_path / "original-completed").touch()
        _success(port, key, request_id)
        chosen = cli(port, key, "chat-completion-status")
        assert chosen["selected_id"] == request_id and not chosen.get("replacement_id")
        cli(port, key, "chat-completion-save", "--output", output)
        saved = json.loads(output.read_text())
        assert saved["request_id"] == request_id and saved["content"] == "late original HTTP result"
        assert cli(port, key, "chat-completion-complete", "--reviewed")["state"] == "completed"
        _wait(lambda: _http(port, key, "GET", endpoint)[1].get("work", {}).get("archive", {}).get("status") == "confirmed")
        cli(port, key, "chat-completion-rework")
        _wait(lambda: _http(port, key, "GET", endpoint)[1].get("work", {}).get("state") == "active")
    assert (tmp_path / "sends.jsonl").read_text().splitlines() == [json.dumps(request_id)]


def test_http_same_step_messages_preserve_order_and_reject_middle_input_drift(tmp_path):
    messages = [{"role": role, "content": value} for role, value in (
        ("system", "fixture rule"), ("user", "first question"),
        ("assistant", "prior answer"), ("user", "current question"))]
    body = {**_input(tmp_path, "ordered-input"), "messages": messages}
    with _server(tmp_path, "ordered-input") as (port, key, _):
        assert _http(port, key, "POST", "/api/chat-requests", body)[0] == 202
        _success(port, key, body["client_request_id"])
        changed = json.loads(json.dumps(body))
        changed["messages"][1]["content"] = "changed earlier input"
        assert _http(port, key, "POST", "/api/chat-requests", changed)[0] == 409
    assert json.loads((tmp_path / "received-messages.json").read_text()) == messages
    assert len((tmp_path / "sends.jsonl").read_text().splitlines()) == 1


def test_http_restored_account_wakes_original_id_without_starving_other_owner_workflows(tmp_path):
    from services.task_store import TaskStore
    def body(task_id, workflow, selected=False):
        return {"client_task_id": task_id, "model": "gpt-image-2", "prompt": "controlled image",
                "scheduling": {"workflow_id": workflow, "workflow_concurrency": 1},
                **({"account_ref": "car_" + "B" * 43} if selected else {})}
    with _server(tmp_path, "image-fair-recovery") as (port, _, _):
        keys = json.loads((tmp_path / "fixture-fair-keys.json").read_text())
        jobs = [("same_one", "bound-recovery", "recovery", True),
                ("same_two", "same-owner-healthy", "healthy-a", False),
                ("other", "other-owner-healthy", "healthy-b", False)]
        for key, task_id, workflow, selected in jobs:
            assert _http(port, keys[key], "POST", "/api/image-tasks/generations", body(task_id, workflow, selected))[0] == 200
        _wait(lambda: (tmp_path / "metadata-reads.jsonl").exists())
        assert not (tmp_path / "image-sends.jsonl").exists()
        def success(key, task_id):
            return _http(port, keys[key], "GET", "/api/image-tasks?ids=" + task_id)[1]["items"][0]["status"] == "success"
        (tmp_path / "healthy-recovered").touch()
        _wait(lambda: success("same_two", "same-owner-healthy") and success("other", "other-owner-healthy"), timeout=45)
        assert not success("same_one", "bound-recovery")
        (tmp_path / "quota-recovered").touch()
        _wait(lambda: success("same_one", "bound-recovery"), timeout=45)
    sends = [json.loads(line) for line in (tmp_path / "image-sends.jsonl").read_text().splitlines()]
    assert {row["request_id"] for row in sends[:2]} == {"same-owner-healthy", "other-owner-healthy"}
    assert len(sends) == 3 and sends[-1]["request_id"] == "bound-recovery"
    assert sends[-1]["account"] == "account-B"
    store = TaskStore(tmp_path / "text_tasks.sqlite3")
    with store.connect() as db:
        rows = {request_id: store.read_receipt(db, kind, owner, request_id)
                for kind, owner, request_id, _ in store.receipts(db)}
    assert rows["bound-recovery"]["_source"] == rows["same-owner-healthy"]["_source"]
    assert rows["bound-recovery"]["_source"] != rows["other-owner-healthy"]["_source"]
    assert rows["bound-recovery"]["_requested_account_identity"] == "account-B"


def _serve_images(root, completion=False, fair_recovery=False):
    import pytest
    import uvicorn
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from api import image_tasks, support
    from services.auth_service import AuthService
    from services.storage.json_storage import JSONStorageBackend
    from services.request_context import current_request
    from test.test_image_account_selection import runtime, fresh

    patches = pytest.MonkeyPatch()
    rt = runtime.__wrapped__(root, patches)
    rt.admission.clock = time.time
    rt.update("B", status="限流", **fresh(0))
    if fair_recovery:
        rt.update("A", status="限流", **fresh(0))
    def metadata(token):
        letter = str(token).rsplit("-", 1)[-1]
        recovered = (root / ("healthy-recovered" if letter == "A" and fair_recovery else "quota-recovered")).exists()
        with (root / "metadata-reads.jsonl").open("a") as log:
            log.write(json.dumps({"account": letter, "at": time.monotonic(), "recovered": recovered}) + "\n")
        amount = 3 if recovered else 0
        return ("fixture-user", ""), {**fresh(amount), "quota": amount, "status": "正常" if amount else "限流"}
    rt.accounts._verified_chat_info = metadata
    rt.accounts.refresh_image_capability = rt.accounts._refresh_pool_chat
    def render(body):
        ctx = current_request.get()
        if completion and ctx.request_id == "image-original":
            exc = RuntimeError("controlled known-unsent failure")
            exc.code, exc.upstream_submitted = "IMAGE_GENERATION_NOT_SUBMITTED", False
            raise exc
        ctx.before_send()
        with (root / "image-sends.jsonl").open("a") as log:
            log.write(json.dumps({"request_id": ctx.request_id, "model": body["model"], "prompt": body["prompt"],
                                  "account": ctx.selected_account()["provider_account_identity"]}) + "\n")
        return {"created": 1, "data": [{"b64_json": _FIXTURE_PNG}],
                "_provider_binding_id": body["provider_binding_id"],
                "_provider_account_identity": body["provider_account_identity"],
                "_conversation_id": "controlled-image-conversation", "_parent_message_id": "controlled-image-result",
                **({"_image_thread_terminal": True} if completion else {})}
    rt.tasks.generation_handler = render
    rt.tasks.edit_handler = render
    auth = AuthService(JSONStorageBackend(root / "accounts.json", root / "auth_keys.json"))
    _, secret = auth.create_key(role="user", owner_subject="fixture-person", routes=["chat"])
    (root / "fixture-key").write_text(secret)
    (root / "fixture-key").chmod(0o600)
    if fair_recovery:
        keys = {}
        for name, person in (("same_one", "same"), ("same_two", "same"), ("other", "other")):
            _, keys[name] = auth.create_key(role="user", routes=["chat"], owner_subject="fixture-person-" + person)
        (root / "fixture-fair-keys.json").write_text(json.dumps(keys))
        (root / "fixture-fair-keys.json").chmod(0o600)
    support.auth_service = auth
    image_tasks.image_task_service = rt.tasks
    if completion:
        from services import text_task_service as text_module, image_task_service as image_module
        from services.generation_completion import GenerationCompletionService
        from services.work_lifecycle import WorkLifecycleService
        text_module.text_task_service = text_module.TextTaskService(rt.store.path, admission=rt.admission)
        image_module.image_task_service = rt.tasks
        rt.admission.work_lifecycle = WorkLifecycleService(text_module.text_task_service, rt.tasks)
        rt.admission.generation_completion = GenerationCompletionService(text_module.text_task_service, rt.tasks, rt.admission.work_lifecycle)
        def archive(identity, task_id, archived):
            task = rt.tasks.list_tasks(identity, [task_id])["items"][0]
            return {"task_id": task_id, "image_thread": task["image_thread"], "archived": archived}
        rt.tasks.set_thread_archived = archive
    @asynccontextmanager
    async def lifespan(_app):
        rt.admission.start()
        try:
            yield
        finally:
            rt.admission.stop()
    app = FastAPI(lifespan=lifespan)
    app.include_router(image_tasks.create_router())
    @app.get("/fixture-ready")
    def ready():
        return {"ready": True}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(64)
        listener.setblocking(False)
        (root / "ready.json").write_text(json.dumps({"port": listener.getsockname()[1]}))
        uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False)).run(sockets=[listener])


@pytest.mark.parametrize("same_session", [False, True])
def test_http_completion_cli_saves_reviews_archives_and_restores_selected_result_after_restart(tmp_path, same_session):
    from examples import image_client
    mode = "same-session-completion" if same_session else "completion"
    state = tmp_path / "original-state.json"
    output = tmp_path / "selected-answer.json"
    def cli(port, key, command, *extra, expected=0):
        result = subprocess.run([sys.executable, str(Path(image_client.__file__)), command, "--state", str(state), *map(str, extra)],
            env={**os.environ, "SERVER_ROOT": f"http://127.0.0.1:{port}", "CHATGPT2API_BEARER_TOKEN": key},
            capture_output=True, text=True, timeout=10)
        assert result.returncode == expected, result.stderr
        return json.loads(result.stdout) if expected == 0 else None
    with _server(tmp_path, mode) as (port, key, _):
        cli(port, key, "chat-submit", "--request-id", "completion-original", "--session-id", "original-work",
            "--model", "fixture-text", "--prompt", "Complete this retained objective")
        _wait(lambda: _http(port, key, "GET", "/api/chat-requests/completion-original")[1]["status"] == "unknown")
        requested = cli(port, key, "chat-completion-recover", *([] if same_session else ["--allow-unconfirmed-retry"]))
        assert requested["original_id"] == "completion-original"
        def selected():
            receipt = _http(port, key, "GET", "/api/chat-requests/completion-original/completion")[1]
            return receipt if receipt.get("selected_id") else None
        chosen = _wait(selected)
        child = chosen["selected_id"]
        assert child != "completion-original" and chosen["original_cleanup"] == ("not_required" if same_session else "pending")
        assert chosen["conversation_mode"] == ("original" if same_session else "reconstructed")
        cli(port, key, "chat-completion-complete", "--reviewed", expected=1)
    with _server(tmp_path, mode) as (port, key, _):
        assert cli(port, key, "chat-completion-status")["selected_id"] == child
        cli(port, key, "chat-completion-save", "--output", output)
        saved = json.loads(output.read_text())
        assert saved["request_id"] == child and saved["content"] == "synthetic saved answer"
        done = cli(port, key, "chat-completion-complete", "--reviewed")
        assert done["state"] == "completed" and done["original_cleanup"] == ("not_required" if same_session else "pending")
        endpoint = "/api/chat-requests/completion-original/completion"
        _wait(lambda: _http(port, key, "GET", endpoint)[1].get("work", {}).get("archive", {}).get("status") == "confirmed")
        cli(port, key, "chat-completion-rework")
        _wait(lambda: _http(port, key, "GET", endpoint)[1].get("work", {}).get("state") == "active")
        (tmp_path / "original-completed").touch()
        _success(port, key, "completion-original")
        assert cli(port, key, "chat-completion-status")["selected_id"] == child
        assert json.loads(state.read_text())["request_id"] == "completion-original"
        assert json.loads(state.read_text())["completion_output"]["selected_id"] == child
    assert [json.loads(line) for line in (tmp_path / "sends.jsonl").read_text().splitlines()] == ["completion-original", child]
    archives = [json.loads(line) for line in (tmp_path / "archives.jsonl").read_text().splitlines()]
    assert any(a["request_id"] == child and a["archived"] for a in archives)
    assert any(a["request_id"] == child and not a["archived"] for a in archives)
    if same_session:
        assert all(a["conversation_id"] == "controlled-original-conversation" for a in archives)


def test_http_image_completion_cli_downloads_real_bytes_before_acknowledging(tmp_path):
    import base64
    from examples import image_client
    state, output = tmp_path / "image-state.json", tmp_path / "selected.png"
    with _server(tmp_path, "image-completion") as (port, key, _):
        def cli(command, *extra):
            result = subprocess.run([sys.executable, str(Path(image_client.__file__)), command, "--state", str(state), *map(str, extra)],
                env={**os.environ, "SERVER_ROOT": f"http://127.0.0.1:{port}", "CHATGPT2API_BEARER_TOKEN": key},
                capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, result.stderr
            return json.loads(result.stdout)
        cli("submit", "--client-task-id", "image-original", "--thread-id", "image-work", "--prompt", "controlled image", "--model", "gpt-image-2")
        _wait(lambda: _http(port, key, "GET", "/api/image-tasks?ids=image-original")[1]["items"][0]["status"] == "error")
        cli("completion-recover")  # Confirmed unsent failure needs no UNKNOWN opt-in.
        endpoint = "/api/image-tasks/image-original/completion"
        chosen = _wait(lambda: (r if (r := _http(port, key, "GET", endpoint)[1]).get("selected_id") else None))
        assert chosen["selected_id"] != "image-original"
        cli("completion-save", "--output", output)
        assert output.read_bytes() == base64.b64decode(_FIXTURE_PNG)
        assert cli("completion-complete", "--reviewed")["state"] == "completed"
        _wait(lambda: _http(port, key, "GET", endpoint)[1].get("work", {}).get("archive", {}).get("status") == "confirmed")
        cli("completion-rework")
        _wait(lambda: _http(port, key, "GET", endpoint)[1].get("work", {}).get("state") == "active")
    assert len((tmp_path / "image-sends.jsonl").read_text().splitlines()) == 1


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
    if mode in {"image-recovery", "image-completion", "image-fair-recovery"}:
        return _serve_images(root, completion=mode == "image-completion", fair_recovery=mode == "image-fair-recovery")

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
    if mode == "completion" and len(accounts.list_accounts()) == 1:
        accounts.add_account_items([{"access_token": "fixture-second-never-sent", "source_type": "web",
                                    "type": "Plus", "status": "正常"}])
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
    from services.config import config
    config.data["ai_review"] = {"enabled": False}
    store = TaskStore(root / "tasks.sqlite3")
    admission = PoolAdmission(store, accounts, settings=lambda: {
        "chat_account_concurrency": 2 if mode == "parallel" else 1, "image_account_concurrency": 1, "codex_max_concurrency": 1},
        model_types=lambda _model: {"Plus"}, pacing=lambda _account, now: {"next_at": now})

    def synthetic_model(body, on_cursor):
        current_request.get().before_send()
        if mode == "ordered-input":
            (root / "received-messages.json").write_text(json.dumps(body["messages"]))
        with (root / "sends.jsonl").open("a") as log:
            log.write(json.dumps(body["client_request_id"]) + "\n")
            log.flush()
            os.fsync(log.fileno())
        if mode in {"stalled", "stalled-completion"} or mode in {"completion", "same-session-completion"} and body["client_request_id"] == "completion-original":
            from services.conversation_binding_service import ConversationBindingError
            current_request.get().record_stage("send_call_started")
            current_request.get().record_stage("response_headers_received", status_code=200)
            on_cursor({"conversation_id": "controlled-original-conversation"})
            raise ConversationBindingError("controlled interrupted stream", code="CONVERSATION_OUTCOME_UNKNOWN")
        if mode == "blocked-send":
            _wait(lambda: (root / "release-send").exists(), timeout=10)
        if mode == "parallel":
            _wait(lambda: (root / ("release-" + body["client_request_id"])).exists(), timeout=10)
        result = {"content": "synthetic saved answer"}
        if mode in {"lifecycle", "parallel", "completion", "same-session-completion"}:
            conversation = body.get("conversation_id") or "upstream-" + body["client_conversation_id"]
            if body.get("_continue_after_terminal_empty"):
                with store.connect() as db:
                    previous = store.read_receipt(db, "text", current_request.get().owner, body["_previous_request_id"])
                assert body["parent_message_id"] == previous["_turn_end_evidence"]["final_message_id"]
            elif body.get("_previous_request_id"):
                assert body["parent_message_id"] == "answer-" + body["_previous_request_id"]
            result.update(provider_binding_id=body.get("provider_binding_id") or "fixture-binding",
                          provider_account_identity=body.get("provider_account_identity") or "fixture-account",
                          conversation_id=conversation, parent_message_id="answer-" + body["client_request_id"],
                          binding_status="bound")
        return result

    tasks = TextTaskService(store.path, runner=synthetic_model, admission=admission)
    if mode in {"stalled", "stalled-completion", "completion", "same-session-completion"}:
        # Accelerate only this isolated child fixture. Production's 900-second
        # boundary is exercised with a manual clock in the service regressions.
        TextTaskService.UNRECOVERABLE_MIN_AGE_SECONDS = .05
        TextTaskService.RECOVERY_BASE_BACKOFF_SECONDS = .03
        TextTaskService.RECOVERY_MAX_BACKOFF_SECONDS = .1
        admission.CLAIM_SECONDS = .1
        from services.conversation_binding_service import ConversationBindingService
        from test.test_unknown_turn_recovery import document
        from test.test_stalled_text_diagnostics import mixed_chain
        def original_read(row):
            doc = document(row)
            mixed_chain(doc)
            message = doc["mapping"]["final-" + row["request_message_id"]]["message"]
            if (root / "original-completed").exists():
                message["content"]["parts"] = ["late original HTTP result"]
            else:
                message.update(status="finished_successfully" if mode == "same-session-completion" else "in_progress",
                               end_turn=True if mode == "same-session-completion" else None)
                message["content"]["parts"] = [""]
            return ConversationBindingService._read_text_request_result(None, row, document=doc)
        tasks.recovery_reader = original_read
    admission.register("text", lambda context, body: tasks._run(context.owner, context.request_id, body))
    chat_requests.text_task_service = tasks
    if mode in {"lifecycle", "stalled-completion", "completion", "same-session-completion"}:
        from services import text_task_service as text_module
        from services.work_lifecycle import WorkLifecycleService
        text_module.text_task_service = tasks
        admission.work_lifecycle = WorkLifecycleService(tasks, SimpleNamespace())
        def synthetic_archive(receipt, archived):
            with (root / "archives.jsonl").open("a") as log:
                log.write(json.dumps({"request_id": receipt["request_id"],
                                      "conversation_id": receipt["conversation_id"], "archived": archived}) + "\n")
            if (root / "fail-archive").exists():
                raise TimeoutError("controlled lost archive response")
            return {"archived": archived}
        text_module.conversation_binding_service.set_archived = synthetic_archive
        if mode in {"stalled-completion", "completion", "same-session-completion"}:
            from services.generation_completion import GenerationCompletionService
            GenerationCompletionService.STALL_SECONDS = .05
            GenerationCompletionService.INVESTIGATION_SECONDS = .05
            GenerationCompletionService.RECHECK_SECONDS = .05
            admission.generation_completion = GenerationCompletionService(tasks, SimpleNamespace(), admission.work_lifecycle)

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
