from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock
from urllib import parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "examples" / "image_client.py"
SPEC = importlib.util.spec_from_file_location("external_image_client", CLIENT_PATH)
assert SPEC is not None and SPEC.loader is not None
image_client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(image_client)

TOKEN = "fixture-bearer-secret"
PNG = b"\x89PNG\r\n\x1a\nfixture"


class FixtureState:
    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, object]] = {}
        self.requests: list[dict[str, object]] = []
        self.expected_state_path: Path | None = None
        self.prepared_state_seen = False
        self.download_redirect = ""


def fixture_handler(state: FixtureState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *_args: object) -> None:
            pass

        def _record(self, body: bytes = b"") -> None:
            state.requests.append({
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "image_client": self.headers.get("X-Workbench-Image-Client"),
                "content_type": self.headers.get("Content-Type", ""),
                "body": body,
            })

        def _json(self, status: int, value: object) -> None:
            body = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def _body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0"))
            return self.rfile.read(length)

        def do_GET(self) -> None:
            self._record()
            parsed = parse.urlsplit(self.path)
            if parsed.path == "/ai/v1/models":
                self._json(200, {"object": "list", "data": [{"id": "fixture-image-model"}]})
                return
            if parsed.path == "/ai/api/image-tasks":
                ids = parse.parse_qs(parsed.query).get("ids", [""])[0].split(",")
                ids = [item for item in ids if item]
                self._json(200, {
                    "items": [state.tasks[item] for item in ids if item in state.tasks],
                    "missing_ids": [item for item in ids if item not in state.tasks],
                })
                return
            prefix = "/ai/api/image-tasks/"
            suffix = "/images/0"
            if parsed.path.startswith(prefix) and parsed.path.endswith(suffix):
                if state.download_redirect:
                    self.send_response(302)
                    self.send_header("Location", state.download_redirect)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = PNG
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._json(404, {"detail": {"error": "not found"}})

        def do_POST(self) -> None:
            body = self._body()
            self._record(body)
            parsed = parse.urlsplit(self.path)
            if parsed.path in {"/ai/api/image-tasks/generations", "/ai/api/image-tasks/edits"}:
                if state.expected_state_path is not None and state.expected_state_path.exists():
                    durable = json.loads(state.expected_state_path.read_text(encoding="utf-8"))
                    state.prepared_state_seen = durable.get("phase") == "prepared"
                if parsed.path.endswith("generations"):
                    payload = json.loads(body)
                    task_id = payload["client_task_id"]
                    prompt = payload["prompt"]
                    mode = "generate"
                else:
                    content_type = self.headers["Content-Type"]
                    boundary = content_type.split("boundary=", 1)[1].encode()
                    task_id = body.split(b'name="client_task_id"\r\n\r\n', 1)[1].split(b"\r\n", 1)[0].decode()
                    prompt = body.split(b'name="prompt"\r\n\r\n', 1)[1].split(b"\r\n", 1)[0].decode()
                    self.server.multipart_boundary = boundary  # type: ignore[attr-defined]
                    mode = "edit"
                task = {"id": task_id, "status": "queued", "mode": mode, "data": []}
                state.tasks[task_id] = task
                if prompt == "timeout-after-admission":
                    time.sleep(0.25)
                self._json(200, task)
                return
            prefix = "/ai/api/image-tasks/"
            suffix = "/resume-poll"
            if parsed.path.startswith(prefix) and parsed.path.endswith(suffix):
                encoded = parsed.path[len(prefix):-len(suffix)]
                task_id = parse.unquote(encoded)
                task = dict(state.tasks[task_id])
                task["status"] = "running"
                state.tasks[task_id] = task
                self._json(200, task)
                return
            self._json(404, {"detail": {"error": "not found"}})

    return Handler


class SinkHandler(BaseHTTPRequestHandler):
    hits: list[str] = []

    def log_message(self, _format: str, *_args: object) -> None:
        pass

    def do_GET(self) -> None:
        type(self).hits.append(self.headers.get("Authorization", ""))
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(PNG)))
        self.end_headers()
        self.wfile.write(PNG)


class ExternalImageClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.work = Path(self.temp_dir.name)
        self.fixture = FixtureState()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), fixture_handler(self.fixture))
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        host, port = self.server.server_address
        self.root = f"http://{host}:{port}/ai"

    def run_client(self, *args: str, timeout: float = 1.0) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = ["--server-root", self.root, "--timeout", str(timeout), *args]
        with (
            mock.patch.dict(os.environ, {"CHATGPT2API_BEARER_TOKEN": TOKEN}, clear=False),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = image_client.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_blank_optional_state_env_uses_safe_default_filename(self) -> None:
        with mock.patch.dict(os.environ, {"IMAGE_CLIENT_STATE": ""}, clear=False):
            self.assertEqual(
                image_client._state_path(argparse.Namespace(state=None)),
                Path(".image-client-task.json"),
            )

    def test_models_uses_v1_beside_api_under_server_root_prefix(self) -> None:
        code, stdout, stderr = self.run_client("models")
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(stdout)["data"][0]["id"], "fixture-image-model")
        self.assertEqual(self.fixture.requests[0]["path"], "/ai/v1/models")
        self.assertEqual(self.fixture.requests[0]["authorization"], f"Bearer {TOKEN}")
        self.assertEqual(self.fixture.requests[0]["image_client"], "1")

    def test_submit_persists_before_post_and_restart_queries_without_repost(self) -> None:
        state_path = self.work / "generation.json"
        self.fixture.expected_state_path = state_path
        command = (
            "submit", "--state", str(state_path), "--client-task-id", "stable-task-1",
            "--prompt", "draw a fixture", "--model", "fixture-image-model",
        )
        code, stdout, stderr = self.run_client(*command)
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(stdout)["id"], "stable-task-1")
        self.assertTrue(self.fixture.prepared_state_seen)
        durable_text = state_path.read_text(encoding="utf-8")
        durable = json.loads(durable_text)
        self.assertEqual(durable["phase"], "accepted")
        self.assertEqual(durable["client_task_id"], "stable-task-1")
        self.assertTrue(durable["input_fingerprint"].startswith("sha256:"))
        self.assertNotIn("draw a fixture", durable_text)
        self.assertNotIn(TOKEN, durable_text)

        code, stdout, stderr = self.run_client(*command)
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(stdout)["id"], "stable-task-1")
        posts = [item for item in self.fixture.requests if item["method"] == "POST"]
        self.assertEqual(len(posts), 1)
        self.assertIn("ids=stable-task-1", str(self.fixture.requests[-1]["path"]))

        changed = list(command)
        changed[changed.index("draw a fixture")] = "different input"
        code, stdout, stderr = self.run_client(*changed)
        self.assertEqual((code, stdout), (1, ""))
        self.assertIn("different immutable input", stderr)
        posts = [item for item in self.fixture.requests if item["method"] == "POST"]
        self.assertEqual(len(posts), 1)

    def test_edit_uses_repeated_local_images_and_content_fingerprints(self) -> None:
        first = self.work / "front.png"
        second = self.work / "detail.jpg"
        first.write_bytes(b"front")
        second.write_bytes(b"detail")
        state_path = self.work / "edit.json"
        self.fixture.expected_state_path = state_path
        code, stdout, stderr = self.run_client(
            "submit", "--state", str(state_path), "--client-task-id", "edit-task",
            "--prompt", "edit fixture", "--model", "fixture-image-model",
            "--image", str(first), "--image", str(second),
        )
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(stdout)["mode"], "edit")
        post = [item for item in self.fixture.requests if item["method"] == "POST"][0]
        self.assertEqual(post["path"], "/ai/api/image-tasks/edits")
        self.assertEqual(bytes(post["body"]).count(b'name="image"'), 2)
        durable = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(durable["input"]["mode"], "edit")
        self.assertEqual(len(durable["input"]["images"]), 2)
        self.assertNotIn("front.png", state_path.read_text(encoding="utf-8"))

    def test_timeout_keeps_original_id_and_next_submit_only_queries(self) -> None:
        state_path = self.work / "timeout.json"
        self.fixture.expected_state_path = state_path
        command = (
            "submit", "--state", str(state_path), "--client-task-id", "timeout-task",
            "--prompt", "timeout-after-admission", "--model", "fixture-image-model",
        )
        code, stdout, stderr = self.run_client(*command, timeout=0.05)
        self.assertEqual((code, stdout), (1, ""))
        self.assertIn("result is unknown", stderr)
        durable = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(durable["phase"], "unknown")
        self.assertEqual(durable["client_task_id"], "timeout-task")

        code, stdout, stderr = self.run_client(*command)
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(stdout)["id"], "timeout-task")
        posts = [item for item in self.fixture.requests if item["method"] == "POST"]
        self.assertEqual(len(posts), 1)

    def test_status_and_resume_use_the_exact_original_client_id(self) -> None:
        task_id = "original.task:1"
        self.fixture.tasks[task_id] = {
            "id": task_id,
            "status": "error",
            "mode": "generate",
            "error_code": "CONVERSATION_OUTCOME_UNKNOWN",
        }
        absent_state = self.work / "absent.json"
        code, stdout, stderr = self.run_client(
            "status", "--state", str(absent_state), "--task-id", task_id,
        )
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(stdout)["id"], task_id)

        code, stdout, stderr = self.run_client(
            "resume", "--state", str(absent_state), "--task-id", task_id,
            "--extra-timeout-secs", "45",
        )
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(stdout)["status"], "running")
        self.assertEqual(
            self.fixture.requests[-1]["path"],
            "/ai/api/image-tasks/original.task%3A1/resume-poll",
        )
        self.assertEqual(json.loads(bytes(self.fixture.requests[-1]["body"]))["extra_timeout_secs"], 45.0)

    def test_task_ids_must_match_the_server_url_safe_contract(self) -> None:
        absent_state = self.work / "absent.json"
        for task_id in ("contains space", "contains/slash", "a" * 201, ""):
            with self.subTest(task_id=task_id):
                code, stdout, stderr = self.run_client(
                    "status", "--state", str(absent_state), "--task-id", task_id,
                )
                self.assertEqual((code, stdout), (1, ""))
                self.assertIn("1..200 characters", stderr)
        self.assertEqual(self.fixture.requests, [])

    def test_download_is_authenticated_safe_and_never_follows_cross_origin(self) -> None:
        task_id = "download-task"
        self.fixture.tasks[task_id] = {
            "id": task_id,
            "status": "success",
            "mode": "generate",
            "data": [{"url": "https://storage.invalid/raw-image.png"}],
        }
        output = self.work / "result.png"
        absent_state = self.work / "absent.json"
        code, stdout, stderr = self.run_client(
            "download", "--state", str(absent_state), "--task-id", task_id,
            "--output", str(output),
        )
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(output.read_bytes(), PNG)
        self.assertEqual(json.loads(stdout)["bytes"], len(PNG))
        download_request = self.fixture.requests[-1]
        self.assertEqual(download_request["path"], "/ai/api/image-tasks/download-task/images/0")
        self.assertEqual(download_request["authorization"], f"Bearer {TOKEN}")

        code, stdout, stderr = self.run_client(
            "download", "--state", str(absent_state), "--task-id", task_id,
            "--output", str(output),
        )
        self.assertEqual((code, stdout), (1, ""))
        self.assertIn("refusing to overwrite", stderr)

        SinkHandler.hits = []
        sink = ThreadingHTTPServer(("127.0.0.1", 0), SinkHandler)
        sink.daemon_threads = True
        sink_thread = threading.Thread(target=sink.serve_forever, daemon=True)
        sink_thread.start()
        self.addCleanup(sink.server_close)
        self.addCleanup(sink.shutdown)
        sink_host, sink_port = sink.server_address
        self.fixture.download_redirect = f"http://{sink_host}:{sink_port}/stolen.png"
        redirected_output = self.work / "redirected.png"
        code, stdout, stderr = self.run_client(
            "download", "--state", str(absent_state), "--task-id", task_id,
            "--output", str(redirected_output),
        )
        self.assertEqual((code, stdout), (1, ""))
        self.assertIn("origin-changing redirect", stderr)
        self.assertEqual(SinkHandler.hits, [])
        self.assertFalse(redirected_output.exists())


if __name__ == "__main__":
    unittest.main()
