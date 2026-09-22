"""Public Chat client persists identity across process exit and uncertain POST."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CLIENT = Path(__file__).resolve().parents[1] / "examples/image_client.py"


@contextmanager
def session_client(directory):
    calls, receipts, disconnect = [], {}, set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, request_id):
            receipt = receipts.get(request_id)
            body = json.dumps(receipt or {"detail": "not found"}).encode()
            self.send_response(200 if receipt else 404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            request_id = payload["client_request_id"]
            saved = json.loads((Path(directory) / f"{request_id}.json").read_text())
            calls.append(("POST", payload, saved))
            receipts.setdefault(request_id, {"request_id": request_id, "route": "chat", "status": "succeeded",
                "content": f"result-{request_id}", "conversation": {"protocol": "sequential-v1",
                    "client_conversation_id": payload.get("client_conversation_id"),
                    "previous_request_id": payload.get("previous_request_id")}})
            if request_id in disconnect:
                self.close_connection = True
            else:
                self.reply(request_id)

        def do_GET(self):
            request_id = self.path.rsplit("/", 1)[-1]
            calls.append(("GET", request_id))
            self.reply(request_id)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def run(request_id, previous=None, session="work", command="chat-submit"):
        args = [sys.executable, str(CLIENT), "--server-root", f"http://127.0.0.1:{server.server_port}/ai",
                "--timeout", "1", command, "--state", str(Path(directory) / f"{request_id}.json")]
        if command == "chat-submit":
            args += ["--request-id", request_id, "--model", "fixture-text", "--prompt", f"delta-{request_id}",
                     "--session-id", session]
            if previous:
                args += ["--previous-request-id", previous]
        return subprocess.run(args, env={**os.environ, "CHATGPT2API_BEARER_TOKEN": "fixture-only"},
                              capture_output=True, text=True)

    try:
        yield run, calls, receipts, disconnect
    finally:
        server.shutdown()
        server.server_close()


class ChatClientTest(unittest.TestCase):
    def test_invalid_session_references_cannot_silently_become_a_new_conversation(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "invalid.json"
            base = [sys.executable, str(CLIENT), "--server-root", "http://127.0.0.1:1/ai", "chat-submit",
                    "--state", str(state), "--request-id", "current", "--model", "fixture", "--prompt", "safe input"]
            variants = [["--session-id", value] for value in ("", ".", "..")]
            variants += [["--previous-request-id", "previous"], ["--session-id", "work", "--previous-request-id", ""],
                         ["--session-id", "work", "--previous-request-id", "current"]]
            for args in variants:
                with self.subTest(args=args):
                    result = subprocess.run(base + args, env={**os.environ, "CHATGPT2API_BEARER_TOKEN": "fixture-only"},
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 1)
                    self.assertNotIn("Connection refused", result.stderr)
                    self.assertFalse(state.exists())

    def test_three_turn_session_uses_delta_and_recovers_original_after_process_exit(self):
        with tempfile.TemporaryDirectory() as directory, session_client(directory) as (run, calls, receipts, disconnect):
            first = run("one")
            self.assertEqual(first.returncode, 0, first.stderr)
            disconnect.add("two")
            second = run("two", "one")
            self.assertEqual(second.returncode, 1)
            self.assertEqual(json.loads((Path(directory) / "two.json").read_text())["phase"], "unknown")
            recovered = run("two", command="chat-status")
            self.assertEqual(recovered.returncode, 0, recovered.stderr)
            self.assertEqual(json.loads(recovered.stdout)["content"], "result-two")
            third = run("three", "two")
            self.assertEqual(third.returncode, 0, third.stderr)
            # Re-entering an older accepted turn reads that result even after its session advanced.
            for request_id, previous in (("one", None), ("two", "one")):
                result = run(request_id, previous)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["content"], f"result-{request_id}")
            before_drift = len(calls)
            for result in (run("two", "three"), run("two", "one", session="other")):
                self.assertEqual(result.returncode, 1)
                self.assertIn("immutable input", result.stderr)
            self.assertEqual(len(calls), before_drift)
            posts = [call for call in calls if call[0] == "POST"]
            self.assertEqual(len(posts), 3)
            for (_, payload, saved), request_id, previous in zip(posts, ("one", "two", "three"), (None, "one", "two")):
                self.assertEqual(payload["client_conversation_id"], "work")
                self.assertEqual(payload.get("previous_request_id"), previous)
                self.assertEqual(payload["messages"], [{"role": "user", "content": [{"type": "text", "text": f"delta-{request_id}"}]}])
                self.assertEqual(saved["phase"], "prepared")
                self.assertEqual(saved["conversation"], receipts[request_id]["conversation"])
                text = (Path(directory) / f"{request_id}.json").read_text()
                self.assertNotIn("fixture-only", text)
                self.assertNotIn(f"delta-{request_id}", text)

    def test_incomplete_or_unconfirmed_predecessor_never_submits_next_turn(self):
        with tempfile.TemporaryDirectory() as directory, session_client(directory) as (run, calls, receipts, _):
            for index, status in enumerate(("queued", "running", "not_started", "unknown", "failed", "missing", "wrong_session", "old_protocol")):
                with self.subTest(status=status):
                    receipt = {"request_id": "previous", "route": "chat", "status": status, "conversation": {
                        "protocol": "sequential-v1", "client_conversation_id": "work", "previous_request_id": None}}
                    if status in {"wrong_session", "old_protocol"}:
                        receipt["status"] = "succeeded"
                        receipt["conversation"]["client_conversation_id" if status == "wrong_session" else "protocol"] = "other"
                    receipts["previous"] = None if status == "missing" else receipt
                    result = run(f"next-{index}", "previous")
                    self.assertEqual(result.returncode, 1)
                    self.assertFalse((Path(directory) / f"next-{index}.json").exists())
            self.assertTrue(all(call[0] == "GET" for call in calls))
            self.assertEqual(len(calls), 8)

    def test_missing_session_confirmation_preserves_original_and_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory, session_client(directory) as (run, calls, receipts, _):
            receipts["one"] = {"request_id": "one", "route": "chat", "status": "succeeded", "content": "unconfirmed"}
            result = run("one")
            self.assertEqual(result.returncode, 1)
            state = json.loads((Path(directory) / "one.json").read_text())
            self.assertEqual(state["phase"], "unknown")
            self.assertEqual(state["conversation"]["client_conversation_id"], "work")
            for result in (run("one"), run("one", command="chat-status")):
                self.assertEqual(result.returncode, 1)
                self.assertIn("does not confirm", result.stderr)
            self.assertEqual([call[0] for call in calls], ["POST", "GET", "GET"])

    def test_dot_segment_ids_rejected_before_state_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            for request_id in (".", ".."):
                with self.subTest(request_id=request_id):
                    state = Path(directory) / "state.json"
                    result = subprocess.run(
                        [sys.executable, str(CLIENT), "--server-root", "http://127.0.0.1:1/ai",
                         "chat-submit", "--state", str(state), "--request-id", request_id,
                         "--model", "fixture-text", "--prompt", "safe test"],
                        env={**os.environ, "CHATGPT2API_BEARER_TOKEN": "fixture-only"},
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertIn("cannot be . or ..", result.stderr)
                    self.assertFalse(state.exists())

    def test_lost_response_restart_queries_same_id_and_drift_never_posts(self):
        posts, gets = [], []
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_):
                    pass

                def do_POST(self):
                    payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    saved = json.loads(state.read_text())
                    posts.append((payload, saved))
                    # The request was accepted, but the response disappeared.
                    self.close_connection = True

                def do_GET(self):
                    gets.append(self.path)
                    body = json.dumps({"request_id": posts[0][0]["client_request_id"],
                                       "route": "chat", "model": "fixture-text", "status": "succeeded", "content": "ok"}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                base = [sys.executable, str(CLIENT), "--server-root", f"http://127.0.0.1:{server.server_port}/ai", "--timeout", "1"]
                env = {**os.environ, "CHATGPT2API_BEARER_TOKEN": "fixture-secret-not-output"}
                submit = ["chat-submit", "--state", str(state), "--request-id", "original", "--model", "fixture-text", "--prompt", "private test input"]
                first = subprocess.run(base + submit, env=env, capture_output=True, text=True)
                self.assertEqual(first.returncode, 1, first.stderr)
                self.assertEqual(posts[0][1]["phase"], "prepared")
                self.assertEqual(posts[0][1]["request_id"], "original")
                self.assertEqual(json.loads(state.read_text())["phase"], "unknown")
                repeat = subprocess.run(base + submit, env=env, capture_output=True, text=True)
                self.assertEqual(repeat.returncode, 0, repeat.stderr)
                self.assertEqual(json.loads(repeat.stdout)["content"], "ok")
                status = subprocess.run(base + ["chat-status", "--state", str(state)], env=env, capture_output=True, text=True)
                self.assertEqual(status.returncode, 0, status.stderr)
                drift = subprocess.run(base + submit[:-1] + ["changed"], env=env, capture_output=True, text=True)
                self.assertEqual(drift.returncode, 1)
                self.assertEqual(len(posts), 1)
                self.assertEqual(gets, ["/ai/api/chat-requests/original"] * 2)
                combined = state.read_text() + first.stdout + first.stderr + repeat.stdout + repeat.stderr
                self.assertNotIn("fixture-secret-not-output", combined)
                self.assertNotIn("private test input", state.read_text())
                self.assertEqual(state.stat().st_mode & 0o777, 0o600)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
