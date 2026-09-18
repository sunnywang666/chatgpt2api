"""Public Chat client persists identity across process exit and uncertain POST."""
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


class ChatClientTest(unittest.TestCase):
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
