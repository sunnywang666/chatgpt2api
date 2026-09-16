from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


CLIENT_DIR = Path(__file__).resolve().parents[1] / "examples" / "codex_client"
sys.path.insert(0, str(CLIENT_DIR))
from list_models import fetch_models  # noqa: E402
from launch_codex import codex_environment  # noqa: E402
from mock_provider import TOOL_CYCLE, function_call_item, function_output_call_ids  # noqa: E402
from prepare_acceptance import GENERATED_MARKER, normalized_base_url, prepare  # noqa: E402


class _ModelsHandler(BaseHTTPRequestHandler):
    seen_authorization = ""

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        type(self).seen_authorization = self.headers.get("authorization", "")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"object": "list", "data": [{"id": "codex-test"}]}).encode())


class CodexClientTests(unittest.TestCase):
    def test_base_url_requires_absolute_http_url_without_query(self) -> None:
        self.assertEqual(normalized_base_url("https://example.test/v1/"), "https://example.test/v1")
        with self.assertRaises(ValueError):
            normalized_base_url("example.test/v1")
        with self.assertRaises(ValueError):
            normalized_base_url("https://example.test/v1?key=not-allowed")

    def test_prepare_isolated_state_has_no_key_and_fixture_starts_broken(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = prepare(root / "state", root / "fixture", "https://provider.example/v1", "codex-test")
            config = Path(result["config"]).read_text(encoding="utf-8")
            self.assertTrue(config.startswith(GENERATED_MARKER))
            self.assertIn('env_key = "OPENAI_API_KEY"', config)
            self.assertIn("request_max_retries = 0", config)
            self.assertIn("stream_max_retries = 0", config)
            self.assertNotIn("sk-", config)
            self.assertNotIn("api_key =", config)
            self.assertIn("return left + right", (root / "fixture" / "calculator.py").read_text(encoding="utf-8"))

    def test_prepare_refuses_to_replace_foreign_state_config(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir()
            (state / "config.toml").write_text("model = 'foreign'\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                prepare(state, root / "fixture", "https://provider.example/v1", "codex-test")

    def test_model_preflight_uses_bearer_header_without_persisting_the_key(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            self.assertEqual(fetch_models(f"http://127.0.0.1:{server.server_port}/v1", "fake-test-key"), ["codex-test"])
            self.assertEqual(_ModelsHandler.seen_authorization, "Bearer fake-test-key")
        finally:
            server.shutdown()
            server.server_close()

    def test_launchers_keep_the_key_out_of_argument_construction(self) -> None:
        shell = (CLIENT_DIR / "run-codex.sh").read_text(encoding="utf-8")
        powershell = (CLIENT_DIR / "run-codex.ps1").read_text(encoding="utf-8")
        bridge = (CLIENT_DIR / "launch_codex.py").read_text(encoding="utf-8")
        self.assertIn("launch_codex.py", shell)
        self.assertIn("launch_codex.py", powershell)
        self.assertIn('child.pop("CODEX_PROVIDER_API_KEY", None)', bridge)
        self.assertIn('child["OPENAI_API_KEY"] = key', bridge)
        self.assertIn("os.execvpe", bridge)
        self.assertNotIn("OPENAI_API_KEY=\"$CODEX_PROVIDER_API_KEY\"", shell)
        self.assertNotIn("--api-key", shell)
        self.assertNotIn("--api-key", powershell)

    def test_exec_bridge_keeps_key_out_of_codex_source_environment(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"CODEX_PROVIDER_API_KEY": "test-key", "CODEX_HOME": "/normal", "OPENAI_API_KEY": "normal-key"},
            clear=True,
        ):
            environment = codex_environment(Path("/isolated"))
        self.assertEqual(environment["CODEX_HOME"], "/isolated")
        self.assertEqual(environment["OPENAI_API_KEY"], "test-key")
        self.assertNotIn("CODEX_PROVIDER_API_KEY", environment)

    def test_mock_logs_header_names_but_not_values(self) -> None:
        mock = (CLIENT_DIR / "mock_provider.py").read_text(encoding="utf-8")
        self.assertIn('"header_names": headers', mock)
        self.assertNotIn('"headers": headers', mock)
        self.assertIn("function output can contain secrets", mock)

    def test_mock_cycle_uses_actual_exec_command_and_tracks_only_call_ids(self) -> None:
        self.assertEqual([call_id for call_id, _command in TOOL_CYCLE], ["call_read_fixture", "call_repair_fixture", "call_test_fixture"])
        item = function_call_item(*TOOL_CYCLE[0])
        self.assertEqual(item["type"], "function_call")
        self.assertEqual(item["name"], "exec_command")
        self.assertEqual(function_output_call_ids({"input": [{"type": "function_call_output", "call_id": "call_read_fixture", "output": "private output"}]}), ["call_read_fixture"])

    def test_candidate_qa_uses_only_harness_url_mapping_and_reports_no_key_material(self) -> None:
        qa = (CLIENT_DIR / "candidate_chain_qa.py").read_text(encoding="utf-8")
        self.assertIn("class MappedSession", qa)
        self.assertIn("CODEX_MODELS_URL", qa)
        self.assertIn("candidate_service_mode = True", qa)
        self.assertIn('"key_material_logged": False', qa)
        self.assertIn("revoked_get, revoked_post, admin_status", qa)


if __name__ == "__main__":
    unittest.main()
