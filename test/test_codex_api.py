from __future__ import annotations

import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.codex as codex_api
from services.codex_service import CodexHTTPResponse, CodexServiceError


AUTH = {"Authorization": "Bearer user-key"}


class FakeCodexService:
    def __init__(self):
        self.calls = []
        self.next_result = CodexHTTPResponse(200, {"content-type": "application/json"}, body=b'{"ok":true}')
        self.error = None

    def list_native_models(self, identity, headers):
        self.calls.append(("models", identity, dict(headers)))
        if self.error:
            raise self.error
        return self.next_result

    def submit(self, identity, payload, headers, *, compact=False):
        self.calls.append(("compact" if compact else "responses", identity, payload, dict(headers)))
        if self.error:
            raise self.error
        return self.next_result


class CodexApiTests(unittest.TestCase):
    def setUp(self):
        self.service = FakeCodexService()
        self.service_patch = mock.patch.object(codex_api, "codex_service", self.service)
        self.service_patch.start()
        self.addCleanup(self.service_patch.stop)
        self.identity_patch = mock.patch.object(
            codex_api,
            "require_identity",
            return_value={"id": "key-id", "name": "CLI", "role": "user", "enabled": True,
                          "policy": {"version": 1, "revision": 1, "capabilities": ["codex_coding"]}},
        )
        self.identity = self.identity_patch.start()
        self.addCleanup(self.identity_patch.stop)
        app = FastAPI()
        app.include_router(codex_api.create_router())
        self.client = TestClient(app)

    def test_models_requires_user_and_returns_native_catalog(self):
        self.service.next_result = CodexHTTPResponse(
            200,
            {"content-type": "application/json", "set-cookie": "must-not-pass"},
            body=b'{"models":[{"slug":"gpt-5.6-codex","raw_details":{"context_window":258400}}]}',
        )
        response = self.client.get("/codex/v1/models", headers={**AUTH, "session-id": "session-1"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["models"][0]["raw_details"]["context_window"], 258400)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertNotIn("set-cookie", response.headers)
        self.assertEqual(self.service.calls[0][0], "models")

    def test_admin_identity_is_forbidden_before_service_call(self):
        self.identity.return_value = {"id": "admin", "role": "admin"}
        response = self.client.get("/codex/v1/models", headers=AUTH)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.service.calls, [])

    def test_responses_preserves_native_tool_and_reasoning_schema(self):
        payload = {
            "model": "gpt-5.6-codex",
            "input": [{"type": "function_call_output", "call_id": "call_1", "output": "done"}],
            "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}}],
            "reasoning": {"effort": "high", "encrypted_content": "opaque"},
            "include": ["reasoning.encrypted_content"],
            "stream": True,
        }
        response = self.client.post(
            "/codex/v1/responses",
            headers={**AUTH, "session-id": "session-1", "x-codex-turn-metadata": '{"turn_id":"t1"}'},
            json=payload,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.service.calls[0][0], "responses")
        self.assertEqual(self.service.calls[0][2], payload)

    def test_true_sse_bytes_are_returned_without_json_wrapping(self):
        chunks = [
            b'data: {"type":"response.output_text.delta","delta":"one"}\n\n',
            b'data: {"type":"response.completed","response":{"id":"r1"}}\n\n',
        ]
        self.service.next_result = CodexHTTPResponse(
            200, {"content-type": "text/event-stream"}, stream=iter(chunks)
        )
        with self.client.stream("POST", "/codex/v1/responses", headers=AUTH, json={"model": "gpt-5.6-codex"}) as response:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(b"".join(response.iter_bytes()), b"".join(chunks))
            self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))

    def test_compact_selects_compact_operation(self):
        response = self.client.post(
            "/codex/v1/responses/compact",
            headers=AUTH,
            json={"model": "gpt-5.6-codex", "input": [{"type": "compaction", "encrypted_content": "cipher"}]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.calls[0][0], "compact")

    def test_websocket_upgrade_is_explicitly_unsupported(self):
        response = self.client.post(
            "/codex/v1/responses",
            headers={**AUTH, "Upgrade": "websocket"},
            json={"model": "gpt-5.6-codex"},
        )
        self.assertEqual(response.status_code, 426)
        self.assertEqual(response.json()["detail"]["error"]["code"], "codex_websocket_unsupported")
        self.assertEqual(self.service.calls, [])

    def test_invalid_json_and_large_declared_body_are_rejected_generically(self):
        invalid = self.client.post(
            "/codex/v1/responses", headers={**AUTH, "content-type": "application/json"}, content=b'{"access_token":"secret"'
        )
        self.assertEqual(invalid.status_code, 400)
        self.assertNotIn("secret", invalid.text)
        large = self.client.post(
            "/codex/v1/responses",
            headers={**AUTH, "content-length": str(codex_api.MAX_REQUEST_BYTES + 1)},
            content=b"{}",
        )
        self.assertEqual(large.status_code, 413)
        self.assertEqual(self.service.calls, [])

    def test_service_error_is_safe_and_never_echoes_raw_upstream_data(self):
        self.service.error = CodexServiceError(502, "codex_upstream_outcome_unknown", "The Codex request outcome is unknown")
        response = self.client.post(
            "/codex/v1/responses", headers=AUTH, json={"model": "gpt-5.6-codex", "input": "private prompt"}
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"]["error"]["code"], "codex_upstream_outcome_unknown")
        self.assertNotIn("private prompt", response.text)


if __name__ == "__main__":
    unittest.main()
