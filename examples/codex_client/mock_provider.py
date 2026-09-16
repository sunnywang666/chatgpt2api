#!/usr/bin/env python3
"""Disposable local Responses mock for Codex CLI transport and tool-cycle checks.

The default cycle requests three real local `exec_command` calls in the
disposable fixture: read its files, repair calculator.py, and run unittest.
It records only request shape and call IDs, never bearer values or tool output.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


TOOL_CYCLE = (
    ("call_read_fixture", "sed -n '1,120p' calculator.py test_calculator.py"),
    (
        "call_repair_fixture",
        "python -c \"from pathlib import Path; path = Path('calculator.py'); path.write_text(path.read_text().replace('return left + right', 'return left * right'))\"",
    ),
    ("call_test_fixture", "python -m unittest -v"),
)


def function_output_call_ids(payload: dict[str, Any]) -> list[str]:
    """Return only call IDs; deliberately never retain tool output text."""
    input_items = payload.get("input")
    if not isinstance(input_items, list):
        return []
    return [
        str(item["call_id"])
        for item in input_items
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and isinstance(item.get("call_id"), str)
    ]


def tool_summary(tools: object) -> list[dict[str, str]]:
    if not isinstance(tools, list):
        return []
    return [
        {"type": str(tool.get("type", "")), "name": str(tool.get("name", ""))}
        for tool in tools
        if isinstance(tool, dict)
    ]


def response_payload(response_id: str, model: object, status: str, output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 0,
        "status": status,
        "model": model,
        "output": output,
    }


def function_call_item(call_id: str, command: str) -> dict[str, Any]:
    return {
        "id": f"fc_{call_id}",
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": "exec_command",
        "arguments": json.dumps({"cmd": command}),
    }


class MockHandler(BaseHTTPRequestHandler):
    server_version = "CodexProviderMock/2"
    cycle_stage = 0
    candidate_service_mode = False
    observations: list[dict[str, Any]] = []

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _observation(self, body: dict[str, Any] | None = None) -> None:
        # Header values, prompt text, and function output can contain secrets. Keep
        # only names and protocol structure in the mock's operator-visible output.
        headers = sorted(str(name).lower() for name in self.headers)
        observation = {
            "method": self.command,
            "path": self.path,
            "header_names": headers,
            "json_keys": sorted((body or {}).keys()),
            "tools": tool_summary((body or {}).get("tools")),
            "function_call_output_ids": function_output_call_ids(body or {}),
        }
        type(self).observations.append(observation)
        print(json.dumps(observation), flush=True)

    def _write_events(self, events: Iterable[dict[str, Any]]) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()
        for event in events:
            event_type = str(event["type"])
            self.wfile.write(f"event: {event_type}\ndata: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        self._observation()
        if self.path == "/v1/usage" and type(self).candidate_service_mode:
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"rate_limit":{"allowed":true,"limit_reached":false,"primary_window":{"used_percent":1,"limit_window_seconds":18000}}}')
            return
        if self.path != "/v1/models":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        # The candidate service refreshes a Codex account using the native
        # catalog projection before it serves the CLI-compatible model list.
        if type(self).candidate_service_mode and self.headers.get("user-agent") == "codex-cli/0.149.1":
            self.wfile.write(b'{"models":[{"slug":"gpt-5.1-codex-mini","display_name":"Mock Codex","supported_reasoning_efforts":["medium"]}]}')
            return
        self.wfile.write(b'{"object":"list","data":[{"id":"gpt-5.1-codex-mini","object":"model"}]}')

    def _terminal_events(self, body: dict[str, Any], text: str) -> list[dict[str, Any]]:
        message = {
            "id": "msg_mock_complete",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        in_progress = response_payload("resp_mock_complete", body.get("model"), "in_progress", [])
        completed = response_payload("resp_mock_complete", body.get("model"), "completed", [message])
        return [
            {"type": "response.created", "response": in_progress},
            {"type": "response.in_progress", "response": in_progress},
            {"type": "response.output_item.added", "output_index": 0, "item": message},
            {"type": "response.output_text.delta", "item_id": message["id"], "output_index": 0, "content_index": 0, "delta": text},
            {"type": "response.output_item.done", "output_index": 0, "item": message},
            {"type": "response.completed", "response": completed},
        ]

    def _tool_events(self, body: dict[str, Any], call_id: str, command: str) -> list[dict[str, Any]]:
        item = function_call_item(call_id, command)
        in_progress = response_payload(f"resp_{call_id}", body.get("model"), "in_progress", [])
        completed = response_payload(f"resp_{call_id}", body.get("model"), "completed", [item])
        return [
            {"type": "response.created", "response": in_progress},
            {"type": "response.in_progress", "response": in_progress},
            {"type": "response.output_item.added", "output_index": 0, "item": item},
            {
                "type": "response.function_call_arguments.done",
                "item_id": item["id"],
                "output_index": 0,
                "call_id": call_id,
                "name": "exec_command",
                "arguments": item["arguments"],
            },
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": completed},
        ]

    def do_POST(self) -> None:  # noqa: N802
        body = self._body()
        self._observation(body)
        if self.path != "/v1/responses":
            self.send_error(404)
            return
        prior_outputs = function_output_call_ids(body)
        stage = type(self).cycle_stage
        if stage:
            expected = TOOL_CYCLE[stage - 1][0]
            if expected not in prior_outputs:
                self._write_events(self._terminal_events(body, f"mock expected function_call_output for {expected}"))
                return
        if stage < len(TOOL_CYCLE):
            call_id, command = TOOL_CYCLE[stage]
            type(self).cycle_stage += 1
            self._write_events(self._tool_events(body, call_id, command))
            return
        self._write_events(self._terminal_events(body, "mock completed read, repair, and test tool cycle"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18787)
    parser.add_argument("--candidate-service", action="store_true", help="serve native catalog and usage payloads for the candidate CodexService refresh path")
    args = parser.parse_args()
    MockHandler.candidate_service_mode = args.candidate_service
    MockHandler.cycle_stage = 0
    MockHandler.observations = []
    print(f"mock provider listening on http://127.0.0.1:{args.port}/v1", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), MockHandler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
