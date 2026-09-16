#!/usr/bin/env python3
"""Prepare an isolated Codex CLI state directory and disposable code fixture.

This program never reads or writes the normal Codex home directory and never
receives an API key.  The launchers inject the key into the Codex child only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


DEFAULT_BASE_URL = "https://app.hugsweetglobal.com/ai/codex/v1"
GENERATED_MARKER = "# generated-by-chatgpt2api-codex-client"


def normalized_base_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("provider base URL must be an absolute http(s) URL")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError("provider base URL must have a hostname and no embedded credentials")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("provider base URL requires HTTPS except for localhost, 127.0.0.1 or ::1 mocks")
    if parsed.query or parsed.fragment:
        raise ValueError("provider base URL must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def provider_config(base_url: str, model: str) -> str:
    """Return config with routing data only; an API key is deliberately absent."""
    return "\n".join(
        [
            GENERATED_MARKER,
            f"model = {json.dumps(model)}",
            'model_provider = "hugsweet-codex"',
            'approval_policy = "never"',
            'sandbox_mode = "workspace-write"',
            "",
            "[model_providers.hugsweet-codex]",
            'name = "Hugsweet Codex Responses"',
            f"base_url = {json.dumps(base_url)}",
            'env_key = "OPENAI_API_KEY"',
            'wire_api = "responses"',
            "request_max_retries = 0",
            "stream_max_retries = 0",
            "",
        ]
    )


def ensure_generated_config(state_root: Path, base_url: str, model: str) -> Path:
    state_root.mkdir(parents=True, exist_ok=True)
    config_path = state_root / "config.toml"
    content = provider_config(base_url, model)
    if config_path.exists():
        existing = config_path.read_text(encoding="utf-8")
        if not existing.startswith(GENERATED_MARKER):
            raise RuntimeError(
                f"refusing to replace a non-generated config: {config_path}; choose another --state-root"
            )
        if existing == content:
            return config_path
    config_path.write_text(content, encoding="utf-8")
    return config_path


def ensure_fixture(fixture_root: Path) -> None:
    fixture_root.mkdir(parents=True, exist_ok=True)
    calculator = fixture_root / "calculator.py"
    test_file = fixture_root / "test_calculator.py"
    if not calculator.exists():
        calculator.write_text(
            "def multiply(left: int, right: int) -> int:\n"
            "    \"\"\"Return the product of two integers.\"\"\"\n"
            "    return left + right\n",
            encoding="utf-8",
        )
    if not test_file.exists():
        test_file.write_text(
            "import unittest\n\n"
            "from calculator import multiply\n\n\n"
            "class CalculatorTests(unittest.TestCase):\n"
            "    def test_multiply(self) -> None:\n"
            "        self.assertEqual(multiply(6, 7), 42)\n",
            encoding="utf-8",
        )


def prepare(state_root: Path, fixture_root: Path, base_url: str, model: str) -> dict[str, str]:
    base_url = normalized_base_url(base_url)
    model = model.strip()
    if not model:
        raise ValueError("model is required")
    config_path = ensure_generated_config(state_root, base_url, model)
    ensure_fixture(fixture_root)
    return {
        "state_root": str(state_root),
        "config": str(config_path),
        "fixture": str(fixture_root),
        "base_url": base_url,
        "model": model,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.state_root, args.fixture_root, args.base_url, args.model)))
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
