#!/usr/bin/env python3
"""List an OpenAI-compatible provider's model IDs without exposing its key."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from prepare_acceptance import DEFAULT_BASE_URL, normalized_base_url


def fetch_models(base_url: str, api_key: str, timeout: float = 20.0) -> list[str]:
    endpoint = f"{normalized_base_url(base_url)}/models"
    request = Request(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310: caller supplied provider endpoint
            payload: Any = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"GET /models returned HTTP {error.code}") from error
    except URLError as error:
        raise RuntimeError("GET /models could not reach the provider") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise RuntimeError("GET /models returned an invalid OpenAI model-list payload")
    ids = sorted(
        str(item["id"])
        for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
    )
    if not ids:
        raise RuntimeError("GET /models returned no model IDs")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", help="require this exact model ID in the returned list")
    args = parser.parse_args()
    api_key = os.environ.get("CODEX_PROVIDER_API_KEY")
    if not api_key:
        parser.error("set CODEX_PROVIDER_API_KEY in the launcher environment")
    try:
        models = fetch_models(args.base_url, api_key)
    except (RuntimeError, ValueError) as error:
        print(f"model preflight failed: {error}", file=sys.stderr)
        return 2
    if args.model and args.model not in models:
        print(f"model preflight failed: requested model {args.model!r} was not returned by GET /models", file=sys.stderr)
        return 3
    print(json.dumps({"object": "list", "data": [{"id": model} for model in models]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
