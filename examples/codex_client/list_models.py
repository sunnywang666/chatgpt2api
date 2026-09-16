#!/usr/bin/env python3
"""List native Codex or OpenAI-compatible model IDs without exposing its key."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from prepare_acceptance import DEFAULT_BASE_URL, normalized_base_url


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # urllib otherwise forwards the bearer header to the Location origin.
        return None


def fetch_models(base_url: str, api_key: str, timeout: float = 20.0) -> list[str]:
    endpoint = f"{normalized_base_url(base_url)}/models"
    request = Request(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with build_opener(_RejectRedirects()).open(request, timeout=timeout) as response:
            payload: Any = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"GET /models returned HTTP {error.code}") from error
    except URLError as error:
        raise RuntimeError("GET /models could not reach the provider") from error
    if isinstance(payload, dict) and isinstance(payload.get("models"), list):
        items, field = payload["models"], "slug"
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        items, field = payload["data"], "id"
    else:
        raise RuntimeError("GET /models returned an invalid model-list payload")
    ids = sorted({
        item[field]
        for item in items
        if isinstance(item, dict) and isinstance(item.get(field), str) and item[field]
    })
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
