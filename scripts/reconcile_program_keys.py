"""Explicit one-time policy assignment on existing key records; dry run by default.

Use during the owned Provider cutover with the old writer stopped. Input contains
key IDs and route names only, never raw keys or upstream account secrets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.auth_service import auth_service
from services.program_key_policy import PolicyError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        assignments = json.loads(args.assignments.read_text(encoding="utf-8"))
        items = auth_service.reconcile_legacy_policies(assignments, apply=args.apply)
    except (PolicyError, ValueError, OSError) as exc:
        print(json.dumps({"error": exc.code if isinstance(exc, PolicyError) else "KEY_RECONCILIATION_FAILED"}))
        return 1
    print(json.dumps({"applied": args.apply, "items": [
        {field: item[field] for field in ("id", "enabled", "policy")}
        for item in items
    ]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
