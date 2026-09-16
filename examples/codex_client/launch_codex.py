#!/usr/bin/env python3
"""Replace this process with Codex after moving its key to a child-only env."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def codex_environment(state_root: Path) -> dict[str, str]:
    key = os.environ.get("CODEX_PROVIDER_API_KEY")
    if not key:
        raise RuntimeError("CODEX_PROVIDER_API_KEY is required by this launcher")
    child = dict(os.environ)
    # Do not inherit an ordinary Codex state or expose the source-key variable to
    # the final Codex process. Neither value is ever constructed as an argument.
    child.pop("CODEX_HOME", None)
    child.pop("OPENAI_API_KEY", None)
    child.pop("CODEX_PROVIDER_API_KEY", None)
    child["CODEX_HOME"] = os.fspath(state_root)
    child["OPENAI_API_KEY"] = key
    return child


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    try:
        environment = codex_environment(args.state_root)
    except RuntimeError as error:
        parser.error(str(error))

    os.chdir(args.fixture_root)
    if args.resume:
        command = ["codex", "exec", "resume", "--skip-git-repo-check", args.resume, args.prompt, "--json", "--model", args.model]
    else:
        command = [
            "codex",
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "--cd",
            os.fspath(args.fixture_root),
            "--model",
            args.model,
            args.prompt,
        ]
    os.execvpe(command[0], command, environment)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
