#!/usr/bin/env python3
"""Build an isolated real-acceptance runtime without printing credentials.

Run this only inside the approved host/container.  It consumes read-only
/input/config/config.json and /input/accounts/accounts.json, copies at most two
operator-selected account records, and writes a new temporary Provider root.
It never starts Provider or sends HTTP requests.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import subprocess
import sys
import time
from typing import Any

def _die(message: str) -> None:
    raise SystemExit(message)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _die(f"bootstrap_error={path.name}_unreadable:{type(exc).__name__}")
    if not isinstance(value, dict):
        _die(f"bootstrap_error={path.name}_must_be_object")
    return value


def _read_list(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _die(f"bootstrap_error={path.name}_unreadable:{type(exc).__name__}")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        _die(f"bootstrap_error={path.name}_must_be_list_of_objects")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _provider_account_identity(record: dict[str, Any]) -> str:
    """Match Provider's stable identity; never select by a token or database id."""
    explicit = str(record.get("provider_account_identity") or "").strip()
    if explicit:
        return explicit
    # AccountService derives this exact public identity when no explicit value
    # has been persisted: ``account_`` plus the opaque pool-account ref body.
    # Bootstrap cannot safely derive it from a raw access token, so legacy
    # records must be selected only after the operator adds the issued identity.
    return ""


# A copied record must be able to authenticate and retain observed model/quota
# capability, but must not carry prior conversation ownership or Codex import
# state into the isolated runtime.  Unknown fields are intentionally excluded.
_ACCOUNT_COPY_FIELDS = frozenset({
    # No refresh/password/Codex material: this acceptance process can use only
    # a still-valid access token and therefore cannot rotate a live credential.
    "access_token",
    "account_id", "provider_account_identity", "managed_pool_account_ref",
    "user_id", "source_type", "type", "status", "quota", "limits_progress",
    "models", "capabilities", "image_capabilities", "image_models",
    "model_catalog", "paid_model_catalog", "last_verified_at", "expires_at",
    "capacity_observed_at", "capacity_read_failed_at", "capacity_used_since_observation",
})


def _isolated_account_record(record: dict[str, Any]) -> dict[str, Any]:
    result = {key: record[key] for key in _ACCOUNT_COPY_FIELDS if key in record}
    # Explicitly make stale per-conversation/Codex state impossible to carry
    # even if a future whitelist expansion names one of these fields.
    for forbidden in ("conversation_binding_ids", "codex_affinities", "codex_import_submissions",
                      "managed_owner", "managed_account_id", "password", "email", "refresh_token",
                      "id_token", "session_token", "cookies", "codex_credentials"):
        result.pop(forbidden, None)
    return result


def _access_token_remaining_seconds(record: dict[str, Any]) -> int:
    """Require a parseable JWT exp without logging the credential or claims."""
    token = str(record.get("access_token") or "")
    parts = token.split(".")
    if len(parts) != 3:
        _die("bootstrap_error=selected_access_token_missing_readable_jwt_exp")
    try:
        segment = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(segment.encode("ascii")).decode("utf-8"))
        exp = float(payload["exp"])
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        _die("bootstrap_error=selected_access_token_missing_readable_jwt_exp")
    remaining = int(exp - time.time())
    if remaining < 30 * 60 + 60 * 60:
        _die("bootstrap_error=selected_access_token_remaining_below_5400_seconds")
    return remaining


def _copy_source(source: Path, destination: Path) -> None:
    if destination.exists():
        _die("bootstrap_error=destination_already_exists")
    if not (source / "main.py").is_file() or not (source / "services").is_dir():
        _die("bootstrap_error=candidate_source_layout_invalid")
    # A public exact-SHA source must not pull an arbitrary host path into the
    # test container through a repository symlink.
    skipped = {".git", "data", "__pycache__", ".pytest_cache"}
    for current, directories, names in os.walk(source, followlinks=False):
        directories[:] = [name for name in directories if name not in skipped]
        if any((Path(current) / name).is_symlink() for name in [*directories, *names]):
            _die("bootstrap_error=candidate_source_symlink_forbidden")
    shutil.copytree(source, destination, symlinks=False, ignore=shutil.ignore_patterns(
        ".git", "data", "config.json", "__pycache__", ".pytest_cache"))


def _candidate_sha(source: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        _die(f"bootstrap_error=candidate_sha_unavailable:{type(exc).__name__}")
    return completed.stdout.strip()


def _safe_proxy_runtime(raw: object) -> dict[str, Any]:
    """Keep only Provider transport configuration; never print its contents."""
    source = raw if isinstance(raw, dict) else {}
    clearance = source.get("clearance") if isinstance(source.get("clearance"), dict) else {}
    return {
        "enabled": bool(source.get("enabled", False)),
        "egress_mode": str(source.get("egress_mode") or "direct"),
        "proxy_url": str(source.get("proxy_url") or ""),
        "resource_proxy_url": str(source.get("resource_proxy_url") or ""),
        "skip_ssl_verify": bool(source.get("skip_ssl_verify", False)),
        "reset_session_status_codes": source.get("reset_session_status_codes", [403]),
        "clearance": {
            "enabled": bool(clearance.get("enabled", False)),
            "mode": str(clearance.get("mode") or "none"),
            "cf_cookies": str(clearance.get("cf_cookies") or ""),
            "cf_clearance": str(clearance.get("cf_clearance") or ""),
            "user_agent": str(clearance.get("user_agent") or ""),
            "browser": str(clearance.get("browser") or "chrome"),
            "flaresolverr_url": str(clearance.get("flaresolverr_url") or ""),
            "timeout_sec": clearance.get("timeout_sec", 60),
            "refresh_interval": clearance.get("refresh_interval", 3600),
            "warm_up_on_start": False,
        },
    }


def _temporary_config(source: dict[str, Any], bootstrap_key: str) -> dict[str, Any]:
    """Explicit allowlist. All writing/cleanup/relogin integrations stay off."""
    return {
        "auth-key": bootstrap_key,
        "proxy_runtime": _safe_proxy_runtime(source.get("proxy_runtime")),
        "default_upstream_model_name": str(source.get("default_upstream_model_name") or "gpt-5-5"),
        "default_thinking_effort": str(source.get("default_thinking_effort") or "auto"),
        "image_poll_timeout_secs": source.get("image_poll_timeout_secs", 120),
        "image_poll_interval_secs": source.get("image_poll_interval_secs", 10),
        "image_poll_initial_wait_secs": source.get("image_poll_initial_wait_secs", 10),
        "image_settle_enabled": bool(source.get("image_settle_enabled", True)),
        "image_check_before_hit_enabled": bool(source.get("image_check_before_hit_enabled", True)),
        "image_settle_secs": source.get("image_settle_secs", 2),
        "image_account_concurrency": 1,
        "chat_account_concurrency": 1,
        "codex_max_concurrency": 1,
        "refresh_account_interval_minute": 1440,
        "ai_review": {"enabled": False},
        "backup": {"enabled": False},
        "image_storage": {"enabled": False, "mode": "local"},
        "image_remove_conversation_after_result": False,
        "image_remove_conversation_always": False,
        "auto_remove_invalid_accounts": False,
        "auto_remove_rate_limited_accounts": False,
        "auto_relogin_after_refresh": False,
        "sensitive_words": [],
        "cors_origins": [],
    }


def _create_ordinary_user_key(source_root: Path, data_root: Path, key_file: Path) -> None:
    """Use the candidate's own AuthService and JSON backend schema.

    The child writes the raw local test key directly to its 0600 file; it does
    not return the key in stdout, a command argument, or an exception.
    """
    code = r'''
from pathlib import Path
import os
from services.auth_service import AuthService
from services.storage.json_storage import JSONStorageBackend

data = Path(os.environ["ACCEPTANCE_DATA_ROOT"])
key_file = Path(os.environ["ACCEPTANCE_KEY_FILE"])
auth = AuthService(JSONStorageBackend(data / "accounts.json", data / "auth_keys.json"))
_item, raw = auth.create_key(
    role="user", name="isolated acceptance user",
    owner_subject="isolated-real-acceptance", routes=["chat"],
)
if auth.authenticate(raw) is None:
    raise RuntimeError("ordinary key did not authenticate")
fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    handle.write(raw + "\n")
    handle.flush()
    os.fsync(handle.fileno())
'''
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "STORAGE_BACKEND": "json",
        "PROVIDER_DATA_DIR": str(data_root),
        "ACCEPTANCE_DATA_ROOT": str(data_root),
        "ACCEPTANCE_KEY_FILE": str(key_file),
    }
    try:
        subprocess.run([sys.executable, "-c", code], cwd=source_root, env=env,
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        _die(f"bootstrap_error=ordinary_key_create_or_authenticate_failed:{type(exc).__name__}")
    os.chmod(key_file, 0o600)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True,
                        help="public exact candidate Git SHA to require")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--input-config", type=Path, default=Path("/input/config/config.json"))
    parser.add_argument("--input-accounts", type=Path, default=Path("/input/accounts/accounts.json"))
    parser.add_argument("--protected-root", action="append", type=Path, required=True,
                        help="production source/data root; runtime root must not be beneath it")
    parser.add_argument("--provider-account-identity", action="append", default=[],
                        help="issued account_… Provider identity; repeat once or twice")
    parser.add_argument("--attest-selected-idle", action="store_true",
                        help="operator attests selected records have no current in-flight execution or pause")
    args = parser.parse_args()

    if not args.attest_selected_idle:
        _die("bootstrap_error=selected_idle_attestation_required")
    expected_sha = str(args.expected_sha or "").strip().lower()
    if len(expected_sha) != 40 or any(char not in "0123456789abcdef" for char in expected_sha):
        _die("bootstrap_error=expected_sha_must_be_40_lowercase_hex")
    selectors = [value.strip() for value in args.provider_account_identity if value.strip()]
    if not 1 <= len(selectors) <= 2 or len(set(selectors)) != len(selectors):
        _die("bootstrap_error=select_one_or_two_distinct_accounts")
    candidate_source = args.candidate_source.resolve()
    protected = [root.resolve() for root in args.protected_root]
    if any(_inside(args.runtime_root.resolve(), root) for root in protected):
        _die("bootstrap_error=runtime_root_is_under_protected_root")
    if any(_inside(candidate_source, root) for root in protected):
        _die("bootstrap_error=candidate_source_is_under_protected_root")

    source_sha = _candidate_sha(candidate_source)
    if source_sha != expected_sha:
        _die("bootstrap_error=candidate_sha_mismatch")

    source_config = _read_object(args.input_config)
    source_accounts = _read_list(args.input_accounts)
    selected = [item for item in source_accounts if _provider_account_identity(item) in selectors]
    if len(selected) != len(selectors):
        _die("bootstrap_error=selected_account_not_found")
    remaining_seconds: list[int] = []
    for record in selected:
        status = str(record.get("status") or "").strip().lower()
        if record.get("managed_disabled") or status in {"paused", "disabled", "禁用", "异常", "限流"}:
            _die("bootstrap_error=selected_account_not_eligible")
        remaining_seconds.append(_access_token_remaining_seconds(record))
    selected = [_isolated_account_record(item) for item in selected]

    runtime_root = args.runtime_root.resolve()
    if runtime_root.exists():
        _die("bootstrap_error=runtime_root_must_not_exist")
    source_root = runtime_root / "provider"
    data_root = runtime_root / "data"
    _copy_source(candidate_source, source_root)
    os.chmod(runtime_root, 0o700)
    data_root.mkdir(mode=0o700, parents=True)
    os.chmod(data_root, 0o700)

    bootstrap_key = "bootstrap-" + secrets.token_urlsafe(32)
    _write_json(source_root / "config.json", _temporary_config(source_config, bootstrap_key))
    _write_json(data_root / "accounts.json", selected)
    (runtime_root / "candidate.sha").write_text(source_sha + "\n", encoding="ascii")
    os.chmod(runtime_root / "candidate.sha", 0o600)
    _write_json(data_root / "auth_keys.json", {"items": []})
    key_file = runtime_root / "ordinary-test.key"
    _create_ordinary_user_key(source_root, data_root, key_file)
    (runtime_root / "results").mkdir(mode=0o700)

    # stdout is intentionally secret-free and contains only safe counts/paths.
    print(json.dumps({
        "status": "prepared", "candidate_sha": source_sha,
        "selected_account_count": len(selected), "runtime_root": str(runtime_root),
        "provider_data_dir": str(data_root), "test_key_file": str(key_file),
        "access_token_remaining_seconds": remaining_seconds,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
