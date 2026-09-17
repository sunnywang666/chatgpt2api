"""Exercise real persistence, independent instances and OS processes."""
import json
import multiprocessing
from pathlib import Path

import pytest

from services.auth_service import AuthService
from services.program_key_policy import PolicyError
from services.storage.json_storage import JSONStorageBackend
from services.storage.database_storage import DatabaseStorageBackend


def backend(kind, root):
    return (JSONStorageBackend(Path(root) / "accounts.json") if kind == "json"
            else DatabaseStorageBackend(f"sqlite:///{Path(root) / 'keys.db'}"))


def stale_reader(kind, root, secret, ready, resume, results):
    service = AuthService(backend(kind, root))
    assert service.authenticate(secret)
    ready.set()
    if not resume.wait(15):
        raise RuntimeError("test writer did not complete")
    results.put(service.authenticate(secret))
    service.create_key(role="user", name="independent-process")


@pytest.mark.parametrize("kind", ["json", "sqlite"])
@pytest.mark.parametrize("operation", ["revoke", "narrow"])
def test_process_cache_cannot_restore_revocation_or_policy(tmp_path, kind, operation):
    a = AuthService(backend(kind, tmp_path))
    item, secret = a.create_key(role="user", owner_subject="owner", capabilities=["chat_image", "codex_coding"])
    context = multiprocessing.get_context("spawn")
    ready, resume, results = context.Event(), context.Event(), context.Queue()
    child = context.Process(target=stale_reader, args=(kind, str(tmp_path), secret, ready, resume, results))
    child.start()
    try:
        assert ready.wait(15)
        if operation == "revoke":
            assert a.revoke_owned_key("owner", item["id"])
        else:
            a.update_owned_policy("owner", item["id"], ["codex_coding"], 1)
        resume.set()
        identity = results.get(timeout=15)
        child.join(15)
        assert child.exitcode == 0
        restarted = AuthService(backend(kind, tmp_path))
        if operation == "revoke":
            assert identity is None
            assert restarted.authenticate(secret) is None
        else:
            assert identity["policy"]["revision"] == 2
            assert restarted.authenticate(secret)["policy"]["capabilities"] == ["codex_coding"]
        assert len(restarted.list_keys()) == 2
    finally:
        if child.is_alive():
            child.terminate()
            child.join()


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_update_conflict_owner_scope_and_secrets_survive_restart(tmp_path, kind):
    first = AuthService(backend(kind, tmp_path))
    item, secret = first.create_key(role="user", owner_subject="owner")
    original = first.storage.load_auth_keys()[0]
    second = AuthService(backend(kind, tmp_path))
    assert second.update_owned_policy("other", item["id"], ["codex_coding"], 1) is None
    updated = second.update_owned_policy("owner", item["id"], ["codex_coding"], 1)
    assert updated["policy"]["revision"] == 2
    with pytest.raises(PolicyError, match="REVISION_CONFLICT"):
        first.update_owned_policy("owner", item["id"], ["chat_image"], 1)
    assert first.authenticate(secret)["policy"] == updated["policy"]
    final = first.storage.load_auth_keys()[0]
    for field in ("id", "owner_subject", "key_hash", "created_at"):
        assert final[field] == original[field]
    assert "key_hash" not in updated and "owner_subject" not in updated and secret not in json.dumps(updated)


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_no_partial_write_on_policy_error_and_legacy_not_implicitly_open(tmp_path, kind):
    service = AuthService(backend(kind, tmp_path))
    item, secret = service.create_key(role="user", owner_subject="owner")
    before = service.storage.load_auth_keys()
    with pytest.raises(PolicyError, match="NOT_READY"):
        service.update_owned_policy("owner", item["id"], ["codex_image"], 1)
    assert service.storage.load_auth_keys() == before
    with service.storage.auth_keys_transaction() as keys:
        keys[0].pop("policy")
    legacy = service.authenticate(secret)
    assert legacy["policy"] is None and legacy["policy_state"] == "reconciliation_required"
    assigned = service.update_owned_policy("owner", item["id"], ["chat_image"], 0)
    assert assigned["policy"]["revision"] == 1


def test_corrupt_json_is_not_overwritten_by_authentication_or_creation(tmp_path):
    storage = backend("json", tmp_path)
    service = AuthService(storage)
    storage.auth_keys_path.write_text("broken")
    with pytest.raises(ValueError):
        service.create_key(role="user")
    with pytest.raises(ValueError):
        service.authenticate("any")
    assert storage.auth_keys_path.read_text() == "broken"


@pytest.mark.parametrize("kind", ["json", "sqlite"])
def test_explicit_legacy_reconciliation_is_atomic_and_dry_run_is_read_only(tmp_path, kind):
    service = AuthService(backend(kind, tmp_path))
    first, secret = service.create_key(role="user", owner_subject="owner")
    second, _ = service.create_key(role="user")
    with service.storage.auth_keys_transaction() as records:
        for item in records:
            item.pop("policy")
    original = service.storage.load_auth_keys()
    assignments = [{"id": first["id"], "capabilities": ["chat_image", "codex_coding"],
                    "legacy_text_compatibility": [{"endpoint": "/v1/chat/completions", "models": ["old-model"]}]},
                   {"id": second["id"], "capabilities": ["chat_image"]}]
    service.reconcile_legacy_policies(assignments)
    assert service.storage.load_auth_keys() == original
    with pytest.raises(PolicyError, match="KEY_SET_CHANGED"):
        service.reconcile_legacy_policies(assignments[:1], apply=True)
    assert service.storage.load_auth_keys() == original
    result = service.reconcile_legacy_policies(assignments, apply=True)
    assert len(result) == 2
    for prior, after in zip(original, service.storage.load_auth_keys()):
        assert all(prior[field] == after[field] for field in ("id", "owner_subject", "key_hash", "enabled"))
    assert service.authenticate(secret)["legacy_text_compatibility"]
    service.update_owned_policy("owner", first["id"], ["chat_image"], 1)
    assert service.authenticate(secret)["legacy_text_compatibility"] == []
    with pytest.raises(PolicyError, match="KEY_SET_CHANGED"):
        service.reconcile_legacy_policies(assignments, apply=True)


def test_json_commit_syncs_file_before_rename_and_directory_after(tmp_path, monkeypatch):
    import os
    import stat
    from services.storage import json_storage
    events = []
    original_sync, original_replace = os.fsync, os.replace
    def sync(fd):
        events.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        original_sync(fd)
    def replace(source, target):
        events.append("rename")
        original_replace(source, target)
    monkeypatch.setattr(json_storage.os, "fsync", sync)
    monkeypatch.setattr(json_storage.os, "replace", replace)
    auth = AuthService(backend("json", tmp_path))
    auth.create_key(role="user")
    assert events == ["file", "rename", "directory"]
