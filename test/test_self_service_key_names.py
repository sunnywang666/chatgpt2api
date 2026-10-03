"""Owner-scoped key naming; no provider/model requests.

Run with the owning suite in an isolated PROVIDER_DATA_DIR after the capability
implementation batch. These tests do not imply Workbench or live acceptance.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture(params=["json", "sqlite"])
def service(tmp_path, monkeypatch, request):
    # Keep singleton initialization, when this file is run alone, off real data.
    monkeypatch.setenv("PROVIDER_DATA_DIR", str(tmp_path / "singletons"))
    from services.auth_service import AuthService
    from services.storage.json_storage import JSONStorageBackend
    from services.storage.database_storage import DatabaseStorageBackend

    storage = (JSONStorageBackend(tmp_path / "accounts.json")
               if request.param == "json"
               else DatabaseStorageBackend(f"sqlite:///{tmp_path / 'keys.db'}"))
    return AuthService(storage)


def test_two_users_can_name_their_own_keys_identically(service):
    first, first_secret = service.create_key(role="user", owner_subject="workbench:org:first", name="工作流")
    second, second_secret = service.create_key(role="user", owner_subject="workbench:org:second", name="工作流")
    assert first["name"] == second["name"] == "工作流"
    assert first["id"] != second["id"] and first_secret != second_secret
    assert [item["id"] for item in service.list_owned_keys("workbench:org:first")] == [first["id"]]
    assert [item["id"] for item in service.list_owned_keys("workbench:org:second")] == [second["id"]]
    assert service.authenticate(first_secret)["id"] == first["id"]
    assert service.authenticate(second_secret)["id"] == second["id"]


def test_same_user_name_conflict_is_atomic_and_trimmed(service):
    item, _ = service.create_key(role="user", owner_subject="workbench:org:first", name="工作流")
    before = service.storage.load_auth_keys()
    with pytest.raises(ValueError, match="名称已经在使用"):
        service.create_key(role="user", owner_subject="workbench:org:first", name="  工作流  ")
    assert service.storage.load_auth_keys() == before
    assert service.list_owned_keys("workbench:org:first")[0]["id"] == item["id"]


def test_default_name_suffixes_are_private_to_owner(service):
    first, _ = service.create_key(role="user", owner_subject="workbench:org:first")
    first_next, _ = service.create_key(role="user", owner_subject="workbench:org:first", name=" ")
    second, _ = service.create_key(role="user", owner_subject="workbench:org:second")
    assert first["name"] == second["name"] == "普通用户"
    assert first_next["name"] == "普通用户 2"


def test_legacy_unowned_and_admin_names_keep_their_own_scopes(service):
    legacy, _ = service.create_key(role="user", name="工作流")
    owned, _ = service.create_key(role="user", owner_subject="workbench:org:first", name="工作流")
    admin, _ = service.create_key(role="admin", name="工作流")
    assert len({legacy["id"], owned["id"], admin["id"]}) == 3
    with pytest.raises(ValueError):
        service.create_key(role="user", name="工作流")
    with pytest.raises(ValueError):
        service.create_key(role="admin", name="工作流")


def test_rename_uses_target_owner_not_other_owners(service):
    first, _ = service.create_key(role="user", owner_subject="workbench:org:first", name="工作流")
    second, _ = service.create_key(role="user", owner_subject="workbench:org:second", name="临时名称")
    renamed = service.update_key(second["id"], {"name": "工作流"}, role="user")
    assert renamed["name"] == "工作流"
    assert service.update_key(first["id"], {"name": "工作流"})["id"] == first["id"]
    third, _ = service.create_key(role="user", owner_subject="workbench:org:second", name="另一工作流")
    before = service.storage.load_auth_keys()
    with pytest.raises(ValueError):
        service.update_key(third["id"], {"name": "工作流"}, role="user")
    assert service.storage.load_auth_keys() == before


def test_same_name_does_not_weaken_secret_uniqueness_or_owner_checks(service):
    first, first_secret = service.create_key(role="user", owner_subject="workbench:org:first", name="工作流")
    second, second_secret = service.create_key(role="user", owner_subject="workbench:org:second", name="工作流")
    with pytest.raises(ValueError, match="专用密钥已经存在"):
        service.update_key(second["id"], {"key": first_secret}, role="user")
    assert not service.revoke_owned_key("workbench:org:second", first["id"])
    assert service.update_owned_policy("workbench:org:second", first["id"], ["codex"], 1) is None
    assert service.authenticate(first_secret) is not None
    assert service.authenticate(second_secret) is not None
    public = json.dumps(service.list_keys(), ensure_ascii=False)
    assert first_secret not in public and second_secret not in public
    assert "key_hash" not in public and "owner_subject" not in public


def test_restart_and_revocation_preserve_other_users_same_named_key(service):
    from services.auth_service import AuthService

    first, first_secret = service.create_key(role="user", owner_subject="workbench:org:first", name="工作流")
    second, second_secret = service.create_key(role="user", owner_subject="workbench:org:second", name="工作流")
    independent = AuthService(service.storage)
    assert independent.authenticate(first_secret) is not None
    before = service.storage.load_auth_keys()
    assert service.revoke_owned_key("workbench:org:first", first["id"])
    assert service.revoke_owned_key("workbench:org:first", first["id"])
    assert independent.authenticate(first_secret) is None
    assert independent.authenticate(second_secret)["id"] == second["id"]
    after = service.storage.load_auth_keys()
    for original, current in zip(before, after):
        for field in ("id", "name", "owner_subject", "key_hash", "created_at", "policy"):
            assert original[field] == current[field]
    assert len(independent.list_owned_keys("workbench:org:first")) == 1
    # A revoked record remains visible; this patch does not add implicit rename,
    # deletion, name recycling or authority to re-enable an owned key.
    with pytest.raises(ValueError):
        independent.create_key(role="user", owner_subject="workbench:org:first", name="工作流")
