"""Owned HTTP operations must reuse the existing protected account authority.

Regression definitions for the C01/C02 implementation batch. All upstream reads
are controlled; these tests must run with the owning suite in an isolated data
root during unified validation, not against a production service.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


OWNER = "workbench:test:owner"
OTHER = "workbench:test:other"
TOKEN = "owned-bridge-synthetic-access"
ROW_ID = "owned-row"
WORKSPACE = "12345678-1234-5678-9234-567812345678"


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setenv("PROVIDER_DATA_DIR", str(tmp_path / "singletons"))
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from api import owned_accounts as bridge
    from services.account_service import AccountService
    from services.auth_service import AuthService
    from services.codex_service import codex_service
    from services.storage.json_storage import JSONStorageBackend

    storage = JSONStorageBackend(tmp_path / "accounts.json")
    accounts = AccountService(storage)
    accounts.add_account_items([{
        "access_token": TOKEN, "source_type": "web", "status": "正常",
        "type": "Plus", "quota": 7, "account_id": WORKSPACE,
        "user_id": "chat-user-synthetic", "managed_owner": OWNER,
        "managed_account_id": ROW_ID, "managed_disabled": False,
        "capacity_observed_at": datetime.now(timezone.utc).isoformat(),
        "limits_progress": [{"feature_name": "image_gen", "remaining": 7}],
        "conversation_binding_ids": ["cb_original_synthetic"],
        "task_receipts": {"original": "keep"},
    }])
    auth = AuthService(storage)
    monkeypatch.setattr(bridge, "account_service", accounts)
    monkeypatch.setattr(bridge, "auth_service", auth)

    def require_admin(value):
        if value != "Bearer internal-bridge-test":
            raise HTTPException(403, detail="internal bridge authentication required")

    monkeypatch.setattr(bridge, "require_admin", require_admin)
    monkeypatch.setattr(accounts, "refresh_access_token", lambda token, **_kwargs: token)
    # Neither legacy method is an acceptable fallback for these HTTP routes.
    monkeypatch.setattr(accounts, "refresh_owned_account", Mock(side_effect=AssertionError("legacy refresh")))
    monkeypatch.setattr(accounts, "set_owned_account_enabled", Mock(side_effect=AssertionError("legacy enable")))
    monkeypatch.setattr(accounts, "fetch_remote_info", Mock(side_effect=AssertionError("legacy metadata read")))
    codex_read = Mock(side_effect=AssertionError("no Codex authorization was saved"))
    monkeypatch.setattr(codex_service, "refresh_account", codex_read)
    read = Mock(return_value=(("chat-user-synthetic", WORKSPACE), {
        "status": "正常", "quota": 5,
        "limits_progress": [{"feature_name": "image_gen", "remaining": 5}],
    }))
    monkeypatch.setattr(accounts, "_verified_chat_info", read)
    app = FastAPI()
    app.include_router(bridge.create_router())
    headers = {"Authorization": "Bearer internal-bridge-test", "X-Workbench-Account-Owner": OWNER}
    with TestClient(app) as client:
        yield SimpleNamespace(
            accounts=accounts, auth=auth, storage=storage, client=client,
            headers=headers, read=read, codex_read=codex_read, bridge=bridge,
        )


def post(h, suffix, body=None, *, headers=None):
    kwargs = {} if body is None else {"json": body}
    return h.client.post("/api/workbench/ai" + suffix, headers=headers or h.headers, **kwargs)


def assert_owned_receipt(response):
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    item = response.json()["item"]
    assert item["id"] == ROW_ID
    assert "account_ref" not in item and "managed_owner" not in item
    assert TOKEN not in response.text and "access_token" not in response.text
    return item


def test_refresh_reuses_verified_routes_and_owned_envelope(harness):
    h = harness
    before = h.accounts.get_account(TOKEN)
    item = assert_owned_receipt(post(h, f"/accounts/{ROW_ID}/refresh"))
    h.read.assert_called_once_with(TOKEN)
    h.codex_read.assert_not_called()
    assert item["capacity"]["remaining"] == 5
    assert item["capacity"]["state"] == "observed"
    after = h.accounts.get_account(TOKEN)
    for field in ("managed_owner", "managed_account_id", "conversation_binding_ids", "task_receipts"):
        assert after[field] == before[field]


def test_repeat_enable_is_read_only_and_reenable_preserves_quota(harness):
    h = harness
    original = h.storage.file_path.read_bytes()
    assert_owned_receipt(post(h, f"/accounts/{ROW_ID}/enabled", {"enabled": True}))
    assert h.storage.file_path.read_bytes() == original
    assert not assert_owned_receipt(post(h, f"/accounts/{ROW_ID}/enabled", {"enabled": False}))["enabled"]
    disabled = h.storage.file_path.read_bytes()
    post(h, f"/accounts/{ROW_ID}/enabled", {"enabled": False})
    assert h.storage.file_path.read_bytes() == disabled
    item = assert_owned_receipt(post(h, f"/accounts/{ROW_ID}/enabled", {"enabled": True}))
    assert item["enabled"] and item["connection_status"] == "unavailable"
    assert item["capacity"]["remaining"] == 7
    assert h.accounts.get_account(TOKEN)["quota"] == 7
    assert h.accounts.get_account(TOKEN)["status"] == "禁用"
    h.read.assert_not_called()
    item = assert_owned_receipt(post(h, f"/accounts/{ROW_ID}/refresh"))
    assert item["connection_status"] == "connected"
    assert h.accounts.get_account(TOKEN)["status"] == "正常"


@pytest.mark.parametrize("operation,body", [("refresh", None), ("enabled", {"enabled": False})])
def test_wrong_owner_and_pool_reference_cannot_select_owned_row(harness, operation, body):
    h = harness
    original = h.storage.file_path.read_bytes()
    headers = {**h.headers, "X-Workbench-Account-Owner": OTHER}
    assert post(h, f"/accounts/{ROW_ID}/{operation}", body, headers=headers).status_code == 404
    ref = h.accounts.list_pool_accounts()[0]["account_ref"]
    assert post(h, f"/accounts/{ref}/{operation}", body).status_code == 404
    assert h.storage.file_path.read_bytes() == original
    h.read.assert_not_called()


def test_duplicate_owned_id_does_not_pick_first_row(harness):
    h = harness
    h.accounts.add_account_items([{
        "access_token": "second-synthetic-access", "source_type": "web",
        "managed_owner": OWNER, "managed_account_id": ROW_ID,
        "user_id": "other-chat-user", "quota": 3,
    }])
    original = h.storage.file_path.read_bytes()
    assert post(h, f"/accounts/{ROW_ID}/refresh").status_code == 409
    assert post(h, f"/accounts/{ROW_ID}/enabled", {"enabled": False}).status_code == 409
    assert h.storage.file_path.read_bytes() == original
    h.read.assert_not_called()


def test_failed_metadata_read_keeps_previous_value_and_specific_state(harness):
    h = harness
    h.read.side_effect = RuntimeError("synthetic private upstream error")
    item = assert_owned_receipt(post(h, f"/accounts/{ROW_ID}/refresh"))
    assert item["capacity"]["state"] == "read_failed"
    assert item["capacity"]["remaining"] == 7
    assert item["chat"]["state"] == "read_failed"
    assert "synthetic private upstream error" not in str(item)
    assert h.accounts.get_account(TOKEN)["task_receipts"] == {"original": "keep"}


def test_owner_is_rechecked_after_refresh_without_leaking_new_owner(harness, monkeypatch):
    h = harness
    def changed_owner(*_args):
        h.accounts.update_account(TOKEN, {"managed_owner": OTHER}, quiet=True)
        return {}
    monkeypatch.setattr(h.accounts, "refresh_pool_account", changed_owner)
    response = post(h, f"/accounts/{ROW_ID}/refresh")
    assert response.status_code == 404
    assert OTHER not in response.text and TOKEN not in response.text


def test_upstream_read_does_not_hold_global_account_lock(harness):
    h = harness
    original_read = h.read.return_value
    def observe(_token):
        assert not h.accounts.admission_transaction()._is_owned()
        return original_read
    h.read.side_effect = observe
    assert_owned_receipt(post(h, f"/accounts/{ROW_ID}/refresh"))


def test_keys_have_no_store_and_revoke_only_the_requested_owners_key(harness):
    h = harness
    created = post(h, "/keys", {"name": "workflow", "routes": ["chat"]})
    assert created.status_code == 200
    assert created.headers["cache-control"] == "private, no-store"
    payload = created.json()
    secret, key_id = payload["key"], payload["item"]["id"]
    listing = h.client.get("/api/workbench/ai/keys", headers=h.headers)
    assert listing.headers["cache-control"] == "private, no-store"
    assert secret not in listing.text and "key_hash" not in listing.text
    wrong = h.client.delete(f"/api/workbench/ai/keys/{key_id}", headers={**h.headers, "X-Workbench-Account-Owner": OTHER})
    assert wrong.status_code == 404 and h.auth.authenticate(secret)
    revoked = h.client.delete(f"/api/workbench/ai/keys/{key_id}", headers=h.headers)
    assert revoked.status_code == 200 and revoked.json() == {"revoked": True}
    assert revoked.headers["cache-control"] == "private, no-store"
    assert h.auth.authenticate(secret) is None
    # A model API key never substitutes for authenticated Workbench management.
    denied = post(h, "/keys", {"name": "not-created"}, headers={**h.headers, "Authorization": "Bearer " + secret})
    assert denied.status_code == 403
