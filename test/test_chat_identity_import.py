"""Chat OAuth subjects and Chat user IDs must not be compared as one namespace."""
import base64
import json
import uuid
from unittest.mock import patch

import pytest

from services.account_service import AccountService, CodexAuthorizationAttachError
from services.chat_login_service import ChatLoginService
from services.storage.json_storage import JSONStorageBackend
from test.test_chat_login_flow import FakeHttp, FakeResponse, OWNER, callback_for


def token(subject, chat_user, workspace=None, marker=""):
    claims = {"user_id": chat_user}
    if workspace:
        claims.update(chatgpt_account_id=workspace, chatgpt_user_id=chat_user)
    payload = {"sub": subject, "https://api.openai.com/auth": claims, "jti": marker}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def material(index, *, codex=False, marker="old"):
    subject, chat_user = f"auth0|person-{index}", f"user-chat-{index}"
    workspace = str(uuid.UUID(int=index + 1))
    return {
        "access_token": token(subject, chat_user, workspace if codex else None, marker),
        "id_token": token(subject, chat_user, workspace if codex else None, marker + "-id"),
        "refresh_token": f"synthetic-{index}-{marker}",
    }


def pool(tmp_path):
    accounts = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    rows = []
    for index in range(4):
        codex = material(index, codex=True)
        rows.append({
            **(codex if index < 2 else material(index)),
            "source_type": "codex" if index < 2 else "oauth_login",
            "account_id": str(uuid.UUID(int=index + 1)),
            "user_id": None if index < 2 else f"user-chat-{index}",
            "capacity_observed_at": None if index < 2 else "2026-09-20T00:00:00Z",
            "codex_credentials": codex,
            "managed_owner": "original-owner",
            "managed_account_id": f"original-row-{index}",
            "conversation_binding_ids": [f"original-binding-{index}"],
            "task_receipts": {"original": "succeeded"},
        })
    accounts.add_account_items(rows)
    return accounts


def observation(index):
    identity = (f"user-chat-{index}", str(uuid.UUID(int=index + 1)))
    return identity, {"user_id": identity[0], "account_id": identity[1], "quota": 19,
                      "limits_progress": [{"feature_name": "image_gen", "remaining": 19}]}


@pytest.mark.parametrize("index", range(4))
def test_official_callback_matches_each_existing_row_and_preserves_other_route(tmp_path, index):
    accounts = pool(tmp_path)
    before = accounts.storage.load_accounts()
    refs = [accounts.pool_account_ref(row) for row in before]
    incoming = material(index, marker="fresh")
    http = FakeHttp([FakeResponse(200, incoming)])
    service = ChatLoginService(tmp_path / "sessions.json", accounts, http.factory)
    started = service.start(OWNER, "owned", "import", str(uuid.uuid4()))
    with patch.object(accounts, "_verified_chat_info", return_value=observation(index)) as protected_read, \
            patch.object(accounts, "_request_access_token_refresh", side_effect=AssertionError("no second exchange")):
        result = service.submit_callback(OWNER, "owned", started["id"], callback_for(started))
    assert result["state"] == "succeeded"
    assert result["import_status"] == "updated"
    assert result["account_ref"] == refs[index]
    assert result["capacity"]["remaining"] == 19
    protected_read.assert_called_once_with(incoming["access_token"])
    assert len(http.calls) == 1
    after = accounts.storage.load_accounts()
    assert len(after) == 4
    by_ref = {accounts.pool_account_ref(row): row for row in after}
    for position, previous in enumerate(before):
        current = by_ref[refs[position]]
        if position != index:
            assert current == previous
        else:
            assert current["access_token"] == incoming["access_token"]
            assert current["user_id"] == f"user-chat-{index}"
            assert current["source_type"] == "oauth_login"
            for key in ("managed_owner", "managed_account_id", "codex_credentials", "task_receipts", "conversation_binding_ids"):
                assert current[key] == previous[key]
    restarted = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    receipt = restarted.chat_login_committed_receipt(incoming)
    assert receipt["authorization_ref"] == refs[index]
    assert service.get(OWNER, "owned", started["id"])["state"] == "succeeded"
    assert len(http.calls) == 1
    assert "pending_credentials" not in (tmp_path / "sessions.json").read_text()


def test_unseen_authenticated_account_is_created_once(tmp_path):
    accounts = pool(tmp_path)
    incoming = material(5, marker="fresh")
    with patch.object(accounts, "_verified_chat_info", return_value=observation(5)) as verify:
        created = accounts.import_owned_account(OWNER, {**incoming, "source_type": "oauth_login"}, verified_oauth=True)
    assert len(accounts.storage.load_accounts()) == 5
    verify.assert_called_once_with(incoming["access_token"])
    saved = next(row for row in accounts.storage.load_accounts() if row["access_token"] == incoming["access_token"])
    assert saved["managed_owner"] == OWNER
    assert saved["user_id"] == "user-chat-5"
    assert created["capacity"]["remaining"] == 19
    restored = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    assert restored.chat_login_committed_receipt(incoming)
    assert len(restored.storage.load_accounts()) == 5


@pytest.mark.parametrize("conflict", ["chat-user", "workspace", "id-subject"])
def test_real_identity_conflicts_do_not_overwrite_or_create(tmp_path, conflict):
    accounts = pool(tmp_path)
    incoming = material(2, marker="fresh")
    identity, info = observation(2)
    if conflict == "chat-user":
        identity = ("user-different", identity[1])
        info["user_id"] = identity[0]
    elif conflict == "workspace":
        incoming["account_id"] = str(uuid.UUID(int=99))
    else:
        incoming["id_token"] = token("auth0|different", "user-chat-2")
    before = accounts.storage.file_path.read_bytes()
    with patch.object(accounts, "_verified_chat_info", return_value=(identity, info)):
        with pytest.raises(CodexAuthorizationAttachError, match="account_conflict"):
            accounts.import_owned_account(OWNER, {**incoming, "source_type": "oauth_login"}, verified_oauth=True)
    assert accounts.storage.file_path.read_bytes() == before


def test_duplicate_principal_is_not_silently_overwritten(tmp_path):
    accounts = pool(tmp_path)
    original = accounts.storage.load_accounts()[2]
    accounts.add_account_items([{**original, **material(2, marker="duplicate"), "managed_account_id": "duplicate-row"}])
    before = accounts.storage.file_path.read_bytes()
    with patch.object(accounts, "_verified_chat_info", return_value=observation(2)):
        with pytest.raises(CodexAuthorizationAttachError, match="account_ambiguous"):
            accounts.import_owned_account(OWNER, {**material(2, marker="fresh"), "source_type": "oauth_login"}, verified_oauth=True)
    assert accounts.storage.file_path.read_bytes() == before


def test_untrusted_id_token_cannot_attach_an_opaque_bearer_to_another_codex_user(tmp_path):
    accounts = pool(tmp_path)
    before = accounts.storage.file_path.read_bytes()
    identity = ("user-attacker", str(uuid.UUID(int=1)))
    info = {"user_id": identity[0], "account_id": identity[1], "quota": 19}
    with patch.object(accounts, "_verified_chat_info", return_value=(identity, info)):
        with pytest.raises(CodexAuthorizationAttachError, match="account_conflict"):
            accounts.import_owned_account(OWNER, {
                "access_token": "opaque-attacker-bearer", "source_type": "web",
                "id_token": token("auth0|person-0", "user-attacker", identity[1]),
            })
    assert accounts.storage.file_path.read_bytes() == before
