"""Company OAuth start/recovery contract in isolated storage.

All HTTP clients and target lookups below are controlled fixtures. These cases
are not real authorization, credential persistence, or production evidence.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock
import uuid

import pytest


OWNER = "workbench:company:manager"
OTHER = "workbench:company:other"
REF = "car_" + "a" * 43


@pytest.fixture(params=["chat", "codex"])
def flow(request, tmp_path, monkeypatch):
    monkeypatch.setenv("PROVIDER_DATA_DIR", str(tmp_path / "isolated-singletons"))
    from services.chat_login_service import ChatLoginService, ChatLoginError
    from services.codex_login_service import CodexLoginService, CodexLoginError

    client = SimpleNamespace(
        post=Mock(return_value=SimpleNamespace(
            status_code=200,
            json=lambda: {"device_auth_id": "test-device", "user_code": "TEST-CODE", "interval": 5},
        )),
        close=Mock(),
    )
    accounts = SimpleNamespace(codex_login_target=Mock(return_value={"revision": "test-revision"}))
    kwargs = dict(path=tmp_path / "sessions.json", accounts=accounts,
                  session_factory=lambda: client, clock=lambda: 1_790_000_000.0)
    if request.param == "chat":
        service, error_type = ChatLoginService(**kwargs), ChatLoginError
    else:
        service, error_type = CodexLoginService(**kwargs, auto_start_workers=False), CodexLoginError
    return SimpleNamespace(service=service, error_type=error_type, client=client,
                           accounts=accounts, kind=request.param)


def test_company_import_is_a_valid_start_and_does_not_select_personal_inventory(flow):
    request_id = str(uuid.uuid4())
    first = flow.service.start(OWNER, "pool", "import", request_id)
    second = flow.service.start(OWNER, "pool", "import", request_id)
    assert first["id"] == second["id"]
    assert flow.service._sessions[first["id"]]["scope"] == "pool"
    flow.accounts.codex_login_target.assert_not_called()
    assert flow.client.post.call_count == (1 if flow.kind == "codex" else 0)


def test_new_legacy_start_rejects_before_upstream_action(flow):
    before = dict(flow.service._sessions)
    with pytest.raises(flow.error_type) as caught:
        flow.service.start(OWNER, "owned", "import", str(uuid.uuid4()))
    assert caught.value.status_code == 409
    assert caught.value.code == "COMPANY_ACCOUNT_ENTRY_REQUIRED"
    assert flow.service._sessions == before
    flow.client.post.assert_not_called()
    flow.accounts.codex_login_target.assert_not_called()


def test_already_accepted_legacy_start_keeps_its_original_scope_id_and_hash(flow):
    request_id = str(uuid.uuid4())
    started = flow.service.start(OWNER, "pool", "import", request_id)
    record = flow.service._sessions[started["id"]]
    # Seed the historical record shape without changing any production data.
    record["scope"] = "owned"
    record["request_hash"] = flow.service._request_hash("owned", "import", "")
    before_hash = record["request_hash"]
    before_http = flow.client.post.call_count
    resumed = flow.service.start(OWNER, "owned", "import", request_id)
    assert resumed["id"] == started["id"]
    assert flow.service.get(OWNER, "owned", started["id"])["id"] == started["id"]
    assert record["request_hash"] == before_hash and record["scope"] == "owned"
    assert flow.client.post.call_count == before_http
    with pytest.raises(flow.error_type) as wrong_scope:
        flow.service.get(OWNER, "pool", started["id"])
    assert wrong_scope.value.status_code == 404
    with pytest.raises(flow.error_type) as wrong_actor:
        flow.service.get(OTHER, "owned", started["id"])
    assert wrong_actor.value.status_code == 404


def test_legacy_id_cannot_be_reused_as_a_new_pool_start(flow):
    request_id = str(uuid.uuid4())
    result = flow.service.start(OWNER, "pool", "import", request_id)
    record = flow.service._sessions[result["id"]]
    record["scope"] = "owned"
    record["request_hash"] = flow.service._request_hash("owned", "import", "")
    before_http = flow.client.post.call_count
    with pytest.raises(flow.error_type) as caught:
        flow.service.start(OWNER, "pool", "import", request_id)
    assert caught.value.status_code == 409
    assert "idempotency_conflict" in caught.value.code
    assert flow.client.post.call_count == before_http


def test_company_attachment_uses_company_target_validation(flow):
    flow.service.start(OWNER, "pool", "attach", str(uuid.uuid4()), REF)
    flow.accounts.codex_login_target.assert_called_once_with(OWNER, REF, pool=True)


@pytest.mark.parametrize("mode,reference", [("import", REF), ("attach", None)])
def test_invalid_target_shape_is_not_silently_downgraded(flow, mode, reference):
    with pytest.raises(flow.error_type) as caught:
        flow.service.start(OWNER, "pool", mode, str(uuid.uuid4()), reference)
    assert caught.value.status_code == 422
    flow.client.post.assert_not_called()
