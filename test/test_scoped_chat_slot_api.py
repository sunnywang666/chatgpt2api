"""Admin control can arm one original Chat request without opening the whole pool."""
import json
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import owned_accounts
from api.errors import install_exception_handlers
from services.account_service import AccountService
from services.auth_service import AuthService
from services.config import ConfigStore
from services.storage.json_storage import JSONStorageBackend
from services.text_task_service import TextTaskService
from test.test_pool_admission import build, Clock


def test_internal_admin_arms_only_a_durable_bound_original_and_can_clear_it(tmp_path: Path):
    row = {"access_token": "fixture-token", "account_id": "upstream-account",
           "provider_account_identity": "account_fixture", "type": "Plus",
           "source_type": "web", "status": "正常", "quota": 100,
           "conversation_binding_ids": ["binding-fixture"]}
    (tmp_path / "accounts.json").write_text(json.dumps([row]))
    (tmp_path / "config.json").write_text(json.dumps({"auth-key": "fixture-admin-secret"}))
    accounts, store, admission = build(tmp_path, Clock())
    config = ConfigStore(tmp_path / "config.json")
    text = TextTaskService(store.path, admission=admission, clock=Clock())
    def submit(request_id, owner):
        return text.submit(owner, {
            "client_request_id": request_id,
            "client_conversation_id": "conversation-" + request_id,
            "model": "fixture-text", "messages": [{"role": "user", "content": "fixture input"}],
            "provider_account_identity": row["provider_account_identity"],
            "provider_binding_id": "binding-fixture",
        })
    submit("active", "happy")
    admission.claim_next().before_send()
    submit("chosen", "wb")
    auth = AuthService(JSONStorageBackend(tmp_path / "auth_keys.json"))
    _, admin = auth.create_key(role="admin", name="admin")
    _, ordinary = auth.create_key(role="user", name="program")
    from services.image_task_service import image_task_service
    with (patch("api.support.auth_service", auth),
          patch("api.owned_accounts.account_service", accounts),
          patch("services.config.config", config),
          patch.object(image_task_service, "admission", admission),
          patch("api.owned_accounts.time.time", return_value=1000)):
        app = FastAPI()
        install_exception_handlers(app)
        app.include_router(owned_accounts.create_router())
        client = TestClient(app)
        route = "/api/workbench/ai/pool/chat-second-slot"
        account_ref = AccountService.pool_account_ref(accounts.list_accounts()[0])
        body = {"expected_revision": 0, "action": "start", "account_ref": account_ref,
                "request_id": "chosen", "ttl_seconds": 120}
        owner = {"X-Workbench-Account-Owner": "workbench:org:boss"}
        assert client.post(route, headers={**owner, "Authorization": "Bearer " + ordinary}, json=body).status_code == 403
        assert client.post(route, headers={"Authorization": "Bearer " + admin}, json=body).status_code == 400
        headers = {**owner, "Authorization": "Bearer " + admin}
        assert client.post(route, headers=headers, json={**body, "request_id": "missing"}).status_code == 409
        response = client.post(route, headers=headers, json=body)
        assert response.status_code == 200, response.text
        assert response.json()["temporary_chat_second_slot"] == {
            "account_identity": "account_fixture", "request_id": "chosen", "expires_at": 1120}
        assert client.post(route, headers=headers, json=body).status_code == 409
        stopped = client.post(route, headers=headers, json={"expected_revision": 1, "action": "stop"})
        assert stopped.status_code == 200, stopped.text
        assert "temporary_chat_second_slot" not in stopped.json()
