import copy
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import ai, codex, image_tasks, owned_accounts
from services.auth_service import AuthService
from services.codex_service import CodexHTTPResponse
from services.storage.json_storage import JSONStorageBackend


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    auth = AuthService(JSONStorageBackend(tmp_path / "accounts.json"))
    monkeypatch.setattr("api.support.auth_service", auth)
    monkeypatch.setattr(owned_accounts, "auth_service", auth)
    upstream = Mock(return_value={"data": []})
    monkeypatch.setattr(ai.openai_v1_image_generations, "handle", upstream)
    monkeypatch.setattr(ai.openai_v1_image_edit, "handle", upstream)
    monkeypatch.setattr(ai.openai_v1_chat_complete, "handle", upstream)
    monkeypatch.setattr(ai.openai_v1_response, "handle", upstream)
    monkeypatch.setattr(ai, "check_request", lambda *_: None)
    monkeypatch.setattr(image_tasks, "check_request", lambda *_: None)
    task_submit = Mock(return_value={"id": "task", "data": []})
    monkeypatch.setattr(image_tasks.image_task_service, "submit_generation", task_submit)
    monkeypatch.setattr(image_tasks.image_task_service, "submit_edit", task_submit)
    native = Mock(return_value=CodexHTTPResponse(200, {"content-type": "application/json"}, body=b'{"ok":true}'))
    monkeypatch.setattr(codex.codex_service, "submit", native)
    app = FastAPI()
    for module in (ai, codex, image_tasks, owned_accounts):
        app.include_router(module.create_router())
    client = TestClient(app)
    return auth, client, upstream, task_submit, native


def key(auth, caps, owner="workbench:company:user"):
    item, secret = auth.create_key(role="user", capabilities=caps, owner_subject=owner)
    return item, {"Authorization": "Bearer " + secret}


@pytest.mark.parametrize("path", ["/v1/images/generations", "/api/image-tasks/generations"])
@pytest.mark.parametrize("model", ["gpt-image-2", "codex-gpt-image-2", "plus-codex-gpt-image-2", "team-codex-gpt-image-2", "pro-codex-gpt-image-2"])
def test_coding_key_cannot_generate_images_on_any_alias(runtime, path, model):
    auth, client, upstream, tasks, native = runtime
    _, headers = key(auth, ["codex_coding"])
    result = client.post(path, headers=headers, json={"prompt": "sample", "model": model, "client_task_id": "task", "route": "chat", "purpose": "coding"})
    assert result.status_code == 403
    upstream.assert_not_called(); tasks.assert_not_called(); native.assert_not_called()


@pytest.mark.parametrize("path", ["/v1/images/edits", "/api/image-tasks/edits"])
def test_multipart_denied_before_fetch_or_generation(runtime, monkeypatch, path):
    auth, client, upstream, tasks, _ = runtime
    _, headers = key(auth, ["codex_coding"])
    read = Mock(side_effect=AssertionError("should not fetch image"))
    monkeypatch.setattr(ai, "read_image_sources", read)
    monkeypatch.setattr(image_tasks, "read_image_sources", read)
    result = client.post(path, headers=headers, data={"prompt": "sample", "client_task_id": "task"}, files={"image": ("sample.png", b"png", "image/png")})
    assert result.status_code == 403
    read.assert_not_called(); upstream.assert_not_called(); tasks.assert_not_called()


@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions", {"model": "gpt-image-2", "messages": [{"role": "user", "content": "image"}]}),
    ("/v1/chat/completions", {"model": "codex-gpt-image-2", "modalities": ["image"]}),
    ("/v1/responses", {"model": "gpt-image-2", "input": "image", "tools": [{"type": "image_generation"}]}),
    ("/v1/responses", {"model": "auto", "input": "text"}),
    ("/v1/messages", {"model": "auto", "messages": []}),
    ("/v1/search", {"prompt": "sample"}),
    ("/v1/ppt/generations", {"prompt": "sample"}),
    ("/v1/psd/generations", {"prompt": "sample"}),
])
def test_compatibility_routes_do_not_bypass_policy(runtime, path, body):
    auth, client, upstream, tasks, native = runtime
    _, headers = key(auth, ["codex_coding"])
    result = client.post(path, headers=headers, json=body)
    assert result.status_code == 403, result.text
    upstream.assert_not_called(); tasks.assert_not_called(); native.assert_not_called()


@pytest.mark.parametrize("path", ["/codex/v1/responses", "/codex/v1/responses/compact"])
@pytest.mark.parametrize("tool_part", [
    {"tools": [{"type": "image_generation"}]},
    {"tools": [{"type": "namespace", "name": "tools", "tools": [{"type": "image_generation"}]}]},
    {"tool_choice": {"type": "image_generation"}},
    {"tool_choice": {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "image_generation"}]}},
    {"tools": [{"type": "unknown_native"}]},
])
def test_native_image_tool_and_forced_selection_cannot_bypass(runtime, path, tool_part):
    auth, client, _, _, native = runtime
    _, headers = key(auth, ["codex_coding"])
    result = client.post(path, headers=headers, json={"model": "gpt-5.6-codex", **tool_part})
    assert result.status_code == 403
    native.assert_not_called()


def test_native_payload_is_unchanged_and_function_names_do_not_imply_images(runtime):
    auth, client, _, _, native = runtime
    _, headers = key(auth, ["codex_coding"])
    payload = {"model": "gpt-5.6-codex", "tools": [{"type": "namespace", "name": "tools", "tools": [
        {"type": "function", "name": "image_generation", "parameters": {"type": "object"}},
        {"type": "custom", "name": "image_gen", "format": {"type": "text"}},
    ]}], "tool_choice": {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "image_generation", "namespace": "tools"}]},
               "input": [{"type": "function_call_output", "call_id": "original", "output": "done"}],
               "reasoning": {"effort": "high"}, "include": ["reasoning.encrypted_content"]}
    expected = copy.deepcopy(payload)
    result = client.post("/codex/v1/responses", headers=headers, json=payload)
    assert result.status_code == 200
    assert native.call_args.args[1] == expected


def test_management_policy_roundtrip_conflict_owner_and_revocation(runtime):
    auth, client, upstream, _, native = runtime
    _, admin = auth.create_key(role="admin")
    headers = {"Authorization": "Bearer " + admin, "X-Workbench-Account-Owner": "workbench:company:user"}
    created = client.post("/api/workbench/ai/keys", headers=headers, json={"name": "program", "capabilities": ["chat_image"]})
    assert created.status_code == 200
    item, raw = created.json()["item"], created.json()["key"]
    ordinary = {"Authorization": "Bearer " + raw}
    assert client.post("/v1/images/generations", headers=ordinary, json={"prompt": "sample"}).status_code == 200
    assert upstream.call_count == 1
    endpoint = f'/api/workbench/ai/keys/{item["id"]}/policy'
    assert client.patch(endpoint, headers={**headers, "X-Workbench-Account-Owner": "workbench:other"}, json={"capabilities": ["codex_coding"], "expected_revision": 1}).status_code == 404
    changed = client.patch(endpoint, headers=headers, json={"capabilities": ["codex_coding"], "expected_revision": 1})
    assert changed.status_code == 200
    assert changed.json()["item"]["policy"]["revision"] == 2
    assert raw not in changed.text and "key_hash" not in changed.text
    assert client.patch(endpoint, headers=headers, json={"capabilities": ["chat_image"], "expected_revision": 1}).status_code == 409
    assert client.post("/v1/images/generations", headers=ordinary, json={"prompt": "sample"}).status_code == 403
    assert client.post("/codex/v1/responses", headers=ordinary, json={"model": "gpt-5.6-codex"}).status_code == 200
    assert native.call_count == 1
    assert client.get("/api/workbench/ai/keys", headers=ordinary).status_code == 403
    assert client.delete(f'/api/workbench/ai/keys/{item["id"]}', headers=headers).status_code == 200
    assert client.post("/codex/v1/responses", headers=ordinary, json={"model": "gpt-5.6-codex"}).status_code == 401
    assert native.call_count == 1


@pytest.mark.parametrize("capabilities", [[], ["chat_text"], ["codex_image"], ["unknown"], ["chat_image", "chat_image"]])
def test_management_cannot_enable_unready_unknown_or_empty_policy(runtime, capabilities):
    auth, client, *_ = runtime
    _, admin = auth.create_key(role="admin")
    response = client.post("/api/workbench/ai/keys", headers={"Authorization": "Bearer " + admin, "X-Workbench-Account-Owner": "workbench:owner"}, json={"name": "test", "capabilities": capabilities})
    assert response.status_code == 422
    assert auth.list_owned_keys("workbench:owner") == []


def test_legacy_text_compatibility_is_exact_nontransferable_and_removed_on_edit(runtime):
    auth, client, upstream, _, _ = runtime
    item, headers = key(auth, ["chat_image"])
    with auth.storage.auth_keys_transaction() as records:
        records[0].pop("policy")
    assert client.post("/v1/images/generations", headers=headers, json={"prompt": "sample"}).status_code == 403
    auth.reconcile_legacy_policies([{"id": item["id"], "capabilities": ["chat_image"],
        "legacy_text_compatibility": [{"endpoint": "/v1/chat/completions", "models": ["old-text"]}]}], apply=True)
    body = {"model": "old-text", "messages": [{"role": "user", "content": "sample"}]}
    assert client.post("/v1/chat/completions", headers=headers, json=body).status_code == 200
    assert client.post("/v1/chat/completions", headers=headers, json={**body, "model": "new-text"}).status_code == 403
    assert client.post("/v1/responses", headers=headers, json={"model": "old-text", "input": "sample"}).status_code == 403
    _, other_headers = key(auth, ["chat_image"])
    assert client.post("/v1/chat/completions", headers=other_headers, json=body).status_code == 403
    auth.update_owned_policy("workbench:company:user", item["id"], ["chat_image"], 1)
    assert client.post("/v1/chat/completions", headers=headers, json=body).status_code == 403
    assert upstream.call_count == 1


def test_server_consumer_admin_path_unchanged(runtime):
    auth, client, upstream, _, _ = runtime
    _, raw = auth.create_key(role="admin")
    headers = {"Authorization": "Bearer " + raw}
    assert client.post("/v1/chat/completions", headers=headers, json={"model": "auto", "messages": []}).status_code == 200
    assert client.post("/v1/images/generations", headers=headers, json={"prompt": "sample"}).status_code == 200
    assert upstream.call_count == 2


@pytest.mark.parametrize("model", ["gpt-image-2", "codex-gpt-image-2"])
def test_native_image_model_without_tools_still_needs_image_permission(runtime, model):
    auth, client, _, _, native = runtime
    _, headers = key(auth, ["codex_coding"])
    response = client.post("/codex/v1/responses", headers=headers, json={"model": model})
    assert response.status_code == 403
    native.assert_not_called()


@pytest.mark.parametrize("caps", [["chat_image"], ["codex_coding"]])
def test_conversation_archive_is_an_upstream_write_not_read_recovery(runtime, monkeypatch, caps):
    auth, client, *_ = runtime
    _, headers = key(auth, caps)
    archive = Mock(return_value={"archived": True})
    monkeypatch.setattr(ai.conversation_binding_service, "archive", archive)
    body = {name: "example" for name in ("provider_binding_id", "provider_account_identity",
            "client_conversation_id", "conversation_id", "parent_message_id")}
    assert client.post("/api/conversation-bindings/archive", headers=headers, json=body).status_code == 403
    archive.assert_not_called()
    _, admin = auth.create_key(role="admin")
    assert client.post("/api/conversation-bindings/archive", headers={"Authorization": "Bearer " + admin}, json=body).status_code == 200
    archive.assert_called_once_with(body)
