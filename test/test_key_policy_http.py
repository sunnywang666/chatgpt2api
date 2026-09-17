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
    item, secret = auth.create_key(role="user", routes=caps, owner_subject=owner)
    return item, {"Authorization": "Bearer " + secret}


@pytest.mark.parametrize("path", ["/v1/images/generations", "/api/image-tasks/generations"])
@pytest.mark.parametrize("model", ["gpt-image-2", "codex-gpt-image-2", "plus-codex-gpt-image-2", "team-codex-gpt-image-2", "pro-codex-gpt-image-2"])
@pytest.mark.parametrize("routes", [["chat"], ["codex"], ["chat", "codex"]])
def test_image_alias_is_checked_against_actual_endpoint(runtime, path, model, routes):
    auth, client, upstream, tasks, native = runtime
    _, headers = key(auth, routes)
    result = client.post(path, headers=headers, json={"prompt": "sample", "model": model, "client_task_id": "task", "route": "chat", "purpose": "coding"})
    actual_route = "codex" if "codex-" in model else "chat"
    assert result.status_code == (200 if actual_route in routes else 403), result.text
    assert upstream.call_count + tasks.call_count == int(actual_route in routes)
    native.assert_not_called()


@pytest.mark.parametrize("path", ["/v1/images/edits", "/api/image-tasks/edits"])
def test_multipart_denied_before_fetch_or_generation(runtime, monkeypatch, path):
    auth, client, upstream, tasks, _ = runtime
    _, headers = key(auth, ["codex"])
    read = Mock(side_effect=AssertionError("should not fetch image"))
    monkeypatch.setattr(ai, "read_image_sources", read)
    monkeypatch.setattr(image_tasks, "read_image_sources", read)
    result = client.post(path, headers=headers, data={"prompt": "sample", "client_task_id": "task"}, files={"image": ("sample.png", b"png", "image/png")})
    assert result.status_code == 403
    read.assert_not_called(); upstream.assert_not_called(); tasks.assert_not_called()


@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions", {"model": "gpt-image-2", "messages": [{"role": "user", "content": "image"}]}),
    ("/v1/responses", {"model": "gpt-image-2", "input": "image", "tools": [{"type": "image_generation"}]}),
    ("/v1/responses", {"model": "auto", "input": "text"}),
    ("/v1/messages", {"model": "auto", "messages": []}),
    ("/v1/search", {"prompt": "sample"}),
    ("/v1/ppt/generations", {"prompt": "sample"}),
    ("/v1/psd/generations", {"prompt": "sample"}),
])
def test_compatibility_routes_do_not_bypass_policy(runtime, path, body):
    auth, client, upstream, tasks, native = runtime
    _, headers = key(auth, ["codex"])
    result = client.post(path, headers=headers, json=body)
    assert result.status_code == 403, result.text
    upstream.assert_not_called(); tasks.assert_not_called(); native.assert_not_called()


@pytest.mark.parametrize("path", ["/codex/v1/responses", "/codex/v1/responses/compact"])
@pytest.mark.parametrize("tool_part", [
    {"tools": [{"type": "image_generation"}]},
    {"tools": [{"type": "namespace", "name": "tools", "tools": [{"type": "image_generation"}]}]},
    {"tool_choice": {"type": "image_generation"}},
    {"tool_choice": {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "image_generation"}]}},
])
@pytest.mark.parametrize("routes", [["chat"], ["codex"], ["chat", "codex"]])
def test_native_tools_require_codex_endpoint_not_image_subpermission(runtime, path, tool_part, routes):
    auth, client, _, _, native = runtime
    _, headers = key(auth, routes)
    result = client.post(path, headers=headers, json={"model": "gpt-5.6-codex", **tool_part})
    assert result.status_code == (200 if "codex" in routes else 403)
    assert native.call_count == int("codex" in routes)


def test_native_payload_is_unchanged_and_function_names_do_not_imply_images(runtime):
    auth, client, _, _, native = runtime
    _, headers = key(auth, ["codex"])
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
    created = client.post("/api/workbench/ai/keys", headers=headers, json={"name": "program", "routes": ["chat"]})
    assert created.status_code == 200
    item, raw = created.json()["item"], created.json()["key"]
    ordinary = {"Authorization": "Bearer " + raw}
    assert client.post("/v1/images/generations", headers=ordinary, json={"prompt": "sample"}).status_code == 200
    assert upstream.call_count == 1
    endpoint = f'/api/workbench/ai/keys/{item["id"]}/policy'
    assert client.patch(endpoint, headers={**headers, "X-Workbench-Account-Owner": "workbench:other"}, json={"routes": ["codex"], "expected_revision": 1}).status_code == 404
    changed = client.patch(endpoint, headers=headers, json={"routes": ["codex"], "expected_revision": 1})
    assert changed.status_code == 200
    assert changed.json()["item"]["policy"]["revision"] == 2
    assert raw not in changed.text and "key_hash" not in changed.text
    assert client.patch(endpoint, headers=headers, json={"routes": ["chat"], "expected_revision": 1}).status_code == 409
    assert client.post("/v1/images/generations", headers=ordinary, json={"prompt": "sample"}).status_code == 403
    assert client.post("/codex/v1/responses", headers=ordinary, json={"model": "gpt-5.6-codex"}).status_code == 200
    assert native.call_count == 1
    assert client.get("/api/workbench/ai/keys", headers=ordinary).status_code == 403
    assert client.delete(f'/api/workbench/ai/keys/{item["id"]}', headers=headers).status_code == 200
    assert client.post("/codex/v1/responses", headers=ordinary, json={"model": "gpt-5.6-codex"}).status_code == 401
    assert native.call_count == 1


@pytest.mark.parametrize("routes", [[], ["chat_text"], ["codex_image"], ["unknown"], ["chat", "chat"]])
def test_management_cannot_enable_unready_unknown_or_empty_policy(runtime, routes):
    auth, client, *_ = runtime
    _, admin = auth.create_key(role="admin")
    response = client.post("/api/workbench/ai/keys", headers={"Authorization": "Bearer " + admin, "X-Workbench-Account-Owner": "workbench:owner"}, json={"name": "test", "routes": routes})
    assert response.status_code == 422
    assert auth.list_owned_keys("workbench:owner") == []


@pytest.mark.parametrize("routes", [["chat"], ["codex"], ["chat", "codex"]])
@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions", {"model": "auto", "messages": [{"role": "user", "content": "plan"}]}),
    ("/v1/responses", {"model": "auto", "input": "describe", "tools": []}),
])
def test_chat_text_has_no_hidden_purpose_permission(runtime, routes, path, body):
    auth, client, upstream, _, _ = runtime
    _, headers = key(auth, routes)
    result = client.post(path, headers=headers, json=body)
    assert result.status_code == (200 if "chat" in routes else 403), result.text
    assert upstream.call_count == int("chat" in routes)


def test_unknown_native_tool_is_technical_unavailable_not_permission(runtime):
    auth, client, _, _, native = runtime
    _, headers = key(auth, ["codex"])
    response = client.post("/codex/v1/responses", headers=headers,
        json={"model": "gpt-5.6-codex", "tools": [{"type": "future_native"}]})
    assert response.status_code == 501
    assert response.json()["detail"]["code"] == "NATIVE_TOOL_NOT_CLASSIFIED"
    native.assert_not_called()


def test_old_capability_management_input_is_not_silently_broadened(runtime):
    auth, client, *_ = runtime
    _, admin = auth.create_key(role="admin")
    result = client.post("/api/workbench/ai/keys", headers={"Authorization": "Bearer " + admin,
        "X-Workbench-Account-Owner": "workbench:company:user"},
        json={"name": "old", "capabilities": ["chat_image"]})
    assert result.status_code == 422
    assert auth.list_owned_keys("workbench:company:user") == []


def test_server_consumer_admin_path_unchanged(runtime):
    auth, client, upstream, _, _ = runtime
    _, raw = auth.create_key(role="admin")
    headers = {"Authorization": "Bearer " + raw}
    assert client.post("/v1/chat/completions", headers=headers, json={"model": "auto", "messages": []}).status_code == 200
    assert client.post("/v1/images/generations", headers=headers, json={"prompt": "sample"}).status_code == 200
    assert upstream.call_count == 2


@pytest.mark.parametrize("model", ["gpt-image-2", "codex-gpt-image-2"])
def test_native_image_model_without_tools_uses_codex_endpoint(runtime, model):
    auth, client, _, _, native = runtime
    _, headers = key(auth, ["codex"])
    response = client.post("/codex/v1/responses", headers=headers, json={"model": model})
    assert response.status_code == 200
    native.assert_called_once()


@pytest.mark.parametrize("caps", [["chat"], ["codex"]])
def test_conversation_archive_is_an_upstream_write_not_read_recovery(runtime, monkeypatch, caps):
    auth, client, *_ = runtime
    _, headers = key(auth, caps)
    archive = Mock(return_value={"archived": True})
    monkeypatch.setattr(ai.conversation_binding_service, "archive", archive)
    body = {name: "example" for name in ("provider_binding_id", "provider_account_identity",
            "client_conversation_id", "conversation_id", "parent_message_id")}
    assert client.post("/api/conversation-bindings/archive", headers=headers, json=body).status_code == 404
    archive.assert_not_called()
    _, admin = auth.create_key(role="admin")
    assert client.post("/api/conversation-bindings/archive", headers={"Authorization": "Bearer " + admin}, json=body).status_code == 200
    archive.assert_called_once_with(body)


@pytest.mark.parametrize("caps", [["chat"], ["codex"]])
def test_ordinary_key_cannot_read_unowned_legacy_text_cursor(runtime, monkeypatch, caps):
    auth, client, *_ = runtime
    item, headers = key(auth, caps)
    read = Mock(return_value={"content": "another caller's private text"})
    monkeypatch.setattr(ai.conversation_binding_service, "read_text", read)
    cursor = {name: "foreign-cursor" for name in ("provider_binding_id", "provider_account_identity",
              "client_conversation_id", "conversation_id", "parent_message_id")}
    response = client.get("/api/conversation-bindings/text", headers=headers, params=cursor)
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "TASK_NOT_FOUND"
    read.assert_not_called()
    auth.revoke_owned_key("workbench:company:user", item["id"])
    assert client.get("/api/conversation-bindings/text", headers=headers, params=cursor).status_code == 401
    read.assert_not_called()


def test_persisted_text_receipt_owner_survives_narrowing_but_not_revocation(runtime, monkeypatch, tmp_path):
    from services.text_task_service import TextTaskService
    auth, client, *_ = runtime
    item, headers = key(auth, ["chat", "codex"])
    _, foreign = key(auth, ["codex"])
    # Seed an already accepted historical receipt through the actual service;
    # no new Chat-text capability is opened and no upstream is called.
    class ImmediateExecutor:
        def submit(self, function, *args):
            function(*args)
    completed = Mock(return_value={"content": "private historical result", "conversation_id": "chat", "parent_message_id": "answer"})
    path = tmp_path / "text-receipts.sqlite3"
    service = TextTaskService(path, completed, ImmediateExecutor())
    body = {"client_request_id": "saved-text", "client_conversation_id": "original",
            "messages": [{"role": "user", "content": "historical"}]}
    service.submit(item["id"], body)
    completed.assert_called_once()
    auth.update_owned_policy("workbench:company:user", item["id"], ["codex"], 1)
    upstream_read = Mock(side_effect=AssertionError("must not read another caller's Provider conversation"))
    monkeypatch.setattr(ai.conversation_binding_service, "read_text", upstream_read)
    monkeypatch.setattr(ai, "text_task_service", TextTaskService(path, completed, ImmediateExecutor()))
    route = "/api/conversation-bindings/text-requests/saved-text"
    response = client.get(route, headers=headers)
    assert response.status_code == 200
    assert response.json()["content"] == "private historical result"
    other = client.get(route, headers=foreign)
    assert other.json()["status"] == "not_found"
    assert "private historical result" not in other.text
    recovery = client.post(route + "/recover", headers=foreign, json={"allow_unrecoverable_retry": True})
    assert recovery.json()["status"] == "not_found"
    upstream_read.assert_not_called()
    completed.assert_called_once()
    auth.revoke_owned_key("workbench:company:user", item["id"])
    assert client.get(route, headers=headers).status_code == 401
    assert client.post(route + "/recover", headers=headers, json={"allow_unrecoverable_retry": True}).status_code == 401
    upstream_read.assert_not_called()


@pytest.mark.parametrize("routes", [["chat"], ["codex"], ["chat", "codex"]])
def test_codex_model_discovery_checks_endpoint_before_account_read(runtime, monkeypatch, routes):
    auth, client, *_ = runtime
    _, headers = key(auth, routes)
    read = Mock(return_value=CodexHTTPResponse(200, {"content-type": "application/json"}, body=b'{"models":[]}'))
    monkeypatch.setattr(codex.codex_service, "list_native_models", read)
    result = client.get("/codex/v1/models", headers=headers)
    assert result.status_code == (200 if "codex" in routes else 403)
    assert read.call_count == int("codex" in routes)


@pytest.mark.parametrize("routes", [["chat"], ["codex"], ["chat", "codex"]])
def test_unowned_binding_submission_is_technical_unavailable(runtime, monkeypatch, routes):
    auth, client, *_ = runtime
    _, headers = key(auth, routes)
    submit = Mock(side_effect=AssertionError("foreign binding must not be submitted"))
    monkeypatch.setattr(ai.text_task_service, "submit", submit)
    monkeypatch.setattr(ai.conversation_binding_service, "complete_text", submit)
    response = client.post("/api/conversation-bindings/text", headers=headers, json={
        "client_conversation_id": "foreign", "client_request_id": "copy",
        "provider_binding_id": "foreign-binding", "provider_account_identity": "foreign-account",
        "conversation_id": "foreign-chat", "parent_message_id": "foreign-turn",
        "messages": [{"role": "user", "content": "hello"}]})
    assert response.status_code == 501
    assert response.json()["detail"]["code"] == "SERVICE_OPERATION_UNAVAILABLE"
    submit.assert_not_called()


def test_endpoint_policy_keeps_existing_detailed_call_logging(runtime, monkeypatch):
    from services import log_service
    auth, client, *_ = runtime
    item, headers = key(auth, ["chat", "codex"])
    logged = Mock()
    monkeypatch.setattr(log_service.log_service, "add", logged)
    assert client.post("/v1/chat/completions", headers=headers,
        json={"model": "auto", "messages": [{"role": "user", "content": "describe"}]}).status_code == 200
    assert client.post("/v1/images/generations", headers=headers,
        json={"prompt": "sample", "model": "codex-gpt-image-2"}).status_code == 200
    details = [call.args[2] for call in logged.call_args_list]
    assert [(d["endpoint"], d["model"], d["key_id"], d["status"]) for d in details] == [
        ("/v1/chat/completions", "auto", item["id"], "success"),
        ("/v1/images/generations", "codex-gpt-image-2", item["id"], "success"),
    ]
    assert all("usage" not in d for d in details)  # Missing upstream usage is not invented as zero.
