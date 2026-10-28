from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import company_requests, external_images, image_tasks, task_results
from api.company_requests import PREFIX, company_identity
from services.public_chat_service import project_public_chat_receipt
from services.auth_service import AuthService
from services.image_task_service import ImageTaskService
from services.storage.json_storage import JSONStorageBackend
from services.text_task_service import TextTaskService

CONNECTOR = "1f084f01-d4b2-4080-8bce-b926f31cc454"


@pytest.fixture
def result_api(tmp_path, monkeypatch):
    auth = AuthService(JSONStorageBackend(tmp_path / "keys.json"))
    admin_item, admin = auth.create_key(role="admin")
    owner_a, key_a = auth.create_key(role="user", routes=["chat"])
    owner_b, key_b = auth.create_key(role="user", routes=["chat"])
    text = TextTaskService(tmp_path / "tasks.sqlite3", runner=Mock(side_effect=AssertionError("must not submit")))
    images = ImageTaskService(
        tmp_path / "images.json", store=text.store,
        generation_handler=Mock(side_effect=AssertionError("must not generate")),
        edit_handler=Mock(side_effect=AssertionError("must not edit")),
    )
    monkeypatch.setattr("api.support.auth_service", auth)
    monkeypatch.setattr(task_results, "text_task_service", text)
    monkeypatch.setattr(task_results, "image_task_service", images)
    monkeypatch.setattr(image_tasks, "image_task_service", images)
    app = FastAPI()
    app.middleware("http")(external_images.external_image_boundary)
    app.middleware("http")(company_requests.company_request_boundary)
    app.include_router(task_results.create_router())
    app.include_router(image_tasks.create_router())

    def headers(key=key_a):
        return {"Authorization": "Bearer " + key}

    def company_headers(user="employee"):
        return {
            "Authorization": "Bearer " + admin,
            "X-Workbench-Company-Org": "company",
            "X-Workbench-Company-User": user,
            "X-Workbench-Company-Connector": CONNECTOR,
            "X-Workbench-Expected-User": user,
        }

    return SimpleNamespace(
        client=TestClient(app), auth=auth, admin_item=admin_item, admin=admin, owner_a=owner_a,
        owner_b=owner_b, key_a=key_a, key_b=key_b, text=text, images=images,
        headers=headers, company_headers=company_headers,
    )


def _put_text(service, owner, request_id, **values):
    receipt = {
        "request_id": request_id,
        "model": "gpt-text",
        "status": "queued",
        "client_conversation_id": "private-conversation",
        **values,
    }
    with service.store.transaction() as db:
        db.execute("INSERT INTO requests VALUES(?,?,?,?)", (owner, request_id, "fixture", json.dumps(receipt)))
    return receipt


def _put_image(service, owner, request_id, **values):
    receipt = {
        "id": request_id,
        "owner_id": owner,
        "model": "gpt-image-2",
        "status": "queued",
        "provider_binding_id": "private-binding",
        "provider_account_identity": "private-account",
        "client_conversation_id": "private-conversation",
        **values,
    }
    with service.store.transaction() as db:
        service.store.write_receipt(db, "image", owner, request_id, receipt)
    return receipt


@pytest.mark.parametrize(("kind", "status", "value", "expected_status", "ready"), [
    ("text", "queued", None, "queued", False),
    ("text", "not_started", None, "not_started", False),
    ("text", "running", None, "running", False),
    ("text", "failed", None, "failed", False),
    ("text", "succeeded", "answer", "succeeded", True),
    ("text", "succeeded", "", "succeeded", False),
    ("text", "succeeded", " \t", "succeeded", False),
    ("text", "succeeded", {"not": "text"}, "succeeded", False),
    ("text", "unknown", None, "unknown", False),
    ("text", "unexpected", None, "unknown", False),
    ("image", "queued", None, "queued", False),
    ("image", "not_started", None, "not_started", False),
    ("image", "running", None, "running", False),
    ("image", "failed", None, "failed", False),
    ("image", "success", [{"b64_json": "aGVsbG8="}], "succeeded", True),
    ("image", "success", [], "succeeded", False),
    ("image", "success", {"not": "a list"}, "succeeded", False),
    ("image", "success", [{"not": "an image"}], "succeeded", False),
    ("image", "success", [{"b64_json": "  "}], "succeeded", False),
    ("image", "success", [{"url": "\t"}], "succeeded", False),
    ("image", "unknown", None, "unknown", False),
    ("image", "error", None, "failed", False),
    ("image", "unexpected", None, "unknown", False),
])
def test_task_result_projects_only_complete_persisted_outputs(
    result_api, kind, status, value, expected_status, ready,
):
    request_id = f"{kind}-{status}-{len(str(value))}"
    if kind == "text":
        _put_text(result_api.text, result_api.owner_a["id"], request_id, status=status, content=value)
    else:
        _put_image(result_api.images, result_api.owner_a["id"], request_id, status=status, data=value)

    response = result_api.client.get(f"/api/task-results/{kind}/{request_id}", headers=result_api.headers())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["protocol"] == "task-result-v1"
    assert {key: body[key] for key in ("kind", "request_id", "status", "result_ready")} == {
        "kind": kind, "request_id": request_id, "status": expected_status, "result_ready": ready,
    }
    expected_result = {"content": value} if kind == "text" else {"data": value}
    if ready:
        assert body["result"] == expected_result
    else:
        assert body["result"] is None
    assert response.headers["Cache-Control"] == "private, no-store"
    assert "private-binding" not in response.text and "private-account" not in response.text


def test_task_result_owner_and_chat_codex_policy_boundaries(result_api):
    _put_text(result_api.text, result_api.owner_a["id"], "owner-only", status="succeeded", content="answer")
    assert result_api.client.get("/api/task-results/text/owner-only", headers=result_api.headers(result_api.key_b)).status_code == 404

    _put_text(result_api.text, result_api.owner_a["id"], "codex-only", status="succeeded", content="codex", _route="codex")
    assert result_api.client.get("/api/task-results/text/codex-only", headers=result_api.headers()).status_code == 403

    policy = result_api.owner_a["policy"]
    result_api.auth.update_owned_policy(result_api.owner_a.get("owner_subject", ""), result_api.owner_a["id"], ["codex"], policy["revision"])
    denied = result_api.client.get("/api/task-results/text/owner-only", headers=result_api.headers())
    assert denied.status_code == 403 and denied.json()["detail"]["code"] == "KEY_ROUTE_DENIED"
    codex = result_api.client.get("/api/task-results/text/codex-only", headers=result_api.headers())
    assert codex.status_code == 200 and codex.json()["result"] == {"content": "codex"}
    assert codex.json()["receipt"]["route"] == "codex"


def test_task_result_preserves_unknown_image_outcome_and_validates_local_data(result_api, monkeypatch):
    owner = result_api.owner_a["id"]
    _put_image(
        result_api.images, owner, "unknown-original", status="error", error_code="CONVERSATION_OUTCOME_UNKNOWN",
        upstream_unfinished=True, _recovery_paused=True, data=[],
    )
    unknown = result_api.client.get("/api/task-results/image/unknown-original", headers=result_api.headers())
    assert unknown.status_code == 200
    assert unknown.json()["status"] == "unknown"
    assert unknown.json()["result_ready"] is False

    _put_image(result_api.images, owner, "confirmed-failure", status="error", error_code="CONTENT_POLICY_VIOLATION",
               upstream_unfinished=False, data=[])
    failed = result_api.client.get("/api/task-results/image/confirmed-failure", headers=result_api.headers())
    assert failed.status_code == 200 and failed.json()["status"] == "failed"

    _put_text(result_api.text, owner, "unrecoverable-text", status="failed", error_code="RESULT_UNRECOVERABLE",
              upstream_outcome="unknown")
    text_unknown = result_api.client.get("/api/task-results/text/unrecoverable-text", headers=result_api.headers())
    assert text_unknown.status_code == 200 and text_unknown.json()["status"] == "unknown"
    _put_image(result_api.images, owner, "unrecoverable-image", status="error", error_code="RESULT_UNRECOVERABLE",
               upstream_outcome="unknown", data=[])
    image_unknown = result_api.client.get("/api/task-results/image/unrecoverable-image", headers=result_api.headers())
    assert image_unknown.status_code == 200 and image_unknown.json()["status"] == "unknown"

    for request_id, data in (("bad-b64", [{"b64_json": "%%%"}]), ("bad-url", [{"url": "https://example.test/no-marker.png"}]),
                             ("malformed-url", [{"url": "https://["}])):
        _put_image(result_api.images, owner, request_id, status="success", data=data)
        response = result_api.client.get(f"/api/task-results/image/{request_id}", headers=result_api.headers())
        assert response.status_code == 200 and response.json()["result_ready"] is False

    exists = Mock(return_value=True)
    monkeypatch.setattr(task_results.image_storage_service, "exists", exists)
    _put_image(result_api.images, owner, "local-url", status="success", data=[{"url": "/images/2026/10/ready.png"}])
    ready = result_api.client.get("/api/task-results/image/local-url", headers=result_api.headers())
    assert ready.status_code == 200 and ready.json()["result"] == {"data": [{"url": "/images/2026/10/ready.png"}]}
    exists.assert_called_once_with("2026/10/ready.png")


def test_company_result_route_accepts_leading_dot_submission_id_and_public_route_is_not_forged(result_api):
    owner = company_identity("company", "employee", CONNECTOR)["id"]
    _put_text(result_api.text, owner, ".original-1", status="succeeded", content="company answer")
    response = result_api.client.get(
        PREFIX + "/api/task-results/text/.original-1", headers=result_api.company_headers(),
    )
    assert response.status_code == 200 and response.json()["result"] == {"content": "company answer"}
    assert project_public_chat_receipt({"request_id": "codex-receipt", "_route": "codex"})["route"] == "codex"
    assert project_public_chat_receipt({"request_id": "legacy", "_route": "not-a-route"})["route"] == "chat"


def test_company_leading_dot_result_download_is_allowed_and_consumable(result_api):
    owner = company_identity("company", "employee", CONNECTOR)["id"]
    _put_image(result_api.images, owner, ".original-1", status="success", data=[{"b64_json": "aGVsbG8="}])
    result = result_api.client.get(
        PREFIX + "/api/task-results/image/.original-1", headers=result_api.company_headers(),
    )
    assert result.status_code == 200
    url = result.json()["result"]["data"][0]["url"]
    assert url == "/api/v1/content/company-ai/api/image-tasks/.original-1/images/0"
    # The BFF maps the public prefix back onto this private company ingress.
    download = result_api.client.get(
        PREFIX + "/api/image-tasks/.original-1/images/0", headers=result_api.company_headers(),
    )
    assert download.status_code == 200 and download.content == b"hello"


def test_task_result_private_admin_bound_receipts_match_snapshot(result_api):
    owner = result_api.admin_item["id"]
    _put_text(
        result_api.text, owner, "admin-bound", status="succeeded", content="answer",
        provider_binding_id="binding", provider_account_identity="account",
        client_conversation_id="client", conversation_id="conversation", parent_message_id="parent",
    )
    snapshot = result_api.text.read_snapshot(owner, "admin-bound")
    response = result_api.client.get(
        "/api/task-results/text/admin-bound", headers={"Authorization": "Bearer " + result_api.admin},
    )
    assert response.status_code == 200
    assert response.json()["receipt"] == snapshot
    for field in ("provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id", "parent_message_id"):
        assert response.json()["receipt"][field]

    _put_image(
        result_api.images, owner, "admin-image-bound", status="success", data=[{"url": "https://example.test/image"}],
        conversation_id="image-conversation", parent_message_id="image-parent",
    )
    image = result_api.client.get(
        "/api/task-results/image/admin-image-bound", headers={"Authorization": "Bearer " + result_api.admin},
    )
    assert image.status_code == 200
    assert image.json()["receipt"]["image_session_id"] == "image-conversation"
    assert image.json()["receipt"]["image_session_parent_id"] == "image-parent"
    external_admin = result_api.client.get(
        "/api/task-results/text/admin-bound",
        headers={"Authorization": "Bearer " + result_api.admin, "X-Workbench-Image-Client": "1"},
    )
    assert external_admin.status_code == 403


def test_task_result_is_read_only_and_survives_restart(result_api, monkeypatch):
    original = _put_text(
        result_api.text, result_api.owner_a["id"], "restartable", status="succeeded", content="saved answer",
        provider_binding_id="private-binding", provider_account_identity="private-account",
    )
    with result_api.text.store.connect() as db:
        before = copy.deepcopy(result_api.text.store.read_receipt(db, "text", result_api.owner_a["id"], "restartable"))
    image_original = _put_image(
        result_api.images, result_api.owner_a["id"], "image-readonly", status="success", data=[{"b64_json": "aGVsbG8="}],
    )
    with result_api.text.store.connect() as db:
        image_before = copy.deepcopy(result_api.text.store.read_receipt(db, "image", result_api.owner_a["id"], "image-readonly"))
    result_api.text.read = Mock(side_effect=AssertionError("legacy read must not run"))
    result_api.text.recovery_reader = Mock(side_effect=AssertionError("recovery must not run"))
    result_api.images.list_tasks = Mock(side_effect=AssertionError("image list must not run"))
    response = result_api.client.get("/api/task-results/text/restartable", headers=result_api.headers())
    assert response.status_code == 200 and response.json()["result"] == {"content": "saved answer"}
    image = result_api.client.get("/api/task-results/image/image-readonly", headers=result_api.headers())
    assert image.status_code == 200 and image.json()["result_ready"] is True
    with result_api.text.store.connect() as db:
        assert result_api.text.store.read_receipt(db, "text", result_api.owner_a["id"], "restartable") == before == original
        assert result_api.text.store.read_receipt(db, "image", result_api.owner_a["id"], "image-readonly") == image_before == image_original
    result_api.text.read.assert_not_called()
    result_api.text.recovery_reader.assert_not_called()
    result_api.images.list_tasks.assert_not_called()
    result_api.text.runner.assert_not_called()
    result_api.images.generation_handler.assert_not_called()
    result_api.images.edit_handler.assert_not_called()

    restarted = TextTaskService(result_api.text.path, runner=result_api.text.runner)
    monkeypatch.setattr(task_results, "text_task_service", restarted)
    after_restart = result_api.client.get("/api/task-results/text/restartable", headers=result_api.headers())
    assert after_restart.status_code == 200 and after_restart.json()["result"] == {"content": "saved answer"}
    restarted.runner.assert_not_called()


def test_task_result_supports_company_owner_and_external_image_projection(result_api):
    owner = company_identity("company", "employee", CONNECTOR)["id"]
    _put_text(result_api.text, owner, "company-text", status="succeeded", content="company answer")
    _put_image(result_api.images, owner, "company-image", status="success", data=[{"b64_json": "aGVsbG8="}])

    text = result_api.client.get(PREFIX + "/api/task-results/text/company-text", headers=result_api.company_headers())
    assert text.status_code == 200 and text.json()["result"] == {"content": "company answer"}
    image = result_api.client.get(PREFIX + "/api/task-results/image/company-image", headers=result_api.company_headers())
    assert image.status_code == 200 and image.json()["result"] == {"data": [
        {"url": "/api/v1/content/company-ai/api/image-tasks/company-image/images/0"},
    ]}
    assert image.headers["Cache-Control"] == "private, no-store"
    assert result_api.client.get(
        PREFIX + "/api/task-results/text/company-text", headers=result_api.company_headers(user="other"),
    ).status_code == 404
