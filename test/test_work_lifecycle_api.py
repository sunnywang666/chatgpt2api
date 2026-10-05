from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from api import chat_requests, image_tasks, external_images, support
from api.company_requests import allowed_route
from services.auth_service import AuthService
from services.durable_forward import respond
from services.storage.json_storage import JSONStorageBackend
from api.external_images import external_image_boundary
from services.work_lifecycle import WorkLifecycleError


class FakeWorkLifecycle:
    def __init__(self):
        self.calls = []

    def get(self, kind, identity, request_id):
        self.calls.append(("get", kind, identity["id"], request_id))
        return {"protocol": "work-v1", "kind": kind, "work_ref": "safe-work", "request_id": request_id,
                "state": "active", "slot_held": True, "version": 1, "results_saved": False,
                "archive": {"status": "not_requested", "desired": None}}

    def set_archived(self, kind, identity, request_id, archived):
        self.calls.append(("archive", kind, identity["id"], request_id, archived))
        return ({"request_id": request_id, "archived": archived, "conversation": {"protocol": "sequential-v1", "client_conversation_id": "safe-work"}}
                if kind == "text" else {"task_id": request_id, "archived": archived, "image_thread": {"id": "safe-work"}})

    def update(self, kind, identity, request_id, state, results_saved=False):
        self.calls.append(("update", kind, identity["id"], request_id, state, results_saved))
        return self.get(kind, identity, request_id) | {"state": state, "results_saved": results_saved}


def test_native_work_routes_map_owner_and_strict_completed_contract(monkeypatch):
    work = FakeWorkLifecycle()
    identity = {"id": "owner-a", "role": "user"}
    monkeypatch.setattr(chat_requests, "_ordinary_identity", lambda *_args: identity)
    monkeypatch.setattr(chat_requests, "_raw_text_receipt", lambda *_args: {"model": "gpt-text", "_route": "chat"})
    monkeypatch.setattr(chat_requests, "require_chat_text_policy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_requests, "get_work_lifecycle_service", lambda: work)
    monkeypatch.setattr(chat_requests.text_task_service, "admission", object())
    monkeypatch.setattr(image_tasks, "require_identity", lambda *_args, **_kwargs: identity)
    monkeypatch.setattr(image_tasks.image_task_service, "list_tasks", lambda *_args: {"items": [{"model": "gpt-image-2"}], "missing_ids": []})
    monkeypatch.setattr(image_tasks, "require_image_policy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(image_tasks, "get_work_lifecycle_service", lambda: work)
    monkeypatch.setattr(image_tasks.image_task_service, "admission", object())

    app = FastAPI()
    app.include_router(chat_requests.create_router())
    app.include_router(image_tasks.create_router())
    client = TestClient(app)

    chat = client.post("/api/chat-requests/chat-1/work", json={"state": "completed", "results_saved": True})
    assert chat.status_code == 200, chat.text
    image = client.get("/api/image-tasks/image-1/work")
    assert image.status_code == 200, image.text
    chat_archive = client.post("/api/chat-requests/chat-1/archive-conversation", json={})
    image_restore = client.post("/api/image-tasks/image-1/restore-thread", json={})
    assert chat_archive.status_code == 200, chat_archive.text
    assert image_restore.status_code == 200, image_restore.text
    invalid = client.post("/api/image-tasks/image-1/work", json={"state": "completed"})
    assert invalid.status_code == 422
    assert ("update", "text", "owner-a", "chat-1", "completed", True) in work.calls
    assert ("get", "image", "owner-a", "image-1") in work.calls
    assert ("archive", "text", "owner-a", "chat-1", True) in work.calls
    assert ("archive", "image", "owner-a", "image-1", False) in work.calls


def test_work_routes_keep_original_codex_policy_and_service_error(monkeypatch):
    work = FakeWorkLifecycle()
    identity = {"id": "owner-a", "role": "user"}
    calls = []
    monkeypatch.setattr(chat_requests, "_ordinary_identity", lambda *_args: identity)
    monkeypatch.setattr(chat_requests, "_raw_text_receipt", lambda *_args: {"model": "gpt-5.6-codex", "_route": "codex"})
    monkeypatch.setattr(chat_requests, "require_codex_endpoint", lambda received: calls.append(received["id"]))
    monkeypatch.setattr(chat_requests, "get_work_lifecycle_service", lambda: work)
    monkeypatch.setattr(chat_requests.text_task_service, "admission", object())
    app = FastAPI(); app.include_router(chat_requests.create_router())
    client = TestClient(app)
    response = client.get("/api/chat-requests/codex-1/work")
    assert response.status_code == 200, response.text
    assert calls == ["owner-a"]

    def unavailable(*_args):
        raise WorkLifecycleError("WORK_RESTORE_PENDING", 409)
    monkeypatch.setattr(chat_requests, "get_work_lifecycle_service", lambda: type("Broken", (), {"get": unavailable})())
    response = client.get("/api/chat-requests/codex-1/work")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "WORK_RESTORE_PENDING"


def test_codex_work_access_uses_the_owner_scoped_raw_route_and_current_key_policy(tmp_path, monkeypatch):
    auth = AuthService(JSONStorageBackend(tmp_path / "keys.json"))
    key, secret = auth.create_key(role="user", owner_subject="workbench:org:person", routes=["codex"])
    work = FakeWorkLifecycle()
    monkeypatch.setattr(support, "auth_service", auth)
    monkeypatch.setattr(
        chat_requests,
        "_raw_text_receipt",
        lambda owner, request_id: {"model": "gpt-5.6-codex", "_route": "codex"}
        if (owner, request_id) == (key["id"], "original-codex") else None,
    )
    monkeypatch.setattr(chat_requests, "get_work_lifecycle_service", lambda: work)
    app = FastAPI(); app.include_router(chat_requests.create_router())
    client = TestClient(app)
    headers = {"Authorization": "Bearer " + secret}

    assert client.get("/api/chat-requests/original-codex/work", headers=headers).status_code == 200
    assert client.post("/api/chat-requests/original-codex/work", headers=headers,
                       json={"state": "paused"}).status_code == 200
    auth.update_owned_policy("workbench:org:person", key["id"], ["chat"], 1)
    assert client.get("/api/chat-requests/original-codex/work", headers=headers).status_code == 403
    assert client.post("/api/chat-requests/original-codex/work", headers=headers,
                       json={"state": "active"}).status_code == 403


def test_submission_entries_preserve_work_lifecycle_error_codes(monkeypatch):
    identity = {"id": "owner-a", "role": "user"}
    error = WorkLifecycleError("WORK_NOT_ACTIVE", 409)
    monkeypatch.setattr(chat_requests, "_ordinary_identity", lambda *_args: identity)
    monkeypatch.setattr(chat_requests.text_task_service, "validate_submission", lambda *_args: None)
    monkeypatch.setattr(chat_requests.text_task_service, "submit", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    monkeypatch.setattr(chat_requests, "require_chat_text_policy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(chat_requests, "require_public_text_model", lambda *_args: None)
    monkeypatch.setattr(chat_requests, "require_public_reasoning_effort", lambda *_args: None)
    monkeypatch.setattr(chat_requests, "check_request", lambda *_args: None)
    monkeypatch.setattr(image_tasks, "require_identity", lambda *_args, **_kwargs: identity)
    monkeypatch.setattr(image_tasks, "require_image_policy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(image_tasks, "check_request", lambda *_args: None)
    monkeypatch.setattr(image_tasks.image_task_service, "submit_generation", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    app = FastAPI(); app.include_router(chat_requests.create_router()); app.include_router(image_tasks.create_router())
    client = TestClient(app)
    chat = client.post("/api/chat-requests", json={"client_request_id": "paused-chat", "model": "gpt-text",
                                                      "messages": [{"role": "user", "content": "hello"}]})
    image = client.post("/api/image-tasks/generations", json={"client_task_id": "paused-image", "prompt": "draw"})
    assert chat.status_code == image.status_code == 409
    assert chat.json()["detail"]["code"] == image.json()["detail"]["code"] == "WORK_NOT_ACTIVE"

    class RejectedTextService:
        def submit(self, *_args, **_kwargs):
            raise WorkLifecycleError("WORK_SCHEDULING_CONFLICT", 409)

    request = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions",
                       "headers": [], "query_string": b"", "server": ("testserver", 80), "scheme": "http"})
    with pytest.raises(HTTPException) as raised:
        asyncio.run(respond(identity, {"model": "gpt-text"}, request, "openai_v1_chat_complete", service=RejectedTextService()))
    assert raised.value.status_code == 409
    assert raised.value.detail == {"code": "WORK_SCHEDULING_CONFLICT"}

    monkeypatch.setattr("services.image_task_service.image_task_service.submit_generation",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(HTTPException) as raised:
        asyncio.run(external_images.synchronous_external_task(identity, {"client_task_id": "paused-external", "prompt": "draw",
                                                                           "model": "gpt-image-2", "_scheduling": {"workflow_id": "w"}}, edit=False))
    assert raised.value.status_code == 409
    assert raised.value.detail == {"code": "WORK_NOT_ACTIVE"}


def test_external_and_company_allow_only_native_work_routes():
    assert allowed_route("GET", "/api/chat-requests/a/work")
    assert allowed_route("POST", "/api/image-tasks/a/work")
    assert not allowed_route("POST", "/v1/chat/completions")

    app = FastAPI()
    app.middleware("http")(external_image_boundary)
    app.get("/api/chat-requests/a/work")(lambda: {"ok": True})
    client = TestClient(app)
    # The boundary reaches the route; it does not turn this native endpoint
    # into a compatibility ingress. Authentication fails before a body read.
    response = client.get("/api/chat-requests/a/work", headers={"X-Workbench-Image-Client": "1"})
    assert response.status_code == 401
