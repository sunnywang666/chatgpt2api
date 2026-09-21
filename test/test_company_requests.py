from __future__ import annotations

import base64
import io
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import ai, chat_requests, image_tasks
from api.app import create_app
from api.company_requests import PREFIX, PUBLIC_PREFIX, company_identity
from services.auth_service import AuthService
from services.image_task_service import ImageTaskService
from services.storage.json_storage import JSONStorageBackend
from services.text_task_service import TextTaskService

CONNECTOR = "1f084f01-d4b2-4080-8bce-b926f31cc454"
OTHER_CONNECTOR = "1f084f01-d4b2-4080-8bce-b926f31cc455"


class Queue:
    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        self.calls.append((function, args))

    def run(self):
        function, args = self.calls.pop(0)
        function(*args)


@pytest.fixture
def company(tmp_path, monkeypatch):
    auth = AuthService(JSONStorageBackend(tmp_path / "accounts.json"))
    _, admin = auth.create_key(role="admin")
    old_key, ordinary = auth.create_key(role="user", routes=["chat"])
    monkeypatch.setattr("api.support.auth_service", auth)
    queue = Queue()
    runner = Mock(return_value={"content": "answer", "provider_binding_id": "private-binding"})
    text_tasks = TextTaskService(tmp_path / "text.sqlite3", runner=runner, executor=queue)
    monkeypatch.setattr(chat_requests, "text_task_service", text_tasks)
    monkeypatch.setattr(chat_requests, "check_request", lambda *_: None)
    monkeypatch.setattr(image_tasks, "check_request", lambda *_: None)
    monkeypatch.setattr("services.log_service.log_service.add", lambda *args, **kwargs: None)
    monkeypatch.setattr("services.public_chat_service.model_catalog_service.route_for_model",
                        lambda _: SimpleNamespace(account_types=frozenset({"Plus"}), allow_anonymous=False))
    monkeypatch.setattr(ai.openai_v1_models, "list_models", lambda: {
        "data": [{"id": "gpt-text"}, {"id": "gpt-image-2"}], "object": "list"})
    png = io.BytesIO()
    Image.new("RGB", (2, 2)).save(png, format="PNG")
    image_calls = []

    def generate(payload):
        image_calls.append(payload)
        return {"data": [{"b64_json": base64.b64encode(png.getvalue()).decode()}],
                "_provider_binding_id": payload["provider_binding_id"],
                "_provider_account_identity": payload["provider_account_identity"],
                "_conversation_id": "original-conversation", "_parent_message_id": "original-message"}

    images = ImageTaskService(tmp_path / "images.json", generation_handler=generate, edit_handler=generate)
    monkeypatch.setattr(image_tasks, "image_task_service", images)
    monkeypatch.setattr("services.account_service.account_service.create_conversation_binding",
                        lambda **_: ("private-binding", "private-account", "private-token"))
    monkeypatch.setattr("services.account_service.account_service.release_image_slot", lambda *_: None)
    monkeypatch.setattr("services.account_service.account_service.get_account", lambda *_: {"quota": 2})

    def headers(user="employee", org="company", connector=CONNECTOR):
        return {"Authorization": "Bearer " + admin, "X-Workbench-Company-Org": org,
                "X-Workbench-Company-User": user, "X-Workbench-Company-Connector": connector,
                "X-Workbench-Expected-User": user}

    return SimpleNamespace(client=TestClient(create_app()), headers=headers, auth=auth, admin=admin,
                           ordinary=ordinary, old_key=old_key, runner=runner, queue=queue,
                           text_tasks=text_tasks, images=images, image_calls=image_calls, png=png.getvalue())


def body(text="hello"):
    return {"client_request_id": "original-chat", "model": "gpt-text",
            "messages": [{"role": "user", "content": text}]}


@pytest.mark.parametrize("change,status", [
    ({"Authorization": "Bearer invalid"}, 401),
    ({"X-Workbench-Expected-User": "someone"}, 409),
    ({"X-Workbench-Company-Org": ""}, 400),
    ({"X-Workbench-Company-Connector": "not-a-uuid"}, 400),
    ({"X-Workbench-Image-Client": "1"}, 403),
])
def test_company_requires_private_credential_and_complete_identity(company, change, status):
    response = company.client.get(PREFIX + "/session", headers={**company.headers(), **change})
    assert response.status_code == status
    assert company.admin not in response.text


def test_company_never_exposes_management_or_codex(company):
    for path in ("/api/accounts", "/api/workbench/ai/keys", "/v1/responses", "/codex/v1/models"):
        assert company.client.get(PREFIX + path, headers=company.headers()).status_code == 404
    assert company.client.get(PREFIX + "/session", headers={
        **company.headers(), "Authorization": "Bearer " + company.ordinary}).status_code == 403
    session = company.client.get(PREFIX + "/session", headers=company.headers()).json()
    assert session == {"contract_version": 1, "org_id": "company", "user_id": "employee", "connector_id": CONNECTOR}
    assert company.client.get(PREFIX + "/v1/models", headers=company.headers()).json()["data"][0]["id"] == "gpt-text"


def test_durable_chat_restart_cookie_renewal_and_old_key_isolation(company, monkeypatch):
    response = company.client.post(PREFIX + "/api/chat-requests", headers=company.headers(), json=body())
    assert response.status_code == 202, response.text
    company.queue.run()
    restarted = TextTaskService(company.text_tasks.path, runner=company.runner, executor=Queue())
    monkeypatch.setattr(chat_requests, "text_task_service", restarted)
    renewed = {**company.headers(), "Cookie": "company-session=renewed"}
    read = company.client.get(PREFIX + "/api/chat-requests/original-chat", headers=renewed)
    assert read.status_code == 200 and read.json()["content"] == "answer"
    assert "private-binding" not in read.text
    assert company.client.post(PREFIX + "/api/chat-requests", headers=renewed, json=body()).status_code == 200
    assert company.client.post(PREFIX + "/api/chat-requests", headers=renewed, json=body("changed")).status_code == 409
    assert company.runner.call_count == 1
    for headers in (company.headers(user="other"), company.headers(org="other"),
                    company.headers(connector=OTHER_CONNECTOR)):
        assert company.client.get(PREFIX + "/api/chat-requests/original-chat", headers=headers).status_code == 404
        assert company.client.post(PREFIX + "/api/chat-requests/original-chat/recover", headers=headers, json={}).status_code == 404
    # Forging company metadata on a normal-key route cannot change its owner.
    public = {**company.headers(), "Authorization": "Bearer " + company.ordinary, "X-Workbench-Image-Client": "1"}
    assert company.client.get("/api/chat-requests/original-chat", headers=public).status_code == 404
    original = company.client.post("/api/chat-requests", headers=public, json=body("legacy"))
    assert original.status_code == 202
    assert restarted.read(company.old_key["id"], "original-chat")["status"] != "not_found"
    assert company.client.get(PREFIX + "/api/chat-requests/original-chat", headers=company.headers()).json()["content"] == "answer"


def test_company_unknown_submission_is_not_reexecuted_after_restart(company, monkeypatch):
    assert company.client.post(PREFIX + "/api/chat-requests", headers=company.headers(), json=body()).status_code == 202
    owner = company_identity("company", "employee", CONNECTOR)["id"]
    company.text_tasks._update(owner, "original-chat", _input_ref=None)  # Legacy input-less receipt.
    queue = Queue()
    restarted = TextTaskService(company.text_tasks.path, runner=company.runner, executor=queue)
    monkeypatch.setattr(chat_requests, "text_task_service", restarted)
    original = company.client.get(PREFIX + "/api/chat-requests/original-chat", headers=company.headers())
    assert original.json()["status"] == "not_started"
    repeated = company.client.post(PREFIX + "/api/chat-requests", headers=company.headers(), json=body())
    assert repeated.status_code == 202
    assert repeated.json()["status"] == "not_started"
    assert queue.calls == []
    company.runner.assert_not_called()


def test_company_multipart_images_query_download_and_recovery(company, monkeypatch):
    response = company.client.post(PREFIX + "/api/image-tasks/edits", headers=company.headers(),
        data={"client_task_id": "original-image", "prompt": "edit", "model": "gpt-image-2"},
        files=[("image", ("one.png", company.png, "image/png")),
               ("image", ("two.png", company.png, "image/png"))])
    assert response.status_code == 200, response.text
    for _ in range(100):
        read = company.client.get(PREFIX + "/api/image-tasks?ids=original-image", headers=company.headers())
        if read.json()["items"][0]["status"] in {"success", "error"}:
            break
        time.sleep(.01)
    task = read.json()["items"][0]
    assert task["status"] == "success", task
    assert task["data"] == [{"url": PUBLIC_PREFIX + "/api/image-tasks/original-image/images/0"}]
    assert "private-" not in read.text
    assert len(company.image_calls) == 1
    assert len(company.image_calls[0]["images"]) == 2
    duplicate = company.client.get(PREFIX + "/api/image-tasks?ids=original-image,original-image", headers=company.headers())
    assert duplicate.status_code == 200
    restarted = ImageTaskService(company.images.path, generation_handler=Mock(), edit_handler=Mock())
    monkeypatch.setattr(image_tasks, "image_task_service", restarted)
    download = company.client.get(PREFIX + "/api/image-tasks/original-image/images/0", headers=company.headers())
    assert download.content == company.png
    assert download.headers["cache-control"] == "private, no-store"
    for headers in (company.headers(user="other"), company.headers(connector=OTHER_CONNECTOR)):
        assert company.client.get(PREFIX + "/api/image-tasks?ids=original-image", headers=headers).status_code == 404
        assert company.client.get(PREFIX + "/api/image-tasks/original-image/images/0", headers=headers).status_code == 404
        assert company.client.post(PREFIX + "/api/image-tasks/original-image/resume-poll", headers=headers, json={}).status_code == 404
    forbidden = company.client.post(PREFIX + "/api/image-tasks/original-image/resume-poll", headers=company.headers(),
                                    json={"allow_unrecoverable_retry": True})
    assert forbidden.status_code == 400
    recovered = company.client.post(PREFIX + "/api/image-tasks/original-image/resume-poll", headers=company.headers(), json={})
    assert recovered.status_code == 200 and recovered.json()["status"] == "success"
    restarted.generation_handler.assert_not_called()
    restarted.edit_handler.assert_not_called()


def test_company_identity_is_stable_and_not_key_ownership():
    a = company_identity("org", "user", CONNECTOR)
    assert a == company_identity("org", "user", CONNECTOR)
    assert a["id"] != company_identity("other", "user", CONNECTOR)["id"]
    assert a["role"] == "user" and a["policy"]["routes"] == ["chat"]
