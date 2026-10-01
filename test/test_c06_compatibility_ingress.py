from __future__ import annotations

import base64
import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from api import ai, codex, image_tasks
from services.auth_service import AuthService
from services.image_thread import ImageThreadError
from services.openai_backend_api import ChatRequirements, OpenAIBackendAPI
from services.request_context import current_request
from services.storage.json_storage import JSONStorageBackend
from services import durable_forward


REF = "car_" + "A" * 43
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGPkEpFjYGBgYgADAALmAEAUQs4PAAAAAElFTkSuQmCC")
DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode()


@pytest.fixture
def client(tmp_path, monkeypatch):
    auth = AuthService(JSONStorageBackend(Path(tmp_path) / "accounts.json"))
    _, secret = auth.create_key(role="admin", name="compat-ingress")
    monkeypatch.setattr("api.support.auth_service", auth)
    app = FastAPI()
    app.include_router(ai.create_router())
    app.include_router(image_tasks.create_router())
    return TestClient(app), {"Authorization": "Bearer " + secret}


@pytest.mark.parametrize(
    ("path", "body", "protocol", "operation"),
    [
        ("/v1/images/generations", {"prompt": "draw", "model": "gpt-image-2"}, "openai_v1_image_generations", "image"),
        ("/v1/chat/completions", {"model": "gpt-text", "messages": [{"role": "user", "content": "hello"}]}, "openai_v1_chat_complete", "text"),
        ("/v1/responses", {"model": "gpt-text", "input": "hello"}, "openai_v1_response", "text"),
        ("/v1/messages", {"model": "gpt-text", "messages": [{"role": "user", "content": "hello"}]}, "anthropic_v1_messages", "text"),
        ("/v1/search", {"prompt": "find product"}, "openai_search", "text"),
    ],
)
def test_selected_compatibility_ingress_uses_durable_forward(
        client, monkeypatch, path, body, protocol, operation):
    http, headers = client
    calls = []

    async def accepted(identity, payload, request, received_protocol, *, operation="text", **_kwargs):
        calls.append((payload, received_protocol, operation))
        return {"accepted": True}

    monkeypatch.setattr(ai.text_task_service, "admission", object())
    monkeypatch.setattr("services.durable_forward.respond", accepted)
    response = http.post(path, headers={**headers, "X-Client-Request-ID": "selected-original"}, json={**body, "account_ref": REF})
    assert response.status_code == 200, response.text
    assert len(calls) == 1
    payload, received_protocol, received_operation = calls[0]
    assert payload["account_ref"] == REF
    assert received_protocol == protocol
    assert received_operation == operation


def test_selected_native_codex_ingress_preserves_only_durable_ref(monkeypatch):
    app = FastAPI()
    app.include_router(codex.create_router())
    client = TestClient(app)
    calls = []

    async def accepted(identity, payload, request, protocol, **kwargs):
        saved = durable_forward.envelope(identity, payload, request, protocol, **kwargs)
        calls.append(saved)
        return {"accepted": True}

    monkeypatch.setattr(codex, "_ordinary_identity", lambda _authorization: {"id": "owner", "role": "user"})
    monkeypatch.setattr(codex, "require_codex_policy", lambda *_args: None)
    monkeypatch.setattr(ai.text_task_service, "admission", object())
    monkeypatch.setattr("services.durable_forward.respond", accepted)
    response = client.post(
        "/codex/v1/responses", headers={"X-Client-Request-ID": "selected-codex"},
        json={"model": "gpt-5.6-codex", "input": "hello", "account_ref": REF},
    )
    assert response.status_code == 200, response.text
    assert calls[0]["_route"] == "codex"
    assert calls[0]["_requested_account_ref"] == REF
    assert "account_ref" not in calls[0]["_forward"]["payload"]


def test_selected_compatibility_requires_original_id_and_durable_admission(client, monkeypatch):
    http, headers = client
    body = {"model": "gpt-text", "messages": [{"role": "user", "content": "hello"}], "account_ref": REF}
    monkeypatch.setattr(ai.text_task_service, "admission", object())
    missing = http.post("/v1/chat/completions", headers=headers, json=body)
    assert missing.status_code == 400
    assert missing.json()["detail"]["code"] == "ACCOUNT_SELECTION_REQUEST_ID_REQUIRED"
    monkeypatch.setattr(ai.text_task_service, "admission", None)
    unavailable = http.post("/v1/chat/completions", headers={**headers, "X-Client-Request-ID": "selected-original"}, json=body)
    assert unavailable.status_code == 503
    assert unavailable.json()["detail"]["code"] == "ACCOUNT_SELECTION_REQUIRES_DURABLE_ADMISSION"


@pytest.mark.parametrize("account_ref", [None, "car_short", 7])
def test_compatibility_rejects_non_advertised_account_refs(client, account_ref):
    http, headers = client
    response = http.post(
        "/v1/chat/completions", headers=headers,
        json={"model": "gpt-text", "messages": [{"role": "user", "content": "hello"}], "account_ref": account_ref},
    )
    assert response.status_code == 422


def test_image_task_json_and_multipart_preserve_only_safe_ref(client, monkeypatch):
    http, headers = client
    calls = []

    def accepted(_identity, **kwargs):
        calls.append(kwargs)
        return {"id": kwargs["client_task_id"], "status": "queued", "data": []}

    monkeypatch.setattr(image_tasks.image_task_service, "submit_generation", accepted)
    response = http.post(
        "/api/image-tasks/generations", headers=headers,
        json={"client_task_id": "selected-image", "prompt": "draw", "account_ref": REF},
    )
    assert response.status_code == 200, response.text
    assert calls[0]["account_ref"] == REF
    assert calls[0]["provider_account_identity"] == ""
    assert "provider_account_identity" not in response.text

    invalid = http.post(
        "/api/image-tasks/generations", headers=headers,
        json={"client_task_id": "bad-ref", "prompt": "draw", "account_ref": None},
    )
    assert invalid.status_code == 422

    edit_calls = []
    monkeypatch.setattr(image_tasks.image_task_service, "submit_edit", lambda _identity, **kwargs: edit_calls.append(kwargs) or {"id": kwargs["client_task_id"], "status": "queued", "data": []})
    edited = http.post(
        "/api/image-tasks/edits", headers=headers,
        data={"client_task_id": "selected-edit", "prompt": "edit", "account_ref": REF},
        files={"image": ("source.png", PNG, "image/png")},
    )
    assert edited.status_code == 200, edited.text
    assert edit_calls[0]["account_ref"] == REF


def test_codex_image_post_rechecks_admission_at_urllib_send(monkeypatch):
    class Context:
        calls = 0

        def before_send(self):
            self.calls += 1

    class Raw:
        headers = {"content-type": "text/event-stream"}
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'data: {"type":"response.completed"}\n\n'

    context = Context()
    monkeypatch.setattr("services.openai_backend_api.account_service.get_account", lambda _token: {"source_type": "codex"})
    backend = OpenAIBackendAPI(access_token="fixture-token")

    def urlopen(_request, *, timeout):
        assert timeout == 1200
        assert context.calls == 1
        return Raw()

    monkeypatch.setattr("services.openai_backend_api.urllib.request.urlopen", urlopen)
    token = current_request.set(context)
    try:
        assert list(backend.iter_codex_image_response_events("draw")) == [{"type": "response.completed"}]
    finally:
        current_request.reset(token)


def test_direct_codex_image_post_rechecks_capacity(monkeypatch):
    class Raw:
        headers = {"content-type": "text/event-stream"}
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'data: {"type":"response.completed"}\n\n'

    checks = []
    monkeypatch.setattr("services.openai_backend_api.account_service.get_account", lambda _token: {"source_type": "codex"})
    monkeypatch.setattr("services.openai_backend_api.account_service.require_image_account", lambda token, model: checks.append((token, model)) or {})
    monkeypatch.setattr("services.openai_backend_api.urllib.request.urlopen", lambda _request, *, timeout: Raw())
    backend = OpenAIBackendAPI(access_token="fixture-token")
    assert list(backend.iter_codex_image_response_events("draw")) == [{"type": "response.completed"}]
    assert checks == [("fixture-token", "codex-gpt-image-2")]


def test_synchronous_selected_image_preserves_core_selection_error(client, monkeypatch):
    http, headers = client

    def rejected(_identity, **_kwargs):
        raise ImageThreadError("IMAGE_ACCOUNT_NOT_FOUND", status=404)

    monkeypatch.setattr(image_tasks.image_task_service, "submit_generation", rejected)
    monkeypatch.setattr(ai.text_task_service, "admission", object())
    response = http.post(
        "/v1/images/generations",
        headers={**headers, "X-Workbench-Image-Client": "1", "X-Client-Request-ID": "selected-image"},
        json={"client_task_id": "selected-image", "prompt": "draw", "account_ref": REF},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "IMAGE_ACCOUNT_NOT_FOUND"


def test_durable_image_selection_error_is_not_converted_to_generic_failure():
    class RejectedService:
        def submit(self, *_args, **_kwargs):
            raise ImageThreadError("IMAGE_ACCOUNT_NOT_FOUND", status=404)

    request = Request({"type": "http", "method": "POST", "path": "/v1/images/generations",
                       "headers": [], "query_string": b"", "server": ("testserver", 80), "scheme": "http"})
    with pytest.raises(HTTPException) as raised:
        asyncio.run(durable_forward.respond(
            {"id": "owner", "role": "user"}, {"model": "gpt-image-2", "account_ref": REF},
            request, "openai_v1_image_generations", operation="image", service=RejectedService(),
        ))
    assert raised.value.status_code == 404
    assert raised.value.detail == {"code": "IMAGE_ACCOUNT_NOT_FOUND"}


def test_chat_image_post_rechecks_capacity_before_paced_send(monkeypatch):
    class Session:
        headers = {}

        def post(self, *_args, **kwargs):
            kwargs["_account_request_before_send"]()
            raise AssertionError("the transport must not post after failed capacity recheck")

    checks = []
    monkeypatch.setattr("services.openai_backend_api.account_service.get_account", lambda _token: {})
    backend = OpenAIBackendAPI(access_token="fixture-token")
    backend.session = Session()
    monkeypatch.setattr(backend, "_bootstrap", lambda: None)
    monkeypatch.setattr(backend, "_get_chat_requirements", lambda: ChatRequirements(token="fixture"))
    monkeypatch.setattr(backend, "_prepare_image_conversation", lambda *_args, **_kwargs: "conduit")

    def unavailable(token, model):
        checks.append((token, model))
        raise RuntimeError("selected image account capability is unavailable")

    monkeypatch.setattr("services.openai_backend_api.account_service.require_image_account", unavailable)
    with pytest.raises(RuntimeError, match="capability is unavailable"):
        list(backend._stream_picture_conversation("draw", "gpt-image-2", []))
    assert checks == [("fixture-token", "gpt-image-2")]
    assert backend.image_submission_started is False


def test_compatibility_scheduling_stays_private_in_the_durable_envelope(client, monkeypatch):
    http, headers = client
    received = []

    async def accepted(identity, payload, request, protocol, **kwargs):
        received.append(durable_forward.envelope(identity, payload, request, protocol, **kwargs))
        return {"accepted": True}

    monkeypatch.setattr(ai.text_task_service, "admission", object())
    monkeypatch.setattr("services.durable_forward.respond", accepted)
    response = http.post(
        "/v1/chat/completions",
        headers={**headers, "X-Client-Request-ID": "scheduled-original"},
        json={"model": "gpt-text", "messages": [{"role": "user", "content": "hello"}],
              "_scheduling": {"workflow_id": "forged-private"},
              "scheduling": {"workflow_id": "catalog-refresh", "workflow_concurrency": 2,
                             "min_send_interval_seconds": 3, "not_before": "2026-10-01T00:00:00Z"}},
    )
    assert response.status_code == 200, response.text
    envelope = received[0]
    assert envelope["_scheduling"] == {
        "workflow_id": "catalog-refresh", "workflow_concurrency": 2,
        "min_send_interval_seconds": 3, "not_before": "2026-10-01T00:00:00Z",
    }
    assert "scheduling" not in envelope["_forward"]["payload"]
    assert "_scheduling" not in envelope["_forward"]["payload"]


def test_compatibility_discards_forged_private_scheduling_without_public_schedule():
    body = ai.ChatCompletionRequest(
        model="gpt-text", messages=[{"role": "user", "content": "hello"}],
        _scheduling={"workflow_id": "forged-private"},
    )
    assert "_scheduling" not in ai._compatibility_payload(body)


def test_compatibility_scheduling_requires_durable_original_id(client, monkeypatch):
    http, headers = client
    monkeypatch.setattr(ai.text_task_service, "admission", object())
    response = http.post(
        "/v1/chat/completions", headers=headers,
        json={"model": "gpt-text", "messages": [{"role": "user", "content": "hello"}],
              "scheduling": {"workflow_id": "catalog-refresh"}},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "SCHEDULING_REQUEST_ID_REQUIRED"


def test_native_image_scheduling_is_normalized_for_json_and_multipart(client, monkeypatch):
    http, headers = client
    calls = []
    monkeypatch.setattr(image_tasks.image_task_service, "submit_generation",
                        lambda _identity, **kwargs: calls.append(kwargs) or {"id": kwargs["client_task_id"], "status": "queued", "data": []})
    generated = http.post(
        "/api/image-tasks/generations", headers=headers,
        json={"client_task_id": "scheduled-image", "prompt": "draw",
              "scheduling": {"workflow_id": "catalog-refresh"}},
    )
    assert generated.status_code == 200, generated.text
    assert calls[-1]["scheduling"] == {"workflow_id": "catalog-refresh", "workflow_concurrency": 1}

    monkeypatch.setattr(image_tasks.image_task_service, "submit_edit",
                        lambda _identity, **kwargs: calls.append(kwargs) or {"id": kwargs["client_task_id"], "status": "queued", "data": []})
    edited = http.post(
        "/api/image-tasks/edits", headers=headers,
        data={"client_task_id": "scheduled-edit", "prompt": "edit",
              "scheduling": '{"workflow_id":"catalog-refresh","workflow_concurrency":2}'},
        files={"image": ("source.png", PNG, "image/png")},
    )
    assert edited.status_code == 200, edited.text
    assert calls[-1]["scheduling"] == {"workflow_id": "catalog-refresh", "workflow_concurrency": 2}


def test_invalid_compatibility_scheduling_is_rejected_without_durable_submit(client):
    http, headers = client
    response = http.post(
        "/v1/chat/completions", headers={**headers, "X-Client-Request-ID": "bad-schedule"},
        json={"model": "gpt-text", "messages": [{"role": "user", "content": "hello"}],
              "scheduling": {"workflow_concurrency": 2}},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "SCHEDULING_INVALID"


def test_native_chat_scheduling_is_private_and_preserves_legacy_absence():
    from api.chat_requests import PublicChatRequest, _payload

    original = {"client_request_id": "scheduled-native", "model": "gpt-text",
                "messages": [{"role": "user", "content": "hello"}],
                "scheduling": {"workflow_id": "catalog-refresh"}}
    scheduled = _payload("owner", PublicChatRequest(**original), original["messages"])
    assert scheduled["_scheduling"] == {"workflow_id": "catalog-refresh", "workflow_concurrency": 1}
    legacy = _payload("owner", PublicChatRequest(**{k: v for k, v in original.items() if k != "scheduling"}), original["messages"])
    assert "_scheduling" not in legacy


def test_codex_scheduling_stays_private_in_durable_envelope(monkeypatch):
    app = FastAPI(); app.include_router(codex.create_router())
    client = TestClient(app)
    received = []

    async def accepted(identity, payload, request, protocol, **kwargs):
        received.append(durable_forward.envelope(identity, payload, request, protocol, **kwargs))
        return {"accepted": True}

    monkeypatch.setattr(codex, "_ordinary_identity", lambda _authorization: {"id": "owner", "role": "user"})
    monkeypatch.setattr(codex, "require_codex_policy", lambda *_args: None)
    monkeypatch.setattr(ai.text_task_service, "admission", object())
    monkeypatch.setattr("services.durable_forward.respond", accepted)
    response = client.post(
        "/codex/v1/responses", headers={"X-Client-Request-ID": "scheduled-codex"},
        json={"model": "gpt-5.6-codex", "input": "hello", "scheduling": {"workflow_id": "catalog-refresh"}},
    )
    assert response.status_code == 200, response.text
    assert received[0]["_scheduling"] == {"workflow_id": "catalog-refresh", "workflow_concurrency": 1}
    assert "scheduling" not in received[0]["_forward"]["payload"]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/chat/completions", {"model": "gpt-text", "messages": [{"role": "user", "content": "hello"}], "scheduling": None}),
        ("/v1/images/generations", {"prompt": "draw", "scheduling": None}),
    ],
)
def test_compatibility_rejects_explicit_null_scheduling(client, path, body):
    http, headers = client
    response = http.post(path, headers={**headers, "X-Client-Request-ID": "null-schedule"}, json=body)
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "SCHEDULING_INVALID"


def test_native_requests_reject_explicit_null_scheduling(client):
    http, headers = client
    chat = http.post("/api/image-tasks/generations", headers=headers,
                     json={"client_task_id": "null-schedule", "prompt": "draw", "scheduling": None})
    assert chat.status_code == 422


def test_public_chat_projection_keeps_normalized_scheduling_and_waiting_reason():
    from services.public_chat_service import project_public_chat_receipt
    result = project_public_chat_receipt({
        "request_id": "scheduled-public", "model": "gpt-text", "status": "queued",
        "scheduling": {"workflow_id": "catalog-refresh", "workflow_concurrency": 2},
        "waiting": {"reasons": ["not_before"], "next_check_at": 1700000000},
        "_requested_account_identity": "private-account",
    })
    assert result["scheduling"] == {"workflow_id": "catalog-refresh", "workflow_concurrency": 2}
    assert result["waiting"] == {"reasons": ["not_before"], "next_check_at": 1700000000}
    assert "private-account" not in repr(result)
