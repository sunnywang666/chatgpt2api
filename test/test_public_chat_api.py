from __future__ import annotations

import base64
import asyncio
import io
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.testclient import TestClient
from PIL import Image

from api import ai, chat_requests
from api import external_images
from api.external_images import external_image_boundary
from services.account_service import AccountService
from services.auth_service import AuthService
from services.storage.json_storage import JSONStorageBackend
from services.text_task_service import (
    ContinuationExecutor,
    TextTaskCapacityError,
    TextTaskService,
)


class QueuedExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        self.calls.append((function, args))

    def run(self):
        function, args = self.calls.pop(0)
        function(*args)


@pytest.fixture
def public_chat(tmp_path, monkeypatch):
    auth = AuthService(JSONStorageBackend(tmp_path / "auth.json"))
    key_a, secret_a = auth.create_key(role="user", routes=["chat"], owner_subject="owner:a")
    key_b, secret_b = auth.create_key(role="user", routes=["chat"], owner_subject="owner:b")
    codex_key, codex_secret = auth.create_key(role="user", routes=["codex"], owner_subject="owner:c")
    _, admin_secret = auth.create_key(role="admin")
    queue = QueuedExecutor()
    upstream = Mock(return_value={
        "content": "public answer",
        "provider_binding_id": "binding-secret",
        "provider_account_identity": "account-secret",
        "conversation_id": "conversation-secret",
        "parent_message_id": "parent-secret",
        "binding_status": "bound",
    })
    tasks = TextTaskService(tmp_path / "chat.sqlite3", runner=upstream, executor=queue)
    monkeypatch.setattr("api.support.auth_service", auth)
    monkeypatch.setattr(chat_requests, "text_task_service", tasks)
    monkeypatch.setattr(chat_requests, "check_request", lambda *_: None)
    monkeypatch.setattr(
        "services.public_chat_service.model_catalog_service.route_for_model",
        lambda model: SimpleNamespace(
            account_types=frozenset({"Plus"}) if model in {"gpt-text", "gpt-text-2", "gpt-5-6-thinking"} else frozenset(),
            allow_anonymous=False,
        ),
    )
    monkeypatch.setattr("services.log_service.log_service.add", lambda *_args, **_kwargs: None)
    app = FastAPI()
    app.middleware("http")(external_image_boundary)
    app.include_router(chat_requests.create_router())
    app.include_router(ai.create_router())
    client = TestClient(app)

    def headers(secret=secret_a, *, public=True):
        value = {"Authorization": "Bearer " + secret}
        if public:
            value["X-Workbench-Image-Client"] = "1"
        return value

    return SimpleNamespace(
        auth=auth,
        client=client,
        headers=headers,
        key_a=key_a,
        key_b=key_b,
        codex_key=codex_key,
        secret_a=secret_a,
        secret_b=secret_b,
        codex_secret=codex_secret,
        admin_secret=admin_secret,
        queue=queue,
        upstream=upstream,
        tasks=tasks,
    )


def png_data_url() -> str:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), "blue").save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def test_completed_native_image_is_queryable_without_public_resubmission(public_chat):
    from services.conversation_binding_service import ConversationBindingService
    from test.test_non_text_recovery import image_document

    submitted = public_chat.client.post("/api/chat-requests", headers=public_chat.headers(), json=request_body())
    assert submitted.status_code == 202
    public_chat.tasks._update(
        public_chat.key_a["id"], "chat-1", status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
        provider_binding_id="private-binding", provider_account_identity="private-account",
        conversation_id="original-chat", request_parent_message_id="prior-answer",
    )
    public_chat.tasks.recovery_reader = Mock(side_effect=lambda receipt: ConversationBindingService._read_text_request_result(
        Mock(), receipt, document=image_document(receipt["request_message_id"]),
    ))
    result = public_chat.client.get("/api/chat-requests/chat-1", headers=public_chat.headers())
    assert result.status_code == 200
    receipt = result.json()
    assert receipt["status"] == "failed"
    assert receipt["error_code"] == "CHAT_RESPONSE_NOT_TEXT"
    assert receipt["result"] == {"type": "non_text", "artifact_type": "image", "artifact_count": 1}
    assert receipt["recovery"]["retryable"] is False
    assert receipt["recovery"]["requires_new_conversation"] is False
    assert receipt["recovery"]["upstream_outcome"] == "completed"
    assert result.headers["cache-control"] == "private, no-store"
    for method, path, body in (
        ("post", "/api/chat-requests/chat-1/recover", {}),
        ("post", "/api/chat-requests", request_body()),
        ("get", "/api/chat-requests/chat-1", None),
    ):
        response = public_chat.client.request(method, path, headers=public_chat.headers(), **({"json": body} if body is not None else {}))
        assert response.status_code == 200
        assert response.json() == receipt
    assert public_chat.client.get("/api/chat-requests/chat-1", headers=public_chat.headers(public_chat.secret_b)).status_code == 404
    assert public_chat.client.post("/api/chat-requests/chat-1/recover", headers=public_chat.headers(public_chat.secret_b), json={}).status_code == 404
    conflict = public_chat.client.post("/api/chat-requests", headers=public_chat.headers(), json=request_body(text="changed"))
    assert conflict.status_code == 409
    assert public_chat.tasks.recovery_reader.call_count == 1
    assert len(public_chat.queue.calls) == 1
    assert public_chat.upstream.call_count == 0
    assert not any(secret in result.text for secret in ("private-binding", "private-account", "original-chat", "file-service://"))


def request_body(request_id="chat-1", *, text="describe", model="gpt-text"):
    return {
        "client_request_id": request_id,
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": png_data_url()},
            ],
        }],
    }


def test_public_chat_high_reaches_runner_without_changing_legacy_payload(public_chat):
    legacy = request_body(request_id="legacy")
    assert public_chat.client.post("/api/chat-requests", headers=public_chat.headers(), json=legacy).status_code == 202
    legacy_payload = public_chat.queue.calls[0][1][2]
    assert "thinking_effort" not in legacy_payload
    assert set(legacy_payload) == {
        "client_request_id", "model", "messages", "client_conversation_id",
        "_text_only_binding", "_public_route",
        "_request_message_id",
    }
    high = request_body(request_id="high", model="gpt-5-6-thinking")
    high["reasoning_effort"] = "high"
    assert public_chat.client.post("/api/chat-requests", headers=public_chat.headers(), json=high).status_code == 202
    assert public_chat.queue.calls[1][1][2]["thinking_effort"] == "high"
    public_chat.queue.run()
    public_chat.queue.run()
    assert "thinking_effort" not in public_chat.upstream.call_args_list[0].args[0]
    assert public_chat.upstream.call_args_list[1].args[0]["thinking_effort"] == "high"
    assert public_chat.client.post("/api/chat-requests", headers=public_chat.headers(), json=high).status_code == 200
    assert public_chat.upstream.call_count == 2
    assert public_chat.client.get("/api/chat-requests/high", headers=public_chat.headers(public_chat.secret_b)).status_code == 404


@pytest.mark.parametrize("first_high", [False, True])
def test_public_chat_reasoning_drift_conflicts_in_both_directions(public_chat, first_high):
    body = request_body(request_id="effort-drift", model="gpt-5-6-thinking")
    if first_high:
        body["reasoning_effort"] = "high"
    assert public_chat.client.post("/api/chat-requests", headers=public_chat.headers(), json=body).status_code == 202
    if first_high:
        del body["reasoning_effort"]
    else:
        body["reasoning_effort"] = "high"
    conflict = public_chat.client.post("/api/chat-requests", headers=public_chat.headers(), json=body)
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "CHAT_REQUEST_CONFLICT"
    assert len(public_chat.queue.calls) == 1


@pytest.mark.parametrize("extra", [
    {"reasoning_effort": None}, {"reasoning_effort": "low"},
    {"reasoning_effort": "extended"}, {"reasoning_effort": True},
    {"reasoning_effort": "high", "tools": []},
    {"reasoning_effort": "high", "stream": False},
])
def test_public_chat_rejects_invalid_reasoning_and_generic_options(public_chat, extra):
    body = request_body(model="gpt-5-6-thinking")
    body.update(extra)
    assert public_chat.client.post("/api/chat-requests", headers=public_chat.headers(), json=body).status_code == 422
    assert public_chat.queue.calls == []


def test_public_chat_high_requires_catalogued_support(public_chat):
    body = request_body()
    body["reasoning_effort"] = "high"
    result = public_chat.client.post("/api/chat-requests", headers=public_chat.headers(), json=body)
    assert result.status_code == 400
    assert result.json()["detail"]["code"] == "CHAT_REASONING_UNSUPPORTED"
    assert public_chat.queue.calls == []


def test_durable_public_chat_submit_query_reuse_conflict_and_owner_isolation(public_chat):
    first = public_chat.client.post(
        "/api/chat-requests", headers=public_chat.headers(), json=request_body(),
    )
    assert first.status_code == 202, first.text
    assert first.json()["status"] == "queued"
    assert set(first.json()) <= {
        "request_id", "route", "model", "status", "content", "error_code",
        "created_at", "updated_at", "started_at", "finished_at", "recovery",
    }
    assert "secret" not in first.text
    assert len(public_chat.queue.calls) == 1

    public_chat.queue.run()
    completed = public_chat.client.get(
        "/api/chat-requests/chat-1", headers=public_chat.headers(),
    )
    assert completed.status_code == 200
    assert completed.json()["content"] == "public answer"
    assert "secret" not in completed.text

    repeated = public_chat.client.post(
        "/api/chat-requests", headers=public_chat.headers(), json=request_body(),
    )
    assert repeated.status_code == 200
    assert repeated.json()["content"] == "public answer"
    assert public_chat.upstream.call_count == 1
    assert public_chat.queue.calls == []

    drift = public_chat.client.post(
        "/api/chat-requests", headers=public_chat.headers(), json=request_body(text="different"),
    )
    assert drift.status_code == 409
    assert drift.json()["detail"]["code"] == "CHAT_REQUEST_CONFLICT"
    assert public_chat.client.get(
        "/api/chat-requests/chat-1", headers=public_chat.headers(public_chat.secret_b),
    ).status_code == 404


def test_public_chat_never_resubmits_cross_process_not_started_receipt(public_chat, monkeypatch):
    body = request_body(request_id="cross-process")
    assert public_chat.client.post(
        "/api/chat-requests", headers=public_chat.headers(), json=body,
    ).status_code == 202
    assert len(public_chat.queue.calls) == 1

    second_queue = QueuedExecutor()
    restarted = TextTaskService(
        public_chat.tasks.path,
        runner=public_chat.upstream,
        executor=second_queue,
    )
    observed = restarted.read(public_chat.key_a["id"], "cross-process")
    assert observed["status"] == "not_started"
    monkeypatch.setattr(chat_requests, "text_task_service", restarted)

    repeated = public_chat.client.post(
        "/api/chat-requests", headers=public_chat.headers(), json=body,
    )
    assert repeated.status_code == 202
    assert repeated.json()["status"] == "not_started"
    assert second_queue.calls == []
    public_chat.queue.run()
    public_chat.upstream.assert_not_called()


def test_text_task_service_atomically_blocks_public_not_started_resubmit(tmp_path):
    first_queue = QueuedExecutor()
    second_queue = QueuedExecutor()
    upstream = Mock(return_value={"content": "must not run"})
    path = tmp_path / "public-toctou.sqlite3"
    body = {
        "client_request_id": "atomic-public",
        "client_conversation_id": "server-owned",
        "model": "gpt-text",
        "messages": [{"role": "user", "content": "hello"}],
        "_public_route": "chat",
        "_text_only_binding": True,
    }
    original = TextTaskService(path, runner=upstream, executor=first_queue)
    assert original.submit("owner", body)["status"] == "queued"
    assert len(first_queue.calls) == 1

    restarted = TextTaskService(path, runner=upstream, executor=second_queue)
    assert restarted.read("owner", "atomic-public")["status"] == "not_started"
    # This calls the atomic service method directly, covering a
    # validate_submission -> submit interleaving in the HTTP layer.
    result = restarted.submit("owner", body)
    assert result["status"] == "not_started"
    assert second_queue.calls == []

    first_queue.run()
    upstream.assert_not_called()


def test_public_chat_auth_policy_admin_and_revocation(public_chat):
    assert public_chat.client.post("/api/chat-requests", json=request_body()).status_code == 401
    assert public_chat.client.post(
        "/api/chat-requests",
        headers=public_chat.headers(public_chat.admin_secret),
        json=request_body(),
    ).status_code == 403
    denied = public_chat.client.post(
        "/api/chat-requests",
        headers=public_chat.headers(public_chat.codex_secret),
        json=request_body(),
    )
    assert denied.status_code == 403
    assert public_chat.queue.calls == []

    assert public_chat.client.post(
        "/api/chat-requests", headers=public_chat.headers(), json=request_body(),
    ).status_code == 202
    public_chat.queue.run()
    public_chat.auth.update_owned_policy("owner:a", public_chat.key_a["id"], ["codex"], 1)
    assert public_chat.client.get(
        "/api/chat-requests/chat-1", headers=public_chat.headers(),
    ).status_code == 200
    public_chat.auth.revoke_owned_key("owner:a", public_chat.key_a["id"])
    assert public_chat.client.get(
        "/api/chat-requests/chat-1", headers=public_chat.headers(),
    ).status_code == 401


@pytest.mark.parametrize("part,code", [
    ({"type": "image_url", "image_url": "https://example.test/image.png"}, "REMOTE_IMAGE_URL_NOT_SUPPORTED"),
    ({"type": "tool_call", "name": "unsafe"}, "CHAT_OPTION_UNSUPPORTED"),
])
def test_public_chat_rejects_remote_images_and_unsupported_parts_before_submit(public_chat, part, code):
    body = request_body()
    body["messages"][0]["content"] = [{"type": "text", "text": "describe"}, part]
    result = public_chat.client.post(
        "/api/chat-requests", headers=public_chat.headers(), json=body,
    )
    assert result.status_code == 400
    assert result.json()["detail"]["code"] == code
    assert public_chat.queue.calls == []
    public_chat.upstream.assert_not_called()


def test_public_chat_model_and_recovery_contract(public_chat):
    unsupported = public_chat.client.post(
        "/api/chat-requests",
        headers=public_chat.headers(),
        json=request_body(model="gpt-image-2"),
    )
    assert unsupported.status_code == 400
    assert unsupported.json()["detail"]["code"] == "CHAT_MODEL_UNSUPPORTED"

    public_chat.client.post(
        "/api/chat-requests", headers=public_chat.headers(), json=request_body(),
    )
    public_chat.queue.run()
    for kwargs in ({}, {"json": {}}):
        result = public_chat.client.post(
            "/api/chat-requests/chat-1/recover",
            headers=public_chat.headers(),
            **kwargs,
        )
        assert result.status_code == 200, result.text
        assert result.json()["content"] == "public answer"
    assert public_chat.client.post(
        "/api/chat-requests/chat-1/recover",
        headers=public_chat.headers(),
        json={"retry": True},
    ).status_code == 422
    assert public_chat.upstream.call_count == 1


def test_public_chat_enforces_body_text_and_image_magic_limits_before_submit(public_chat):
    headers = public_chat.headers()
    headers["Content-Length"] = str(140 * 1024 * 1024 + 1)
    too_large = public_chat.client.post(
        "/api/chat-requests", headers=headers, json=request_body(),
    )
    assert too_large.status_code == 413
    assert too_large.json()["detail"]["code"] == "CHAT_REQUEST_TOO_LARGE"

    text_body = request_body()
    text_body["messages"] = [{"role": "user", "content": "x" * (1024 * 1024 + 1)}]
    text = public_chat.client.post(
        "/api/chat-requests", headers=public_chat.headers(), json=text_body,
    )
    assert text.status_code == 400
    assert "1MiB" in text.text

    invalid_image = request_body()
    invalid_image["messages"][0]["content"][1]["image_url"] = (
        "data:image/png;base64," + base64.b64encode(b"not a png").decode()
    )
    image = public_chat.client.post(
        "/api/chat-requests", headers=public_chat.headers(), json=invalid_image,
    )
    assert image.status_code == 400
    assert "valid raster" in image.text
    assert public_chat.queue.calls == []


def _asgi_request(path, *, authorization="Bearer valid", body=b"{}"):
    consumed = {"value": False}
    delivered = {"value": False}

    async def receive():
        consumed["value"] = True
        if not delivered["value"]:
            delivered["value"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    headers = [(b"authorization", authorization.encode()), (b"content-length", str(len(body)).encode())]
    request = Request({
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("test", 1),
        "server": ("test", 80),
    }, receive)
    return request, consumed


def test_public_chat_boundary_authenticates_before_consuming_body(monkeypatch):
    request, consumed = _asgi_request("/api/chat-requests", body=b"not-json")

    def reject(_authorization, **_kwargs):
        raise HTTPException(401, detail={"code": "INVALID_API_KEY"})

    async def downstream(_request):
        raise AssertionError("unauthorized request reached the application")

    monkeypatch.setattr(external_images, "require_identity", reject)
    response = asyncio.run(external_image_boundary(request, downstream))

    assert response.status_code == 401
    assert json.loads(response.body)["detail"]["code"] == "INVALID_API_KEY"
    assert consumed["value"] is False


def test_public_chat_body_reader_capacity_is_nonblocking_and_released(monkeypatch):
    monkeypatch.setattr(external_images, "require_identity", lambda _authorization, **_kwargs: {"role": "user"})
    with external_images._public_chat_body_reader_lock:
        external_images._public_chat_body_readers = (
            external_images.MAX_CONCURRENT_PUBLIC_CHAT_BODY_READERS
        )
    saturated, consumed = _asgi_request("/api/chat-requests", body=b"not-json")
    try:
        response = asyncio.run(external_image_boundary(saturated, lambda _request: None))
        assert response.status_code == 429
        assert json.loads(response.body)["detail"]["code"] == "CHAT_BODY_READER_CAPACITY_EXCEEDED"
        assert consumed["value"] is False
    finally:
        with external_images._public_chat_body_reader_lock:
            external_images._public_chat_body_readers = 0

    async def succeeds(_request):
        return Response(status_code=204)

    successful, _ = _asgi_request("/api/chat-requests")
    assert asyncio.run(external_image_boundary(successful, succeeds)).status_code == 204
    assert external_images._public_chat_body_readers == 0

    async def fails(_request):
        raise RuntimeError("downstream failed")

    failed, _ = _asgi_request("/api/chat-requests")
    with pytest.raises(RuntimeError, match="downstream failed"):
        asyncio.run(external_image_boundary(failed, fails))
    assert external_images._public_chat_body_readers == 0


def test_continuation_executor_bounds_and_releases_capacity():
    executor = ContinuationExecutor(
        max_workers=1,
        max_outstanding=1,
        max_retained_bytes=1024 * 1024,
    )
    entered = threading.Event()
    release = threading.Event()

    def blocked(_body):
        entered.set()
        assert release.wait(2)
        return "done"

    try:
        first = executor.submit(blocked, {})
        assert entered.wait(2)
        with pytest.raises(TextTaskCapacityError):
            executor.submit(lambda _body: None, {})
        assert executor.outstanding == 1
        release.set()
        assert first.result(timeout=2) == "done"
        assert executor.outstanding == 0
        assert executor.retained_bytes == 0

        def fails(_body):
            raise RuntimeError("runner failed")

        failed = executor.submit(fails, {})
        with pytest.raises(RuntimeError, match="runner failed"):
            failed.result(timeout=2)
        assert executor.outstanding == 0
        assert executor.retained_bytes == 0

        with pytest.raises(TextTaskCapacityError):
            executor.submit(lambda _body: None, {"blob": b"x" * (2 * 1024 * 1024)})
        assert executor.outstanding == 0
        assert executor.retained_bytes == 0
    finally:
        release.set()
        executor.shutdown()


def test_public_chat_queue_overflow_is_durable_and_never_duplicates(public_chat, monkeypatch):
    class CapacityThenQueue:
        def __init__(self):
            self.calls = 0
            self.queued = []

        def submit(self, function, *args):
            self.calls += 1
            if self.calls == 1:
                raise TextTaskCapacityError("full")
            self.queued.append((function, args))

        def run(self):
            function, args = self.queued.pop(0)
            function(*args)

    capacity = CapacityThenQueue()
    tasks = TextTaskService(
        public_chat.tasks.path.parent / "capacity.sqlite3",
        runner=public_chat.upstream,
        executor=capacity,
    )
    monkeypatch.setattr(chat_requests, "text_task_service", tasks)
    body = request_body(request_id="capacity")

    first = public_chat.client.post(
        "/api/chat-requests", headers=public_chat.headers(), json=body,
    )
    assert first.status_code == 429
    assert first.json()["status"] == "failed"
    assert first.json()["error_code"] == "TEXT_TASK_CAPACITY_EXCEEDED"
    assert first.json()["recovery"]["upstream_outcome"] == "not_sent"
    assert capacity.calls == 1
    public_chat.upstream.assert_not_called()

    query = public_chat.client.get(
        "/api/chat-requests/capacity", headers=public_chat.headers(),
    )
    assert query.status_code == 200
    assert query.json()["error_code"] == "TEXT_TASK_CAPACITY_EXCEEDED"
    repeated = public_chat.client.post(
        "/api/chat-requests", headers=public_chat.headers(), json=body,
    )
    assert repeated.status_code == 202
    assert repeated.json()["status"] == "queued"
    assert capacity.calls == 2
    # A concurrent/repeated POST observes the one accepted queued receipt and
    # cannot enqueue a second copy.
    assert public_chat.client.post(
        "/api/chat-requests", headers=public_chat.headers(), json=body,
    ).status_code == 202
    assert capacity.calls == 2
    public_chat.upstream.assert_not_called()
    capacity.run()
    assert public_chat.upstream.call_count == 1
    assert tasks.read(public_chat.key_a["id"], "capacity")["status"] == "succeeded"


@pytest.mark.parametrize("request_id", [".", ".."])
def test_public_chat_rejects_dot_segment_ids_in_body_and_path(public_chat, request_id):
    body_result = public_chat.client.post(
        "/api/chat-requests", headers=public_chat.headers(), json=request_body(request_id=request_id),
    )
    assert body_result.status_code == 422
    encoded = "%2E" if request_id == "." else "%2E%2E"
    path_result = public_chat.client.get(
        f"/api/chat-requests/{encoded}", headers=public_chat.headers(),
    )
    assert path_result.status_code == 400
    assert path_result.json()["detail"]["code"] == "CHAT_REQUEST_ID_INVALID"
    with pytest.raises(HTTPException) as invalid_path:
        chat_requests._validated_request_id(request_id)
    assert invalid_path.value.status_code == 400
    assert invalid_path.value.detail["code"] == "CHAT_REQUEST_ID_INVALID"
    assert public_chat.queue.calls == []


def test_text_only_binding_uses_existing_pool_without_image_quota(tmp_path, monkeypatch):
    accounts = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    accounts.add_account_items([{
        "access_token": "paid-text-token",
        "type": "Plus",
        "source_type": "web",
        "status": "正常",
        "quota": 0,
    }])
    monkeypatch.setattr(
        "services.model_service.model_catalog_service.route_for_model",
        lambda _model: SimpleNamespace(account_types=frozenset({"Plus"})),
    )
    monkeypatch.setattr(accounts, "refresh_access_token", lambda token, **_kwargs: token)
    monkeypatch.setattr(
        accounts,
        "get_available_access_token",
        Mock(side_effect=AssertionError("text binding must not acquire image quota")),
    )

    binding, account_identity = accounts.create_text_conversation_binding(text_model="gpt-text")

    assert binding.startswith("cb_")
    assert account_identity.startswith("account_")
    assert accounts.get_bound_account_identity(binding) == account_identity
    assert accounts._image_inflight == {}


def test_public_model_discovery_describes_text_image_input_and_generation(public_chat, monkeypatch):
    catalogue = {
        "object": "list",
        "data": [
            {"id": "gpt-image-2", "object": "model"},
            {"id": "gpt-text", "object": "model"},
            {"id": "gpt-5-6-thinking", "object": "model"},
            {"id": "unavailable", "object": "model"},
            {"id": "codex-gpt-image-2", "object": "model"},
        ],
    }
    monkeypatch.setattr(ai.openai_v1_models, "list_models", lambda: catalogue)
    public = public_chat.client.get("/v1/models", headers=public_chat.headers())
    assert public.status_code == 200, public.text
    by_id = {item["id"]: item for item in public.json()["data"]}
    assert set(by_id) == {"gpt-image-2", "gpt-text", "gpt-5-6-thinking"}
    assert by_id["gpt-5-6-thinking"]["reasoning_efforts"] == ["high"]
    assert "reasoning_efforts" not in by_id["gpt-text"]
    assert by_id["gpt-text"]["capabilities"] == ["text", "image_input"]
    assert by_id["gpt-text"]["input_limits"]["max_text_bytes"] == 1024 * 1024
    assert by_id["gpt-image-2"]["capabilities"] == ["image_generation", "image_edit"]

    internal = public_chat.client.get("/v1/models", headers=public_chat.headers(public=False))
    assert internal.json() == catalogue
