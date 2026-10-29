"""Real HTTP and durable-input boundaries; no external service calls."""
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import ai, task_results
from services.work_lifecycle import WorkLifecycleError
from test.test_category_directory_successor import install
from test.test_explicit_unknown_successor import task, row, update
from test.test_workbench_derived_input import payload

PREFIX = "/api/conversation-bindings/text-requests/old"


@pytest.fixture
def boundary(task, monkeypatch):
    h = install(task)
    identity = {"role": "admin", "id": "owner"}
    lane = {"source": "internal:listing", "company": False}

    def authenticate(_authorization, *, request=None):
        if request is not None and lane["company"]:
            request.state.company_identity = {"id": "company"}
        return identity

    monkeypatch.setattr(ai, "text_task_service", h.service)
    monkeypatch.setattr(task_results, "text_task_service", h.service)
    monkeypatch.setattr(ai, "require_identity", authenticate)
    monkeypatch.setattr(task_results, "require_identity", authenticate)
    monkeypatch.setattr(ai, "trusted_source", lambda *_: lane["source"])
    review = AsyncMock()
    monkeypatch.setattr(ai, "filter_or_log", review)
    app = FastAPI()
    app.include_router(ai.create_router())
    app.include_router(task_results.create_router())
    with TestClient(app) as client:
        yield h, identity, lane, review, client


def test_original_input_get_and_child_get_never_recover_or_submit(boundary, monkeypatch):
    h, identity, lane, review, client = boundary
    monkeypatch.setattr(h.service, "recover", lambda *_a, **_k: pytest.fail("implicit recovery"))
    response = client.get(PREFIX + "/retained-input")
    assert response.status_code == 200 and response.json()["input"] == h.body
    assert response.headers["cache-control"] == "private, no-store"
    request = payload(h.service, h.old)
    created = client.post(PREFIX + "/derived-successor", json=request)
    assert created.status_code == 200, created.text
    audit = created.json()["workbench_derived_input"]
    assert audit["original_input_hash"] == response.json()["input_hash"]
    assert audit["derivation_kind"] == request["derivation_kind"]
    assert created.json()["supersedes_request_id"] == "old"
    assert created.json()["status"] == "queued" and h.sent == []
    review.assert_awaited_once()
    before = deepcopy(row(h.service, "workbench-child"))
    read = client.get("/api/task-results/text/workbench-child")
    assert read.status_code == 200 and read.json()["receipt"]["workbench_derived_input"] == audit
    assert row(h.service, "workbench-child") == before and row(h.service) == h.old
    assert client.post(PREFIX + "/derived-successor", json=request).json() == created.json()
    review.assert_awaited_once()
    request["messages"][0]["content"][0]["text"] += " changed"
    assert client.post(PREFIX + "/derived-successor", json=request).status_code == 409
    review.assert_awaited_once()


@pytest.mark.parametrize("change", ["ordinary", "foreign", "wrong_lane", "company"])
def test_private_input_and_submit_authorization(boundary, change):
    h, identity, lane, review, client = boundary
    request = payload(h.service, h.old)
    if change == "ordinary": identity["role"] = "user"
    if change == "foreign": identity["id"] = "other-owner"
    if change == "wrong_lane": lane["source"] = "internal:content"
    if change == "company": lane["company"] = True
    assert client.get(PREFIX + "/retained-input").status_code == 404
    assert client.post(PREFIX + "/derived-successor", json=request).status_code == 404
    review.assert_not_awaited()
    assert row(h.service, "workbench-child") is None


@pytest.mark.parametrize("extra", [
    {"provider_binding_id": "other"}, {"provider_account_identity": "other"},
    {"conversation_id": "other"}, {"parent_message_id": "other"},
    {"product_work": {"product_ref": "other", "step_ref": "other"}},
    {"allow_unconfirmed_retry": True}, {"_workbench_derived_input": {}},
    {"source": "internal:listing"}, {"continue_after_no_final": True},
])
def test_route_accepts_business_content_not_transport_authority(boundary, extra):
    h, identity, lane, review, client = boundary
    request = {**payload(h.service, h.old), **extra}
    assert client.post(PREFIX + "/derived-successor", json=request).status_code == 422
    review.assert_not_awaited()
    assert row(h.service, "workbench-child") is None


def test_success_and_pause_rejected_before_review_and_rechecked_after_review(boundary):
    h, identity, lane, review, client = boundary
    request = payload(h.service, h.old)
    update(h.service, _recovery_paused=True)
    assert client.post(PREFIX + "/derived-successor", json=request).status_code == 409
    review.assert_not_awaited()
    update(h.service, _recovery_paused=False)

    async def late_result(*_args):
        update(h.service, status="succeeded", content="late original result")
    review.side_effect = late_result
    assert client.post(PREFIX + "/derived-successor", json=request).status_code == 409
    assert row(h.service, "workbench-child") is None and h.sent == []
    assert row(h.service)["content"] == "late original result"


def test_private_marker_cannot_enter_legacy_body(boundary):
    h, identity, lane, review, client = boundary
    body = {**h.body, "_workbench_derived_input": {"mode": "no_final"}}
    assert client.post("/api/conversation-bindings/text", json=body).status_code == 422
    review.assert_not_awaited()


@pytest.mark.parametrize("phase", ["validate_workbench_derived_input", "submit_workbench_derived_input"])
def test_product_work_rejection_keeps_its_explicit_http_reason(boundary, monkeypatch, phase):
    h, identity, lane, review, client = boundary
    request = payload(h.service, h.old)

    def closed_work(*_args, **_kwargs):
        raise WorkLifecycleError("WORK_NOT_ACTIVE")

    monkeypatch.setattr(h.service, phase, closed_work)
    response = client.post(PREFIX + "/derived-successor", json=request)
    assert response.status_code == 409
    assert response.json() == {"detail": {"code": "WORK_NOT_ACTIVE"}}
    assert row(h.service, "workbench-child") is None and h.sent == []
