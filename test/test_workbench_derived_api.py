"""Real HTTP and durable-input boundaries; no external service calls."""
from copy import deepcopy
import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import ai, task_results
from services.work_lifecycle import WorkLifecycleError
from test.test_category_directory_successor import install
from test.test_explicit_unknown_successor import task, row, update
from test.test_workbench_derived_input import payload
from test.test_advanced_session_successor import advanced
from test.test_no_final_successor import product_no_final

PREFIX = "/api/conversation-bindings/text-requests/old"


@pytest.fixture
def advanced_http(advanced, monkeypatch):
    h = advanced
    monkeypatch.setattr(ai, "text_task_service", h.service)
    monkeypatch.setattr(task_results, "text_task_service", h.service)
    identity = lambda *_a, **_k: {"role": "admin", "id": "owner"}
    monkeypatch.setattr(ai, "require_identity", identity)
    monkeypatch.setattr(task_results, "require_identity", identity)
    monkeypatch.setattr(ai, "trusted_source", lambda *_a: "internal:listing")
    review = AsyncMock()
    monkeypatch.setattr(ai, "filter_or_log", review)
    app = FastAPI()
    app.include_router(ai.create_router())
    app.include_router(task_results.create_router())
    with TestClient(app) as client:
        yield h, review, client


def test_advanced_wire_idempotency_and_saved_result_read(advanced_http):
    h, review, client = advanced_http
    before = deepcopy(row(h.service))
    created = client.post(PREFIX + "/derived-successor", json=h.payload)
    assert created.status_code == 200, created.text
    first = created.json()
    assert first["workbench_derived_input"]["mode"] == "advanced_new_session"
    proof = first["advanced_session"]
    assert proof["context_ref"] == h.payload["context_ref"]
    assert proof["original_request_id"] == "old" and proof["child_request_id"] == "isolated"
    assert proof["provider_binding_id"] is None and proof["conversation_id"] is None
    assert proof["product_work"] == h.old["_product_work"]
    assert client.post(PREFIX + "/derived-successor", json=h.payload).json() == first
    result = client.get("/api/task-results/text/isolated").json()
    assert result["result_ready"] is False and result["receipt"]["advanced_session"] == proof
    assert row(h.service) == before and not h.sent
    review.assert_awaited_once()


@pytest.mark.parametrize("mutation", ["missing", "null", "foreign_field", "malformed_hash", "wrong_mode"])
def test_advanced_context_schema_fails_before_review(advanced_http, mutation):
    h, review, client = advanced_http
    request = deepcopy(h.payload)
    if mutation == "missing": request.pop("context_ref")
    elif mutation == "null": request["context_ref"] = None
    elif mutation == "foreign_field": request["context_ref"]["conversation_id"] = "caller-cursor"
    elif mutation == "malformed_hash": request["context_ref"]["result_sha256"] = "md5-not-sha256"
    else: request["mode"] = "incomplete"
    assert client.post(PREFIX + "/derived-successor", json=request).status_code == 422
    assert row(h.service, "isolated") is None
    review.assert_not_awaited()


def test_advanced_actual_wire_with_scheduling_and_multiple_external_turns(advanced_http, tmp_path):
    """Export actual HTTP serialization for the paired Workbench contract test."""
    h, review, client = advanced_http
    # ProductWork and nonempty legacy workflow scheduling are mutually exclusive.
    # Preserve the real admission guard; this is the supported non-null wire shape.
    scheduling = {}
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        body = h.service.store.load_input(original["_input_ref"])
        body["_scheduling"] = scheduling
        original["_scheduling"] = scheduling
        original["_input_ref"] = h.service.store.save_input(body)
        original["_completed_successor_evidence"]["tail_steps"] = [
            {"request_message_id": "external-user-2", "final_message_id": "external-final-2"},
            {"request_message_id": "external-user-3", "final_message_id": "external-final-3"}]
        digest = h.service._submission_identity("owner", body)[1]
        db.execute("UPDATE requests SET request_hash=?,receipt=? WHERE owner='owner' AND id='old'",
                   (digest, json.dumps(original)))
    h.payload["original_input_hash"] = digest
    queued_response = client.post(PREFIX + "/derived-successor", json=h.payload)
    assert queued_response.status_code == 200, queued_response.text
    queued = queued_response.json()
    assert queued["advanced_session"]["scheduling"] == scheduling
    assert queued["advanced_session"]["original_head_message_id"] == "external-final-3"
    assert queued["advanced_session"]["provider_binding_id"] is None
    assert queued["advanced_session"]["conversation_id"] is None
    with h.service.store.transaction() as db:
        receipt = h.service.store.read_receipt(db, "text", "owner", "isolated")
        receipt.update(status="succeeded", content="fixture complete business result", _submission_started=True,
            _upstream_terminal=True, _executing=False, _turn_reserved=False,
            provider_binding_id="fixture-new-binding", conversation_id="fixture-new-chat",
            request_message_id="fixture-new-user", parent_message_id="fixture-new-final",
            request_parent_message_id="fixture-new-root", _submission_parent_message_id="fixture-new-root")
        h.service.store.write_receipt(db, "text", "owner", "isolated", receipt)
    result_response = client.get("/api/task-results/text/isolated")
    assert result_response.status_code == 200
    result = result_response.json()
    assert result["result_ready"] and result["receipt"]["advanced_session"]["conversation_id"] == "fixture-new-chat"
    assert result["receipt"]["parent_message_id"] == "fixture-new-final"
    for key in set(queued["advanced_session"]) - {"provider_binding_id", "conversation_id"}:
        assert queued["advanced_session"][key] == result["receipt"]["advanced_session"][key]
    (tmp_path / "advanced-queued.json").write_text(json.dumps(queued, ensure_ascii=False, indent=2) + "\n")
    (tmp_path / "advanced-success-task-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    assert not h.sent


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
