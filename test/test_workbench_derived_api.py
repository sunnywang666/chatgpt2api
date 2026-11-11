"""Real HTTP and durable-input boundaries; no external service calls."""
from copy import deepcopy
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import ai, task_results
from services.work_lifecycle import WorkLifecycleError
from services.text_task_service import TextTaskService
from services.product_work import _key
from test.test_category_directory_successor import install
from test.test_explicit_unknown_successor import task, row, update
from test.test_workbench_derived_input import payload, install_invalid_image, invalid_image_payload, valid_png
from test.test_incomplete_compact_successor import install_incomplete
from test.test_advanced_session_successor import advanced
from test.test_no_final_successor import product_no_final
from test.test_no_final_successor import product_lifecycle
from test.test_pool_admission import build

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


@pytest.fixture
def incomplete_http(task, monkeypatch):
    h = install_incomplete(task)
    h.payload = payload(h.service, h.old, request_id="successor", mode="incomplete")
    monkeypatch.setattr(ai, "text_task_service", h.service)
    monkeypatch.setattr(task_results, "text_task_service", h.service)
    identity = {"role": "admin", "id": "owner"}
    monkeypatch.setattr(ai, "require_identity", lambda *_a, **_k: identity)
    monkeypatch.setattr(task_results, "require_identity", lambda *_a, **_k: identity)
    monkeypatch.setattr(ai, "trusted_source", lambda *_a: "internal:listing")
    review = AsyncMock()
    monkeypatch.setattr(ai, "filter_or_log", review)
    app = FastAPI()
    app.include_router(ai.create_router())
    app.include_router(task_results.create_router())
    with TestClient(app) as client:
        yield h, review, client, identity


INPUT_PARENT = "566459da-c92c-4d1c-9f8f-97991b2f8abf"
SENT_PARENT = "94f12781-769f-4249-9719-4b1a72601a99"
FINAL_CURSOR = "3b460b3e-ca3f-49c4-b6c6-8e975188112f"
INPUT_PARENT_FIELD = "workbench_derived_input_parent_message_id"


def split_attribute_result(h, client):
    """Synthetic incident shape, with real durable acceptance and HTTP reads."""
    h.body["parent_message_id"] = INPUT_PARENT
    update(h.service, parent_message_id=INPUT_PARENT,
           request_parent_message_id=INPUT_PARENT, _submission_parent_message_id=INPUT_PARENT)
    install_incomplete(h)
    h.payload = {**payload(h.service, h.old, request_id="successor", mode="incomplete"),
                 "derivation_kind": "split_attributes_v1"}
    submitted = client.post(PREFIX + "/derived-successor", json=h.payload)
    assert submitted.status_code == 200, submitted.text
    assert INPUT_PARENT_FIELD not in submitted.json()
    update(h.service, "successor", status="succeeded", content="synthetic attribute result",
           request_parent_message_id=SENT_PARENT, _submission_parent_message_id=INPUT_PARENT,
           parent_message_id=FINAL_CURSOR, _submission_started=True, _executing=False,
           _turn_reserved=False, _upstream_terminal=True)


def test_invalid_image_successor_http_never_reviews_or_accepts_caller_content(task, monkeypatch, tmp_path):
    h = install_invalid_image(task)
    monkeypatch.setattr(ai, "text_task_service", h.service)
    monkeypatch.setattr(task_results, "text_task_service", h.service)
    monkeypatch.setattr(ai, "require_identity", lambda *_a, **_k: {"role": "admin", "id": "owner"})
    monkeypatch.setattr(task_results, "require_identity", lambda *_a, **_k: {"role": "admin", "id": "owner"})
    monkeypatch.setattr(ai, "trusted_source", lambda *_a: "internal:listing")
    review = AsyncMock()
    monkeypatch.setattr(ai, "filter_or_log", review)
    request = invalid_image_payload(h.service)
    app = FastAPI(); app.include_router(ai.create_router()); app.include_router(task_results.create_router())
    with TestClient(app) as client:
        malformed = client.post(PREFIX + "/derived-successor", json={
            **request, "client_request_id": "bool-index", "omit_image_parts": [{"message_index": True, "part_index": 6}],
        })
        submitted = client.post(PREFIX + "/derived-successor", json=request)
        accepted = client.post(PREFIX + "/derived-successor/acceptance", json=request)
    assert submitted.status_code == accepted.status_code == 200
    assert malformed.status_code == 422
    assert review.await_count == 0
    assert submitted.json()["workbench_derived_input"]["omit_image_parts"] == [{"message_index": 0, "part_index": 6}]
    assert accepted.json()["request_id"] == "image-child"
    output_dir = Path(os.environ.get("INVALID_IMAGE_SUCCESSOR_FIXTURE_DIR", tmp_path))
    for name, value in (("invalid-image-successor-submit", submitted.json()),
                        ("invalid-image-successor-acceptance", accepted.json()),
                        ("invalid-image-request", request)):
        (output_dir / (name + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def listing_invalid_product(h):
    """Turn the existing ProductWork fixture into a typed listing image failure."""
    svg = "data:image/svg+xml;base64,PHN2Zy8+"
    body = h.service.store.load_input(h.old["_input_ref"])
    body["messages"] = [{"role": "user", "content": [
        {"type": "text", "text": "frozen listing prompt"},
        *[{"type": "image_url", "image_url": {"url": valid_png()}} for _ in range(5)],
        {"type": "image_url", "image_url": {"url": svg}},
    ]}]
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        old_key = original["_product_work_key"]
        parent = h.service.store.runtime(db, old_key)
        key = _key("internal:listing", "owner", parent["product_ref"])
        parent.update(key=key, source="internal:listing")
        h.service.store.set_runtime(db, key, parent)
        db.execute("DELETE FROM task_runtime WHERE name=?", (old_key,))
        original.update(_source="internal:listing", _product_work_key=key,
                        _input_ref=h.service.store.save_input(body), status="failed",
                        error_code="INVALID_IMAGE_INPUT", upstream_outcome="not_sent",
                        _submission_started=False, _executing=False, _turn_reserved=False,
                        upstream_unfinished=False, recovery_retryable=False,
                        recovery_requires_new_conversation=False, recovery_next_at=None,
                        original_failure_phase="image_input_validation",
                        original_exception_category="invalid_image_input", _last_sent_sequence=None,
                        send_count=0, _execution_timeline=[{"stage": "accepted"}, {"stage": "execution_claimed"},
                                                           {"stage": "pre_submit_failure"}, {"stage": "task_finished"}])
        original.pop("_scheduling_owner", None)
        h.service.store.write_receipt(db, "text", "owner", "old", original)
        request_hash = TextTaskService._submission_identity("owner", body)[1]
        db.execute("UPDATE requests SET request_hash=? WHERE owner=? AND id='old'", (request_hash, "owner"))
    h.old = row(h.service)
    return {"client_request_id": "image-product-child", "original_input_hash": request_hash,
            "mode": "invalid_image_input", "derivation_kind": "omit_invalid_inline_image_v1",
            "omit_image_parts": [{"message_index": 0, "part_index": 6}]}


@pytest.mark.parametrize("product_no_final", [{"organization_id": "org", "subject_id": "subject", "root_workflow_id": "root"}], indirect=True)
def test_invalid_image_product_http_acceptance_restart_and_result_fixture(product_no_final, monkeypatch, tmp_path):
    h = product_no_final
    request = listing_invalid_product(h)
    raster_calls = []
    raster_validator = TextTaskService._valid_inline_raster
    def track_raster(_cls, part):
        raster_calls.append(part)
        return raster_validator(part)
    monkeypatch.setattr(TextTaskService, "_valid_inline_raster", classmethod(track_raster))
    monkeypatch.setattr(ai, "text_task_service", h.service)
    monkeypatch.setattr(task_results, "text_task_service", h.service)
    monkeypatch.setattr(ai, "require_identity", lambda *_a, **_k: {"role": "admin", "id": "owner"})
    monkeypatch.setattr(task_results, "require_identity", lambda *_a, **_k: {"role": "admin", "id": "owner"})
    monkeypatch.setattr(ai, "trusted_source", lambda *_a: "internal:listing")
    review = AsyncMock(); monkeypatch.setattr(ai, "filter_or_log", review)
    app = FastAPI(); app.include_router(ai.create_router()); app.include_router(task_results.create_router())
    with TestClient(app) as client:
        absent = client.post(PREFIX + "/derived-successor/acceptance", json=request)
        bad_hash = client.post(PREFIX + "/derived-successor/acceptance", json={**request, "original_input_hash": "0" * 64})
        bad_selector = client.post(PREFIX + "/derived-successor", json={**request, "client_request_id": "bad-selector",
                                                                            "omit_image_parts": [{"message_index": 0, "part_index": 5}]})
        submitted = client.post(PREFIX + "/derived-successor", json=request)
        lost_response_replay = client.post(PREFIX + "/derived-successor", json=request)
    assert absent.status_code == 200 and absent.json()["state"] == "absent_at_observation"
    assert bad_hash.status_code == bad_selector.status_code == 409
    assert submitted.status_code == lost_response_replay.status_code == 200
    assert submitted.json() == lost_response_replay.json() and review.await_count == 0
    assert submitted.json()["product_work"] == {
        "protocol": "product-work-v1", "product_ref": "fixture-product", "step_ref": "title",
        "fairness": {"organization_id": "org", "subject_id": "subject", "root_workflow_id": "root"},
    }
    child = row(h.service, "image-product-child")
    assert child["_product_work"]["fairness"] == {"organization_id": "org", "subject_id": "subject", "root_workflow_id": "root"}
    assert child["_product_work"]["step_ref"] == "title"
    _unused, _store, restarted_admission = build(h.root, h.clock)
    restarted = TextTaskService(h.service.store.path, runner=h.service.runner, clock=h.clock, admission=restarted_admission)
    restarted_admission.register("text", lambda ctx, body: restarted._run(ctx.owner, ctx.request_id, body))
    monkeypatch.setattr(ai, "text_task_service", restarted)
    monkeypatch.setattr(task_results, "text_task_service", restarted)
    before = deepcopy(row(restarted, "image-product-child"))
    with TestClient(app) as client:
        restarted_acceptance = client.post(PREFIX + "/derived-successor/acceptance", json=request)
    assert restarted_acceptance.status_code == 200 and restarted_acceptance.json() == submitted.json()
    assert row(restarted, "image-product-child") == before
    context = restarted_admission.claim_next()
    assert context and context.request_id == "image-product-child"
    decoded_before_execute = len(raster_calls)
    restarted_admission.execute(context)
    assert len(raster_calls) == decoded_before_execute
    with TestClient(app) as client:
        successful = client.post(PREFIX + "/derived-successor/acceptance", json=request)
        result = client.get("/api/task-results/text/image-product-child")
    assert successful.status_code == result.status_code == 200
    assert successful.json()["status"] == "succeeded" and result.json()["result_ready"] is True
    decoded_before_close = len(raster_calls)
    product, lifecycle, calls, identity = product_lifecycle(h)
    identity["_trusted_source"] = "internal:listing"
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    assert lifecycle.process_one() and calls == [("owner", "image-product-child", True)]
    assert len(raster_calls) == decoded_before_close
    output_dir = Path(os.environ.get("INVALID_IMAGE_SUCCESSOR_FIXTURE_DIR", tmp_path))
    for name, value in (("invalid-image-successor-product-absent", absent.json()),
                        ("invalid-image-successor-product-submit", submitted.json()),
                        ("invalid-image-successor-product-restart-acceptance", restarted_acceptance.json()),
                        ("invalid-image-successor-product-success-acceptance", successful.json()),
                        ("invalid-image-successor-product-task-result", result.json())):
        (output_dir / (name + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@pytest.mark.parametrize("second_observation", [False, True])
def test_split_attributes_acceptance_attests_input_parent_without_changing_actual_cursor(
        incomplete_http, monkeypatch, tmp_path, second_observation):
    h, review, client, _identity = incomplete_http
    split_attribute_result(h, client)
    review.reset_mock()
    h.service.recovery_reader = lambda *_a: pytest.fail("acceptance read upstream")
    if second_observation:
        # A child may appear after the initial local observation; the second
        # owner/hash/full-body check must provide the same attestation.
        monkeypatch.setattr(h.service, "validate_workbench_derived_input", lambda *_a, **_k: None)
    before = [deepcopy(row(h.service, rid)) for rid in ("old", "successor")]
    response = client.post(PREFIX + "/derived-successor/acceptance", json=h.payload)
    assert response.status_code == 200, response.text
    accepted = response.json()
    assert accepted[INPUT_PARENT_FIELD] == INPUT_PARENT
    assert accepted["request_parent_message_id"] == SENT_PARENT
    assert accepted["parent_message_id"] == FINAL_CURSOR
    assert accepted["status"] == "succeeded"
    normal = client.get("/api/task-results/text/successor")
    assert normal.status_code == 200 and normal.json()["result_ready"]
    assert INPUT_PARENT_FIELD not in normal.json()["receipt"]
    assert normal.json()["receipt"]["request_parent_message_id"] == SENT_PARENT
    assert [row(h.service, rid) for rid in ("old", "successor")] == before
    assert not h.sent and not review.await_count
    # Complete synthetic HTTP wire responses for the paired Node parser test.
    for name, value in (("split-attributes-acceptance", accepted),
                        ("split-attributes-task-result", normal.json()),
                        ("split-attributes-retained-input", h.service.read_retained_input(
                            "owner", "old", source="internal:listing")),
                        ("split-attributes-request", h.payload)):
        (tmp_path / (name + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@pytest.mark.parametrize("corruption", ["owner", "original_hash", "child_hash", "body",
                                       "missing_input", "missing_original_input", "audit"])
def test_split_attributes_acceptance_requires_original_and_child_durable_proof(
        incomplete_http, corruption):
    h, review, client, identity = incomplete_http
    split_attribute_result(h, client)
    review.reset_mock()
    if corruption == "owner":
        identity["id"] = "foreign-owner"
    elif corruption == "original_hash":
        h.payload["original_input_hash"] = "0" * 64
    else:
        rid = "old" if corruption == "missing_original_input" else "successor"
        with h.service.store.transaction() as db:
            r = h.service.store.read_receipt(db, "text", "owner", rid)
            if corruption == "child_hash":
                db.execute("UPDATE requests SET request_hash=? WHERE owner=? AND id=?",
                           ("0" * 64, "owner", rid))
            elif corruption == "body":
                body = h.service.store.load_input(r["_input_ref"])
                body["parent_message_id"] = "different-source"
                r["_input_ref"] = h.service.store.save_input(body)
            elif corruption in {"missing_input", "missing_original_input"}:
                r["_input_ref"] = "0" * 32 + ".json"
            elif corruption == "audit":
                r["_workbench_derived_input"]["child_input_hash"] = "0" * 64
            r[INPUT_PARENT_FIELD] = INPUT_PARENT  # A stored forged proof cannot rescue invalid input.
            h.service.store.write_receipt(db, "text", "owner", rid, r)
    response = client.post(PREFIX + "/derived-successor/acceptance", json=h.payload)
    assert response.status_code in {404, 409}, response.text
    assert INPUT_PARENT_FIELD not in response.json()
    assert not h.sent and not review.await_count


def test_ordinary_get_discards_forged_input_parent_and_other_kind_gets_no_proof(incomplete_http):
    h, review, client, _identity = incomplete_http
    response = client.post(PREFIX + "/derived-successor", json=h.payload)
    assert response.status_code == 200, response.text
    update(h.service, "successor", **{INPUT_PARENT_FIELD: INPUT_PARENT})
    review.reset_mock()
    normal = client.get("/api/task-results/text/successor")
    accepted = client.post(PREFIX + "/derived-successor/acceptance", json=h.payload)
    assert normal.status_code == accepted.status_code == 200
    assert INPUT_PARENT_FIELD not in normal.json()["receipt"]
    assert INPUT_PARENT_FIELD not in accepted.json()
    assert not h.sent and not review.await_count


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


def test_incomplete_acceptance_is_local_and_returns_exact_existing_receipt(incomplete_http):
    h, review, client, identity = incomplete_http
    h.service.recovery_reader = lambda _original: pytest.fail("acceptance observation read upstream")
    before = deepcopy(row(h.service))

    absent = client.post(PREFIX + "/derived-successor/acceptance", json=h.payload)

    body = h.service.prepare_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    assert absent.status_code == 200, absent.text
    assert absent.json() == {
        "state": "absent_at_observation", "original_request_id": "old",
        "client_request_id": "successor", "original_input_hash": h.payload["original_input_hash"],
        "child_input_hash": TextTaskService._submission_identity("owner", body)[1],
    }
    assert row(h.service, "successor") is None and row(h.service) == before
    assert getattr(h, "sent", []) == [] and not review.await_count

    accepted = h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    existing = client.post(PREFIX + "/derived-successor/acceptance", json=h.payload)
    assert existing.status_code == 200 and existing.json() == accepted
    assert getattr(h, "sent", []) == [] and not review.await_count

    changed = {**h.payload, "thinking_effort": "standard"}
    conflict = client.post(PREFIX + "/derived-successor/acceptance", json=changed)
    assert conflict.status_code == 409 and conflict.json()["detail"]["code"] == "CONVERSATION_REQUEST_CONFLICT"

    identity["id"] = "other-owner"
    foreign = client.post(PREFIX + "/derived-successor/acceptance", json=h.payload)
    assert foreign.status_code == 404
    assert getattr(h, "sent", []) == [] and not review.await_count


def test_incomplete_acceptance_rejects_unqualified_absent_child_locally(incomplete_http):
    h, review, client, _identity = incomplete_http
    before = deepcopy(row(h.service))
    update(h.service, status="succeeded", content="late original")

    response = client.post(PREFIX + "/derived-successor/acceptance", json=h.payload)

    assert response.status_code == 409
    assert row(h.service, "successor") is None
    assert getattr(h, "sent", []) == [] and not review.await_count
    assert row(h.service)["status"] == "succeeded" and before["request_id"] == "old"


@pytest.mark.parametrize("suffix", ["/derived-successor/acceptance", "/derived-successor"])
@pytest.mark.parametrize("changes", [
    {"_result_no_progress_reads": 1},
    {"_result_last_progress_at": 999.0},
    {"_result_wait_ended_at": None},
    {"_result_observation_started_at": None, "_result_last_checked_at": None,
     "_result_wait_ended_at": None, "_result_no_progress_reads": None,
     "_original_result_observation": None},
    {"recovery_claim_id": "reading-original", "recovery_claimed_at": 1000.0,
     "recovery_lease_until": 1100.0, "recovery_phase": "read_text_request"},
])
def test_incomplete_waiting_qualification_is_busy_on_both_public_paths(incomplete_http, suffix, changes):
    h, review, client, _identity = incomplete_http
    update(h.service, **changes)
    before = deepcopy(row(h.service))
    h.service.recovery_reader = lambda *_a: pytest.fail("validation read upstream")
    response = client.post(PREFIX + suffix, json=h.payload)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "CHAT_SUPERSEDE_PREDECESSOR_BUSY"
    assert row(h.service) == before and row(h.service, "successor") is None
    assert not h.sent and not review.await_count


@pytest.mark.parametrize("suffix", ["/derived-successor/acceptance", "/derived-successor"])
@pytest.mark.parametrize("changes", [
    {"_recovery_paused": True}, {"_recovery_suppressed": True},
    {"_upstream_terminal": True}, {"upstream_outcome": None},
    {"_original_result_observation": {"invalid": True}},
    {"_result_no_progress_reads": "1"}, {"_result_last_checked_at": -1},
    {"status": "succeeded", "content": "late original"},
])
def test_incomplete_permanent_ineligibility_stays_invalid(incomplete_http, suffix, changes):
    h, review, client, _identity = incomplete_http
    update(h.service, **changes)
    before = deepcopy(row(h.service))
    response = client.post(PREFIX + suffix, json=h.payload)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "CHAT_WORKBENCH_DERIVED_INPUT_INVALID"
    assert row(h.service) == before and row(h.service, "successor") is None
    assert not h.sent and not review.await_count


@pytest.mark.parametrize("suffix", ["/derived-successor/acceptance", "/derived-successor"])
def test_incomplete_bad_input_is_not_masked_by_pending_observation(incomplete_http, suffix):
    h, review, client, _identity = incomplete_http
    update(h.service, _result_no_progress_reads=1)
    response = client.post(PREFIX + suffix, json={**h.payload, "original_input_hash": "0" * 64})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] != "CHAT_SUPERSEDE_PREDECESSOR_BUSY"
    assert row(h.service, "successor") is None and not h.sent and not review.await_count


def test_incomplete_new_progress_after_acceptance_keeps_child_retryable(incomplete_http):
    from services.conversation_binding_service import ConversationBindingError
    h, review, client, _identity = incomplete_http
    accepted = client.post(PREFIX + "/derived-successor", json=h.payload)
    assert accepted.status_code == 200, accepted.text
    child = row(h.service, "successor")
    update(h.service, _result_last_progress_at=999.0)
    with h.service.store.connect() as db, pytest.raises(ConversationBindingError) as exc:
        h.service._validate_supersede(h.service.store, db, "owner", child, h.clock())
    assert exc.value.code == "CHAT_SUPERSEDE_PREDECESSOR_BUSY"
    assert row(h.service, "successor") == child and not h.sent


def test_incomplete_acceptance_preserves_work_lifecycle_http_status(incomplete_http, monkeypatch):
    h, review, client, _identity = incomplete_http

    def blocked(*_args, **_kwargs):
        raise WorkLifecycleError("WORK_NOT_ACTIVE", status=423)

    monkeypatch.setattr(h.service, "observe_workbench_derived_acceptance", blocked)
    response = client.post(PREFIX + "/derived-successor/acceptance", json=h.payload)

    assert response.status_code == 423
    assert response.json() == {"detail": {"code": "WORK_NOT_ACTIVE"}}
    assert row(h.service, "successor") is None and getattr(h, "sent", []) == []
    assert not review.await_count


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
