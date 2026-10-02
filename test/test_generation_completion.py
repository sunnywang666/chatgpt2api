"""Controlled pure-generation recovery; these are not real upstream samples."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.generation_completion import GenerationCompletionService, CompletionError
from services.image_task_service import ImageTaskService
from services.request_context import current_request, AdmissionLost, executing
from services.work_lifecycle import WorkLifecycleService, WorkLifecycleError, ensure_work
from test.test_unknown_turn_recovery import migration
from test.test_stalled_text_diagnostics import incomplete_reader


IDENTITY = {"id": "owner", "role": "user", "external_image_client": True}


def row(service, kind="text", request_id="old-0"):
    with service.store.connect() as db:
        return service.store.read_receipt(db, kind, "owner", request_id)


def patch_row(service, kind="text", request_id="old-0", **changes):
    with service.store.transaction() as db:
        receipt = service.store.read_receipt(db, kind, "owner", request_id)
        receipt.update(changes)
        service.store.write_receipt(db, kind, "owner", request_id, receipt)


@pytest.fixture
def setup(tmp_path):
    text, admission, backend, _ = migration(tmp_path)
    admission.clock.now = 3000
    body = {"client_request_id": "old-0", "client_conversation_id": "client-0", "_public_route": "chat",
            "_public_session_ref": "original-session", "model": "fixture-text",
            "messages": [{"role": "user", "content": "Retained original input"}]}
    with text.store.transaction() as db:
        original = text.store.read_receipt(db, "text", "owner", "old-0")
        original.update(_input_ref=text.store.save_input(body), route="chat", _public_session_ref="original-session",
                        _submission_started=True, _turn_reserved=True, _executing=False,
                        _execution_timeline=[{"stage": "send_call_started", "at": 10}])
        ensure_work(text.store, db, "text", "owner", "old-0", original)
        text.store.write_receipt(db, "text", "owner", "old-0", original)
        db.execute("UPDATE requests SET request_hash=? WHERE owner='owner' AND id='old-0'", (text._submission_identity("owner", body)[1],))
    text.recovery_reader = incomplete_reader(backend, lambda doc, msg: doc.update(is_archived=False))
    calls = []
    def runner(body, on_cursor):
        context = current_request.get()
        context.before_send()
        calls.append(copy.deepcopy(body))
        return {"content": "Completed original objective", "conversation_id": body.get("conversation_id", "conv-0"),
                "parent_message_id": "new-final", "_upstream_terminal": True, "upstream_outcome": "completed"}
    text.runner = runner
    admission.register("text", lambda ctx, body: text._run(ctx.owner, ctx.request_id, body))
    images = ImageTaskService(tmp_path / "images.json", admission=admission, store=text.store)
    lifecycle = WorkLifecycleService(text, images, clock=admission.clock)
    service = GenerationCompletionService(text, images, lifecycle, clock=admission.clock)
    admission.generation_completion = service
    return service, admission, calls


def start(service, kind="text", rid="old-0"):
    return service.start(kind, IDENTITY, rid, allow_unconfirmed_retry=True)


def ended_original(service):
    from services.conversation_binding_service import ConversationBindingService
    from test.test_unknown_turn_recovery import document
    service.text.recovery_reader = lambda receipt: ConversationBindingService._read_text_request_result(
        None, receipt, document=document(receipt))
    patch_row(service, _sequence=0)
    service.text.read("owner", "old-0")
    return row(service)


@pytest.mark.parametrize("status", ["unknown", "failed"])
def test_confirmed_original_turn_retries_same_account_session_work_and_closes(setup, status):
    service, admission, calls = setup
    original = ended_original(service)
    if status == "failed":
        patch_row(service, status="failed", error_code="RESULT_UNRECOVERABLE")
    # No permission to create a new conversation and no second workflow slot.
    result = service.start("text", IDENTITY, "old-0")
    child_id = result["replacement_id"]
    assert result["conversation_mode"] == "original" and result["original_turn_ended"]
    child = row(service, request_id=child_id)
    for key in ("provider_account_identity", "provider_binding_id", "conversation_id",
                "client_conversation_id", "_public_session_ref", "_work_key"):
        assert child[key] == original[key]
    assert child["parent_message_id"] == original["_turn_end_evidence"]["final_message_id"]
    assert child["request_message_id"] != original["request_message_id"]
    assert child["_previous_request_id"] == "old-0"
    with service.store.connect() as db:
        assert db.execute("SELECT count(*) FROM task_runtime WHERE name LIKE 'work:%'").fetchone()[0] == 1
    def run(body, on_cursor):
        current_request.get().before_send()
        calls.append(copy.deepcopy(body))
        return {"content": "Completed original objective", "conversation_id": original["conversation_id"],
                "parent_message_id": "corrected-final", "_upstream_terminal": True, "upstream_outcome": "completed"}
    service.text.runner = run
    restarted = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=admission.clock)
    assert restarted.start("text", IDENTITY, "old-0")["replacement_id"] == child_id
    ctx = admission.claim_next()
    assert ctx and ctx.request_id == child_id
    admission.execute(ctx)
    assert len(calls) == 1
    assert calls[0]["messages"] == service.store.load_input(original["_input_ref"])["messages"]
    assert restarted.read("text", IDENTITY, "old-0")["selected_id"] == child_id
    done = restarted.complete("text", IDENTITY, "old-0", child_id)
    assert done["state"] == "completed" and done["original_cleanup"] == "not_required"
    assert done["work"]["archive"]["status"] == "pending" and not done["work"]["slot_held"]
    archive_calls = []
    def archive(owner, request_id, archived):
        archive_calls.append((request_id, row(service, request_id=request_id)["conversation_id"], archived))
        return {"request_id": request_id, "archived": archived,
                "conversation": {"client_conversation_id": original["_public_session_ref"]}}
    service.text.set_public_session_archived = archive
    service.lifecycle.process_one()
    assert restarted.read("text", IDENTITY, "old-0")["work"]["archive"]["status"] == "confirmed"
    restarted.rework("text", IDENTITY, "old-0", child_id)
    service.lifecycle.process_one()
    assert archive_calls == [(child_id, original["conversation_id"], True), (child_id, original["conversation_id"], False)]
    assert restarted.read("text", IDENTITY, "old-0")["work"]["state"] == "active"
    assert row(service)["status"] == status and row(service)["_input_ref"] == original["_input_ref"]
    assert len(calls) == 1


def test_ended_empty_alone_cannot_close_work_and_drift_before_retry_never_sends(setup):
    service, admission, calls = setup
    ended_original(service)
    with pytest.raises(WorkLifecycleError, match="WORK_TURN_UNFINISHED"):
        service.lifecycle.update("text", IDENTITY, "old-0", "completed", results_saved=True)
    child_id = service.start("text", IDENTITY, "old-0")["replacement_id"]
    service.text.recovery_reader = Mock(return_value={"status": "running", "recovery_reason": "REQUEST_RESULT_INCOMPLETE"})
    ctx = admission.claim_next()
    assert ctx and ctx.request_id == child_id
    admission.execute(ctx)
    assert calls == [] and row(service, request_id=child_id)["upstream_outcome"] == "not_sent"
    assert row(service)["status"] == "unknown"


@pytest.mark.parametrize("after_claim", [False, True])
def test_original_recovered_before_same_session_send_closes_and_allows_later_continuation(setup, after_claim):
    service, admission, calls = setup
    original = ended_original(service)
    child_id = service.start("text", IDENTITY, "old-0")["replacement_id"]
    ctx = admission.claim_next() if after_claim else None
    patch_row(service, status="succeeded", content="late original result", upstream_outcome="completed",
              parent_message_id="late-final", recovery_next_at=None)
    if ctx:
        with executing(ctx), pytest.raises(AdmissionLost):
            ctx.before_send()
    result = service.read("text", IDENTITY, "old-0")
    assert result["selected_id"] == "old-0" and row(service, request_id=child_id)["upstream_outcome"] == "not_sent"
    assert result["work"]["request_id"] == "old-0"
    assert admission.claim_next() is None and calls == []
    assert service.complete("text", IDENTITY, "old-0", "old-0")["state"] == "completed"
    # Stand in only for confirmed archive/restore; the full chain is tested above.
    with service.store.transaction() as db:
        work = service.store.runtime(db, original["_work_key"])
        work.update(state="active", archive={"status": "confirmed", "desired": False, "archived": False})
        service.store.set_runtime(db, work["key"], work)
    body = service.store.load_input(original["_input_ref"])
    response = service.text.submit("owner", {**body, "client_request_id": "later-user-turn",
                                            "_previous_request_id": "old-0"})
    assert response["status"] == "queued"
    assert row(service, request_id="later-user-turn")["conversation_id"] == original["conversation_id"]


def test_failed_label_with_inconsistent_end_metadata_cannot_start_same_session_retry(setup):
    service, admission, calls = setup
    ended_original(service)
    patch_row(service, status="failed", error_code="UNRELATED_FAILURE", upstream_outcome="failed")
    result = service.start("text", IDENTITY, "old-0")
    assert result["reason"] == "COMPLETION_ORIGINAL_END_UNCONFIRMED"
    assert "replacement_id" not in result and not calls


def test_same_session_retry_pause_restart_keeps_one_child_and_unknown_occupancy_needs_proof(setup):
    service, admission, calls = setup
    original = ended_original(service)
    child_id = service.start("text", IDENTITY, "old-0")["replacement_id"]
    with service.store.transaction() as db:
        work = service.store.runtime(db, original["_work_key"])
        work["state"] = "paused"
        service.store.set_runtime(db, work["key"], work)
    for _ in range(3):
        assert admission.claim_next() is None
        admission.clock.now += 31
        assert service.start("text", IDENTITY, "old-0")["replacement_id"] == child_id
    with service.store.transaction() as db:
        work = service.store.runtime(db, original["_work_key"])
        work["state"] = "active"
        service.store.set_runtime(db, work["key"], work)
    assert admission.claim_next().request_id == child_id
    assert calls == []


def test_concurrent_authorization_restart_single_replacement_and_actual_completion(setup):
    service, admission, calls = setup
    original = row(service)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: start(service), range(4)))
    result = service.read("text", IDENTITY, "old-0")
    child_id = result["replacement_id"]
    assert result["state"] == "replacement_pending"
    assert len({r.get("replacement_id") for r in results if r.get("replacement_id")}) == 1
    saved = row(service)
    assert {k: v for k, v in saved.items() if k not in {"_completion", "updated_at"} and not k.startswith("recovery") and not k.startswith("_result") and k != "_original_result_observation"}.items() >= {
        k: original[k] for k in ("status", "_input_ref", "request_message_id", "provider_account_identity", "conversation_id")}.items()
    restarted = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=admission.clock)
    assert start(restarted)["replacement_id"] == child_id
    assert row(service)["_turn_reserved"] is False
    ctx = admission.claim_next()
    assert ctx and ctx.request_id == child_id
    assert row(service, request_id=child_id)["provider_account_identity"] == original["provider_account_identity"]
    admission.execute(ctx)
    assert len(calls) == 1
    assert calls[0]["messages"] == service.store.load_input(original["_input_ref"])["messages"]
    selected = restarted.read("text", IDENTITY, "old-0")
    assert selected["selected_id"] == child_id and selected["state"] == "result_ready"
    assert selected["result"]["content"] == "Completed original objective"
    assert selected["stop"] == {"capability": "unsupported", "confirmed": False}
    with pytest.raises(CompletionError, match="MISMATCH"):
        service.complete("text", IDENTITY, "old-0", "unrelated")
    done = service.complete("text", IDENTITY, "old-0", child_id)
    assert done["state"] == "completed" and done["original_cleanup"] == "completed"
    assert done["work"]["archive"]["status"] == "pending"
    original_work = done["original_work"]
    assert original_work["state"] == "completed" and not original_work["slot_held"]
    assert original_work["request_id"] == child_id
    assert row(service)["status"] == "unknown" and row(service)["_turn_reserved"] is False
    assert start(restarted)["selected_id"] == child_id and len(calls) == 1
    with pytest.raises(CompletionError, match="ATTEMPT_LIMIT"):
        start(restarted, rid=child_id)


def test_late_original_before_send_cancels_unsent_child_and_selects_original(setup):
    service, admission, calls = setup
    child_id = start(service)["replacement_id"]
    ctx = admission.claim_next()
    patch_row(service, status="succeeded", content="late original", _upstream_terminal=True, upstream_outcome="completed")
    admission.execute(ctx)
    assert calls == []
    assert row(service, request_id=child_id)["status"] == "failed"
    assert row(service, request_id=child_id)["upstream_outcome"] == "not_sent"
    assert service.read("text", IDENTITY, "old-0")["selected_id"] == "old-0"


def test_late_original_after_selection_does_not_replace_saved_result(setup):
    service, admission, calls = setup
    child_id = start(service)["replacement_id"]
    admission.execute(admission.claim_next())
    assert service.read("text", IDENTITY, "old-0")["selected_id"] == child_id
    patch_row(service, status="succeeded", content="late original", upstream_outcome="completed")
    result = service.read("text", IDENTITY, "old-0")
    assert result["selected_id"] == child_id and result["result"]["content"] != "late original"
    assert row(service)["content"] == "late original" and len(calls) == 1


def test_paused_original_work_holds_existing_replacement_without_repeated_claims(setup):
    service, admission, calls = setup
    child_id = start(service)["replacement_id"]
    def state(value):
        with service.store.transaction() as db:
            original = service.store.read_receipt(db, "text", "owner", "old-0")
            work = service.store.runtime(db, original["_work_key"])
            work["state"] = value
            service.store.set_runtime(db, work["key"], work)
    state("paused")
    for _ in range(3):
        admission.clock.now += 31
        service.advance("text", "owner", "old-0")
        assert admission.claim_next() is None
        assert row(service, request_id=child_id).get("_claim_id") is None
    result = service.read("text", IDENTITY, "old-0")
    assert result["reason"] == "COMPLETION_WORK_NOT_ACTIVE"
    assert "work_not_active" in result["waiting"]["reasons"]
    state("active")
    admission.execute(admission.claim_next())
    assert len(calls) == 1 and service.read("text", IDENTITY, "old-0")["selected_id"] == child_id


@pytest.mark.parametrize("problem", ["too_young", "input_missing", "wrong_route", "paused", "read_failure", "no_permission"])
def test_eligibility_never_invents_unsent_or_sends_on_missing_evidence(setup, problem):
    service, admission, calls = setup
    if problem == "too_young":
        patch_row(service, _execution_timeline=[{"stage": "send_call_started", "at": 2950}])
    elif problem == "input_missing":
        patch_row(service, _input_ref="missing")
    elif problem == "wrong_route":
        patch_row(service, _forward_protocol="business_mutation")
    elif problem == "paused":
        with service.store.transaction() as db:
            work = service.store.runtime(db, row(service)["_work_key"])
            work["state"] = "paused"
            service.store.set_runtime(db, work["key"], work)
    elif problem == "read_failure":
        service.text.recovery_reader = Mock(side_effect=ConnectionError("controlled"))
    if problem == "wrong_route":
        with pytest.raises(CompletionError, match="PURE_GENERATION"):
            start(service)
    else:
        result = service.start("text", IDENTITY, "old-0", allow_unconfirmed_retry=problem != "no_permission")
        assert "replacement_id" not in result and result.get("reason")
    assert calls == [] and row(service)["status"] == "unknown"


def test_crash_after_reservation_reuses_prepared_input_and_rejects_drift(setup, monkeypatch):
    service, admission, calls = setup
    real_submit = service.text.submit
    monkeypatch.setattr(service.text, "submit", Mock(side_effect=RuntimeError("process exit before insert")))
    with pytest.raises(RuntimeError):
        start(service)
    state = row(service)["_completion"]
    prepared = service.store.load_input(state["prepared_input"])
    changed = {**prepared, "messages": [{"role": "user", "content": "drift"}]}
    with pytest.raises(CompletionError, match="INPUT_CHANGED"):
        real_submit("owner", changed)
    monkeypatch.setattr(service.text, "submit", real_submit)
    admission.clock.now += 31
    assert start(service)["replacement_id"] == state["replacement_id"]
    admission.execute(admission.claim_next())
    assert len(calls) == 1 and calls[0]["messages"] == prepared["messages"]


def test_explicit_later_authorization_can_enable_the_one_unused_attempt(setup):
    service, admission, calls = setup
    initial = service.start("text", IDENTITY, "old-0")
    assert initial["reason"] == "COMPLETION_ORIGINAL_END_UNCONFIRMED"
    assert "replacement_id" not in initial
    authorized = start(service)
    assert authorized["replacement_id"] and authorized["max_extra_requests"] == 1
    assert start(service)["replacement_id"] == authorized["replacement_id"]


@pytest.mark.parametrize("mode", ["generate", "edit"])
def test_image_replacement_retains_exact_input_and_download_evidence_prevents_redraw(setup, mode):
    service, admission, calls = setup
    kwargs = dict(client_task_id="image-original", prompt="retained image instruction", model="gpt-image-2", size=None)
    if mode == "edit":
        service.images.submit_edit(IDENTITY, **kwargs, images=[(b"exact-original-image", "image/png", "original.png")])
    else:
        service.images.submit_generation(IDENTITY, **kwargs)
    patch_row(service, "image", "image-original", status="error", upstream_outcome="not_sent",
              error_code="IMAGE_GENERATION_NOT_SUBMITTED", _submission_started=False,
              active_attempt_started_at=1000, active_attempt_deadline_at=1300)
    old = row(service, "image", "image-original")
    result = start(service, "image", "image-original")
    retried = row(service, "image", "image-original")
    assert "replacement_id" not in result and retried["status"] == "queued"
    assert retried["_completion"]["same_request_retry"]
    assert retried["_input_ref"] == old["_input_ref"]
    assert retried["request_hash"] == old["request_hash"]
    assert retried["active_attempt_started_at"] is None and retried["active_attempt_deadline_at"] is None
    assert retried["_execution_timeline"][-1]["previous_active_deadline_at"] == 1300
    patch_row(service, "image", "image-original", status="error", upstream_outcome="not_sent",
              error_code="IMAGE_GENERATION_NOT_SUBMITTED")
    admission.clock.now += 31
    assert start(service, "image", "image-original")["reason"] == "COMPLETION_ORIGINAL_NOT_RETRYABLE"
    # A different original with a known generated artifact must only download.
    service.images.submit_generation(IDENTITY, **{**kwargs, "client_task_id": "download-original"})
    patch_row(service, "image", "download-original", status="error", upstream_outcome="generated", result_file_ids=["saved-file"])
    result = start(service, "image", "download-original")
    assert result["reason"] == "COMPLETION_DOWNLOAD_ORIGINAL_RESULT" and "replacement_id" not in result


@pytest.fixture
def failed_unsent_image(setup):
    service, admission, calls = setup
    service.images.submit_generation(IDENTITY, client_task_id="repair-image", prompt="retained input",
                                     model="gpt-image-2", size=None)
    patch_row(service, "image", "repair-image", status="error", upstream_outcome="not_submitted",
              _submission_started=False, upstream_submission_started=False, upstream_unfinished=False,
              error_code="RESULT_UNRECOVERABLE", recovery_retryable=True,
              last_recovery_failure={"at": 2900.0, "type": "ImageGenerationError"},
              active_attempt_started_at=2000, active_attempt_deadline_at=2300,
              _completion={"state": "needs_attention", "allow_unconfirmed_retry": False,
                           "same_request_retry": True, "max_extra_requests": 1, "next_at": None})
    return service


def test_explicit_unsent_image_repair_is_same_id_and_idempotent_across_restart(failed_unsent_image):
    service = failed_unsent_image
    old = row(service, "image", "repair-image")
    service.start("image", IDENTITY, "repair-image", retry_not_sent_failure_at=2900.0)
    retried = row(service, "image", "repair-image")
    assert retried["status"] == "queued" and retried["_completion"]["same_request_retry"]
    assert retried["_input_ref"] == old["_input_ref"] and retried["request_hash"] == old["request_hash"]
    assert retried["active_attempt_deadline_at"] is None
    restarted = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=service.clock)
    restarted.start("image", IDENTITY, "repair-image", retry_not_sent_failure_at=2900.0)
    assert row(service, "image", "repair-image")["_execution_timeline"] == retried["_execution_timeline"]
    patch_row(service, "image", "repair-image", status="error", last_recovery_failure={"at": 2950.0},
              _completion={**retried["_completion"], "state": "needs_attention", "next_at": None})
    # A delayed repeat of the old authorization cannot retry the new failure.
    restarted.start("image", IDENTITY, "repair-image", retry_not_sent_failure_at=2900.0)
    assert row(service, "image", "repair-image")["status"] == "error"
    restarted.start("image", IDENTITY, "repair-image", retry_not_sent_failure_at=2950.0)
    assert row(service, "image", "repair-image")["status"] == "queued"


@pytest.mark.parametrize("changes", [
    {"upstream_outcome": "unknown"}, {"upstream_submission_started": True}, {"_submission_started": True},
    {"_executing": True}, {"result_file_ids": ["known-result"]}, {"data": [{"b64_json": "saved"}]},
    {"_execution_timeline": [{"stage": "send_call_started", "at": 2000}]},
    {"last_recovery_failure": {"at": 2901.0}}, {"_recovery_paused": True}, {"_recovery_suppressed": True},
])
def test_explicit_unsent_repair_rejects_sent_unknown_results_and_pause(failed_unsent_image, changes):
    service = failed_unsent_image
    patch_row(service, "image", "repair-image", **changes)
    before = row(service, "image", "repair-image")
    with pytest.raises(CompletionError):
        service.start("image", IDENTITY, "repair-image", retry_not_sent_failure_at=2900.0)
    assert row(service, "image", "repair-image") == before


def test_explicit_unsent_repair_rejects_missing_original_input(failed_unsent_image):
    service = failed_unsent_image
    patch_row(service, "image", "repair-image", _input_ref="missing-original.json")
    with pytest.raises(CompletionError, match="COMPLETION_ORIGINAL_INPUT_UNAVAILABLE"):
        service.start("image", IDENTITY, "repair-image", retry_not_sent_failure_at=2900.0)
    assert row(service, "image", "repair-image")["status"] == "error"


def test_image_repair_api_owner_and_exact_failure_contract(failed_unsent_image, monkeypatch):
    import api.generation_completion as api
    service = failed_unsent_image
    monkeypatch.setattr(api, "get_generation_completion_service", lambda: service)
    monkeypatch.setattr(api, "require_identity", lambda authorization, request: {**IDENTITY, "id": authorization or "other"})
    monkeypatch.setattr(api, "require_image_policy", lambda *a, **kw: None)
    app = FastAPI()
    app.include_router(api.create_router("image"))
    client = TestClient(app)
    endpoint = "/api/image-tasks/repair-image/completion"
    payload = {"action": "recover", "retry_not_sent_failure_at": 2900.0}
    assert client.post(endpoint, headers={"Authorization": "other"}, json=payload).status_code == 404
    assert client.post(endpoint, headers={"Authorization": "owner"}, json={**payload, "allow_unconfirmed_retry": True}).status_code == 422
    assert client.post(endpoint, headers={"Authorization": "owner"}, json={**payload, "retry_not_sent_failure_at": 2899.0}).status_code == 409
    response = client.post(endpoint, headers={"Authorization": "owner"}, json=payload)
    assert response.status_code == 200 and response.json()["original_id"] == "repair-image"
    assert row(service, "image", "repair-image")["status"] == "queued"


def test_api_owner_isolation_and_explicit_saved_reviewed_ack(setup, monkeypatch):
    import api.generation_completion as api
    service, admission, calls = setup
    monkeypatch.setattr(api, "get_generation_completion_service", lambda: service)
    monkeypatch.setattr(api, "require_identity", lambda authorization, request: {**IDENTITY, "id": authorization or "other"})
    monkeypatch.setattr(api, "require_chat_text_policy", lambda *a, **kw: None)
    app = FastAPI()
    app.include_router(api.create_router("text"))
    client = TestClient(app)
    endpoint = "/api/chat-requests/old-0/completion"
    assert client.get(endpoint, headers={"Authorization": "other"}).status_code == 404
    assert client.post(endpoint, headers={"Authorization": "owner"}, json={"prompt": "injection"}).status_code == 422
    result = client.post(endpoint, headers={"Authorization": "owner"}, json={"allow_unconfirmed_retry": True})
    assert result.status_code == 200 and result.json()["replacement_id"]
    assert client.post(endpoint, headers={"Authorization": "owner"}, json={"action": "complete", "selected_id": "wrong", "results_saved": True}).status_code == 422
    assert not calls


def test_recovery_pause_fences_prepared_completion_and_preserves_original(setup):
    from services.generation_completion import replacement_send_allowed
    service, admission, calls = setup
    ended_original(service)
    state = service.start("text", IDENTITY, "old-0")
    child_id = state["replacement_id"]
    original = row(service)
    service.store.set_recovery_paused("text", "owner", "old-0", True)
    service.process_one()
    assert admission.claim_next() is None
    with service.store.connect() as db:
        child = service.store.read_receipt(db, "text", "owner", child_id)
        assert not replacement_send_allowed(service.store, db, "text", "owner", child_id, child)
    assert calls == [] and row(service)["_completion"]["replacement_id"] == child_id
    service.store.set_recovery_paused("text", "owner", "old-0", False)
    assert admission.claim_next().request_id == child_id
    assert row(service)["conversation_id"] == original["conversation_id"]
