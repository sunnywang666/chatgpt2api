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
from services.work_lifecycle import WorkLifecycleService, ensure_work
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
    text.recovery_reader = incomplete_reader(backend)
    calls = []
    def runner(body, on_cursor):
        context = current_request.get()
        context.before_send()
        calls.append(copy.deepcopy(body))
        return {"content": "Completed original objective", "conversation_id": "new-conversation",
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
        k: original[k] for k in ("status", "_input_ref", "request_message_id", "provider_account_identity", "conversation_id", "_turn_reserved")}.items()
    restarted = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=admission.clock)
    assert start(restarted)["replacement_id"] == child_id
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    ctx = admission.claim_next()
    assert ctx and ctx.request_id == child_id
    assert row(service, request_id=child_id)["provider_account_identity"] != original["provider_account_identity"]
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
    assert done["state"] == "completed" and done["original_cleanup"] == "pending"
    assert done["work"]["archive"]["status"] == "pending"
    original_work = service.lifecycle.get("text", IDENTITY, "old-0")
    assert original_work["state"] == "completed" and original_work["cleanup_pending"] is True
    assert original_work["completion_result_id"] == child_id
    assert row(service)["status"] == "unknown" and row(service)["_turn_reserved"] is True
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
    assert initial["reason"] == "COMPLETION_UNCONFIRMED_RETRY_NOT_AUTHORIZED"
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
    patch_row(service, "image", "image-original", status="error", upstream_outcome="not_sent")
    old = row(service, "image", "image-original")
    result = start(service, "image", "image-original")
    new = row(service, "image", result["replacement_id"])
    original_input = service.store.load_input(old["_input_ref"])
    replacement_input = service.store.load_input(new["_input_ref"])
    assert new["_completion_of"] == "image-original"
    assert replacement_input["payload"]["prompt"] == original_input["payload"]["prompt"]
    if mode == "edit":
        assert replacement_input["payload"]["images"] == original_input["payload"]["images"]
    assert not new["conversation_id"] and not new["provider_binding_id"]
    assert start(service, "image", "image-original")["replacement_id"] == new["id"]
    # A different original with a known generated artifact must only download.
    service.images.submit_generation(IDENTITY, **{**kwargs, "client_task_id": "download-original"})
    patch_row(service, "image", "download-original", status="error", upstream_outcome="generated", result_file_ids=["saved-file"])
    result = start(service, "image", "download-original")
    assert result["reason"] == "COMPLETION_DOWNLOAD_ORIGINAL_RESULT" and "replacement_id" not in result


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
