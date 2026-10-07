"""Controlled pure-generation recovery; these are not real upstream samples."""
import copy
import json
import threading
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.external_images import external_image_boundary
from services.generation_completion import GenerationCompletionService, CompletionError
from services.image_task_service import ImageTaskService
from services.request_context import current_request, AdmissionLost, executing
from services.work_lifecycle import WorkLifecycleService, WorkLifecycleError, ensure_work
from test.test_unknown_turn_recovery import migration
from test.test_stalled_text_diagnostics import incomplete_reader


IDENTITY = {"id": "owner", "role": "user", "external_image_client": True}


@pytest.mark.parametrize('change', ['none', 'active', 'similar_text', 'later_user', 'branch', 'asset',
                                  'archived', 'wrong_conversation', 'stale', 'future', 'saved_asset', 'paused',
                                  'cursor_before_error', 'cursor_before_error_code', 'cursor_before_finished'])
def test_terminal_image_failure_requires_fresh_closed_asset_free_original(change):
    from services.generation_completion import retry_cursor, verified_image_failure
    root = {'status': 'error', 'error_code': 'NO_IMAGE_GENERATED', 'upstream_outcome': 'unknown',
            'upstream_unfinished': False, 'conversation_id': 'conversation', 'request_message_id': 'request',
            'provider_binding_id': 'binding', 'provider_account_identity': 'account', 'client_conversation_id': 'client'}
    document = {'conversation_id': 'conversation', 'is_archived': False, 'current_node': 'answer', 'mapping': {
        'request': {'parent': 'old', 'message': {'author': {'role': 'user'}}},
        'answer': {'parent': 'request', 'message': {'author': {'role': 'assistant'}, 'status': 'finished_successfully',
                   'end_turn': True, 'content': {'content_type': 'text',
                   'parts': ['Something went wrong while generating your image. Sorry about that.']}}}}}
    message = document['mapping']['answer']['message']
    if change == 'active': message.update(status='in_progress', end_turn=False)
    if change == 'similar_text': message['content']['parts'] = ['Maybe something went wrong.']
    if change == 'later_user':
        document['mapping']['later'] = {'parent': 'answer', 'message': {'author': {'role': 'user'}}}
        document['current_node'] = 'later'
    if change == 'branch': document['mapping']['branch'] = {'parent': 'request', 'message': {'author': {'role': 'assistant'}}}
    if change == 'asset': message['content']['parts'].append({'asset_pointer': 'file-service://saved-file'})
    if change == 'archived': document['is_archived'] = True
    if change == 'wrong_conversation': document['conversation_id'] = 'other'
    root['_retry_cursor'] = retry_cursor(document, root, kind='image', now=100)
    if change == 'saved_asset': root['result_file_ids'] = ['already-generated']
    if change == 'paused': root['_recovery_paused'] = True
    # The cursor may be persisted before the authoritative terminal update.
    # A crash in that interval must not make the observation alone retryable.
    if change == 'cursor_before_error': root['status'] = 'running'
    if change == 'cursor_before_error_code': root['error_code'] = 'CONVERSATION_OUTCOME_UNKNOWN'
    if change == 'cursor_before_finished': root['upstream_unfinished'] = True
    now = 401 if change == 'stale' else 99 if change == 'future' else 100
    assert verified_image_failure(root, now, 300) is (change == 'none')


def row(service, kind="text", request_id="old-0"):
    with service.store.connect() as db:
        return service.store.read_receipt(db, kind, "owner", request_id)


def patch_row(service, kind="text", request_id="old-0", **changes):
    with service.store.transaction() as db:
        receipt = service.store.read_receipt(db, kind, "owner", request_id)
        receipt.update(changes)
        service.store.write_receipt(db, kind, "owner", request_id, receipt)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    text, admission, backend, _ = migration(tmp_path)
    admission.clock.now = 3000
    # This suite fakes archive I/O; its fake business clock must not read or
    # write a real account clock, including clocks left by another test run.
    monkeypatch.setattr("services.account_request_pacing.account_pacing_snapshot",
                        lambda account, now, **kwargs: {"next_at": now, "cooldown_until": 0})
    monkeypatch.setattr("services.account_request_pacing.reserve_account_archive_read", lambda *a, **k: True)
    monkeypatch.setattr("services.account_request_pacing.release_account_archive_read", lambda *a, **k: None)
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


def test_completion_discovery_leaves_writer_free_and_rechecks_concurrent_pause(setup, monkeypatch):
    service, admission, calls = setup
    patch_row(service, _automatic_generation_recovery=True)
    with service.store.transaction() as db:
        for index in range(200):
            rid = f"historical-failure-{index}"
            service.store.write_receipt(db, "image", "owner", rid, {
                "id": rid, "owner_id": "owner", "status": "error", "upstream_outcome": "rejected",
                "error_code": "IMAGE_CONTENT_POLICY", "data": [{"b64_json": "fixture" * 1000}]})
    scanned, release = threading.Event(), threading.Event()
    receipts = service.store.receipts
    discovered = []
    first = True

    def snapshot(db, **kwargs):
        nonlocal first
        rows = list(receipts(db, **kwargs))
        if first:
            first = False
            discovered.extend((kind, rid) for kind, _, rid, _ in rows)
            scanned.set()
            assert release.wait(3)
        yield from rows

    monkeypatch.setattr(service.store, "receipts", snapshot)
    service.advance = Mock(side_effect=AssertionError("paused work must not advance"))
    with ThreadPoolExecutor(max_workers=2) as workers:
        scan = workers.submit(service.process_one)
        try:
            assert scanned.wait(1)
            # A paused read snapshot must not own SQLite's writer lock.
            update = workers.submit(patch_row, service, _recovery_paused=True)
            update.result(timeout=1)
        finally:
            release.set()
        scan.result(timeout=2)
    assert discovered == [("text", "old-0")]
    assert row(service)["_recovery_paused"] is True
    assert not row(service).get("_completion")
    service.advance.assert_not_called()
    assert calls == []


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
    # A later user turn follows the selected successful receipt. The failed
    # original remains readable evidence, not a successful continuation cursor.
    from services.conversation_binding_service import ConversationBindingError
    body = service.store.load_input(original["_input_ref"])
    later = {**body, "client_request_id": "after-selected-result",
             "messages": [{"role": "user", "content": "Continue from the recovered answer."}]}
    with pytest.raises(ConversationBindingError) as rejected:
        service.text.submit("owner", {**later, "_previous_request_id": "old-0"})
    assert rejected.value.code == "CHAT_CONVERSATION_CONFLICT"
    assert row(service, request_id=later["client_request_id"]) is None
    submitted = service.text.submit("owner", {**later, "_previous_request_id": child_id})
    assert submitted["status"] == "queued"
    next_turn = row(service, request_id=later["client_request_id"])
    selected = row(service, request_id=child_id)
    for key in ("provider_account_identity", "provider_binding_id", "conversation_id",
                "client_conversation_id", "_public_session_ref", "parent_message_id", "_work_key"):
        assert next_turn[key] == selected[key]
    assert next_turn["_previous_request_id"] == child_id
    assert len(calls) == 1


@pytest.mark.parametrize("terminal_empty", [False, True])
@pytest.mark.parametrize("effort", [None, "high"])
def test_text_completion_preserves_public_effort_through_durable_retry(setup, terminal_empty, effort):
    from api.chat_requests import PublicChatRequest, _payload

    service, admission, calls = setup
    messages = [{"role": "user", "content": 'Return only JSON with keys "answer" and "steps".'}]
    public = PublicChatRequest(client_request_id="old-0", model="fixture-text", messages=messages,
                               **({"reasoning_effort": effort} if effort else {}))
    mapped = _payload("owner", public, messages)
    with service.store.transaction() as db:
        root = service.store.read_receipt(db, "text", "owner", "old-0")
        body = service.store.load_input(root["_input_ref"])
        body["messages"] = messages
        if "thinking_effort" in mapped:
            body["thinking_effort"] = mapped["thinking_effort"]
        root["_input_ref"] = service.store.save_input(body)
        service.store.write_receipt(db, "text", "owner", "old-0", root)
        db.execute("UPDATE requests SET request_hash=? WHERE owner='owner' AND id='old-0'",
                   (service.text._submission_identity("owner", body)[1],))
    if terminal_empty:
        ended_original(service)
    result = service.start("text", IDENTITY, "old-0", allow_unconfirmed_retry=not terminal_empty)
    child_id = result["replacement_id"]
    prepared = service.store.load_input(row(service)["_completion"]["prepared_input"])
    assert prepared.get("thinking_effort") == effort
    assert ("thinking_effort" in prepared) is (effort is not None)
    assert prepared["messages"] == messages
    assert prepared["_previous_request_id"] == prepared["_completion_of"] == "old-0"
    restarted = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=admission.clock)
    assert restarted.start("text", IDENTITY, "old-0", allow_unconfirmed_retry=not terminal_empty)["replacement_id"] == child_id
    ctx = admission.claim_next()
    assert ctx and ctx.request_id == child_id
    admission.execute(ctx)
    assert len(calls) == 1
    assert calls[0].get("thinking_effort") == effort
    assert ("thinking_effort" in calls[0]) is (effort is not None)
    assert calls[0]["messages"] == messages


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


def test_same_session_fresh_original_success_clears_active_error_and_preserves_history(setup):
    service, admission, calls = setup
    original = row(service)
    child_id = start(service)["replacement_id"]
    ctx = admission.claim_next()
    assert ctx and ctx.request_id == child_id
    patch_row(service, error_code="CONVERSATION_OUTCOME_UNKNOWN",
              waiting={"reason": "previous_result_unverified"}, original_failure_phase="cursor_read")
    service.text.recovery_reader = Mock(return_value={
        "status": "succeeded", "content": "late original answer", "binding_status": "bound",
        "conversation_id": original["conversation_id"], "parent_message_id": "late-final",
    })
    admission.execute(ctx)
    recovered = row(service)
    assert recovered["status"] == "succeeded" and recovered["content"] == "late original answer"
    assert recovered.get("error_code") is None and recovered.get("waiting") is None
    assert recovered["upstream_outcome"] == "completed"
    assert recovered["original_failure_phase"] == "cursor_read"
    assert calls == []


@pytest.mark.parametrize("read_available", [True, False])
def test_successful_missing_result_read_allows_bounded_automatic_same_session_retry(setup, read_available):
    from services.conversation_binding_service import ConversationBindingService
    from test.test_unknown_turn_recovery import document
    service, admission, calls = setup
    admission.clock.now = 1000
    patch_row(service, _sequence=0, _automatic_generation_recovery=True)

    def read(original):
        doc = document(original)
        del doc["mapping"][doc["current_node"]]
        doc.update(current_node=original["request_message_id"], is_archived=False)
        return ConversationBindingService._read_text_request_result(None, original, document=doc)

    service.text.recovery_reader = read
    service.process_one()
    original = row(service)
    assert original["recovery_reason"] == "REQUEST_RESULT_NOT_FOUND"
    assert original["_completion"]["reason"] == "COMPLETION_INVESTIGATING_ORIGINAL"
    admission.clock.now += service.STALL_SECONDS + service.INVESTIGATION_SECONDS
    if not read_available:
        service.text.recovery_reader = Mock(side_effect=ConnectionError("unavailable"))
    service.process_one()
    state = service.read("text", IDENTITY, "old-0")
    if not read_available:
        assert state["reason"] == "COMPLETION_ORIGINAL_READ_UNAVAILABLE"
        assert not state.get("replacement_id") and not calls
        assert row(service)["_result_last_checked_at"] == 1000
        return
    assert state.get("replacement_id"), state
    child_id = state["replacement_id"]
    child = row(service, request_id=child_id)
    assert row(service)["_result_last_checked_at"] == admission.clock.now
    assert row(service)["status"] == "unknown"
    assert row(service).get("upstream_outcome") != "not_sent"
    for key in ("conversation_id", "provider_account_identity", "provider_binding_id", "_work_key"):
        assert child[key] == original[key]
    assert child["parent_message_id"] == original["request_message_id"]
    admission.execute(admission.claim_next())
    assert len(calls) == 1
    assert service.read("text", IDENTITY, "old-0")["selected_id"] == child_id
    assert sum(e["stage"] == "send_call_started" for e in row(service)["_execution_timeline"]) == 1


def test_selected_completion_drops_active_wait_reason_and_ended_recovery_count(setup):
    service, admission, calls = setup
    assert admission.resource_snapshot()["queue"]["recovering_original"] == 1
    child_id = start(service)["replacement_id"]
    with service.store.transaction() as db:
        root = service.store.read_receipt(db, "text", "owner", "old-0")
        root["_completion"]["reason"] = "COMPLETION_ATTEMPT_PENDING"
        service.store.write_receipt(db, "text", "owner", "old-0", root)
    admission.execute(admission.claim_next())
    ready = service.read("text", IDENTITY, "old-0")
    assert ready["state"] == "result_ready" and "reason" not in ready
    assert admission.resource_snapshot()["queue"]["recovering_original"] == 0
    done = service.complete("text", IDENTITY, "old-0", child_id)
    assert done["state"] == "completed" and "reason" not in done
    assert row(service)["status"] == "unknown"
    # Already stored historical completion states must also project correctly.
    with service.store.transaction() as db:
        root = service.store.read_receipt(db, "text", "owner", "old-0")
        root["_completion"]["reason"] = "COMPLETION_ATTEMPT_PENDING"
        service.store.write_receipt(db, "text", "owner", "old-0", root)
    assert "reason" not in service.read("text", IDENTITY, "old-0")


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


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("proof_source", [None, "terminal_image_failure"])
@pytest.mark.parametrize("error_code", ["content_policy_violation", "CONTENT_POLICY_VIOLATION"])
def test_policy_refusal_never_replays_original_image_from_saved_cursor(failed_unsent_image, automatic, proof_source, error_code):
    service = failed_unsent_image
    rid = "repair-image"
    proof = {"conversation_id": "original-chat", "request_message_id": "original-user",
             "retry_parent_message_id": "refusal", "observed_at": service.clock()}
    if proof_source:
        proof["source"] = proof_source
    patch_row(service, "image", rid, retain_receipt=True, status="error",
              error_code=error_code, upstream_outcome="rejected",
              _submission_started=True, upstream_submission_started=True, upstream_unfinished=False,
              provider_binding_id="original-binding", provider_account_identity="original-account",
              client_conversation_id="original-client", conversation_id="original-chat",
              request_message_id="original-user", parent_message_id="refusal",
              _attempt_finished_at=service.clock(), _turn_reserved=False,
              _retry_cursor=proof, _completion=None, _automatic_generation_recovery=automatic)
    original = row(service, "image", rid)
    service.images._submit = Mock()
    if automatic:
        service.process_one()
    else:
        service.start("image", IDENTITY, rid, allow_unconfirmed_retry=True)
    after = row(service, "image", rid)
    assert after["_completion"]["state"] == "needs_attention"
    assert after["_completion"]["reason"] == "COMPLETION_ORIGINAL_NOT_RETRYABLE"
    assert not after["_completion"].get("replacement_id")
    assert after["_retry_cursor"] == original["_retry_cursor"]
    assert after["error_code"] == error_code
    service.images._submit.assert_not_called()


def test_late_policy_refusal_blocks_already_prepared_image_replacement(failed_unsent_image):
    from services.generation_completion import replacement_send_allowed
    service = failed_unsent_image
    patch_row(service, "image", "repair-image", status="error", error_code="content_policy_violation",
              upstream_outcome="rejected", _completion={"state": "replacement_pending", "replacement_id": "child"})
    with service.store.connect() as db:
        assert not replacement_send_allowed(service.store, db, "image", "owner", "child",
                                            {"_completion_of": "repair-image"})


def test_verified_empty_image_still_prepares_retry_in_original_conversation(failed_unsent_image):
    service = failed_unsent_image
    rid = "repair-image"
    patch_row(service, "image", rid, retain_receipt=True, status="error", error_code="NO_IMAGE_GENERATED",
              upstream_outcome="failed", _submission_started=True, upstream_submission_started=True,
              upstream_unfinished=False, provider_binding_id="original-binding",
              provider_account_identity="original-account", client_conversation_id="original-client",
              conversation_id="original-chat", request_message_id="original-user", parent_message_id="empty-final",
              _attempt_finished_at=service.clock(), _turn_reserved=False, _completion=None,
              _retry_cursor={"conversation_id": "original-chat", "request_message_id": "original-user",
                             "retry_parent_message_id": "empty-final", "observed_at": service.clock(),
                             "source": "terminal_image_failure"})
    service.images._submit = Mock()
    service.start("image", IDENTITY, rid)
    root = row(service, "image", rid)
    assert root["_completion"]["state"] == "replacement_pending"
    service.images._submit.assert_called_once()
    submitted = service.images._submit.call_args.kwargs
    assert submitted["client_task_id"] == root["_completion"]["replacement_id"]
    for field in ("provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id"):
        assert submitted["payload"][field] == root[field]
    assert submitted["payload"]["parent_message_id"] == "empty-final"
    assert submitted["payload"]["prompt"] == "retained input"


@pytest.mark.parametrize("kind", ["text", "image"])
@pytest.mark.parametrize("pending_delay", [None, 30])
def test_background_settles_saved_original_without_waiting_for_investigation(setup, kind, pending_delay):
    service, admission, calls = setup
    rid = "old-0"
    if kind == "image":
        service.images.submit_generation(IDENTITY, client_task_id=rid, prompt="retained image",
                                         model="gpt-image-2", size=None)
    result = {"content": "Recovered original answer"} if kind == "text" else {
        "data": [{"b64_json": "c2F2ZWQtb3JpZ2luYWw="}]}
    patch_row(service, kind, rid, status="succeeded" if kind == "text" else "success",
              **result, _completion={"state": "needs_attention", "next_at": service.clock()+pending_delay if pending_delay else None,
                                    "reason": "COMPLETION_ORIGINAL_READ_UNAVAILABLE"})
    before = row(service, kind, rid)
    service.text.read = Mock(side_effect=AssertionError("no original HTTP read"))
    service.images.resume_poll = Mock(side_effect=AssertionError("no original image read"))
    restarted = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=service.clock)
    restarted.process_one()
    after = row(service, kind, rid)
    assert after["_completion"]["state"] == "result_ready"
    assert after["_completion"]["selected_id"] == rid
    assert "reason" not in after["_completion"] and "replacement_id" not in after["_completion"]
    assert {k: v for k, v in after.items() if k != "_completion"} == {
        k: v for k, v in before.items() if k != "_completion"}
    restarted.process_one()
    assert row(service, kind, rid) == after
    assert calls == []


@pytest.mark.parametrize("guard", ["no_result", "paused", "suppressed"])
def test_background_settlement_never_reopens_unavailable_or_paused_original(setup, guard):
    service, admission, calls = setup
    changes = {"status": "succeeded", "content": "Recovered original answer"}
    if guard == "no_result":
        changes["content"] = ""
    elif guard == "paused":
        changes["_recovery_paused"] = True
    else:
        changes["_recovery_suppressed"] = True
    patch_row(service, **changes, _completion={"state": "needs_attention", "next_at": None,
                                              "reason": "COMPLETION_ORIGINAL_READ_UNAVAILABLE"})
    before = row(service)
    service.text.read = Mock(side_effect=AssertionError("must not resume original read"))
    service.process_one()
    assert row(service) == before and calls == []


def test_background_settlement_preserves_sent_replacement_and_other_ready_work(setup):
    service, admission, calls = setup
    child_id = start(service)["replacement_id"]
    state = {**row(service)["_completion"], "state": "needs_attention", "next_at": None}
    patch_row(service, status="succeeded", content="late original", _completion=state)
    patch_row(service, request_id=child_id, status="running", _submission_started=True)
    service.images.submit_generation(IDENTITY, client_task_id="other-image", prompt="other image",
                                     model="gpt-image-2", size=None)
    patch_row(service, "image", "other-image", status="success", data=[{"b64_json": "c2F2ZWQ="}],
              _completion={"state": "needs_attention", "next_at": None})
    service.text.read = Mock(side_effect=AssertionError("no original read"))
    service.process_one()
    assert "selected_id" not in row(service)["_completion"]
    assert row(service, request_id=child_id)["status"] == "running"
    assert row(service, "image", "other-image")["_completion"]["selected_id"] == "other-image"
    patch_row(service, request_id=child_id, status="succeeded", content="saved replacement")
    service.process_one()
    assert row(service)["_completion"]["selected_id"] == child_id
    assert row(service)["content"] == "late original" and calls == []


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
    import api.external_images as external_images
    service = failed_unsent_image
    monkeypatch.setattr(api, "get_generation_completion_service", lambda: service)
    monkeypatch.setattr(api, "require_identity", lambda authorization, request: {**IDENTITY, "id": authorization or "other"})
    monkeypatch.setattr(external_images, "require_identity", lambda authorization, request: {**IDENTITY, "id": authorization or "other"})
    monkeypatch.setattr(api, "require_image_policy", lambda *a, **kw: None)
    app = FastAPI()
    app.middleware("http")(external_image_boundary)
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


def test_admin_image_completion_only_reads_its_legacy_original(failed_unsent_image, monkeypatch):
    import api.generation_completion as api
    import api.external_images as external_images

    service = failed_unsent_image
    admin_identity = {"id": "admin", "role": "admin", "external_image_client": True}
    with service.store.transaction() as db:
        legacy = service.store.read_receipt(db, "image", "owner", "repair-image")
        source = service.store.load_input(legacy["_input_ref"])
        source["identity"] = admin_identity
        legacy.update(
            owner_id="admin",
            _input_ref=service.store.save_input(source),
            status="error",
            error_code="RESULT_UNRECOVERABLE",
            upstream_outcome="unknown",
            upstream_unfinished=False,
            recovery_no_result_reads=3,
            next_poll_at=0,
            conversation_id="legacy-conversation",
            request_message_id="legacy-request",
            provider_binding_id="legacy-binding",
            provider_account_identity="legacy-account",
            client_conversation_id="legacy-client",
            _submission_started=True,
            upstream_submission_started=True,
        )
        for key in ("_completion", "_attempt_finished_at", "_attempt_reason", "_image_thread",
                    "_recovery_paused", "_recovery_suppressed", "recovery_claim_id"):
            legacy.pop(key, None)
        service.store.write_receipt(db, "image", "admin", "repair-image", legacy)
        db.execute("DELETE FROM image_requests WHERE task_key=?", ("owner:repair-image",))

    monkeypatch.setattr(api, "get_generation_completion_service", lambda: service)
    monkeypatch.setattr(
        api, "require_identity",
        lambda authorization, request: admin_identity if authorization == "admin" else {**IDENTITY, "id": authorization or "other"},
    )
    monkeypatch.setattr(
        external_images, "require_identity",
        lambda authorization, request: admin_identity if authorization == "admin" else {**IDENTITY, "id": authorization or "other"},
    )
    monkeypatch.setattr(api, "require_image_policy", lambda *a, **kw: None)
    original_read = Mock()
    service.images.resume_poll = original_read
    app = FastAPI()
    app.middleware("http")(external_image_boundary)
    app.include_router(api.create_router("image"))
    app.include_router(api.create_router("text"))
    client = TestClient(app)
    image_endpoint = "/api/image-tasks/repair-image/completion"

    public_rejected = client.post(
        image_endpoint,
        headers={"Authorization": "admin", "X-Workbench-Image-Client": "1"},
        json={"action": "recover", "allow_unconfirmed_retry": False},
    )
    assert public_rejected.status_code == 403, public_rejected.text
    original_read.assert_not_called()

    oversized = client.post(
        image_endpoint, headers={"Authorization": "admin"},
        json={"action": "recover", "padding": "x" * 1024},
    )
    assert oversized.status_code == 413, oversized.text
    assert oversized.json()["detail"]["code"] == "COMPLETION_BODY_TOO_LARGE"
    original_read.assert_not_called()

    response = client.post(
        image_endpoint, headers={"Authorization": "admin"},
        json={"action": "recover", "allow_unconfirmed_retry": False},
    )
    assert response.status_code == 200, response.text
    original_read.assert_called_once_with(
        {"id": "admin"}, "repair-image", extra_timeout_secs=5,
        allow_unrecoverable_retry=True, completion_recheck=True,
    )
    with service.store.connect() as db:
        current = service.store.read_receipt(db, "image", "admin", "repair-image")
        assert current["recovery_no_result_reads"] == 3
        assert current["_completion"]["max_extra_requests"] == 0
        assert "replacement_id" not in current["_completion"]
        assert db.execute("SELECT count(*) FROM image_requests WHERE task_key LIKE 'admin:%'").fetchone()[0] == 1
    assert "replacement_id" not in response.json()
    assert service.text.admission.claim_next() is None

    before = copy.deepcopy(current)
    assert client.get(image_endpoint, headers={"Authorization": "admin"}).status_code == 403
    for body in (
        {"action": "complete", "selected_id": "repair-image", "results_saved": True, "reviewed": True},
        {"action": "rework", "selected_id": "repair-image"},
        {"action": "recover", "allow_unconfirmed_retry": True},
        {"action": "recover", "retry_not_sent_failure_at": 2900.0},
        {"action": "recover", "reviewed": False},
    ):
        rejected = client.post(image_endpoint, headers={"Authorization": "admin"}, json=body)
        assert rejected.status_code == 403, rejected.text
        with service.store.connect() as db:
            assert service.store.read_receipt(db, "image", "admin", "repair-image") == before

    text_rejected = client.post("/api/chat-requests/old-0/completion", headers={"Authorization": "admin"},
                                json={"action": "recover"})
    assert text_rejected.status_code == 403


def test_full_app_legacy_admin_original_receipt_stays_read_only_and_can_offer_terminal_cursor(
        failed_unsent_image, monkeypatch, tmp_path):
    import api.app as app_module
    import api.generation_completion as api
    import api.support as support
    from api.company_requests import PREFIX, company_identity
    from services.auth_service import AuthService
    from services.storage.json_storage import JSONStorageBackend

    service = failed_unsent_image
    admin_identity = {"id": "admin", "role": "admin"}
    auth = AuthService(JSONStorageBackend(tmp_path / "keys.json"))
    user_key, user_secret = auth.create_key(role="user", name="ordinary")
    monkeypatch.setattr(support, "auth_service", auth)
    monkeypatch.setattr(
        support, "_legacy_admin_identity",
        lambda token: admin_identity if token == "legacy-admin" else None,
    )
    completion_service = Mock(return_value=service)
    monkeypatch.setattr(api, "get_generation_completion_service", completion_service)
    monkeypatch.setattr(api, "require_image_policy", lambda *args, **kwargs: None)
    wake = Mock()
    monkeypatch.setattr(service.text.admission, "wake", wake)
    generation = Mock()
    service.images.generation_handler = generation

    def legacy_receipt(owner, request_id, *, retain=False, **changes):
        with service.store.transaction() as db:
            original = service.store.read_receipt(db, "image", "owner", "repair-image")
            receipt = copy.deepcopy(original)
            receipt.pop("_completion", None)
            receipt.pop("_completion_of", None)
            receipt.pop("_route", None)
            values = {
                "id": request_id,
                "owner_id": owner,
                "retain_receipt": retain,
                "mode": "edit",
                "model": "gpt-image-2",
                "status": "error",
                "error_code": "RESULT_UNRECOVERABLE",
                "upstream_outcome": "unknown",
                "upstream_unfinished": False,
                "recovery_no_result_reads": 3,
                "provider_binding_id": "binding-legacy",
                "provider_account_identity": "account-legacy",
                "client_conversation_id": "client-legacy",
                "conversation_id": "conversation-legacy",
                "request_message_id": "request-legacy",
                "binding_status": "bound",
            }
            values.update(changes)
            receipt.update(values)
            if retain is None:
                receipt.pop("retain_receipt", None)
            service.store.write_receipt(db, "image", owner, request_id, receipt)
        service.images._tasks[f"{owner}:{request_id}"] = receipt
        return receipt

    legacy_receipt("admin", "legacy-false")
    legacy_receipt("admin", "legacy-missing", retain=None)
    legacy_receipt(user_key["id"], "ordinary-false")
    connector = "1f084f01-d4b2-4080-8bce-b926f31cc454"
    company_owner = company_identity("company", "worker", connector)["id"]
    legacy_receipt(company_owner, "company-false")
    legacy_receipt("admin", "wrong-owner", owner_id="other")
    legacy_receipt("admin", "wrong-model", model="gpt-image-1")
    legacy_receipt("admin", "codex-route", _route="codex")
    legacy_receipt("admin", "missing-binding", provider_binding_id="")

    reads = []
    def settle_original(identity, request_id, **_kwargs):
        reads.append((identity, request_id))
        service.images._update_task(
            f"admin:{request_id}", status="running", error_code="RESULT_UNRECOVERABLE",
            upstream_outcome="unknown", upstream_unfinished=True,
        )
    service.images.resume_poll = settle_original

    app = app_module.create_app()
    client = TestClient(app)
    endpoint = "/api/image-tasks/legacy-false/completion"
    admin_headers = {"Authorization": "Bearer legacy-admin"}
    with service.store.connect() as db:
        service._root(db, "image", "admin", "legacy-false", read_only_original=True)
    response = client.post(endpoint, headers=admin_headers, json={"action": "recover", "allow_unconfirmed_retry": False})
    assert completion_service.called
    assert response.status_code == 200, response.text
    assert reads == [({"id": "admin"}, "legacy-false")]
    assert response.json()["state"] == "checking_original"
    assert generation.call_count == 0 and wake.call_count == 0
    service.images._update_task(
        "admin:legacy-false", status="error", error_code="NO_IMAGE_GENERATED", upstream_outcome="failed",
        upstream_unfinished=False, _completion_read_at=service.clock(),
    )
    with service.store.transaction() as db:
        current = service.store.read_receipt(db, "image", "admin", "legacy-false")
        current["_completion"]["next_at"] = 0
        service.store.write_receipt(db, "image", "admin", "legacy-false", current)
    service.images._tasks["admin:legacy-false"] = current
    service.images.resume_poll = Mock(side_effect=AssertionError("terminal original must not be reread"))
    restarted = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=service.clock)
    restarted.advance("image", "admin", "legacy-false")
    with service.store.connect() as db:
        current = service.store.read_receipt(db, "image", "admin", "legacy-false")
    assert current["retain_receipt"] is False
    assert current["error_code"] == "NO_IMAGE_GENERATED"
    assert current["_completion"].get("state") == "needs_attention"
    assert current["_completion"].get("reason") == "COMPLETION_ORIGINAL_ONLY"
    assert current["_completion"].get("next_at") is None
    assert current["_completion"].get("read_only_original") is True
    assert current["_completion"].get("max_extra_requests") == 0
    assert current["_completion"].get("allow_unconfirmed_retry") is False
    assert not current["_completion"].get("replacement_id")
    assert not current["_completion"].get("prepared_input")
    assert generation.call_count == 0 and wake.call_count == 0
    assert service.text.admission.claim_next() is None

    assert restarted.read("image", admin_identity, "legacy-false")["reason"] == "COMPLETION_ORIGINAL_ONLY"
    before_upgrade = copy.deepcopy(current["_completion"])
    with pytest.raises(CompletionError, match="COMPLETION_POLICY_CONFLICT"):
        restarted.start("image", admin_identity, "legacy-false", allow_unconfirmed_retry=True)
    with service.store.connect() as db:
        assert service.store.read_receipt(db, "image", "admin", "legacy-false")["_completion"] == before_upgrade
    assert generation.call_count == 0 and wake.call_count == 0
    assert service.text.admission.claim_next() is None

    service.images._update_task("admin:legacy-false", _claim_id="read-lease", _claim_until=time.time() + 30)

    class TerminalCursorBackend:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def _has_image_asset_pointer(_payload):
            return False

        def _get_conversation(self, _conversation_id):
            return {
                "conversation_id": "conversation-legacy", "is_archived": False, "current_node": "terminal",
                "mapping": {
                    "request-legacy": {"parent": "anchor", "message": {
                        "id": "request-legacy", "author": {"role": "user"},
                    }},
                    "terminal": {"parent": "request-legacy", "message": {
                        "id": "terminal", "author": {"role": "assistant"},
                        "status": "finished_successfully", "end_turn": True,
                    }},
                },
            }

        def close(self):
            pass

    with (
        patch("services.account_service.account_service.get_bound_account_identity", return_value="account-legacy"),
        patch("services.account_service.account_service.get_bound_text_access_token", return_value="read-token"),
        patch("services.account_service.account_service.conversation_binding_lock", return_value=nullcontext()),
        patch("services.openai_backend_api.OpenAIBackendAPI", TerminalCursorBackend),
    ):
        assert service.images.failure_continuation(admin_identity, "legacy-false") is None
        service.images._update_task("admin:legacy-false", _claim_until=time.time() - 1)
        assert service.images.failure_continuation(admin_identity, "legacy-false") == {
            "source_task_id": "legacy-false", "source_request_message_id": "request-legacy",
            "provider_binding_id": "binding-legacy", "provider_account_identity": "account-legacy",
            "client_conversation_id": "client-legacy", "conversation_id": "conversation-legacy",
            "parent_message_id": "terminal",
        }
    assert generation.call_count == 0 and wake.call_count == 0

    legacy_receipt(
        "admin", "legacy-success", status="success", error_code="", upstream_outcome="completed",
        data=[{"b64_json": "c2F2ZWQ="}],
    )
    with service.store.transaction() as db:
        succeeded = service.store.read_receipt(db, "image", "admin", "legacy-success")
        succeeded["_completion"] = {
            "state": "checking_original", "next_at": 0,
            "read_only_original": True, "max_extra_requests": 0, "allow_unconfirmed_retry": False,
        }
        service.store.write_receipt(db, "image", "admin", "legacy-success", succeeded)
    restarted.advance("image", "admin", "legacy-success")
    selected = restarted.read("image", admin_identity, "legacy-success")
    assert selected["state"] == "result_ready"
    assert selected["selected_id"] == "legacy-success"
    assert selected["result"] == {"id": "legacy-success", "status": "success", "image_count": 1}
    after_success_restart = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=service.clock)
    assert after_success_restart.read("image", admin_identity, "legacy-success")["selected_id"] == "legacy-success"
    assert generation.call_count == 0 and wake.call_count == 0
    assert service.text.admission.claim_next() is None

    service.images.resume_poll = settle_original
    missing = client.post(
        "/api/image-tasks/legacy-missing/completion", headers=admin_headers,
        json={"action": "recover", "allow_unconfirmed_retry": False},
    )
    assert missing.status_code == 200, missing.text
    public = client.post(endpoint, headers={**admin_headers, "X-Workbench-Image-Client": "1"}, json={"action": "recover"})
    assert public.status_code == 403
    oversized = client.post(endpoint, headers=admin_headers, json={"action": "recover", "padding": "x" * 1024})
    assert oversized.status_code == 413
    for body in (
        {"action": "recover", "allow_unconfirmed_retry": True},
        {"action": "complete", "selected_id": "legacy-false", "results_saved": True, "reviewed": True},
        {"action": "rework", "selected_id": "legacy-false"},
    ):
        assert client.post(endpoint, headers=admin_headers, json=body).status_code == 403
    assert client.post(
        "/api/image-tasks/ordinary-false/completion", headers={"Authorization": "Bearer " + user_secret}, json={"action": "recover"},
    ).status_code == 409
    company_headers = {
        "Authorization": "Bearer legacy-admin", "X-Workbench-Company-Org": "company",
        "X-Workbench-Company-User": "worker", "X-Workbench-Company-Connector": connector,
        "X-Workbench-Expected-User": "worker",
    }
    assert client.post(PREFIX + "/api/image-tasks/company-false/completion", headers=company_headers,
                       json={"action": "recover"}).status_code == 409
    for request_id in ("wrong-owner", "wrong-model", "codex-route", "missing-binding"):
        assert client.post(f"/api/image-tasks/{request_id}/completion", headers=admin_headers,
                           json={"action": "recover"}).status_code == 409


def test_admin_image_completion_does_not_queue_a_proven_not_sent_original(failed_unsent_image, monkeypatch):
    import api.generation_completion as api

    service = failed_unsent_image
    admin_identity = {"id": "admin", "role": "admin", "external_image_client": True}
    with service.store.transaction() as db:
        receipt = service.store.read_receipt(db, "image", "owner", "repair-image")
        source = service.store.load_input(receipt["_input_ref"])
        source["identity"] = admin_identity
        receipt.update(owner_id="admin", _input_ref=service.store.save_input(source))
        receipt.pop("_completion", None)
        service.store.write_receipt(db, "image", "admin", "repair-image", receipt)
        db.execute("DELETE FROM image_requests WHERE task_key=?", ("owner:repair-image",))

    monkeypatch.setattr(api, "get_generation_completion_service", lambda: service)
    monkeypatch.setattr(api, "require_identity", lambda _authorization, request: admin_identity)
    monkeypatch.setattr(api, "require_image_policy", lambda *a, **kw: None)
    wake = Mock()
    monkeypatch.setattr(service.text.admission, "wake", wake)
    original_read = Mock()
    service.images.resume_poll = original_read
    app = FastAPI()
    app.include_router(api.create_router("image"))
    client = TestClient(app)

    response = client.post("/api/image-tasks/repair-image/completion", headers={"Authorization": "admin"},
                           json={"action": "recover", "allow_unconfirmed_retry": False})
    assert response.status_code == 200, response.text
    original_read.assert_not_called()
    wake.assert_not_called()
    with service.store.connect() as db:
        current = service.store.read_receipt(db, "image", "admin", "repair-image")
        assert current["status"] == "error"
        assert current["upstream_outcome"] == "not_submitted"
        assert current["_completion"]["reason"] == "COMPLETION_ORIGINAL_ONLY"
        assert current["_completion"]["max_extra_requests"] == 0
        assert "replacement_id" not in current["_completion"]
        assert db.execute("SELECT count(*) FROM image_requests WHERE task_key LIKE 'admin:%'").fetchone()[0] == 1
    assert service.text.admission.claim_next() is None


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


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("finish_after,no_result_reads", [(61, 4), (601, 3)])
def test_original_only_read_handoff_stops_after_one_fresh_check(
        failed_unsent_image, read_only, finish_after, no_result_reads):
    service = failed_unsent_image
    rid = "repair-image"
    patch_row(service, "image", rid, retain_receipt=True, _completion=None,
              upstream_outcome="unknown", upstream_unfinished=False,
              _submission_started=True, upstream_submission_started=True,
              _turn_reserved=True, _executing=False, recovery_claim_id=None,
              conversation_id="original-conversation", request_message_id="original-request",
              provider_binding_id="original-binding", provider_account_identity="original-account",
              client_conversation_id="original-client", next_poll_at=0,
              recovery_no_result_reads=3, _completion_read_at=0)
    with service.store.transaction() as db:
        original = service.store.read_receipt(db, "image", "owner", rid)
        original["_public_session_ref"] = "original-image-session"
        work = ensure_work(service.store, db, "image", "owner", rid, original)
        work["slot_held"] = True
        service.store.set_runtime(db, work["key"], work)
        service.store.write_receipt(db, "image", "owner", rid, original)
        other = service.store.read_receipt(db, "text", "owner", "old-0")
        other_work = service.store.runtime(db, other["_work_key"])
        other_work["slot_held"] = True
        service.store.set_runtime(db, other_work["key"], other_work)
    before = row(service, "image", rid)
    service.text.admission.wake = Mock()
    service._prepare = Mock(side_effect=AssertionError("no replacement allowed"))

    def start_original_read(*args, **kwargs):
        patch_row(service, "image", rid, status="running", _executing=True)

    service.images.resume_poll = Mock(side_effect=start_original_read)
    service.start("image", IDENTITY, rid, original_only=True, read_only_original=read_only)
    assert row(service, "image", rid)["_completion"]["state"] == "checking_original"
    # The original reader survives an orchestration restart. It must finish;
    # the next due orchestration pass must not launch it all over again.
    restarted = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=service.clock)
    restarted._prepare = service._prepare
    service.clock.now += 30
    restarted.advance("image", "owner", rid)
    assert row(service, "image", rid)["_executing"] is True
    assert not row(service, "image", rid).get("_attempt_finished_at")
    patch_row(service, "image", rid, status="error", _executing=False,
              error_code="RESULT_UNRECOVERABLE", recovery_no_result_reads=no_result_reads)
    service.clock.now = 3000 + finish_after
    for _ in range(4):
        restarted.advance("image", "owner", rid)
        service.clock.now += 301
    after = row(service, "image", rid)
    assert service.images.resume_poll.call_count == 1
    assert after["_completion"]["state"] == "needs_attention"
    assert after["_completion"]["reason"] == "COMPLETION_ORIGINAL_ONLY"
    assert after["_completion"]["next_at"] is None
    assert after["_attempt_finished_at"] and after["_turn_reserved"] is False
    assert after["upstream_outcome"] == "unknown"
    assert after["recovery_no_result_reads"] == no_result_reads
    with service.store.connect() as db:
        assert service.store.runtime(db, work["key"])["slot_held"] is False
        assert service.store.runtime(db, other_work["key"])["slot_held"] is True
    for key in ("conversation_id", "request_message_id", "provider_binding_id",
                "provider_account_identity", "_input_ref", "active_attempt_deadline_at"):
        assert after[key] == before[key]
    service._prepare.assert_not_called()
    if read_only:
        service.text.admission.wake.assert_not_called()
    # A late saved original can still be selected; stopping queries is not a
    # statement that the upstream generation failed or never existed.
    patch_row(service, "image", rid, status="success", data=[{"b64_json": "saved-original"}],
              upstream_outcome="generated")
    restarted.advance("image", "owner", rid)
    assert row(service, "image", rid)["_completion"]["selected_id"] == rid
    assert service.images.resume_poll.call_count == 1


@pytest.mark.parametrize("asset", [
    {"result_file_ids": ["original-file"]},
    {"result_sediment_ids": ["original-sediment"]},
    {"_pending_image_result_ids": {"file_ids": ["pending-original-file"]}},
])
def test_original_only_read_budget_preserves_captured_result(failed_unsent_image, asset):
    service = failed_unsent_image
    rid = "repair-image"
    patch_row(service, "image", rid, retain_receipt=True, upstream_outcome="unknown",
              conversation_id="original-conversation", request_message_id="original-request",
              _submission_started=True, upstream_submission_started=True,
              _executing=False, recovery_claim_id=None, next_poll_at=0,
              recovery_no_result_reads=4, _completion_read_at=0,
              _completion={"state": "checking_original", "allow_unconfirmed_retry": False,
                           "max_extra_requests": 0, "next_at": 0,
                           "original_read_requested_at": 2000,
                           "original_read_no_result_baseline": 3}, **asset)
    service._prepare = Mock(side_effect=AssertionError("never regenerate a captured result"))

    def collect_original(*args, **kwargs):
        patch_row(service, "image", rid, status="success", data=[{"b64_json": "saved-original"}],
                  upstream_outcome="generated")

    service.images.resume_poll = Mock(side_effect=collect_original)
    service.advance("image", "owner", rid)
    service.images.resume_poll.assert_called_once()
    service._prepare.assert_not_called()
    after = row(service, "image", rid)
    assert after["_completion"]["selected_id"] == rid
    assert not after.get("_attempt_finished_at")


@pytest.mark.parametrize("pause_key", ["_recovery_paused", "_recovery_suppressed"])
def test_original_only_read_errors_obey_cooldown_and_persistent_investigation_window(
        failed_unsent_image, pause_key):
    service = failed_unsent_image
    rid = "repair-image"
    patch_row(service, "image", rid, retain_receipt=True, upstream_outcome="unknown",
              conversation_id="original-conversation", request_message_id="original-request",
              _submission_started=True, upstream_submission_started=True,
              _executing=False, recovery_claim_id=None, recovery_no_result_reads=3,
              _completion_read_at=0, next_poll_at=3060,
              _completion={"state": "checking_original", "allow_unconfirmed_retry": False,
                           "max_extra_requests": 0, "next_at": 0,
                           "original_read_requested_at": 3000,
                           "original_read_no_result_baseline": 3})
    original_deadline = row(service, "image", rid)["active_attempt_deadline_at"]
    service.images.resume_poll = Mock(
        side_effect=lambda *a, **k: patch_row(service, "image", rid, status="running", _executing=True))
    # Observe the 429/transport backoff without converting it into a qualified
    # no-result read. The reader still owns a live handoff during this pass.
    patch_row(service, "image", rid, _executing=True)
    service.clock.now = 3030
    service.advance("image", "owner", rid)
    service.images.resume_poll.assert_not_called()
    patch_row(service, "image", rid, _executing=False)
    service.clock.now = 3060
    service.advance("image", "owner", rid)
    service.images.resume_poll.assert_called_once()
    patch_row(service, "image", rid, status="error", _executing=False,
              next_poll_at=3120, **{pause_key: True})
    restarted = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=service.clock)
    service.clock.now = 3400
    restarted.advance("image", "owner", rid)
    assert not row(service, "image", rid).get("_attempt_finished_at")
    patch_row(service, "image", rid, **{pause_key: False})
    restarted.advance("image", "owner", rid)
    after = row(service, "image", rid)
    assert service.images.resume_poll.call_count == 1
    assert after["_completion"]["reason"] == "COMPLETION_ORIGINAL_ONLY"
    assert after["_completion"]["next_at"] is None
    assert after["_completion"]["original_read_requested_at"] == 3000
    assert after["recovery_no_result_reads"] == 3
    assert after["upstream_outcome"] == "unknown"
    assert after["active_attempt_deadline_at"] == original_deadline


def test_original_only_image_policy_survives_restart_and_prevents_successor(setup):
    service, admission, calls = setup
    service.images.submit_generation(IDENTITY, client_task_id="original-only", prompt="retained input",
                                     model="gpt-image-2", size=None)
    patch_row(service, "image", "original-only", status="error", upstream_outcome="failed",
              _submission_started=True, upstream_submission_started=True,
              _completion={"state": "checking_original", "allow_unconfirmed_retry": False,
                           "automatic_failure_retry": True, "max_extra_requests": 1, "next_at": None})
    service._prepare = Mock(side_effect=AssertionError("must not prepare a successor"))
    result = service.start("image", IDENTITY, "original-only", original_only=True)
    assert result["reason"] == "COMPLETION_ORIGINAL_ONLY" and result["max_extra_requests"] == 0
    assert "replacement_id" not in result
    patch_row(service, "image", "original-only", _completion={
        **row(service, "image", "original-only")["_completion"], "state": "checking_original", "next_at": 0})
    restarted = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=service.clock)
    restarted._prepare = service._prepare
    restarted.advance("image", "owner", "original-only")
    assert "replacement_id" not in row(service, "image", "original-only")["_completion"]
    service._prepare.assert_not_called()
    with service.store.connect() as db:
        assert db.execute("SELECT count(*) FROM image_requests").fetchone()[0] == 1
    assert calls == []


def test_image_send_without_cursor_stops_local_investigation_and_preserves_unknown(setup):
    service, admission, calls = setup
    rid = "missing-cursor-image"
    service.images.submit_generation(IDENTITY, client_task_id=rid, prompt="retained image input",
                                     model="gpt-image-2", size=None)
    patch_row(service, "image", rid, status="error", error_code="CONVERSATION_OUTCOME_UNKNOWN",
              upstream_outcome="unknown", upstream_unfinished=True, conversation_id="",
              request_message_id="original-user-message", _submission_started=True,
              upstream_submission_started=True, _turn_reserved=True, _executing=False,
              recovery_error_code="RECOVERY_READ_FAILED",
              _execution_timeline=[{"stage": "send_call_started", "at": 10}],
              _completion={"state": "checking_original", "allow_unconfirmed_retry": False,
                           "automatic_failure_retry": True, "max_extra_requests": 1, "next_at": 0})
    original = row(service, "image", rid)
    service.images.resume_poll = Mock(side_effect=AssertionError("no original cursor to read"))
    service._prepare = Mock(side_effect=AssertionError("must not replace unknown original"))
    service.advance("image", "owner", rid)
    result = service.read("image", IDENTITY, rid)
    assert result["state"] == "needs_attention"
    assert result["reason"] == "COMPLETION_ORIGINAL_CURSOR_UNAVAILABLE"
    assert result["next_at"] is None and "replacement_id" not in result
    assert result["original_status"] == "error" and not result["original_turn_ended"]
    assert result["stop"]["confirmed"] is False
    after = row(service, "image", rid)
    for key in ("_input_ref", "_execution_timeline", "request_message_id", "conversation_id",
                "provider_account_identity", "provider_binding_id", "upstream_outcome",
                "error_code", "recovery_error_code"):
        assert after.get(key) == original.get(key)
    assert after["_attempt_finished_at"] and not after["_turn_reserved"]
    assert result["local_reservation"] == "released"
    restarted = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=service.clock)
    restarted.process_one()
    restarted.process_one()
    restarted.advance("image", "owner", rid)
    service.images.resume_poll.assert_not_called()
    service._prepare.assert_not_called()
    with service.store.connect() as db:
        assert db.execute("SELECT count(*) FROM image_requests").fetchone()[0] == 1
    assert calls == []


@pytest.mark.parametrize("code", ["RECOVERY_READ_FAILED", "RECOVERY_RATE_LIMITED"])
def test_image_cursor_scan_wait_survives_completion_ticks_and_restart(setup, code):
    service, admission, calls = setup
    rid = "cursor-scan-wait"
    service.images.submit_generation(IDENTITY, client_task_id=rid, prompt="retained image input",
                                     model="gpt-image-2", size=None)
    patch_row(service, "image", rid, status="error", upstream_outcome="unknown", conversation_id="",
              provider_binding_id="original-binding", provider_account_identity="original-account",
              client_conversation_id="original-client-session", request_message_id="original-user-message",
              _submission_started=True, upstream_submission_started=True, _executing=False,
              recovery_error_code=code, next_poll_at=service.clock()+73,
              _recovery_conversation_scan={"next_offset": 20},
              _image_cursor_lookup_error="REQUEST_CONVERSATION_SCAN_INCOMPLETE",
              _completion={"state": "checking_original", "allow_unconfirmed_retry": False,
                           "automatic_failure_retry": True, "max_extra_requests": 1, "next_at": 0})
    service.images.resume_poll = Mock()
    service._prepare = Mock(side_effect=AssertionError("scan must not generate"))
    service.advance("image", "owner", rid)
    first = row(service, "image", rid)
    assert first["_completion"]["state"] == "checking_original"
    assert first["_completion"]["next_at"] == first["next_poll_at"]
    restarted = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=service.clock)
    restarted._prepare = service._prepare
    admission.clock.now += 72
    restarted.process_one()
    service.images.resume_poll.assert_not_called()
    admission.clock.now += 1
    restarted.process_one()
    service.images.resume_poll.assert_called_once_with({"id": "owner"}, rid, extra_timeout_secs=5,
                                                     allow_unrecoverable_retry=True, completion_recheck=True)
    final = row(service, "image", rid)
    assert final["_completion"]["state"] == "checking_original"
    assert final["_recovery_conversation_scan"] == first["_recovery_conversation_scan"]
    assert "replacement_id" not in final["_completion"]
    service._prepare.assert_not_called()
    assert calls == []


def test_original_only_policy_does_not_cancel_existing_successor(failed_unsent_image):
    service = failed_unsent_image
    prior = row(service, "image", "repair-image")
    patch_row(service, "image", "repair-image", _completion={
        **prior["_completion"], "replacement_id": "existing-authorized-attempt"})
    before = row(service, "image", "repair-image")
    with pytest.raises(CompletionError, match="COMPLETION_POLICY_CONFLICT"):
        service.start("image", IDENTITY, "repair-image", original_only=True)
    assert row(service, "image", "repair-image") == before
