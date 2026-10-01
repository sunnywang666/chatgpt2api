"""Automatic one-retry handling for observed empty native Chat responses."""
import copy
from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from services.conversation_binding_service import ConversationBindingService, ConversationBindingError
from services.generation_completion import GenerationCompletionService
from services.request_context import current_request
from test.test_generation_completion import setup, row, patch_row, IDENTITY
from test.test_unknown_turn_recovery import document
from test.test_stalled_text_diagnostics import external_continuation, mixed_chain


def empty_reader(original, *, external=False, mixed=False):
    doc = document(original)
    final = doc["current_node"]
    doc["mapping"][final]["message"].update(status="in_progress", end_turn=None)
    if mixed:
        mixed_chain(doc)
        if mixed == "stale_reasoning":
            for node in doc["mapping"].values():
                message = node.get("message") or {}
                if (message.get("content") or {}).get("content_type") == "thoughts":
                    message["status"] = "in_progress"
    if external:
        external_continuation(doc)
    return ConversationBindingService._read_text_request_result(None, original, document=doc)


def seed_empty(service, *, external=False, mixed=False, end="done", legacy=False):
    events = [{"stage": "send_call_started", "at": 10},
              {"stage": "response_headers_received", "at": 11},
              {"stage": "task_finished", "at": 13}]
    if not legacy:
        events.insert(2, {"stage": "stream_finished", "at": 12, "stream_end": end,
                          "sse_data_count": 3, "sse_parse_errors": 0, "sse_error_event": False})
    patch_row(service, _sequence=0, _execution_timeline=events,
              **({"_result_wait_ended_at": 1000, "_result_no_progress_reads": 3} if legacy else {}))
    service.text.recovery_reader = lambda original: empty_reader(original, external=external, mixed=mixed)
    service.text.read("owner", "old-0")


@pytest.mark.parametrize("external,mixed", [(False, False), (True, False), (False, True), (True, "stale_reasoning")])
def test_empty_stream_automatically_retries_same_conversation_and_transfers_own_slot(setup, external, mixed):
    service, admission, calls = setup
    seed_empty(service, external=external, mixed=mixed)
    original = row(service)
    assert original["status"] == "unknown" and not original.get("_upstream_terminal")
    assert original["_completion"]["automatic_empty_retry"]
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    service.process_one()  # no external recovery action required
    state = service.read("text", IDENTITY, "old-0")
    child_id = state["replacement_id"]
    assert state["conversation_mode"] == "original" and not state["original_turn_ended"]
    child = row(service, request_id=child_id)
    assert child["parent_message_id"] == ("external-final" if external else "final-user-0")
    for key in ("provider_account_identity", "provider_binding_id", "conversation_id", "_work_key"):
        assert child[key] == original[key]
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    restarted = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=admission.clock)
    assert restarted.start("text", IDENTITY, "old-0")["replacement_id"] == child_id
    def run(body, on_cursor):
        current_request.get().before_send()
        calls.append(copy.deepcopy(body))
        return {"content": "Actual complete result", "conversation_id": original["conversation_id"],
                "parent_message_id": "retry-final", "_upstream_terminal": True, "upstream_outcome": "completed"}
    service.text.runner = run
    ctx = admission.claim_next()
    assert ctx and ctx.request_id == child_id
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    admission.execute(ctx)
    assert len(calls) == 1
    assert calls[0]["messages"] == service.store.load_input(original["_input_ref"])["messages"]
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert row(service)["status"] == "unknown" and not row(service).get("content")
    assert restarted.read("text", IDENTITY, "old-0")["selected_id"] == child_id
    completed = restarted.complete("text", IDENTITY, "old-0", child_id)
    assert completed["state"] == "completed" and not completed["work"]["slot_held"]
    assert admission.claim_next() is None


@pytest.mark.parametrize("end", ["eof", "transport_error", "hard_timeout", "consumer_closed", None])
def test_transport_uncertainty_does_not_become_confirmed_empty(setup, end):
    service, _, _ = setup
    seed_empty(service, end=end)
    assert "_completion" not in row(service)
    service.process_one()
    assert service.text._verified_retryable_empty(row(service)) is None


def test_old_closed_response_with_bounded_empty_observation_can_retry(setup):
    service, _, _ = setup
    seed_empty(service, external=True, legacy=True)
    assert row(service)["_completion"]["automatic_empty_retry"]
    service.process_one()
    assert service.read("text", IDENTITY, "old-0")["replacement_id"]


def test_empty_retry_is_not_retried_again_and_preserves_both_records(setup):
    service, admission, calls = setup
    seed_empty(service)
    service.process_one()
    child_id = service.read("text", IDENTITY, "old-0")["replacement_id"]
    def run(body, on_cursor):
        ctx = current_request.get()
        ctx.before_send()
        ctx.record_stage("send_call_started")
        ctx.record_stage("response_headers_received", http_status=200)
        ctx.record_stage("stream_finished", stream_end="done", sse_data_count=2, sse_parse_errors=0, sse_error_event=False)
        calls.append(body)
        raise ConversationBindingError("empty", code="CONVERSATION_OUTCOME_UNKNOWN")
    service.text.runner = run
    admission.execute(admission.claim_next())
    service.text.recovery_reader = lambda receipt: empty_reader(receipt)
    service.text.read("owner", child_id)
    assert row(service, request_id=child_id)["status"] == "unknown"
    assert "_completion" not in row(service, request_id=child_id)
    admission.clock.now += 31
    service.process_one()
    state = service.read("text", IDENTITY, "old-0")
    assert state["reason"] == "COMPLETION_ATTEMPT_EXHAUSTED"
    assert state["replacement_id"] == child_id and not state.get("selected_id")
    assert len(calls) == 1 and admission.claim_next() is None
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert state["local_reservation"] == "released"
    from services.pool_admission import unfinished
    assert unfinished("text", row(service, request_id=child_id))
    public = service.text.read("owner", child_id)
    assert public["execution"]["resources"]["account_turn"] == "released"
    assert public["execution"]["resources"]["conversation"] == "protected"
    paused = service.lifecycle.update("text", IDENTITY, child_id, "paused")
    assert paused["state"] == "paused" and not paused["slot_held"]
    assert row(service)["status"] == row(service, request_id=child_id)["status"] == "unknown"


@pytest.mark.parametrize("verified", [True, False])
def test_user_can_pause_confirmed_empty_without_clearing_unknown(setup, verified):
    from services.work_lifecycle import WorkLifecycleError
    from services.pool_admission import unfinished
    service, admission, _ = setup
    seed_empty(service, end="done" if verified else "transport_error")
    if not verified:
        patch_row(service, _turn_reserved=False)
        with pytest.raises(WorkLifecycleError, match="WORK_TURN_UNFINISHED"):
            service.lifecycle.update("text", IDENTITY, "old-0", "paused")
        assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
        return
    paused = service.lifecycle.update("text", IDENTITY, "old-0", "paused")
    assert paused["state"] == "paused" and not paused["slot_held"]
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert unfinished("text", row(service)) and row(service)["status"] == "unknown"
    restarted = GenerationCompletionService(service.text,service.images,service.lifecycle,clock=admission.clock)
    restarted.process_one()
    assert not restarted.read("text",IDENTITY,"old-0").get("replacement_id")


@pytest.mark.parametrize("case", ["head_drift", "late_original", "missing_proof"])
def test_retry_rechecks_empty_and_current_head_before_send(setup, case):
    service, admission, calls = setup
    seed_empty(service)
    service.process_one()
    child_id = service.read("text", IDENTITY, "old-0")["replacement_id"]
    def changed(receipt):
        if case == "head_drift":
            return empty_reader(receipt, external=True)
        doc = document(receipt)
        if case == "late_original":
            doc["mapping"][doc["current_node"]]["message"]["content"]["parts"] = ["late actual answer"]
        else:
            doc["mapping"][doc["current_node"]]["message"].update(status="in_progress", end_turn=None)
            doc["mapping"][doc["current_node"]]["message"]["content"] = {"content_type": "unknown"}
        return ConversationBindingService._read_text_request_result(None, receipt, document=doc)
    service.text.recovery_reader = changed
    admission.execute(admission.claim_next())
    assert calls == []
    assert row(service, request_id=child_id)["upstream_outcome"] == "not_sent"


def test_late_original_after_retry_send_cannot_strand_retry(setup):
    service, admission, calls = setup
    seed_empty(service)
    service.process_one()
    child_id = service.read("text", IDENTITY, "old-0")["replacement_id"]
    def run(body, on_cursor):
        current_request.get().before_send()
        calls.append(body)
        patch_row(service, status="succeeded", content="late original", _upstream_terminal=True,
                  upstream_outcome="completed")
        during = service.read("text", IDENTITY, "old-0")
        assert not during.get("selected_id")
        assert row(service, request_id=child_id)["_submission_started"]
        assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
        return {"content": "retry result", "conversation_id": row(service)["conversation_id"],
                "parent_message_id": "retry-final", "_upstream_terminal": True, "upstream_outcome": "completed"}
    service.text.runner = run
    admission.execute(admission.claim_next())
    assert service.read("text", IDENTITY, "old-0")["selected_id"] == child_id
    assert row(service)["content"] == "late original" and len(calls) == 1
    assert service.complete("text", IDENTITY, "old-0", child_id)["state"] == "completed"


@pytest.mark.parametrize("drift", [False, True])
def test_binding_pre_send_reads_real_empty_proof_before_post(setup, monkeypatch, drift):
    from services import conversation_binding_service as binding
    service, admission, calls = setup
    seed_empty(service)
    service.process_one()
    child_id = service.read("text", IDENTITY, "old-0")["replacement_id"]
    original, child = row(service), row(service, request_id=child_id)
    doc = document(original)
    doc["is_archived"] = False
    doc["mapping"][doc["current_node"]]["message"].update(status="in_progress", end_turn=None)
    changed = copy.deepcopy(doc)
    if drift:
        external_continuation(changed)
    backend = Mock()
    backend._get_conversation.side_effect = [doc, changed]
    monkeypatch.setattr(binding, "OpenAIBackendAPI", lambda **kw: backend)
    monkeypatch.setattr(binding.account_service, "get_bound_account_identity", lambda *a: original["provider_account_identity"])
    monkeypatch.setattr(binding.account_service, "get_bound_text_access_token", lambda *a, **kw: "fixture")
    monkeypatch.setattr(binding.account_service, "conversation_binding_lock", lambda *a: nullcontext())
    def events(actual, **kw):
        actual.text_pre_send_check(object())
        calls.append("POST")
        # End here: this test verifies the real binding preflight, not a model.
        raise RuntimeError("fixture stopped after send boundary")
        yield
    monkeypatch.setattr(binding, "conversation_events", events)
    body = {**service.store.load_input(original["_input_ref"]),
            **{k: child[k] for k in ("provider_binding_id", "provider_account_identity", "client_conversation_id",
                                     "conversation_id", "parent_message_id")},
            "_empty_retry_original": original, "_empty_retry_receipt": child}
    with pytest.raises(Exception) as caught:
        ConversationBindingService().complete_text(body)
    assert calls == ([] if drift else ["POST"])
    if drift:
        assert caught.value.code == "CHAT_TERMINAL_EMPTY_UNVERIFIED"
