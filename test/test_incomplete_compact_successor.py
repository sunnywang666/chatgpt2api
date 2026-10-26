"""One explicit compact successor may continue a strictly observed incomplete Chat turn."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from services.combined_copy_derivation import KIND, MODEL, EFFORT
from services.conversation_binding_service import ConversationBindingError, ConversationBindingService
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_combined_copy_derivation import WBOZ_CURRENT_CASES, branch, legacy_prompt
from test.test_explicit_unknown_successor import task, row, update
from test.test_pool_admission import build


def incomplete_document(original, *, change=None):
    user = original["request_message_id"]
    document = {"conversation_id": original["conversation_id"], "current_node": "active", "is_archived": False,
                "mapping": {
                    user: {"parent": original["parent_message_id"], "children": ["active"], "message": {
                        "id": user, "author": {"role": "user"}, "status": "finished_successfully",
                        "end_turn": True, "content": {"content_type": "text", "parts": ["original input"]}}},
                    "active": {"parent": user, "children": [], "message": {
                        "id": "active", "author": {"role": "assistant"}, "status": "in_progress",
                        "end_turn": False, "content": {"content_type": "thoughts", "thoughts": [
                            {"summary": "working", "content": "", "finished": False, "chunks": []}]},
                        "update_time": 10}},
                }}
    if change == "late_final":
        message = document["mapping"]["active"]["message"]
        message.update(status="finished_successfully", end_turn=True,
                       content={"content_type": "text", "parts": ["late answer"]})
    elif change == "progress":
        document["mapping"]["active"]["message"]["content"]["thoughts"][0]["content"] = "new measured progress"
        document["mapping"]["active"]["message"]["update_time"] = 11
    elif change == "heartbeat":
        document["mapping"]["active"]["message"]["update_time"] = 11
    elif change == "branch":
        document["mapping"]["active"]["children"] = ["later"]
        document["mapping"]["later"] = {"parent": "active", "children": [], "message": {
            "id": "later", "author": {"role": "user"}, "status": "finished_successfully",
            "end_turn": True, "content": {"content_type": "text", "parts": ["later work"]}}}
    return document


def install_incomplete(h, *, prompt=None, input_body=None):
    if input_body is not None:
        h.body.update(deepcopy(input_body))
    else:
        prompt = legacy_prompt([branch(0), branch(1)]) if prompt is None else prompt
        h.body.update(model=MODEL, thinking_effort=EFFORT, messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "text", "text": '{"image_ref":"source-a"}'},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDRA=="}},
        ]}])
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        original.update(_input_ref=h.service.store.save_input(h.body), model=MODEL, thinking_effort=EFFORT)
        document = incomplete_document(original)
        observed = ConversationBindingService._read_text_request_result(None, original, document=document)
        assert observed["recovery_reason"] == "REQUEST_RESULT_INCOMPLETE"
        original.update(status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN", upstream_outcome="unknown",
                        recovery_reason="REQUEST_RESULT_INCOMPLETE", recovery_error_code="UPSTREAM_OUTCOME_UNKNOWN",
                        _submission_started=True, _executing=False, _turn_reserved=False,
                        _claim_id=None, _claim_until=None, recovery_claim_id=None, recovery_claimed_at=None,
                        recovery_lease_until=None, _result_wait_ended_at=901.0,
                        _result_observation_started_at=1.0, _result_last_progress_at=None,
                        _result_last_checked_at=1000.0, _result_no_progress_reads=3,
                        _original_result_observation=observed["_original_result_observation"])
        h.service.store.write_receipt(db, "text", "owner", "old", original)
        db.execute("UPDATE requests SET request_hash=? WHERE owner='owner' AND id='old'",
                   (TextTaskService._submission_identity("owner", h.body)[1],))
    h.old = row(h.service)
    h.envelope = {key: h.body[key] for key in TextTaskService.SUPERSEDE_FIELDS if key in h.body}
    h.envelope.update(client_request_id="successor", supersedes_request_id="old",
                      derived_input={"kind": KIND}, continue_after_incomplete=True)
    h.document = document
    return h


@pytest.fixture
def incomplete(task):
    return install_incomplete(task)


def patch_bound_runner(monkeypatch, h, *, before_send_change=None, before_send_document=None, before_send_once=False):
    import services.conversation_binding_service as binding
    calls = []
    pending_change = [before_send_change]
    class Backend:
        def __init__(self, access_token):
            assert access_token == "fixture-token"
            self.retain_bound_conversation = False
            self.text_request_message_id = ""
            self.text_cursor_callback = None
            self.text_pre_send_check = None
        def _get_conversation(self, conversation_id, _send=None, *, deadline_monotonic=None):
            assert conversation_id == h.body["conversation_id"]
            calls.append("read")
            return deepcopy(h.document)
        def get_conversation_parent_message_id(self, conversation_id):
            return "successor-answer"
        def close(self):
            pass
    accounts = Mock()
    accounts.get_bound_account_identity.return_value = "original-account"
    accounts.get_bound_text_access_token.return_value = "fixture-token"
    accounts.conversation_binding_lock.side_effect = lambda *_: nullcontext()
    monkeypatch.setattr(binding, "account_service", accounts)
    monkeypatch.setattr(binding, "OpenAIBackendAPI", Backend)
    def events(backend, **kwargs):
        assert kwargs["conversation_id"] == h.body["conversation_id"]
        parent_message_id = h.document["current_node"]
        assert kwargs["parent_message_id"] == parent_message_id
        if before_send_document is not None:
            h.document = deepcopy(before_send_document)
        elif pending_change[0]:
            h.document = incomplete_document(h.old, change=pending_change[0])
            if before_send_once:
                pending_change[0] = None
        backend.text_pre_send_check(None)
        # Match the real runner boundary: the durable admission context must
        # still permit the POST after the second live original-result read.
        current_request.get().before_send()
        calls.append("send")
        request_id = backend.text_request_message_id
        h.document["mapping"][parent_message_id]["children"] = [request_id]
        h.document["mapping"][request_id] = {"parent": parent_message_id, "children": ["successor-answer"], "message": {
            "id": request_id, "author": {"role": "user"}, "status": "finished_successfully",
            "end_turn": True, "content": {"content_type": "text", "parts": ["compact request"]}}}
        h.document["mapping"]["successor-answer"] = {"parent": request_id, "children": [], "message": {
            "id": "successor-answer", "author": {"role": "assistant"}, "status": "finished_successfully",
            "end_turn": True, "content": {"content_type": "text", "parts": ["compact answer"]}}}
        h.document["current_node"] = "successor-answer"
        yield {"type": "conversation.delta", "conversation_id": h.body["conversation_id"], "delta": "compact answer"}
    monkeypatch.setattr(binding, "conversation_events", events)
    h.service.runner = ConversationBindingService().complete_text
    return calls


def set_original_outcome(h, outcome):
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        if outcome == "missing":
            original.pop("upstream_outcome", None)
        else:
            original["upstream_outcome"] = outcome
        h.service.store.write_receipt(db, "text", "owner", "old", original)
    h.old = row(h.service)


@pytest.mark.parametrize("outcome", ["unknown", "missing", "not_sent"])
def test_incomplete_compact_successor_uses_pool_and_two_fresh_original_reads(incomplete, monkeypatch, outcome):
    h = incomplete
    set_original_outcome(h, outcome)
    calls = patch_bound_runner(monkeypatch, h)
    accepted = h.service.submit("owner", h.envelope)
    assert accepted["status"] == "queued" and accepted["continue_after_incomplete"] is True
    assert "continue_after_no_final" not in accepted
    queued = row(h.service, "successor")
    saved = h.service.store.load_input(queued["_input_ref"])
    assert saved["continue_after_incomplete"] is True and "continue_after_no_final" not in saved
    _, store, restarted = build(h.root, h.clock)
    service = TextTaskService(store.path, runner=h.service.runner, clock=h.clock, admission=restarted)
    restarted.register("text", lambda ctx, body: service._run(ctx.owner, ctx.request_id, body))
    assert service.read("owner", "successor") == accepted
    context = restarted.claim_next()
    assert context and context.request_id == "successor"
    running = service.read("owner", "successor")
    assert running["status"] == "running" and running["continue_after_incomplete"] is True
    assert running["derived_input"] == accepted["derived_input"]
    restarted.execute(context)
    result = service.submit("owner", h.envelope)
    saved_receipt = row(service, "successor")
    assert result["status"] == "succeeded" and result["continue_after_incomplete"] is True, (calls, {
        key: saved_receipt.get(key) for key in ("error_code", "original_failure_phase", "original_exception_category",
            "original_upstream_request_stage", "_execution_timeline", "_supersedes_incomplete")})
    assert result["derived_input"] == accepted["derived_input"]
    assert calls == ["read", "read", "send", "read"]
    assert row(service) == h.old
    Path("/tmp/provider-incomplete-contract-receipts.json").write_text(json.dumps({
        "request": {
            "clientRequestId": "successor",
            "supersedesRequestId": "old",
            "cursor": {
                "providerBindingId": h.envelope["provider_binding_id"],
                "providerAccountIdentity": h.envelope["provider_account_identity"],
                "clientConversationId": h.envelope["client_conversation_id"],
                "conversationId": h.envelope["conversation_id"],
                "parentMessageId": h.envelope["parent_message_id"],
            },
        },
        "queued": accepted,
        "succeeded": result,
    }, indent=2, sort_keys=True), encoding="utf-8")


@pytest.mark.parametrize("change", ["late_final", "progress", "branch"])
@pytest.mark.parametrize("outcome", ["unknown", "missing", "not_sent"])
def test_incomplete_send_edge_rejects_late_result_progress_or_branch(incomplete, monkeypatch, change, outcome):
    h = incomplete
    set_original_outcome(h, outcome)
    calls = patch_bound_runner(monkeypatch, h, before_send_change=change)
    h.service.submit("owner", h.envelope)
    h.admission.execute(h.admission.claim_next())
    successor = row(h.service, "successor")
    assert successor["status"] == "failed" and successor["upstream_outcome"] == "not_sent"
    assert not successor["_submission_started"] and "send" not in calls
    assert row(h.service) == h.old


def test_incomplete_compact_successor_accepts_heartbeat_only_at_initial_read(incomplete, monkeypatch):
    h = incomplete
    h.document = incomplete_document(h.old, change="heartbeat")
    calls = patch_bound_runner(monkeypatch, h)

    h.service.submit("owner", h.envelope)
    h.admission.execute(h.admission.claim_next())

    successor = row(h.service, "successor")
    assert successor["status"] == "succeeded"
    assert calls == ["read", "read", "send", "read"]


def test_incomplete_compact_successor_accepts_heartbeat_only_before_send(incomplete, monkeypatch):
    h = incomplete
    calls = patch_bound_runner(monkeypatch, h, before_send_change="heartbeat")

    h.service.submit("owner", h.envelope)
    h.admission.execute(h.admission.claim_next())

    successor = row(h.service, "successor")
    assert successor["status"] == "succeeded"
    assert calls == ["read", "read", "send", "read"]


def test_incomplete_mode_is_closed_idempotent_and_rejects_queued_resume(incomplete):
    h = incomplete
    first = h.service.submit("owner", h.envelope)
    assert h.service.submit("owner", h.envelope) == first
    with pytest.raises(ConversationBindingError) as error:
        h.service.submit("owner", {**h.envelope, "client_request_id": "other"})
    assert error.value.code == "CHAT_SUPERSEDE_CONFLICT"
    with pytest.raises(ConversationBindingError) as error:
        h.service.resume_unsent_successor("owner", "successor", h.envelope)
    assert error.value.code == "CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE"
    with pytest.raises(ConversationBindingError) as error:
        h.service.submit("owner", {**h.body, "continue_after_incomplete": True})
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"
    from api.ai import ConversationBindingTextRequest
    assert ConversationBindingTextRequest.model_validate(h.envelope).model_dump(exclude_unset=True) == h.envelope
    for bad in ({**h.envelope, "continue_after_no_final": True},
                {**h.envelope, "continue_after_incomplete": False},
                {**h.envelope, "derived_input": {"kind": "attributes_required_only_v4"}},
                {**h.envelope, "messages": []}):
        with pytest.raises(ValidationError):
            ConversationBindingTextRequest.model_validate(bad)


@pytest.mark.parametrize("case", WBOZ_CURRENT_CASES, ids=lambda case: case["name"])
def test_current_compact_incomplete_successor_retains_original_messages_and_idempotency(task, case):
    h = install_incomplete(task, input_body=case["input"])
    original_body = h.service.store.load_input(row(h.service)["_input_ref"])
    first = h.service.submit("owner", h.envelope)
    saved = h.service.store.load_input(row(h.service, "successor")["_input_ref"])

    assert saved["messages"] == original_body["messages"]
    assert first["derived_input"]["kind"] == KIND
    assert first["derived_input"]["preparation_count"] == len(case["preparations"])
    assert first["derived_input"]["evidence_set_count"] == case["evidenceSetCount"]
    assert h.service.submit("owner", h.envelope) == first


@pytest.mark.parametrize("change", [
    {"upstream_outcome": None},
    {"upstream_outcome": "not_submitted"},
    {"upstream_outcome": "submitted"},
    {"upstream_outcome": "completed"},
    {"upstream_outcome": "rejected"},
    {"upstream_outcome": "not_sent", "_submission_started": False},
    {"recovery_reason": "REQUEST_RESULT_NOT_FOUND"},
    {"_result_no_progress_reads": 0},
    {"_result_no_progress_reads": 2},
    {"_result_last_progress_at": 500.0},
    {"_result_last_checked_at": 800.0},
    {"_claim_id": "live", "_claim_until": 1001.0},
    {"_claim_id": "nonfinite", "_claim_until": float("inf")},
    {"_claim_id": "unpaired", "_claim_until": None},
    {"_claim_id": None, "_claim_until": 999.0},
    {"_claim_id": "expired", "_claim_until": 999.0, "_executing": True},
    {"_claim_id": "expired", "_claim_until": 999.0, "_turn_reserved": True},
    {"recovery_claim_id": "reader", "recovery_lease_until": 1001.0},
    {"_recovery_claim_id": "internal", "_recovery_claim_until": 999.0},
    {"_missing_successor_resume": {}},
    {"status": "succeeded", "content": "late answer"},
])
def test_only_qualified_incomplete_original_can_register(incomplete, change):
    h = incomplete
    update(h.service, **change)
    with pytest.raises(ConversationBindingError):
        h.service.submit("owner", h.envelope)
    assert row(h.service, "successor") is None


def test_expired_paired_sender_claim_can_continue_without_mutating_original(incomplete):
    h = incomplete
    update(h.service, _claim_id="historical-sender", _claim_until=999.0)
    before = row(h.service)
    assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(before, h.clock.now)
    accepted = h.service.submit("owner", h.envelope)
    assert accepted["status"] == "queued" and accepted["continue_after_incomplete"] is True
    assert row(h.service) == before


def test_incomplete_source_accepts_explicit_nonterminal_false(incomplete):
    h = incomplete
    update(h.service, _upstream_terminal=False)
    accepted = h.service.submit("owner", h.envelope)
    assert accepted["status"] == "queued" and accepted["continue_after_incomplete"] is True


def test_qualified_incomplete_successor_keeps_original_planner_position(incomplete):
    h = incomplete
    # The unrelated work is accepted after the original but before its compact
    # child.  The child keeps its own stored sequence, while the planner must
    # project the proved original position for this one continuation.
    h.service.submit("owner", {
        "client_request_id": "peer", "client_conversation_id": "peer-session",
        "model": MODEL, "thinking_effort": EFFORT,
        "messages": [{"role": "user", "content": "independent durable input"}],
    }, source="internal:content")
    accepted = h.service.submit("owner", h.envelope, source="internal:content")
    saved = row(h.service, "successor")
    peer_saved = row(h.service, "peer")
    assert saved["_sequence"] > peer_saved["_sequence"] > h.old["_sequence"]
    assert accepted["status"] == "queued"

    context = h.admission.claim_next()

    assert context is not None and context.request_id == "successor"
    # Projection cannot rewrite the receipt's accepted ordering or lineage.
    assert row(h.service, "successor")["_sequence"] == saved["_sequence"]
    assert row(h.service, "successor")["_supersedes_request_id"] == "old"



def request_hash(service, request_id):
    with service.store.connect() as db:
        return db.execute(
            "SELECT request_hash FROM requests WHERE owner=? AND id=?", ("owner", request_id)
        ).fetchone()[0]


def failed_unsent_incomplete_successor(h, monkeypatch):
    """Reach the real second guard, fail before POST, then restore its live input."""
    original_document = deepcopy(h.document)
    calls = patch_bound_runner(monkeypatch, h, before_send_change="progress", before_send_once=True)
    h.service.submit("owner", h.envelope)
    context = h.admission.claim_next()
    assert context and context.request_id == "successor"
    h.admission.execute(context)
    failed = row(h.service, "successor")
    assert failed["status"] == "failed"
    assert failed["error_code"] == "CHAT_SUPERSEDE_CURSOR_CHANGED"
    assert failed["upstream_outcome"] == "not_sent"
    assert failed["_submission_started"] is False
    assert "send" not in calls
    h.document = original_document
    calls.clear()
    return failed, calls


def finished_thought_document(original, *, channel=None, recipient=None,
                              metadata_channel=None, is_complete=None):
    """A still-running original with a completed analysis segment at its leaf."""
    document = incomplete_document(original)
    active = document["mapping"]["active"]
    active["children"] = ["finished"]
    message = {
        "id": "finished", "author": {"role": "assistant"},
        "status": "finished_successfully", "end_turn": False,
        "content": {"content_type": "thoughts", "thoughts": [{
            "summary": "completed", "content": "completed thought", "finished": True, "chunks": [],
        }]},
        "update_time": 11,
    }
    if channel is not None:
        message["channel"] = channel
    if recipient is not None:
        message["recipient"] = recipient
    if metadata_channel is not None or is_complete is not None:
        metadata = {}
        if metadata_channel is not None:
            metadata["channel"] = metadata_channel
        if is_complete is not None:
            metadata["is_complete"] = is_complete
        message["metadata"] = metadata
    document["mapping"]["finished"] = {"parent": "active", "children": [], "message": message}
    document["current_node"] = "finished"
    return document


def save_incomplete_observation(h, document):
    """Store the same redacted node metrics that the two live guards compare."""
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        observed = ConversationBindingService._read_text_request_result(None, original, document=document)
        assert observed["status"] == "running"
        assert observed["recovery_reason"] == "REQUEST_RESULT_INCOMPLETE"
        original["_original_result_observation"] = observed["_original_result_observation"]
        h.service.store.write_receipt(db, "text", "owner", "old", original)
    h.old = row(h.service)
    h.document = document


def test_incomplete_known_unsent_snapshot_and_read_advertise_without_mutation(incomplete, monkeypatch):
    h = incomplete
    failed, calls = failed_unsent_incomplete_successor(h, monkeypatch)
    before = deepcopy(failed)

    snapshot = h.service.read_snapshot("owner", "successor")
    ordinary = h.service.read("owner", "successor")

    for result in (snapshot, ordinary):
        assert result["status"] == "failed"
        assert result["bound_successor_resume_retryable"] is True
        assert result["upstream_outcome"] == "not_submitted"
        assert result["upstream_submission_started"] is False
        assert result["continue_after_incomplete"] is True
    assert row(h.service, "successor") == before
    assert h.admission.claim_next() is None
    assert calls == []


@pytest.mark.parametrize("changes", [
    {"send_count": 1},
    {"status": "unknown", "upstream_outcome": "unknown", "recovery_next_at": 1100.0},
])
def test_incomplete_resume_advertisement_hides_sent_or_unknown_receipt(incomplete, monkeypatch, changes):
    h = incomplete
    failed_unsent_incomplete_successor(h, monkeypatch)
    update(h.service, "successor", **changes)
    before = deepcopy(row(h.service, "successor"))

    snapshot = h.service.read_snapshot("owner", "successor")
    ordinary = h.service.read("owner", "successor")

    assert "bound_successor_resume_retryable" not in snapshot
    assert "bound_successor_resume_retryable" not in ordinary
    assert row(h.service, "successor") == before


@pytest.mark.parametrize("finished_tail", [False, True])
def test_incomplete_known_unsent_resumes_once_after_restart_without_changing_identity(incomplete, monkeypatch, finished_tail):
    h = incomplete
    if finished_tail:
        save_incomplete_observation(h, finished_thought_document(h.old))
    failed, calls = failed_unsent_incomplete_successor(h, monkeypatch)
    frozen_input = h.service.store.load_input(failed["_input_ref"])
    frozen_hash = request_hash(h.service, "successor")
    original_before = deepcopy(row(h.service, "old"))

    resumed = h.service.resume_unsent_successor("owner", "successor", h.envelope)

    assert resumed["status"] == "queued"
    queued = row(h.service, "successor")
    assert queued["_bound_successor_resume_count"] == 1
    assert queued["_bound_successor_resume_failure"]["error_code"] == "CHAT_SUPERSEDE_CURSOR_CHANGED"
    for key in ("request_id", "request_message_id", "_input_ref", "_supersedes_request_id",
                "_supersedes_input_hash", "_sequence", "provider_binding_id",
                "provider_account_identity", "conversation_id", "client_conversation_id"):
        assert queued[key] == failed[key]
    assert h.service.store.load_input(queued["_input_ref"]) == frozen_input
    assert request_hash(h.service, "successor") == frozen_hash
    assert row(h.service, "old") == original_before

    _, store, restarted = build(h.root, h.clock)
    service = TextTaskService(store.path, runner=h.service.runner, clock=h.clock, admission=restarted)
    restarted.register("text", lambda ctx, body: service._run(ctx.owner, ctx.request_id, body))
    assert service.resume_unsent_successor("owner", "successor", h.envelope)["status"] == "queued"
    context = restarted.claim_next()
    assert context and context.request_id == "successor"
    restarted.execute(context)

    succeeded = row(service, "successor")
    assert succeeded["status"] == "succeeded"
    assert succeeded["_submission_started"] is True and succeeded["_last_sent_sequence"] == 0
    assert succeeded["request_message_id"] == failed["request_message_id"]
    assert service.store.load_input(succeeded["_input_ref"]) == frozen_input
    assert request_hash(service, "successor") == frozen_hash
    assert calls == ["read", "read", "send", "read"]
    assert service.resume_unsent_successor("owner", "successor", h.envelope)["status"] == "succeeded"
    assert calls == ["read", "read", "send", "read"]
    assert restarted.claim_next() is None
    assert row(service, "old") == original_before


@pytest.mark.parametrize("changes", [
    {"_claim_id": "live", "_claim_until": 1001.0},
    {"send_count": 1},
    {"_last_sent_sequence": 0},
    {"_sequence": 0},
    {"_submission_started": True},
    {"status": "running"},
    {"status": "unknown"},
])
def test_incomplete_resume_rejects_claim_send_or_sequence_ambiguity(incomplete, monkeypatch, changes):
    h = incomplete
    failed_unsent_incomplete_successor(h, monkeypatch)
    update(h.service, "successor", **changes)
    before = deepcopy(row(h.service, "successor"))

    with pytest.raises(ConversationBindingError) as error:
        h.service.resume_unsent_successor("owner", "successor", h.envelope)

    assert error.value.code == "CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE"
    assert row(h.service, "successor") == before
    assert h.admission.claim_next() is None


@pytest.mark.parametrize("mutation", ["input_hash", "parent_late_result"])
def test_incomplete_resume_rejects_changed_frozen_or_parent_proof(incomplete, monkeypatch, mutation):
    h = incomplete
    failed_unsent_incomplete_successor(h, monkeypatch)
    if mutation == "input_hash":
        with h.service.store.transaction() as db:
            db.execute("UPDATE requests SET request_hash='changed' WHERE owner=? AND id=?", ("owner", "successor"))
    else:
        update(h.service, "old", status="succeeded", content="late answer", upstream_outcome="completed")
    before = deepcopy(row(h.service, "successor"))

    with pytest.raises(ConversationBindingError):
        h.service.resume_unsent_successor("owner", "successor", h.envelope)

    assert row(h.service, "successor") == before
    assert h.admission.claim_next() is None


@pytest.mark.parametrize("channel,recipient,metadata_channel,is_complete", [
    (None, None, None, None),
    ("analysis", "all", "analysis", None),
    (None, None, None, True),
])
def test_completed_thought_tail_can_continue_when_the_whole_branch_is_still_incomplete(
        incomplete, monkeypatch, channel, recipient, metadata_channel, is_complete):
    h = incomplete
    document = finished_thought_document(h.old, channel=channel, recipient=recipient,
                                         metadata_channel=metadata_channel, is_complete=is_complete)
    save_incomplete_observation(h, document)
    calls = patch_bound_runner(monkeypatch, h)

    h.service.submit("owner", h.envelope)
    h.admission.execute(h.admission.claim_next())

    assert row(h.service, "successor")["status"] == "succeeded"
    assert calls == ["read", "read", "send", "read"]


def invalid_finished_tail(original, kind):
    document = finished_thought_document(original)
    tail = document["current_node"]
    message = document["mapping"][tail]["message"]
    if kind == "final":
        message.update(status="finished_successfully", end_turn=True,
                       content={"content_type": "text", "parts": ["answer"]})
    elif kind == "tool":
        message["author"] = {"role": "tool"}
    elif kind == "code":
        message["content"] = {"content_type": "code", "text": "x"}
    elif kind == "end_turn":
        message["end_turn"] = True
    elif kind == "channel":
        message["channel"] = "final"
    elif kind == "recipient":
        message["recipient"] = "tool"
    elif kind == "metadata_final":
        message["metadata"] = {"channel": "final"}
    elif kind == "children":
        document["mapping"][tail]["children"] = ["later"]
        document["mapping"]["later"] = {"parent": tail, "children": [], "message": {
            "id": "later", "author": {"role": "user"}, "status": "finished_successfully",
            "end_turn": True, "content": {"content_type": "text", "parts": ["later"]}}}
    elif kind == "missing_finished":
        message["content"]["thoughts"][0]["finished"] = False
    elif kind == "no_active_ancestor":
        document = incomplete_document(original)
        message = document["mapping"]["active"]["message"]
        message.update(status="finished_successfully", end_turn=False)
        message["content"]["thoughts"][0].update(content="completed thought", finished=True)
    else:
        raise AssertionError(kind)
    return document


@pytest.mark.parametrize("stage", ["initial", "pre_send"])
@pytest.mark.parametrize("kind", ["final", "tool", "code", "end_turn", "channel", "recipient",
                                    "metadata_final", "children", "missing_finished", "no_active_ancestor"])
def test_completed_thought_tail_rejects_final_or_branch_changes_at_both_live_guards(
        incomplete, monkeypatch, stage, kind):
    h = incomplete
    good = finished_thought_document(h.old)
    save_incomplete_observation(h, good)
    bad = invalid_finished_tail(h.old, kind)
    if stage == "initial":
        h.document = bad
        calls = patch_bound_runner(monkeypatch, h)
    else:
        calls = patch_bound_runner(monkeypatch, h, before_send_document=bad)

    h.service.submit("owner", h.envelope)
    h.admission.execute(h.admission.claim_next())

    failed = row(h.service, "successor")
    assert failed["status"] == "failed"
    assert failed["error_code"] in {"CHAT_SUPERSEDE_CURSOR_CHANGED", "CHAT_SUPERSEDE_ORIGINAL_FOUND"}
    assert failed["upstream_outcome"] == "not_sent"
    assert "send" not in calls


@pytest.mark.parametrize("stage", ["initial", "pre_send"])
@pytest.mark.parametrize("change", ["progress", "branch"])
def test_incomplete_resume_rechecks_late_progress_or_branch_at_both_live_guards(incomplete, monkeypatch, stage, change):
    h = incomplete
    failed_unsent_incomplete_successor(h, monkeypatch)
    if stage == "initial":
        h.document = incomplete_document(h.old, change=change)
        calls = patch_bound_runner(monkeypatch, h)
    else:
        calls = patch_bound_runner(monkeypatch, h, before_send_change=change)
    h.service.resume_unsent_successor("owner", "successor", h.envelope)
    h.admission.execute(h.admission.claim_next())

    failed = row(h.service, "successor")
    assert failed["status"] == "failed"
    assert failed["error_code"] == "CHAT_SUPERSEDE_CURSOR_CHANGED"
    assert failed["upstream_outcome"] == "not_sent"
    assert "send" not in calls


def test_incomplete_resume_endpoint_is_admin_only_existing_only_and_has_no_review(incomplete, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import api.ai as api

    h = incomplete
    failed_unsent_incomplete_successor(h, monkeypatch)
    app = FastAPI()
    app.include_router(api.create_router())
    monkeypatch.setattr(api, "text_task_service", h.service)
    monkeypatch.setattr(api, "require_identity", lambda token: {
        "id": "owner", "role": "admin" if token == "internal" else "user",
    })
    monkeypatch.setattr(api, "require_chat_text_policy", lambda _: None)
    review = AsyncMock()
    monkeypatch.setattr(api, "filter_or_log", review)
    submit = Mock(side_effect=AssertionError("incomplete resume must not create a new submit"))
    monkeypatch.setattr(h.service, "submit", submit)
    url = "/api/conversation-bindings/text-requests/successor/resume-unsent-successor"

    with TestClient(app) as client:
        assert client.post(url, json=h.envelope).status_code == 501
        first = client.post(url, json=h.envelope, headers={"Authorization": "internal"})
        assert first.status_code == 200 and first.json()["status"] == "queued", first.text
        missing = client.post(url.replace("/successor/", "/absent/"),
                              json={**h.envelope, "client_request_id": "absent"},
                              headers={"Authorization": "internal"})
        assert missing.status_code == 409
        assert missing.json()["detail"]["code"] == "CHAT_REQUEST_NOT_FOUND"
    assert submit.call_count == 0
    assert review.await_count == 0
