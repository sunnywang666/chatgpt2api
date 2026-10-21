"""One explicit compact successor may continue a strictly observed incomplete Chat turn."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from services.combined_copy_derivation import KIND, MODEL, EFFORT
from services.conversation_binding_service import ConversationBindingError, ConversationBindingService
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_combined_copy_derivation import branch, legacy_prompt
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
    elif change == "branch":
        document["mapping"]["active"]["children"] = ["later"]
        document["mapping"]["later"] = {"parent": "active", "children": [], "message": {
            "id": "later", "author": {"role": "user"}, "status": "finished_successfully",
            "end_turn": True, "content": {"content_type": "text", "parts": ["later work"]}}}
    return document


def install_incomplete(h):
    h.body.update(model=MODEL, thinking_effort=EFFORT, messages=[{"role": "user", "content": [
        {"type": "text", "text": legacy_prompt([branch(0), branch(1)])},
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


def patch_bound_runner(monkeypatch, h, *, before_send_change=None):
    import services.conversation_binding_service as binding
    calls = []
    class Backend:
        def __init__(self, access_token):
            assert access_token == "fixture-token"
            self.retain_bound_conversation = False
            self.text_request_message_id = ""
            self.text_cursor_callback = None
            self.text_pre_send_check = None
        def _get_conversation(self, conversation_id, _send=None):
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
        assert kwargs["parent_message_id"] == "active"
        if before_send_change:
            h.document = incomplete_document(h.old, change=before_send_change)
        backend.text_pre_send_check(None)
        calls.append("send")
        request_id = backend.text_request_message_id
        h.document["mapping"]["active"]["children"] = [request_id]
        h.document["mapping"][request_id] = {"parent": "active", "children": ["successor-answer"], "message": {
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


def test_incomplete_compact_successor_uses_pool_and_two_fresh_original_reads(incomplete, monkeypatch):
    h = incomplete
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
def test_incomplete_send_edge_rejects_late_result_progress_or_branch(incomplete, monkeypatch, change):
    h = incomplete
    calls = patch_bound_runner(monkeypatch, h, before_send_change=change)
    h.service.submit("owner", h.envelope)
    h.admission.execute(h.admission.claim_next())
    successor = row(h.service, "successor")
    assert successor["status"] == "failed" and successor["upstream_outcome"] == "not_sent"
    assert not successor["_submission_started"] and "send" not in calls
    assert row(h.service) == h.old


def test_incomplete_mode_is_closed_idempotent_and_never_resume_unsent(incomplete):
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


@pytest.mark.parametrize("change", [
    {"recovery_reason": "REQUEST_RESULT_NOT_FOUND"},
    {"_result_no_progress_reads": 2},
    {"_claim_id": "expired", "_claim_until": 1.0},
    {"recovery_claim_id": "reader", "recovery_lease_until": 1001.0},
    {"_missing_successor_resume": {}},
    {"status": "succeeded", "content": "late answer"},
])
def test_only_qualified_incomplete_original_can_register(incomplete, change):
    h = incomplete
    update(h.service, **change)
    with pytest.raises(ConversationBindingError):
        h.service.submit("owner", h.envelope)
    assert row(h.service, "successor") is None


def test_incomplete_source_accepts_explicit_nonterminal_false(incomplete):
    h = incomplete
    update(h.service, _upstream_terminal=False)
    accepted = h.service.submit("owner", h.envelope)
    assert accepted["status"] == "queued" and accepted["continue_after_incomplete"] is True
