"""Narrow three-field successors for proved pre-conversation model rejections."""
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from api.ai import ConversationBindingTextRequest
from services.conversation_binding_service import ConversationBindingError
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_pool_admission import Clock, build, claim_worker
from test.test_bound_text_archive import bound_chat


def receipt(service, request_id="old"):
    with service.store.connect() as db:
        return service.store.read_receipt(db, "text", "owner", request_id)


def update(service, request_id="old", **changes):
    with service.store.transaction() as db:
        row = service.store.read_receipt(db, "text", "owner", request_id)
        row.update(changes)
        service.store.write_receipt(db, "text", "owner", request_id, row)


def rewrite_input(service, request_id, body):
    request_id, request_hash = service._submission_identity("owner", body)
    with service.store.transaction() as db:
        row = service.store.read_receipt(db, "text", "owner", request_id)
        row["_input_ref"] = service.store.save_input(body)
        service.store.write_receipt(db, "text", "owner", request_id, row)
        db.execute("UPDATE requests SET request_hash=? WHERE owner=? AND id=?", (request_hash, "owner", request_id))


def prove_model_rejection(service, *, cursor, input_lacks_identity=False):
    old = receipt(service)
    original_input = service.store.load_input(old["_input_ref"])
    if not cursor:
        original_input.update(conversation_id=None, parent_message_id=None)
        if input_lacks_identity:
            original_input.update(provider_binding_id=None, provider_account_identity=None)
        rewrite_input(service, "old", original_input)
    update(service, **({"conversation_id": None, "parent_message_id": None} if not cursor else {}),
           status="failed", error_code="MODEL_NOT_FOUND", upstream_outcome="rejected",
           _submission_started=True, _executing=False, _turn_reserved=False,
           original_failure_phase="stream_open", original_exception_category="http",
           original_upstream_request_stage="conversation", original_http_status=404,
           original_upstream_error_form="detail_text",
           original_upstream_error={"message": "Model not found"})


@pytest.fixture
def task(tmp_path):
    (tmp_path / "accounts.json").write_text(json.dumps([{
        "access_token": "fixture", "account_id": "physical", "provider_account_identity": "original-account",
        "conversation_binding_ids": ["original-binding"], "type": "Plus", "status": "正常", "quota": 99,
        "source_type": "web",
    }]))
    clock = Clock()
    _, store, admission = build(tmp_path, clock)
    sent = []
    def runner(body, on_cursor):
        current_request.get().before_send()
        sent.append(deepcopy(body))
        return {"content": "successor answer", "parent_message_id": "new-answer"}
    service = TextTaskService(store.path, runner=runner, clock=clock, admission=admission)
    admission.register("text", lambda ctx, body: service._run(ctx.owner, ctx.request_id, body))
    body = {
        "client_request_id": "old", "client_conversation_id": "session", "model": "fixture-text",
        "image_model": "gpt-image-2", "thinking_effort": "high",
        "provider_binding_id": "original-binding", "provider_account_identity": "original-account",
        "conversation_id": "original-chat", "parent_message_id": "previous-answer",
        "messages": [{"role": "user", "content": "retain exactly"}],
    }
    service.submit("owner", body, source="internal:content")
    return SimpleNamespace(service=service, admission=admission, root=tmp_path, body=body, sent=sent)


def envelope(request_id="successor"):
    return {"client_request_id": request_id, "supersedes_request_id": "old", "client_conversation_id": "session"}


def prove_no_final_model_rejection(task):
    update(task.service, status="failed", error_code="RESULT_UNRECOVERABLE", upstream_outcome="unknown",
           recovery_reason=None, recovery_error_code="RECOVERY_READ_FAILED",
           _attempt_reason="ORIGINAL_RESULT_NO_FINAL", _attempt_finished_at=500,
           recovery_no_result_reads=9, _execution_wait_ended_at=500,
           _submission_started=True, _executing=False, _turn_reserved=False, _claim_id=None,
           _claim_until=None, send_count=1)
    unknown_child = task.service.submit("owner", {
        **{key: task.body[key] for key in TextTaskService.SUPERSEDE_FIELDS - {"supersedes_request_id"}},
        "client_request_id": "rejected-parent", "supersedes_request_id": "old",
        "continue_after_no_final": True,
    }, source="internal:content")
    assert unknown_child["supersedes_request_id"] == "old"
    update(task.service, "rejected-parent", status="failed", error_code="MODEL_NOT_FOUND",
           upstream_outcome="rejected", _submission_started=True, _executing=False, _turn_reserved=False,
           request_parent_message_id="actual-no-final-tail",
           original_failure_phase="stream_open", original_exception_category="http",
           original_upstream_request_stage="conversation", original_http_status=404,
           original_upstream_error_form="detail_text",
           original_upstream_error={"message": "Model not found"})
    return {"client_request_id": "model-retry", "supersedes_request_id": "rejected-parent",
            "client_conversation_id": "session"}


@pytest.mark.parametrize("cursor", [True, False])
def test_typed_rejection_creates_frozen_successor_with_or_without_cursor(task, cursor):
    prove_model_rejection(task.service, cursor=cursor, input_lacks_identity=not cursor)
    old = deepcopy(receipt(task.service))
    result = task.service.submit("owner", envelope(), source="internal:content")
    assert result["supersedes_request_id"] == "old" and result["status"] == "queued"
    child = receipt(task.service, "successor")
    saved = task.service.store.load_input(child["_input_ref"])
    assert saved["provider_binding_id"] == "original-binding"
    assert saved["provider_account_identity"] == "original-account"
    if cursor:
        assert saved["conversation_id"] == "original-chat" and child["_supersedes_request_message_id"] == old["request_message_id"]
    else:
        assert saved["conversation_id"] is None and "_supersedes_request_message_id" not in child
    assert receipt(task.service) == old
    claim = task.admission.claim_next()
    assert claim and claim.request_id == "successor"
    task.admission.execute(claim)
    assert receipt(task.service, "successor")["status"] == "succeeded"
    assert len(task.sent) == 1
    assert ("_supersedes_request_message_id" in task.sent[0]) is cursor


def test_same_id_reads_frozen_child_after_parent_changes(task):
    prove_model_rejection(task.service, cursor=False, input_lacks_identity=True)
    first = task.service.submit("owner", envelope(), source="internal:content")
    update(task.service, status="succeeded", content="late original", upstream_outcome="completed")
    assert task.service.submit("owner", envelope()) == first
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", envelope("other"))
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"


def test_model_rejection_after_one_explicit_unknown_successor_can_retry_once(task):
    """A typed rejection may retry the exact input of one proved UNKNOWN child."""
    def cursor_runner(body, on_cursor):
        current_request.get().before_send()
        # openai_backend_api reports both values from the actual POST tail.
        # C must retain its frozen submission root in its durable receipt.
        on_cursor({"request_parent_message_id": body["parent_message_id"],
                   "_submission_parent_message_id": body["parent_message_id"]})
        task.sent.append(deepcopy(body))
        return {"content": "successor answer", "parent_message_id": "new-answer"}
    task.service.runner = cursor_runner
    retry = task.service.submit("owner", prove_no_final_model_rejection(task), source="internal:content")

    saved = task.service.store.load_input(receipt(task.service, "model-retry")["_input_ref"])
    assert retry["supersedes_request_id"] == "rejected-parent"
    assert saved == {**task.body, "client_request_id": "model-retry", "supersedes_request_id": "rejected-parent"}
    assert "continue_after_no_final" not in saved
    child = receipt(task.service, "model-retry")
    assert "_supersedes_no_final" not in child
    assert child["request_parent_message_id"] == "previous-answer"
    assert receipt(task.service, "rejected-parent")["request_parent_message_id"] == "actual-no-final-tail"
    assert TextTaskService._supersede_order_link(receipt(task.service), receipt(task.service, "rejected-parent"))
    assert TextTaskService._supersede_order_link(receipt(task.service, "rejected-parent"), child)
    claim = task.admission.claim_next()
    assert claim and claim.request_id == "model-retry"
    task.admission.execute(claim)
    assert receipt(task.service, "model-retry")["status"] == "succeeded"
    assert len(task.sent) == 1
    assert task.sent[-1]["parent_message_id"] == "actual-no-final-tail"
    finished = receipt(task.service, "model-retry")
    assert finished["request_parent_message_id"] == "actual-no-final-tail"
    assert finished["_submission_parent_message_id"] == "previous-answer"
    task.service.submit("owner", {**task.body, "client_request_id": "after-model-retry",
                                   "parent_message_id": "new-answer"}, source="internal:content")
    later = task.admission.claim_next()
    assert later and later.request_id == "after-model-retry"


@pytest.mark.parametrize("changes", [
    {"status": "succeeded", "content": "late original", "upstream_outcome": "completed"},
    {"_claim_id": "active", "_claim_until": 2000},
])
def test_model_retry_rejects_late_or_busy_no_final_ancestor(task, changes):
    retry = prove_no_final_model_rejection(task)
    update(task.service, **changes)
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", retry, source="internal:content")
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"
    assert receipt(task.service, "model-retry") is None


def test_model_retry_same_id_is_frozen_and_different_child_conflicts(task):
    retry = prove_no_final_model_rejection(task)
    first = task.service.submit("owner", retry, source="internal:content")
    assert task.service.submit("owner", retry) == first
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", {**retry, "client_request_id": "other-model-retry"}, source="internal:content")
    assert error.value.code == "CHAT_SUPERSEDE_CONFLICT"


def test_model_retry_cannot_create_another_rejected_successor(task):
    retry = prove_no_final_model_rejection(task)
    task.service.submit("owner", retry, source="internal:content")
    update(task.service, "model-retry", status="failed", error_code="MODEL_NOT_FOUND",
           upstream_outcome="rejected", _submission_started=True, _executing=False, _turn_reserved=False,
           original_failure_phase="stream_open", original_exception_category="http",
           original_upstream_request_stage="conversation", original_http_status=404,
           original_upstream_error_form="detail_text",
           original_upstream_error={"message": "Model not found"})
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", {**retry, "client_request_id": "third-child",
                                       "supersedes_request_id": "model-retry"}, source="internal:content")
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"
    account_ref = task.admission.accounts.pool_account_ref(task.admission.accounts.list_accounts()[0])
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit_codex_fallback("owner", "model-retry", "codex-third", "gpt-5.6-codex", account_ref)
    assert error.value.code == "CODEX_FALLBACK_ORIGINAL_INELIGIBLE"


def test_model_retry_cannot_reserve_a_missing_successor_resend(task):
    retry = prove_no_final_model_rejection(task)
    task.service.submit("owner", retry, source="internal:content")
    update(task.service, "model-retry", status="failed", error_code="RESULT_UNRECOVERABLE",
           upstream_outcome="unknown", recovery_reason="REQUEST_MESSAGE_NOT_FOUND",
           _attempt_reason="ORIGINAL_REQUEST_MESSAGE_NOT_FOUND", _attempt_finished_at=500,
           _execution_wait_ended_at=500, _submission_started=True, _executing=False, _turn_reserved=False)
    saved = task.service.store.load_input(receipt(task.service, "model-retry")["_input_ref"])
    envelope = {key: saved[key] for key in TextTaskService.SUPERSEDE_FIELDS}
    with pytest.raises(ConversationBindingError) as error:
        task.service.resume_missing_successor("owner", "model-retry", envelope)
    assert error.value.code == "CHAT_MISSING_SUCCESSOR_NOT_RESUMABLE"


@pytest.mark.parametrize("parent_id,changes", [
    ("old", {"status": "succeeded", "content": "late root", "upstream_outcome": "completed"}),
    ("rejected-parent", {"status": "succeeded", "content": "late parent", "upstream_outcome": "completed",
                         "parent_message_id": "late-parent-answer"}),
])
def test_consumed_model_retry_stays_closed_after_ancestor_state_changes(task, parent_id, changes):
    retry = prove_no_final_model_rejection(task)
    task.service.submit("owner", retry, source="internal:content")
    update(task.service, "model-retry", status="failed", error_code="MODEL_NOT_FOUND",
           upstream_outcome="rejected", _submission_started=True, _executing=False, _turn_reserved=False,
           original_failure_phase="stream_open", original_exception_category="http",
           original_upstream_request_stage="conversation", original_http_status=404,
           original_upstream_error_form="detail_text",
           original_upstream_error={"message": "Model not found"})
    update(task.service, parent_id, **changes)
    account_ref = task.admission.accounts.pool_account_ref(task.admission.accounts.list_accounts()[0])
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit_codex_fallback("owner", "model-retry", "codex-third", "gpt-5.6-codex", account_ref)
    assert error.value.code == "CODEX_FALLBACK_ORIGINAL_INELIGIBLE"


def test_model_retry_rechecks_ancestor_before_send(task):
    retry = prove_no_final_model_rejection(task)
    task.service.submit("owner", retry, source="internal:content")
    claim = task.admission.claim_next()
    assert claim and claim.request_id == "model-retry"
    update(task.service, status="succeeded", content="late original", upstream_outcome="completed")
    task.admission.execute(claim)
    child = receipt(task.service, "model-retry")
    assert not task.sent
    assert child["upstream_outcome"] == "not_sent"


@pytest.mark.parametrize("changes, code", [
    ({"status": "unknown", "upstream_outcome": "unknown"}, "CHAT_SUPERSEDE_INVALID"),
    ({"status": "succeeded", "content": "late"}, "CHAT_SUPERSEDE_INVALID"),
    ({"_executing": True}, "CHAT_SUPERSEDE_INVALID"),
    ({"recovery_claim_id": "lease", "recovery_lease_until": 9999}, "CHAT_SUPERSEDE_INVALID"),
    ({"original_upstream_error": {"message": "not this model"}}, "CHAT_SUPERSEDE_INVALID"),
])
def test_only_typed_quiescent_rejection_is_eligible(task, changes, code):
    prove_model_rejection(task.service, cursor=False)
    update(task.service, **changes)
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", envelope())
    assert error.value.code == code and receipt(task.service, "successor") is None


def test_rejected_parent_refuses_full_identity_envelope_and_schema_accepts_exact_three(task):
    prove_model_rejection(task.service, cursor=False)
    assert ConversationBindingTextRequest.model_validate(envelope()).model_dump(exclude_unset=True) == envelope()
    with pytest.raises(ValidationError):
        ConversationBindingTextRequest.model_validate({**envelope(), "model": "fixture-text"})
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", {**task.body, "client_request_id": "successor", "supersedes_request_id": "old"})
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"


def submit_worker(root, body, ready, start, results):
    service = TextTaskService(Path(root) / "text_tasks.sqlite3", admission=Mock())
    ready.put(True)
    start.wait(10)
    try:
        service.submit("owner", body)
        results.put("accepted")
    except ConversationBindingError as exc:
        results.put(exc.code)


def test_cross_process_only_one_rejected_successor_claims(task):
    prove_model_rejection(task.service, cursor=False)
    mp = multiprocessing.get_context("spawn")
    ready, start, results = mp.Queue(), mp.Event(), mp.Queue()
    workers = [mp.Process(target=submit_worker, args=(str(task.root), envelope(name), ready, start, results))
               for name in ("a", "b")]
    for worker in workers: worker.start()
    for _ in workers: ready.get(timeout=20)
    start.set()
    assert sorted(results.get(timeout=20) for _ in workers) == ["CHAT_SUPERSEDE_CONFLICT", "accepted"]
    for worker in workers:
        worker.join(20)
        assert worker.exitcode == 0
    # Existing PoolAdmission transaction remains the only claim lock.
    ready, start, results = mp.Queue(), mp.Event(), mp.Queue()
    workers = [mp.Process(target=claim_worker, args=(str(task.root), ready, start, results)) for _ in range(2)]
    for worker in workers: worker.start()
    for _ in workers: ready.get(timeout=20)
    start.set()
    assert sum(results.get(timeout=20) is not None for _ in workers) == 1
    for worker in workers:
        worker.join(20)
        assert worker.exitcode == 0


def test_dispatch_bound_identity_is_inherited_without_reselecting_account(task):
    # This is the production shape: acceptance stored no provider identity, but
    # the real runner persisted the dispatch choice before the typed 404.
    unbound = {
        "client_request_id": "unbound-old", "client_conversation_id": "unbound-session", "model": "fixture-text",
        "messages": [{"role": "user", "content": "retain original prompt"}],
    }
    def rejected_runner(body, on_cursor):
        on_cursor({"provider_binding_id": "original-binding", "provider_account_identity": "original-account"})
        raise ConversationBindingError("model missing", code="MODEL_NOT_FOUND",
            provider_binding_id="original-binding", provider_account_identity="original-account",
            original_failure_phase="stream_open", original_exception_category="http",
            original_upstream_request_stage="conversation", original_http_status=404,
            original_upstream_error_form="detail_text", original_upstream_error={"message": "Model not found"})
    task.service.runner = rejected_runner
    task.service.submit("owner", unbound, source="internal:content")
    # The fixture's unrelated old receipt was accepted first; drain it so the
    # unbound request executes through the same admission path.
    task.admission.execute(task.admission.claim_next())
    task.admission.execute(task.admission.claim_next())
    original = receipt(task.service, "unbound-old")
    assert original["status"] == "failed"
    assert original["provider_binding_id"] == "original-binding"
    assert original["provider_account_identity"] == "original-account"
    child = task.service.submit("owner", {
        "client_request_id": "unbound-child", "supersedes_request_id": "unbound-old",
        "client_conversation_id": "unbound-session",
    }, source="internal:content")
    saved = task.service.store.load_input(receipt(task.service, "unbound-child")["_input_ref"])
    assert saved["messages"] == unbound["messages"]
    assert saved["provider_binding_id"] == original["provider_binding_id"]
    assert saved["provider_account_identity"] == original["provider_account_identity"]
    assert saved.get("conversation_id") is None and child["supersedes_request_id"] == "unbound-old"


def test_chat_and_codex_fallback_cannot_both_succeed(task):
    prove_model_rejection(task.service, cursor=False)
    account_ref = task.admission.accounts.pool_account_ref(task.admission.accounts.list_accounts()[0])
    task.service.submit("owner", envelope(), source="internal:content")
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit_codex_fallback("owner", "old", "codex-child", "gpt-5.6-codex", account_ref)
    assert error.value.code == "CODEX_FALLBACK_CONFLICT"


def test_codex_then_chat_successor_conflicts(task):
    prove_model_rejection(task.service, cursor=False)
    account_ref = task.admission.accounts.pool_account_ref(task.admission.accounts.list_accounts()[0])
    task.service.submit_codex_fallback("owner", "old", "codex-child", "gpt-5.6-codex", account_ref)
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", envelope(), source="internal:content")
    assert error.value.code == "CHAT_SUPERSEDE_CONFLICT"


def test_physical_rejected_successor_rechecks_original_cursor_before_send(task, bound_chat):
    prove_model_rejection(task.service, cursor=True)
    task.service.runner = bound_chat.service.complete_text
    original = receipt(task.service)
    bound_chat.document["mapping"][original["request_message_id"]] = {"message": {"id": original["request_message_id"]}}
    task.service.submit("owner", envelope(), source="internal:content")
    task.admission.execute(task.admission.claim_next())
    child = receipt(task.service, "successor")
    assert child["status"] == "failed" and child["error_code"] == "CHAT_SUPERSEDE_ORIGINAL_FOUND"
    assert child["upstream_outcome"] == "not_sent"
    assert "send" not in bound_chat.calls


@pytest.mark.parametrize("change, expected", [
    ("root_late_final", "CHAT_SUPERSEDE_CURSOR_CHANGED"),
    ("rejected_user_appears", "CHAT_SUPERSEDE_ORIGINAL_FOUND"),
    ("tail_content_changes", "CHAT_SUPERSEDE_CURSOR_CHANGED"),
])
def test_model_retry_bound_guard_rechecks_root_tail_and_rejected_parent(task, bound_chat, monkeypatch,
                                                                         change, expected):
    """The C POST reuses the binding layer's R→tail preflight, not P's cursor."""
    retry = prove_no_final_model_rejection(task)
    root = receipt(task.service)
    rejected = receipt(task.service, "rejected-parent")

    def node(message_id, parent, children, role, content, *, end=None, channel=None):
        return {"parent": parent, "children": children, "message": {
            "id": message_id, "author": {"role": role}, "status": "finished_successfully",
            "end_turn": end, "channel": channel, "content": content,
        }}
    bound_chat.document = {
        "conversation_id": "original-chat", "current_node": "actual-no-final-tail", "is_archived": False,
        "mapping": {
            root["request_message_id"]: node(
                root["request_message_id"], "previous-answer", ["no-final-code"], "user",
                {"content_type": "text", "parts": ["original input"]}),
            "no-final-code": node(
                "no-final-code", root["request_message_id"], ["actual-no-final-tail"], "assistant",
                {"content_type": "code", "text": "inspect input"}, end=False, channel="analysis"),
            "actual-no-final-tail": node(
                "actual-no-final-tail", "no-final-code", [], "tool",
                {"content_type": "multimodal_text", "parts": ["retained tool context"]}, end=False),
        },
    }
    import services.conversation_binding_service as binding_module
    backend_type = binding_module.OpenAIBackendAPI
    original_get = backend_type._get_conversation

    def get_with_preflight_send(self, chat, _send=None):
        return original_get(self, chat)
    monkeypatch.setattr(backend_type, "_get_conversation", get_with_preflight_send)

    def guarded_events(backend, **kwargs):
        if change == "root_late_final":
            bound_chat.document["mapping"][root["request_message_id"]]["children"].append("late-final")
            bound_chat.document["mapping"]["late-final"] = node(
                "late-final", root["request_message_id"], [], "assistant",
                {"content_type": "text", "parts": ["late answer"]}, end=True)
        elif change == "rejected_user_appears":
            bound_chat.document["mapping"][rejected["request_message_id"]] = node(
                rejected["request_message_id"], "actual-no-final-tail", [], "user",
                {"content_type": "text", "parts": ["rejected request"]})
        else:
            bound_chat.document["mapping"]["actual-no-final-tail"]["message"]["content"] = {
                "content_type": "multimodal_text", "parts": ["changed tool context"]}
        backend.text_pre_send_check(None)
        bound_chat.calls.append("send")
        yield {"type": "conversation.delta", "conversation_id": "original-chat", "delta": "answer"}

    monkeypatch.setattr(binding_module, "conversation_events", guarded_events)
    task.service.runner = bound_chat.service.complete_text
    task.service.submit("owner", retry, source="internal:content")
    task.admission.execute(task.admission.claim_next())
    child = receipt(task.service, "model-retry")
    assert child["status"] == "failed" and child["error_code"] == expected
    assert child["upstream_outcome"] == "not_sent" and "send" not in bound_chat.calls


def test_rejected_successor_rejects_local_bound_auth_unavailable_before_insert(task, monkeypatch):
    prove_model_rejection(task.service, cursor=False)
    import services.text_task_service as text_module
    monkeypatch.setattr(text_module.account_service, "bound_text_recovery_auth_unavailable", lambda _binding: True)
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", envelope(), source="internal:content")
    assert error.value.code == "CHAT_SUPERSEDE_AUTH_REQUIRED"
    assert receipt(task.service, "successor") is None
