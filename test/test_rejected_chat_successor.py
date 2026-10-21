"""Narrow three-field successors for proved pre-conversation model rejections."""
from copy import deepcopy
import json
import multiprocessing
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from api.ai import ConversationBindingTextRequest
from services.conversation_binding_service import ConversationBindingError
from services.request_context import AdmissionLost, current_request
from services.text_task_service import TextTaskService
from services.image_task_service import ImageTaskService
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
        "source_type": "web", "limits_progress": [{"feature_name": "image_gen", "remaining": 9}],
        "capacity_observed_at": datetime.now(timezone.utc).isoformat(),
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


def missing_node_model_rejection_with_queued_image(task, *, image_changes=None):
    """Build R -> P -> C locally; C is the only image-linked retry shape."""
    rewrite_input(task.service, "old", {**task.body, "product_work": {"product_ref": "fixture-product", "step_ref": "text"}})
    update(task.service, status="failed", error_code="RESULT_UNRECOVERABLE", upstream_outcome="unknown",
           recovery_reason="REQUEST_MESSAGE_NOT_FOUND",
           _attempt_reason="ORIGINAL_REQUEST_MESSAGE_NOT_FOUND", _attempt_finished_at=500,
           recovery_no_result_reads=9, _execution_wait_ended_at=500,
           _submission_started=True, _executing=False, _turn_reserved=False,
           _claim_id=None, _claim_until=None, send_count=1)
    task.service.submit("owner", {
        **{key: task.body[key] for key in TextTaskService.SUPERSEDE_FIELDS - {"supersedes_request_id"}},
        "client_request_id": "missing-parent", "supersedes_request_id": "old",
    }, source="internal:content")
    update(task.service, "missing-parent", status="failed", error_code="MODEL_NOT_FOUND",
           upstream_outcome="rejected", _submission_started=True, _executing=False, _turn_reserved=False,
           original_failure_phase="stream_open", original_exception_category="http",
           original_upstream_request_stage="conversation", original_http_status=404,
           original_upstream_error_form="detail_text", original_upstream_error={"message": "Model not found"})
    parent = receipt(task.service, "missing-parent")
    update(task.service, _product_work_key=parent["_product_work_key"])
    images = ImageTaskService(task.root / "image_tasks.json", admission=task.admission, store=task.service.store)
    images.submit_generation(
        {"id": "owner", "role": "admin", "external_image_client": True, "_trusted_source": "internal:content"},
        client_task_id="queued-image", prompt="fixture only", model="gpt-image-2", size=None,
        provider_binding_id=parent["provider_binding_id"], provider_account_identity=parent["provider_account_identity"],
        client_conversation_id=parent["client_conversation_id"], conversation_id=parent["conversation_id"],
        parent_message_id=parent["parent_message_id"], retain_conversation=True,
        product_work={"product_ref": "fixture-product", "step_ref": "image"},
    )
    with task.service.store.transaction() as db:
        image = task.service.store.read_receipt(db, "image", "owner", "queued-image")
        image.update(image_changes or {})
        task.service.store.write_receipt(db, "image", "owner", "queued-image", image)
    return {"client_request_id": "missing-model-retry", "supersedes_request_id": "missing-parent",
            "client_conversation_id": "session", "continue_before_queued_image_id": "queued-image"}


def test_missing_node_rejected_child_can_link_only_its_queued_unsent_image(task):
    envelope = missing_node_model_rejection_with_queued_image(task)
    accepted = task.service.submit("owner", envelope, source="internal:content")
    retry = receipt(task.service, "missing-model-retry")
    with task.service.store.connect() as db:
        image = task.service.store.read_receipt(db, "image", "owner", "queued-image")
    assert accepted["status"] == "queued"
    assert retry[TextTaskService.MISSING_NODE_RETRY_FIELD] == "queued-image"
    assert retry["_sequence"] > image["_sequence"]  # persisted receipts are never reordered
    context = task.admission.claim_next()
    assert context.request_id == "missing-model-retry"  # A4 cannot claim ahead of its linked C.
    task.admission.execute(context)  # Exercises the admission before-send revalidation.
    assert receipt(task.service, "missing-model-retry")["status"] == "succeeded"
    next_context = task.admission.claim_next()
    with task.service.store.connect() as db:
        queued_image = task.service.store.read_receipt(db, "image", "owner", "queued-image")
    assert next_context is not None, queued_image
    assert next_context.request_id == "queued-image"  # A4 resumes only after C is terminal.
    # The image's final admission guard must release only this proved R/P
    # chain.  It exercises the successor-success gate rather than merely the
    # planner's next-item ordering.
    next_context.before_send()
    with task.service.store.connect() as db:
        assert task.service.store.read_receipt(db, "image", "owner", "queued-image")["_submission_started"] is True
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", {**envelope, "client_request_id": "second-missing-model-retry"},
                            source="internal:content")
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"


def test_missing_node_parent_cannot_use_three_fields_or_rebind_its_image(task):
    envelope = missing_node_model_rejection_with_queued_image(task)
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", {key: value for key, value in envelope.items()
                                       if key != TextTaskService.MISSING_NODE_RETRY_FIELD}, source="internal:content")
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"
    accepted = task.service.submit("owner", envelope, source="internal:content")
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", {**envelope, TextTaskService.MISSING_NODE_RETRY_FIELD: "other-image"},
                            source="internal:content")
    assert error.value.code == "CONVERSATION_REQUEST_CONFLICT"
    assert task.service.submit("owner", envelope, source="internal:content") == accepted


def test_missing_node_late_root_change_blocks_a4_before_send(task):
    envelope = missing_node_model_rejection_with_queued_image(task)
    task.service.submit("owner", envelope, source="internal:content")
    retry = task.admission.claim_next()
    task.admission.execute(retry)
    assert receipt(task.service, "missing-model-retry")["status"] == "succeeded"
    # Structural linkage keeps the queue order durable, but a late root state
    # must still fence the actual image send rather than letting A4 overtake.
    update(task.service, status="succeeded", upstream_outcome="completed", content="late root result")
    image = task.admission.claim_next()
    assert image and image.request_id == "queued-image"
    with pytest.raises(AdmissionLost):
        image.before_send()


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_missing_node_failed_c_keeps_a4_behind_unknown_root(task, status):
    envelope = missing_node_model_rejection_with_queued_image(task)
    task.service.submit("owner", envelope, source="internal:content")
    update(task.service, "missing-model-retry", status=status, upstream_outcome="not_sent",
           _submission_started=False, _executing=False, _turn_reserved=False)
    assert task.admission.claim_next() is None


def test_missing_node_a4_execution_uses_c_terminal_parent_without_mutating_frozen_input(task):
    envelope = missing_node_model_rejection_with_queued_image(task)
    task.service.submit("owner", envelope, source="internal:content")
    retry = task.admission.claim_next()
    task.admission.execute(retry)
    assert receipt(task.service, "missing-model-retry")["parent_message_id"] == "new-answer"
    image = task.admission.claim_next()
    observed = []
    def run_image(_context, body):
        current_request.get().before_send()
        observed.append(deepcopy(body["payload"]))
    task.admission.register("image", run_image)
    task.admission.execute(image)
    assert observed and observed[0]["parent_message_id"] == "new-answer"
    with task.service.store.connect() as db:
        saved = task.service.store.read_receipt(db, "image", "owner", "queued-image")
        frozen = task.service.store.load_input(saved["_input_ref"])
    assert frozen["payload"]["parent_message_id"] == "previous-answer"
    assert saved["_missing_node_retry_request_parent_message_id"] == "new-answer"
    assert saved["request_parent_message_id"] == "new-answer"


@pytest.mark.parametrize("changes", [
    {"_recovery_paused": True}, {"_recovery_suppressed": True}, {"_work_key": "owned-work"},
    {"_completion": {"state": "pending"}}, {"_completion_of": "other"},
    {"_claim_id": "live", "_claim_until": 1100},
])
def test_missing_node_root_live_blockers_refuse_c(task, changes):
    envelope = missing_node_model_rejection_with_queued_image(task)
    update(task.service, **changes)
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", envelope, source="internal:content")
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"


def test_missing_node_retry_rejects_changed_image_frozen_input(task):
    envelope = missing_node_model_rejection_with_queued_image(task)
    with task.service.store.transaction() as db:
        image = task.service.store.read_receipt(db, "image", "owner", "queued-image")
        frozen = task.service.store.load_input(image["_input_ref"])
        frozen["payload"] = {**frozen["payload"], "prompt": "changed after acceptance"}
        image["_input_ref"] = task.service.store.save_input(frozen)
        task.service.store.write_receipt(db, "image", "owner", "queued-image", image)
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", envelope, source="internal:content")
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"


def test_missing_node_retry_rejects_other_unfinished_same_conversation(task):
    envelope = missing_node_model_rejection_with_queued_image(task)
    with task.service.store.transaction() as db:
        other = task.service.store.read_receipt(db, "image", "owner", "queued-image")
        other.update(id="unrelated-pending", _sequence=other["_sequence"] + 1)
        task.service.store.write_receipt(db, "image", "owner", "unrelated-pending", other)
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", envelope, source="internal:content")
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"


def test_missing_node_snapshot_reuses_one_receipt_index_and_preserves_image_guard(task):
    envelope = missing_node_model_rejection_with_queued_image(task)
    task.service.submit("owner", envelope, source="internal:content")
    original_validate = TextTaskService._missing_node_model_rejection_retry
    # `_snapshot` consults C once for both order-head and A4 linkage
    # projections.  Its supplied transaction index must prevent the helper
    # from independently decoding all historical receipts again.
    with patch.object(TextTaskService, "_missing_node_model_rejection_retry", wraps=original_validate) as validated:
        context = task.admission.claim_next()
    assert context and context.request_id == "missing-model-retry"
    assert validated.call_count == 1
    index = validated.call_args.kwargs["unfinished_by_conversation"]
    assert ("text", "old") in next(iter(index.values()))
    assert ("image", "queued-image") in next(iter(index.values()))

    # The indexed path must retain the old full-scan rejection: an unrelated
    # unfinished image in the exact physical conversation still blocks C.
    with task.admission.store.transaction() as db:
        successor = task.admission.store.read_receipt(db, "text", "owner", "missing-model-retry")
        image = task.admission.store.read_receipt(db, "image", "owner", "queued-image")
        other = {**image, "id": "other-pending-image", "_sequence": image["_sequence"] + 10}
        task.admission.store.write_receipt(db, "image", "owner", "other-pending-image", other)
        receipts = list(task.admission.store.receipts(db))
        indexed = task.admission._missing_node_unfinished_by_conversation(receipts)
        assert not TextTaskService._missing_node_model_rejection_retry(
            task.admission.store, db, "owner", successor, 1000, unfinished_by_conversation=indexed)
        # The send-edge fallback remains the full current read and must agree.
        assert not TextTaskService._missing_node_model_rejection_retry(
            task.admission.store, db, "owner", successor, 1000)


def test_missing_node_retry_revalidates_root_before_send_and_same_id_is_idempotent(task):
    envelope = missing_node_model_rejection_with_queued_image(task)
    task.service.submit("owner", envelope, source="internal:content")
    # A restart/control replay can only read back this C; it cannot create D.
    assert task.service.submit("owner", envelope, source="internal:content")["request_id"] == "missing-model-retry"
    context = task.admission.claim_next()
    assert context.request_id == "missing-model-retry"
    update(task.service, status="succeeded", content="late root result")
    with pytest.raises(ConversationBindingError):
        context.before_send()
    assert receipt(task.service, "missing-model-retry")["_submission_started"] is False
    with task.service.store.connect() as db:
        retry = task.service.store.read_receipt(db, "text", "owner", "missing-model-retry")
        assert TextTaskService._missing_node_model_rejection_retry_consumed(
            task.service.store, db, "owner", retry)


@pytest.mark.parametrize("changes", [
    {"_submission_started": True}, {"request_message_id": "already-started"},
    {"status": "error", "upstream_unfinished": True}, {"provider_binding_id": "foreign-binding"},
])
def test_missing_node_retry_rejects_changed_or_started_linked_image(task, changes):
    envelope = missing_node_model_rejection_with_queued_image(task, image_changes=changes)
    with pytest.raises(ConversationBindingError) as error:
        task.service.submit("owner", envelope, source="internal:content")
    assert error.value.code == "CHAT_SUPERSEDE_INVALID"
    assert receipt(task.service, "missing-model-retry") is None


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
