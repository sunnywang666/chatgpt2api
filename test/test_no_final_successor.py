"""An explicit legacy text continuation keeps the original task and context."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from services.conversation_binding_service import ConversationBindingError, ConversationBindingService
from services.product_work import ProductWorkService
from services.request_context import current_request
from services.text_task_service import TextTaskService
from services.work_lifecycle import WorkLifecycleError, WorkLifecycleService
from test.test_explicit_unknown_successor import task, row, update
from test.test_bound_text_archive import bound_chat
from test.test_pool_admission import Clock, build


def no_final(h):
    update(h.service, recovery_reason="REQUEST_RESULT_NOT_FOUND", recovery_no_result_reads=3,
           _attempt_reason="ORIGINAL_RESULT_NO_FINAL", _attempt_finished_at=h.clock(),
           request_parent_message_id="previous-answer")
    h.old = row(h.service)
    h.envelope["continue_after_no_final"] = True
    return h


@pytest.fixture
def product_no_final(tmp_path):
    (tmp_path / "accounts.json").write_text(json.dumps([{"access_token": "fixture",
        "account_id": "physical", "provider_account_identity": "original-account",
        "conversation_binding_ids": ["original-binding"], "type": "Plus", "status": "正常",
        "quota": 99, "source_type": "web"}]))
    clock = Clock()
    _, store, admission = build(tmp_path, clock)
    sent = []
    def runner(body, on_cursor):
        current_request.get().before_send()
        sent.append(deepcopy(body))
        return {"content": "successful explicit successor", "parent_message_id": "successor-answer"}
    service = TextTaskService(store.path, runner=runner, clock=clock, admission=admission)
    admission.register("text", lambda ctx, body: service._run(ctx.owner, ctx.request_id, body))
    metadata = {"product_ref": "fixture-product", "step_ref": "title"}
    body = {"client_request_id": "old", "client_conversation_id": "product-session",
        "provider_binding_id": "original-binding", "provider_account_identity": "original-account",
        "conversation_id": "original-chat", "parent_message_id": "previous-answer", "model": "fixture-text",
        "messages": [{"role": "user", "content": "original retained instructions"}], "product_work": metadata}
    service.submit("owner", body, source="internal:content")
    update(service, status="failed", error_code="RESULT_UNRECOVERABLE", upstream_outcome="unknown",
           recovery_reason="REQUEST_RESULT_NOT_FOUND", recovery_no_result_reads=3,
           _attempt_reason="ORIGINAL_RESULT_NO_FINAL", _attempt_finished_at=clock(),
           _execution_wait_ended_at=clock(), _submission_started=True, _last_sent_sequence=0,
           _executing=False, _claim_id=None, _claim_until=None, _turn_reserved=False, send_count=1)
    envelope = {k: v for k, v in body.items() if k in service.SUPERSEDE_FIELDS}
    envelope.update(client_request_id="successor", supersedes_request_id="old", continue_after_no_final=True)
    return SimpleNamespace(service=service, admission=admission, clock=clock, sent=sent,
                           envelope=envelope, old=row(service), root=tmp_path)


def complete_product_no_final_successor(h):
    h.service.submit("owner", h.envelope, source="internal:content")
    context = h.admission.claim_next()
    assert context and context.request_id == "successor"
    h.admission.execute(context)
    h.admission.update_claim(context, _claim_id=None, _claim_until=0, _turn_reserved=False)
    return row(h.service), row(h.service, "successor")


def product_lifecycle(h):
    calls = []
    def archive(owner, request_id, archived):
        calls.append((owner, request_id, archived))
        with h.service.store.connect() as db:
            receipt = h.service.store.read_receipt(db, "text", owner, request_id)
        return {"request_id": request_id, "conversation_id": receipt["conversation_id"], "archived": archived}
    text = SimpleNamespace(store=h.service.store, admission=None, archive_product_work=archive)
    product = ProductWorkService(text, SimpleNamespace(archive_product_work=archive), clock=h.clock)
    lifecycle = WorkLifecycleService(text, SimpleNamespace(archive_product_work=archive), clock=h.clock)
    identity = {"id": "owner", "role": "admin", "_trusted_source": "internal:content"}
    return product, lifecycle, calls, identity


def graph(h):
    user = h.old["request_message_id"]
    def node(key, parent, children, role, content, end=None):
        return {"parent": parent, "children": children, "message": {"id": key,
            "author": {"role": role}, "status": "finished_successfully", "end_turn": end,
            "channel": None, "content": content}}
    return {"conversation_id": "original-chat", "current_node": "tool", "is_archived": False,
        "mapping": {
            user: node(user, "previous-answer", ["code"], "user", {"content_type": "text", "parts": ["original input"]}),
            "code": node("code", user, ["tool"], "assistant", {"content_type": "code", "text": "inspect input"}, False),
            "tool": node("tool", "code", [], "tool", {"content_type": "multimodal_text", "parts": ["retained tool context"]})}}


def user_leaf_graph(h):
    document = graph(h)
    user = h.old["request_message_id"]
    document["mapping"] = {user: document["mapping"][user]}
    document["mapping"][user]["children"] = []
    document["current_node"] = user
    return document


def test_explicit_mode_preserves_original_and_immutable_input_after_restart(task):
    h = no_final(task)
    first = h.service.submit("owner", h.envelope)
    assert first["status"] == "queued"
    original_body = h.service.store.load_input(row(h.service, "successor")["_input_ref"])
    assert original_body == {**h.body, "client_request_id": "successor", "supersedes_request_id": "old",
                             "continue_after_no_final": True}
    _, store, admission = build(h.root, h.clock)
    service = type(h.service)(store.path, runner=h.service.runner, clock=h.clock, admission=admission)
    admission.register("text", lambda ctx, body: service._run(ctx.owner, ctx.request_id, body))
    admission.execute(admission.claim_next())
    assert len(h.sent) == 1 and h.sent[0]["_supersedes_no_final_original"] == h.old
    assert service.submit("owner", h.envelope)["status"] == "succeeded"
    assert row(service) == h.old
    with pytest.raises(ConversationBindingError):
        service.submit("owner", {**h.envelope, "client_request_id": "second"})


def test_verified_stopped_retry_cursor_allows_the_single_no_final_successor(task):
    h = no_final(task)
    original = row(h.service)
    proof = {
        "conversation_id": original["conversation_id"],
        "request_message_id": original["request_message_id"],
        "retry_parent_message_id": original["request_message_id"],
        "observed_at": h.clock(),
    }
    # Match the production receipt shape: recovery stopped after repeated
    # missing-result reads while the exact original-branch observation remained.
    update(h.service, recovery_no_result_reads=18, _retry_cursor=proof)
    original = row(h.service)

    accepted = h.service.submit("owner", h.envelope)

    assert accepted["status"] == "queued"
    assert row(h.service) == original
    successor = row(h.service, "successor")
    assert successor["_supersedes_no_final"] is True
    assert successor["_supersedes_input_hash"]
    assert h.admission.claim_next().request_id == "successor"


def test_no_final_successor_keeps_unrelated_queued_turn_ahead(task):
    h = no_final(task)
    # A distinct queued request in the same conversation is never skipped by
    # an explicit no-final successor, even though the original is stopped.
    # The successful single-successor path is covered by the preceding test.
    h.service.submit("owner", {**h.body, "client_request_id": "unrelated-queued",
                               "parent_message_id": "unrelated-parent"})
    with pytest.raises(ConversationBindingError) as error:
        h.service.submit("owner", h.envelope)
    assert error.value.code == "CHAT_SUPERSEDE_CONFLICT"
    assert row(h.service, "successor") is None
    assert row(h.service, "unrelated-queued")["status"] == "queued"
    assert row(h.service) == h.old and not h.sent


def test_product_no_final_successor_allows_authorized_close_and_archives_successor(product_no_final):
    h = product_no_final
    original_before = row(h.service)
    with h.service.store.connect() as db:
        original_hash = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", ("owner", "old")).fetchone()[0]

    original, successor = complete_product_no_final_successor(h)
    with h.service.store.connect() as db:
        successor_hash = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", ("owner", "successor")).fetchone()[0]
    assert original == original_before
    assert successor["status"] == "succeeded" and successor["upstream_outcome"] == "completed"
    assert successor.get("upstream_unfinished") is not True and successor["content"] == "successful explicit successor"
    assert successor["_product_work_key"] == original["_product_work_key"]
    assert h.service.store.load_input(original["_input_ref"])["client_request_id"] == "old"
    assert h.service.store.load_input(successor["_input_ref"])["client_request_id"] == "successor"
    assert original_hash == successor["_supersedes_input_hash"] and successor_hash

    product, lifecycle, calls, identity = product_lifecycle(h)

    closing = product.close(identity, "fixture-product", results_saved=True)
    assert closing["state"] == "closing" and closing["results_saved"]
    assert lifecycle.process_one()
    assert product.get(identity, "fixture-product")["state"] == "completed"
    assert calls == [("owner", "successor", True)]
    assert row(h.service) == original_before and len(h.sent) == 1


def test_cancelled_later_product_turn_no_longer_blocks_exact_no_final_successor(product_no_final):
    h = product_no_final
    later = {**h.service.store.load_input(h.old["_input_ref"]), "client_request_id": "later",
             "messages": [{"role": "user", "content": "different later business attempt"}],
             "product_work": {"product_ref": "fixture-product", "step_ref": "later"}}
    h.service.submit("owner", later, source="internal:content")
    product, _, _, identity = product_lifecycle(h)
    cancelled = product.cancel_step(identity, "fixture-product", "later", 1)
    assert cancelled["cancelled_unsent"]["request_id"] == "later"
    accepted = h.service.submit("owner", h.envelope, source="internal:content")
    assert accepted["status"] == "queued" and accepted["supersedes_request_id"] == "old"


def test_other_queued_product_turn_still_blocks_exact_no_final_successor_after_cancel(product_no_final):
    h = product_no_final
    original_body = h.service.store.load_input(h.old["_input_ref"])
    h.service.submit("owner", {**original_body, "client_request_id": "later",
                                 "messages": [{"role": "user", "content": "different later business attempt"}],
                                 "product_work": {"product_ref": "fixture-product", "step_ref": "later"}},
                     source="internal:content")
    product, _, _, identity = product_lifecycle(h)
    product.cancel_step(identity, "fixture-product", "later", 1)
    h.service.submit("owner", {**original_body, "client_request_id": "still-queued",
                                 "messages": [{"role": "user", "content": "independent unsent business turn"}],
                                 "product_work": {"product_ref": "fixture-product", "step_ref": "blocked"}},
                     source="internal:content")
    with pytest.raises(ConversationBindingError, match="cannot advance") as error:
        h.service.submit("owner", h.envelope, source="internal:content")
    assert error.value.code == "CHAT_SUPERSEDE_CONFLICT"


def test_product_no_final_successor_survives_resume_and_later_cycle(product_no_final):
    h = product_no_final
    original_before, successor = complete_product_no_final_successor(h)
    product, lifecycle, calls, identity = product_lifecycle(h)
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    assert lifecycle.process_one()
    assert product.get(identity, "fixture-product")["state"] == "completed"

    restarted = ProductWorkService(lifecycle.text, lifecycle.images, clock=h.clock)
    assert restarted.resume(identity, "fixture-product", expected_version=1)["state"] == "active"
    original_body = h.service.store.load_input(original_before["_input_ref"])
    h.service.submit("owner", {**original_body, "client_request_id": "v2-normal",
                                 "parent_message_id": successor["parent_message_id"],
                                 "product_work": {"product_ref": "fixture-product", "step_ref": "v2-normal"}},
                     source="internal:content")
    context = h.admission.claim_next()
    assert context and context.request_id == "v2-normal"
    h.admission.execute(context)
    h.admission.update_claim(context, _claim_id=None, _claim_until=0, _turn_reserved=False)

    closing = restarted.close(identity, "fixture-product", results_saved=True, expected_version=2)
    assert closing["state"] == "closing" and closing["version"] == 2
    assert lifecycle.process_one()
    result = restarted.get(identity, "fixture-product")
    assert result["state"] == "completed" and result["version"] == 2
    assert calls == [("owner", "successor", True), ("owner", "v2-normal", True)]
    assert row(h.service) == original_before


def test_product_no_final_v4_successor_rederives_then_closes_and_archives(product_no_final):
    from test.test_required_attributes_successor import KIND, fields, prompt

    h = product_no_final
    original_body = h.service.store.load_input(row(h.service)["_input_ref"])
    original_body.update(model="gpt-5-6-instant", thinking_effort="standard", messages=[{
        "role": "user", "content": [
            {"type": "text", "text": prompt(fields())},
            {"type": "text", "text": '{"image_ref":"actual-image-not-source-1"}'},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDRA=="}},
            {"type": "text", "text": '{"image_ref":"another-original-image"}'},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,RUZH"}},
        ],
    }])
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        original.update(_input_ref=h.service.store.save_input(original_body), model="gpt-5-6-instant",
                        thinking_effort="standard")
        h.service.store.write_receipt(db, "text", "owner", "old", original)
        db.execute("UPDATE requests SET request_hash=? WHERE owner=? AND id=?",
                   (TextTaskService._submission_identity("owner", original_body)[1], "owner", "old"))
    h.envelope["derived_input"] = {"kind": KIND}

    original, successor = complete_product_no_final_successor(h)
    assert successor["_derived_input"]["kind"] == KIND
    assert successor["model"] == "gpt-5-6-thinking"
    assert h.service.store.load_input(successor["_input_ref"])["thinking_effort"] == "high"
    product, lifecycle, calls, identity = product_lifecycle(h)
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    assert lifecycle.process_one()
    assert product.get(identity, "fixture-product")["state"] == "completed"
    assert calls == [("owner", "successor", True)]
    assert row(h.service) == original


def test_product_no_final_successor_rejects_tampered_retained_input_hash(product_no_final):
    h = product_no_final
    complete_product_no_final_successor(h)
    with h.service.store.transaction() as db:
        db.execute("UPDATE requests SET request_hash=? WHERE owner=? AND id=?", ("tampered", "owner", "successor"))
    product, _, calls, identity = product_lifecycle(h)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_TURN_UNFINISHED"):
        product.close(identity, "fixture-product", results_saved=True)
    assert calls == []


@pytest.mark.parametrize("request_id,changes", [
    ("old", {"_recovery_paused": True}),
    ("old", {"_claim_id": "active", "_executing": True}),
    ("successor", {"status": "failed", "upstream_outcome": "unknown"}),
    ("successor", {"provider_account_identity": "different-account"}),
    ("successor", {"_product_work_key": "different-product"}),
    ("successor", {"_derived_input": {"kind": "attributes_required_only_v4", "original_input_hash": "fake"}}),
])
def test_product_no_final_successor_requires_unchanged_exact_proof(product_no_final, request_id, changes):
    h = product_no_final
    complete_product_no_final_successor(h)
    with h.service.store.transaction() as db:
        receipt = h.service.store.read_receipt(db, "text", "owner", request_id)
        receipt.update(changes)
        h.service.store.write_receipt(db, "text", "owner", request_id, receipt)
    product, _, calls, identity = product_lifecycle(h)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_"):
        product.close(identity, "fixture-product", results_saved=True)
    assert calls == []


def test_product_no_final_successor_rejects_second_registered_fallback(product_no_final):
    h = product_no_final
    original, successor = complete_product_no_final_successor(h)
    with h.service.store.transaction() as db:
        fallback = {**successor, "request_id": "fallback", "_codex_fallback_of": "old",
                    "_supersedes_request_id": None, "_sequence": successor["_sequence"] + 1}
        db.execute("INSERT INTO requests VALUES(?,?,?,?)",
                   ("owner", "fallback", "fallback-hash", json.dumps(fallback)))
    product, _, calls, identity = product_lifecycle(h)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_TURN_UNFINISHED"):
        product.close(identity, "fixture-product", results_saved=True)
    assert row(h.service) == original and calls == []


def test_product_no_final_archive_rechecks_late_original_change(product_no_final):
    h = product_no_final
    original, _ = complete_product_no_final_successor(h)
    product, lifecycle, calls, identity = product_lifecycle(h)
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    with h.service.store.transaction() as db:
        changed = h.service.store.read_receipt(db, "text", "owner", "old")
        changed["_recovery_paused"] = True
        h.service.store.write_receipt(db, "text", "owner", "old", changed)
    assert not lifecycle.process_one()
    result = product.get(identity, "fixture-product")
    assert result["state"] == "closing" and result["slot_held"] and result["archive"]["pending"] == 1
    assert row(h.service)["_recovery_paused"] is True and row(h.service)["_input_ref"] == original["_input_ref"]
    assert calls == []


def test_failed_explicit_read_keeps_qualified_no_final_source_and_sanitizes_error(task):
    h = no_final(task)
    h.service.recovery_reader = Mock(side_effect=RuntimeError(
        "token=secret-value prompt=do-not-persist"
    ))

    h.service.recover("owner", "old", explicit_ended_recheck=True)
    original = row(h.service)
    assert original["_attempt_reason"] == "ORIGINAL_RESULT_NO_FINAL"
    assert original["recovery_reason"] is None
    assert original["recovery_error_code"] == "RECOVERY_READ_FAILED"
    assert original["recovery_last_read_error"] == {
        "phase": "conversation_detail", "category": "other",
        "http_status": None, "retry_after_seconds": None, "candidate_ref": None,
    }
    assert "secret-value" not in json.dumps(original)
    assert h.service._no_final_successor_source(original, h.clock())

    accepted = h.service.submit("owner", h.envelope)
    assert accepted["status"] == "queued"
    assert h.service.submit("owner", h.envelope) == accepted
    assert row(h.service, "successor")["_supersedes_no_final"] is True
    assert not h.sent


def test_legacy_detail_free_read_failure_keeps_stopped_no_final_eligibility(task):
    h = no_final(task)
    update(h.service, recovery_reason=None, recovery_error_code="RECOVERY_READ_FAILED",
           recovery_last_read_error=None, recovery_no_result_reads=40)
    original = row(h.service)
    assert h.service._no_final_successor_source(original, h.clock())

    accepted = h.service.submit("owner", h.envelope)
    assert accepted["status"] == "queued"
    assert row(h.service, "successor")["_supersedes_no_final"] is True


def test_legacy_no_final_does_not_create_child_while_bound_account_is_invalidated(task, monkeypatch):
    h = no_final(task)
    # This matches the three pre-typed receipts already stopped in production:
    # durable no-final proof, generic read failure, and no saved error detail.
    update(h.service, recovery_reason=None, recovery_error_code="RECOVERY_READ_FAILED",
           recovery_last_read_error=None, recovery_no_result_reads=9)
    import services.text_task_service as text_module
    monkeypatch.setattr(text_module.account_service, "bound_text_recovery_auth_unavailable", lambda _binding: True)
    with pytest.raises(ConversationBindingError) as error:
        h.service.submit("owner", h.envelope)
    assert error.value.code == "CHAT_SUPERSEDE_AUTH_REQUIRED"
    assert row(h.service, "successor") is None and not h.sent

    monkeypatch.setattr(text_module.account_service, "bound_text_recovery_auth_unavailable", lambda _binding: False)
    assert h.service.submit("owner", h.envelope)["status"] == "queued"
    assert row(h.service, "successor")["_supersedes_no_final"] is True


def test_known_local_bound_auth_failure_is_typed_without_erasing_no_final_proof(task, monkeypatch):
    h = no_final(task)
    import services.conversation_binding_service as binding
    fake_accounts = SimpleNamespace(
        get_bound_account_identity=lambda _binding: "original-account",
        get_bound_text_access_token=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("token=secret-value prompt=do-not-persist")
        ),
        bound_text_recovery_auth_unavailable=lambda _binding: True,
    )
    monkeypatch.setattr(binding, "account_service", fake_accounts)
    h.service.recovery_reader = binding.ConversationBindingService().read_text_request

    h.service.recover("owner", "old", explicit_ended_recheck=True)
    original = row(h.service)
    assert original["recovery_error_code"] == "RECOVERY_AUTH_REQUIRED"
    assert original["recovery_last_read_error"] == {
        "phase": "conversation_detail", "category": "other",
        "http_status": None, "retry_after_seconds": None, "candidate_ref": None,
    }
    assert "secret-value" not in json.dumps(original)
    assert h.service._no_final_successor_source(original, h.clock())


@pytest.mark.parametrize("changes", [
    {"status": "succeeded", "content": "new upstream answer", "error_code": None,
     "upstream_outcome": "completed"},
    {"recovery_reason": "REQUEST_CONVERSATION_ADVANCED", "recovery_error_code": None,
     "recovery_last_read_error": None},
    {"provider_account_identity": "changed-account"},
])
def test_new_success_branch_or_identity_evidence_still_blocks_no_final_successor(task, changes):
    h = no_final(task)
    update(h.service, **changes)
    with pytest.raises(ConversationBindingError):
        h.service.submit("owner", h.envelope)
    assert row(h.service, "successor") is None and not h.sent


@pytest.mark.parametrize("change", ["mismatched-conversation", "missing-observed-at", "image-source"])
def test_unverified_or_non_text_retry_cursor_still_blocks_no_final_successor(task, change):
    h = no_final(task)
    original = row(h.service)
    proof = {
        "conversation_id": original["conversation_id"],
        "request_message_id": original["request_message_id"],
        "retry_parent_message_id": original["request_message_id"],
        "observed_at": h.clock(),
    }
    if change == "mismatched-conversation":
        proof["conversation_id"] = "different-conversation"
    elif change == "missing-observed-at":
        proof.pop("observed_at")
    else:
        proof["source"] = "absent_image_thread_request"
    update(h.service, _retry_cursor=proof)

    with pytest.raises(ConversationBindingError):
        h.service.submit("owner", h.envelope)

    assert row(h.service, "successor") is None and not h.sent


@pytest.mark.parametrize("changes", [
    {"recovery_no_result_reads": 2}, {"_attempt_finished_at": None}, {"_attempt_finished_at": float("inf")},
    {"_attempt_reason": "other"}, {"_execution_wait_ended_at": None}, {"_work_key": "work"},
    {"_completion": {"state": "checking"}}, {"_retry_cursor": {"anything": True}},
    {"_retry_cursor": {"conversation_id": "wrong", "request_message_id": "wrong",
                         "retry_parent_message_id": "tail", "observed_at": 1}},
    {"_retry_cursor": {"conversation_id": "original-chat", "request_message_id": "missing-time",
                         "retry_parent_message_id": "tail"}},
    {"_retry_cursor": {"conversation_id": "original-chat", "request_message_id": "missing-source",
                         "retry_parent_message_id": "tail", "observed_at": 1,
                         "source": "absent_image_thread_request"}},
    {"_recovery_paused": True}, {"_executing": True}, {"_turn_reserved": True},
])
def test_unqualified_original_cannot_enqueue(task, changes):
    h = no_final(task); update(h.service, **changes)
    with pytest.raises(ConversationBindingError):
        h.service.submit("owner", h.envelope)
    assert row(h.service, "successor") is None and not h.sent


def test_mode_is_explicit_and_cannot_be_changed_on_same_id(task):
    h = no_final(task)
    without = {k: v for k, v in h.envelope.items() if k != "continue_after_no_final"}
    with pytest.raises(ConversationBindingError): h.service.submit("owner", without)
    h.service.submit("owner", h.envelope)
    with pytest.raises(ConversationBindingError): h.service.submit("owner", without)
    from api.ai import ConversationBindingTextRequest
    assert ConversationBindingTextRequest.model_validate(h.envelope).model_dump(exclude_unset=True) == h.envelope
    for bad in ({**h.envelope, "continue_after_no_final": False},
                {**h.envelope, "continue_after_no_final": "true"},
                {**h.body, "continue_after_no_final": True},
                {**h.envelope, "derived_input": {"kind": "any"}}):
        with pytest.raises(ValidationError): ConversationBindingTextRequest.model_validate(bad)


@pytest.mark.parametrize("change", ["archived", "parent", "head", "fork", "later_user", "code_running",
                                  "tool_running", "final", "end_turn", "children", "alias"])
def test_graph_changes_fail_closed(task, change):
    h = no_final(task); d = graph(h)
    assert ConversationBindingService._no_final_successor_tail(d, h.old)[0] == "tool"
    if change == "archived": d["is_archived"] = True
    elif change == "parent": d["mapping"][h.old["request_message_id"]]["parent"] = "wrong"
    elif change == "head": d["current_node"] = "code"
    elif change == "fork": d["mapping"]["other"] = {"parent": h.old["request_message_id"]}
    elif change == "later_user": d["mapping"]["tool"]["message"]["author"]["role"] = "user"
    elif change.endswith("running"): d["mapping"][change.split("_")[0]]["message"]["status"] = "in_progress"
    elif change == "final": d["mapping"]["code"]["message"]["content"] = {"content_type": "text", "parts": ["partial answer"]}
    elif change == "end_turn": d["mapping"]["code"]["message"]["end_turn"] = True
    elif change == "children": d["mapping"]["tool"]["children"] = ["missing"]
    elif change == "alias": d["mapping"]["tool"]["message"]["id"] = "alias"
    with pytest.raises(ConversationBindingError): ConversationBindingService._no_final_successor_tail(d, h.old)


@pytest.mark.parametrize("tail,drift", [
    ("tool", None), ("tool", "end_turn"), ("tool", "tool_content"), ("tool", "code_content"),
    ("user", None), ("user", "user_content"), ("user", "late_answer"), ("user", "in_progress"),
    ("user", "head"), ("user", "unlisted_child"), ("user", "archived"), ("user", "parent"),
    ("user", "alias"),
])
def test_fresh_tail_is_used_and_rechecked_without_rewriting_original(task, bound_chat, monkeypatch, tail, drift):
    import services.conversation_binding_service as module
    h, chat = no_final(task), bound_chat
    chat.document = user_leaf_graph(h) if tail == "user" else graph(h)
    expected_tail = h.old["request_message_id"] if tail == "user" else "tool"
    h.service.runner = chat.service.complete_text
    original_get = module.OpenAIBackendAPI._get_conversation
    reads = []
    def read(backend, cid, **kw):
        reads.append(cid)
        return original_get(backend, cid)
    monkeypatch.setattr(module.OpenAIBackendAPI, "_get_conversation", read)
    sends = []
    def events(backend, **kwargs):
        if drift == "end_turn":
            chat.document["mapping"]["code"]["message"]["end_turn"] = True
        elif drift == "tool_content":
            chat.document["mapping"]["tool"]["message"]["content"]["parts"] = ["changed context"]
        elif drift == "code_content":
            chat.document["mapping"]["code"]["message"]["content"]["text"] = "changed code"
        elif drift == "user_content":
            chat.document["mapping"][expected_tail]["message"]["content"]["parts"] = ["changed input"]
        elif drift == "late_answer":
            chat.document["mapping"][expected_tail]["children"] = ["late"]
            chat.document["mapping"]["late"] = {"parent": expected_tail, "children": [], "message": {
                "id": "late", "author": {"role": "assistant"}, "status": "finished_successfully",
                "end_turn": True, "content": {"content_type": "text", "parts": ["original answer"]}}}
            chat.document["current_node"] = "late"
        elif drift == "in_progress":
            chat.document["mapping"][expected_tail]["message"]["status"] = "in_progress"
        elif drift == "head":
            chat.document["current_node"] = "different"
        elif drift == "unlisted_child":
            chat.document["mapping"]["hidden"] = {"parent": expected_tail}
        elif drift == "archived":
            chat.document["is_archived"] = True
        elif drift == "parent":
            chat.document["mapping"][expected_tail]["parent"] = "different"
        elif drift == "alias":
            chat.document["mapping"][expected_tail]["message"]["id"] = "alias"
        backend.text_pre_send_check(None)
        current_request.get().before_send()
        sends.append(kwargs)
        assert kwargs["parent_message_id"] == expected_tail and kwargs["conversation_id"] == "original-chat"
        assert kwargs["messages"] == h.body["messages"] and kwargs["model"] == h.body["model"]
        backend.text_cursor_callback({"request_parent_message_id": expected_tail, "_submission_parent_message_id": expected_tail})
        yield {"type": "conversation.delta", "conversation_id": "original-chat", "delta": "new answer"}
    monkeypatch.setattr(module, "conversation_events", events)
    monkeypatch.setattr(ConversationBindingService, "_read_text_request_result", staticmethod(
        lambda backend, receipt: {"status": "succeeded", "content": "new answer", "parent_message_id": "new-answer"}))
    h.service.submit("owner", h.envelope)
    h.admission.execute(h.admission.claim_next())
    result = row(h.service, "successor")
    assert reads == ["original-chat", "original-chat"]
    assert row(h.service) == h.old
    assert result["_submission_parent_message_id"] == "previous-answer"
    if drift:
        assert not sends and result["upstream_outcome"] == "not_sent"
        assert result["status"] == "failed" and result["error_code"] == "CHAT_SUPERSEDE_CURSOR_CHANGED"
    else:
        assert len(sends) == 1 and result["status"] == "succeeded"
        assert result["request_parent_message_id"] == expected_tail
        assert h.service.submit("owner", h.envelope)["status"] == "succeeded"
        with pytest.raises(ConversationBindingError):
            h.service.submit("owner", {**h.envelope, "client_request_id": "second"})
        assert len(sends) == 1


def test_no_final_two_processes_keep_one_successor_and_one_claim(task):
    from test.test_explicit_unknown_successor import test_two_processes_cannot_register_two_successors_and_only_one_can_claim
    test_two_processes_cannot_register_two_successors_and_only_one_can_claim(no_final(task))
