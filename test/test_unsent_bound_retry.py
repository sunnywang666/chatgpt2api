"""One fresh-ID retry for a proved pre-send binding mismatch (no upstream I/O)."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from contextlib import nullcontext
import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.ai import ConversationBindingTextRequest
from services.conversation_binding_service import ConversationBindingError
from services.conversation_binding_service import ConversationBindingService
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_pool_admission import Clock, build


SOURCE = "internal:listing"


def read(service, request_id):
    with service.store.connect() as db:
        return service.store.read_receipt(db, "text", "owner", request_id)


@pytest.fixture
def retry(tmp_path):
    (tmp_path / "accounts.json").write_text(json.dumps([{
        "access_token": "fixture", "account_id": "fixture-account",
        "provider_account_identity": "account-0", "conversation_binding_ids": ["binding-0"],
        "type": "Plus", "status": "正常", "quota": 99, "source_type": "web",
    }]))
    clock = Clock()
    _, store, admission = build(tmp_path, clock)
    sent, phase = [], {"binding_failure": True}
    def runner(body, on_cursor):
        if phase["binding_failure"]:
            raise ConversationBindingError("fixture parent changed", code="CONVERSATION_BINDING_MISMATCH")
        current_request.get().before_send()
        sent.append(deepcopy(body))
        return {"content": "fixture answer", "parent_message_id": "C-final"}
    service = TextTaskService(store.path, runner=runner, admission=admission, clock=clock)
    admission.register("text", lambda ctx, body: service._run(ctx.owner, ctx.request_id, body))
    original = {
        "client_request_id": "P", "client_conversation_id": "client-0", "model": "fixture-text",
        "provider_binding_id": "binding-0", "provider_account_identity": "account-0",
        "conversation_id": "conversation-0", "parent_message_id": "A",
        "messages": [{"role": "user", "content": "retained private input"}],
        "product_work": {"product_ref": "product-0", "step_ref": "direct-listing.content.copy:P"},
    }
    service.submit("owner", original, source=SOURCE)
    original_context = admission.claim_next()
    assert original_context and original_context.request_id == "P"
    admission.execute(original_context)
    original_receipt = read(service, "P")
    assert original_receipt["error_code"] == "CONVERSATION_BINDING_MISMATCH"
    assert original_receipt["upstream_outcome"] == "not_sent"
    assert [event["stage"] for event in original_receipt["_execution_timeline"]] == [
        "accepted", "execution_claimed", "task_finished"]
    # The failed runner's historical scheduler claim must be expired, just as
    # the observed P receipt is; an active claim is deliberately ineligible.
    clock.now = 2000.0
    phase["binding_failure"] = False
    envelope = {
        "client_request_id": "N", "retry_unsent_request_id": "P",
        "provider_binding_id": "binding-0", "provider_account_identity": "account-0",
        "client_conversation_id": "client-0", "conversation_id": "conversation-0",
        "parent_message_id": "B",
    }
    with store.connect() as db:
        before = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", ("owner", "P")).fetchone()
    return service, admission, store, clock, original, envelope, sent, before, tmp_path


def test_retry_retains_p_byte_for_byte_and_derives_fresh_product_step(retry):
    service, admission, store, _, original, envelope, sent, before, _ = retry
    accepted = service.submit("owner", envelope, source=SOURCE)
    child = read(service, "N")
    saved = service.store.load_input(child["_input_ref"])
    assert accepted["status"] == "queued"
    assert child["_retry_unsent_of"] == "P"
    assert child["_retry_unsent_input_hash"] == before[0]
    assert saved["messages"] == original["messages"]
    assert saved["parent_message_id"] == "B"
    assert saved["product_work"] == {"product_ref": "product-0", "step_ref": "direct-listing.content.copy:N"}
    with store.connect() as db:
        assert db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", ("owner", "P")).fetchone() == before
    context = admission.claim_next()
    assert context and context.request_id == "N"
    admission.execute(context)
    assert read(service, "N")["status"] == "succeeded", (read(service, "N").get("error_code"),
                                                               read(service, "N").get("upstream_outcome"))
    assert len(sent) == 1 and sent[0]["parent_message_id"] == "B"
    assert service.submit("owner", envelope, source=SOURCE)["status"] == "succeeded"


def test_same_id_is_frozen_after_original_changes_and_second_id_is_rejected(retry):
    service, _, store, _, _, envelope, _, _, _ = retry
    first = service.submit("owner", envelope, source=SOURCE)
    with pytest.raises(ConversationBindingError) as error:
        service.submit("owner", {**envelope, "client_request_id": "N2"}, source=SOURCE)
    assert error.value.code == "CHAT_UNSENT_RETRY_CONFLICT"
    with store.transaction() as db:
        original = store.read_receipt(db, "text", "owner", "P")
        original.update(status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN", upstream_outcome="unknown",
                        _submission_started=True, send_count=1)
        store.write_receipt(db, "text", "owner", "P", original)
    assert service.submit("owner", envelope, source=SOURCE) == first
    with pytest.raises(ConversationBindingError) as error:
        service.submit("owner", {**envelope, "parent_message_id": "C"}, source=SOURCE)
    assert error.value.code == "CONVERSATION_REQUEST_CONFLICT"


@pytest.mark.parametrize("changes", [
    {"status": "unknown", "error_code": "CONVERSATION_OUTCOME_UNKNOWN", "upstream_outcome": "unknown"},
    {"send_count": 1}, {"_submission_started": True}, {"_recovery_suppressed": True},
    {"_execution_timeline": [{"stage": "send_call_started"}]}, {"_claim_id": "live", "_claim_until": 3000},
    {"_execution_timeline": [{"stage": "accepted"}, {"stage": "execution_claimed"},
                               {"stage": "task_finished", "task_status": "succeeded"}]},
])
def test_only_exact_known_unsent_binding_failure_is_eligible(retry, changes):
    service, _, store, _, _, envelope, _, before, _ = retry
    with store.transaction() as db:
        row = store.read_receipt(db, "text", "owner", "P")
        row.update(changes)
        store.write_receipt(db, "text", "owner", "P", row)
    with pytest.raises(ConversationBindingError) as error:
        service.submit("owner", envelope, source=SOURCE)
    assert error.value.code == "CHAT_UNSENT_RETRY_INVALID"
    with store.connect() as db:
        assert db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", ("owner", "P")).fetchone() != before
    assert read(service, "N") is None


def test_concurrent_new_ids_accept_one_and_restart_keeps_it(retry):
    service, _, _, clock, _, envelope, _, _, root = retry
    def submit(request_id):
        try:
            return service.submit("owner", {**envelope, "client_request_id": request_id}, source=SOURCE)["request_id"]
        except ConversationBindingError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=2) as workers:
        outcome = set(workers.map(submit, ("N", "N2")))
    assert outcome & {"N", "N2"}
    assert "CHAT_UNSENT_RETRY_CONFLICT" in outcome
    _, _, restarted_admission = build(root, clock)
    restarted = TextTaskService(service.store.path, runner=service.runner, admission=restarted_admission, clock=clock)
    restarted_admission.register("text", lambda ctx, body: restarted._run(ctx.owner, ctx.request_id, body))
    child_id = next(value for value in outcome if value in {"N", "N2"})
    replay = {**envelope, "client_request_id": child_id}
    assert restarted.submit("owner", replay, source=SOURCE)["request_id"] == child_id
    with restarted.store.connect() as db:
        assert db.execute("SELECT count(*) FROM requests WHERE owner=? AND json_extract(receipt,'$._retry_unsent_of')=?",
                          ("owner", "P")).fetchone()[0] == 1


def test_execution_rechecks_original_and_never_sends_after_drift(retry):
    service, admission, store, _, _, envelope, sent, _, _ = retry
    service.submit("owner", envelope, source=SOURCE)
    with store.transaction() as db:
        original = store.read_receipt(db, "text", "owner", "P")
        original.update(_submission_started=True, send_count=1)
        store.write_receipt(db, "text", "owner", "P", original)
    context = admission.claim_next()
    assert context and context.request_id == "N"
    admission.execute(context)
    assert sent == []
    assert read(service, "N")["error_code"] == "CHAT_UNSENT_RETRY_ORIGINAL_CHANGED"


def test_send_guard_rechecks_original_after_run_validation_before_post(retry):
    service, admission, store, _, _, envelope, sent, _, _ = retry
    service.submit("owner", envelope, source=SOURCE)

    def pause_between_run_and_send(body, on_cursor):
        with store.transaction() as db:
            original = store.read_receipt(db, "text", "owner", "P")
            original["_recovery_paused"] = True
            store.write_receipt(db, "text", "owner", "P", original)
        current_request.get().before_send()
        sent.append(body)
        return {"content": "must not post"}

    service.runner = pause_between_run_and_send
    context = admission.claim_next()
    assert context and context.request_id == "N"
    admission.execute(context)
    assert sent == []
    assert read(service, "N")["upstream_outcome"] == "not_sent"


def test_bound_unsent_cursor_requires_absent_original_and_single_nonuser_chain(retry):
    service, _, store, _, _, _, _, _, _ = retry
    with store.transaction() as db:
        original = store.read_receipt(db, "text", "owner", "P")
        original["request_message_id"] = "P-user"
        store.write_receipt(db, "text", "owner", "P", original)
    document = {
        "conversation_id": "conversation-0", "is_archived": False, "current_node": "B",
        "mapping": {
            "A": {"parent": "root", "children": ["B"], "message": {
                "id": "A", "author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True}},
            "B": {"parent": "A", "children": [], "message": {
                "id": "B", "author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True}},
            "root": {"parent": None, "children": ["A"], "message": {"id": "root", "author": {"role": "user"}}},
        },
    }
    proof = ConversationBindingService._check_bound_unsent_retry_cursor(document, original, "B")
    assert proof["retry_parent_message_id"] == "B"
    document["mapping"]["B"]["message"]["author"] = {"role": "user"}
    with pytest.raises(ConversationBindingError) as error:
        ConversationBindingService._check_bound_unsent_retry_cursor(document, original, "B")
    assert error.value.code == "CHAT_BOUND_RETRY_CURSOR_UNAVAILABLE"

    document["mapping"]["B"]["message"]["author"] = {"role": "assistant"}
    document["mapping"]["B"]["message"]["status"] = "finished_successfully"
    document["mapping"]["X"] = {"parent": "A", "children": [], "message": {
        "id": "X", "author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True}}
    with pytest.raises(ConversationBindingError) as error:
        ConversationBindingService._check_bound_unsent_retry_cursor(document, original, "B")
    assert error.value.code == "CHAT_BOUND_RETRY_CURSOR_UNAVAILABLE"

    document["mapping"].pop("X")
    document["mapping"]["A"]["children"] = ["tool"]
    document["mapping"]["tool"] = {"parent": "A", "children": ["B"], "message": {
        "id": "tool", "author": {"role": "tool"}, "status": "in_progress", "end_turn": False}}
    document["mapping"]["B"]["parent"] = "tool"
    with pytest.raises(ConversationBindingError) as error:
        ConversationBindingService._check_bound_unsent_retry_cursor(document, original, "B")
    assert error.value.code == "CHAT_BOUND_RETRY_CURSOR_UNAVAILABLE"


def test_bound_retry_uses_verified_fresh_cursor_and_reuses_attempt_after_restart(retry, monkeypatch):
    import services.conversation_binding_service as binding_module

    service, admission, store, clock, _, _, _, _, root = retry
    with store.transaction() as db:
        original = store.read_receipt(db, "text", "owner", "P")
        original["request_message_id"] = "P-user"
        store.write_receipt(db, "text", "owner", "P", original)
    document = {
        "conversation_id": "conversation-0", "is_archived": False, "current_node": "B",
        "mapping": {
            "root": {"parent": None, "children": ["A"], "message": {"id": "root", "author": {"role": "user"}}},
            "A": {"parent": "root", "children": ["B"], "message": {
                "id": "A", "author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True}},
            "B": {"parent": "A", "children": [], "message": {
                "id": "B", "author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True}},
        },
    }
    class Backend:
        def __init__(self, **_kwargs):
            pass
        def _get_conversation(self, _conversation_id):
            return deepcopy(document)
        def close(self):
            pass
    monkeypatch.setattr(binding_module, "OpenAIBackendAPI", Backend)
    monkeypatch.setattr(binding_module.account_service, "get_bound_account_identity", lambda _binding: "account-0")
    monkeypatch.setattr(binding_module.account_service, "get_bound_text_access_token", lambda *_args, **_kwargs: "fixture")
    monkeypatch.setattr(binding_module.account_service, "conversation_binding_lock", lambda *_args: nullcontext())

    first = service.retry_bound_attempt("owner", "P")
    assert first["existing"] is False and first["result_ready"] is False
    child_id = first["attempt"]["request_id"]
    child = read(service, child_id)
    assert child["_retry_unsent_of"] == "P" and child["parent_message_id"] == "B"

    with store.transaction() as db:
        original = store.read_receipt(db, "text", "owner", "P")
        original.update(_recovery_paused=True, status="unknown", upstream_outcome="unknown")
        store.write_receipt(db, "text", "owner", "P", original)
    restarted = TextTaskService(service.store.path, runner=service.runner, admission=admission, clock=clock)
    replay = restarted.retry_bound_attempt("owner", "P")
    assert replay["existing"] is True and replay["attempt"]["request_id"] == child_id
    with store.transaction() as db:
        child = store.read_receipt(db, "text", "owner", child_id)
        child.update(status="unknown", upstream_outcome="unknown", _submission_started=True, send_count=1)
        store.write_receipt(db, "text", "owner", child_id, child)
        original = store.read_receipt(db, "text", "owner", "P")
        original.update(status="succeeded", content="late original", upstream_outcome="completed")
        store.write_receipt(db, "text", "owner", "P", original)
    after_late_original = restarted.retry_bound_attempt("owner", "P")
    assert after_late_original["attempt"]["request_id"] == child_id
    assert after_late_original["result_ready"] is False
    with restarted.store.connect() as db:
        assert db.execute("SELECT count(*) FROM requests WHERE owner=? AND json_extract(receipt,'$._retry_unsent_of')=?",
                          ("owner", "P")).fetchone()[0] == 1


def test_retained_hash_and_fresh_parent_failure_never_send(retry):
    service, admission, store, _, _, envelope, sent, _, _ = retry
    with store.connect() as db:
        db.execute("UPDATE requests SET request_hash='corrupt' WHERE owner=? AND id=?", ("owner", "P"))
    with pytest.raises(ConversationBindingError) as error:
        service.submit("owner", envelope, source=SOURCE)
    assert error.value.code == "CHAT_UNSENT_RETRY_INVALID"
    with store.connect() as db:
        original = store.read_receipt(db, "text", "owner", "P")
        body = service.store.load_input(original["_input_ref"])
        db.execute("UPDATE requests SET request_hash=? WHERE owner=? AND id=?",
                   (service._submission_identity("owner", body)[1], "owner", "P"))
    service.runner = lambda body, on_cursor: (_ for _ in ()).throw(
        ConversationBindingError("B is no longer the terminal parent", code="CONVERSATION_BINDING_MISMATCH"))
    service.submit("owner", envelope, source=SOURCE)
    context = admission.claim_next()
    assert context and context.request_id == "N"
    admission.execute(context)
    assert sent == [] and read(service, "N")["upstream_outcome"] == "not_sent"


def test_http_schema_is_internal_reference_only(retry, monkeypatch):
    import api.ai as api
    service, _, _, _, _, envelope, _, _, _ = retry
    app = FastAPI(); app.include_router(api.create_router())
    monkeypatch.setattr(api, "text_task_service", service)
    monkeypatch.setattr(api, "require_identity", lambda token: {"id": "owner", "role": "admin" if token == "internal" else "user"})
    monkeypatch.setattr(api, "require_chat_text_policy", lambda _: None)
    monkeypatch.setattr(api, "trusted_source", lambda *_: SOURCE)
    review = AsyncMock(); monkeypatch.setattr(api, "filter_or_log", review)
    with pytest.raises(ValidationError):
        ConversationBindingTextRequest.model_validate({**envelope, "messages": []})
    with TestClient(app) as client:
        assert client.post("/api/conversation-bindings/text", json=envelope,
                           headers={"Authorization": "ordinary"}).status_code == 501
        response = client.post("/api/conversation-bindings/text", json=envelope,
                               headers={"Authorization": "internal"})
        assert response.status_code == 200, response.text
        assert "retained private input" not in response.text and "_input_ref" not in response.text
        for public in (response.json(),
                       client.get("/api/conversation-bindings/text-requests/N", headers={"Authorization": "internal"}).json(),
                       client.get("/api/conversation-bindings/text-requests/N/snapshot", headers={"Authorization": "internal"}).json()):
            assert public["retry_unsent_request_id"] == "P"
            assert not {"supersedes_request_id", "continue_after_no_final", "continue_after_incomplete", "derived_input"} & set(public)
        assert review.await_count == 1
