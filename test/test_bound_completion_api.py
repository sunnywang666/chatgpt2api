"""The internal retry command must not expand caller authority.

These route-level negative cases deliberately stop before the execution
services. Stateful attempt, cursor and send-boundary coverage lives with those
services' tests; no upstream account or request is used here.
"""
from contextlib import nullcontext
import json
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.generation_completion as api
from test.test_generation_completion import failed_unsent_image, setup
from test.test_no_final_successor import no_final, product_no_final
from test.test_explicit_unknown_successor import task


@pytest.fixture
def boundary(monkeypatch):
    service = Mock()
    service.store.connect.side_effect = lambda: nullcontext(None)
    service._root.return_value = {"model": "fixture-model"}
    identity = {"role": "admin", "id": "admin"}
    company = {"enabled": False}

    def authenticate(_authorization, *, request):
        if company["enabled"]:
            request.state.company_identity = {"id": "fixture-company"}
        return dict(identity)

    monkeypatch.setattr(api, "require_identity", authenticate)
    monkeypatch.setattr(api, "get_generation_completion_service", lambda: service)
    monkeypatch.setattr(api, "require_chat_text_policy", lambda *a, **kw: None)
    monkeypatch.setattr(api, "require_image_policy", lambda *a, **kw: None)
    app = FastAPI()
    app.include_router(api.create_router("text"))
    app.include_router(api.create_router("image"))
    return TestClient(app), identity, company, service


def assert_no_attempt(service):
    service.start.assert_not_called()
    service.text.retry_bound_attempt.assert_not_called()
    service.text.recover.assert_not_called()
    service.complete.assert_not_called()
    service.rework.assert_not_called()


def admin_bound_completion(h):
    """Move this isolated stored fixture to the trusted internal owner."""
    from services.generation_completion import GenerationCompletionService

    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        db.execute("UPDATE requests SET owner='admin' WHERE owner='owner' AND id='old'")
        h.service.store.write_receipt(db, "text", "admin", "old", original)
    return GenerationCompletionService(h.service, Mock(), Mock())


def admin_attempt_client(service, monkeypatch, *, company=False):
    identity = {"id": "admin", "role": "admin"}
    def authenticate(_authorization, *, request):
        if company:
            request.state.company_identity = {"id": "company"}
        return dict(identity)
    monkeypatch.setattr(api, "get_generation_completion_service", lambda: service)
    monkeypatch.setattr(api, "require_identity", authenticate)
    monkeypatch.setattr(api, "require_chat_text_policy", lambda *_a, **_kw: None)
    app = FastAPI()
    app.include_router(api.create_router("text"))
    return TestClient(app), identity


@pytest.mark.parametrize("path", ["chat-requests", "image-tasks"])
@pytest.mark.parametrize("company_client", [False, True])
def test_retry_is_not_an_ordinary_or_company_client_permission(boundary, path, company_client):
    client, identity, company, service = boundary
    identity.update(role="user", id="fixture-client")
    company["enabled"] = company_client
    response = client.post(f"/api/{path}/original/completion", json={"action": "retry"})
    assert response.status_code == 403
    assert_no_attempt(service)


@pytest.mark.parametrize("path", ["chat-requests", "image-tasks"])
@pytest.mark.parametrize("extra", [
    {"messages": [{"role": "user", "content": "replacement prompt"}]},
    {"model": "replacement-model"},
    {"provider_binding_id": "replacement-binding"},
    {"provider_account_identity": "replacement-account"},
    {"client_conversation_id": "replacement-client-conversation"},
    {"conversation_id": "replacement-conversation"},
    {"parent_message_id": "replacement-parent"},
    {"client_request_id": "caller-selected-child"},
    {"allow_unconfirmed_retry": True},
    {"allow_unconfirmed_retry": False},
    {"retry_not_sent_failure_at": 123.0},
    {"selected_id": "caller-selected-result"},
    {"results_saved": False},
    {"reviewed": False},
])
def test_internal_retry_accepts_no_caller_input_cursor_or_retry_proof(boundary, path, extra):
    client, _identity, _company, service = boundary
    response = client.post(f"/api/{path}/original/completion", json={"action": "retry", **extra})
    assert response.status_code in {403, 422}
    assert_no_attempt(service)


@pytest.mark.parametrize("path", ["chat-requests", "image-tasks"])
def test_admin_recover_cannot_opt_into_unknown_resend(boundary, path):
    client, _identity, _company, service = boundary
    response = client.post(f"/api/{path}/original/completion", json={
        "action": "recover", "allow_unconfirmed_retry": True,
    })
    assert response.status_code == 403
    assert_no_attempt(service)


def test_admin_explicit_recover_can_read_late_original_after_automatic_reads_end(setup, monkeypatch):
    from services.conversation_binding_service import ConversationBindingService
    from test.test_stalled_text_diagnostics import completed_tool_chain
    from test.test_unknown_turn_recovery import document

    service, _admission, sends = setup
    admin = {"id": "admin", "role": "admin"}
    reason = "COMPLETION_ORIGINAL_READ_UNAVAILABLE"
    with service.store.transaction() as db:
        original = service.store.read_receipt(db, "text", "owner", "old-0")
        original.pop("route", None)
        original.pop("_public_session_ref", None)
        original.pop("_work_key", None)
        original.update(_attempt_finished_at=1, _attempt_reason=reason, _completion={
            "state": "needs_attention", "reason": reason, "next_at": None,
            "allow_unconfirmed_retry": False, "max_extra_requests": 0,
        })
        db.execute("UPDATE requests SET owner='admin' WHERE owner='owner' AND id='old-0'")
        service.store.write_receipt(db, "text", "admin", "old-0", original)
    reads = []

    def original_reader(receipt):
        assert receipt["request_id"] == "old-0"
        reads.append(receipt["request_id"])
        saved_document = document(receipt)
        completed_tool_chain(saved_document)
        return ConversationBindingService._read_text_request_result(None, receipt, document=saved_document)

    service.text.recovery_reader = original_reader
    assert service.text.recover("admin", "old-0")["status"] == "unknown"
    assert reads == []
    monkeypatch.setattr(api, "get_generation_completion_service", lambda: service)
    monkeypatch.setattr(api, "require_identity", lambda *_a, **_kw: admin)
    monkeypatch.setattr(api, "require_chat_text_policy", lambda *_a, **_kw: None)
    app = FastAPI()
    app.include_router(api.create_router("text"))
    response = TestClient(app).post("/api/chat-requests/old-0/completion", json={})
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "succeeded", (response.text, reads)
    assert response.json()["content"] == "original calculated stock review"
    assert reads == ["old-0"] and sends == []
    with service.store.connect() as db:
        assert service.store.read_receipt(db, "text", "admin", "old-0")["content"] == "original calculated stock review"
        assert db.execute("SELECT count(*) FROM requests WHERE owner='admin'").fetchone()[0] == 1


def test_admin_retry_get_recovers_persisted_text_attempt_without_writes_or_upstream(task, monkeypatch):
    h = no_final(task)
    completion = admin_bound_completion(h)
    client, _identity = admin_attempt_client(completion, monkeypatch)
    posted = client.post("/api/chat-requests/old/completion", json={"action": "retry"})
    assert posted.status_code == 200, posted.text
    attempt_id = posted.json()["attempt_id"]
    with h.service.store.connect() as db:
        before = list(db.execute("SELECT owner,id,request_hash,receipt FROM requests ORDER BY id"))
    h.service.retry_bound_attempt = Mock(side_effect=AssertionError("GET must not retry"))
    h.service.read = Mock(side_effect=AssertionError("GET must not recover/read upstream"))
    first = client.get("/api/chat-requests/old/completion?action=retry")
    assert first.status_code == 200, first.text
    assert first.headers["cache-control"] == "private, no-store"
    assert first.json() == {"protocol": "task-attempt-v1", "kind": "text", "original_id": "old",
                            "attempt_id": attempt_id, "status": "queued", "result_ready": False,
                            "existing": True, "created": False}
    with h.service.store.connect() as db:
        assert list(db.execute("SELECT owner,id,request_hash,receipt FROM requests ORDER BY id")) == before

    from services.text_task_service import TextTaskService
    from services.generation_completion import GenerationCompletionService
    restarted_text = TextTaskService(h.service.store.path, runner=h.service.runner, clock=h.clock,
                                     admission=h.admission)
    restarted = GenerationCompletionService(restarted_text, Mock(), Mock())
    restarted_client, _ = admin_attempt_client(restarted, monkeypatch)
    second = restarted_client.get("/api/chat-requests/old/completion?action=retry")
    assert second.status_code == 200 and second.json() == first.json()
    assert h.sent == []


def test_admin_retry_get_has_no_attempt_for_new_unsuccessful_original(task, monkeypatch):
    h = no_final(task)
    completion = admin_bound_completion(h)
    client, _ = admin_attempt_client(completion, monkeypatch)
    response = client.get("/api/chat-requests/old/completion?action=retry")
    assert response.status_code == 404 and response.json()["detail"]["code"] == "TASK_ATTEMPT_NOT_FOUND"
    assert h.sent == [] and h.admission.claim_next() is None


def test_retry_get_after_late_original_uses_retained_submission_cursor(task, monkeypatch):
    h = no_final(task)
    completion = admin_bound_completion(h)
    client, _ = admin_attempt_client(completion, monkeypatch)
    posted = client.post("/api/chat-requests/old/completion", json={"action": "retry"})
    assert posted.status_code == 200
    child_id = posted.json()["attempt_id"]
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "admin", "old")
        original.pop("_submission_parent_message_id", None)
        original.update(status="succeeded", content="late answer", upstream_outcome="completed",
                        parent_message_id="late-final-node")
        h.service.store.write_receipt(db, "text", "admin", "old", original)
    selected = client.post("/api/chat-requests/old/completion", json={"action": "retry"})
    assert selected.status_code == 200 and selected.json()["attempt_id"] == "old"
    with h.service.store.connect() as db:
        child = h.service.store.read_receipt(db, "text", "admin", child_id)
        assert child["error_code"] == "COMPLETION_ORIGINAL_RECOVERED"
        before = list(db.execute("SELECT owner,id,request_hash,receipt FROM requests ORDER BY id"))
    found = client.get("/api/chat-requests/old/completion?action=retry")
    assert found.status_code == 200, found.text
    assert found.json()["attempt_id"] == "old" and found.json()["result_ready"] is True
    with h.service.store.connect() as db:
        assert list(db.execute("SELECT owner,id,request_hash,receipt FROM requests ORDER BY id")) == before
    assert h.sent == []


def test_retry_attempt_get_remains_admin_only_for_company_and_user(boundary, monkeypatch):
    client, identity, company, service = boundary
    identity.update(role="user", id="client")
    response = client.get("/api/chat-requests/original/completion?action=retry")
    assert response.status_code == 403
    identity.update(role="admin", id="admin")
    company["enabled"] = True
    response = client.get("/api/chat-requests/original/completion?action=retry")
    assert response.status_code == 403
    service.read_bound_attempt.assert_not_called()
    company["enabled"] = False
    monkeypatch.setattr(api, "is_external", lambda *_: True)
    assert client.get("/api/chat-requests/original/completion?action=retry").status_code == 403
    service.read_bound_attempt.assert_not_called()


def test_retry_attempt_get_rejects_foreign_owner_and_private_route(task, monkeypatch):
    h = no_final(task)
    completion = admin_bound_completion(h)
    client, identity = admin_attempt_client(completion, monkeypatch)
    identity["id"] = "other-admin"
    foreign = client.get("/api/chat-requests/old/completion?action=retry")
    assert foreign.status_code == 404
    identity["id"] = "admin"
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "admin", "old")
        original["_forward_protocol"] = "codex"
        h.service.store.write_receipt(db, "text", "admin", "old", original)
    private = client.get("/api/chat-requests/old/completion?action=retry")
    assert private.status_code == 409 and private.json()["detail"]["code"] == "COMPLETION_PURE_GENERATION_REQUIRED"


def test_bound_attempt_reader_rejects_multiple_children_and_keeps_sent_child(task):
    from services.generation_completion import CompletionError

    h = no_final(task)
    completion = admin_bound_completion(h)
    child_id = h.service.retry_bound_attempt("admin", "old")["attempt"]["request_id"]
    with h.service.store.transaction() as db:
        child = h.service.store.read_receipt(db, "text", "admin", child_id)
        child.update(request_id="second-bound-child")
        db.execute("INSERT INTO requests VALUES(?,?,?,?)", (
            "admin", "second-bound-child", "second-bound-hash", json.dumps(child),
        ))
    with pytest.raises(CompletionError) as exc:
        completion.read_bound_attempt("text", {"id": "admin"}, "old")
    assert exc.value.code == "TASK_ATTEMPT_CONFLICT" and exc.value.status == 409

    with h.service.store.transaction() as db:
        db.execute("DELETE FROM requests WHERE owner='admin' AND id='second-bound-child'")
        original = h.service.store.read_receipt(db, "text", "admin", "old")
        original.update(status="succeeded", content="late original", upstream_outcome="completed")
        child = h.service.store.read_receipt(db, "text", "admin", child_id)
        child.update(status="unknown", upstream_outcome="unknown", _submission_started=True)
        h.service.store.write_receipt(db, "text", "admin", "old", original)
        h.service.store.write_receipt(db, "text", "admin", child_id, child)
    assert completion.read_bound_attempt("text", {"id": "admin"}, "old")["request_id"] == child_id
    with h.service.store.transaction() as db:
        child = h.service.store.read_receipt(db, "text", "admin", child_id)
        child.update(status="failed", error_code="COMPLETION_ORIGINAL_RECOVERED",
                     upstream_outcome="not_sent", _submission_started=False)
        h.service.store.write_receipt(db, "text", "admin", child_id, child)
    assert completion.read_bound_attempt("text", {"id": "admin"}, "old")["request_id"] == "old"


def test_image_retry_get_returns_only_same_id_bound_retry_and_never_replacement(failed_unsent_image, monkeypatch):
    service = failed_unsent_image
    admin = {"id": "admin", "role": "admin"}
    with service.store.transaction() as db:
        receipt = service.store.read_receipt(db, "image", "owner", "repair-image")
        source = service.store.load_input(receipt["_input_ref"])
        source["identity"] = admin
        receipt.update(owner_id="admin", _input_ref=service.store.save_input(source))
        service.store.write_receipt(db, "image", "admin", "repair-image", receipt)
        db.execute("DELETE FROM image_requests WHERE task_key=?", ("owner:repair-image",))
    monkeypatch.setattr(api, "get_generation_completion_service", lambda: service)
    monkeypatch.setattr(api, "require_identity", lambda *_a, **_kw: admin)
    monkeypatch.setattr(api, "require_image_policy", lambda *_a, **_kw: None)
    app = FastAPI(); app.include_router(api.create_router("image")); client = TestClient(app)
    assert client.post("/api/image-tasks/repair-image/completion", json={"action": "retry"}).status_code == 200
    with service.store.connect() as db:
        persisted = service.store.read_receipt(db, "image", "admin", "repair-image")
    assert persisted.get("_completion", {}).get("bound_retry") is True, persisted
    before_wake = Mock()
    service.text.admission.wake = before_wake
    found = client.get("/api/image-tasks/repair-image/completion?action=retry")
    assert found.status_code == 200, found.text
    assert found.json()["attempt_id"] == "repair-image"
    assert found.json()["existing"] is True and found.json()["created"] is False
    before_wake.assert_not_called()
    with service.store.transaction() as db:
        receipt = service.store.read_receipt(db, "image", "admin", "repair-image")
        receipt["_completion"]["replacement_id"] = "existing-other-image"
        receipt.update(status="success", data=[{"url": "https://example.test/original-image"}])
        service.store.write_receipt(db, "image", "admin", "repair-image", receipt)
    conflict = client.get("/api/image-tasks/repair-image/completion?action=retry")
    assert conflict.status_code == 409 and conflict.json()["detail"]["code"] == "TASK_ATTEMPT_CONFLICT"


@pytest.mark.parametrize("claimed", [False, True])
def test_late_original_cancels_only_a_proven_unsent_child(product_no_final, claimed):
    from services.request_context import AdmissionLost

    h = product_no_final
    child_id = h.service.retry_bound_attempt("owner", "old")["attempt"]["request_id"]
    context = h.admission.claim_next() if claimed else None
    if claimed:
        assert context and context.request_id == child_id
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        original.update(status="succeeded", content="late original answer", upstream_outcome="completed",
                        upstream_unfinished=False, _upstream_terminal=True)
        h.service.store.write_receipt(db, "text", "owner", "old", original)
    selected = h.service.retry_bound_attempt("owner", "old")
    assert selected["result_ready"] and selected["existing"]
    assert selected["attempt"]["request_id"] == "old"
    with h.service.store.connect() as db:
        child = h.service.store.read_receipt(db, "text", "owner", child_id)
        assert child["error_code"] == "COMPLETION_ORIGINAL_RECOVERED"
        assert child["upstream_outcome"] == "not_sent"
        assert child["_claim_id"] is None and child["_executing"] is False
    if context:
        with pytest.raises(AdmissionLost):
            context.before_send()
    repeat = h.service.retry_bound_attempt("owner", "old")
    assert repeat["attempt"]["request_id"] == "old" and repeat["result_ready"]
    assert h.admission.claim_next() is None and h.sent == []
