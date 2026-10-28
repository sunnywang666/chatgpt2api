"""The internal retry command must not expand caller authority.

These route-level negative cases deliberately stop before the execution
services. Stateful attempt, cursor and send-boundary coverage lives with those
services' tests; no upstream account or request is used here.
"""
from contextlib import nullcontext
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.generation_completion as api
from test.test_generation_completion import setup
from test.test_no_final_successor import product_no_final


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
