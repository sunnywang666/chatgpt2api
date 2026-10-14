"""HTTP boundary for the explicit, once-only same-ID recovery operation."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import api.ai as api
from services.conversation_binding_service import ConversationBindingError
from test.test_explicit_unknown_successor import task, row


@pytest.fixture
def missing_resume_api(monkeypatch):
    envelope = {
        "client_request_id": "existing-successor",
        "supersedes_request_id": "original",
        "provider_binding_id": "original-binding",
        "provider_account_identity": "original-account",
        "client_conversation_id": "original-client-session",
        "conversation_id": "original-chat",
        "parent_message_id": "actual-submitted-parent",
    }
    receipt = {"request_id": "existing-successor", "status": "failed",
               "upstream_outcome": "unknown"}
    resume = Mock(return_value=receipt)
    monkeypatch.setattr(api, "text_task_service", SimpleNamespace(resume_missing_successor=resume))
    monkeypatch.setattr(api, "require_identity", lambda token: {
        "id": "original-owner", "role": "admin" if token == "internal" else "user"})
    policy = Mock()
    monkeypatch.setattr(api, "require_chat_text_policy", policy)
    app = FastAPI()
    app.include_router(api.create_router())
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, envelope=envelope, receipt=receipt, resume=resume, policy=policy,
            headers={"Authorization": "internal"},
            url="/api/conversation-bindings/text-requests/existing-successor/resume-missing-successor")


def test_internal_owner_and_exact_envelope_reach_only_explicit_operation(missing_resume_api):
    h = missing_resume_api
    response = h.client.post(h.url, json=h.envelope, headers=h.headers)
    assert response.status_code == 200 and response.json() == h.receipt
    assert response.headers["Cache-Control"] == "private, no-store"
    h.resume.assert_called_once_with("original-owner", "existing-successor", h.envelope)
    h.policy.assert_called_once()


def test_ordinary_client_cannot_invoke_new_send_operation(missing_resume_api):
    h = missing_resume_api
    response = h.client.post(h.url, json=h.envelope)
    assert response.status_code == 501
    assert response.json()["detail"]["code"] == "SERVICE_OPERATION_UNAVAILABLE"
    h.resume.assert_not_called()


@pytest.mark.parametrize("extra", [
    {"messages": []}, {"model": "different"}, {"image_model": "different"},
    {"thinking_effort": "high"}, {"derived_input": {"kind": "attributes_required_only_v4"}},
    {"continue_after_no_final": True}, {"product_work": {}}, {"owner": "other"},
])
def test_input_and_mode_overrides_rejected_before_service(missing_resume_api, extra):
    h = missing_resume_api
    response = h.client.post(h.url, json={**h.envelope, **extra}, headers=h.headers)
    assert response.status_code == 422
    h.resume.assert_not_called()


@pytest.mark.parametrize("field", [
    "client_request_id", "supersedes_request_id", "provider_binding_id", "provider_account_identity",
    "client_conversation_id", "conversation_id", "parent_message_id",
])
def test_all_original_identity_fields_are_required(missing_resume_api, field):
    h = missing_resume_api
    body = {key: value for key, value in h.envelope.items() if key != field}
    assert h.client.post(h.url, json=body, headers=h.headers).status_code == 422
    h.resume.assert_not_called()


def test_path_cannot_target_a_different_request(missing_resume_api):
    h = missing_resume_api
    response = h.client.post(h.url.replace("existing-successor", "different"), json=h.envelope, headers=h.headers)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "CHAT_MISSING_SUCCESSOR_NOT_RESUMABLE"
    h.resume.assert_not_called()


def test_existing_chat_policy_still_applies(missing_resume_api):
    h = missing_resume_api
    h.policy.side_effect = HTTPException(403, detail={"code": "CHAT_POLICY_DENIED"})
    assert h.client.post(h.url, json=h.envelope, headers=h.headers).status_code == 403
    h.resume.assert_not_called()


def test_service_rejection_exposes_only_machine_reason(missing_resume_api):
    h = missing_resume_api
    h.resume.side_effect = ConversationBindingError("private upstream text", code="CHAT_MISSING_SUCCESSOR_BUSY")
    response = h.client.post(h.url, json=h.envelope, headers=h.headers)
    assert response.status_code == 409
    assert response.json() == {"detail": {"code": "CHAT_MISSING_SUCCESSOR_BUSY"}}
    assert "private upstream" not in response.text


def test_http_resume_uses_real_receipt_and_admission_once(task, monkeypatch):
    from services.request_context import current_request
    from test.test_missing_successor_resume import stopped_successor

    h = stopped_successor(task)
    original = row(h.service, "successor")
    sent = []

    def runner(body, on_cursor):
        current_request.get().before_send()
        sent.append(body["_request_message_id"])
        raise RuntimeError("response lost after dispatch")

    h.service.runner = runner
    monkeypatch.setattr(api, "text_task_service", h.service)
    monkeypatch.setattr(api, "require_identity", lambda token: {"id": "owner", "role": "admin"})
    monkeypatch.setattr(api, "require_chat_text_policy", lambda _: None)
    app = FastAPI()
    app.include_router(api.create_router())
    with TestClient(app) as client:
        url = "/api/conversation-bindings/text-requests/successor/resume-missing-successor"
        accepted = client.post(url, json=h.envelope)
        assert accepted.status_code == 200
        assert accepted.headers["cache-control"] == "private, no-store"
        assert accepted.json()["missing_successor_resume"]["state"] == "reserved"
        assert client.post(url, json=h.envelope).json() == accepted.json()
        h.admission.execute(h.admission.claim_next())
        uncertain = client.post(url, json=h.envelope)
        assert uncertain.status_code == 200
        assert uncertain.json()["missing_successor_resume"]["state"] == "submitted"
        assert uncertain.json()["upstream_outcome"] == "unknown"

    saved = row(h.service, "successor")
    assert sent == [saved["_missing_successor_resume"]["request_message_id"]]
    assert saved["request_message_id"] == original["request_message_id"]
    assert saved["_input_ref"] == original["_input_ref"]
    assert h.admission.claim_next() is None
