"""One explicit retry for a terminal empty-wire Codex fallback rejection."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.text_task_service import TextTaskService
from services.product_work import ProductWorkService
from test.test_pool_admission import Clock, build


IDENTITY = {"id": "owner", "role": "admin", "_trusted_source": "internal:listing"}


@pytest.fixture
def retry_runtime(tmp_path, monkeypatch):
    account = {
        "access_token": "fixture-only-token", "account_id": "physical-account",
        "provider_account_identity": "account-0", "source_type": "web", "type": "Plus",
        "status": "正常", "quota": 99, "conversation_binding_ids": ["binding-0"],
        "limits_progress": [{"feature_name": "image_gen", "remaining": 99}],
        "capacity_observed_at": datetime.now(timezone.utc).isoformat(),
    }
    (tmp_path / "accounts.json").write_text(json.dumps([account]))
    clock = Clock()
    accounts, store, admission = build(tmp_path, clock)
    service = TextTaskService(store.path, admission=admission, clock=clock)
    admission.register("text", lambda ctx, body: service._run(ctx.owner, ctx.request_id, body))
    return SimpleNamespace(accounts=accounts, store=store, admission=admission, service=service,
                           clock=clock, account=account)


def _read(runtime, request_id):
    with runtime.store.connect() as db:
        return runtime.store.read_receipt(db, "text", "owner", request_id)


def _update(runtime, request_id, **changes):
    with runtime.store.transaction() as db:
        receipt = runtime.store.read_receipt(db, "text", "owner", request_id)
        receipt.update(changes)
        runtime.store.write_receipt(db, "text", "owner", request_id, receipt)


def _prove_model_rejection(runtime):
    _update(runtime, "original", status="failed", error_code="MODEL_NOT_FOUND", upstream_outcome="rejected",
            _submission_started=True, _executing=False, _turn_reserved=False,
            original_failure_phase="stream_open", original_exception_category="http",
            original_upstream_request_stage="conversation", original_http_status=404,
            original_upstream_error_form="detail_text", original_upstream_error={"message": "Model not found"})


def _prepare_rejected_fallback(runtime):
    original = {
        "client_request_id": "original", "client_conversation_id": "source-client", "model": "source-model",
        "messages": [{"role": "user", "content": "synthetic retained original"}],
        "product_work": {"product_ref": "synthetic-product", "step_ref": "copy"},
    }
    runtime.service.submit("owner", original, source="internal:listing")
    _prove_model_rejection(runtime)
    account_ref = runtime.accounts.pool_account_ref(runtime.account)
    runtime.service.submit_codex_fallback("owner", "original", "rejected-native", "gpt-5.6-codex", account_ref,
                                          source="internal:listing")
    output = runtime.store.create_output()
    _update(runtime, "rejected-native", status="failed", error_code="codex_request_rejected",
            upstream_outcome="rejected", _submission_started=True, _upstream_terminal=True,
            _turn_reserved=False, _executing=False, _wire_output=output, _wire_size=0,
            _wire_error_status=400,
            _wire_error_payload={"error": {"code": "codex_request_rejected", "type": "invalid_request_error"}})


def _install_native_success(runtime, monkeypatch):
    from services.codex_service import CodexService
    from test.test_codex_service import FakeResponse, FakeSession, SessionFactory, observation

    runtime.accounts.update_account(runtime.account["access_token"], {"codex_observation": observation()})
    response = {"id": "resp-retry", "status": "completed", "output": [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "saved retry result"}]},
    ]}
    session = FakeSession(post_response=FakeResponse(payload=response))
    native = CodexService(runtime.accounts, SessionFactory([session]))
    runtime.admission.codex = native
    monkeypatch.setattr("services.codex_service.codex_service", native)
    return session


def test_admin_rejected_retry_registers_one_frozen_child_and_runner_sends_once(retry_runtime, monkeypatch):
    from api import ai

    _prepare_rejected_fallback(retry_runtime)
    session = _install_native_success(retry_runtime, monkeypatch)
    monkeypatch.setattr(ai, "text_task_service", retry_runtime.service)
    identity = dict(IDENTITY)
    monkeypatch.setattr(ai, "require_identity", lambda _token: identity)
    app = FastAPI()
    app.include_router(ai.create_router())

    with TestClient(app) as client:
        identity["role"] = "user"
        assert client.post("/api/conversation-bindings/text-requests/rejected-native/codex-rejected-retry",
                           headers={"Authorization": "ordinary"}, json={"client_request_id": "retry-native"}).status_code == 404
        identity["role"] = "admin"
        malformed = client.post("/api/conversation-bindings/text-requests/rejected-native/codex-rejected-retry",
                                headers={"Authorization": "admin"}, json={"client_request_id": "retry-native", "model": "other"})
        assert malformed.status_code == 422
        first = client.post("/api/conversation-bindings/text-requests/rejected-native/codex-rejected-retry",
                            headers={"Authorization": "admin", "x-workbench-consumer": "listing"}, json={"client_request_id": "retry-native"})
        repeated = client.post("/api/conversation-bindings/text-requests/rejected-native/codex-rejected-retry",
                               headers={"Authorization": "admin", "x-workbench-consumer": "listing"}, json={"client_request_id": "retry-native"})
        conflict = client.post("/api/conversation-bindings/text-requests/rejected-native/codex-rejected-retry",
                               headers={"Authorization": "admin", "x-workbench-consumer": "listing"}, json={"client_request_id": "other-retry"})
    assert first.status_code == repeated.status_code == 200
    assert conflict.status_code == 409
    before = _read(retry_runtime, "rejected-native")
    child = _read(retry_runtime, "retry-native")
    saved_parent = retry_runtime.store.load_input(before["_input_ref"])
    saved_child = retry_runtime.store.load_input(child["_input_ref"])
    assert child["_codex_rejected_retry_of"] == "rejected-native"
    assert child["_codex_fallback_input_hash"] != before["_codex_fallback_input_hash"]
    assert saved_child["model"] == saved_parent["model"]
    assert saved_child["_requested_account_ref"] == saved_parent["_requested_account_ref"]
    assert saved_child["_forward"]["payload"] == saved_parent["_forward"]["payload"]
    assert saved_child["_forward"]["headers"]["session-id"] == saved_parent["_forward"]["headers"]["session-id"]
    assert saved_child["_forward"]["headers"]["x-client-request-id"] == "retry-native"
    assert child["client_conversation_id"] == before["client_conversation_id"]
    assert before["_wire_error_status"] == 400 and before["_wire_size"] == 0

    claimed = retry_runtime.admission.claim_next()
    assert claimed and claimed.request_id == "retry-native"
    retry_runtime.admission.execute(claimed)
    assert len(session.calls) == 1
    assert retry_runtime.service.read("owner", "retry-native")["content"] == "saved retry result"
    assert retry_runtime.admission.claim_next() is None

    work = ProductWorkService(retry_runtime.service, SimpleNamespace(), clock=retry_runtime.clock)
    closed = work.close(IDENTITY, "synthetic-product", results_saved=True, outcome="completed")
    assert closed["state"] == "completed" and closed["archive"] == {"total": 0, "confirmed": 0, "pending": 0}


@pytest.mark.parametrize("changes", [
    {"status": "unknown", "upstream_outcome": "unknown"},
    {"_wire_size": 1},
    {"_claim_id": "live", "_claim_until": 2000.0, "_executing": True},
    {"_recovery_paused": True},
    {"content": "late result"},
    {"_wire_error_status": 408},
])
def test_rejected_retry_refuses_non_terminal_or_changed_parent(retry_runtime, changes):
    from services.conversation_binding_service import ConversationBindingError

    _prepare_rejected_fallback(retry_runtime)
    if changes.get("_wire_size"):
        parent = _read(retry_runtime, "rejected-native")
        with retry_runtime.store.output_file(parent["_wire_output"], append=True) as handle:
            handle.write(b"x")
    _update(retry_runtime, "rejected-native", **changes)
    with pytest.raises(ConversationBindingError) as error:
        retry_runtime.service.submit_codex_rejected_retry("owner", "rejected-native", "retry-native",
                                                          source="internal:listing")
    assert error.value.code == "CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE"
    assert retry_runtime.admission.claim_next() is None


def test_rejected_retry_cannot_retry_its_own_failed_child(retry_runtime):
    from services.conversation_binding_service import ConversationBindingError

    _prepare_rejected_fallback(retry_runtime)
    retry_runtime.service.submit_codex_rejected_retry("owner", "rejected-native", "retry-native",
                                                      source="internal:listing")
    _update(retry_runtime, "retry-native", status="failed", error_code="codex_request_rejected",
            upstream_outcome="rejected", _submission_started=True, _upstream_terminal=True,
            _turn_reserved=False, _executing=False)
    with pytest.raises(ConversationBindingError) as error:
        retry_runtime.service.submit_codex_rejected_retry("owner", "retry-native", "retry-grandchild",
                                                          source="internal:listing")
    assert error.value.code == "CODEX_REJECTED_RETRY_PARENT_UNAVAILABLE"
