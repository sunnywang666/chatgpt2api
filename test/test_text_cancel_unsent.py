"""Bounded ordinary Chat withdrawal before a PoolAdmission claim or send."""
from types import SimpleNamespace
from unittest.mock import Mock
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import ai
from services.text_task_service import TextTaskService
from services.work_lifecycle import WorkLifecycleError
from test.test_pool_admission import Clock, build


IDENTITY = {"id": "owner", "role": "admin", "_trusted_source": "internal:content"}
BINDING = {
    "provider_binding_id": "binding", "provider_account_identity": "account",
    "client_conversation_id": "client", "conversation_id": "conversation",
    "parent_message_id": "parent",
}


@pytest.fixture
def service(tmp_path):
    return TextTaskService(tmp_path / "tasks.sqlite3", admission=SimpleNamespace(wake=Mock()),
                           runner=Mock(), clock=lambda: 1000.0)


def body(request_id="withdraw-me"):
    return {"client_request_id": request_id, "messages": [{"role": "user", "content": "original"}], **BINDING}


def pinned(service, request_id="withdraw-me"):
    snapshot = service.read_snapshot("owner", request_id, include_cancel_unsent_context=True,
                                     cancel_unsent_source="internal:content")
    return snapshot["cancel_unsent_context"]


def test_context_is_source_scoped_and_apply_is_durable_idempotent(service):
    service.submit("owner", body(), source="internal:content")
    assert "cancel_unsent_context" not in service.read_snapshot("owner", "withdraw-me")
    assert "cancel_unsent_context" not in service.read_snapshot(
        "owner", "withdraw-me", include_cancel_unsent_context=True, cancel_unsent_source="internal:listing")
    context = pinned(service)
    dry = service.cancel_unsent(IDENTITY, "withdraw-me", context["request_hash"],
                                context["expected_binding"], context["reason"], "dry-run")
    assert dry["status"] == "queued" and dry["cancel_unsent"]["mode"] == "dry-run"
    applied = service.cancel_unsent(IDENTITY, "withdraw-me", context["request_hash"],
                                    context["expected_binding"], context["reason"], "apply")
    assert applied["status"] == "cancelled"
    assert service.cancel_unsent(IDENTITY, "withdraw-me", context["request_hash"],
                                 context["expected_binding"], context["reason"], "apply") == applied
    assert service.read("owner", "withdraw-me")["cancelled_unsent"] == applied["cancelled_unsent"]
    assert service.submit("owner", body(), source="internal:content")["status"] == "cancelled"
    restarted = TextTaskService(service.store.path, admission=SimpleNamespace(wake=Mock()),
                                runner=Mock(), clock=lambda: 1000.0)
    assert restarted.read_snapshot("owner", "withdraw-me")["cancelled_unsent"] == applied["cancelled_unsent"]


def test_cancel_http_requires_trusted_source_and_uses_the_server_owner(service, monkeypatch):
    service.submit("owner", body(), source="internal:content")
    identity = [IDENTITY]
    monkeypatch.setattr(ai, "text_task_service", service)
    monkeypatch.setattr(ai, "require_identity", lambda *args, **kwargs: identity[0])
    app = FastAPI(); app.include_router(ai.create_router())
    client = TestClient(app)
    header = {"x-workbench-consumer": "content"}
    assert "cancel_unsent_context" not in client.get(
        "/api/conversation-bindings/text-requests/withdraw-me/snapshot", headers=header).json()
    assert client.get("/api/conversation-bindings/text-requests/withdraw-me/snapshot",
                      params={"include_cancel_unsent_context": "true"}).status_code == 403
    context_response = client.get("/api/conversation-bindings/text-requests/withdraw-me/snapshot",
                                  params={"include_cancel_unsent_context": "true"}, headers=header)
    assert context_response.status_code == 200
    context = context_response.json()["cancel_unsent_context"]
    payload = {"mode": "dry-run", "expected_request_hash": context["request_hash"],
               "expected_binding": context["expected_binding"], "reason": context["reason"]}
    dry = client.post("/api/conversation-bindings/text-requests/withdraw-me/cancel-unsent",
                      headers=header, json=payload)
    assert dry.status_code == 200 and dry.json()["cancel_unsent"]["mode"] == "dry-run"
    identity[0] = {**IDENTITY, "id": "other"}
    assert client.post("/api/conversation-bindings/text-requests/withdraw-me/cancel-unsent",
                       headers=header, json=payload).status_code == 404
    identity[0] = IDENTITY
    applied = client.post("/api/conversation-bindings/text-requests/withdraw-me/cancel-unsent",
                          headers=header, json={**payload, "mode": "apply"})
    assert applied.status_code == 200 and applied.json()["status"] == "cancelled"
    assert client.post("/api/conversation-bindings/text-requests/withdraw-me/cancel-unsent",
                       headers=header, json={**payload, "mode": "apply"}).json() == applied.json()


def test_cancel_refuses_source_hash_binding_and_proof_drift(service):
    service.submit("owner", body(), source="internal:content")
    context = pinned(service)
    with pytest.raises(WorkLifecycleError, match="SOURCE_CONFLICT"):
        service.cancel_unsent({**IDENTITY, "_trusted_source": "internal:listing"}, "withdraw-me",
                              context["request_hash"], context["expected_binding"], context["reason"], "apply")
    with pytest.raises(WorkLifecycleError, match="IDENTITY_CONFLICT"):
        service.cancel_unsent(IDENTITY, "withdraw-me", "0" * 64,
                              context["expected_binding"], context["reason"], "apply")
    changed = {**context["expected_binding"], "conversation_id": "other"}
    with pytest.raises(WorkLifecycleError, match="IDENTITY_CONFLICT"):
        service.cancel_unsent(IDENTITY, "withdraw-me", context["request_hash"], changed, context["reason"], "apply")
    applied = service.cancel_unsent(IDENTITY, "withdraw-me", context["request_hash"],
                                    context["expected_binding"], context["reason"], "apply")
    with service.store.transaction() as db:
        receipt = service.store.read_receipt(db, "text", "owner", "withdraw-me")
        receipt["_ordinary_unsent_cancelled"]["request_hash"] = "f" * 64
        service.store.write_receipt(db, "text", "owner", "withdraw-me", receipt)
    assert "cancelled_unsent" not in service.read_snapshot("owner", "withdraw-me")
    with pytest.raises(WorkLifecycleError, match="PROOF_INVALID"):
        service.cancel_unsent(IDENTITY, "withdraw-me", context["request_hash"],
                              context["expected_binding"], context["reason"], "apply")
    assert applied["cancelled_unsent"]["request_hash"] == context["request_hash"]


def test_cancel_refuses_immutable_input_or_child_lineage_drift(service):
    service.submit("owner", body(), source="internal:content")
    context = pinned(service)
    with service.store.transaction() as db:
        receipt = service.store.read_receipt(db, "text", "owner", "withdraw-me")
        receipt["_input_ref"] = service.store.save_input({**body(), "messages": [{"role": "user", "content": "changed"}]})
        service.store.write_receipt(db, "text", "owner", "withdraw-me", receipt)
    with pytest.raises(WorkLifecycleError, match="IDENTITY_CONFLICT"):
        service.cancel_unsent(IDENTITY, "withdraw-me", context["request_hash"],
                              context["expected_binding"], context["reason"], "apply")

    service.submit("owner", body("with-child"), source="internal:content")
    child_context = pinned(service, "with-child")
    with service.store.transaction() as db:
        db.execute("INSERT INTO requests VALUES(?,?,?,?)", (
            "owner", "child", "fixture", '{"_supersedes_request_id":"with-child"}',
        ))
    with pytest.raises(WorkLifecycleError, match="NOT_UNSENT"):
        service.cancel_unsent(IDENTITY, "with-child", child_context["request_hash"],
                              child_context["expected_binding"], child_context["reason"], "apply")


def test_cancel_service_rejects_invalid_mode_and_wrong_owner(service):
    service.submit("owner", body(), source="internal:content")
    context = pinned(service)
    with pytest.raises(WorkLifecycleError, match="CANCEL_INVALID"):
        service.cancel_unsent(IDENTITY, "withdraw-me", context["request_hash"],
                              context["expected_binding"], context["reason"], "implicit-apply")
    with pytest.raises(WorkLifecycleError, match="CANCEL_NOT_FOUND"):
        service.cancel_unsent({**IDENTITY, "id": "other"}, "withdraw-me", context["request_hash"],
                              context["expected_binding"], context["reason"], "apply")


def test_cancel_proof_and_snapshot_refuse_a_relocated_request_row(service):
    service.submit("owner", body(), source="internal:content")
    context = pinned(service)
    service.cancel_unsent(IDENTITY, "withdraw-me", context["request_hash"],
                          context["expected_binding"], context["reason"], "apply")
    with service.store.transaction() as db:
        db.execute("UPDATE requests SET id=? WHERE owner=? AND id=?", ("relocated", "owner", "withdraw-me"))
    snapshot = service.read_snapshot("owner", "relocated", include_cancel_unsent_context=True,
                                     cancel_unsent_source="internal:content")
    assert "cancelled_unsent" not in snapshot and "cancel_unsent_context" not in snapshot
    with pytest.raises(WorkLifecycleError, match="PROOF_INVALID"):
        service.cancel_unsent(IDENTITY, "relocated", context["request_hash"],
                              context["expected_binding"], context["reason"], "apply")


@pytest.mark.parametrize("changes", [
    {"_claim_id": None, "_claim_until": 0, "_executing": False},
    {"_submission_started": True, "_execution_timeline": [{"stage": "send_call_started", "at": 1000}]},
    {"status": "unknown", "error_code": "CONVERSATION_OUTCOME_UNKNOWN"},
    {"result": {"content": "already returned"}},
    {"_recovery_paused": True},
])
def test_cancel_rejects_claim_history_unknown_results_or_recovery(service, changes):
    service.submit("owner", body(), source="internal:content")
    context = pinned(service)
    with service.store.transaction() as db:
        receipt = service.store.read_receipt(db, "text", "owner", "withdraw-me")
        receipt.update(changes)
        service.store.write_receipt(db, "text", "owner", "withdraw-me", receipt)
    with pytest.raises(WorkLifecycleError, match="NOT_UNSENT"):
        service.cancel_unsent(IDENTITY, "withdraw-me", context["request_hash"],
                              context["expected_binding"], context["reason"], "apply")


def test_cancel_and_real_pool_claim_serialize_in_both_orders(tmp_path, monkeypatch):
    (tmp_path / "accounts.json").write_text(json.dumps([{
        "access_token": "fixture", "account_id": "upstream", "provider_account_identity": "account-0",
        "type": "Plus", "status": "正常", "quota": 99, "source_type": "web",
        "conversation_binding_ids": ["binding-0"],
        "limits_progress": [{"feature_name": "image_gen", "remaining": 99}],
        "capacity_observed_at": datetime.now(timezone.utc).isoformat(),
    }]))
    clock = Clock()
    _, store, admission = build(tmp_path, clock)
    monkeypatch.setattr(admission, "_types", lambda _: {"Plus"})
    text = TextTaskService(store.path, admission=admission, runner=Mock(), clock=clock)
    pool_binding = {**BINDING, "provider_binding_id": "binding-0", "provider_account_identity": "account-0"}
    text.submit("owner", {"client_request_id": "cancel-first", "messages": [{"role": "user", "content": "original"}],
                          **pool_binding}, source="internal:content")
    first = text.read_snapshot("owner", "cancel-first", include_cancel_unsent_context=True,
                               cancel_unsent_source="internal:content")["cancel_unsent_context"]
    cancel_entered, release_cancel, claim_started, claim_done = (
        threading.Event(), threading.Event(), threading.Event(), threading.Event())
    original_context = text._ordinary_cancel_context

    def paused_cancel_context(*args, **kwargs):
        cancel_entered.set()
        assert release_cancel.wait(3)
        return original_context(*args, **kwargs)

    monkeypatch.setattr(text, "_ordinary_cancel_context", paused_cancel_context)
    with ThreadPoolExecutor(max_workers=2) as workers:
        cancelled = workers.submit(text.cancel_unsent, IDENTITY, "cancel-first", first["request_hash"],
                                   first["expected_binding"], first["reason"], "apply")
        assert cancel_entered.wait(3)
        def claim_after_cancel():
            claim_started.set()
            try:
                return admission.claim_next()
            finally:
                claim_done.set()

        claimed_after_cancel = workers.submit(claim_after_cancel)
        assert claim_started.wait(3)
        assert not claim_done.wait(0.1)
        release_cancel.set()
        assert cancelled.result(timeout=3)["status"] == "cancelled"
        assert claimed_after_cancel.result(timeout=3) is None

    monkeypatch.setattr(text, "_ordinary_cancel_context", original_context)
    text.submit("owner", {"client_request_id": "claim-first", "messages": [{"role": "user", "content": "original"}],
                          **pool_binding}, source="internal:content")
    second = text.read_snapshot("owner", "claim-first", include_cancel_unsent_context=True,
                                cancel_unsent_source="internal:content")["cancel_unsent_context"]
    claim_entered, release_claim, cancel_done = threading.Event(), threading.Event(), threading.Event()
    original_snapshot = admission._snapshot

    def paused_snapshot(*args, **kwargs):
        claim_entered.set()
        assert release_claim.wait(3)
        return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(admission, "_snapshot", paused_snapshot)
    with ThreadPoolExecutor(max_workers=2) as workers:
        claimed = workers.submit(admission.claim_next)
        assert claim_entered.wait(3)

        def cancel_after_claim():
            try:
                return text.cancel_unsent(IDENTITY, "claim-first", second["request_hash"],
                                          second["expected_binding"], second["reason"], "apply")
            finally:
                cancel_done.set()

        rejected = workers.submit(cancel_after_claim)
        assert not cancel_done.wait(0.1)
        release_claim.set()
        context = claimed.result(timeout=3)
        assert context and context.request_id == "claim-first"
        with pytest.raises(WorkLifecycleError, match="NOT_UNSENT"):
            rejected.result(timeout=3)
