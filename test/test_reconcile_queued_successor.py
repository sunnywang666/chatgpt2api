"""A proven internal listing child may gain its missing Provider lineage once."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.ai as api
from services.conversation_binding_service import ConversationBindingError
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_explicit_unknown_successor import row, update
from test.test_pool_admission import Clock, build


def cursor(body):
    return {key: body[key] for key in (
        "provider_binding_id", "provider_account_identity", "client_conversation_id",
        "conversation_id", "parent_message_id",
    )}


@pytest.fixture
def reconciled(tmp_path):
    (tmp_path / "accounts.json").write_text(json.dumps([{
        "access_token": "fixture", "account_id": "physical", "provider_account_identity": "account",
        "conversation_binding_ids": ["binding"], "type": "Plus", "status": "正常", "quota": 99,
        "source_type": "web",
    }]))
    clock = Clock()
    _, store, admission = build(tmp_path, clock)
    sent = []

    def runner(body, on_cursor):
        current_request.get().before_send()
        sent.append(deepcopy(body))
        return {"content": "answer", "parent_message_id": "answer"}

    service = TextTaskService(store.path, runner=runner, clock=clock, admission=admission)
    admission.register("text", lambda context, body: service._run(context.owner, context.request_id, body))
    root_body = {
        "client_request_id": "root", "client_conversation_id": "session",
        "provider_binding_id": "binding", "provider_account_identity": "account",
        "conversation_id": "conversation", "parent_message_id": "parent", "model": "fixture-text",
        "image_model": "gpt-image-2", "thinking_effort": "standard",
        "messages": [{"role": "user", "content": "frozen listing request"}],
    }
    child_body = {**root_body, "client_request_id": "queued"}
    service.submit("owner", root_body, source="internal:listing")
    service.submit("owner", child_body, source="internal:listing")
    update(service, "root", status="failed", error_code="RESULT_UNRECOVERABLE", upstream_outcome="unknown",
           recovery_reason="REQUEST_MESSAGE_NOT_FOUND", _attempt_reason="ORIGINAL_REQUEST_MESSAGE_NOT_FOUND",
           _attempt_finished_at=900, _execution_wait_ended_at=900, _submission_started=True,
           _executing=False, _turn_reserved=False, _claim_id=None, _claim_until=None,
           recovery_claim_id=None, recovery_lease_until=None, _scheduling_owner="internal:listing")
    update(service, "queued", _scheduling_owner="internal:listing")
    return service, admission, root_body, child_body, sent, tmp_path


def reconcile(service, body, *, dry_run=True):
    return service.reconcile_queued_successor(
        "owner", "queued", "root", cursor(body), source="internal:listing", dry_run=dry_run,
    )


def test_dry_run_is_read_only_then_apply_claims_same_frozen_receipt(reconciled):
    service, admission, root_body, _, sent, _ = reconciled
    assert admission.claim_next() is None
    # Admission records the ordinary waiting projection above; dry-run itself
    # must not add a second write or alter either immutable identity.
    root_before, queued_before = deepcopy(row(service, "root")), deepcopy(row(service, "queued"))
    with service.store.connect() as db:
        hashes_before = dict(db.execute("SELECT id,request_hash FROM requests"))
    assert reconcile(service, root_body) == {
        "request_id": "queued", "original_request_id": "root", "type": "queued_successor_reconciliation", "eligible": True,
        "reconciled": False, "dry_run": True,
    }
    assert row(service, "root") == root_before and row(service, "queued") == queued_before

    applied = reconcile(service, root_body, dry_run=False)
    assert applied["reconciled"] is True and applied["dry_run"] is False
    queued = row(service, "queued")
    for key in ("request_id", "request_message_id", "_input_ref", "_sequence"):
        assert queued[key] == queued_before[key]
    with service.store.connect() as db:
        assert dict(db.execute("SELECT id,request_hash FROM requests")) == hashes_before
        assert service.store.load_input(queued["_input_ref"]) == {**root_body, "client_request_id": "queued"}
    assert row(service, "root") == root_before

    claim = admission.claim_next()
    assert claim and claim.request_id == "queued"
    admission.execute(claim)
    assert row(service, "queued")["status"] == "succeeded"
    assert len(sent) == 1 and sent[0]["messages"] == root_body["messages"]
    assert row(service, "root") == root_before

    # A caller that lost the successful apply response gets the same typed
    # acknowledgement after the normal execution mutates only live receipt
    # fields. It neither rewrites lineage nor makes another upstream send.
    acknowledgement = reconcile(service, root_body, dry_run=False)
    assert acknowledgement == {
        "request_id": "queued", "original_request_id": "root", "type": "queued_successor_reconciliation",
        "eligible": True, "reconciled": True, "dry_run": False,
    }
    with pytest.raises(ConversationBindingError):
        service.reconcile_queued_successor(
            "owner", "queued", "root", {**cursor(root_body), "parent_message_id": "wrong"},
            source="internal:listing", dry_run=False,
        )
    assert len(sent) == 1


@pytest.mark.parametrize("kind", ["prompt", "model", "source", "peer", "claim", "send-count", "last-sent", "root-result", "hash"])
def test_reconcile_rejects_any_unproven_or_busy_state(reconciled, kind):
    service, _, root_body, child_body, _, _ = reconciled
    if kind == "prompt":
        # A new queued receipt preserves all normal identity but carries different frozen input.
        with service.store.transaction() as db:
            receipt = service.store.read_receipt(db, "text", "owner", "queued")
            service.store.write_receipt(db, "text", "owner", "queued", {**receipt, "_input_ref": service.store.save_input(
                {**child_body, "messages": [{"role": "user", "content": "different"}]})})
    elif kind == "model":
        update(service, "queued", model="other")
    elif kind == "source":
        update(service, "queued", _source="internal:content")
    elif kind == "peer":
        service.submit("owner", {**child_body, "client_request_id": "peer"}, source="internal:listing")
        update(service, "peer", _scheduling_owner="internal:listing")
    elif kind == "claim":
        update(service, "queued", _claim_id="busy", _claim_until=2000)
    elif kind == "send-count":
        update(service, "queued", send_count=1)
    elif kind == "last-sent":
        update(service, "queued", _last_sent_sequence=0)
    elif kind == "hash":
        with service.store.connect() as db:
            db.execute("UPDATE requests SET request_hash='changed' WHERE owner=? AND id=?", ("owner", "root"))
    else:
        update(service, "root", status="succeeded", upstream_outcome="completed", content="late")
    before = deepcopy(row(service, "queued"))
    with pytest.raises(ConversationBindingError):
        reconcile(service, root_body, dry_run=False)
    assert row(service, "queued") == before


def test_late_original_success_after_apply_blocks_send_without_replaying(reconciled):
    service, admission, root_body, _, sent, _ = reconciled
    reconcile(service, root_body, dry_run=False)
    update(service, "root", status="succeeded", upstream_outcome="completed", content="late")

    claim = admission.claim_next()
    assert claim and claim.request_id == "queued"
    admission.execute(claim)

    queued = row(service, "queued")
    assert queued["status"] == "failed"
    assert queued["error_code"] == "CHAT_SUPERSEDE_ORIGINAL_FOUND"
    assert queued["upstream_outcome"] == "not_sent"
    assert sent == []


def test_reconcile_is_idempotent_across_concurrent_apply_and_restart(reconciled):
    service, admission, root_body, _, sent, root = reconciled
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: reconcile(service, root_body, dry_run=False), range(2)))
    assert all(result["reconciled"] for result in results)
    queued = row(service, "queued")
    assert queued[TextTaskService._RECONCILED_QUEUED_SUCCESSOR] is True

    _, _, restarted = build(root, service.clock)
    restarted.register("text", lambda context, body: service._run(context.owner, context.request_id, body))
    claim = restarted.claim_next()
    assert claim and claim.request_id == "queued"
    restarted.execute(claim)
    assert len(sent) == 1 and row(service, "root")["status"] == "failed"


def test_http_requires_admin_listing_source_and_has_strict_body(reconciled, monkeypatch):
    service, admission, root_body, _, sent, _ = reconciled
    monkeypatch.setattr(api, "text_task_service", service)
    app = FastAPI()
    app.include_router(api.create_router())
    monkeypatch.setattr(api, "require_identity", lambda _authorization, request=None: {"id": "owner", "role": "admin"})
    monkeypatch.setattr(api, "trusted_source", lambda _identity, _request: "internal:listing")
    body = {"client_request_id": "queued", "original_request_id": "root", **cursor(root_body)}
    with TestClient(app) as client:
        response = client.post("/api/conversation-bindings/text-requests/queued/reconcile-queued-successor", json=body)
        assert response.status_code == 200 and response.json()["dry_run"] is True
        assert response.headers["cache-control"] == "private, no-store"
        assert client.post("/api/conversation-bindings/text-requests/queued/reconcile-queued-successor",
                           json={**body, "messages": []}).status_code == 422
        assert client.post("/api/conversation-bindings/text-requests/wrong/reconcile-queued-successor", json=body).status_code == 409
        changed_cursor = {**body, "parent_message_id": "other"}
        assert client.post("/api/conversation-bindings/text-requests/queued/reconcile-queued-successor",
                           json=changed_cursor).status_code == 409
        applied = client.post("/api/conversation-bindings/text-requests/queued/reconcile-queued-successor",
                              json={**body, "dry_run": False})
        assert applied.status_code == 200 and applied.json()["type"] == "queued_successor_reconciliation"
        claim = admission.claim_next()
        assert claim and claim.request_id == "queued"
        admission.execute(claim)
        # The ordinary live cursor has advanced, but a lost apply response is
        # still acknowledged from the original frozen body and relation.
        retried = client.post("/api/conversation-bindings/text-requests/queued/reconcile-queued-successor",
                              json={**body, "dry_run": False})
        assert retried.status_code == 200 and retried.json()["reconciled"] is True
        assert len(sent) == 1
    monkeypatch.setattr(api, "trusted_source", lambda _identity, _request: "internal:content")
    with TestClient(app) as client:
        assert client.post("/api/conversation-bindings/text-requests/queued/reconcile-queued-successor", json=body).status_code == 404
