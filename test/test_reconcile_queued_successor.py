"""A proven internal listing child may gain its missing Provider lineage once."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.ai as api
from services.conversation_binding_service import ConversationBindingError
from services.product_work import ProductWorkService, ensure_product_work
from services.request_context import current_request
from services.text_task_service import TextTaskService
from services.work_lifecycle import WorkLifecycleError, WorkLifecycleService
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


def reconcile(service, body, *, dry_run=True, continue_after_no_final=False):
    return service.reconcile_queued_successor(
        "owner", "queued", "root", cursor(body), source="internal:listing", dry_run=dry_run,
        continue_after_no_final=continue_after_no_final,
    )


def qualify_no_final(service):
    """Turn the frozen root into the separately proved no-final outcome."""
    update(service, "root", status="failed", error_code="RESULT_UNRECOVERABLE", upstream_outcome="unknown",
           recovery_reason="REQUEST_RESULT_NOT_FOUND", recovery_no_result_reads=3,
           _attempt_reason="ORIGINAL_RESULT_NO_FINAL", _attempt_finished_at=900,
           _execution_wait_ended_at=900, _submission_started=True, _last_sent_sequence=0,
           _executing=False, _turn_reserved=False, _claim_id=None, _claim_until=None,
           recovery_claim_id=None, recovery_lease_until=None, _scheduling_owner="internal:listing")


def attach_product_work_to_reconciled_pair(service):
    """Seed the existing accepted pair's durable product association for close checks."""
    metadata = {"product_ref": "queued-reconcile-product", "step_ref": "title"}
    with service.store.transaction() as db:
        root = service.store.read_receipt(db, "text", "owner", "root")
        child = service.store.read_receipt(db, "text", "owner", "queued")
        root_body = {**service.store.load_input(root["_input_ref"]), "product_work": metadata}
        child_body = {**root_body, "client_request_id": "queued"}
        root["_input_ref"] = service.store.save_input(root_body)
        child["_input_ref"] = service.store.save_input(child_body)
        root_hash = service._submission_identity("owner", root_body)[1]
        child_hash = service._submission_identity("owner", child_body)[1]
        db.execute("UPDATE requests SET request_hash=? WHERE owner=? AND id=?", (root_hash, "owner", "root"))
        db.execute("UPDATE requests SET request_hash=? WHERE owner=? AND id=?", (child_hash, "owner", "queued"))
        child["_supersedes_input_hash"] = root_hash
        ensure_product_work(service.store, db, "text", "owner", "root", root, metadata, source="internal:listing")
        ensure_product_work(service.store, db, "text", "owner", "queued", child, metadata, source="internal:listing")
        service.store.write_receipt(db, "text", "owner", "root", root)
        service.store.write_receipt(db, "text", "owner", "queued", child)
    return metadata


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


def test_qualified_no_final_requires_explicit_mode_and_preserves_the_frozen_child(reconciled):
    service, admission, root_body, _, sent, root = reconciled
    qualify_no_final(service)
    root_before, child_before = deepcopy(row(service, "root")), deepcopy(row(service, "queued"))
    with pytest.raises(ConversationBindingError):
        reconcile(service, root_body, dry_run=False)
    assert row(service, "root") == root_before and row(service, "queued") == child_before

    assert reconcile(service, root_body, continue_after_no_final=True) == {
        "request_id": "queued", "original_request_id": "root", "type": "queued_successor_reconciliation",
        "eligible": True, "reconciled": False, "dry_run": True,
    }
    assert row(service, "root") == root_before and row(service, "queued") == child_before
    applied = reconcile(service, root_body, dry_run=False, continue_after_no_final=True)
    assert applied["reconciled"] is True
    child = row(service, "queued")
    assert child["_supersedes_no_final"] is True
    for key in ("request_id", "request_message_id", "_input_ref", "_sequence"):
        assert child[key] == child_before[key]
    with service.store.connect() as db:
        assert service.store.load_input(child["_input_ref"]) == {**root_body, "client_request_id": "queued"}
    assert row(service, "root") == root_before

    claim = admission.claim_next()
    assert claim and claim.request_id == "queued"
    admission.execute(claim)
    assert len(sent) == 1 and sent[0]["_supersedes_no_final_original"] == root_before
    assert reconcile(service, root_body, dry_run=False, continue_after_no_final=True)["reconciled"] is True
    with pytest.raises(ConversationBindingError):
        reconcile(service, root_body, dry_run=False)
    with pytest.raises(ConversationBindingError):
        reconcile(service, {**root_body, "parent_message_id": "other"}, dry_run=False,
                  continue_after_no_final=True)

    _, _, restarted = build(root, service.clock)
    restarted.register("text", lambda context, body: service._run(context.owner, context.request_id, body))
    restarted_service = TextTaskService(service.store.path, runner=service.runner, clock=service.clock, admission=restarted)
    assert reconcile(restarted_service, root_body, dry_run=False, continue_after_no_final=True)["reconciled"] is True


@pytest.mark.parametrize("changes,expected_status", [
    ({"status": "succeeded", "upstream_outcome": "completed", "content": "late"}, "failed"),
    ({"recovery_no_result_reads": 2}, "failed"),
    ({"_claim_id": "busy", "_claim_until": 2000}, "queued"),
])
def test_qualified_no_final_reconcile_rechecks_original_before_send(reconciled, changes, expected_status):
    service, admission, root_body, _, sent, _ = reconciled
    qualify_no_final(service)
    reconcile(service, root_body, dry_run=False, continue_after_no_final=True)
    update(service, "root", **changes)
    claim = admission.claim_next()
    assert claim and claim.request_id == "queued"
    admission.execute(claim)
    child = row(service, "queued")
    assert child["status"] == expected_status and sent == []
    if expected_status == "failed":
        assert child["upstream_outcome"] == "not_sent"


@pytest.mark.parametrize("value", [1, False, "true"])
def test_qualified_no_final_reconcile_rejects_non_boolean_persisted_mode_before_send(reconciled, value):
    service, admission, root_body, _, sent, _ = reconciled
    qualify_no_final(service)
    reconcile(service, root_body, dry_run=False, continue_after_no_final=True)
    update(service, "queued", _supersedes_no_final=value)
    claim = admission.claim_next()
    assert claim and claim.request_id == "queued"
    admission.execute(claim)
    child = row(service, "queued")
    assert child["status"] == "failed" and child["upstream_outcome"] == "not_sent" and sent == []


@pytest.mark.parametrize("value", [1, False, "true"])
def test_qualified_no_final_reconcile_acknowledgement_rejects_non_boolean_persisted_mode(reconciled, value):
    service, admission, root_body, _, sent, _ = reconciled
    qualify_no_final(service)
    reconcile(service, root_body, dry_run=False, continue_after_no_final=True)
    claim = admission.claim_next()
    assert claim and claim.request_id == "queued"
    admission.execute(claim)
    assert len(sent) == 1
    update(service, "queued", _supersedes_no_final=value)
    with pytest.raises(ConversationBindingError):
        reconcile(service, root_body, dry_run=False, continue_after_no_final=True)


def test_qualified_no_final_reconciled_result_closes_and_archives_product_work(reconciled, monkeypatch):
    service, admission, root_body, _, sent, root = reconciled
    qualify_no_final(service)
    reconcile(service, root_body, dry_run=False, continue_after_no_final=True)
    metadata = attach_product_work_to_reconciled_pair(service)
    claim = admission.claim_next()
    assert claim and claim.request_id == "queued"
    admission.execute(claim)
    assert row(service, "queued")["status"] == "succeeded" and len(sent) == 1

    archived = []
    monkeypatch.setattr("services.text_task_service.conversation_binding_service.set_archived",
                        lambda receipt, value: archived.append((receipt["request_id"], value)))
    images = SimpleNamespace(archive_product_work=lambda *_args: None)
    product = ProductWorkService(service, images, clock=service.clock)
    lifecycle = WorkLifecycleService(service, images, clock=service.clock)
    identity = {"id": "owner", "role": "admin", "_trusted_source": "internal:listing"}
    assert product.close(identity, metadata["product_ref"], results_saved=True)["state"] == "closing"
    assert lifecycle.process_one()
    assert archived == [("queued", True)]
    assert product.get(identity, metadata["product_ref"])["state"] == "completed"

    _, store, restarted_admission = build(root, service.clock)
    restarted = TextTaskService(store.path, runner=service.runner, clock=service.clock, admission=restarted_admission)
    restarted_admission.register("text", lambda context, body: restarted._run(context.owner, context.request_id, body))
    restarted_product = ProductWorkService(restarted, images, clock=service.clock)
    assert restarted_product.get(identity, metadata["product_ref"])["state"] == "completed"


@pytest.mark.parametrize("target,changes", [
    ("queued", {"_reconciled_queued_successor": False}),
    ("queued", {"_product_work": {"product_ref": "queued-reconcile-product", "step_ref": "other"}}),
    ("hash", {}),
])
def test_qualified_no_final_reconciled_product_close_rejects_forged_contract(reconciled, target, changes):
    service, admission, root_body, _, _, _ = reconciled
    qualify_no_final(service)
    reconcile(service, root_body, dry_run=False, continue_after_no_final=True)
    metadata = attach_product_work_to_reconciled_pair(service)
    claim = admission.claim_next()
    assert claim and claim.request_id == "queued"
    admission.execute(claim)
    if target == "hash":
        with service.store.connect() as db:
            db.execute("UPDATE requests SET request_hash=? WHERE owner=? AND id=?", ("tampered", "owner", "queued"))
    else:
        update(service, target, **changes)
    images = SimpleNamespace(archive_product_work=lambda *_args: None)
    product = ProductWorkService(service, images, clock=service.clock)
    identity = {"id": "owner", "role": "admin", "_trusted_source": "internal:listing"}
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_"):
        product.close(identity, metadata["product_ref"], results_saved=True)


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
        assert client.post("/api/conversation-bindings/text-requests/queued/reconcile-queued-successor",
                           json={**body, "continue_after_no_final": "true"}).status_code == 422
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


def test_http_explicit_no_final_mode_reaches_reconcile_service(reconciled, monkeypatch):
    service, admission, root_body, _, sent, _ = reconciled
    qualify_no_final(service)
    monkeypatch.setattr(api, "text_task_service", service)
    app = FastAPI()
    app.include_router(api.create_router())
    monkeypatch.setattr(api, "require_identity", lambda _authorization, request=None: {"id": "owner", "role": "admin"})
    monkeypatch.setattr(api, "trusted_source", lambda _identity, _request: "internal:listing")
    body = {"client_request_id": "queued", "original_request_id": "root", **cursor(root_body),
            "continue_after_no_final": True}
    with TestClient(app) as client:
        assert client.post("/api/conversation-bindings/text-requests/queued/reconcile-queued-successor", json=body).status_code == 200
        applied = client.post("/api/conversation-bindings/text-requests/queued/reconcile-queued-successor",
                              json={**body, "dry_run": False})
        assert applied.status_code == 200 and applied.json()["reconciled"] is True
    assert row(service, "queued")["_supersedes_no_final"] is True
    claim = admission.claim_next()
    assert claim and claim.request_id == "queued"
    admission.execute(claim)
    assert len(sent) == 1 and sent[0]["_supersedes_no_final_original"]["request_id"] == "root"


@pytest.fixture
def attempt_steps(tmp_path):
    """Actual Node contract: the business step address contains its write UUID."""
    (tmp_path / "accounts.json").write_text(json.dumps([{
        "access_token": "fixture", "account_id": "physical", "provider_account_identity": "account",
        "conversation_binding_ids": ["binding"], "type": "Plus", "status": "正常", "quota": 99,
        "source_type": "web",
    }]))
    clock = Clock()
    _, store, admission = build(tmp_path, clock)
    sent, before_send = [], []
    def runner(body, on_cursor):
        for action in before_send:
            action()
        current_request.get().before_send()
        sent.append(deepcopy(body))
        return {"content": "answer", "parent_message_id": "answer"}
    service = TextTaskService(store.path, runner=runner, clock=clock, admission=admission)
    admission.register("text", lambda ctx, body: service._run(ctx.owner, ctx.request_id, body))
    original_id = "00000000-0000-4000-8000-000000000001"
    queued_id = "00000000-0000-4000-8000-000000000002"
    base = {"client_conversation_id": "session", "provider_binding_id": "binding",
            "provider_account_identity": "account", "conversation_id": "conversation",
            "parent_message_id": "parent", "model": "fixture-text", "image_model": "gpt-image-2",
            "thinking_effort": "standard", "messages": [{"role": "user", "content": "frozen recognition"}]}
    bodies = []
    for rid in (original_id, queued_id):
        body = {**base, "client_request_id": rid, "product_work": {
            "product_ref": "same-product", "step_ref": "direct-listing.accessory.recognition:" + rid}}
        service.submit("owner", body, source="internal:listing")
        update(service, rid, _scheduling_owner="internal:listing")
        bodies.append(body)
    update(service, original_id, status="failed", error_code="RESULT_UNRECOVERABLE", upstream_outcome="unknown",
           recovery_reason="REQUEST_RESULT_NOT_FOUND", recovery_no_result_reads=3,
           _attempt_reason="ORIGINAL_RESULT_NO_FINAL", _attempt_finished_at=900,
           _execution_wait_ended_at=900, _submission_started=True, _last_sent_sequence=0,
           _executing=False, _turn_reserved=False, _claim_id=None, _claim_until=None,
           recovery_claim_id=None, recovery_lease_until=None)
    return SimpleNamespace(service=service, admission=admission, sent=sent, before_send=before_send,
                           original=original_id, queued=queued_id, bodies=bodies, root=tmp_path)


def reconcile_attempt(h, **overrides):
    options = {"source": "internal:listing", "dry_run": False, "continue_after_no_final": True,
               "reconcile_attempt_step": True, **overrides}
    return h.service.reconcile_queued_successor("owner", h.queued, h.original, cursor(h.bodies[0]), **options)


def rewrite_attempt_input(h, transform):
    """Model a different accepted input, retaining its independently valid hash."""
    with h.service.store.transaction() as db:
        child = h.service.store.read_receipt(db, "text", "owner", h.queued)
        body = transform(deepcopy(h.service.store.load_input(child["_input_ref"])))
        child["_input_ref"] = h.service.store.save_input(body)
        child["_product_work"] = body.get("product_work")
        h.service.store.write_receipt(db, "text", "owner", h.queued, child)
        db.execute("UPDATE requests SET request_hash=? WHERE owner=? AND id=?",
                   (h.service._submission_identity("owner", body)[1], "owner", h.queued))


def test_attempt_addressed_reconcile_preserves_inputs_and_completes_archive_after_restart(attempt_steps, monkeypatch):
    h = attempt_steps
    before = [deepcopy(row(h.service, rid)) for rid in (h.original, h.queued)]
    with h.service.store.connect() as db:
        hashes = list(db.execute("SELECT id,request_hash FROM requests ORDER BY id"))
    with pytest.raises(ConversationBindingError):
        reconcile_attempt(h, reconcile_attempt_step=False)
    assert reconcile_attempt(h, dry_run=True)["reconciled"] is False
    assert [row(h.service, rid) for rid in (h.original, h.queued)] == before
    assert reconcile_attempt(h)["reconciled"] is True
    assert row(h.service, h.original) == before[0]
    for field in ("request_id", "request_message_id", "_input_ref", "_sequence", "_product_work", "_product_work_key"):
        assert row(h.service, h.queued)[field] == before[1][field]
    with h.service.store.connect() as db:
        assert list(db.execute("SELECT id,request_hash FROM requests ORDER BY id")) == hashes
        assert [h.service.store.load_input(row(h.service, rid)["_input_ref"]) for rid in (h.original, h.queued)] == h.bodies
    claim = h.admission.claim_next()
    assert claim and claim.request_id == h.queued
    h.admission.execute(claim)
    assert len(h.sent) == 1 and row(h.service, h.queued)["status"] == "succeeded"
    assert reconcile_attempt(h)["reconciled"] is True
    with pytest.raises(ConversationBindingError):
        reconcile_attempt(h, reconcile_attempt_step=False)

    archived = []
    monkeypatch.setattr("services.text_task_service.conversation_binding_service.set_archived",
                        lambda receipt, value: archived.append((receipt["request_id"], value)))
    images = SimpleNamespace(archive_product_work=lambda *_args: None)
    product = ProductWorkService(h.service, images, clock=h.service.clock)
    identity = {"id": "owner", "role": "admin", "_trusted_source": "internal:listing"}
    assert product.close(identity, "same-product", results_saved=True)["state"] == "closing"
    assert WorkLifecycleService(h.service, images, clock=h.service.clock).process_one()
    assert archived == [(h.queued, True)]
    assert product.get(identity, "same-product")["state"] == "completed"
    _, store, admission = build(h.root, h.service.clock)
    restarted = TextTaskService(store.path, runner=h.service.runner, clock=h.service.clock, admission=admission)
    assert ProductWorkService(restarted, images, clock=h.service.clock).get(identity, "same-product")["state"] == "completed"
    assert restarted.reconcile_queued_successor("owner", h.queued, h.original, cursor(h.bodies[0]),
        source="internal:listing", dry_run=False, continue_after_no_final=True, reconcile_attempt_step=True)["reconciled"]
    assert row(h.service, h.original)["upstream_outcome"] == "unknown"
    assert len(h.sent) == 1


@pytest.mark.parametrize("mutation", ["prompt", "model", "product", "fairness", "operation", "wrong_attempt", "member", "hash", "third_child"])
def test_attempt_reconcile_rejects_different_input_or_unproved_members(attempt_steps, mutation):
    h = attempt_steps
    if mutation in {"prompt", "model", "product", "fairness", "operation", "wrong_attempt"}:
        def change(body):
            if mutation == "prompt": body["messages"] = [{"role": "user", "content": "different"}]
            if mutation == "model": body["model"] = "different"
            if mutation == "product": body["product_work"]["product_ref"] = "other-product"
            if mutation == "fairness": body["product_work"]["fairness"] = {"organization_id": "org", "subject_id": "subject", "root_workflow_id": "root"}
            if mutation == "operation": body["product_work"]["step_ref"] = "direct-listing.gallery.review:" + h.queued
            if mutation == "wrong_attempt": body["product_work"]["step_ref"] = "direct-listing.accessory.recognition:" + h.original
            return body
        rewrite_attempt_input(h, change)
    elif mutation == "member":
        with h.service.store.transaction() as db:
            key = row(h.service, h.queued)["_product_work_key"]
            parent = h.service.store.runtime(db, key)
            parent["members"] = [m for m in parent["members"] if m["request_id"] != h.queued]
            h.service.store.set_runtime(db, key, parent)
    elif mutation == "hash":
        with h.service.store.transaction() as db:
            db.execute("UPDATE requests SET request_hash='bad' WHERE owner='owner' AND id=?", (h.queued,))
    else:
        h.service.submit("owner", {**h.bodies[0], "client_request_id": "other-child", "product_work": {
            "product_ref": "same-product", "step_ref": "other-step"}}, source="internal:listing")
        update(h.service, "other-child", _supersedes_request_id=h.original)
    with pytest.raises(ConversationBindingError):
        reconcile_attempt(h)
    assert h.sent == [] and not row(h.service, h.queued).get("_reconciled_attempt_step")


@pytest.mark.parametrize("mutation", ["late_success", "mode_int", "member", "metadata"])
def test_attempt_reconcile_rechecks_at_actual_send_edge(attempt_steps, mutation):
    h = attempt_steps
    reconcile_attempt(h)
    def drift():
        if mutation == "late_success": update(h.service, h.original, status="succeeded", content="late", upstream_outcome="completed")
        elif mutation == "mode_int": update(h.service, h.queued, _reconciled_attempt_step=1)
        elif mutation == "metadata": update(h.service, h.queued, _product_work={"product_ref": "same-product", "step_ref": "changed"})
        else:
            with h.service.store.transaction() as db:
                key = row(h.service, h.queued)["_product_work_key"]
                parent = h.service.store.runtime(db, key)
                parent["members"][1]["step_ref"] = "changed"
                h.service.store.set_runtime(db, key, parent)
    h.before_send.append(drift)
    claim = h.admission.claim_next()
    assert claim and claim.request_id == h.queued
    h.admission.execute(claim)
    assert h.sent == [] and row(h.service, h.queued)["upstream_outcome"] == "not_sent"


@pytest.mark.parametrize("mode", [False, 1, "true"])
def test_attempt_reconcile_close_and_ack_reject_persisted_mode_drift(attempt_steps, mode):
    h = attempt_steps
    reconcile_attempt(h)
    claim = h.admission.claim_next()
    h.admission.execute(claim)
    assert len(h.sent) == 1
    update(h.service, h.queued, _reconciled_attempt_step=mode)
    with pytest.raises(ConversationBindingError): reconcile_attempt(h)
    product = ProductWorkService(h.service, SimpleNamespace(archive_product_work=lambda *_: None), clock=h.service.clock)
    with pytest.raises(WorkLifecycleError):
        product.close({"id": "owner", "role": "admin", "_trusted_source": "internal:listing"}, "same-product", results_saved=True)


def test_http_attempt_step_requires_explicit_strict_no_final_mode(attempt_steps, monkeypatch):
    h = attempt_steps
    monkeypatch.setattr(api, "text_task_service", h.service)
    monkeypatch.setattr(api, "require_identity", lambda _authorization, request=None: {"id": "owner", "role": "admin"})
    monkeypatch.setattr(api, "trusted_source", lambda _identity, _request: "internal:listing")
    app = FastAPI(); app.include_router(api.create_router())
    body = {"client_request_id": h.queued, "original_request_id": h.original, **cursor(h.bodies[0]),
            "reconcile_attempt_step": True}
    url = f"/api/conversation-bindings/text-requests/{h.queued}/reconcile-queued-successor"
    with TestClient(app) as client:
        assert client.post(url, json=body).status_code == 409
        assert client.post(url, json={**body, "continue_after_no_final": True, "reconcile_attempt_step": 1}).status_code == 422
        assert client.post(url, json={**body, "continue_after_no_final": True}).status_code == 200
        assert not row(h.service, h.queued).get("_reconciled_attempt_step")
        assert client.post(url, json={**body, "continue_after_no_final": True, "dry_run": False}).status_code == 200
        assert row(h.service, h.queued)["_reconciled_attempt_step"] is True
