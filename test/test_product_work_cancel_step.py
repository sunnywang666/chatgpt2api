"""Bounded local cancellation of one never-sent ProductWork text step."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from services.product_work import ProductWorkService, before_send
from services.text_task_service import TextTaskService
from services.work_lifecycle import WorkLifecycleError, WorkLifecycleService


IDENTITY = {"id": "owner", "role": "admin", "_trusted_source": "internal:content"}
PRODUCT = "fixture-product"


def body(request_id, step, content="original task"):
    return {"client_request_id": request_id, "client_conversation_id": "fixture-client",
            "messages": [{"role": "user", "content": content}],
            "product_work": {"product_ref": PRODUCT, "step_ref": step}}


@pytest.fixture
def runtime(tmp_path):
    admission = SimpleNamespace(wake=Mock())
    text = TextTaskService(tmp_path / "tasks.sqlite3", admission=admission, runner=Mock(), clock=lambda: 1000.0)
    calls = []

    def archive(owner, request_id, archived):
        calls.append((owner, request_id, archived))
        with text.store.connect() as db:
            receipt = text.store.read_receipt(db, "text", owner, request_id)
        return {"request_id": request_id, "conversation_id": receipt["conversation_id"], "archived": archived}

    text.archive_product_work = archive
    images = SimpleNamespace(archive_product_work=archive)
    service = ProductWorkService(text, images, clock=lambda: 1000.0)
    lifecycle = WorkLifecycleService(text, images, clock=lambda: 1000.0)
    return text, service, lifecycle, calls


def submit(text, request_id, step, content="original task"):
    return text.submit("owner", body(request_id, step, content), source=IDENTITY["_trusted_source"])


def succeed(text, request_id):
    with text.store.transaction() as db:
        receipt = text.store.read_receipt(db, "text", "owner", request_id)
        receipt.update(status="succeeded", content="saved result", upstream_outcome="completed",
                       _submission_started=True, provider_binding_id="binding",
                       provider_account_identity="account", conversation_id="conversation",
                       parent_message_id="parent")
        text.store.write_receipt(db, "text", "owner", request_id, receipt)


def receipt(text, request_id):
    with text.store.connect() as db:
        return text.store.read_receipt(db, "text", "owner", request_id)


def test_cancelled_step_is_readable_never_sent_and_cannot_claim_or_resubmit(runtime):
    text, service, _, _ = runtime
    submit(text, "later", "later")
    result = service.cancel_step(IDENTITY, PRODUCT, "later", 1)
    proof = result["cancelled_unsent"]
    assert proof == result["members"][0]["cancelled_unsent"]
    assert proof["request_id"] == "later" and proof["upstream_outcome"] == "not_sent"
    saved = receipt(text, "later")
    assert saved["status"] == "cancelled" and saved["_input_ref"]
    assert text.read("owner", "later")["cancelled_unsent"] == proof
    assert text.read_snapshot("owner", "later")["cancelled_unsent"] == proof
    assert text.submit("owner", body("later", "later"), source=IDENTITY["_trusted_source"])["status"] == "cancelled"
    with text.store.transaction() as db:
        with pytest.raises(WorkLifecycleError, match="STEP_CANCELLED"):
            before_send(text.store, db, saved)
    assert service.cancel_step(IDENTITY, PRODUCT, "later", 1)["cancelled_unsent"] == proof


@pytest.mark.parametrize("changes", [
    {"_claim_id": "claim", "_executing": True, "status": "running"},
    {"_recovery_paused": True},
    {"_supersedes_request_id": "old"},
    {"response_id": "upstream"},
    {"_execution_timeline": [{"stage": "send_call_started", "at": 1}]},
])
def test_cancel_rejects_claim_recovery_lineage_or_send_evidence_without_mutation(runtime, changes):
    text, service, _, _ = runtime
    submit(text, "later", "later")
    before = receipt(text, "later")
    with text.store.transaction() as db:
        current = text.store.read_receipt(db, "text", "owner", "later")
        current.update(changes)
        text.store.write_receipt(db, "text", "owner", "later", current)
    with pytest.raises(WorkLifecycleError, match="CANCEL_STEP_NOT_UNSENT"):
        service.cancel_step(IDENTITY, PRODUCT, "later", 1)
    assert receipt(text, "later") == {**before, **changes}


def test_cancel_proof_input_or_parent_drift_is_not_public_or_a_close_waiver(runtime):
    text, service, _, _ = runtime
    submit(text, "later", "later")
    proof = service.cancel_step(IDENTITY, PRODUCT, "later", 1)["cancelled_unsent"]
    with text.store.transaction() as db:
        current = text.store.read_receipt(db, "text", "owner", "later")
        current["_product_unsent_cancelled"]["request_hash"] = "changed"
        text.store.write_receipt(db, "text", "owner", "later", current)
    assert "cancelled_unsent" not in text.read("owner", "later")
    assert "cancelled_unsent" not in service.get(IDENTITY, PRODUCT)["members"][0]
    with pytest.raises(WorkLifecycleError, match="TURN_UNFINISHED"):
        service.close(IDENTITY, PRODUCT, results_saved=True)
    assert proof["request_hash"] != "changed"


def test_cancel_rejects_receipt_binding_drift_from_its_immutable_input(runtime):
    text, service, _, _ = runtime
    bound = {**body("later", "later"), "provider_binding_id": "binding",
             "provider_account_identity": "account", "conversation_id": "conversation",
             "parent_message_id": "parent"}
    text.submit("owner", bound, source=IDENTITY["_trusted_source"])
    with text.store.transaction() as db:
        current = text.store.read_receipt(db, "text", "owner", "later")
        current["conversation_id"] = "different-conversation"
        text.store.write_receipt(db, "text", "owner", "later", current)
    with pytest.raises(WorkLifecycleError, match="CANCEL_STEP_INPUT_INVALID"):
        service.cancel_step(IDENTITY, PRODUCT, "later", 1)
    assert receipt(text, "later")["status"] == "queued"


@pytest.mark.parametrize("changes", [
    {"_product_work": {"product_ref": PRODUCT, "step_ref": "other"}},
    {"request_id": "another-request"},
])
def test_cancel_rejects_parent_member_metadata_drift_without_writing(runtime, changes):
    text, service, _, _ = runtime
    submit(text, "later", "later")
    with text.store.transaction() as db:
        current = text.store.read_receipt(db, "text", "owner", "later")
        current.update(changes)
        text.store.write_receipt(db, "text", "owner", "later", current)
        before = (list(db.execute("SELECT * FROM requests")), list(db.execute("SELECT * FROM task_runtime")))
    with pytest.raises(WorkLifecycleError, match="CANCEL_STEP_INPUT_INVALID"):
        service.cancel_step(IDENTITY, PRODUCT, "later", 1)
    assert receipt(text, "later")["status"] == "queued"
    with text.store.connect() as db:
        assert (list(db.execute("SELECT * FROM requests")), list(db.execute("SELECT * FROM task_runtime"))) == before


def test_cancel_then_close_resume_and_second_cycle_preserves_old_cancelled_member(runtime):
    text, service, lifecycle, calls = runtime
    submit(text, "saved-v1", "saved-v1")
    submit(text, "later", "later")
    service.cancel_step(IDENTITY, PRODUCT, "later", 1)
    succeed(text, "saved-v1")
    assert service.close(IDENTITY, PRODUCT, results_saved=True)["state"] == "closing"
    assert lifecycle.process_one()
    assert service.get(IDENTITY, PRODUCT)["state"] == "completed"
    assert service.cancel_step(IDENTITY, PRODUCT, "later", 1)["state"] == "completed"
    assert service.resume(IDENTITY, PRODUCT, 1)["version"] == 2
    with pytest.raises(WorkLifecycleError, match="VERSION_CONFLICT"):
        service.cancel_step(IDENTITY, PRODUCT, "later", 1)
    submit(text, "saved-v2", "saved-v2")
    succeed(text, "saved-v2")
    assert service.close(IDENTITY, PRODUCT, results_saved=True, expected_version=2)["state"] == "closing"
    assert lifecycle.process_one()
    assert service.get(IDENTITY, PRODUCT)["state"] == "completed"
    assert receipt(text, "later")["status"] == "cancelled"
    assert calls == [("owner", "saved-v1", True), ("owner", "saved-v2", True)]


def test_archive_refuses_cancel_proof_drift_after_close(runtime):
    text, service, lifecycle, calls = runtime
    submit(text, "saved", "saved")
    submit(text, "later", "later")
    service.cancel_step(IDENTITY, PRODUCT, "later", 1)
    succeed(text, "saved")
    assert service.close(IDENTITY, PRODUCT, results_saved=True)["state"] == "closing"
    with text.store.transaction() as db:
        current = text.store.read_receipt(db, "text", "owner", "later")
        current["_product_unsent_cancelled"]["request_hash"] = "drifted"
        text.store.write_receipt(db, "text", "owner", "later", current)
    assert not lifecycle.process_one()
    assert service.get(IDENTITY, PRODUCT)["state"] == "closing"
    assert calls == []
