"""Lifecycle close/archive coverage for the one missing-node continuation chain."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from services.product_work import ProductWorkService, ensure_product_work
from services.image_task_service import ImageTaskService
from services.request_context import current_request
from services.work_lifecycle import WorkLifecycleError, WorkLifecycleService
from test.test_rejected_chat_successor import (
    missing_node_model_rejection_with_queued_image,
    task,
)


IDENTITY = {"id": "owner", "role": "admin", "_trusted_source": "internal:content"}
PRODUCT = {"product_ref": "fixture-product", "step_ref": "text"}


def _chain(task):
    """Build R/P/C/A4 and execute C plus ordinary retained A4 locally."""
    with task.service.store.transaction() as db:
        root = task.service.store.read_receipt(db, "text", "owner", "old")
        ensure_product_work(task.service.store, db, "text", "owner", "old", root, PRODUCT,
                            source="internal:content")
        task.service.store.write_receipt(db, "text", "owner", "old", root)
    envelope = missing_node_model_rejection_with_queued_image(task)
    task.service.submit("owner", envelope, source="internal:content")
    context = task.admission.claim_next()
    assert context and context.request_id == "missing-model-retry"
    task.admission.execute(context)
    # The local runner stands in for the bound transport's terminal handoff.
    # Product closure deliberately waits for that execution lease to end.
    task.admission.update_claim(context, _claim_id=None, _claim_until=0, _turn_reserved=False)
    seen_image_parents = []
    def generation(payload):
        current_request.get().before_send()
        seen_image_parents.append(payload["parent_message_id"])
        return {"data": [{"url": "https://fixture.invalid/image.png"}],
                "_provider_binding_id": "original-binding",
                "_provider_account_identity": "original-account",
                "_conversation_id": "original-chat", "_parent_message_id": "a4-final"}
    images = ImageTaskService(task.root / "image_tasks.json", store=task.service.store,
                              admission=task.admission, generation_handler=generation)
    task.admission.register("image", lambda ctx, body: images._run_task(
        ctx.owner + ":" + ctx.request_id, body["mode"], body["payload"], body["identity"], body["payload"]["model"]))
    image_context = task.admission.claim_next()
    assert image_context and image_context.request_id == "queued-image"
    task.admission.execute(image_context)
    task.admission.update_claim(image_context, _claim_id=None, _claim_until=0, _turn_reserved=False)
    with task.service.store.transaction() as db:
        continued = task.service.store.read_receipt(db, "text", "owner", "missing-model-retry")
        image = task.service.store.read_receipt(db, "image", "owner", "queued-image")
        assert continued["status"] == "succeeded" and continued["upstream_outcome"] == "completed"
        assert image["status"] == "success" and image.get("_image_thread_terminal") is not True
        assert image["parent_message_id"] == "a4-final" and image["binding_status"] == "bound"
        assert image["_missing_node_retry_request_parent_message_id"] == continued["parent_message_id"]
        assert seen_image_parents == [continued["parent_message_id"]]
    calls, fail = [], [False]
    def archive(owner, request_id, archived):
        calls.append((owner, request_id, archived))
        if fail[0]:
            raise RuntimeError("archive readback unavailable")
        with task.service.store.connect() as db:
            receipt = task.service.store.read_receipt(db, "image", owner, request_id)
        return {"request_id": request_id, "conversation_id": receipt["conversation_id"], "archived": archived}
    text = SimpleNamespace(store=task.service.store, admission=None, archive_product_work=archive)
    images = SimpleNamespace(archive_product_work=archive)
    product = ProductWorkService(text, images, clock=lambda: 1000)
    lifecycle = WorkLifecycleService(text, images, clock=lambda: 1000)
    with task.service.store.transaction() as db:
        parent = product._load(db, IDENTITY, "fixture-product")
        members = product._members(db, parent)
        chains = product._missing_node_chains(db, parent, members)
        assert chains
    return product, lifecycle, calls, fail


def test_completed_missing_node_chain_closes_archives_a4_and_releases_slot(task):
    product, lifecycle, calls, _ = _chain(task)

    closing = product.close(IDENTITY, "fixture-product", results_saved=True)
    assert closing["state"] == "closing" and closing["results_saved"] and closing["slot_held"]
    assert lifecycle.process_one()
    completed = product.get(IDENTITY, "fixture-product")
    assert completed["state"] == "completed" and not completed["slot_held"]
    assert calls == [("owner", "queued-image", True)]


@pytest.mark.parametrize("kind, request_id, changes", [
    ("text", "old", {"_recovery_paused": True}),
    ("text", "missing-parent", {"_claim_id": "late-claim"}),
    ("text", "old", {"status": "succeeded", "upstream_outcome": "completed", "content": "late root"}),
    ("text", "missing-parent", {"status": "succeeded", "upstream_outcome": "completed", "content": "late rejection"}),
    ("text", "missing-model-retry", {"content": ""}),
    ("image", "queued-image", {"status": "queued", "upstream_outcome": "not_sent"}),
])
def test_missing_node_chain_requires_exact_terminal_evidence(task, kind, request_id, changes):
    product, _, calls, _ = _chain(task)
    with task.service.store.transaction() as db:
        receipt = task.service.store.read_receipt(db, kind, "owner", request_id)
        receipt.update(changes)
        task.service.store.write_receipt(db, kind, "owner", request_id, receipt)

    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_"):
        product.close(IDENTITY, "fixture-product", results_saved=True)
    assert calls == []


def test_missing_node_chain_requires_saved_results_and_no_external_member(task):
    product, _, calls, _ = _chain(task)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_RESULTS_SAVE_REQUIRED"):
        product.close(IDENTITY, "fixture-product", results_saved=False)
    with task.service.store.transaction() as db:
        image = task.service.store.read_receipt(db, "image", "owner", "queued-image")
        external = deepcopy(image)
        external.update(id="other-member", owner_id="other", _product_work_key="product_work:external")
        task.service.store.write_receipt(db, "image", "other", "other-member", external)

    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_CONVERSATION_CONFLICT"):
        product.close(IDENTITY, "fixture-product", results_saved=True)
    assert calls == []


def test_archive_commit_rechecks_chain_and_keeps_slot_on_late_change(task):
    product, lifecycle, calls, _ = _chain(task)
    product.close(IDENTITY, "fixture-product", results_saved=True)
    def changed_archive(owner, request_id, archived):
        with task.service.store.transaction() as db:
            root = task.service.store.read_receipt(db, "text", owner, "old")
            root["_recovery_suppressed"] = True
            task.service.store.write_receipt(db, "text", owner, "old", root)
        calls.append((owner, request_id, archived))
        with task.service.store.connect() as db:
            image = task.service.store.read_receipt(db, "image", owner, request_id)
        return {"request_id": request_id, "conversation_id": image["conversation_id"], "archived": archived}
    lifecycle.images.archive_product_work = changed_archive

    assert lifecycle.process_one()
    result = product.get(IDENTITY, "fixture-product")
    assert result["state"] == "closing" and result["slot_held"]
    assert result["archive"]["pending"] == 1 and calls == [("owner", "queued-image", True)]


def test_archive_failure_keeps_missing_node_chain_slot_held(task):
    product, lifecycle, calls, fail = _chain(task)
    product.close(IDENTITY, "fixture-product", results_saved=True)
    fail[0] = True

    assert lifecycle.process_one()
    result = product.get(IDENTITY, "fixture-product")
    assert result["state"] == "closing" and result["slot_held"]
    assert result["archive"]["pending"] == 1 and calls == [("owner", "queued-image", True)]


def _complete_chain(task):
    product, lifecycle, calls, _ = _chain(task)
    assert product.close(IDENTITY, "fixture-product", results_saved=True)["state"] == "closing"
    assert lifecycle.process_one()
    completed = product.get(IDENTITY, "fixture-product")
    assert completed["state"] == "completed" and completed["version"] == 1
    return product, lifecycle, calls


def _new_v2_text(task, request_id="v2-text"):
    return task.service.submit("owner", {
        **task.body,
        "client_request_id": request_id,
        "parent_message_id": "a4-final",
        "product_work": {"product_ref": "fixture-product", "step_ref": request_id},
    }, source="internal:content")


def _archive_any(task, calls):
    def archive(owner, request_id, archived):
        calls.append((owner, request_id, archived))
        with task.service.store.connect() as db:
            receipt = (task.service.store.read_receipt(db, "text", owner, request_id)
                       or task.service.store.read_receipt(db, "image", owner, request_id))
        return {"request_id": request_id, "conversation_id": receipt["conversation_id"], "archived": archived}
    return archive


def test_resumed_missing_node_chain_archives_latest_normal_text_cursor(task):
    product, lifecycle, calls = _complete_chain(task)
    resumed = product.resume(IDENTITY, "fixture-product", expected_version=1)
    assert resumed["state"] == "active" and resumed["version"] == 2

    _new_v2_text(task)
    context = task.admission.claim_next()
    assert context and context.request_id == "v2-text"
    task.admission.execute(context)
    task.admission.update_claim(context, _claim_id=None, _claim_until=0, _turn_reserved=False)

    lifecycle.text.archive_product_work = _archive_any(task, calls)
    closing = product.close(IDENTITY, "fixture-product", results_saved=True, expected_version=2)
    assert closing["state"] == "closing" and closing["version"] == 2
    assert lifecycle.process_one()
    assert calls == [("owner", "queued-image", True), ("owner", "v2-text", True)]
    assert product.get(IDENTITY, "fixture-product")["state"] == "completed"


@pytest.mark.parametrize("tamper", [
    lambda child: child.update(_product_archive_parent="tampered-archive-proof"),
    lambda child: child.update(_product_archive_version=2),
    lambda child: child["target"].update(provider_account_identity="tampered-account"),
    lambda child: child["target"].update(conversation_id="tampered-conversation"),
    lambda child: child.update(last_request_id="tampered-request"),
], ids=["parent", "epoch", "target-account", "target-conversation", "last-request"])
def test_resume_requires_untampered_confirmed_missing_node_archive(task, tamper):
    product, _, _ = _complete_chain(task)
    with task.service.store.transaction() as db:
        parent = product._load(db, IDENTITY, "fixture-product")
        child = task.service.store.runtime(db, parent["archive_keys"][0])
        tamper(child)
        task.service.store.set_runtime(db, child["key"], child)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_ARCHIVE_PENDING"):
        product.resume(IDENTITY, "fixture-product", expected_version=1)


@pytest.mark.parametrize("changes", [
    {"status": "unknown", "upstream_outcome": "unknown", "_attempt_finished_at": None},
    {"_claim_id": "active-v2-claim", "_claim_until": 1100, "_executing": True, "_turn_reserved": True},
])
def test_resumed_missing_node_chain_does_not_waive_unknown_or_active_new_text(task, changes):
    product, _, _ = _complete_chain(task)
    product.resume(IDENTITY, "fixture-product", expected_version=1)
    _new_v2_text(task)
    with task.service.store.transaction() as db:
        current = task.service.store.read_receipt(db, "text", "owner", "v2-text")
        current.update(changes)
        task.service.store.write_receipt(db, "text", "owner", "v2-text", current)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_TURN_UNFINISHED"):
        product.close(IDENTITY, "fixture-product", results_saved=True, expected_version=2)


@pytest.mark.parametrize("kind, request_id, changes", [
    ("text", "old", {"_recovery_paused": True}),
    ("text", "missing-parent", {"_claim_id": "late-historical-claim"}),
    ("text", "missing-model-retry", {"content": ""}),
    ("image", "queued-image", {"status": "unknown", "upstream_outcome": "unknown"}),
], ids=["root-paused", "rejected-claimed", "continue-empty", "a4-unknown"])
def test_resumed_missing_node_chain_rereads_historical_proof_before_close(task, kind, request_id, changes):
    product, _, _ = _complete_chain(task)
    product.resume(IDENTITY, "fixture-product", expected_version=1)
    _new_v2_text(task)
    context = task.admission.claim_next()
    assert context and context.request_id == "v2-text"
    task.admission.execute(context)
    task.admission.update_claim(context, _claim_id=None, _claim_until=0, _turn_reserved=False)
    with task.service.store.transaction() as db:
        historical = task.service.store.read_receipt(db, kind, "owner", request_id)
        historical.update(changes)
        task.service.store.write_receipt(db, kind, "owner", request_id, historical)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_TURN_UNFINISHED"):
        product.close(IDENTITY, "fixture-product", results_saved=True, expected_version=2)


def test_resumed_missing_node_chain_without_new_step_archives_original_a4(task):
    product, lifecycle, calls = _complete_chain(task)
    product.resume(IDENTITY, "fixture-product", expected_version=1)

    assert product.close(IDENTITY, "fixture-product", results_saved=True, expected_version=2)["state"] == "closing"
    assert lifecycle.process_one()
    assert calls == [("owner", "queued-image", True)] * 2
    assert product.get(IDENTITY, "fixture-product")["state"] == "completed"


def test_resumed_missing_node_chain_with_other_conversation_keeps_original_a4_target(task):
    product, lifecycle, calls = _complete_chain(task)
    product.resume(IDENTITY, "fixture-product", expected_version=1)
    task.service.submit("owner", {
        **task.body,
        "client_request_id": "other-conversation-v2",
        "conversation_id": "other-conversation",
        "parent_message_id": "other-parent",
        "product_work": {"product_ref": "fixture-product", "step_ref": "other-conversation-v2"},
    }, source="internal:content")
    context = task.admission.claim_next()
    assert context and context.request_id == "other-conversation-v2"
    task.admission.execute(context)
    task.admission.update_claim(context, _claim_id=None, _claim_until=0, _turn_reserved=False)

    archive = _archive_any(task, calls)
    lifecycle.text.archive_product_work = archive
    lifecycle.images.archive_product_work = archive
    assert product.close(IDENTITY, "fixture-product", results_saved=True, expected_version=2)["state"] == "closing"
    while product.get(IDENTITY, "fixture-product")["state"] == "closing":
        assert lifecycle.process_one()
    assert set(calls[1:]) == {("owner", "queued-image", True), ("owner", "other-conversation-v2", True)}
    assert ("owner", "missing-model-retry", True) not in calls
