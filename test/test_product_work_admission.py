"""Opt-in product parents in the existing durable PoolAdmission scheduler."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json

import pytest

from services.image_task_service import ImageTaskService
from services.product_work import record_slot
from services.text_task_service import TextTaskService
from test.test_pool_admission import Clock, build

SOURCE_A = "internal:listing"
SOURCE_B = "internal:content"


@pytest.fixture
def runtime(tmp_path):
    rows = [{"access_token": f"fixture-token-{i}", "account_id": f"upstream-{i}",
             "provider_account_identity": f"account-{i}", "type": "Plus", "status": "正常",
             "quota": 99, "source_type": "web", "conversation_binding_ids": [f"binding-{i}"],
             "limits_progress": [{"feature_name": "image_gen", "remaining": 99}],
             "capacity_observed_at": datetime.now(timezone.utc).isoformat()}
            for i in range(2)]
    (tmp_path / "accounts.json").write_text(json.dumps(rows))
    clock = Clock()
    accounts, store, admission = build(tmp_path, clock)
    text = TextTaskService(store.path, admission=admission, clock=clock)
    images = ImageTaskService(tmp_path / "images.json", store=store, admission=admission)
    return clock, accounts, store, admission, text, images



def text_submit(rt, request_id, *, owner="owner-a", source=SOURCE_A, product="product-a", step=None):
    _, _, _, _, text, _ = rt
    return text.submit(owner, {"client_request_id": request_id,
        "client_conversation_id": f"session-{request_id}", "model": "fixture-text",
        "messages": [{"role": "user", "content": "synthetic"}],
        **({"product_work": {"product_ref": product, "step_ref": step or request_id}} if product else {})}, source=source)


def image_submit(rt, request_id, *, owner="owner-a", source=SOURCE_A, product="product-a", step=None):
    _, _, _, _, _, images = rt
    return images.submit_generation({"id": owner, "role": "admin", "_trusted_source": source,
                                     "external_image_client": True}, client_task_id=request_id,
        prompt="synthetic", model="gpt-image-2", size=None,
        client_conversation_id=f"image-{request_id}",
        **({"product_work": {"product_ref": product, "step_ref": step or request_id}} if product else {}))


def receipt(rt, kind, owner, request_id):
    _, _, store, _, _, _ = rt
    with store.connect() as db:
        return store.read_receipt(db, kind, owner, request_id)


def finish(rt, kind, owner, request_id):
    clock, _, store, _, _, _ = rt
    with store.transaction() as db:
        row = store.read_receipt(db, kind, owner, request_id)
        row.update(status="success" if kind == "image" else "succeeded", _executing=False,
                   _turn_reserved=False, _attempt_finished_at=clock.now,
                   upstream_unfinished=False, upstream_outcome="completed")
        store.write_receipt(db, kind, owner, request_id, row)


def parent(rt, row):
    _, _, store, _, _, _ = rt
    with store.connect() as db:
        return store.runtime(db, row["_product_work_key"])


def held_parent(rt, kind, owner, request_id):
    _, _, store, _, _, _ = rt
    with store.transaction() as db:
        row = store.read_receipt(db, kind, owner, request_id)
        record_slot(store, db, row, True)
        store.write_receipt(db, kind, owner, request_id, row)


def test_mixed_text_image_share_one_parent_and_continuation_skips_new_slot(runtime):
    text_submit(runtime, "text")
    image_submit(runtime, "image", step="image")
    first = runtime[3].claim_next()
    assert first and first.request_id == "text"
    assert parent(runtime, receipt(runtime, "text", "owner-a", "text"))["slot_held"]
    finish(runtime, "text", "owner-a", "text")
    second = runtime[3].claim_next()
    assert second and second.kind == "image" and second.request_id == "image"
    image_parent = parent(runtime, receipt(runtime, "image", "owner-a", "image"))
    assert image_parent["state"] == "active" and image_parent["slot_held"]


def test_new_product_capacity_is_shared_across_sources_and_claim_is_atomic(runtime):
    text_submit(runtime, "a", owner="owner-a", source=SOURCE_A, product="a")
    text_submit(runtime, "b", owner="owner-b", source=SOURCE_B, product="b")
    admission = runtime[3]
    runtime[1].update_account("fixture-token-1", {"managed_disabled": True}, quiet=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = list(pool.map(lambda _: admission.claim_next(), range(2)))
    claimed = [item for item in claimed if item is not None]
    assert len(claimed) == 1
    assert parent(runtime, receipt(runtime, "text", claimed[0].owner, claimed[0].request_id))["slot_held"]
    queued = "b" if claimed[0].request_id == "a" else "a"
    assert receipt(runtime, "text", "owner-b" if queued == "b" else "owner-a", queued)["status"] == "queued"
    assert admission.claim_next() is None


def test_capacity_shrink_and_unknown_parent_block_only_new_products(runtime):
    text_submit(runtime, "a-first", owner="owner-a", source=SOURCE_A, product="a")
    text_submit(runtime, "b-first", owner="owner-b", source=SOURCE_B, product="b")
    admission = runtime[3]
    first, second = admission.claim_next(), admission.claim_next()
    assert {first.request_id, second.request_id} == {"a-first", "b-first"}
    finish(runtime, "text", "owner-a", "a-first")
    finish(runtime, "text", "owner-b", "b-first")
    text_submit(runtime, "a-next", owner="owner-a", source=SOURCE_A, product="a")
    text_submit(runtime, "new", owner="owner-c", source=SOURCE_B, product="new")
    # Losing an eligible account lowers the snapshot total below its two held
    # parents. The existing parent can continue on account-0; a new one cannot.
    runtime[1].update_account("fixture-token-1", {"managed_disabled": True}, quiet=True)
    continuation = admission.claim_next()
    assert continuation and continuation.request_id == "a-next"
    finish(runtime, "text", "owner-a", "a-next")
    assert receipt(runtime, "text", "owner-c", "new")["status"] == "queued"
    assert admission.claim_next() is None

    # An unresolved held parent also remains counted. New work must not infer
    # its capacity from a free worker or an unrelated source.
    with runtime[2].transaction() as db:
        first_row = runtime[2].read_receipt(db, "text", "owner-a", "a-first")
        first_row.update(status="unknown", upstream_outcome="unknown", _attempt_finished_at=None)
        runtime[2].write_receipt(db, "text", "owner-a", "a-first", first_row)
    assert admission.claim_next() is None


def test_fixed_product_lanes_rotate_with_legacy_without_idling_on_unready_continuation(runtime):
    admission = runtime[3]
    # A terminal first member retains the product slot while its ready second
    # member uses the fixed continuation lane rather than a per-product lane.
    text_submit(runtime, "seed", product="held")
    held_parent(runtime, "text", "owner-a", "seed")
    finish(runtime, "text", "owner-a", "seed")
    text_submit(runtime, "continuation", product="held")
    text_submit(runtime, "legacy-a", product=None)
    text_submit(runtime, "legacy-b", owner="owner-b", source=SOURCE_B, product=None)
    first = admission.claim_next()
    finish(runtime, "text", first.owner, first.request_id)
    second = admission.claim_next()
    finish(runtime, "text", second.owner, second.request_id)
    third = admission.claim_next()
    assert [item.request_id for item in (first, second, third)] == ["legacy-b", "legacy-a", "continuation"]
    finish(runtime, "text", third.owner, third.request_id)

    text_submit(runtime, "blocked-continuation", product="held")
    with runtime[2].transaction() as db:
        row = runtime[2].read_receipt(db, "text", "owner-a", "blocked-continuation")
        row["_ready_at"] = runtime[0].now + 100
        runtime[2].write_receipt(db, "text", "owner-a", "blocked-continuation", row)
    text_submit(runtime, "legacy-ready", product=None)
    selected = admission.claim_next()
    assert selected and selected.request_id == "legacy-ready"
