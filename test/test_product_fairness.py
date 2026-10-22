"""Trusted person/root fairness does not change receipt ownership or quotas."""
from collections import Counter
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from api.product_works import ProductWorkInput
from api.company_requests import company_identity
from services.product_work import fair_subject
from services.work_lifecycle import WorkLifecycleError
from test.test_product_work_admission import runtime, receipt, finish, parent


def metadata(person="alice", root="root-a", product="product", step="text"):
    return {"product_ref": product, "step_ref": step, "fairness": {
        "organization_id": "company", "subject_id": person, "root_workflow_id": root}}


def submit(rt, rid, *, owner="key-a", person="alice", root="root-a", product=None, step="text", meta=None):
    return rt[4].submit(owner, {"client_request_id": rid, "client_conversation_id": "session-"+rid,
        "model": "fixture-text", "messages": [{"role": "user", "content": "synthetic"}],
        "product_work": meta or metadata(person, root, product or rid, step)}, source="internal:listing")


def test_person_lane_matches_company_identity_and_owner_stays_unchanged(runtime):
    submit(runtime, "one")
    row = receipt(runtime, "text", "key-a", "one")
    before = deepcopy(row)
    work = parent(runtime, row)
    expected = company_identity("company", "alice", "11111111-1111-4111-8111-111111111111")["_fair_source"]
    assert fair_subject(work) == expected
    context = runtime[3].claim_next()
    assert context.owner == "key-a" and context.request_id == "one"
    saved = receipt(runtime, "text", "key-a", "one")
    assert saved["_source"] == saved["_scheduling_owner"] == "internal:listing"
    assert saved["_product_work"] == before["_product_work"]
    assert not saved.get("_scheduling") and not saved.get("_work_key")


def test_multikey_people_roots_rotate_without_default_concurrency_one(runtime):
    for person in ("alice", "bob"):
        for root in ("one", "two"):
            for i in range(4):
                submit(runtime, f"{person}-{root}-{i}", owner=f"{person}-key-{root}", person=person, root=root)
    contexts = [runtime[3].claim_next(), runtime[3].claim_next()]
    assert all(contexts)  # No workflow concurrency=1 was introduced.
    chosen = []
    for _ in range(8):
        context = contexts.pop(0) if contexts else runtime[3].claim_next()
        assert context is not None
        row = receipt(runtime, "text", context.owner, context.request_id)
        chosen.append(row["_product_work"]["fairness"])
        finish(runtime, "text", context.owner, context.request_id)
    assert [x["subject_id"] for x in chosen[:2]] in (["alice", "bob"], ["bob", "alice"])
    assert Counter(x["subject_id"] for x in chosen) == {"alice": 4, "bob": 4}
    assert Counter((x["subject_id"], x["root_workflow_id"]) for x in chosen) == {
        ("alice", "one"): 2, ("alice", "two"): 2, ("bob", "one"): 2, ("bob", "two"): 2}


def test_blocked_root_does_not_block_other_root_and_cursor_survives_restart(runtime):
    submit(runtime, "blocked", root="a")
    submit(runtime, "ready", root="b")
    with runtime[2].transaction() as db:
        row = runtime[2].read_receipt(db, "text", "key-a", "blocked")
        row["_ready_at"] = runtime[0]() + 100
        runtime[2].write_receipt(db, "text", "key-a", "blocked", row)
    context = runtime[3].claim_next()
    assert context.request_id == "ready"
    with runtime[2].connect() as db:
        cursor = runtime[2].runtime(db, "fairness")
        assert cursor["groups"][cursor["source"]] == "workflow:b"
    from test.test_pool_admission import build
    _, store, restarted = build(runtime[2].path.parent, runtime[0])
    with store.connect() as db:
        assert store.runtime(db, "fairness") == cursor
    finish(runtime, "text", "key-a", "ready")
    runtime[0].now += 101
    assert restarted.claim_next().request_id == "blocked"


def test_image_and_text_keep_rotation_inside_original_root(runtime):
    submit(runtime, "text-1")
    submit(runtime, "text-2")
    runtime[5].submit_generation({"id": "key-a", "role": "admin", "_trusted_source": "internal:listing",
        "external_image_client": True}, client_task_id="image", prompt="synthetic", model="gpt-image-2",
        size=None, client_conversation_id="image-session", product_work=metadata(product="image-product", step="image"))
    first = runtime[3].claim_next()
    assert first.kind == "text"
    finish(runtime, "text", first.owner, first.request_id)
    second = runtime[3].claim_next()
    assert second.kind == "image"
    row = receipt(runtime, "image", "key-a", "image")
    assert row["_product_work"]["fairness"] == metadata()["fairness"]


@pytest.mark.parametrize("change", ["subject", "root", "missing", "add_to_legacy"])
def test_frozen_product_fairness_cannot_be_reassigned(runtime, change):
    original = metadata(product="same")
    if change == "add_to_legacy":
        original.pop("fairness")
    submit(runtime, "first", meta=original)
    modified = metadata(product="same", step="next")
    if change == "subject": modified["fairness"]["subject_id"] = "bob"
    elif change == "root": modified["fairness"]["root_workflow_id"] = "other"
    elif change == "missing": modified.pop("fairness")
    with runtime[2].connect() as db:
        before = list(db.execute("SELECT * FROM requests"))
    with pytest.raises(WorkLifecycleError, match="FAIRNESS_CONFLICT"):
        submit(runtime, "next", meta=modified)
    with runtime[2].connect() as db:
        assert list(db.execute("SELECT * FROM requests")) == before


@pytest.mark.parametrize("change", ["empty", "space", "extra", "missing", "wrong_type", "null"])
def test_fairness_api_is_strict(change):
    value = metadata()
    if change == "empty": value["fairness"]["subject_id"] = ""
    elif change == "space": value["fairness"]["organization_id"] = "bad org"
    elif change == "extra": value["fairness"]["priority"] = 9
    elif change == "missing": value["fairness"].pop("root_workflow_id")
    elif change == "null": value["fairness"] = None
    else: value["fairness"]["subject_id"] = 9
    with pytest.raises(ValidationError):
        ProductWorkInput.model_validate(value)


def test_legacy_metadata_bytes_and_fairness_roundtrip():
    old = {"product_ref": "p", "step_ref": "s"}
    assert ProductWorkInput.model_validate(old).model_dump() == old
    assert ProductWorkInput.model_validate(metadata()).model_dump() == metadata()
