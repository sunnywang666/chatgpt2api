"""Opt-in product parents in the existing durable PoolAdmission scheduler."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json

import pytest

from services.image_task_service import ImageTaskService
from services.product_work import ProductWorkService, record_slot
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



def text_submit(rt, request_id, *, owner="owner-a", source=SOURCE_A, product="product-a", step=None,
                conversation=None):
    _, _, _, _, text, _ = rt
    return text.submit(owner, {"client_request_id": request_id,
        "client_conversation_id": conversation or f"session-{request_id}", "model": "fixture-text",
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


def rejected_without_conversation(rt, owner, request_id):
    clock, _, store, _, _, _ = rt
    with store.transaction() as db:
        row = store.read_receipt(db, "text", owner, request_id)
        row.update(status="failed", error_code="MODEL_NOT_FOUND", upstream_outcome="rejected",
                   _submission_started=True, _executing=False, _turn_reserved=False,
                   _attempt_finished_at=clock.now, upstream_unfinished=False,
                   original_failure_phase="stream_open", original_exception_category="http",
                   original_upstream_request_stage="conversation", original_http_status=404,
                   original_upstream_error_form="detail_text",
                   original_upstream_error={"message": "Model not found"})
        store.write_receipt(db, "text", owner, request_id, row)
        return row


def update_receipt(rt, kind, owner, request_id, **changes):
    _, _, store, _, _, _ = rt
    with store.transaction() as db:
        row = store.read_receipt(db, kind, owner, request_id)
        row.update(changes)
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


def test_actual_chat_capacity_is_shared_across_sources_and_claim_is_atomic(runtime):
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


def test_admission_pause_is_cas_fenced_and_reacquires_only_via_claim(runtime):
    text_submit(runtime, "pause-me", product="paused-product")
    service = ProductWorkService(runtime[4], runtime[5], clock=runtime[0])
    identity = {"id": "owner-a", "role": "admin", "_trusted_source": SOURCE_A}

    paused = service.admission_control(identity, "paused-product", True, 0)
    assert paused["admission_control"] == {"state": "paused", "version": 1, "slot_held": False}
    assert runtime[3].claim_next() is None
    with pytest.raises(Exception, match="PRODUCT_WORK_ADMISSION_VERSION_CONFLICT"):
        service.admission_control(identity, "paused-product", True, 0)

    resumed = service.admission_control(identity, "paused-product", False, 1)
    assert resumed["admission_control"] == {"state": "active", "version": 2, "slot_held": False}
    claimed = runtime[3].claim_next()
    assert claimed and claimed.request_id == "pause-me"
    assert parent(runtime, receipt(runtime, "text", "owner-a", "pause-me"))["admission_control"]["slot_held"]


def test_pause_keeps_live_admission_until_lease_ends_then_releases(runtime):
    text_submit(runtime, "live", product="live-product")
    claimed = runtime[3].claim_next()
    assert claimed and claimed.request_id == "live"
    service = ProductWorkService(runtime[4], runtime[5], clock=runtime[0])
    identity = {"id": "owner-a", "role": "admin", "_trusted_source": SOURCE_A}
    pausing = service.admission_control(identity, "live-product", True, 0)
    assert pausing["admission_control"] == {"state": "pausing", "version": 1, "slot_held": True}
    with runtime[2].transaction() as db:
        current = runtime[2].read_receipt(db, "text", "owner-a", "live")
        from services.work_lifecycle import WorkLifecycleError
        with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_NOT_ACTIVE"):
            from services import product_work
            product_work.before_send(runtime[2], db, current)
    finish(runtime, "text", "owner-a", "live")
    # A safe GET must report the expired/finished pause as paused even before
    # a later admission transaction persists that projection.
    assert service.get(identity, "live-product")["admission_control"] == {
        "state": "paused", "version": 1, "slot_held": False}
    text_submit(runtime, "other", owner="owner-b", source=SOURCE_B, product="other-product")
    next_claim = runtime[3].claim_next()
    assert next_claim and next_claim.request_id == "other"
    assert parent(runtime, receipt(runtime, "text", "owner-a", "live"))["admission_control"] == {
        "state": "paused", "version": 1, "slot_held": False}


def test_resume_cannot_drop_a_live_pausing_lease_or_overadmit(runtime):
    text_submit(runtime, "a", product="product-a")
    claim_a = runtime[3].claim_next()
    assert claim_a and claim_a.request_id == "a"
    # Restrict the physical denominator to A's account.  A's admission marker
    # must remain counted while pause is still waiting for its lease to end.
    account = receipt(runtime, "text", "owner-a", "a")["provider_account_identity"]
    disabled = "fixture-token-1" if account == "account-0" else "fixture-token-0"
    runtime[1].update_account(disabled, {"managed_disabled": True}, quiet=True)
    text_submit(runtime, "b", owner="owner-b", source=SOURCE_B, product="product-b")
    service = ProductWorkService(runtime[4], runtime[5], clock=runtime[0])
    identity = {"id": "owner-a", "role": "admin", "_trusted_source": SOURCE_A}
    assert service.admission_control(identity, "product-a", True, 0)["admission_control"] == {
        "state": "pausing", "version": 1, "slot_held": True}
    from services.work_lifecycle import WorkLifecycleError
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_ADMISSION_PAUSING"):
        service.admission_control(identity, "product-a", False, 1)
    assert service.get(identity, "product-a")["admission_control"] == {
        "state": "pausing", "version": 1, "slot_held": True}
    assert runtime[3].resource_snapshot()["products"]["slots_held"] == 1
    assert runtime[3].claim_next() is None

    # GET projects this as paused/version 1; one matching resume CAS atomically
    # normalizes it and makes the product eligible to acquire a fresh slot.
    runtime[0].now += runtime[3].CLAIM_SECONDS + 1
    assert service.get(identity, "product-a")["admission_control"] == {
        "state": "paused", "version": 1, "slot_held": False}
    resumed = service.admission_control(identity, "product-a", False, 1)
    assert resumed["admission_control"] == {"state": "active", "version": 2, "slot_held": False}


def test_product_capacity_releases_only_after_completed_close(runtime):
    """The actual 24-turn configuration replenishes only its three closures."""
    runtime[3].settings = lambda: {"image_account_concurrency": 4, "codex_max_concurrency": 4,
                                   "chat_account_concurrency": 12}
    jobs = [(f"product-{number}", f"owner-{number}", SOURCE_A if number % 2 else SOURCE_B)
            for number in range(28)]
    for request_id, owner, source in jobs:
        text_submit(runtime, request_id, owner=owner, source=source, product=request_id + "-product")
    claimed = [runtime[3].claim_next() for _ in range(24)]
    assert len({item.request_id for item in claimed}) == 24
    assert runtime[3].claim_next() is None
    service = ProductWorkService(runtime[4], runtime[5], clock=runtime[0])
    sources = {owner: source for _request_id, owner, source in jobs}
    for item in claimed[:3]:
        finish(runtime, "text", item.owner, item.request_id)
        # This fixture models a completed local result with no physical archive
        # target; the archive-pending close release has its own regression.
        update_receipt(runtime, "text", item.owner, item.request_id, upstream_outcome="not_sent")
        closed = service.close({"id": item.owner, "role": "admin", "_trusted_source": sources[item.owner]},
                               item.request_id + "-product")
        assert closed["admission_control"]["slot_held"] is False
    replenished = [runtime[3].claim_next() for _ in range(3)]
    assert all(replenished)
    assert {item.request_id for item in replenished}.isdisjoint({item.request_id for item in claimed})
    assert runtime[3].claim_next() is None


def test_restart_retains_product_admission_between_ai_steps(runtime):
    text_submit(runtime, "first", product="same-product", step="copy")
    first = runtime[3].claim_next()
    assert first and first.request_id == "first"
    finish(runtime, "text", "owner-a", "first")
    restarted = TextTaskService(runtime[2].path, admission=runtime[3], clock=runtime[0])
    restarted.submit("owner-a", {"client_request_id": "second", "client_conversation_id": "session-second",
                     "model": "fixture-text", "messages": [{"role": "user", "content": "synthetic"}],
                     "product_work": {"product_ref": "same-product", "step_ref": "image-copy"}}, source=SOURCE_A)
    assert runtime[3].resource_snapshot()["products"]["slots_held"] == 1
    second = runtime[3].claim_next()
    assert second and second.request_id == "second"


def test_ready_continuations_stay_ahead_of_fresh_products_when_cursor_favors_new(runtime):
    # Four physical Chat turns leave fresh capacity available; priority rather
    # than a full cap must still select the two retained continuations first.
    runtime[3].settings = lambda: {"image_account_concurrency": 4, "codex_max_concurrency": 4,
                                   "chat_account_concurrency": 2}
    text_submit(runtime, "a-seed", product="product-a")
    text_submit(runtime, "b-seed", owner="owner-b", product="product-b")
    seeded = [runtime[3].claim_next(), runtime[3].claim_next()]
    assert {item.request_id for item in seeded} == {"a-seed", "b-seed"}
    for item in seeded:
        finish(runtime, "text", item.owner, item.request_id)
    text_submit(runtime, "a-next", product="product-a", step="second")
    text_submit(runtime, "b-next", owner="owner-b", product="product-b", step="second")
    text_submit(runtime, "fresh-a", owner="owner-c", product="fresh-a")
    text_submit(runtime, "fresh-b", owner="owner-d", product="fresh-b")
    with runtime[2].transaction() as db:
        runtime[2].set_runtime(db, "fairness", {
            "source": SOURCE_A, "account": "account-0", "owners": {},
            "groups": {SOURCE_A: "workflow:product-new"}, "subgroups": {},
        })
    chosen = [runtime[3].claim_next(), runtime[3].claim_next()]
    assert {item.request_id for item in chosen} == {"a-next", "b-next"}


def test_legacy_live_claim_counts_but_paused_unknown_history_does_not(runtime):
    text_submit(runtime, "old-paused", product="old-paused")
    text_submit(runtime, "old-live", owner="owner-b", source=SOURCE_B, product="old-live")
    with runtime[2].transaction() as db:
        paused = runtime[2].read_receipt(db, "text", "owner-a", "old-paused")
        paused.update(status="unknown", upstream_outcome="unknown", _recovery_paused=True)
        runtime[2].write_receipt(db, "text", "owner-a", "old-paused", paused)
        paused_parent = runtime[2].runtime(db, paused["_product_work_key"])
        paused_parent.update(state="active", slot_held=True)
        paused_parent.pop("admission_control", None)
        runtime[2].set_runtime(db, paused_parent["key"], paused_parent)
        live = runtime[2].read_receipt(db, "text", "owner-b", "old-live")
        live.update(status="running", _claim_id="legacy-live", _claim_until=runtime[0].now + 30,
                    _executing=True, _turn_reserved=True)
        runtime[2].write_receipt(db, "text", "owner-b", "old-live", live)
        live_parent = runtime[2].runtime(db, live["_product_work_key"])
        live_parent.update(state="active", slot_held=True)
        live_parent.pop("admission_control", None)
        runtime[2].set_runtime(db, live_parent["key"], live_parent)
    assert runtime[3].resource_snapshot()["products"]["slots_held"] == 1
    runtime[0].now += 31
    assert runtime[3].resource_snapshot()["products"]["slots_held"] == 0


@pytest.mark.parametrize("claim_first", [False, True])
def test_cancel_step_and_real_admission_cannot_both_win(runtime, claim_first):
    from services.work_lifecycle import WorkLifecycleError
    clock, _, store, admission, text, images = runtime
    text_submit(runtime, "withdraw")
    service = ProductWorkService(text, images, clock=clock)
    identity = {"id": "owner-a", "role": "admin", "_trusted_source": SOURCE_A}
    if claim_first:
        claim = admission.claim_next()
        assert claim and claim.request_id == "withdraw"
        original = receipt(runtime, "text", "owner-a", "withdraw")
        with pytest.raises(WorkLifecycleError, match="CANCEL_STEP_NOT_UNSENT"):
            service.cancel_step(identity, "product-a", "withdraw", 1)
        assert receipt(runtime, "text", "owner-a", "withdraw") == original
    else:
        service.cancel_step(identity, "product-a", "withdraw", 1)
        assert admission.claim_next() is None
        restarted = TextTaskService(store.path, admission=admission, clock=clock)
        result = restarted.submit("owner-a", {
            "client_request_id": "withdraw", "client_conversation_id": "session-withdraw", "model": "fixture-text",
            "messages": [{"role": "user", "content": "synthetic"}],
            "product_work": {"product_ref": "product-a", "step_ref": "withdraw"}}, source=SOURCE_A)
        assert result["status"] == "cancelled"
        assert restarted.read_snapshot("owner-a", "withdraw")["cancelled_unsent"]
        assert admission.claim_next() is None


def test_rejected_parent_releases_one_slot_without_rewriting_receipt_and_recovery_reclaims(runtime):
    _, accounts, _, admission, text, images = runtime
    accounts.update_account("fixture-token-1", {"managed_disabled": True}, quiet=True)
    text_submit(runtime, "rejected", product="rejected-product")
    claimed = admission.claim_next()
    assert claimed and claimed.request_id == "rejected"
    original = rejected_without_conversation(runtime, "owner-a", "rejected")
    original_parent = parent(runtime, original)
    assert original_parent["state"] == "active" and original_parent["slot_held"] is True

    text_submit(runtime, "waiting", owner="owner-b", source=SOURCE_B, product="waiting-product")
    # GET is a projection only: it reports retained parent identity without
    # releasing the historical rejection outside the normal claim transaction.
    snapshot = admission.resource_snapshot()
    assert snapshot["products"] == {"slots_total": 1, "slots_held": 1, "slots_free": 0,
                                    "lifecycle_slots_held": 1, "admission_limited": True,
                                    "queued_products": 1, "active_products": 1}
    assert parent(runtime, original) == original_parent

    next_claim = admission.claim_next()
    assert next_claim and next_claim.request_id == "waiting"
    assert receipt(runtime, "text", "owner-a", "rejected") == original
    released_parent = parent(runtime, original)
    assert released_parent["state"] == "queued" and released_parent["slot_held"] is False

    work = ProductWorkService(text, images, clock=lambda: runtime[0].now)
    identity = {"id": "owner-a", "role": "admin", "_trusted_source": SOURCE_A}
    assert work.adopt(identity, "rejected-product", [
        {"kind": "text", "request_id": "rejected", "step_ref": "rejected"},
    ])["slot_held"] is False

    text_submit(runtime, "recovery", product="rejected-product", step="recovery")
    assert work.adopt(identity, "rejected-product", [
        {"kind": "text", "request_id": "rejected", "step_ref": "rejected"},
        {"kind": "text", "request_id": "recovery", "step_ref": "recovery"},
    ])["slot_held"] is False
    assert admission.claim_next() is None  # The only usable Chat account is occupied by "waiting".
    assert parent(runtime, receipt(runtime, "text", "owner-a", "recovery"))["slot_held"] is False

    accounts.update_account("fixture-token-1", {"managed_disabled": False, "status": "正常"}, quiet=True)
    recovery = admission.claim_next()
    assert recovery and recovery.request_id == "recovery"
    recovered_parent = parent(runtime, receipt(runtime, "text", "owner-a", "recovery"))
    assert recovered_parent["state"] == "active" and recovered_parent["slot_held"] is True


@pytest.mark.parametrize("protected", [
    "unknown", "physical", "success", "active_lease", "recovery_claim", "legacy_recovery_claim",
    "unproven_rejection", "queued_mixed",
])
def test_rejected_parent_never_releases_when_a_member_or_lease_is_protected(runtime, protected):
    text_submit(runtime, "rejected", product="protected-product")
    claimed = runtime[3].claim_next()
    assert claimed and claimed.request_id == "rejected"
    rejected_without_conversation(runtime, "owner-a", "rejected")

    if protected == "unknown":
        update_receipt(runtime, "text", "owner-a", "rejected", status="unknown", upstream_outcome="unknown")
    elif protected == "physical":
        update_receipt(runtime, "text", "owner-a", "rejected", conversation_id="physical-conversation",
                       provider_account_identity="account-0")
    elif protected == "success":
        update_receipt(runtime, "text", "owner-a", "rejected", status="succeeded", upstream_outcome="completed",
                       content="saved original result")
    elif protected == "active_lease":
        update_receipt(runtime, "text", "owner-a", "rejected", _executing=True, _turn_reserved=True)
    elif protected == "recovery_claim":
        update_receipt(runtime, "text", "owner-a", "rejected", recovery_claim_id="active-recovery")
    elif protected == "legacy_recovery_claim":
        update_receipt(runtime, "text", "owner-a", "rejected", _recovery_claim_id="active-recovery")
    elif protected == "unproven_rejection":
        update_receipt(runtime, "text", "owner-a", "rejected", original_upstream_error=None)
    else:
        text_submit(runtime, "queued-child", product="protected-product", step="recovery")
        update_receipt(runtime, "text", "owner-a", "queued-child", _ready_at=runtime[0].now + 60)

    runtime[3].claim_next()
    held = parent(runtime, receipt(runtime, "text", "owner-a", "rejected"))
    assert held["state"] == "active" and held["slot_held"] is True


def test_rejected_parent_never_releases_when_receipt_membership_is_not_bidirectional(runtime):
    text_submit(runtime, "rejected", product="protected-product")
    claimed = runtime[3].claim_next()
    assert claimed and claimed.request_id == "rejected"
    original = rejected_without_conversation(runtime, "owner-a", "rejected")
    text_submit(runtime, "unlisted", product="other-product")
    update_receipt(runtime, "text", "owner-a", "unlisted", _product_work_key=original["_product_work_key"],
                   _ready_at=runtime[0].now + 60)

    assert runtime[3].claim_next() is None
    held = parent(runtime, original)
    assert held["state"] == "active" and held["slot_held"] is True


def test_capacity_shrink_blocks_new_work_only_while_actual_chat_is_occupied(runtime):
    text_submit(runtime, "a-first", owner="owner-a", source=SOURCE_A, product="a")
    text_submit(runtime, "b-first", owner="owner-b", source=SOURCE_B, product="b")
    admission = runtime[3]
    first, second = admission.claim_next(), admission.claim_next()
    assert {first.request_id, second.request_id} == {"a-first", "b-first"}
    finish(runtime, "text", "owner-a", "a-first")
    finish(runtime, "text", "owner-b", "b-first")
    text_submit(runtime, "a-next", owner="owner-a", source=SOURCE_A, product="a")
    text_submit(runtime, "new", owner="owner-c", source=SOURCE_B, product="new")
    update_receipt(runtime, "text", "owner-c", "new", _ready_at=runtime[0].now + 60)
    # One eligible Chat account remains. Its active continuation occupies that
    # actual account capacity, independent of the two retained product parents.
    first_account = receipt(runtime, "text", "owner-a", "a-first")["provider_account_identity"]
    disabled_token = "fixture-token-1" if first_account == "account-0" else "fixture-token-0"
    runtime[1].update_account(disabled_token, {"managed_disabled": True}, quiet=True)
    continuation = admission.claim_next()
    assert continuation and continuation.request_id == "a-next"
    finish(runtime, "text", "owner-a", "a-next")
    assert receipt(runtime, "text", "owner-c", "new")["status"] == "queued"
    assert admission.claim_next() is None

    # UNKNOWN retains both original identity and its actual account occupancy;
    # removing a product-count cap must not make that account reusable.
    with runtime[2].transaction() as db:
        first_row = runtime[2].read_receipt(db, "text", "owner-a", "a-first")
        first_row.update(status="unknown", upstream_outcome="unknown", _attempt_finished_at=None)
        runtime[2].write_receipt(db, "text", "owner-a", "a-first", first_row)
    update_receipt(runtime, "text", "owner-c", "new", _ready_at=runtime[0].now)
    assert admission.claim_next() is None


def test_legacy_retained_product_history_does_not_block_a_new_product(runtime):
    """Only a live pre-marker claim counts; retained legacy history does not."""
    admission = runtime[3]
    for number in range(25):
        request_id = f"retained-{number}"
        text_submit(runtime, request_id, product=f"retained-product-{number}")
        held_parent(runtime, "text", "owner-a", request_id)
        finish(runtime, "text", "owner-a", request_id)
        with runtime[2].transaction() as db:
            row = runtime[2].read_receipt(db, "text", "owner-a", request_id)
            saved = runtime[2].runtime(db, row["_product_work_key"])
            saved.pop("admission_control", None)  # pre-marker durable parent
            runtime[2].set_runtime(db, saved["key"], saved)

    snapshot = admission.resource_snapshot()
    assert snapshot["products"]["slots_held"] == 0
    assert snapshot["products"]["lifecycle_slots_held"] == 25

    text_submit(runtime, "new-product", owner="owner-b", source=SOURCE_B, product="new-product")
    claimed = admission.claim_next()
    assert claimed and claimed.request_id == "new-product"


def test_real_chat_account_capacity_still_queues_a_new_product(runtime):
    admission = runtime[3]
    text_submit(runtime, "first", product="first-product")
    text_submit(runtime, "second", owner="owner-b", source=SOURCE_B, product="second-product")
    text_submit(runtime, "waiting", owner="owner-c", source=SOURCE_A, product="waiting-product")

    first = admission.claim_next()
    second = admission.claim_next()
    assert {first.request_id, second.request_id} == {"first", "second"}
    assert admission.claim_next() is None
    assert receipt(runtime, "text", "owner-c", "waiting")["status"] == "queued"


def test_chat_cooldown_still_queues_a_new_product(runtime):
    _, _, _, admission, _, _ = runtime
    admission._pacing = lambda _account, now: {"next_at": now + 60, "cooldown_until": now + 60}
    text_submit(runtime, "cooling", product="cooling-product")

    assert admission.claim_next() is None
    assert receipt(runtime, "text", "owner-a", "cooling")["status"] == "queued"


def test_same_product_conversation_members_cannot_start_concurrently(runtime):
    admission = runtime[3]
    text_submit(runtime, "first", product="one-product", step="first", conversation="same-conversation")
    text_submit(runtime, "duplicate", product="one-product", step="duplicate", conversation="same-conversation")

    first = admission.claim_next()
    assert first and first.request_id == "first"
    assert admission.claim_next() is None
    finish(runtime, "text", "owner-a", "first")
    second = admission.claim_next()
    assert second and second.request_id == "duplicate"


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


def test_rejected_chat_successor_reclaims_released_product_slot(runtime):
    """The three-field Chat retry is a new product admission after release."""
    clock, _, store, admission, text, _ = runtime
    text_submit(runtime, "rejected", product="rejected-product")
    claimed = admission.claim_next()
    assert claimed and claimed.request_id == "rejected"
    with store.transaction() as db:
        original = store.read_receipt(db, "text", "owner-a", "rejected")
        retained = text.store.load_input(original["_input_ref"])
        original.update(status="failed", error_code="MODEL_NOT_FOUND", upstream_outcome="rejected",
                        _submission_started=True, _executing=False, _turn_reserved=False,
                        provider_binding_id="binding-0", provider_account_identity="account-0",
                        conversation_id=None, parent_message_id=None,
                        original_failure_phase="stream_open", original_exception_category="http",
                        original_upstream_request_stage="conversation", original_http_status=404,
                        original_upstream_error_form="detail_text",
                        original_upstream_error={"message": "Model not found"})
        original["_input_ref"] = text.store.save_input(retained)
        store.write_receipt(db, "text", "owner-a", "rejected", original)

    text_submit(runtime, "waiting", owner="owner-b", source=SOURCE_B, product="waiting-product")
    waiting = admission.claim_next()
    assert waiting and waiting.request_id == "waiting"
    released = parent(runtime, receipt(runtime, "text", "owner-a", "rejected"))
    assert released["state"] == "queued" and released["slot_held"] is False

    successor = text.submit("owner-a", {
        "client_request_id": "chat-successor", "supersedes_request_id": "rejected",
        "client_conversation_id": "session-rejected",
    }, source=SOURCE_A)
    assert successor["product_work"]["product_ref"] == "rejected-product"
    work = ProductWorkService(text, runtime[5], clock=lambda: clock.now)
    parent_after_adopt = work.adopt({"id": "owner-a", "role": "admin", "_trusted_source": SOURCE_A},
        "rejected-product", [
            {"kind": "text", "request_id": "rejected", "step_ref": "rejected"},
            {"kind": "text", "request_id": "chat-successor", "step_ref": "rejected"},
        ])
    assert parent_after_adopt["slot_held"] is False
    finish(runtime, "text", "owner-b", "waiting")
    retried = admission.claim_next()
    assert retried and retried.request_id == "chat-successor"
    reclaimed = parent(runtime, receipt(runtime, "text", "owner-a", "chat-successor"))
    assert reclaimed["state"] == "active" and reclaimed["slot_held"] is True
