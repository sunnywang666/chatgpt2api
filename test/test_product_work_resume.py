"""Completed product-work cycles may resume without replaying original requests."""
import copy

import pytest

from services.product_work import ProductWorkService, state_snapshot
from services.work_lifecycle import WorkLifecycleError
from test.test_product_work import IDENTITY, runtime


def complete_first_cycle(runtime):
    service, lifecycle, store, _, calls, _, add = runtime
    original = add("original", conversation="original-cursor")
    closed = service.close(IDENTITY, "P")  # Version 1 remains backward compatible.
    assert closed["state"] == "closing"
    assert lifecycle.process_one()
    completed = service.get(IDENTITY, "P")
    assert completed["state"] == "completed" and completed["results_saved"] and completed["version"] == 1
    with store.connect() as db:
        saved = store.read_receipt(db, "text", "owner", "original")
        request_hash = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?",
                                  ("owner", "original")).fetchone()[0]
        parent = next(iter(state_snapshot(store, db).values()))
        old_child = copy.deepcopy(store.runtime(db, parent["archive_keys"][0]))
    return service, lifecycle, original, saved, request_hash, parent, old_child, calls, add


def test_resume_completed_product_preserves_originals_and_recloses_latest_cursor(runtime):
    service, lifecycle, original, saved, request_hash, parent, old_child, calls, add = complete_first_cycle(runtime)
    assert len(calls) == 1

    resumed = service.resume(IDENTITY, "P", expected_version=1)
    assert resumed["state"] == "active"
    assert resumed["version"] == 2
    assert resumed["slot_held"] is True
    assert resumed["admission_control"] == {"state": "active", "version": 0, "slot_held": False}
    assert resumed["results_saved"] is False
    assert resumed["outcome"] is None
    assert resumed["archive"] == {"total": 0, "confirmed": 0, "pending": 0}
    assert len(calls) == 1  # Resume is local ownership state, never upstream I/O.

    _, _, store, _, _, _, _ = runtime
    with store.connect() as db:
        assert store.read_receipt(db, "text", "owner", "original") == saved == original
        assert db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?",
                          ("owner", "original")).fetchone()[0] == request_hash
        active = next(iter(state_snapshot(store, db).values()))
        assert active["previous_closures"][0]["outcome"] == "completed"

    # The resumed cycle can attach a later cursor in the same conversation.
    add("next", step="next", conversation="original-cursor", _sequence=2)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_VERSION_REQUIRED"):
        service.close(IDENTITY, "P")
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_VERSION_CONFLICT"):
        service.close(IDENTITY, "P", expected_version=1)
    closing = service.close(IDENTITY, "P", expected_version=2)
    assert closing["state"] == "closing" and closing["version"] == 2

    with store.transaction() as db:
        current = next(iter(state_snapshot(store, db).values()))
        assert len(current["archive_keys"]) == 1
        new_child = store.runtime(db, current["archive_keys"][0])
        assert new_child["key"] != old_child["key"]
        assert new_child["last_request_id"] == "next"
        service.archive_confirmed(db, old_child)
        after_old_callback = next(iter(state_snapshot(store, db).values()))
        assert after_old_callback["state"] == "closing" and after_old_callback["slot_held"] is True

    assert lifecycle.process_one()
    assert calls[-1] == ("owner", "next", True)
    assert service.get(IDENTITY, "P")["state"] == "completed"


def test_same_resume_version_is_idempotent_after_restart_and_completion(runtime):
    service, lifecycle, _, _, _, _, _, _, add = complete_first_cycle(runtime)
    first = service.resume(IDENTITY, "P", expected_version=1)
    add("next", step="next", conversation="original-cursor", _sequence=2)
    service.close(IDENTITY, "P", expected_version=first["version"])
    assert lifecycle.process_one()

    restarted = ProductWorkService(service.text, service.images, clock=service.clock)
    replay = restarted.resume(IDENTITY, "P", expected_version=1)
    assert replay["state"] == "completed"
    assert replay["version"] == first["version"]
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_VERSION_CONFLICT"):
        restarted.resume(IDENTITY, "P", expected_version=3)
    v3 = restarted.resume(IDENTITY, "P", expected_version=2)
    assert v3["state"] == "active" and v3["version"] == 3
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_VERSION_CONFLICT"):
        restarted.resume(IDENTITY, "P", expected_version=1)
    assert restarted.resume(IDENTITY, "P", expected_version=2) == v3


def test_resume_rejects_closing_and_persisted_incomplete_archives(runtime):
    service, _, store, _, _, _, add = runtime
    add("original")
    service.close(IDENTITY, "P")
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_RESUME_NOT_COMPLETED"):
        service.resume(IDENTITY, "P", expected_version=1)

    # A persisted completed projection may be stale while its original archive
    # child remains unconfirmed; resume must retain that closeout instead of
    # reopening the original conversation.
    with store.transaction() as db:
        parent = next(iter(state_snapshot(store, db).values()))
        parent.update(state="completed", slot_held=False, cleanup_pending=False)
        store.set_runtime(db, parent["key"], parent)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_ARCHIVE_PENDING"):
        service.resume(IDENTITY, "P", expected_version=1)


def test_resume_rejects_unresolved_original_after_completed_close(runtime):
    service, *_ = complete_first_cycle(runtime)
    store = service.store
    with store.transaction() as db:
        original = store.read_receipt(db, "text", "owner", "original")
        original.update(status="unknown", upstream_outcome="unknown", _attempt_finished_at=None)
        store.write_receipt(db, "text", "owner", "original", original)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_"):
        service.resume(IDENTITY, "P", expected_version=1)


@pytest.mark.parametrize("outcome, status, upstream_outcome", [
    ("failed", "error", "failed"),
    ("cancelled", "succeeded", "completed"),  # A sent original still requires its archive closeout.
])
def test_failed_or_cancelled_completed_products_cannot_resume(runtime, outcome, status, upstream_outcome):
    service, lifecycle, _, _, _, _, add = runtime
    add("original", status=status, upstream_outcome=upstream_outcome)
    service.close(IDENTITY, "P", outcome=outcome)
    assert lifecycle.process_one()
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_RESUME_NOT_COMPLETED"):
        service.resume(IDENTITY, "P", expected_version=1)


def test_resume_rejects_missing_saved_result_and_untrusted_or_other_identity(runtime):
    service, *_ = complete_first_cycle(runtime)
    store = service.store
    with store.transaction() as db:
        parent = next(iter(state_snapshot(store, db).values()))
        parent["results_saved"] = False
        store.set_runtime(db, parent["key"], parent)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_RESULTS_SAVE_REQUIRED"):
        service.resume(IDENTITY, "P", expected_version=1)

    for identity in ({**IDENTITY, "id": "other"}, {**IDENTITY, "_trusted_source": "internal:content"}):
        with pytest.raises(WorkLifecycleError) as error:
            service.resume(identity, "P", expected_version=1)
        assert error.value.code == "PRODUCT_WORK_NOT_FOUND"
        assert error.value.status == 404
    with pytest.raises(WorkLifecycleError) as error:
        service.resume({"id": "owner", "role": "user"}, "P", expected_version=1)
    assert error.value.code == "PRODUCT_WORK_SOURCE_REQUIRED"
    assert error.value.status == 403


@pytest.mark.parametrize("expected_version", [True, False, 1.0, "1", None])
def test_resume_rejects_non_integer_versions_with_bad_request(runtime, expected_version):
    service, *_ = complete_first_cycle(runtime)
    with pytest.raises(WorkLifecycleError) as error:
        service.resume(IDENTITY, "P", expected_version=expected_version)
    assert error.value.code == "PRODUCT_WORK_VERSION_INVALID"
    assert error.value.status == 400
