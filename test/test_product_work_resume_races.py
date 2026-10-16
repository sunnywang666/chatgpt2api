"""The completed-product resume fence shares the physical archive lock."""
import copy
import threading
from contextlib import contextmanager

import services.product_work as product_work_module
from services.account_service import account_service
from services.product_work import state_snapshot
from test.test_product_work import IDENTITY, runtime


def _complete_first_cycle(runtime):
    service, lifecycle, store, now, calls, _, add = runtime
    add("original", conversation="original-cursor")
    assert service.close(IDENTITY, "P")["state"] == "closing"
    assert lifecycle.process_one()
    assert service.get(IDENTITY, "P")["state"] == "completed"
    with store.connect() as db:
        parent = next(iter(state_snapshot(store, db).values()))
        old_child = copy.deepcopy(store.runtime(db, parent["archive_keys"][0]))
    return service, lifecycle, store, now, calls, add, old_child


def test_resume_first_makes_retained_v1_archive_child_non_dispatchable(runtime):
    service, lifecycle, store, now, calls, _, old_child = _complete_first_cycle(runtime)

    assert service.resume(IDENTITY, "P", expected_version=1)["version"] == 2
    with store.transaction() as db:
        # Model a stale scheduler wakeup for the retained V1 child. It may be
        # selected, but its parent no longer names it as the active archive.
        stale = store.runtime(db, old_child["key"])
        stale["archive"].update(status="pending", claim=None, claim_until=None, next_at=now[0])
        store.set_runtime(db, stale["key"], stale)

    assert not lifecycle.process_one(target_key=old_child["key"])
    assert calls == [("owner", "original", True)]
    with store.connect() as db:
        parent = next(iter(state_snapshot(store, db).values()))
        stale = store.runtime(db, old_child["key"])
    assert parent["state"] == "active" and parent["version"] == 2
    assert stale["archive"]["status"] == "unknown"
    assert stale["archive"]["error_code"] == "PRODUCT_WORK_ARCHIVE_CONFLICT"


def test_archive_lock_finishes_before_resume_and_old_callback_cannot_confirm_v2(runtime, monkeypatch):
    service, lifecycle, store, _, calls, _, add = runtime
    add("original", conversation="original-cursor")
    assert service.close(IDENTITY, "P")["state"] == "closing"
    with store.connect() as db:
        first_parent = next(iter(state_snapshot(store, db).values()))
        old_child = copy.deepcopy(store.runtime(db, first_parent["archive_keys"][0]))

    entered, release = threading.Event(), threading.Event()
    resume_acquiring, resumed, archive_confirmed = threading.Event(), threading.Event(), threading.Event()
    original_archive = lifecycle.text.archive_product_work
    original_release = lifecycle._release_operation
    original_lock = product_work_module.locking_product_conversations
    physical_lock = account_service.conversation_binding_lock("product-archive:account", "original-cursor")

    def hold_archive(*args):
        entered.set()
        assert release.wait(3)
        return original_archive(*args)

    @contextmanager
    def observe_physical_lock(receipts):
        if threading.current_thread().name == "product-work-resume-race":
            resume_acquiring.set()
        with original_lock(receipts):
            yield

    def record_archive_confirmation(account_key):
        try:
            return original_release(account_key)
        finally:
            archive_confirmed.set()

    lifecycle.text.archive_product_work = hold_archive
    lifecycle._release_operation = record_archive_confirmation
    monkeypatch.setattr(product_work_module, "locking_product_conversations", observe_physical_lock)
    assert lifecycle.process_one(background=True)
    assert entered.wait(3)
    assert physical_lock.locked()

    outcome = {}

    def resume():
        try:
            outcome["result"] = service.resume(IDENTITY, "P", expected_version=1)
        except BaseException as exc:  # Preserve a thread failure for the assertion below.
            outcome["error"] = exc
        finally:
            resumed.set()

    thread = threading.Thread(target=resume, name="product-work-resume-race")
    thread.start()
    assert resume_acquiring.wait(3)
    assert not resumed.is_set()

    # Resume waits on the physical conversation mutex, before opening its
    # writer transaction. The archive's held mutex therefore does not strand
    # SQLite while the resume thread is waiting.
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        db.rollback()

    release.set()
    assert archive_confirmed.wait(3)
    assert resumed.wait(3)
    thread.join(3)
    assert not thread.is_alive()
    if "error" in outcome:
        # The physical lock releases before WorkLifecycle's final SQLite
        # confirmation. A resume that reaches the database in that interval
        # must observe the still-closing V1 and retry only after confirmation.
        assert getattr(outcome["error"], "code", None) == "PRODUCT_WORK_RESUME_NOT_COMPLETED"
        outcome["result"] = service.resume(IDENTITY, "P", expected_version=1)
    assert outcome["result"]["state"] == "active"
    assert outcome["result"]["version"] == 2
    assert calls == [("owner", "original", True)]

    # V2 receives a new archive child. A delayed V1 completion callback is
    # retained as history and cannot complete the new closeout.
    add("next", step="next", conversation="original-cursor", _sequence=2)
    assert service.close(IDENTITY, "P", expected_version=2)["state"] == "closing"
    with store.transaction() as db:
        service.archive_confirmed(db, old_child)
        parent = next(iter(state_snapshot(store, db).values()))
        new_child = store.runtime(db, parent["archive_keys"][0])
    assert parent["state"] == "closing" and parent["version"] == 2
    assert new_child["key"] != old_child["key"]
    assert new_child["archive"]["status"] == "pending"
