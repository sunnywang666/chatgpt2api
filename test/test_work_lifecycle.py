import json
import hashlib
import threading
from types import SimpleNamespace

import pytest

from services.task_store import TaskStore
from services.work_lifecycle import (
    WorkLifecycleError, WorkLifecycleService, ensure_work, record_slot, read_work,
)


@pytest.fixture
def runtime(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    clock = [1000.0]
    calls = []
    fail = [False]
    def archive(owner, request_id, desired):
        calls.append((owner, request_id, desired))
        if fail[0]:
            raise RuntimeError("fixture response lost")
        return {"request_id": request_id, "archived": desired,
                "conversation": {"protocol": "sequential-v1", "client_conversation_id": request_id.split("-")[0]}}
    text = SimpleNamespace(store=store, admission=None, set_public_session_archived=archive)
    service = WorkLifecycleService(text, SimpleNamespace(), clock=lambda: clock[0])
    def add(request_id, *, status="succeeded", previous=None, owner="one", work=None):
        receipt = {"request_id": request_id, "status": status, "route": "chat", "_route": "chat",
                   "client_conversation_id": "internal-" + (work or request_id.split("-")[0]),
                   "_public_session_ref": work or request_id.split("-")[0], "_source": "person:one",
                   "created_at": clock[0], "_sequence": int(request_id.split("-")[-1]),
                   "_previous_request_id": previous, "_submission_started": status != "queued",
                   "upstream_outcome": "unknown" if status == "unknown" else "completed" if status == "succeeded" else "not_sent"}
        with store.transaction() as db:
            ensure_work(store, db, "text", owner, request_id, receipt,
                        scheduling={"workflow_id": "batch", "workflow_concurrency": 1})
            db.execute("INSERT INTO requests VALUES(?,?,?,?)", (owner, request_id, "original-hash", json.dumps(receipt)))
            record_slot(store, db, "text", owner, receipt, status != "queued")
        return receipt
    return service, store, clock, calls, fail, add


def test_completion_persists_original_archive_intent_and_releases_only_its_work(runtime):
    service, store, _, calls, _, add = runtime
    a = add("A-1")
    add("B-2")
    result = service.update("text", {"id": "one"}, "A-1", "completed", True)
    assert result["state"] == "completed" and result["slot_held"] is False
    assert result["archive"]["status"] == "pending" and calls == []
    assert service.get("text", {"id": "one"}, "B-2")["slot_held"] is True
    assert service.process_one()
    assert calls == [("one", "A-1", True)]
    assert service.get("text", {"id": "one"}, "A-1")["archive"]["archived"] is True
    with store.connect() as db:
        assert store.read_receipt(db, "text", "one", "A-1") == a
    assert not service.process_one()


@pytest.mark.parametrize("background", [False, True])
def test_waiting_archive_does_not_block_another_account_or_overlap_same_account(runtime, background):
    service, store, _, _, _, add = runtime
    keys = {}
    for rid, account in (("A-1", "account-a"), ("A2-2", "account-a"), ("B-3", "account-b")):
        receipt = add(rid)
        with store.transaction() as db:
            receipt["provider_account_identity"] = account
            store.write_receipt(db, "text", "one", rid, receipt)
        keys[rid] = receipt["_work_key"]
        service.update("text", {"id": "one"}, rid, "completed", True)
    entered, release = threading.Event(), threading.Event()
    finished = threading.Event()
    def wake():
        if service.get("text", {"id": "one"}, "A-1")["archive"]["status"] == "confirmed":
            finished.set()
    service.text.admission = SimpleNamespace(wake=wake)
    calls = []
    def archive(owner, rid, desired):
        calls.append(rid)
        if rid == "A-1":
            entered.set()
            assert release.wait(3)
        return {"request_id": rid, "archived": desired,
                "conversation": {"client_conversation_id": rid.split("-")[0]}}
    service.text.set_public_session_archived = archive
    first = threading.Thread(target=lambda: service.process_one(target_key=keys["A-1"], background=background))
    first.start()
    try:
        assert entered.wait(1)
        assert not service.process_one(target_key=keys["A2-2"])
        assert service.process_one(target_key=keys["B-3"])
        assert calls == ["A-1", "B-3"]
        assert service.get("text", {"id": "one"}, "B-3")["archive"]["status"] == "confirmed"
    finally:
        release.set()
        first.join(3)
        assert finished.wait(2)
    assert not first.is_alive()
    assert service.process_one(target_key=keys["A2-2"])


def test_background_archive_does_not_occupy_original_result_recovery(runtime):
    from services.pool_admission import PoolAdmission
    from services.request_context import current_archive_guard, current_archive_read_owner
    service, store, now, calls, _, add = runtime
    add("A-1")
    service.update("text", {"id": "one"}, "A-1", "completed", True)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    service.text.admission = SimpleNamespace(wake=finished.set)
    def archive(owner, rid, desired):
        assert callable(current_archive_guard.get()) and current_archive_read_owner.get()
        current_archive_guard.get()()
        entered.set()
        assert release.wait(3)
        calls.append(rid)
        return {"request_id": rid, "archived": desired, "conversation": {"client_conversation_id": "A"}}
    service.text.set_public_session_archived = archive
    admission = SimpleNamespace(store=store, clock=lambda: now[0], recoveries={}, work_lifecycle=service)
    try:
        PoolAdmission.recover_one(admission)
        assert entered.wait(1) and not finished.is_set()
        # The recovery loop already returned while its original archive waits.
        assert service.get("text", {"id": "one"}, "A-1")["archive"]["status"] == "running"
        assert not service.process_one()
    finally:
        release.set()
        assert finished.wait(2)
    assert calls == ["A-1"]
    assert service.get("text", {"id": "one"}, "A-1")["archive"]["status"] == "confirmed"


@pytest.mark.parametrize("previous_error", [None, "UPSTREAM_READ_FAILED"])
def test_background_archive_start_failure_releases_original_claim(runtime, monkeypatch, previous_error):
    service, store, now, calls, _, add = runtime
    receipt = add("A-1")
    service.update("text", {"id": "one"}, "A-1", "completed", True)
    if previous_error:
        with store.transaction() as db:
            work = store.runtime(db, receipt["_work_key"])
            work["archive"].update(status="unknown", error_code=previous_error)
            store.set_runtime(db, work["key"], work)
    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", lambda _: (_ for _ in ()).throw(RuntimeError("worker start failed")))
        with pytest.raises(RuntimeError, match="worker start failed"):
            service.process_one(background=True)
    with store.connect() as db:
        archive = store.runtime(db, receipt["_work_key"])["archive"]
    assert archive["status"] == "pending" and archive["claim"] is None and archive["claim_until"] is None
    assert archive["error_code"] == (previous_error or "WORK_ARCHIVE_DISPATCH_FAILED") and calls == []
    now[0] += 1
    assert service.process_one()
    assert service.get("text", {"id": "one"}, "A-1")["archive"]["status"] == "confirmed"


def test_archive_workers_are_bounded_and_release_capacity(runtime):
    service, store, _, _, _, add = runtime
    release = threading.Event()
    entered = [threading.Event() for _ in range(4)]
    finished = threading.Event()
    count, lock = [0], threading.Lock()
    def wake():
        with lock:
            count[0] += 1
            if count[0] == 4:
                finished.set()
    keys = []
    for i in range(5):
        rid = f"work{i}-{i}"
        keys.append(add(rid)["_work_key"])
        service.update("text", {"id": "one"}, rid, "completed", True)
    service.text.admission = SimpleNamespace(wake=wake)
    def archive(owner, rid, desired):
        i = int(rid.split("-")[-1])
        if i < 4:
            entered[i].set()
            assert release.wait(3)
        return {"request_id": rid, "archived": desired, "conversation": {"client_conversation_id": rid.split("-")[0]}}
    service.text.set_public_session_archived = archive
    try:
        for key, event in zip(keys, entered):
            assert service.process_one(target_key=key, background=True)
            assert event.wait(1)
        assert not service.process_one(background=True)
        assert service.get("text", {"id": "one"}, "work4-4")["archive"]["status"] == "pending"
    finally:
        release.set()
        assert finished.wait(2)
    assert service.process_one()
    assert service.get("text", {"id": "one"}, "work4-4")["archive"]["status"] == "confirmed"


def test_archive_timestamps_follow_confirmation_not_patch_or_failed_attempt(runtime):
    service, _, now, _, fail, add = runtime
    add("A-1")
    completed = service.update("text", {"id": "one"}, "A-1", "completed", True)
    assert completed["results_saved_at"] == completed["archive"]["requested_at"] == 1000
    assert "confirmed_at" not in completed["archive"]
    now[0] = 1001
    assert service.update("text", {"id": "one"}, "A-1", "completed", True)["results_saved_at"] == 1000
    fail[0] = True
    assert service.process_one()
    failed = service.get("text", {"id": "one"}, "A-1")
    assert failed["archive"]["attempt_started_at"] == 1001
    assert "confirmed_at" not in failed["archive"]
    now[0] = 1010
    fail[0] = False
    assert service.process_one()
    confirmed = service.get("text", {"id": "one"}, "A-1")
    assert confirmed["archive"]["confirmed_at"] == 1010
    assert confirmed["archive"]["attempt_started_at"] == 1010
    assert confirmed["archive"]["requested_at"] == confirmed["results_saved_at"] == 1000
    now[0] = 1020
    restoring = service.update("text", {"id": "one"}, "A-1", "active")
    assert "confirmed_at" not in restoring["archive"]
    assert service.process_one()
    assert "results_saved_at" not in service.get("text", {"id": "one"}, "A-1")


def test_archive_http_steps_are_attributed_without_changing_read_pacing(runtime, monkeypatch):
    from services.account_request_pacing import AccountRequestClock
    from services.config import config
    from services.openai_backend_api import OpenAIBackendAPI
    from services.request_context import current_archive_observation, current_archive_step
    service, _, now, _, _, add = runtime
    receipt = add("A-1")
    service.update("text", {"id": "one"}, "A-1", "completed", True)
    monkeypatch.setattr("services.account_request_pacing.time.monotonic", lambda: now[0])
    monkeypatch.setattr("services.account_request_pacing.time.time", lambda: now[0])
    monkeypatch.setattr("services.account_request_pacing.time.sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(type(config), "account_request_interval_secs", property(lambda _: 1))
    monkeypatch.setattr(type(config), "account_conversation_read_interval_secs", property(lambda _: 60))
    events, reads, archived = [], [], [False]
    monkeypatch.setattr("services.account_request_pacing.logger.info", lambda event: events.append(event))
    clock = AccountRequestClock()
    def send(method, url, **kwargs):
        if method == "PATCH":
            archived[0] = kwargs["json"]["is_archived"]
        else:
            reads.append(now[0])
        now[0] += 1
        return SimpleNamespace(status_code=200, headers={}, json=lambda: {
            "current_node": "parent", "mapping": {"parent": {}}, "is_archived": archived[0]})
    backend = object.__new__(OpenAIBackendAPI)
    backend.base_url = "https://fixture.test"
    backend._headers = lambda *args: {}
    backend._get_conversation = lambda cid: clock.request(send, "GET", backend.base_url + "/conversation/" + cid, timeout=60).json()
    backend.session = SimpleNamespace(patch=lambda url, **kw: clock.request(send, "PATCH", url, **kw))
    def archive(owner, rid, desired):
        result = backend.set_conversation_archived("private-conversation", "parent", desired)
        return {**result, "request_id": rid, "conversation": {"client_conversation_id": "A"}}
    service.text.set_public_session_archived = archive
    assert service.process_one()
    attempts = [e for e in events if e.get("event") == "account_http_attempt"]
    assert [e["archive_step"] for e in attempts] == ["precheck", "patch", "readback"]
    assert [e["phase"] for e in attempts] == ["conversation_read", "conversation_archive", "conversation_read"]
    assert {e["request_ref"] for e in attempts} == {hashlib.sha256(b"one:A-1").hexdigest()[:24]}
    assert {e["work_ref"] for e in attempts} == {hashlib.sha256(receipt["_work_key"].encode()).hexdigest()[:24]}
    assert reads[1] - reads[0] >= 60
    assert "private-conversation" not in json.dumps(attempts)
    assert "one:A-1" not in json.dumps(attempts)
    assert current_archive_observation.get() is None and current_archive_step.get() is None
    assert service.get("text", {"id": "one"}, "A-1")["archive"]["confirmed_at"] >= reads[-1]


def test_read_deferral_books_oldest_archive_outside_receipt_transaction(runtime, monkeypatch):
    from services import account_request_pacing as pacing
    service, store, now, calls, _, add = runtime
    first = add("Z-1")
    service.update("text", {"id": "one"}, "Z-1", "completed", True)
    now[0] += 1
    second = add("A-2")
    service.update("text", {"id": "one"}, "A-2", "completed", True)
    with store.transaction() as db:
        for row in (first, second):
            row["provider_account_identity"] = "same-account"
            store.write_receipt(db, "text", "one", row["request_id"], row)
    service.text.admission = SimpleNamespace(accounts=SimpleNamespace(
        admission_accounts=lambda: [{"provider_account_identity": "same-account"}]))
    monkeypatch.setattr(pacing, "account_pacing_snapshot", lambda *a, **k: {"next_at": 1060, "cooldown_until": 0})
    booked = []
    def book(account, owner):
        # A second connection can take a write transaction only after the
        # scheduler committed its selection; no DB -> clock lock inversion.
        with store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.rollback()
        booked.append(owner)
    monkeypatch.setattr(pacing, "reserve_account_archive_read", book)
    assert not service.process_one()
    from services.work_lifecycle import _archive_read_owner
    with store.connect() as db:
        expected = _archive_read_owner(store.runtime(db, first["_work_key"]))
    assert booked == [expected]
    assert calls == []
    assert service.get("text", {"id": "one"}, "Z-1")["archive"]["attempts"] == 0


def test_archive_failure_and_restart_recover_original_target_without_occupying_new_work(runtime):
    service, store, clock, calls, fail, add = runtime
    add("A-1")
    service.update("text", {"id": "one"}, "A-1", "completed", True)
    fail[0] = True
    service.process_one()
    assert service.get("text", {"id": "one"}, "A-1")["archive"]["status"] == "unknown"
    add("B-2")
    with pytest.raises(WorkLifecycleError, match="WORK_ARCHIVE_PENDING"):
        service.update("text", {"id": "one"}, "A-1", "active")
    restarted = WorkLifecycleService(service.text, service.images, clock=lambda: clock[0])
    fail[0] = False
    clock[0] += 5
    assert restarted.process_one()
    assert calls == [("one", "A-1", True), ("one", "A-1", True)]
    assert restarted.get("text", {"id": "one"}, "B-2")["state"] == "active"
    assert restarted.get("text", {"id": "one"}, "B-2")["slot_held"] is True


@pytest.mark.parametrize("background", [False, True])
def test_stale_archive_cannot_patch_after_another_worker_restores(runtime, monkeypatch, background):
    from services.account_request_pacing import AccountRequestClock
    from services.config import config
    service, store, now, calls, _, add = runtime
    add("A-1")
    service.update("text", {"id": "one"}, "A-1", "completed", True)
    other = WorkLifecycleService(service.text, service.images, clock=lambda: now[0])
    # Give the replacement worker an independent transport, as in another process.
    other.text = SimpleNamespace(store=store, admission=None,
        set_public_session_archived=lambda owner, rid, desired: {
            "request_id": rid, "archived": desired, "conversation": {"client_conversation_id": "A"}})
    finished = threading.Event()
    service.text.admission = SimpleNamespace(wake=finished.set)
    monkeypatch.setattr(type(config), "account_request_interval_secs", property(lambda _: 0))
    monkeypatch.setattr(type(config), "account_conversation_read_interval_secs", property(lambda _: 0))
    clock = AccountRequestClock()
    def send(method, url, **kwargs):
        calls.append(method)
        if method == "GET":
            now[0] += 301
            assert other.process_one()
            other.update("text", {"id": "one"}, "A-1", "active")
            assert other.process_one()
        return SimpleNamespace(status_code=200, headers={})
    def stale_archive(*args):
        clock.request(send, "GET", "https://provider/conversation/original", timeout=60)
        clock.request(send, "PATCH", "https://provider/conversation/original",
                      json={"is_archived": True}, timeout=60)
        pytest.fail("stale worker reached PATCH")
    service.text.set_public_session_archived = stale_archive
    assert service.process_one(background=background)
    assert finished.wait(2)
    assert calls == ["GET"]
    current = service.get("text", {"id": "one"}, "A-1")
    assert current["state"] == "active" and current["archive"]["archived"] is False


def test_archive_guard_rechecks_after_pacing_and_bounds_each_wait(runtime, monkeypatch):
    from services.account_request_pacing import AccountRequestClock
    from services.config import config
    from services.request_context import current_archive_guard
    service, store, now, calls, _, add = runtime
    receipt = add("A-1")
    service.update("text", {"id": "one"}, "A-1", "completed", True)
    clock = AccountRequestClock()
    import time
    clock.next_request = time.monotonic() + 10
    def steal_claim(seconds):
        with store.transaction() as db:
            work = store.runtime(db, receipt["_work_key"])
            work["archive"]["claim"] = "new-worker"
            work["archive"]["claim_until"] = now[0] + 300
            store.set_runtime(db, work["key"], work)
    monkeypatch.setattr("services.account_request_pacing.time.sleep", steal_claim)
    monkeypatch.setattr(type(config), "account_request_interval_secs", property(lambda _: 0))
    def archive(*args):
        assert current_archive_guard.get() is not None
        clock.request(lambda *a, **kw: calls.append("PATCH"), "PATCH",
                      "https://provider/conversation/original", json={"is_archived": True}, timeout=60)
    service.text.set_public_session_archived = archive
    assert service.process_one() and calls == []
    assert current_archive_guard.get() is None
    with store.connect() as db:
        assert store.runtime(db, receipt["_work_key"])["archive"]["claim"] == "new-worker"


def test_archive_renews_claim_before_readback_wait(runtime, monkeypatch):
    from services.account_request_pacing import AccountRequestClock
    from services.config import config
    service, store, now, calls, _, add = runtime
    receipt = add("A-1")
    service.update("text", {"id": "one"}, "A-1", "completed", True)
    other = WorkLifecycleService(service.text, service.images, clock=lambda: now[0])
    monkeypatch.setattr("services.account_request_pacing.time.monotonic", lambda: now[0])
    monkeypatch.setattr("services.account_request_pacing.time.sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(type(config), "account_request_interval_secs", property(lambda _: 0))
    monkeypatch.setattr(type(config), "account_conversation_read_interval_secs", property(lambda _: 300))
    clock = AccountRequestClock()
    def send(method, url, **kwargs):
        calls.append(method)
        assert kwargs["timeout"] <= 60
        with store.connect() as db:
            work = store.runtime(db, receipt["_work_key"])
            assert work["archive"]["claim_until"] == now[0] + 300
        now[0] += 50
        assert not other.process_one()
        return SimpleNamespace(status_code=200, headers={})
    def archive(owner, rid, desired):
        for method in ["GET", "PATCH", "GET"]:
            clock.request(send, method, "https://provider/conversation/original",
                          json={"is_archived": True} if method == "PATCH" else None, timeout=60)
        return {"request_id": rid, "archived": desired, "conversation": {"client_conversation_id": "A"}}
    service.text.set_public_session_archived = archive
    assert service.process_one()
    assert now[0] > 1300  # The original unrenewed claim would already have expired.
    assert calls == ["GET", "PATCH", "GET"]
    assert service.get("text", {"id": "one"}, "A-1")["archive"]["status"] == "confirmed"


def test_archive_read_wait_exceeding_claim_budget_does_not_send(runtime, monkeypatch):
    from services.account_request_pacing import AccountRequestClock
    from services.config import config
    import time
    service, _, _, calls, _, add = runtime
    add("A-1")
    service.update("text", {"id": "one"}, "A-1", "completed", True)
    clock = AccountRequestClock()
    clock.next_conversation_read = time.monotonic() + 300
    monkeypatch.setattr(type(config), "account_request_interval_secs", property(lambda _: 0))
    service.text.set_public_session_archived = lambda *a: clock.request(
        lambda *a, **kw: calls.append("GET"), "GET", "https://provider/conversation/original", timeout=60)
    assert service.process_one() and calls == []
    assert service.get("text", {"id": "one"}, "A-1")["archive"]["status"] == "unknown"
    assert not clock.lock.locked()


def test_rework_requires_restore_confirmation_before_next_turn(runtime):
    service, _, _, calls, _, add = runtime
    add("A-1")
    with pytest.raises(WorkLifecycleError, match="WORK_RESULTS_SAVE_REQUIRED"):
        service.update("text", {"id": "one"}, "A-1", "completed")
    service.update("text", {"id": "one"}, "A-1", "completed", True)
    service.process_one()
    result = service.update("text", {"id": "one"}, "A-1", "active")
    assert result["state"] == "restoring" and result["slot_held"] is False
    with pytest.raises(WorkLifecycleError, match="WORK_NOT_ACTIVE"):
        add("A-2", previous="A-1")
    service.process_one()
    add("A-2", previous="A-1")
    assert calls == [("one", "A-1", True), ("one", "A-1", False)]
    with pytest.raises(WorkLifecycleError, match="WORK_SUPERSEDED"):
        service.update("text", {"id": "one"}, "A-1", "completed", True)


def test_pause_releases_work_slot_only_when_original_turn_is_safe(runtime):
    service, _, _, _, _, add = runtime
    add("queued-1", status="queued")
    assert service.update("text", {"id": "one"}, "queued-1", "paused")["state"] == "paused"
    # Resuming a paused unsent request is safe; it must not need a finished result.
    assert service.update("text", {"id": "one"}, "queued-1", "active")["state"] == "active"
    add("unknown-2", status="unknown")
    for state in ("paused", "completed"):
        with pytest.raises(WorkLifecycleError, match="WORK_TURN_UNFINISHED"):
            service.update("text", {"id": "one"}, "unknown-2", state, True)
    assert service.get("text", {"id": "one"}, "unknown-2")["slot_held"] is True


def test_other_key_and_old_completion_cannot_change_current_work(runtime):
    service, _, _, calls, _, add = runtime
    add("A-1")
    add("A-2", previous="A-1")
    with pytest.raises(WorkLifecycleError, match="WORK_REQUEST_NOT_FOUND"):
        service.get("text", {"id": "another-key"}, "A-2")
    with pytest.raises(WorkLifecycleError, match="WORK_SUPERSEDED"):
        service.update("text", {"id": "one"}, "A-1", "completed", True)
    assert calls == []


def test_single_request_protocol_work_closes_locally_without_claiming_upstream_archive(runtime):
    service, store, _, calls, _, _ = runtime
    receipt = {"request_id": "native", "client_conversation_id": "native-session", "status": "succeeded", "_route": "codex",
               "_scheduling": {"workflow_id": "batch", "workflow_concurrency": 1}}
    with store.transaction() as db:
        ensure_work(store, db, "text", "one", "native", receipt)
        db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("one", "native", "hash", json.dumps(receipt)))
    result = service.update("text", {"id": "one"}, "native", "completed", True)
    assert result["state"] == "completed" and result["archive"]["status"] == "not_applicable"
    assert result["archive"]["scope"] == "provider_work" and "archived" not in result["archive"]
    assert service.update("text", {"id": "one"}, "native", "active")["state"] == "active"
    assert not service.process_one() and calls == []


def test_legacy_archived_session_requires_explicit_restore_readback(runtime):
    service, _, _, calls, _, add = runtime
    add("legacy-1")
    assert service.get("text", {"id": "one"}, "legacy-1")["archive"]["status"] == "not_requested"
    assert service.update("text", {"id": "one"}, "legacy-1", "active")["state"] == "restoring"
    assert service.process_one()
    result = service.get("text", {"id": "one"}, "legacy-1")
    assert result["state"] == "active" and result["archive"]["archived"] is False
    assert calls == [("one", "legacy-1", False)]


def test_unsent_expired_work_closes_without_retrying_a_nonexistent_conversation(runtime):
    service, store, _, calls, _, add = runtime
    receipt = add("expired-1", status="queued")
    receipt.update(status="failed", error_code="WAIT_DEADLINE_EXCEEDED")
    with store.transaction() as db:
        store.write_receipt(db, "text", "one", "expired-1", receipt)
    result = service.update("text", {"id": "one"}, "expired-1", "completed", True)
    assert result["state"] == "completed" and result["slot_held"] is False
    assert result["archive"]["status"] == "not_applicable"
    assert result["archive"]["error_code"] == "UPSTREAM_CONVERSATION_NOT_CREATED"
    assert "archived" not in result["archive"]
    assert service.update("text", {"id": "one"}, "expired-1", "active")["state"] == "active"
    assert not service.process_one() and calls == []


def test_later_real_conversation_cannot_create_a_second_slot_via_another_key(runtime):
    _, store, _, _, _, add = runtime
    first = add("A-1")
    with store.transaction() as db:
        first.update(provider_binding_id="original-binding", conversation_id="original-upstream")
        store.write_receipt(db, "text", "one", "A-1", first)
        impostor = {"request_id": "other", "_source": "person:one", "client_conversation_id": "other-alias",
                    "provider_binding_id": "original-binding", "conversation_id": "original-upstream",
                    "_scheduling": {"workflow_id": "batch", "workflow_concurrency": 1}}
        with pytest.raises(WorkLifecycleError, match="WORK_OWNER_CONFLICT"):
            ensure_work(store, db, "text", "another-key", "other", impostor)
        assert impostor["_work_key"] == first["_work_key"]


def test_binding_alias_cannot_bypass_paused_work_and_other_kind_unknown_blocks_archive(runtime):
    service, store, _, calls, _, add = runtime
    first = add("A-1")
    with store.transaction() as db:
        first.update(provider_binding_id="binding-A", provider_account_identity="physical-account", conversation_id="physical-conversation")
        store.write_receipt(db, "text", "one", "A-1", first)
    service.update("text", {"id": "one"}, "A-1", "paused")
    with store.transaction() as db:
        alias = {**first, "provider_binding_id": "binding-B", "client_conversation_id": "another-alias", "request_id": "A-2"}
        alias.pop("_work_key")
        with pytest.raises(WorkLifecycleError, match="WORK_NOT_ACTIVE"):
            ensure_work(store, db, "text", "one", "A-2", alias)
    service.update("text", {"id": "one"}, "A-1", "active")
    with store.transaction() as db:
        store.write_receipt(db, "image", "other-key", "image-unknown", {
            "id": "image-unknown", "owner_id": "other-key", "status": "error", "upstream_unfinished": True,
            "provider_binding_id": "binding-C", "provider_account_identity": "physical-account", "conversation_id": "physical-conversation"})
    with pytest.raises(WorkLifecycleError, match="WORK_TURN_UNFINISHED"):
        service.update("text", {"id": "one"}, "A-1", "completed", True)
    assert calls == []


@pytest.mark.parametrize("kind", ["text", "image"])
@pytest.mark.parametrize("desired", [True, False])
@pytest.mark.parametrize("read_only_wait", [False, True])
def test_cooling_account_does_not_block_other_original_archive_or_restore(runtime, monkeypatch, tmp_path, kind, desired, read_only_wait):
    from services import account_request_pacing as pacing
    service, store, clock, calls, _, add = runtime
    def image_archive(identity, request_id, archived):
        calls.append((identity["id"], request_id, archived))
        return {"task_id": request_id, "archived": archived, "image_thread": {"id": request_id.split("-")[0]}}
    service.images.set_thread_archived = image_archive
    rows = [add("A-1"), add("B-2")]
    if kind == "image":
        with store.transaction() as db:
            for receipt in rows:
                request_id = receipt["request_id"]
                db.execute("DELETE FROM requests WHERE owner=? AND id=?", ("one", request_id))
                receipt.update(id=request_id, owner_id="one", status="success", _image_thread={"id": request_id.split("-")[0]})
                work = store.runtime(db, receipt["_work_key"])
                work["kind"] = "image"
                store.set_runtime(db, work["key"], work)
                store.write_receipt(db, "image", "one", request_id, receipt)
    for receipt in rows:
        service.update(kind, {"id": "one"}, receipt["request_id"], "completed", True)
    if not desired:
        assert service.process_one() and service.process_one()
        calls.clear()
        for receipt in rows:
            service.update(kind, {"id": "one"}, receipt["request_id"], "active")
    cold, ready = sorted(rows, key=lambda r: r["_work_key"])
    accounts = []
    with store.transaction() as db:
        for label, receipt in [("cold", cold), ("ready", ready)]:
            receipt["provider_account_identity"] = label
            store.write_receipt(db, kind, "one", receipt["request_id"], receipt)
            accounts.append({"provider_account_identity": label, "account_id": "workspace-" + label})
    service.text.admission = SimpleNamespace(accounts=SimpleNamespace(admission_accounts=lambda: accounts), wake=lambda: None)
    monkeypatch.setattr(pacing, "DATA_DIR", tmp_path)
    folder = tmp_path / "account_request_clocks"
    folder.mkdir()
    cold_path = folder / (hashlib.sha256(b"workspace-cold").hexdigest() + ".json")
    cold_path.write_text(json.dumps({"next_request": 999, "next_turn": 1200, "cooldown_until": 0, "next_conversation_read": 1100}
                                    if read_only_wait else {"next_request": 1100, "next_turn": 1200, "cooldown_until": 1060}))
    ready_path = folder / (hashlib.sha256(b"workspace-ready").hexdigest() + ".json")
    # A future model-message slot must not delay a GET/PATCH archive operation.
    ready_path.write_text(json.dumps({"next_request": 999, "next_turn": 2000, "cooldown_until": 0}))
    assert service.process_one()
    assert calls == [("one", ready["request_id"], desired)]
    deferred = service.get(kind, {"id": "one"}, cold["request_id"])
    assert deferred["archive"]["status"] == "pending"
    assert deferred["archive"]["attempts"] == 0
    assert deferred["archive"]["next_at"] == 1100
    assert deferred["slot_held"] is False
    restarted = WorkLifecycleService(service.text, service.images, clock=lambda: clock[0])
    assert not restarted.process_one(target_key=cold["_work_key"])
    clock[0] = 1100
    assert restarted.process_one(target_key=cold["_work_key"])
    assert calls[-1] == ("one", cold["request_id"], desired)
    result = restarted.get(kind, {"id": "one"}, cold["request_id"])
    assert result["archive"]["status"] == "confirmed" and result["archive"]["attempts"] == 1
    assert result["state"] == ("completed" if desired else "active")


def test_unreadable_account_clock_defers_archive_without_claim_or_http(runtime, monkeypatch, tmp_path):
    from services import account_request_pacing as pacing
    service, store, _, calls, _, add = runtime
    receipt = add("A-1")
    receipt["provider_account_identity"] = "original-account"
    with store.transaction() as db:
        store.write_receipt(db, "text", "one", "A-1", receipt)
    service.update("text", {"id": "one"}, "A-1", "completed", True)
    service.text.admission = SimpleNamespace(accounts=SimpleNamespace(admission_accounts=lambda: [
        {"provider_account_identity": "original-account", "account_id": "workspace"}]))
    monkeypatch.setattr(pacing, "DATA_DIR", tmp_path)
    folder = tmp_path / "account_request_clocks"
    folder.mkdir()
    (folder / (hashlib.sha256(b"workspace").hexdigest() + ".json")).write_text("broken")
    assert not service.process_one()
    assert calls == []
    result = service.get("text", {"id": "one"}, "A-1")
    assert result["archive"]["status"] == "unknown"
    assert result["archive"]["error_code"] == "ACCOUNT_PACING_UNAVAILABLE"
    assert result["archive"]["next_at"] == 1060
    assert result["archive"]["attempts"] == 0 and result["slot_held"] is False


@pytest.mark.parametrize("failure", [
    {"last_refresh_error": "token invalidated (/backend-api/me)"},
    {"last_token_refresh_error": "oauth_refresh_http_401: refresh_token_invalidated"},
    {"last_token_refresh_error": "app_session_terminated"},
])
@pytest.mark.parametrize("desired", [True, False])
def test_invalidated_authorization_preserves_intent_until_same_account_restored(runtime, monkeypatch, failure, desired):
    from services import account_request_pacing as pacing
    service, store, clock, calls, _, add = runtime
    rows = [add("A-1"), add("B-2")]
    for row in rows:
        service.update("text", {"id": "one"}, row["request_id"], "completed", True)
    if not desired:
        assert service.process_one() and service.process_one()
        calls.clear()
        for row in rows:
            service.update("text", {"id": "one"}, row["request_id"], "active")
    bad, healthy = sorted(rows, key=lambda row: row["_work_key"])
    accounts = [{"provider_account_identity": "bad", "status": "异常", **failure},
                {"provider_account_identity": "healthy", "status": "正常"}]
    with store.transaction() as db:
        for label, row in [("bad", bad), ("healthy", healthy)]:
            row["provider_account_identity"] = label
            store.write_receipt(db, "text", "one", row["request_id"], row)
    service.text.admission = SimpleNamespace(accounts=SimpleNamespace(admission_accounts=lambda: accounts), wake=lambda: None)
    monkeypatch.setattr(pacing, "account_pacing_snapshot", lambda account, now, **kwargs: {"next_at": now})
    assert service.process_one()
    assert calls == [("one", healthy["request_id"], desired)]
    deferred = service.get("text", {"id": "one"}, bad["request_id"])
    assert deferred["archive"]["status"] == "unknown"
    assert deferred["archive"]["error_code"] == "RECOVERY_AUTH_REQUIRED"
    assert deferred["archive"]["attempts"] == 0
    assert deferred["archive"]["desired"] is desired and deferred["archive"]["next_at"] == 1060
    assert deferred["slot_held"] is False
    accounts[0] = {"provider_account_identity": "bad", "status": "正常"}
    clock[0] = 1060
    restarted = WorkLifecycleService(service.text, service.images, clock=lambda: clock[0])
    assert restarted.process_one(target_key=bad["_work_key"])
    assert calls[-1] == ("one", bad["request_id"], desired)
    result = restarted.get("text", {"id": "one"}, bad["request_id"])
    assert result["archive"]["status"] == "confirmed" and result["archive"]["error_code"] is None


@pytest.mark.parametrize("account", [
    {"status": "异常", "last_refresh_error": "connection failed"},
    {"status": "限流", "last_token_refresh_error": "refresh_token_invalidated"},
    {"status": "正常", "last_refresh_error": "token invalidated"},
])
def test_other_account_states_are_not_misclassified_as_expired_authorization(runtime, monkeypatch, account):
    from services import account_request_pacing as pacing
    service, store, _, calls, _, add = runtime
    row = add("A-1")
    row["provider_account_identity"] = "original"
    with store.transaction() as db:
        store.write_receipt(db, "text", "one", row["request_id"], row)
    service.text.admission = SimpleNamespace(accounts=SimpleNamespace(admission_accounts=lambda: [
        {"provider_account_identity": "original", **account}]), wake=lambda: None)
    monkeypatch.setattr(pacing, "account_pacing_snapshot", lambda account, now, **kwargs: {"next_at": now})
    service.update("text", {"id": "one"}, row["request_id"], "completed", True)
    assert service.process_one()
    assert calls == [("one", row["request_id"], True)]
