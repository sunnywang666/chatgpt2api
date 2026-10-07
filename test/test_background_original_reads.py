"""Ordinary background recovery with isolated storage and controlled I/O."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from services.text_task_service import TextTaskService
from test.test_pool_admission import build, Clock
from test.test_text_task_service import QueuedExecutor


@pytest.fixture
def recovery(tmp_path):
    accounts = [{"provider_account_identity": f"account-{i}", "status": "正常",
                 "access_token": f"fixture-token-{i}", "account_id": f"upstream-{i}"}
                for i in range(2)]
    (tmp_path / "accounts.json").write_text(json.dumps(accounts))
    clock = Clock()
    _, store, admission = build(tmp_path, clock)
    queue = QueuedExecutor()
    text = TextTaskService(store.path, executor=queue, clock=clock)
    release = threading.Event()
    workers = []

    def original(rid, account=0, conversation=None, **changes):
        text.submit("owner", {"client_request_id": rid, "client_conversation_id": rid,
                              "messages": [{"role": "user", "content": "original input"}]})
        text._update("owner", rid, status="unknown", provider_account_identity=f"account-{account}",
                     provider_binding_id=f"binding-{account}", conversation_id=conversation or rid,
                     recovery_next_at=0, **changes)
        with store.connect() as db:
            return store.read_receipt(db, "text", "owner", rid)

    # Capture only threads started by this dispatcher so cleanup never leaves
    # an I/O callback using the temporary store after the test ends.
    real_thread = threading.Thread
    def thread(*args, **kwargs):
        worker = real_thread(*args, **kwargs)
        workers.append(worker)
        return worker

    with patch("services.pool_admission.threading.Thread", side_effect=thread):
        yield SimpleNamespace(admission=admission, text=text, store=store, clock=clock,
                              original=original, release=release, queue=queue)
        release.set()
        for worker in workers:
            if worker.ident is not None:
                worker.join(2)
                assert not worker.is_alive()


def test_slow_original_does_not_block_other_account_or_independent_conversation(recovery):
    r = recovery
    for rid, account, chat in [("slow", 0, "same"), ("duplicate-chat", 0, "same"),
                                ("independent", 0, "other"), ("other-account", 1, "same")]:
        r.original(rid, account, chat)
    started = {rid: threading.Event() for rid in ("slow", "duplicate-chat", "independent", "other-account")}
    calls = []
    def read(owner, rid):
        calls.append(rid)
        started[rid].set()
        assert r.release.wait(3)
    r.admission.recoveries["text"] = read
    r.admission.recover_one(background=True)
    for rid in ("slow", "independent", "other-account"):
        assert started[rid].wait(1)
    r.admission.recover_one(background=True)
    assert sorted(calls) == ["independent", "other-account", "slow"]
    assert not started["duplicate-chat"].is_set()
    assert len(r.queue.calls) == 4  # No generation executed or resubmitted.


def test_cooldown_disabled_and_paused_originals_do_not_take_workers(recovery):
    r = recovery
    r.original("cooldown")
    r.original("ready", 1)
    r.original("paused", 1, _recovery_paused=True)
    r.original("suppressed", 1, _recovery_suppressed=True)
    r.admission.pacing = lambda a, now: {"next_at": now + 60 if a["provider_account_identity"] == "account-0" else now}
    ready = threading.Event()
    calls = []
    def read(owner, rid):
        calls.append(rid)
        ready.set()
        r.release.wait(3)
    r.admission.recoveries["text"] = read
    r.admission.recover_one(background=True)
    assert ready.wait(1)
    assert calls == ["ready"]
    receipt = r.original("disabled")
    with patch.object(r.admission, "_rows", return_value=[{"provider_account_identity": "account-0", "managed_disabled": True}]):
        assert not r.admission._dispatch_original_read("text", "owner", "disabled", read, receipt)


def test_unlocated_same_client_conversation_does_not_fill_all_workers(recovery):
    r = recovery
    for i in range(4):
        rid = f"unlocated-{i}"
        r.original(rid)
        r.text._update("owner", rid, conversation_id=None, client_conversation_id="same-client")
    r.original("other-account", 1)
    calls = []
    other_started = threading.Event()
    def read(owner, rid):
        calls.append(rid)
        if rid == "other-account":
            other_started.set()
        r.release.wait(3)
    r.admission.recoveries["text"] = read
    r.admission.recover_one(background=True)
    assert other_started.wait(1)
    assert sorted(calls) == ["other-account", "unlocated-0"]


@pytest.mark.parametrize("completion_first", [False, True])
def test_blocked_image_recoveries_overlap_and_leave_capacity_for_text(recovery, completion_first):
    r = recovery
    with r.store.transaction() as db:
        for i in range(2):
            rid = f"image-{i}"
            r.store.write_receipt(db, "image", "owner", rid, {
                "id": rid, "owner_id": "owner", "status": "error", "error_code": "CONVERSATION_OUTCOME_UNKNOWN",
                "provider_account_identity": "account-0", "conversation_id": rid,
                "request_message_id": f"message-{i}", "next_poll_at": 0})
    r.original("text")
    text_started = threading.Event()
    image_started = [threading.Event(), threading.Event()]
    image_calls = []
    def image(owner, rid):
        image_calls.append(rid)
        image_started[int(rid.rsplit("-", 1)[1])].set()
        assert r.release.wait(3)  # The configured handler owns the actual I/O.
    if completion_first:
        with r.store.connect() as db:
            receipt = r.store.read_receipt(db, "image", "owner", "image-0")
        r.admission.generation_completion = SimpleNamespace(process_one=lambda *, dispatch:
            dispatch("image", "owner", "image-0", image, receipt))
    r.admission.recoveries.update(image=image, text=lambda owner, rid: text_started.set())
    r.admission.recover_one(background=True)
    assert all(event.wait(1) for event in image_started) and text_started.wait(1)
    r.admission.recover_one(background=True)
    assert sorted(image_calls) == ["image-0", "image-1"]
    assert all(("image", "owner", rid) in r.admission._original_reads.values()
               for rid in ("image-0", "image-1"))


def test_worker_bound_and_start_failure_release_permit(recovery):
    r = recovery
    callbacks = []
    def read(owner, rid):
        callbacks.append(rid)
        r.release.wait(3)
    receipts = [r.original(str(i)) for i in range(5)]
    with patch("services.pool_admission.threading.Thread", side_effect=RuntimeError("fixture start failure")):
        with pytest.raises(RuntimeError):
            r.admission._dispatch_original_read("text", "owner", "0", read, receipts[0])
    assert not r.admission._original_reads
    for i in range(4):
        assert r.admission._dispatch_original_read("text", "owner", str(i), read, receipts[i])
    assert not r.admission._dispatch_original_read("text", "owner", "4", read, receipts[4])
    assert len(r.admission._original_reads) == 4


def test_due_saved_image_joins_fifo_before_read_edge_without_occupying_worker(recovery, tmp_path):
    """A failed download must not race every already-booked result/archive."""
    import hashlib
    import services.account_request_pacing as pacing
    from services.request_context import executing

    r = recovery
    r.admission.pacing = None
    receipt = {"id": "saved-image", "owner_id": "owner", "status": "error",
               "error_code": "CONVERSATION_OUTCOME_UNKNOWN", "provider_account_identity": "account-0",
               "conversation_id": "original", "request_message_id": "original-message",
               "result_file_ids": ["original-file"], "upstream_outcome": "generated",
               "upstream_unfinished": False, "next_poll_at": 999}
    key = hashlib.sha256(b"upstream-0").hexdigest()
    path = tmp_path / "account_request_clocks" / (key + ".json")
    finished = threading.Event()
    calls = []
    errors = []

    class Context:
        kind, owner, request_id = "image", "owner", "saved-image"

    def read(owner, rid):
        try:
            with executing(Context()):
                pacing.AccountRequestClock(key[:12], path).request(
                    lambda method, url, **kw: calls.append((method, r.clock())) or
                        SimpleNamespace(status_code=200, headers={}),
                    "GET", "https://fixture.invalid/conversation/original")
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    with patch.object(pacing, "DATA_DIR", tmp_path), patch.object(pacing, "_clocks", {}), \
         patch.object(pacing, "config", SimpleNamespace(account_request_interval_secs=0,
                                                        account_conversation_read_interval_secs=52.5)), \
         patch.object(pacing.time, "time", side_effect=r.clock), \
         patch.object(pacing.time, "monotonic", side_effect=r.clock):
        clock = pacing.AccountRequestClock(key[:12], path)
        clock.next_conversation_read = 1052.5
        clock.last_read_was_archive = True
        clock._save()
        assert not r.admission._dispatch_original_read("image", "owner", "saved-image", read, receipt)
        queued = pacing.AccountRequestClock(key[:12], path)
        assert len(queued.ordinary_read_queue) == 1, "due recovery never joined the read FIFO"
        first_owner = queued.ordinary_read_queue[0]["owner"]
        assert not r.admission._original_reads and not calls
        with queued.lock:
            assert not queued._ordinary_read_turn("later-result", r.clock(), 52.5)
            assert not queued._reserve_archive_read("later-archive", r.clock())
        # Repeated scheduler scans renew the same entry, never spend read credit.
        r.clock.now = 1030
        for _ in range(4):
            assert r.admission._original_read_slots.acquire(blocking=False)
        try:
            with patch.object(r.admission, "_account_guard", side_effect=AssertionError("full workers need no account lock")):
                for _ in range(3):
                    assert not r.admission._dispatch_original_read("image", "owner", "saved-image", read, receipt)
            queued = pacing.AccountRequestClock(key[:12], path)
            assert [v["owner"] for v in queued.ordinary_read_queue] == [first_owner, "later-result"]
        finally:
            for _ in range(4):
                r.admission._original_read_slots.release()
        assert not r.admission._dispatch_original_read("image", "owner", "saved-image", read, receipt)
        queued = pacing.AccountRequestClock(key[:12], path)
        assert [v["owner"] for v in queued.ordinary_read_queue] == [first_owner, "later-result"]
        assert queued.next_conversation_read == 1052.5
        r.clock.now = 1052.5
        assert r.admission._dispatch_original_read("image", "owner", "saved-image", read, receipt)
        assert finished.wait(2) and not errors
        assert calls == [("GET", 1052.5)]
        queued = pacing.AccountRequestClock(key[:12], path)
        assert [v["owner"] for v in queued.ordinary_read_queue] == ["later-result"]
        assert queued.next_conversation_read >= 1105
        assert receipt["upstream_outcome"] == "generated" and receipt["result_file_ids"] == ["original-file"]


@pytest.mark.parametrize("limit", ["global", "conversation"])
def test_image_recovery_fifo_keeps_cooldown_owner_isolation_and_dead_lease_expiry(tmp_path, limit):
    import hashlib
    import services.account_request_pacing as pacing

    now = [1000.]
    account = {"account_id": "upstream"}
    key = hashlib.sha256(b"upstream").hexdigest()
    path = tmp_path / "account_request_clocks" / (key + ".json")
    with patch.object(pacing, "DATA_DIR", tmp_path), patch.object(pacing, "_clocks", {}), \
         patch.object(pacing, "config", SimpleNamespace(account_request_interval_secs=0,
                                                        account_conversation_read_interval_secs=52.5)), \
         patch.object(pacing.time, "time", side_effect=lambda: now[0]), \
         patch.object(pacing.time, "monotonic", side_effect=lambda: now[0]):
        clock = pacing.AccountRequestClock(key[:12], path)
        clock.next_conversation_read = 1052.5
        if limit == "global":
            clock.cooldown_until = 1120
        else:
            clock.conversation_read_rate_failures = 1
            clock.last_conversation_read_rate_limit = 1000
        clock._save()
        assert not pacing.reserve_account_image_recovery_read(account, "a", "same-id")
        assert pacing.AccountRequestClock(key[:12], path).ordinary_read_queue == []
        # Passing the rate edge does not pass a later provider cooldown.
        now[0] = 1052.5
        assert not pacing.reserve_account_image_recovery_read(account, "a", "same-id")
        now[0] = 1200
        assert pacing.reserve_account_image_recovery_read(account, "a", "same-id")
        pacing._clocks.clear()  # Same persisted lease across a fresh service.
        assert not pacing.reserve_account_image_recovery_read(account, "b", "same-id")
        assert not pacing.reserve_account_image_recovery_read(account, "a", "different-id")
        queued = pacing.AccountRequestClock(key[:12], path)
        owners = [v["owner"] for v in queued.ordinary_read_queue]
        assert len(set(owners)) == 3
        assert queued.next_conversation_read == 1052.5  # Booking is not a GET.
        # A stopped task no longer renews. Its old place must expire instead
        # of blocking other accounts/tasks indefinitely or clearing evidence.
        now[0] = 1231
        assert pacing.reserve_account_image_recovery_read(account, "b", "same-id")
        assert [v["owner"] for v in pacing.AccountRequestClock(key[:12], path).ordinary_read_queue] == [owners[1]]


@pytest.mark.parametrize("blocked", ["same_conversation", "start_failure", "paused", "suppressed", "new_cooldown"])
def test_image_recovery_cancels_undispatched_fifo_place(recovery, tmp_path, blocked):
    import hashlib
    import services.account_request_pacing as pacing
    from services.pool_admission import physical_conversation_key

    r = recovery
    r.admission.pacing = None
    account = {"account_id": "upstream-0"}
    key = hashlib.sha256(b"upstream-0").hexdigest()
    path = tmp_path / "account_request_clocks" / (key + ".json")
    receipt = {"status": "error", "error_code": "CONVERSATION_OUTCOME_UNKNOWN",
               "provider_account_identity": "account-0", "conversation_id": "same",
               "request_message_id": "original-message", "result_file_ids": ["original-file"], "next_poll_at": 0}
    read = Mock(side_effect=AssertionError("blocked recovery must not start"))
    with patch.object(pacing, "DATA_DIR", tmp_path), patch.object(pacing, "_clocks", {}), \
         patch.object(pacing, "config", SimpleNamespace(account_request_interval_secs=0,
                                                        account_conversation_read_interval_secs=52.5)), \
         patch.object(pacing.time, "time", side_effect=r.clock), \
         patch.object(pacing.time, "monotonic", side_effect=r.clock):
        assert pacing.reserve_account_image_recovery_read(account, "owner", "image")
        assert len(pacing.AccountRequestClock(key[:12], path).ordinary_read_queue) == 1
        if blocked == "same_conversation":
            r.admission._original_reads[physical_conversation_key(receipt)] = ("image", "owner", "other-original")
        elif blocked == "paused":
            receipt["_recovery_paused"] = True
        elif blocked == "suppressed":
            receipt["_recovery_suppressed"] = True
        elif blocked == "new_cooldown":
            clock = pacing.AccountRequestClock(key[:12], path)
            clock.cooldown_until = 1120
            clock._save()
        if blocked == "start_failure":
            with patch("services.pool_admission.threading.Thread", side_effect=RuntimeError("fixture start")), \
                 pytest.raises(RuntimeError, match="fixture start"):
                r.admission._dispatch_original_read("image", "owner", "image", read, receipt)
        else:
            assert not r.admission._dispatch_original_read("image", "owner", "image", read, receipt)
        read.assert_not_called()
        clock = pacing.AccountRequestClock(key[:12], path)
        assert clock.ordinary_read_queue == []
        if blocked == "new_cooldown":
            assert clock.cooldown_until == 1120
        r.admission._original_reads.clear()
        assert r.admission._original_read_slots.acquire(blocking=False)
        r.admission._original_read_slots.release()


def test_zero_read_interval_keeps_http_wait_out_of_recovery_workers(recovery, tmp_path):
    import hashlib
    import services.account_request_pacing as pacing
    r = recovery
    r.admission.pacing = None
    key = hashlib.sha256(b"upstream-0").hexdigest()
    path = tmp_path / "account_request_clocks" / (key + ".json")
    receipt = {"status": "error", "error_code": "CONVERSATION_OUTCOME_UNKNOWN",
               "provider_account_identity": "account-0", "conversation_id": "original",
               "request_message_id": "original-message", "result_file_ids": ["original-file"], "next_poll_at": 0}
    with patch.object(pacing, "DATA_DIR", tmp_path), patch.object(pacing, "_clocks", {}), \
         patch.object(pacing, "config", SimpleNamespace(account_request_interval_secs=5,
                                                        account_conversation_read_interval_secs=0)), \
         patch.object(pacing.time, "time", side_effect=r.clock), \
         patch.object(pacing.time, "monotonic", side_effect=r.clock):
        clock = pacing.AccountRequestClock(key[:12], path)
        clock.next_request = 1005
        clock._save()
        read = Mock(side_effect=AssertionError("HTTP interval has not elapsed"))
        assert not r.admission._dispatch_original_read("image", "owner", "image", read, receipt)
        assert not r.admission._original_reads
        assert pacing.AccountRequestClock(key[:12], path).ordinary_read_queue == []
        read.assert_not_called()


def test_archive_dispatch_and_other_original_continue_during_completion_read(recovery):
    r = recovery
    receipt = r.original("completion")
    r.original("ordinary", 1)
    started = {name: threading.Event() for name in ("completion", "ordinary")}
    def read(owner, rid):
        started[rid].set()
        r.release.wait(3)
    r.admission.generation_completion = SimpleNamespace(process_one=lambda *, dispatch:
        dispatch("text", "owner", "completion", read, receipt))
    r.admission.work_lifecycle = SimpleNamespace(process_one=Mock())
    r.admission.recoveries["text"] = read
    r.admission.recover_one(background=True)
    assert started["completion"].wait(1)
    assert started["ordinary"].wait(1)
    assert r.admission.work_lifecycle.process_one.call_count == 4
    assert all(call.kwargs == {"background": True} for call in r.admission.work_lifecycle.process_one.call_args_list)


def test_slow_read_renews_same_durable_claim_and_preserves_pause(recovery):
    r = recovery
    r.original("slow")
    started, renewed = threading.Event(), threading.Event()
    calls = []
    def read(receipt):
        calls.append(receipt["recovery_claim_id"])
        started.set()
        assert r.release.wait(3)
        return {"status": "succeeded", "content": "original saved result", "conversation_id": "slow",
                "parent_message_id": "answer", "binding_status": "bound"}
    r.text.recovery_reader = read
    r.text.RECOVERY_LEASE_SECONDS = .15
    update = r.text._update_recovery_claim
    def observe(*args, **kwargs):
        result = update(*args, **kwargs)
        renewed.set()
        return result
    restarted = TextTaskService(r.store.path, executor=r.queue, clock=r.clock,
                                recovery_reader=Mock(side_effect=AssertionError("duplicate upstream read")))
    with patch.object(r.text, "_update_recovery_claim", side_effect=observe), ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(r.text.read, "owner", "slow")
        try:
            assert started.wait(1)
            r.clock.now += 1  # Beyond the original lease; live worker renews it.
            assert renewed.wait(1)
            assert restarted.read("owner", "slow")["status"] == "unknown"
            restarted.recovery_reader.assert_not_called()
            r.text._update("owner", "slow", _recovery_paused=True, _recovery_suppressed=True)
            renewed.clear()
            assert renewed.wait(1)
            with r.store.connect() as db:
                current = r.store.read_receipt(db, "text", "owner", "slow")
            assert current["recovery_claim_id"] == calls[0]
            assert current["_recovery_paused"] and current["_recovery_suppressed"]
            r.release.set()
            assert pending.result(2)["content"] == "original saved result"
        finally:
            r.release.set()
    assert len(calls) == 1
    assert len(r.queue.calls) == 1


def test_lease_thread_start_failure_finishes_claim_without_upstream_read(recovery):
    r = recovery
    r.original("start-failure")
    r.text.recovery_reader = Mock(side_effect=AssertionError("must not call upstream"))
    failed_thread = Mock()
    failed_thread.start.side_effect = RuntimeError("fixture thread failure")
    with patch("services.text_task_service.threading.Thread", return_value=failed_thread):
        result = r.text.read("owner", "start-failure")
    assert result["status"] == "unknown"
    with r.store.connect() as db:
        current = r.store.read_receipt(db, "text", "owner", "start-failure")
    assert not current.get("recovery_claim_id")
    assert current["recovery_next_at"] > r.clock()
    r.text.recovery_reader.assert_not_called()
    failed_thread.join.assert_not_called()
