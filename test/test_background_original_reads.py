"""Ordinary background recovery with isolated storage and controlled I/O."""
import json
import hashlib
import queue
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

    def original(rid, account=0, conversation=None, owner="owner", **changes):
        text.submit(owner, {"client_request_id": rid, "client_conversation_id": rid,
                              "messages": [{"role": "user", "content": "original input"}]})
        text._update(owner, rid, status="unknown", provider_account_identity=f"account-{account}",
                     provider_binding_id=f"binding-{account}", conversation_id=conversation or rid,
                     recovery_next_at=0, **changes)
        with store.connect() as db:
            return store.read_receipt(db, "text", owner, rid)

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


def test_due_history_on_one_account_cannot_take_all_original_read_slots(recovery):
    r = recovery
    # The persisted retry clock is intentionally identical.  SQLite insertion
    # order makes account-0's history appear first, which previously consumed
    # every one of the four local original-read slots before account-1's due
    # original was considered.
    for index in range(4):
        r.original(f"account-0-history-{index}", 0, f"history-{index}")
    r.original("account-1-due", 1, "due")
    started = {"account-1-due": threading.Event()}
    calls = []

    def read(owner, request_id):
        calls.append(request_id)
        if request_id == "account-1-due":
            started[request_id].set()
        r.release.wait(3)

    r.admission.recoveries["text"] = read
    r.admission.recover_one(background=True)

    assert started["account-1-due"].wait(1)
    assert len(calls) == 4
    assert sum(request_id.startswith("account-0-history-") for request_id in calls) == 3


def test_live_recovery_leases_do_not_take_a_background_dispatch_turn(recovery):
    r = recovery
    for index in range(4):
        request_id = f"leased-history-{index}"
        r.original(request_id, 0, f"leased-{index}", recovery_claim_id=f"lease-{index}",
                   recovery_claimed_at=r.clock(), recovery_lease_until=r.clock() + 60)
    r.original("other-account-due", 1, "other-due")
    started = threading.Event()
    calls = []

    def read(owner, request_id):
        calls.append(request_id)
        if request_id == "other-account-due":
            started.set()
        r.release.wait(3)

    r.admission.recoveries["text"] = read
    r.admission.recover_one(background=True)

    assert started.wait(1)
    assert calls == ["other-account-due"]


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
def test_async_images_keep_one_dispatch_per_scan_and_do_not_block_text(recovery, completion_first):
    r = recovery
    with r.store.transaction() as db:
        for i in range(5):
            rid = f"image-{i}"
            r.store.write_receipt(db, "image", "owner", rid, {
                "id": rid, "owner_id": "owner", "status": "error", "error_code": "CONVERSATION_OUTCOME_UNKNOWN",
                "provider_account_identity": "account-0", "conversation_id": rid,
                "request_message_id": f"message-{i}", "next_poll_at": 0})
    r.original("text")
    text_started = threading.Event()
    image_started = threading.Event()
    image_calls = []
    def image(owner, rid):
        image_calls.append(rid)
        image_started.set()  # Existing image handler returns after scheduling.
    if completion_first:
        with r.store.connect() as db:
            receipt = r.store.read_receipt(db, "image", "owner", "image-0")
        r.admission.generation_completion = SimpleNamespace(process_one=lambda *, dispatch:
            dispatch("image", "owner", "image-0", image, receipt))
    r.admission.recoveries.update(image=image, text=lambda owner, rid: text_started.set())
    r.admission.recover_one(background=True)
    assert image_started.wait(1) and text_started.wait(1)
    assert len(image_calls) == 1


def test_fifo_deadline_before_get_releases_all_reader_permits_and_other_account_can_continue(recovery):
    from services.account_request_pacing import AccountRequestClock, AccountRequestDeadlineExceeded
    import time

    r = recovery
    # A durable exact-result reader must not renew its lease forever merely
    # because a prior same-account ordinary reader owns the FIFO head. Run the
    # real clock path four times: if any local permit leaked, a different
    # account's recovery could not enter the shared reader pool.
    clock = AccountRequestClock("shared-account")
    clock.ordinary_read_queue = [{"owner": "earlier", "until": time.monotonic() + 900}]
    clock.ordinary_read_wait_until = clock.ordinary_read_queue[0]["until"]
    sent = []

    def no_send_reader(_receipt):
        with pytest.raises(AccountRequestDeadlineExceeded):
            clock.request(
                lambda method, url, **kwargs: sent.append((method, url, kwargs)),
                "GET", "https://chatgpt.com/backend-api/conversation/original",
                timeout=60,
                _account_request_deadline_monotonic=time.monotonic() + .01,
                _account_request_connect_timeout_secs=10,
                _account_request_minimum_budget_secs=10,
            )
        raise AccountRequestDeadlineExceeded("deadline elapsed before original result GET")

    r.text.recovery_reader = no_send_reader
    r.admission.recoveries["text"] = lambda owner, request_id: r.text.read(owner, request_id)
    for index in range(4):
        request_id = f"fifo-{index}"
        receipt = r.original(request_id, 0, request_id)
        original_identity = {key: receipt.get(key) for key in (
            "request_id", "request_hash", "_input_ref", "provider_binding_id",
            "provider_account_identity", "client_conversation_id", "conversation_id",
            "request_message_id",
        )}
        finished = threading.Event()

        def recover(owner, rid, finished=finished):
            try:
                return r.text.read(owner, rid)
            finally:
                finished.set()

        assert r.admission._dispatch_original_read("text", "owner", request_id, recover, receipt)
        assert finished.wait(1)
        with r.store.connect() as db:
            current = r.store.read_receipt(db, "text", "owner", request_id)
        assert current["status"] == "unknown"
        assert current.get("recovery_claim_id") is None
        assert current["recovery_next_at"] > r.clock()
        # No transport began: preserve the durable request/binding and do not
        # treat this pacing timeout as a qualified original-result read.
        assert {key: current.get(key) for key in original_identity} == original_identity
        assert current.get("_result_last_checked_at") is None
        assert current.get("recovery_no_result_reads", 0) == 0
        assert current.get("_execution_wait_ended_at") is None

    assert sent == []
    assert [item["owner"] for item in clock.ordinary_read_queue] == ["earlier"]
    assert not r.admission._original_reads

    other = r.original("other-account", 1, "other-account")
    progressed = threading.Event()
    assert r.admission._dispatch_original_read(
        "text", "owner", "other-account",
        lambda *_args: progressed.set(), other,
    )
    assert progressed.wait(1)


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
    r.admission.work_lifecycle.process_one.assert_called_once_with(background=True)


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


def test_capacity_contributing_originals_precede_history_without_losing_account_fairness(recovery):
    r = recovery
    # These are still due to read but already have the bounded local-wait
    # marker, so they are historical UNKNOWNs rather than account-slot
    # contributors.  Insert them first to reproduce the old starvation order.
    for index in range(4):
        request_id = f"history-{index}"
        r.original(request_id, 0, request_id)
        r.text._update("owner", request_id, status="failed", error_code="RESULT_UNRECOVERABLE",
                       upstream_outcome="unknown", _execution_wait_ended_at=r.clock())
    for index in range(4):
        r.original(f"contributor-{index}", 1, f"contributor-{index}")

    started = {f"contributor-{index}": threading.Event() for index in range(4)}
    calls = []

    def read(owner, request_id):
        calls.append(request_id)
        if request_id in started:
            started[request_id].set()
        r.release.wait(3)

    r.admission.recoveries["text"] = read
    r.admission.recover_one(background=True)

    assert all(event.wait(1) for event in started.values())
    assert set(calls) == set(started)


def test_transferred_empty_roots_do_not_precede_later_capacity_contributor(recovery):
    r = recovery
    # The exact same-session retry proof moves each root's local turn to its
    # queued child.  The roots remain UNKNOWN and due, but cannot consume the
    # first four original-read permits ahead of a real capacity contributor.
    for index in range(4):
        root_id = f"transferred-root-{index}"
        child_id = f"transferred-child-{index}"
        root = r.original(root_id, 0, f"transferred-conversation-{index}")
        r.original(child_id, 0, f"transferred-conversation-{index}")
        r.text._update("owner", root_id,
                       _retry_cursor={"conversation_id": root["conversation_id"],
                                      "request_message_id": root["request_message_id"],
                                      "retry_parent_message_id": f"retry-parent-{index}",
                                      "observed_at": r.clock()},
                       _completion={"replacement_id": child_id})
        r.text._update("owner", child_id, status="queued", _same_session_retry_of=root_id,
                       _completion_of=root_id, _submission_parent_message_id=f"retry-parent-{index}",
                       client_conversation_id=root["client_conversation_id"],
                       provider_account_identity=root["provider_account_identity"],
                       provider_binding_id=root["provider_binding_id"],
                       conversation_id=root["conversation_id"])
    r.original("actual-contributor", 0, "actual-contributor")
    scheduled = []

    def dispatch(kind, owner, request_id, function, receipt):
        scheduled.append(request_id)
        return True

    r.admission.recoveries["text"] = lambda *_: None
    with patch.object(r.admission, "_dispatch_original_read", side_effect=dispatch):
        r.admission.recover_one(background=True)

    assert scheduled[0] == "actual-contributor"
    assert set(scheduled[1:]) <= {f"transferred-root-{index}" for index in range(4)}


@pytest.mark.parametrize("completion_first", [False, True])
def test_released_original_gets_a_read_turn_during_continuous_contributor_recovery(recovery, completion_first):
    """A released UNKNOWN still needs its original result under steady load."""
    from services import account_request_pacing as pacing
    from services.account_request_pacing import AccountRequestClock
    from services.config import config

    r = recovery
    root = r.store.path.parent
    for rid in ("contributor-a", "contributor-b", "released-original"):
        r.original(rid, 0, rid, _submission_started=True, _executing=False,
                   request_message_id=f"message-{rid}", upstream_outcome="unknown")
    released = r.text._update("owner", "released-original", _turn_reserved=False,
        recovery_reason="REQUEST_RESULT_INCOMPLETE", _result_observation_started_at=1.0,
        _result_last_checked_at=r.clock(), _result_wait_ended_at=r.clock(),
        _result_last_progress_at=None, _result_no_progress_reads=3,
        _original_result_observation={"conversation_id": "released-original",
            "request_message_id": "message-released-original", "upstream_updated_at": 100.0,
            "nodes": [{"id": "unfinished-node", "role": "assistant", "status": "in_progress",
                       "end_turn": False, "text_chars": 0}]})
    with r.store.connect() as db:
        released = r.store.read_receipt(db, "text", "owner", "released-original")
    assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(released, r.clock())

    key = hashlib.sha256(b"upstream-0").hexdigest()
    path = root / "account_request_clocks" / f"{key}.json"
    pending, sent, finished = queue.Queue(), [], threading.Event()
    def read(owner, request_id):
        allow = threading.Event()
        pending.put((request_id, allow))
        assert allow.wait(2)
        def send(method, *_args, **_kwargs):
            assert method == "GET"
            sent.append(request_id)
            return SimpleNamespace(status_code=200, headers={})
        AccountRequestClock(key[:12], path).request(send, "GET", f"https://provider/conversation/{request_id}")
        # A normal local outcome can be due again by the following account
        # read edge; the other contributor keeps a continuous backlog.
        r.text._update(owner, request_id, recovery_next_at=r.clock() + 30, updated_at=r.clock(),
                       _completion={"next_at": r.clock() + 30})

    if completion_first:
        def offer_completion(*, dispatch):
            with r.store.connect() as db:
                receipts = [r.store.read_receipt(db, "text", "owner", rid)
                            for rid in ("contributor-a", "contributor-b")]
            for receipt in receipts:
                if receipt.get("_completion", {}).get("next_at", 0) <= r.clock():
                    dispatch("text", "owner", receipt["request_id"], read, receipt)
        r.admission.generation_completion = SimpleNamespace(process_one=offer_completion)

    r.admission.pacing = None
    r.admission.recoveries["text"] = read
    with patch.object(pacing, "DATA_DIR", root), patch.object(pacing, "_clocks", {}), \
         patch("services.account_request_pacing.time.monotonic", side_effect=r.clock), \
         patch("services.account_request_pacing.time.time", side_effect=r.clock), \
         patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)), \
         patch.object(type(config), "account_conversation_read_interval_secs", property(lambda _: 45)), \
         patch.object(type(config), "account_conversation_read_burst", property(lambda _: 1)), \
         patch.object(r.admission, "wake", side_effect=finished.set):
        for _ in range(8):
            finished.clear()
            r.admission.recover_one(background=True)
            request_id, allow = pending.get(timeout=1)
            allow.set()
            assert finished.wait(2)
            # The scanner also runs before the next read edge.  Its first
            # eligible candidate books the only dormant original FIFO ticket.
            r.clock.now += 1
            r.admission.recover_one(background=True)
            r.clock.now += 44
        assert sent[0].startswith("contributor-")
        assert "released-original" in sent[:4]
        assert any(rid.startswith("contributor-") for rid in sent[4:])
    assert len(r.queue.calls) == 3  # Only the three fixture submissions; no replacement or resend.


def released_original(recovery, request_id, account=0, next_at=0, conversation=None, **changes):
    recovery.original(request_id, account, conversation)
    recovery.text._update("owner", request_id, status="failed", error_code="RESULT_UNRECOVERABLE",
                          upstream_outcome="unknown", _execution_wait_ended_at=recovery.clock(),
                          recovery_next_at=next_at, **changes)


def explicit_original(recovery, request_id, account=0, **changes):
    return recovery.original(request_id, account, _submission_started=True,
                             _turn_reserved=False, _executing=False, upstream_outcome="unknown",
                             request_message_id="message-" + request_id, **changes)


def compact_incomplete_leaf(recovery, parent_id="compact-parent", child_id="compact-child",
                            *, parent_owner="owner", child_owner="owner"):
    """Persist a sent compact child and a stale exact-result observation."""
    r = recovery
    shared = "compact-conversation"
    parent = r.original(parent_id, 0, shared, owner=parent_owner,
                        client_conversation_id="compact-client",
                        parent_message_id="compact-root",
                        request_message_id="message-" + parent_id,
                        _submission_started=True, _turn_reserved=False, _executing=False,
                        upstream_outcome="unknown")
    child = r.original(child_id, 0, shared, owner=child_owner,
                       client_conversation_id="compact-client",
                       parent_message_id="compact-root",
                       request_message_id="message-" + child_id,
                       _submission_started=True, _turn_reserved=False, _executing=False,
                       upstream_outcome="unknown")
    with r.store.connect() as db:
        parent_hash = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?",
                                 (parent_owner, parent_id)).fetchone()[0]
    assert child.get("_input_ref")
    stalled = {
        "recovery_reason": "REQUEST_RESULT_INCOMPLETE",
        "_result_observation_started_at": 1.0,
        "_result_last_checked_at": r.clock(),
        "_result_wait_ended_at": r.clock(),
        "_result_last_progress_at": None,
        "_result_no_progress_reads": 3,
    }
    for owner, request_id in ((parent_owner, parent_id), (child_owner, child_id)):
        r.text._update(owner, request_id, **stalled,
                       _original_result_observation={
                           "conversation_id": shared,
                           "request_message_id": "message-" + request_id,
                           "upstream_updated_at": 100.0,
                           "nodes": [{"id": "unfinished-" + request_id,
                                      "role": "assistant", "status": "in_progress",
                                      "end_turn": False, "text_chars": 0}],
                       })
    r.text._update(child_owner, child_id,
                   _supersedes_request_id=parent_id,
                   _supersedes_input_hash=parent_hash,
                   _supersedes_request_message_id="message-" + parent_id,
                   _submission_parent_message_id="compact-root",
                   _supersedes_incomplete=True,
                   _derived_input={"kind": "combined_copy_compact_v1",
                                   "original_input_hash": parent_hash})
    return parent_id, child_id


def prepare_explicit_recovery(recovery, monkeypatch):
    from services import account_request_pacing as pacing
    from services.config import config
    r = recovery
    r.text.admission = r.admission
    r.admission.pacing = None
    monkeypatch.setattr(pacing, "DATA_DIR", r.store.path.parent)
    monkeypatch.setattr(pacing, "_clocks", {})
    monkeypatch.setattr(pacing.time, "monotonic", r.clock)
    monkeypatch.setattr(pacing.time, "time", r.clock)
    monkeypatch.setattr(type(config), "account_request_interval_secs", property(lambda _: 0))
    monkeypatch.setattr(type(config), "account_conversation_read_interval_secs", property(lambda _: 45))
    monkeypatch.setattr(type(config), "account_conversation_read_burst", property(lambda _: 1))
    calls = []

    def reader(receipt):
        account = receipt["provider_account_identity"].replace("account-", "upstream-")
        _, clock = pacing._clock_for_account(account)
        def send(method, *_args, **_kwargs):
            calls.append((method, receipt["request_id"]))
            return SimpleNamespace(status_code=200, headers={})
        clock.request(send, "GET", "https://provider/conversation/" + receipt["conversation_id"])
        return {**receipt, "status": "succeeded", "binding_status": "bound",
                "parent_message_id": "final-" + receipt["request_id"], "content": "original final"}
    r.text.recovery_reader = reader
    r.admission.recoveries["text"] = r.text.read
    return pacing, calls


def test_explicit_recover_defers_without_claim_and_background_adopts_same_fifo_ticket(recovery, monkeypatch):
    from services.public_chat_service import project_public_chat_receipt
    r = recovery
    explicit_original(r, "waiting-original", recovery_attempt=8)
    explicit_original(r, "same-account")
    explicit_original(r, "other-account", 1)
    pacing, calls = prepare_explicit_recovery(r, monkeypatch)
    _, clock = pacing._clock_for_account("upstream-0")
    clock.request(lambda *_a, **_k: SimpleNamespace(status_code=200, headers={}),
                  "GET", "https://provider/conversation/previous")
    with r.store.connect() as db:
        before = db.execute("SELECT request_hash,receipt FROM requests WHERE id=?", ("waiting-original",)).fetchone()
    for _ in range(3):
        result = r.text.recover("owner", "waiting-original", explicit_ended_recheck=True)
        assert result["recovery_deferred"] is True
        assert project_public_chat_receipt(result)["recovery"]["deferred"] is True
    ticket = list(clock.ordinary_read_queue)
    assert len(ticket) == 1
    assert r.text.recover("owner", "same-account")["recovery_deferred"] is True
    assert clock.ordinary_read_queue == ticket  # No second dormant ticket or queue jump.
    with r.store.connect() as db:
        assert db.execute("SELECT request_hash,receipt FROM requests WHERE id=?", ("waiting-original",)).fetchone() == before
    assert not calls and not r.admission._original_reads
    assert r.text.recover("owner", "other-account")["content"] == "original final"
    assert calls == [("GET", "other-account")]

    finished = threading.Event()
    monkeypatch.setattr(r.admission, "wake", finished.set)
    r.clock.now += 45
    r.admission.recover_one(background=True)
    assert finished.wait(2)
    result = r.text.read_snapshot("owner", "waiting-original")
    assert result["status"] == "succeeded" and result["content"] == "original final"
    assert "recovery_deferred" not in result
    assert calls == [("GET", "other-account"), ("GET", "waiting-original")]
    assert not r.admission._original_reads


def test_background_registered_read_reserves_fifo_then_claims_at_the_read_edge(recovery, monkeypatch):
    """A due background original retains its FIFO place without claiming early."""
    r = recovery
    r.original("paced-background")
    pacing, calls = prepare_explicit_recovery(r, monkeypatch)
    _, clock = pacing._clock_for_account("upstream-0")
    # Consume the only read credit first.  The background dispatcher must
    # reserve the due original, then invoke its registered ``text.read`` only
    # when that durable read edge arrives.
    clock.request(lambda *_args, **_kwargs: SimpleNamespace(status_code=200, headers={}),
                  "GET", "https://provider/conversation/previous")
    completed = threading.Event()
    monkeypatch.setattr(r.admission, "wake", completed.set)

    r.admission.recover_one(background=True)
    with r.store.connect() as db:
        waiting = r.store.read_receipt(db, "text", "owner", "paced-background")
    assert waiting.get("recovery_attempt") is None
    assert not calls
    assert len(clock.ordinary_read_queue) == 1
    assert clock.ordinary_read_queue[0]["owner"].startswith("original-result:")

    r.clock.now += 45
    r.admission.recover_one(background=True)
    assert completed.wait(2)
    result = r.text.read_snapshot("owner", "paced-background")
    assert result["status"] == "succeeded" and result["content"] == "original final"
    assert calls == [("GET", "paced-background")]
    assert not clock.ordinary_read_queue and not r.admission._original_reads


def test_explicit_and_background_recovery_share_physical_read_exclusion(recovery, monkeypatch):
    r = recovery
    receipt = explicit_original(r, "shared-original")
    _, calls = prepare_explicit_recovery(r, monkeypatch)
    acquired, finished = threading.Event(), threading.Event()
    def background(owner, request_id):
        acquired.set()
        assert r.release.wait(2)
        return r.text.read(owner, request_id)
    monkeypatch.setattr(r.admission, "wake", finished.set)
    assert r.admission._dispatch_original_read("text", "owner", "shared-original", background, receipt)
    assert acquired.wait(1)
    assert r.text.recover("owner", "shared-original")["recovery_deferred"] is True
    assert calls == []
    r.release.set()
    assert finished.wait(2)
    assert r.text.read_snapshot("owner", "shared-original")["status"] == "succeeded"
    assert calls == [("GET", "shared-original")] and not r.admission._original_reads


@pytest.mark.parametrize("change", [
    {"status": "succeeded", "content": "already saved"},
    {"_recovery_paused": True},
    {"recovery_claim_id": "other-reader", "recovery_lease_until": 1100},
])
def test_explicit_read_revalidates_receipt_after_admission(recovery, monkeypatch, change):
    r = recovery
    explicit_original(r, "changed-original")
    pacing, calls = prepare_explicit_recovery(r, monkeypatch)
    original_read = r.text.read
    def changed(owner, request_id, **kwargs):
        r.text._update(owner, request_id, **change)
        return original_read(owner, request_id, **kwargs)
    monkeypatch.setattr(r.text, "read", changed)
    result = r.text.recover("owner", "changed-original")
    assert result["status"] == change.get("status", "unknown")
    assert calls == [] and not r.admission._original_reads
    assert not pacing._clock_for_account("upstream-0")[1].ordinary_read_queue


@pytest.mark.parametrize("change", [
    {"status": "succeeded", "content": "saved concurrently"},
    {"_recovery_paused": True},
    {"_recovery_suppressed": True},
    {"recovery_claim_id": "other-reader", "recovery_lease_until": 1100},
    {"_attempt_finished_at": 900},
])
def test_deferred_projection_rechecks_latest_state_without_writing(recovery, monkeypatch, change):
    r = recovery
    explicit_original(r, "deferred-race")
    _, calls = prepare_explicit_recovery(r, monkeypatch)
    changed = []
    def defer(*_args):
        r.text._update("owner", "deferred-race", **change)
        with r.store.connect() as db:
            changed.append(db.execute("SELECT receipt FROM requests WHERE id='deferred-race'").fetchone())
        return False
    monkeypatch.setattr(r.admission, "read_original_if_ready", defer)
    result = r.text.recover("owner", "deferred-race")
    assert "recovery_deferred" not in result
    with r.store.connect() as db:
        assert db.execute("SELECT receipt FROM requests WHERE id='deferred-race'").fetchone() == changed[0]
    assert calls == []


def test_explicit_claim_busy_releases_admitted_fifo_and_local_slot(recovery, monkeypatch):
    import sqlite3
    from services.text_task_service import TextRecoveryLeaseBusy
    r = recovery
    explicit_original(r, "busy-original")
    pacing, calls = prepare_explicit_recovery(r, monkeypatch)
    connect = r.text.store._connect
    def bounded_connect(**kwargs):
        db = connect(**kwargs)
        db.execute("PRAGMA busy_timeout=10")
        return db
    monkeypatch.setattr(r.text.store, "_connect", bounded_connect)
    with sqlite3.connect(r.store.path) as writer:
        writer.execute("BEGIN IMMEDIATE")
        with pytest.raises(TextRecoveryLeaseBusy):
            r.text.recover("owner", "busy-original")
    assert not calls and not r.admission._original_reads
    assert not pacing._clock_for_account("upstream-0")[1].ordinary_read_queue
    assert r.text.recover("owner", "busy-original")["status"] == "succeeded"


@pytest.mark.parametrize("legacy", ["ended", "allow_unrecoverable", "no_admission", "custom_pacing", "missing_account", "disabled"])
def test_explicit_admission_preserves_legacy_recovery_modes(recovery, monkeypatch, legacy):
    r = recovery
    explicit_original(r, "legacy-original")
    prepare_explicit_recovery(r, monkeypatch)
    if legacy == "ended":
        r.text._update("owner", "legacy-original", _attempt_finished_at=900,
                       _attempt_reason="ORIGINAL_RESULT_NO_FINAL", recovery_reason="REQUEST_RESULT_NOT_FOUND")
    elif legacy == "no_admission":
        r.text.admission = None
    elif legacy == "custom_pacing":
        r.admission.pacing = lambda *_: {"next_at": 9999}
    elif legacy == "missing_account":
        monkeypatch.setattr(r.admission, "_rows", lambda: [])
    elif legacy == "disabled":
        monkeypatch.setattr(r.admission, "_rows", lambda: [{"provider_account_identity": "account-0", "managed_disabled": True}])
    read = Mock(return_value={"status": "unknown", "legacy_path": True})
    monkeypatch.setattr(r.text, "read", read)
    assert r.text.recover("owner", "legacy-original", legacy == "allow_unrecoverable",
                          explicit_ended_recheck=True)["legacy_path"] is True
    assert read.call_count == 1
    assert read.call_args.kwargs["_explicit_ended_recheck"] is True
    assert read.call_args.kwargs["allow_unrecoverable_retry"] == (legacy == "allow_unrecoverable")


def test_same_scan_rechecks_priority_and_serves_released_originals_in_due_order(recovery):
    r = recovery
    released_original(r, "newer-released", next_at=900)
    released_original(r, "older-released", next_at=800)
    for i in range(6):
        r.original(f"contributor-{i}")
    admitted = []
    r.admission.recoveries["text"] = lambda *_: None
    with patch.object(r.admission, "_dispatch_original_read",
                      side_effect=lambda k, o, rid, f, receipt: admitted.append(rid) or True):
        r.admission.recover_one(background=True)
    assert admitted == ["contributor-0", "contributor-1", "contributor-2", "older-released",
                        "contributor-3", "contributor-4", "contributor-5", "newer-released"]


def test_bounded_get_timeout_stays_a_contributor_before_released_history(recovery):
    """A failed exact GET frees capacity without proving its original stopped."""
    r = recovery
    r.original("timeout-original", _submission_started=True, _turn_reserved=False,
               _executing=False, upstream_outcome="unknown")

    def unavailable(_receipt):
        error = TimeoutError("fixture exact original GET timed out")
        error._original_result_get_attempted = True
        raise error

    r.text.recovery_reader = unavailable
    assert r.text.read("owner", "timeout-original")["status"] == "unknown"
    with r.store.connect() as db:
        timed_out = r.store.read_receipt(db, "text", "owner", "timeout-original")
    assert timed_out["recovery_error_code"] == "RECOVERY_READ_FAILED"
    assert TextTaskService._bounded_unreadable_result_capacity_boundary(timed_out, r.clock())
    assert not TextTaskService._qualified_stalled_incomplete_capacity_boundary(timed_out, r.clock())

    # Make the proven live original and three ordinary contributors due in
    # order.  The old rows model the 178 released originals ahead of the
    # incident child in the production snapshot without retaining their IDs.
    r.text._update("owner", "timeout-original", recovery_next_at=0)
    for index in range(3):
        request_id = f"contributor-{index}"
        r.original(request_id)
        r.text._update("owner", request_id, recovery_next_at=index + 1)
    for index in range(178):
        released_original(r, f"released-history-{index}", next_at=10 + index)

    submitted = len(r.queue.calls)
    admitted = []
    r.admission.recoveries["text"] = lambda *_: None
    with patch.object(r.admission, "_dispatch_original_read",
                      side_effect=lambda kind, owner, request_id, function, receipt:
                      admitted.append(request_id) or True):
        r.admission.recover_one(background=True)

    # The timeout child must not wait behind released history.  Once three
    # P0 originals have been served, the existing 3:1 fairness edge gives the
    # earliest P1 row a turn before another P0 contributor.
    assert admitted[:4] == ["timeout-original", "contributor-0", "contributor-1",
                            "released-history-0"]
    assert len(r.queue.calls) == submitted  # Recovery scheduled reads only; no new POST.


def test_qualified_stalled_original_remains_released_history(recovery):
    r = recovery
    r.original("qualified-stalled", _submission_started=True, _turn_reserved=False,
               _executing=False, request_message_id="message-qualified-stalled",
               upstream_outcome="unknown")
    r.text._update("owner", "qualified-stalled",
                   recovery_reason="REQUEST_RESULT_INCOMPLETE",
                   _result_observation_started_at=1.0,
                   _result_last_checked_at=r.clock(), _result_wait_ended_at=r.clock(),
                   _result_last_progress_at=None, _result_no_progress_reads=3,
                   _original_result_observation={
                       "conversation_id": "qualified-stalled",
                       "request_message_id": "message-qualified-stalled",
                       "upstream_updated_at": 100.0,
                       "nodes": [{"id": "unfinished-node", "role": "assistant",
                                  "status": "in_progress", "end_turn": False,
                                  "text_chars": 0}],
                   })
    with r.store.connect() as db:
        stalled = r.store.read_receipt(db, "text", "owner", "qualified-stalled")
    assert TextTaskService._qualified_stalled_incomplete_capacity_boundary(stalled, r.clock())

    for index in range(4):
        request_id = f"contributor-{index}"
        r.original(request_id)
        r.text._update("owner", request_id, recovery_next_at=index + 1)
    admitted = []
    r.admission.recoveries["text"] = lambda *_: None
    with patch.object(r.admission, "_dispatch_original_read",
                      side_effect=lambda kind, owner, request_id, function, receipt:
                      admitted.append(request_id) or True):
        r.admission.recover_one(background=True)
    assert admitted[:5] == ["contributor-0", "contributor-1", "contributor-2",
                            "qualified-stalled", "contributor-3"]


def test_sent_compact_leaf_keeps_current_read_priority_and_preserves_three_to_one(recovery):
    r = recovery
    parent_id, child_id = compact_incomplete_leaf(r)
    for index in range(2):
        r.original(f"contributor-{index}")
        r.text._update("owner", f"contributor-{index}", recovery_next_at=index + 1)
    released_original(r, "older-released", next_at=4)

    admitted = []
    r.admission.recoveries["text"] = lambda *_: None
    with patch.object(r.admission, "_dispatch_original_read",
                      side_effect=lambda kind, owner, request_id, function, receipt:
                      admitted.append(request_id) or True):
        r.admission.recover_one(background=True)

    # The verified sent child is P0 despite its capacity-release observation.
    # Three P0 reads (including that child) still give released history a turn.
    assert admitted[:4] == [child_id, "contributor-0", "contributor-1", parent_id]


@pytest.mark.parametrize("invalid", [
    "fake-flags", "hash-drift", "audit-hash-drift", "other-owner", "newer-child",
    "wrong-binding", "wrong-account", "wrong-conversation", "wrong-client", "sibling-fallback",
])
def test_unverified_compact_lineage_stays_released_history(recovery, invalid):
    r = recovery
    parent_owner, child_owner = ("owner", "other-owner") if invalid == "other-owner" else ("owner", "owner")
    parent_id, child_id = compact_incomplete_leaf(
        r, f"{invalid}-parent", f"{invalid}-child",
        parent_owner=parent_owner, child_owner=child_owner)
    if invalid == "fake-flags":
        r.text._update(child_owner, child_id, _derived_input={"kind": "untrusted",
                                                               "original_input_hash": "not-used"})
    elif invalid == "hash-drift":
        r.text._update(child_owner, child_id, _supersedes_input_hash="drifted-hash")
    elif invalid == "audit-hash-drift":
        r.text._update(child_owner, child_id, _derived_input={
            "kind": "combined_copy_compact_v1", "original_input_hash": "drifted-hash"})
    elif invalid.startswith("wrong-"):
        field = {"wrong-binding": "provider_binding_id", "wrong-account": "provider_account_identity",
                 "wrong-conversation": "conversation_id", "wrong-client": "client_conversation_id"}[invalid]
        r.text._update(child_owner, child_id, **{field: "mismatched-identity"})
        if invalid == "wrong-conversation":
            # Keep the child's own stalled observation valid while breaking
            # only its lineage relationship to the parent's conversation.
            with r.store.connect() as db:
                receipt = r.store.read_receipt(db, "text", child_owner, child_id)
            r.text._update(child_owner, child_id, _original_result_observation={
                **receipt["_original_result_observation"], "conversation_id": "mismatched-identity"})
    elif invalid == "sibling-fallback":
        r.original("competing-fallback", owner=child_owner, _attempt_finished_at=r.clock())
        r.text._update(child_owner, "competing-fallback", _codex_fallback_of=parent_id)
    elif invalid == "newer-child":
        r.original(f"{invalid}-newer", 0, "compact-conversation", owner=child_owner,
                   _attempt_finished_at=r.clock())
        r.text._update(child_owner, f"{invalid}-newer",
                       _supersedes_request_id=child_id)

    from services.pool_admission import _verified_current_incomplete_leaves
    with r.store.connect() as db:
        rows = [(owner, rid, digest, json.loads(raw)) for owner, rid, digest, raw in
                db.execute("SELECT owner,id,request_hash,receipt FROM requests")]
    assert (child_owner, child_id) not in _verified_current_incomplete_leaves(rows)
    r.original("ordinary-contributor")
    r.text._update("owner", "ordinary-contributor", recovery_next_at=1)
    admitted = []
    r.admission.recoveries["text"] = lambda *_: None
    with patch.object(r.admission, "_dispatch_original_read",
                      side_effect=lambda kind, owner, request_id, function, receipt:
                      admitted.append(request_id) or True):
        r.admission.recover_one(background=True)

    # The ordinary due original remains ahead: mutable/fake receipt flags,
    # hash drift, owner mismatch, and a newer child fail closed to P1.
    if invalid != "wrong-account":
        assert admitted[0] == "ordinary-contributor"


def test_compact_leaf_and_parent_keep_one_physical_original_read(recovery):
    r = recovery
    _parent_id, child_id = compact_incomplete_leaf(r)
    child_started = threading.Event()
    calls = []

    def read(_owner, request_id):
        calls.append(request_id)
        if request_id == child_id:
            child_started.set()
            assert r.release.wait(3)

    r.admission.recoveries["text"] = read
    r.admission.recover_one(background=True)
    assert child_started.wait(1)
    # The parent is due too, but shares the exact physical conversation key
    # and cannot start another original-result reader while the child is live.
    r.admission.recover_one(background=True)
    assert calls == [child_id]


def test_each_account_keeps_its_turn_budget_across_pacing_failures_and_empty_scans(recovery):
    r = recovery
    for account in range(2):
        for i in range(2):
            r.original(f"account-{account}-contributor-{i}", account)
        released_original(r, f"account-{account}-released", account)
    available = {}
    admitted = {f"account-{i}": [] for i in range(2)}
    def dispatch(kind, owner, request_id, function, receipt):
        account = receipt["provider_account_identity"]
        if not available.get(account):
            return False
        available[account] = False
        admitted[account].append(request_id)
        r.text._update(owner, request_id, recovery_next_at=r.clock() + 1)
        return True
    r.admission.recoveries["text"] = lambda *_: None
    with patch.object(r.admission, "_dispatch_original_read", side_effect=dispatch):
        for tick in range(12):
            available.update({"account-0": True, "account-1": tick % 3 == 0})
            r.admission.recover_one(background=True)
            for _ in range(2):
                r.admission.recover_one(background=True)  # No pacing opportunity.
            r.clock.now += 1
    for account, calls in admitted.items():
        assert all("contributor" in rid for rid in calls[:3])
        assert calls[3] == f"{account}-released"


@pytest.mark.parametrize("blocker", ["absent", "paused", "future", "leased", "physical_duplicate"])
def test_unavailable_released_original_does_not_stop_contributor_fallback(recovery, blocker):
    r = recovery
    for i in range(4):
        r.original(f"contributor-{i}")
    changes = {"paused": {"_recovery_paused": True},
               "future": {"next_at": r.clock() + 60},
               "leased": {"recovery_lease_until": r.clock() + 60},
               "physical_duplicate": {"conversation": "contributor-0"}}
    if blocker != "absent":
        released_original(r, "unavailable-released", **changes[blocker])
    started = {f"contributor-{i}": threading.Event() for i in range(4)}
    calls = []
    def read(owner, request_id):
        calls.append(request_id)
        if request_id in started:
            started[request_id].set()
        r.release.wait(3)
    r.admission.recoveries["text"] = read
    r.admission.recover_one(background=True)
    assert all(event.wait(1) for event in started.values())
    assert set(calls) == set(started)


def test_completion_only_ended_recheck_survives_and_duplicate_uses_completion_callback_once(recovery):
    r = recovery
    r.original("completion-only", _attempt_finished_at=r.clock())
    r.original("duplicate")
    r.original("ordinary")
    completion_calls, ordinary_calls = [], []
    def offer_completion(*, dispatch):
        with r.store.connect() as db:
            receipts = [r.store.read_receipt(db, "text", "owner", rid)
                        for rid in ("completion-only", "duplicate")]
        for receipt in receipts:
            dispatch("text", "owner", receipt["request_id"], lambda o, rid: completion_calls.append(rid), receipt)
    def dispatch(kind, owner, request_id, function, receipt):
        function(owner, request_id)
        return True
    r.admission.generation_completion = SimpleNamespace(process_one=offer_completion)
    r.admission.recoveries["text"] = lambda o, rid: ordinary_calls.append(rid)
    with patch.object(r.admission, "_dispatch_original_read", side_effect=dispatch):
        r.admission.recover_one(background=True)
    assert sorted(completion_calls) == ["completion-only", "duplicate"]
    assert ordinary_calls == ["ordinary"]


@pytest.mark.parametrize("mixed_busy_account", [False, True])
def test_released_only_account_progresses_after_initial_contributor_head_start(recovery, mixed_busy_account):
    r = recovery
    # One variant keeps four accounts continuously busy. The other spends
    # every scan on three contributors plus one released read on a mixed
    # account, which must not reset another account's chance to zero forever.
    for i in range(4):
        r.original(f"busy-{i}", 0 if mixed_busy_account else i)
    if mixed_busy_account:
        released_original(r, "mixed-account-released", 0)
    released_original(r, "released-only-account", 9)
    scans, available = [], [0]
    def dispatch(kind, owner, request_id, function, receipt):
        if available[0] == 0:
            return False
        available[0] -= 1
        scans[-1].append(request_id)
        return True
    r.admission.recoveries["text"] = lambda *_: None
    with patch.object(r.admission, "_dispatch_original_read", side_effect=dispatch):
        for _ in range(4):
            scans.append([])
            available[0] = 4
            r.admission.recover_one(background=True)
    if not mixed_busy_account:
        assert all(rid.startswith("busy-") for rid in scans[0])
    assert "released-only-account" in scans[1]
    assert any(rid.startswith("busy-") for rid in scans[-1])
