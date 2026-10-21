"""Pre-dispatch metadata liveness with local receipts and controlled I/O."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from services.account_service import AccountService
from services.account_request_pacing import AccountRequestClock
from services.config import config
from services.image_task_service import ImageTaskService
from services.openai_backend_api import OpenAIBackendAPI
from services.pool_admission import PoolAdmission
from services.task_store import TaskStore
from services.text_task_service import TextTaskService
from test.test_account_request_pacing import Response
from test.test_pool_admission import build, Clock


@pytest.fixture
def runtime(tmp_path):
    rows = [{"access_token": f"fixture-token-{i}", "account_id": f"upstream-{i}",
             "provider_account_identity": f"account-{i}", "type": "Plus", "status": "正常",
             "source_type": "web", "conversation_binding_ids": [f"binding-{i}"],
             "quota": 0 if i == 0 else 100,
             "limits_progress": [{"feature_name": "image_gen", "remaining": 0 if i == 0 else 100}],
             "capacity_observed_at": datetime.now(timezone.utc).isoformat()}
            for i in range(2)]
    (tmp_path / "accounts.json").write_text(json.dumps(rows))
    clock = Clock()
    accounts, store, admission = build(tmp_path, clock)
    generation = []
    executed = threading.Event()

    def text_runner(body, on_cursor):
        from services.request_context import current_request
        context = current_request.get()
        context.before_send()
        generation.append(context.request_id)
        executed.set()
        return {"content": "fixture answer"}

    text = TextTaskService(store.path, runner=text_runner, admission=admission, clock=clock)
    admission.register("text", lambda ctx, body: text._run(ctx.owner, ctx.request_id, body))
    images = ImageTaskService(tmp_path / "images.json", admission=admission, store=store)
    images.submit_generation({"id": "owner", "role": "user", "external_image_client": True},
        client_task_id="waiting-image", prompt="fixture", model="gpt-image-2", size=None,
        provider_account_identity="account-0", provider_binding_id="binding-0")
    release = threading.Event()
    workers = []
    real_thread = threading.Thread

    def thread(*args, **kwargs):
        worker = real_thread(*args, **kwargs)
        workers.append(worker)
        return worker

    def read(kind, rid):
        with store.connect() as db:
            return store.read_receipt(db, kind, "owner", rid)

    with patch("services.pool_admission.threading.Thread", side_effect=thread):
        yield SimpleNamespace(accounts=accounts, store=store, admission=admission, clock=clock,
            text=text, rows=rows, root=tmp_path, release=release, read=read,
            generation=generation, executed=executed)
        admission.stop()
        release.set()
        for worker in workers:
            if worker.ident is not None:
                worker.join(3)
                assert not worker.is_alive(), worker.name


def join_refresh(runtime):
    worker = runtime.admission._image_refresh_thread
    assert worker is not None
    worker.join(2)
    assert not worker.is_alive()
    return worker


def test_credential_lock_wait_does_not_block_other_claim_or_original_recovery(runtime):
    r = runtime
    for rid in ("unknown-original", "successful-original", "other-account"):
        r.text.submit("owner", {"client_request_id": rid, "client_conversation_id": rid,
            "model": "fixture-text", "messages": [{"role": "user", "content": "fixture"}],
            "provider_account_identity": "account-1" if rid == "other-account" else "account-0",
            "provider_binding_id": "binding-1" if rid == "other-account" else "binding-0"})
    r.text._update("owner", "unknown-original", status="unknown", upstream_outcome="unknown",
        _submission_started=True, recovery_next_at=0, conversation_id="original-chat")
    r.text._update("owner", "successful-original", status="succeeded", content="saved success",
        _upstream_terminal=True, _turn_reserved=False, _executing=False)
    originals = {rid: r.read("text", rid) for rid in ("unknown-original", "successful-original")}
    recovering, token_waiting = threading.Event(), threading.Event()
    recovered = []

    def read_original(owner, rid):
        recovered.append((owner, rid))
        recovering.set()
        assert r.release.wait(3)

    original_refresh = r.accounts.refresh_access_token

    def credential_refresh(*args, **kwargs):
        token_waiting.set()
        return original_refresh(*args, **kwargs)

    r.admission.recoveries["text"] = read_original
    # Keep the real credential path: no deadline keyword that silently skips it.
    r.accounts.refresh_image_capability = lambda ref: AccountService.refresh_image_capability(r.accounts, ref)
    r.accounts._token_refresh_lock.acquire()
    try:
        with patch.object(r.accounts, "refresh_access_token", side_effect=credential_refresh), \
             patch.object(r.accounts, "_verified_chat_info", side_effect=RuntimeError("fixture read failed")), \
             patch.object(r.accounts, "_request_access_token_refresh", side_effect=AssertionError("unexpected OAuth POST")) as oauth:
            r.admission.start()
            assert token_waiting.wait(2)
            assert r.executed.wait(2), "an unrelated account could not execute"
            assert recovering.wait(2), "new original-result recovery was not dispatched"
            assert r.admission._image_refresh_thread.is_alive()
            assert r.accounts._lock.acquire(timeout=.2), "credential wait retained the account-state lock"
            r.accounts._lock.release()
            assert r.generation == ["other-account"]
            assert recovered == [("owner", "unknown-original")]
            assert r.read("image", "waiting-image")["status"] == "queued"
            assert not r.read("image", "waiting-image").get("_submission_started")
            for rid, original in originals.items():
                assert r.read("text", rid) == original
            r.accounts._token_refresh_lock.release()
            join_refresh(r)
            oauth.assert_not_called()
    finally:
        if r.accounts._token_refresh_lock.locked():
            r.accounts._token_refresh_lock.release()


def test_one_inflight_refresh_survives_repeated_ticks_and_concurrent_starters(runtime):
    r = runtime
    entered = threading.Event()
    calls = []

    def refresh(ref):
        calls.append(ref)
        entered.set()
        assert r.release.wait(3)

    r.accounts.refresh_image_capability = refresh
    r.admission._start_image_capability_refresh()
    assert entered.wait(2)
    worker = r.admission._image_refresh_thread
    r.clock.now += 120  # Expiring the probe cadence cannot replace a live worker.
    with ThreadPoolExecutor(max_workers=8) as callers:
        list(callers.map(lambda _: r.admission._start_image_capability_refresh(), range(40)))
    assert r.admission._image_refresh_thread is worker
    assert len(calls) == 1
    assert r.admission.claim_next() is None  # Unknown/zero capacity remains closed.
    assert r.generation == []
    r.release.set()
    join_refresh(r)


def test_failed_refresh_retries_after_existing_cadence_then_wakes_claim(runtime):
    r = runtime
    calls = []

    def refresh(ref):
        calls.append(ref)
        if len(calls) == 1:
            raise RuntimeError("fixture metadata unavailable")
        r.accounts.update_account("fixture-token-0", {
            "quota": 100, "limits_progress": [{"feature_name": "image_gen", "remaining": 100}],
            "capacity_observed_at": datetime.now(timezone.utc).isoformat()}, quiet=True)

    r.accounts.refresh_image_capability = refresh
    r.admission.wake = Mock()
    r.admission._start_image_capability_refresh()
    first = join_refresh(r)
    assert r.admission.claim_next() is None
    r.admission._start_image_capability_refresh()
    assert r.admission._image_refresh_thread is first
    assert len(calls) == 1
    r.clock.now += 31
    r.admission._start_image_capability_refresh()
    join_refresh(r)
    assert len(calls) == 2
    assert r.admission.wake.call_count == 2
    claim = r.admission.claim_next()
    assert claim is not None and claim.request_id == "waiting-image"
    assert claim.receipt()["provider_account_identity"] == "account-0"
    assert not claim.receipt()["_submission_started"]
    assert r.generation == []  # Metadata only; no image generation was executed.


def test_scan_failure_does_not_latch_worker_or_block_later_refresh(runtime):
    r = runtime
    r.accounts.refresh_image_capability = Mock()
    original = r.admission._metadata_receipts
    scans = []

    def scan(*args):
        scans.append(True)
        if len(scans) == 2:  # Fail the worker's recheck, after the candidate precheck.
            raise RuntimeError("fixture scan failure")
        return original(*args)

    with patch.object(r.admission, "_metadata_receipts", side_effect=scan):
        r.admission._start_image_capability_refresh()
        first = join_refresh(r)
    r.accounts.refresh_image_capability.assert_not_called()
    r.admission._start_image_capability_refresh()
    join_refresh(r)
    assert r.admission._image_refresh_thread is not first
    r.accounts.refresh_image_capability.assert_called_once()


def test_empty_metadata_queue_does_not_wake_dispatcher_into_busy_loop(runtime):
    r = runtime
    with r.store.transaction() as db:
        db.execute("DELETE FROM image_requests")
    r.admission.wake = Mock()
    r.accounts.refresh_image_capability = Mock()
    r.admission._start_image_capability_refresh()
    assert r.admission._image_refresh_thread is None
    r.admission.wake.assert_not_called()
    r.accounts.refresh_image_capability.assert_not_called()


def test_uninitialized_store_does_not_start_metadata_worker(runtime):
    r = runtime
    store = TaskStore(r.root / "never-initialized.sqlite3")
    admission = PoolAdmission(store, r.accounts)
    with patch.object(store, "initialize", side_effect=AssertionError("metadata initialized idle store")):
        admission._start_image_capability_refresh()
    assert admission._image_refresh_thread is None
    assert not store.path.exists()


def test_real_metadata_clock_wait_releases_account_state_for_other_claims(runtime):
    r = runtime
    pace = AccountRequestClock("metadata-fixture", r.root / "metadata-clock.json")
    at_clock = threading.Event()
    transports = []

    class MetadataResponse(Response):
        def __init__(self, payload):
            super().__init__()
            self.payload = payload

        def json(self):
            return self.payload

    def send(method, url, **kwargs):
        transports.append((method, url))
        if url.endswith("/me"):
            return MetadataResponse({"id": "fixture-user", "email": "fixture@example.invalid"})
        if url.endswith("/init"):
            return MetadataResponse({"limits_progress": [{"feature_name": "image_gen", "remaining": 0}]})
        return MetadataResponse({"accounts": {"default": {"account": {
            "account_id": "11111111-1111-4111-8111-111111111111", "plan_type": "plus"}}}})

    def request(method, url, **kwargs):
        at_clock.set()
        return pace.request(send, method, url, **kwargs)

    # Keep the real _verified_chat_info/get_user_info and its three real metadata
    # methods. Only the backend transport is local, paced by the actual clock.
    backend = object.__new__(OpenAIBackendAPI)
    backend.access_token = "fixture-token-0"
    backend.base_url = "https://fixture"
    backend._headers = lambda *args: {}
    backend.session = SimpleNamespace(
        get=lambda url, **kw: request("GET", url, **kw),
        post=lambda url, **kw: request("POST", url, **kw), close=lambda: None)
    r.accounts.refresh_image_capability = lambda ref: AccountService.refresh_image_capability(r.accounts, ref)
    r.text.submit("owner", {"client_request_id": "other-clock-account", "client_conversation_id": "independent",
        "model": "fixture-text", "messages": [{"role": "user", "content": "fixture"}],
        "provider_account_identity": "account-1", "provider_binding_id": "binding-1"})
    pace.lock.acquire()
    clock_held = True
    try:
        with patch("services.openai_backend_api.OpenAIBackendAPI", return_value=backend), \
             patch.object(type(config), "account_request_interval_secs", property(lambda _: 0)):
            r.admission.start()
            assert at_clock.wait(2)
            assert r.executed.wait(2), "metadata's clock wait blocked an unrelated claim"
            assert r.accounts._lock.acquire(timeout=.2), "metadata retained the account-state lock"
            r.accounts._lock.release()
            assert len(r.accounts.list_accounts()) == 2
            assert transports == []
            assert r.generation == ["other-clock-account"]
            r.admission.stop()
            pace.lock.release()
            clock_held = False
            join_refresh(r)
            assert sorted(method for method, _ in transports) == ["GET", "GET", "POST"]
            assert all("/conversation" not in url or url.endswith("/conversation/init") for _, url in transports)
    finally:
        if clock_held:
            pace.lock.release()


def test_stop_does_not_join_stuck_metadata_or_start_next_account(runtime):
    r = runtime
    # Two eligible accounts in a sweep; stopping during the first read must
    # prevent the second. An already-started read is not falsely called killed.
    with r.store.transaction() as db:
        receipt = r.store.read_receipt(db, "image", "owner", "waiting-image")
        receipt.pop("provider_account_identity", None)
        r.store.write_receipt(db, "image", "owner", "waiting-image", receipt)
    r.accounts.update_account("fixture-token-1", {
        "quota": 0, "limits_progress": [{"feature_name": "image_gen", "remaining": 0}]}, quiet=True)
    entered = threading.Event()
    calls = []

    def refresh(ref):
        calls.append(ref)
        entered.set()
        assert r.release.wait(3)

    r.accounts.refresh_image_capability = refresh
    r.admission.start()
    assert entered.wait(2)
    worker = r.admission._image_refresh_thread
    started = time.monotonic()
    r.admission.stop()
    assert time.monotonic() - started < 1
    assert not r.admission._thread.is_alive()
    assert worker.is_alive()
    r.admission._start_image_capability_refresh()
    assert r.admission._image_refresh_thread is worker
    r.release.set()
    join_refresh(r)
    r.clock.now += 31
    r.admission._start_image_capability_refresh()
    assert r.admission._image_refresh_thread is worker
    assert len(calls) == 1
    assert r.generation == []
