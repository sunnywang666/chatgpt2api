"""Real mutex contention and local blocking HTTP at original recovery boundaries."""
import json
import threading
import time
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from curl_cffi import requests

from services.account_service import AccountService
from services.account_request_pacing import AccountRequestDeadlineExceeded
from services.conversation_binding_service import ConversationBindingService
from services.image_task_service import ImageTaskService
from services.openai_backend_api import OpenAIBackendAPI as RealBackend
from services.text_task_service import TextTaskService
from services.request_context import operation_deadline, bounded_lock, executing
from test.test_work_lifecycle import runtime
from test.test_background_original_reads import recovery


@pytest.fixture
def local_upstream(tmp_path, monkeypatch):
    import services.account_request_pacing as pacing
    from services.config import ConfigStore
    state = SimpleNamespace(calls=[], backends=[], blocked=threading.Event(), release=threading.Event(),
                            block=None, delays={}, archived=False, status=200, invalid_json=False,
                            disconnect=False)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def handle_request(self):
            state.calls.append((self.command, self.path))
            if state.disconnect:
                self.close_connection = True
                return
            if self.command == "PATCH":
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                state.archived = body["is_archived"]
            delay = state.delays.get(self.path, 0)
            if delay:
                state.release.wait(delay)
            if state.block == (self.command, self.path):
                state.blocked.set()
                state.release.wait(3)
            if self.path.startswith("/backend-api/files/"):
                payload = {"download_url": f"http://127.0.0.1:{self.server.server_port}/asset"}
            elif self.path == "/asset":
                payload = None
            else:
                payload = {"conversation_id": "original", "current_node": "final", "is_archived": state.archived,
                           "mapping": {"user": {"parent": "root", "message": {
                               "id": "user", "author": {"role": "user"}}},
                           "final": {"parent": "user", "message": {"id": "final", "author": {"role": "assistant"},
                               "content": {"content_type": "text", "parts": ["original answer"]},
                               "status": "finished_successfully", "end_turn": True}}}}
            data = b"original-image-bytes" if payload is None else json.dumps(payload).encode()
            if state.invalid_json:
                data = b"invalid-json"
            try:
                self.send_response(state.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The bounded caller already closed its connection.
        do_GET = handle_request
        do_PATCH = handle_request
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    (tmp_path / "config.json").write_text(json.dumps({"auth-key": "test-only"}))
    config = ConfigStore(tmp_path / "config.json")
    config.update({"account_request_interval_secs": 0, "account_conversation_read_interval_secs": 0})
    monkeypatch.setattr(pacing, "config", config)
    monkeypatch.setattr(pacing, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pacing, "_clocks", {})
    def backend(access_token):
        instance = RealBackend.__new__(RealBackend)
        instance._closed = False
        instance.base_url = "https://chatgpt.com"
        instance._headers = lambda _path, extra: extra
        session = requests.Session(trust_env=False)
        raw_send = session.request
        def send(method, url, **kwargs):
            if urlsplit(url).hostname == "chatgpt.com":
                url = f"http://127.0.0.1:{server.server_port}" + urlsplit(url).path
            return raw_send(method, url, **kwargs)
        session.request = send
        pacing.pace_account_session(session, {"account_id": access_token}, access_token)
        instance.session = session
        state.backends.append(instance)
        return instance
    accounts = AccountService.__new__(AccountService)
    accounts._lock = threading.Lock()
    accounts._conversation_binding_locks = {}
    accounts.get_bound_account_identity = lambda binding: "account"
    accounts.get_bound_text_access_token = lambda binding, **kwargs: "fixture-token"
    monkeypatch.setattr("services.account_service.account_service", accounts)
    monkeypatch.setattr("services.conversation_binding_service.account_service", accounts)
    monkeypatch.setattr("services.conversation_binding_service.OpenAIBackendAPI", backend)
    monkeypatch.setattr("services.openai_backend_api.OpenAIBackendAPI", backend)
    state.backend, state.accounts = backend, accounts
    yield state
    state.release.set()
    for instance in state.backends:
        instance.close()
    server.shutdown()
    server.server_close()
    serving.join(1)


def cursor(session="blocked"):
    return {"provider_binding_id": "binding", "provider_account_identity": "account",
            "client_conversation_id": session, "conversation_id": "original",
            "parent_message_id": "final", "request_message_id": "user", "request_parent_message_id": "root"}


@pytest.mark.parametrize("method", ["read_text_request", "read_text", "archive"])
def test_real_session_lock_timeout_does_not_send_and_other_session_advances(local_upstream, method):
    state = local_upstream
    lock = state.accounts.conversation_binding_lock("binding", "blocked")
    lock.acquire()
    failures = []
    def blocked():
        try:
            with operation_deadline(time.monotonic() + .15):
                getattr(ConversationBindingService(), method)(cursor())
        except Exception as exc:
            failures.append(exc)
    worker = threading.Thread(target=blocked)
    worker.start()
    try:
        worker.join(1)
        assert not worker.is_alive()
        assert len(failures) == 1 and isinstance(failures[0], AccountRequestDeadlineExceeded)
        assert not getattr(failures[0], "_original_result_get_attempted", False)
        assert lock.locked() and state.calls == [] and state.backends == []
        # The holder remains blocked; the independent session can still use real HTTP.
        with operation_deadline(time.monotonic() + 20):
            assert ConversationBindingService().read_text_request(cursor("independent"))["content"] == "original answer"
        assert len(state.calls) == 1 and state.backends[-1]._closed
    finally:
        lock.release()
        worker.join(1)
    assert len(state.calls) == 1, "an expired waiter must not send after the holder releases"


@pytest.mark.parametrize("response_error", ["http", "json"])
def test_received_original_get_error_marks_actual_attempt(local_upstream, response_error):
    state = local_upstream
    state.status = 500 if response_error == "http" else 200
    state.invalid_json = response_error == "json"
    with pytest.raises(Exception) as error:
        ConversationBindingService().read_text_request(cursor())
    assert getattr(error.value, "_original_result_get_attempted", False) is True
    assert state.calls == [("GET", "/backend-api/conversation/original")]
    assert state.backends[-1]._closed


def seed_image(tmp_path, *, claim="original-claim"):
    images = ImageTaskService(tmp_path / "images.json", generation_handler=lambda *_: pytest.fail("no generation"))
    row = {"id": "original-image", "owner_id": "owner", "status": "error", "mode": "generate",
           "model": "gpt-image-2", "error_code": "CONVERSATION_OUTCOME_UNKNOWN", "upstream_unfinished": True,
           **cursor(), "result_file_ids": ["original-file"], "_claim_id": claim,
           "next_poll_at": 0, "updated_at": "2026-10-21T00:00:00Z"}
    with images.store.transaction() as db:
        images.store.write_receipt(db, "image", "owner", row["id"], row)
    return images


def image_row(images):
    with images.store.connect() as db:
        return images.store.read_receipt(db, "image", "owner", "original-image")


def run_image(images, seconds):
    images._run_resume_poll("owner:original-image", "original", seconds, "", {"id": "owner"},
                            "generate", "gpt-image-2", False, False)


def test_image_real_lock_wait_obeys_whole_recovery_budget(tmp_path, local_upstream):
    images = seed_image(tmp_path)
    lock = local_upstream.accounts.conversation_binding_lock("binding", "blocked")
    lock.acquire()
    worker = threading.Thread(target=run_image, args=(images, .15))
    worker.start()
    try:
        worker.join(1)
        assert not worker.is_alive() and lock.locked()
        assert local_upstream.calls == []
        row = image_row(images)
        assert row["status"] == "error" and row["error_code"] == "CONVERSATION_OUTCOME_UNKNOWN"
        assert row["result_file_ids"] == ["original-file"] and row.get("recovery_no_result_reads", 0) == 0
    finally:
        lock.release()
        worker.join(1)


@pytest.mark.parametrize("blocked_path", ["/asset", "/backend-api/conversation/original"])
def test_image_http_download_and_confirmation_share_remaining_budget(tmp_path, local_upstream, blocked_path):
    state = local_upstream
    images = seed_image(tmp_path)
    state.delays["/backend-api/files/original-file/download"] = .10
    state.block = ("GET", blocked_path)
    started = time.monotonic()
    run_image(images, .25)
    elapsed = time.monotonic() - started
    assert state.blocked.is_set() and .18 <= elapsed < .9
    row = image_row(images)
    assert row["status"] == "error" and row["error_code"] == "CONVERSATION_OUTCOME_UNKNOWN"
    assert row["result_file_ids"] == ["original-file"]
    assert not row.get("data") and state.backends[-1]._closed
    assert all(method == "GET" for method, _ in state.calls)
    if blocked_path.endswith("original"):
        assert row.get("_pending_image_output"), "downloaded bytes remain private until confirmation"


def test_image_late_worker_cannot_overwrite_replacement_claim(tmp_path, local_upstream):
    state = local_upstream
    images = seed_image(tmp_path)
    state.block = ("GET", "/asset")
    errors = []
    context = SimpleNamespace(kind="image", owner="owner", request_id="original-image", claim="original-claim")
    def run():
        try:
            with executing(context):
                run_image(images, 1)
        except Exception as exc:
            errors.append(exc)
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert state.blocked.wait(1)
        with images.store.transaction() as db:
            row = images.store.read_receipt(db, "image", "owner", "original-image")
            row.update(_claim_id="replacement-claim", status="success", data=[{"b64_json": "bmV3"}])
            images.store.write_receipt(db, "image", "owner", "original-image", row)
        state.release.set()
        worker.join(2)
        assert not worker.is_alive()
        assert image_row(images) == row
        assert state.backends[-1]._closed
    finally:
        state.release.set()
        worker.join(2)


def test_archive_applied_patch_timeout_remains_unknown_then_reads_before_any_retry(runtime, local_upstream):
    service, store, clock, _, _, add = runtime
    original = add("A-1")
    service.update("text", {"id": "one"}, "A-1", "completed", True)
    binding = ConversationBindingService()
    def archive(_owner, rid, desired):
        result = binding.set_archived(cursor("archive"), desired)
        return {"request_id": rid, "archived": result["archived"],
                "conversation": {"client_conversation_id": "A"}}
    service.text.set_public_session_archived = archive
    state = local_upstream
    state.block = ("PATCH", "/backend-api/conversation/original")
    with operation_deadline(time.monotonic() + .25):
        assert service.process_one()
    row = service.get("text", {"id": "one"}, "A-1")
    assert state.archived is True and row["archive"]["status"] == "unknown"
    with store.connect() as db:
        assert store.runtime(db, original["_work_key"])["archive"]["claim"] is None
    assert not service._active_accounts
    assert state.backends[-1]._closed
    state.release.set()
    state.block = None
    clock[0] += 1000
    with operation_deadline(time.monotonic() + 1):
        assert service.process_one()
    assert service.get("text", {"id": "one"}, "A-1")["archive"]["status"] == "confirmed"
    assert [method for method, _ in state.calls] == ["GET", "PATCH", "GET"]


def test_product_multilock_timeout_releases_only_acquired_locks(local_upstream):
    from services.product_work import locking_product_conversations
    accounts = local_upstream.accounts
    first = accounts.conversation_binding_lock("product-archive:account", "a")
    held = accounts.conversation_binding_lock("product-archive:account", "b")
    held.acquire()
    try:
        with operation_deadline(time.monotonic() + .1):
            with pytest.raises(AccountRequestDeadlineExceeded):
                with locking_product_conversations([
                    {"provider_account_identity": "account", "conversation_id": "a"},
                    {"provider_account_identity": "account", "conversation_id": "b"},
                ]):
                    pytest.fail("the blocked second lock cannot be bypassed")
        assert not first.locked() and held.locked()
    finally:
        held.release()


def test_token_refresh_mutex_also_consumes_original_operation_budget():
    account = AccountService.__new__(AccountService)
    account._token_refresh_lock = threading.Lock()
    account._token_refresh_lock.acquire()
    try:
        with operation_deadline(time.monotonic() + .1):
            with pytest.raises(AccountRequestDeadlineExceeded):
                account.refresh_access_token("fixture-token")
        assert account._token_refresh_lock.locked()
    finally:
        account._token_refresh_lock.release()


@pytest.mark.parametrize("success", [False, True])
def test_refresh_deadline_bounds_error_record_but_preserves_rotated_credentials(success):
    account = AccountService.__new__(AccountService)
    account._lock = threading.RLock()
    account._image_slot_condition = threading.Condition(account._lock)
    account._token_refresh_lock = threading.Lock()
    account._accounts = {"old": {"access_token": "old", "refresh_token": "old-refresh"}}
    account._token_aliases, account._image_inflight = {}, {}
    account._normalize_account = lambda value: dict(value)
    account._token_needs_refresh = lambda *args, **kwargs: True
    saved = []
    account._save_accounts = lambda: saved.append(deepcopy(account._accounts))
    response_ready, return_response = threading.Event(), threading.Event()
    results, errors = [], []
    def refresh(*args):
        response_ready.set()
        assert return_response.wait(2)
        if not success:
            raise RuntimeError("fixture upstream refresh failure")
        return {"access_token": "new", "refresh_token": "rotated-refresh"}
    account._request_access_token_refresh = refresh
    def run():
        try:
            with operation_deadline(time.monotonic() + .15):
                results.append(account.refresh_access_token("old", force=True))
        except Exception as exc:
            errors.append(exc)
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert response_ready.wait(1)
        with account._lock:
            return_response.set()
            worker.join(.3)
            assert worker.is_alive() is success
            assert saved == []
    finally:
        return_response.set()
        worker.join(2)
    assert not worker.is_alive()
    if success:
        assert not errors and results == ["new"]
        assert account._token_aliases == {"old": "new"}
        assert saved[-1]["new"]["refresh_token"] == "rotated-refresh"
        assert "old" not in saved[-1]
    else:
        assert len(errors) == 1 and isinstance(errors[0], AccountRequestDeadlineExceeded)
        assert not results and saved == []
        assert account._accounts["old"] == {"access_token": "old", "refresh_token": "old-refresh"}


def test_auth_failure_probe_cannot_outlive_original_read_budget(local_upstream):
    state = local_upstream
    token_lookups = []
    def unavailable(*args, **kwargs):
        token_lookups.append(True)
        raise RuntimeError("bound account is unavailable")
    state.accounts.get_bound_text_access_token = unavailable
    state.accounts._lock.acquire()
    try:
        with operation_deadline(time.monotonic() + .1):
            with pytest.raises(AccountRequestDeadlineExceeded):
                ConversationBindingService().read_text_request(cursor())
        assert state.accounts._lock.locked()
        assert token_lookups == [True]
        assert not state.calls and not state.backends
    finally:
        state.accounts._lock.release()


def test_real_lock_timeout_retires_durable_claim_and_reader_permit(recovery, local_upstream):
    r, state = recovery, local_upstream
    state.accounts.get_bound_account_identity = lambda _binding: "account-0"
    receipt = r.original("locked-original", 0, "original", _submission_started=True)
    lock = state.accounts.conversation_binding_lock(receipt["provider_binding_id"], receipt["client_conversation_id"])
    lock.acquire()
    def read(row):
        with operation_deadline(time.monotonic() + .1):
            return ConversationBindingService().read_text_request(row)
    r.text.recovery_reader = read
    finished = threading.Event()
    def recover(owner, rid):
        try:
            return r.text.read(owner, rid)
        finally:
            finished.set()
    try:
        assert r.admission._dispatch_original_read("text", "owner", "locked-original", recover, receipt)
        assert finished.wait(1)
        until = time.monotonic() + 1
        while r.admission._original_reads and time.monotonic() < until:
            threading.Event().wait(.01)
        assert not r.admission._original_reads
        with r.store.connect() as db:
            saved = r.store.read_receipt(db, "text", "owner", "locked-original")
        assert saved["status"] == "unknown" and saved.get("recovery_claim_id") is None
        assert saved.get("recovery_no_result_reads", 0) == 0
        assert saved["request_message_id"] == receipt["request_message_id"]
        assert "_result_read_capacity_released_at" not in saved
        assert state.calls == [] and lock.locked()
        progressed = threading.Event()
        other = r.original("independent-original", 1, "other")
        assert r.admission._dispatch_original_read("text", "owner", "independent-original",
                                                   lambda *_: progressed.set(), other)
        assert progressed.wait(1)
    finally:
        lock.release()


@pytest.mark.parametrize("failure", ["http", "json", "transport", "pacer", "credential", "candidate_scan"])
def test_actual_get_evidence_reaches_durable_capacity_boundary(recovery, local_upstream, failure):
    r, state = recovery, local_upstream
    state.accounts.get_bound_account_identity = lambda _binding: "account-0"
    original = r.original("read-proof", 0, "original", _submission_started=True,
                          _turn_reserved=False, _executing=False, upstream_outcome="unknown")
    if failure == "http":
        state.status = 500
    elif failure == "json":
        state.invalid_json = True
    elif failure == "transport":
        state.disconnect = True
    elif failure == "credential":
        def unavailable(*_args, **_kwargs):
            raise TimeoutError("credential refresh did not reach conversation GET")
        state.accounts.get_bound_text_access_token = unavailable
    elif failure == "candidate_scan":
        r.text._update("owner", "read-proof", conversation_id=None)

    def read(row):
        # The real pacer rejects insufficient transport budget before GET.
        with operation_deadline(time.monotonic() + (.1 if failure == "pacer" else 20)):
            return ConversationBindingService().read_text_request(row)
    r.text.recovery_reader = read
    assert r.text.read("owner", "read-proof")["status"] == "unknown"
    with r.store.connect() as db:
        saved = r.store.read_receipt(db, "text", "owner", "read-proof")
    actually_read_original = failure in {"http", "json", "transport"}
    if actually_read_original:
        assert state.calls == [("GET", "/backend-api/conversation/original")]
        assert saved["_result_read_capacity_released_at"] == r.clock()
        assert TextTaskService._bounded_unreadable_result_capacity_boundary(saved, r.clock())
        assert r.admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    else:
        assert "_result_read_capacity_released_at" not in saved
        assert not TextTaskService._bounded_unreadable_result_capacity_boundary(saved, r.clock())
        if failure == "candidate_scan":
            assert state.calls and all(urlsplit(path).path == "/backend-api/conversations" for _, path in state.calls)
        else:
            assert state.calls == []
    assert saved["recovery_claim_id"] is None
    assert saved.get("_attempt_finished_at") is None and saved.get("_upstream_terminal") is not True
    assert saved["request_message_id"] == original["request_message_id"]
    assert len(r.queue.calls) == 1  # Only the original submitted fixture; no resend.
