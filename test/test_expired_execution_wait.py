"""Bound local Chat waiting without asserting that an unknown upstream turn ended."""
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from services.conversation_binding_service import ConversationBindingService
from services.text_task_service import TextTaskService
from services.pool_admission import ExecutionContext
from services.request_context import AdmissionLost, executing
from test.test_pool_admission import build, claim_worker
from test.test_unknown_turn_recovery import document, migration


def saved(service):
    with service.store.connect() as db:
        return service.store.read_receipt(db, "text", "owner", "old-0")


def patch(service, **changes):
    with service.store.transaction() as db:
        row = service.store.read_receipt(db, "text", "owner", "old-0")
        row.update(changes)
        service.store.write_receipt(db, "text", "owner", "old-0", row)


def missing_reader(backend):
    def read(row):
        doc = document(row)
        del doc["mapping"][row["request_message_id"]]
        return ConversationBindingService._read_text_request_result(backend, row, document=doc)
    return read


def finish_wait(service, admission, backend):
    service.recovery_reader = missing_reader(backend)
    admission.recoveries["text"] = service.read
    for attempt in range(1, 4):
        admission.recover_one()
        row = saved(service)
        assert row["recovery_no_result_reads"] == attempt
        assert row["status"] == ("failed" if attempt == 3 else "unknown")
        admission.clock.now = row["recovery_next_at"] + 1
    return row


def waiter(service, name="waiting", session="independent"):
    return service.submit("owner", {
        "client_request_id": name, "client_conversation_id": session,
        "provider_binding_id": "binding-0", "provider_account_identity": "account-0",
        "model": "fixture-text", "messages": [{"role": "user", "content": "new work"}],
    })


def test_background_read_ends_local_wait_but_preserves_original_and_session_order(tmp_path):
    service, admission, backend, legacy = migration(tmp_path)
    waiter(service)
    waiter(service, "same-session-next", "client-0")
    assert admission.claim_next() is None
    row = finish_wait(service, admission, backend)
    assert row["error_code"] == "RESULT_UNRECOVERABLE"
    assert row["upstream_outcome"] == "unknown"
    assert row["_execution_wait_ended_at"] > 0
    assert row.get("_upstream_terminal") is not True
    for key in ("request_id", "request_message_id", "request_parent_message_id", "conversation_id",
                "client_conversation_id", "provider_account_identity", "provider_binding_id"):
        assert row[key] == legacy[0][key]
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    claimed = admission.claim_next()
    assert claimed.request_id == "waiting"
    assert admission.claim_next() is None
    _, _, restarted = build(tmp_path, admission.clock)
    assert restarted.resource_snapshot()["chat_turn"]["inflight"] == 1
    assert restarted.claim_next() is None
    assert saved(service) == row


@pytest.mark.parametrize("changes", [
    {"created_at": 200}, {"recovery_no_result_reads": 2},
    {"recovery_reason": "REQUEST_RESULT_INCOMPLETE"}, {"recovery_reason": None},
    {"recovery_reason": "REQUEST_BRANCH_AMBIGUOUS"}, {"_executing": True},
    {"_claim_id": "live-worker", "_claim_until": 1100},
    {"_claim_id": "unknown-lease", "_claim_until": None},
    {"recovery_claim_id": "live-reader", "recovery_lease_until": 1100},
    {"request_message_id": None}, {"provider_binding_id": None},
    {"_route": "codex"}, {"_operation": "image"},
    {"_forward_protocol": "openai_v1_chat_completions"},
    {"_recovery_suppressed": True},
])
def test_unqualified_or_active_receipt_does_not_end_local_wait(tmp_path, changes):
    service, admission, _, _ = migration(tmp_path)
    patch(service, recovery_no_result_reads=3, recovery_reason="REQUEST_MESSAGE_NOT_FOUND")
    patch(service, **changes)
    before = saved(service)
    assert TextTaskService._end_execution_wait(before, admission.clock()) is False
    assert "_execution_wait_ended_at" not in before
    admission.claim_next()
    assert saved(service) == before


@pytest.mark.parametrize("status", ["unknown", "failed"])
@pytest.mark.parametrize("executing_flag", [False, True])
def test_existing_qualified_evidence_is_applied_atomically_on_restart(tmp_path, status, executing_flag):
    service, admission, _, _ = migration(tmp_path)
    patch(service, status=status, error_code="RESULT_UNRECOVERABLE" if status == "failed" else "CONVERSATION_OUTCOME_UNKNOWN",
          recovery_no_result_reads=3, recovery_reason="REQUEST_MESSAGE_NOT_FOUND",
          _claim_id="expired-worker", _claim_until=10, _executing=executing_flag)
    waiter(service)
    _, _, restarted = build(tmp_path, admission.clock)
    assert restarted.resource_snapshot()["chat_turn"]["inflight"] == 1
    assert restarted.claim_next().request_id == "waiting"
    assert saved(service)["status"] == "failed"
    assert saved(service)["upstream_outcome"] == "unknown"
    assert restarted.resource_snapshot()["chat_turn"]["inflight"] == 1
    stale = ExecutionContext(admission, "text", "owner", "old-0", "expired-worker")
    with pytest.raises(AdmissionLost):
        admission.update_claim(stale, _claim_until=2000, _executing=True)
    with pytest.raises(AdmissionLost):
        admission.before_send(stale)
    with executing(stale), pytest.raises(AdmissionLost):
        service._update("owner", "old-0", status="running")


@pytest.mark.parametrize("late", ["transport", "incomplete", "success"])
def test_late_read_never_reopens_slot_and_exact_original_success_can_still_be_saved(tmp_path, late):
    service, admission, backend, legacy = migration(tmp_path)
    before = finish_wait(service, admission, backend)
    def read(row):
        if late == "transport":
            raise TimeoutError("synthetic GET timeout")
        doc = document(row)
        answer = doc["mapping"]["final-user-0"]["message"]
        answer["content"]["parts"] = ["original late answer"]
        if late == "incomplete":
            answer["status"] = "in_progress"
        return ConversationBindingService._read_text_request_result(backend, row, document=doc)
    service.recovery_reader = read
    result = service.read("owner", "old-0")
    row = saved(service)
    assert row["_execution_wait_ended_at"] == before["_execution_wait_ended_at"]
    assert row["request_message_id"] == legacy[0]["request_message_id"]
    assert row["provider_account_identity"] == legacy[0]["provider_account_identity"]
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    if late == "success":
        assert result["status"] == "succeeded"
        assert result["content"] == "original late answer"
        assert result["upstream_outcome"] == "completed"
    else:
        assert result["status"] == "failed"
        assert result["upstream_outcome"] == "unknown"


def test_stale_runner_cannot_reopen_wait_or_replace_original_identity(tmp_path):
    service, admission, backend, _ = migration(tmp_path)
    before = finish_wait(service, admission, backend)
    service._update("owner", "old-0", status="running", _turn_reserved=True, conversation_id="wrong-chat")
    assert saved(service) == before
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0


def test_verified_empty_terminal_keeps_existing_correction_contract(tmp_path):
    service, admission, _, _ = migration(tmp_path)
    for _ in range(4):
        result = service.read("owner", "old-0")
        row = saved(service)
        assert result["status"] == "unknown"
        assert row["_upstream_terminal"] is True
        assert "_execution_wait_ended_at" not in row
        assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
        admission.clock.now = row["recovery_next_at"] + 1


def test_same_original_submission_does_not_resend_after_local_failure(tmp_path):
    service, admission, backend, _ = migration(tmp_path)
    body = {"client_request_id": "old-0", "client_conversation_id": "client-0",
            "model": "fixture-text", "messages": [{"role": "user", "content": "original input"}]}
    _, request_hash = service._submission_identity("owner", body)
    with service.store.transaction() as db:
        db.execute("UPDATE requests SET request_hash=? WHERE owner=? AND id=?", (request_hash, "owner", "old-0"))
    before = finish_wait(service, admission, backend)
    assert service.submit("owner", body)["status"] == "failed"
    after = saved(service)
    # Same-ID submission may perform a due original-result GET, but cannot
    # queue another model send or replace identity, outcome, or release time.
    assert after["recovery_no_result_reads"] == before["recovery_no_result_reads"] + 1
    for key in ("request_id", "request_message_id", "request_parent_message_id", "conversation_id",
                "client_conversation_id", "provider_account_identity", "provider_binding_id",
                "status", "error_code", "upstream_outcome", "_execution_wait_ended_at"):
        assert after[key] == before[key]
    assert admission.claim_next() is None


def test_two_processes_release_old_wait_and_claim_new_request_once(tmp_path):
    service, _, _, _ = migration(tmp_path)
    patch(service, recovery_no_result_reads=3, recovery_reason="REQUEST_MESSAGE_NOT_FOUND")
    waiter(service)
    ctx = multiprocessing.get_context("spawn")
    ready, result, start = ctx.Queue(), ctx.Queue(), ctx.Event()
    workers = [ctx.Process(target=claim_worker, args=(str(tmp_path), ready, start, result)) for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        for _ in workers:
            assert ready.get(timeout=15)
        start.set()
        outcomes = [result.get(timeout=15) for _ in workers]
        assert [row[0] for row in outcomes if row] == ["waiting"]
        assert outcomes.count(None) == 1
        assert saved(service)["status"] == "failed"
    finally:
        start.set()
        for worker in workers:
            worker.join(timeout=15)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        for queue in (ready, result):
            queue.close()


def test_competing_recovery_read_counts_once_and_stale_claim_cannot_finish(tmp_path):
    service, admission, backend, _ = migration(tmp_path)
    patch(service, recovery_no_result_reads=2, recovery_reason="REQUEST_MESSAGE_NOT_FOUND")
    started, finish = Event(), Event()
    calls = []
    def read(row):
        calls.append(row["recovery_claim_id"])
        started.set()
        assert finish.wait(5)
        return missing_reader(backend)(row)
    service.recovery_reader = read
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.read, "owner", "old-0")
        assert started.wait(5)
        try:
            # The age/count policy cannot release while the original read owns its claim.
            assert service.read("owner", "old-0")["status"] == "unknown"
            assert admission.claim_next() is None
            assert "_execution_wait_ended_at" not in saved(service)
        finally:
            finish.set()
        assert first.result()["status"] == "failed"
    assert len(calls) == 1
    before = saved(service)
    service._finish_recovery("owner", "old-0", calls[0], {"content": "stale"})
    assert saved(service) == before
    assert before["recovery_no_result_reads"] == 3
