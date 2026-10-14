"""Durable one-dispatch boundary across competing callers and interruptions."""
from copy import deepcopy
import multiprocessing
from pathlib import Path
from unittest.mock import Mock

import pytest

from services.conversation_binding_service import ConversationBindingError
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_bound_text_archive import bound_chat
from test.test_explicit_unknown_successor import task, row, update
from test.test_missing_successor_resume import stopped_successor
from test.test_pool_admission import build


def reserve_worker(root, envelope, ready, start, results):
    service = TextTaskService(Path(root) / "text_tasks.sqlite3", admission=Mock())
    ready.put(True)
    start.wait(10)
    try:
        results.put(service.resume_missing_successor("owner", "successor", envelope)["missing_successor_resume"])
    except ConversationBindingError as exc:
        results.put({"error": exc.code})


def restarted(h, runner=None):
    _, store, admission = build(h.root, h.clock)
    service = TextTaskService(store.path, runner=runner or h.service.runner, clock=h.clock, admission=admission)
    admission.register("text", lambda ctx, body: service._run(ctx.owner, ctx.request_id, body))
    return service, admission


def test_competing_processes_and_restart_reserve_one_dispatch(task):
    h = stopped_successor(task)
    original = deepcopy(row(h.service, "successor"))
    mp = multiprocessing.get_context("spawn")
    ready, results, start = mp.Queue(), mp.Queue(), mp.Event()
    workers = [mp.Process(target=reserve_worker, args=(str(h.root), h.envelope, ready, start, results)) for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        for _ in workers:
            ready.get(timeout=20)
        start.set()
        answers = [results.get(timeout=20) for _ in workers]
        assert answers[0] == answers[1] and answers[0]["state"] == "reserved"
        for worker in workers:
            worker.join(20)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(5)
    service, admission = restarted(h)
    admission.execute(admission.claim_next())
    assert len(h.sent) == 1
    assert h.sent[0]["_request_message_id"] == answers[0]["request_message_id"]
    assert service.resume_missing_successor("owner", "successor", h.envelope)["status"] == "succeeded"
    assert row(service, "successor")["request_message_id"] == original["request_message_id"]
    assert admission.claim_next() is None
    with service.store.connect() as db:
        assert db.execute("SELECT count(*) FROM requests").fetchone()[0] == 2


def test_crash_after_transport_fence_never_sends_on_restart_or_stale_requeue(task):
    h = stopped_successor(task)
    h.service.resume_missing_successor("owner", "successor", h.envelope)
    dispatches = []

    def crash(body, on_cursor):
        current_request.get().before_send()
        dispatches.append(body["_request_message_id"])
        raise SystemExit("simulated process loss")

    h.service.runner = crash
    with pytest.raises(SystemExit):
        h.admission.execute(h.admission.claim_next())
    marker = deepcopy(row(h.service, "successor")["_missing_successor_resume"])
    assert marker["state"] == "submitted"
    h.clock.now += h.admission.CLAIM_SECONDS + 1
    service, admission = restarted(h, runner=Mock(side_effect=AssertionError("must not dispatch")))
    assert admission.claim_next() is None
    assert service.resume_missing_successor("owner", "successor", h.envelope)["missing_successor_resume"] == {
        "attempt": 1, "state": "submitted", "request_message_id": marker["request_message_id"]}
    # A generic stale writer cannot clear the dedicated transport fence merely
    # by putting the old row back in the queue.
    update(service, "successor", status="queued", _claim_id=None, _claim_until=None,
           _submission_started=False, _turn_reserved=False, _executing=False)
    assert admission.claim_next() is None
    assert len(dispatches) == 1
    service.runner.assert_not_called()


@pytest.mark.parametrize("marker", [{}, {"version": 99}, "invalid"])
def test_unrecognized_attempt_marker_is_not_a_fresh_send_permission(task, marker):
    h = stopped_successor(task)
    update(h.service, "successor", _missing_successor_resume=marker)
    before = deepcopy(row(h.service, "successor"))
    with pytest.raises(ConversationBindingError):
        h.service.resume_missing_successor("owner", "successor", h.envelope)
    assert row(h.service, "successor") == before
    assert h.admission.claim_next() is None


def test_expired_result_reader_cannot_finish_over_reserved_attempt(task):
    h = stopped_successor(task)
    update(h.service, "successor", recovery_claim_id="expired-reader", recovery_claimed_at=800,
           recovery_lease_until=900)
    h.service.resume_missing_successor("owner", "successor", h.envelope)
    reserved = deepcopy(row(h.service, "successor"))
    assert not reserved.get("recovery_claim_id")
    h.service._finish_recovery("owner", "successor", "expired-reader", recovered={
        "status": "succeeded", "content": "stale read cannot replace new attempt"})
    assert row(h.service, "successor") == reserved


@pytest.mark.parametrize("wrong_parent", [False, True])
def test_actual_binding_reader_uses_new_attempt_and_its_persisted_parent(task, bound_chat, wrong_parent):
    h = stopped_successor(task)
    h.service.resume_missing_successor("owner", "successor", h.envelope)
    old_id = row(h.service, "successor")["request_message_id"]
    attempt = row(h.service, "successor")["_missing_successor_resume"]["request_message_id"]

    def lost(body, on_cursor):
        on_cursor({"request_parent_message_id": "actual-attempt-parent"})
        current_request.get().before_send()
        raise RuntimeError("lost response")

    h.service.runner = lost
    h.admission.execute(h.admission.claim_next())
    bound_chat.document = {"conversation_id": "original-chat", "is_archived": False,
        "current_node": "unrelated-answer", "mapping": {
            old_id: {"parent": "old-batched-parent", "message": {"id": old_id, "author": {"role": "user"}}},
            "unrelated-answer": {"parent": old_id, "message": {"id": "unrelated-answer", "author": {"role": "assistant"},
                "status": "finished_successfully", "end_turn": True, "content": {"content_type": "text", "parts": ["old late answer"]}}},
            attempt: {"parent": "wrong-parent" if wrong_parent else "actual-attempt-parent",
                "message": {"id": attempt, "author": {"role": "user"}}},
            "attempt-answer": {"parent": attempt, "message": {"id": "attempt-answer", "author": {"role": "assistant"},
                "status": "finished_successfully", "end_turn": True, "content": {"content_type": "text", "parts": ["verified attempt answer"]}}},
        }}
    h.service.recovery_reader = bound_chat.service.read_text_request
    result = h.service.read("owner", "successor")
    saved = row(h.service, "successor")
    assert bound_chat.calls == ["read"]
    assert saved["request_message_id"] == old_id
    assert saved["request_parent_message_id"] == "old-batched-parent"
    if wrong_parent:
        assert result["status"] != "succeeded"
        assert result.get("content") not in {"old late answer", "verified attempt answer"}
        assert saved["recovery_reason"] == "REQUEST_PARENT_MISMATCH"
    else:
        assert result["status"] == "succeeded" and result["content"] == "verified attempt answer"
        assert saved["_missing_successor_resume"]["request_parent_message_id"] == "actual-attempt-parent"
        assert saved["_missing_successor_resume"]["state"] == "recovered"
    assert h.admission.claim_next() is None
