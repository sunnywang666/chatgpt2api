"""A stagnant original turn ends result waiting, never ownership or recovery."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from services.conversation_binding_service import ConversationBindingService
from services.public_chat_service import project_public_chat_receipt
from services.text_task_service import TextTaskService
from test.test_expired_execution_wait import saved, patch, waiter
from test.test_unknown_turn_recovery import document, migration


def incomplete_reader(backend, mutate=None):
    def read(row):
        doc = document(row)
        message = doc["mapping"]["final-user-0"]["message"]
        message.update(status="in_progress", end_turn=None, update_time=10)
        message["content"]["parts"] = [""]
        if mutate:
            mutate(doc, message)
        return ConversationBindingService._read_text_request_result(backend, row, document=doc)
    return read


def stalled(service, admission, backend):
    service.recovery_reader = incomplete_reader(backend)
    for index in range(4):
        if index:
            admission.clock.now += 301
        result = service.read("owner", "old-0")
    return result


def test_stalled_original_ends_wait_persistently_without_release_resend_or_lost_late_result(tmp_path):
    service, admission, backend, legacy = migration(tmp_path)
    patch(service, _submission_started=True, _turn_reserved=True, _executing=False,
          _execution_timeline=[{"stage": "send_call_started", "at": 3},
                               {"stage": "response_headers_received", "at": 4},
                               {"stage": "first_output", "at": 5}, {"stage": "task_finished", "at": 20}])
    result = stalled(service, admission, backend)
    row = saved(service)
    assert result["status"] == "unknown"
    assert row["_result_no_progress_reads"] == 3
    assert row["_result_wait_ended_at"] == admission.clock.now
    assert "_execution_wait_ended_at" not in row
    assert row["_turn_reserved"] is True
    assert row["recovery_no_result_reads"] == 0
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    waiter(service, "next", "client-0")
    assert admission.claim_next() is None
    execution = project_public_chat_receipt(result)["execution"]
    assert execution["phase"] == "stalled"
    assert execution["wait_state"] == "ended"
    assert execution["send_state"] == "response_received"
    assert execution["local_state"] == "ended"
    assert execution["upstream_outcome"] == "unknown"
    assert execution["observation_started_at"] < execution["last_checked_at"]
    assert "last_progress_at" not in execution  # First empty observation is not real progress.
    assert execution["resources"] == {"local_worker": "idle", "account_turn": "held", "conversation": "protected"}
    restarted = TextTaskService(service.path, admission=admission, clock=admission.clock,
                                recovery_reader=incomplete_reader(backend))
    assert restarted.read("owner", "old-0")["execution"] == execution
    assert saved(service)["_result_wait_ended_at"] == row["_result_wait_ended_at"]
    def complete(original):
        doc = document(original)
        doc["mapping"]["final-user-0"]["message"]["content"]["parts"] = ["late original result"]
        return ConversationBindingService._read_text_request_result(backend, original, document=doc)
    restarted.recovery_reader = complete
    admission.clock.now = row["recovery_next_at"] + 1
    result = restarted.read("owner", "old-0")
    assert result["status"] == "succeeded" and result["content"] == "late original result"
    assert result["execution"]["wait_state"] == "completed"
    assert result["execution"]["resources"]["account_turn"] == "released"
    for key in ("request_id", "request_message_id", "provider_account_identity", "conversation_id"):
        assert saved(service)[key] == legacy[0][key]
    assert saved(service)["_result_wait_ended_at"] == row["_result_wait_ended_at"]
    assert backend.mock_calls == []  # The reader receives controlled GET documents; no model POST.


def test_only_original_branch_content_progress_resets_clock_not_heartbeat_or_unrelated_nodes(tmp_path):
    service, admission, backend, _ = migration(tmp_path)
    service.recovery_reader = incomplete_reader(backend)
    service.read("owner", "old-0")
    first = saved(service)["_result_observation_started_at"]
    admission.clock.now += 901
    def unrelated(doc, message):
        message["update_time"] = 99999
        doc["mapping"]["unrelated"] = {"parent": "elsewhere", "message": {"id": "unrelated"}}
    service.recovery_reader = incomplete_reader(backend, unrelated)
    service.read("owner", "old-0")
    assert saved(service)["_result_observation_started_at"] == first
    assert saved(service)["_result_last_progress_at"] is None
    assert saved(service)["_result_no_progress_reads"] == 1
    admission.clock.now += 901
    service.recovery_reader = incomplete_reader(backend, lambda _doc, message: message["content"].update(parts=["new verified text"]))
    service.read("owner", "old-0")
    assert saved(service)["_result_last_progress_at"] == admission.clock.now
    assert saved(service)["_result_no_progress_reads"] == 0
    assert "_result_wait_ended_at" not in saved(service)


@pytest.mark.parametrize("failure", ["transport", "429", "401", "mismatch", "branch", "invalid", "foreign_current"])
def test_unqualified_reads_cannot_count_or_finish_stagnant_wait(tmp_path, failure):
    service, admission, backend, _ = migration(tmp_path)
    service.recovery_reader = incomplete_reader(backend)
    service.read("owner", "old-0")
    original = saved(service)
    admission.clock.now += 901
    def read(row):
        if failure in {"transport", "429", "401"}:
            error = TimeoutError("private diagnostic must not leak")
            error.status_code = {"transport": None, "429": 429, "401": 401}[failure]
            raise error
        def mutate(doc, message):
            if failure == "mismatch":
                doc["mapping"]["user-0"]["parent"] = "different-parent"
            elif failure == "branch":
                doc["mapping"]["sibling"] = copy.deepcopy(doc["mapping"]["final-user-0"])
                doc["mapping"]["sibling"]["message"]["id"] = "sibling"
            elif failure == "foreign_current":
                del doc["mapping"]["user-0"]
            else:
                message["content"]["parts"] = [{"secret": "not text"}]
        return incomplete_reader(backend, mutate)(row)
    service.recovery_reader = read
    service.read("owner", "old-0")
    row = saved(service)
    for key in ("_result_last_checked_at", "_result_last_progress_at", "_result_no_progress_reads"):
        assert row[key] == original[key]
    assert "_result_wait_ended_at" not in row
    # Merely aging persisted evidence on a GET-before-recovery cannot close it.
    patch(service, _result_no_progress_reads=3, recovery_reason="REQUEST_RESULT_INCOMPLETE")
    old = saved(service)
    assert not TextTaskService._end_execution_wait(old, admission.clock.now)


def test_parallel_reader_counts_once_and_old_claim_cannot_overwrite_observation(tmp_path):
    service, admission, backend, _ = migration(tmp_path)
    service.recovery_reader = incomplete_reader(backend)
    service.read("owner", "old-0")
    admission.clock.now += 901
    entered, release = Event(), Event()
    claims = []
    def read(row):
        claims.append(row["recovery_claim_id"])
        entered.set()
        assert release.wait(5)
        return incomplete_reader(backend)(row)
    service.recovery_reader = read
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(service.read, "owner", "old-0")
        assert entered.wait(5)
        service.read("owner", "old-0")
        release.set()
        pending.result(5)
    assert len(claims) == 1
    assert saved(service)["_result_no_progress_reads"] == 1
    before = saved(service)
    service._finish_recovery("owner", "old-0", claims[0], {"content": "stale overwrite"})
    assert saved(service) == before


def test_public_execution_is_allowlisted_and_legacy_send_is_unknown():
    raw = {"request_id": "legacy", "status": "unknown", "finished_at": 1,
           "provider_account_identity": "PRIVATE_ACCOUNT", "_execution_timeline": [{"stage": "PRIVATE_STAGE", "at": 2}],
           "execution": {"send_state": "not_sent", "secret": "PRIVATE_BODY"}}
    receipt = project_public_chat_receipt(TextTaskService._public(raw))
    assert receipt["execution"]["send_state"] == "unknown"
    assert receipt["execution"]["upstream_outcome"] == "unknown"
    assert "PRIVATE" not in json.dumps(receipt)
    receipt = project_public_chat_receipt({"request_id": "x", "execution": {
        "send_state": "response_received", "sent_at": float("inf"), "last_progress_at": True,
        "secret": "PRIVATE_BODY", "resources": {"account_turn": "held", "secret": "PRIVATE_ACCOUNT"}}})
    assert receipt["execution"] == {"send_state": "response_received", "resources": {"account_turn": "held"}}
