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


def mixed_chain(doc):
    """Synthetic bodies reproduce only the live GET's verified envelope shapes."""
    final = doc["current_node"]
    parent = doc["mapping"][final]["parent"]
    envelopes = [
        ("code", "assistant", {"content_type": "code", "text": "PRIVATE_CODE"}),
        ("output", "tool", {"content_type": "execution_output", "text": "PRIVATE_TOOL"}),
        ("thought", "assistant", {"content_type": "thoughts", "thoughts": [
            {"summary": "PRIVATE_SUMMARY", "content": "", "chunks": [], "finished": False}]}),
        ("recap", "assistant", {"content_type": "reasoning_recap", "content": "PRIVATE_RECAP"}),
    ]
    for suffix, role, content in envelopes:
        node_id = final + "-" + suffix
        doc["mapping"][node_id] = {"parent": parent, "message": {
            "id": node_id, "author": {"role": role}, "status": "finished_successfully",
            "end_turn": False, "content": content}}
        parent = node_id
    doc["mapping"][final]["parent"] = parent


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


@pytest.mark.parametrize("part", ["code", "output", "thought", "recap"])
def test_mixed_original_chain_observes_progress_then_stall_without_exposing_bodies(tmp_path, part):
    service, admission, backend, _ = migration(tmp_path)
    def shape(doc, message):
        mixed_chain(doc)
        content = doc["mapping"][doc["current_node"] + "-" + part]["message"]["content"]
        if progressed[0]:
            if part == "thought":
                content["thoughts"][0]["finished"] = True
            else:
                content["content" if part == "recap" else "text"] += " growth"
    progressed = [False]
    service.recovery_reader = incomplete_reader(backend, shape)
    result = service.read("owner", "old-0")
    assert result["execution"]["unchanged_reads"] == 0
    assert "PRIVATE" not in json.dumps(saved(service))
    progressed[0] = True
    admission.clock.now += 901
    service.read("owner", "old-0")
    assert saved(service)["_result_last_progress_at"] == admission.clock.now
    for _ in range(3):
        admission.clock.now += 301
        result = service.read("owner", "old-0")
    assert result["execution"]["phase"] == "stalled"
    assert result["execution"]["resources"]["account_turn"] == "held"
    assert "PRIVATE" not in json.dumps(result)
    assert all(key not in json.dumps(result) for key in ("text_chars", "content_items", "finished_items", "nodes"))


@pytest.mark.parametrize("malformed", ["media", "object_output", "unknown_thought_chunk", "later_user", "cycle", "kind_change"])
def test_mixed_observation_rejects_unsupported_or_ambiguous_branch(tmp_path, malformed):
    service, admission, backend, _ = migration(tmp_path)
    def mutate(doc, message):
        mixed_chain(doc)
        output = doc["mapping"][doc["current_node"] + "-output"]["message"]
        if malformed == "media":
            output["content"] = {"content_type": "image", "parts": ["PRIVATE_IMAGE"]}
        elif malformed == "object_output":
            output["content"]["text"] = {"PRIVATE": "BODY"}
        elif malformed == "unknown_thought_chunk":
            thought = doc["mapping"][doc["current_node"] + "-thought"]["message"]["content"]["thoughts"][0]
            thought["chunks"] = [{"PRIVATE": "BODY"}]
        elif malformed == "later_user":
            output["author"]["role"] = "user"
        elif malformed == "cycle":
            doc["mapping"]["user-0"]["parent"] = doc["current_node"]
        else:
            output["content"]["content_type"] = []
    service.recovery_reader = incomplete_reader(backend, mutate)
    result = service.read("owner", "old-0")
    assert "observation_started_at" not in result["execution"]
    assert "_result_wait_ended_at" not in saved(service)


def test_pure_text_observation_upgrade_preserves_progress_clock(tmp_path):
    service, admission, backend, _ = migration(tmp_path)
    service.recovery_reader = incomplete_reader(backend)
    service.read("owner", "old-0")
    original = saved(service)
    observation = original["_original_result_observation"]
    for node in observation["nodes"]:
        for key in ("content_type", "content_items", "finished_items"):
            node.pop(key)
    patch(service, _original_result_observation=observation)
    admission.clock.now += 301
    service.read("owner", "old-0")
    row = saved(service)
    assert row["_result_observation_started_at"] == original["_result_observation_started_at"]
    assert row["_result_last_progress_at"] is None
    assert row["_result_no_progress_reads"] == 1
