"""Migration shapes are synthetic; only an exact original GET can end occupancy."""
import copy
import json
import sqlite3
from unittest.mock import Mock

import pytest

from services.conversation_binding_service import ConversationBindingService, TURN_END_EVIDENCE_FIELD
from services.text_task_service import TextTaskService
from test.test_pool_admission import build, Clock


@pytest.mark.parametrize("case", ["proven", "missing_stage", "bootstrap", "timeout", "stream_event", "5xx", "image", "codex"])
def test_only_exact_original_chat_422_rejection_excludes_legacy_turn(tmp_path, case):
    service, admission, backend, legacy = migration(tmp_path)
    evidence = {"original_failure_phase": "stream_open", "original_exception_category": "http",
                "original_upstream_request_stage": "conversation", "original_http_status": 422}
    if case == "missing_stage":
        evidence.pop("original_upstream_request_stage")
    elif case == "bootstrap":
        evidence["original_upstream_request_stage"] = "bootstrap"
    elif case == "timeout":
        evidence["original_exception_category"] = "timeout"
    elif case == "stream_event":
        evidence["original_failure_phase"] = "stream_event"
    elif case == "5xx":
        evidence["original_http_status"] = 503
    elif case == "image":
        evidence["_forward_protocol"] = "openai_v1_image_generations"
    elif case == "codex":
        evidence["_route"] = "codex"
    service._update("owner", legacy[0]["request_id"], **evidence)
    resource = "codex" if case == "codex" else "chat_turn"
    assert admission.resource_snapshot()[resource]["inflight"] == (0 if case == "proven" else 1)
    with service.store.connect() as db:
        saved = service.store.read_receipt(db, "text", "owner", legacy[0]["request_id"])
    assert saved["status"] == "unknown"
    assert saved["request_message_id"] == legacy[0]["request_message_id"]
    assert saved.get("upstream_outcome") != "not_sent"


def document(receipt):
    user = receipt["request_message_id"]
    final = "final-" + user
    return {"conversation_id": receipt["conversation_id"], "current_node": final, "mapping": {
        user: {"parent": "parent", "message": {"id": user, "author": {"role": "user"}}},
        final: {"parent": user, "message": {"id": final, "author": {"role": "assistant"},
            "status": "finished_successfully", "end_turn": True, "channel": "final",
            "content": {"content_type": "text", "parts": []}}},
    }}


def migration(tmp_path, count=1):
    rows = [{"access_token": "fixture-" + str(i), "account_id": "physical-" + str(i),
             "provider_account_identity": "account-" + str(i), "type": "Plus", "status": "正常",
             "quota": 99, "source_type": "web", "conversation_binding_ids": ["binding-" + str(i)]}
            for i in range(2)]
    (tmp_path / "accounts.json").write_text(json.dumps(rows))
    legacy = [{"request_id": "old-" + str(i), "status": "unknown", "boot": "legacy-boot",
               "created_at": 1, "finished_at": 2, "model": "fixture-text",
               "provider_binding_id": "binding-" + str(i % 2),
               "provider_account_identity": "account-" + str(i % 2),
               "client_conversation_id": "client-" + str(i), "conversation_id": "conversation-" + str(i),
               "request_message_id": "user-" + str(i), "request_parent_message_id": "parent",
               "error_code": "CONVERSATION_OUTCOME_UNKNOWN", "upstream_outcome": "unknown"}
              for i in range(count)]
    with sqlite3.connect(tmp_path / "text_tasks.sqlite3") as db:
        db.execute("CREATE TABLE requests (owner TEXT,id TEXT,request_hash TEXT,receipt TEXT,PRIMARY KEY(owner,id))")
        for row in legacy:
            db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("owner", row["request_id"], "original-hash-" + row["request_id"], json.dumps(row)))
    clock = Clock()
    accounts, store, admission = build(tmp_path, clock)
    with store.connect() as db:
        assert [json.loads(raw[0]) for raw in db.execute("SELECT receipt FROM requests ORDER BY id")] == legacy
    backend = Mock()
    def original_reader(row):
        backend._get_conversation.return_value = document(row)
        return ConversationBindingService._read_text_request_result(backend, row)
    service = TextTaskService(store.path, admission=admission, clock=clock, recovery_reader=original_reader)
    return service, admission, backend, legacy


def test_nine_legacy_unknowns_keep_results_and_order_but_release_proven_ended_turns(tmp_path):
    service, admission, backend, legacy = migration(tmp_path, 9)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 9
    assert admission.resource_snapshot()["chat_turn"]["slots_free"] == 0
    before_accounts = (tmp_path / "accounts.json").read_bytes()
    for row in legacy:
        recovered = service.read("owner", row["request_id"])
        assert recovered["status"] == "unknown"
        assert recovered["upstream_outcome"] == "unknown"
        assert recovered["request_message_id"] == row["request_message_id"]
        assert recovered["provider_account_identity"] == row["provider_account_identity"]
        assert "content" not in recovered
    assert (tmp_path / "accounts.json").read_bytes() == before_accounts
    assert backend._get_conversation.call_count == 9
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert admission.resource_snapshot()["chat_turn"]["dispatchable_now"] == 2
    # A free physical account is not permission to repeat the unresolved old turn
    # or advance that same conversation past its unresolved result.
    service.submit("owner", {"client_request_id": "blocked-next", "client_conversation_id": "client-0",
                             "model": "fixture-text", "messages": [{"role": "user", "content": "next"}]})
    assert admission.claim_next() is None
    service.submit("other", {"client_request_id": "waiting", "client_conversation_id": "new-client",
                             "model": "fixture-text", "messages": [{"role": "user", "content": "original new input"}]})
    assert admission.claim_next().request_id == "waiting"
    # Independent restart sees the evidence; it cannot reopen the nine turns.
    _, _, restarted = build(tmp_path, admission.clock)
    assert restarted.resource_snapshot()["chat_turn"]["inflight"] == 1


@pytest.mark.parametrize("recovery", ["transport", "ended_empty", "completed"])
def test_unrecoverable_result_is_not_proof_of_an_ended_model_turn(tmp_path, recovery):
    service, admission, backend, legacy = migration(tmp_path)
    original_id = legacy[0]["request_id"]
    service._update("owner", original_id, status="failed", error_code="RESULT_UNRECOVERABLE",
                    upstream_outcome="unknown", recovery_retryable=True)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    service.submit("owner", {"client_request_id": "next", "client_conversation_id": "client-0",
                             "model": "fixture-text", "messages": []})
    assert admission.claim_next() is None
    def read(receipt):
        if recovery == "transport":
            raise ConnectionError("synthetic read failure")
        doc = document(receipt)
        if recovery == "completed":
            doc["mapping"]["final-user-0"]["message"]["content"]["parts"] = ["original completed result"]
        return ConversationBindingService._read_text_request_result(backend, receipt, document=doc)
    service.recovery_reader = read
    admission.recoveries["text"] = service.read
    admission.recover_one()
    with service.store.connect() as db:
        actual = service.store.read_receipt(db, "text", "owner", original_id)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == (1 if recovery == "transport" else 0)
    assert actual["request_message_id"] == legacy[0]["request_message_id"]
    assert actual["provider_account_identity"] == legacy[0]["provider_account_identity"]
    if recovery == "completed":
        assert actual["status"] == "succeeded"
        assert actual["upstream_outcome"] == "completed"
        assert actual["recovery_retryable"] is False
    else:
        assert actual["status"] == "failed" and actual["upstream_outcome"] == "unknown"
        assert admission.claim_next() is None


@pytest.mark.parametrize("case", ["active", "no_result", "missing_user", "wrong_parent", "branch", "missing_status", "not_end_turn", "transport"])
def test_age_missing_or_ambiguous_result_never_proves_idle(tmp_path, case):
    service, admission, backend, legacy = migration(tmp_path)
    doc = document(legacy[0])
    mapping = doc["mapping"]
    final = mapping["final-user-0"]["message"]
    if case == "active":
        final["status"] = "in_progress"
    elif case == "no_result":
        del mapping["final-user-0"]
    elif case == "missing_user":
        del mapping["user-0"]
    elif case == "wrong_parent":
        mapping["user-0"]["parent"] = "other-parent"
    elif case == "branch":
        mapping["sibling"] = copy.deepcopy(mapping["final-user-0"])
        mapping["sibling"]["message"]["id"] = "sibling"
    elif case == "missing_status":
        final.pop("status")
    elif case == "not_end_turn":
        final["end_turn"] = False
    backend._get_conversation.return_value = doc
    if case == "transport":
        backend._get_conversation.side_effect = TimeoutError()
    service.recovery_reader = lambda row: ConversationBindingService._read_text_request_result(backend, row)
    admission.clock.now += 1_000_000
    result = service.read("owner", "old-0")
    assert result["status"] == "unknown"
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    with admission.store.connect() as db:
        row = admission.store.read_receipt(db, "text", "owner", "old-0")
    assert row.get("_upstream_terminal") is not True


@pytest.mark.parametrize("field,value", [("conversation_id", "wrong"), ("request_message_id", "other-user"),
                                        ("final_message_id", "user-0"), ("observed_at", float("nan"))])
def test_terminal_evidence_must_match_the_original_request(tmp_path, field, value):
    service, admission, backend, legacy = migration(tmp_path)
    recovered = ConversationBindingService._read_text_request_result(backend, legacy[0], document=document(legacy[0]))
    recovered[TURN_END_EVIDENCE_FIELD][field] = value
    service.recovery_reader = lambda row: recovered
    assert service.read("owner", "old-0")["recovery_error_code"] == "RECOVERY_INVALID_RESULT"
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
