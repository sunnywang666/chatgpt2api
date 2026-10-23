"""Migration shapes are synthetic; only an exact original GET can end occupancy."""
import copy
import json
import sqlite3
import time
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


@pytest.mark.parametrize("case", ["active_sibling", "empty_sibling", "analysis_sibling", "active_child", "later_user"])
def test_completed_answer_needs_original_branch_terminal_proof(tmp_path, case):
    service, admission, backend, legacy = migration(tmp_path)
    doc = document(legacy[0])
    mapping = doc["mapping"]
    mapping["final-user-0"]["message"]["content"]["parts"] = ["original answer"]
    parent = "final-user-0" if case in {"active_child", "later_user"} else "user-0"
    if case == "later_user":
        mapping["later-user"] = {"parent": parent, "message": {"id": "later-user", "author": {"role": "user"}}}
        parent = "later-user"
    mapping["other"] = {"parent": parent, "message": {
        "id": "other", "author": {"role": "assistant"},
        "status": "finished_successfully" if case in {"empty_sibling", "analysis_sibling"} else "in_progress",
        "end_turn": case == "empty_sibling", "channel": "analysis" if case == "analysis_sibling" else "final",
        "content": {"content_type": "text", "parts": [] if case == "empty_sibling" else ["other output"]},
    }}
    doc["current_node"] = "other"
    backend._get_conversation.return_value = doc
    service.recovery_reader = lambda row: ConversationBindingService._read_text_request_result(backend, row)
    result = service.read("owner", "old-0")
    proven = case == "later_user"
    assert result["status"] == ("succeeded" if proven else "unknown")
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == (0 if proven else 1)
    assert result.get("content") == ("original answer" if proven else None)
    for key in ("request_id", "request_message_id", "provider_binding_id", "provider_account_identity", "client_conversation_id"):
        assert result[key] == legacy[0][key]
    assert backend._get_conversation.call_count == 1
    assert not backend.session.post.called


@pytest.mark.parametrize("field,value", [("conversation_id", "wrong"), ("request_message_id", "other-user"),
                                        ("final_message_id", "user-0"), ("observed_at", float("nan"))])
def test_terminal_evidence_must_match_the_original_request(tmp_path, field, value):
    service, admission, backend, legacy = migration(tmp_path)
    recovered = ConversationBindingService._read_text_request_result(backend, legacy[0], document=document(legacy[0]))
    recovered[TURN_END_EVIDENCE_FIELD][field] = value
    service.recovery_reader = lambda row: recovered
    assert service.read("owner", "old-0")["recovery_error_code"] == "RECOVERY_INVALID_RESULT"
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1


@pytest.mark.parametrize("expired_sender", [False, True])
def test_exact_recovered_final_retires_original_turn_after_sender_exits_or_expires(tmp_path, expired_sender):
    service, _, backend, legacy = migration(tmp_path)
    original = legacy[0]
    service._update(
        "owner", original["request_id"], _submission_started=True, _turn_reserved=True,
        _executing=expired_sender, _claim_id="old-sender", _claim_until=10 if expired_sender else 2000,
    )
    doc = document(original)
    doc["mapping"]["final-" + original["request_message_id"]]["message"]["content"]["parts"] = ["saved original final"]
    service.recovery_reader = lambda row: ConversationBindingService._read_text_request_result(
        backend, row, document=doc,
    )

    result = service.read("owner", original["request_id"])
    assert result["status"] == "succeeded" and result["content"] == "saved original final"
    with service.store.connect() as db:
        saved = service.store.read_receipt(db, "text", "owner", original["request_id"])
    assert saved["_turn_reserved"] is False
    assert saved[TURN_END_EVIDENCE_FIELD]["request_message_id"] == original["request_message_id"]
    if expired_sender:
        assert saved["_executing"] is False
        assert saved["_claim_id"] is None and saved["_claim_until"] is None
    else:
        # A sender which already ran its finally block keeps only its normal
        # attribution claim; it is no longer a turn owner.
        assert saved["_executing"] is False and saved["_claim_id"] == "old-sender"
    restarted = TextTaskService(service.store.path, admission=service.admission, clock=service.clock,
                                recovery_reader=lambda _row: (_ for _ in ()).throw(AssertionError("no reread")))
    assert restarted.read("owner", original["request_id"])["status"] == "succeeded"


def test_paused_recovery_success_keeps_original_turn_reserved(tmp_path):
    service, _, backend, legacy = migration(tmp_path)
    original = legacy[0]
    with service.store.transaction() as db:
        saved = service.store.read_receipt(db, "text", "owner", original["request_id"])
        saved.update(_submission_started=True, _turn_reserved=True, _executing=False,
                     recovery_claim_id="current-reader", recovery_claimed_at=900,
                     recovery_lease_until=1100, recovery_phase="read_text_request",
                     _recovery_paused=True)
        service.store.write_receipt(db, "text", "owner", original["request_id"], saved)
    doc = document(original)
    doc["mapping"]["final-" + original["request_message_id"]]["message"]["content"]["parts"] = ["saved final"]
    recovered = ConversationBindingService._read_text_request_result(backend, original, document=doc)
    safe, error, phase, reason = service._safe_recovery_result(recovered, original)
    assert error is None
    assert service._finish_recovery("owner", original["request_id"], "current-reader", safe, error, phase, reason)["status"] == "succeeded"
    with service.store.connect() as db:
        saved = service.store.read_receipt(db, "text", "owner", original["request_id"])
    assert saved["_turn_reserved"] is True and saved["_recovery_paused"] is True


def test_existing_saved_success_rechecks_exact_original_before_retiring_residual_turn(tmp_path):
    service, _, backend, legacy = migration(tmp_path)
    original = legacy[0]
    service._update(
        "owner", original["request_id"], status="succeeded", content="exact original final",
        upstream_outcome="completed", error_code=None, finished_at=900,
        _submission_started=True, _turn_reserved=True, _executing=False,
        _claim_id="retained-attribution", _claim_until=2000,
        _attempt_finished_at=850,
        parent_message_id="final-" + original["request_message_id"],
    )
    doc = document(original)
    doc["mapping"]["final-" + original["request_message_id"]]["message"]["content"]["parts"] = ["exact original final"]
    service.recovery_reader = lambda row: ConversationBindingService._read_text_request_result(
        backend, row, document=doc,
    )

    repaired = service.read("owner", original["request_id"])
    assert repaired["status"] == "succeeded" and repaired["content"] == "exact original final"
    with service.store.connect() as db:
        saved = service.store.read_receipt(db, "text", "owner", original["request_id"])
    assert saved["_turn_reserved"] is False
    assert saved[TURN_END_EVIDENCE_FIELD]["request_message_id"] == original["request_message_id"]


@pytest.mark.parametrize("initial_status", ["unknown", "succeeded"])
def test_historical_terminal_proof_cannot_replace_this_read_proof(tmp_path, initial_status):
    service, _, backend, legacy = migration(tmp_path)
    original = legacy[0]
    final = "final-" + original["request_message_id"]
    historical = {
        "conversation_id": original["conversation_id"],
        "request_message_id": original["request_message_id"],
        "final_message_id": final,
        "observed_at": 800,
    }
    service._update(
        "owner", original["request_id"], status=initial_status,
        content="saved original final" if initial_status == "succeeded" else None,
        upstream_outcome="completed" if initial_status == "succeeded" else "unknown",
        error_code=None if initial_status == "succeeded" else "CONVERSATION_OUTCOME_UNKNOWN",
        finished_at=900 if initial_status == "succeeded" else None,
        _submission_started=True, _turn_reserved=True, _executing=False,
        _attempt_finished_at=850 if initial_status == "succeeded" else None,
        parent_message_id=final,
        **{TURN_END_EVIDENCE_FIELD: historical},
    )
    # This is a nonempty recovered answer, but it deliberately lacks the
    # current exact-branch proof.  A historical receipt proof cannot retire
    # either the UNKNOWN or already-succeeded reservation.
    service.recovery_reader = lambda row: {
        "status": "succeeded", "binding_status": "bound",
        "conversation_id": row["conversation_id"], "parent_message_id": final,
        "content": "saved original final",
    }

    result = service.read("owner", original["request_id"])
    assert result["status"] == "succeeded" and result["content"] == "saved original final"
    with service.store.connect() as db:
        saved = service.store.read_receipt(db, "text", "owner", original["request_id"])
    assert saved["_turn_reserved"] is True
    assert saved[TURN_END_EVIDENCE_FIELD] == historical


@pytest.mark.parametrize("case", ["live_sender", "newer_recovery_claim", "wrong_identity", "content_mismatch", "parent_mismatch"])
def test_nonempty_recovered_final_never_retires_protected_or_mismatched_turn(tmp_path, case):
    service, _, backend, legacy = migration(tmp_path)
    original = legacy[0]
    final = "final-" + original["request_message_id"]
    service._update(
        "owner", original["request_id"], _submission_started=True, _turn_reserved=True,
        _executing=case == "live_sender", _claim_id="live-sender" if case == "live_sender" else None,
        _claim_until=2000 if case == "live_sender" else None,
        status="succeeded" if case in {"content_mismatch", "parent_mismatch"} else "unknown",
        content="saved content" if case == "content_mismatch" else "fresh recovered final" if case == "parent_mismatch" else None,
        upstream_outcome="completed" if case in {"content_mismatch", "parent_mismatch"} else "unknown",
        error_code=None if case in {"content_mismatch", "parent_mismatch"} else "CONVERSATION_OUTCOME_UNKNOWN",
        finished_at=900 if case in {"content_mismatch", "parent_mismatch"} else None,
        parent_message_id="older-final" if case == "parent_mismatch" else final if case == "content_mismatch" else None,
    )
    doc = document(original)
    doc["mapping"][final]["message"]["content"]["parts"] = ["fresh recovered final"]
    if case == "wrong_identity":
        doc["mapping"][final]["message"]["id"] = "wrong-final"
    if case == "newer_recovery_claim":
        def reader(row):
            with service.store.transaction() as db:
                saved = service.store.read_receipt(db, "text", "owner", original["request_id"])
                saved.update(recovery_claim_id="newer-reader", recovery_claimed_at=900,
                             recovery_lease_until=1100)
                service.store.write_receipt(db, "text", "owner", original["request_id"], saved)
            return ConversationBindingService._read_text_request_result(backend, row, document=doc)
        service.recovery_reader = reader
    else:
        service.recovery_reader = lambda row: ConversationBindingService._read_text_request_result(
            backend, row, document=doc,
        )

    result = service.read("owner", original["request_id"])
    with service.store.connect() as db:
        saved = service.store.read_receipt(db, "text", "owner", original["request_id"])
    assert saved["_turn_reserved"] is True
    if case == "newer_recovery_claim":
        assert result["status"] == "unknown" and saved["recovery_claim_id"] == "newer-reader"
    elif case == "content_mismatch":
        assert result["status"] == "succeeded" and result["content"] == "saved content"
        assert saved["parent_message_id"] == final
    elif case == "parent_mismatch":
        assert result["status"] == "succeeded" and saved["parent_message_id"] == "older-final"
    else:
        assert result["status"] == ("succeeded" if case == "live_sender" else "unknown")


def test_existing_saved_success_without_exact_final_proof_keeps_residual_turn(tmp_path):
    service, _, backend, legacy = migration(tmp_path)
    original = legacy[0]
    service._update(
        "owner", original["request_id"], status="succeeded", content="legacy saved content",
        upstream_outcome="completed", error_code=None, finished_at=900,
        _submission_started=True, _turn_reserved=True, _executing=False,
    )
    doc = document(original)
    doc["mapping"]["final-" + original["request_message_id"]]["message"]["content"]["parts"] = []
    service.recovery_reader = lambda row: ConversationBindingService._read_text_request_result(
        backend, row, document=doc,
    )

    assert service.read("owner", original["request_id"])["status"] == "succeeded"
    with service.store.connect() as db:
        saved = service.store.read_receipt(db, "text", "owner", original["request_id"])
    assert saved["_turn_reserved"] is True


@pytest.mark.parametrize("entrypoint", ["ordinary", "explicit", "scanner", "scanner_background"])
def test_expired_saved_success_reader_lease_is_fenced_by_all_recovery_entrypoints(tmp_path, entrypoint):
    service, admission, backend, legacy = migration(tmp_path)
    original = legacy[0]
    now = service.clock.now
    final = "final-" + original["request_message_id"]
    service._update(
        "owner", original["request_id"], status="succeeded", content="legacy saved content",
        upstream_outcome="completed", error_code=None, finished_at=now - 100,
        _submission_started=True, _turn_reserved=True, _executing=False,
        parent_message_id=final,
    )
    class ReaderProcessDied(BaseException):
        pass
    service.recovery_reader = lambda _receipt: (_ for _ in ()).throw(ReaderProcessDied())
    # This is an actual ordinary compatibility read: it durably claims the
    # original, then its process dies before it can call _finish_recovery.
    with pytest.raises(ReaderProcessDied):
        service.read("owner", original["request_id"])
    with service.store.connect() as db:
        crashed = service.store.read_receipt(db, "text", "owner", original["request_id"])
    assert crashed["recovery_claim_id"] and crashed["recovery_phase"] == "read_text_request"
    crashed_claim = crashed["recovery_claim_id"]
    service.clock.now = crashed["recovery_lease_until"] + 1
    # Reopen the same SQLite file after the old reader has died. Its late
    # completion cannot change the newer fenced claim created by this read.
    reopened = TextTaskService(service.store.path, admission=admission, clock=service.clock)
    reads = []
    def reader(receipt):
        reads.append(receipt["recovery_claim_id"])
        late_doc = document(receipt)
        late_doc["mapping"][final]["message"]["content"]["parts"] = ["old reader result"]
        late = ConversationBindingService._read_text_request_result(backend, receipt, document=late_doc)
        reopened._finish_recovery("owner", original["request_id"], crashed_claim, late)
        with reopened.store.connect() as db:
            assert reopened.store.read_receipt(db, "text", "owner", original["request_id"])["recovery_claim_id"] == reads[0]
        fresh_doc = document(receipt)
        fresh_doc["mapping"][final]["message"]["content"]["parts"] = ["legacy saved content"]
        return ConversationBindingService._read_text_request_result(backend, receipt, document=fresh_doc)
    reopened.recovery_reader = reader

    if entrypoint == "ordinary":
        result = reopened.read("owner", original["request_id"])
    elif entrypoint == "explicit":
        result = reopened.recover("owner", original["request_id"])
    else:
        admission.recoveries["text"] = reopened.read
        admission.recover_one(background=entrypoint == "scanner_background")
        if entrypoint == "scanner_background":
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with reopened.store.connect() as db:
                    current = reopened.store.read_receipt(db, "text", "owner", original["request_id"])
                if current.get("status") == "succeeded" and current.get("_turn_reserved") is False:
                    break
                time.sleep(0.01)
        result = reopened.read_snapshot("owner", original["request_id"])

    assert result["status"] == "succeeded" and result["content"] == "legacy saved content"
    assert len(reads) == 1 and reads[0] != crashed_claim
    with reopened.store.connect() as db:
        saved = reopened.store.read_receipt(db, "text", "owner", original["request_id"])
    assert saved["_turn_reserved"] is False
    assert saved.get("recovery_claim_id") is None
    assert saved[TURN_END_EVIDENCE_FIELD]["final_message_id"] == final


@pytest.mark.parametrize("field,value", [
    ("_claim_id", "different-sender"), ("content", "late changed content"),
    ("parent_message_id", "late-final"), ("provider_binding_id", "late-binding"),
    ("provider_account_identity", "late-account"), ("client_conversation_id", "late-client"),
])
def test_sender_finally_marker_rejects_late_claim_content_cursor_or_identity_change(field, value):
    receipt = {
        "status": "succeeded", "content": "exact saved final", "upstream_outcome": "completed",
        "error_code": None, "_route": "chat", "_operation": "text", "_submission_started": True,
        "_turn_reserved": True, "_executing": False, "_claim_id": "sender-claim", "_claim_until": 2000,
        "provider_binding_id": "binding", "provider_account_identity": "account", "client_conversation_id": "client",
        "conversation_id": "conversation", "request_message_id": "request", "parent_message_id": "final",
        TURN_END_EVIDENCE_FIELD: {
            "conversation_id": "conversation", "request_message_id": "request",
            "final_message_id": "final", "observed_at": 1000,
        },
    }
    recovered = {
        "provider_binding_id": "binding", "provider_account_identity": "account", "client_conversation_id": "client",
        TURN_END_EVIDENCE_FIELD: dict(receipt[TURN_END_EVIDENCE_FIELD]),
    }
    receipt["_recovered_final_sender"] = TextTaskService._recovered_final_sender_marker(receipt, recovered)
    receipt[field] = value
    assert not TextTaskService.retire_recovered_final_after_sender_exit(receipt, "sender-claim")
    assert receipt["_turn_reserved"] is True


def test_marker_saved_before_sender_finally_requires_a_fresh_read_after_sender_crash(tmp_path):
    service, _, backend, legacy = migration(tmp_path)
    original = legacy[0]
    final = "final-" + original["request_message_id"]
    service._update(
        "owner", original["request_id"], _submission_started=True, _turn_reserved=True,
        _executing=True, _claim_id="crashed-sender", _claim_until=service.clock.now + 10,
    )
    reads = []
    def reader(receipt):
        reads.append(receipt["request_id"])
        upstream = document(receipt)
        upstream["mapping"][final]["message"]["content"]["parts"] = ["exact original final"]
        return ConversationBindingService._read_text_request_result(backend, receipt, document=upstream)
    service.recovery_reader = reader

    assert service.read("owner", original["request_id"])["content"] == "exact original final"
    with service.store.connect() as db:
        saved = service.store.read_receipt(db, "text", "owner", original["request_id"])
    assert saved["_turn_reserved"] is True and saved["_executing"] is True
    assert saved["_recovered_final_sender"]["claim_id"] == "crashed-sender"

    service.clock.now = saved["_claim_until"] + 1
    reopened = TextTaskService(service.store.path, admission=service.admission, clock=service.clock,
                               recovery_reader=reader)
    assert reopened.read("owner", original["request_id"])["content"] == "exact original final"
    with reopened.store.connect() as db:
        recovered = reopened.store.read_receipt(db, "text", "owner", original["request_id"])
    assert reads == [original["request_id"], original["request_id"]]
    assert recovered["_turn_reserved"] is False and recovered["_executing"] is False
    assert recovered["_claim_id"] is None and "_recovered_final_sender" not in recovered


@pytest.mark.parametrize("block", ["missing_binding", "missing_account", "missing_client", "reader_lease", "paused", "suppressed"])
def test_sender_finally_marker_keeps_reserved_for_missing_identity_or_protected_recovery(block):
    receipt = {
        "status": "succeeded", "content": "exact saved final", "upstream_outcome": "completed",
        "error_code": None, "_route": "chat", "_operation": "text", "_submission_started": True,
        "_turn_reserved": True, "_executing": False, "_claim_id": "sender-claim", "_claim_until": 2000,
        "provider_binding_id": "binding", "provider_account_identity": "account", "client_conversation_id": "client",
        "conversation_id": "conversation", "request_message_id": "request", "parent_message_id": "final",
        TURN_END_EVIDENCE_FIELD: {
            "conversation_id": "conversation", "request_message_id": "request",
            "final_message_id": "final", "observed_at": 1000,
        },
    }
    recovered = {
        "provider_binding_id": "binding", "provider_account_identity": "account", "client_conversation_id": "client",
        TURN_END_EVIDENCE_FIELD: dict(receipt[TURN_END_EVIDENCE_FIELD]),
    }
    receipt["_recovered_final_sender"] = TextTaskService._recovered_final_sender_marker(receipt, recovered)
    if block == "missing_binding":
        receipt.pop("provider_binding_id")
    elif block == "missing_account":
        receipt.pop("provider_account_identity")
    elif block == "missing_client":
        receipt.pop("client_conversation_id")
    elif block == "reader_lease":
        receipt.update(recovery_claim_id="live-reader", recovery_claimed_at=1000,
                       recovery_lease_until=2000, recovery_phase="read_text_request")
    elif block == "paused":
        receipt["_recovery_paused"] = True
    else:
        receipt["_recovery_suppressed"] = True
    assert not TextTaskService.retire_recovered_final_after_sender_exit(receipt, "sender-claim")
    assert receipt["_turn_reserved"] is True


@pytest.mark.parametrize("field", ["provider_binding_id", "provider_account_identity", "client_conversation_id"])
@pytest.mark.parametrize("saved_success", [False, True])
def test_original_read_cannot_restore_identity_changed_during_get(tmp_path, field, saved_success):
    service, _, backend, legacy = migration(tmp_path)
    original = legacy[0]
    rid = original["request_id"]
    final = "final-" + original["request_message_id"]
    changes = dict(_submission_started=True, _turn_reserved=True, _executing=False)
    if saved_success:
        changes.update(status="succeeded", content="saved final", upstream_outcome="completed",
                       error_code=None, parent_message_id=final)
    service._update("owner", rid, **changes)

    def reader(receipt):
        upstream = document(receipt)
        upstream["mapping"][final]["message"]["content"]["parts"] = ["saved final"]
        result = ConversationBindingService._read_text_request_result(backend, receipt, document=upstream)
        with service.store.transaction() as db:
            current = service.store.read_receipt(db, "text", "owner", rid)
            current[field] = "changed-during-get"
            service.store.write_receipt(db, "text", "owner", rid, current)
        return result

    service.recovery_reader = reader
    service.read("owner", rid)
    with service.store.connect() as db:
        current = service.store.read_receipt(db, "text", "owner", rid)
    assert current[field] == "changed-during-get"
    assert current["_turn_reserved"] is True
    assert current.get("_recovered_final_sender") is None
    assert current["status"] == ("succeeded" if saved_success else "unknown")
    assert current.get("content") == ("saved final" if saved_success else None)
