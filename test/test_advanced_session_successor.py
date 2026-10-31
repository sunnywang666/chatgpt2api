"""Isolated ADVANCED successor preserves original history and frozen context."""
from copy import deepcopy
import hashlib
import json

import pytest

from services.advanced_session_successor import FIELD, MODE, validate_successor
from services.conversation_binding_service import ConversationBindingError
from services.product_work import _key
from services.text_task_service import TextTaskService
from test.test_no_final_successor import product_no_final
from test.test_explicit_unknown_successor import row


@pytest.fixture
def advanced(product_no_final):
    h = product_no_final
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        old_key = original["_product_work_key"]
        parent = h.service.store.runtime(db, old_key)
        key = _key("internal:listing", "owner", "fixture-product")
        parent.update(key=key, source="internal:listing")
        h.service.store.set_runtime(db, key, parent)
        db.execute("DELETE FROM task_runtime WHERE name=?", (old_key,))
        original.update(status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
            recovery_reason="REQUEST_CONVERSATION_ADVANCED", upstream_unfinished=False,
            _source="internal:listing", _scheduling_owner="internal:listing", _product_work_key=key,
            request_parent_message_id="previous-answer", _submission_parent_message_id="previous-answer",
            _completed_successor_evidence={"conversation_id": "original-chat", "request_message_id": original["request_message_id"],
                "successor_request_message_id": "external-user", "successor_request_parent_message_id": "thoughts",
                "successor_final_message_id": "external-final", "observed_at": h.clock()})
        body = h.service.store.load_input(original["_input_ref"])
        body["messages"] = [{"role": "user", "content": [{"type": "text", "text": "original prompt"},
            {"type": "text", "text": '{"image_ref":"source-a"}'},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDRA=="}}]}]
        original["_input_ref"] = h.service.store.save_input(body)
        digest = TextTaskService._submission_identity("owner", body)[1]
        db.execute("UPDATE requests SET request_hash=?,receipt=? WHERE owner='owner' AND id='old'", (digest, json.dumps(original)))
        context_body = deepcopy(body)
        context_body.update(client_request_id="gallery", parent_message_id="gallery-input",
                            product_work={**body["product_work"], "step_ref": "gallery"})
        context_hash = TextTaskService._submission_identity("owner", context_body)[1]
        gallery = {**original, "request_id": "gallery", "status": "succeeded", "content": "saved exact gallery result",
            "_upstream_terminal": True, "parent_message_id": "previous-answer", "request_message_id": "gallery-user",
            "request_parent_message_id": "gallery-input", "_submission_parent_message_id": "gallery-input",
            "_product_work": context_body["product_work"], "_input_ref": h.service.store.save_input(context_body), "_sequence": 1}
        db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("owner", "gallery", context_hash, json.dumps(gallery)))
    h.old = row(h.service)
    h.payload = {"client_request_id": "isolated", "original_input_hash": digest, "messages": deepcopy(body["messages"]),
        "model": "fixture-text", "thinking_effort": "standard", "mode": MODE, "derivation_kind": "new-business-prompt",
        "context_ref": {"request_id": "gallery", "input_hash": context_hash,
                        "result_sha256": hashlib.sha256(gallery["content"].encode()).hexdigest()}}
    h.payload["messages"][0]["content"][0]["text"] = "Full replacement instruction\n" + gallery["content"]
    h.service.recovery_reader = lambda original: {**original, "status": "unknown"}
    return h


def prepared(h):
    body = h.service.prepare_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    candidate = {"request_id": body["client_request_id"], "client_conversation_id": body["client_conversation_id"]}
    with h.service.store.connect() as db:
        h.service._prepare_supersede(db, "owner", body, candidate)
    return body, candidate


def test_prepared_child_is_new_session_account_locked_original_unchanged(advanced):
    h = advanced
    body, child = prepared(h)
    assert not any(k in body for k in ("provider_binding_id", "conversation_id", "parent_message_id"))
    assert body["provider_account_identity"] == h.old["provider_account_identity"]
    assert body["client_conversation_id"] != h.old["client_conversation_id"]
    assert body["product_work"] == h.old["_product_work"]
    assert child[FIELD]["original_head_message_id"] == "external-final"
    assert row(h.service) == h.old


@pytest.mark.parametrize("drift", ["reason", "paused", "hold", "claim", "recovery", "success", "context_hash", "context_result",
    "media", "omitted_result", "caller_parent", "original_hash", "duplicate", "other_unknown"])
def test_creation_rejects_ineligible_or_drifted_evidence(advanced, drift):
    h = advanced
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        if drift == "reason": original["recovery_reason"] = "REQUEST_RESULT_INCOMPLETE"
        elif drift == "paused": original["_recovery_paused"] = True
        elif drift == "hold": original["_send_hold"] = True
        elif drift == "claim": original.update(_claim_id="live", _claim_until=h.clock()+100)
        elif drift == "recovery": original.update(recovery_claim_id="read", recovery_lease_until=h.clock()+100)
        elif drift == "success": original.update(status="succeeded", content="late answer")
        elif drift == "context_hash": h.payload["context_ref"]["input_hash"] = "0"*64
        elif drift == "context_result": h.payload["context_ref"]["result_sha256"] = "0"*64
        elif drift == "media": h.payload["messages"][0]["content"][-1]["image_url"]["url"] = "changed"
        elif drift == "omitted_result": h.payload["messages"][0]["content"][0]["text"] = "without gallery"
        elif drift == "caller_parent": h.payload["parent_message_id"] = "previous-answer"
        elif drift == "original_hash": h.payload["original_input_hash"] = "0"*64
        elif drift in {"duplicate", "other_unknown"}:
            other = {**original, "request_id": "other", "_supersedes_request_id": "old" if drift == "duplicate" else None}
            db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("owner", "other", "unused", json.dumps(other)))
        h.service.store.write_receipt(db, "text", "owner", "old", original)
    with pytest.raises(ConversationBindingError):
        prepared(h)


def test_before_send_rechecks_hash_and_original_late_answer(advanced):
    h = advanced
    body, child = prepared(h)
    child["_workbench_derived_body"] = body
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        original.update(status="succeeded", content="late original result", _upstream_terminal=True)
        original.pop("_completed_successor_evidence", None)
        h.service.store.write_receipt(db, "text", "owner", "old", original)
        with pytest.raises(ConversationBindingError):
            validate_successor(h.service.store, db, "owner", child, h.clock())
        child["_submission_started"] = True
        assert validate_successor(h.service.store, db, "owner", child, h.clock(), after_send=True)["content"] == "late original result"
        child[FIELD]["original_binding_id"] = "drift"
        with pytest.raises(ConversationBindingError):
            validate_successor(h.service.store, db, "owner", child, h.clock(), after_send=True)


def test_stale_graph_rejected_before_registration(advanced):
    h = advanced
    h.service.recovery_reader = lambda original: {**original, "_completed_successor_evidence": {
        **original["_completed_successor_evidence"], "successor_final_message_id": "newer-external-final"}}
    with pytest.raises(ConversationBindingError):
        h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    assert row(h.service, "isolated") is None
    assert row(h.service) == h.old


def test_submit_is_atomic_unique_and_idempotent(advanced):
    h = advanced
    first = h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    assert first["status"] == "queued"
    assert first["advanced_session"]["original_request_id"] == "old"
    assert first["workbench_derived_input"]["mode"] == MODE
    assert h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing") == first
    different = {**h.payload, "client_request_id": "second-isolated"}
    with pytest.raises(ConversationBindingError):
        h.service.submit_workbench_derived_input("owner", "old", different, source="internal:listing")
    assert row(h.service, "second-isolated") is None
    assert row(h.service) == h.old


def test_frozen_evidence_refresh_allows_timestamp_only(advanced):
    h = advanced
    body, child = prepared(h)
    child["_workbench_derived_body"] = body
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        original["_completed_successor_evidence"]["observed_at"] -= 1
        h.service.store.write_receipt(db, "text", "owner", "old", original)
        assert validate_successor(h.service.store, db, "owner", child, h.clock()) == original
        original["_completed_successor_evidence"]["successor_final_message_id"] = "another-head"
        h.service.store.write_receipt(db, "text", "owner", "old", original)
        with pytest.raises(ConversationBindingError):
            validate_successor(h.service.store, db, "owner", child, h.clock())


@pytest.mark.parametrize("field,value", [("provider_account_identity", "foreign"),
    ("client_conversation_id", "old-session"), ("provider_binding_id", "original-binding"),
    ("conversation_id", "original-chat"), ("_product_work_key", "foreign-product")])
def test_before_send_rejects_child_identity_drift(advanced, field, value):
    h = advanced
    body, child = prepared(h)
    child.update(_workbench_derived_body=body)
    child[field] = value
    with h.service.store.connect() as db, pytest.raises(ConversationBindingError):
        validate_successor(h.service.store, db, "owner", child, h.clock())


def test_concurrent_different_children_register_only_one(advanced):
    from concurrent.futures import ThreadPoolExecutor
    h = advanced
    def submit(request_id):
        try:
            return h.service.submit_workbench_derived_input("owner", "old", {**h.payload, "client_request_id": request_id}, source="internal:listing")
        except ConversationBindingError:
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(submit, ["one-child", "other-child"]))
    assert sum(result is not None for result in results) == 1
    assert row(h.service) == h.old


def test_claim_and_run_pass_isolated_guard_and_locked_account(advanced):
    h = advanced
    seen = []
    def inspect(body, on_cursor):
        seen.append(deepcopy(body))
        raise ConversationBindingError("stop before test transport", code="CHAT_ADVANCED_SESSION_INVALID")
    h.service.runner = inspect
    h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    context = h.admission.claim_next()
    assert context is not None and context.request_id == "isolated"
    h.admission.execute(context)
    assert len(seen) == 1, {k: row(h.service, "isolated").get(k) for k in ("error_code", "provider_binding_id", "provider_account_identity", "_requested_account_identity", "_execution_timeline")}
    body = seen[0]
    assert body["_requested_account_identity"] == h.old["provider_account_identity"]
    assert body["_text_only_binding"] is True
    assert body["_advanced_session_original"] == h.old
    assert "_supersedes_request_message_id" not in body
    assert not body.get("conversation_id")
    assert row(h.service) == h.old


def test_registered_message_collision_is_not_external_discussion(advanced):
    h = advanced
    with h.service.store.transaction() as db:
        gallery = h.service.store.read_receipt(db, "text", "owner", "gallery")
        gallery["request_message_id"] = "external-user"
        h.service.store.write_receipt(db, "text", "owner", "gallery", gallery)
    with pytest.raises(ConversationBindingError):
        prepared(h)


def test_same_id_cannot_change_context_reference(advanced):
    h = advanced
    first = h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    assert first["status"] == "queued"
    with h.service.store.transaction() as db:
        gallery = h.service.store.read_receipt(db, "text", "owner", "gallery")
        body = h.service.store.load_input(gallery["_input_ref"])
        body["client_request_id"] = "another-gallery"
        gallery.update(request_id="another-gallery", _input_ref=h.service.store.save_input(body))
        digest = TextTaskService._submission_identity("owner", body)[1]
        db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("owner", "another-gallery", digest, json.dumps(gallery)))
    changed = deepcopy(h.payload)
    changed["context_ref"].update(request_id="another-gallery", input_hash=digest)
    with pytest.raises(ConversationBindingError):
        h.service.submit_workbench_derived_input("owner", "old", changed, source="internal:listing")


@pytest.mark.parametrize("drift", ["fourth_turn", "read_failure", "incomplete_again", "read_claim"])
def test_sent_child_relationship_survives_independent_old_session_reads(advanced, drift):
    h = advanced
    body, child = prepared(h)
    child.update(_workbench_derived_body=body, _submission_started=True)
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        if drift == "fourth_turn":
            original["_completed_successor_evidence"]["tail_steps"] = [{"request_message_id": "fourth-user", "final_message_id": "fourth-final"}]
        elif drift == "read_failure":
            original.update(recovery_reason=None, recovery_error_code="RECOVERY_RATE_LIMITED")
            original.pop("_completed_successor_evidence")
        elif drift == "incomplete_again":
            original.update(recovery_reason="REQUEST_RESULT_INCOMPLETE", upstream_unfinished=True)
            original.pop("_completed_successor_evidence")
        else:
            original.update(recovery_claim_id="active-reader", recovery_lease_until=h.clock()+100)
        h.service.store.write_receipt(db, "text", "owner", "old", original)
        with pytest.raises(ConversationBindingError):
            validate_successor(h.service.store, db, "owner", child, h.clock())
        result = validate_successor(h.service.store, db, "owner", child, h.clock(), after_send=True)
        assert result == original  # Active old claims remain visible to capacity accounting.


def first_session_context(h):
    with h.service.store.transaction() as db:
        gallery = h.service.store.read_receipt(db, "text", "owner", "gallery")
        body = h.service.store.load_input(gallery["_input_ref"])
        for key in ("provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id"):
            body.pop(key, None)
        gallery.pop("_upstream_terminal", None)
        gallery.pop("_submission_parent_message_id", None)
        gallery["request_parent_message_id"] = ""
        gallery["_input_ref"] = h.service.store.save_input(body)
        digest = TextTaskService._submission_identity("owner", body)[1]
        db.execute("UPDATE requests SET request_hash=?,receipt=? WHERE owner='owner' AND id='gallery'", (digest, json.dumps(gallery)))
    h.payload["context_ref"]["input_hash"] = digest
    return body


def test_context_first_session_input_and_legacy_success_are_supported(advanced):
    h = advanced
    first_session_context(h)
    body, child = prepared(h)
    assert child[FIELD]["context_ref"] == h.payload["context_ref"]
    assert body["messages"] == h.payload["messages"]


def test_http_retained_first_gallery_builds_context_and_submits_once(advanced, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock
    from api import ai
    h = advanced
    expected = first_session_context(h)
    monkeypatch.setattr(ai, "text_task_service", h.service)
    monkeypatch.setattr(ai, "require_identity", lambda *_a, **_kw: {"role": "admin", "id": "owner"})
    monkeypatch.setattr(ai, "trusted_source", lambda *_a: "internal:listing")
    monkeypatch.setattr(ai, "filter_or_log", AsyncMock())
    app = FastAPI()
    app.include_router(ai.create_router())
    with TestClient(app) as client:
        exported = client.get("/api/conversation-bindings/text-requests/gallery/retained-input")
        assert exported.status_code == 200, exported.text
        assert exported.json()["input"] == expected
        request = deepcopy(h.payload)
        request["context_ref"]["input_hash"] = exported.json()["input_hash"]
        posted = client.post("/api/conversation-bindings/text-requests/old/derived-successor", json=request)
        assert posted.status_code == 200, posted.text
        assert posted.json()["advanced_session"]["context_ref"] == request["context_ref"]
        assert client.post("/api/conversation-bindings/text-requests/old/derived-successor", json=request).json() == posted.json()
    assert row(h.service) == h.old and h.sent == []


@pytest.mark.parametrize("drift", ["unsent", "unknown", "wrong_identity", "hash", "no_content", "foreign_owner"])
def test_first_session_context_export_does_not_relax_unknown_or_identity(advanced, drift):
    h = advanced
    first_session_context(h)
    with h.service.store.transaction() as db:
        gallery = h.service.store.read_receipt(db, "text", "owner", "gallery")
        if drift == "unsent": gallery["_submission_started"] = False
        elif drift == "unknown": gallery["status"] = "unknown"
        elif drift == "no_content": gallery["content"] = ""
        elif drift == "wrong_identity": gallery["client_conversation_id"] = "unrelated"
        elif drift == "hash": db.execute("UPDATE requests SET request_hash=? WHERE owner='owner' AND id='gallery'", ("0"*64,))
        h.service.store.write_receipt(db, "text", "owner", "gallery", gallery)
    with pytest.raises(ConversationBindingError):
        h.service.read_retained_input("foreign" if drift == "foreign_owner" else "owner", "gallery", source="internal:listing")


def test_after_send_mode_requires_a_recorded_send_boundary(advanced):
    h = advanced
    body, child = prepared(h)
    child.update(_workbench_derived_body=body, status="succeeded", _submission_started=False)
    with h.service.store.connect() as db, pytest.raises(ConversationBindingError):
        validate_successor(h.service.store, db, "owner", child, h.clock(), after_send=True)


def test_durable_child_cannot_lose_product_or_source_identity(advanced):
    h = advanced
    h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    with h.service.store.connect() as db:
        child = h.service.store.read_receipt(db, "text", "owner", "isolated")
        for field in ("_source", "_product_work", "_product_work_key"):
            corrupt = deepcopy(child)
            corrupt.pop(field)
            with pytest.raises(ConversationBindingError):
                validate_successor(h.service.store, db, "owner", corrupt, h.clock())
