"""Isolated ADVANCED successor preserves original history and frozen context."""
from copy import deepcopy
import hashlib
import json
from unittest.mock import patch

import pytest

from services.advanced_session_successor import FIELD, MODE, validate_successor
from services.conversation_binding_service import ConversationBindingError
from services.product_work import _key
from services.text_task_service import TextTaskService
from services.work_lifecycle import WorkLifecycleError
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
    # The PoolAdmission fixture uses a synthetic pacing callback, so model the
    # real FIFO's admitted read without changing any advanced-session proof.
    h.admission.read_original_if_ready = lambda owner, request_id, reader, receipt, **_kwargs: reader(owner, request_id)
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


def test_preaccept_admission_unavailable_creates_no_child(advanced):
    h = advanced
    calls = []
    h.service.recovery_reader = lambda original: calls.append(original) or {**original, "status": "unknown"}
    h.service.admission = None

    with pytest.raises(WorkLifecycleError) as error:
        h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")

    assert error.value.code == "CHAT_ADVANCED_SESSION_PREACCEPT_READ_DEFERRED"
    assert error.value.status == 503
    assert calls == []
    assert row(h.service, "isolated") is None
    assert row(h.service) == h.old


def test_preaccept_deferred_logs_exact_fifo_reason_without_creating_child(advanced, monkeypatch):
    from services import text_task_service as text_tasks

    h = advanced
    events = []
    calls = []

    def admit(_owner, _request_id, _reader, _receipt, *, diagnostic=None, **_kwargs):
        diagnostic.update(reason="fifo_not_head", account_clock_prefix="a" * 12,
                          ticket_state="own", ticket_retained=True, fifo_position=2)
        return False

    h.admission.read_original_if_ready = admit
    h.service.recovery_reader = lambda original: calls.append(original) or {**original, "status": "unknown"}
    monkeypatch.setattr(text_tasks.logger, "info", events.append)

    with pytest.raises(WorkLifecycleError) as error:
        h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")

    assert error.value.code == "CHAT_ADVANCED_SESSION_PREACCEPT_READ_DEFERRED"
    assert calls == []
    assert events == [{"event": "advanced_preaccept_read_admission", "reason": "fifo_not_head",
                       "root_ref": hashlib.sha256(b"owner\0old").hexdigest()[:24],
                       "account_clock_prefix": "a" * 12, "ticket_state": "own",
                       "ticket_retained": True, "fifo_position": 2}]
    assert row(h.service, "isolated") is None
    assert row(h.service) == h.old


def test_preaccept_admitted_fresh_proof_submits_once(advanced):
    h = advanced
    admissions, reads = [], []

    def admit(owner, request_id, reader, receipt, **kwargs):
        admissions.append((owner, request_id, receipt["request_id"]))
        assert kwargs["reservation_scope"] == "advanced-preaccept"
        assert kwargs["wait_for_ready_seconds"] == 720.0
        return reader(owner, request_id)

    def fresh(original):
        reads.append(original["request_id"])
        return {**original, "status": "unknown"}

    h.admission.read_original_if_ready = admit
    h.service.recovery_reader = fresh
    result = h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")

    assert result["status"] == "queued"
    assert admissions == [("owner", "old", "old")]
    assert reads == ["old"]
    assert row(h.service, "isolated")["_submission_started"] is False
    assert row(h.service) == h.old


def test_preaccept_read_deadline_creates_no_child(advanced):
    from services.account_request_pacing import AccountRequestDeadlineExceeded

    h = advanced
    h.admission.read_original_if_ready = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AccountRequestDeadlineExceeded("fixture read deadline"))

    with pytest.raises(WorkLifecycleError) as error:
        h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")

    assert error.value.code == "CHAT_ADVANCED_SESSION_PREACCEPT_READ_DEFERRED"
    assert error.value.status == 503
    assert row(h.service, "isolated") is None
    assert row(h.service) == h.old


def test_preaccept_deferred_http_response_has_no_child(advanced, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock
    from api import ai

    h = advanced
    h.admission.read_original_if_ready = lambda *_args, **_kwargs: False
    monkeypatch.setattr(ai, "text_task_service", h.service)
    monkeypatch.setattr(ai, "require_identity", lambda *_a, **_kw: {"role": "admin", "id": "owner"})
    monkeypatch.setattr(ai, "trusted_source", lambda *_a: "internal:listing")
    monkeypatch.setattr(ai, "filter_or_log", AsyncMock())
    app = FastAPI()
    app.include_router(ai.create_router())

    with TestClient(app) as client:
        response = client.post("/api/conversation-bindings/text-requests/old/derived-successor", json=h.payload)

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "CHAT_ADVANCED_SESSION_PREACCEPT_READ_DEFERRED"}}
    assert row(h.service, "isolated") is None
    assert row(h.service) == h.old


def test_preaccept_deferred_does_not_claim_global_absence(advanced, monkeypatch):
    h = advanced
    body = h.service.prepare_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    submit_prepared = h.service._submit_prepared
    accepted = []

    def concurrent_acceptance(*_args, **_kwargs):
        accepted.append(submit_prepared("owner", body, source="internal:listing"))
        return False

    h.admission.read_original_if_ready = concurrent_acceptance
    with pytest.raises(WorkLifecycleError) as error:
        h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")

    assert error.value.code == "CHAT_ADVANCED_SESSION_PREACCEPT_READ_DEFERRED"
    assert row(h.service, "isolated") is not None
    # A subsequent identical request must return the durable concurrent receipt
    # without attempting another submit/send boundary.
    monkeypatch.setattr(h.service, "_submit_prepared", lambda *_a, **_kw: pytest.fail("submitted twice"))
    assert h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing") == accepted[0]


def test_concurrent_same_id_conflict_is_revalidated_before_existing_return(advanced, monkeypatch):
    h = advanced
    conflicting = {**h.payload, "model": "different-fixture-model"}
    conflicting_body = h.service.prepare_workbench_derived_input(
        "owner", "old", conflicting, source="internal:listing")
    prepare = h.service.prepare_workbench_derived_input
    submit_prepared = h.service._submit_prepared
    inserted = False

    def prepare_then_concurrently_accept(owner, request_id, payload, *, source):
        nonlocal inserted
        body = prepare(owner, request_id, payload, source=source)
        if not inserted and payload is h.payload:
            inserted = True
            submit_prepared("owner", conflicting_body, source="internal:listing")
        return body

    monkeypatch.setattr(h.service, "prepare_workbench_derived_input", prepare_then_concurrently_accept)
    with pytest.raises(ConversationBindingError) as error:
        h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")

    assert error.value.code == "CONVERSATION_REQUEST_CONFLICT"
    assert row(h.service, "isolated")["model"] == "different-fixture-model"
    assert row(h.service) == h.old


def test_acceptance_observation_absent_is_local_only(advanced):
    h = advanced
    h.service.recovery_reader = lambda _original: pytest.fail("acceptance observation read upstream")

    observed = h.service.observe_workbench_derived_acceptance(
        "owner", "old", h.payload, source="internal:listing")

    body = h.service.prepare_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    assert observed == {
        "state": "absent_at_observation",
        "original_request_id": "old",
        "client_request_id": "isolated",
        "original_input_hash": h.payload["original_input_hash"],
        "child_input_hash": TextTaskService._submission_identity("owner", body)[1],
    }
    assert row(h.service, "isolated") is None
    assert row(h.service) == h.old
    assert h.sent == []


def test_acceptance_observation_returns_exact_existing_receipt(advanced, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock
    from api import ai

    h = advanced
    accepted = h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    content_filter = AsyncMock()
    monkeypatch.setattr(ai, "text_task_service", h.service)
    monkeypatch.setattr(ai, "require_identity", lambda *_a, **_kw: {"role": "admin", "id": "owner"})
    monkeypatch.setattr(ai, "trusted_source", lambda *_a: "internal:listing")
    monkeypatch.setattr(ai, "filter_or_log", content_filter)
    app = FastAPI()
    app.include_router(ai.create_router())

    with TestClient(app) as client:
        observed = client.post(
            "/api/conversation-bindings/text-requests/old/derived-successor/acceptance", json=h.payload)

    assert observed.status_code == 200, observed.text
    assert observed.json() == accepted
    content_filter.assert_not_awaited()


def test_acceptance_observation_rejects_conflicting_existing_child(advanced):
    h = advanced
    h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    changed = {**h.payload, "model": "different-fixture-model"}

    with pytest.raises(ConversationBindingError) as error:
        h.service.observe_workbench_derived_acceptance("owner", "old", changed, source="internal:listing")

    assert error.value.code == "CONVERSATION_REQUEST_CONFLICT"


def test_acceptance_http_is_restricted_and_requires_advanced_payload(advanced, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api import ai

    h = advanced
    identity = {"role": "user", "id": "owner"}
    monkeypatch.setattr(ai, "text_task_service", h.service)
    monkeypatch.setattr(ai, "require_identity", lambda *_a, **_kw: identity)
    monkeypatch.setattr(ai, "trusted_source", lambda *_a: "internal:listing")
    app = FastAPI()
    app.include_router(ai.create_router())

    with TestClient(app) as client:
        forbidden = client.post(
            "/api/conversation-bindings/text-requests/old/derived-successor/acceptance", json=h.payload)
        identity["role"] = "admin"
        malformed = client.post(
            "/api/conversation-bindings/text-requests/old/derived-successor/acceptance",
            json={key: value for key, value in h.payload.items() if key != "messages"},
        )
        legacy = {key: value for key, value in h.payload.items() if key != "context_ref"}
        legacy["mode"] = "no_final"
        not_advanced = client.post(
            "/api/conversation-bindings/text-requests/old/derived-successor/acceptance", json=legacy)

    assert forbidden.status_code == 404
    assert malformed.status_code == 422
    assert not_advanced.status_code == 409
    assert not_advanced.json() == {"detail": {"code": "CHAT_ADVANCED_SESSION_INVALID"}}


def test_ordinary_postaccept_unknown_retains_durable_send_boundary(tmp_path):
    class Queue:
        def __init__(self):
            self.call = None

        def submit(self, function, *args):
            self.call = function, args

        def run(self):
            function, args = self.call
            function(*args)

    queue = Queue()
    service = TextTaskService(tmp_path / "ordinary.sqlite3", executor=queue)
    body = {"client_request_id": "ordinary", "client_conversation_id": "ordinary-client",
            "messages": [{"role": "user", "content": "ordinary request"}]}

    def unknown_after_send(_body, on_cursor):
        service._update("owner", "ordinary", _submission_started=True, _turn_reserved=True)
        raise ConversationBindingError("uncertain model outcome", code="CONVERSATION_OUTCOME_UNKNOWN")

    service.runner = unknown_after_send
    service.submit("owner", body)
    queue.run()

    with service.store.connect() as db:
        receipt = service.store.read_receipt(db, "text", "owner", "ordinary")
    assert receipt["status"] == "unknown"
    assert receipt["error_code"] == "CONVERSATION_OUTCOME_UNKNOWN"
    assert receipt["_submission_started"] is True


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


def test_advanced_validation_keeps_unrelated_terminal_payload_compact(advanced):
    """The existing successor proof may read its two roots, never another saved result body."""
    h = advanced
    body, child = prepared(h)
    child["_workbench_derived_body"] = body
    marker = "advanced-validation-must-not-read-unrelated-terminal-image"
    with h.service.store.transaction() as db:
        db.execute("INSERT INTO image_requests(task_key,receipt) VALUES(?,?)", (
            "other-owner:unrelated-terminal",
            json.dumps({"id": "unrelated-terminal", "owner_id": "other-owner", "status": "success",
                        "upstream_unfinished": False, "data": [{"b64_json": marker * 30000}]}),
        ))
    decode = json.loads

    def compact_only(raw, *args, **kwargs):
        assert marker not in raw
        return decode(raw, *args, **kwargs)

    with h.service.store.connect() as db, \
         patch("services.task_store.json.loads", side_effect=compact_only):
        original = validate_successor(h.service.store, db, "owner", child, h.clock())

    assert original["request_id"] == "old"
    with h.service.store.connect() as db:
        unrelated = h.service.store.read_receipt(db, "image", "other-owner", "unrelated-terminal")
    assert unrelated["data"] == [{"b64_json": marker * 30000}]


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
        raise ConversationBindingError("stop before test transport", code="CHAT_ADVANCED_SESSION_INVALID",
                                       provider_binding_id="foreign-observed-binding",
                                       provider_account_identity="foreign-account",
                                       conversation_id="foreign-observed-conversation")
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


def test_pre_send_reentry_reuses_allocated_advanced_binding_only_in_execution_copy(advanced):
    h = advanced
    seen = []
    h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    h.service._update("owner", "isolated", provider_binding_id="allocated-child-binding",
                      provider_account_identity=h.old["provider_account_identity"])

    def inspect(body, on_cursor):
        seen.append(deepcopy(body))
        on_cursor({"provider_binding_id": "allocated-child-binding",
                   "provider_account_identity": h.old["provider_account_identity"],
                   "client_conversation_id": body["client_conversation_id"]})
        raise ConversationBindingError("stop before test transport", code="CHAT_ADVANCED_SESSION_INVALID",
                                       provider_binding_id="foreign-observed-binding",
                                       provider_account_identity="foreign-account",
                                       conversation_id="foreign-observed-conversation",
                                       parent_message_id="foreign-observed-parent")

    h.service.runner = inspect
    with h.service.store.connect() as db:
        durable = h.service.store.load_input(h.service.store.read_receipt(db, "text", "owner", "isolated")["_input_ref"])
    assert "provider_binding_id" not in durable
    h.service._run("owner", "isolated", durable)
    assert seen and seen[0]["provider_binding_id"] == "allocated-child-binding"
    assert seen[0]["provider_account_identity"] == h.old["provider_account_identity"]
    with h.service.store.connect() as db:
        saved = h.service.store.read_receipt(db, "text", "owner", "isolated")
    assert saved["provider_binding_id"] == "allocated-child-binding"
    assert not saved.get("conversation_id")
    assert saved["status"] == "failed" and saved["error_code"] == "CHAT_ADVANCED_SESSION_INVALID"


def test_initial_foreign_cursor_exception_does_not_allocate_advanced_child(advanced):
    h = advanced
    h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")

    def foreign(_body, on_cursor):
        raise ConversationBindingError("foreign cursor", code="CHAT_ADVANCED_SESSION_INVALID",
                                       provider_binding_id="foreign-binding",
                                       provider_account_identity="foreign-account",
                                       conversation_id="foreign-conversation",
                                       parent_message_id="foreign-parent")

    h.service.runner = foreign
    with h.service.store.connect() as db:
        durable = h.service.store.load_input(h.service.store.read_receipt(db, "text", "owner", "isolated")["_input_ref"])
    h.service._run("owner", "isolated", durable)
    with h.service.store.connect() as db:
        saved = h.service.store.read_receipt(db, "text", "owner", "isolated")
    assert not saved.get("provider_binding_id") and not saved.get("conversation_id")
    assert saved["status"] == "failed" and saved["error_code"] == "CHAT_ADVANCED_SESSION_INVALID"


@pytest.mark.parametrize("change", [None, "binding", "account", "conversation", "before_send"])
def test_failure_cursor_keeps_matching_post_send_identity_only(advanced, change):
    h = advanced
    h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")

    def fail(body, on_cursor):
        on_cursor({"provider_binding_id": "allocated-child-binding",
                   "provider_account_identity": "original-account"})
        if change != "before_send":
            on_cursor({"_submission_started": True})
        if change == "conversation":
            on_cursor({"conversation_id": "already-saved-chat"})
        raise ConversationBindingError("result read failed", code="CONVERSATION_OUTCOME_UNKNOWN",
            provider_binding_id="other-binding" if change == "binding" else "allocated-child-binding",
            provider_account_identity="other-account" if change == "account" else "original-account",
            conversation_id="new-chat", parent_message_id="new-parent")

    h.service.runner = fail
    durable = h.service.store.load_input(row(h.service, "isolated")["_input_ref"])
    h.service._run("owner", "isolated", durable)
    saved = row(h.service, "isolated")
    assert saved["status"] == "unknown"
    assert saved["provider_binding_id"] == "allocated-child-binding"
    assert saved["provider_account_identity"] == "original-account"
    if change is None:
        assert saved["conversation_id"] == "new-chat" and saved["parent_message_id"] == "new-parent"
    else:
        assert saved.get("conversation_id") == ("already-saved-chat" if change == "conversation" else None)
        assert not saved.get("parent_message_id")


def test_post_send_allocation_mismatch_preserves_unknown(advanced):
    h = advanced
    h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")

    def fail(body, on_cursor):
        on_cursor({"provider_binding_id": "allocated-child-binding", "_submission_started": True})
        on_cursor({"provider_binding_id": "foreign-binding"})

    h.service.runner = fail
    h.service._run("owner", "isolated", h.service.store.load_input(row(h.service, "isolated")["_input_ref"]))
    saved = row(h.service, "isolated")
    assert saved["status"] == "unknown" and saved["upstream_outcome"] == "unknown"
    assert saved["provider_binding_id"] == "allocated-child-binding"
    assert saved["_submission_started"] is True


def test_invalid_advanced_execution_copy_is_recorded_as_unsent_failure(advanced):
    h = advanced
    h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    h.service._update("owner", "isolated", provider_binding_id="allocated-child-binding",
                      provider_account_identity=h.old["provider_account_identity"],
                      conversation_id="unexpected-pre-send-conversation")
    with h.service.store.connect() as db:
        durable = h.service.store.load_input(h.service.store.read_receipt(db, "text", "owner", "isolated")["_input_ref"])
    h.service._run("owner", "isolated", durable)
    with h.service.store.connect() as db:
        saved = h.service.store.read_receipt(db, "text", "owner", "isolated")
    assert saved["status"] == "failed" and saved["upstream_outcome"] == "not_sent"
    assert saved["error_code"] == "CHAT_ADVANCED_SESSION_INVALID"


@pytest.mark.parametrize(("field", "first", "second"), [
    ("provider_binding_id", "allocated-child-binding", "reassigned-child-binding"),
    ("conversation_id", "allocated-child-conversation", "reassigned-child-conversation"),
])
def test_advanced_child_allocated_cursor_is_monotonic(advanced, field, first, second):
    h = advanced
    h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    h.service._update("owner", "isolated", provider_binding_id="allocated-child-binding",
                      provider_account_identity=h.old["provider_account_identity"])
    if field == "conversation_id":
        h.service._update("owner", "isolated", conversation_id=first)
    with pytest.raises(ConversationBindingError):
        h.service._update("owner", "isolated", **{field: second})
    with h.service.store.connect() as db:
        saved = h.service.store.read_receipt(db, "text", "owner", "isolated")
    assert saved[field] == (first if field == "conversation_id" else "allocated-child-binding")


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
