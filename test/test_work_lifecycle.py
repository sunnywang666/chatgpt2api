import json
from types import SimpleNamespace

import pytest

from services.task_store import TaskStore
from services.work_lifecycle import (
    WorkLifecycleError, WorkLifecycleService, ensure_work, record_slot, read_work,
)


@pytest.fixture
def runtime(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    clock = [1000.0]
    calls = []
    fail = [False]
    def archive(owner, request_id, desired):
        calls.append((owner, request_id, desired))
        if fail[0]:
            raise RuntimeError("fixture response lost")
        return {"request_id": request_id, "archived": desired,
                "conversation": {"protocol": "sequential-v1", "client_conversation_id": request_id.split("-")[0]}}
    text = SimpleNamespace(store=store, admission=None, set_public_session_archived=archive)
    service = WorkLifecycleService(text, SimpleNamespace(), clock=lambda: clock[0])
    def add(request_id, *, status="succeeded", previous=None, owner="one", work=None):
        receipt = {"request_id": request_id, "status": status, "route": "chat", "_route": "chat",
                   "client_conversation_id": "internal-" + (work or request_id.split("-")[0]),
                   "_public_session_ref": work or request_id.split("-")[0], "_source": "person:one",
                   "created_at": clock[0], "_sequence": int(request_id.split("-")[-1]),
                   "_previous_request_id": previous, "_submission_started": status != "queued",
                   "upstream_outcome": "unknown" if status == "unknown" else "completed" if status == "succeeded" else "not_sent"}
        with store.transaction() as db:
            ensure_work(store, db, "text", owner, request_id, receipt,
                        scheduling={"workflow_id": "batch", "workflow_concurrency": 1})
            db.execute("INSERT INTO requests VALUES(?,?,?,?)", (owner, request_id, "original-hash", json.dumps(receipt)))
            record_slot(store, db, "text", owner, receipt, status != "queued")
        return receipt
    return service, store, clock, calls, fail, add


def test_completion_persists_original_archive_intent_and_releases_only_its_work(runtime):
    service, store, _, calls, _, add = runtime
    a = add("A-1")
    add("B-2")
    result = service.update("text", {"id": "one"}, "A-1", "completed", True)
    assert result["state"] == "completed" and result["slot_held"] is False
    assert result["archive"]["status"] == "pending" and calls == []
    assert service.get("text", {"id": "one"}, "B-2")["slot_held"] is True
    assert service.process_one()
    assert calls == [("one", "A-1", True)]
    assert service.get("text", {"id": "one"}, "A-1")["archive"]["archived"] is True
    with store.connect() as db:
        assert store.read_receipt(db, "text", "one", "A-1") == a
    assert not service.process_one()


def test_archive_failure_and_restart_recover_original_target_without_occupying_new_work(runtime):
    service, store, clock, calls, fail, add = runtime
    add("A-1")
    service.update("text", {"id": "one"}, "A-1", "completed", True)
    fail[0] = True
    service.process_one()
    assert service.get("text", {"id": "one"}, "A-1")["archive"]["status"] == "unknown"
    add("B-2")
    with pytest.raises(WorkLifecycleError, match="WORK_ARCHIVE_PENDING"):
        service.update("text", {"id": "one"}, "A-1", "active")
    restarted = WorkLifecycleService(service.text, service.images, clock=lambda: clock[0])
    fail[0] = False
    clock[0] += 5
    assert restarted.process_one()
    assert calls == [("one", "A-1", True), ("one", "A-1", True)]
    assert restarted.get("text", {"id": "one"}, "B-2")["state"] == "active"
    assert restarted.get("text", {"id": "one"}, "B-2")["slot_held"] is True


def test_rework_requires_restore_confirmation_before_next_turn(runtime):
    service, _, _, calls, _, add = runtime
    add("A-1")
    with pytest.raises(WorkLifecycleError, match="WORK_RESULTS_SAVE_REQUIRED"):
        service.update("text", {"id": "one"}, "A-1", "completed")
    service.update("text", {"id": "one"}, "A-1", "completed", True)
    service.process_one()
    result = service.update("text", {"id": "one"}, "A-1", "active")
    assert result["state"] == "restoring" and result["slot_held"] is False
    with pytest.raises(WorkLifecycleError, match="WORK_NOT_ACTIVE"):
        add("A-2", previous="A-1")
    service.process_one()
    add("A-2", previous="A-1")
    assert calls == [("one", "A-1", True), ("one", "A-1", False)]
    with pytest.raises(WorkLifecycleError, match="WORK_SUPERSEDED"):
        service.update("text", {"id": "one"}, "A-1", "completed", True)


def test_pause_releases_work_slot_only_when_original_turn_is_safe(runtime):
    service, _, _, _, _, add = runtime
    add("queued-1", status="queued")
    assert service.update("text", {"id": "one"}, "queued-1", "paused")["state"] == "paused"
    # Resuming a paused unsent request is safe; it must not need a finished result.
    assert service.update("text", {"id": "one"}, "queued-1", "active")["state"] == "active"
    add("unknown-2", status="unknown")
    for state in ("paused", "completed"):
        with pytest.raises(WorkLifecycleError, match="WORK_TURN_UNFINISHED"):
            service.update("text", {"id": "one"}, "unknown-2", state, True)
    assert service.get("text", {"id": "one"}, "unknown-2")["slot_held"] is True


def test_other_key_and_old_completion_cannot_change_current_work(runtime):
    service, _, _, calls, _, add = runtime
    add("A-1")
    add("A-2", previous="A-1")
    with pytest.raises(WorkLifecycleError, match="WORK_REQUEST_NOT_FOUND"):
        service.get("text", {"id": "another-key"}, "A-2")
    with pytest.raises(WorkLifecycleError, match="WORK_SUPERSEDED"):
        service.update("text", {"id": "one"}, "A-1", "completed", True)
    assert calls == []


def test_single_request_protocol_work_closes_locally_without_claiming_upstream_archive(runtime):
    service, store, _, calls, _, _ = runtime
    receipt = {"request_id": "native", "client_conversation_id": "native-session", "status": "succeeded", "_route": "codex",
               "_scheduling": {"workflow_id": "batch", "workflow_concurrency": 1}}
    with store.transaction() as db:
        ensure_work(store, db, "text", "one", "native", receipt)
        db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("one", "native", "hash", json.dumps(receipt)))
    result = service.update("text", {"id": "one"}, "native", "completed", True)
    assert result["state"] == "completed" and result["archive"]["status"] == "not_applicable"
    assert result["archive"]["scope"] == "provider_work" and "archived" not in result["archive"]
    assert service.update("text", {"id": "one"}, "native", "active")["state"] == "active"
    assert not service.process_one() and calls == []


def test_legacy_archived_session_requires_explicit_restore_readback(runtime):
    service, _, _, calls, _, add = runtime
    add("legacy-1")
    assert service.get("text", {"id": "one"}, "legacy-1")["archive"]["status"] == "not_requested"
    assert service.update("text", {"id": "one"}, "legacy-1", "active")["state"] == "restoring"
    assert service.process_one()
    result = service.get("text", {"id": "one"}, "legacy-1")
    assert result["state"] == "active" and result["archive"]["archived"] is False
    assert calls == [("one", "legacy-1", False)]


def test_unsent_expired_work_closes_without_retrying_a_nonexistent_conversation(runtime):
    service, store, _, calls, _, add = runtime
    receipt = add("expired-1", status="queued")
    receipt.update(status="failed", error_code="WAIT_DEADLINE_EXCEEDED")
    with store.transaction() as db:
        store.write_receipt(db, "text", "one", "expired-1", receipt)
    result = service.update("text", {"id": "one"}, "expired-1", "completed", True)
    assert result["state"] == "completed" and result["slot_held"] is False
    assert result["archive"]["status"] == "not_applicable"
    assert result["archive"]["error_code"] == "UPSTREAM_CONVERSATION_NOT_CREATED"
    assert "archived" not in result["archive"]
    assert service.update("text", {"id": "one"}, "expired-1", "active")["state"] == "active"
    assert not service.process_one() and calls == []


def test_later_real_conversation_cannot_create_a_second_slot_via_another_key(runtime):
    _, store, _, _, _, add = runtime
    first = add("A-1")
    with store.transaction() as db:
        first.update(provider_binding_id="original-binding", conversation_id="original-upstream")
        store.write_receipt(db, "text", "one", "A-1", first)
        impostor = {"request_id": "other", "_source": "person:one", "client_conversation_id": "other-alias",
                    "provider_binding_id": "original-binding", "conversation_id": "original-upstream",
                    "_scheduling": {"workflow_id": "batch", "workflow_concurrency": 1}}
        with pytest.raises(WorkLifecycleError, match="WORK_OWNER_CONFLICT"):
            ensure_work(store, db, "text", "another-key", "other", impostor)
        assert impostor["_work_key"] == first["_work_key"]


def test_binding_alias_cannot_bypass_paused_work_and_other_kind_unknown_blocks_archive(runtime):
    service, store, _, calls, _, add = runtime
    first = add("A-1")
    with store.transaction() as db:
        first.update(provider_binding_id="binding-A", provider_account_identity="physical-account", conversation_id="physical-conversation")
        store.write_receipt(db, "text", "one", "A-1", first)
    service.update("text", {"id": "one"}, "A-1", "paused")
    with store.transaction() as db:
        alias = {**first, "provider_binding_id": "binding-B", "client_conversation_id": "another-alias", "request_id": "A-2"}
        alias.pop("_work_key")
        with pytest.raises(WorkLifecycleError, match="WORK_NOT_ACTIVE"):
            ensure_work(store, db, "text", "one", "A-2", alias)
    service.update("text", {"id": "one"}, "A-1", "active")
    with store.transaction() as db:
        store.write_receipt(db, "image", "other-key", "image-unknown", {
            "id": "image-unknown", "owner_id": "other-key", "status": "error", "upstream_unfinished": True,
            "provider_binding_id": "binding-C", "provider_account_identity": "physical-account", "conversation_id": "physical-conversation"})
    with pytest.raises(WorkLifecycleError, match="WORK_TURN_UNFINISHED"):
        service.update("text", {"id": "one"}, "A-1", "completed", True)
    assert calls == []
