import json
import hashlib
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


@pytest.mark.parametrize("kind", ["text", "image"])
@pytest.mark.parametrize("desired", [True, False])
def test_cooling_account_does_not_block_other_original_archive_or_restore(runtime, monkeypatch, tmp_path, kind, desired):
    from services import account_request_pacing as pacing
    service, store, clock, calls, _, add = runtime
    def image_archive(identity, request_id, archived):
        calls.append((identity["id"], request_id, archived))
        return {"task_id": request_id, "archived": archived, "image_thread": {"id": request_id.split("-")[0]}}
    service.images.set_thread_archived = image_archive
    rows = [add("A-1"), add("B-2")]
    if kind == "image":
        with store.transaction() as db:
            for receipt in rows:
                request_id = receipt["request_id"]
                db.execute("DELETE FROM requests WHERE owner=? AND id=?", ("one", request_id))
                receipt.update(id=request_id, owner_id="one", status="success", _image_thread={"id": request_id.split("-")[0]})
                work = store.runtime(db, receipt["_work_key"])
                work["kind"] = "image"
                store.set_runtime(db, work["key"], work)
                store.write_receipt(db, "image", "one", request_id, receipt)
    for receipt in rows:
        service.update(kind, {"id": "one"}, receipt["request_id"], "completed", True)
    if not desired:
        assert service.process_one() and service.process_one()
        calls.clear()
        for receipt in rows:
            service.update(kind, {"id": "one"}, receipt["request_id"], "active")
    cold, ready = sorted(rows, key=lambda r: r["_work_key"])
    accounts = []
    with store.transaction() as db:
        for label, receipt in [("cold", cold), ("ready", ready)]:
            receipt["provider_account_identity"] = label
            store.write_receipt(db, kind, "one", receipt["request_id"], receipt)
            accounts.append({"provider_account_identity": label, "account_id": "workspace-" + label})
    service.text.admission = SimpleNamespace(accounts=SimpleNamespace(admission_accounts=lambda: accounts), wake=lambda: None)
    monkeypatch.setattr(pacing, "DATA_DIR", tmp_path)
    folder = tmp_path / "account_request_clocks"
    folder.mkdir()
    cold_path = folder / (hashlib.sha256(b"workspace-cold").hexdigest() + ".json")
    cold_path.write_text(json.dumps({"next_request": 1100, "next_turn": 1200, "cooldown_until": 1060}))
    ready_path = folder / (hashlib.sha256(b"workspace-ready").hexdigest() + ".json")
    # A future model-message slot must not delay a GET/PATCH archive operation.
    ready_path.write_text(json.dumps({"next_request": 999, "next_turn": 2000, "cooldown_until": 0}))
    assert service.process_one()
    assert calls == [("one", ready["request_id"], desired)]
    deferred = service.get(kind, {"id": "one"}, cold["request_id"])
    assert deferred["archive"]["status"] == "pending"
    assert deferred["archive"]["attempts"] == 0
    assert deferred["archive"]["next_at"] == 1100
    assert deferred["slot_held"] is False
    restarted = WorkLifecycleService(service.text, service.images, clock=lambda: clock[0])
    assert not restarted.process_one(target_key=cold["_work_key"])
    clock[0] = 1100
    assert restarted.process_one(target_key=cold["_work_key"])
    assert calls[-1] == ("one", cold["request_id"], desired)
    result = restarted.get(kind, {"id": "one"}, cold["request_id"])
    assert result["archive"]["status"] == "confirmed" and result["archive"]["attempts"] == 1
    assert result["state"] == ("completed" if desired else "active")


def test_unreadable_account_clock_defers_archive_without_claim_or_http(runtime, monkeypatch, tmp_path):
    from services import account_request_pacing as pacing
    service, store, _, calls, _, add = runtime
    receipt = add("A-1")
    receipt["provider_account_identity"] = "original-account"
    with store.transaction() as db:
        store.write_receipt(db, "text", "one", "A-1", receipt)
    service.update("text", {"id": "one"}, "A-1", "completed", True)
    service.text.admission = SimpleNamespace(accounts=SimpleNamespace(admission_accounts=lambda: [
        {"provider_account_identity": "original-account", "account_id": "workspace"}]))
    monkeypatch.setattr(pacing, "DATA_DIR", tmp_path)
    folder = tmp_path / "account_request_clocks"
    folder.mkdir()
    (folder / (hashlib.sha256(b"workspace").hexdigest() + ".json")).write_text("broken")
    assert not service.process_one()
    assert calls == []
    result = service.get("text", {"id": "one"}, "A-1")
    assert result["archive"]["status"] == "unknown"
    assert result["archive"]["error_code"] == "ACCOUNT_PACING_UNAVAILABLE"
    assert result["archive"]["next_at"] == 1060
    assert result["archive"]["attempts"] == 0 and result["slot_held"] is False
