"""Original-result pause is durable control, not deletion or upstream cancellation."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest

from api import ai
from api.company_requests import PREFIX, company_identity
from services.conversation_binding_service import ConversationBindingService
from services.image_task_service import ImageTaskService
from services.pool_admission import unfinished, unknown_text_result
from services.text_task_service import TextTaskService
from services.task_store import RecoveryControlConflict, TaskStore
from test.test_company_requests import company, CONNECTOR
from test.test_unknown_turn_recovery import migration, document


def raw(service, kind="text", owner="owner", rid="old-0"):
    with service.store.connect() as db:
        return service.store.read_receipt(db, kind, owner, rid)


def released_original(tmp_path, proof):
    from test.test_stalled_text_diagnostics import incomplete_reader, stalled

    service, admission, backend, _ = migration(tmp_path)
    service._update("owner", "old-0", _submission_started=True,
                    _turn_reserved=False, _executing=False)
    if proof == "stalled":
        stalled(service, admission, backend)
    else:
        service.recovery_reader = incomplete_reader(backend)
        service.read("owner", "old-0")
        admission.clock.now += TextTaskService.RECOVERY_LEASE_SECONDS + 1

        def unavailable(_receipt):
            error = ConnectionError("synthetic original GET unavailable")
            error._original_result_get_attempted = True
            raise error

        service.recovery_reader = unavailable
        service.read("owner", "old-0")
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    return service, admission, backend


@pytest.mark.parametrize("proof", ["stalled", "unreadable"])
def test_pause_keeps_proven_capacity_released_without_recovery_or_same_chat_send(tmp_path, monkeypatch, proof):
    from services.public_chat_service import project_text_execution
    from services.work_lifecycle import _blocked
    from test.test_expired_execution_wait import waiter
    from test.test_pool_admission import build

    service, admission, _ = released_original(tmp_path, proof)
    monkeypatch.setattr("services.public_chat_service.time.time", lambda: admission.clock.now)
    original = raw(service)
    control = service.store.set_recovery_paused("text", "owner", "old-0", True)
    paused = raw(service)
    assert {k: v for k, v in paused.items() if not k.startswith("_recovery_")} == original
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert project_text_execution(paused)["resources"]["account_turn"] == "released"
    assert _blocked(paused)  # No permission to close/archive the unresolved work.
    assert not service._incomplete_compact_successor_source(paused, admission.clock.now)
    assert not service._no_final_successor_source(paused, admission.clock.now)

    _, _, restarted_admission = build(tmp_path, admission.clock)
    reader, runner = Mock(), Mock()
    restarted = TextTaskService(service.path, admission=restarted_admission,
                                clock=admission.clock, recovery_reader=reader, runner=runner)
    restarted_admission.recoveries["text"] = restarted.read
    admission.clock.now += 301
    assert restarted.read("owner", "old-0")["status"] == "unknown"
    assert restarted.recover("owner", "old-0", True)["status"] == "unknown"
    restarted_admission.recover_one()
    reader.assert_not_called()
    runner.assert_not_called()
    assert raw(restarted) == paused
    assert restarted_admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    waiter(restarted, "same-chat", "client-0")
    assert restarted_admission.claim_next() is None
    waiter(restarted, "independent", "independent")
    assert restarted_admission.claim_next().request_id == "independent"
    assert restarted.store.set_recovery_paused("text", "owner", "old-0", False,
                                               expected_version=control["version"])["state"] == "active"


@pytest.mark.parametrize("progress", [False, True])
def test_pause_during_stalled_read_preserves_ended_capacity_window_even_with_progress(tmp_path, progress):
    from test.test_stalled_text_diagnostics import incomplete_reader

    service, admission, backend = released_original(tmp_path, "stalled")

    def read_and_pause(receipt):
        service.store.set_recovery_paused("text", "owner", "old-0", True)
        transform = (lambda _doc, message: message["content"].update(parts=["verified progress"])) if progress else None
        return incomplete_reader(backend, transform)(receipt)

    service.recovery_reader = read_and_pause
    admission.clock.now += 301
    assert service.read("owner", "old-0")["status"] == "unknown"
    paused = raw(service)
    assert paused["_recovery_paused"] is True
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert "_result_wait_ended_at" in paused
    service.recovery_reader = Mock()
    admission.clock.now += 301
    service.recover("owner", "old-0", True)
    service.recovery_reader.assert_not_called()


def test_paused_capacity_proof_keeps_progress_grace_and_rejects_invalid_controls(tmp_path):
    service, admission, _ = released_original(tmp_path, "unreadable")
    original = raw(service)
    progress = original["_result_observation_started_at"]
    marker = original["_result_read_capacity_released_at"]
    paused = {**original, "_recovery_paused": True,
              "_result_read_capacity_released_at": progress + 1}
    # Pausing cannot shorten the original progress grace. Clock passage alone
    # can mature the same proof, exactly as it would without a pause.
    assert not service._bounded_unreadable_result_capacity_boundary(paused, progress + 59)
    assert service._bounded_unreadable_result_capacity_boundary(paused, marker)
    for changes in ({"_recovery_paused": 1}, {"_recovery_paused": "paused"},
                    {"_recovery_suppressed": True}, {"_executing": True},
                    {"_result_read_capacity_released_at": None}):
        assert not service._bounded_unreadable_result_capacity_boundary(
            {**paused, **changes}, admission.clock.now)


def complete(receipt):
    doc = document(receipt)
    doc["mapping"]["final-" + receipt["request_message_id"]]["message"]["content"]["parts"] = ["saved original"]
    return ConversationBindingService._read_text_request_result(None, receipt, document=doc)


def test_pause_restart_resume_preserves_identity_protection_and_other_scopes(tmp_path):
    text, admission, _, _ = migration(tmp_path)
    text._update("owner", "old-0", _turn_reserved=True, recovery_next_at=2000,
                 recovery_no_result_reads=99, recovery_reason="request_not_found",
                 _input_ref="retained-input")
    with text.store.transaction() as db:
        text.store.set_runtime(db, "work:original", {"state": "paused"})
    original = raw(text)
    reader = Mock(side_effect=complete)
    text.recovery_reader = reader
    assert text.store.set_recovery_paused("text", "owner", "old-0", True)["state"] == "paused"
    paused = raw(text)
    assert {k:v for k,v in paused.items() if not k.startswith("_recovery_")} == original
    assert text.store.set_recovery_paused("text", "other", "old-0", False) is None
    assert text.store.set_recovery_paused("text", "owner", "old-0", True)["state"] == "paused"
    repeated = raw(text)
    assert repeated.pop("_recovery_control_version") == paused["_recovery_control_version"] + 1
    assert repeated == {k: v for k, v in paused.items() if k != "_recovery_control_version"}
    paused = raw(text)  # state and timestamp are idempotent; each command advances the version
    restarted = TextTaskService(text.store.path, admission=admission, clock=admission.clock,
                                recovery_reader=reader, runner=Mock())
    admission.clock.now = 3000
    admission.recoveries["text"] = restarted.read
    for _ in range(3):
        assert restarted.read("owner", "old-0")["status"] == "unknown"
        assert restarted.recover("owner", "old-0", True)["status"] == "unknown"
        admission.recover_one()
    reader.assert_not_called()
    assert raw(text) == paused
    assert unknown_text_result(paused) and unfinished("text", paused)
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    assert restarted.store.set_recovery_paused("text", "owner", "old-0", False)["state"] == "active"
    with text.store.connect() as db:
        assert text.store.runtime(db, "work:original")["state"] == "paused"
    admission.recover_one()
    assert reader.call_count == 1
    assert reader.call_args.args[0]["request_message_id"] == original["request_message_id"]
    assert raw(text)["content"] == "saved original"
    assert raw(text)["_input_ref"] == "retained-input"
    restarted.runner.assert_not_called()


def test_resume_never_bypasses_cooldown_or_operator_stop(tmp_path):
    text, admission, _, _ = migration(tmp_path)
    text._update("owner", "old-0", recovery_next_at=2000, _recovery_suppressed=True)
    text.recovery_reader = Mock(side_effect=complete)
    for desired in (True, False):
        control = text.store.set_recovery_paused("text", "owner", "old-0", desired)
        assert control["state"] == ("paused" if desired else "stopped")
        assert control["operator_stopped"] is True
    assert raw(text)["recovery_next_at"] == 2000
    assert raw(text)["_recovery_suppressed"] is True
    assert text.read("owner", "old-0")["recovery_control"]["state"] == "stopped"
    text.read("owner", "old-0")
    text.recovery_reader.assert_not_called()
    text._update("owner", "old-0", _recovery_suppressed=False)
    text.read("owner", "old-0")
    text.recovery_reader.assert_not_called()
    admission.clock.now = 2001
    assert text.read("owner", "old-0")["content"] == "saved original"


def test_pause_during_claim_accepts_late_result_without_a_second_reader(tmp_path):
    text, admission, _, _ = migration(tmp_path)
    started, finish = Event(), Event()
    def read(receipt):
        started.set()
        assert finish.wait(5)
        return complete(receipt)
    text.recovery_reader = Mock(side_effect=read)
    peer = TextTaskService(text.store.path, admission=admission, recovery_reader=Mock(), clock=admission.clock)
    with ThreadPoolExecutor(1) as workers:
        future = workers.submit(text.read, "owner", "old-0")
        assert started.wait(3)
        try:
            control = peer.store.set_recovery_paused("text", "owner", "old-0", True)
            assert control["state"] == "pausing" and control["in_flight"]
            assert peer.read("owner", "old-0")["status"] == "unknown"
            peer.recovery_reader.assert_not_called()
        finally:
            finish.set()
        assert future.result(timeout=3)["content"] == "saved original"
    assert text.read("owner", "old-0")["recovery_control"]["state"] == "paused"
    assert raw(text)["_recovery_paused"] is True
    assert text.recovery_reader.call_count == 1


def image_original(images, owner="owner", rid="old-image"):
    task = {"id": rid, "owner_id": owner, "status": "error", "model": "gpt-image-2",
            "mode": "generate", "error_code": "CONVERSATION_OUTCOME_UNKNOWN", "upstream_unfinished": True,
            "conversation_id": "original-conversation", "request_message_id": "original-message",
            "provider_binding_id": "original-binding", "provider_account_identity": "original-account",
            "next_poll_at": 0, "result_file_ids": ["original-file"], "updated_at": "2000-01-01T00:00:00Z"}
    with images.store.transaction() as db:
        images.store.write_receipt(db, "image", owner, rid, task)
    return task


def test_image_pause_survives_restart_retention_and_resume_never_generates(tmp_path, monkeypatch):
    images = ImageTaskService(tmp_path / "images.json", generation_handler=Mock())
    task = image_original(images)
    images.store.set_recovery_paused("image", "owner", "old-image", True)
    restarted = ImageTaskService(tmp_path / "images.json", generation_handler=Mock())
    assert raw(restarted, "image", rid="old-image")["result_file_ids"] == ["original-file"]
    poll = Mock()
    monkeypatch.setattr(restarted, "_run_resume_poll", poll)
    for _ in range(2):
        assert restarted.resume_poll({"id": "owner"}, "old-image")["recovery_control"]["state"] == "paused"
    poll.assert_not_called()
    assert unfinished("image", raw(restarted, "image", rid="old-image"))
    restarted.store.set_recovery_paused("image", "owner", "old-image", False)
    class InlineThread:
        def __init__(self, target, args, **kwargs): self.target, self.args = target, args
        def start(self): self.target(*self.args)
    monkeypatch.setattr("services.image_task_service.threading.Thread", InlineThread)
    restarted.resume_poll({"id": "owner"}, "old-image")
    assert poll.call_count == 1 and poll.call_args.args[1] == task["conversation_id"]
    restarted.generation_handler.assert_not_called()


@pytest.mark.parametrize("kind", ["text", "image"])
def test_company_and_ordinary_control_are_owner_scoped_and_do_not_read(company, kind):
    owner = company_identity("company", "employee", CONNECTOR)["id"]
    service = company.text_tasks if kind == "text" else company.images
    rid = "original"
    if kind == "text":
        with service.store.transaction() as db:
            db.execute("INSERT INTO requests VALUES(?,?,?,?)", (owner, rid, "hash", json.dumps({
                "request_id":rid, "status":"unknown", "boot":service.boot, "model":"gpt-text"})))
        service.recovery_reader = Mock()
    else:
        image_original(service, owner, rid)
    path = f"/api/{'chat-requests' if kind == 'text' else 'image-tasks'}/{rid}/recovery-control"
    response = company.client.post(PREFIX + path, headers=company.headers(), json={"state":"paused"})
    assert response.status_code == 200, response.text
    assert response.json()["recovery_control"]["state"] == "paused"
    assert response.headers["cache-control"] == "private, no-store"
    assert company.client.post(PREFIX + path, headers=company.headers(user="other"), json={"state":"active"}).status_code == 404
    assert company.client.post(path, headers={"Authorization":"Bearer " + company.ordinary}, json={"state":"active"}).status_code == 404
    assert company.client.post(PREFIX + path, headers=company.headers(), json={"state":"active", "owner":owner}).status_code == 422
    assert company.client.post(PREFIX + path, headers=company.headers(), json={"state":"clear"}).status_code == 422
    assert company.client.post(PREFIX + path, headers=company.headers(), json={"state":"active"}).status_code == 200
    if kind == "text": service.recovery_reader.assert_not_called()


def test_legacy_happy_control_uses_original_authenticated_owner(company, monkeypatch):
    monkeypatch.setattr(ai, "text_task_service", company.text_tasks)
    identity = company.auth.authenticate(company.admin)
    owner = identity["id"]
    with company.text_tasks.store.transaction() as db:
        db.execute("INSERT INTO requests VALUES(?,?,?,?)", (owner, "legacy", "hash", json.dumps({
            "request_id":"legacy", "status":"unknown", "boot":company.text_tasks.boot})))
    response = company.client.post("/api/conversation-bindings/text-requests/legacy/recovery-control",
                                   headers={"Authorization":"Bearer " + company.admin}, json={"state":"paused"})
    assert response.status_code == 200, response.text
    assert response.json()["recovery_control"]["state"] == "paused"


def test_paused_image_interrupted_worker_stays_unknown_and_keeps_backoff(tmp_path):
    images = ImageTaskService(tmp_path / "images.json")
    image_original(images)
    with images.store.transaction() as db:
        task = images.store.read_receipt(db, "image", "owner", "old-image")
        task.update(status="running", next_poll_at=9999999999, _recovery_paused=True)
        images.store.write_receipt(db, "image", "owner", "old-image", task)
    restarted = ImageTaskService(tmp_path / "images.json")
    receipt = raw(restarted, "image", rid="old-image")
    assert receipt["status"] == "error" and receipt["error_code"] == "CONVERSATION_OUTCOME_UNKNOWN"
    assert receipt["_recovery_paused"] is True and receipt["next_poll_at"] == 9999999999
    assert receipt["result_file_ids"] == ["original-file"]
    assert restarted.resume_poll({"id":"owner"}, "old-image")["recovery_control"]["state"] == "paused"


def test_cli_reports_other_stop_without_clearing_it(tmp_path, capsys):
    from examples.image_client import _command_recovery_control, ClientError
    from argparse import Namespace
    args = Namespace(command="chat-recovery-resume", state=str(tmp_path / "none"), request_id="original")
    api = Mock()
    api.json.return_value = {"request_id":"original", "recovery_control":{
        "state":"stopped", "operator_stopped":True, "in_flight":False}}
    with pytest.raises(ClientError, match="another operator scope"):
        _command_recovery_control(api, args)
    assert json.loads(capsys.readouterr().out)["recovery_control"]["state"] == "stopped"
    assert api.json.call_count == 1
    assert api.json.call_args.kwargs["payload"] == {"state":"active"}


def test_pause_during_failed_read_preserves_unknown_and_turn(tmp_path):
    from services.conversation_binding_service import ConversationBindingError
    text, admission, _, _ = migration(tmp_path)
    text._update("owner", "old-0", _turn_reserved=True, recovery_no_result_reads=99)
    def reader(receipt):
        text.store.set_recovery_paused("text", "owner", "old-0", True)
        raise ConversationBindingError("fixture failure", code="UPSTREAM_OUTCOME_UNKNOWN")
    text.recovery_reader = reader
    receipt = text.recover("owner", "old-0", True)
    assert receipt["status"] == "unknown" and receipt["recovery_control"]["state"] == "paused"
    assert raw(text)["_turn_reserved"] is True
    assert admission.resource_snapshot()["chat_turn"]["inflight"] == 1


def test_private_image_admin_control_and_body_limit(company):
    owner = company.auth.authenticate(company.admin)["id"]
    image_original(company.images, owner, "legacy-image")
    path = "/api/image-tasks/legacy-image/recovery-control"
    headers = {"Authorization":"Bearer " + company.admin}
    result = company.client.post(path, headers=headers, json={"state":"paused"})
    assert result.status_code == 200 and result.json()["recovery_control"]["state"] == "paused"
    assert company.client.post(path, headers=headers, json={"state":"active", "padding":"a"*1100}).status_code == 413
    assert company.client.post(path, headers={**headers,"X-Workbench-Image-Client":"1"}, json={"state":"active"}).status_code == 403
    assert raw(company.images,"image",owner,"legacy-image")["_recovery_paused"] is True


def seed_control_receipt(store, kind, owner, receipt):
    with store.transaction() as db:
        if kind == "text":
            db.execute("INSERT INTO requests VALUES(?,?,?,?)", (owner, "original", "original-hash", json.dumps(receipt)))
        else:
            store.write_receipt(db, kind, owner, "original", receipt)


@pytest.mark.parametrize("kind", ["text", "image"])
def test_conditional_restore_does_not_override_a_later_same_state_command(tmp_path, kind):
    store = TaskStore(tmp_path / "tasks.sqlite")
    original = {"status": "unknown", "content": "original", "recovery_next_at": 2000,
                "recovery_claim_id": "original-claim", "_turn_reserved": True,
                "_input_ref": "original-input", "conversation_id": "original-conversation"}
    seed_control_receipt(store, kind, "owner", original)
    paused = store.set_recovery_paused(kind, "owner", "original", True, expected_version=0)
    assert paused["version"] == 1 and paused["state"] == "pausing"
    # A legacy caller repeats pause while the first caller holds version 1.
    peer = TaskStore(store.path)
    assert peer.set_recovery_paused(kind, "owner", "original", True)["version"] == 2
    with store.connect() as db:
        before = store.read_receipt(db, kind, "owner", "original")
    with pytest.raises(RecoveryControlConflict) as error:
        store.set_recovery_paused(kind, "owner", "original", False, expected_version=1)
    assert error.value.control["version"] == 2
    with store.connect() as db:
        assert store.read_receipt(db, kind, "owner", "original") == before
    restarted = TaskStore(store.path)
    assert restarted.set_recovery_paused(kind, "owner", "original", False, expected_version=2)["version"] == 3
    with store.connect() as db:
        restored = store.read_receipt(db, kind, "owner", "original")
    assert {k: v for k, v in restored.items() if not k.startswith("_recovery_")} == original


def test_competing_control_commands_compare_and_write_atomically(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite")
    seed_control_receipt(store, "text", "owner", {"status": "unknown"})
    def change():
        try:
            return TaskStore(store.path).set_recovery_paused("text", "owner", "original", True, expected_version=0)
        except RecoveryControlConflict:
            return "conflict"
    with ThreadPoolExecutor(2) as workers:
        outcomes = list(workers.map(lambda _: change(), range(2)))
    assert outcomes.count("conflict") == 1
    assert [v["version"] for v in outcomes if isinstance(v, dict)] == [1]


@pytest.mark.parametrize("kind", ["text", "image"])
def test_company_control_conflict_does_not_mutate_or_wake(company, kind):
    owner = company_identity("company", "employee", CONNECTOR)["id"]
    service = company.text_tasks if kind == "text" else company.images
    seed_control_receipt(service.store, kind, owner, {"status": "unknown"})
    service.admission = Mock()
    path = PREFIX + f"/api/{'chat-requests' if kind == 'text' else 'image-tasks'}/original/recovery-control"
    paused = company.client.post(path, headers=company.headers(), json={"state": "paused", "expected_version": 0})
    assert paused.status_code == 200 and paused.json()["recovery_control"]["version"] == 1
    before = raw(service, kind, owner, "original")
    stale = company.client.post(path, headers=company.headers(), json={"state": "active", "expected_version": 0})
    assert stale.status_code == 409, stale.text
    assert stale.json()["detail"]["code"] == "RECOVERY_CONTROL_CONFLICT"
    assert stale.json()["detail"]["recovery_control"]["version"] == 1
    assert raw(service, kind, owner, "original") == before
    service.admission.wake.assert_not_called()
    assert company.client.post(path, headers=company.headers(user="other"),
                               json={"state": "active", "expected_version": 1}).status_code == 404
    resumed = company.client.post(path, headers=company.headers(), json={"state": "active", "expected_version": 1})
    assert resumed.status_code == 200 and resumed.json()["recovery_control"]["version"] == 2
    service.admission.wake.assert_called_once()


@pytest.mark.parametrize("value", [None, True, False, -1, 1.0, "1"])
def test_control_expected_version_rejects_ambiguous_values(value):
    from pydantic import ValidationError
    from api.recovery_control import RecoveryControlRequest
    with pytest.raises(ValidationError):
        RecoveryControlRequest(state="active", expected_version=value)
    assert RecoveryControlRequest(state="active").expected_version is None


@pytest.mark.parametrize("version", [None, True, -1, "1"])
def test_invalid_stored_version_never_resets_control(tmp_path, version):
    store = TaskStore(tmp_path / "tasks.sqlite")
    original = {"status": "unknown", "_recovery_paused": True, "_recovery_control_version": version}
    seed_control_receipt(store, "text", "owner", original)
    with pytest.raises(RecoveryControlConflict):
        store.set_recovery_paused("text", "owner", "original", False)
    with store.connect() as db:
        assert store.read_receipt(db, "text", "owner", "original") == original
