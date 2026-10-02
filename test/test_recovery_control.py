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
from test.test_company_requests import company, CONNECTOR
from test.test_unknown_turn_recovery import migration, document


def raw(service, kind="text", owner="owner", rid="old-0"):
    with service.store.connect() as db:
        return service.store.read_receipt(db, kind, owner, rid)


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
    assert raw(text) == paused  # idempotence, including timestamp
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
        text.store.set_recovery_paused("text", "owner", "old-0", desired)
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
