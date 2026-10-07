from __future__ import annotations

import base64
import io
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import ai, chat_requests, image_tasks
from api.app import create_app
from api.company_requests import PREFIX, PUBLIC_PREFIX, company_identity
from services.auth_service import AuthService
from services.image_task_service import ImageTaskService
from services.storage.json_storage import JSONStorageBackend
from services.text_task_service import TextTaskService

CONNECTOR = "1f084f01-d4b2-4080-8bce-b926f31cc454"
OTHER_CONNECTOR = "1f084f01-d4b2-4080-8bce-b926f31cc455"


class Queue:
    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        self.calls.append((function, args))

    def run(self):
        function, args = self.calls.pop(0)
        function(*args)


@pytest.fixture
def company(tmp_path, monkeypatch):
    tmp_path = tmp_path / "company"
    tmp_path.mkdir()
    auth = AuthService(JSONStorageBackend(tmp_path / "accounts.json"))
    _, admin = auth.create_key(role="admin")
    old_key, ordinary = auth.create_key(role="user", routes=["chat"])
    monkeypatch.setattr("api.support.auth_service", auth)
    queue = Queue()
    runner = Mock(return_value={"content": "answer", "provider_binding_id": "private-binding"})
    text_tasks = TextTaskService(tmp_path / "text.sqlite3", runner=runner, executor=queue)
    monkeypatch.setattr(chat_requests, "text_task_service", text_tasks)
    monkeypatch.setattr(chat_requests, "check_request", lambda *_: None)
    monkeypatch.setattr(image_tasks, "check_request", lambda *_: None)
    monkeypatch.setattr("services.log_service.log_service.add", lambda *args, **kwargs: None)
    monkeypatch.setattr("services.public_chat_service.model_catalog_service.route_for_model",
                        lambda _: SimpleNamespace(account_types=frozenset({"Plus"}), allow_anonymous=False))
    monkeypatch.setattr(ai.openai_v1_models, "list_models", lambda: {
        "data": [{"id": "gpt-text"}, {"id": "gpt-image-2"}], "object": "list"})
    png = io.BytesIO()
    Image.new("RGB", (2, 2)).save(png, format="PNG")
    image_calls = []

    def generate(payload):
        image_calls.append(payload)
        return {"data": [{"b64_json": base64.b64encode(png.getvalue()).decode()}],
                "_provider_binding_id": payload["provider_binding_id"],
                "_provider_account_identity": payload["provider_account_identity"],
                "_conversation_id": "original-conversation", "_parent_message_id": "original-message"}

    images = ImageTaskService(tmp_path / "images.json", generation_handler=generate, edit_handler=generate)
    monkeypatch.setattr(image_tasks, "image_task_service", images)
    monkeypatch.setattr("services.account_service.account_service.create_conversation_binding",
                        lambda **_: ("private-binding", "private-account", "private-token"))
    monkeypatch.setattr("services.account_service.account_service.release_image_slot", lambda *_: None)
    # Image dispatch now requires fresh observed image_gen evidence. This is
    # the real capacity gate, not a mocked bypass of final dispatch.
    monkeypatch.setattr("services.account_service.account_service.get_account", lambda *_: {
        "access_token": "private-token", "quota": 2, "status": "正常", "source_type": "web", "type": "Plus",
        "limits_progress": [{"feature_name": "image_gen", "remaining": 2}],
        "capacity_observed_at": datetime.now(timezone.utc).isoformat(),
    })

    def headers(user="employee", org="company", connector=CONNECTOR):
        return {"Authorization": "Bearer " + admin, "X-Workbench-Company-Org": org,
                "X-Workbench-Company-User": user, "X-Workbench-Company-Connector": connector,
                "X-Workbench-Expected-User": user}

    return SimpleNamespace(client=TestClient(create_app()), headers=headers, auth=auth, admin=admin,
                           ordinary=ordinary, old_key=old_key, runner=runner, queue=queue,
                           text_tasks=text_tasks, images=images, image_calls=image_calls, png=png.getvalue())


def body(text="hello"):
    return {"client_request_id": "original-chat", "model": "gpt-text",
            "messages": [{"role": "user", "content": text}]}


@pytest.mark.parametrize("change,status", [
    ({"Authorization": "Bearer invalid"}, 401),
    ({"X-Workbench-Expected-User": "someone"}, 409),
    ({"X-Workbench-Company-Org": ""}, 400),
    ({"X-Workbench-Company-Connector": "not-a-uuid"}, 400),
    ({"X-Workbench-Image-Client": "1"}, 403),
])
def test_company_requires_private_credential_and_complete_identity(company, change, status):
    response = company.client.get(PREFIX + "/session", headers={**company.headers(), **change})
    assert response.status_code == status
    assert company.admin not in response.text


def test_company_never_exposes_management_or_codex(company):
    for path in ("/api/accounts", "/api/workbench/ai/keys", "/v1/responses", "/codex/v1/models"):
        assert company.client.get(PREFIX + path, headers=company.headers()).status_code == 404
    assert company.client.get(PREFIX + "/session", headers={
        **company.headers(), "Authorization": "Bearer " + company.ordinary}).status_code == 403
    session = company.client.get(PREFIX + "/session", headers=company.headers()).json()
    assert session == {"contract_version": 1, "org_id": "company", "user_id": "employee", "connector_id": CONNECTOR}
    assert company.client.get(PREFIX + "/v1/models", headers=company.headers()).json()["data"][0]["id"] == "gpt-text"


def test_durable_chat_restart_cookie_renewal_and_old_key_isolation(company, monkeypatch):
    response = company.client.post(PREFIX + "/api/chat-requests", headers=company.headers(), json=body())
    assert response.status_code == 202, response.text
    company.queue.run()
    restarted = TextTaskService(company.text_tasks.path, runner=company.runner, executor=Queue())
    monkeypatch.setattr(chat_requests, "text_task_service", restarted)
    renewed = {**company.headers(), "Cookie": "company-session=renewed"}
    read = company.client.get(PREFIX + "/api/chat-requests/original-chat", headers=renewed)
    assert read.status_code == 200 and read.json()["content"] == "answer"
    assert "private-binding" not in read.text
    assert company.client.post(PREFIX + "/api/chat-requests", headers=renewed, json=body()).status_code == 200
    assert company.client.post(PREFIX + "/api/chat-requests", headers=renewed, json=body("changed")).status_code == 409
    assert company.runner.call_count == 1
    for headers in (company.headers(user="other"), company.headers(org="other"),
                    company.headers(connector=OTHER_CONNECTOR)):
        assert company.client.get(PREFIX + "/api/chat-requests/original-chat", headers=headers).status_code == 404
        assert company.client.post(PREFIX + "/api/chat-requests/original-chat/recover", headers=headers, json={}).status_code == 404
    # Forging company metadata on a normal-key route cannot change its owner.
    public = {**company.headers(), "Authorization": "Bearer " + company.ordinary, "X-Workbench-Image-Client": "1"}
    assert company.client.get("/api/chat-requests/original-chat", headers=public).status_code == 404
    original = company.client.post("/api/chat-requests", headers=public, json=body("legacy"))
    assert original.status_code == 202
    assert restarted.read(company.old_key["id"], "original-chat")["status"] != "not_found"
    assert company.client.get(PREFIX + "/api/chat-requests/original-chat", headers=company.headers()).json()["content"] == "answer"


def test_company_unknown_submission_is_not_reexecuted_after_restart(company, monkeypatch):
    assert company.client.post(PREFIX + "/api/chat-requests", headers=company.headers(), json=body()).status_code == 202
    owner = company_identity("company", "employee", CONNECTOR)["id"]
    company.text_tasks._update(owner, "original-chat", _input_ref=None)  # Legacy input-less receipt.
    queue = Queue()
    restarted = TextTaskService(company.text_tasks.path, runner=company.runner, executor=queue)
    monkeypatch.setattr(chat_requests, "text_task_service", restarted)
    original = company.client.get(PREFIX + "/api/chat-requests/original-chat", headers=company.headers())
    assert original.json()["status"] == "not_started"
    repeated = company.client.post(PREFIX + "/api/chat-requests", headers=company.headers(), json=body())
    assert repeated.status_code == 202
    assert repeated.json()["status"] == "not_started"
    assert queue.calls == []
    company.runner.assert_not_called()


def test_company_explicitly_resumes_only_the_same_known_unsent_correction(company):
    owner = company_identity("company", "employee", CONNECTOR)["id"]
    original_body = {**body(), "client_conversation_id": "work"}
    original = company.client.post(PREFIX + "/api/chat-requests", headers=company.headers(), json=original_body)
    assert original.status_code == 202, original.text
    with company.text_tasks._db() as db:
        original_receipt = company.text_tasks.store.read_receipt(db, "text", owner, "original-chat")
    evidence = {"conversation_id": "original-conversation",
                "request_message_id": original_receipt["request_message_id"],
                "final_message_id": "empty-final", "observed_at": 100.0}
    company.text_tasks._update(owner, "original-chat", status="unknown",
                               error_code="CONVERSATION_OUTCOME_UNKNOWN",
                               recovery_reason="REQUEST_RESULT_TERMINAL_EMPTY", _upstream_terminal=True,
                               provider_binding_id="private-binding",
                               provider_account_identity="private-account",
                               conversation_id="original-conversation", _turn_end_evidence=evidence)
    company.queue.calls.clear()

    def read_original(receipt):
        return {"status": "unknown", "recovery_reason": "REQUEST_RESULT_TERMINAL_EMPTY",
                "provider_binding_id": receipt["provider_binding_id"],
                "provider_account_identity": receipt["provider_account_identity"],
                "client_conversation_id": receipt["client_conversation_id"],
                "conversation_id": receipt["conversation_id"],
                "_turn_end_evidence": {**evidence, "observed_at": 101.0}}

    company.text_tasks.recovery_reader = read_original
    correction_body = {**body("correction"), "client_request_id": "correction-chat",
                       "client_conversation_id": "work", "previous_request_id": "original-chat",
                       "continue_after_terminal_empty": True}
    accepted = company.client.post(PREFIX + "/api/chat-requests", headers=company.headers(), json=correction_body)
    assert accepted.status_code == 202, accepted.text
    company.queue.calls.clear()
    company.text_tasks._update(owner, "correction-chat", status="failed",
                               error_code="CHAT_TERMINAL_EMPTY_UNVERIFIED", upstream_outcome="not_sent",
                               _submission_started=False, _turn_reserved=False, _executing=False,
                               _claim_until=0)
    wrong_owner = company.client.post(PREFIX + "/api/chat-requests/correction-chat/recover",
                                      headers=company.headers(connector=OTHER_CONNECTOR),
                                      json={"resume_unsent_correction": True})
    assert wrong_owner.status_code == 404
    resumed = company.client.post(PREFIX + "/api/chat-requests/correction-chat/recover",
                                  headers=company.headers(), json={"resume_unsent_correction": True})
    assert resumed.status_code == 202, resumed.text
    assert resumed.json()["request_id"] == "correction-chat"
    assert len(company.queue.calls) == 1
    company.queue.run()
    result = company.client.get(PREFIX + "/api/chat-requests/correction-chat", headers=company.headers())
    assert result.status_code == 200 and result.json()["content"] == "answer"
    assert company.runner.call_count == 1


def test_company_multipart_images_query_download_and_recovery(company, monkeypatch):
    response = company.client.post(PREFIX + "/api/image-tasks/edits", headers=company.headers(),
        data={"client_task_id": "original-image", "prompt": "edit", "model": "gpt-image-2"},
        files=[("image", ("one.png", company.png, "image/png")),
               ("image", ("two.png", company.png, "image/png"))])
    assert response.status_code == 200, response.text
    for _ in range(100):
        read = company.client.get(PREFIX + "/api/image-tasks?ids=original-image", headers=company.headers())
        if read.json()["items"][0]["status"] in {"success", "error"}:
            break
        time.sleep(.01)
    task = read.json()["items"][0]
    assert task["status"] == "success", task
    assert task["data"] == [{"url": PUBLIC_PREFIX + "/api/image-tasks/original-image/images/0"}]
    assert "private-" not in read.text
    assert len(company.image_calls) == 1
    assert len(company.image_calls[0]["images"]) == 2
    duplicate = company.client.get(PREFIX + "/api/image-tasks?ids=original-image,original-image", headers=company.headers())
    assert duplicate.status_code == 200
    restarted = ImageTaskService(company.images.path, generation_handler=Mock(), edit_handler=Mock())
    monkeypatch.setattr(image_tasks, "image_task_service", restarted)
    download = company.client.get(PREFIX + "/api/image-tasks/original-image/images/0", headers=company.headers())
    assert download.content == company.png
    assert download.headers["cache-control"] == "private, no-store"
    for headers in (company.headers(user="other"), company.headers(connector=OTHER_CONNECTOR)):
        assert company.client.get(PREFIX + "/api/image-tasks?ids=original-image", headers=headers).status_code == 404
        assert company.client.get(PREFIX + "/api/image-tasks/original-image/images/0", headers=headers).status_code == 404
        assert company.client.post(PREFIX + "/api/image-tasks/original-image/resume-poll", headers=headers, json={}).status_code == 404
    forbidden = company.client.post(PREFIX + "/api/image-tasks/original-image/resume-poll", headers=company.headers(),
                                    json={"allow_unrecoverable_retry": True})
    assert forbidden.status_code == 400
    recovered = company.client.post(PREFIX + "/api/image-tasks/original-image/resume-poll", headers=company.headers(), json={})
    assert recovered.status_code == 200 and recovered.json()["status"] == "success"
    restarted.generation_handler.assert_not_called()
    restarted.edit_handler.assert_not_called()


def test_company_identity_is_stable_and_not_key_ownership():
    a = company_identity("org", "user", CONNECTOR)
    assert a == company_identity("org", "user", CONNECTOR)
    assert a["id"] != company_identity("other", "user", CONNECTOR)["id"]
    assert a["role"] == "user" and a["policy"]["routes"] == ["chat"]


def test_multiple_company_connectors_share_user_fairness_without_sharing_receipts():
    from services.request_context import trusted_source
    one = company_identity("org", "person", CONNECTOR)
    two = company_identity("org", "person", OTHER_CONNECTOR)
    other = company_identity("org", "colleague", CONNECTOR)
    request = SimpleNamespace(state=SimpleNamespace(company_identity=True), headers={"x-workbench-consumer": "happy"})
    assert one["id"] != two["id"]
    assert trusted_source(one, request) == trusted_source(two, request)
    assert trusted_source(one, request) != trusted_source(other, request)
    # The same field on a normal external credential grants no scheduling lane.
    request.state.company_identity = None
    assert trusted_source(one, request) == "key:" + one["id"]


# Reuse the real durable recovery fixture; only its upstream is controlled.
from test.test_generation_completion import setup, failed_unsent_image


@pytest.fixture
def company_repair(company, failed_unsent_image, monkeypatch):
    import api.generation_completion as completion_api
    service = failed_unsent_image
    identity = company_identity("company", "employee", CONNECTOR)
    with service.store.transaction() as db:
        original = service.store.read_receipt(db, "image", "owner", "repair-image")
        original["owner_id"] = identity["id"]
        source = service.store.load_input(original["_input_ref"])
        source["identity"] = identity
        original["_input_ref"] = service.store.save_input(source)
        service.store.write_receipt(db, "image", identity["id"], "repair-image", original)
        db.execute("DELETE FROM image_requests WHERE task_key=?", ("owner:repair-image",))
    monkeypatch.setattr(completion_api, "get_generation_completion_service", lambda: service)
    return company, service, identity


def company_image_row(service, identity):
    with service.store.connect() as db:
        return service.store.read_receipt(db, "image", identity["id"], "repair-image")


def test_company_completion_restores_only_owned_unsent_original(company_repair, monkeypatch):
    from services.generation_completion import GenerationCompletionService
    company, service, identity = company_repair
    endpoint = PREFIX + "/api/image-tasks/repair-image/completion"
    payload = {"action": "recover", "allow_unconfirmed_retry": False,
               "retry_not_sent_failure_at": 2900.0}
    original = company_image_row(service, identity)
    for headers in (company.headers(user="other"), company.headers(org="other"),
                    company.headers(connector=OTHER_CONNECTOR)):
        assert company.client.get(endpoint, headers=headers).status_code == 404
        assert company.client.post(endpoint, headers=headers, json=payload).status_code == 404
    assert company_image_row(service, identity) == original
    assert company.client.get(endpoint, headers=company.headers()).status_code == 200
    assert company.client.post(endpoint, headers=company.headers(),
                               json={**payload, "retry_not_sent_failure_at": 2899.0}).status_code == 409
    assert company_image_row(service, identity) == original
    response = company.client.post(endpoint, headers=company.headers(), json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["original_id"] == "repair-image"
    assert "replacement_id" not in response.json()
    restored = company_image_row(service, identity)
    assert restored["status"] == "queued" and restored["_completion"]["max_extra_requests"] == 0
    assert restored["_input_ref"] == original["_input_ref"]
    assert restored["request_hash"] == original["request_hash"]
    restarted = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=service.clock)
    monkeypatch.setattr("api.generation_completion.get_generation_completion_service", lambda: restarted)
    assert company.client.post(endpoint, headers=company.headers(), json=payload).status_code == 200
    assert company_image_row(service, identity)["_execution_timeline"] == restored["_execution_timeline"]
    with service.store.connect() as db:
        assert db.execute("SELECT count(*) FROM image_requests").fetchone()[0] == 1


@pytest.mark.parametrize("payload", [
    {"action": "recover", "allow_unconfirmed_retry": True},
    {"action": "recover", "reviewed": False},
    {"action": "recover", "results_saved": False},
    {"action": "recover", "selected_id": None},
    {"action": "recover", "unexpected": "field"},
])
def test_company_completion_rejects_recovery_authority_and_extra_fields(company_repair, payload):
    company, service, identity = company_repair
    before = company_image_row(service, identity)
    response = company.client.post(PREFIX + "/api/image-tasks/repair-image/completion",
                                   headers=company.headers(), json=payload)
    assert response.status_code in (403, 422), response.text
    assert company_image_row(service, identity) == before


def test_company_completion_saved_original_never_regenerates(company_repair):
    company, service, identity = company_repair
    with service.store.transaction() as db:
        original = service.store.read_receipt(db, "image", identity["id"], "repair-image")
        original.update(status="success", data=[{"b64_json": base64.b64encode(company.png).decode()}])
        service.store.write_receipt(db, "image", identity["id"], "repair-image", original)
    service.images.resume_poll = Mock(side_effect=AssertionError("saved result must not be polled"))
    response = company.client.post(PREFIX + "/api/image-tasks/repair-image/completion",
                                   headers=company.headers(), json={"action": "recover"})
    assert response.status_code == 200, response.text
    assert response.json()["selected_id"] == "repair-image"
    assert "replacement_id" not in response.json()
    assert company_image_row(service, identity)["data"] == original["data"]
    service.images.resume_poll.assert_not_called()


@pytest.fixture
def company_selected_result(company_repair, monkeypatch):
    """Saved selected result and a separate unresolved original physical work."""
    import copy
    from services.work_lifecycle import ensure_work

    company, service, identity = company_repair
    selected_id = "company-selected-image"
    with service.store.transaction() as db:
        original = service.store.read_receipt(db, "image", identity["id"], "repair-image")
        original.update(status="error", upstream_outcome="unknown", error_code="CONTENT_POLICY_VIOLATION",
                        _submission_started=True, upstream_unfinished=False, _attempt_finished_at=3000,
                        conversation_id="unknown-original", provider_binding_id="binding-original",
                        provider_account_identity="account-original", request_message_id="original-user",
                        _image_thread={"id": "original-thread"})
        original.pop("_work_key", None)
        original.pop("_work_ref", None)
        selected = copy.deepcopy(original)
        selected.update(id=selected_id, status="success", upstream_outcome="completed", error_code=None,
                        conversation_id="selected-conversation", parent_message_id="selected-final",
                        request_message_id="selected-user", _image_thread={"id": "selected-thread"},
                        _completion_of="repair-image", data=[{"b64_json": base64.b64encode(company.png).decode()}])
        selected.pop("_completion", None)
        original["_completion"] = {"state": "result_ready", "replacement_id": selected_id,
                                   "selected_id": selected_id, "max_extra_requests": 1, "next_at": None}
        for rid, receipt in (("repair-image", original), (selected_id, selected)):
            work = ensure_work(service.store, db, "image", identity["id"], rid, receipt)
            work["slot_held"] = True
            service.store.set_runtime(db, work["key"], work)
            service.store.write_receipt(db, "image", identity["id"], rid, receipt)
    monkeypatch.setattr(image_tasks, "image_task_service", service.images)
    monkeypatch.setattr(image_tasks, "get_work_lifecycle_service", lambda: service.lifecycle)
    service.images.generation_handler = Mock(side_effect=AssertionError("lifecycle must not generate"))
    service.images.resume_poll = Mock(side_effect=AssertionError("selected result must not repoll original"))

    def archive(identity_arg, rid, desired):
        assert identity_arg["id"] == identity["id"] and rid == selected_id
        return {"task_id": rid, "image_thread": {"id": "selected-thread"}, "archived": desired}

    service.images.set_thread_archived = Mock(side_effect=archive)
    return company, service, identity, selected_id


def test_company_selected_image_save_complete_archive_rework_keeps_original_unknown(company_selected_result):
    company, service, identity, selected_id = company_selected_result
    endpoint = PREFIX + "/api/image-tasks/repair-image/completion"
    headers = company.headers()
    ready = company.client.get(endpoint, headers=headers).json()
    assert ready["state"] == "result_ready" and ready["original_turn_ended"] is False
    assert ready["original_cleanup"] == "pending"
    downloaded = company.client.get(PREFIX + f"/api/image-tasks/{selected_id}/images/0", headers=headers)
    assert downloaded.status_code == 200 and downloaded.content == company.png
    original = company_image_row(service, identity)
    done = company.client.post(endpoint, headers=headers, json={
        "action": "complete", "selected_id": selected_id, "results_saved": True, "reviewed": True})
    assert done.status_code == 200, done.text
    result = done.json()
    assert result["state"] == "completed" and result["selected_id"] == selected_id
    assert result["work"]["slot_held"] is False and result["work"]["archive"]["status"] == "pending"
    assert result["original_cleanup"] == "pending" and result["original_work"]["slot_held"] is True
    assert result["original_turn_ended"] is False
    with service.store.connect() as db:
        chosen = service.store.read_receipt(db, "image", identity["id"], selected_id)
    pending = company.client.post(endpoint, headers=headers, json={"action": "rework", "selected_id": selected_id})
    assert pending.status_code == 409 and pending.json()["detail"]["code"] == "WORK_ARCHIVE_PENDING"
    service.lifecycle.process_one(target_key=chosen["_work_key"])
    rework = company.client.post(endpoint, headers=headers, json={"action": "rework", "selected_id": selected_id})
    assert rework.status_code == 200, rework.text
    assert rework.json()["work"]["state"] == "restoring"
    service.lifecycle.process_one(target_key=chosen["_work_key"])
    assert service.lifecycle.get("image", identity, selected_id)["state"] == "active"
    # The pre-existing thread endpoints also operate on the selected receipt.
    for action in ("archive-thread", "restore-thread"):
        response = company.client.post(PREFIX + f"/api/image-tasks/{selected_id}/{action}", headers=headers, json={})
        assert response.status_code == 200, response.text
    after = company_image_row(service, identity)
    assert after["status"] == original["status"] and after["upstream_outcome"] == "unknown"
    assert after["conversation_id"] == original["conversation_id"]
    assert service.read("image", identity, "repair-image")["original_work"]["slot_held"] is True
    service.images.generation_handler.assert_not_called()
    service.images.resume_poll.assert_not_called()


@pytest.mark.parametrize("action", ["complete", "rework"])
def test_company_selected_completion_checks_owner_and_exact_selected_result(company_selected_result, action):
    company, service, identity, selected_id = company_selected_result
    endpoint = PREFIX + "/api/image-tasks/repair-image/completion"
    payload = {"action": action, "selected_id": selected_id}
    if action == "complete":
        payload.update(results_saved=True, reviewed=True)
    before = company_image_row(service, identity)
    for headers in (company.headers(user="other"), company.headers(org="other"), company.headers(connector=OTHER_CONNECTOR)):
        assert company.client.post(endpoint, headers=headers, json=payload).status_code == 404
    response = company.client.post(endpoint, headers=company.headers(), json={**payload, "selected_id": "unrelated-image"})
    assert response.status_code == 409, response.text
    assert company_image_row(service, identity) == before
    service.images.set_thread_archived.assert_not_called()


def test_company_same_thread_selected_closes_only_after_saved_success(company_selected_result):
    from services.work_lifecycle import ensure_work
    company, service, identity, selected_id = company_selected_result
    with service.store.transaction() as db:
        original = service.store.read_receipt(db, "image", identity["id"], "repair-image")
        selected = service.store.read_receipt(db, "image", identity["id"], selected_id)
        db.execute("DELETE FROM task_runtime WHERE name=?", (selected["_work_key"],))
        original.update(_sequence=1, client_conversation_id="same-client", error_code="RESULT_UNRECOVERABLE",
                        _retry_cursor={"conversation_id": original["conversation_id"],
                                       "request_message_id": original["request_message_id"],
                                       "retry_parent_message_id": "verified-empty-final", "observed_at": service.clock()})
        selected.update({k: original[k] for k in ("provider_binding_id", "provider_account_identity",
                        "client_conversation_id", "conversation_id", "_work_key", "_image_thread")})
        selected.update(_sequence=2, _same_session_retry_of="repair-image",
                        _submission_parent_message_id="verified-empty-final")
        service.store.write_receipt(db, "image", identity["id"], "repair-image", original)
        ensure_work(service.store, db, "image", identity["id"], selected_id, selected)
        service.store.write_receipt(db, "image", identity["id"], selected_id, selected)
    endpoint = PREFIX + "/api/image-tasks/repair-image/completion"
    response = company.client.post(endpoint, headers=company.headers(), json={
        "action": "complete", "selected_id": selected_id, "results_saved": True, "reviewed": True})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["state"] == "completed" and result["original_status"] == "error"
    assert result["work"]["slot_held"] is False and result["work"]["archive"]["status"] == "pending"
    assert result["original_work"]["request_id"] == selected_id
    service.images.set_thread_archived.side_effect = lambda who, rid, desired: {
        "task_id": rid, "image_thread": {"id": "original-thread"}, "archived": desired}
    service.lifecycle.process_one(target_key=selected["_work_key"])
    service.images.set_thread_archived.assert_called_once_with({"id": identity["id"], "role": "user"}, selected_id, True)
    assert company_image_row(service, identity)["upstream_outcome"] == "unknown"
    response = company.client.post(endpoint, headers=company.headers(),
                                   json={"action": "rework", "selected_id": selected_id})
    assert response.status_code == 200 and response.json()["work"]["state"] == "restoring"
    service.images.generation_handler.assert_not_called()
    service.images.resume_poll.assert_not_called()


@pytest.mark.parametrize("action,extra", [
    ("complete", {"allow_unconfirmed_retry": False}),
    ("complete", {"retry_not_sent_failure_at": None}),
    ("complete", {"results_saved": "true"}),
    ("complete", {"reviewed": False}),
    ("rework", {"allow_unconfirmed_retry": False}),
    ("rework", {"results_saved": False}),
    ("rework", {"retry_not_sent_failure_at": None}),
])
def test_company_selected_lifecycle_does_not_expand_recovery_fields(company_selected_result, action, extra):
    company, service, identity, selected_id = company_selected_result
    payload = {"action": action, "selected_id": selected_id}
    if action == "complete":
        payload.update(results_saved=True, reviewed=True)
    before = company_image_row(service, identity)
    response = company.client.post(PREFIX + "/api/image-tasks/repair-image/completion",
                                   headers=company.headers(), json={**payload, **extra})
    assert response.status_code in (403, 422), response.text
    assert company_image_row(service, identity) == before
    service.images.set_thread_archived.assert_not_called()


@pytest.mark.parametrize("later_status", ["success", "running"])
@pytest.mark.parametrize("invalid", [None, "thread", "work_ref", "sequence", "work", "selected_asset"])
def test_company_intermediate_selected_ack_preserves_newer_work_and_restart(company_selected_result, monkeypatch,
                                                                          later_status, invalid):
    import copy
    from services.generation_completion import GenerationCompletionService
    from services.work_lifecycle import ensure_work

    company, service, identity, selected_id = company_selected_result
    with service.store.transaction() as db:
        selected = service.store.read_receipt(db, "image", identity["id"], selected_id)
        selected["_sequence"] = 2
        service.store.write_receipt(db, "image", identity["id"], selected_id, selected)
        later = copy.deepcopy(selected)
        later.update(id="later-image", _sequence=3, status=later_status, _completion_of=None,
                     _previous_request_id=selected_id, _executing=later_status == "running",
                     request_message_id="later-user")
        later["_image_thread"]["previous_task_id"] = selected_id
        if later_status == "running":
            later.pop("data", None)
        ensure_work(service.store, db, "image", identity["id"], "later-image", later)
        if invalid == "thread":
            later["_image_thread"]["id"] = "unrelated-thread"
        elif invalid == "work_ref":
            selected["_image_thread"]["id"] = later["_image_thread"]["id"] = "unrelated-thread"
            service.store.write_receipt(db, "image", identity["id"], selected_id, selected)
        elif invalid == "sequence":
            later["_sequence"] = selected["_sequence"]
        elif invalid == "work":
            later["_work_key"] = "unrelated-work"
        elif invalid == "selected_asset":
            selected["data"] = []
            service.store.write_receipt(db, "image", identity["id"], selected_id, selected)
        service.store.write_receipt(db, "image", identity["id"], "later-image", later)
        before_work = service.store.runtime(db, selected["_work_key"])
    endpoint = PREFIX + "/api/image-tasks/repair-image/completion"
    payload = {"action": "complete", "selected_id": selected_id, "results_saved": True, "reviewed": True}
    for restarted in (False, True):
        if restarted:
            service = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=service.clock)
            monkeypatch.setattr("api.generation_completion.get_generation_completion_service", lambda: service)
        response = company.client.post(endpoint, headers=company.headers(), json=payload)
        if invalid:
            assert response.status_code == 409, response.text
            assert company_image_row(service, identity)["_completion"]["state"] == "result_ready"
            with service.store.connect() as db:
                assert service.store.runtime(db, selected["_work_key"]) == before_work
            continue
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["state"] == "completed" and result["selected_id"] == selected_id
        assert result["results_saved"] is True
        assert result["work"]["request_id"] == "later-image" and result["work"]["state"] == "active"
        assert result["work"]["slot_held"] is True and result["work"]["archive"]["status"] == "not_requested"
        assert result["original_status"] == "error" and result["original_cleanup"] == "pending"
        with service.store.connect() as db:
            assert service.store.runtime(db, selected["_work_key"]) == before_work
            assert service.store.read_receipt(db, "image", identity["id"], "later-image") == later
        # Rework must not roll a shared conversation back over its later turn.
        before_root = company_image_row(service, identity)
        response = company.client.post(endpoint, headers=company.headers(),
                                       json={"action": "rework", "selected_id": selected_id})
        assert response.status_code == 409 and response.json()["detail"]["code"] == "WORK_SUPERSEDED"
        assert company_image_row(service, identity) == before_root
        assert not service.lifecycle.process_one()
        with service.store.connect() as db:
            assert service.store.runtime(db, selected["_work_key"]) == before_work
            assert service.store.read_receipt(db, "image", identity["id"], "later-image") == later
    service.images.generation_handler.assert_not_called()
    service.images.resume_poll.assert_not_called()
    service.images.set_thread_archived.assert_not_called()
