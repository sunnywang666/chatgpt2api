import copy
import json
from datetime import datetime, timezone
import pytest

from services.image_quota_successor import (
    FAILURE_REASON,
    excluded_account,
    eligible_original,
    is_successor,
    prepared_payload,
    registration_successor,
)
from services.generation_completion import GenerationCompletionService
from services.work_lifecycle import WorkLifecycleService


def root():
    return {
        "request_id": "original", "request_hash": "immutable-original", "owner_id": "owner",
        "status": "error", "error_code": "NO_IMAGE_GENERATED", "upstream_outcome": "failed",
        "upstream_unfinished": False, "_authoritative_failure_reason": FAILURE_REASON,
        "_submission_started": True, "provider_binding_id": "old-binding",
        "provider_account_identity": "old-account", "client_conversation_id": "old-client",
        "conversation_id": "old-conversation", "_product_work_key": "product-work",
        "_product_work": {"product_ref": "P", "step_ref": "image"}, "_source": "internal:listing",
    }


def child(original, *, sent=False, account="new-account"):
    replacement_id = "completion-child"
    payload = prepared_payload({"prompt": "persisted prompt", "model": "gpt-image-2"}, original, replacement_id)
    result = {
        "request_id": replacement_id, "owner_id": "owner", "_completion_of": "original",
        "_image_quota_failover": payload["_image_quota_failover"], "_product_work_key": "product-work",
        "_product_work": original["_product_work"], "_source": "internal:listing",
        "_submission_started": sent,
    }
    original["_completion"] = {"quota_failover": True, "replacement_id": replacement_id,
                               "max_extra_requests": 1, "selected_id": None}
    if sent:
        result.update(provider_account_identity=account, provider_binding_id="new-binding",
                      client_conversation_id="new-client", conversation_id="new-conversation")
    return result


def test_strict_terminal_only_derives_one_unbound_portable_generate_child():
    original = root()
    assert eligible_original(original)
    payload = prepared_payload({"prompt": "persisted prompt", "provider_binding_id": "forged"}, original, "completion-child")
    assert payload["prompt"] == "persisted prompt"
    assert not {"provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id"} & payload.keys()
    assert payload["_image_quota_failover"]["excluded_account_identity"] == "old-account"
    original["upstream_unfinished"] = True
    assert not eligible_original(original)


def test_only_marked_child_can_cross_account_and_never_reuse_old_account():
    original = root()
    candidate = child(original)
    assert registration_successor(original, candidate)
    assert excluded_account(candidate) == "old-account"
    assert not is_successor(original, candidate, after_send=True)
    candidate = child(original, sent=True)
    assert is_successor(original, candidate, after_send=True)
    candidate["provider_account_identity"] = "old-account"
    assert not is_successor(original, candidate, after_send=True)


def test_unknown_or_ordinary_no_image_never_gets_a_quota_successor_marker():
    original = root()
    original["_authoritative_failure_reason"] = None
    assert not eligible_original(original)
    original = root()
    original.update(status="unknown", upstream_outcome="unknown")
    assert not eligible_original(original)


def test_image_receipt_id_is_accepted_as_the_durable_request_identity():
    original = root()
    original["id"] = original.pop("request_id")
    candidate = child(original, sent=True)
    candidate["id"] = candidate.pop("request_id")
    assert is_successor(original, candidate, after_send=True)


@pytest.mark.parametrize("review_retry", [False, True])
@pytest.mark.parametrize("upstream_model", ["", "gpt-5-5-extended"])
def test_legacy_admin_edit_quota_api_reuses_saved_bytes_on_new_account_then_closes_both_sessions(tmp_path, monkeypatch, review_retry, upstream_model):
    """API -> frozen edit bytes -> alternate-account admission -> close/archive."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import api.generation_completion as api
    from services.image_task_service import ImageTaskService
    from services.product_work import ProductWorkService
    from test.test_unknown_turn_recovery import migration

    text, admission, _backend, _legacy = migration(tmp_path)
    accounts_path = tmp_path / "accounts.json"
    accounts = json.loads(accounts_path.read_text())
    for index, account in enumerate(accounts):
        account.update(limits_progress=[{"feature_name": "image_gen", "remaining": 9}],
                       capacity_observed_at=datetime.now(timezone.utc).isoformat())
        if index == 1:
            account["limits_progress"][0]["remaining"] = 0
    accounts_path.write_text(json.dumps(accounts))
    images = ImageTaskService(tmp_path / "images.json", admission=admission, store=text.store)
    service = GenerationCompletionService(text, images, WorkLifecycleService(text, images, clock=admission.clock),
                                          clock=admission.clock)
    admin = {"id": "admin", "role": "admin", "external_image_client": True,
             "_trusted_source": "internal:listing"}
    source = (b"\x89PNG\r\n\x1a\noriginal-main-image", "main.png", "image/png")
    mask = (b"\x89PNG\r\n\x1a\noriginal-mask", "mask.png", "image/png")
    images.submit_edit(admin, client_task_id="legacy-2344", prompt="keep this main product image",
                       model="gpt-image-2", size=None, images=[source], masks=[mask], upstream_model=upstream_model,
                       product_work={"product_ref": "P-2344", "step_ref": "main-image"})
    with text.store.transaction() as db:
        root = text.store.read_receipt(db, "image", "admin", "legacy-2344")
        root.update(retain_receipt=False, status="error", error_code="NO_IMAGE_GENERATED",
                    upstream_outcome="failed", upstream_unfinished=False, _submission_started=True,
                    _authoritative_failure_reason="image_generation_quota_exhausted",
                    provider_binding_id="binding-0", provider_account_identity="account-0",
                    client_conversation_id="old-client", conversation_id="old-conversation",
                    request_message_id="old-request")
        parent = text.store.runtime(db, root["_product_work_key"])
        parent.update(state="active", slot_held=True)
        text.store.set_runtime(db, parent["key"], parent)
        text.store.write_receipt(db, "image", "admin", "legacy-2344", root)
    monkeypatch.setattr(api, "get_generation_completion_service", lambda: service)
    from api.support import require_identity
    from types import SimpleNamespace
    from api.company_requests import company_request_boundary, PREFIX
    from api.external_images import external_image_boundary
    monkeypatch.setattr("api.support.config", SimpleNamespace(auth_key="test-only-quota-admin"))
    app = FastAPI(); app.include_router(api.create_router("image")); app.include_router(api.create_router("text"))
    app.middleware("http")(external_image_boundary)
    app.middleware("http")(company_request_boundary)
    client = TestClient(app, raise_server_exceptions=False, headers={"Authorization": "Bearer test-only-quota-admin"})
    company_headers = {"X-Workbench-Company-Org": "org", "X-Workbench-Company-User": "user",
                       "X-Workbench-Expected-User": "user",
                       "X-Workbench-Company-Connector": "1f084f01-d4b2-4080-8bce-b926f31cc454"}
    path = "/api/image-tasks/legacy-2344/completion"
    for url, headers in ((PREFIX + path, company_headers), (path, {"X-Workbench-Image-Client": "1"})):
        assert client.post(url, json={"action": "quota_failover"}, headers=headers).status_code == 403
        assert client.get(url + "?action=quota_failover", headers=headers).status_code == 403
    no_child = client.get(path + "?action=quota_failover")
    assert no_child.status_code == 409 and no_child.json()["detail"]["code"] == "COMPLETION_QUOTA_FAILOVER_NOT_AUTHORIZED"
    real_submit = images._submit
    rejected_submit = []
    monkeypatch.setattr(images, "_submit", lambda *_args, **_kwargs: rejected_submit.append(True))
    with text.store.connect() as db:
        baseline = copy.deepcopy(text.store.read_receipt(db, "image", "admin", "legacy-2344"))
    def rejected(**changes):
        with text.store.transaction() as db:
            text.store.write_receipt(db, "image", "admin", "legacy-2344", {**baseline, **changes})
        return client.post("/api/image-tasks/legacy-2344/completion", json={"action": "quota_failover"})
    assert rejected(status="unknown", upstream_outcome="unknown").status_code == 409
    assert rejected(_authoritative_failure_reason=None).status_code == 409
    assert rejected(_source="public").status_code == 403
    assert rejected(_completion={"quota_failover": True, "replacement_id": "forged", "max_extra_requests": 1}).status_code == 409
    monkeypatch.setattr(api, "require_identity", lambda *_args, **_kwargs: {**admin, "id": "foreign"})
    assert client.post("/api/image-tasks/legacy-2344/completion", json={"action": "quota_failover"}).status_code == 404
    monkeypatch.setattr(api, "require_identity", require_identity)
    with text.store.transaction() as db:
        text.store.write_receipt(db, "image", "admin", "legacy-2344", baseline)
    with text.store.connect() as db:
        multi_input = text.store.load_input(baseline["_input_ref"])
    multi_input["payload"]["n"] = 2
    multi_ref = text.store.save_input(multi_input)
    multiple = rejected(_input_ref=multi_ref)
    assert multiple.status_code == 409
    assert multiple.json()["detail"]["code"] == "COMPLETION_SINGLE_OUTPUT_REQUIRED"
    with text.store.transaction() as db:
        text.store.write_receipt(db, "image", "admin", "legacy-2344", baseline)
    assert rejected_submit == []
    monkeypatch.setattr(images, "_submit", real_submit)
    # A process loss after the durable prepared input, before child acceptance,
    # leaves exactly one reserved ID; the repeat must submit that same input.
    monkeypatch.setattr(images, "_submit", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")))
    interrupted = client.post("/api/image-tasks/legacy-2344/completion", json={"action": "quota_failover"})
    assert interrupted.status_code == 500
    with text.store.connect() as db:
        reserved = text.store.read_receipt(db, "image", "admin", "legacy-2344")["_completion"]["replacement_id"]
    monkeypatch.setattr(images, "_submit", real_submit)
    response = client.post("/api/image-tasks/legacy-2344/completion", json={"action": "quota_failover"})
    assert response.status_code == 200, response.text
    attempt_id = response.json()["attempt_id"]
    assert attempt_id == reserved
    assert response.json()["created"] is True and response.json()["status"] == "queued"
    assert client.post("/api/chat-requests/legacy-2344/completion", json={"action": "quota_failover"}).status_code == 400
    with text.store.connect() as db:
        child = text.store.read_receipt(db, "image", "admin", attempt_id)
        frozen = text.store.load_input(child["_input_ref"])
    assert frozen["mode"] == "edit" and frozen["payload"]["images"] == [source]
    assert frozen["payload"]["mask"] == [mask] and frozen["payload"]["n"] == 1
    assert frozen["payload"].get("upstream_model") == upstream_model
    assert not {"provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id"} & frozen["payload"].keys()
    with text.store.connect() as db:
        queued_before = (copy.deepcopy(text.store.read_receipt(db, "image", "admin", "legacy-2344")),
                         copy.deepcopy(text.store.read_receipt(db, "image", "admin", attempt_id)))
    queued_read = client.get("/api/image-tasks/legacy-2344/completion?action=quota_failover")
    assert queued_read.status_code == 200 and queued_read.json()["attempt_id"] == attempt_id
    assert {k: v for k, v in response.json().items() if k not in {"created", "existing"}} == {k: v for k, v in queued_read.json().items() if k not in {"created", "existing"}}
    assert queued_read.json()["product_work"] == {"protocol": "product-work-v1", "product_ref": "P-2344", "step_ref": "main-image"}
    assert queued_read.json()["attempt_authority"] == {key: None for key in (
        "provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id", "parent_message_id")}
    with text.store.connect() as db:
        assert queued_before == (text.store.read_receipt(db, "image", "admin", "legacy-2344"),
                                 text.store.read_receipt(db, "image", "admin", attempt_id))
    def image_handler(payload):
        from services.request_context import current_request
        context = current_request.get()
        context.before_send()
        assert payload["provider_account_identity"] == "account-1"
        assert payload["prompt"] == "keep this main product image"
        assert payload["upstream_model"] == upstream_model
        assert payload["images"] == [source] and payload["mask"] == [mask]
        assert not payload.get("conversation_id")
        assert not payload.get("parent_message_id")
        context.release_turn()  # The paced upstream transport releases its request turn.
        return {"data": [{"b64_json": "c2F2ZWQ="}],
                "_provider_binding_id": payload["provider_binding_id"],
                "_provider_account_identity": payload["provider_account_identity"],
                "_conversation_id": "new-conversation", "_parent_message_id": "new-parent"}
    images.edit_handler = image_handler
    def finish(context, body):
        images._run_task("admin:" + context.request_id, body["mode"], body["payload"], body["identity"], "gpt-image-2")
    admission.register("image", finish)
    # No eligible second account initially: child remains the one queued row.
    assert admission.claim_next() is None
    accounts = json.loads(accounts_path.read_text())
    accounts[1]["limits_progress"][0]["remaining"] = 9
    accounts[1]["capacity_observed_at"] = datetime.now(timezone.utc).isoformat()
    accounts_path.write_text(json.dumps(accounts))
    context = admission.claim_next()
    assert context and context.request_id == attempt_id
    bound_read = client.get(path + "?action=quota_failover")
    assert bound_read.status_code == 200
    assert bound_read.json()["attempt_authority"]["provider_account_identity"] == "account-1"
    admission.execute(context)
    selected = service.read("image", admin, "legacy-2344")
    assert selected["selected_id"] == attempt_id
    repeated = client.post("/api/image-tasks/legacy-2344/completion", json={"action": "quota_failover"})
    assert repeated.status_code == 200 and repeated.json()["attempt_id"] == attempt_id
    assert repeated.json()["existing"] is True and repeated.json()["created"] is False
    assert repeated.json()["status"] == "succeeded"
    archive_tail = (_complete_unsent_review_retry(client, text, admission, attempt_id, monkeypatch)
                    if review_retry else attempt_id)
    product = ProductWorkService(text, images, clock=admission.clock)
    closed = product.close(admin, "P-2344")
    assert closed["archive"]["total"] == 2 and closed["state"] == "closing"
    closing_repeat = client.post("/api/image-tasks/legacy-2344/completion", json={"action": "quota_failover"})
    assert closing_repeat.status_code == 200 and closing_repeat.json()["attempt_id"] == attempt_id
    archived = []
    def archive(owner, request_id, value):
        with text.store.connect() as db:
            target = (text.store.read_receipt(db, "image", owner, request_id)
                      or text.store.read_receipt(db, "text", owner, request_id))
        archived.append((request_id, value))
        return {"request_id": request_id, "conversation_id": target["conversation_id"], "archived": value}
    images.archive_product_work = archive
    text.archive_product_work = archive
    for _ in range(8):
        service.lifecycle.process_one()
        if product.get(admin, "P-2344")["state"] == "completed":
            break
    assert {request_id for request_id, value in archived if value} == {"legacy-2344", archive_tail}
    completed = product.get(admin, "P-2344")
    assert completed["state"] == "completed" and completed["archive"] == {"total": 2, "confirmed": 2, "pending": 0}
    final_repeat = client.post("/api/image-tasks/legacy-2344/completion", json={"action": "quota_failover"})
    assert final_repeat.status_code == 200 and final_repeat.json()["attempt_id"] == attempt_id
    closed_read = client.get("/api/image-tasks/legacy-2344/completion?action=quota_failover")
    assert closed_read.status_code == 200 and closed_read.json()["attempt_authority"] == {
        "provider_binding_id": "binding-1", "provider_account_identity": "account-1",
        "client_conversation_id": closed_read.json()["attempt_authority"]["client_conversation_id"],
        "conversation_id": "new-conversation", "parent_message_id": "new-parent",
    }
    with text.store.connect() as db:
        original = text.store.read_receipt(db, "image", "admin", "legacy-2344")
    assert original["status"] == "error" and original["error_code"] == "NO_IMAGE_GENERATED"


def _complete_unsent_review_retry(client, text, admission, image_id, monkeypatch):
    from services.conversation_binding_service import ConversationBindingError, conversation_binding_service
    from services.request_context import current_request
    with text.store.connect() as db:
        image = text.store.read_receipt(db, "image", "admin", image_id)
    metadata = {"product_ref": "P-2344", "step_ref": "direct-listing.content.main-image-review:review-Q"}
    authority = {k: image[k] for k in ("provider_binding_id", "provider_account_identity",
                                      "client_conversation_id", "conversation_id", "parent_message_id")}
    body = {"client_request_id": "review-Q", "model": "fixture-text", **authority,
            "messages": [{"role": "user", "content": "review the saved image"}], "product_work": metadata}
    def fail_before_send(_body, on_cursor):
        raise ConversationBindingError("fixture stale parent", code="CONVERSATION_BINDING_MISMATCH")
    text.runner = fail_before_send
    admission.register("text", lambda ctx, value: text._run(ctx.owner, ctx.request_id, value))
    text.submit("admin", body, source="internal:listing")
    ctx = admission.claim_next()
    assert ctx and ctx.request_id == "review-Q"
    admission.execute(ctx)
    admission.clock.now += 1000  # Retire the historical lease, without clearing the failed original.
    with text.store.connect() as db:
        original = text.store.read_receipt(db, "text", "admin", "review-Q")
    assert original["upstream_outcome"] == "not_sent" and original["_submission_started"] is False
    monkeypatch.setattr(conversation_binding_service, "bound_unsent_retry_cursor", lambda _r: authority)
    path = "/api/chat-requests/review-Q/completion"
    missing = client.get(path + "?action=retry")
    assert missing.status_code == 404 and missing.json()["detail"]["code"] == "TASK_ATTEMPT_NOT_FOUND"
    response = client.post(path, json={"action": "retry"})
    assert response.status_code == 200, response.text
    retry_id = response.json()["attempt_id"]
    assert client.post(path, json={"action": "retry"}).json()["attempt_id"] == retry_id
    def finish_review(value, on_cursor):
        assert value["product_work"] == metadata and value["messages"] == body["messages"]
        current_request.get().before_send()
        return {"content": "review saved", "parent_message_id": "review-final"}
    text.runner = finish_review
    ctx = admission.claim_next()
    assert ctx and ctx.request_id == retry_id
    admission.execute(ctx)
    admission.update_claim(ctx, _claim_id=None, _claim_until=0, _turn_reserved=False)
    with text.store.connect() as db:
        assert text.store.read_receipt(db, "text", "admin", "review-Q") == original
        saved = text.store.read_receipt(db, "text", "admin", retry_id)
        assert saved["status"] == "succeeded" and saved["content"] == "review saved"
    result = client.get(path + "?action=retry")
    assert result.json()["attempt_id"] == retry_id and result.json()["result_ready"] is True
    return retry_id

