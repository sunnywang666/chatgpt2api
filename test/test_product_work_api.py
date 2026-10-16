"""API/receipt integration with no upstream calls or controller threads."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import ai, image_tasks, product_works
from api.image_inputs import _payload_from_fields
from services.text_task_service import TextTaskService
from services.image_task_service import ImageTaskService, _request_hash
from services.product_work import ProductWorkService
from services.work_lifecycle import WorkLifecycleError


IDENTITY = {"id": "owner", "role": "admin", "_trusted_source": "internal:wb-to-ozon"}
META = {"product_ref": "org/batch/product/rev1", "step_ref": "title"}


@pytest.fixture
def services(tmp_path):
    admission = SimpleNamespace(wake=Mock())
    text = TextTaskService(tmp_path / "tasks.sqlite3", admission=admission, runner=Mock())
    images = ImageTaskService(tmp_path / "images.json", store=text.store, admission=admission,
                              generation_handler=Mock(), edit_handler=Mock())
    return text, images, ProductWorkService(text, images)


def text_body(rid="text-1", metadata=META):
    return {"client_request_id": rid, "client_conversation_id": "original-client",
            "messages": [{"role": "user", "content": "original task"}],
            **({"product_work": metadata} if metadata is not None else {})}


def image_submit(images, rid="image-1", metadata=None, identity=IDENTITY):
    return images.submit_generation(identity, client_task_id=rid, prompt="original image", model="gpt-image-2",
                                    size=None, product_work=metadata)


def test_mixed_acceptance_shares_parent_without_generic_work_or_upstream(services):
    text, images, product = services
    first = text.submit("owner", text_body(), source=IDENTITY["_trusted_source"])
    second = image_submit(images, metadata={**META, "step_ref": "main-image"})
    state = product.get(IDENTITY, META["product_ref"])
    assert state["state"] == "queued" and not state["slot_held"]
    assert {(m["kind"], m["request_id"]) for m in state["members"]} == {("text", "text-1"), ("image", "image-1")}
    assert first["product_work"] == {"protocol": "product-work-v1", **META}
    assert second["product_work"]["product_ref"] == META["product_ref"]
    with text.store.connect() as db:
        rows = list(text.store.receipts(db))
        assert len({r["_product_work_key"] for _, _, _, r in rows}) == 1
        assert not any(r.get("_work_key") for _, _, _, r in rows)
        image = text.store.read_receipt(db, "image", "owner", "image-1")
        assert text.store.load_input(image["_input_ref"])["payload"]["retain_conversation"] is True
    text.runner.assert_not_called(); images.generation_handler.assert_not_called()


def test_original_ids_remain_idempotent_and_changed_product_conflicts(services):
    text, images, product = services
    source = IDENTITY["_trusted_source"]
    text.submit("owner", text_body(), source=source)
    text.submit("owner", text_body(), source=source)
    image_submit(images, metadata={**META, "step_ref": "image"})
    image_submit(images, metadata={**META, "step_ref": "image"})
    assert len(product.get(IDENTITY, META["product_ref"])["members"]) == 2
    from services.conversation_binding_service import ConversationBindingError
    with pytest.raises(ConversationBindingError, match="different input"):
        text.submit("owner", text_body(metadata={**META, "product_ref": "different"}), source=source)
    with pytest.raises(ValueError, match="different immutable"):
        image_submit(images, metadata={**META, "product_ref": "different", "step_ref": "image"})
    with pytest.raises(WorkLifecycleError) as exc:
        text.validate_submission("owner", text_body(), source="internal:content")
    assert exc.value.code == "PRODUCT_WORK_SOURCE_CONFLICT"
    with pytest.raises(WorkLifecycleError) as exc:
        image_submit(images, metadata={**META, "step_ref": "image"},
                     identity={**IDENTITY, "_trusted_source": "internal:content"})
    assert exc.value.code == "PRODUCT_WORK_SOURCE_CONFLICT"


def test_optional_contract_preserves_legacy_payload_and_rejects_null_extra_fields():
    model = ai.ConversationBindingTextRequest(client_conversation_id="c", messages=[])
    assert "product_work" not in model.model_dump()
    for bad in (None, {}, {**META, "priority": 100}, {**META, "step_ref": 1}):
        with pytest.raises(ValidationError):
            ai.ConversationBindingTextRequest(client_conversation_id="c", messages=[], product_work=bad)
        with pytest.raises(ValidationError):
            image_tasks.ImageGenerationTaskRequest(client_task_id="i", prompt="p", product_work=bad)
    payload = {"prompt": "p", "model": "gpt-image-2"}
    assert _request_hash("generate", payload) != _request_hash("generate", {**payload, "_product_work": META})
    for encoded in (META, json.dumps(META)):
        parsed = _payload_from_fields({"prompt": "p", "product_work": encoded})
        assert parsed["product_work"] == META
    assert "product_work" not in _payload_from_fields({"prompt": "p"})


def test_explicit_product_never_falls_back_without_admission_or_source(services):
    text, images, _ = services
    with pytest.raises(WorkLifecycleError) as exc:
        text.submit("owner", text_body(), source="key:owner")
    assert exc.value.status == 403
    with pytest.raises(WorkLifecycleError) as exc:
        image_submit(images, metadata=META, identity={**IDENTITY, "role": "user"})
    assert exc.value.status == 403
    text.admission = images.admission = None
    for call in (lambda: text.submit("owner", text_body(), source=IDENTITY["_trusted_source"]),
                 lambda: image_submit(images, metadata=META)):
        with pytest.raises(WorkLifecycleError) as exc:
            call()
        assert exc.value.code == "PRODUCT_WORK_UNAVAILABLE"
    with text.store.connect() as db:
        assert not list(text.store.receipts(db))


def test_parent_api_strict_auth_and_durable_close(services, monkeypatch):
    text, _, service = services
    text.submit("owner", text_body(), source=IDENTITY["_trusted_source"])
    monkeypatch.setattr(product_works, "require_identity", lambda *a, **k: IDENTITY)
    monkeypatch.setattr(product_works, "get_product_work_service", lambda: service)
    app = FastAPI(); app.include_router(product_works.create_router())
    client = TestClient(app)
    headers = {"x-workbench-consumer": "wb-to-ozon"}
    assert client.get("/api/product-works/capabilities").status_code == 403
    caps = client.get("/api/product-works/capabilities", headers=headers)
    assert caps.json()["shared_capacity_scope"] == "provider"
    assert caps.json()["capacity_source"] == "eligible_chat_turn_total"
    assert caps.json()["admission_scope"] == "account_execution"
    assert caps.json()["product_count_limit"] is None
    assert caps.json()["retained_products_consume_capacity"] is False
    assert caps.json()["resume"] is True
    assert caps.json()["cancel_unsent_text_step"] is True
    assert caps.headers["cache-control"] == "private, no-store"
    status = client.get("/api/product-works", params={"product_ref": META["product_ref"]}, headers=headers)
    assert status.json()["members"][0]["request_id"] == "text-1"
    assert status.json()["version"] == 1
    for invalid in ({"results_saved": 1}, {"results_saved": False}, {"results_saved": True, "priority": 3}):
        assert client.post("/api/product-works/close", headers=headers,
            json={"product_ref": META["product_ref"], **invalid}).status_code == 422
    assert client.post("/api/product-works/close", headers=headers,
        json={"product_ref": META["product_ref"], "results_saved": True}).status_code == 409
    cancel = client.post("/api/product-works/close", headers=headers,
        json={"product_ref": META["product_ref"], "results_saved": True, "outcome": "cancelled"})
    assert cancel.status_code == 200, cancel.text
    assert cancel.json()["state"] == "cancelled"
    assert text.read("owner", "text-1")["status"] == "cancelled"


def test_cancel_step_route_rejects_unauthorized_or_expanded_scope_without_mutation(services, monkeypatch):
    text, images, service = services
    text.submit("owner", text_body(), source=IDENTITY["_trusted_source"])
    identity = [IDENTITY]
    monkeypatch.setattr(product_works, "require_identity", lambda *a, **k: identity[0])
    monkeypatch.setattr(product_works, "get_product_work_service", lambda: service)
    app = FastAPI(); app.include_router(product_works.create_router())
    client = TestClient(app)
    headers = {"x-workbench-consumer": "wb-to-ozon"}
    payload = {"product_ref": META["product_ref"], "request_id": "text-1", "expected_version": 1}

    def saved_rows():
        with text.store.connect() as db:
            return (list(db.execute("SELECT * FROM requests ORDER BY owner,id")),
                    list(db.execute("SELECT * FROM task_runtime ORDER BY name")))

    before = saved_rows()
    assert client.post("/api/product-works/cancel-step", json=payload).status_code == 403
    identity[0] = {**IDENTITY, "role": "user"}
    assert client.post("/api/product-works/cancel-step", headers=headers, json=payload).status_code == 403
    identity[0] = {**IDENTITY, "id": "other-owner"}
    assert client.post("/api/product-works/cancel-step", headers=headers, json=payload).status_code == 404
    identity[0] = IDENTITY
    for patch in ({"expected_version": True}, {"expected_version": "1"}, {"expected_version": 0},
                  {"request_id": " text-1"}, {"request_id": "text-1\n"}, {"product_ref": ""},
                  {"kind": "image"}, {"messages": []}, {"outcome": "cancelled"},
                  {"provider_account_identity": "other"}, {"supersedes_request_id": "old"}):
        assert client.post("/api/product-works/cancel-step", headers=headers,
                           json={**payload, **patch}).status_code == 422
    for omitted in payload:
        assert client.post("/api/product-works/cancel-step", headers=headers,
                           json={k: v for k, v in payload.items() if k != omitted}).status_code == 422
    stale = client.post("/api/product-works/cancel-step", headers=headers,
                        json={**payload, "expected_version": 2})
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "PRODUCT_WORK_VERSION_CONFLICT"
    assert saved_rows() == before
    text.runner.assert_not_called(); images.generation_handler.assert_not_called()


def test_cancel_step_route_returns_same_verified_proof_as_original_read(services, monkeypatch):
    text, images, service = services
    text.submit("owner", text_body(), source=IDENTITY["_trusted_source"])
    text.submit("owner", text_body("text-2", {**META, "step_ref": "another-step"}),
                source=IDENTITY["_trusted_source"])
    monkeypatch.setattr(product_works, "require_identity", lambda *a, **k: IDENTITY)
    monkeypatch.setattr(product_works, "get_product_work_service", lambda: service)
    app = FastAPI(); app.include_router(product_works.create_router())
    client = TestClient(app)
    headers = {"x-workbench-consumer": "wb-to-ozon"}
    payload = {"product_ref": META["product_ref"], "request_id": "text-1", "expected_version": 1}
    response = client.post("/api/product-works/cancel-step", headers=headers, json=payload)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    state = response.json()
    proof = state["cancelled_unsent"]
    assert set(proof) == {"request_id", "product_ref", "parent_version", "request_hash", "cancelled_at",
                          "code", "upstream_outcome"}
    assert proof["request_id"] == "text-1" and proof["product_ref"] == META["product_ref"]
    assert proof["code"] == "PRODUCT_WORK_STEP_CANCELLED" and proof["upstream_outcome"] == "not_sent"
    assert state["state"] == "queued" and not state["results_saved"]
    assert {m["request_id"]: m["status"] for m in state["members"]} == {
        "text-1": "cancelled", "text-2": "queued"}
    assert text.read("owner", "text-1")["cancelled_unsent"] == proof
    assert client.post("/api/product-works/cancel-step", headers=headers, json=payload).json() == state
    text.runner.assert_not_called(); images.generation_handler.assert_not_called()


def test_resume_route_requires_source_and_version_and_fences_close(services, monkeypatch):
    text, images, service = services
    text.submit("owner", text_body(), source=IDENTITY["_trusted_source"])
    # API transport checks use a completed no-conversation parent; actual
    # archive proof/cursors and concurrency are exercised by service tests.
    monkeypatch.setattr(product_works, "require_identity", lambda *a, **k: IDENTITY)
    monkeypatch.setattr(product_works, "get_product_work_service", lambda: service)
    app = FastAPI(); app.include_router(product_works.create_router())
    client = TestClient(app)
    headers = {"x-workbench-consumer": "wb-to-ozon"}
    payload = {"product_ref": META["product_ref"], "expected_version": 1}
    assert client.post("/api/product-works/resume", json=payload).status_code == 403
    for invalid in ({}, {"expected_version": True}, {"expected_version": "1"},
                    {"expected_version": 0}, {"expected_version": 1, "priority": 1}):
        assert client.post("/api/product-works/resume", headers=headers,
                           json={"product_ref": META["product_ref"], **invalid}).status_code == 422
    response = client.post("/api/product-works/resume", headers=headers, json=payload)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "PRODUCT_WORK_RESUME_NOT_COMPLETED"
    with text.store.transaction() as db:
        receipt = text.store.read_receipt(db, "text", "owner", "text-1")
        receipt.update(status="succeeded", content="saved original", upstream_outcome="completed")
        text.store.write_receipt(db, "text", "owner", "text-1", receipt)
        parent = text.store.runtime(db, receipt["_product_work_key"])
        parent.update(state="completed", outcome="completed", results_saved=True, slot_held=False)
        text.store.set_runtime(db, parent["key"], parent)
    resumed = client.post("/api/product-works/resume", headers=headers, json=payload)
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["version"] == 2
    assert resumed.headers["cache-control"] == "private, no-store"
    assert client.post("/api/product-works/resume", headers=headers, json=payload).json() == resumed.json()
    for expected, code in ((None, "PRODUCT_WORK_VERSION_REQUIRED"), (1, "PRODUCT_WORK_VERSION_CONFLICT")):
        closed = client.post("/api/product-works/close", headers=headers, json={
            "product_ref": META["product_ref"], "results_saved": True,
            **({"expected_version": expected} if expected is not None else {})})
        assert closed.status_code == 409
        assert closed.json()["detail"]["code"] == code
    text.runner.assert_not_called(); images.generation_handler.assert_not_called()


def test_adoption_keeps_successful_original_hash_input_and_generation_count(services):
    text, _, service = services
    source = IDENTITY["_trusted_source"]
    text.submit("owner", text_body(metadata=None), source=source)
    text._update("owner", "text-1", status="succeeded", content="saved original", send_count=1,
                 _submission_started=True, upstream_outcome="completed")
    with text.store.connect() as db:
        before = db.execute("SELECT request_hash,receipt FROM requests WHERE id='text-1'").fetchone()
    adopted = service.adopt(IDENTITY, META["product_ref"], [{"kind": "text", "request_id": "text-1", "step_ref": "title"}])
    assert adopted["slot_held"]
    with text.store.connect() as db:
        after = db.execute("SELECT request_hash,receipt FROM requests WHERE id='text-1'").fetchone()
    assert before[0] == after[0]
    original, changed = json.loads(before[1]), json.loads(after[1])
    for key in ("_product_work_key", "_product_work"):
        changed.pop(key)
    assert original == changed
    assert text.submit("owner", text_body(metadata=None), source=source)["content"] == "saved original"
    text.runner.assert_not_called()


def test_actual_submission_routes_share_parent_and_project_metadata(services, monkeypatch):
    text, images, product = services
    monkeypatch.setattr(ai, "require_identity", lambda *a, **k: IDENTITY)
    monkeypatch.setattr(image_tasks, "require_identity", lambda *a, **k: IDENTITY)
    monkeypatch.setattr(ai, "text_task_service", text)
    monkeypatch.setattr(image_tasks, "image_task_service", images)
    async def no_review(*args):
        return None
    monkeypatch.setattr(ai, "filter_or_log", no_review)
    monkeypatch.setattr(image_tasks, "filter_or_log", no_review)
    app = FastAPI(); app.include_router(ai.create_router()); app.include_router(image_tasks.create_router())
    client = TestClient(app)
    headers = {"x-workbench-consumer": "wb-to-ozon"}
    response = client.post("/api/conversation-bindings/text", headers=headers, json=text_body())
    assert response.status_code == 200, response.text
    image = client.post("/api/image-tasks/generations", headers=headers,
        json={"client_task_id": "image-route", "prompt": "main image", "product_work": {**META, "step_ref": "image"}})
    assert image.status_code == 200, image.text
    assert image.json()["product_work"]["product_ref"] == META["product_ref"]
    assert len(product.get(IDENTITY, META["product_ref"])["members"]) == 2
    wrong_source = client.post("/api/conversation-bindings/text", headers={"x-workbench-consumer": "content"}, json=text_body())
    assert wrong_source.status_code == 403, wrong_source.text
    text.runner.assert_not_called(); images.generation_handler.assert_not_called()


def test_product_original_receipts_outlive_legacy_image_retention(services):
    text, images, product = services
    image_submit(images, metadata=META)
    with text.store.transaction() as db:
        receipt = text.store.read_receipt(db, "image", "owner", "image-1")
        receipt.update(status="success", data=[{"url": "saved.png"}], updated_at="2001-01-01 00:00:00")
        text.store.write_receipt(db, "image", "owner", "image-1", receipt)
    assert images.list_tasks(IDENTITY, ["image-1"])["missing_ids"] == []
    assert product.get(IDENTITY, META["product_ref"])["members"][0]["status"] == "success"


def test_raw_archive_cannot_bypass_product_close_or_repeat_completed_archive(services, monkeypatch):
    text, _, _ = services
    text.submit("owner", text_body(), source=IDENTITY["_trusted_source"])
    bound = {"provider_binding_id": "binding", "provider_account_identity": "account",
             "client_conversation_id": "original-client", "conversation_id": "original-chat",
             "parent_message_id": "original-answer"}
    text._update("owner", "text-1", **bound)
    monkeypatch.setattr(ai, "require_identity", lambda *a, **k: IDENTITY)
    monkeypatch.setattr(ai, "text_task_service", text)
    archive = Mock()
    monkeypatch.setattr(ai.conversation_binding_service, "archive", archive)
    app = FastAPI(); app.include_router(ai.create_router())
    client = TestClient(app)
    for state in ("active", "closing", "completed"):
        with text.store.transaction() as db:
            receipt = text.store.read_receipt(db, "text", "owner", "text-1")
            parent = text.store.runtime(db, receipt["_product_work_key"])
            parent["state"] = state
            text.store.set_runtime(db, parent["key"], parent)
        response = client.post("/api/conversation-bindings/archive", json=bound)
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "PRODUCT_WORK_REQUIRED"
    archive.assert_not_called()


def test_raw_archive_rechecks_product_adopted_while_waiting(services, monkeypatch):
    from services.request_context import current_archive_guard
    text, _, _ = services
    text.submit("owner", text_body(), source=IDENTITY["_trusted_source"])
    bound = {"provider_binding_id": "binding", "provider_account_identity": "account",
             "client_conversation_id": "original-client", "conversation_id": "original-chat",
             "parent_message_id": "original-answer"}
    monkeypatch.setattr(ai, "require_identity", lambda *a, **k: IDENTITY)
    monkeypatch.setattr(ai, "text_task_service", text)
    sent = []

    def waiting_archive(payload):
        text._update("owner", "text-1", **bound)
        current_archive_guard.get()()
        sent.append(payload)
        return {"archived": True}

    monkeypatch.setattr(ai.conversation_binding_service, "archive", waiting_archive)
    app = FastAPI(); app.include_router(ai.create_router())
    response = TestClient(app).post("/api/conversation-bindings/archive", json=bound)
    assert response.status_code == 409 and not sent


def test_unowned_raw_archive_keeps_legacy_behavior(services, monkeypatch):
    from services.request_context import current_archive_guard
    text, _, _ = services
    monkeypatch.setattr(ai, "require_identity", lambda *a, **k: IDENTITY)
    monkeypatch.setattr(ai, "text_task_service", text)
    bound = {"provider_binding_id": "binding", "provider_account_identity": "account",
             "client_conversation_id": "original-client", "conversation_id": "original-chat",
             "parent_message_id": "original-answer"}

    def archive(payload):
        current_archive_guard.get()()
        assert payload == bound
        return {**payload, "archived": True}

    monkeypatch.setattr(ai.conversation_binding_service, "archive", archive)
    app = FastAPI(); app.include_router(ai.create_router())
    response = TestClient(app).post("/api/conversation-bindings/archive", json=bound)
    assert response.status_code == 200 and response.json()["archived"] is True


@pytest.mark.parametrize("kind", ["text", "image"])
def test_first_bound_product_waits_for_raw_archive_even_with_another_client_alias(services, monkeypatch, kind):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock
    from services.account_service import account_service
    from services.request_context import current_archive_guard
    text, images, product = services
    entered, release, attachment_waiting = Event(), Event(), Event()
    physical_lock = Lock()
    attempts = []

    class ObservedLock:
        def __enter__(self):
            attempts.append(1)
            if len(attempts) > 1:
                attachment_waiting.set()
            physical_lock.acquire()

        def __exit__(self, *args):
            physical_lock.release()

    def canonical_lock(binding, conversation):
        assert (binding, conversation) == ("product-archive:account", "original-chat")
        return ObservedLock()

    monkeypatch.setattr(account_service, "conversation_binding_lock", canonical_lock)
    monkeypatch.setattr(ai, "require_identity", lambda *a, **k: IDENTITY)
    monkeypatch.setattr(ai, "text_task_service", text)
    bound = {"provider_binding_id": "binding", "provider_account_identity": "account",
             "client_conversation_id": "archive-alias", "conversation_id": "original-chat",
             "parent_message_id": "original-answer"}

    def archive(payload):
        current_archive_guard.get()()
        entered.set()
        assert release.wait(3)
        current_archive_guard.get()()
        return {**payload, "archived": True}

    def attach():
        original = {**bound, "client_conversation_id": "actual-client"}
        if kind == "text":
            return text.submit("owner", {**text_body(), **original}, source=IDENTITY["_trusted_source"])
        return images.submit_generation(IDENTITY, client_task_id="image-1", prompt="original",
            model="gpt-image-2", size=None, product_work=META, **original)

    monkeypatch.setattr(ai.conversation_binding_service, "archive", archive)
    app = FastAPI(); app.include_router(ai.create_router())
    client = TestClient(app)
    with ThreadPoolExecutor(max_workers=2) as pool:
        archiving = pool.submit(client.post, "/api/conversation-bindings/archive", json=bound)
        try:
            assert entered.wait(3)
            attaching = pool.submit(attach)
            assert attachment_waiting.wait(3)
            with text.store.connect() as db:
                assert not list(text.store.receipts(db))
        finally:
            release.set()
        assert archiving.result(timeout=3).status_code == 200
        assert attaching.result(timeout=3)["product_work"]["product_ref"] == META["product_ref"]
    assert product.get(IDENTITY, META["product_ref"])["state"] == "queued"
