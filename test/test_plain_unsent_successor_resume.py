"""One-shot resume for a stopped plain no-final successor.

All documents and IDs are synthetic.  The first run reaches the normal
pre-send no-final guard and fails before any upstream send; the explicit
resume must not broaden the derived-category recovery path.
"""
from copy import deepcopy

import pytest

from services.conversation_binding_service import ConversationBindingError, ConversationBindingService
from services.request_context import current_request
from test.test_bound_text_archive import bound_chat
from test.test_explicit_unknown_successor import row, task, update
from test.test_no_final_successor import graph, no_final


def failed_plain_no_final_successor(task, bound_chat):
    h = no_final(task)
    h.service.runner = bound_chat.service.complete_text
    bound_chat.document = graph(h)
    # Make the real binding-layer no-final tail guard stop before the send.
    bound_chat.document["current_node"] = "different-synthetic-head"
    h.service.submit("owner", h.envelope)
    context = h.admission.claim_next()
    assert context and context.request_id == "successor"
    h.admission.execute(context)
    failed = row(h.service, "successor")
    assert failed["status"] == "failed"
    assert failed["error_code"] == "CHAT_SUPERSEDE_CURSOR_CHANGED"
    assert failed["upstream_outcome"] == "not_sent"
    assert failed["_supersedes_no_final"] is True
    assert failed.get("_derived_input") is None
    assert "send" not in bound_chat.calls
    bound_chat.document = graph(h)
    bound_chat.calls.clear()
    return h


def request_hash(service, request_id):
    with service.store.connect() as db:
        return db.execute(
            "SELECT request_hash FROM requests WHERE owner=? AND id=?", ("owner", request_id)
        ).fetchone()[0]


def test_plain_no_final_unsent_successor_resumes_once_without_changing_frozen_identity(task, bound_chat):
    h = failed_plain_no_final_successor(task, bound_chat)
    before = deepcopy(row(h.service, "successor"))
    frozen_input = h.service.store.load_input(before["_input_ref"])
    frozen_hash = request_hash(h.service, "successor")
    original_before = deepcopy(row(h.service, "old"))

    resumed = h.service.resume_unsent_successor("owner", "successor", h.envelope)

    assert resumed["status"] == "queued"
    queued = row(h.service, "successor")
    assert queued["_bound_successor_resume_count"] == 1
    assert queued["_bound_successor_resume_failure"]["error_code"] == "CHAT_SUPERSEDE_CURSOR_CHANGED"
    for key in ("request_id", "request_message_id", "_input_ref", "_supersedes_request_id",
                "_supersedes_input_hash", "_supersedes_no_final", "_sequence"):
        assert queued[key] == before[key]
    assert queued.get("_derived_input") is None
    assert h.service.store.load_input(queued["_input_ref"]) == frozen_input
    assert request_hash(h.service, "successor") == frozen_hash
    assert row(h.service, "old") == original_before
    assert h.service.resume_unsent_successor("owner", "successor", h.envelope) == resumed


@pytest.mark.parametrize("changes", [
    {"_supersedes_no_final": False},
    {"send_count": 1},
    {"_claim_id": "synthetic-live-lease", "_claim_until": 2000},
    {"_recovery_paused": True},
    {"_execution_timeline": [{"stage": "send_guard_passed"}]},
])
def test_plain_no_final_resume_rejects_missing_proof_or_any_send_ambiguity(task, bound_chat, changes):
    h = failed_plain_no_final_successor(task, bound_chat)
    update(h.service, "successor", **changes)
    before = deepcopy(row(h.service, "successor"))

    with pytest.raises(ConversationBindingError) as error:
        h.service.resume_unsent_successor("owner", "successor", h.envelope)

    assert error.value.code == "CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE"
    assert row(h.service, "successor") == before
    assert h.admission.claim_next() is None
    assert "send" not in bound_chat.calls


@pytest.mark.parametrize(("parent_sequence", "child_sequence"), [
    (1, 0), (1, "2"), (1, True), (2, 2), (3, 2),
])
def test_plain_no_final_resume_requires_strict_existing_sequence_lineage(
        task, bound_chat, parent_sequence, child_sequence):
    h = failed_plain_no_final_successor(task, bound_chat)
    update(h.service, "old", _sequence=parent_sequence)
    update(h.service, "successor", _sequence=child_sequence)
    before = deepcopy(row(h.service, "successor"))

    with pytest.raises(ConversationBindingError) as error:
        h.service.resume_unsent_successor("owner", "successor", h.envelope)

    assert error.value.code == "CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE"
    assert row(h.service, "successor") == before
    assert h.admission.claim_next() is None
    assert "send" not in bound_chat.calls


def test_plain_no_final_resume_repeats_fresh_tail_guard_before_send(task, bound_chat):
    h = failed_plain_no_final_successor(task, bound_chat)
    h.service.resume_unsent_successor("owner", "successor", h.envelope)
    # This change happens after the durable row is requeued.  The runner must
    # use the live no-final tail, not rewrite the frozen submitted cursor.
    bound_chat.document["current_node"] = "different-synthetic-head"
    h.admission.execute(h.admission.claim_next())

    failed = row(h.service, "successor")
    assert failed["status"] == "failed"
    assert failed["error_code"] == "CHAT_SUPERSEDE_CURSOR_CHANGED"
    assert failed["upstream_outcome"] == "not_sent"
    assert failed["_submission_parent_message_id"] == "previous-answer"
    assert h.service.store.load_input(failed["_input_ref"])["parent_message_id"] == "previous-answer"
    assert "send" not in bound_chat.calls


def test_plain_no_final_resume_sends_once_from_the_verified_tail_and_preserves_original(task, bound_chat, monkeypatch):
    import services.conversation_binding_service as module

    h = failed_plain_no_final_successor(task, bound_chat)
    before = deepcopy(row(h.service, "successor"))
    original_before = deepcopy(row(h.service, "old"))
    frozen_input = h.service.store.load_input(before["_input_ref"])
    frozen_hash = request_hash(h.service, "successor")
    sends = []
    original_get = module.OpenAIBackendAPI._get_conversation

    def read(backend, conversation_id, **_kwargs):
        return original_get(backend, conversation_id)

    def events(backend, **kwargs):
        backend.text_pre_send_check(None)
        current_request.get().before_send()
        sends.append(deepcopy(kwargs))
        assert kwargs["conversation_id"] == "original-chat"
        assert kwargs["parent_message_id"] == "tool"
        backend.text_cursor_callback({"request_parent_message_id": "tool"})
        yield {"type": "conversation.delta", "conversation_id": "original-chat", "delta": "synthetic answer"}

    monkeypatch.setattr(module.OpenAIBackendAPI, "_get_conversation", read)
    monkeypatch.setattr(module, "conversation_events", events)
    monkeypatch.setattr(ConversationBindingService, "_read_text_request_result", staticmethod(
        lambda _backend, _receipt: {"status": "succeeded", "content": "synthetic answer", "parent_message_id": "new-answer"}
    ))

    h.service.resume_unsent_successor("owner", "successor", h.envelope)
    h.admission.execute(h.admission.claim_next())

    succeeded = row(h.service, "successor")
    assert succeeded["status"] == "succeeded" and succeeded["content"] == "synthetic answer", {
        key: succeeded.get(key) for key in ("status", "error_code", "upstream_outcome", "waiting", "_ready_at", "_submission_started")
    }
    assert succeeded["_bound_successor_resume_count"] == 1
    for key in ("request_id", "request_message_id", "_input_ref", "_supersedes_request_id",
                "_supersedes_input_hash", "_supersedes_no_final", "_sequence"):
        assert succeeded[key] == before[key]
    assert h.service.store.load_input(succeeded["_input_ref"]) == frozen_input
    assert request_hash(h.service, "successor") == frozen_hash
    assert row(h.service, "old") == original_before
    assert len(sends) == 1
    assert h.service.resume_unsent_successor("owner", "successor", h.envelope)["status"] == "succeeded"
    assert len(sends) == 1 and h.admission.claim_next() is None


def test_plain_no_final_resume_endpoint_requires_admin_and_accepts_exact_envelope(task, bound_chat, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import api.ai as api

    h = failed_plain_no_final_successor(task, bound_chat)
    app = FastAPI()
    app.include_router(api.create_router())
    monkeypatch.setattr(api, "text_task_service", h.service)
    monkeypatch.setattr(api, "require_identity", lambda token: {
        "id": "owner", "role": "admin" if token == "internal" else "user",
    })
    monkeypatch.setattr(api, "require_chat_text_policy", lambda _: None)
    url = "/api/conversation-bindings/text-requests/successor/resume-unsent-successor"

    with TestClient(app) as client:
        assert client.post(url, json=h.envelope).status_code == 501
        first = client.post(url, json=h.envelope, headers={"Authorization": "internal"})
        assert first.status_code == 200 and first.json()["status"] == "queued"
        assert client.post(url, json=h.envelope, headers={"Authorization": "internal"}).json() == first.json()


def test_plain_no_final_resume_endpoint_rejects_invalid_sequence_without_requeue(task, bound_chat, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import api.ai as api

    h = failed_plain_no_final_successor(task, bound_chat)
    update(h.service, "successor", _sequence=0)
    before = deepcopy(row(h.service, "successor"))
    app = FastAPI()
    app.include_router(api.create_router())
    monkeypatch.setattr(api, "text_task_service", h.service)
    monkeypatch.setattr(api, "require_identity", lambda _token: {"id": "owner", "role": "admin"})
    monkeypatch.setattr(api, "require_chat_text_policy", lambda _: None)

    with TestClient(app) as client:
        response = client.post(
            "/api/conversation-bindings/text-requests/successor/resume-unsent-successor",
            json=h.envelope,
            headers={"Authorization": "internal"},
        )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE"
    assert row(h.service, "successor") == before
    assert h.admission.claim_next() is None
    assert "send" not in bound_chat.calls
