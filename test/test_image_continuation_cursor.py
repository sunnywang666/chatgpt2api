"""Read-only context continuation, never adoption or replay of a refused image."""
import copy
import json
import time
from contextlib import nullcontext
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.image_tasks as routes
from services.image_task_service import ImageTaskService, _similarity_failure_tail
from services.image_thread import ImageThreadError
from services.openai_backend_api import OpenAIBackendAPI
from test.test_image_task_service import OWNER, write_policy_task, failure_cursor_document


ANCHORS = dict(provider_binding_id="binding-1", provider_account_identity="account-1",
               client_conversation_id="client-chat-1", conversation_id="conversation-1")
REFUSAL = "非常抱歉，生成的图片可能违反了关于与第三方内容相似性的防护限制。如果你认为此判断有误，请重试或修改提示语。"


def completed_context(turns=15):
    doc = failure_cursor_document()
    mapping = doc["mapping"]
    mapping["fresh-terminal"]["message"]["content"]["parts"] = [REFUSAL]
    parent = "fresh-terminal"
    for i in range(turns):
        user, final = f"manual-{i}", f"final-{i}"
        mapping[user] = {"parent": parent, "message": {
            "id": user, "author": {"role": "user"}, "content": {"content_type": "text", "parts": ["manual context"]}}}
        mapping[final] = {"parent": user, "message": {
            "id": final, "author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True,
            "channel": "final", "content": {"content_type": "text", "parts": ["manual result"]}}}
        parent = final
    doc["current_node"] = parent
    return doc


class ReadBackend(OpenAIBackendAPI):
    def __init__(self, document):
        self.document, self.reads, self.task_reads, self.closed = document, 0, 0, False
        self.tasks = []
        self.on_read = lambda: None

    def _get_conversation(self, conversation_id, **kwargs):
        assert conversation_id == "conversation-1"
        self.reads += 1
        self.on_read()
        return copy.deepcopy(self.document)

    def _query_backend_tasks(self, **kwargs):
        assert kwargs == dict(conversation_id="conversation-1", timeout_secs=5.0, strict_schema=True)
        self.task_reads += 1
        return copy.deepcopy(self.tasks)

    def close(self):
        self.closed = True


@pytest.fixture
def harness(tmp_path, monkeypatch):
    path = tmp_path / "image_tasks.json"
    write_policy_task(path, upstream_outcome="unknown", request_parent_message_id="anchor")
    service = ImageTaskService(path, generation_handler=mock.Mock(), edit_handler=mock.Mock(),
                               retention_days_getter=lambda: 30)
    update_receipt(service, request_parent_message_id="anchor")
    backend = ReadBackend(completed_context())
    # Keep real static asset parsers while replacing only construction.
    class BackendFactory(ReadBackend):
        def __new__(cls, *args, **kwargs):
            return backend
    monkeypatch.setattr("services.openai_backend_api.OpenAIBackendAPI", BackendFactory)
    accounts = mock.Mock()
    accounts.get_bound_account_identity.return_value = "account-1"
    accounts.get_bound_text_access_token.return_value = "token-never-returned"
    accounts.conversation_binding_lock.side_effect = lambda *a: nullcontext()
    monkeypatch.setattr("services.account_service.account_service", accounts)
    return service, backend, accounts


def read_receipt(service):
    with service.store.connect() as db:
        return service.store.read_receipt(db, "image", OWNER["id"], "policy-task")


def update_receipt(service, **changes):
    with service.store.transaction() as db:
        task = service.store.read_receipt(db, "image", OWNER["id"], "policy-task")
        task.update(changes)
        service.store.write_receipt(db, "image", OWNER["id"], "policy-task", task)


@pytest.mark.parametrize("turns", [0, 1, 15])
def test_completed_manual_tail_keeps_original_failure_and_has_no_write(harness, turns):
    service, backend, _ = harness
    backend.document = completed_context(turns)
    # A manual image remains context; it is never adopted by the old task.
    if turns:
        backend.document["mapping"]["final-0"]["message"]["content"] = {
            "content_type": "multimodal_text", "parts": [{"content_type": "image_asset_pointer", "asset_pointer": "file-service://manual-image"}]}
    before = read_receipt(service)
    proof = service.continuation_cursor(OWNER, "policy-task", ANCHORS)
    assert proof == {**ANCHORS, "source_task_id": "policy-task", "source_request_message_id": "original-request",
        "parent_message_id": backend.document["current_node"], "original_terminal_message_id": "fresh-terminal",
        "failure_reason": "third_party_similarity", "observed_at": proof["observed_at"]}
    assert time.time() - proof["observed_at"] < 5
    assert read_receipt(service) == before
    assert backend.reads == backend.task_reads == 1 and backend.closed
    assert "token" not in json.dumps(proof) and "manual result" not in json.dumps(proof)


@pytest.mark.parametrize("change", ["active", "user", "cycle", "missing", "fork", "declared_fork", "wrong_message",
    "wrong_chat", "archived", "different_branch", "original_asset", "generic_policy", "later_only_refusal", "unfinished_middle", "wrong_parent", "missing_anchor", "tool_recipient"])
def test_invalid_context_never_proves_continuation(harness, change):
    service, backend, _ = harness
    doc, mapping = backend.document, backend.document["mapping"]
    if change == "active": mapping[doc["current_node"]]["message"]["status"] = "in_progress"
    elif change == "tool_recipient": mapping[doc["current_node"]]["message"]["recipient"] = "python"
    elif change == "user": doc["current_node"] = "manual-14"
    elif change == "cycle": mapping["manual-14"]["parent"] = "final-14"
    elif change == "missing": del mapping["manual-3"]
    elif change == "fork": mapping["fork"] = {"parent": "fresh-terminal", "message": {"id": "fork", "author": {"role": "user"}}}
    elif change == "declared_fork": mapping["fresh-terminal"]["children"] = ["manual-0", "missing"]
    elif change == "wrong_message": mapping["original-request"]["message"]["id"] = "other"
    elif change == "wrong_chat": doc["conversation_id"] = "other"
    elif change == "archived": doc["is_archived"] = True
    elif change == "different_branch": mapping["manual-0"]["parent"] = "anchor"
    elif change == "original_asset": mapping["fresh-terminal"]["message"]["metadata"] = {"asset_pointer": "file-service://old-output"}
    elif change in {"generic_policy", "later_only_refusal"}:
        mapping["fresh-terminal"]["message"]["content"]["parts"] = ["I cannot generate this due to content policy."]
        mapping["final-14"]["message"]["content"]["parts"] = [REFUSAL]
    elif change == "unfinished_middle": mapping["final-2"]["message"]["end_turn"] = False
    elif change == "wrong_parent": mapping["original-request"]["parent"] = "other-anchor"
    elif change == "missing_anchor": del mapping["anchor"]
    before = read_receipt(service)
    with pytest.raises(ImageThreadError, match="IMAGE_CONTINUATION_UNAVAILABLE"):
        service.continuation_cursor(OWNER, "policy-task", ANCHORS)
    assert read_receipt(service) == before and backend.task_reads == 0


@pytest.mark.parametrize("case", ["later_completed", "original", "active", "not_ended", "empty", "non_text"])
def test_commentary_is_only_completed_later_context(harness, case):
    service, backend, _ = harness
    doc = backend.document = completed_context(2)
    message = doc["mapping"]["fresh-terminal" if case == "original" else "final-0"]["message"]
    message["channel"] = "commentary"
    if case == "active": message["status"] = "in_progress"
    elif case == "not_ended": message["end_turn"] = False
    elif case == "empty": message["content"]["parts"] = [" "]
    elif case == "non_text": message["content"] = {"content_type": "code", "text": "context"}
    before = read_receipt(service)
    if case == "later_completed":
        proof = service.continuation_cursor(OWNER, "policy-task", ANCHORS)
        assert proof["parent_message_id"] == "final-1"
        assert proof["original_terminal_message_id"] == "fresh-terminal"
        from services.conversation_binding_service import _completed_request_turn
        children = {}
        for key, node in doc["mapping"].items(): children.setdefault(node.get("parent"), []).append(key)
        assert _completed_request_turn(doc["mapping"], children, "manual-0", "conversation-1") is None
    else:
        with pytest.raises(ImageThreadError, match="IMAGE_CONTINUATION_UNAVAILABLE"):
            service.continuation_cursor(OWNER, "policy-task", ANCHORS)
        assert backend.task_reads == 0
    assert read_receipt(service) == before


@pytest.mark.parametrize("case", ["past", "original", "current", "active_tool", "active_recap",
    "wrong_recipient", "recap_recipient", "no_tool", "unended_text", "fork"])
def test_completed_past_tool_recap_is_context_only(harness, case):
    service, backend, _ = harness
    doc = backend.document = completed_context(2)
    mapping = doc["mapping"]
    final = "fresh-terminal" if case == "original" else "final-0"
    user = mapping[final]["parent"]
    previous = user
    for index, kind in enumerate(["multimodal_text", "multimodal_text", "text"]):
        node = f"past-tool-{index}"
        mapping[node] = {"parent": previous, "message": {"id": node,
            "author": {"role": "tool"}, "status": "finished_successfully",
            "content": {"content_type": kind, "parts": ["past context"]}}}
        previous = node
    mapping[final]["parent"] = previous
    mapping[final]["message"].update(end_turn=False, channel=None,
        content={"content_type": "reasoning_recap"})
    if case == "current":
        del mapping["manual-1"], mapping["final-1"]
        doc["current_node"] = final
    elif case == "active_tool": mapping["past-tool-1"]["message"]["status"] = "in_progress"
    elif case == "active_recap": mapping[final]["message"]["status"] = "in_progress"
    elif case == "wrong_recipient": mapping["past-tool-1"]["message"]["recipient"] = "python"
    elif case == "recap_recipient": mapping[final]["message"]["recipient"] = "python"
    elif case == "no_tool": mapping["past-tool-1"]["message"]["author"] = {"role": "assistant"}
    elif case == "unended_text": mapping[final]["message"]["content"] = {"content_type": "text", "parts": ["unfinished"]}
    elif case == "fork": mapping["fork"] = {"parent": previous, "message": {"id": "fork", "author": {"role": "user"}}}
    before = read_receipt(service)
    if case == "past":
        proof = service.continuation_cursor(OWNER, "policy-task", ANCHORS)
        assert proof["parent_message_id"] == "final-1"
        assert proof["original_terminal_message_id"] == "fresh-terminal"
        assert backend.task_reads == 1
    else:
        with pytest.raises(ImageThreadError, match="IMAGE_CONTINUATION_UNAVAILABLE"):
            service.continuation_cursor(OWNER, "policy-task", ANCHORS)
        assert backend.task_reads == 0
    assert read_receipt(service) == before


@pytest.mark.parametrize("changes", [dict(status="success"), dict(error_code="RESULT_UNRECOVERABLE"),
    dict(upstream_unfinished=True), dict(data=[{"url": "old"}]), dict(_pending_image_result_ids=["old"]),
    dict(_recovery_paused=True), dict(_recovery_suppressed=True), dict(_executing=True), dict(_turn_reserved=True),
    dict(_claim_id="active", _claim_until=time.time()+1000), dict(recovery_claim_id="active"),
    dict(_completion={"replacement_id": "child", "state": "retry_queued"})])
def test_ineligible_receipt_never_contacts_upstream(harness, changes):
    service, backend, _ = harness
    update_receipt(service, **changes)
    with pytest.raises(ImageThreadError): service.continuation_cursor(OWNER, "policy-task", ANCHORS)
    assert backend.reads == 0


@pytest.mark.parametrize("field", list(ANCHORS))
def test_four_anchors_are_exact_before_read(harness, field):
    service, backend, _ = harness
    with pytest.raises(ImageThreadError, match="IMAGE_CONTINUATION_IDENTITY_MISMATCH"):
        service.continuation_cursor(OWNER, "policy-task", {**ANCHORS, field: "other"})
    assert backend.reads == 0


@pytest.mark.parametrize("identity", [dict(id="other-admin", role="admin"), dict(id="owner-1", role="user")])
def test_receipt_owner_and_admin_both_required(harness, identity):
    service, backend, _ = harness
    with pytest.raises(ImageThreadError) as error: service.continuation_cursor(identity, "policy-task", ANCHORS)
    assert error.value.status == 404 and backend.reads == 0


@pytest.mark.parametrize("kind", ["text", "image"])
def test_other_local_inflight_on_same_chat_blocks_read(harness, kind):
    service, backend, _ = harness
    with service.store.transaction() as db:
        if kind == "text":
            db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("other-owner", "new-task", "hash", "{}"))
        service.store.write_receipt(db, kind, "other-owner", "new-task", {
            **ANCHORS, "id": "new-task", "request_id": "new-task", "status": "running"})
    with pytest.raises(ImageThreadError, match="IMAGE_CONTINUATION_BUSY"):
        service.continuation_cursor(OWNER, "policy-task", ANCHORS)
    assert backend.reads == 0


@pytest.mark.parametrize("identities", [["changed"], ["account-1", "changed"], ["account-1", "account-1", "changed"]])
def test_account_rotation_is_rechecked(harness, identities):
    service, backend, accounts = harness
    accounts.get_bound_account_identity.side_effect = identities
    with pytest.raises(ImageThreadError, match="IMAGE_CONTINUATION_IDENTITY_MISMATCH"):
        service.continuation_cursor(OWNER, "policy-task", ANCHORS)
    assert backend.reads == (1 if len(identities) == 3 else 0)


@pytest.mark.parametrize("kind", ["text", "image"])
@pytest.mark.parametrize("state,next_at,blocked", [("checking_original", None, True), ("needs_attention", 100, True),
    ("completed", None, False), ("result_ready", None, False)])
def test_completion_scheduled_on_another_receipt_blocks_proof(harness, kind, state, next_at, blocked):
    service, backend, _ = harness
    with service.store.transaction() as db:
        if kind == "text":
            db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("other-owner", "other", "hash", "{}"))
        service.store.write_receipt(db, kind, "other-owner", "other", {**ANCHORS, "status": "error",
            "_completion": {"state": state, "next_at": next_at}})
    if blocked:
        with pytest.raises(ImageThreadError, match="IMAGE_CONTINUATION_BUSY"):
            service.continuation_cursor(OWNER, "policy-task", ANCHORS)
        assert backend.reads == 0
    else:
        assert service.continuation_cursor(OWNER, "policy-task", ANCHORS)["source_task_id"] == "policy-task"


@pytest.mark.parametrize("status,code", [(429, "RECOVERY_RATE_LIMITED"), (401, "RECOVERY_AUTH_REQUIRED"),
    (403, "RECOVERY_AUTH_REQUIRED"), (502, "RECOVERY_READ_FAILED")])
def test_upstream_failure_never_returns_a_cursor(harness, monkeypatch, status, code):
    service, backend, _ = harness
    error = RuntimeError("secret upstream response must not be returned")
    error.status_code, error.retry_after = status, 17
    monkeypatch.setattr(backend, "_get_conversation", mock.Mock(side_effect=error))
    with pytest.raises(ImageThreadError, match=code) as result:
        service.continuation_cursor(OWNER, "policy-task", ANCHORS)
    assert backend.closed and backend.task_reads == 0
    assert "secret" not in str(result.value)
    if status == 429: assert result.value.retry_after == 17


def test_active_upstream_task_or_local_read_race_never_returns_proof(harness):
    service, backend, _ = harness
    backend.tasks = [{"status": "in_progress"}]
    with pytest.raises(ImageThreadError, match="IMAGE_CONTINUATION_BUSY"):
        service.continuation_cursor(OWNER, "policy-task", ANCHORS)
    backend.tasks = []
    backend.on_read = lambda: update_receipt(service, request_message_id="changed")
    with pytest.raises(ImageThreadError, match="IMAGE_CONTINUATION_IDENTITY_MISMATCH"):
        service.continuation_cursor(OWNER, "policy-task", ANCHORS)


def test_tail_changes_after_proof_existing_send_recheck_blocks_generation(harness, monkeypatch):
    from services.conversation_binding_service import ConversationBindingError, ConversationBindingService
    service, backend, accounts = harness
    proof = service.continuation_cursor(OWNER, "policy-task", ANCHORS)
    backend.document["current_node"] = "new-manual-request"
    monkeypatch.setattr("services.conversation_binding_service.account_service", accounts)
    monkeypatch.setattr("services.conversation_binding_service.OpenAIBackendAPI", lambda **kw: backend)
    generate = mock.Mock(side_effect=AssertionError("must not send after cursor drift"))
    monkeypatch.setattr("services.conversation_binding_service.conversation_events", generate)
    with pytest.raises(ConversationBindingError) as error:
        ConversationBindingService().complete_text({**ANCHORS, "parent_message_id": proof["parent_message_id"],
            "messages": [{"role": "user", "content": "Describe the package quantity."}], "model": "auto"})
    assert error.value.code == "CONVERSATION_BINDING_MISMATCH"
    generate.assert_not_called()


def test_exact_formal_get_response_and_permission_boundary(harness, monkeypatch):
    service, backend, _ = harness
    identity = mock.Mock(return_value=OWNER)
    monkeypatch.setattr(routes, "require_identity", identity)
    monkeypatch.setattr(routes, "image_task_service", service)
    app = FastAPI(); app.include_router(routes.create_router())
    with TestClient(app) as client:
        url = "/api/image-tasks/policy-task/continuation-cursor"
        response = client.get(url, params=ANCHORS)
        assert response.status_code == 200 and response.json()["parent_message_id"] == "final-14"
        assert response.headers["cache-control"] == "private, no-store"
        identity.return_value = {**OWNER, "role": "user"}
        assert client.get(url, params=ANCHORS).status_code == 404
        identity.return_value = OWNER
        assert client.get(url, params={}).status_code == 422
        backend.tasks = [{"status": "running"}]
        assert client.get(url, params=ANCHORS).json()["detail"]["code"] == "IMAGE_CONTINUATION_BUSY"
        error = ImageThreadError("RECOVERY_RATE_LIMITED", status=429)
        error.retry_after = 23
        monkeypatch.setattr(service, "continuation_cursor", mock.Mock(side_effect=error))
        limited = client.get(url, params=ANCHORS)
        assert limited.status_code == 429 and limited.headers["retry-after"] == "23"
        assert client.post(url, params=ANCHORS).status_code == 405
