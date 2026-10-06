"""Completed image-tool parents and one-shot recovery of the original unsent ID."""
from copy import deepcopy
import multiprocessing
from pathlib import Path
from unittest.mock import Mock

import pytest

from services.conversation_binding_service import ConversationBindingError, ConversationBindingService
from services.text_task_service import TextTaskService
from test.test_bound_text_archive import bound_chat
from test.test_category_directory_successor import install
from test.test_explicit_unknown_successor import task, row, update
from test.test_pool_admission import build


def image_branch():
    # The sanitized 581 shape: user -> assistant code call -> terminal image tool.
    return {"conversation_id": "original-chat", "current_node": "previous-answer", "is_archived": False,
        "mapping": {
            "image-user": {"parent": "older-node", "children": ["image-call"], "message": {
                "id": "image-user", "author": {"role": "user"}, "status": "finished_successfully", "end_turn": None}},
            "image-call": {"parent": "image-user", "children": ["previous-answer"], "message": {
                "id": "image-call", "author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": False,
                "recipient": "image-tool", "content": {"content_type": "code", "text": "synthetic tool input"}}},
            "previous-answer": {"parent": "image-call", "children": [], "message": {
                "id": "previous-answer", "author": {"role": "tool", "name": "image-tool"}, "status": "finished_successfully",
                "end_turn": None, "recipient": "all", "content": {"content_type": "multimodal_text", "parts": [
                    {"content_type": "image_asset_pointer", "asset_pointer": "file-service://synthetic_image"}]}}}}}


def failed_successor(task, bound_chat):
    h = install(task)
    h.service.runner = bound_chat.service.complete_text
    # Produce the real terminal local no-send shape through the normal runner.
    bound_chat.document["mapping"]["previous-answer"]["message"]["end_turn"] = False
    h.service.submit("owner", h.envelope)
    h.admission.execute(h.admission.claim_next())
    assert row(h.service, "successor")["error_code"] == "CHAT_SUPERSEDE_CURSOR_CHANGED"
    bound_chat.document = image_branch()
    bound_chat.calls.clear()
    return h


def test_complete_image_tool_shape_is_valid_without_generic_null_end_turn_permission():
    ConversationBindingService._check_superseded_original(image_branch(), "original-chat", "previous-answer", "old-user")


@pytest.mark.parametrize("bad", ["active_tool", "active_call", "active_user", "no_image", "bad_pointer", "bad_recipient",
    "missing_call", "missing_user", "wrong_message", "declared_branch", "reverse_branch", "head_changed", "child",
    "cycle", "generic_tool", "original_present", "missing_leaf_children"])
def test_image_tool_compatibility_rejects_incomplete_or_foreign_branches(bad):
    doc = image_branch(); mapping = doc["mapping"]; tool = mapping["previous-answer"]["message"]
    if bad.startswith("active_"):
        key = {"active_tool": "previous-answer", "active_call": "image-call", "active_user": "image-user"}[bad]
        mapping[key]["message"]["status"] = "in_progress"
    elif bad == "no_image": tool["content"]["parts"] = ["tool returned no image"]
    elif bad == "bad_pointer": tool["content"]["parts"][0]["asset_pointer"] = "https://example.invalid/not-an-image-proof"
    elif bad == "bad_recipient": mapping["image-call"]["message"]["recipient"] = "different-tool"
    elif bad == "missing_call": del mapping["image-call"]
    elif bad == "missing_user": del mapping["image-user"]
    elif bad == "wrong_message": mapping["image-call"]["message"]["id"] = "different"
    elif bad == "declared_branch": mapping["image-user"]["children"].append("other")
    elif bad == "reverse_branch": mapping["other"] = {"parent": "image-call"}
    elif bad == "head_changed": doc["current_node"] = "image-call"
    elif bad == "child": mapping["later"] = {"parent": "previous-answer"}
    elif bad == "cycle": mapping["image-call"]["parent"] = "previous-answer"
    elif bad == "generic_tool": tool["content"] = {"content_type": "text", "parts": ["done"]}
    elif bad == "missing_leaf_children": del mapping["previous-answer"]["children"]
    else: mapping["old-user"] = {"message": {"id": "old-user"}}
    with pytest.raises(ConversationBindingError) as exc:
        ConversationBindingService._check_superseded_original(doc, "original-chat", "previous-answer", "old-user")
    assert exc.value.code == ("CHAT_SUPERSEDE_ORIGINAL_FOUND" if bad == "original_present" else "CHAT_SUPERSEDE_CURSOR_CHANGED")


def test_read_advertises_only_verified_no_send_and_never_resumes(task, bound_chat):
    h = failed_successor(task, bound_chat)
    before = deepcopy(row(h.service, "successor"))
    result = h.service.read("owner", "successor")
    assert result["bound_successor_resume_retryable"] is True
    assert result["upstream_outcome"] == "not_submitted" and result["upstream_submission_started"] is False
    h.service.submit("owner", h.envelope)
    h.service.recover("owner", "successor", True)
    assert row(h.service, "successor") == before and not bound_chat.calls
    assert h.admission.claim_next() is None


def test_explicit_resume_preserves_identity_history_and_runs_only_once(task, bound_chat, monkeypatch):
    h = failed_successor(task, bound_chat)
    before = deepcopy(row(h.service, "successor"))
    result = h.service.resume_unsent_successor("owner", "successor", h.envelope)
    assert result["status"] == "queued" and "bound_successor_resume_retryable" not in result
    queued = row(h.service, "successor")
    assert queued["_bound_successor_resume_count"] == 1
    assert queued["_bound_successor_resume_failure"]["error_code"] == "CHAT_SUPERSEDE_CURSOR_CHANGED"
    for key in ("request_id", "request_message_id", "_input_ref", "_supersedes_request_id", "_supersedes_input_hash", "_derived_input", "_sequence"):
        assert queued[key] == before[key]
    assert h.service.resume_unsent_successor("owner", "successor", h.envelope) == result
    from test.test_explicit_unknown_successor import test_real_paced_send_checks_after_cooldown_and_saves_only_own_result
    test_real_paced_send_checks_after_cooldown_and_saves_only_own_result(h, bound_chat, monkeypatch, "success")
    assert h.service.resume_unsent_successor("owner", "successor", h.envelope)["status"] == "succeeded"
    assert h.admission.claim_next() is None and row(h.service) == h.old


def test_resume_requires_durable_admission_without_consuming_one_shot(task, bound_chat):
    h = failed_successor(task, bound_chat)
    before = deepcopy(row(h.service, "successor"))
    h.service.admission = None
    h.service.executor = Mock()
    h.service.executor.submit.side_effect = RuntimeError("executor unavailable")
    assert "bound_successor_resume_retryable" not in h.service.read("owner", "successor")
    with pytest.raises(ConversationBindingError) as exc:
        h.service.resume_unsent_successor("owner", "successor", h.envelope)
    assert exc.value.code == "CHAT_UNSENT_SUCCESSOR_NOT_RESUMABLE"
    assert row(h.service, "successor") == before
    h.service.executor.submit.assert_not_called()
    assert not bound_chat.calls
    h.service.admission = h.admission
    assert h.service.resume_unsent_successor("owner", "successor", h.envelope)["status"] == "queued"
    assert row(h.service, "successor")["_bound_successor_resume_count"] == 1


@pytest.mark.parametrize("outcome", ["late_user", "late_parent", "429", "timeout"])
def test_image_tool_resume_retains_final_paced_pre_send_check(task, bound_chat, monkeypatch, outcome):
    from test.test_explicit_unknown_successor import test_real_paced_send_checks_after_cooldown_and_saves_only_own_result
    h = failed_successor(task, bound_chat)
    h.service.resume_unsent_successor("owner", "successor", h.envelope)
    test_real_paced_send_checks_after_cooldown_and_saves_only_own_result(h, bound_chat, monkeypatch, outcome)


@pytest.mark.parametrize("changes", [
    {"_submission_started": True}, {"_turn_reserved": True}, {"_executing": True}, {"_last_sent_sequence": 0},
    {"send_count": 1}, {"_claim_id": "active", "_claim_until": 2000}, {"_claim_id": "unknown", "_claim_until": None},
    {"recovery_claim_id": "active", "recovery_lease_until": 2000}, {"_recovery_paused": True},
    {"_execution_timeline": [{"stage": "accepted"}, {"stage": "send_call_started"}]},
    {"upstream_outcome": "unknown"}, {"status": "unknown"}, {"_derived_input": None},
    {"_execution_wait_ended_at": 900}, {"_attempt_finished_at": 900},
])
def test_ineligible_successor_never_advertises_or_requeues(task, bound_chat, changes):
    h = failed_successor(task, bound_chat)
    update(h.service, "successor", **changes)
    assert "bound_successor_resume_retryable" not in h.service._public(row(h.service, "successor"))
    with pytest.raises(ConversationBindingError):
        h.service.resume_unsent_successor("owner", "successor", h.envelope)
    assert h.admission.claim_next() is None and not bound_chat.calls


@pytest.mark.parametrize("field", ["provider_account_identity", "provider_binding_id", "client_conversation_id", "conversation_id", "parent_message_id", "supersedes_request_id", "derived_input", "messages"])
def test_changed_envelope_cannot_resume(task, bound_chat, field):
    h = failed_successor(task, bound_chat)
    with pytest.raises(ConversationBindingError):
        h.service.resume_unsent_successor("owner", "successor", {**h.envelope, field: "different"})
    assert h.admission.claim_next() is None


def test_original_hash_tamper_disables_advertisement_and_resume(task, bound_chat):
    h = failed_successor(task, bound_chat)
    with h.service.store.transaction() as db:
        db.execute("UPDATE requests SET request_hash='changed' WHERE id='old'")
    assert "bound_successor_resume_retryable" not in h.service.read("owner", "successor")
    with pytest.raises(ConversationBindingError): h.service.resume_unsent_successor("owner", "successor", h.envelope)


@pytest.mark.parametrize("after", ["branch", "original"])
def test_resumed_request_fresh_check_refuses_new_branch_or_original_result(task, bound_chat, after):
    h = failed_successor(task, bound_chat)
    h.service.resume_unsent_successor("owner", "successor", h.envelope)
    if after == "branch": bound_chat.document["current_node"] = "image-call"
    else: bound_chat.document["mapping"][h.old["request_message_id"]] = {"message": {"id": h.old["request_message_id"]}}
    h.admission.execute(h.admission.claim_next())
    failed = deepcopy(row(h.service, "successor"))
    assert failed["status"] == "failed" and "send" not in bound_chat.calls
    assert "bound_successor_resume_retryable" not in h.service.read("owner", "successor")
    assert h.service.resume_unsent_successor("owner", "successor", h.envelope)["status"] == "failed"
    assert row(h.service, "successor") == failed and h.admission.claim_next() is None


def resume_worker(root, envelope, ready, start, results):
    service = TextTaskService(Path(root) / "text_tasks.sqlite3", admission=Mock())
    ready.put(True); start.wait(10)
    try: results.put(service.resume_unsent_successor("owner", "successor", envelope)["status"])
    except ConversationBindingError as exc: results.put(exc.code)


def test_concurrent_same_id_and_restart_share_one_atomic_resume(task, bound_chat):
    h = failed_successor(task, bound_chat)
    mp = multiprocessing.get_context("spawn"); ready, results, start = mp.Queue(), mp.Queue(), mp.Event()
    workers = [mp.Process(target=resume_worker, args=(str(h.root), h.envelope, ready, start, results)) for _ in range(2)]
    for worker in workers: worker.start()
    for _ in workers: ready.get(timeout=20)
    start.set(); assert [results.get(timeout=20) for _ in workers] == ["queued", "queued"]
    for worker in workers:
        worker.join(20); assert worker.exitcode == 0
    assert row(h.service, "successor")["_bound_successor_resume_count"] == 1
    _, store, admission = build(h.root, h.clock)
    service = TextTaskService(store.path, runner=h.service.runner, clock=h.clock, admission=admission)
    admission.register("text", lambda ctx, body: service._run(ctx.owner, ctx.request_id, body))
    assert service.resume_unsent_successor("owner", "successor", h.envelope)["status"] == "queued"
    admission.execute(admission.claim_next())
    assert bound_chat.calls.count("send") == 1 and admission.claim_next() is None


def test_explicit_endpoint_authority_path_and_closed_body(task, bound_chat, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import api.ai as api
    h = failed_successor(task, bound_chat); app = FastAPI(); app.include_router(api.create_router())
    monkeypatch.setattr(api, "text_task_service", h.service)
    monkeypatch.setattr(api, "require_identity", lambda token: {"id": "owner", "role": "admin" if token == "internal" else "user"})
    monkeypatch.setattr(api, "require_chat_text_policy", lambda _: None)
    with TestClient(app) as client:
        url = "/api/conversation-bindings/text-requests/successor/resume-unsent-successor"
        assert client.post(url, json=h.envelope).status_code == 501
        assert client.post(url, json={**h.envelope, "messages": []}, headers={"Authorization": "internal"}).status_code == 422
        assert client.post(url.replace('/successor/', '/other/'), json=h.envelope, headers={"Authorization": "internal"}).status_code == 409
        first = client.post(url, json=h.envelope, headers={"Authorization": "internal"})
        assert first.status_code == 200 and first.json()["status"] == "queued"
        assert client.post(url, json=h.envelope, headers={"Authorization": "internal"}).json() == first.json()
