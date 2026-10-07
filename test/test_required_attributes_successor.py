"""Closed, same-chat legacy attributes upgrade; no live accounts or products."""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from services.category_directory_derivation import compact_json
from services.required_attributes_derivation import (
    KIND, MODEL, EFFORT, LEGACY_CATEGORY_PREFIX, LEGACY_ATTRIBUTES_PREFIX,
    derive_required_attributes_input,
)
from services.conversation_binding_service import ConversationBindingError
from services.text_task_service import TextTaskService
from services.request_context import current_request
from test.test_bound_text_archive import bound_chat
from test.test_explicit_unknown_successor import task, row, update
from test.test_explicit_unknown_successor import (
    test_two_processes_cannot_register_two_successors_and_only_one_can_claim as assert_single_successor,
)
from test.test_no_final_successor import no_final, graph
from test.test_pool_admission import build


def fields(category=False):
    required = {"attribute_id": 9, "name": "Пол", "required": True,
                "provided_by_category": False, "already_satisfied": False,
                "dictionary_id": 12, "dictionary_values": ["Женский", "Мужской"],
                "dictionary_complete": True, "is_collection": True, "max_count": 2}
    common = {"SOURCE_platform": "WILDBERRIES", "SOURCE_category": {"name": "Decorative ring"},
              "SOURCE_title": "Original pearl ring", "SOURCE_description": None}
    schema = [required]
    if category:
        start = common | {"compact_directory": {"paths": [["Jewelry", "Rings"]], "parent_names": ["Rings"],
                                                   "leaves": [[10, 20, "Ring", 0, 0]]}}
        schema += [dict(required, attribute_id=10, provided_by_category=True),
                   dict(required, attribute_id=11, already_satisfied=True), dict(required, attribute_id=12, required=False)]
    else:
        start = {"task_phase": "attributes", "fixed_target_category": {"description_category_id": 10,
                 "type_id": 20, "name": "Ring", "description_category_name": "Rings", "category_path": ["Jewelry", "Rings"]},
                 "required_attribute_targets": [{"attribute_id": 9, "name": "Пол"}], **common}
    return {**start, "schema": schema, "source_attributes": [
        {"source_attribute_index": 0, "name": "size", "value": "16", "source_extra": "retained"},
        {"source_attribute_index": 1, "name": "detail", "value": "pearl"}]}


def prompt(data, category=False):
    prefix = LEGACY_CATEGORY_PREFIX if category else LEGACY_ATTRIBUTES_PREFIX.replace(
        "__FIXED_DESCRIPTION_CATEGORY_ID__", str(data["fixed_target_category"]["description_category_id"])).replace(
        "__FIXED_TYPE_ID__", str(data["fixed_target_category"]["type_id"]))
    return prefix + "".join("\n" + k + "=" + compact_json(v) for k, v in data.items())


def install(h, category=False, data=None):
    h.body.update(model="gpt-5-6-instant", thinking_effort="standard", messages=[{"role": "user", "content": [
        {"type": "text", "text": prompt(data or fields(category), category)},
        {"type": "text", "text": '{"image_ref":"actual-image-not-source-1"}'},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDRA=="}},
        {"type": "text", "text": '{"image_ref":"another-original-image"}'},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,RUZH"}},
    ]}])
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        original.update(_input_ref=h.service.store.save_input(h.body), model="gpt-5-6-instant")
        h.service.store.write_receipt(db, "text", "owner", "old", original)
        db.execute("UPDATE requests SET request_hash=? WHERE owner='owner' AND id='old'",
                   (TextTaskService._submission_identity("owner", h.body)[1],))
    no_final(h)
    h.envelope["derived_input"] = {"kind": KIND}
    return h


def decoded(body):
    data = body["messages"][0]["content"][0]["text"].split("\ntask_phase=", 1)[1]
    return {k: json.loads(v) for k, v in (line.split("=", 1) for line in ("task_phase="+data).splitlines())}


@pytest.mark.parametrize("category", [False, True])
def test_closed_upgrade_preserves_facts_images_and_original_then_runs_once(task, category):
    h = install(task, category); before = deepcopy(h.body)
    first = h.service.submit("owner", h.envelope)
    assert first["status"] == "queued" and first["continue_after_no_final"] is True
    proof = first["derived_input"]
    assert proof["model"] == MODEL and proof["thinking_effort"] == EFFORT
    assert proof["original_input_hash"] == TextTaskService._submission_identity("owner", before)[1]
    assert proof["target"] == {"description_category_id": 10, "type_id": 20}
    saved = h.service.store.load_input(row(h.service, "successor")["_input_ref"])
    assert saved["model"] == MODEL and saved["thinking_effort"] == EFFORT
    assert saved["messages"][0]["content"][1:] == before["messages"][0]["content"][1:]
    data = decoded(saved); original = fields(category)
    for key in ("SOURCE_platform", "SOURCE_category", "SOURCE_title", "SOURCE_description", "source_attributes"):
        assert data[key] == original[key]
    assert data["schema"] == [original["schema"][0]]
    assert data["required_attribute_targets"] == [{"attribute_id": 9, "name": "Пол"}]
    assert proof["schema_count"] == (4 if category else 1) and proof["required_target_count"] == 1 and proof["image_count"] == 2
    text = saved["messages"][0]["content"][0]["text"]
    assert '"image_refs":["actual-image-not-source-1"]' in text and '"source-1"' not in text
    assert "A fine decorative ring with visible pearl" in text
    assert "Do not return both merely because use is nonexclusive" in text
    assert h.body == before and row(h.service) == h.old
    _, store, admission = build(h.root, h.clock)
    service = TextTaskService(store.path, runner=h.service.runner, clock=h.clock, admission=admission)
    admission.register("text", lambda ctx, body: service._run(ctx.owner, ctx.request_id, body))
    ctx = admission.claim_next(); assert ctx and ctx.request_id == "successor"
    admission.execute(ctx)
    assert service.submit("owner", h.envelope)["status"] == "succeeded"
    assert len(h.sent) == 1 and h.sent[0]["model"] == MODEL and h.sent[0]["thinking_effort"] == EFFORT
    assert h.sent[0]["_supersedes_no_final_original"] == h.old
    assert row(service) == h.old and admission.claim_next() is None
    with pytest.raises(ConversationBindingError): service.submit("owner", {**h.envelope, "client_request_id": "third"})


@pytest.mark.parametrize("extra", [{"messages": []}, {"model": MODEL}, {"thinking_effort": "high"},
    {"continue_after_no_final": False}, {"continue_after_no_final": "true"},
    {"derived_input": {"kind": KIND, "prompt": "changed"}}, {"derived_input": {"kind": "category_directory_parent_v1"}}])
def test_illegal_envelope_cannot_enqueue(task, extra):
    from api.ai import ConversationBindingTextRequest
    h = install(task)
    body = {**h.envelope, **extra}
    with pytest.raises(ValidationError): ConversationBindingTextRequest.model_validate(body)
    with pytest.raises(ConversationBindingError): h.service.submit("owner", body)
    assert row(h.service, "successor") is None


def test_both_opt_in_fields_are_required(task):
    from api.ai import ConversationBindingTextRequest
    h = install(task)
    assert ConversationBindingTextRequest.model_validate(h.envelope).model_dump(exclude_unset=True) == h.envelope
    body = {k:v for k,v in h.envelope.items() if k != "continue_after_no_final"}
    with pytest.raises(ValidationError): ConversationBindingTextRequest.model_validate(body)
    with pytest.raises(ConversationBindingError): h.service.submit("owner", body)


@pytest.mark.parametrize("change", ["two_leaves", "same_ids_different_path", "invalid_reference", "no_path",
    "duplicate_schema", "invalid_flag", "no_required", "source_index", "source_value"])
def test_ambiguous_or_incomplete_original_never_derived(task, change):
    data = fields(True)
    if change == "two_leaves": data["compact_directory"]["leaves"].append([30,40,"Other",0,0])
    elif change == "same_ids_different_path":
        data["compact_directory"]["paths"].append(["Another", "Rings"])
        data["compact_directory"]["leaves"].append([10,20,"Ring",0,1])
    elif change == "invalid_reference": data["compact_directory"]["leaves"][0][4] = 7
    elif change == "no_path": data["compact_directory"]["leaves"][0][4] = -1
    elif change == "duplicate_schema": data["schema"].append(deepcopy(data["schema"][0]))
    elif change == "invalid_flag": data["schema"][0]["required"] = 1
    elif change == "no_required": data["schema"][0]["already_satisfied"] = True
    elif change == "source_index": data["source_attributes"][0]["source_attribute_index"] = 7
    elif change == "source_value": data["source_attributes"][0]["value"] = None
    h = install(task, True, data)
    with pytest.raises(ConversationBindingError): h.service.submit("owner", h.envelope)
    assert row(h.service, "successor") is None and not h.sent


@pytest.mark.parametrize("change", ["prefix", "extra_field", "duplicate_key", "duplicate_nested_key", "unpaired_image",
    "reordered_images", "duplicate_ref", "extra_instruction", "old_model", "old_effort", "schema_target_mismatch"])
def test_unrecognized_input_is_rejected_without_silent_truncation(task, change):
    h = install(task); body = deepcopy(h.body); c = body["messages"][0]["content"]
    if change == "prefix": c[0]["text"] = "New instruction\n" + c[0]["text"]
    elif change == "extra_field": c[0]["text"] += '\nextra="instruction"'
    elif change == "duplicate_key": c[0]["text"] += '\nSOURCE_title="another"'
    elif change == "duplicate_nested_key": c[0]["text"] = c[0]["text"].replace('"required":true', '"required":true,"required":false')
    elif change == "unpaired_image": c.pop()
    elif change == "reordered_images": c[1], c[2] = c[2], c[1]
    elif change == "duplicate_ref": c[3] = deepcopy(c[1])
    elif change == "extra_instruction": c += [{"type":"text","text":"additional task"}]
    elif change == "old_model": body["model"] = "gpt-5-6-pro"
    elif change == "old_effort": body["thinking_effort"] = "high"
    elif change == "schema_target_mismatch": c[0]["text"] = c[0]["text"].replace('required_attribute_targets=[{"attribute_id":9', 'required_attribute_targets=[{"attribute_id":8')
    with pytest.raises(ConversationBindingError): derive_required_attributes_input(body, "successor", "old", {"kind":KIND})


def test_arbitrary_counts_and_no_image_instruction(task):
    data = fields(True)
    data["schema"].append(dict(data["schema"][0], attribute_id=77, name="Color"))
    data["source_attributes"].append({"source_attribute_index": 2, "name": "color", "value": "white"})
    h = install(task, True, data); h.body["messages"][0]["content"] = h.body["messages"][0]["content"][:1]
    body, proof = derive_required_attributes_input(h.body, "next", "old", {"kind":KIND})
    assert proof["required_target_count"] == 2 and proof["image_count"] == 0
    assert len(decoded(body)["source_attributes"]) == 3
    assert '"source_title":true' in body["messages"][0]["content"][0]["text"]


@pytest.mark.parametrize("changes", [{"recovery_no_result_reads": 2}, {"_work_key": "work"},
    {"_attempt_reason": "other"}, {"_executing": True}, {"_completion": {"state": "checking"}},
    {"_recovery_suppressed": True}, {"_attempt_finished_at": None}])
def test_original_no_final_proof_stays_required(task, changes):
    h = install(task); update(h.service, **changes)
    with pytest.raises(ConversationBindingError): h.service.submit("owner", h.envelope)
    assert row(h.service, "successor") is None


@pytest.mark.parametrize("field", ["provider_binding_id", "provider_account_identity", "conversation_id",
                                  "client_conversation_id", "parent_message_id"])
def test_original_bindings_cannot_drift(task, field):
    h = install(task)
    with pytest.raises(ConversationBindingError): h.service.submit("owner", {**h.envelope, field:"changed"})
    assert row(h.service, "successor") is None


@pytest.mark.parametrize("change", ["model", "effort", "audit", "parent"])
def test_persisted_successor_revalidated_before_send(task, change):
    h = install(task); h.service.submit("owner", h.envelope)
    saved = row(h.service, "successor")
    if change == "model": saved["model"] = "gpt-5-6-instant"
    elif change == "effort": saved["_derived_input"]["thinking_effort"] = "standard"
    elif change == "audit": saved["_derived_input"]["required_target_count"] = 50
    else: saved["_submission_parent_message_id"] = "changed"
    with h.service.store.transaction() as db:
        h.service.store.write_receipt(db, "text", "owner", "successor", saved)
    ctx = h.admission.claim_next()
    if ctx:
        h.admission.execute(ctx)
        assert row(h.service, "successor")["upstream_outcome"] == "not_sent"
    assert not h.sent


@pytest.mark.parametrize("failure", [None, "changed_tail", "unsupported_model", "read_unavailable"])
def test_upgraded_send_keeps_same_binding_and_fresh_no_final_checks(task, bound_chat, monkeypatch, failure):
    import services.conversation_binding_service as module
    h, chat = install(task), bound_chat
    chat.document = graph(h)
    h.service.runner = chat.service.complete_text
    original_get = module.OpenAIBackendAPI._get_conversation
    reads, sends = [], []
    def read(backend, cid, **kwargs):
        reads.append(cid)
        if failure == "read_unavailable":
            raise RuntimeError("fixture connection failure")
        return original_get(backend, cid)
    monkeypatch.setattr(module.OpenAIBackendAPI, "_get_conversation", read)
    if failure == "unsupported_model":
        chat.accounts.get_bound_text_access_token.side_effect = RuntimeError("bound model unavailable")
    def events(backend, **kwargs):
        if failure == "changed_tail":
            chat.document["mapping"]["tool"]["message"]["content"]["parts"] = ["changed context"]
        backend.text_pre_send_check(None)
        current_request.get().before_send()
        sends.append(kwargs)
        assert kwargs["conversation_id"] == "original-chat" and kwargs["parent_message_id"] == "tool"
        assert kwargs["model"] == MODEL and kwargs["thinking_effort"] == "high"
        backend.text_cursor_callback({"request_parent_message_id": "tool", "_submission_parent_message_id": "tool"})
        yield {"type": "conversation.delta", "conversation_id": "original-chat", "delta": "new answer"}
    monkeypatch.setattr(module, "conversation_events", events)
    monkeypatch.setattr(module.ConversationBindingService, "_read_text_request_result", staticmethod(
        lambda *_: {"status": "succeeded", "content": "new answer", "parent_message_id": "new-answer"}))
    h.service.submit("owner", h.envelope)
    h.admission.execute(h.admission.claim_next())
    result = row(h.service, "successor")
    assert row(h.service) == h.old
    chat.accounts.create_conversation_binding.assert_not_called()
    if failure:
        assert not sends and result["upstream_outcome"] == "not_sent"
    else:
        assert len(sends) == 1 and reads == ["original-chat", "original-chat"] and result["status"] == "succeeded"
        chat.accounts.get_bound_text_access_token.assert_called_with("original-binding", model=MODEL, for_message=True)


def test_http_upgrade_is_internal_idempotent_and_exposes_only_proof(task, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock
    import api.ai as api
    h = install(task)
    app = FastAPI(); app.include_router(api.create_router())
    monkeypatch.setattr(api, "text_task_service", h.service)
    monkeypatch.setattr(api, "require_identity", lambda token: {"id": "owner", "role": "admin" if token == "internal" else "user"})
    monkeypatch.setattr(api, "require_chat_text_policy", lambda _: None)
    review = AsyncMock(); monkeypatch.setattr(api, "filter_or_log", review)
    with TestClient(app) as client:
        url, headers = "/api/conversation-bindings/text", {"Authorization": "internal"}
        assert client.post(url, json=h.envelope, headers={"Authorization": "ordinary"}).status_code == 501
        assert client.post(url, json={**h.envelope, "model": MODEL}, headers=headers).status_code == 422
        first = client.post(url, json=h.envelope, headers=headers)
        assert first.status_code == 200, first.text
        assert first.json()["continue_after_no_final"] is True and first.json()["model"] == MODEL
        assert client.post(url, json=h.envelope, headers=headers).json() == first.json()
        reread = client.get("/api/conversation-bindings/text-requests/successor", headers=headers)
        assert reread.status_code == 200 and reread.json()["derived_input"] == first.json()["derived_input"]
        assert review.await_count == 1 and "actual-image-not-source-1" in review.await_args.args[1]
        assert "Original pearl ring" not in first.text and "_input_ref" not in first.text and "QUJDRA" not in first.text
        assert row(h.service) == h.old


def test_upgrade_two_processes_still_register_and_claim_one_successor(task):
    assert_single_successor(install(task))


def test_large_retained_text_stays_rejected(task):
    data = fields(); data["SOURCE_title"] = "x" * 65536
    h = install(task, data=data)
    with pytest.raises(ConversationBindingError) as exc:
        h.service.submit("owner", h.envelope)
    assert exc.value.code == "CHAT_DERIVED_INPUT_TOO_LARGE" and row(h.service, "successor") is None
