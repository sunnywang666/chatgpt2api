"""Retained combined-copy handling stays one explicit no-final child."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from services.category_directory_derivation import compact_json
from services.combined_copy_derivation import (KIND, LEGACY_BOUNDARY, LEGACY_SUFFIX, CURRENT_SUFFIX, MODEL, EFFORT,
                                                compact_retained_messages, derive_combined_copy_input,
                                                is_current_combined_copy_input,
                                                is_legacy_combined_copy_input,
                                                recognize_compact_retained_messages)
from services.conversation_binding_service import ConversationBindingError
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_explicit_unknown_successor import task, row, update
from test.test_no_final_successor import no_final
from test.test_pool_admission import build


def branch(index, *, category="Ring", variant="one", evidence=None):
    evidence = evidence or {"source_attributes": [{"source_attribute_index": 0, "name": "size", "values": ["16"]}],
                            "source_images": [{"image_ref": "source-a", "url": "asset://a"}],
                            "source_title": "Pearl ring", "source_description": "Synthetic fixture"}
    return {"preparation_index": index, "target_platform": "ozon",
            "category": {"name": category, "variant_note": variant},
            "schema": [{"attribute_id": index + 1, "name": "Color", "required": True}], **evidence}


def legacy_prompt(rows, *, json_envelope=True):
    instruction = LEGACY_SUFFIX + LEGACY_BOUNDARY + compact_json(rows)
    if not json_envelope:
        return "Keep the original copy text and response contract exactly." + instruction
    return compact_json({"task": "copy", "source_facts": {"title": "Synthetic fixture", "count": 2},
                         "keep_nested": {"enabled": True, "labels": ["a", "b"]},
                         "category_attribute_instructions": instruction})


WBOZ_CURRENT_CASES = tuple(json.loads(
    (Path(__file__).parent / "fixtures" / "wboz-current-combined-copy.json").read_text(encoding="utf-8")
)["cases"])


WBOZ_STRUCTURED_CASES = tuple(json.loads(
    (Path(__file__).parent / "fixtures" / "wboz-structured-combined-copy.json").read_text(encoding="utf-8")
)["cases"])


def install(h, rows=None):
    rows = rows or [branch(0), branch(1, category="Brooch", variant="two")]
    h.body.update(model=MODEL, thinking_effort=EFFORT, messages=[{"role": "user", "content": [
        {"type": "text", "text": legacy_prompt(rows)},
        {"type": "text", "text": '{"image_ref":"source-a"}'},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDRA=="}},
        {"type": "text", "text": '{"image_ref":"source-b"}'},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,RUZH"}},
    ]}])
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        original.update(_input_ref=h.service.store.save_input(h.body), model=MODEL, thinking_effort=EFFORT)
        h.service.store.write_receipt(db, "text", "owner", "old", original)
        db.execute("UPDATE requests SET request_hash=? WHERE owner='owner' AND id='old'",
                   (TextTaskService._submission_identity("owner", h.body)[1],))
    no_final(h)
    h.envelope["derived_input"] = {"kind": KIND}
    return h


def compacted(body):
    text = body["messages"][0]["content"][0]["text"]
    if text.lstrip().startswith("{"):
        text = json.loads(text)["category_attribute_instructions"]
    evidence = json.loads(text.split("\nFrozen evidence sets: ", 1)[1].split("\nFrozen attribute preparations: ", 1)[0])
    preparations = json.loads(text.rsplit("\nFrozen attribute preparations: ", 1)[1])
    return evidence, preparations


def test_only_evidence_is_deduplicated_and_original_input_stays_immutable(task):
    rows = [branch(index, category="Ring" if index % 2 == 0 else "Brooch", variant=f"variant-{index}") for index in range(10)]
    h = install(task, rows); before = deepcopy(h.body)
    first = h.service.submit("owner", h.envelope)
    assert first["status"] == "queued" and first["continue_after_no_final"] is True
    proof = first["derived_input"]
    assert proof == {"kind": KIND, "original_model": MODEL, "original_thinking_effort": EFFORT,
                     "model": MODEL, "thinking_effort": EFFORT, "preparation_count": 10,
                     "evidence_set_count": 1, "image_count": 2, "text_utf8_bytes": proof["text_utf8_bytes"],
                     "original_input_hash": TextTaskService._submission_identity("owner", before)[1]}
    saved = h.service.store.load_input(row(h.service, "successor")["_input_ref"])
    evidence, preparations = compacted(saved)
    assert len(evidence) == 1 and [p["preparation_index"] for p in preparations] == list(range(10))
    assert [p["category"] for p in preparations] == [item["category"] for item in rows]
    assert all(p["evidence_set_index"] == 0 for p in preparations)
    assert saved["messages"][0]["content"][1:] == before["messages"][0]["content"][1:]
    assert h.body == before and row(h.service) == h.old


def test_pure_message_compactor_has_no_identity_or_input_side_effect(task):
    h = install(task); before = deepcopy(h.body)
    messages, proof = compact_retained_messages(h.body)
    assert h.body == before and messages != before["messages"]
    assert proof["kind"] == KIND and proof["preparation_count"] == 2 and "original_input_hash" not in proof
    assert all(key not in messages[0] for key in ("client_request_id", "supersedes_request_id", "derived_input"))


def test_json_envelope_compacts_only_the_known_instruction_field(task):
    h = install(task); before = deepcopy(h.body)
    messages, proof = compact_retained_messages(h.body)
    original, changed = json.loads(before["messages"][0]["content"][0]["text"]), json.loads(messages[0]["content"][0]["text"])
    assert {key: value for key, value in changed.items() if key != "category_attribute_instructions"} == {
        key: value for key, value in original.items() if key != "category_attribute_instructions"}
    assert changed["category_attribute_instructions"].startswith(LEGACY_SUFFIX)
    assert proof["preparation_count"] == 2 and before == h.body


def test_plain_legacy_form_remains_supported(task):
    h = install(task)
    body = deepcopy(h.body)
    body["messages"][0]["content"][0]["text"] = legacy_prompt([branch(0)], json_envelope=False)
    messages, proof = compact_retained_messages(body)
    assert messages[0]["content"][0]["text"].startswith("Keep the original copy text")
    assert proof["preparation_count"] == 1


def test_new_json_contract_passes_through_recognizer_without_compaction(task):
    h = install(task); body = deepcopy(h.body)
    payload = json.loads(body["messages"][0]["content"][0]["text"])
    payload["category_attribute_instructions"] = LEGACY_SUFFIX + LEGACY_BOUNDARY + compact_json({
        "preparations": [], "branch_sets": [], "evidence_sets": []})
    body["messages"][0]["content"][0]["text"] = compact_json(payload)
    assert is_legacy_combined_copy_input(body) is False
    assert recognize_compact_retained_messages(body) is None
    with pytest.raises(ConversationBindingError): compact_retained_messages(body)


@pytest.mark.parametrize("case", WBOZ_CURRENT_CASES + WBOZ_STRUCTURED_CASES, ids=lambda case: case["name"])
def test_current_json_contract_derives_losslessly_for_wboz_producer_fixture(task, case):
    h = install(task); body = deepcopy(h.body)
    body.update(deepcopy(case["input"]))
    before = deepcopy(body)

    assert is_legacy_combined_copy_input(body) is False
    assert is_current_combined_copy_input(body) is True
    messages, proof = compact_retained_messages(body)
    saved, derived = derive_combined_copy_input(body, "successor", "old", {"kind": KIND})

    assert messages == before["messages"]
    assert saved["messages"] == before["messages"]
    assert proof == derived
    assert proof["preparation_count"] == len(case["preparations"])
    assert proof["evidence_set_count"] == case["evidenceSetCount"]
    assert saved["continue_after_no_final"] is True
    assert before == body


@pytest.mark.parametrize("change", ["branch_reference", "reply_reference", "empty_evidence", "unused_evidence",
                                      "empty_payload_branch", "duplicate_key", "changed_prompt"])
def test_current_json_contract_rejects_corrupt_references_before_derivation(task, change):
    h = install(task); body = deepcopy(h.body)
    fixture = WBOZ_CURRENT_CASES[-1] if change == "empty_payload_branch" else WBOZ_CURRENT_CASES[0]
    payload = json.loads(fixture["input"]["messages"][0]["content"][0]["text"])
    frozen = json.loads(payload["category_attribute_instructions"].split("\nFrozen attribute preparations: ", 1)[1])
    if change == "branch_reference":
        frozen["preparations"][0]["branch_set_index"] = 9
    elif change == "reply_reference":
        frozen["branch_sets"][0]["reply_preparation_index"] = 1
    elif change == "empty_evidence":
        frozen["branch_sets"][1]["evidence_set_index"] = 0
    elif change == "unused_evidence":
        frozen["evidence_sets"].append(deepcopy(frozen["evidence_sets"][0]))
    elif change == "empty_payload_branch":
        frozen["branch_sets"].append({"target_platform": "ozon", "source_variant_id": None, "schema": []})
    elif change == "duplicate_key":
        encoded = compact_json(frozen).replace('"branch_set_index":0', '"branch_set_index":0,"branch_set_index":0', 1)
        payload["category_attribute_instructions"] = CURRENT_SUFFIX + "\nFrozen attribute preparations: " + encoded
        body["messages"][0]["content"][0]["text"] = compact_json(payload)
    else:
        payload["category_attribute_instructions"] = payload["category_attribute_instructions"].replace(
            "Return no row", "Also summarize. Return no row", 1)
        body["messages"][0]["content"][0]["text"] = compact_json(payload)
    if change not in {"duplicate_key", "changed_prompt"}:
        payload["category_attribute_instructions"] = CURRENT_SUFFIX + "\nFrozen attribute preparations: " + compact_json(frozen)
        body["messages"][0]["content"][0]["text"] = compact_json(payload)
    assert is_current_combined_copy_input(body) is True
    with pytest.raises(ConversationBindingError):
        compact_retained_messages(body)


def test_our_compact_output_passes_through_lightweight_recognizer(task):
    h = install(task)
    messages, _ = compact_retained_messages(h.body)
    body = {**h.body, "messages": messages}
    assert is_legacy_combined_copy_input(body) is False
    assert recognize_compact_retained_messages(body) is None


def test_lightweight_recognizer_never_parses_ordinary_json_but_marks_malformed_legacy(task):
    h = install(task); body = deepcopy(h.body)
    body["messages"][0]["content"][0]["text"] = '{"ordinary":1,"ordinary":2}'
    assert is_legacy_combined_copy_input(body) is False
    malformed = deepcopy(h.body)
    text = malformed["messages"][0]["content"][0]["text"]
    malformed["messages"][0]["content"][0]["text"] = ('{"category_attribute_instructions":"x",'
        '"category_attribute_instructions":"' + text.replace('"', '\\"') + '"}')
    assert is_legacy_combined_copy_input(malformed) is True
    with pytest.raises(ConversationBindingError): compact_retained_messages(malformed)


def test_different_evidence_never_merges_or_reindexes_preparations(task):
    second = branch(1, evidence={"source_attributes": [{"source_attribute_index": 0, "name": "color", "values": ["blue"]}],
                                 "source_images": [{"image_ref": "source-b", "url": "asset://b"}],
                                 "source_title": "Blue brooch", "source_description": "Other fixture"})
    h = install(task, [branch(0), second])
    saved, proof = derive_combined_copy_input(h.body, "next", "old", {"kind": KIND})
    evidence, preparations = compacted(saved)
    assert proof["preparation_count"] == proof["evidence_set_count"] == 2
    assert preparations == [{"preparation_index": 0, "target_platform": "ozon", "schema": h.body["messages"][0]["content"] and branch(0)["schema"],
                             "category": branch(0)["category"], "evidence_set_index": 0},
                            {"preparation_index": 1, "target_platform": "ozon", "schema": second["schema"],
                             "category": second["category"], "evidence_set_index": 1}]
    assert evidence[0]["source_attributes"] != evidence[1]["source_attributes"]


def test_compact_prompt_retains_legacy_omission_and_evidence_rules_verbatim(task):
    h = install(task)
    messages, _ = compact_retained_messages(h.body)
    text = messages[0]["content"][0]["text"]
    if text.lstrip().startswith("{"):
        text = json.loads(text)["category_attribute_instructions"]
    assert LEGACY_SUFFIX in text
    assert "Omit nonmatching alternative branches." in text
    assert "translate a nonnumeric dictionary label only to an offered valid option" in text
    assert "A derived basis must cite actual source_title/source_description/image_refs" in text
    assert "Color alone, a generic category, or uncertainty does not justify an audience." in text
    assert "return one category_attributes row for every original preparation" not in text
    assert "evidence_set_index" in text and "lossless storage only" in text


@pytest.mark.parametrize("change", ["model", "effort", "suffix", "extra_row_key", "bad_index", "bad_content", "duplicate_json", "duplicate_envelope_key"])
def test_unrecognized_or_malformed_legacy_input_rejects_closed(task, change):
    h = install(task); body = deepcopy(h.body); text = body["messages"][0]["content"][0]["text"]
    if change == "model": body["model"] = "gpt-5-6-instant"
    elif change == "effort": body["thinking_effort"] = "standard"
    elif change == "suffix": body["messages"][0]["content"][0]["text"] = text.replace("Fill only", "Also summarize and Fill only", 1)
    elif change == "extra_row_key":
        payload = json.loads(text); ci = payload["category_attribute_instructions"]
        payload["category_attribute_instructions"] = ci[:-1] + ',"new_instruction":true}]'
        body["messages"][0]["content"][0]["text"] = compact_json(payload)
    elif change == "bad_index":
        payload = json.loads(text)
        payload["category_attribute_instructions"] = payload["category_attribute_instructions"].replace(
            '"preparation_index":1', '"preparation_index":4')
        body["messages"][0]["content"][0]["text"] = compact_json(payload)
    elif change == "bad_content": body["messages"][0]["content"].append({"type": "audio", "audio": "x"})
    elif change == "duplicate_json":
        payload = json.loads(text); ci = payload["category_attribute_instructions"]
        payload["category_attribute_instructions"] = ci.replace('"preparation_index":0', '"preparation_index":0,"preparation_index":0', 1)
        body["messages"][0]["content"][0]["text"] = compact_json(payload)
    else:
        body["messages"][0]["content"][0]["text"] = ('{"category_attribute_instructions":"x",'
            '"category_attribute_instructions":"' + text.replace('"', '\\"') + '"}')
    with pytest.raises(ConversationBindingError):
        derive_combined_copy_input(body, "successor", "old", {"kind": KIND})


@pytest.mark.parametrize("extra", [{"messages": []}, {"model": MODEL}, {"thinking_effort": "high"},
    {"continue_after_no_final": False}, {"derived_input": {"kind": KIND, "prompt": "changed"}},
    {"derived_input": {"kind": "unknown"}}])
def test_http_envelope_is_seven_identities_plus_fixed_kind_and_true(task, extra):
    from api.ai import ConversationBindingTextRequest
    h = install(task); body = {**h.envelope, **extra}
    with pytest.raises(ValidationError): ConversationBindingTextRequest.model_validate(body)
    with pytest.raises(ConversationBindingError): h.service.submit("owner", body)
    assert row(h.service, "successor") is None


def test_text_endpoint_accepts_only_the_compact_reference_and_returns_proof(task, monkeypatch):
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
        assert client.post(url, json={**h.envelope, "messages": []}, headers=headers).status_code == 422
        first = client.post(url, json=h.envelope, headers=headers)
        assert first.status_code == 200, first.text
        proof = first.json()["derived_input"]
        assert first.json()["continue_after_no_final"] is True and proof["kind"] == KIND
        assert proof["original_model"] == proof["model"] == MODEL and proof["thinking_effort"] == EFFORT
        assert client.post(url, json=h.envelope, headers=headers).json() == first.json()
        assert review.await_count == 1 and "source-a" in review.await_args.args[1]
        assert "Synthetic fixture" not in first.text and "_input_ref" not in first.text


def test_restart_same_child_and_send_guards_keep_original_no_final_qualification(task):
    h = install(task)
    first = h.service.submit("owner", h.envelope)
    _, store, admission = build(h.root, h.clock)
    service = TextTaskService(store.path, runner=h.service.runner, clock=h.clock, admission=admission)
    admission.register("text", lambda ctx, body: service._run(ctx.owner, ctx.request_id, body))
    assert service.submit("owner", h.envelope) == first
    update(service, status="succeeded", content="late original")
    admission.execute(admission.claim_next())
    assert not h.sent and row(service, "successor")["upstream_outcome"] == "not_sent"


def test_generic_incomplete_is_not_no_final_eligibility(task):
    h = install(task)
    update(h.service, recovery_reason="REQUEST_RESULT_INCOMPLETE")
    with pytest.raises(ConversationBindingError) as exc:
        h.service.submit("owner", h.envelope)
    assert exc.value.code == "CHAT_SUPERSEDE_INVALID" and row(h.service, "successor") is None


def test_bound_runner_preserves_same_account_and_chat_with_fresh_send_guard(task):
    h = install(task)
    observed = []
    def runner(body, on_cursor):
        current_request.get().before_send()
        observed.append(body)
        return {"content": "synthetic compact answer", "parent_message_id": "new-answer"}
    h.service.runner = runner
    accepted = h.service.submit("owner", h.envelope)
    proof = accepted["derived_input"]
    h.admission.execute(h.admission.claim_next())
    assert len(observed) == 1 and observed[0]["provider_binding_id"] == h.body["provider_binding_id"]
    assert observed[0]["conversation_id"] == h.body["conversation_id"] and observed[0]["model"] == MODEL
    finished = h.service.submit("owner", h.envelope)
    assert finished["status"] == "succeeded" and finished["derived_input"] == proof


@pytest.mark.parametrize("change", ["branch_reference", "reply_reference", "empty_evidence", "unused_evidence",
                                      "empty_payload_branch", "duplicate_key", "changed_prompt", "extra_field"])
def test_structured_contract_keeps_strict_reference_and_instruction_validation(task, change):
    h = install(task); body = deepcopy(h.body)
    fixture = WBOZ_STRUCTURED_CASES[-1] if change == "empty_payload_branch" else WBOZ_STRUCTURED_CASES[0]
    payload = json.loads(fixture["input"]["messages"][0]["content"][0]["text"])
    frozen = payload["category_attribute_instructions"]
    if change == "branch_reference":
        frozen["preparations"][0]["branch_set_index"] = 9
    elif change == "reply_reference":
        frozen["branch_sets"][0]["reply_preparation_index"] = 1
    elif change == "empty_evidence":
        frozen["branch_sets"][1]["evidence_set_index"] = 0
    elif change == "unused_evidence":
        frozen["evidence_sets"].append(deepcopy(frozen["evidence_sets"][0]))
    elif change == "empty_payload_branch":
        frozen["branch_sets"].append({"target_platform": "ozon", "source_variant_id": None, "schema": []})
    elif change == "changed_prompt":
        frozen["instructions"][0] += " Invent missing facts."
    elif change == "extra_field":
        frozen["override"] = True
    encoded = compact_json(payload)
    if change == "duplicate_key":
        encoded = encoded.replace('"branch_set_index":0', '"branch_set_index":0,"branch_set_index":0', 1)
    body["messages"][0]["content"][0]["text"] = encoded
    assert is_current_combined_copy_input(body) is True
    with pytest.raises(ConversationBindingError):
        compact_retained_messages(body)
