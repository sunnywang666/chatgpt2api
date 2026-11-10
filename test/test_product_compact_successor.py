"""Saved compact results close only their exact product and ended original turn."""
from copy import deepcopy
import json

import pytest

from services.conversation_binding_service import ConversationBindingError, ConversationBindingService
from services.advanced_session_successor import source_eligible
from services.product_work import ProductWorkService
from services.request_context import current_request
from services.text_task_service import TextTaskService
from services.work_lifecycle import WorkLifecycleError
from test.test_combined_copy_derivation import WBOZ_CURRENT_CASES, branch, legacy_prompt
from test.test_incomplete_compact_successor import install_incomplete
from test.test_no_final_successor import (product_no_final, product_lifecycle,
    complete_product_no_final_successor)
from test.test_explicit_unknown_successor import row, update


def compact(h, incomplete=False, *, prompt=None, input_body=None, workbench=False):
    prompt = legacy_prompt([branch(0), branch(1)]) if prompt is None else prompt
    h.body = h.service.store.load_input(row(h.service)["_input_ref"])
    if input_body is not None:
        h.body.update(deepcopy(input_body))
    else:
        h.body.update(model="gpt-5-6-thinking", thinking_effort="high", messages=[{
            "role": "user", "content": [{"type": "text", "text": prompt},
            {"type": "text", "text": '{"image_ref":"source-a"}'},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDRA=="}}]}])
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        original.update(_input_ref=h.service.store.save_input(h.body), model=h.body["model"],
                        thinking_effort=h.body["thinking_effort"])
        if incomplete:
            original.pop("_attempt_finished_at", None)
            original.pop("_attempt_reason", None)
        h.service.store.write_receipt(db, "text", "owner", "old", original)
        db.execute("UPDATE requests SET request_hash=? WHERE owner='owner' AND id='old'",
                   (TextTaskService._submission_identity("owner", h.body)[1],))
    if incomplete:
        install_incomplete(h, prompt=prompt, input_body=input_body)
    else:
        h.envelope = {k: v for k, v in h.body.items() if k in TextTaskService.SUPERSEDE_FIELDS}
        h.envelope.update(client_request_id="successor", supersedes_request_id="old",
                          continue_after_no_final=True, derived_input={"kind": "combined_copy_compact_v1"})
    if workbench:
        from services.product_work import _key
        from test.test_workbench_derived_input import payload
        with h.service.store.transaction() as db:
            original = h.service.store.read_receipt(db, "text", "owner", "old")
            old_key = original["_product_work_key"]
            parent = h.service.store.runtime(db, old_key)
            new_key = _key("internal:listing", "owner", "fixture-product")
            parent.update(key=new_key, source="internal:listing")
            h.service.store.set_runtime(db, new_key, parent)
            db.execute("DELETE FROM task_runtime WHERE name=?", (old_key,))
            original.update(_source="internal:listing", _product_work_key=new_key)
            original.pop("_scheduling_owner", None)
            h.service.store.write_receipt(db, "text", "owner", "old", original)
        request = payload(h.service, original, request_id="successor", mode="incomplete")
        request["derivation_kind"] = "source_category_review"
        h.derived_request = request
        assert h.service.submit_workbench_derived_input("owner", "old", request,
                                                        source="internal:listing")["status"] == "queued"
        context = h.admission.claim_next()
        assert context and context.request_id == "successor"
        h.admission.execute(context)
        h.admission.update_claim(context, _claim_id=None, _claim_until=0, _turn_reserved=False)
        return row(h.service), row(h.service, "successor")
    return complete_product_no_final_successor(h)


def ended_original(h, monkeypatch):
    document = deepcopy(h.document)
    document["mapping"]["active"]["message"].update(status="finished_successfully", end_turn=True,
        content={"content_type": "text", "parts": [""]})
    monkeypatch.setattr("services.conversation_binding_service.time.time", h.clock)
    h.service.recovery_reader = lambda receipt: ConversationBindingService._read_text_request_result(
        None, receipt, document=document)
    h.service.recover("owner", "old")
    original = row(h.service)
    assert original.get("_upstream_terminal") is True, {k: original.get(k) for k in ("status", "recovery_reason", "recovery_error_code", "recovery_next_at", "recovery_last_read_error", "_attempt_finished_at")}
    assert original["recovery_reason"] == "REQUEST_RESULT_TERMINAL_EMPTY"
    assert original["upstream_outcome"] == "unknown" and "content" not in original
    return original


def advanced_to_own_successor(h, monkeypatch, *, workbench=False):
    """Persist the original reader's exact, redacted local-branch proof."""
    original, _successor = compact(h, True, workbench=workbench)
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        successor = h.service.store.read_receipt(db, "text", "owner", "successor")
        # The fixture runner normally returns only the final cursor. Model the
        # bound send/read path's persisted child-user parent before reading it.
        successor["request_parent_message_id"] = "active"
        h.service.store.write_receipt(db, "text", "owner", "successor", successor)
    document = deepcopy(h.document)
    active = document["mapping"]["active"]
    active["children"] = [successor["request_message_id"]]
    document["mapping"][successor["request_message_id"]] = {
        "parent": "active", "children": [successor["parent_message_id"]], "message": {
            "id": successor["request_message_id"], "author": {"role": "user"},
            "status": "finished_successfully", "end_turn": True,
            "content": {"content_type": "text", "parts": ["compact child"]},
        },
    }
    document["mapping"][successor["parent_message_id"]] = {
        "parent": successor["request_message_id"], "children": [], "message": {
            "id": successor["parent_message_id"], "author": {"role": "assistant"},
            "status": "finished_successfully", "end_turn": True, "channel": "final",
            "content": {"content_type": "text", "parts": [successor["content"]]},
        },
    }
    document["current_node"] = successor["parent_message_id"]
    monkeypatch.setattr("services.conversation_binding_service.time.time", h.clock)
    child_read = ConversationBindingService._read_text_request_result(None, successor, document=document)
    assert child_read["status"] == "succeeded" and child_read["content"] == successor["content"]
    assert child_read["_turn_end_evidence"]["final_message_id"] == successor["parent_message_id"]
    with h.service.store.transaction() as db:
        successor = h.service.store.read_receipt(db, "text", "owner", "successor")
        successor["_turn_end_evidence"] = child_read["_turn_end_evidence"]
        h.service.store.write_receipt(db, "text", "owner", "successor", successor)
    h.service.recovery_reader = lambda receipt: ConversationBindingService._read_text_request_result(
        None, receipt, document=document)
    h.service.recover("owner", "old")
    original = row(h.service)
    assert original["upstream_unfinished"] is False
    assert original["status"] == "unknown"
    assert original["recovery_reason"] == "REQUEST_CONVERSATION_ADVANCED"
    assert original["_completed_successor_evidence"] == {
        "conversation_id": original["conversation_id"],
        "request_message_id": original["request_message_id"],
        "successor_request_message_id": successor["request_message_id"],
        "successor_request_parent_message_id": "active",
        "successor_final_message_id": successor["parent_message_id"],
        "observed_at": h.clock(),
    }
    if workbench:
        # The writer's persisted output, not a fixture-side repair, is the
        # strict eligibility input consumed by the isolated successor path.
        assert source_eligible(original, h.clock()) is True
    return original, row(h.service, "successor")


def test_invalid_advanced_successor_evidence_does_not_clear_unfinished(product_no_final):
    h = product_no_final
    compact(h, True, workbench=True)
    original = row(h.service)
    assert original.get("upstream_unfinished") is None

    # This resembles an advanced reader response, but mismatches the original
    # submitted turn.  `_safe_recovery_result` must reject it before the
    # writer is allowed to persist either the proof or the explicit boundary.
    h.service.recovery_reader = lambda receipt: {
        "status": "unknown",
        "binding_status": "unknown",
        "recovery_reason": "REQUEST_CONVERSATION_ADVANCED",
        "conversation_id": receipt["conversation_id"],
        "provider_binding_id": receipt["provider_binding_id"],
        "provider_account_identity": receipt["provider_account_identity"],
        "client_conversation_id": receipt["client_conversation_id"],
        "_completed_successor_evidence": {
            "conversation_id": receipt["conversation_id"],
            "request_message_id": "wrong-original-request",
            "successor_request_message_id": "external-user",
            "successor_request_parent_message_id": "external-parent",
            "successor_final_message_id": "external-final",
            "observed_at": h.clock(),
        },
    }
    h.service.recover("owner", "old")

    saved = row(h.service)
    assert saved.get("upstream_unfinished") is None
    assert "_completed_successor_evidence" not in saved
    assert saved["recovery_error_code"] == "RECOVERY_INVALID_RESULT"


def add_finished_gallery(h, request_id="gallery", *, recovered=False):
    """Attach an already-finished earlier product step on the same Chat."""
    original = row(h.service)
    body = h.service.store.load_input(original["_input_ref"])
    body.update(client_request_id=request_id, product_work={"product_ref": "fixture-product", "step_ref": request_id})
    h.service.submit("owner", body, source=original["_source"])
    with h.service.store.transaction() as db:
        gallery = h.service.store.read_receipt(db, "text", "owner", request_id)
        old = h.service.store.read_receipt(db, "text", "owner", "old")
        successor = h.service.store.read_receipt(db, "text", "owner", "successor")
        gallery.update(status="succeeded", upstream_outcome="completed", content="saved gallery review",
                       _upstream_terminal=not recovered, _executing=False, _turn_reserved=False,
                       _claim_id=None, _claim_until=None, _sequence=4746,
                       **({"_turn_end_evidence": {"conversation_id": gallery["conversation_id"],
                                                   "request_message_id": gallery["request_message_id"],
                                                   "final_message_id": gallery["parent_message_id"],
                                                   "observed_at": h.clock()}} if recovered else {}))
        old["_sequence"] = 5945
        successor["_sequence"] = 7122
        h.service.store.write_receipt(db, "text", "owner", request_id, gallery)
        h.service.store.write_receipt(db, "text", "owner", "old", old)
        h.service.store.write_receipt(db, "text", "owner", "successor", successor)
    return row(h.service, request_id)


def add_later_product_member(h, request_id="later"):
    """Attach a deliberately later same-conversation member without supersession."""
    original = row(h.service)
    body = h.service.store.load_input(original["_input_ref"])
    body.update(client_request_id=request_id, product_work={"product_ref": "fixture-product", "step_ref": request_id})
    h.service.submit("owner", body, source="internal:content")
    with h.service.store.transaction() as db:
        later = h.service.store.read_receipt(db, "text", "owner", request_id)
        successor = h.service.store.read_receipt(db, "text", "owner", "successor")
        later.update(status="succeeded", upstream_outcome="completed", content="later result",
                     _upstream_terminal=True, _executing=False, _turn_reserved=False,
                     _claim_id=None, _claim_until=None, _sequence=successor["_sequence"] + 1)
        h.service.store.write_receipt(db, "text", "owner", request_id, later)
    return row(h.service, request_id)


def add_completed_tail(h, request_id="copy-1", *, previous_id="successor", before_send=None, expect_sent=True, before_claim=None, expect_claim=True, lose_response=False, after_send=None):
    """Use the normal registered submit path after the owned derived child."""
    successor = row(h.service, previous_id)
    body = h.service.store.load_input(row(h.service)['_input_ref'])
    body.update(client_request_id=request_id, parent_message_id=successor['parent_message_id'],
                messages=[{'role': 'user', 'content': 'normal copy after category'}],
                product_work={**successor['_product_work'], 'step_ref': request_id})

    def runner(submitted, on_cursor):
        if before_send:
            before_send()
        current_request.get().before_send()
        if after_send:
            after_send()
        h.sent.append(deepcopy(submitted))
        if lose_response:
            raise ConversationBindingError('fixture response lost after send', code='CONVERSATION_OUTCOME_UNKNOWN')
        return {'content': 'normal copy result', 'parent_message_id': request_id + '-answer',
                'upstream_outcome': 'completed', 'upstream_unfinished': False, '_upstream_terminal': True}

    h.tail_body = deepcopy(body)
    h.service.runner = runner
    assert h.service.submit('owner', body, source=row(h.service)['_source'])['status'] == 'queued'
    if before_claim:
        before_claim()
    context = h.admission.claim_next()
    if not expect_claim:
        assert context is None or context.request_id != request_id
        return row(h.service, request_id)
    snapshot = h.admission.resource_snapshot()
    assert context and context.request_id == request_id, {
        'chat_turn': snapshot['chat_turn'], 'products': snapshot['products'],
        'accounts': snapshot['accounts'],
    }
    h.admission.execute(context)
    tail = row(h.service, request_id)
    if lose_response:
        assert tail['status'] == 'unknown' and tail['_submission_started'] is True
        return tail
    if expect_sent:
        assert tail['status'] == 'succeeded', {key: tail.get(key) for key in ('status', 'error_code', 'waiting')}
    else:
        assert tail.get('_submission_started') is False
    if expect_sent:
        h.admission.update_claim(context, _claim_id=None, _claim_until=0, _turn_reserved=False)
    return row(h.service, request_id)


def advanced_document_with_tail(h, successor, *tails):
    document = deepcopy(h.document)
    previous = "active"
    for turn in (successor, *tails):
        user, final = turn["request_message_id"], turn["parent_message_id"]
        document["mapping"][previous]["children"] = [user]
        document["mapping"][user] = {
            "parent": previous, "children": [final], "message": {
                "id": user, "author": {"role": "user"}, "status": "finished_successfully",
                "end_turn": True, "content": {"content_type": "text", "parts": ["normal instruction"]}}}
        document["mapping"][final] = {
            "parent": user, "children": [], "message": {
                "id": final, "author": {"role": "assistant"}, "status": "finished_successfully",
                "end_turn": True, "channel": "final",
                "content": {"content_type": "text", "parts": [turn["content"]]}}}
        previous = final
    document["current_node"] = previous
    return document


def refresh_advanced_original(h, monkeypatch, document):
    h.clock.now = max(h.clock.now + 60, (row(h.service).get("recovery_next_at") or 0) + 1)
    monkeypatch.setattr('services.conversation_binding_service.time.time', h.clock)
    h.service.recovery_reader = lambda receipt: ConversationBindingService._read_text_request_result(
        None, receipt, document=document)
    h.service.recover('owner', 'old')
    return row(h.service)


def add_completed_image_tail(h, successor, monkeypatch):
    """Persist a bound generated image and reread its exact empty-final graph."""
    from services.image_task_service import _request_hash

    payload = {
        "prompt": "render the approved product image", "model": "gpt-image-2", "size": "1024x1024",
        "quality": "auto", "provider_binding_id": successor["provider_binding_id"],
        "provider_account_identity": successor["provider_account_identity"],
        "client_conversation_id": successor["client_conversation_id"],
        "conversation_id": successor["conversation_id"], "parent_message_id": successor["parent_message_id"],
        "retain_conversation": True, "_product_work": {**successor["_product_work"], "step_ref": "image-tail"},
    }
    with h.service.store.transaction() as db:
        parent = h.service.store.runtime(db, successor["_product_work_key"])
        image = {
            "id": "image-tail", "owner_id": "owner", "status": "success", "mode": "generate",
            "provider_binding_id": successor["provider_binding_id"],
            "provider_account_identity": successor["provider_account_identity"],
            "client_conversation_id": successor["client_conversation_id"],
            "conversation_id": successor["conversation_id"], "parent_message_id": "image-final",
            "request_message_id": "image-user", "binding_status": "bound", "request_hash": _request_hash("generate", payload),
            "_input_ref": h.service.store.save_input({"payload": payload, "identity": {
                "id": "owner", "role": "admin", "_trusted_source": successor["_source"]}, "mode": "generate"}),
            "_product_work_key": successor["_product_work_key"], "_product_work": payload["_product_work"],
            "_source": successor["_source"], "_scheduling_owner": successor["_source"],
            "_sequence": successor["_sequence"] + 1, "_submission_started": True,
            "upstream_outcome": "generated", "upstream_unfinished": False,
            "result_file_ids": ["image-file"], "result_sediment_ids": [], "data": [{"b64_json": "AA=="}],
            "_executing": False, "_turn_reserved": False, "_claim_id": None, "_claim_until": None,
        }
        parent["members"].append({"kind": "image", "owner": "owner", "request_id": "image-tail",
                                  "step_ref": "image-tail"})
        h.service.store.set_runtime(db, parent["key"], parent)
        h.service.store.write_receipt(db, "image", "owner", "image-tail", image)
    document = advanced_document_with_tail(h, successor)
    text_final = successor["parent_message_id"]
    document["mapping"][text_final]["children"] = ["image-user"]
    document["mapping"].update({
        "image-user": {"parent": text_final, "children": ["image-code"], "message": {
            "id": "image-user", "author": {"role": "user"}, "status": "finished_successfully", "end_turn": True,
            "content": {"content_type": "text", "parts": ["make the final image"]}}},
        "image-code": {"parent": "image-user", "children": ["image-tool"], "message": {
            "id": "image-code", "author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": False,
            "content": {"content_type": "code", "text": "generate"}}},
        "image-tool": {"parent": "image-code", "children": ["image-final"], "message": {
            "id": "image-tool", "author": {"role": "tool"}, "status": "finished_successfully",
            "content": {"content_type": "multimodal_text", "parts": [{
                "content_type": "image_asset_pointer", "asset_pointer": "file-service://image-file"}]}}},
        "image-final": {"parent": "image-tool", "children": [], "message": {
            "id": "image-final", "author": {"role": "assistant"}, "status": "finished_successfully",
            "end_turn": True, "channel": "final", "content": {"content_type": "text", "parts": [""]}}},
    })
    document["current_node"] = "image-final"
    original = refresh_advanced_original(h, monkeypatch, document)
    assert original["_completed_successor_evidence"]["tail_steps"] == [{
        "request_message_id": "image-user", "final_message_id": "image-final"}]
    return original, row(h.service, "image-tail")


def test_incomplete_advanced_successor_image_tail_closes_archives_and_restarts(product_no_final, monkeypatch):
    from services.image_task_service import ImageTaskService
    from services.work_lifecycle import WorkLifecycleService

    h = product_no_final
    images = ImageTaskService(h.root / "images.json", store=h.service.store, admission=h.admission)
    _original, successor = advanced_to_own_successor(h, monkeypatch, workbench=True)
    original, _image = add_completed_image_tail(h, successor, monkeypatch)
    product, lifecycle, calls, identity = product_lifecycle(h)
    identity["_trusted_source"] = original["_source"]

    def archive_image(owner, request_id, archived):
        calls.append((owner, request_id, archived))
        return {"request_id": request_id, "conversation_id": original["conversation_id"], "archived": archived}

    monkeypatch.setattr(images, "archive_product_work", archive_image)
    lifecycle = WorkLifecycleService(lifecycle.text, images, clock=h.clock)
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    with h.service.store.connect() as db:
        parent = h.service.store.runtime(db, original["_product_work_key"])
        child = h.service.store.runtime(db, parent["archive_keys"][0])
        members = product._members(db, parent)
        assert product.archive_turns_ready(db, child, [receipt for _, receipt in members])
        assert child["target"]["conversation_id"] == original["conversation_id"]
        assert child["archive"]["desired"] is True
    assert lifecycle.process_one()
    completed = product.get(identity, "fixture-product")
    assert completed["state"] == "completed"
    assert calls == [("owner", "image-tail", True)]
    restarted = ProductWorkService(lifecycle.text, lifecycle.images, clock=h.clock)
    assert restarted.resume(identity, "fixture-product", 1)["state"] == "active"
    with h.service.store.connect() as db:
        image_after = h.service.store.read_receipt(db, "image", "owner", "image-tail")
    assert row(h.service) == original and image_after["request_hash"]


@pytest.mark.parametrize("drift", ["wrong_parent", "wrong_result", "foreign", "active", "paused", "fork"])
def test_incomplete_advanced_successor_image_tail_rejects_drift(product_no_final, monkeypatch, drift):
    h = product_no_final
    _original, successor = advanced_to_own_successor(h, monkeypatch, workbench=True)
    original, _image = add_completed_image_tail(h, successor, monkeypatch)
    with h.service.store.transaction() as db:
        image = h.service.store.read_receipt(db, "image", "owner", "image-tail")
        if drift == "wrong_parent":
            image["parent_message_id"] = "wrong-final"
        elif drift == "wrong_result":
            image["data"] = []
        elif drift == "foreign":
            foreign = {**image, "owner_id": "foreign"}
            h.service.store.write_receipt(db, "image", "foreign", "foreign-image", foreign)
        elif drift == "active":
            image["_executing"] = True
        elif drift == "paused":
            image["_recovery_paused"] = True
        else:
            original["_completed_successor_evidence"]["tail_steps"].append({
                "request_message_id": "fork-user", "final_message_id": "fork-final"})
            h.service.store.write_receipt(db, "text", "owner", "old", original)
        h.service.store.write_receipt(db, "image", "owner", "image-tail", image)
    product, _, calls, identity = product_lifecycle(h)
    identity["_trusted_source"] = original["_source"]
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_"):
        product.close(identity, "fixture-product", results_saved=True)
    assert calls == []


@pytest.mark.parametrize("step_ref", ["gallery", "attributes"])
def test_incomplete_advanced_image_tail_accepts_recovered_completed_predecessor(
        product_no_final, monkeypatch, step_ref):
    h = product_no_final
    _original, successor = advanced_to_own_successor(h, monkeypatch, workbench=True)
    predecessor = add_finished_gallery(h, step_ref, recovered=True)
    original, _image = add_completed_image_tail(h, row(h.service, "successor"), monkeypatch)
    product, _, _, identity = product_lifecycle(h)
    identity["_trusted_source"] = original["_source"]

    assert predecessor.get("_upstream_terminal") is False
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    with h.service.store.connect() as db:
        parent = h.service.store.runtime(db, original["_product_work_key"])
        child = h.service.store.runtime(db, parent["archive_keys"][0])
        product.load_archive_target(db, child)


@pytest.mark.parametrize("proof_change", ["wrong_final", "wrong_request", "future_timestamp"])
def test_incomplete_advanced_image_tail_rejects_invalid_recovered_predecessor(
        product_no_final, monkeypatch, proof_change):
    h = product_no_final
    _original, successor = advanced_to_own_successor(h, monkeypatch, workbench=True)
    predecessor = add_finished_gallery(h, recovered=True)
    original, _image = add_completed_image_tail(h, row(h.service, "successor"), monkeypatch)
    with h.service.store.transaction() as db:
        gallery = h.service.store.read_receipt(db, "text", "owner", predecessor["request_id"])
        if proof_change == "wrong_final":
            gallery["_turn_end_evidence"]["final_message_id"] = "wrong-final"
        elif proof_change == "wrong_request":
            gallery["_turn_end_evidence"]["request_message_id"] = "wrong-request"
        else:
            gallery["_turn_end_evidence"]["observed_at"] = h.clock() + 1
        h.service.store.write_receipt(db, "text", "owner", predecessor["request_id"], gallery)
    product, _, calls, identity = product_lifecycle(h)
    identity["_trusted_source"] = original["_source"]
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_"):
        product.close(identity, "fixture-product", results_saved=True)
    assert calls == []


def test_incomplete_advanced_own_successor_closes_archives_and_resumes(product_no_final, monkeypatch):
    h = product_no_final
    original, successor = advanced_to_own_successor(h, monkeypatch)
    gallery = add_finished_gallery(h)
    original = row(h.service)
    successor = row(h.service, "successor")
    assert gallery["_sequence"] < original["_sequence"] < successor["_sequence"]
    product, lifecycle, calls, identity = product_lifecycle(h)

    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    assert lifecycle.process_one()
    assert product.get(identity, "fixture-product")["state"] == "completed"
    assert calls == [("owner", "successor", True)]
    # The branch proof only waives ProductWork close; it never rewrites the
    # original's UNKNOWN/ADVANCED lifecycle or turns it terminal.
    assert row(h.service) == original
    assert row(h.service)["recovery_reason"] == "REQUEST_CONVERSATION_ADVANCED"
    assert "_upstream_terminal" not in row(h.service)
    restarted = ProductWorkService(lifecycle.text, lifecycle.images, clock=h.clock)
    assert restarted.resume(identity, "fixture-product", 1)["state"] == "active"
    assert row(h.service, "successor")["_turn_end_evidence"] == successor["_turn_end_evidence"]


@pytest.mark.parametrize("product_no_final", [None, {"organization_id": "org", "subject_id": "member", "root_workflow_id": "root"}], indirect=True)
@pytest.mark.parametrize("workbench,tail_count", [(False, 1), (True, 1), (True, 3)])
def test_incomplete_advanced_successor_then_normal_tail_rechecks_and_archives_tail(product_no_final, monkeypatch, workbench, tail_count):
    h = product_no_final
    original, successor = advanced_to_own_successor(h, monkeypatch, workbench=workbench)
    retained = h.service.store.load_input(original["_input_ref"])
    assert not original.get("_attempt_finished_at")
    tails = []
    for index in range(tail_count):
        tail = add_completed_tail(h, f"normal-{index}",
                                  previous_id=tails[-1]["request_id"] if tails else "successor")
        assert tail["request_id"] != tail["request_message_id"]
        assert tail["_source"] == tail["_scheduling_owner"] == successor["_source"] == successor["_scheduling_owner"]
        tails.append(tail)
        document = advanced_document_with_tail(h, successor, *tails)
        original = refresh_advanced_original(h, monkeypatch, document)
        assert len(original["_completed_successor_evidence"]["tail_steps"]) == index + 1
        # A restart / lost caller response only reads this same saved request.
        h.service = TextTaskService(h.service.store.path, runner=h.service.runner,
                                    clock=h.clock, admission=h.admission)
        h.admission.register("text", lambda ctx, body: h.service._run(ctx.owner, ctx.request_id, body))
        assert h.service.submit("owner", h.tail_body, source=original["_source"])["status"] == "succeeded"
        assert h.service.read("owner", tail["request_id"])["status"] == "succeeded"
    assert len(h.sent) == tail_count + 1
    assert original["status"] == "unknown" and not original.get("_attempt_finished_at")
    assert original["recovery_reason"] == "REQUEST_CONVERSATION_ADVANCED"
    assert h.service.store.load_input(original["_input_ref"]) == retained
    product, lifecycle, calls, identity = product_lifecycle(h)
    identity["_trusted_source"] = original["_source"]
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    assert lifecycle.process_one()
    assert product.get(identity, "fixture-product")["state"] == "completed"
    assert calls == [("owner", tails[-1]["request_id"], True)]
    assert row(h.service) == original


@pytest.mark.parametrize("drift", [
    "no_proof", "wrong_child", "wrong_final", "input_hash", "original_hash", "product", "step",
    "sibling", "late_member", "foreign_same_conversation", "live_claim", "paused", "unfinished_gallery",
    "unfinished_boundary_missing", "unfinished_boundary_true",
])
def test_incomplete_advanced_close_rejects_unowned_or_changed_branch(product_no_final, monkeypatch, drift):
    h = product_no_final
    advanced_to_own_successor(h, monkeypatch)
    if drift == "unfinished_gallery":
        add_finished_gallery(h)
    elif drift == "late_member":
        add_later_product_member(h)
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        successor = h.service.store.read_receipt(db, "text", "owner", "successor")
        if drift == "no_proof":
            original.pop("_completed_successor_evidence")
        elif drift == "unfinished_boundary_missing":
            original.pop("upstream_unfinished")
        elif drift == "unfinished_boundary_true":
            original["upstream_unfinished"] = True
        elif drift == "wrong_child":
            original["_completed_successor_evidence"]["successor_request_message_id"] = "external-user"
        elif drift == "wrong_final":
            original["_completed_successor_evidence"]["successor_final_message_id"] = "external-final"
        elif drift in {"input_hash", "original_hash"}:
            db.execute("UPDATE requests SET request_hash='wrong' WHERE owner='owner' AND id=?",
                       ("old" if drift == "original_hash" else "successor",))
        elif drift == "product":
            successor["_product_work_key"] = "wrong-product"
        elif drift == "step":
            successor["_product_work"] = {**successor["_product_work"], "step_ref": "other-step"}
        elif drift == "sibling":
            duplicate = {**successor, "request_id": drift}
            duplicate["_supersedes_request_id"] = "old"
            db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("owner", drift, "different", json.dumps(duplicate)))
        elif drift == "foreign_same_conversation":
            foreign = {**successor, "request_id": "successor"}
            db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("other-owner", "successor", "different", json.dumps(foreign)))
        elif drift == "live_claim":
            successor.update(recovery_claim_id="live-reader", recovery_lease_until=h.clock()+60)
        elif drift == "unfinished_gallery":
            gallery = h.service.store.read_receipt(db, "text", "owner", "gallery")
            gallery.update(status="unknown", upstream_outcome="unknown", upstream_unfinished=True,
                           _upstream_terminal=False)
            h.service.store.write_receipt(db, "text", "owner", "gallery", gallery)
        else:
            original["_recovery_paused"] = True
        h.service.store.write_receipt(db, "text", "owner", "old", original)
        h.service.store.write_receipt(db, "text", "owner", "successor", successor)
    product, _, calls, identity = product_lifecycle(h)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_"):
        product.close(identity, "fixture-product", results_saved=True)
    assert calls == []


def test_incomplete_advanced_archive_rechecks_original_branch_proof(product_no_final, monkeypatch):
    h = product_no_final
    advanced_to_own_successor(h, monkeypatch)
    product, lifecycle, calls, identity = product_lifecycle(h)
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        original["_completed_successor_evidence"]["successor_final_message_id"] = "changed-final"
        h.service.store.write_receipt(db, "text", "owner", "old", original)
    assert not lifecycle.process_one()
    assert calls == []
    assert product.get(identity, "fixture-product")["state"] == "closing"


def test_incomplete_advanced_archive_rejects_later_same_product_member(product_no_final, monkeypatch):
    h = product_no_final
    original, _successor = advanced_to_own_successor(h, monkeypatch)
    product, lifecycle, calls, identity = product_lifecycle(h)
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    with h.service.store.transaction() as db:
        parent = h.service.store.runtime(db, original["_product_work_key"])
        successor = h.service.store.read_receipt(db, "text", "owner", "successor")
        later = {**successor, "request_id": "post-close-later", "_sequence": successor["_sequence"] + 1,
                 "_supersedes_request_id": None,
                 "_product_work": {"product_ref": "fixture-product", "step_ref": "post-close-later"}}
        parent["members"].append({"kind": "text", "owner": "owner", "request_id": "post-close-later",
                                  "step_ref": "post-close-later"})
        h.service.store.set_runtime(db, parent["key"], parent)
        db.execute("INSERT INTO requests VALUES(?,?,?,?)",
                   ("owner", "post-close-later", "different", json.dumps(later)))
    assert not lifecycle.process_one()
    assert calls == []
    assert product.get(identity, "fixture-product")["state"] == "closing"


@pytest.mark.parametrize("incomplete", [False, True])
@pytest.mark.parametrize("product_no_final", [None, {
    "organization_id": "company", "subject_id": "alice", "root_workflow_id": "root-a",
}], indirect=True)
def test_compact_save_close_readback_restart_resume(product_no_final, monkeypatch, incomplete):
    h = product_no_final
    original, successor = compact(h, incomplete)
    product, lifecycle, calls, identity = product_lifecycle(h)
    if incomplete:
        # A completed child does not authorize archive of the still-active old turn.
        with pytest.raises(WorkLifecycleError, match="TURN_UNFINISHED"):
            product.close(identity, "fixture-product", results_saved=True)
        assert calls == [] and row(h.service) == original
        original = ended_original(h, monkeypatch)
    with pytest.raises(WorkLifecycleError, match="RESULTS_SAVE_REQUIRED"):
        product.close(identity, "fixture-product", results_saved=False)
    closing = product.close(identity, "fixture-product", results_saved=True)
    assert closing["state"] == "closing"
    # Lost close response: exact repeat observes the same durable intent.
    assert product.close(identity, "fixture-product", results_saved=True) == closing
    assert lifecycle.process_one()
    assert product.get(identity, "fixture-product")["state"] == "completed"
    assert calls == [("owner", "successor", True)]
    restarted = ProductWorkService(lifecycle.text, lifecycle.images, clock=h.clock)
    assert restarted.resume(identity, "fixture-product", 1)["state"] == "active"
    assert restarted.resume(identity, "fixture-product", 1)["version"] == 2
    assert row(h.service) == original and len(h.sent) == 1
    assert row(h.service, "successor")["content"] == successor["content"]
    assert successor["_product_work"] == original["_product_work"]
    with h.service.store.connect() as db:
        parent = h.service.store.runtime(db, original["_product_work_key"])
    assert parent.get("fairness") == original["_product_work"].get("fairness")


@pytest.mark.parametrize("case", WBOZ_CURRENT_CASES, ids=lambda case: case["name"])
def test_current_compact_no_final_close_revalidation_keeps_frozen_input(product_no_final, monkeypatch, case):
    h = product_no_final
    original, successor = compact(h, input_body=case["input"])
    product, lifecycle, calls, identity = product_lifecycle(h)

    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    assert lifecycle.process_one()
    assert calls == [("owner", "successor", True)]
    assert h.service.store.load_input(row(h.service)["_input_ref"])["messages"] == h.body["messages"]
    assert h.service.store.load_input(row(h.service, "successor")["_input_ref"])["messages"] == h.body["messages"]
    assert row(h.service, "successor")["content"] == successor["content"]
    assert row(h.service) == original


@pytest.mark.parametrize("incomplete", [False, True])
@pytest.mark.parametrize("drift", ["input_hash", "original_hash", "product", "step", "claim", "second_child"])
def test_compact_close_rejects_identity_claim_and_child_drift(product_no_final, monkeypatch, incomplete, drift):
    h = product_no_final
    compact(h, incomplete)
    if incomplete:
        ended_original(h, monkeypatch)
    with h.service.store.transaction() as db:
        child = h.service.store.read_receipt(db, "text", "owner", "successor")
        if drift in {"input_hash", "original_hash"}:
            db.execute("UPDATE requests SET request_hash='wrong' WHERE owner='owner' AND id=?",
                       ("old" if drift == "original_hash" else "successor",))
        elif drift == "product":
            child["_product_work_key"] = "wrong-product"
        elif drift == "step":
            child["_product_work"] = {**child["_product_work"], "step_ref": "other-step"}
        elif drift == "claim":
            child.update(recovery_claim_id="new-live-reader", recovery_lease_until=h.clock()+60)
        else:
            duplicate = {**child, "request_id": "duplicate"}
            db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("owner", "duplicate", "different", json.dumps(duplicate)))
        h.service.store.write_receipt(db, "text", "owner", "successor", child)
    product, _, calls, identity = product_lifecycle(h)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_"):
        product.close(identity, "fixture-product", results_saved=True)
    assert calls == []


@pytest.mark.parametrize("drift", ["identity", "timestamp", "missing_proof", "live_claim", "active_original"])
def test_incomplete_child_success_never_replaces_original_end_proof(product_no_final, monkeypatch, drift):
    h = product_no_final
    compact(h, True)
    original = ended_original(h, monkeypatch)
    if drift == "identity":
        original["_turn_end_evidence"]["request_message_id"] = "other-original"
    elif drift == "timestamp":
        original["_turn_end_evidence"]["observed_at"] = h.clock()+1
    elif drift == "missing_proof":
        original.pop("_turn_end_evidence")
    elif drift == "live_claim":
        original.update(_claim_id="live", _claim_until=h.clock()+60)
    else:
        original.update(_executing=True)
    with h.service.store.transaction() as db:
        h.service.store.write_receipt(db, "text", "owner", "old", original)
    product, _, calls, identity = product_lifecycle(h)
    with pytest.raises(WorkLifecycleError, match="TURN_UNFINISHED"):
        product.close(identity, "fixture-product", results_saved=True)
    assert calls == []


def test_compact_archive_rechecks_original_after_close(product_no_final, monkeypatch):
    h = product_no_final
    compact(h, True)
    original = ended_original(h, monkeypatch)
    product, lifecycle, calls, identity = product_lifecycle(h)
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    update(h.service, _executing=True)
    assert not lifecycle.process_one()
    assert calls == []
    assert product.get(identity, "fixture-product")["state"] == "closing"
    assert row(h.service)["_input_ref"] == original["_input_ref"]


@pytest.mark.parametrize("legacy_outcome", ["missing", None])
@pytest.mark.parametrize("original_ended", [False, True])
def test_legacy_missing_outcome_needs_own_end_proof_before_compact_close(
        product_no_final, monkeypatch, legacy_outcome, original_ended):
    h = product_no_final
    compact(h, True)
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        if legacy_outcome == "missing":
            original.pop("upstream_outcome", None)
        else:
            original["upstream_outcome"] = None
        original.update(_claim_id="historical-sender", _claim_until=h.clock()-1,
                        _executing=False, _turn_reserved=False)
        h.service.store.write_receipt(db, "text", "owner", "old", original)
    product, lifecycle, calls, identity = product_lifecycle(h)
    with pytest.raises(WorkLifecycleError, match="TURN_UNFINISHED"):
        product.close(identity, "fixture-product", results_saved=True)
    assert calls == []
    if not original_ended:
        assert row(h.service) == original
        return
    ended = ended_original(h, monkeypatch)
    assert ended["status"] == "unknown" and "content" not in ended
    assert ended["_claim_id"] == original["_claim_id"]
    assert ended["_claim_until"] == original["_claim_until"]
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    assert lifecycle.process_one()
    assert product.get(identity, "fixture-product")["state"] == "completed"
    assert calls == [("owner", "successor", True)] and len(h.sent) == 1
    assert row(h.service) == ended


@pytest.mark.parametrize("drift", ["parent", "binding", "account", "source", "client", "physical",
    "step", "hash", "original_hash", "original_succeeded", "original_paused", "original_claim",
    "original_source", "child_source",
    "child_unknown", "child_empty", "admission_paused"])
def test_normal_tail_send_edge_rechecks_owned_chain(product_no_final, monkeypatch, drift):
    h = product_no_final
    advanced_to_own_successor(h, monkeypatch, workbench=True)
    sent = len(h.sent)
    def change():
        with h.service.store.transaction() as db:
            tail = h.service.store.read_receipt(db, "text", "owner", "copy-1")
            old = h.service.store.read_receipt(db, "text", "owner", "old")
            child = h.service.store.read_receipt(db, "text", "owner", "successor")
            if drift == "parent": tail["request_parent_message_id"] = "previous-answer"
            elif drift == "binding": tail["provider_binding_id"] = "other"
            elif drift == "account": tail["provider_account_identity"] = "other"
            elif drift == "source": tail["_source"] = "internal:content"
            elif drift == "client": tail["client_conversation_id"] = "other"
            elif drift == "physical": tail["conversation_id"] = "other"
            elif drift == "step": tail["_product_work"]["step_ref"] = "title"
            elif drift == "hash": db.execute("UPDATE requests SET request_hash='changed' WHERE id='copy-1'")
            elif drift == "original_hash": db.execute("UPDATE requests SET request_hash='changed' WHERE id='old'")
            elif drift == "original_succeeded": old.update(status="succeeded", content="late result")
            elif drift == "original_paused": old["_recovery_paused"] = True
            elif drift == "original_source": old["_source"] = "internal:other"
            elif drift == "child_source": child["_source"] = "internal:other"
            elif drift == "original_claim": old.update(_claim_id="live", _claim_until=h.clock()+60)
            elif drift == "child_unknown": child.update(status="unknown", upstream_outcome="unknown")
            elif drift == "child_empty": child["content"] = ""
            elif drift == "admission_paused":
                parent = h.service.store.runtime(db, old["_product_work_key"])
                parent["admission_control"]["state"] = "paused"
                h.service.store.set_runtime(db, parent["key"], parent)
            for key, value in (("old", old), ("successor", child), ("copy-1", tail)):
                h.service.store.write_receipt(db, "text", "owner", key, value)
    add_completed_tail(h, before_send=change, expect_sent=False)
    assert len(h.sent) == sent


@pytest.mark.parametrize("drift", ["hash", "parent", "binding", "source", "product", "step", "unknown",
    "claim", "paused", "branch", "foreign", "descendant", "proof_removed", "message_collision", "child_parent", "unsent_success"])
@pytest.mark.parametrize("after_close", [False, True])
def test_multi_tail_close_and_archive_reject_drift(product_no_final, monkeypatch, drift, after_close):
    h = product_no_final
    _, successor = advanced_to_own_successor(h, monkeypatch, workbench=True)
    tails = []
    for index in range(2):
        tails.append(add_completed_tail(h, f"copy-{index}", previous_id=tails[-1]["request_id"] if tails else "successor"))
        refresh_advanced_original(h, monkeypatch, advanced_document_with_tail(h, successor, *tails))
    product, lifecycle, calls, identity = product_lifecycle(h)
    identity["_trusted_source"] = "internal:listing"
    if after_close:
        assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    with h.service.store.transaction() as db:
        tail = h.service.store.read_receipt(db, "text", "owner", "copy-0")
        old = h.service.store.read_receipt(db, "text", "owner", "old")
        if drift == "hash": db.execute("UPDATE requests SET request_hash='changed' WHERE id='copy-0'")
        elif drift == "parent": tail["request_parent_message_id"] = "active"
        elif drift == "binding": tail["provider_binding_id"] = "other"
        elif drift == "source": tail["_source"] = "internal:content"
        elif drift == "product": tail["_product_work"]["product_ref"] = "other"
        elif drift == "step": tail["_product_work"]["step_ref"] = "title"
        elif drift == "unknown": tail.update(status="unknown", upstream_outcome="unknown")
        elif drift == "unsent_success": tail["_submission_started"] = False
        elif drift == "claim": tail.update(_claim_id="live", _executing=True, _claim_until=h.clock()+60)
        elif drift == "paused": tail["_recovery_paused"] = True
        elif drift == "proof_removed": old.pop("_completed_successor_evidence")
        elif drift == "message_collision": old["request_message_id"] = tail["request_message_id"]
        elif drift == "child_parent":
            child = h.service.store.read_receipt(db, "text", "owner", "successor")
            child["request_parent_message_id"] = "wrong-actual-parent"
            h.service.store.write_receipt(db, "text", "owner", "successor", child)
        else:
            foreign = deepcopy(tail)
            foreign.update(request_id="foreign", _sequence=tail["_sequence"]+10)
            if drift == "descendant": foreign["_supersedes_request_id"] = "copy-0"
            owner = "other-owner" if drift == "foreign" else "owner"
            db.execute("INSERT INTO requests(owner,id,request_hash,receipt) VALUES(?,?,?,?)",
                       (owner, "foreign", "foreign", json.dumps(foreign)))
        h.service.store.write_receipt(db, "text", "owner", "copy-0", tail)
        h.service.store.write_receipt(db, "text", "owner", "old", old)
    if after_close:
        lifecycle.process_one()
        assert product.get(identity, "fixture-product")["state"] != "completed"
    else:
        with pytest.raises(WorkLifecycleError):
            product.close(identity, "fixture-product", results_saved=True)
    assert calls == []


@pytest.mark.parametrize("drift", ["branch", "wrong_parent", "active", "empty"])
def test_fresh_original_read_replaces_and_invalidates_tail_proof(product_no_final, monkeypatch, drift):
    h = product_no_final
    _, successor = advanced_to_own_successor(h, monkeypatch, workbench=True)
    tail = add_completed_tail(h)
    document = advanced_document_with_tail(h, successor, tail)
    original = refresh_advanced_original(h, monkeypatch, document)
    assert original["_completed_successor_evidence"]["tail_steps"]
    final = tail["parent_message_id"]
    if drift == "branch":
        other = deepcopy(document["mapping"][final])
        other["message"]["id"] = "foreign"
        document["mapping"]["foreign"] = other
        document["mapping"][tail["request_message_id"]]["children"].append("foreign")
    elif drift == "wrong_parent": document["mapping"][tail["request_message_id"]]["parent"] = "active"
    elif drift == "active": document["mapping"][final]["message"]["status"] = "in_progress"
    else: document["mapping"][final]["message"]["content"]["parts"] = [""]
    original = refresh_advanced_original(h, monkeypatch, document)
    assert "_completed_successor_evidence" not in original
    assert original["status"] == "unknown" and not original.get("_attempt_finished_at")
    product, _, calls, identity = product_lifecycle(h)
    identity["_trusted_source"] = "internal:listing"
    with pytest.raises(WorkLifecycleError): product.close(identity, "fixture-product", results_saved=True)
    assert calls == []


@pytest.mark.parametrize("drift", ["no_proof", "wrong_parent", "child_queued", "child_unknown", "child_empty", "pause"])
def test_invalid_chain_cannot_claim_even_if_stalled_policy_releases_capacity(product_no_final, monkeypatch, drift):
    h = product_no_final
    advanced_to_own_successor(h, monkeypatch, workbench=True)
    sent = len(h.sent)
    # The new gate must not rely on the original consuming a capacity slot.
    monkeypatch.setattr(TextTaskService, "_qualified_stalled_incomplete_capacity_boundary",
                        staticmethod(lambda receipt, now: receipt.get("request_id") == "old"))
    def change():
        with h.service.store.transaction() as db:
            old = h.service.store.read_receipt(db, "text", "owner", "old")
            child = h.service.store.read_receipt(db, "text", "owner", "successor")
            tail = h.service.store.read_receipt(db, "text", "owner", "copy-1")
            if drift == "no_proof": old.pop("_completed_successor_evidence")
            elif drift == "wrong_parent": tail["request_parent_message_id"] = "active"
            elif drift == "child_queued": child["status"] = "queued"
            elif drift == "child_unknown": child.update(status="unknown", upstream_outcome="unknown")
            elif drift == "child_empty": child["content"] = ""
            elif drift == "pause": old["_recovery_paused"] = True
            for request_id, receipt in (("old", old), ("successor", child), ("copy-1", tail)):
                h.service.store.write_receipt(db, "text", "owner", request_id, receipt)
    add_completed_tail(h, before_claim=change, expect_claim=False)
    assert len(h.sent) == sent


def test_legacy_single_child_proof_requires_fresh_parent_read_before_normal_send(product_no_final, monkeypatch):
    h = product_no_final
    _, successor = advanced_to_own_successor(h, monkeypatch, workbench=True)
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        original["_completed_successor_evidence"].pop("successor_request_parent_message_id")
        h.service.store.write_receipt(db, "text", "owner", "old", original)
    add_completed_tail(h, expect_claim=False)
    assert len(h.sent) == 1
    refresh_advanced_original(h, monkeypatch, advanced_document_with_tail(h, successor))
    add_completed_tail(h)
    assert len(h.sent) == 2


def test_legacy_single_child_close_proof_remains_compatible(product_no_final, monkeypatch):
    h = product_no_final
    advanced_to_own_successor(h, monkeypatch)
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        original["_completed_successor_evidence"].pop("successor_request_parent_message_id")
        h.service.store.write_receipt(db, "text", "owner", "old", original)
    product, lifecycle, calls, identity = product_lifecycle(h)
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    assert lifecycle.process_one()
    assert product.get(identity, "fixture-product")["state"] == "completed"
    assert calls == [("owner", "successor", True)]


def test_unknown_tail_after_lost_response_recovers_exact_id_after_restart_without_resend(product_no_final, monkeypatch):
    h = product_no_final
    _, successor = advanced_to_own_successor(h, monkeypatch, workbench=True)
    tail = add_completed_tail(h, lose_response=True)
    assert h.admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    assert len(h.sent) == 2
    h.service = TextTaskService(h.service.store.path, runner=h.service.runner, clock=h.clock, admission=h.admission)
    h.admission.register("text", lambda ctx, body: h.service._run(ctx.owner, ctx.request_id, body))
    assert h.service.submit("owner", h.tail_body, source="internal:listing")["status"] == "unknown"
    assert h.admission.claim_next() is None and len(h.sent) == 2
    h.clock.now += 60
    recovery_occupancy = []
    def unavailable(receipt):
        recovery_occupancy.append((bool(receipt.get("recovery_claim_id")),
                                   h.admission.resource_snapshot()["chat_turn"]["inflight"]))
        raise TimeoutError("fixture exact result temporarily unavailable")
    h.service.recovery_reader = unavailable
    h.service.recover("owner", "copy-1")
    assert recovery_occupancy == [(True, 1)]
    assert row(h.service, "copy-1")["status"] == "unknown"
    assert h.admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    product, lifecycle, calls, identity = product_lifecycle(h)
    identity["_trusted_source"] = "internal:listing"
    with pytest.raises(WorkLifecycleError): product.close(identity, "fixture-product", results_saved=True)
    document = advanced_document_with_tail(h, successor, {
        **tail, "parent_message_id": "copy-1-answer", "content": "normal copy result"})
    h.clock.now = max(h.clock.now + 60, (row(h.service, "copy-1").get("recovery_next_at") or 0) + 1)
    h.service.recovery_reader = lambda receipt: ConversationBindingService._read_text_request_result(
        None, receipt, document=document)
    h.service.recover("owner", "copy-1")
    recovered = row(h.service, "copy-1")
    assert h.admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert recovered["status"] == "succeeded" and recovered["request_message_id"] == tail["request_message_id"]
    refresh_advanced_original(h, monkeypatch, document)
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    assert lifecycle.process_one()
    assert calls == [("owner", "copy-1", True)] and len(h.sent) == 2


def test_owned_chain_capacity_does_not_rebound_when_tail_starts_or_finishes(product_no_final, monkeypatch):
    h = product_no_final
    _, successor = advanced_to_own_successor(h, monkeypatch, workbench=True)
    observations = {}
    def capture(stage):
        observations[stage] = h.admission.resource_snapshot()["chat_turn"]["inflight"]
    capture("child_completed")
    tail = add_completed_tail(h, before_claim=lambda: capture("tail_queued"),
                             before_send=lambda: capture("tail_claimed"),
                             after_send=lambda: capture("tail_sent"))
    capture("tail_completed_before_original_read")
    refresh_advanced_original(h, monkeypatch, advanced_document_with_tail(h, successor, tail))
    capture("tail_completed_after_original_read")
    second = add_completed_tail(h, "copy-2", previous_id="copy-1", before_claim=lambda: capture("next_tail_queued"))
    refresh_advanced_original(h, monkeypatch, advanced_document_with_tail(h, successor, tail, second))
    product, lifecycle, _, identity = product_lifecycle(h)
    identity["_trusted_source"] = "internal:listing"
    product.close(identity, "fixture-product", results_saved=True)
    capture("product_closing")
    lifecycle.process_one()
    capture("product_archived")
    assert observations == {"child_completed": 0, "tail_queued": 0, "tail_claimed": 1,
                            "tail_sent": 1, "tail_completed_before_original_read": 0,
                            "tail_completed_after_original_read": 0, "next_tail_queued": 0,
                            "product_closing": 0, "product_archived": 0}


@pytest.mark.parametrize("extra", ["original_claim", "original_pause", "original_hash", "leaf_parent",
                                   "double_leaf", "foreign_unknown", "foreign_same_conversation"])
def test_capacity_transfer_preserves_active_claims_and_other_unknowns(product_no_final, monkeypatch, extra):
    h = product_no_final
    advanced_to_own_successor(h, monkeypatch, workbench=True)
    observations = []
    def during_send():
        observations.append(h.admission.resource_snapshot()["chat_turn"]["inflight"])
        with h.service.store.transaction() as db:
            original = h.service.store.read_receipt(db, "text", "owner", "old")
            leaf = h.service.store.read_receipt(db, "text", "owner", "copy-1")
            assert leaf["status"] == "running" and leaf["_claim_id"] and leaf["_executing"] is True
            if extra == "original_claim":
                original.update(_claim_id="live-owner", _claim_until=h.clock()+60)
                h.service.store.write_receipt(db, "text", "owner", "old", original)
            elif extra == "original_pause":
                original["_recovery_paused"] = True
                h.service.store.write_receipt(db, "text", "owner", "old", original)
            elif extra == "original_hash":
                db.execute("UPDATE requests SET request_hash='changed' WHERE id='old'")
            elif extra == "leaf_parent":
                leaf["request_parent_message_id"] = "active"
                h.service.store.write_receipt(db, "text", "owner", "copy-1", leaf)
            elif extra == "double_leaf":
                second = deepcopy(leaf)
                second.update(request_id="second", request_message_id="second-user", _sequence=100)
                second["_product_work"]["step_ref"] = "second"
                body = h.service.store.load_input(leaf["_input_ref"])
                body.update(client_request_id="second", product_work=second["_product_work"])
                second["_input_ref"] = h.service.store.save_input(body)
                db.execute("INSERT INTO requests(owner,id,request_hash,receipt) VALUES(?,?,?,?)",
                           ("owner", "second", TextTaskService._submission_identity("owner", body)[1], json.dumps(second)))
                parent = h.service.store.runtime(db, original["_product_work_key"])
                parent["members"].append({"kind": "text", "owner": "owner", "request_id": "second", "step_ref": "second"})
                h.service.store.set_runtime(db, parent["key"], parent)
            else:
                foreign = deepcopy(original)
                for key in ("_product_work_key", "_product_work", "_completed_successor_evidence"):
                    foreign.pop(key, None)
                foreign.update(request_id="foreign", request_message_id="foreign-user", _sequence=100)
                if extra == "foreign_unknown":
                    foreign.update(conversation_id="other-chat", client_conversation_id="other-session")
                db.execute("INSERT INTO requests(owner,id,request_hash,receipt) VALUES(?,?,?,?)",
                           ("other-owner", "foreign", "foreign", json.dumps(foreign)))
        observations.append(h.admission.resource_snapshot()["chat_turn"]["inflight"])
    add_completed_tail(h, after_send=during_send)
    assert observations == [1, 3 if extra in {"foreign_same_conversation", "double_leaf"} else 2]


def queued_image_after_completed_copy(h, monkeypatch):
    """0309 relation: UNKNOWN attrs -> verified replacement -> copy -> image."""
    from datetime import datetime, timezone
    from services.image_task_service import ImageTaskService

    advanced_to_own_successor(h, monkeypatch, workbench=True)
    # Match 0309's saved successful replacement; the copy changes the final cursor.
    update(h.service, 'successor', _upstream_terminal=True)
    copy = add_completed_tail(h)
    h.admission.settings = lambda: {'chat_account_concurrency': 6, 'image_account_concurrency': 4}
    h.admission.accounts.update_account('fixture', {
        'limits_progress': [{'feature_name': 'image_gen', 'remaining': 100}],
        'capacity_observed_at': datetime.now(timezone.utc).isoformat(),
    })
    images = ImageTaskService(h.root / 'image_tasks.json', admission=h.admission, store=h.service.store)
    binding = {key: copy[key] for key in ('provider_account_identity', 'provider_binding_id',
                                         'client_conversation_id', 'conversation_id', 'parent_message_id')}
    images.submit_generation({'id': 'owner', 'role': 'admin', '_trusted_source': 'internal:listing'},
        client_task_id='original-image', prompt='original main image instruction', size=None,
        model='gpt-image-2', **binding,
        product_work={**copy['_product_work'], 'step_ref': 'image'})
    context = h.admission.claim_next()
    with h.service.store.connect() as db:
        image = h.service.store.read_receipt(db, 'image', 'owner', 'original-image')
    assert context and context.request_id == 'original-image', json.dumps(image)
    return context


@pytest.mark.parametrize('mode', ['terminal', 'recovered', 'restart'])
def test_image_after_completed_copy_sends_once_preserving_original_unknown(product_no_final, monkeypatch, mode):
    from services.request_context import AdmissionLost
    h = product_no_final
    context = queued_image_after_completed_copy(h, monkeypatch)
    if mode == 'recovered':
        update(h.service, 'successor', _upstream_terminal=False)
    elif mode == 'restart':
        from test.test_pool_admission import build
        from services.pool_admission import ExecutionContext
        _, _, restarted = build(h.root, h.clock)
        restarted.settings = h.admission.settings
        context = ExecutionContext(restarted, context.kind, context.owner, context.request_id, context.claim)
    old = row(h.service)
    copy = row(h.service, 'copy-1')
    with h.service.store.connect() as db:
        before = h.service.store.read_receipt(db, 'image', 'owner', 'original-image')
    context.before_send_with_marker()
    with h.service.store.connect() as db:
        after = h.service.store.read_receipt(db, 'image', 'owner', 'original-image')
    assert after['_submission_started'] is True
    for key in ('id', 'parent_message_id', 'provider_account_identity', 'provider_binding_id',
                'conversation_id', '_input_ref', 'request_hash'):
        assert after[key] == before[key]
    assert after['parent_message_id'] == copy['parent_message_id']
    assert row(h.service) == old and old['status'] == 'unknown'
    with pytest.raises(AdmissionLost, match='already submitted'):
        context.before_send()


@pytest.mark.parametrize('drift', [
    'image-parent', 'image-work', 'image-input', 'image-owner', 'image-binding',
    'image-account', 'image-conversation', 'copy-parent', 'copy-work', 'copy-binding',
    'copy-account', 'copy-conversation', 'copy-unknown', 'copy-empty', 'copy-inflight',
    'copy-lease', 'copy-hold', 'copy-terminal', 'copy-input', 'copy-sequence',
    'child-missing', 'child-unknown', 'child-inflight', 'child-final', 'child-audit',
    'old-inflight', 'old-upstream-inflight', 'old-recovery-lease', 'old-hold', 'parent-paused', 'foreign-member',
    'duplicate-member', 'competing-copy',
])
def test_image_after_completed_copy_rejects_changed_chain(product_no_final, monkeypatch, drift):
    from services.request_context import AdmissionLost
    h = product_no_final
    context = queued_image_after_completed_copy(h, monkeypatch)
    store = h.service.store
    with store.transaction() as db:
        image = store.read_receipt(db, 'image', 'owner', 'original-image')
        copy = store.read_receipt(db, 'text', 'owner', 'copy-1')
        child = store.read_receipt(db, 'text', 'owner', 'successor')
        old = store.read_receipt(db, 'text', 'owner', 'old')
        parent = store.runtime(db, image['_product_work_key'])
        fields = {'parent': 'parent_message_id', 'work': '_product_work_key',
                  'binding': 'provider_binding_id', 'account': 'provider_account_identity',
                  'conversation': 'conversation_id'}
        target, change = drift.split('-', 1)
        if target == 'image':
            if change in fields:
                image[fields[change]] = 'wrong'
            elif change == 'input':
                image['request_hash'] = 'wrong'
            else:
                member = next(m for m in parent['members'] if m['kind'] == 'image')
                member['owner'] = 'other'
        elif target == 'copy':
            if change in fields:
                copy[fields[change]] = 'wrong'
            elif change == 'unknown':
                copy['status'] = 'unknown'
            elif change == 'empty':
                copy['content'] = ''
            elif change == 'inflight':
                copy['_executing'] = True
            elif change == 'lease':
                copy['recovery_claim_id'] = 'live'
                copy['recovery_lease_until'] = h.clock() + 60
            elif change == 'hold':
                copy['_send_hold'] = True
            elif change == 'terminal':
                copy['_upstream_terminal'] = False
                copy.pop('_turn_end_evidence', None)
            elif change == 'input':
                body = store.load_input(copy['_input_ref'])
                body['parent_message_id'] = 'different-branch'
                copy['_input_ref'] = store.save_input(body)
                db.execute('UPDATE requests SET request_hash=? WHERE owner=? AND id=?',
                           (TextTaskService._submission_identity('owner', body)[1], 'owner', 'copy-1'))
            else:
                copy['_sequence'] = image['_sequence']
        elif target == 'child':
            if change == 'unknown':
                child['status'] = 'unknown'
            elif change == 'inflight':
                child['_executing'] = True
            elif change == 'final':
                child['_turn_end_evidence']['final_message_id'] = 'wrong'
            elif change == 'audit':
                child['_workbench_derived_input']['child_input_hash'] = 'wrong'
        elif target == 'old':
            if change == 'inflight':
                old['_executing'] = True
            elif change == 'upstream-inflight':
                old['upstream_unfinished'] = True
            elif change == 'hold':
                old['_send_hold'] = True
            else:
                old['recovery_claim_id'] = 'live'
                old['recovery_lease_until'] = h.clock() + 60
        elif target == 'parent':
            parent['admission_control'] = {'state': 'paused', 'version': 1, 'slot_held': False}
        elif target == 'foreign':
            next(m for m in parent['members'] if m['request_id'] == 'copy-1')['owner'] = 'other'
        elif target == 'duplicate':
            parent['members'].append(deepcopy(next(m for m in parent['members'] if m['request_id'] == 'copy-1')))
        elif target == 'competing':
            extra = dict(copy, request_id='competing', _sequence=copy['_sequence'])
            store.write_receipt(db, 'text', 'owner', 'competing', extra)
            parent['members'].append({'kind': 'text', 'owner': 'owner', 'request_id': 'competing', 'step_ref': 'other'})
        for kind, rid, value in [('image', 'original-image', image), ('text', 'copy-1', copy),
                                 ('text', 'old', old), ('text', 'successor', child)]:
            store.write_receipt(db, kind, 'owner', rid, value)
        if drift == 'child-missing':
            db.execute('DELETE FROM requests WHERE owner=? AND id=?', ('owner', 'successor'))
        store.set_runtime(db, parent['key'], parent)
    old_before = row(h.service)
    from services.pool_admission import _image_follows_completed_text
    with store.connect() as db:
        assert not _image_follows_completed_text(store, db, 'owner', image, old_before, h.clock())
    # A different physical conversation does not enter this occupancy guard.
    # Its proof must still fail; independent binding validation owns that route.
    if drift == 'image-conversation':
        return
    with pytest.raises(AdmissionLost):
        context.before_send_with_marker()
    with store.connect() as db:
        image = store.read_receipt(db, 'image', 'owner', 'original-image')
    assert image['_submission_started'] is False
    assert row(h.service) == old_before
