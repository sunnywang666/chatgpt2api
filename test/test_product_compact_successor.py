"""Saved compact results close only their exact product and ended original turn."""
from copy import deepcopy
import json

import pytest

from services.conversation_binding_service import ConversationBindingService
from services.product_work import ProductWorkService
from services.text_task_service import TextTaskService
from services.work_lifecycle import WorkLifecycleError
from test.test_combined_copy_derivation import WBOZ_CURRENT_CASES, branch, legacy_prompt
from test.test_incomplete_compact_successor import install_incomplete
from test.test_no_final_successor import (product_no_final, product_lifecycle,
    complete_product_no_final_successor)
from test.test_explicit_unknown_successor import row, update


def compact(h, incomplete=False, *, prompt=None, input_body=None):
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


def advanced_to_own_successor(h, monkeypatch):
    """Persist the original reader's exact, redacted local-branch proof."""
    original, _successor = compact(h, True)
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        successor = h.service.store.read_receipt(db, "text", "owner", "successor")
        # Preserve the explicit legacy boundary observed on the real receipt;
        # recovery must carry it through without a post-read repair.
        original["upstream_unfinished"] = False
        h.service.store.write_receipt(db, "text", "owner", "old", original)
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
        "successor_final_message_id": successor["parent_message_id"],
        "observed_at": h.clock(),
    }
    return original, row(h.service, "successor")


def add_finished_gallery(h):
    """Attach an already-finished earlier product step on the same Chat."""
    original = row(h.service)
    body = h.service.store.load_input(original["_input_ref"])
    body.update(client_request_id="gallery", product_work={"product_ref": "fixture-product", "step_ref": "gallery"})
    h.service.submit("owner", body, source="internal:content")
    with h.service.store.transaction() as db:
        gallery = h.service.store.read_receipt(db, "text", "owner", "gallery")
        old = h.service.store.read_receipt(db, "text", "owner", "old")
        successor = h.service.store.read_receipt(db, "text", "owner", "successor")
        gallery.update(status="succeeded", upstream_outcome="completed", content="saved gallery review",
                       _upstream_terminal=True, _executing=False, _turn_reserved=False,
                       _claim_id=None, _claim_until=None, _sequence=4746)
        old["_sequence"] = 5945
        successor["_sequence"] = 7122
        h.service.store.write_receipt(db, "text", "owner", "gallery", gallery)
        h.service.store.write_receipt(db, "text", "owner", "old", old)
        h.service.store.write_receipt(db, "text", "owner", "successor", successor)
    return row(h.service, "gallery")


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
