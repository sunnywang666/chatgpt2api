"""Saved R -> rejected P -> native no-final N -> successful C product closure."""
import json

import pytest

from services.product_work import ProductWorkService
from services.text_task_service import TextTaskService
from services.work_lifecycle import WorkLifecycleError
from test.test_no_final_successor import product_no_final, product_lifecycle
from test.test_rejected_chat_successor import prove_no_final_model_rejection, receipt, update


IDS = ("old", "rejected-parent", "model-retry", "native-no-final-child")


def complete_chain(h):
    h.body = h.service.store.load_input(h.old["_input_ref"])
    retry = prove_no_final_model_rejection(h)
    retry["client_conversation_id"] = h.body["client_conversation_id"]
    h.service.submit("owner", retry, source="internal:content")
    update(h.service, "old", request_parent_message_id="previous-answer")
    update(h.service, "model-retry", status="failed", error_code="RESULT_UNRECOVERABLE",
           upstream_outcome="unknown", recovery_reason="REQUEST_RESULT_NOT_FOUND",
           _attempt_reason="ORIGINAL_RESULT_NO_FINAL", _attempt_finished_at=500,
           recovery_no_result_reads=4, _execution_wait_ended_at=500,
           _submission_started=True, _last_sent_sequence=0, _executing=False,
           _turn_reserved=False, _claim_id=None, _claim_until=None,
           recovery_claim_id=None, recovery_lease_until=None,
           request_parent_message_id="native-root-tail")
    originals = [receipt(h.service, rid) for rid in IDS[:-1]]
    body = h.service.store.load_input(originals[-1]["_input_ref"])
    envelope = {key: body[key] for key in TextTaskService.SUPERSEDE_FIELDS}
    envelope.update(client_request_id=IDS[-1], supersedes_request_id="model-retry",
                    continue_after_no_final=True)
    h.service.submit("owner", envelope, source="internal:content")
    context = h.admission.claim_next()
    assert context and context.request_id == IDS[-1]
    h.admission.execute(context)
    h.admission.update_claim(context, _claim_id=None, _claim_until=0, _turn_reserved=False)
    assert receipt(h.service, IDS[-1])["status"] == "succeeded"
    return originals


def test_saved_native_chain_closes_archives_and_survives_resume(product_no_final):
    h = product_no_final
    originals = complete_chain(h)
    product, lifecycle, calls, identity = product_lifecycle(h)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_RESULTS_SAVE_REQUIRED"):
        product.close(identity, "fixture-product", results_saved=False)
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    assert lifecycle.process_one()
    assert product.get(identity, "fixture-product")["state"] == "completed"
    assert calls == [("owner", IDS[-1], True)]
    assert [receipt(h.service, rid) for rid in IDS[:-1]] == originals

    restarted = ProductWorkService(lifecycle.text, lifecycle.images, clock=h.clock)
    assert restarted.resume(identity, "fixture-product", expected_version=1)["state"] == "active"
    h.service.submit("owner", {**h.body, "client_request_id": "v2-normal",
                     "parent_message_id": receipt(h.service, IDS[-1])["parent_message_id"],
                     "product_work": {"product_ref": "fixture-product", "step_ref": "v2-normal"}},
                     source="internal:content")
    context = h.admission.claim_next()
    assert context and context.request_id == "v2-normal"
    h.admission.execute(context)
    h.admission.update_claim(context, _claim_id=None, _claim_until=0, _turn_reserved=False)
    assert restarted.close(identity, "fixture-product", results_saved=True, expected_version=2)["state"] == "closing"
    assert lifecycle.process_one()
    assert restarted.get(identity, "fixture-product")["state"] == "completed"
    assert calls[-1] == ("owner", "v2-normal", True)
    assert [receipt(h.service, rid) for rid in IDS[:-1]] == originals


@pytest.mark.parametrize("rid,changes", [
    (IDS[-1], {"status": "failed", "upstream_outcome": "unknown"}),
    (IDS[-1], {"content": " "}),
    (IDS[-1], {"_supersedes_request_message_id": "wrong-branch"}),
    (IDS[-1], {"provider_account_identity": "different-account"}),
    (IDS[-1], {"_product_work": {"product_ref": "other", "step_ref": "title"}}),
    (IDS[0], {"content": "late original answer"}),
    (IDS[1], {"upstream_outcome": "unknown"}),
    (IDS[1], {"_submission_parent_message_id": "wrong-branch"}),
    (IDS[1], {"model": "different-model"}),
    (IDS[0], {"_scheduling_owner": "different-source"}),
    (IDS[2], {"_product_work_key": "different-product"}),
    (IDS[2], {"_supersedes_no_final": True}),
    (IDS[2], {"_sequence": 1}),
] + [(rid, {"_recovery_paused": True}) for rid in IDS]
  + [(rid, {"_claim_id": "new-generation", "_executing": True}) for rid in IDS]
  + [(rid, {"recovery_claim_id": "new-read", "recovery_lease_until": 2000}) for rid in IDS])
def test_changed_native_proof_blocks_close(product_no_final, rid, changes):
    h = product_no_final
    complete_chain(h)
    update(h.service, rid, **changes)
    product, _, calls, identity = product_lifecycle(h)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_"):
        product.close(identity, "fixture-product", results_saved=True)
    assert calls == []


@pytest.mark.parametrize("rid", IDS)
def test_changed_native_input_hash_blocks_close(product_no_final, rid):
    h = product_no_final
    complete_chain(h)
    with h.service.store.transaction() as db:
        db.execute("UPDATE requests SET request_hash=? WHERE owner=? AND id=?", ("tampered", "owner", rid))
    product, _, calls, identity = product_lifecycle(h)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_TURN_UNFINISHED"):
        product.close(identity, "fixture-product", results_saved=True)
    assert calls == []


def test_retired_original_execution_claim_keeps_existing_lifecycle_contract(product_no_final):
    h = product_no_final
    complete_chain(h)
    update(h.service, IDS[0], _claim_id="retired-attribution", _claim_until=500,
           _executing=False, _turn_reserved=False)
    product, lifecycle, calls, identity = product_lifecycle(h)
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    assert lifecycle.process_one()
    assert calls == [("owner", IDS[-1], True)]
    assert receipt(h.service)["_claim_id"] == "retired-attribution"


@pytest.mark.parametrize("rid", IDS[:-1])
def test_second_native_chain_child_blocks_close(product_no_final, rid):
    h = product_no_final
    complete_chain(h)
    with h.service.store.transaction() as db:
        fallback = {**receipt(h.service, IDS[-1]), "request_id": "fallback",
                    "_codex_fallback_of": rid, "_supersedes_request_id": None}
        db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("owner", "fallback", "fixture", json.dumps(fallback)))
    product, _, calls, identity = product_lifecycle(h)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_TURN_UNFINISHED"):
        product.close(identity, "fixture-product", results_saved=True)
    assert calls == []


@pytest.mark.parametrize("rid", IDS[:-1])
def test_archive_rechecks_native_ancestors(product_no_final, rid):
    h = product_no_final
    complete_chain(h)
    product, lifecycle, calls, identity = product_lifecycle(h)
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    update(h.service, rid, _recovery_paused=True)
    assert not lifecycle.process_one()
    assert product.get(identity, "fixture-product")["state"] == "closing"
    assert calls == []


@pytest.mark.parametrize("rid", IDS[:-1])
def test_native_ancestor_must_remain_a_product_member(product_no_final, rid):
    h = product_no_final
    complete_chain(h)
    with h.service.store.transaction() as db:
        key = receipt(h.service)["_product_work_key"]
        parent = h.service.store.runtime(db, key)
        parent["members"] = [member for member in parent["members"] if member["request_id"] != rid]
        h.service.store.set_runtime(db, key, parent)
    product, _, calls, identity = product_lifecycle(h)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_TURN_UNFINISHED"):
        product.close(identity, "fixture-product", results_saved=True)
    assert calls == []


@pytest.mark.parametrize("unknown", [False, True])
def test_unrelated_product_turn_is_not_waived(product_no_final, unknown):
    h = product_no_final
    complete_chain(h)
    h.service.submit("owner", {**h.body, "client_request_id": "other",
                     "product_work": {"product_ref": "fixture-product", "step_ref": "other"}},
                     source="internal:content")
    if unknown:
        update(h.service, "other", status="failed", upstream_outcome="unknown", _submission_started=True)
    product, _, calls, identity = product_lifecycle(h)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_TURN_UNFINISHED"):
        product.close(identity, "fixture-product", results_saved=True)
    assert calls == []
