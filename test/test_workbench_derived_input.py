"""Trusted Workbench-built successors keep Provider ownership and send guards."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import pytest

from services.conversation_binding_service import ConversationBindingError
from services.product_work import _key
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_category_directory_successor import install
from test.test_explicit_unknown_successor import row, task, update
from test.test_incomplete_compact_successor import incomplete
from test.test_no_final_successor import product_lifecycle, product_no_final
from test.test_pool_admission import build
from test.test_required_attributes_successor import install as install_no_final


def payload(service, original, *, request_id="workbench-child", mode="413_request_missing"):
    body = service.read_retained_input("owner", "old", source="internal:listing")
    messages = deepcopy(body["input"]["messages"])
    if isinstance(messages[0]["content"], list):
        messages[0]["content"][0]["text"] = "Workbench owns this derived business instruction."
    else:
        messages[0]["content"] = "Workbench owns this derived business instruction."
    return {
        "client_request_id": request_id,
        "original_input_hash": body["input_hash"],
        "messages": messages,
        "model": "gpt-5-6-thinking",
        "thinking_effort": "high",
        "mode": mode,
        "derivation_kind": "wb-business-rule-v1",
    }


def test_retained_input_is_exact_owner_scoped_and_workbench_child_runs_once(task):
    h = install(task)
    read = h.service.read_retained_input("owner", "old", source="internal:listing")
    assert read["protocol"] == "task-input-v1" and read["input"] == h.body
    assert read["input_hash"] == TextTaskService._submission_identity("owner", h.body)[1]
    first = h.service.submit_workbench_derived_input("owner", "old", payload(h.service, h.old),
                                                     source="internal:listing")
    child = row(h.service, "workbench-child")
    assert first["status"] == "queued"
    assert first["workbench_derived_input"] == {
        "protocol": "workbench-derived-input-v1",
        "original_input_hash": read["input_hash"],
        "child_input_hash": TextTaskService._submission_identity(
            "owner", h.service.store.load_input(child["_input_ref"]),
        )[1],
        "mode": "413_request_missing", "derivation_kind": "wb-business-rule-v1",
    }
    assert child["_workbench_derived_input"] == {
        "protocol": "workbench-derived-input-v1",
        "original_input_hash": read["input_hash"],
        "child_input_hash": TextTaskService._submission_identity(
            "owner", h.service.store.load_input(child["_input_ref"]),
        )[1],
        "mode": "413_request_missing", "derivation_kind": "wb-business-rule-v1",
    }
    stored = h.service.store.load_input(child["_input_ref"])
    assert stored["messages"][0]["content"][1:] == h.body["messages"][0]["content"][1:]
    assert stored.get("product_work") == h.body.get("product_work")
    assert h.service.submit_workbench_derived_input("owner", "old", payload(h.service, h.old),
                                                     source="internal:listing") == first
    h.admission.execute(h.admission.claim_next())
    assert h.service.submit_workbench_derived_input("owner", "old", payload(h.service, h.old),
                                                     source="internal:listing")["status"] == "succeeded"
    assert len(h.sent) == 1 and row(h.service) == h.old


@pytest.mark.parametrize("field,value", [
    ("provider_binding_id", "other"), ("parent_message_id", "different-parent"),
    ("product_work", {"product_ref": "other", "step_ref": "other"}),
])
def test_caller_cannot_supply_binding_cursor_or_product_work(task, field, value):
    h = install(task)
    request = payload(h.service, h.old)
    request[field] = value
    with pytest.raises(ConversationBindingError) as exc:
        h.service.submit_workbench_derived_input("owner", "old", request, source="internal:listing")
    assert exc.value.code == "CHAT_WORKBENCH_DERIVED_INPUT_INVALID"
    assert row(h.service, "workbench-child") is None


def test_source_media_hash_and_original_eligibility_fail_closed(task):
    h = install(task)
    request = payload(h.service, h.old)
    with pytest.raises(ConversationBindingError) as exc:
        h.service.read_retained_input("owner", "old", source="untrusted")
    assert exc.value.code == "CHAT_WORKBENCH_DERIVED_SOURCE_REQUIRED"
    request["messages"][0]["content"][1]["text"] = '{"image_ref":"replacement"}'
    with pytest.raises(ConversationBindingError):
        h.service.submit_workbench_derived_input("owner", "old", request, source="internal:listing")
    request = payload(h.service, h.old)
    request["original_input_hash"] = "0" * 64
    with pytest.raises(ConversationBindingError):
        h.service.submit_workbench_derived_input("owner", "old", request, source="internal:listing")
    update(h.service, status="succeeded", content="late original")
    with pytest.raises(ConversationBindingError) as exc:
        h.service.submit_workbench_derived_input("owner", "old", payload(h.service, h.old),
                                                 source="internal:listing")
    assert exc.value.code == "CHAT_SUPERSEDE_ORIGINAL_FOUND"


def test_same_child_id_rejects_changed_business_input(task):
    h = install(task)
    request = payload(h.service, h.old)
    h.service.submit_workbench_derived_input("owner", "old", request, source="internal:listing")
    changed = deepcopy(request)
    changed["thinking_effort"] = "standard"
    with pytest.raises(ConversationBindingError) as exc:
        h.service.submit_workbench_derived_input("owner", "old", changed, source="internal:listing")
    assert exc.value.code == "CONVERSATION_REQUEST_CONFLICT"


def test_local_validation_returns_only_an_exact_existing_child(task):
    h = install(task)
    request = payload(h.service, h.old)
    assert h.service.validate_workbench_derived_input("owner", "old", request,
                                                      source="internal:listing") is None
    first = h.service.submit_workbench_derived_input("owner", "old", request, source="internal:listing")
    assert h.service.validate_workbench_derived_input("owner", "old", request,
                                                      source="internal:listing") == first


def test_new_local_validation_rejects_ineligible_original_before_content_review(task):
    h = install(task)
    update(h.service, status="succeeded", content="late original")
    with pytest.raises(ConversationBindingError) as exc:
        h.service.validate_workbench_derived_input(
            "owner", "old", payload(h.service, h.old), source="internal:listing",
        )
    assert exc.value.code == "CHAT_SUPERSEDE_ORIGINAL_FOUND"
    assert row(h.service, "workbench-child") is None


def test_retained_input_refuses_receipt_binding_or_private_route_drift(task):
    h = install(task)
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        original["provider_binding_id"] = "other-binding"
        h.service.store.write_receipt(db, "text", "owner", "old", original)
    with pytest.raises(ConversationBindingError) as exc:
        h.service.read_retained_input("owner", "old", source="internal:listing")
    assert exc.value.code == "CHAT_WORKBENCH_DERIVED_INPUT_UNAVAILABLE"


@pytest.mark.parametrize("drift", ["original_succeeded", "original_paused", "original_hash"])
def test_send_guard_rechecks_late_original_state_pause_and_hash(task, drift):
    h = install(task)
    def runner(body, on_cursor):
        if drift == "original_succeeded":
            update(h.service, status="succeeded", content="late original")
        elif drift == "original_paused":
            update(h.service, _recovery_paused=True)
        else:
            with h.service.store.transaction() as db:
                db.execute("UPDATE requests SET request_hash='late-tampered' WHERE owner='owner' AND id='old'")
        current_request.get().before_send()
        h.sent.append(deepcopy(body))
        return {"content": "must not be written", "parent_message_id": "must-not-send"}

    h.service.runner = runner
    h.service.submit_workbench_derived_input("owner", "old", payload(h.service, h.old),
                                              source="internal:listing")
    h.admission.execute(h.admission.claim_next())
    child = row(h.service, "workbench-child")
    assert not h.sent and child["status"] != "succeeded"


def test_two_workbench_children_race_to_one_registered_send(task):
    h = install(task)
    requests = [payload(h.service, h.old, request_id="workbench-child-a"),
                payload(h.service, h.old, request_id="workbench-child-b")]
    def submit(request):
        try:
            return h.service.submit_workbench_derived_input(
                "owner", "old", request, source="internal:listing",
            )
        except ConversationBindingError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(submit, requests))
    accepted = [result for result in results if isinstance(result, dict)]
    assert len(accepted) == 1
    assert any(result == "CHAT_SUPERSEDE_CONFLICT" for result in results if isinstance(result, str))
    context = h.admission.claim_next()
    assert context and context.request_id in {"workbench-child-a", "workbench-child-b"}
    h.admission.execute(context)
    assert len(h.sent) == 1 and h.admission.claim_next() is None


def test_workbench_child_restart_claims_and_sends_once(task):
    h = install(task)
    request = payload(h.service, h.old)
    h.service.submit_workbench_derived_input("owner", "old", request, source="internal:listing")
    _, store, restarted_admission = build(h.root, h.clock)
    restarted = TextTaskService(store.path, runner=h.service.runner, clock=h.clock, admission=restarted_admission)
    restarted_admission.register("text", lambda context, body: restarted._run(context.owner, context.request_id, body))
    context = restarted_admission.claim_next()
    assert context and context.request_id == "workbench-child" and h.admission.claim_next() is None
    restarted_admission.execute(context)
    assert len(h.sent) == 1 and row(restarted) == h.old
    assert restarted.submit_workbench_derived_input("owner", "old", request,
                                                    source="internal:listing")["status"] == "succeeded"


def test_no_final_mode_reuses_only_the_existing_no_final_qualification(task):
    h = install_no_final(task)
    request = payload(h.service, h.old, mode="no_final")
    result = h.service.submit_workbench_derived_input("owner", "old", request, source="internal:listing")
    receipt = row(h.service, "workbench-child")
    assert result["status"] == "queued" and receipt["_supersedes_no_final"] is True
    assert receipt["_workbench_derived_input"]["mode"] == "no_final"


def test_incomplete_mode_reuses_only_the_existing_incomplete_qualification(incomplete):
    h = incomplete
    request = payload(h.service, h.old, mode="incomplete")
    result = h.service.submit_workbench_derived_input("owner", "old", request, source="internal:listing")
    receipt = row(h.service, "workbench-child")
    assert result["status"] == "queued" and receipt["_supersedes_incomplete"] is True
    assert receipt["_workbench_derived_input"]["mode"] == "incomplete"


def test_workbench_no_final_same_step_product_work_closes_archives_and_survives_restart(product_no_final):
    h = product_no_final
    # This fixture predates the trusted listing boundary; change only its local
    # saved source before ProductWork is first created, then exercise the real
    # Workbench-only path and its restart-safe lifecycle proof.
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        old_key = original["_product_work_key"]
        parent = h.service.store.runtime(db, old_key)
        new_key = _key("internal:listing", "owner", "fixture-product")
        parent.update(key=new_key, source="internal:listing")
        h.service.store.set_runtime(db, new_key, parent)
        db.execute("DELETE FROM task_runtime WHERE name=?", (old_key,))
        original["_source"] = "internal:listing"
        original.pop("_scheduling_owner", None)
        original["_product_work_key"] = new_key
        h.service.store.write_receipt(db, "text", "owner", "old", original)
    h.old = row(h.service)
    request = payload(h.service, h.old, mode="no_final")
    assert h.service.submit_workbench_derived_input(
        "owner", "old", request, source="internal:listing",
    )["status"] == "queued"
    context = h.admission.claim_next()
    assert context and context.request_id == "workbench-child"
    h.admission.execute(context)
    h.admission.update_claim(context, _claim_id=None, _claim_until=0, _turn_reserved=False)

    child = row(h.service, "workbench-child")
    assert child["status"] == "succeeded" and child["_product_work"] == h.old["_product_work"]
    product, lifecycle, calls, identity = product_lifecycle(h)
    identity["_trusted_source"] = "internal:listing"
    assert product.close(identity, "fixture-product", results_saved=True)["state"] == "closing"
    assert lifecycle.process_one()
    assert product.get(identity, "fixture-product")["state"] == "completed"
    assert calls == [("owner", "workbench-child", True)]

    restarted = TextTaskService(h.service.store.path, runner=h.service.runner, clock=h.clock,
                                admission=h.admission)
    with restarted._db() as db:
        restarted_public = restarted._public(
            restarted.store.read_receipt(db, "text", "owner", "workbench-child")
        )
    assert restarted_public["workbench_derived_input"] == {
        "protocol": "workbench-derived-input-v1",
        "original_input_hash": child["_supersedes_input_hash"],
        "child_input_hash": TextTaskService._submission_identity(
            "owner", restarted.store.load_input(child["_input_ref"]),
        )[1],
        "mode": "no_final", "derivation_kind": "wb-business-rule-v1",
    }
