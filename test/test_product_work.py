import copy
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from services.task_store import TaskStore
from services.work_lifecycle import WorkLifecycleError, WorkLifecycleService
from services.product_work import (ProductWorkService, normalize_product_work, ensure_product_work,
                                   record_slot, before_send, state_snapshot, _claims)

IDENTITY = {"id": "owner", "role": "admin", "_trusted_source": "internal:listing"}


@pytest.fixture
def runtime(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    now, calls, fail = [1000.0], [], [False]
    def archive(owner, rid, archived):
        calls.append((owner, rid, archived))
        if fail[0]:
            raise RuntimeError("lost response")
        with store.connect() as db:
            r = store.read_receipt(db, "text", owner, rid) or store.read_receipt(db, "image", owner, rid)
        return {"request_id": rid, "conversation_id": r["conversation_id"], "archived": archived}
    text = SimpleNamespace(store=store, admission=None, archive_product_work=archive)
    images = SimpleNamespace(archive_product_work=archive)
    service = ProductWorkService(text, images, clock=lambda: now[0])
    lifecycle = WorkLifecycleService(text, images, clock=lambda: now[0])
    def add(rid, *, kind="text", product="P", step=None, status=None, conversation=None, physical=True,
            source="internal:listing", owner="owner", **extras):
        status = status or ("succeeded" if kind == "text" else "success")
        r = {"request_id" if kind == "text" else "id": rid, "owner_id": owner, "status": status,
             "_source": source, "_sequence": 1, "_input_ref": "original-input", "_submission_started": status != "queued",
             "upstream_outcome": "not_sent" if status == "queued" else "unknown" if status == "unknown" else "completed"}
        if status != "queued":
            r.update(provider_binding_id="binding", provider_account_identity="account",
                     conversation_id=(conversation or rid) if physical else None, parent_message_id="parent-" + rid)
        r.update(extras)
        with store.transaction() as db:
            if product:
                ensure_product_work(store, db, kind, owner, rid, r,
                                    {"product_ref": product, "step_ref": step or rid}, source=source)
            if kind == "text":
                db.execute("INSERT INTO requests VALUES(?,?,?,?)", (owner, rid, "immutable-hash", json.dumps(r)))
            else:
                store.write_receipt(db, kind, owner, rid, r)
            if product and status != "queued":
                record_slot(store, db, r)
        return r
    return service, lifecycle, store, now, calls, fail, add


def test_shared_text_image_close_waits_for_every_archive_and_restart(runtime):
    service, lifecycle, store, now, calls, fail, add = runtime
    originals = [add("text"), add("image", kind="image")]
    result = service.close(IDENTITY, "P")
    assert result["state"] == "closing" and result["slot_held"] and result["cleanup_pending"]
    assert result["archive"] == {"total": 2, "confirmed": 0, "pending": 2}
    assert calls == []
    assert lifecycle.process_one()
    assert service.get(IDENTITY, "P")["slot_held"]
    lifecycle = WorkLifecycleService(lifecycle.text, lifecycle.images, clock=lambda: now[0])
    assert lifecycle.process_one()
    result = service.get(IDENTITY, "P")
    assert result["state"] == "completed" and not result["slot_held"] and not result["cleanup_pending"]
    assert result["archive"]["confirmed"] == 2
    assert service.close(IDENTITY, "P") == result
    with store.connect() as db:
        assert store.read_receipt(db, "text", "owner", "text") == originals[0]
        assert store.read_receipt(db, "image", "owner", "image") == originals[1]
        assert db.execute("SELECT request_hash FROM requests").fetchone()[0] == "immutable-hash"
    with pytest.raises(WorkLifecycleError, match="CLOSE_CONFLICT"):
        service.close(IDENTITY, "P", outcome="failed")


def test_archive_unknown_retains_slot_and_retries_original_target(runtime):
    service, lifecycle, _, now, calls, fail, add = runtime
    add("text")
    service.close(IDENTITY, "P")
    fail[0] = True
    assert lifecycle.process_one()
    assert service.get(IDENTITY, "P")["slot_held"]
    assert not lifecycle.process_one()
    fail[0], now[0] = False, now[0] + 10
    assert lifecycle.process_one()
    assert calls == [("owner", "text", True)] * 2
    assert not service.get(IDENTITY, "P")["slot_held"]


@pytest.mark.parametrize("extra", [{"status": "unknown", "_recovery_paused": True},
    {"status": "failed", "upstream_outcome": "unknown", "_recovery_suppressed": True},
    {"_claim_id": "ordinary"}, {"recovery_claim_id": "read"}, {"_executing": True},
    {"_turn_reserved": True}])
def test_claim_or_unknown_blocks_without_mutation(runtime, extra):
    service, _, store, _, _, _, add = runtime
    original = add("text", **extra)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_"):
        service.close(IDENTITY, "P")
    assert service.get(IDENTITY, "P")["state"] == "active"
    with store.connect() as db:
        assert store.read_receipt(db, "text", "owner", "text") == original


def test_retired_recovery_claim_stops_blocking_without_closing_parent(runtime):
    service, _, store, _, _, _, add = runtime
    receipt = add("text", recovery_claim_id="expired-reader", recovery_claimed_at=900.0,
                  recovery_lease_until=999.0, _executing=False, _turn_reserved=False)
    assert _claims(receipt)
    with store.transaction() as db:
        current = store.read_receipt(db, "text", "owner", "text")
        current.update(recovery_claim_id=None, recovery_claimed_at=None, recovery_lease_until=None)
        db.execute("UPDATE requests SET receipt=? WHERE owner=? AND id=?",
                   (json.dumps(current), "owner", "text"))
    assert not _claims(current)
    assert service.get(IDENTITY, "P")["state"] == "active"


def test_cancel_unsent_atomic_and_cannot_send_or_join_after_close(runtime):
    service, _, store, _, _, _, add = runtime
    original = add("queued", status="queued")
    with store.transaction() as db:
        with pytest.raises(WorkLifecycleError, match="NOT_ACTIVE"):
            before_send(store, db, original)
    result = service.close(IDENTITY, "P", outcome="cancelled")
    assert result["state"] == "cancelled" and not result["slot_held"]
    with store.connect() as db:
        receipt = store.read_receipt(db, "text", "owner", "queued")
        assert receipt["status"] == "cancelled" and receipt["_input_ref"] == "original-input"
    with pytest.raises(WorkLifecycleError, match="NOT_ACTIVE"):
        add("later", status="queued")


def test_close_freezes_joins_and_provisional_release_does_not_release(runtime):
    service, _, store, _, _, _, add = runtime
    receipt = add("text")
    with store.transaction() as db:
        record_slot(store, db, receipt, held=False)
    assert service.get(IDENTITY, "P")["slot_held"]
    service.close(IDENTITY, "P")
    with store.transaction() as db:
        with pytest.raises(WorkLifecycleError, match="NOT_ACTIVE"):
            before_send(store, db, receipt)
    with pytest.raises(WorkLifecycleError, match="NOT_ACTIVE"):
        add("later")


def test_adoption_preserves_receipts_and_grandfathers_unknown(runtime):
    service, _, store, _, _, _, add = runtime
    original = add("old", product=None, status="unknown")
    result = service.adopt(IDENTITY, "P", [{"kind": "text", "request_id": "old", "step_ref": "text"}])
    assert result["slot_held"] and result["state"] == "active"
    with store.connect() as db:
        adopted = store.read_receipt(db, "text", "owner", "old")
        assert {k: v for k, v in adopted.items() if not k.startswith("_product_work")} == original
    with pytest.raises(WorkLifecycleError, match="TURN_UNFINISHED"):
        service.close(IDENTITY, "P", outcome="failed")


def test_adoption_queued_waits_and_batch_validation_rolls_back(runtime):
    service, _, store, _, _, _, add = runtime
    old = add("old", product=None, status="queued")
    add("wrong", product=None, _work_key="work:public")
    with pytest.raises(WorkLifecycleError, match="LEGACY_REQUIRED"):
        service.adopt(IDENTITY, "P", [{"kind": "text", "request_id": "old", "step_ref": "a"},
                                      {"kind": "text", "request_id": "wrong", "step_ref": "b"}])
    with store.connect() as db:
        assert store.read_receipt(db, "text", "owner", "old") == old
        assert state_snapshot(store, db) == {}
    result = service.adopt(IDENTITY, "P", [{"kind": "text", "request_id": "old", "step_ref": "a"}])
    assert not result["slot_held"] and result["state"] == "queued"


def test_cross_parent_physical_conversation_and_other_legacy_turn_rejected(runtime):
    service, _, _, _, _, _, add = runtime
    add("first", conversation="shared")
    with pytest.raises(WorkLifecycleError, match="CONVERSATION_CONFLICT"):
        add("second", product="Q", kind="image", conversation="shared")
    add("legacy", product=None, conversation="shared", status="unknown")
    with pytest.raises(WorkLifecycleError, match="CONVERSATION_CONFLICT"):
        service.close(IDENTITY, "P")


def test_canonical_archive_deduplicates_shared_text_image_conversation(runtime):
    service, lifecycle, _, _, calls, _, add = runtime
    add("text", conversation="shared")
    add("image", kind="image", conversation="shared", _sequence=2)
    assert service.close(IDENTITY, "P")["archive"]["total"] == 1
    assert lifecycle.process_one()
    assert calls == [("owner", "image", True)]


def test_source_owner_and_duplicate_step_security(runtime):
    service, _, store, _, _, _, add = runtime
    first = add("first")
    for identity in ({**IDENTITY, "id": "other"}, {**IDENTITY, "_trusted_source": "internal:content"}):
        with pytest.raises(WorkLifecycleError, match="NOT_FOUND"):
            service.get(identity, "P")
    with pytest.raises(WorkLifecycleError, match="SOURCE_REQUIRED"):
        service.get({**IDENTITY, "role": "user"}, "P")
    with store.transaction() as db:
        second = {**first, "request_id": "second"}
        with pytest.raises(WorkLifecycleError, match="STEP_CONFLICT"):
            ensure_product_work(store, db, "text", "owner", "second", second,
                                {"product_ref": "P", "step_ref": "first"})
    with pytest.raises(WorkLifecycleError, match="SOURCE_CONFLICT"):
        service.adopt({**IDENTITY, "_trusted_source": "internal:content"}, "Q",
                      [{"kind": "text", "request_id": "first", "step_ref": "a"}])


@pytest.mark.parametrize("value", [{}, {"product_ref": "x", "step_ref": ""},
    {"product_ref": " x", "step_ref": "y"}, {"product_ref": "x", "step_ref": "y "},
    {"product_ref": "x", "step_ref": "y", "source": "evil"}, {"product_ref": "x" * 201, "step_ref": "y"}])
def test_strict_metadata(value):
    with pytest.raises(WorkLifecycleError):
        normalize_product_work(value)


def test_legacy_receipt_cannot_bypass_product_after_binding(runtime):
    _, _, store, _, _, _, add = runtime
    add("product", conversation="owned")
    legacy = {"request_id": "legacy", "_source": "internal:listing", "provider_binding_id": "binding",
              "provider_account_identity": "account", "conversation_id": "owned"}
    with store.transaction() as db:
        with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_REQUIRED"):
            ensure_product_work(store, db, "text", "owner", "legacy", legacy, None)
        with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_REQUIRED"):
            before_send(store, db, legacy)


def test_late_external_member_blocks_archive_after_close(runtime):
    service, lifecycle, store, _, calls, _, add = runtime
    add("text", conversation="shared")
    service.close(IDENTITY, "P")
    add("legacy", product=None, conversation="shared", status="unknown")
    assert not lifecycle.process_one()
    assert calls == [] and service.get(IDENTITY, "P")["slot_held"]


def test_unconfirmed_archive_response_does_not_release(runtime):
    service, lifecycle, _, _, _, _, add = runtime
    add("text")
    service.close(IDENTITY, "P")
    lifecycle.text.archive_product_work = lambda *args: {"request_id": "text", "conversation_id": "other", "archived": True}
    assert lifecycle.process_one()
    assert service.get(IDENTITY, "P")["archive"]["confirmed"] == 0
    assert service.get(IDENTITY, "P")["slot_held"]


def test_archive_claim_restart_waits_for_lease_then_recovers(runtime):
    service, lifecycle, store, now, calls, _, add = runtime
    add("text")
    service.close(IDENTITY, "P")
    with store.transaction() as db:
        parent = next(iter(state_snapshot(store, db).values()))
        child = store.runtime(db, parent["archive_keys"][0])
        child["archive"].update(status="running", claim="crashed", claim_until=now[0] + 300)
        store.set_runtime(db, child["key"], child)
    assert not lifecycle.process_one() and calls == []
    now[0] += 301
    restarted = WorkLifecycleService(lifecycle.text, lifecycle.images, clock=lambda: now[0])
    assert restarted.process_one()
    assert service.get(IDENTITY, "P")["state"] == "completed"


def test_failed_partial_results_require_saved_and_archive(runtime):
    service, lifecycle, _, _, _, _, add = runtime
    add("success")
    add("failed", kind="image", status="error", upstream_outcome="failed")
    with pytest.raises(WorkLifecycleError, match="RESULTS_SAVE_REQUIRED"):
        service.close(IDENTITY, "P", results_saved=False, outcome="failed")
    assert service.close(IDENTITY, "P", outcome="failed")["state"] == "closing"
    assert lifecycle.process_one() and lifecycle.process_one()
    result = service.get(IDENTITY, "P")
    assert result["state"] == "failed" and result["results_saved"] and not result["slot_held"]


def test_adopt_queued_image_without_text_outcome_is_unsent(runtime):
    service, _, store, _, _, _, add = runtime
    add("queued", kind="image", status="queued", product=None, upstream_outcome=None)
    result = service.adopt(IDENTITY, "P", [{"kind": "image", "request_id": "queued", "step_ref": "image"}])
    assert result["state"] == "queued" and not result["slot_held"]
    assert service.close(IDENTITY, "P", outcome="cancelled")["state"] == "cancelled"


def test_public_generic_and_nonchat_product_acceptance_rejected(runtime):
    _, _, _, _, _, _, add = runtime
    for extra in ({"_work_key": "generic"}, {"_image_thread": {"id": "public"}},
                  {"_public_session_ref": "public"}, {"_route": "codex"}):
        with pytest.raises(WorkLifecycleError, match="LEGACY_REQUIRED"):
            add("bad", **extra)


def test_proven_existing_same_session_correction_closes_original_history(runtime):
    service, lifecycle, _, _, _, _, add = runtime
    root = add("root", product=None, status="unknown", conversation="shared", request_message_id="request-root",
               client_conversation_id="client", _retry_cursor={"conversation_id": "shared", "request_message_id": "request-root",
               "retry_parent_message_id": "cursor", "observed_at": 900.0},
               _completion={"replacement_id": "child", "selected_id": "child"})
    add("child", product=None, conversation="shared", client_conversation_id="client", _sequence=2,
        _same_session_retry_of="root", _completion_of="root", _submission_parent_message_id="cursor", content="recovered")
    service.adopt(IDENTITY, "P", [{"kind": "text", "request_id": "root", "step_ref": "text"},
                                 {"kind": "text", "request_id": "child", "step_ref": "text"}])
    assert service.close(IDENTITY, "P")["state"] == "closing"
    assert lifecycle.process_one()
    assert service.get(IDENTITY, "P")["state"] == "completed"


def test_generic_lifecycle_cannot_archive_or_release_product(runtime):
    _, lifecycle, _, _, _, _, add = runtime
    add("text")
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_REQUIRED"):
        lifecycle.update("text", IDENTITY, "text", "completed", results_saved=True)


@pytest.mark.parametrize("extra", [{"_last_sent_sequence": 1}, {"send_count": 1},
                                    {"upstream_unfinished": True}, {"error_code": "CONVERSATION_OUTCOME_UNKNOWN"}])
def test_queued_with_prior_send_evidence_is_not_cancellable(runtime, extra):
    service, _, _, _, _, _, add = runtime
    add("queued", status="queued", **extra)
    with pytest.raises(WorkLifecycleError, match="TURN_UNFINISHED"):
        service.close(IDENTITY, "P", outcome="cancelled")


def test_real_image_admission_retires_execution_but_keeps_claim_identity(tmp_path):
    from datetime import datetime, timezone
    from test.test_pool_admission import build, Clock
    from services.text_task_service import TextTaskService
    from services.image_task_service import ImageTaskService
    clock = Clock()
    (tmp_path / "accounts.json").write_text(json.dumps([{
        "access_token": "fixture-token", "account_id": "upstream", "provider_account_identity": "account",
        "type": "Plus", "status": "正常", "quota": 99, "source_type": "web",
        "conversation_binding_ids": ["binding"], "limits_progress": [{"feature_name": "image_gen", "remaining": 99}],
        "capacity_observed_at": datetime.now(timezone.utc).isoformat()}]))
    _, store, admission = build(tmp_path, clock)
    text = TextTaskService(store.path, admission=admission, clock=clock)
    images = ImageTaskService(tmp_path / "images.json", store=store, admission=admission)
    def finish(context, body):
        context.before_send()
        context.terminal(True)
        images._update_task("owner:image", status="success", upstream_outcome="generated", upstream_unfinished=False,
                            conversation_id="conversation", parent_message_id="parent", data=[{"url": "fixture-result"}])
    admission.register("image", finish)
    images.submit_generation(IDENTITY, client_task_id="image", prompt="fixture", model="gpt-image-2", size=None,
                             client_conversation_id="client", product_work={"product_ref": "P", "step_ref": "image"})
    context = admission.claim_next()
    assert context and context.kind == "image"
    product = ProductWorkService(text, images, clock=clock)
    with pytest.raises(WorkLifecycleError, match="TURN_UNFINISHED"):
        product.close(IDENTITY, "P")
    admission.execute(context)
    with store.connect() as db:
        receipt = store.read_receipt(db, "image", "owner", "image")
    assert receipt["status"] == "success" and receipt["_claim_id"] == context.claim
    assert receipt["_claim_until"] > clock.now
    assert receipt["_executing"] is False and receipt["_turn_reserved"] is False
    result = product.close(IDENTITY, "P")
    assert result["state"] == "closing" and result["slot_held"] and result["cleanup_pending"]


def test_real_text_admission_retires_turn_and_allows_product_close(tmp_path):
    from datetime import datetime, timezone
    from services.request_context import current_request
    from services.text_task_service import TextTaskService
    from test.test_pool_admission import build, Clock

    clock = Clock()
    (tmp_path / "accounts.json").write_text(json.dumps([{
        "access_token": "fixture-token", "account_id": "upstream", "provider_account_identity": "account",
        "type": "Plus", "status": "正常", "quota": 99, "source_type": "web",
        "conversation_binding_ids": ["binding"],
        "capacity_observed_at": datetime.now(timezone.utc).isoformat(),
    }]))
    _, store, admission = build(tmp_path, clock)

    def runner(body, on_cursor):
        current_request.get().before_send()
        return {"content": "saved normal text result", "upstream_outcome": "completed",
                "upstream_unfinished": False, "provider_binding_id": "binding",
                "provider_account_identity": "account", "conversation_id": "conversation",
                "parent_message_id": "final-parent"}

    text = TextTaskService(store.path, runner=runner, admission=admission, clock=clock)
    admission.register("text", lambda context, body: text._run(context.owner, context.request_id, body))
    text.submit("owner", {
        "client_request_id": "text", "client_conversation_id": "client", "model": "fixture-text",
        "messages": [{"role": "user", "content": "fixture"}],
        "product_work": {"product_ref": "P", "step_ref": "text"},
    }, source="internal:listing")

    context = admission.claim_next()
    assert context and context.kind == "text"
    product = ProductWorkService(text, SimpleNamespace(archive_product_work=lambda *args: None), clock=clock)
    with pytest.raises(WorkLifecycleError, match="TURN_UNFINISHED"):
        product.close(IDENTITY, "P")
    admission.execute(context)
    with store.connect() as db:
        receipt = store.read_receipt(db, "text", "owner", "text")
    assert receipt["status"] == "succeeded" and receipt["content"] == "saved normal text result"
    assert receipt["_claim_id"] == context.claim and receipt["_claim_until"] > clock.now
    assert receipt["_executing"] is False and receipt["_turn_reserved"] is False
    result = product.close(IDENTITY, "P")
    assert result["state"] == "closing" and result["slot_held"] and result["cleanup_pending"]


def test_recovered_original_final_retires_turn_and_allows_same_product_close(tmp_path):
    from datetime import datetime, timezone
    from unittest.mock import Mock
    from services.conversation_binding_service import ConversationBindingService
    from services.request_context import current_request
    from services.text_task_service import TextTaskService
    from test.test_pool_admission import build, Clock
    from test.test_unknown_turn_recovery import document

    clock = Clock()
    (tmp_path / "accounts.json").write_text(json.dumps([{
        "access_token": "fixture-token", "account_id": "upstream", "provider_account_identity": "account",
        "type": "Plus", "status": "正常", "quota": 99, "source_type": "web",
        "conversation_binding_ids": ["binding"],
        "capacity_observed_at": datetime.now(timezone.utc).isoformat(),
    }]))
    _, store, admission = build(tmp_path, clock)

    runner_calls = []
    def runner(_body, on_cursor):
        runner_calls.append("send")
        current_request.get().before_send()
        on_cursor({"provider_binding_id": "binding", "provider_account_identity": "account",
                   "conversation_id": "conversation", "parent_message_id": "parent"})
        raise TimeoutError("synthetic sender timeout")

    text = TextTaskService(store.path, runner=runner, admission=admission, clock=clock)
    admission.register("text", lambda context, body: text._run(context.owner, context.request_id, body))
    text.submit("owner", {
        "client_request_id": "recovered-text", "client_conversation_id": "client", "model": "fixture-text",
        "messages": [{"role": "user", "content": "fixture"}],
        "product_work": {"product_ref": "P", "step_ref": "text"},
    }, source="internal:listing")
    context = admission.claim_next()
    assert context and context.kind == "text"
    admission.execute(context)
    with store.connect() as db:
        timed_out = store.read_receipt(db, "text", "owner", "recovered-text")
    assert timed_out["status"] == "unknown" and timed_out["_turn_reserved"] is True

    backend = Mock()
    upstream = document(timed_out)
    upstream["mapping"]["final-" + timed_out["request_message_id"]]["message"]["content"]["parts"] = ["recovered final"]
    text.recovery_reader = lambda row: ConversationBindingService._read_text_request_result(
        backend, row, document=upstream,
    )
    recovered = text.read("owner", "recovered-text")
    assert recovered["status"] == "succeeded" and recovered["content"] == "recovered final"
    with store.connect() as db:
        saved = store.read_receipt(db, "text", "owner", "recovered-text")
    assert saved["_turn_reserved"] is False and saved["_executing"] is False

    images = SimpleNamespace(archive_product_work=lambda *args: None)
    product = ProductWorkService(text, images, clock=clock)
    closed = product.close(IDENTITY, "P")
    assert closed["state"] == "closing" and closed["cleanup_pending"] is True
    # A repeated close retains the one durable archive intent.  Simulate a
    # remote archive that accepted its PATCH but lost the response: restart
    # reconciliation observes that remote state read-only and never resends
    # either the model turn or the archive PATCH before resume.
    assert product.close(IDENTITY, "P")["state"] == "closing"
    remote = {"archived": False}
    archive_calls = []
    def archive(_owner, _request_id, desired):
        archive_calls.append("precheck")
        if remote["archived"]:
            return {"request_id": "recovered-text", "conversation_id": "conversation", "archived": desired}
        archive_calls.append("patch")
        remote["archived"] = True
        raise TimeoutError("synthetic lost archive response")

    text.archive_product_work = archive
    from services.work_lifecycle import WorkLifecycleService
    lifecycle = WorkLifecycleService(text, images, clock=clock)
    assert lifecycle.process_one()
    assert product.get(IDENTITY, "P")["archive"]["pending"] == 1
    clock.now += 61
    restarted_lifecycle = WorkLifecycleService(text, images, clock=clock)
    assert restarted_lifecycle.process_one()
    completed = product.get(IDENTITY, "P")
    assert completed["state"] == "completed" and archive_calls.count("patch") == 1
    assert product.resume(IDENTITY, "P", completed["version"])["state"] == "active"
    assert runner_calls == ["send"]


def test_expired_sender_claim_scanner_recovers_original_final_and_allows_product_close(tmp_path):
    from datetime import datetime, timezone
    from unittest.mock import Mock
    from services.conversation_binding_service import ConversationBindingService
    from services.request_context import executing
    from services.text_task_service import TextTaskService
    from test.test_pool_admission import build, Clock
    from test.test_unknown_turn_recovery import document

    clock = Clock()
    (tmp_path / "accounts.json").write_text(json.dumps([{
        "access_token": "fixture-token", "account_id": "upstream", "provider_account_identity": "account",
        "type": "Plus", "status": "正常", "quota": 99, "source_type": "web",
        "conversation_binding_ids": ["binding"],
        "capacity_observed_at": datetime.now(timezone.utc).isoformat(),
    }]))
    _, store, admission = build(tmp_path, clock)
    text = TextTaskService(store.path, admission=admission, clock=clock)
    text.submit("owner", {
        "client_request_id": "expired-text", "client_conversation_id": "client", "model": "fixture-text",
        "messages": [{"role": "user", "content": "fixture"}],
        "product_work": {"product_ref": "expired-product", "step_ref": "text"},
    }, source="internal:listing")
    sender = admission.claim_next()
    assert sender and sender.kind == "text"
    # This is the real claim/send edge.  No execute() finally runs afterwards,
    # modelling a sender process that exits after durable cursor persistence.
    with executing(sender):
        sender.before_send()
        text._update("owner", "expired-text", provider_binding_id="binding",
                     provider_account_identity="account", conversation_id="conversation",
                     parent_message_id="parent")
    with store.connect() as db:
        stranded = store.read_receipt(db, "text", "owner", "expired-text")
    assert stranded["_executing"] is True and stranded["_turn_reserved"] is True
    clock.now = stranded["_claim_until"] + 1
    # The real claim scanner converts only this expired, sent claim to
    # UNKNOWN; it does not clear the reserved original turn or resubmit it.
    assert admission.claim_next() is None

    backend = Mock()
    upstream = document(stranded)
    upstream["mapping"]["final-" + stranded["request_message_id"]]["message"]["content"]["parts"] = ["recovered expired final"]
    text.recovery_reader = lambda row: ConversationBindingService._read_text_request_result(
        backend, row, document=upstream,
    )
    admission.recoveries["text"] = text.read
    admission.recover_one()
    with store.connect() as db:
        recovered = store.read_receipt(db, "text", "owner", "expired-text")
    assert recovered["status"] == "succeeded" and recovered["_turn_reserved"] is False
    assert recovered["_executing"] is False and recovered["_claim_id"] is None
    product = ProductWorkService(text, SimpleNamespace(archive_product_work=lambda *args: None), clock=clock)
    assert product.close(IDENTITY, "expired-product")["state"] == "closing"


@pytest.mark.parametrize("extra", [{"_forward_protocol": "responses"}, {"_editable_task_id": "file"},
    {"_editable_kind": "spreadsheet"}, {"_editable": {"kind": "file"}}, {"_operation": "image"}])
def test_adoption_rejects_nonpure_legacy_text(runtime, extra):
    service, _, store, _, _, _, add = runtime
    original = add("legacy", product=None, **extra)
    with pytest.raises(WorkLifecycleError, match="LEGACY_REQUIRED"):
        service.adopt(IDENTITY, "P", [{"kind": "text", "request_id": "legacy", "step_ref": "text"}])
    with store.connect() as db:
        assert store.read_receipt(db, "text", "owner", "legacy") == original
        assert not state_snapshot(store, db)


@pytest.mark.parametrize("change_binding", [False, True])
def test_adoption_waits_for_archive_binding_lock_then_fresh_reads(runtime, monkeypatch, change_binding):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from services.account_service import account_service
    service, _, store, _, _, _, add = runtime
    add("legacy", product=None, client_conversation_id="client")
    lock = account_service.conversation_binding_lock("product-archive:account", "legacy")
    acquiring = threading.Event()
    class ObservedLock:
        def acquire(self, *args, **kwargs):
            acquiring.set()
            return lock.acquire(*args, **kwargs)
        def release(self):
            lock.release()
        def __enter__(self):
            self.acquire()
        def __exit__(self, *exc):
            self.release()
    monkeypatch.setattr(account_service, "conversation_binding_lock", lambda binding, client: ObservedLock())
    lock.acquire()  # An existing raw archive owns this across PATCH and readback.
    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(service.adopt, IDENTITY, "P", [{"kind": "text", "request_id": "legacy", "step_ref": "text"}])
        try:
            assert acquiring.wait(2)
            assert not future.done()
            # Adopter is not holding SQLite while waiting on the upstream lock.
            with store.transaction() as db:
                assert not state_snapshot(store, db)
                receipt = store.read_receipt(db, "text", "owner", "legacy")
                if change_binding:
                    receipt["provider_binding_id"] = "changed-binding"
                    store.write_receipt(db, "text", "owner", "legacy", receipt)
        finally:
            lock.release()
        if change_binding:
            with pytest.raises(WorkLifecycleError, match="BINDING_CHANGED"):
                future.result(timeout=2)
        else:
            assert future.result(timeout=2)["state"] == "active"


def test_completed_supersedes_child_does_not_prove_original_unknown_resolved(runtime):
    service, _, _, _, _, _, add = runtime
    add("original", product=None, status="failed", error_code="RESULT_UNRECOVERABLE", upstream_outcome="unknown",
        conversation="shared", request_message_id="original-request", recovery_reason="REQUEST_MESSAGE_NOT_FOUND")
    add("successor", product=None, conversation="shared", _supersedes_request_id="original", content="new result",
        _supersedes_input_hash="immutable-hash", _upstream_terminal=True)
    service.adopt(IDENTITY, "P", [{"kind": "text", "request_id": "original", "step_ref": "original"},
                                 {"kind": "text", "request_id": "successor", "step_ref": "successor"}])
    with pytest.raises(WorkLifecycleError, match="TURN_UNFINISHED"):
        service.close(IDENTITY, "P")


def test_physical_conversation_without_account_cannot_attach_or_adopt(runtime):
    from services.product_work import locking_product_conversations
    service, _, store, _, _, _, add = runtime
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_BINDING_REQUIRED"):
        with locking_product_conversations([{"conversation_id": "bound", "provider_binding_id": "binding"}]):
            pytest.fail("unknown physical account was not fenced")
    original = add("legacy", product=None, provider_account_identity=None)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_BINDING_REQUIRED"):
        service.adopt(IDENTITY, "P", [{"kind": "text", "request_id": "legacy", "step_ref": "text"}])
    with store.connect() as db:
        assert store.read_receipt(db, "text", "owner", "legacy") == original
        assert state_snapshot(store, db) == {}
    with locking_product_conversations([]), locking_product_conversations([{"status": "unknown"}]):
        pass


def _codex_bridge(*, origin, parent, parent_hash="immutable-hash", fallback=False):
    return {
        "_route": "codex", "_operation": "text", "_forward_protocol": "codex",
        "_codex_fallback_protocol": "codex-responses-v1",
        "_codex_origin_request_id": origin,
        "_codex_fallback_input_hash": parent_hash,
        "_codex_fallback_of" if fallback else "_codex_continue_of": parent,
    }


def test_rejected_chat_codex_successor_preserves_parent_and_archives_only_chat(runtime):
    service, lifecycle, store, _, calls, _, add = runtime
    original = add("chat-rejected", step="gallery", status="failed", conversation="chat-cursor",
                   error_code="MODEL_NOT_FOUND", upstream_outcome="rejected")
    fallback = add("codex-gallery", step="gallery", physical=False,
                   **_codex_bridge(origin="chat-rejected", parent="chat-rejected", fallback=True))
    assert original["status"] == "failed" and original["error_code"] == "MODEL_NOT_FOUND"
    assert fallback.get("conversation_id") is None

    closing = service.close(IDENTITY, "P")
    assert closing["state"] == "closing" and closing["archive"] == {"total": 1, "confirmed": 0, "pending": 1}
    assert lifecycle.process_one()
    assert calls == [("owner", "chat-rejected", True)]
    assert service.get(IDENTITY, "P")["state"] == "completed"
    with store.connect() as db:
        saved_parent = store.read_receipt(db, "text", "owner", "chat-rejected")
        saved_child = store.read_receipt(db, "text", "owner", "codex-gallery")
    assert saved_parent["status"] == "failed" and saved_parent["error_code"] == "MODEL_NOT_FOUND"
    assert saved_child.get("conversation_id") is None


def test_cursorless_rejected_chat_requires_valid_terminal_codex_child_and_allows_new_step(runtime):
    service, _, _, _, calls, _, add = runtime
    add("chat-rejected", step="gallery", status="failed", physical=False,
        error_code="MODEL_NOT_FOUND", upstream_outcome="rejected")
    fallback = add("codex-gallery", step="gallery", physical=False,
                   **_codex_bridge(origin="chat-rejected", parent="chat-rejected", fallback=True))
    add("codex-copy", step="copy", physical=False,
        **_codex_bridge(origin="chat-rejected", parent="codex-gallery"))

    projection = service.get(IDENTITY, "P")
    assert [(m["request_id"], m["step_ref"]) for m in projection["members"]] == [
        ("chat-rejected", "gallery"), ("codex-gallery", "gallery"), ("codex-copy", "copy"),
    ]
    completed = service.close(IDENTITY, "P")
    assert completed["state"] == "completed" and not completed["slot_held"]
    assert completed["archive"] == {"total": 0, "confirmed": 0, "pending": 0}
    assert calls == []
    assert fallback["_codex_fallback_of"] == "chat-rejected"


def test_cursorless_rejected_chat_unknown_codex_child_does_not_release_product(runtime):
    service, _, _, _, calls, _, add = runtime
    add("chat-rejected", step="gallery", status="failed", physical=False,
        error_code="MODEL_NOT_FOUND", upstream_outcome="rejected")
    add("codex-unknown", step="gallery", status="unknown", physical=False,
        **_codex_bridge(origin="chat-rejected", parent="chat-rejected", fallback=True))

    with pytest.raises(WorkLifecycleError, match="TURN_UNFINISHED"):
        service.close(IDENTITY, "P")
    assert service.get(IDENTITY, "P")["state"] == "active"
    assert calls == []


def test_get_keeps_one_query_only_snapshot_without_blocking_wal_writer(runtime, monkeypatch):
    """A product projection reads one point in time without claiming the writer lock."""
    service, _, store, _, _, _, add = runtime
    original = add("original", conversation="shared")

    # TaskStore itself does not choose WAL. Configure this isolated real SQLite
    # fixture explicitly so this test proves the configured WAL concurrency
    # behavior without claiming it for every deployment.
    with sqlite3.connect(store.path) as configured:
        assert configured.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"

    entered, release = threading.Event(), threading.Event()
    result, failure, observed = {}, [], {}
    projection = service._projection

    def paused_projection(db, parent):
        observed["query_only"] = db.execute("PRAGMA query_only").fetchone()[0]
        observed["in_transaction"] = db.in_transaction
        entered.set()  # _load has already established the read snapshot.
        assert release.wait(3)
        return projection(db, parent)

    monkeypatch.setattr(service, "_projection", paused_projection)

    def read_projection():
        try:
            result["value"] = service.get(IDENTITY, "P")
        except BaseException as exc:  # Preserve the thread exception for the assertion below.
            failure.append(exc)

    reader = threading.Thread(target=read_projection)
    reader.start()
    assert entered.wait(2)
    try:
        # Same physical conversation but a foreign product binding would be a
        # normal PRODUCT_WORK_CONVERSATION_CONFLICT on a later read. It must not
        # appear midway through this projection's deferred snapshot.
        foreign = copy.deepcopy(original)
        foreign.update(request_id="foreign", _product_work_key="product_work:foreign",
                       _product_work={"product_ref": "Q", "step_ref": "foreign"})
        with store.transaction(busy_timeout_secs=0.5) as db:
            db.execute("INSERT INTO requests VALUES(?,?,?,?)",
                       ("owner", "foreign", "foreign-hash", json.dumps(foreign)))
    finally:
        release.set()
        reader.join(timeout=3)

    assert not reader.is_alive() and not failure
    assert observed == {"query_only": 1, "in_transaction": True}
    assert "PRODUCT_WORK_CONVERSATION_CONFLICT" not in result["value"]["blocked_reasons"]
    assert "PRODUCT_WORK_CONVERSATION_CONFLICT" in service.get(IDENTITY, "P")["blocked_reasons"]


def test_get_in_delete_mode_reads_while_an_existing_writer_is_reserved(runtime, monkeypatch):
    service, _, store, _, _, _, add = runtime
    add("original")
    with store.connect() as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    connect = store._connect
    monkeypatch.setattr(store, "_connect", lambda **kwargs: connect(busy_timeout_secs=0.2))
    with store.transaction() as writer:
        store.set_runtime(writer, "uncommitted:writer", {"pending": True})
        # A read can see the last committed snapshot while another connection
        # owns RESERVED. BEGIN IMMEDIATE would instead contend with that writer.
        assert service.get(IDENTITY, "P")["state"] == "active"
    # DELETE-mode readers can still delay a writer's COMMIT. No test here claims
    # that a long projection permits concurrent commit outside WAL mode.


def test_get_read_snapshot_releases_transaction_when_projection_raises(runtime, monkeypatch):
    service, _, store, _, _, _, add = runtime
    add("original")

    def fail_projection(db, parent):
        assert db.execute("PRAGMA query_only").fetchone()[0] == 1
        assert db.in_transaction
        raise RuntimeError("projection failed")

    monkeypatch.setattr(service, "_projection", fail_projection)
    with pytest.raises(RuntimeError, match="projection failed"):
        service.get(IDENTITY, "P")

    # A new real writer proves the failed read released its deferred transaction.
    with store.transaction(busy_timeout_secs=0.2) as db:
        store.set_runtime(db, "product_work:test_after_failed_read", {"released": True})
