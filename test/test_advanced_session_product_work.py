"""Product lifecycle boundaries for an isolated ADVANCED replacement session."""
from types import SimpleNamespace

import pytest

from services.product_work import ProductWorkService, advanced_session_capacity_originals, state_snapshot
from services.work_lifecycle import WorkLifecycleError
from test.test_advanced_session_successor import advanced as advanced_fixture
from test.test_no_final_successor import product_no_final


def _product_service(h, calls):
    def archive(owner, request_id, archived):
        calls.append((owner, request_id, archived))
        with h.service.store.connect() as db:
            receipt = h.service.store.read_receipt(db, "text", owner, request_id)
        return {"request_id": request_id, "conversation_id": receipt["conversation_id"], "archived": archived}

    text = SimpleNamespace(store=h.service.store, admission=None, archive_product_work=archive)
    return ProductWorkService(text, SimpleNamespace(archive_product_work=archive), clock=h.clock)


def _saved_child(h):
    h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    with h.service.store.transaction() as db:
        original = h.service.store.read_receipt(db, "text", "owner", "old")
        child = h.service.store.read_receipt(db, "text", "owner", "isolated")
        parent = h.service.store.runtime(db, original["_product_work_key"])
        gallery = h.service.store.read_receipt(db, "text", "owner", "gallery")
        gallery.update(upstream_outcome="completed", upstream_unfinished=False,
                       error_code=None, recovery_reason=None, _upstream_terminal=True,
                       _executing=False, _turn_reserved=False, _claim_id=None, _claim_until=None)
        h.service.store.write_receipt(db, "text", "owner", "gallery", gallery)
        # Keep the already saved gallery as an owned old-session member: close
        # must retain it but never archive that old human-discussion session.
        parent["members"].append({"kind": "text", "owner": "owner", "request_id": "gallery", "step_ref": "gallery"})
        h.service.store.set_runtime(db, parent["key"], parent)
        child.update(provider_binding_id="new-binding", provider_account_identity=original["provider_account_identity"],
                     conversation_id="new-chat", parent_message_id="new-final", request_message_id="new-user",
                     request_parent_message_id="new-parent", _submission_parent_message_id="new-parent",
                     status="succeeded", upstream_outcome="completed", _upstream_terminal=True,
                     _submission_started=True, _last_sent_sequence=1, _executing=False,
                     _turn_reserved=False, _claim_id=None, _claim_until=None, content="new product result")
        h.service.store.write_receipt(db, "text", "owner", "isolated", child)
    return original, gallery


def test_advanced_new_session_uses_one_turn_and_archives_only_new_session(advanced_fixture):
    h = advanced_fixture
    # Before the child obtains its new provider binding, its immutable proof is
    # enough to transfer the old local slot for normal FIFO claiming.
    h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    with h.service.store.transaction() as db:
        old = h.service.store.read_receipt(db, "text", "owner", "old")
        parent = h.service.store.runtime(db, old["_product_work_key"])
        # The fixture turns an already-sent root into UNKNOWN directly; model
        # the durable active parent that the real claim path would have kept.
        parent.update(state="active", slot_held=True,
                      admission_control={"state": "active", "version": 0, "slot_held": True})
        h.service.store.set_runtime(db, parent["key"], parent)
        released = advanced_session_capacity_originals(h.service.store, db, old["_product_work_key"], h.clock())
    assert released == {("owner", "old")}

    # Reuse that registered child rather than constructing another successor.
    original, gallery = _saved_child(h)
    calls = []
    product = _product_service(h, calls)
    identity = {"id": "owner", "role": "admin", "_trusted_source": "internal:listing"}
    closing = product.close(identity, "fixture-product", results_saved=True)

    assert closing["state"] == "closing"
    with h.service.store.connect() as db:
        parent = next(iter(state_snapshot(h.service.store, db).values()))
        targets = [h.service.store.runtime(db, key) for key in parent["archive_keys"]]
        old_after = h.service.store.read_receipt(db, "text", "owner", "old")
        gallery_after = h.service.store.read_receipt(db, "text", "owner", "gallery")
    assert [(target["last_request_id"], target["target"]["conversation_id"]) for target in targets] == [("isolated", "new-chat")]
    assert old_after == original and gallery_after == gallery
    assert calls == []


def test_ordinary_fresh_binding_cannot_waive_old_unknown(advanced_fixture):
    h = advanced_fixture
    with h.service.store.connect() as db:
        old = h.service.store.read_receipt(db, "text", "owner", "old")
        body = h.service.store.load_input(old["_input_ref"])
    normal = {**body, "client_request_id": "ordinary-new-session", "client_conversation_id": "ordinary-client",
              "provider_binding_id": "ordinary-binding", "provider_account_identity": old["provider_account_identity"],
              "conversation_id": "ordinary-chat", "parent_message_id": "ordinary-parent",
              "product_work": {**body["product_work"], "step_ref": "ordinary"}}
    h.service.submit("owner", normal, source="internal:listing")
    with h.service.store.transaction() as db:
        child = h.service.store.read_receipt(db, "text", "owner", "ordinary-new-session")
        child.update(status="succeeded", upstream_outcome="completed", _upstream_terminal=True,
                     _submission_started=True, _executing=False, _turn_reserved=False,
                     _claim_id=None, _claim_until=None, content="unrelated later result")
        h.service.store.write_receipt(db, "text", "owner", "ordinary-new-session", child)
    product = _product_service(h, [])
    identity = {"id": "owner", "role": "admin", "_trusted_source": "internal:listing"}
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_TURN_UNFINISHED"):
        product.close(identity, "fixture-product", results_saved=True)


def test_live_old_root_never_donates_capacity_to_advanced_child(advanced_fixture):
    h = advanced_fixture
    h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    with h.service.store.transaction() as db:
        old = h.service.store.read_receipt(db, "text", "owner", "old")
        parent = h.service.store.runtime(db, old["_product_work_key"])
        parent.update(state="active", slot_held=True,
                      admission_control={"state": "active", "version": 0, "slot_held": True})
        h.service.store.set_runtime(db, parent["key"], parent)
        old.update(_executing=True, _turn_reserved=True, _claim_id="still-live", _claim_until=h.clock() + 60)
        h.service.store.write_receipt(db, "text", "owner", "old", old)
        assert advanced_session_capacity_originals(h.service.store, db, old["_product_work_key"], h.clock()) == set()


def test_failed_advanced_child_never_falls_back_to_archiving_old_discussion(advanced_fixture):
    h = advanced_fixture
    h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    with h.service.store.transaction() as db:
        old = h.service.store.read_receipt(db, "text", "owner", "old")
        child = h.service.store.read_receipt(db, "text", "owner", "isolated")
        old.update(status="succeeded", upstream_outcome="completed", _upstream_terminal=True,
                   content="late original result", error_code=None, recovery_reason=None,
                   _executing=False, _turn_reserved=False)
        child.update(status="failed", upstream_outcome="not_sent", error_code="PREFLIGHT_REJECTED",
                     _submission_started=False, provider_binding_id=None, conversation_id=None,
                     parent_message_id=None, content=None, _executing=False, _turn_reserved=False)
        h.service.store.write_receipt(db, "text", "owner", "old", old)
        h.service.store.write_receipt(db, "text", "owner", "isolated", child)
    product = _product_service(h, [])
    identity = {"id": "owner", "role": "admin", "_trusted_source": "internal:listing"}
    closed = product.close(identity, "fixture-product", results_saved=True, outcome="failed")
    assert closed["state"] == "failed" and closed["cleanup_pending"] is False
    with h.service.store.connect() as db:
        parent = next(iter(state_snapshot(h.service.store, db).values()))
    assert parent["archive_keys"] == []


def test_invalid_advanced_marker_cannot_authorize_old_archive(advanced_fixture):
    h = advanced_fixture
    # A tampered marker is not permission to revive the old archive target.
    h.service.submit_workbench_derived_input("owner", "old", h.payload, source="internal:listing")
    with h.service.store.transaction() as db:
        old = h.service.store.read_receipt(db, "text", "owner", "old")
        child = h.service.store.read_receipt(db, "text", "owner", "isolated")
        old.update(status="succeeded", upstream_outcome="completed", _upstream_terminal=True,
                   content="late original result", error_code=None, recovery_reason=None,
                   _executing=False, _turn_reserved=False)
        child.update(status="failed", upstream_outcome="not_sent", error_code="PREFLIGHT_REJECTED",
                     _submission_started=False, _executing=False, _turn_reserved=False)
        child["_advanced_session"]["original_conversation_id"] = "tampered"
        h.service.store.write_receipt(db, "text", "owner", "old", old)
        h.service.store.write_receipt(db, "text", "owner", "isolated", child)
    product = _product_service(h, [])
    identity = {"id": "owner", "role": "admin", "_trusted_source": "internal:listing"}
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_TURN_UNFINISHED"):
        product.close(identity, "fixture-product", results_saved=True, outcome="failed")
