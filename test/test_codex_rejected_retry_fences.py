"""Execution and product fences for the original P -> rejected D -> retry E."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from services.conversation_binding_service import ConversationBindingError
from services.product_work import ProductWorkService
from services.request_context import current_request
from services.work_lifecycle import WorkLifecycleError, WorkLifecycleService
from test.test_codex_rejected_retry import (
    IDENTITY, retry_runtime, _prepare_rejected_fallback, _install_native_success, _read, _update,
)


IDS = ("original", "rejected-native", "retry-native")
RESULT_EVIDENCE = {"content": "late result", "result": None, "data": None,
                   "result_file_ids": [], "result_sediment_ids": []}
RETRY_MARKERS = ("_codex_rejected_retry_of", "_codex_origin_request_id",
                 "_codex_fallback_input_hash", "_codex_fallback_protocol")


def _queue(runtime, monkeypatch):
    _prepare_rejected_fallback(runtime)
    session = _install_native_success(runtime, monkeypatch)
    runtime.service.submit_codex_rejected_retry("owner", IDS[1], IDS[2], source="internal:listing")
    return session


def _complete(runtime, monkeypatch):
    session = _queue(runtime, monkeypatch)
    claimed = runtime.admission.claim_next()
    assert claimed and claimed.request_id == IDS[2]
    runtime.admission.execute(claimed)
    result = runtime.service.read("owner", IDS[2])
    assert result["status"] == "succeeded" and result["content"] == "saved retry result"
    assert len(session.calls) == 1
    return session


@pytest.mark.parametrize("already_claimed", [False, True])
@pytest.mark.parametrize("changes", [
    {"content": "late original result"},
    {"_recovery_paused": True},
    {"recovery_claim_id": "new-original-reader", "recovery_lease_until": 2000.0},
])
def test_changed_d_after_e_queued_never_sends_native(retry_runtime, monkeypatch, already_claimed, changes):
    h = retry_runtime
    session = _queue(h, monkeypatch)
    claimed = h.admission.claim_next() if already_claimed else None
    if already_claimed:
        assert claimed and claimed.request_id == IDS[2]
    _update(h, IDS[1], **changes)
    original = _read(h, IDS[1])
    claimed = claimed or h.admission.claim_next()
    if claimed:
        h.admission.execute(claimed)
    assert session.calls == []
    child = _read(h, IDS[2])
    assert child["status"] == "failed" and child["upstream_outcome"] == "not_sent"
    assert not child.get("send_count")
    assert h.admission.claim_next() is None
    assert _read(h, IDS[1]) == original


@pytest.mark.parametrize("rid", IDS)
@pytest.mark.parametrize("drift", ["source", "hash", "pause", "claim", "member"])
def test_changed_chain_evidence_cannot_complete_product(retry_runtime, monkeypatch, rid, drift):
    h = retry_runtime
    session = _complete(h, monkeypatch)
    product = ProductWorkService(h.service, SimpleNamespace(), clock=h.clock)
    key = _read(h, IDS[0])["_product_work_key"]
    if drift == "source":
        _update(h, rid, _source="internal:other", _scheduling_owner="internal:other")
    elif drift == "pause":
        _update(h, rid, _recovery_paused=True)
    elif drift == "claim":
        _update(h, rid, recovery_claim_id="active-read", recovery_lease_until=2000.0)
    else:
        with h.store.transaction() as db:
            if drift == "hash":
                db.execute("UPDATE requests SET request_hash=? WHERE owner=? AND id=?", ("changed", "owner", rid))
            else:
                parent = h.store.runtime(db, key)
                parent["members"] = [member for member in parent["members"] if member["request_id"] != rid]
                h.store.set_runtime(db, key, parent)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_"):
        product.close(IDENTITY, "synthetic-product", results_saved=True, outcome="completed")
    with h.store.connect() as db:
        parent = h.store.runtime(db, key)
    assert parent["state"] != "completed" and not parent.get("archive_keys")
    assert len(session.calls) == 1


@pytest.mark.parametrize("archive_drift", [None] + [
    (rid, drift) for rid in IDS for drift in ("hash", "pause", "claim", "member", "source")
] + [pytest.param((IDS[0], {"content": "late result"}), id="newproof-original-result"),
     pytest.param((IDS[1], {"result_file_ids": []}), id="newproof-rejected-result"),
     pytest.param((IDS[2], {"provider_account_identity": "different-actual-account"}), id="actual-provider-account")]
  + [pytest.param((IDS[2], {field: "changed"}), id="newproof-" + field) for field in RETRY_MARKERS])
def test_saved_retry_close_restart_resume_new_cycle_keeps_original_d(retry_runtime, monkeypatch, archive_drift):
    h = retry_runtime
    session = _complete(h, monkeypatch)
    originals = [_read(h, rid) for rid in IDS[:2]]
    with h.store.connect() as db:
        frozen_rows = list(db.execute("SELECT id,request_hash FROM requests WHERE id IN (?,?) ORDER BY id", IDS[:2]))
    product = ProductWorkService(h.service, SimpleNamespace(), clock=h.clock)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_RESULTS_SAVE_REQUIRED"):
        product.close(IDENTITY, "synthetic-product", results_saved=False)
    assert product.close(IDENTITY, "synthetic-product", results_saved=True)["state"] == "completed"
    # Native Codex has no Chat conversation; successful close must not invent
    # a Chat archive target merely to release the product.
    assert product.get(IDENTITY, "synthetic-product")["archive"]["total"] == 0
    archives = []
    def archive(owner, request_id, archived):
        archives.append((owner, request_id, archived))
        return {"request_id": request_id, "conversation_id": "new-cycle-chat", "archived": archived}
    text = SimpleNamespace(store=h.store, admission=None, archive_product_work=archive)
    restarted = ProductWorkService(text, SimpleNamespace(), clock=h.clock)
    lifecycle = WorkLifecycleService(text, SimpleNamespace(), clock=h.clock)
    assert restarted.resume(IDENTITY, "synthetic-product", expected_version=1)["version"] == 2
    sent = []
    def run(body, on_cursor):
        current_request.get().before_send()
        sent.append(deepcopy(body))
        return {"content": "new cycle saved result", "conversation_id": "new-cycle-chat",
                "parent_message_id": "new-cycle-answer", "provider_binding_id": "binding-0",
                "provider_account_identity": "account-0"}
    h.service.runner = run
    body = h.store.load_input(originals[0]["_input_ref"])
    h.service.submit("owner", {**body, "client_request_id": "new-cycle",
                     "product_work": {"product_ref": "synthetic-product", "step_ref": "new-cycle"}},
                     source="internal:listing")
    context = h.admission.claim_next()
    assert context and context.request_id == "new-cycle"
    h.admission.execute(context)
    assert _read(h, "new-cycle")["status"] == "succeeded" and len(sent) == 1
    closed = restarted.close(IDENTITY, "synthetic-product", results_saved=True, expected_version=2)
    assert closed["state"] == "closing" and closed["version"] == 2
    if archive_drift:
        rid, drift = archive_drift
        if isinstance(drift, dict):
            _update(h, rid, **drift)
        elif drift in {"hash", "member"}:
            with h.store.transaction() as db:
                if drift == "hash":
                    db.execute("UPDATE requests SET request_hash=? WHERE owner=? AND id=?", ("late-change", "owner", rid))
                else:
                    key = originals[0]["_product_work_key"]
                    parent = h.store.runtime(db, key)
                    parent["members"] = [member for member in parent["members"] if member["request_id"] != rid]
                    h.store.set_runtime(db, key, parent)
        else:
            changes = {"pause": {"_recovery_paused": True},
                       "claim": {"recovery_claim_id": "late-read", "recovery_lease_until": 2000.0},
                       "source": {"_source": "internal:other", "_scheduling_owner": "internal:other"}}[drift]
            _update(h, rid, **changes)
        assert not lifecycle.process_one()
        assert archives == []
        with h.store.connect() as db:
            assert h.store.runtime(db, originals[0]["_product_work_key"])["state"] == "closing"
        assert len(session.calls) == 1
        return
    assert lifecycle.process_one()
    assert restarted.get(IDENTITY, "synthetic-product")["state"] == "completed"
    assert archives == [("owner", "new-cycle", True)]
    assert [_read(h, rid) for rid in IDS[:2]] == originals
    with h.store.connect() as db:
        assert list(db.execute("SELECT id,request_hash FROM requests WHERE id IN (?,?) ORDER BY id", IDS[:2])) == frozen_rows
    assert len(session.calls) == 1


def test_failed_e_never_creates_grandchild_or_completes_product(retry_runtime, monkeypatch):
    h = retry_runtime
    session = _queue(h, monkeypatch)
    _update(h, IDS[2], status="failed", error_code="codex_request_rejected", upstream_outcome="rejected",
            _submission_started=True, _upstream_terminal=True, _executing=False, _turn_reserved=False)
    with h.store.connect() as db:
        before = list(db.execute("SELECT * FROM requests ORDER BY id"))
    with pytest.raises(ConversationBindingError):
        h.service.submit_codex_rejected_retry("owner", IDS[2], "grandchild", source="internal:listing")
    with h.store.connect() as db:
        assert list(db.execute("SELECT * FROM requests ORDER BY id")) == before
    product = ProductWorkService(h.service, SimpleNamespace(), clock=h.clock)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_"):
        product.close(IDENTITY, "synthetic-product", results_saved=True)
    assert h.admission.claim_next() is None and session.calls == []


def test_non_product_prepared_retry_requires_admission_before_persisting(retry_runtime):
    from services.codex_text_fallback import build_rejected_retry_body

    h = retry_runtime
    _prepare_rejected_fallback(h)
    with h.store.connect() as db:
        body = build_rejected_retry_body(h.store, db, "owner", IDS[1], IDS[2])
        before = list(db.execute("SELECT * FROM requests ORDER BY id"))
    body.pop("product_work")
    h.service.admission = None
    with pytest.raises(WorkLifecycleError, match="CODEX_FALLBACK_ADMISSION_REQUIRED") as error:
        h.service._submit_prepared("owner", body, source="internal:listing")
    assert error.value.status == 503
    with h.store.connect() as db:
        assert list(db.execute("SELECT * FROM requests ORDER BY id")) == before


@pytest.mark.parametrize("field", RESULT_EVIDENCE)
def test_newproof_original_result_presence_refuses_retry_registration(retry_runtime, monkeypatch, field):
    h = retry_runtime
    _prepare_rejected_fallback(h)
    session = _install_native_success(h, monkeypatch)
    _update(h, IDS[0], **{field: RESULT_EVIDENCE[field]})
    parent = _read(h, IDS[0])
    assert (parent["status"], parent["error_code"], parent["upstream_outcome"]) == (
        "failed", "MODEL_NOT_FOUND", "rejected")
    with h.store.connect() as db:
        before = list(db.execute("SELECT * FROM requests ORDER BY id"))
    with pytest.raises(ConversationBindingError):
        h.service.submit_codex_rejected_retry("owner", IDS[1], IDS[2], source="internal:listing")
    with h.store.connect() as db:
        assert list(db.execute("SELECT * FROM requests ORDER BY id")) == before
    assert session.calls == []


@pytest.mark.parametrize("rid", IDS[:2])
@pytest.mark.parametrize("field", RESULT_EVIDENCE)
def test_newproof_result_presence_on_rejected_ancestor_blocks_close(retry_runtime, monkeypatch, rid, field):
    h = retry_runtime
    session = _complete(h, monkeypatch)
    _update(h, rid, **{field: RESULT_EVIDENCE[field]})
    product = ProductWorkService(h.service, SimpleNamespace(), clock=h.clock)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_"):
        product.close(IDENTITY, "synthetic-product", results_saved=True)
    assert len(session.calls) == 1


@pytest.mark.parametrize("field", RETRY_MARKERS)
@pytest.mark.parametrize("clear_marker", [False, True])
def test_newproof_queued_receipt_marker_drift_stops_once_before_post(retry_runtime, monkeypatch, field, clear_marker):
    h = retry_runtime
    session = _queue(h, monkeypatch)
    saved = h.store.load_input(_read(h, IDS[2])["_input_ref"])
    if clear_marker:
        with h.store.transaction() as db:
            child = h.store.read_receipt(db, "text", "owner", IDS[2])
            child.pop(field)
            h.store.write_receipt(db, "text", "owner", IDS[2], child)
    else:
        _update(h, IDS[2], **{field: "changed"})
    claimed = h.admission.claim_next()
    if claimed:
        h.admission.execute(claimed)
    child = _read(h, IDS[2])
    assert child["status"] == "failed" and child["upstream_outcome"] == "not_sent"
    assert not child.get("send_count") and session.calls == []
    assert h.admission.claim_next() is None
    assert h.store.load_input(child["_input_ref"]) == saved


@pytest.mark.parametrize("field", RETRY_MARKERS)
@pytest.mark.parametrize("clear_marker", [False, True])
def test_newproof_success_receipt_marker_drift_blocks_close(retry_runtime, monkeypatch, field, clear_marker):
    h = retry_runtime
    session = _complete(h, monkeypatch)
    saved = h.store.load_input(_read(h, IDS[2])["_input_ref"])
    if clear_marker:
        with h.store.transaction() as db:
            child = h.store.read_receipt(db, "text", "owner", IDS[2])
            child.pop(field)
            h.store.write_receipt(db, "text", "owner", IDS[2], child)
    else:
        _update(h, IDS[2], **{field: "changed"})
    product = ProductWorkService(h.service, SimpleNamespace(), clock=h.clock)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_"):
        product.close(IDENTITY, "synthetic-product", results_saved=True)
    assert h.store.load_input(_read(h, IDS[2])["_input_ref"]) == saved
    assert len(session.calls) == 1


def test_actual_provider_account_drift_on_success_blocks_close(retry_runtime, monkeypatch):
    h = retry_runtime
    session = _complete(h, monkeypatch)
    before = _read(h, IDS[2])
    _update(h, IDS[2], provider_account_identity="different-actual-account")
    assert _read(h, IDS[2])["_requested_account_identity"] == before["_requested_account_identity"]
    product = ProductWorkService(h.service, SimpleNamespace(), clock=h.clock)
    with pytest.raises(WorkLifecycleError, match="PRODUCT_WORK_"):
        product.close(IDENTITY, "synthetic-product", results_saved=True)
    assert len(session.calls) == 1
