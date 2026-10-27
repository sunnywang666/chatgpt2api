"""Opt-in product ownership over original text/image receipts in task_runtime.

A product retains ownership until its durable conversation archives are confirmed,
except for a proved no-conversation model rejection returned to the queue.
The legacy slot_held flag tracks that ownership, not a generation-capacity unit.
Each actual request is admitted against account resources by PoolAdmission.
No result, input hash, request sequence or ordinary work identity is replaced.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import ExitStack, contextmanager

from services.work_lifecycle import WorkLifecycleError, _blocked, _same_conversation


def _ref(value):
    if (not isinstance(value, str) or not value.strip() or value != value.strip() or len(value) > 200
            or any(ord(c) < 32 for c in value)):
        raise WorkLifecycleError("PRODUCT_WORK_REFERENCE_INVALID", 400)
    return value


def normalize_product_work(value):
    if value is None:
        return None
    if (not isinstance(value, dict) or not {"product_ref", "step_ref"} <= set(value)
            or set(value) - {"product_ref", "step_ref", "fairness"}):
        raise WorkLifecycleError("PRODUCT_WORK_INVALID", 400)
    result = {k: _ref(value[k]) for k in ("product_ref", "step_ref")}
    if "fairness" in value:
        fair = value["fairness"]
        keys = {"organization_id", "subject_id", "root_workflow_id"}
        if (not isinstance(fair, dict) or set(fair) != keys
                or any(not isinstance(v, str) or not v or len(v) > 200
                       or any(ord(c) < 33 or ord(c) == 127 for c in v) for v in fair.values())):
            raise WorkLifecycleError("PRODUCT_WORK_FAIRNESS_INVALID", 400)
        result["fairness"] = dict(fair)
    return result


def _metadata(parent, step_ref):
    metadata = {"product_ref": parent["product_ref"], "step_ref": step_ref}
    if parent.get("fairness") is not None:
        metadata["fairness"] = parent["fairness"]
    return metadata


def fair_subject(parent):
    """Use the same org/person lane as company requests, never the service key."""
    fair = parent.get("fairness")
    if fair is None:
        return None
    normalize_product_work(_metadata(parent, "fairness-validation"))
    identity = json.dumps([fair["organization_id"], fair["subject_id"]],
                          separators=(",", ":"), ensure_ascii=True)
    return "company:" + hashlib.sha256(identity.encode()).hexdigest()


def require_product_source(identity):
    source = identity.get("_trusted_source")
    if identity.get("role") != "admin" or source not in {"internal:wb-to-ozon", "internal:ozon-to-wb", "internal:listing", "internal:content"}:
        raise WorkLifecycleError("PRODUCT_WORK_SOURCE_REQUIRED", 403)
    return source


def _physical_lock_keys(receipts):
    if any(r.get("conversation_id") and not r.get("provider_account_identity") for r in receipts):
        raise WorkLifecycleError("PRODUCT_WORK_BINDING_REQUIRED")
    return sorted({(r["provider_account_identity"], r["conversation_id"]) for r in receipts
                   if r.get("provider_account_identity") and r.get("conversation_id")})


@contextmanager
def locking_product_conversations(receipts, *, budget_seconds=60.0):
    """Serialize ownership and raw archive for a physical conversation.

    Call outside SQLite transactions. Reuse the account service's in-process
    mutex map, with a separate namespace from generation's client cursor locks.
    The default remains the normal 60-second ProductWork lock budget; archive
    execution passes its already bounded archive-chain budget explicitly.
    """
    from services.account_service import account_service
    from services.request_context import bounded_lock, operation_deadline
    if (type(budget_seconds) not in (int, float) or not math.isfinite(budget_seconds)
            or budget_seconds <= 0):
        raise ValueError("finite product conversation lock budget required")
    keys = _physical_lock_keys(list(receipts))
    with operation_deadline(time.monotonic() + float(budget_seconds)), ExitStack() as stack:
        locks = [account_service.conversation_binding_lock("product-archive:" + account, conversation)
                 for account, conversation in keys]
        seen = set()
        for lock in locks:
            if id(lock) not in seen:
                stack.enter_context(bounded_lock(lock))
                seen.add(id(lock))
        yield


def _has_physical(receipt):
    return bool(receipt.get("conversation_id") and (receipt.get("provider_account_identity") or receipt.get("provider_binding_id")))


def _source(receipt):
    return receipt.get("_scheduling_owner") or receipt.get("_source")


_CANCELLED_UNSENT_STEP = "_product_unsent_cancelled"


def _cancel_unsent_eligible(receipt):
    """Keep this narrow: it is a local withdrawal before any model turn."""
    if not _unsent(receipt):
        return False
    if receipt.get("_submission_started") is not False or receipt.get("_turn_reserved") is not False:
        return False
    if receipt.get("_work_key") or any(receipt.get(key) for key in (
            "_recovery_paused", "_recovery_suppressed", "_completion", "_completion_of",
            "_retry_cursor", "_same_session_retry_of", "_previous_request_id",
            "_terminal_empty_correction_of", "_forward_protocol", "_forward")):
        return False
    if any(key.startswith(("_supersedes", "_codex", "recovery_", "_recovery")) for key in receipt):
        return False
    timeline = receipt.get("_execution_timeline")
    if timeline is not None and (not isinstance(timeline, list)
                                 or any(not isinstance(item, dict) or item.get("stage") != "accepted" for item in timeline)):
        return False
    if any(key in receipt for key in ("content", "result", "data", "result_file_ids", "result_sediment_ids",
                                      "response_id", "original_upstream_response_ids", "original_upstream_error",
                                      "original_http_status", "original_exception_category", "original_failure_phase",
                                      "original_upstream_request_stage", "finished_at")):
        return False
    return not any(receipt.get(key) for key in (
        "_attempt_started_at", "_attempt_finished_at", "_last_sent_sequence", "_send_sequence",
        "_attempt_reason", "upstream_submission_started", "_upstream_terminal",
        "_execution_wait_ended_at"))


def _input_matches_cancelled_member(store, owner, member, receipt, request_hash):
    """Check only fields that the immutable original actually supplied."""
    try:
        from services.conversation_binding_service import ConversationBindingError
        from services.text_task_service import TextTaskService
        body = store.load_input(receipt["_input_ref"])
        if TextTaskService._submission_identity(owner, body) != (member["request_id"], request_hash):
            return False
    except (OSError, ValueError, TypeError, KeyError, ConversationBindingError):
        return False
    metadata = receipt.get("_product_work")
    if body.get("product_work") != metadata:
        return False
    for field in ("provider_binding_id", "provider_account_identity", "client_conversation_id",
                  "conversation_id", "parent_message_id"):
        if field in body and body[field] != receipt.get(field):
            return False
    return True


def _closed_cancellation_cycle(store, db, parent, member, version):
    """A cancelled step remains a resolved historical member after resume."""
    closures = [row for row in parent.get("previous_closures", [])
                if isinstance(row, dict) and row.get("version") == version]
    if len(closures) != 1:
        return False
    closure = closures[0]
    if closure.get("outcome") != "completed" or closure.get("results_saved") is not True:
        return False
    members = closure.get("members")
    current = {(item.get("kind"), item.get("owner"), item.get("request_id")): item
               for item in parent.get("members", []) if isinstance(item, dict)}
    if (not isinstance(members, list) or not members or sum(item == member for item in members) != 1
            or any(not isinstance(item, dict)
                   or current.get((item.get("kind"), item.get("owner"), item.get("request_id"))) != item
                   for item in members)):
        return False
    keys = closure.get("archive_keys")
    if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
        return False
    originals = {}
    for item in members:
        receipt = store.read_receipt(db, item["kind"], item["owner"], item["request_id"])
        if not receipt:
            return False
        originals[(item["kind"], item["owner"], item["request_id"])] = receipt
    for key in keys:
        child = store.runtime(db, key)
        archive = (child or {}).get("archive") or {}
        original = originals.get(((child or {}).get("kind"), (child or {}).get("owner"),
                                  (child or {}).get("last_request_id")))
        if (not child or child.get("key") != key
                or child.get("_product_archive_parent") != parent["key"]
                or child.get("_product_archive_version", 1) != version or child.get("version") != version
                or archive.get("version") != version or archive.get("status") != "confirmed"
                or archive.get("desired") is not True or archive.get("claim")
                or archive.get("request_id") != child.get("last_request_id") or not original
                or any(not original.get(field) or child.get("target", {}).get(field) != original[field]
                       for field in ("provider_binding_id", "provider_account_identity", "conversation_id"))):
            return False
    return True


def _cancelled_unsent_step(store, db, parent, member, receipt, *, allow_historical=True):
    """Revalidate the one durable local cancellation proof before consuming it."""
    proof = receipt.get(_CANCELLED_UNSENT_STEP)
    if not isinstance(proof, dict) or set(proof) != {"parent_key", "parent_version", "request_hash", "cancelled_at"}:
        return False
    version = parent.get("version", 1)
    if (member.get("kind") != "text" or member.get("owner") != parent.get("owner")
            or member.get("request_id") != receipt.get("request_id")
            or receipt.get("_product_work_key") != parent.get("key")
            or _source(receipt) != parent.get("source")
            or receipt.get("_product_work") != _metadata(parent, member.get("step_ref"))
            or receipt.get("status") != "cancelled"
            or receipt.get("error_code") != "PRODUCT_WORK_STEP_CANCELLED"
            or receipt.get("upstream_outcome") != "not_sent"
            or receipt.get("upstream_unfinished") is not False
            or proof.get("parent_key") != parent.get("key")
            or type(proof.get("parent_version")) is not int
            or not isinstance(proof.get("request_hash"), str) or not proof["request_hash"]
            or type(proof.get("cancelled_at")) not in (int, float)
            or not math.isfinite(proof["cancelled_at"])):
        return False
    proof_version = proof["parent_version"]
    if proof_version != version:
        if not allow_historical or proof_version > version or not _closed_cancellation_cycle(
                store, db, parent, member, proof_version):
            return False
    row = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?",
                     (member["owner"], member["request_id"])).fetchone()
    if not row or row[0] != proof["request_hash"]:
        return False
    if not _input_matches_cancelled_member(store, member["owner"], member, receipt, row[0]):
        return False
    original = dict(receipt)
    original.pop(_CANCELLED_UNSENT_STEP, None)
    original.pop("error_code", None)
    original["status"] = "queued"
    return _cancel_unsent_eligible(original)


def cancelled_unsent_public(store, db, owner, request_hash, receipt):
    """Return the small caller-verifiable cancellation record, or nothing."""
    key = receipt.get("_product_work_key")
    parent = store.runtime(db, key) if isinstance(key, str) else None
    if not parent or parent.get("owner") != owner:
        return None
    member = next((item for item in parent.get("members", [])
                   if item.get("kind") == "text" and item.get("owner") == owner
                   and item.get("request_id") == receipt.get("request_id")), None)
    if not member or not _cancelled_unsent_step(store, db, parent, member, receipt):
        return None
    proof = receipt[_CANCELLED_UNSENT_STEP]
    if proof["request_hash"] != request_hash:
        return None
    return {"request_id": receipt["request_id"], "product_ref": parent["product_ref"],
            "parent_version": proof["parent_version"], "request_hash": proof["request_hash"],
            "cancelled_at": proof["cancelled_at"], "code": "PRODUCT_WORK_STEP_CANCELLED",
            "upstream_outcome": "not_sent"}


def _key(source, owner, ref):
    raw = json.dumps([source, owner, ref], separators=(",", ":"))
    return "product_work:" + hashlib.sha256(raw.encode()).hexdigest()


def state_snapshot(store, db):
    return {key: json.loads(raw) for key, raw in db.execute(
        "SELECT name,value FROM task_runtime WHERE name LIKE 'product_work:%'")}


def _save(store, db, parent):
    store.set_runtime(db, parent["key"], parent)


def admission_control(parent):
    """Return the durable product-admission state without reusing lifecycle slots.

    Older parents deliberately have no admission marker.  They are not counted
    merely because they retained a lifecycle/archive owner; the admission
    transaction recognizes a genuinely live legacy claim separately.
    """
    value = parent.get("admission_control")
    if (isinstance(value, dict) and value.get("state") in {"active", "pausing", "paused"}
            and type(value.get("version")) is int and value["version"] >= 0
            and type(value.get("slot_held")) is bool):
        return dict(value)
    return {"state": "active", "version": 0, "slot_held": False}


def admission_slot_held(parent):
    return admission_control(parent)["slot_held"]


def _live_product_claim(receipt, now):
    """True only for an unexpired executing/reserved generation, never history."""
    until = receipt.get("_claim_until")
    return bool(receipt.get("status") == "running" and receipt.get("_claim_id")
                and isinstance(until, (int, float)) and not isinstance(until, bool) and until > now
                and (receipt.get("_executing") or receipt.get("_turn_reserved")))


def legacy_live_admission(parent, receipts, now):
    """Compatibility count for a pre-marker parent that is actually executing."""
    return ("admission_control" not in parent and parent.get("state") == "active"
            and any(_live_product_claim(receipt, now) for receipt in receipts))


def reconcile_paused_admission_slots(store, db, receipts=None, *, now=None):
    """Finish an explicit pause once its already-live lease has ended.

    This is intentionally called from the existing admission transaction.  It
    never releases active/UNKNOWN work and it never infers a pause from old
    recovery flags or archive state.
    """
    now = time.time() if now is None else float(now)
    rows = list(store.receipts(db)) if receipts is None else list(receipts)
    by_parent = {}
    for _kind, _owner, _request_id, receipt in rows:
        if isinstance(receipt, dict) and isinstance(receipt.get("_product_work_key"), str):
            by_parent.setdefault(receipt["_product_work_key"], []).append(receipt)
    changed = 0
    for key, parent in state_snapshot(store, db).items():
        control = admission_control(parent)
        if ("admission_control" not in parent or control["state"] != "pausing"
                or control["slot_held"] is not True):
            continue
        if any(_live_product_claim(receipt, now) for receipt in by_parent.get(key, ())):
            continue
        parent["admission_control"] = {**control, "state": "paused", "slot_held": False}
        _save(store, db, parent)
        changed += 1
    return changed


def _attached(store, db, receipt):
    key = receipt.get("_product_work_key")
    if not key:
        return None
    parent = store.runtime(db, key)
    if not parent or parent.get("source") != _source(receipt):
        raise WorkLifecycleError("PRODUCT_WORK_BINDING_CONFLICT")
    return parent


def before_send(store, db, receipt):
    parent = _attached(store, db, receipt)
    if not parent and _has_physical(receipt):
        for _, _, _, other in store.receipts(db):
            if other.get("_product_work_key") and _same_conversation(receipt, other):
                raise WorkLifecycleError("PRODUCT_WORK_REQUIRED")
    if parent:
        member = next((item for item in parent.get("members", [])
                       if item.get("kind") == "text" and item.get("owner") == parent.get("owner")
                       and item.get("request_id") == receipt.get("request_id")), None)
        if receipt.get(_CANCELLED_UNSENT_STEP):
            if not member or not _cancelled_unsent_step(store, db, parent, member, receipt):
                raise WorkLifecycleError("PRODUCT_WORK_CANCEL_STEP_PROOF_INVALID")
            raise WorkLifecycleError("PRODUCT_WORK_STEP_CANCELLED")
    control = admission_control(parent) if parent else None
    if parent and (parent["state"] != "active" or not parent.get("slot_held")
                   or control["state"] != "active" or not control["slot_held"]):
        raise WorkLifecycleError("PRODUCT_WORK_NOT_ACTIVE")
    if parent:
        _check_physical(store, db, receipt, parent["key"])
    return parent


def record_slot(store, db, receipt, held=True):
    parent = _attached(store, db, receipt)
    if parent and held and parent["state"] not in {"queued", "active"}:
        raise WorkLifecycleError("PRODUCT_WORK_NOT_ACTIVE")
    if parent and held:
        control = admission_control(parent)
        if control["state"] != "active":
            raise WorkLifecycleError("PRODUCT_WORK_ADMISSION_PAUSED")
        _check_physical(store, db, receipt, parent["key"])
    if parent and held:
        parent.update(state="active", slot_held=True,
                      admission_control={**control, "slot_held": True})
        _save(store, db, parent)
    # Ordinary turn completion/failure does not release product ownership.
    return parent


def _check_physical(store, db, receipt, key):
    if not _has_physical(receipt):
        return
    for _, _, _, other in store.receipts(db):
        if (_same_conversation(receipt, other) and other.get("_product_work_key")
                and other["_product_work_key"] != key):
            raise WorkLifecycleError("PRODUCT_WORK_CONVERSATION_CONFLICT")


def _codex_product_bridge(receipt):
    forward = receipt.get("_forward_protocol")
    fallback = receipt.get("_codex_fallback_of")
    continuation = receipt.get("_codex_continue_of")
    rejected_retry = receipt.get("_codex_rejected_retry_of")
    return bool(
        receipt.get("_route") == "codex"
        and forward == "codex"
        and receipt.get("_codex_fallback_protocol") == "codex-responses-v1"
        and isinstance(receipt.get("_codex_origin_request_id"), str)
        and bool(receipt.get("_codex_origin_request_id"))
        and sum(bool(marker) for marker in (fallback, continuation, rejected_retry)) == 1
        and isinstance(receipt.get("_codex_fallback_input_hash"), str)
    )


def _codex_bridge_successor(store, db, owner, previous, receipt):
    rid = previous.get("request_id") or previous.get("id")
    if not rid or not _codex_product_bridge(receipt):
        return False
    saved = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", (owner, rid)).fetchone()
    if not saved or saved[0] != receipt.get("_codex_fallback_input_hash"):
        return False
    if receipt.get("_codex_fallback_of") == rid:
        return bool(
            previous.get("status") == "failed"
            and previous.get("error_code") == "MODEL_NOT_FOUND"
            and previous.get("upstream_outcome") == "rejected"
            and previous.get("_submission_started") is True
            and previous.get("_route", "chat") == "chat"
            and previous.get("_operation", "text") == "text"
            and not previous.get("_forward_protocol")
            and receipt.get("_codex_origin_request_id") == rid
        )
    if receipt.get("_codex_rejected_retry_of") == rid:
        return bool(
            previous.get("_codex_fallback_of")
            and not previous.get("_codex_continue_of")
            and not previous.get("_codex_rejected_retry_of")
            and _codex_product_bridge(previous)
            and receipt.get("_codex_origin_request_id") == previous.get("_codex_origin_request_id")
        )
    return bool(
        receipt.get("_codex_continue_of") == rid
        and _codex_product_bridge(previous)
        and receipt.get("_codex_origin_request_id") == previous.get("_codex_origin_request_id")
    )


def _rejected_chat_replaced(store, db, owner, receipt, members):
    rid = receipt.get("request_id") or receipt.get("id")
    return bool(
        rid
        and receipt.get("status") == "failed"
        and receipt.get("error_code") == "MODEL_NOT_FOUND"
        and receipt.get("upstream_outcome") == "rejected"
        and receipt.get("_submission_started") is True
        and receipt.get("_route", "chat") == "chat"
        and any(_codex_bridge_successor(store, db, owner, receipt, child)
                for _, child in members if child.get("_codex_fallback_of") == rid)
    )


def _successor(store, db, owner, previous, receipt):
    from services.generation_completion import same_session_retry
    if same_session_retry(previous, receipt):
        return True
    rid = previous.get("request_id") or previous.get("id")
    if receipt.get("_terminal_empty_correction_of") == rid:
        from services.text_task_service import TextTaskService
        proof = TextTaskService._verified_retryable_empty(previous)
        return bool(proof and receipt.get("_previous_request_id") == rid
                    and receipt.get("_submission_parent_message_id") == proof["retry_parent_message_id"]
                    and receipt.get("_work_key") == previous.get("_work_key")
                    and _same_conversation(previous, receipt))
    if receipt.get("_supersedes_request_id") == rid:
        from services.text_task_service import TextTaskService
        saved = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", (owner, rid)).fetchone()
        return bool(saved and saved[0] == receipt.get("_supersedes_input_hash")
                    and TextTaskService._supersede_order_link(previous, receipt))
    if _codex_bridge_successor(store, db, owner, previous, receipt):
        return True
    return False


def ensure_product_work(store, db, kind, owner, request_id, receipt, product_work, *, source=None):
    metadata = normalize_product_work(product_work)
    if metadata is None:
        if not _has_physical(receipt):
            return None
        for _, _, _, other in store.receipts(db):
            if other.get("_product_work_key") and _same_conversation(receipt, other):
                raise WorkLifecycleError("PRODUCT_WORK_REQUIRED")
        return None
    if (receipt.get("_work_key") or receipt.get("_public_session_ref") or receipt.get("_image_thread")
            or receipt.get("_route", receipt.get("route", "chat")) != "chat"
            or receipt.get("_forward_protocol") or receipt.get("_editable_task_id") or receipt.get("_editable_kind")
            or receipt.get("_editable") or receipt.get("_forward")
            or kind == "text" and receipt.get("_operation", "text") != "text") and not _codex_product_bridge(receipt):
        raise WorkLifecycleError("PRODUCT_WORK_LEGACY_REQUIRED")
    owner = str(owner)
    source = source or _source(receipt)
    if not source or source != _source(receipt) or kind not in {"text", "image"}:
        raise WorkLifecycleError("PRODUCT_WORK_SOURCE_CONFLICT")
    if metadata.get("fairness") is not None and source not in {
            "internal:listing", "internal:content", "internal:wb-to-ozon", "internal:ozon-to-wb"}:
        raise WorkLifecycleError("PRODUCT_WORK_SOURCE_REQUIRED", 403)
    key = _key(source, owner, metadata["product_ref"])
    if receipt.get("_product_work_key") not in {None, key}:
        raise WorkLifecycleError("PRODUCT_WORK_BINDING_CONFLICT")
    _check_physical(store, db, receipt, key)
    parent = store.runtime(db, key)
    member = {"kind": kind, "owner": owner, "request_id": request_id, "step_ref": metadata["step_ref"]}
    if parent:
        if parent["owner"] != owner or parent["source"] != source:
            raise WorkLifecycleError("PRODUCT_WORK_OWNER_CONFLICT")
        if parent.get("fairness") != metadata.get("fairness"):
            raise WorkLifecycleError("PRODUCT_WORK_FAIRNESS_CONFLICT")
        existing = [m for m in parent["members"] if (m["kind"], m["owner"], m["request_id"]) == (kind, owner, request_id)]
        if existing:
            if existing != [member]:
                raise WorkLifecycleError("PRODUCT_WORK_STEP_CONFLICT")
            receipt.update(_product_work_key=key, _product_work=metadata)
            return parent
        if parent["state"] not in {"queued", "active"}:
            raise WorkLifecycleError("PRODUCT_WORK_NOT_ACTIVE")
        same_step = [m for m in parent["members"] if m["step_ref"] == member["step_ref"]]
        if same_step:
            while same_step:
                old = same_step[-1]
                previous = store.read_receipt(db, old["kind"], old["owner"], old["request_id"])
                if (old["kind"] == "text" and previous
                        and _cancelled_unsent_step(store, db, parent, old, previous)):
                    same_step.pop()
                    continue
                break
            if not same_step:
                raise WorkLifecycleError("PRODUCT_WORK_STEP_CONFLICT")
            old = same_step[-1]
            previous = store.read_receipt(db, old["kind"], old["owner"], old["request_id"])
            if old["kind"] != kind or not previous or not _successor(store, db, owner, previous, receipt):
                raise WorkLifecycleError("PRODUCT_WORK_STEP_CONFLICT")
    else:
        parent = {"key": key, "owner": owner, "source": source, "product_ref": metadata["product_ref"],
                  "state": "queued", "slot_held": False, "members": [], "archive_keys": [],
                  "results_saved": False, "outcome": None, "cleanup_pending": False,
                  "created_at": time.time(),
                  "admission_control": {"state": "active", "version": 0, "slot_held": False}}
        if metadata.get("fairness") is not None:
            parent["fairness"] = metadata["fairness"]
    parent["members"].append(member)
    receipt.update(_product_work_key=key, _product_work=metadata)
    _save(store, db, parent)
    return parent


def _claims(receipt):
    if any(receipt.get(k) for k in ("_executing", "recovery_claim_id", "_recovery_claim_id", "_turn_reserved")):
        return True
    # PoolAdmission keeps the original claim ID for outcome attribution after
    # execute() exits. A terminal receipt with both explicit retirement flags
    # cannot pass its send guard (which requires status=running). An ambiguous
    # claim, or a running/queued task, still blocks regardless of wall-clock TTL.
    retired = (receipt.get("status") in {"succeeded", "success", "failed", "error", "cancelled"}
               and receipt.get("_executing") is False and receipt.get("_turn_reserved") is False)
    return bool(receipt.get("_claim_id") and not retired)


def _no_saved_result(kind, receipt):
    """Return whether a rejected original has no result to preserve.

    Presence, rather than truthiness, is deliberate for text receipts: an empty
    saved response is still evidence that this is no longer the no-conversation
    rejection case.  Image result references likewise remain protected.
    """
    if "content" in receipt or "result" in receipt or "data" in receipt:
        return False
    if receipt.get("result_file_ids") or receipt.get("result_sediment_ids"):
        return False
    from services.task_store import pending_image_result_ids
    return not pending_image_result_ids(receipt)


def _rejected_without_conversation(kind, receipt):
    """Recognize the narrow terminal rejection that never obtained a Chat."""
    from services.conversation_binding_service import is_model_not_found_rejection
    return bool(
        kind == "text"
        and receipt.get("status") == "failed"
        and receipt.get("error_code") == "MODEL_NOT_FOUND"
        and receipt.get("upstream_outcome") == "rejected"
        and receipt.get("_submission_started") is True
        and not receipt.get("conversation_id")
        and not _claims(receipt)
        and not receipt.get("upstream_unfinished")
        and _no_saved_result(kind, receipt)
        and is_model_not_found_rejection(receipt)
    )


def release_rejected_product_slots(store, db, receipts=None):
    """Return only stale no-conversation model rejections to the product queue.

    This runs inside PoolAdmission's existing claim transaction.  It changes
    only the parent allocation; original receipts, their request IDs, outcome
    evidence and retired claim IDs remain untouched for a later authorized
    successor.  `receipts` is the claim transaction's current read set so this
    check cannot miss an attached queued or UNKNOWN member.
    """
    rows = list(store.receipts(db)) if receipts is None else receipts
    attached = {}
    for kind, owner, request_id, receipt in rows:
        key = receipt.get("_product_work_key") if isinstance(receipt, dict) else None
        if isinstance(key, str):
            attached.setdefault(key, []).append((kind, owner, request_id, receipt))
    released = 0
    for key, parent in state_snapshot(store, db).items():
        if (not isinstance(parent, dict) or parent.get("key") != key
                or parent.get("state") != "active" or parent.get("slot_held") is not True
                or parent.get("results_saved") is not False or parent.get("outcome") is not None
                or parent.get("cleanup_pending") is not False or parent.get("archive_keys") != []):
            continue
        members = parent.get("members")
        if not isinstance(members, list) or not members:
            continue
        expected = set()
        valid_members = True
        for member in members:
            if (not isinstance(member, dict) or member.get("kind") not in {"text", "image"}
                    or not isinstance(member.get("owner"), str) or not isinstance(member.get("request_id"), str)
                    or not isinstance(member.get("step_ref"), str)):
                valid_members = False
                break
            expected.add((member["kind"], member["owner"], member["request_id"]))
        actual = {(kind, owner, request_id) for kind, owner, request_id, _ in attached.get(key, [])}
        if not valid_members or len(expected) != len(members) or actual != expected:
            continue
        receipts = []
        for kind, owner, request_id, receipt in attached[key]:
            if (not isinstance(receipt, dict) or receipt.get("_product_work_key") != key
                    or _source(receipt) != parent.get("source") or owner != parent.get("owner")
                    or not _rejected_without_conversation(kind, receipt)):
                receipts = []
                break
            receipts.append(receipt)
        if not receipts:
            continue
        parent.update(state="queued", slot_held=False)
        _save(store, db, parent)
        released += 1
    return released


def _unsent(receipt):
    return (receipt.get("status") == "queued" and receipt.get("upstream_outcome") in {None, "not_sent", "not_submitted"}
            and not any(receipt.get(k) for k in ("_submission_started", "upstream_submission_started",
                                                 "_attempt_started_at"))
            and receipt.get("_last_sent_sequence") is None
            and type(receipt.get("send_count", 0)) is int and receipt.get("send_count", 0) == 0
            and not receipt.get("upstream_unfinished")
            and receipt.get("error_code") != "CONVERSATION_OUTCOME_UNKNOWN"
            and not _claims(receipt))


def _unresolved(receipt, members):
    # Correction proof is shared with existing lifecycle, but never overrides
    # an ordinary execution or recovery lease or a paused unresolved branch.
    if _claims(receipt):
        return True
    if (receipt.get("status") in {"failed", "error"}
            and receipt.get("upstream_outcome") not in {"failed", "completed", "not_sent", "not_submitted", "unknown"}
            and receipt.get("_upstream_terminal") is not True):
        return True
    unknown = (receipt.get("status") == "unknown" or receipt.get("upstream_outcome") == "unknown"
               or receipt.get("upstream_unfinished") or receipt.get("error_code") == "CONVERSATION_OUTCOME_UNKNOWN")
    if unknown and (receipt.get("_recovery_paused") or receipt.get("_recovery_suppressed")):
        return True
    if unknown and receipt.get("error_code") == "CONVERSATION_OUTCOME_UNKNOWN" and not (
            receipt.get("upstream_outcome") == "unknown" or receipt.get("status") == "unknown" or receipt.get("upstream_unfinished")):
        return True
    return _blocked(receipt, members=members)


def _missing_node_chain(store, db, parent, receipt, members, now):
    """Return the one terminal R/P/C/A4 chain which can retire R and P.

    The text service owns the immutable request/input/hash proof.  Product
    lifecycle adds only the live terminal-result and ownership constraints
    needed before an UNKNOWN historical root can stop blocking close/archive.
    """
    from services.text_task_service import TextTaskService

    if (receipt.get("status") != "succeeded"
            or receipt.get("upstream_outcome") != "completed"
            or not isinstance(receipt.get("content"), str) or not receipt["content"].strip()
            or not TextTaskService._missing_node_model_rejection_retry_consumed(
                store, db, parent["owner"], receipt)):
        return None
    parent_id = receipt.get("_supersedes_request_id")
    rejected = store.read_receipt(db, "text", parent["owner"], parent_id)
    original_id = rejected.get("_supersedes_request_id") if rejected else None
    original = store.read_receipt(db, "text", parent["owner"], original_id) if isinstance(original_id, str) else None
    image_id = receipt.get(TextTaskService.MISSING_NODE_RETRY_FIELD)
    image = store.read_receipt(db, "image", parent["owner"], image_id) if isinstance(image_id, str) else None
    if not (original and rejected and image
            and TextTaskService._rejected_missing_node_successor_source(
                store, db, parent["owner"], rejected, now)):
        return None
    chain = (original, rejected, receipt, image)
    keys = ("_product_work_key", "provider_binding_id", "provider_account_identity",
            "client_conversation_id", "conversation_id")
    if any(any(item.get(key) != original.get(key) or not item.get(key) for key in keys) for item in chain):
        return None
    if any(item.get("_product_work_key") != parent["key"] for item in chain):
        return None
    if (image.get("status") != "success" or image.get("upstream_outcome") != "generated"
            or image.get("upstream_unfinished") is not False or image.get("binding_status") != "bound"
            or not isinstance(image.get("data"), list) or not image["data"]
            or not isinstance(image.get("parent_message_id"), str) or not image["parent_message_id"]
            or image.get("_missing_node_retry_request_parent_message_id") != receipt.get("parent_message_id")):
        return None
    # Do not turn this narrow exception into a generic cleanup of paused,
    # claimed, completion-owned, or later-mutated historical work.
    controls = ("_recovery_paused", "_recovery_suppressed", "_completion", "_completion_of")
    # R/P are only retired through the immutable successor proof.  Preserve
    # their existing no-claim constraint; only successful result receipts may
    # retain the normal attribution claim after execution has ended.
    if original.get("_claim_id") or rejected.get("_claim_id"):
        return None
    # Terminal receipts deliberately retain their old admission claim for
    # attribution.  `_claims` recognizes that retired claim while retaining
    # the block for executing, reserved, or otherwise ambiguous turns.
    if any(_claims(item) or any(item.get(key) for key in controls) for item in chain):
        return None
    sequence = [item.get("_sequence") for item in chain]
    if not all(type(value) is int and value > 0 for value in sequence) or not sequence[0] < sequence[1] < sequence[3] < sequence[2]:
        return None
    member_ids = {(member["kind"], row.get("request_id") or row.get("id")) for member, row in members}
    expected_ids = {("text", original_id), ("text", parent_id),
                    ("text", receipt.get("request_id")), ("image", image_id)}
    if not expected_ids <= member_ids:
        return None
    # C is intentionally newer than A4 because A4 was queued first.  Keep
    # earlier members as historical work; then accept only a linear later text
    # chain whose immutable inputs start at A4's resulting cursor.  This
    # preserves the original R/P/C/A4 proof but archives the actual latest
    # safe cursor.  A parallel, unproved, or non-text later member remains an
    # ambiguity rather than a reason to retire an UNKNOWN root.
    later = []
    for member, other in members:
        kind = member["kind"]
        other_id = other.get("request_id") or other.get("id")
        if (kind, other_id) not in expected_ids and _same_conversation(other, receipt):
            if type(other.get("_sequence")) is not int:
                return None
            if other["_sequence"] <= receipt["_sequence"]:
                continue
            later.append((other["_sequence"], member, other))
    later.sort(key=lambda item: item[0])
    if any(left[0] == right[0] for left, right in zip(later, later[1:])):
        return None
    successors = []
    cursor = image["parent_message_id"]
    for _, member, other in later:
        kind = member["kind"]
        other_id = other.get("request_id") or other.get("id")
        if not isinstance(other_id, str) or not other_id:
            return None
        if (kind != "text"
                or other.get("status") != "succeeded"
                or other.get("upstream_outcome") != "completed"
                or not isinstance(other.get("content"), str) or not other["content"].strip()
                or not isinstance(other.get("parent_message_id"), str) or not other["parent_message_id"]
                or other.get("request_parent_message_id") not in (None, "", cursor)
                or _claims(other) or any(other.get(key) for key in controls)
                or _unresolved(other, [row for _, row in members])):
            return None
        if any(other.get(key) != original.get(key) or not other.get(key) for key in keys):
            return None
        try:
            row = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?",
                             (parent["owner"], other_id)).fetchone()
            body = store.load_input(other["_input_ref"])
            if not row or TextTaskService._submission_identity(parent["owner"], body) != (other_id, row[0]):
                return None
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            return None
        if (other.get("_product_work") != _metadata(parent, member["step_ref"])
                or body.get("product_work") != other.get("_product_work")
                or body.get("parent_message_id") != cursor
                or any(body.get(key) != image.get(key) for key in (
                    "provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id"))):
            return None
        successors.append((member, other))
        cursor = other["parent_message_id"]
    if successors:
        archive_member, archive = successors[-1]
    else:
        archive_member = next((member for member, row in members
                               if member["kind"] == "image" and row.get("id") == image_id), None)
        archive = image
    if not isinstance(archive_member, dict):
        return None
    return {"original": original, "rejected": rejected, "continue": receipt, "image": image,
            "archive": archive, "archive_member": archive_member}


def _no_final_successor_chain(store, db, parent, receipt, members, now):
    """Resolve one saved successor using the exact original recovery proof.

    This is deliberately a lifecycle proof, not a call to the send-time
    supersede validator.  Close and archive must re-read the frozen inputs,
    hashes, ownership and live controls after the successor has completed.
    """
    from services.category_directory_derivation import derive_retained_input
    from services.conversation_binding_service import ConversationBindingError
    from services.text_task_service import TextTaskService

    successor_id = receipt.get("request_id")
    original_id = receipt.get("_supersedes_request_id")
    incomplete = receipt.get("_supersedes_incomplete") is True
    if (not isinstance(successor_id, str) or not successor_id
            or not isinstance(original_id, str) or not original_id or successor_id == original_id
            or (receipt.get("_supersedes_no_final") is True) == incomplete
            or receipt.get("status") != "succeeded"
            or receipt.get("upstream_outcome") != "completed"
            or not isinstance(receipt.get("content"), str) or not receipt["content"].strip()
            or receipt.get("upstream_unfinished")):
        return None
    original_row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                              (parent["owner"], original_id)).fetchone()
    successor_row = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?",
                               (parent["owner"], successor_id)).fetchone()
    if not original_row or not successor_row or receipt.get("_supersedes_input_hash") != original_row[0]:
        return None
    try:
        original = json.loads(original_row[1])
        original_body = store.load_input(original["_input_ref"])
        successor_body = store.load_input(receipt["_input_ref"])
        original_identity = TextTaskService._submission_identity(parent["owner"], original_body)
        successor_identity = TextTaskService._submission_identity(parent["owner"], successor_body)
    except (ConversationBindingError, OSError, ValueError, TypeError, KeyError):
        return None
    ancestors = []
    if original.get("_supersedes_request_id"):
        if incomplete:
            return None
        try:
            native_chain = TextTaskService._native_no_final_successor_source(
                store, db, parent["owner"], original, now,
                successor_id=successor_id, successor_sequence=receipt.get("_sequence"))
        except (ConversationBindingError, OSError, ValueError, TypeError, KeyError):
            return None
        if not native_chain:
            return None
        ancestors = [native_chain["original"], native_chain["parent"]]
    advanced_by_own_successor = False
    if incomplete:
        # Child success never terminates the original. Only the original
        # reader's exact persisted terminal-empty proof can waive UNKNOWN;
        # a nonempty original success needs no waiver in the first place.
        ended = original.get("_turn_end_evidence")
        original_ended = bool(
            original.get("status") in {"unknown", "failed"}
            and original.get("error_code") in {"CONVERSATION_OUTCOME_UNKNOWN", "RESULT_UNRECOVERABLE"}
            and original.get("recovery_reason") == "REQUEST_RESULT_TERMINAL_EMPTY"
            and original.get("_upstream_terminal") is True
            and isinstance(ended, dict)
            and set(ended) == {"conversation_id", "request_message_id", "final_message_id", "observed_at"}
            and all(isinstance(ended.get(key), str) and ended[key].strip() for key in (
                "conversation_id", "request_message_id", "final_message_id"))
            and ended["conversation_id"] == original.get("conversation_id")
            and ended["request_message_id"] == original.get("request_message_id")
            and ended["final_message_id"] != ended["request_message_id"]
            and type(ended.get("observed_at")) in {int, float}
            and math.isfinite(ended["observed_at"]) and 0 < ended["observed_at"] <= now)
        # This is deliberately not another terminal-result case.  The bound
        # original reader may prove that its still-UNKNOWN branch advanced to
        # this exact, locally-owned compact child.  Keep the original UNKNOWN
        # and use that branch observation only as a ProductWork close waiver.
        advanced = original.get("_completed_successor_evidence")
        successor_ended = receipt.get("_turn_end_evidence")
        advanced_by_own_successor = bool(
            original.get("status") == "unknown"
            and original.get("error_code") == "CONVERSATION_OUTCOME_UNKNOWN"
            and original.get("recovery_reason") == "REQUEST_CONVERSATION_ADVANCED"
            and original.get("upstream_outcome") in {None, "unknown"}
            and original.get("upstream_unfinished") is False
            and original.get("_upstream_terminal") is not True
            and not original.get("_turn_end_evidence")
            and isinstance(advanced, dict)
            and set(advanced) == {"conversation_id", "request_message_id", "successor_request_message_id",
                                  "successor_final_message_id", "observed_at"}
            and all(isinstance(advanced.get(key), str) and advanced[key].strip() for key in (
                "conversation_id", "request_message_id", "successor_request_message_id",
                "successor_final_message_id"))
            and advanced["conversation_id"] == original.get("conversation_id")
            and advanced["request_message_id"] == original.get("request_message_id")
            and advanced["successor_request_message_id"] == receipt.get("request_message_id")
            and advanced["successor_final_message_id"] == receipt.get("parent_message_id")
            and advanced["successor_request_message_id"] != advanced["request_message_id"]
            and advanced["successor_final_message_id"] != advanced["successor_request_message_id"]
            and type(advanced.get("observed_at")) in {int, float}
            and math.isfinite(advanced["observed_at"]) and 0 < advanced["observed_at"] <= now
            and isinstance(successor_ended, dict)
            and set(successor_ended) == {"conversation_id", "request_message_id", "final_message_id", "observed_at"}
            and all(isinstance(successor_ended.get(key), str) and successor_ended[key].strip() for key in (
                "conversation_id", "request_message_id", "final_message_id"))
            and successor_ended["conversation_id"] == receipt.get("conversation_id")
            and successor_ended["request_message_id"] == receipt.get("request_message_id")
            and successor_ended["final_message_id"] == receipt.get("parent_message_id")
            and successor_ended["final_message_id"] != successor_ended["request_message_id"]
            and type(successor_ended.get("observed_at")) in {int, float}
            and math.isfinite(successor_ended["observed_at"])
            and 0 < successor_ended["observed_at"] <= advanced["observed_at"])
    else:
        original_ended = (original.get("status") == "failed"
                          and original.get("error_code") == "RESULT_UNRECOVERABLE"
                          and not original.get("_upstream_terminal")
                          and TextTaskService._no_final_successor_source(original, now))
    if (original.get("request_id") != original_id
            or not (original_ended or advanced_by_own_successor)
            or (original.get("upstream_outcome") not in {None, "unknown"} if advanced_by_own_successor
                else original.get("upstream_outcome") != "unknown")
            or original.get("_supersedes_no_final") is True
            or original.get("upstream_unfinished")
            or any(key in original for key in ("content", "result", "data"))
            or original.get("result_file_ids") or original.get("result_sediment_ids")
            or original_identity != (original_id, original_row[0])
            or successor_identity != (successor_id, successor_row[0])):
        return None
    controls = ("_executing", "_turn_reserved", "recovery_claim_id", "_recovery_claim_id",
                "_recovery_paused", "_recovery_suppressed", "_completion", "_completion_of")
    chain_members = [*ancestors, original, receipt]
    def has_claim(item):
        if incomplete and item is original and item.get("_claim_id"):
            until = item.get("_claim_until")
            if (isinstance(item["_claim_id"], str) and type(until) in {int, float}
                    and math.isfinite(until) and until <= now
                    and item.get("_executing") is False and item.get("_turn_reserved") is False):
                return _claims({**item, "_claim_id": None})
        return _claims(item)
    if (any(has_claim(item) or any(item.get(key) for key in controls) for item in chain_members)
            or _unresolved(receipt, [row for _, row in members])):
        return None
    keys = ("_product_work_key", "provider_binding_id", "provider_account_identity",
            "client_conversation_id", "conversation_id")
    if (any(any(item.get(key) != original.get(key) or not item.get(key) for key in keys)
            for item in chain_members)
            or any(item.get("_product_work_key") != parent["key"] for item in chain_members)
            or any(_source(item) != parent["source"] for item in chain_members)):
        return None
    if (type(original.get("_sequence")) is not int or type(receipt.get("_sequence")) is not int
            or original["_sequence"] <= 0 or receipt["_sequence"] <= original["_sequence"]
            or not TextTaskService._supersede_order_link(original, receipt)):
        return None
    if ((original_body.get("supersedes_request_id") and not ancestors)
            or original_body.get("continue_after_no_final") is not None
            or original_body.get("continue_after_incomplete") is not None
            or original_body.get("derived_input") is not None):
        return None
    audit = receipt.get("_derived_input")
    reconciled = receipt.get(TextTaskService._RECONCILED_QUEUED_SUCCESSOR) is True
    if reconciled:
        if incomplete or ancestors or audit is not None:
            return None
        try:
            TextTaskService._validate_reconciled_queued_successor(
                store, db, parent["owner"], receipt, original, original_row[0], original_body, now,
            )
        except ConversationBindingError:
            return None
    elif audit is not None:
        allowed = {"combined_copy_compact_v1"} if incomplete else {
            "attributes_required_only_v4", "combined_copy_compact_v1"}
        if ancestors or not isinstance(audit, dict) or audit.get("kind") not in allowed:
            return None
        try:
            expected_body, expected_audit = derive_retained_input(
                original_body, successor_id, original_id, {"kind": audit["kind"]},
                no_final=not incomplete, incomplete=incomplete)
        except (ConversationBindingError, ValueError, TypeError, KeyError):
            return None
        if audit != {**expected_audit, "original_input_hash": original_row[0]} or successor_body != expected_body:
            return None
    elif incomplete or successor_body != {**original_body, "client_request_id": successor_id,
                            "supersedes_request_id": original_id, "continue_after_no_final": True}:
        return None
    children = db.execute(
        "SELECT id FROM requests WHERE owner=? AND "
        "(json_extract(receipt,'$._supersedes_request_id')=? OR json_extract(receipt,'$._codex_fallback_of')=?)",
        (parent["owner"], original_id, original_id),
    ).fetchall()
    if len(children) != 1 or children[0][0] != successor_id:
        return None
    if advanced_by_own_successor:
        descendants = db.execute(
            "SELECT id FROM requests WHERE owner=? AND "
            "(json_extract(receipt,'$._supersedes_request_id')=? OR json_extract(receipt,'$._codex_fallback_of')=?)",
            (parent["owner"], successor_id, successor_id),
        ).fetchall()
        if descendants:
            return None
    member_ids = {(member["kind"], row.get("request_id") or row.get("id")) for member, row in members}
    if not {("text", item["request_id"]) for item in chain_members} <= member_ids:
        return None
    if advanced_by_own_successor:
        chain_ids = {("text", parent["owner"], original_id), ("text", parent["owner"], successor_id)}
        registered = {(member["kind"], member["owner"], member["request_id"]) for member, _ in members}
        all_rows = list(store.receipts(db))
        all_receipts = [other for _, _, _, other in all_rows]
        controls = ("_executing", "_turn_reserved", "recovery_claim_id", "_recovery_claim_id",
                    "_recovery_paused", "_recovery_suppressed", "_completion", "_completion_of")
        # A finished, registered step before the original is historical work,
        # not a competing successor. Everything else in the same physical
        # conversation remains a conflict, including another owner's receipt.
        def safe_predecessor(kind, owner, request_id, other):
            return bool(
                (kind, owner, request_id) in registered
                and owner == parent["owner"]
                and type(other.get("_sequence")) is int
                and other["_sequence"] > 0 and other["_sequence"] < original["_sequence"]
                and other.get("status") in {"succeeded", "success"}
                and other.get("upstream_outcome") == "completed"
                and other.get("upstream_unfinished") is not True
                and other.get("_upstream_terminal") is True
                and not _claims(other)
                and not any(other.get(key) for key in controls)
                and not _unresolved(other, all_receipts))
        if any(_same_conversation(other, receipt)
               and (kind, owner, request_id) not in chain_ids
               and not safe_predecessor(kind, owner, request_id, other)
               for kind, owner, request_id, other in all_rows):
            return None
    member_refs = {member["request_id"]: member for member, _ in members if member["kind"] == "text"}
    metadata = original_body.get("product_work")
    attempt_steps = reconciled and receipt.get(TextTaskService._RECONCILED_ATTEMPT_STEP) is True
    if attempt_steps:
        # The registered pair retains two attempt-addressed members. Each must
        # still match its own frozen input and the same durable parent.
        for item, body in ((original, original_body), (receipt, successor_body)):
            item_metadata = body.get("product_work")
            if (not isinstance(item_metadata, dict)
                    or item_metadata != _metadata(parent, member_refs[item["request_id"]].get("step_ref"))
                    or item.get("_product_work") != item_metadata):
                return None
    elif (not isinstance(metadata, dict) or metadata != _metadata(parent, metadata.get("step_ref"))
            or any(item.get("_product_work") != metadata
                   or member_refs[item["request_id"]].get("step_ref") != metadata.get("step_ref")
                   for item in chain_members)):
        return None
    if ancestors:
        if any(not TextTaskService._supersede_order_link(previous, successor)
               for previous, successor in zip(chain_members, chain_members[1:])):
            return None
        member_refs = {member["request_id"]: member for member, _ in members if member["kind"] == "text"}
        for item in chain_members:
            member = member_refs.get(item["request_id"])
            if not member:
                return None
            metadata = _metadata(parent, member["step_ref"])
            if item.get("_product_work") != metadata or original_body.get("product_work") != metadata:
                return None
    return {"original": original, "successor": receipt, "resolved": [*ancestors, original],
            "advanced_by_own_successor": advanced_by_own_successor}


def _codex_rejected_retry_chain(store, db, parent, receipt, members):
    """Return P/D/E only after the registered retry has a saved final result."""
    retry_id = receipt.get("request_id")
    rejected_id = receipt.get("_codex_rejected_retry_of")
    controls = ("_executing", "_turn_reserved", "recovery_claim_id", "_recovery_claim_id",
                "_recovery_paused", "_recovery_suppressed", "_completion", "_completion_of",
                "upstream_unfinished")
    leases = ("recovery_claimed_at", "recovery_lease_until", "_recovery_claim_until")
    if (not isinstance(retry_id, str) or not retry_id or not isinstance(rejected_id, str) or not rejected_id
            or receipt.get("status") != "succeeded" or receipt.get("upstream_outcome") != "completed"
            or not isinstance(receipt.get("content"), str) or not receipt["content"].strip()
            or _claims(receipt) or any(receipt.get(key) for key in controls)
            or any(receipt.get(key) is not None for key in leases)
            or _unresolved(receipt, [row for _, row in members])):
        return None
    row = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", (parent["owner"], retry_id)).fetchone()
    if not row:
        return None
    try:
        from services.conversation_binding_service import ConversationBindingError
        from services.codex_text_fallback import validate_prepared_rejected_retry
        from services.text_task_service import TextTaskService
        body = store.load_input(receipt["_input_ref"])
        if TextTaskService._submission_identity(parent["owner"], body) != (retry_id, row[0]):
            return None
        validate_prepared_rejected_retry(store, db, parent["owner"], body, receipt=receipt)
        rejected = store.read_receipt(db, "text", parent["owner"], rejected_id)
        original_id = rejected.get("_codex_fallback_of") if rejected else None
        original = store.read_receipt(db, "text", parent["owner"], original_id) if isinstance(original_id, str) else None
    except (ConversationBindingError, OSError, ValueError, TypeError, KeyError):
        return None
    chain = (original, rejected, receipt)
    if (not all(isinstance(item, dict) for item in chain)
            or any(item.get("_product_work_key") != parent["key"] or _source(item) != parent["source"] for item in chain)
            or any(_claims(item) or any(item.get(key) for key in controls)
                   or any(item.get(key) is not None for key in leases) for item in chain)
            or any(item.get("_product_work") != _metadata(parent,
                        next((member["step_ref"] for member, row in members
                              if row.get("request_id") == item.get("request_id")), None))
                   for item in chain)):
        return None
    try:
        for item in chain:
            row = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?",
                             (parent["owner"], item["request_id"])).fetchone()
            saved = store.load_input(item["_input_ref"])
            if not row or TextTaskService._submission_identity(parent["owner"], saved) != (item["request_id"], row[0]):
                return None
            if saved.get("product_work") != item.get("_product_work"):
                return None
    except (OSError, ValueError, TypeError, KeyError):
        return None
    sequence = [item.get("_sequence") for item in chain]
    if not all(type(value) is int and value > 0 for value in sequence) or not sequence[0] < sequence[1] < sequence[2]:
        return None
    member_ids = {(member["kind"], row.get("request_id") or row.get("id")) for member, row in members}
    if not {("text", item["request_id"]) for item in chain} <= member_ids:
        return None
    return {"original": original, "rejected": rejected, "retry": receipt, "resolved": [original, rejected]}


class ProductWorkService:
    def __init__(self, text_service, image_service, *, clock=time.time):
        self.text, self.images, self.store, self.clock = text_service, image_service, text_service.store, clock

    def product_capacity_limit(self):
        admission = getattr(self.text, "admission", None)
        read = getattr(admission, "product_capacity_limit", None)
        # Lightweight legacy adapters may expose product work but not a pool;
        # zero is safer than advertising an invented admission capacity.
        return int(read()) if callable(read) else 0

    def _load(self, db, identity, product_ref):
        source = require_product_source(identity)
        key = _key(source, str(identity["id"]), _ref(product_ref))
        parent = self.store.runtime(db, key)
        if not parent:
            raise WorkLifecycleError("PRODUCT_WORK_NOT_FOUND", 404)
        if parent["owner"] != str(identity["id"]) or parent["source"] != source:
            raise WorkLifecycleError("PRODUCT_WORK_OWNER_CONFLICT", 403)
        return parent

    def _members(self, db, parent):
        result = []
        for member in parent["members"]:
            receipt = self.store.read_receipt(db, member["kind"], member["owner"], member["request_id"])
            if (not receipt or receipt.get("_product_work_key") != parent["key"]
                    or _source(receipt) != parent["source"] or member["owner"] != parent["owner"]):
                raise WorkLifecycleError("PRODUCT_WORK_BINDING_CONFLICT")
            result.append((member, receipt))
        return result

    def _missing_node_chains(self, db, parent, members):
        chains = {}
        now = float(self.clock())
        for _, receipt in members:
            chain = _missing_node_chain(self.store, db, parent, receipt, members, now)
            if chain:
                chains[receipt["request_id"]] = chain
        return chains

    def _confirmed_archives(self, db, parent, version, keys, members):
        """Read confirmed child receipts tied to this cycle's original members."""
        if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
            return None
        originals = {(m["kind"], m["owner"], m["request_id"]): r for m, r in members}
        archives = []
        for key in keys:
            row = self.store.runtime(db, key)
            if not row:
                return None
            archive = row.get("archive") or {}
            original = originals.get((row.get("kind"), row.get("owner"), row.get("last_request_id")))
            if (row.get("key") != key or row.get("_product_archive_parent") != parent["key"]
                    or row.get("_product_archive_version", 1) != version or row.get("version") != version
                    or archive.get("version") != version or archive.get("status") != "confirmed"
                    or archive.get("desired") is not True or archive.get("claim")
                    or archive.get("request_id") != row.get("last_request_id") or not original
                    or any(not original.get(k) or row.get("target", {}).get(k) != original[k] for k in (
                        "provider_binding_id", "provider_account_identity", "conversation_id"))):
                return None
            archives.append(row)
        return archives

    def _closed_missing_node_chains(self, db, parent, members):
        """Revalidate resolved historical chains within their confirmed cycle.

        A later step may legitimately advance the same conversation after
        resume. It must not erase the prior closure's R/P resolution proof,
        nor let that old A4 cursor override the new step's archive target.
        Only member identities are saved; all evidence/claims are reread now.
        """
        chains = {}
        current = {(m["kind"], m["owner"], m["request_id"]): (m, r) for m, r in members}
        for closure in parent.get("previous_closures", []):
            version = closure.get("version")
            saved_members, keys = closure.get("members"), closure.get("archive_keys")
            if (type(version) is not int or not 1 <= version < parent.get("version", 1)
                    or closure.get("outcome") != "completed" or closure.get("results_saved") is not True
                    or not isinstance(saved_members, list) or not saved_members
                    or not isinstance(keys, list) or not keys):
                continue
            previous = []
            for member in saved_members:
                if not isinstance(member, dict):
                    break
                found = current.get(tuple(member.get(k) for k in ("kind", "owner", "request_id")))
                if not found or found[0] != member:
                    break
                previous.append(found)
            if len(previous) != len(saved_members):
                continue
            archives = self._confirmed_archives(db, parent, version, keys, previous)
            if archives is None:
                continue
            for request_id, chain in self._missing_node_chains(db, parent, previous).items():
                archive = chain["archive"]
                if any(row.get("kind") == chain["archive_member"]["kind"]
                       and row.get("last_request_id") == chain["archive_member"]["request_id"]
                       and row.get("owner") == parent["owner"]
                       and all(row.get("target", {}).get(k) == archive.get(k) for k in (
                           "provider_binding_id", "provider_account_identity", "conversation_id"))
                       for row in archives):
                    chains[request_id] = chain
        return chains

    def _missing_node_waivers(self, db, parent, members):
        waived = set()
        chains = self._missing_node_chains(db, parent, members)
        chains.update(self._closed_missing_node_chains(db, parent, members))
        for chain in chains.values():
            waived.update(chain[name]["request_id"] for name in ("original", "rejected"))
        return waived

    def _no_final_successor_waivers(self, db, parent, members):
        waived = set()
        now = float(self.clock())
        for _, receipt in members:
            chain = _no_final_successor_chain(self.store, db, parent, receipt, members, now)
            if chain:
                waived.update(item["request_id"] for item in chain["resolved"])
        return waived

    def _owned_advanced_successor_chains(self, db, parent, members):
        """Return only compact children proven by the original-reader branch proof."""
        chains = {}
        now = float(self.clock())
        for _, receipt in members:
            chain = _no_final_successor_chain(self.store, db, parent, receipt, members, now)
            if chain and chain.get("advanced_by_own_successor"):
                chains[receipt["request_id"]] = chain
        return chains

    def _codex_rejected_retry_waivers(self, db, parent, members):
        waived = set()
        for _, receipt in members:
            chain = _codex_rejected_retry_chain(self.store, db, parent, receipt, members)
            if chain:
                waived.update(item["request_id"] for item in chain["resolved"])
        return waived

    def archive_turns_ready(self, db, child, members):
        """Apply exact resolved-history exceptions after the generic recheck."""
        parent = self.store.runtime(db, child["_product_archive_parent"])
        if not parent:
            return False
        owned = self._members(db, parent)
        waived = self._missing_node_waivers(db, parent, owned)
        waived.update(self._no_final_successor_waivers(db, parent, owned))
        waived.update(self._codex_rejected_retry_waivers(db, parent, owned))
        for member, receipt in owned:
            if receipt.get(_CANCELLED_UNSENT_STEP) and not _cancelled_unsent_step(
                    self.store, db, parent, member, receipt):
                return False
        return not any(_blocked(receipt, members=members) and receipt.get("request_id") not in waived
                       for receipt in members)

    def _reasons(self, db, parent, members, outcome=None):
        reasons = set()
        receipts = [r for _, r in members]
        chains = self._missing_node_chains(db, parent, members)
        chains.update(self._closed_missing_node_chains(db, parent, members))
        waived = set()
        valid_continues = set()
        for chain in chains.values():
            waived.update(chain[name]["request_id"] for name in ("original", "rejected"))
            valid_continues.add(chain["continue"]["request_id"])
        waived.update(self._no_final_successor_waivers(db, parent, members))
        waived.update(self._codex_rejected_retry_waivers(db, parent, members))
        from services.text_task_service import TextTaskService
        for member, receipt in members:
            cancelled_unsent = _cancelled_unsent_step(self.store, db, parent, member, receipt)
            if receipt.get(_CANCELLED_UNSENT_STEP) and not cancelled_unsent:
                reasons.add("PRODUCT_WORK_TURN_UNFINISHED")
                continue
            if cancelled_unsent:
                continue
            if outcome == "cancelled" and _unsent(receipt):
                continue
            # A registered C is never an ordinary successful tail.  If one of
            # its R/P/A4 facts changes later, retain the product rather than
            # silently choosing the normal max-sequence archive target.
            if (receipt.get(TextTaskService.MISSING_NODE_RETRY_FIELD)
                    and receipt.get("request_id") not in valid_continues):
                reasons.add("PRODUCT_WORK_TURN_UNFINISHED")
            replaced = _rejected_chat_replaced(self.store, db, member["owner"], receipt, members)
            if not replaced and receipt.get("request_id") not in waived and _unresolved(receipt, receipts):
                reasons.add("PRODUCT_WORK_TURN_UNFINISHED")
            status = receipt.get("status")
            if outcome == "completed" and status not in {"succeeded", "success"}:
                if not (replaced or receipt.get("request_id") in waived
                        or status in {"unknown", "failed", "error"} and not _unresolved(receipt, receipts)
                        and (receipt.get("upstream_outcome") == "unknown" or receipt.get("upstream_unfinished"))):
                    reasons.add("PRODUCT_WORK_RESULT_REQUIRED")
            if outcome in {"failed", "cancelled"} and status not in {"succeeded", "success", "failed", "error", "cancelled"}:
                reasons.add("PRODUCT_WORK_TURN_UNFINISHED")
        if outcome == "failed" and not any(r.get("status") in {"failed", "error"} for r in receipts):
            reasons.add("PRODUCT_WORK_FAILURE_REQUIRED")
        for kind, owner, rid, receipt in self.store.receipts(db):
            if any(_same_conversation(receipt, own) for own in receipts):
                if receipt.get("_product_work_key") != parent["key"]:
                    # Archiving even another owner's completed conversation is
                    # outside this parent. Adoption must include exact originals.
                    reasons.add("PRODUCT_WORK_CONVERSATION_CONFLICT")
        return sorted(reasons, key=lambda reason: (reason != "PRODUCT_WORK_TURN_UNFINISHED", reason))

    def _projection(self, db, parent):
        members = self._members(db, parent)
        control = admission_control(parent)
        # GET remains read-only, but must not report a completed pause lease as
        # live.  The next admission transaction persists this same transition.
        if (control["state"] == "pausing"
                and not any(_live_product_claim(receipt, float(self.clock())) for _member, receipt in members)):
            control = {**control, "state": "paused", "slot_held": False}
        archives = [self.store.runtime(db, key) for key in parent.get("archive_keys", [])]
        confirmed = sum(bool(row and (row.get("archive") or {}).get("status") == "confirmed") for row in archives)
        reasons = self._reasons(db, parent, members)
        if parent["state"] == "closing" and confirmed != len(archives):
            reasons.append("PRODUCT_WORK_ARCHIVE_PENDING")
        projected_members = []
        for member, receipt in members:
            item = {"kind": member["kind"], "request_id": member["request_id"],
                    "step_ref": member["step_ref"], "status": receipt.get("status")}
            if member["kind"] == "text" and receipt.get(_CANCELLED_UNSENT_STEP):
                row = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?",
                                 (member["owner"], member["request_id"])).fetchone()
                if row:
                    proof = cancelled_unsent_public(self.store, db, member["owner"], row[0], receipt)
                    if proof:
                        item["cancelled_unsent"] = proof
            projected_members.append(item)
        return {"protocol": "product-work-v1", "product_ref": parent["product_ref"], "state": parent["state"],
                "version": parent.get("version", 1),
                "slot_held": bool(parent["slot_held"]), "results_saved": bool(parent.get("results_saved")),
                "outcome": parent.get("outcome"), "cleanup_pending": bool(parent.get("cleanup_pending")),
                "admission_control": control,
                "members": projected_members,
                "archive": {"total": len(archives), "confirmed": confirmed, "pending": len(archives) - confirmed},
                "blocked_reasons": sorted(set(reasons))}

    def get(self, identity, product_ref):
        # A product projection spans parent, member and conflict receipts.  Hold
        # one deferred snapshot for those reads, but never acquire the store's
        # writer transaction merely to render it.  `query_only` applies only to
        # this short-lived connection; schema initialization happens before it.
        with self.store.connect() as db:
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN DEFERRED")
            try:
                return self._projection(db, self._load(db, identity, product_ref))
            finally:
                # End an explicit read transaction before connection teardown on
                # both the normal and exceptional paths.
                if db.in_transaction:
                    db.rollback()

    def cancel_step(self, identity, product_ref, request_id, expected_version):
        if type(expected_version) is not int or expected_version < 1:
            raise WorkLifecycleError("PRODUCT_WORK_VERSION_INVALID", 400)
        _ref(request_id)
        with self.store.transaction() as db:
            parent = self._load(db, identity, product_ref)
            version = parent.get("version", 1)
            if version != expected_version:
                raise WorkLifecycleError("PRODUCT_WORK_VERSION_CONFLICT")
            members = self._members(db, parent)
            selected = next(((member, receipt) for member, receipt in members
                             if member["kind"] == "text" and member["request_id"] == request_id), None)
            if not selected:
                raise WorkLifecycleError("PRODUCT_WORK_REQUEST_NOT_FOUND", 404)
            member, receipt = selected
            if _cancelled_unsent_step(self.store, db, parent, member, receipt, allow_historical=False):
                result = self._projection(db, parent)
                row = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?",
                                 (member["owner"], member["request_id"])).fetchone()
                proof = cancelled_unsent_public(self.store, db, member["owner"], row[0], receipt) if row else None
                if proof:
                    result["cancelled_unsent"] = proof
                return result
            if parent.get("state") not in {"queued", "active"}:
                raise WorkLifecycleError("PRODUCT_WORK_NOT_ACTIVE")
            if receipt.get(_CANCELLED_UNSENT_STEP):
                raise WorkLifecycleError("PRODUCT_WORK_CANCEL_STEP_PROOF_INVALID")
            if not _cancel_unsent_eligible(receipt):
                raise WorkLifecycleError("PRODUCT_WORK_CANCEL_STEP_NOT_UNSENT")
            if (receipt.get("request_id") != member["request_id"]
                    or receipt.get("_product_work") != _metadata(parent, member["step_ref"])):
                raise WorkLifecycleError("PRODUCT_WORK_CANCEL_STEP_INPUT_INVALID")
            row = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?",
                             (member["owner"], member["request_id"])).fetchone()
            if not row or not _input_matches_cancelled_member(self.store, member["owner"], member, receipt, row[0]):
                raise WorkLifecycleError("PRODUCT_WORK_CANCEL_STEP_INPUT_INVALID") from None
            receipt.update(status="cancelled", error_code="PRODUCT_WORK_STEP_CANCELLED",
                           upstream_outcome="not_sent", upstream_unfinished=False,
                           **{_CANCELLED_UNSENT_STEP: {"parent_key": parent["key"], "parent_version": version,
                                                       "request_hash": row[0], "cancelled_at": float(self.clock())}})
            self.store.write_receipt(db, "text", member["owner"], member["request_id"], receipt)
            result = self._projection(db, parent)
            result["cancelled_unsent"] = cancelled_unsent_public(self.store, db, member["owner"], row[0], receipt)
            return result

    def resume(self, identity, product_ref, expected_version):
        """Reopen a saved, archived product without sending or reserving a turn.

        The physical lock is shared with archive execution and attachment. A
        delayed archive worker must revalidate its old parent version under
        that lock before it can touch the original conversation.
        """
        if type(expected_version) is not int or expected_version < 1:
            raise WorkLifecycleError("PRODUCT_WORK_VERSION_INVALID", 400)

        def scope(db, parent):
            receipts = [receipt for _, receipt in self._members(db, parent)]
            return _physical_lock_keys(receipts), [tuple(receipt.get(k) for k in (
                "provider_binding_id", "client_conversation_id", "provider_account_identity", "conversation_id"))
                for receipt in receipts]

        with self.store.connect() as db:
            expected_scope = scope(db, self._load(db, identity, product_ref))
        lock_receipts = [{"provider_account_identity": account, "conversation_id": conversation}
                         for account, conversation in expected_scope[0]]
        with locking_product_conversations(lock_receipts):
            with self.store.transaction() as db:
                parent = self._load(db, identity, product_ref)
                version = parent.get("version", 1)
                # A repeated request observes its already-created cycle even
                # after that cycle closes. It must never reopen it again.
                if version == expected_version + 1 and parent.get("resumed_from_version") == expected_version:
                    return self._projection(db, parent)
                if version != expected_version:
                    raise WorkLifecycleError("PRODUCT_WORK_VERSION_CONFLICT")
                if parent["state"] != "completed" or parent.get("outcome") != "completed":
                    raise WorkLifecycleError("PRODUCT_WORK_RESUME_NOT_COMPLETED")
                if parent.get("results_saved") is not True:
                    raise WorkLifecycleError("PRODUCT_WORK_RESULTS_SAVE_REQUIRED")
                if scope(db, parent) != expected_scope:
                    raise WorkLifecycleError("PRODUCT_WORK_BINDING_CHANGED")
                members = self._members(db, parent)
                reasons = self._reasons(db, parent, members, "completed")
                if reasons:
                    raise WorkLifecycleError(reasons[0])
                keys = parent.get("archive_keys", [])
                archives = self._confirmed_archives(db, parent, version, keys, members)
                if (parent.get("cleanup_pending") or parent.get("slot_held")
                        or not keys and any(r.get("conversation_id") for _, r in members)
                        or archives is None):
                    raise WorkLifecycleError("PRODUCT_WORK_ARCHIVE_PENDING")
                parent.setdefault("previous_closures", []).append({"version": version, **{
                    key: parent.get(key) for key in ("outcome", "results_saved", "results_saved_at",
                                                     "archive_keys", "updated_at")},
                    "members": [dict(member) for member in parent["members"]]})
                # Reopening preserves lifecycle/physical ownership.  It
                # creates no admission reservation: the queued successor must
                # reacquire that independently in PoolAdmission's claim.
                control = admission_control(parent)
                parent.update(version=version + 1, resumed_from_version=version, state="active",
                              slot_held=True, results_saved=False, outcome=None, cleanup_pending=False,
                              admission_control={**control, "state": "active", "slot_held": False},
                              archive_keys=[], updated_at=float(self.clock()))
                parent.pop("results_saved_at", None)
                _save(self.store, db, parent)
                return self._projection(db, parent)

    def adopt(self, identity, product_ref, members):
        source, owner = require_product_source(identity), str(identity["id"])
        _ref(product_ref)
        if not isinstance(members, list) or not members:
            raise WorkLifecycleError("PRODUCT_WORK_MEMBERS_INVALID", 400)
        seen = set()
        for member in members:
            if (not isinstance(member, dict) or set(member) != {"kind", "request_id", "step_ref"}
                    or member["kind"] not in {"text", "image"}):
                raise WorkLifecycleError("PRODUCT_WORK_MEMBERS_INVALID", 400)
            _ref(member["request_id"])
            _ref(member["step_ref"])
            identity_key = member["kind"], member["request_id"]
            if identity_key in seen:
                raise WorkLifecycleError("PRODUCT_WORK_MEMBERS_INVALID", 400)
            seen.add(identity_key)

        def originals(db):
            rows = []
            for member in members:
                receipt = self.store.read_receipt(db, member["kind"], owner, member["request_id"])
                if not receipt:
                    raise WorkLifecycleError("PRODUCT_WORK_REQUEST_NOT_FOUND", 404)
                if _source(receipt) != source:
                    raise WorkLifecycleError("PRODUCT_WORK_SOURCE_CONFLICT", 403)
                rows.append(receipt)
            return rows

        def lock_scope(db, rows):
            # Include alternate saved bindings for the same physical conversation
            # so a legacy archive through an alias cannot race ownership adoption.
            related = rows + [r for _, _, _, r in self.store.receipts(db)
                              if any(_same_conversation(r, own) for own in rows)]
            keys = _physical_lock_keys(related)
            anchors = [tuple(r.get(k) for k in ("provider_binding_id", "client_conversation_id",
                                               "provider_account_identity", "conversation_id")) for r in rows]
            return sorted(keys), anchors

        # Match raw archive/first attachment: physical lifecycle lock, then
        # optional legacy cursor lock, then SQLite. Never wait inside SQLite.
        with self.store.connect() as db:
            expected_scope = lock_scope(db, originals(db))
        lock_receipts = [{"provider_account_identity": account, "conversation_id": conversation}
                         for account, conversation in expected_scope[0]]
        with locking_product_conversations(lock_receipts):
            with self.store.transaction() as db:
                current = originals(db)
                if lock_scope(db, current) != expected_scope:
                    raise WorkLifecycleError("PRODUCT_WORK_BINDING_CHANGED")
                parent = None
                for member, receipt in zip(members, current):
                    rid, step = member["request_id"], member["step_ref"]
                    parent = ensure_product_work(self.store, db, member["kind"], owner, rid, receipt,
                        {"product_ref": product_ref, "step_ref": step}, source=source)
                    if (not _unsent(receipt) and not _rejected_without_conversation(member["kind"], receipt)
                            and parent["state"] in {"queued", "active"}):
                        # This is lifecycle/physical ownership evidence only.
                        # Adoption never reserves the new product-capacity
                        # marker; a future send claims it atomically.
                        parent.update(state="active", slot_held=True)
                        _save(self.store, db, parent)
                    self.store.write_receipt(db, member["kind"], owner, rid, receipt)
                return self._projection(db, parent)

    def close(self, identity, product_ref, results_saved=True, outcome="completed", expected_version=None):
        if outcome not in {"completed", "failed", "cancelled"}:
            raise WorkLifecycleError("PRODUCT_WORK_OUTCOME_INVALID", 400)
        if results_saved is not True:
            raise WorkLifecycleError("PRODUCT_WORK_RESULTS_SAVE_REQUIRED", 400)
        if expected_version is not None and (type(expected_version) is not int or expected_version < 1):
            raise WorkLifecycleError("PRODUCT_WORK_VERSION_INVALID", 400)
        with self.store.transaction() as db:
            parent = self._load(db, identity, product_ref)
            version = parent.get("version", 1)
            if expected_version is None and version > 1:
                raise WorkLifecycleError("PRODUCT_WORK_VERSION_REQUIRED")
            if expected_version is not None and expected_version != version:
                raise WorkLifecycleError("PRODUCT_WORK_VERSION_CONFLICT")
            if parent.get("outcome"):
                if parent["outcome"] != outcome or parent["results_saved"] != results_saved:
                    raise WorkLifecycleError("PRODUCT_WORK_CLOSE_CONFLICT")
                return self._projection(db, parent)
            members = self._members(db, parent)
            reasons = self._reasons(db, parent, members, outcome)
            if reasons:
                raise WorkLifecycleError(reasons[0])
            targets = {}
            chains = self._missing_node_chains(db, parent, members)
            advanced_chains = self._owned_advanced_successor_chains(db, parent, members)
            chain_member_ids = set()
            special_targets = {}
            members_by_identity = {(member["kind"], member["request_id"]): (member, receipt)
                                   for member, receipt in members}
            for chain in chains.values():
                chain_member_ids.update(("text", chain[name]["request_id"])
                                        for name in ("original", "rejected", "continue"))
                chain_member_ids.add(("image", chain["image"]["id"]))
                archive_member, archive = chain["archive_member"], chain["archive"]
                chain_member_ids.add((archive_member["kind"], archive_member["request_id"]))
                canonical = (archive["provider_account_identity"], archive["conversation_id"])
                target = archive_member["kind"], archive_member["request_id"]
                if canonical in special_targets and special_targets[canonical] != target:
                    raise WorkLifecycleError("PRODUCT_WORK_ARCHIVE_BINDING_CHANGED")
                special_targets[canonical] = target
            for chain in advanced_chains.values():
                archive_member, archive = members_by_identity[("text", chain["successor"]["request_id"])]
                chain_member_ids.update(("text", chain[name]["request_id"])
                                        for name in ("original", "successor"))
                canonical = archive["provider_account_identity"], archive["conversation_id"]
                target = archive_member["kind"], archive_member["request_id"]
                if canonical in special_targets and special_targets[canonical] != target:
                    raise WorkLifecycleError("PRODUCT_WORK_ARCHIVE_BINDING_CHANGED")
                special_targets[canonical] = target
            chains.update(advanced_chains)
            for member, receipt in members:
                identity = member["kind"], member["request_id"]
                if identity in chain_member_ids:
                    chain = chains.get(receipt.get("request_id"))
                    if not chain:
                        continue
                    if chain.get("advanced_by_own_successor"):
                        member, receipt = members_by_identity[("text", chain["successor"]["request_id"])]
                    else:
                        member, receipt = members_by_identity[(chain["archive_member"]["kind"],
                                                               chain["archive_member"]["request_id"])]
                if _cancelled_unsent_step(self.store, db, parent, member, receipt):
                    continue
                if _unsent(receipt) and outcome == "cancelled":
                    receipt.update(status="cancelled", error_code="PRODUCT_WORK_CANCELLED")
                    self.store.write_receipt(db, member["kind"], member["owner"], member["request_id"], receipt)
                conversation = receipt.get("conversation_id")
                if conversation:
                    account = receipt.get("provider_account_identity") or receipt.get("provider_binding_id")
                    if not account or not receipt.get("provider_binding_id"):
                        raise WorkLifecycleError("PRODUCT_WORK_ARCHIVE_BINDING_REQUIRED")
                    canonical = next((key for key, (_, other) in targets.items() if _same_conversation(receipt, other)),
                                     (account, conversation))
                    if canonical in special_targets and (member["kind"], member["request_id"]) != special_targets[canonical]:
                        continue
                    previous = targets.get(canonical)
                    if (canonical in special_targets or previous is None
                            or receipt.get("_sequence", 0) > previous[1].get("_sequence", 0)):
                        targets[canonical] = member, receipt
                elif _codex_product_bridge(receipt):
                    continue
                elif _rejected_chat_replaced(self.store, db, member["owner"], receipt, members):
                    continue
                elif (receipt.get("_submission_started") or receipt.get("upstream_submission_started")
                      or receipt.get("upstream_outcome") not in {"not_sent", "not_submitted"}
                      and not (outcome == "cancelled" and receipt.get("status") == "cancelled")):
                    raise WorkLifecycleError("PRODUCT_WORK_ARCHIVE_BINDING_REQUIRED")
            now = float(self.clock())
            control = admission_control(parent)
            # AI completion releases product admission before archive or later
            # publication work.  The lifecycle slot stays intact until the
            # existing archive closeout decides otherwise.
            parent.update(state="closing", outcome=outcome, results_saved=True, results_saved_at=now,
                          cleanup_pending=bool(targets), updated_at=now,
                          admission_control={**control, "slot_held": False})
            for canonical, (member, receipt) in targets.items():
                archive_identity = [parent["key"], canonical]
                if version > 1:
                    archive_identity.append(version)
                digest = hashlib.sha256(json.dumps(archive_identity).encode()).hexdigest()
                key = "work:product_archive:" + digest
                child = {"key": key, "_product_archive_parent": parent["key"], "kind": member["kind"],
                         "_product_archive_version": version,
                         "owner": member["owner"], "last_request_id": member["request_id"], "work_ref": canonical[1],
                         "target": {k: receipt.get(k) for k in ("provider_binding_id", "provider_account_identity", "conversation_id")},
                         "state": "completed", "slot_held": False, "version": version, "updated_at": now,
                         "archive": {"status": "pending", "desired": True, "version": version, "request_id": member["request_id"],
                                     "requested_at": now, "next_at": now, "attempts": 0}}
                self.store.set_runtime(db, key, child)
                parent["archive_keys"].append(key)
            if not targets:
                parent.update(state=outcome, slot_held=False, cleanup_pending=False)
            _save(self.store, db, parent)
            result = self._projection(db, parent)
        admission = getattr(self.text, "admission", None)
        if admission:
            admission.wake()
        return result

    def admission_control(self, identity, product_ref, paused, expected_version):
        """CAS pause/resume for admission only; never sends or reserves a turn."""
        if type(paused) is not bool or type(expected_version) is not int or expected_version < 0:
            raise WorkLifecycleError("PRODUCT_WORK_ADMISSION_VERSION_INVALID", 400)
        with self.store.transaction() as db:
            parent = self._load(db, identity, product_ref)
            control = admission_control(parent)
            if control["version"] != expected_version:
                raise WorkLifecycleError("PRODUCT_WORK_ADMISSION_VERSION_CONFLICT")
            members = self._members(db, parent)
            if paused:
                if control["state"] in {"pausing", "paused"}:
                    return self._projection(db, parent)
                live = any(_live_product_claim(receipt, float(self.clock())) for _member, receipt in members)
                parent["admission_control"] = {
                    "state": "pausing" if live else "paused",
                    "version": control["version"] + 1,
                    # A pre-marker parent can already have a genuine live
                    # claim.  Preserve that occupancy through the lease even
                    # though it had no durable marker before this control.
                    "slot_held": bool(live),
                }
            else:
                if control["state"] == "active":
                    return self._projection(db, parent)
                if control["state"] == "pausing":
                    # Never turn a still-live paused lease into an unheld
                    # active product: its original worker can still reach the
                    # send edge.  Once the lease is observably over, publish
                    # the paused state first; a separate versioned resume may
                    # then compete for capacity normally.
                    if any(_live_product_claim(receipt, float(self.clock())) for _member, receipt in members):
                        raise WorkLifecycleError("PRODUCT_WORK_ADMISSION_PAUSING")
                    # Normalize this transient state in the same transaction
                    # without consuming a version, then honor this exact CAS
                    # resume as paused(version) -> active(version + 1).
                    control = {**control, "state": "paused", "slot_held": False}
                if control["state"] != "paused" or control["slot_held"]:
                    raise WorkLifecycleError("PRODUCT_WORK_ADMISSION_PAUSING")
                # Resume authorizes the existing durable parent to compete for
                # a new admission slot; it cannot revive a paused send claim.
                parent["admission_control"] = {
                    "state": "active", "version": control["version"] + 1, "slot_held": False,
                }
            _save(self.store, db, parent)
            return self._projection(db, parent)

    def load_archive_target(self, db, child):
        parent = self.store.runtime(db, child["_product_archive_parent"])
        if (not parent or parent["state"] != "closing" or child["key"] not in parent["archive_keys"]
                or child.get("_product_archive_version", 1) != parent.get("version", 1)):
            raise WorkLifecycleError("PRODUCT_WORK_ARCHIVE_CONFLICT")
        members = self._members(db, parent)
        reasons = self._reasons(db, parent, members, parent["outcome"])
        if reasons:
            raise WorkLifecycleError(reasons[0])
        if not self.archive_turns_ready(db, child, [receipt for _, receipt in members]):
            raise WorkLifecycleError("WORK_TURN_UNFINISHED")
        for chain in self._missing_node_chains(db, parent, members).values():
            archive_member, archive = chain["archive_member"], chain["archive"]
            if (child["target"].get("provider_account_identity") == archive["provider_account_identity"]
                    and child["target"].get("conversation_id") == archive["conversation_id"]
                    and (child["kind"], child["last_request_id"]) != (archive_member["kind"], archive_member["request_id"])):
                raise WorkLifecycleError("PRODUCT_WORK_ARCHIVE_BINDING_CHANGED")
        for chain in self._owned_advanced_successor_chains(db, parent, members).values():
            successor = chain["successor"]
            if (child["target"].get("provider_account_identity") == successor["provider_account_identity"]
                    and child["target"].get("conversation_id") == successor["conversation_id"]
                    and (child["kind"], child["last_request_id"]) != ("text", successor["request_id"])):
                raise WorkLifecycleError("PRODUCT_WORK_ARCHIVE_BINDING_CHANGED")
        receipt = self.store.read_receipt(db, child["kind"], child["owner"], child["last_request_id"])
        if not receipt or any(receipt.get(k) != v for k, v in child["target"].items()):
            raise WorkLifecycleError("PRODUCT_WORK_ARCHIVE_BINDING_CHANGED")
        return receipt, child, [r for _, r in members]

    def archive_confirmed(self, db, child):
        parent = self.store.runtime(db, child["_product_archive_parent"])
        if (not parent or parent["state"] != "closing" or child["key"] not in parent["archive_keys"]
                or child.get("_product_archive_version", 1) != parent.get("version", 1)):
            return
        if all((self.store.runtime(db, key) or {}).get("archive", {}).get("status") == "confirmed"
               for key in parent["archive_keys"]):
            parent.update(state=parent["outcome"], slot_held=False, cleanup_pending=False, updated_at=float(self.clock()))
            _save(self.store, db, parent)
