"""Opt-in product ownership over original text/image receipts in task_runtime.

A product owns capacity until its durable conversation archives are confirmed.
No result, input hash, request sequence or ordinary work identity is replaced.
"""
from __future__ import annotations

import hashlib
import json
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
    if not isinstance(value, dict) or set(value) != {"product_ref", "step_ref"}:
        raise WorkLifecycleError("PRODUCT_WORK_INVALID", 400)
    return {k: _ref(value[k]) for k in ("product_ref", "step_ref")}


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
def locking_product_conversations(receipts):
    """Serialize ownership and raw archive for a physical conversation.

    Call outside SQLite transactions. Reuse the account service's in-process
    mutex map, with a separate namespace from generation's client cursor locks.
    """
    from services.account_service import account_service
    keys = _physical_lock_keys(list(receipts))
    locks = [account_service.conversation_binding_lock("product-archive:" + account, conversation)
             for account, conversation in keys]
    with ExitStack() as stack:
        seen = set()
        for lock in locks:
            if id(lock) not in seen:
                stack.enter_context(lock)
                seen.add(id(lock))
        yield


def _has_physical(receipt):
    return bool(receipt.get("conversation_id") and (receipt.get("provider_account_identity") or receipt.get("provider_binding_id")))


def _source(receipt):
    return receipt.get("_scheduling_owner") or receipt.get("_source")


def _key(source, owner, ref):
    raw = json.dumps([source, owner, ref], separators=(",", ":"))
    return "product_work:" + hashlib.sha256(raw.encode()).hexdigest()


def state_snapshot(store, db):
    return {key: json.loads(raw) for key, raw in db.execute(
        "SELECT name,value FROM task_runtime WHERE name LIKE 'product_work:%'")}


def _save(store, db, parent):
    store.set_runtime(db, parent["key"], parent)


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
    if parent and (parent["state"] != "active" or not parent.get("slot_held")):
        raise WorkLifecycleError("PRODUCT_WORK_NOT_ACTIVE")
    if parent:
        _check_physical(store, db, receipt, parent["key"])
    return parent


def record_slot(store, db, receipt, held=True):
    parent = _attached(store, db, receipt)
    if parent and held and parent["state"] not in {"queued", "active"}:
        raise WorkLifecycleError("PRODUCT_WORK_NOT_ACTIVE")
    if parent and held:
        _check_physical(store, db, receipt, parent["key"])
    if parent and held:
        parent.update(state="active", slot_held=True)
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
    return bool(
        receipt.get("_route") == "codex"
        and forward == "codex"
        and receipt.get("_codex_fallback_protocol") == "codex-responses-v1"
        and isinstance(receipt.get("_codex_origin_request_id"), str)
        and bool(receipt.get("_codex_origin_request_id"))
        and (bool(fallback) ^ bool(continuation))
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
    key = _key(source, owner, metadata["product_ref"])
    if receipt.get("_product_work_key") not in {None, key}:
        raise WorkLifecycleError("PRODUCT_WORK_BINDING_CONFLICT")
    _check_physical(store, db, receipt, key)
    parent = store.runtime(db, key)
    member = {"kind": kind, "owner": owner, "request_id": request_id, "step_ref": metadata["step_ref"]}
    if parent:
        if parent["owner"] != owner or parent["source"] != source:
            raise WorkLifecycleError("PRODUCT_WORK_OWNER_CONFLICT")
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
            old = same_step[-1]
            previous = store.read_receipt(db, old["kind"], old["owner"], old["request_id"])
            if old["kind"] != kind or not previous or not _successor(store, db, owner, previous, receipt):
                raise WorkLifecycleError("PRODUCT_WORK_STEP_CONFLICT")
    else:
        parent = {"key": key, "owner": owner, "source": source, "product_ref": metadata["product_ref"],
                  "state": "queued", "slot_held": False, "members": [], "archive_keys": [],
                  "results_saved": False, "outcome": None, "cleanup_pending": False,
                  "created_at": time.time()}
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


class ProductWorkService:
    def __init__(self, text_service, image_service, *, clock=time.time):
        self.text, self.images, self.store, self.clock = text_service, image_service, text_service.store, clock

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

    def _reasons(self, db, parent, members, outcome=None):
        reasons = set()
        receipts = [r for _, r in members]
        for member, receipt in members:
            if outcome == "cancelled" and _unsent(receipt):
                continue
            replaced = _rejected_chat_replaced(self.store, db, member["owner"], receipt, members)
            if not replaced and _unresolved(receipt, receipts):
                reasons.add("PRODUCT_WORK_TURN_UNFINISHED")
            status = receipt.get("status")
            if outcome == "completed" and status not in {"succeeded", "success"}:
                if not (replaced
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
        archives = [self.store.runtime(db, key) for key in parent.get("archive_keys", [])]
        confirmed = sum(bool(row and (row.get("archive") or {}).get("status") == "confirmed") for row in archives)
        reasons = self._reasons(db, parent, members)
        if parent["state"] == "closing" and confirmed != len(archives):
            reasons.append("PRODUCT_WORK_ARCHIVE_PENDING")
        return {"protocol": "product-work-v1", "product_ref": parent["product_ref"], "state": parent["state"],
                "slot_held": bool(parent["slot_held"]), "results_saved": bool(parent.get("results_saved")),
                "outcome": parent.get("outcome"), "cleanup_pending": bool(parent.get("cleanup_pending")),
                "members": [{"kind": m["kind"], "request_id": m["request_id"], "step_ref": m["step_ref"],
                             "status": r.get("status")} for m, r in members],
                "archive": {"total": len(archives), "confirmed": confirmed, "pending": len(archives) - confirmed},
                "blocked_reasons": sorted(set(reasons))}

    def get(self, identity, product_ref):
        with self.store.transaction() as db:
            return self._projection(db, self._load(db, identity, product_ref))

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
                    if not _unsent(receipt) and parent["state"] in {"queued", "active"}:
                        parent.update(state="active", slot_held=True)
                        _save(self.store, db, parent)
                    self.store.write_receipt(db, member["kind"], owner, rid, receipt)
                return self._projection(db, parent)

    def close(self, identity, product_ref, results_saved=True, outcome="completed"):
        if outcome not in {"completed", "failed", "cancelled"}:
            raise WorkLifecycleError("PRODUCT_WORK_OUTCOME_INVALID", 400)
        if results_saved is not True:
            raise WorkLifecycleError("PRODUCT_WORK_RESULTS_SAVE_REQUIRED", 400)
        with self.store.transaction() as db:
            parent = self._load(db, identity, product_ref)
            if parent.get("outcome"):
                if parent["outcome"] != outcome or parent["results_saved"] != results_saved:
                    raise WorkLifecycleError("PRODUCT_WORK_CLOSE_CONFLICT")
                return self._projection(db, parent)
            members = self._members(db, parent)
            reasons = self._reasons(db, parent, members, outcome)
            if reasons:
                raise WorkLifecycleError(reasons[0])
            targets = {}
            for member, receipt in members:
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
                    previous = targets.get(canonical)
                    if previous is None or receipt.get("_sequence", 0) > previous[1].get("_sequence", 0):
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
            parent.update(state="closing", outcome=outcome, results_saved=True, results_saved_at=now,
                          cleanup_pending=bool(targets), updated_at=now)
            for canonical, (member, receipt) in targets.items():
                digest = hashlib.sha256(json.dumps([parent["key"], canonical]).encode()).hexdigest()
                key = "work:product_archive:" + digest
                child = {"key": key, "_product_archive_parent": parent["key"], "kind": member["kind"],
                         "owner": member["owner"], "last_request_id": member["request_id"], "work_ref": canonical[1],
                         "target": {k: receipt.get(k) for k in ("provider_binding_id", "provider_account_identity", "conversation_id")},
                         "state": "completed", "slot_held": False, "version": 1, "updated_at": now,
                         "archive": {"status": "pending", "desired": True, "version": 1, "request_id": member["request_id"],
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

    def load_archive_target(self, db, child):
        parent = self.store.runtime(db, child["_product_archive_parent"])
        if not parent or parent["state"] != "closing" or child["key"] not in parent["archive_keys"]:
            raise WorkLifecycleError("PRODUCT_WORK_ARCHIVE_CONFLICT")
        members = self._members(db, parent)
        reasons = self._reasons(db, parent, members, parent["outcome"])
        if reasons:
            raise WorkLifecycleError(reasons[0])
        receipt = self.store.read_receipt(db, child["kind"], child["owner"], child["last_request_id"])
        if not receipt or any(receipt.get(k) != v for k, v in child["target"].items()):
            raise WorkLifecycleError("PRODUCT_WORK_ARCHIVE_BINDING_CHANGED")
        return receipt, child, [r for _, r in members]

    def archive_confirmed(self, db, child):
        parent = self.store.runtime(db, child["_product_archive_parent"])
        if not parent or parent["state"] != "closing":
            return
        if all((self.store.runtime(db, key) or {}).get("archive", {}).get("status") == "confirmed"
               for key in parent["archive_keys"]):
            parent.update(state=parent["outcome"], slot_held=False, cleanup_pending=False, updated_at=float(self.clock()))
            _save(self.store, db, parent)
