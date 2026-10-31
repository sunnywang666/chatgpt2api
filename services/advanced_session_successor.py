"""Local proof checks for one explicitly requested isolated ADVANCED successor.

No transport, scheduling, or persistence outside the caller's existing transaction.
"""
from copy import deepcopy
import hashlib
import json
import math

from services.conversation_binding_service import ConversationBindingError

MODE = "advanced_new_session"
FIELD = "_advanced_session"
PROTOCOL = "advanced-session-successor-v1"


def reject():
    raise ConversationBindingError("advanced session successor evidence changed", code="CHAT_ADVANCED_SESSION_INVALID")


def client_conversation_id(original_id, child_id):
    return "advanced-session:" + hashlib.sha256((original_id + "\0" + child_id).encode()).hexdigest()


def _idle(receipt, now):
    if receipt.get("_executing") is not False or receipt.get("_turn_reserved") is not False:
        return False
    if any(receipt.get(k) for k in ("_recovery_paused", "_recovery_suppressed", "_send_hold", "_completion", "_completion_of")):
        return False
    for claim, until in (("_claim_id", "_claim_until"), ("recovery_claim_id", "recovery_lease_until"),
                         ("_recovery_claim_id", "_recovery_claim_until")):
        value = receipt.get(until)
        if claim != "_claim_id" and (receipt.get(claim) or value is not None):
            return False
        if receipt.get(claim) and (type(value) not in (int, float) or not math.isfinite(value) or value > now):
            return False
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value > now):
            return False
    return True


def _evidence(receipt):
    evidence = receipt.get("_completed_successor_evidence")
    required = {"conversation_id", "request_message_id", "successor_request_message_id",
                "successor_request_parent_message_id", "successor_final_message_id", "observed_at"}
    if not isinstance(evidence, dict) or set(evidence) not in (required, required | {"tail_steps"}):
        reject()
    if any(not isinstance(evidence.get(k), str) or not evidence[k].strip() for k in required - {"observed_at"}):
        reject()
    if evidence["conversation_id"] != receipt.get("conversation_id") or evidence["request_message_id"] != receipt.get("request_message_id"):
        reject()
    if type(evidence["observed_at"]) not in (int, float) or not math.isfinite(evidence["observed_at"]) or evidence["observed_at"] <= 0:
        reject()
    ids = [evidence["request_message_id"], evidence["successor_request_message_id"], evidence["successor_final_message_id"]]
    if "tail_steps" in evidence:
        steps = evidence["tail_steps"]
        if not isinstance(steps, list) or not 1 <= len(steps) <= 127:
            reject()
        for step in steps:
            if (not isinstance(step, dict) or set(step) != {"request_message_id", "final_message_id"}
                    or any(not isinstance(value, str) or not value.strip() for value in step.values())):
                reject()
            ids.extend(step.values())
    if len(set(ids)) != len(ids):
        reject()
    return evidence


def source_eligible(original, now):
    try:
        evidence = _evidence(original)
    except ConversationBindingError:
        return False
    return bool(_idle(original, now) and original.get("status") == "unknown"
        and original.get("_route", "chat") == "chat" and original.get("_operation", "text") == "text"
        and type(original.get("_expected_sends", 1)) is int and original.get("_expected_sends", 1) == 1
        and original.get("error_code") == "CONVERSATION_OUTCOME_UNKNOWN"
        and original.get("recovery_reason") == "REQUEST_CONVERSATION_ADVANCED"
        and original.get("_submission_started") is True and original.get("upstream_unfinished") is False
        and original.get("_upstream_terminal") is not True
        and original.get("upstream_outcome") in (None, "unknown")
        and not any(original.get(k) for k in ("_supersedes_request_id", "_derived_input", FIELD,
             "_public_session_ref", "_forward_protocol", "_codex_fallback_of", "_turn_end_evidence"))
        and not any(k in original for k in ("content", "result", "data"))
        and original.get("_source") == "internal:listing"
        and original.get("_scheduling_owner", "internal:listing") == "internal:listing"
        and isinstance(original.get("_product_work"), dict) and bool(original.get("_product_work_key"))
        and evidence["observed_at"] <= now)


def _media(value):
    result = []
    if isinstance(value, list):
        for item in value:
            result.extend(_media(item))
    elif isinstance(value, dict):
        if "type" in value:
            if value.get("type") != "text":
                return [value]
            try:
                label = json.loads(value.get("text", ""))
            except (TypeError, ValueError):
                label = None
            if isinstance(label, dict) and "image_ref" in label:
                return [value]
        else:
            result.extend(_media(value.get("content")))
    return result


def _texts(messages):
    for message in messages:
        if not isinstance(message, dict):
            reject()
        content = message.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    yield part["text"]


def _load(store, db, owner, request_id):
    row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?", (owner, request_id)).fetchone()
    if not row:
        reject()
    receipt = json.loads(row[1])
    body = store.load_input(receipt["_input_ref"])
    from services.text_task_service import TextTaskService
    if TextTaskService._submission_identity(owner, body) != (request_id, row[0]):
        reject()
    return row[0], receipt, body


def retained_context_input(owner, request_id, digest, receipt, body):
    """Successful first-turn inputs may predate their physical session IDs."""
    from services.text_task_service import TextTaskService as Text
    from services.product_work import _key
    metadata = receipt.get("_product_work")
    content = receipt.get("content")
    return bool(
        receipt.get("request_id") == request_id and receipt.get("status") == "succeeded"
        and receipt.get("_submission_started") is True and receipt.get("_upstream_terminal") is not False
        and isinstance(content, str) and content.strip() and receipt.get("_source") == "internal:listing"
        and isinstance(metadata, dict) and isinstance(metadata.get("product_ref"), str)
        and receipt.get("_product_work_key") == _key("internal:listing", owner, metadata["product_ref"])
        and Text._submission_identity(owner, body) == (request_id, digest) and Text._json_exportable(body)
        and all(isinstance(receipt.get(k), str) and receipt[k] for k in (
            "provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id",
            "request_message_id", "parent_message_id"))
        and body.get("client_conversation_id") == receipt["client_conversation_id"]
        and body.get("product_work") == metadata
        and body.get("_route", "chat") == receipt.get("_route", "chat") == "chat"
        and body.get("_operation", "text") == receipt.get("_operation", "text") == "text"
        and not any(receipt.get(k) or body.get(k) for k in (
            "_public_session_ref", "_forward_protocol", "_forward", "_codex_fallback_of", "_editable"))
        and not any(body.get(k) and body[k] != receipt.get(k) for k in (
            "provider_binding_id", "provider_account_identity", "conversation_id"))
        and (not body.get("parent_message_id") or body["parent_message_id"] == (
            receipt.get("_submission_parent_message_id") or receipt.get("request_parent_message_id"))))


def _context(store, db, owner, original, ref, messages):
    if (not isinstance(ref, dict) or set(ref) != {"request_id", "input_hash", "result_sha256"}
            or any(not isinstance(v, str) or not v for v in ref.values())):
        reject()
    digest, context, body = _load(store, db, owner, ref["request_id"])
    content = context.get("content")
    if (digest != ref["input_hash"] or context.get("status") != "succeeded"
            or context.get("_upstream_terminal") is False or context.get("_submission_started") is not True
            or not isinstance(content, str) or not content.strip()
            or hashlib.sha256(content.encode()).hexdigest() != ref["result_sha256"]
            or any(context.get(k) != original.get(k) for k in ("_product_work_key", "_source", "provider_account_identity",
                                                               "provider_binding_id", "conversation_id", "client_conversation_id"))
            or context.get("parent_message_id") != (original.get("_submission_parent_message_id") or original.get("request_parent_message_id"))
            or not any(content in text for text in _texts(messages))
            or _media(body.get("messages")) != _media(messages)):
        reject()
    if not retained_context_input(owner, ref["request_id"], digest, context, body):
        reject()
    return context


def build_body(store, db, owner, original_id, payload, now, *, frozen_proof=None):
    from services.text_task_service import TextTaskService as Text
    required = {"client_request_id", "original_input_hash", "messages", "model", "thinking_effort", "mode", "derivation_kind", "context_ref"}
    if (not isinstance(payload, dict) or set(payload) != required or payload.get("mode") != MODE
            or any(not isinstance(payload.get(k), str) or not payload[k].strip() for k in required - {"messages", "context_ref"})
            or payload["client_request_id"] == original_id or payload["client_request_id"] != payload["client_request_id"].strip()
            or len(payload["client_request_id"]) > 200 or len(payload["model"]) > 200
            or len(payload["thinking_effort"]) > 100 or len(payload["derivation_kind"]) > 128
            or not Text._json_exportable(payload["messages"])
            or not isinstance(payload["messages"], list) or not payload["messages"]):
        reject()
    digest, original, retained = _load(store, db, owner, original_id)
    if digest != payload["original_input_hash"] or not Text._bound_workbench_retained_input(owner, original_id, digest, original, retained):
        reject()
    _context(store, db, owner, original, payload["context_ref"], payload["messages"])
    if frozen_proof is None:
        if not source_eligible(original, now):
            reject()
        evidence = deepcopy(_evidence(original))
        proof = {"protocol": PROTOCOL, "original_request_id": original_id, "original_input_hash": digest,
            "original_account_identity": original["provider_account_identity"], "original_binding_id": original["provider_binding_id"],
            "original_conversation_id": original["conversation_id"], "original_request_message_id": original["request_message_id"],
            "original_head_message_id": (evidence.get("tail_steps") or [{"final_message_id": evidence["successor_final_message_id"]}])[-1]["final_message_id"],
            "completed_successor_evidence": evidence, "context_ref": deepcopy(payload["context_ref"]),
            "product_work": deepcopy(original["_product_work"]), "scheduling": deepcopy(original.get("_scheduling")),
            "child_request_id": payload["client_request_id"],
            "child_client_conversation_id": client_conversation_id(original_id, payload["client_request_id"])}
    else:
        if (not isinstance(frozen_proof, dict) or frozen_proof.get("context_ref") != payload["context_ref"]
                or frozen_proof.get("original_request_id") != original_id
                or frozen_proof.get("child_request_id") != payload["client_request_id"]):
            reject()
        proof = deepcopy(frozen_proof)
    body = deepcopy(retained)
    for key in ("provider_binding_id", "conversation_id", "parent_message_id", "_requested_account_ref"):
        body.pop(key, None)
    body.update(client_request_id=payload["client_request_id"], supersedes_request_id=original_id,
        client_conversation_id=client_conversation_id(original_id, payload["client_request_id"]),
        provider_account_identity=original["provider_account_identity"], _text_only_binding=True,
        messages=deepcopy(payload["messages"]),
        model=payload["model"], thinking_effort=payload["thinking_effort"], **{FIELD: proof,
        Text._WORKBENCH_DERIVED_MARKER: {"protocol": Text._WORKBENCH_DERIVED_PROTOCOL, "original_input_hash": digest,
                                      "mode": MODE, "derivation_kind": payload["derivation_kind"]}})
    return body


def public_proof(receipt):
    proof = receipt.get(FIELD)
    if not isinstance(proof, dict) or proof.get("protocol") != PROTOCOL:
        return None
    return {**deepcopy(proof), "provider_binding_id": receipt.get("provider_binding_id"),
            "conversation_id": receipt.get("conversation_id")}


def prepare_receipt(store, db, owner, body, receipt, now):
    from services.text_task_service import TextTaskService as Text
    proof = body.get(FIELD)
    if not isinstance(proof, dict):
        reject()
    child_id, child_hash = Text._submission_identity(owner, body)
    receipt.update({FIELD: deepcopy(proof), "_supersedes_request_id": proof["original_request_id"],
        "_supersedes_input_hash": proof["original_input_hash"], "provider_account_identity": proof["original_account_identity"],
        "_requested_account_identity": proof["original_account_identity"],
        Text._WORKBENCH_DERIVED_MARKER: Text._workbench_derived_marker(original_input_hash=proof["original_input_hash"],
            child_input_hash=child_hash, mode=MODE, derivation_kind=body[Text._WORKBENCH_DERIVED_MARKER]["derivation_kind"])})
    validate_successor(store, db, owner, {**receipt, "_workbench_derived_body": body}, now)


def validate_successor(store, db, owner, child, now, *, after_send=False):
    """Validate frozen identities; only a sent child may retain a late root answer."""
    from services.text_task_service import TextTaskService as Text
    try:
        proof = child[FIELD]
        original_id = child["_supersedes_request_id"]
        digest, original, retained = _load(store, db, owner, original_id)
        body = child.get("_workbench_derived_body") or store.load_input(child["_input_ref"])
        marker = body[Text._WORKBENCH_DERIVED_MARKER]
        payload = {"client_request_id": body["client_request_id"], "original_input_hash": digest,
            "messages": body["messages"], "model": body["model"], "thinking_effort": body["thinking_effort"],
            "mode": MODE, "derivation_kind": marker["derivation_kind"], "context_ref": proof["context_ref"]}
        expected = build_body(store, db, owner, original_id, payload, now, frozen_proof=proof)
        sent = after_send and child.get("_submission_started") is True
        if after_send and not sent:
            reject()
        if not sent and not source_eligible(original, now):
            reject()
        # A late original answer is retained independently; no new retry is authorized.
        # After this child crossed its send boundary, the old physical session
        # is independent history. New discussion, read failures or a late answer
        # there cannot revoke this frozen relationship. Capacity owners still
        # inspect the returned original's live claims separately.
        historical = {**original, "status": "unknown", "error_code": "CONVERSATION_OUTCOME_UNKNOWN",
                      "recovery_reason": "REQUEST_CONVERSATION_ADVANCED", "_completed_successor_evidence": proof["completed_successor_evidence"]}
        for key in ("content", "result", "data", "_turn_end_evidence"):
            historical.pop(key, None)
        historical.update(_upstream_terminal=False, upstream_unfinished=False, upstream_outcome="unknown")
        # Reconstruct the full immutable proof, never trust arbitrary extra fields.
        ev = _evidence(historical)
        exact = {"protocol": PROTOCOL, "original_request_id": original_id, "original_input_hash": digest,
            "original_account_identity": original["provider_account_identity"], "original_binding_id": original["provider_binding_id"],
            "original_conversation_id": original["conversation_id"], "original_request_message_id": original["request_message_id"],
            "original_head_message_id": (ev.get("tail_steps") or [{"final_message_id": ev["successor_final_message_id"]}])[-1]["final_message_id"],
            "completed_successor_evidence": ev, "context_ref": proof["context_ref"], "product_work": original["_product_work"],
            "scheduling": original.get("_scheduling"), "child_request_id": child["request_id"],
            "child_client_conversation_id": client_conversation_id(original_id, child["request_id"])}
        from services.product_work import _key
        product_key = _key("internal:listing", owner, proof["product_work"]["product_ref"])
        if (proof != exact or body != expected or body.get(FIELD) != proof
                or any(original.get(k) for k in ("_supersedes_request_id", "_derived_input", FIELD))
                or original.get("_source") != "internal:listing"
                or original.get("_scheduling_owner", "internal:listing") != "internal:listing"
                or original.get("_product_work_key") != product_key
                or child.get("_product_work_key", product_key) != product_key
                or child.get("_supersedes_input_hash") != digest
                or child.get("client_conversation_id") != proof["child_client_conversation_id"]
                or child.get("provider_account_identity") != proof["original_account_identity"]
                or child.get("_requested_account_identity") != proof["original_account_identity"]
                or child.get("_source", "internal:listing") != "internal:listing"
                or child.get("_product_work", proof["product_work"]) != proof["product_work"]
                or child.get("provider_binding_id") == proof["original_binding_id"]
                or child.get("conversation_id") == proof["original_conversation_id"]):
            reject()
        if not sent:
            live = _evidence(original)
            if {k:v for k,v in live.items() if k != "observed_at"} != {k:v for k,v in ev.items() if k != "observed_at"}:
                reject()
        child_id, child_hash = Text._submission_identity(owner, body)
        if child_id != child["request_id"] or child.get(Text._WORKBENCH_DERIVED_MARKER) != Text._workbench_derived_marker(
                original_input_hash=digest, child_input_hash=child_hash, mode=MODE, derivation_kind=marker["derivation_kind"]):
            reject()
        row = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", (owner, child_id)).fetchone()
        if row and (row[0] != child_hash or child.get("_source") != "internal:listing"
                    or child.get("_product_work_key") != product_key or child.get("_product_work") != proof["product_work"]):
            reject()
        from services.pool_admission import unfinished
        external_users = {ev["successor_request_message_id"]} | {
            step["request_message_id"] for step in ev.get("tail_steps", [])}
        for kind, other_owner, other_id, other in store.receipts(db):
            if other_owner == owner and kind == "text" and other_id in {original_id, child_id}:
                continue
            if other_owner == owner and (other.get("_supersedes_request_id") == original_id or other.get("_codex_fallback_of") == original_id):
                reject()
            if (other.get("provider_account_identity") == original["provider_account_identity"]
                    and other.get("request_message_id") in external_users):
                # Registered or colliding identities cannot be relabelled external discussion.
                reject()
            same_product = other.get("_product_work_key") == original["_product_work_key"]
            same_chat = (other.get("provider_account_identity") == original["provider_account_identity"]
                         and other.get("conversation_id") == original["conversation_id"])
            if (same_product or same_chat) and unfinished(kind, other):
                reject()
        return original
    except (KeyError, TypeError, ValueError, OSError):
        reject()


def check_fresh_read(fresh, original, proof):
    """Creation uses an exact reader without writing its observation to the root."""
    if (not isinstance(fresh, dict) or fresh.get("status") not in {"running", "unknown"}
            or fresh.get("recovery_reason") != "REQUEST_CONVERSATION_ADVANCED"
            or any(fresh.get(k) != original.get(k) for k in (
                "provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id"))):
        reject()
    live = _evidence({**original, "_completed_successor_evidence": fresh.get("_completed_successor_evidence")})
    if {k:v for k,v in live.items() if k != "observed_at"} != {
            k:v for k,v in proof["completed_successor_evidence"].items() if k != "observed_at"}:
        reject()
