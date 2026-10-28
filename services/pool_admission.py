"""Dispatch the existing durable receipts against one account occupancy view.

No waiting request owns an executor thread. Selection, occupancy, the original
account binding and fairness cursor commit in the existing receipt database.
Expired claims are fenced at the model-send edge, never inferred to be unsent.
"""
from __future__ import annotations

from services.image_thread import bind_waiting_threads, predecessor_state

from collections import deque
from contextlib import nullcontext
import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid

from services.admission_planner import (Need, Resource, Offer, OfferDiagnostic,
                                        RequestRef, WaitingRequest, Snapshot,
                                        choose_next, fairness_lane, fairness_subgroup_key)
from services.request_context import AdmissionLost, executing, safe_account_ref, reserving_original_read
from services.task_store import TaskStore, pending_image_result_ids
from services import workflow_scheduling, product_work
from utils.log import logger


def reset_unsent_image_attempt(kind, receipt):
    if not receipt.get("_submission_started") and (kind == "image" or receipt.get("_operation") == "image"):
        # Queue/cooldown time is outside a not-yet-sent active attempt.
        receipt.update(active_attempt_started_at=None,
                       active_attempt_deadline_at=None, started_ts=None)


_COMBINED_COPY_SOURCE = "internal:listing"
_COMBINED_COPY_MODEL = "gpt-5-6-thinking"
_COMBINED_COPY_EFFORT = "high"
_FIRSTSEND_TRIAL_MODEL = "gpt-5-6-instant"
_FIRSTSEND_TRIAL_EFFORT = "none"
_ORIGINAL_READ_CONTRIBUTOR_BURST = 3
_COMBINED_COPY_SENT_STAGES = frozenset({
    "send_guard_passed", "send_call_started", "response_headers_received",
    "first_output", "artifact_saved", "task_finished",
})
_COMBINED_COPY_RESULT_FIELDS = frozenset({
    "content", "result", "data", "result_file_ids", "result_sediment_ids",
})
_SAFE_SSE_EVENT_TYPES = ("resume_conversation_token", "server_ste_metadata", "stream_handoff")
_SAFE_RESUME_TOKEN_KINDS = frozenset({"topic", "other", "unknown", "multiple"})
_SAFE_STREAM_HANDOFF_OPTION_TYPES = ("resume_sse_endpoint", "subscribe_ws_topic")
_SAFE_STREAM_HANDOFF_STAGES = frozenset({
    "candidate_absent", "candidate_invalid", "candidate_bound",
    "ws_endpoint_get_entered", "ws_endpoint_get_completed", "ws_endpoint_invalid",
    "ws_connected", "ws_subscribed", "ws_terminal", "ws_no_terminal",
    "ws_budget_exhausted", "ws_transport_error", "ws_payload_rejected",
})


_EXPIRED_UNSENT_BEFORE_SEND = object()


def _verified_incomplete_compact_links(text_rows):
    """Return immutable direct parent/child compact links from one snapshot.

    This is intentionally a receipt-snapshot check, not the full supersede
    validator.  Original-result reads must remain available after the child
    has already been sent even when its parent is no longer eligible to send.
    Missing immutable lineage evidence therefore fails closed to released
    history priority without changing recovery eligibility or making a send.
    """
    from services.text_task_service import TextTaskService

    receipts = {}
    hashes = {}
    descendants = {}
    for owner, request_id, request_hash, receipt in text_rows:
        key = (owner, request_id)
        receipts[key] = receipt
        hashes[key] = request_hash
        for field in ("_supersedes_request_id", "_codex_fallback_of"):
            parent_id = receipt.get(field)
            if isinstance(parent_id, str) and parent_id:
                descendants.setdefault((owner, parent_id), set()).add(request_id)

    verified = []
    for (owner, request_id), successor in receipts.items():
        parent_id = successor.get("_supersedes_request_id")
        audit = successor.get("_derived_input")
        if (successor.get("_supersedes_incomplete") is not True
                or successor.get("_supersedes_no_final") is True
                or not isinstance(parent_id, str) or not parent_id
                or not isinstance(audit, dict)
                or audit.get("kind") != "combined_copy_compact_v1"):
            continue
        parent_key = (owner, parent_id)
        parent = receipts.get(parent_key)
        parent_hash = hashes.get(parent_key)
        successor_hash = successor.get("_supersedes_input_hash")
        original_hash = audit.get("original_input_hash")
        if (parent is None
                or parent.get("request_id") != parent_id
                or successor.get("request_id") != request_id
                or not all(isinstance(value, str) and value
                           for value in (parent_hash, successor_hash, original_hash))
                or parent_hash != successor_hash
                or successor_hash != original_hash
                or not TextTaskService._supersede_order_link(parent, successor)
                # Any owner-scoped competing child or descendant makes this
                # no longer the current compact leaf.  Do not infer which one
                # should receive an expedited original read.
                or descendants.get(parent_key) != {request_id}
                or descendants.get((owner, request_id))):
            continue
        verified.append((owner, parent_id, parent, successor))
    return tuple(verified)


def _verified_current_incomplete_leaves(text_rows):
    """Return sent compact-successor leaves safe to keep in the contributor lane."""
    from services.text_task_service import TextTaskService

    return frozenset(
        (owner, successor["request_id"])
        for owner, _parent_id, _parent, successor in _verified_incomplete_compact_links(text_rows)
        if TextTaskService._normal_sent_chat_result_recovery(successor)
    )


def _verified_saved_incomplete_successor_parents(text_rows):
    """Return old originals whose one saved compact child needs an exact final read.

    This only changes recovery scheduling priority.  A saved child is never
    treated as a terminal result for its old request or as a product close
    waiver; the old request remains subject to its normal exact-result reader.
    """
    controls = ("_executing", "_turn_reserved", "recovery_claim_id", "_recovery_claim_id",
                "_recovery_paused", "_recovery_suppressed", "_completion", "_completion_of",
                "upstream_unfinished")
    leases = ("_claim_id", "_claim_until", "recovery_claimed_at", "recovery_lease_until",
              "_recovery_claim_id", "_recovery_claim_until")
    verified = set()
    for owner, parent_id, parent, successor in _verified_incomplete_compact_links(text_rows):
        product_key = parent.get("_product_work_key")
        product_step = parent.get("_product_work")
        if (successor.get("status") != "succeeded"
                or successor.get("upstream_outcome") != "completed"
                or not isinstance(successor.get("content"), str) or not successor["content"].strip()
                or any(successor.get(key) for key in controls)
                or any(successor.get(key) is not None for key in leases)
                or not isinstance(product_key, str) or not product_key
                or successor.get("_product_work_key") != product_key
                or not isinstance(product_step, dict)
                or not all(isinstance(product_step.get(key), str) and product_step[key]
                           for key in ("product_ref", "step_ref"))
                or successor.get("_product_work") != product_step):
            continue
        verified.add((owner, parent_id))
    return frozenset(verified)


def _safe_stream_handoff_event_metadata(extra):
    """Persist only fixed continuation progress, never endpoint or payload data."""
    stage = extra.get("handoff_stage")
    if stage not in _SAFE_STREAM_HANDOFF_STAGES:
        return {}
    fields = {"handoff_stage": stage}
    for key in ("handoff_frame_count", "handoff_byte_count"):
        value = extra.get(key)
        if type(value) is int and 0 <= value <= 2_147_483_647:
            fields[key] = value
    if type(extra.get("handoff_terminal_seen")) is bool:
        fields["handoff_terminal_seen"] = extra["handoff_terminal_seen"]
    return fields


def _safe_stream_finished_event_metadata(extra):
    """Persist only fixed SSE diagnostics, never an event body or token."""
    fields = {}
    for key in ("resume_token_seen", "server_ste_metadata_seen", "stream_handoff_seen",
                "resume_sse_endpoint_seen", "subscribe_ws_topic_seen"):
        if type(extra.get(key)) is bool:
            fields[key] = extra[key]
    kind = extra.get("resume_token_kind")
    if isinstance(kind, str) and kind in _SAFE_RESUME_TOKEN_KINDS:
        fields["resume_token_kind"] = kind
    counts = extra.get("sse_event_type_counts")
    if isinstance(counts, dict):
        safe_counts = {
            name: count for name in _SAFE_SSE_EVENT_TYPES
            if type(count := counts.get(name)) is int and 0 <= count <= 2147483647
        }
        if safe_counts:
            fields["sse_event_type_counts"] = safe_counts
    handoff_counts = extra.get("stream_handoff_option_counts")
    if isinstance(handoff_counts, dict):
        safe_handoff_counts = {
            name: count for name in _SAFE_STREAM_HANDOFF_OPTION_TYPES
            if type(count := handoff_counts.get(name)) is int and 0 <= count <= 2147483647
        }
        if safe_handoff_counts:
            fields["stream_handoff_option_counts"] = safe_handoff_counts
    return fields


def _recognizes_legacy_combined_copy(body):
    """Use the derivation module's exact legacy recognizer.

    This is a non-throwing, non-transforming scope check.  The strict helper
    remains inside the controlled pre-send path below.
    """
    from services.combined_copy_derivation import is_legacy_combined_copy_input
    return is_legacy_combined_copy_input(body)


def _legacy_combined_copy_scope(context, receipt, body):
    """The one original request family eligible for retained-copy handling."""
    return bool(
        context.kind == "text"
        and _legacy_combined_copy_receipt_family(receipt)
        and _recognizes_legacy_combined_copy(body)
    )


def _legacy_combined_copy_receipt_family(receipt):
    """Receipt-only subset safe for pre-input claim fencing."""
    return bool(
        receipt.get("_source") == _COMBINED_COPY_SOURCE
        and receipt.get("_route", "chat") == "chat"
        and receipt.get("_operation", "text") == "text"
        and not receipt.get("_forward_protocol")
        and not receipt.get("_forward")
        and not receipt.get("_supersedes_request_id")
        and not receipt.get("_previous_request_id")
        and not receipt.get("_completion_of")
        and not receipt.get("_codex_fallback_of")
        and not receipt.get("_codex_continue_of")
        and not receipt.get("_codex_rejected_retry_of")
        and not receipt.get("_derived_input")
        and receipt.get("model") == _COMBINED_COPY_MODEL
    )


def _combined_copy_send_evidence(receipt):
    """Whether this original has already crossed, or may have crossed, send."""
    timeline = receipt.get("_execution_timeline")
    return bool(
        receipt.get("_submission_started") is True
        or receipt.get("_last_sent_sequence") is not None
        or receipt.get("send_count", 0) not in (None, 0)
        or receipt.get("upstream_outcome") in {"unknown", "completed", "rejected"}
        or any(bool(receipt.get(field)) for field in _COMBINED_COPY_RESULT_FIELDS)
        or isinstance(timeline, list) and any(
            isinstance(item, dict) and item.get("stage") in _COMBINED_COPY_SENT_STAGES
            for item in timeline
        )
    )


def _queued_combined_copy_candidate(context, receipt, body):
    """Whether one retained input may use the closed in-memory compactor."""
    timeline = receipt.get("_execution_timeline")
    return bool(
        _legacy_combined_copy_scope(context, receipt, body)
        and receipt.get("status") == "running"
        and receipt.get("_claim_id") == context.claim
        and receipt.get("_executing") is True
        and receipt.get("_submission_started") is False
        and receipt.get("upstream_outcome") in (None, "not_sent")
        and isinstance(timeline, list)
        and not _combined_copy_send_evidence(receipt)
    )


def firstsend_model_trial_scope(context, receipt, body):
    """The one product original family eligible for an in-memory model trial.

    This is deliberately narrower than compaction: it is an exact normal
    product Chat original, still before every durable send boundary.  The
    frozen input, receipt model and request identity remain authoritative.
    """
    return bool(
        _legacy_combined_copy_scope(context, receipt, body)
        and isinstance(body.get("product_work"), dict)
        and bool(body["product_work"].get("product_ref"))
        and bool(body["product_work"].get("step_ref"))
        and body.get("model") == _COMBINED_COPY_MODEL
        and body.get("thinking_effort") == _COMBINED_COPY_EFFORT
    )


def _firstsend_model_trial_target(trial, request_id):
    """Return this request's exact configured account binding, if any.

    Keep the original one-item control shape readable while allowing one
    bounded batch control to carry independently receipt-derived bindings.
    """
    if not isinstance(trial, dict):
        return None
    if set(trial) == {"account_identity", "request_id", "expires_at"}:
        target = {"account_identity": trial.get("account_identity"), "request_id": trial.get("request_id"),
                  "expires_at": trial.get("expires_at")}
        return target if target["request_id"] == request_id else None
    targets = trial.get("targets")
    if set(trial) != {"targets", "expires_at"} or not isinstance(targets, list):
        return None
    matches = [target for target in targets if isinstance(target, dict) and target.get("request_id") == request_id]
    if len(matches) != 1:
        return None
    target = matches[0]
    if set(target) != {"account_identity", "request_id"}:
        return None
    return {**target, "expires_at": trial.get("expires_at")}


def _firstsend_model_trial_marker(context, receipt, body, settings, now):
    marker = receipt.get("_temporary_firstsend_model_trial")
    trial = settings.get("temporary_firstsend_model_trial")
    target = _firstsend_model_trial_target(trial, context.request_id)
    if (not isinstance(marker, dict) or not isinstance(trial, dict)
            or type(marker.get("expires_at")) is not int
            or settings.get("revision", 0) != marker.get("revision")
            or not isinstance(target, dict) or any(target.get(key) != marker.get(key)
                   for key in ("account_identity", "request_id", "expires_at"))
            or receipt.get("provider_account_identity") != marker.get("account_identity")
            or context.request_id != marker.get("request_id")
            or now >= marker.get("expires_at", 0)
            or not _queued_combined_copy_candidate(context, receipt, body)
            or not firstsend_model_trial_scope(context, receipt, body)):
        return None
    return marker


def _firstsend_model_execution_body(context, receipt, body, store, settings, clock, *, scope_body=None):
    """Return an ephemeral instant/none body only for one armed original.

    Nothing durable is rewritten here.  The marker is rechecked at the
    model-send edge after its fresh bound-account catalog proof is prepared.
    """
    scope_body = body if scope_body is None else scope_body
    marker = _firstsend_model_trial_marker(context, receipt, scope_body, settings, float(clock()))
    if marker is None:
        return body
    with store.transaction() as db:
        current = store.read_receipt(db, context.kind, context.owner, context.request_id)
        if not current or current.get("_claim_id") != context.claim or current.get("status") != "running":
            raise AdmissionLost("original task claim changed before first-send model selection")
        if _firstsend_model_trial_marker(context, current, scope_body, settings, float(clock())) is None:
            return body
    context._firstsend_model_trial = {
        "original_model": _COMBINED_COPY_MODEL,
        "original_thinking_effort": _COMBINED_COPY_EFFORT,
        "effective_model": _FIRSTSEND_TRIAL_MODEL,
        "effective_thinking_effort": _FIRSTSEND_TRIAL_EFFORT,
    }
    return {**body, "model": _FIRSTSEND_TRIAL_MODEL, "thinking_effort": _FIRSTSEND_TRIAL_EFFORT}


def _compact_queued_combined_copy(context, receipt, body, store, clock):
    """Build one non-durable compact execution body for a proved-unsent input.

    The caller already verified the stored input hash.  Reloading the receipt
    under its existing transaction keeps a lost claim, cancellation, or a late
    send/result marker from being transformed after that verification.
    """
    if not _queued_combined_copy_candidate(context, receipt, body):
        return body
    from services.combined_copy_derivation import compact_retained_messages
    from services.conversation_binding_service import ConversationBindingError

    # The successor helper's 64 KiB bound is deliberately tighter than the
    # ordinary retained-input contract.  An original which it cannot compact
    # still has a valid first-send path and must retain that path unchanged.
    try:
        messages, audit = compact_retained_messages(body)
    except ConversationBindingError as exc:
        if getattr(exc, "code", None) == "CHAT_DERIVED_INPUT_TOO_LARGE":
            return body
        raise
    try:
        original_bytes = len(json.dumps({"messages": body["messages"]}, ensure_ascii=False,
                                        separators=(",", ":")).encode("utf-8"))
        compact_bytes = len(json.dumps({"messages": messages}, ensure_ascii=False,
                                       separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError, UnicodeError):
        return body
    # The execution-only variant exists solely to relieve the retained input
    # size.  Do not replace a valid original with an equal or larger body.
    if compact_bytes >= original_bytes:
        return body
    with store.transaction() as db:
        current = store.read_receipt(db, context.kind, context.owner, context.request_id)
        if not current or current.get("_claim_id") != context.claim or current.get("status") != "running":
            raise AdmissionLost("original task claim changed before retained-input compaction")
        if not _queued_combined_copy_candidate(context, current, body):
            return body
        timeline = list(current.get("_execution_timeline") or [])
        timeline.append({"stage": "retained_input_compacted", "at": float(clock()),
                         "kind": audit["kind"], "preparation_count": audit["preparation_count"],
                         "evidence_set_count": audit["evidence_set_count"],
                         "image_count": audit["image_count"]})
        current["_execution_timeline"] = timeline[-32:]
        store.write_receipt(db, context.kind, context.owner, context.request_id, current)
    return {**body, "messages": messages}


def _missing_node_retry_image_parent(store, db, owner, image_id, now):
    """Return C's only safe terminal parent for its linked A4, if any.

    This is intentionally a narrow database lookup rather than an all-receipt
    scan: routine image sends do not need to decode unrelated text receipts.
    An existing marker is authoritative.  If it no longer proves the exact
    live R -> P -> C chain, A4 must fail closed instead of falling back to its
    stale frozen parent.
    """
    from services.text_task_service import TextTaskService
    rows = db.execute(
        "SELECT receipt FROM requests WHERE owner=? AND "
        "json_extract(receipt,'$.continue_before_queued_image_id')=?",
        (owner, image_id),
    ).fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise AdmissionLost("missing-node image continuation is ambiguous")
    try:
        candidate = json.loads(rows[0][0])
    except (TypeError, ValueError):
        raise AdmissionLost("missing-node image continuation is invalid")
    parent_id = candidate.get("_supersedes_request_id")
    parent = (store.read_receipt(db, "text", owner, parent_id)
              if isinstance(parent_id, str) else None)
    parent_message_id = candidate.get("parent_message_id")
    if (candidate.get(TextTaskService.MISSING_NODE_RETRY_FIELD) != image_id
            or candidate.get("status") != "succeeded"
            or candidate.get("upstream_outcome") != "completed"
            or not isinstance(candidate.get("content"), str) or not candidate["content"].strip()
            or not isinstance(parent_message_id, str) or not parent_message_id
            or not isinstance(parent, dict)
            or not TextTaskService._missing_node_model_rejection_retry_consumed(store, db, owner, candidate)
            or not TextTaskService._rejected_missing_node_successor_source(store, db, owner, parent, now)):
        raise AdmissionLost("missing-node image continuation changed before send")
    root_id = parent.get("_supersedes_request_id")
    if not isinstance(root_id, str) or not root_id:
        raise AdmissionLost("missing-node image continuation is invalid")
    return parent_message_id, parent_id, root_id


def record_unsent_failure(context, receipt, at, error_type=None, reason=None):
    """Keep why an unsent attempt returned to its original queue, without text."""
    # progress can belong to an older attempt. Only the submission fence proves
    # the phase here; do not attribute the failure to a stale detailed step.
    entry = {"stage": "pre_submit_failure", "at": at, "phase": "before_submit",
             "request_ref": hashlib.sha256((context.owner + ":" + context.request_id).encode()).hexdigest()[:24]}
    if error_type:
        entry["error_type"] = error_type
    if reason in {"claim_expired", "image_capacity_changed", "scheduling_changed",
                  "work_not_active", "send_guard_rejected", "account_unavailable",
                  "image_capability_unavailable", "image_binding_changed"}:
        entry["reason"] = reason
    code = receipt.get("error_code")
    if code in {"CONVERSATION_OUTCOME_UNKNOWN", "CONVERSATION_BINDING_UNAVAILABLE", "IMAGE_RESOURCE_UNAVAILABLE"}:
        entry["error_code"] = code
    if code == "CONVERSATION_BINDING_UNAVAILABLE":
        from services.image_task_service import _public_failure_details
        reason = _public_failure_details(receipt.get("last_recovery_failure")).get("binding_reason")
        if reason:
            entry["binding_reason"] = reason
    receipt["_execution_timeline"] = [*(receipt.get("_execution_timeline") or []), entry][-32:]
    logger.warning({"event": "pool_execution_stage", **entry})


def account_clock_key(account):
    # Use the identity already owned by AccountRequestClock. Duplicate imports
    # of that upstream identity share both its turn and image constraints.
    identity = str(account.get("account_id") or account.get("provider_account_identity") or account.get("access_token") or "")
    return hashlib.sha256(identity.encode()).hexdigest()


def _waiting_diagnostic_key(namespace, value):
    """Return a stable receipt-local pseudonym without exposing pool keys."""
    return hashlib.sha256((namespace + "\0" + value).encode()).hexdigest()[:24]


_WAITING_RESOURCE_FAMILIES = frozenset({
    "execution_input_bytes", "chat_executor", "image_executor", "codex_server",
    "chat_turn", "image", "codex_turn", "image_thread", "public_chat_previous",
    "completion_work", "product_parent_state", "conversation_turn",
    "conversation_order", "missing_node_retry", "work_turn", "workflow_slots",
})
_WAITING_ELIGIBILITY_CODES = frozenset({
    "account_disabled", "source_not_authorized", "paid_plan_required",
    "identity_not_unique", "model_plan_not_allowed", "model_account_not_allowed",
    "model_limit_zero", "matching_offer_missing",
})
_WAITING_RESOURCE_BLOCKERS = frozenset({
    "occupancy_unknown", "resource_full", "availability_time_unknown",
    "account_or_shared_pacing",
})
WAITING_DIAGNOSTIC_MAX_ACCOUNT_ENTRIES = 32
WAITING_DIAGNOSTIC_MAX_RESOURCE_ENTRIES = 64
WAITING_DIAGNOSTIC_MAX_JSON_BYTES = 16 * 1024


def _waiting_resource_family(resource):
    # Resource names are internal and may include a hashed account or request
    # suffix.  An unknown prefix is deliberately not exposed as a new channel.
    family = resource.split(":", 1)[0]
    return family if family in _WAITING_RESOURCE_FAMILIES else "other"


def _same_waiting_projection(previous, proposed):
    """Ignore only the per-snapshot diagnostic stamps when no blocker changed."""
    if not isinstance(previous, dict) or not isinstance(proposed, dict):
        return previous == proposed

    def stable(value):
        result = dict(value)
        diagnostics = result.get("admission_diagnostics")
        if isinstance(diagnostics, dict):
            result["admission_diagnostics"] = {
                key: item for key, item in diagnostics.items()
                if key not in {"observed_at", "snapshot_revision"}
            }
        return result

    return stable(previous) == stable(proposed)


def waiting_admission_diagnostics(observed_at, snapshot_revision, offer_diagnostics):
    """Redact planner diagnostics before attaching them to a receipt.

    The planner is allowed to use stable account/resource identities so it can
    select correctly. Waiting receipts must never retain those raw identities,
    credentials, input, or another receipt's payload.
    """
    accounts = {}
    resources = {}
    for offer in offer_diagnostics:
        if not isinstance(offer, OfferDiagnostic):
            continue
        account_key = _waiting_diagnostic_key("account", offer.account)
        eligibility = set(offer.ineligible_reasons) & _WAITING_ELIGIBILITY_CODES
        if eligibility:
            accounts.setdefault(account_key, set()).update(eligibility)
        for block in offer.resource_blocks:
            blockers = set(block.blockers) & _WAITING_RESOURCE_BLOCKERS
            if not blockers:
                continue
            resource_key = _waiting_diagnostic_key("resource", block.resource)
            key = (account_key, resource_key, block.required_units)
            entry = resources.setdefault(key, {
                "account_key": account_key,
                "resource_key": resource_key,
                "family": _waiting_resource_family(block.resource),
                "capacity": block.capacity,
                "occupied_known": block.occupied is not None,
                "required_units": block.required_units,
                "next_at_known": block.next_at is not None,
                "blockers": set(),
            })
            if block.occupied is not None:
                entry["occupied"] = block.occupied
            if block.next_at is not None:
                entry["next_at"] = block.next_at
            entry["blockers"].update(blockers)
    account_entries = [
        {"account_key": account_key, "failed_checks": sorted(checks)}
        for account_key, checks in sorted(accounts.items())
    ]
    resource_entries = [
        {**entry, "blockers": sorted(entry["blockers"])}
        for _, entry in sorted(resources.items())
    ]
    if not account_entries and not resource_entries:
        return None
    # Snapshot revisions are generated UUID hex values. Do not turn an adapter
    # test value or future opaque identifier into an unbounded receipt field.
    revision = (snapshot_revision if isinstance(snapshot_revision, str)
                and len(snapshot_revision) == 32
                and all(char in "0123456789abcdef" for char in snapshot_revision)
                else "unknown")
    omitted_accounts = max(0, len(account_entries) - WAITING_DIAGNOSTIC_MAX_ACCOUNT_ENTRIES)
    omitted_resources = max(0, len(resource_entries) - WAITING_DIAGNOSTIC_MAX_RESOURCE_ENTRIES)
    account_entries = account_entries[:WAITING_DIAGNOSTIC_MAX_ACCOUNT_ENTRIES]
    resource_entries = resource_entries[:WAITING_DIAGNOSTIC_MAX_RESOURCE_ENTRIES]

    def projection():
        result = {"observed_at": observed_at, "snapshot_revision": revision}
        if account_entries:
            result["account_eligibility"] = account_entries
        if resource_entries:
            result["resource_limits"] = resource_entries
        if omitted_accounts or omitted_resources:
            result.update(truncated=True,
                          omitted_account_eligibility=omitted_accounts,
                          omitted_resource_limits=omitted_resources)
        return result

    # Keep the durable representation bounded as well as its entry counts.
    # Prefer retaining account qualification evidence; remove resource entries
    # first because they are more numerous per account.
    while resource_entries and len(json.dumps(projection(), ensure_ascii=False, separators=(",", ":")).encode()) > WAITING_DIAGNOSTIC_MAX_JSON_BYTES:
        resource_entries.pop()
        omitted_resources += 1
    while account_entries and len(json.dumps(projection(), ensure_ascii=False, separators=(",", ":")).encode()) > WAITING_DIAGNOSTIC_MAX_JSON_BYTES:
        account_entries.pop()
        omitted_accounts += 1
    return projection()


def physical_conversation_key(receipt):
    account = receipt.get("provider_account_identity") or receipt.get("provider_binding_id")
    conversation = receipt.get("conversation_id")
    if not account or not conversation:
        return None
    return "conversation_turn:" + hashlib.sha256((str(account) + "\0" + str(conversation)).encode()).hexdigest()


def recovery_suppressed(receipt):
    """Return whether an explicit operator marker stops old recovery work.

    The marker is deliberately separate from status/error fields: setting it
    does not rewrite the original outcome or any request identity, and removing
    it is the reversible way to resume the existing recovery path.
    """
    return receipt.get("_recovery_suppressed") is True


def unknown_text_result(receipt):
    return (not recovery_suppressed(receipt)
            and (receipt.get("status") == "unknown"
            or receipt.get("status") == "failed" and receipt.get("error_code") == "RESULT_UNRECOVERABLE"
            and receipt.get("upstream_outcome") == "unknown"))


def unresolved_result(kind, receipt):
    if kind == "text":
        return unknown_text_result(receipt)
    return not recovery_suppressed(receipt) and (receipt.get("status") == "unknown"
           or receipt.get("status") == "error" and receipt.get("upstream_unfinished") is True)


def image_original_recovery_pending(receipt):
    """The existing scheduler can still collect this original, including cooldown.

    Captured assets may outlive the active generation attempt. Explicit pause
    or suppression still stops their recovery; an old error alone does not.
    This predicate reads a receipt and never starts a recovery or a model send.
    """
    saved = bool(receipt.get("result_file_ids") or receipt.get("result_sediment_ids")
                 or pending_image_result_ids(receipt))
    return bool(not recovery_suppressed(receipt) and receipt.get("_recovery_paused") is not True
                and (not receipt.get("_attempt_finished_at") or saved)
                and receipt.get("status") == "error"
                and (receipt.get("error_code") == "CONVERSATION_OUTCOME_UNKNOWN"
                     or saved and receipt.get("error_code") == "RESULT_UNRECOVERABLE")
                and receipt.get("conversation_id") and receipt.get("request_message_id"))


def image_result_stage(receipt):
    """Safe progress derived from the receipt, never from a new upstream read.

    Asset discovery and private download are not a publishable result. Keep
    explicit stops and terminal failures visible even when cached assets exist.
    """
    status = receipt.get("status")
    if status == "success":
        return "result_ready" if isinstance(receipt.get("data"), list) and receipt["data"] else "needs_attention"
    if (receipt.get("_recovery_paused") is True or recovery_suppressed(receipt)
            or status not in {"queued", "running"} and not image_original_recovery_pending(receipt)):
        return "needs_attention"
    cached = receipt.get("_pending_image_output")
    if isinstance(cached, dict) and cached.get("output_ref") and isinstance(cached.get("coverage"), dict):
        return "downloaded_waiting_original_confirmation"
    if (int(receipt.get("_expected_sends") or 1) == 1
            and int(receipt.get("_send_sequence") or 1) == 1
            and any(isinstance(e, dict) and e.get("stage") == "upstream_terminal" and e.get("known") is True
                    for e in receipt.get("_execution_timeline", []))):
        # An observed terminal turn is progress, not a saved/downloadable result
        # and not permission to release the work slot or submit another turn.
        return "upstream_finished"
    if (receipt.get("result_file_ids") or receipt.get("result_sediment_ids")
            or pending_image_result_ids(receipt)):
        return "assets_discovered"
    if receipt.get("upstream_submission_started") is True:
        return "submitted"
    if status == "queued":
        return "queued"
    if receipt.get("upstream_submission_started") is False:
        return "preparing"
    # Older receipts may lack a submit marker even after an upstream write.
    return "submission_unconfirmed"


def unfinished(kind, receipt):
    if recovery_suppressed(receipt):
        return False
    if kind == "image":
        return receipt.get("upstream_unfinished") is True or receipt.get("status") in {"queued", "running"}
    return receipt.get("status") in {"queued", "running", "not_started"} or unknown_text_result(receipt)


def original_turn_ended(kind, receipt):
    if receipt.get("_upstream_terminal") is True:
        return True
    if kind == "text":
        from services.conversation_binding_service import is_model_not_found_rejection
        if receipt.get("status") == "failed" and receipt.get("error_code") == "MODEL_NOT_FOUND":
            return is_model_not_found_rejection(receipt)
    # Older public Chat code recorded a definitive validation rejection as
    # UNKNOWN. Only its exact model endpoint / pre-SSE HTTP evidence proves
    # that turn ended. Missing stages, timeouts and result age prove nothing.
    return (kind == "text" and unknown_text_result(receipt)
            and receipt.get("_route", "chat") == "chat" and not receipt.get("_forward_protocol")
            and receipt.get("original_failure_phase") == "stream_open"
            and receipt.get("original_exception_category") == "http"
            and receipt.get("original_upstream_request_stage") == "conversation"
            and type(receipt.get("original_http_status")) is int and receipt["original_http_status"] == 422)


def image_capacity(account, settings, model="gpt-image-2"):
    from services.owned_accounts import image_dispatch_capacity
    return min(max(0, int(settings["image_account_concurrency"])), image_dispatch_capacity(account, model))


def image_generation_active(kind, receipt, active):
    """Count generation only; confirmed output download/save has its own work.

    An EOF, old request or suppressed recovery is not proof of generated output.
    Native image receipts persist original result IDs before downloading; wire
    image receipts persist the original turn's explicit terminal evidence.
    """
    if receipt.get("_attempt_finished_at") and not receipt.get("_executing"):
        return False
    if kind != "image" and receipt.get("_operation") != "image":
        return False
    single = int(receipt.get("_expected_sends") or 1) <= 1
    generated = (single and kind == "image" and receipt.get("upstream_unfinished") is False
                 and bool(receipt.get("result_file_ids") or receipt.get("result_sediment_ids"))
                 and bool(receipt.get("request_message_id")))
    ended = original_turn_ended(kind, receipt) and single
    return not (generated or ended) and (receipt.get("upstream_unfinished") is True or active)


class ExecutionContext:
    def __init__(self, admission, kind, owner, request_id, claim):
        self.admission, self.kind, self.owner = admission, kind, owner
        self.request_id, self.claim = request_id, claim

    def before_send(self):
        self.admission.before_send(self)

    def image_capability_refresh(self):
        return self.admission.image_capability_refresh(self)

    def release_turn(self):
        if int(self.receipt().get("_expected_sends") or 1) > 1:
            return  # Bounded legacy multi-image request runs its slots serially.
        self.admission.update_claim(self, _turn_reserved=False)

    def image_slot(self, index):
        r = self.receipt()
        if index >= int(r.get("_expected_sends") or 1) or (index and r.get("_completed_slot") != index - 1):
            raise AdmissionLost("original image sequence is not complete")
        self.admission.update_claim(self, _send_sequence=index)
        # A legacy n>1 request already owns an executing original. Delay its
        # next slot in the existing worker; never release/replay a partial send.
        if index:
            while True:
                receipt = self.receipt()
                interval = (receipt.get("_scheduling") or {}).get("min_send_interval_seconds", 0)
                with self.admission.store.connect() as db:
                    last = self.admission.store.runtime(db, "workflow_send:" + workflow_scheduling.group(receipt), 0)
                delay = float(last) + interval - float(self.admission.clock())
                if delay <= 0 or not interval:
                    break
                time.sleep(min(1.0, delay))

    def image_slot_complete(self, index):
        self.admission.update_claim(self, _completed_slot=index)

    def record_limit(self, evidence):
        self.admission.update_claim(self, rate_limit=evidence)

    def record_stage(self, stage, **extra):
        """Persist and log a safe stage in the original request timeline."""
        from utils.upstream_error import safe_upstream_error
        receipt = self.receipt()
        settings = self.admission._settings()
        accepted = next((item.get("at") for item in receipt.get("_execution_timeline", [])
                         if item.get("stage") == "accepted"), receipt.get("created_ts") or receipt.get("created_at"))
        claimed = next((item.get("at") for item in receipt.get("_execution_timeline", [])
                        if item.get("stage") == "execution_claimed"), None)
        entry = {
            "stage": str(stage),
            "at": time.time(),
            "request_ref": hashlib.sha256((self.owner + ":" + self.request_id).encode()).hexdigest()[:24],
            "account_ref": safe_account_ref(receipt.get("provider_account_identity")),
            "model": receipt.get("model"),
            "operation": receipt.get("_operation", "image" if self.kind == "image" else "text"),
            "source": receipt.get("_source"),
            "input_bytes": receipt.get("_input_bytes"),
            "queue_wait_seconds": max(0, claimed - accepted) if type(accepted) in (int, float) and type(claimed) in (int, float) else None,
            "config_revision": settings.get("revision") if isinstance(settings, dict) else None,
            **{key: value for key, value in extra.items()
               if key in {"status_code", "upstream_request_id", "output_ref", "known", "task_status", "image_count",
                          "stream_end", "sse_data_count", "sse_parse_errors", "sse_error_event", "sse_error_category",
                          "sse_message_snapshot_events", "sse_terminal_assistant_events", "sse_terminal_id_events",
                          "sse_terminal_final_channel_events", "sse_terminal_parent_events",
                          "sse_terminal_direct_parent_match_events"}},
        }
        if stage == "stream_finished":
            entry.update(_safe_stream_finished_event_metadata(extra))
            detail = safe_upstream_error(extra.get("upstream_error"))
            if detail:
                entry["upstream_error"] = detail
        elif stage == "stream_handoff":
            entry.update(_safe_stream_handoff_event_metadata(extra))
        timeline = list(receipt.get("_execution_timeline") or [])
        timeline.append(entry)
        try:
            self.admission.update_claim(self, _execution_timeline=timeline[-32:])
        except AdmissionLost:
            # Telemetry must never turn an expired original into a retry.
            pass
        # The owner-scoped receipt holds the redacted reason. Global operational
        # logs keep structural evidence only, not even sanitized message text.
        logger.info({"event": "pool_execution_stage", **{k: v for k, v in entry.items() if k != "upstream_error"}})
        return entry

    def record_outcome(self):
        receipt = self.receipt()
        status = receipt.get("status")
        if status not in {"success", "succeeded", "failed", "error", "unknown", "cancelled"}:
            return
        if (receipt.get("upstream_outcome") == "generated" and status == "error"
                and (receipt.get("result_file_ids") or receipt.get("result_sediment_ids"))):
            return  # Output download is unfinished, not a failed generation.
        outcome = "succeeded" if status in {"success", "succeeded"} else "unknown" if status == "unknown" or receipt.get("upstream_unfinished") is True or receipt.get("upstream_outcome") == "unknown" else "failed"
        if any(item.get("stage") == "task_finished" and item.get("task_status") == outcome
               for item in receipt.get("_execution_timeline", [])):
            return
        images = len(receipt["data"]) if outcome == "succeeded" and self.kind == "image" and isinstance(receipt.get("data"), list) else None
        if outcome == "succeeded" and receipt.get("_operation") == "image" and type(receipt.get("_completed_slot")) is int:
            images = receipt["_completed_slot"] + 1
        self.record_stage("task_finished", task_status=outcome, image_count=images)

    def log_fields(self):
        r = self.receipt()
        settings = self.admission._settings()
        return {"model": r.get("model"), "operation": r.get("_operation", "image" if self.kind == "image" else "text"),
                "retained_input_bytes": r.get("_input_bytes"), "send_sequence": r.get("_send_sequence", 0),
                "route": r.get("_route", "chat"), "source": r.get("_source"),
                "config_revision": settings.get("revision") if isinstance(settings, dict) else None,
                "account_ref": safe_account_ref(r.get("provider_account_identity"))}

    def receipt(self):
        with self.admission.store.connect() as db:
            return self.admission.store.read_receipt(db, self.kind, self.owner, self.request_id)

    def selected_account(self):
        identity = self.receipt().get("provider_account_identity")
        return next((a for a in self.admission._rows() if a.get("provider_account_identity") == identity), None)

    def terminal(self, known):
        self.admission.update_claim(self, _upstream_terminal=bool(known), _turn_reserved=not known)
        self.record_stage("upstream_terminal", known=bool(known))


class PoolAdmission:
    CLAIM_SECONDS = 30.0
    # Heartbeat writes must never consume the store's normal 30-second writer
    # wait: a queued admission transaction would then outlive this claim.
    CLAIM_HEARTBEAT_WRITER_TIMEOUT_SECS = 0.25
    CLAIM_HEARTBEAT_RETRY_SECS = 0.25
    # Existing text executor's retained-input protection; waiting inputs live
    # on disk. This is an execution memory ceiling, not an account/user budget.
    MAX_ACTIVE_INPUT_BYTES = 256 * 1024 * 1024

    def __init__(self, store: TaskStore, accounts, *, clock=time.time,
                 settings=None, model_types=None, pacing=None, codex=None, text_workers=None):
        self.store, self.accounts, self.clock = store, accounts, clock
        self.settings = settings
        self.model_types = model_types
        self.pacing = pacing
        self.codex = codex
        if codex is not None:
            codex.admission = self
        self.text_workers = text_workers
        self.handlers = {}
        self.recoveries = {}
        self._recovery_thread = None
        # A local I/O bound, not an upstream concurrency or rate limit. Waiting
        # originals stay in the existing store, never in an executor queue.
        self._original_read_slots = threading.BoundedSemaphore(4)
        self._original_read_lock = threading.Lock()
        self._original_reads = set()
        # A durable FIFO ticket belongs to one dispatcher instance.  The
        # original request identity remains part of its hash, while an expired
        # worker/restart falls back to the queue's existing short expiry.
        self._original_read_ticket_nonce = uuid.uuid4().hex
        # The reader slots are shared across accounts.  Remember the last
        # account selected so a large, overdue historical backlog on one
        # account cannot keep every other due original behind its first rows.
        # This is only an in-process scheduling cursor; receipt retry clocks
        # remain the durable recovery ordering.
        self._recovery_account_cursor = None
        # Prefer reads that can free capacity, but give released originals a
        # turn after three successful admissions on their own account. Scans
        # and failed reservations do not consume or reset this volatile budget.
        self._recovery_contributor_streak = {}
        # An initial capacity-recovery head start must not permanently exclude
        # an account that has only released originals and cannot earn a streak.
        self._recovery_contributor_warmup = 0
        # Scoped to one recovery scan. A shared FIFO block must not let a
        # later lane create a new future ticket before the next scan applies
        # the current-vs-history priority again.
        self._recovery_fifo_scan = threading.local()
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._next_codex_probe = 0.0
        self._next_image_probe = 0.0
        self._image_probe_index = 0
        self._image_refresh_lock = threading.Lock()
        self._image_refresh_thread = None
        self._codex_probe_index = 0

    def register(self, kind, handler):
        self.handlers[kind] = handler

    def wake(self):
        self._event.set()

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="original-task-admission", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        self.wake()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _loop(self):
        while not self._stop.is_set():
            self._event.clear()
            try:
                self._start_image_capability_refresh()
                # Model catalog discovery for queued first sends is bounded,
                # but it can still take a synchronous wait in ``claim_next``.
                # Unknown originals never need that discovery, so give their
                # independent read dispatcher its turn before entering the
                # allocation loop.
                if self._recovery_thread is None or not self._recovery_thread.is_alive():
                    self._recovery_thread = threading.Thread(target=self._recovery_loop,
                                                             name="original-result-read", daemon=True)
                    self._recovery_thread.start()
                while not self._stop.is_set():
                    claim = self.claim_next()
                    if claim is None:
                        break
                    threading.Thread(target=self.execute, args=(claim,), name="original-task-execute", daemon=True).start()
            except Exception:
                # Invalid snapshots/storage are not permission to bypass
                # admission. The existing receipt remains queryable.
                from utils.log import logger
                logger.warning({"event": "task_admission_unavailable", "layer": "provider", "phase": "claim"})
            # Account/config changes in another worker are seen without a
            # client retry. Timed waits never occupy execution workers.
            self._event.wait(1.0)

    def _recovery_loop(self):
        """Scan due original results independently of first-send admission."""
        while not self._stop.is_set():
            try:
                self.recover_one(background=True)
            except Exception:
                # A malformed or transiently unreadable recovery candidate
                # must not stop later due originals while claim admission is
                # busy.  Preserve its receipt; the next bounded scan retries.
                from utils.log import logger
                logger.warning({"event": "task_admission_unavailable", "layer": "provider", "phase": "recovery"})
            # Keep the existing one-second recovery cadence even when wakeups
            # or a continuously productive claim loop are busy.
            if self._stop.wait(1.0):
                return

    def _acquire_original_read(self, kind, owner, request_id, receipt, *, require_text_reservation=False):
        if self._stop.is_set():
            return False
        # Do not fill the local workers with known pacing waits. The transport
        # still checks the same persisted clock immediately before its GET.
        with self._account_guard():
            account = next((a for a in self._rows()
                            if a.get("provider_account_identity") == receipt.get("provider_account_identity")), None)
        if require_text_reservation and (
                self.pacing is not None or account is None or account.get("managed_disabled")
                or account.get("status") in {"禁用", "异常"}):
            return NotImplemented
        read_owner = "original-result:" + hashlib.sha256(
            (self._original_read_ticket_nonce + "\0" + kind + "\0" + owner + "\0" + request_id).encode()
        ).hexdigest()[:24]
        reserve_text_read = kind == "text" and account is not None and self.pacing is None

        def release_read_ticket():
            if reserve_text_read:
                from services.account_request_pacing import release_account_original_read
                release_account_original_read(account, read_owner)

        key = physical_conversation_key(receipt)
        if key is None:
            account_key = receipt.get("provider_account_identity") or receipt.get("provider_binding_id")
            client_conversation = receipt.get("client_conversation_id")
            key = ((account_key, client_conversation) if account_key and client_conversation
                   else (kind, owner, request_id))
        with self._original_read_lock:
            # Check before booking: another receipt for this physical read
            # must not leave an unused FIFO head after the active GET starts.
            if key in self._original_reads:
                return False
            if account is not None:
                if account.get("managed_disabled") or account.get("status") in {"禁用", "异常"}:
                    release_read_ticket()
                    return False
                if reserve_text_read:
                    from services.account_request_pacing import reserve_account_original_read
                    # Nonblocking account lock; no reader slot is held until
                    # this original is at the FIFO head and its clock is due.
                    scan_blocked = getattr(self._recovery_fifo_scan, "blocked_accounts", None)
                    account_key = (account.get("provider_account_identity") or account.get("provider_binding_id")
                                   or f"{kind}:{owner}:{request_id}")
                    if not reserve_account_original_read(
                            account, read_owner,
                            allow_new=scan_blocked is None or account_key not in scan_blocked):
                        if scan_blocked is not None:
                            scan_blocked.add(account_key)
                        return False
                else:
                    now = float(self.clock())
                    if self.pacing:
                        pacing = self.pacing(account, now)
                    else:
                        # Image recovery owns a separate asynchronous worker;
                        # retain its existing clock gate and owner lifetime.
                        from services.account_request_pacing import account_pacing_snapshot
                        pacing = account_pacing_snapshot(account, now, include_turn=False,
                                                         include_conversation_read=True)
                    if pacing.get("next_at") is None or pacing["next_at"] > now:
                        return False
            if not self._original_read_slots.acquire(blocking=False):
                release_read_ticket()
                return False
            self._original_reads.add(key)

        def release():
            release_read_ticket()
            with self._original_read_lock:
                self._original_reads.discard(key)
                self._original_read_slots.release()
            self.wake()
        return read_owner if reserve_text_read else None, release

    @staticmethod
    def _run_original_read(permit, function, owner, request_id):
        read_owner, release = permit
        try:
            with reserving_original_read(read_owner):
                return function(owner, request_id)
        finally:
            release()

    def read_original_if_ready(self, owner, request_id, function, receipt):
        """Explicit reads share background FIFO admission before claiming work.

        False means deferred without touching the receipt. NotImplemented
        retains the legacy recovery path for accounts/adapters this scheduler
        cannot pre-admit. A dormant FIFO ticket belongs to the original, so a
        deferred caller must leave it for the background dispatcher to adopt.
        """
        permit = self._acquire_original_read("text", owner, request_id, receipt, require_text_reservation=True)
        if permit is NotImplemented:
            return NotImplemented
        if not permit:
            return False
        return self._run_original_read(permit, function, owner, request_id)

    def _dispatch_original_read(self, kind, owner, request_id, function, receipt):
        permit = self._acquire_original_read(kind, owner, request_id, receipt)
        if not permit:
            return False

        def run():
            try:
                self._run_original_read(permit, function, owner, request_id)
            except Exception:
                pass  # The original handler persists the bounded failure.
        try:
            threading.Thread(target=run, name="original-result-io", daemon=True).start()
        except Exception:
            permit[1]()
            raise
        return True

    def recover_one(self, *, background=False):
        now = float(self.clock())
        candidates = {}
        def collect_completion(kind, owner, request_id, function, receipt):
            # Completion retains its own eligibility and callback, including
            # permitted rechecks of ended attempts. It must share scheduling
            # with ordinary reads rather than always taking the next FIFO turn.
            candidates[(kind, owner, request_id)] = (
                kind, owner, request_id, function, receipt,
                float(receipt.get("_completion", {}).get("next_at") or 0))
            return False
        lifecycle = getattr(self, "work_lifecycle", None)
        if background and lifecycle is not None:
            # Its existing bounded, per-account workers must get a chance even
            # while there is a continuous backlog of original result reads.
            lifecycle.process_one(background=True)
        completion = getattr(self, "generation_completion", None)
        if completion is not None:
            if background:
                completion.process_one(dispatch=collect_completion)
            else:
                completion.process_one()
        with self.store.connect() as db:
            # Recovery needs all text lineage to recognize a transferred empty
            # root, but must keep the existing status prefilter for potentially
            # large, completed image payloads.
            text_rows = [(owner, request_id, request_hash, json.loads(raw))
                         for owner, request_id, request_hash, raw in db.execute(
                             "SELECT owner,id,request_hash,receipt FROM requests")]
            verified_current_incomplete_leaves = _verified_current_incomplete_leaves(text_rows)
            verified_saved_incomplete_successor_parents = _verified_saved_incomplete_successor_parents(text_rows)
            text_receipts = [("text", owner, request_id, receipt)
                             for owner, request_id, _request_hash, receipt in text_rows]
            image_receipts = [("image", receipt["owner_id"], receipt["id"], receipt)
                              for _, raw in db.execute(
                                  "SELECT task_key,receipt FROM image_requests WHERE "
                                  "json_extract(receipt,'$.status') IN (?,?,?)",
                                  ("unknown", "failed", "error"))
                              for receipt in (json.loads(raw),)]
        receipts = text_receipts + image_receipts
        from services.text_task_service import TextTaskService
        rows = [row for row in receipts if (
            row[3].get("status") in {"unknown", "failed", "error"}
            or row[0] == "text" and TextTaskService._completed_reserved_original_recheck(row[3], now)
        )]
        _, transferred_empty_turns = self._empty_turn_transfers(receipts)
        # Use the existing persisted retry timestamps. A short-backoff legacy
        # scan must not monopolize recovery ahead of other overdue originals.
        # Missing/zero retry times are immediately due. Image created_at is a
        # formatted string, so it must not enter this numeric retry ordering.
        rows.sort(key=lambda row: float((TextTaskService._recovery_due_at(row[3])
                                        if row[0] == "text" else row[3].get("next_poll_at")) or 0))
        for kind, owner, request_id, r in rows:
            saved_image = kind == "image" and bool(r.get("result_file_ids") or r.get("result_sediment_ids") or pending_image_result_ids(r))
            compatibility_recheck = (kind == "text"
                                     and TextTaskService._completed_reserved_original_recheck(r, now))
            if (recovery_suppressed(r) or r.get("_recovery_paused") is True
                    or r.get("_attempt_finished_at") and not saved_image and not compatibility_recheck):
                continue
            if kind not in self.recoveries:
                continue
            if r.get("_forward_protocol"):
                from services.durable_forward import chat_recovery_supported
                if not chat_recovery_supported(r):
                    continue
            if kind == "text":
                from services.text_task_service import TextTaskService
                due = ((unknown_text_result(r) or compatibility_recheck) and r.get("provider_binding_id")
                       and TextTaskService._recovery_due(r, now))
            else:
                due = image_original_recovery_pending(r) and float(r.get("next_poll_at") or 0) <= now
            if due:
                if background:
                    # Keep completion's callback if both paths select this
                    # exact request; one scan must not advance it twice.
                    candidates.setdefault((kind, owner, request_id), (
                        kind, owner, request_id, self.recoveries[kind], r,
                        float((TextTaskService._recovery_due_at(r) if kind == "text"
                               else r.get("next_poll_at")) or 0)))
                    continue
                try:
                    self.recoveries[kind](owner, request_id)
                except Exception:
                    pass  # Original read handlers retain bounded retry evidence.
                return
        if background:
            self._dispatch_recovery_candidates(
                candidates.values(), transferred_empty_turns, now,
                verified_current_incomplete_leaves=verified_current_incomplete_leaves,
                verified_saved_incomplete_successor_parents=verified_saved_incomplete_successor_parents)
        if not background and lifecycle is not None:
            lifecycle.process_one(background=True)

    def _dispatch_recovery_candidates(self, candidates, transferred_empty_turns, now,
                                      *, verified_current_incomplete_leaves=frozenset(),
                                      verified_saved_incomplete_successor_parents=frozenset()):
        from services.text_task_service import TextTaskService

        def recovery_priority(candidate):
            # Retain the existing capacity-release proofs. Priority never
            # changes occupancy, original ownership, or recovery eligibility.
            kind, owner, request_id, _, receipt, _ = candidate
            if kind != "text" or not unresolved_result(kind, receipt):
                return 1
            status = receipt.get("status")
            active = status == "running" or not receipt.get("_attempt_finished_at")
            wait_ended = (receipt.get("_route", "chat") == "chat"
                          and receipt.get("_operation", "text") == "text"
                          and not receipt.get("_forward_protocol")
                          and status == "failed"
                          and receipt.get("error_code") == "RESULT_UNRECOVERABLE"
                          and receipt.get("upstream_outcome") == "unknown"
                          and type(receipt.get("_execution_wait_ended_at")) in (int, float)
                          and math.isfinite(receipt["_execution_wait_ended_at"]))
            released = (receipt.get("_turn_reserved") is False
                        and TextTaskService._verified_retryable_empty(receipt))
            # A bounded exact-result GET failure releases local Chat capacity,
            # but establishes nothing about whether the upstream turn remains
            # active.  Keep that original in the contributor lane; only a
            # qualified stalled observation proves it belongs with released
            # history for scheduling.
            stalled = TextTaskService._qualified_stalled_incomplete_capacity_boundary(receipt, now)
            return 0 if (active and (owner, request_id) not in transferred_empty_turns
                         and not original_turn_ended(kind, receipt)
                         and not wait_ended and not released
                         and (not stalled
                              or (owner, request_id) in verified_current_incomplete_leaves
                              or (owner, request_id) in verified_saved_incomplete_successor_parents)) else 1

        by_account = {}
        for candidate in sorted(candidates, key=lambda candidate: candidate[-1]):
            kind, owner, request_id, _, receipt, _ = candidate
            # A receipt-owning reader already renews this durable lease while
            # its GET is in flight. Completion shares the same exclusion.
            lease = receipt.get("recovery_lease_until")
            if kind == "text" and type(lease) in (int, float) and math.isfinite(lease) and lease > now:
                continue
            account = (receipt.get("provider_account_identity") or receipt.get("provider_binding_id")
                       or f"{kind}:{owner}:{request_id}")
            by_account.setdefault(account, (deque(), deque()))[recovery_priority(candidate)].append(candidate)
        accounts = list(by_account)
        if self._recovery_account_cursor in accounts:
            start = (accounts.index(self._recovery_account_cursor) + 1) % len(accounts)
            accounts = accounts[start:] + accounts[:start]

        missing = object()
        previous_blocked = getattr(self._recovery_fifo_scan, "blocked_accounts", missing)
        self._recovery_fifo_scan.blocked_accounts = set()
        try:
            image_dispatched = False
            # Snapshot the one-time head start at scan entry. Thereafter all due
            # accounts share rotation; mixed accounts retain their own 3:1 bias.
            include_released_accounts = self._recovery_contributor_warmup >= _ORIGINAL_READ_CONTRIBUTOR_BURST
            while True:
                preferred = []
                for account in accounts:
                    contributing, released = by_account[account]
                    if released and self._recovery_contributor_streak.get(account, 0) >= _ORIGINAL_READ_CONTRIBUTOR_BURST:
                        preferred.append((account, 1))
                    elif contributing:
                        preferred.append((account, 0))
                    elif released and include_released_accounts:
                        preferred.append((account, 1))
                if not preferred:
                    preferred = [(account, 1) for account in accounts if by_account[account][1]]
                if not preferred:
                    return
                # One candidate per account per round preserves account rotation.
                # Reevaluate after each round so even a single burst-capable scan
                # cannot admit a fourth contributor ahead of its released original.
                for account, priority in preferred:
                    kind, owner, request_id, function, receipt, _ = by_account[account][priority].popleft()
                    # Image recovery owns a separate async worker; retain its
                    # one-dispatch-per-scan cadence.
                    if kind == "image" and image_dispatched:
                        continue
                    if self._dispatch_original_read(kind, owner, request_id, function, receipt):
                        if kind == "image":
                            image_dispatched = True
                        self._recovery_account_cursor = account
                        self._recovery_contributor_streak[account] = (
                            min(_ORIGINAL_READ_CONTRIBUTOR_BURST,
                                self._recovery_contributor_streak.get(account, 0) + 1)
                            if priority == 0 else 0)
                        if priority == 0:
                            self._recovery_contributor_warmup = min(
                                _ORIGINAL_READ_CONTRIBUTOR_BURST, self._recovery_contributor_warmup + 1)
        finally:
            if previous_blocked is missing:
                del self._recovery_fifo_scan.blocked_accounts
            else:
                self._recovery_fifo_scan.blocked_accounts = previous_blocked

    def run_recovery(self, context, function, args):
        done = threading.Event()
        self._start_claim_heartbeat(context, done, name="original-read-heartbeat")
        try:
            with executing(context):
                function(*args)
        finally:
            done.set()
            try:
                context.record_outcome()
            except Exception:
                logger.warning({"event": "pool_outcome_unavailable", "layer": "provider"})
            try:
                self.update_claim(context, _executing=False)
            except AdmissionLost:
                pass
            self.wake()

    def _settings(self):
        if self.settings:
            return self.settings()
        from services.config import config
        return config.resource_settings()

    def _settings_guard(self):
        if self.settings:
            provider = getattr(self.settings, "__self__", None)
        else:
            from services.config import config
            provider = config
        guard = getattr(provider, "resource_settings_guard", None)
        return guard() if guard else nullcontext()

    def _types(self, model):
        if self.model_types:
            return self.model_types(model)
        if model == "auto":
            return {"Plus", "Pro", "ProLite", "Team", "Enterprise"}
        from services.model_service import model_catalog_service
        return model_catalog_service.route_for_model(model)

    def _fresh_bound_firstsend_model_capability(self, receipt, trial):
        """Require a just-read directory row for this bound physical account.

        The ordinary account refresh and model-directory transport remain in
        use.  This does not persist a token, catalog payload or a new routing
        state; it only proves the one effective model immediately before send.
        """
        binding_id = receipt.get("provider_binding_id")
        identity = receipt.get("provider_account_identity")
        if (not isinstance(binding_id, str) or not binding_id
                or not isinstance(identity, str) or not identity
                or trial.get("effective_model") != _FIRSTSEND_TRIAL_MODEL):
            raise AdmissionLost("first-send model trial binding is invalid")
        try:
            access_token = self.accounts.get_bound_text_access_token(
                binding_id, model="auto", for_message=True,
                requested_account_identity=identity,
            )
            from services.model_service import model_catalog_service
            observed = model_catalog_service.observe_bound_model_at_send(
                access_token, identity, _FIRSTSEND_TRIAL_MODEL,
            )
        except Exception:
            raise AdmissionLost("first-send model capability discovery unavailable before send") from None
        if not observed:
            raise AdmissionLost("first-send model is unavailable before send")

    def _prepare_firstsend_model_capability(self, context):
        """Fetch the trial directory before the account send clock is held.

        The resulting proof stays only on this execution context.  The final
        send guard validates its cache freshness, physical binding and current
        control revision; it never performs a catalog network request.
        """
        trial = getattr(context, "_firstsend_model_trial", None)
        if trial is None:
            return
        if (not isinstance(trial, dict)
                or trial.get("original_model") != _COMBINED_COPY_MODEL
                or trial.get("original_thinking_effort") != _COMBINED_COPY_EFFORT
                or trial.get("effective_model") != _FIRSTSEND_TRIAL_MODEL
                or trial.get("effective_thinking_effort") != _FIRSTSEND_TRIAL_EFFORT):
            raise AdmissionLost("first-send model trial is invalid")
        with self.store.connect() as db:
            receipt = self.store.read_receipt(db, context.kind, context.owner, context.request_id) or {}
        if (context.kind != "text" or receipt.get("_claim_id") != context.claim
                or receipt.get("status") != "running" or receipt.get("_submission_started") is not False
                or float(receipt.get("_claim_until") or 0) <= float(self.clock())):
            raise AdmissionLost("original task changed before first-send model discovery")
        self._fresh_bound_firstsend_model_capability(receipt, trial)
        from services.model_service import model_catalog_service
        route = model_catalog_service.prepared_bound_model_route_at_send(
            receipt.get("provider_account_identity"), _FIRSTSEND_TRIAL_MODEL)
        if route is None:
            raise AdmissionLost("first-send model capability discovery unavailable before send")
        with self.store.connect() as db:
            current = self.store.read_receipt(db, context.kind, context.owner, context.request_id) or {}
        if (current.get("_claim_id") != context.claim or current.get("status") != "running"
                or current.get("_submission_started") is not False
                or current.get("provider_binding_id") != receipt.get("provider_binding_id")
                or current.get("provider_account_identity") != receipt.get("provider_account_identity")
                or float(current.get("_claim_until") or 0) <= float(self.clock())):
            raise AdmissionLost("original task changed during first-send model discovery")
        context._prepared_firstsend_model_route = route

    @staticmethod
    def _model_account_matches(account, route) -> bool:
        """Keep legacy injected type sets while catalog routes stay account-exact."""
        if isinstance(route, set):
            return route and account is not None
        account_types = getattr(route, "account_types", route)
        if account_types is None:
            return False
        identities = getattr(route, "account_identities", None)
        if identities is None:
            return True
        return str(account.get("provider_account_identity") or "") in identities

    def _pacing(self, account, now):
        if self.pacing:
            return self.pacing(account, now)
        from services.account_request_pacing import account_pacing_snapshot
        return account_pacing_snapshot(account, now)

    def _account_guard(self):
        # AccountService owns this lock, including the persisted-account
        # refresh. Synthetic adapters can provide an equivalent transaction.
        return getattr(self.accounts, "admission_transaction", nullcontext)()

    def _rows(self):
        read = getattr(self.accounts, "admission_accounts", None)
        return read() if read else self.accounts.list_accounts()

    def _product_capacity_limit(self, rows, settings):
        """Capacity of existing unique physical Chat turns, excluding trials."""
        identities = {}
        for account in rows:
            identity = str(account.get("provider_account_identity") or "")
            if identity:
                identities[identity] = identities.get(identity, 0) + 1
        keys = set()
        for account in rows:
            identity = str(account.get("provider_account_identity") or "")
            if (not identity or identities.get(identity) != 1 or account.get("managed_disabled")
                    or account.get("status") in {"禁用", "异常", "限流"}
                    or str(account.get("source_type") or "web") not in {"web", "oauth_login", "password"}
                    or not account.get("access_token")
                    or self.accounts._normalize_account_type(account.get("type"))
                    not in {"Plus", "Pro", "ProLite", "Team", "Enterprise"}):
                continue
            keys.add(account_clock_key(account))
        # Match ordinary chat_turn.capacity, while deliberately excluding the
        # one-request temporary second-slot trial from general product slots.
        return len(keys) * max(1, int(settings.get("chat_account_concurrency", 1)))

    def product_capacity_limit(self):
        with self._settings_guard(), self._account_guard():
            return self._product_capacity_limit(self._rows(), self._settings())

    @staticmethod
    def _empty_turn_transfers(receipts):
        """Return proven empty roots whose local turn moved to one child."""
        from services.generation_completion import same_session_retry
        from services.text_task_service import TextTaskService
        text_receipts = {(owner, request_id): receipt for kind, owner, request_id, receipt in receipts
                         if kind == "text"}
        released, transferred = set(), set()
        all_receipts = {(kind, owner, request_id): receipt
                        for kind, owner, request_id, receipt in receipts}
        for kind, owner, request_id, receipt in receipts:
            original_id = receipt.get("_same_session_retry_of")
            original = all_receipts.get((kind, owner, original_id))
            if original and same_session_retry(original, receipt):
                released.add((owner, original_id))
                transferred.add((owner, original_id))
        corrections = {}
        for kind, owner, request_id, receipt in receipts:
            previous_id = receipt.get("_terminal_empty_correction_of")
            if kind == "text" and isinstance(previous_id, str) and previous_id:
                corrections.setdefault((owner, previous_id), []).append((request_id, receipt))
        for key, candidates in corrections.items():
            if len(candidates) != 1:
                continue
            previous = text_receipts.get(key)
            evidence = TextTaskService._verified_retryable_empty(previous or {})
            request_id, correction = candidates[0]
            correction_parent = (correction.get("_submission_parent_message_id")
                                 if correction.get("status") == "succeeded"
                                 else correction.get("parent_message_id"))
            if (not evidence or not previous or request_id == key[1]
                    or correction.get("status") not in {"queued", "running", "unknown", "succeeded"}
                    or not correction.get("_input_ref")
                    or correction.get("route") != "chat"
                    or correction.get("_route", "chat") != "chat"
                    or correction.get("_previous_request_id") != key[1]
                    or correction.get("_public_session_ref") != previous.get("_public_session_ref")
                    or correction.get("client_conversation_id") != previous.get("client_conversation_id")
                    or correction.get("model") != previous.get("model")
                    or correction_parent != evidence["retry_parent_message_id"]
                    or correction.get("request_message_id") == previous.get("request_message_id")
                    or type(correction.get("_sequence")) is not int
                    or type(previous.get("_sequence")) is not int
                    or correction["_sequence"] <= previous["_sequence"]
                    or any(correction.get(field) != previous.get(field) for field in (
                        "provider_binding_id", "provider_account_identity", "conversation_id"))):
                continue
            released.add(key)
            if previous.get("_empty_reply_evidence") and correction.get("_completion_of") == key[1]:
                transferred.add(key)
        return released, transferred

    @staticmethod
    def _missing_node_unfinished_by_conversation(receipts):
        """Index unfinished receipts by the exact C/A4 conversation boundary."""
        fields = ("provider_binding_id", "provider_account_identity",
                  "client_conversation_id", "conversation_id")
        indexed = {}
        for kind, owner, request_id, receipt in receipts:
            if not unfinished(kind, receipt):
                continue
            values = tuple(receipt.get(field) for field in fields)
            if all(isinstance(value, str) and value for value in values):
                indexed.setdefault((owner, *values), set()).add((kind, request_id))
        return indexed

    def _snapshot(self, rows, receipts, settings, types, now, cursor, workflow_state=None, product_state=None, *, db=None):
        from services.text_task_service import TextTaskService
        works, workflow_clocks = workflow_state or ({}, {})
        product_works = product_state or {}
        product_capacity = self._product_capacity_limit(rows, settings)
        product_members = {}
        for _kind, _owner, _request_id, receipt in receipts:
            if isinstance(receipt, dict) and isinstance(receipt.get("_product_work_key"), str):
                product_members.setdefault(receipt["_product_work_key"], []).append(receipt)
        # Explicit admission markers survive ordinary AI-turn gaps.  A legacy
        # parent counts only during a genuine live claim, never for retained
        # archive/history ownership or paused/UNKNOWN history.
        product_occupied = sum(
            bool(parent.get("state") == "active" and (
                product_work.admission_slot_held(parent)
                or product_work.legacy_live_admission(parent, product_members.get(key, ()), now)))
            for key, parent in product_works.items() if isinstance(parent, dict)
        )
        # Product parents are opt-in and intentionally separate from generic
        # work rows: one product may contain both text and image originals.
        # Retained parents track ownership/closeout, not generation capacity.
        # Validate each request's own parent below; an unrelated unfinished or
        # malformed parent must not consume another product's account turns.
        # A verified empty upstream turn may have one explicitly accepted
        # correction in the same public Chat session. Its UNKNOWN receipt must
        # remain intact for recovery, but it cannot remain the session's order
        # head forever. This is a planner-only projection, never a status edit.
        text_receipts = {(owner, request_id): r for kind, owner, request_id, r in receipts
                         if kind == "text"}
        # The missing-node C proof needs to reject any unfinished receipt in
        # its exact physical conversation.  `receipts` is already the current
        # transaction snapshot; index it once instead of re-decoding the full
        # database for every C while the admission writer is held.
        missing_node_unfinished = self._missing_node_unfinished_by_conversation(receipts)
        missing_node_retry_valid = {}

        def valid_missing_node_retry(owner, request_id, receipt):
            key = (owner, request_id)
            if key not in missing_node_retry_valid:
                missing_node_retry_valid[key] = TextTaskService._missing_node_model_rejection_retry(
                    self.store, db, owner, receipt, now,
                    unfinished_by_conversation=missing_node_unfinished)
            return missing_node_retry_valid[key]
        released_order_heads, transferred_empty_turns = self._empty_turn_transfers(receipts)
        successors = {}
        for (owner, request_id), r in text_receipts.items():
            if r.get("_supersedes_request_id"):
                successors.setdefault((owner, r["_supersedes_request_id"]), []).append((request_id, r))
        # Only an exact incomplete compact continuation inherits its original
        # acceptance position for planner ordering.  This is a projection: the
        # successor retains its own durable sequence and identity.  A model
        # rejection/no-final branch, a fallback, duplicate child, or malformed
        # chain never gets an older queue position.
        continuation_sequences = {}
        children_by_parent = {}
        for (owner, request_id), receipt in text_receipts.items():
            parent_id = receipt.get("_supersedes_request_id") or receipt.get("_codex_fallback_of")
            if isinstance(parent_id, str) and parent_id:
                children_by_parent.setdefault((owner, parent_id), []).append((request_id, receipt))
        if successors:
            from services.text_task_service import TextTaskService
            for key, candidates in successors.items():
                if len(candidates) != 1:
                    continue
                request_id, successor = candidates[0]
                previous = text_receipts.get(key)
                # Failed known-unsent successors keep the old order block. A
                # submitted/unknown successor remains the new blocking head.
                advances = (successor.get("status") in {"queued", "running", "unknown", "succeeded"}
                            or unknown_text_result(successor))
                # A registered model-rejection retry is the sole exception:
                # its parent is a terminal, typed rejection of the one
                # validated no-final successor. Its one registered child is
                # revalidated immediately before send and remains the current
                # order head while queued, running, unknown, or succeeded.
                # Keep the root blocked until that exact child exists; never
                # release it for a failed rejection by itself or another turn.
                if not advances and db is not None:
                    retries = successors.get((key[0], request_id), [])
                    advances = bool(
                        len(retries) == 1
                        and (retries[0][1].get("status") in {"queued", "running", "unknown", "succeeded"}
                             or unknown_text_result(retries[0][1]))
                        and (TextTaskService._no_final_model_rejection_retry(
                                self.store, db, key[0], retries[0][1], now)
                             or valid_missing_node_retry(key[0], retries[0][0], retries[0][1])
                             or TextTaskService._missing_node_model_rejection_retry_consumed(
                                self.store, db, key[0], retries[0][1]))
                        and TextTaskService._supersede_order_link(successor, retries[0][1])
                    )
                if advances and TextTaskService._supersede_order_link(previous, successor):
                    released_order_heads.add(key)
                    # Preserve the original business request's place only for
                    # the one qualified incomplete compact child.  The source
                    # predicate is the same one rechecked at the send edge;
                    # the unique-child guard also rejects fallback/model
                    # rejection forks before they can move ahead of peers.
                    if (successor.get("_supersedes_incomplete") is True
                            and len(children_by_parent.get(key, ())) == 1
                            and TextTaskService._incomplete_compact_successor_source(previous, now)
                            and type(previous.get("_sequence")) is int):
                        continuation_sequences[(key[0], request_id)] = previous["_sequence"]
        missing_node_image_links = {}
        if db is not None:
            from services.text_task_service import TextTaskService
            for (owner, request_id), receipt in text_receipts.items():
                image_id = receipt.get(TextTaskService.MISSING_NODE_RETRY_FIELD)
                live_link = (receipt.get("status") in {"queued", "running", "unknown"}
                             and isinstance(image_id, str) and image_id
                             and valid_missing_node_retry(owner, request_id, receipt))
                # Once C has positively succeeded, late observations on R/P
                # must not revive their UNKNOWN queue barrier.  This is the
                # durable four-receipt proof, deliberately separate from live
                # eligibility. Failed/cancelled C never releases A4.
                completed_link = (receipt.get("status") == "succeeded"
                                  and receipt.get("upstream_outcome") == "completed"
                                  and isinstance(receipt.get("content"), str) and bool(receipt["content"].strip())
                                  and TextTaskService._missing_node_model_rejection_retry_consumed(
                                      self.store, db, owner, receipt))
                if live_link or completed_link:
                    missing_node_image_links[(owner, request_id)] = image_id
        image_retry_predecessors = {(owner, image_id): request_id
                                    for (owner, request_id), image_id in missing_node_image_links.items()}
        physical_pending = {}
        for kind, owner, request_id, receipt in receipts:
            key = physical_conversation_key(receipt)
            if key and unfinished(kind, receipt) and (owner, request_id) not in released_order_heads:
                physical_pending.setdefault(key, []).append((kind, owner, request_id, receipt))
        by_identity = {str(a.get("provider_account_identity") or ""): a for a in rows}
        identity_counts = {}
        for row in rows:
            identity = str(row.get("provider_account_identity") or "")
            if identity:
                identity_counts[identity] = identity_counts.get(identity, 0) + 1
        occupied = {}
        unknown_unbound = False
        active_bytes = 0
        active_text = 0
        active_images = 0
        live_turns = {}
        requests = []
        thread_resources = []
        image_receipts = {}
        for kind, owner, request_id, r in receipts:
            if kind == "image":
                image_receipts.setdefault(owner, {})[request_id] = r
        ordered_receipts = [row for row in receipts if (row[1], row[2]) not in released_order_heads]
        active_work_counts = workflow_scheduling.active_work_counts(ordered_receipts)
        for kind, owner, request_id, r in receipts:
            status = r.get("status")
            pending = unfinished(kind, r)
            unknown = unresolved_result(kind, r)
            account = by_identity.get(str(r.get("provider_account_identity") or ""))
            resource = account_clock_key(account) if account else r.get("_account_resource")
            active = status == "running" or unknown and not r.get("_attempt_finished_at")
            # Keep an unresolved result as its conversation's order head even
            # after the existing qualified-read policy ends local waiting.
            # This is not proof that the upstream turn ended or permission to
            # send the original request again.
            wait_ended = (kind == "text" and r.get("_route", "chat") == "chat"
                          and r.get("_operation", "text") == "text" and not r.get("_forward_protocol")
                          and status == "failed" and r.get("error_code") == "RESULT_UNRECOVERABLE"
                          and r.get("upstream_outcome") == "unknown"
                          and type(r.get("_execution_wait_ended_at")) in (int, float)
                          and math.isfinite(r["_execution_wait_ended_at"]))
            empty_local_released = (kind == "text" and r.get("_turn_reserved") is False
                                    and TextTaskService._verified_retryable_empty(r))
            # A bounded, observed no-progress result leaves the original
            # UNKNOWN receipt and physical/order barriers intact. It only stops
            # consuming this account's local execution slot while no worker or
            # recovery lease is active.
            stalled_local_capacity_released = (kind == "text"
                                               and (
                                                   TextTaskService._qualified_stalled_incomplete_capacity_boundary(r, now)
                                                   or TextTaskService._bounded_unreadable_result_capacity_boundary(r, now)
                                               ))
            turn_active = (active and not original_turn_ended(kind, r) and not wait_ended and not empty_local_released
                           and not stalled_local_capacity_released
                           and (owner, request_id) not in transferred_empty_turns)
            if status == "running" and r.get("_executing") and float(r.get("_claim_until") or 0) > now:
                active_bytes += int(r.get("_input_bytes") or 0)
                if kind == "image" or r.get("_operation") == "image":
                    active_images += 1
                if kind == "text" and r.get("_route", "chat") == "chat" and r.get("_operation") != "image":
                    active_text += 1
            if turn_active and not resource:
                unknown_unbound = True
            route = r.get("_route", "chat")
            # Closing a text POST stream releases its transport reservation
            # before the runner persists success/UNKNOWN. Keep the submitted
            # turn counted through that gap; otherwise another claim can send
            # and the later UNKNOWN transition restores an extra occupied slot.
            text_result_pending = (kind == "text" and route == "chat"
                                   and r.get("_operation", "text") == "text"
                                   and r.get("_submission_started") is True)
            if resource and turn_active and (unknown or r.get("_turn_reserved", True) or text_result_pending):
                key = route + "_turn:" + resource
                occupied[key] = occupied.get(key, 0) + 1
                if (kind == "text" and route == "chat" and r.get("_operation") != "image"
                        and status == "running" and r.get("_executing")
                        and r.get("_submission_started")
                        and float(r.get("_claim_until") or 0) > now):
                    live_turns[key] = live_turns.get(key, 0) + 1
            if resource and image_generation_active(kind, r, active):
                key = "image:" + resource
                occupied[key] = occupied.get(key, 0) + 1
            # A durable not_started original has no proof that its first send
            # may be resumed. Keep it as a non-terminal conversation barrier,
            # but do not place it in the planner's claimable queued set.
            state = ("unknown" if unknown else "queued" if status == "queued"
                     else "active" if status == "not_started" or active else "succeeded")
            thread_needs = ()
            if kind == "image" and r.get("_image_thread"):
                _, blocked = predecessor_state(r, image_receipts.get(owner, {}))
                dependency = "image_thread:" + hashlib.sha256((owner + "\0" + request_id).encode()).hexdigest()
                thread_resources.append(Resource(dependency, 0 if blocked else 1, 0, now))
                thread_needs = (Need(dependency),)
            if kind == "text" and r.get("_public_previous_waiting"):
                dependency = "public_chat_previous:" + hashlib.sha256((owner + "\0" + request_id).encode()).hexdigest()
                thread_resources.append(Resource(dependency, 0, 0, now))
                thread_needs += (Need(dependency),)
            if r.get("_completion_of"):
                original = (text_receipts.get((owner, r["_completion_of"])) if kind == "text"
                            else image_receipts.get(owner, {}).get(r["_completion_of"])) or {}
                original_work = works.get(original.get("_work_key")) or {}
                enabled = bool(original and not recovery_suppressed(original)
                               and original.get("_recovery_paused") is not True
                               and original_work.get("state", "active") == "active"
                               and not original.get("_completion", {}).get("selected_id"))
                dependency = "completion_work:" + hashlib.sha256((kind + "\0" + owner + "\0" + request_id).encode()).hexdigest()
                thread_resources.append(Resource(dependency, int(enabled), 0, now))
                thread_needs += (Need(dependency),)
            scheduling_resources, scheduling_needs, scheduling_ready = workflow_scheduling.constraints(
                r, works, workflow_clocks, ordered_receipts, now, active_by_work=active_work_counts)
            thread_resources.extend(scheduling_resources)
            product_needs = ()
            product_lane = None
            product_fair_group = None
            product_fair_source = None
            product_fair_root = None
            product_key = r.get("_product_work_key")
            if "_product_work_key" in r:
                parent = product_works.get(product_key)
                source = r.get("_scheduling_owner") or r.get("_source") or "key:" + owner
                valid_parent = (isinstance(product_key, str) and isinstance(parent, dict)
                                and parent.get("key") == product_key
                                and parent.get("owner") == owner
                                and parent.get("source") == source
                                and (r.get("_product_work") or {}).get("fairness") == parent.get("fairness"))
                if valid_parent and parent.get("fairness") is not None:
                    try:
                        product_fair_source = product_work.fair_subject(parent)
                        product_fair_root = parent["fairness"]["root_workflow_id"]
                    except (product_work.WorkLifecycleError, KeyError, TypeError):
                        valid_parent = False
                control = product_work.admission_control(parent) if valid_parent else None
                if valid_parent and control["state"] != "active":
                    dependency = "product_admission_paused:" + hashlib.sha256(
                        (owner + "\0" + kind + "\0" + request_id).encode()).hexdigest()
                    thread_resources.append(Resource(dependency, 0, 0, now))
                    product_needs = (Need(dependency),)
                elif valid_parent and parent.get("state") == "active":
                    product_fair_group = "product-continuation"
                    product_lane = "continuation"
                elif valid_parent and parent.get("state") == "queued" and parent.get("slot_held") is False:
                    product_fair_group = "product-new"
                    product_lane = "new"
                else:
                    # Do not dispatch a closed, mismatched, or otherwise
                    # invalid parent while preserving its original receipt.
                    dependency = "product_parent_state:" + hashlib.sha256(
                        (owner + "\0" + kind + "\0" + request_id).encode()).hexdigest()
                    thread_resources.append(Resource(dependency, 0, 0, now))
                    product_needs = (Need(dependency),)
                if valid_parent and control["state"] == "active" and not control["slot_held"]:
                    # Continuations keep their own fair lane, but a resumed or
                    # adopted product must reacquire capacity through this
                    # claim transaction before any next turn can run.
                    product_needs += (Need("product_admission"),)
            physical_needs = ()
            physical = physical_conversation_key(r)
            if physical:
                members = physical_pending.get(physical, [])
                active_physical = sum(item[3].get("status") == "running" or unresolved_result(item[0], item[3]) for item in members)
                thread_resources.append(Resource(physical, 1, active_physical, now))
                physical_needs = (Need(physical),)
                predecessors = [item for item in members if (item[1], item[2]) != (owner, request_id)
                                and int(item[3].get("_sequence") or 0) < int(r.get("_sequence") or 0)
                                and not (missing_node_image_links.get((owner, request_id)) == item[2]
                                         and item[0] == "image")]
                if predecessors:
                    order = "conversation_order:" + hashlib.sha256((kind + "\0" + owner + "\0" + request_id).encode()).hexdigest()
                    thread_resources.append(Resource(order, 0, 0, now))
                    physical_needs += (Need(order),)
            retry_predecessor = image_retry_predecessors.get((owner, request_id)) if kind == "image" else None
            if retry_predecessor:
                retry = text_receipts.get((owner, retry_predecessor)) or {}
                terminal = (retry.get("status") in {"succeeded", "failed", "cancelled"}
                            and not unresolved_result("text", retry))
                dependency = "missing_node_retry:" + hashlib.sha256(
                    (owner + "\0" + retry_predecessor + "\0" + request_id).encode()).hexdigest()
                thread_resources.append(Resource(dependency, int(terminal), 0, now))
                physical_needs += (Need(dependency),)
            # Legacy pending rows still protect their conversation order, but
            # cannot execute without a durable input reference.
            scheduling = r.get("_scheduling") or {}
            workflow_id = scheduling.get("workflow_id")
            fair_group = (product_fair_group if product_fair_group is not None
                          else workflow_id if isinstance(workflow_id, str) and workflow_id else None)
            # Product text and images draw from the same bound Chat turn, but
            # an image additionally needs its image-generation slot.  Keep the
            # existing durable sequence and conversation order checks, while
            # letting the already-persisted fairness cursor rotate a ready
            # product image after a product-text claim instead of allowing a
            # continuous older text backlog to consume every released turn.
            # This applies only to the trusted product lanes; ordinary and
            # workflow-only traffic retains its established fair group.
            if product_fair_group is not None and (kind == "image" or r.get("_operation") == "image"):
                fair_group = product_fair_group + ":image"
            fair_subgroup = fair_group if product_fair_source is not None else None
            if product_fair_source is not None:
                fair_group = product_fair_root
            if r.get("_send_hold") is True:
                continue
            requests.append(WaitingRequest(
                RequestRef(owner, kind, request_id), product_fair_source or r.get("_source") or "key:" + owner,
                continuation_sequences.get((owner, request_id), int(r.get("_sequence") or 0)),
                route, str(r.get("model") or "auto"),
                "image" if kind == "image" else r.get("_operation", "text"),
                bool(r.get("_input_ref")), state=state,
                bound_account=str(r.get("_requested_account_identity") or r.get("provider_account_identity") or "") or None,
                order_group=(None if (owner, request_id) in released_order_heads
                             or (owner, request_id) in missing_node_image_links
                             else str(r.get("client_conversation_id") or "") or None),
                # `source` is the trusted, stable caller.  The planner scopes
                # this workflow ID to that source, so workflows cannot gain a
                # second top-level caller turn by using multiple key owners.
                fair_group=fair_group,
                fair_subgroup=fair_subgroup,
                product_lane=product_lane,
                ready_at=max(float(r.get("_ready_at") or 0), scheduling_ready),
                needs=thread_needs + physical_needs + product_needs + tuple(scheduling_needs) + (Need("execution_input_bytes", max(1, int(r.get("_input_bytes") or 1))),)
                      + ((Need("chat_executor"),) if kind == "text" and route == "chat" and r.get("_operation") != "image" else ())
                      + ((Need("image_executor"),) if kind == "image" or r.get("_operation") == "image" else ()),
            ))
        codex_used = sum(value for key, value in occupied.items() if key.startswith("codex_turn:"))
        resources = {"execution_input_bytes": Resource("execution_input_bytes", self.MAX_ACTIVE_INPUT_BYTES, active_bytes, now),
                     "chat_executor": Resource("chat_executor", self.text_workers or 0, active_text, now),
                     "codex_server": Resource("codex_server", int(settings.get("codex_max_concurrency", 4)), codex_used, now),
                     "product_admission": Resource("product_admission", product_capacity, product_occupied, now)}
        resources.update({resource.key: resource for resource in thread_resources})
        offers = []
        scoped_extra_turn = None
        trial = settings.get("temporary_chat_second_slot")
        trial_active = (isinstance(trial, dict)
                        and int(settings.get("chat_account_concurrency", 1)) == 1
                        and type(trial.get("expires_at")) is int
                        and now < trial["expires_at"])
        for account in rows:
            identity = str(account.get("provider_account_identity") or "")
            if not identity:
                continue
            key = account_clock_key(account)
            pace = self._pacing(account, now)
            disabled = bool(account.get("managed_disabled") or account.get("status") in {"禁用", "异常", "限流"})
            chat_saved = str(account.get("source_type") or "web") in {"web", "oauth_login", "password"} and bool(account.get("access_token"))
            plan = self.accounts._normalize_account_type(account.get("type"))
            paid = plan in {"Plus", "Pro", "ProLite", "Team", "Enterprise"}
            image_slots = image_capacity(account, settings)
            turn_key, image_key = "chat_turn:" + key, "image:" + key
            # Account activity and send pacing are separate controls. The
            # default remains one active Chat turn per account; a validated
            # setting may raise it without changing the 10/60s pacing clocks.
            chat_capacity = max(1, int(settings.get("chat_account_concurrency", 1)))
            if (trial_active and identity == trial.get("account_identity")
                    and identity_counts.get(identity) == 1
                    and not disabled and chat_saved and paid
                    and occupied.get(turn_key, 0) == 1 and live_turns.get(turn_key, 0) == 1):
                candidates = [request for request in requests
                              if request.ref.kind == "text" and request.ref.request_id == trial.get("request_id")
                              and request.state == "queued" and request.route == "chat"
                              and request.operation == "text" and request.payload_saved
                              and request.bound_account == identity]
                if len(candidates) == 1:
                    chat_capacity = 2
                    scoped_extra_turn = (candidates[0].ref, turn_key)
            # Capacity admits bounded preparation; the durable HTTP clock still
            # gates every request and the final model POST. Waiting for a prior
            # POST's message interval here needlessly delays the next independent
            # conversation's bootstrap/requirements/prepare work as well.
            ready = pace.get("next_at")
            if ready is not None and "cooldown_until" in pace:
                ready = max(now, float(pace.get("cooldown_until") or 0))
            new = Resource(turn_key, chat_capacity, None if unknown_unbound else occupied.get(turn_key, 0), ready)
            resources[turn_key] = new
            previous = resources.get(image_key)
            resources[image_key] = Resource(image_key, min(previous.capacity, image_slots) if previous else image_slots,
                                           None if unknown_unbound else occupied.get(image_key, 0), now)
            codex_key = "codex_turn:" + key
            native_pending = any(isinstance(v, dict) and v.get("state") in {"pending", "unknown"}
                                 for v in (account.get("codex_affinities") or {}).values())
            resources[codex_key] = Resource(codex_key, 1, max(occupied.get(codex_key, 0), int(native_pending)), now)
            for request in requests:
                if request.state != "queued":
                    continue
                model = request.model
                if request.route == "codex":
                    eligible = self.codex is not None and self.codex._eligible_account(account, "" if request.operation == "image" else model, allow_probe=False) is not None
                    if request.operation == "image":
                        eligible = (eligible and account.get("source_type") == "codex"
                                    and image_capacity(account, settings, model) > 0)
                    decision = self.codex.quota_decision(account, model) if eligible and hasattr(self.codex, "quota_decision") else {}
                    offers.append(Offer(identity, "codex", model, request.operation,
                                        (Need(codex_key), Need("codex_server")), enabled=bool(eligible),
                                        preference=0 if decision.get("quota_bucket") == "gpt-reserve" else 1))
                    continue
                ineligible = []
                if disabled:
                    ineligible.append("account_disabled")
                if not chat_saved:
                    ineligible.append("source_not_authorized")
                if not paid:
                    ineligible.append("paid_plan_required")
                if identity_counts.get(identity) != 1:
                    ineligible.append("identity_not_unique")
                enabled = not disabled and chat_saved and paid and identity_counts.get(identity) == 1
                if request.operation == "image":
                    from utils.helper import is_codex_image_model, split_image_model
                    required_plan, _ = split_image_model(model)
                    model_plan_allowed = (not required_plan
                                          or plan == self.accounts._normalize_account_type(required_plan))
                    model_account_allowed = not is_codex_image_model(model)
                    enabled = enabled and image_capacity(account, settings, model) > 0 and not is_codex_image_model(model)
                    if required_plan:
                        enabled = enabled and model_plan_allowed
                    if not model_plan_allowed:
                        ineligible.append("model_plan_not_allowed")
                    if not model_account_allowed:
                        ineligible.append("model_account_not_allowed")
                    needs = (Need(turn_key), Need(image_key))
                else:
                    route = types.get(model, set())
                    model_plan_allowed = plan in getattr(route, "account_types", route)
                    model_account_allowed = self._model_account_matches(account, route)
                    enabled = enabled and model_plan_allowed and model_account_allowed
                    if not model_plan_allowed:
                        ineligible.append("model_plan_not_allowed")
                    if not model_account_allowed:
                        ineligible.append("model_account_not_allowed")
                    needs = (Need(turn_key),)
                # Explicit per-model exhausted observations apply only to that
                # model. Unknown limits are not invented as an extra balance.
                for limit in account.get("limits_progress") or []:
                    if isinstance(limit, dict) and limit.get("feature_name") == model and limit.get("remaining") == 0:
                        enabled = False
                        ineligible.append("model_limit_zero")
                offers.append(Offer(identity, "chat", model, request.operation, needs, enabled=bool(enabled),
                                    ineligible_reasons=tuple(dict.fromkeys(ineligible))))
        if self.text_workers is None:
            # Execution threads follow the same usable physical account turns.
            # The retained-input memory ceiling still bounds all work; image
            # executions use their image budget rather than the text budget.
            enabled_turns = {need.resource for offer in offers if offer.enabled and offer.route == "chat"
                             for need in offer.needs if need.resource.startswith("chat_turn:")}
            # Offers exist only for queued models. Account eligibility must also
            # be visible when idle, without triggering catalog or model calls.
            for account in rows:
                if (not account.get("managed_disabled") and account.get("status") not in {"禁用", "异常", "限流"}
                        and str(account.get("source_type") or "web") in {"web", "oauth_login", "password"}
                        and account.get("access_token") and account.get("provider_account_identity")
                        and self.accounts._normalize_account_type(account.get("type")) in {"Plus", "Pro", "ProLite", "Team", "Enterprise"}):
                    enabled_turns.add("chat_turn:" + account_clock_key(account))
            resources["chat_executor"] = Resource("chat_executor", sum(resources[key].capacity for key in enabled_turns), active_text, now)
        # Leave room for confirmed images to save while new generation runs,
        # but never allow slow downloads to create an unbounded thread backlog.
        # This is a local retained-work ceiling, separate from upstream slots.
        image_keys = {need.resource for offer in offers if offer.enabled
                      for need in offer.needs if need.resource.startswith("image:") or (offer.operation == "image" and need.resource.startswith("codex_turn:"))}
        usable_chat_keys = {"image:" + account_clock_key(account) for account in rows
                            if account.get("provider_account_identity") and account.get("access_token")
                            and not account.get("managed_disabled") and account.get("status") not in {"禁用", "异常", "限流"}
                            and str(account.get("source_type") or "web") in {"web", "oauth_login", "password"}
                            and self.accounts._normalize_account_type(account.get("type")) in {"Plus", "Pro", "ProLite", "Team", "Enterprise"}}
        image_keys.update(usable_chat_keys)
        if self.codex is not None:
            image_keys.update("codex_turn:" + account_clock_key(account) for account in rows
                              if account.get("provider_account_identity") and account.get("source_type") == "codex"
                              and self.codex._eligible_account(account, allow_probe=False) is not None)
        resources["image_executor"] = Resource("image_executor", 2 * sum(resources[key].capacity for key in image_keys), active_images, now)
        # Sources and owners are server-authenticated; cursors move only on a claim.
        unique = {offer.key: offer for offer in offers}
        groups = cursor.get("groups") or {}
        if not isinstance(groups, dict):
            groups = {}
        subgroups = cursor.get("subgroups") or {}
        if not isinstance(subgroups, dict):
            subgroups = {}
        return Snapshot(uuid.uuid4().hex, now, now + 1, tuple(resources.values()), tuple(unique.values()), tuple(requests),
                        last_source=cursor.get("source"), last_account=cursor.get("account"),
                        last_owner_by_source=tuple((cursor.get("owners") or {}).items()),
                        last_group_by_source=tuple((source, group) for source, group in groups.items()
                                                   if isinstance(source, str) and isinstance(group, str)),
                        scoped_extra_turn=scoped_extra_turn,
                        last_subgroup_by_group=tuple((key, value) for key, value in subgroups.items()
                                                    if isinstance(key, str) and isinstance(value, str)))

    def _recover_claims(self, db, receipts, now):
        from services.text_task_service import TextTaskService
        for kind, owner, request_id, r in receipts:
            # Older expired-unsent recovery cleared the claim but left its
            # execution marker behind. No worker without a claim can send;
            # clear only that fenced, unsent marker so work can be paused.
            if (r.get("status") == "queued" and r.get("_claim_id") is None
                    and r.get("_executing") is True and r.get("_submission_started") is False):
                r.update(_executing=False, _claim_until=None)
                self.store.write_receipt(db, kind, owner, request_id, r)
            if kind == "text" and TextTaskService._end_execution_wait(r, now):
                self.store.write_receipt(db, kind, owner, request_id, r)
            if not r.get("_claim_id") or r.get("status") != "running" or float(r.get("_claim_until") or 0) > now:
                continue
            if r.get("_submission_started"):
                r.update(status="unknown" if kind == "text" else "error", error_code="CONVERSATION_OUTCOME_UNKNOWN")
                if kind == "image":
                    r["upstream_unfinished"] = not bool(r.get("result_file_ids") or r.get("result_sediment_ids"))
                    if not r["upstream_unfinished"]:
                        r.update(upstream_outcome="generated",
                                 recovery_error_code=r.get("recovery_error_code") or "RECOVERY_RESULT_INCOMPLETE",
                                 recovery_phase="download_image_result", next_poll_at=0)
            else:
                r.update(status="queued", _turn_reserved=False, _claim_id=None,
                         _claim_until=None, _executing=False, upstream_unfinished=False)
            self.store.write_receipt(db, kind, owner, request_id, r)

    def resource_snapshot(self):
        """Read-only projection of the same persisted occupancy used to claim.

        Null means unresolved occupancy; zero is a known empty/closed resource.
        Session counts are intentionally absent from execution capacity.
        """
        now = float(self.clock())
        # Match claim/send lock order: a read holding the account lock must
        # never wait for settings while admission holds settings for accounts.
        with self._settings_guard(), self.store.connect() as db, self._account_guard():
            rows = self._rows()
            receipts = list(self.store.receipts(db))
            settings = self._settings()
            workflow_state = workflow_scheduling.state_snapshot(self.store, db)
            product_state = product_work.state_snapshot(self.store, db)
            snapshot = self._snapshot(rows, receipts, settings, {}, now, {}, workflow_state, product_state, db=db)
        resources = {r.key: r for r in snapshot.resources}
        result = []
        seen = set()
        for account in rows:
            key = account_clock_key(account)
            if key in seen:
                continue
            seen.add(key)
            identity = account.get("provider_account_identity")
            enabled = not account.get("managed_disabled") and account.get("status") not in {"禁用", "异常", "限流"}
            chat_ok = enabled and str(account.get("source_type") or "web") in {"web", "oauth_login", "password"} and bool(account.get("access_token")) and self.accounts._normalize_account_type(account.get("type")) in {"Plus", "Pro", "ProLite", "Team", "Enterprise"}
            native_ok = self.codex is not None and self.codex._eligible_account(account, allow_probe=False) is not None
            item = {"provider_account_identity": identity, "account_ref": self.accounts.pool_account_ref(account),
                    "enabled": bool(enabled), "status": str(account.get("status") or ""),
                    "chat_eligible": bool(chat_ok)}
            for name, prefix, usable in (("chat_turn", "chat_turn:", chat_ok), ("image", "image:", chat_ok), ("codex", "codex_turn:", native_ok)):
                resource = resources.get(prefix + key)
                if resource is None:
                    item[name] = {"capacity": 0, "occupied": None, "free": None, "dispatchable_now": None}
                    continue
                cap = resource.capacity if usable else 0
                free = None if resource.occupied is None else max(0, cap - resource.occupied)
                ready = resource.next_at is not None and resource.next_at <= now
                item[name] = {"capacity": cap, "occupied": resource.occupied, "free": free,
                              "dispatchable_now": None if resource.next_at is None else free if ready else 0, "next_at": resource.next_at}
            image = item["image"]
            turn = item["chat_turn"]
            image["dispatchable_now"] = None if image["free"] is None or turn["dispatchable_now"] is None else min(image["free"], turn["dispatchable_now"])
            result.append(item)
        def aggregate(name):
            values = [item[name] for item in result]
            return {"slots_total": sum(v["capacity"] for v in values),
                    "inflight": None if any(v["occupied"] is None for v in values) else sum(v["occupied"] for v in values),
                    "slots_free": None if any(v["free"] is None for v in values) else sum(v["free"] for v in values),
                    "dispatchable_now": None if any(v["dispatchable_now"] is None for v in values) else sum(v["dispatchable_now"] for v in values)}
        image_summary = aggregate("image")
        image_workers_free = max(0, resources["image_executor"].capacity - resources["image_executor"].occupied)
        if resources["execution_input_bytes"].occupied >= self.MAX_ACTIVE_INPUT_BYTES:
            image_workers_free = 0
        if image_summary["dispatchable_now"] is not None:
            image_summary["dispatchable_now"] = min(image_summary["dispatchable_now"], image_workers_free)
        for item in result:
            if item["image"]["dispatchable_now"] is not None:
                item["image"]["dispatchable_now"] = min(item["image"]["dispatchable_now"], image_workers_free)
        native = aggregate("codex")
        native["server_limit"] = settings["codex_max_concurrency"]
        native["slots_total"] = min(native["slots_total"], native["server_limit"])
        if native["slots_free"] is not None:
            native["slots_free"] = min(native["slots_free"], max(0, native["server_limit"] - native["inflight"]))
            native["dispatchable_now"] = min(native["dispatchable_now"], native["slots_free"])
        queued = [(kind, owner, r) for kind, owner, _, r in receipts if r.get("status") == "queued"]
        sources, reasons = {}, {}
        oldest = 0.0
        for _, owner, r in queued:
            source = r.get("_source") or "key:" + owner
            sources[source] = sources.get(source, 0) + 1
            accepted = next((stage.get("at") for stage in r.get("_execution_timeline", [])
                             if stage.get("stage") == "accepted"), r.get("created_ts") or r.get("created_at"))
            if type(accepted) in (int, float):
                oldest = max(oldest, max(0, now - accepted))
            for reason in (r.get("waiting") or {}).get("reasons", ["awaiting_dispatch"]):
                reasons[reason] = reasons.get(reason, 0) + 1
        recovery = sum(1 for kind, _, _, r in receipts if not recovery_suppressed(r)
                       and r.get("_recovery_paused") is not True
                       and (not r.get("_attempt_finished_at") or kind == "image"
                            and bool(r.get("result_file_ids") or r.get("result_sediment_ids") or pending_image_result_ids(r))) and
                       (unknown_text_result(r) if kind == "text" else r.get("upstream_unfinished") is True
                        and r.get("status") in {"error", "unknown"}))
        saving = sum(1 for kind, _, _, r in receipts if r.get("status") == "running"
                     and (kind == "image" or r.get("_operation") == "image")
                     and not image_generation_active(kind, r, True))
        product_parents = [parent for parent in product_state.values() if isinstance(parent, dict)]
        return {"accounts": result, "settings": settings, "chat_turn": aggregate("chat_turn"), "image": image_summary, "codex": native,
                "workflows": {"slots_held": sum(bool(w.get("slot_held")) for w in workflow_state[0].values()),
                              "active_work": sum(w.get("state") == "active" for w in workflow_state[0].values()),
                              "paused_work": sum(w.get("state") == "paused" for w in workflow_state[0].values())},
                "products": {"admission_limited": True,
                             "slots_total": resources["product_admission"].capacity,
                             "slots_held": resources["product_admission"].occupied,
                             "slots_free": (None if resources["product_admission"].occupied is None else max(
                                 0, resources["product_admission"].capacity - resources["product_admission"].occupied)),
                             # Existing lifecycle ownership remains observable
                             # separately; archive/history cannot be mistaken
                             # for a product-admission reservation.
                             "lifecycle_slots_held": sum(parent.get("slot_held") is True for parent in product_parents),
                             "queued_products": sum(parent.get("state") == "queued" for parent in product_parents),
                             "active_products": sum(parent.get("state") == "active" for parent in product_parents)},
                "queue": {"mode": "durable_original_receipts", "queued": len(queued), "by_source": sources,
                          "by_reason": reasons, "oldest_wait_seconds": oldest,
                          "recovering_original": recovery, "saving_images": saving},
                "execution": {"active_input_bytes": resources["execution_input_bytes"].occupied, "max_input_bytes": self.MAX_ACTIVE_INPUT_BYTES,
                              "chat_workers_active": resources["chat_executor"].occupied,
                              "chat_workers_limit": resources["chat_executor"].capacity,
                              "image_workers_active": resources["image_executor"].occupied,
                              "image_workers_limit": resources["image_executor"].capacity}}

    def model_resources(self, model_ids):
        """Exact-model readback of original physical turns, no additional pool."""
        from services.owned_accounts import public_pool_account
        now = float(self.clock())
        with self._settings_guard(), self.store.connect() as db, self._account_guard():
            rows = self._rows()
            snapshot = self._snapshot(rows, list(self.store.receipts(db)), self._settings(), {}, now, {},
                                      workflow_scheduling.state_snapshot(self.store, db), product_work.state_snapshot(self.store, db),
                                      db=db)
        resources = {resource.key: resource for resource in snapshot.resources}
        server = resources["codex_server"]
        server_free = None if server.occupied is None else max(0, server.capacity - server.occupied)
        result = {}
        for model in model_ids:
            seen = set()
            details = []
            for account in rows:
                key = account_clock_key(account)
                if key in seen:
                    continue
                seen.add(key)
                projection = self.codex.account_projection(account)
                known = {item["id"] for item in projection["models"]}
                if "gpt-reserve" in known and any(limit.get("id") == "gpt-reserve" and limit.get("normal_model_slug") == "gpt-5.6-luna" for limit in projection["limits"]):
                    known.add("gpt-5.6-luna")
                if model not in known:
                    continue
                decision = self.codex.quota_decision(account, model)
                if decision["reason"] == "model_not_supported":
                    continue
                resource = resources.get("codex_turn:" + key)
                occupied = resource.occupied if resource else None
                eligible = decision["state"] == "available"
                unknown = decision["state"] == "unknown" or occupied is None or server_free is None
                free = None if unknown else int(eligible and occupied == 0 and server_free > 0)
                reason = decision["reason"]
                if eligible:
                    reason = "occupancy_unknown" if unknown else "account_busy" if occupied else "server_busy" if server_free == 0 else "ready"
                details.append({"account_ref": public_pool_account(account)["account_ref"],
                                "eligible": eligible, "state": decision["state"], "reason": decision["reason"],
                                "quota_bucket": decision["quota_bucket"], "occupied": occupied, "dispatchable_now": free,
                                "next_at": now if free else decision["next_at"], "dispatch_reason": reason})
            eligible_rows = [row for row in details if row["eligible"]]
            ready = None if any(row["dispatchable_now"] is None for row in details) else min(server_free or 0, sum(row["dispatchable_now"] for row in details))
            occupied = None if any(row["occupied"] is None for row in details) else sum(row["occupied"] for row in details)
            next_times = [row["next_at"] for row in details if row["next_at"] is not None]
            result[model] = {"eligible_accounts": len(eligible_rows), "occupied": occupied,
                             "dispatchable_now": ready, "next_at": min(next_times) if next_times else None,
                             "accounts": details}
        return result

    def _metadata_receipts(self, db, statuses):
        # Allocation metadata is useful only for work that may dispatch.
        # Paused/completed/restoring work must not keep probing an account just
        # because its original receipt still says queued. Legacy rows without
        # a work projection retain their existing allocation behavior.
        works, _ = workflow_scheduling.state_snapshot(self.store, db)
        parents = product_work.state_snapshot(self.store, db)
        for item in self.store.receipts(db, statuses=statuses):
            if item[3].get("_send_hold") is True:
                continue
            work = works.get(item[3].get("_work_key")) or {}
            parent = parents.get(item[3].get("_product_work_key")) or {}
            if work.get("state", "active") == "active" and parent.get("state", "active") in {"queued", "active"}:
                yield item

    def _start_image_capability_refresh(self):
        # Credential recovery and metadata clocks can wait before HTTP timeouts
        # apply. Keep them off the admission/recovery dispatcher, with one live
        # sweep per admission instance even when a previous read is stuck.
        # Do not time out and replace the worker: Python cannot cancel that I/O.
        with self._image_refresh_lock:
            if (self._stop.is_set() or float(self.clock()) < self._next_image_probe
                    or not callable(getattr(self.accounts, "refresh_image_capability", None))
                    or (self._image_refresh_thread is not None and self._image_refresh_thread.is_alive())):
                return
            # Do not initialize a store or launch idle workers just to discover
            # an empty queue. The normal claim path owns store initialization.
            if not self.store.path.is_file():
                return
            with self.store.connect() as db:
                waiting = any((kind == "image" or r.get("_operation") == "image")
                              and r.get("status") == "queued" and not r.get("_submission_started")
                              for kind, _, _, r in self._metadata_receipts(db, ("queued",)))
            if not waiting or self._stop.is_set():
                return

            def run():
                try:
                    refreshed = self._refresh_waiting_image_capabilities()
                except Exception:
                    from utils.log import logger
                    logger.warning({"event": "image_capability_refresh_unavailable", "layer": "provider"})
                else:
                    # An empty queue must not wake itself into a busy loop.
                    if refreshed and not self._stop.is_set():
                        self.wake()

            self._image_refresh_thread = threading.Thread(
                target=run, name="image-capability-refresh", daemon=True)
            self._image_refresh_thread.start()

    def _refresh_waiting_image_capabilities(self):
        refresh = getattr(self.accounts, "refresh_image_capability", None)
        if self._stop.is_set() or not callable(refresh) or float(self.clock()) < self._next_image_probe:
            return
        with self.store.connect() as db:
            waiting = [r for kind, _, _, r in self._metadata_receipts(db, ("queued",))
                       if (kind == "image" or r.get("_operation") == "image")
                       and r.get("status") == "queued" and not r.get("_submission_started")]
        if not waiting:
            return
        self._next_image_probe = float(self.clock()) + 30
        from services.owned_accounts import observed_capacity
        from utils.helper import is_codex_image_model
        with self._account_guard():
            rows = [dict(row) for row in self._rows()]
        probes = 0
        start = self._image_probe_index % max(1, len(rows))
        for offset in range(len(rows)):
            if self._stop.is_set():
                break  # Let an already-started read finish, but start no next probe.
            account = rows[(start + offset) % len(rows)]
            if account.get("managed_disabled") or account.get("status") == "禁用":
                continue
            matching = [r for r in waiting if (
                not (r.get("_requested_account_identity") or r.get("provider_account_identity"))
                or (r.get("_requested_account_identity") or r.get("provider_account_identity")) == account.get("provider_account_identity"))
                and (r.get("_route") == "codex" or is_codex_image_model(r.get("model", "gpt-image-2"))) == (account.get("source_type") == "codex")]
            if not matching:
                continue
            capacity = observed_capacity(account)
            if (account.get("source_type") != "codex" and capacity["state"] == "observed"
                    and capacity["remaining"] is not None and capacity["remaining"] > 0
                    and account.get("status") not in {"异常", "限流"}):
                continue
            try:
                refresh(self.accounts.pool_account_ref(account))
            except Exception:
                pass  # A metadata failure keeps the same queue/identity intact.
            probes += 1
            self._image_probe_index = (start + offset + 1) % max(1, len(rows))
            if probes >= 3:
                break
        return probes > 0

    def _queued_codex_probe_accounts(self, rows, queued):
        """Return exact native-original accounts that may need observation.

        A selected or already-bound original must only ever observe its own
        account.  Resolve the same anchors as the later claim-time binding
        check, but leave conflicts alone for that transaction to reject; a
        metadata probe must not turn an invalid anchor into a fallback choice.
        """
        by_identity = {}
        for account in rows:
            identity = account.get("provider_account_identity")
            if isinstance(identity, str) and identity:
                by_identity.setdefault(identity, []).append(account)
        targeted = []
        has_unbound = False
        for owner, receipt in queued:
            anchors = []
            invalid = False
            constrained = False
            account_ref = receipt.get("_requested_account_ref")
            if account_ref is not None and account_ref != "":
                constrained = True
                matches = [account for account in rows if self.accounts.pool_account_ref(account) == account_ref] if isinstance(account_ref, str) else []
                if isinstance(account_ref, str) and len(matches) == 1:
                    anchors.append(matches[0].get("provider_account_identity"))
                else:
                    invalid = True
            for key in ("_requested_account_identity", "provider_account_identity"):
                identity = receipt.get(key)
                if identity is not None and identity != "":
                    constrained = True
                    matches = by_identity.get(identity, []) if isinstance(identity, str) else []
                    if isinstance(identity, str) and len(matches) == 1:
                        anchors.append(identity)
                    else:
                        invalid = True
            affinity = receipt.get("client_conversation_id")
            if affinity is not None and affinity != "":
                matches = ([account for account in rows if affinity in (account.get("codex_affinities") or {})]
                           if isinstance(affinity, str) else [])
                if matches:
                    constrained = True
                    if len(matches) == 1:
                        anchors.append(matches[0].get("provider_account_identity"))
                    else:
                        invalid = True
                elif not isinstance(affinity, str):
                    constrained = True
                    invalid = True
            previous = receipt.get("_previous_response_id")
            if previous is not None and previous != "":
                constrained = True
                digest = hashlib.sha256(previous.encode()).hexdigest() if isinstance(previous, str) else None
                owner_digest = hashlib.sha256(owner.encode()).hexdigest()
                matches = ([
                    account for account in rows
                    if (account.get("codex_response_ids", {}).get(digest) or {}).get("owner") == owner_digest
                ] if digest is not None else [])
                if digest is not None and len(matches) == 1:
                    anchors.append(matches[0].get("provider_account_identity"))
                else:
                    invalid = True
            if not constrained:
                has_unbound = True
                continue
            identities = {identity for identity in anchors if isinstance(identity, str) and identity}
            if invalid or len(identities) != 1:
                continue
            identity = next(iter(identities))
            if not any(account.get("provider_account_identity") == identity for account in targeted):
                targeted.append(by_identity[identity][0])
        if not has_unbound:
            return targeted
        # An unrelated unbound original still needs normal pool discovery.
        # Prioritize exact originals first, then keep every remaining account
        # in the bounded round-robin so one invalid/failed selection cannot
        # block independent work on the same queue.
        return targeted + [account for account in rows if account not in targeted]

    def claim_next(self):
        # Catalog lookup may perform a metadata read; never do that under the
        # database/account transaction or occupy an execution worker waiting.
        # UNKNOWN originals only need result recovery, never a new allocation.
        # Keeping their order fence must not continually refresh model metadata
        # (including after a completion child has finished) and starve archive
        # reads on the same account's paced HTTP clock.
        with self.store.connect() as db:
            models = {str(r.get("model") or "auto") for kind, _, _, r in self._metadata_receipts(db, ("queued", "running", "not_started"))
                      if kind == "text" and unfinished(kind, r) and not unknown_text_result(r)
                      and r.get("_route", "chat") == "chat"}
        types = {}
        for model in models:
            try:
                types[model] = self._types(model)
            except Exception:
                types[model] = set()
        if self.codex is not None:
            with self.store.connect() as db:
                native_queued = [(owner, r) for _, owner, _, r in self._metadata_receipts(db, ("queued",))
                                 if r.get("status") == "queued" and r.get("_route") == "codex"]
                native_models = {r.get("model") for _, r in native_queued}
            # Reuse the existing bounded metadata refresh, outside the claim
            # transaction. This never sends a model request.
            if native_models and float(self.clock()) >= self._next_codex_probe:
                self._next_codex_probe = float(self.clock()) + 30
                probes = 0
                with self._account_guard():
                    probe_rows = [dict(account) for account in self._rows()]
                candidates = self._queued_codex_probe_accounts(probe_rows, native_queued)
                start = self._codex_probe_index % max(1, len(candidates))
                scanned = 0
                for offset in range(len(candidates)):
                    account = candidates[(start + offset) % len(candidates)]
                    scanned += 1
                    projection = self.codex.account_projection(account)
                    # A normal pool/Web account can already have a verified
                    # Codex observation without separately stored Codex
                    # credentials.  Once that observation expires, keep its
                    # queued original on the same account's ordinary bounded
                    # observation path.  Do not turn a Chat-only account with
                    # no prior Codex evidence into a new probe candidate.
                    cached_observation = isinstance(account.get("codex_observation"), dict) and bool(
                        projection.get("observed_at")
                    )
                    if (account.get("managed_disabled") or account.get("status") in {"禁用", "异常"}
                            or projection.get("state") == "auth_required"
                            or (projection.get("authorization_status") != "saved" and not cached_observation)):
                        continue
                    if self.codex._observation_fresh(projection) and projection.get("state") != "unknown":
                        continue
                    if float((account.get("codex_rate_limit") or {}).get("cooldown_until") or 0) > time.time():
                        continue
                    self.codex._eligible_account(account, allow_probe=True)
                    probes += 1
                    if probes >= 3:
                        break
                if candidates:
                    self._codex_probe_index = (start + scanned) % len(candidates)
        with self._settings_guard(), self.store.transaction() as db, self._account_guard():
            now = float(self.clock())
            receipts = list(self.store.receipts(db))
            workflow_scheduling.expire_unsent(self.store, db, receipts, now)
            self._recover_claims(db, receipts, now)
            product_work.reconcile_paused_admission_slots(self.store, db, receipts, now=now)
            bind_waiting_threads(self.store, db, receipts)
            from services.text_task_service import bind_waiting_public_sessions
            bind_waiting_public_sessions(self.store, db, receipts)
            product_work.release_rejected_product_slots(self.store, db, receipts)
            if self.codex is not None:
                rows = self._rows()
                for kind, owner, request_id, r in receipts:
                    if r.get("_route") != "codex" or r.get("status") != "queued":
                        continue
                    bound = [a for a in rows if r.get("client_conversation_id") in (a.get("codex_affinities") or {})]
                    previous = r.get("_previous_response_id")
                    if previous:
                        digest = hashlib.sha256(previous.encode()).hexdigest()
                        prior = [a for a in rows if (a.get("codex_response_ids", {}).get(digest) or {}).get("owner") == hashlib.sha256(owner.encode()).hexdigest()]
                        if len(prior) != 1 or (bound and bound[0]["provider_account_identity"] != prior[0]["provider_account_identity"]):
                            r.update(status="failed", error_code="codex_response_owner_unknown")
                        else:
                            bound = prior
                    if len(bound) > 1:
                        r.update(status="failed", error_code="codex_binding_conflict")
                    elif bound:
                        if (r.get("_requested_account_identity")
                                and r["_requested_account_identity"] != bound[0]["provider_account_identity"]):
                            r.update(status="failed", error_code="IMAGE_ACCOUNT_SELECTION_CONFLICT" if kind == "image" or r.get("_operation") == "image" else "CHAT_ACCOUNT_SELECTION_CONFLICT")
                        else:
                            r["provider_account_identity"] = bound[0]["provider_account_identity"]
                    self.store.write_receipt(db, kind, owner, request_id, r)
            settings = self._settings()
            workflow_state = workflow_scheduling.state_snapshot(self.store, db)
            product_state = product_work.state_snapshot(self.store, db)
            snapshot = self._snapshot(self._rows(), receipts, settings, types, now,
                                      self.store.runtime(db, "fairness", {}), workflow_state, product_state, db=db)
            selection = choose_next(snapshot, now)
            waiting_active_work_counts = workflow_scheduling.active_work_counts(receipts)
            for deferred in selection.deferred:
                r = self.store.read_receipt(db, deferred.ref.kind, deferred.ref.owner, deferred.ref.request_id)
                dependency_reason = r.get("_image_thread_waiting_reason") or r.get("_public_previous_waiting_reason")
                reasons = [dependency_reason] if dependency_reason else list(deferred.reasons)
                works, clocks = workflow_state
                reasons = workflow_scheduling.waiting_reasons(
                    r, works, clocks, receipts, now, reasons, active_by_work=waiting_active_work_counts)
                if r.get("_completion_of"):
                    original = self.store.read_receipt(db, deferred.ref.kind, deferred.ref.owner, r["_completion_of"]) or {}
                    original_work = works.get(original.get("_work_key")) or {}
                    if original.get("_recovery_paused") is True:
                        reasons = list(dict.fromkeys([*reasons, "recovery_paused"]))
                    if recovery_suppressed(original) or original_work.get("state", "active") != "active":
                        reasons = list(dict.fromkeys([*reasons, "work_not_active"]))
                if ((deferred.ref.kind == "image" or r.get("_operation") == "image")
                        and r.get("_requested_account_identity")):
                    selected = [a for a in self._rows() if a.get("provider_account_identity") == r["_requested_account_identity"]]
                    if len(selected) == 1 and image_capacity(selected[0], settings, r.get("model", "gpt-image-2")) <= 0:
                        from services.owned_accounts import image_capability_projection
                        reason = image_capability_projection(selected[0]).get("reason") or "capability_unknown"
                        if selected[0].get("source_type") == "codex":
                            reason = "capability_unknown"
                        elif reason == "image_capability_observed":
                            reason = "model_not_supported"
                        reasons.append("image_" + reason)
                waiting = {"reasons": reasons, "next_check_at": deferred.next_at}
                diagnostics = waiting_admission_diagnostics(snapshot.observed_at, snapshot.revision,
                                                            deferred.offer_diagnostics)
                if diagnostics is not None:
                    waiting["admission_diagnostics"] = diagnostics
                previous_id = (r.get("_image_thread") or {}).get("previous_task_id")
                if previous_id and r.get("_image_thread_waiting_reason"):
                    # Same-owner original receipt is the recovery entry. This is
                    # diagnostic only: never replace or resend its failed turn.
                    waiting.update(previous_task_id=previous_id, action="read_original_predecessor")
                if r.get("_public_previous_waiting"):
                    waiting.update(previous_request_id=r["_previous_request_id"], action="read_original_predecessor")
                if not _same_waiting_projection(r.get("waiting"), waiting):
                    r["waiting"] = waiting
                    self.store.write_receipt(db, deferred.ref.kind, deferred.ref.owner, deferred.ref.request_id, r)
            if selection.dispatch is None:
                return None
            pick = selection.dispatch
            ref = pick.ref
            r = self.store.read_receipt(db, ref.kind, ref.owner, ref.request_id)
            if r.get("status") != "queued" or not r.get("_input_ref"):
                return None
            if r.get("_send_hold") is True:
                return None
            if ref.kind == "text":
                # A stale queue status never reopens a model POST after its
                # durable send fence. Preserve the original as UNKNOWN for
                # result recovery; an armed first-send control cannot make it
                # an effective-model retry.
                if (_legacy_combined_copy_receipt_family(r)
                        and (r.get("_submission_started") is True
                        or r.get("_last_sent_sequence") is not None)):
                    r.update(status="unknown", upstream_outcome="unknown",
                             error_code="CONVERSATION_OUTCOME_UNKNOWN",
                             _claim_id=None, _claim_until=None,
                             _executing=False, _turn_reserved=False)
                    self.store.write_receipt(db, ref.kind, ref.owner, ref.request_id, r)
                    return None
                from services.text_task_service import TextTaskService
                missing_attempt = TextTaskService._missing_successor_attempt(r)
                if ("_missing_successor_resume" in r and missing_attempt is None) or (
                        missing_attempt and missing_attempt["state"] != "reserved"):
                    # A stale generic requeue must never turn the durable
                    # one-use resend into a second transport attempt.
                    r.update(status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                             _claim_id=None, _claim_until=None, _executing=False, _turn_reserved=False)
                    self.store.write_receipt(db, ref.kind, ref.owner, ref.request_id, r)
                    return None
            requested = r.get("_requested_account_identity")
            if (r.get("_requested_account_ref") and not requested
                    or requested and (pick.account != requested
                                      or r.get("provider_account_identity") not in (None, "", requested))):
                raise AdmissionLost("requested Chat account changed before claim")
            # This only records a local account binding. It never probes or
            # refreshes an upstream credential in the claiming transaction.
            binding = r.get("provider_binding_id") or (self.accounts.admission_binding(pick.account) if r.get("_route", "chat") == "chat" else None)
            rows = {a["provider_account_identity"]: a for a in self._rows()}
            claim = uuid.uuid4().hex
            timeline = list(r.get("_execution_timeline") or [])
            timeline.append({"stage": "execution_claimed", "at": now})
            trial_marker = None
            if snapshot.scoped_extra_turn and ref == snapshot.scoped_extra_turn[0]:
                trial_marker = {**settings["temporary_chat_second_slot"],
                                "revision": settings.get("revision", 0)}
            firstsend_trial_marker = None
            firstsend_trial = settings.get("temporary_firstsend_model_trial")
            firstsend_target = _firstsend_model_trial_target(firstsend_trial, ref.request_id)
            if (ref.kind == "text" and isinstance(firstsend_trial, dict)
                    and type(firstsend_trial.get("expires_at")) is int and now < firstsend_trial["expires_at"]
                    and isinstance(firstsend_target, dict)
                    and firstsend_target.get("account_identity") == pick.account
                    and r.get("provider_account_identity") in (None, "", pick.account)
                    and r.get("_submission_started") in (None, False)
                    and not _combined_copy_send_evidence(r)):
                firstsend_trial_marker = {**firstsend_target, "expires_at": firstsend_trial["expires_at"],
                                          "revision": settings.get("revision", 0)}
            r.update(status="running", _claim_id=claim, _claim_until=now + self.CLAIM_SECONDS,
                     _turn_reserved=True, _submission_started=False, _executing=True,
                     _account_resource=account_clock_key(rows[pick.account]),
                     _temporary_chat_second_slot=trial_marker,
                     _temporary_firstsend_model_trial=firstsend_trial_marker,
                     _execution_timeline=timeline[-32:],
                     provider_binding_id=binding, provider_account_identity=pick.account)
            if r.get("_route") == "codex" and self.codex is not None and r.get("_forward_protocol") == "codex":
                account = rows[pick.account]
                affinity = r["client_conversation_id"]
                if affinity not in (account.get("codex_affinities") or {}):
                    self.codex._persist_mapping(account["access_token"], "codex_affinities", affinity,
                                                {"state": "bound", "bound_at": now}, required=True)
            r.pop("waiting", None)
            if ref.kind == "image":
                r["upstream_unfinished"] = True
                r["client_conversation_id"] = r.get("client_conversation_id") or "image-task-" + uuid.uuid4().hex
                r["binding_status"] = "bound"
            from services.work_lifecycle import WorkLifecycleError, record_slot
            record_slot(self.store, db, ref.kind, ref.owner, r, True)
            try:
                product_work.record_slot(self.store, db, r, True)
            except WorkLifecycleError as exc:
                raise AdmissionLost(exc.code) from None
            self.store.write_receipt(db, ref.kind, ref.owner, ref.request_id, r)
            owners = {source: owner for source, owner in snapshot.last_owner_by_source
                      if any(request.source == source and request.state == "queued" for request in snapshot.requests)}
            owners[pick.source] = ref.owner
            groups = {source: group for source, group in snapshot.last_group_by_source
                      if any(request.source == source and request.state == "queued" for request in snapshot.requests)}
            selected_group = next((request.fair_group for request in snapshot.requests if request.ref == ref), None)
            groups[pick.source] = fairness_lane(selected_group)
            active_subgroups = {fairness_subgroup_key(request.source, request.fair_group)
                                for request in snapshot.requests if request.state == "queued"
                                and request.fair_subgroup is not None}
            subgroups = {key: value for key, value in snapshot.last_subgroup_by_group if key in active_subgroups}
            selected = next(request for request in snapshot.requests if request.ref == ref)
            if selected.fair_subgroup is not None:
                subgroups[fairness_subgroup_key(pick.source, selected_group)] = fairness_lane(selected.fair_subgroup)
            self.store.set_runtime(db, "fairness", {"source": pick.source, "account": pick.account,
                                                      "owners": owners, "groups": groups, "subgroups": subgroups})
            return ExecutionContext(self, ref.kind, ref.owner, ref.request_id, claim)

    def update_claim(self, context, **changes):
        with self.store.transaction() as db:
            r = self.store.read_receipt(db, context.kind, context.owner, context.request_id)
            if r is None or r.get("_claim_id") != context.claim:
                raise AdmissionLost("original task claim changed")
            r.update(changes)
            self.store.write_receipt(db, context.kind, context.owner, context.request_id, r)
        self.wake()

    def renew_claim(self, context, *, busy_timeout_secs=None):
        """Extend one live claim from the instant its SQLite writer is held."""
        timeout = (self.CLAIM_HEARTBEAT_WRITER_TIMEOUT_SECS
                   if busy_timeout_secs is None else busy_timeout_secs)
        with self.store.transaction(busy_timeout_secs=timeout) as db:
            r = self.store.read_receipt(db, context.kind, context.owner, context.request_id)
            if r is None or r.get("_claim_id") != context.claim:
                raise AdmissionLost("original task claim changed")
            # `BEGIN IMMEDIATE` may have waited behind a broad admission pass.
            # Do not persist a deadline computed before that wait.
            r["_claim_until"] = float(self.clock()) + self.CLAIM_SECONDS
            self.store.write_receipt(db, context.kind, context.owner, context.request_id, r)
        self.wake()

    @staticmethod
    def _sqlite_lock_contention(exc):
        """Only retry a SQLite BUSY/LOCKED error; schema/I/O faults must surface."""
        if not isinstance(exc, sqlite3.OperationalError):
            return False
        code = getattr(exc, "sqlite_errorcode", None)
        return (type(code) is int and (code & 0xff) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})

    def image_capability_refresh(self, context):
        """Return one exact-account metadata read, never while send locks are held."""
        refresh = getattr(self.accounts, "refresh_image_capability", None)
        if not callable(refresh):
            return None
        with self.store.connect() as db:
            receipt = self.store.read_receipt(db, context.kind, context.owner, context.request_id) or {}
        if (receipt.get("_route") == "codex" or receipt.get("_submission_started")
                or not (context.kind == "image" or receipt.get("_operation") == "image")
                or receipt.get("_claim_id") != context.claim or receipt.get("status") != "running"
                or float(receipt.get("_claim_until") or 0) <= float(self.clock())):
            return None
        from services.owned_accounts import observed_capacity
        with self._account_guard():
            matches = [a for a in self._rows() if a.get("provider_account_identity") == receipt.get("provider_account_identity")]
            if len(matches) != 1:
                return None
            account = matches[0]
            capacity = observed_capacity(account)
            if (account.get("managed_disabled") or account.get("status") in {"禁用", "异常", "限流"}
                    or capacity["state"] != "stale" or not capacity["remaining"]
                    or account.get("source_type") not in {None, "web", "oauth_login", "password"}):
                return None
            account_ref = self.accounts.pool_account_ref(account)
        identity, binding = receipt.get("provider_account_identity"), receipt.get("provider_binding_id")

        def still_original():
            with self.store.connect() as db:
                current = self.store.read_receipt(db, context.kind, context.owner, context.request_id) or {}
                from services.generation_completion import replacement_send_allowed
                allowed = replacement_send_allowed(self.store, db, context.kind, context.owner, context.request_id, current)
                work = self.store.runtime(db, current.get("_work_key", ""), {})
                if work and (work.get("state", "active") != "active" or not work.get("slot_held")):
                    raise AdmissionLost("WORK_NOT_ACTIVE")
                if current.get("_completion_of"):
                    original = self.store.read_receipt(db, context.kind, context.owner, current["_completion_of"]) or {}
                    original_work = self.store.runtime(db, original.get("_work_key", ""), {})
                    if original_work and original_work.get("state") != "active":
                        raise AdmissionLost("WORK_NOT_ACTIVE")
            if (current.get("_claim_id") != context.claim or current.get("status") != "running"
                    or float(current.get("_claim_until") or 0) <= float(self.clock())
                    or current.get("_submission_started") or not allowed
                    or current.get("provider_account_identity") != identity or current.get("provider_binding_id") != binding):
                raise AdmissionLost("original task changed during capability read")
            if binding and self.accounts.get_bound_account_identity(binding) != identity:
                raise AdmissionLost("original image binding changed before send")

        def run(deadline):
            still_original()
            refresh(account_ref, deadline=deadline, before_read=still_original)
            still_original()
        return run

    def before_send(self, context):
        while True:
            try:
                guarded = self._before_send_guard(context)
                if guarded is _EXPIRED_UNSENT_BEFORE_SEND:
                    raise AdmissionLost("WAIT_DEADLINE_EXCEEDED")
                return guarded
            except AdmissionLost as exc:
                # A different account in this workflow may have sent after a
                # partial n-image request waited, but before its final guard.
                # Retry only this known-unsent next slot, without rerunning the
                # handler or releasing the original claim/completed results.
                if str(exc) == "SCHEDULING_SEND_INTERVAL":
                    with self.store.connect() as db:
                        r = self.store.read_receipt(db, context.kind, context.owner, context.request_id) or {}
                        now = float(self.clock())
                        sequence = int(r.get("_send_sequence") or 0)
                        partial = (r.get("_claim_id") == context.claim and r.get("status") == "running"
                                   and float(r.get("_claim_until") or 0) > now
                                   and (context.kind == "image" or r.get("_operation") == "image")
                                   and int(r.get("_expected_sends") or 1) > 1 and r.get("_submission_started")
                                   and sequence > 0 and r.get("_completed_slot") == sequence - 1
                                   and r.get("_last_sent_sequence") == sequence - 1)
                        delay = workflow_scheduling.send_delay(self.store, db, r, now) if partial else 0
                    if partial:
                        if delay:
                            time.sleep(min(1.0, delay))
                        continue  # All identity, claim and send checks run again.
                with self.store.transaction() as db:
                    r = self.store.read_receipt(db, context.kind, context.owner, context.request_id)
                    if (r and r.get("_claim_id") == context.claim and not r.get("_submission_started")
                            and r.get("status") == "running"):
                        now = float(self.clock())
                        reasons = {"SCHEDULING_NOT_BEFORE": "not_before", "SCHEDULING_SEND_INTERVAL": "send_interval",
                                   "WORK_NOT_ACTIVE": "work_not_active",
                                   "original Chat account capacity changed before send": "chat_capacity_changed"}
                        reason = reasons.get(str(exc), "send_constraints_changed")
                        from services.generation_completion import replacement_send_allowed
                        cancelled = not replacement_send_allowed(self.store, db, context.kind, context.owner, context.request_id, r)
                        diagnostic = {"original task claim expired": "claim_expired",
                                      "original image capacity decreased before send": "image_capacity_changed",
                                      "original Chat account capacity changed before send": "chat_capacity_changed",
                                      "original account is unavailable before send": "account_unavailable",
                                      "original image capability is unavailable before send": "image_capability_unavailable",
                                      "original image binding changed before send": "image_binding_changed",
                                      "SCHEDULING_NOT_BEFORE": "scheduling_changed",
                                      "SCHEDULING_SEND_INTERVAL": "scheduling_changed",
                                      "WORK_NOT_ACTIVE": "work_not_active"}.get(str(exc), "send_guard_rejected")
                        record_unsent_failure(context, r, now, "AdmissionLost", reason=diagnostic)
                        r.update(status="queued", upstream_outcome="not_sent", upstream_unfinished=False,
                                 _claim_id=None, _claim_until=0, _turn_reserved=False, _executing=False,
                                 _ready_at=now + 1, waiting={"reasons": [reason], "next_check_at": now + 1})
                        reset_unsent_image_attempt(context.kind, r)
                        if cancelled:
                            r.update(status="failed" if context.kind == "text" else "error",
                                     error_code="COMPLETION_ORIGINAL_RECOVERED", waiting=None)
                        self.store.write_receipt(db, context.kind, context.owner, context.request_id, r)
                        workflow_scheduling.release_provisional_slot(self.store, db, r)
                raise

    def _before_send_guard(self, context):
        # Resolve capability outside account/store locks, then recheck the exact
        # selected row under the send guard. Discovery failure must fail closed.
        with self.store.connect() as db:
            observed = self.store.read_receipt(db, context.kind, context.owner, context.request_id) or {}
        firstsend_trial = getattr(context, "_firstsend_model_trial", None)
        execution_model = observed.get("model", "")
        selected_route = None
        if firstsend_trial is not None:
            if (not isinstance(firstsend_trial, dict)
                    or firstsend_trial.get("original_model") != _COMBINED_COPY_MODEL
                    or firstsend_trial.get("original_thinking_effort") != _COMBINED_COPY_EFFORT
                    or firstsend_trial.get("effective_model") != _FIRSTSEND_TRIAL_MODEL
                    or firstsend_trial.get("effective_thinking_effort") != _FIRSTSEND_TRIAL_EFFORT):
                raise AdmissionLost("first-send model trial is invalid")
            prepared = getattr(context, "_prepared_firstsend_model_route", None)
            from services.model_service import model_catalog_service
            current_route = model_catalog_service.prepared_bound_model_route_at_send(
                observed.get("provider_account_identity"), _FIRSTSEND_TRIAL_MODEL)
            if prepared is None or current_route is None or prepared != current_route:
                raise AdmissionLost("first-send model capability discovery unavailable before send")
            selected_route = current_route
            execution_model = _FIRSTSEND_TRIAL_MODEL
        if (observed.get("_requested_account_identity") and context.kind == "text"
                and observed.get("_operation", "text") != "image" and observed.get("_route", "chat") == "chat"):
            if selected_route is None:
                try:
                    selected_route = self._types(execution_model)
                except Exception:
                    raise AdmissionLost("selected Chat model discovery unavailable before send") from None
        with self._settings_guard(), self.store.transaction() as db, self._account_guard():
            now = float(self.clock())
            r = self.store.read_receipt(db, context.kind, context.owner, context.request_id)
            # Preserve the former whole-queue expiry ordering for this exact
            # receipt, even if its local claim was concurrently replaced.
            if r is not None and workflow_scheduling.expire_unsent_receipt(
                    self.store, db, context.kind, context.owner, context.request_id, r, now):
                # Return through the transaction context so this terminal,
                # known-unsent state commits before the outer fencing path.
                return _EXPIRED_UNSENT_BEFORE_SEND
            if (r is None or r.get("_claim_id") != context.claim or r.get("status") != "running"
                    or float(r.get("_claim_until") or 0) <= now):
                raise AdmissionLost("original task claim expired")
            if r.get("_send_hold") is True:
                raise AdmissionLost("original task send hold is active")
            from services.generation_completion import replacement_send_allowed
            if not replacement_send_allowed(self.store, db, context.kind, context.owner, context.request_id, r):
                raise AdmissionLost("completion original changed before send")
            if r.get("_completion_of"):
                original = self.store.read_receipt(db, context.kind, context.owner, r["_completion_of"]) or {}
                original_work = self.store.runtime(db, original.get("_work_key", ""))
                if original_work and original_work.get("state") != "active":
                    raise AdmissionLost("WORK_NOT_ACTIVE")
            if context.kind == "text" and r.get("_retry_unsent_of"):
                # _run validates this relationship before binding/network I/O.
                # Re-run the same retained-input and original-eligibility
                # derivation at the irreversible send edge: P may have been
                # paused or its input/hash may have changed after that first
                # check but before this transaction.
                from services.conversation_binding_service import ConversationBindingError
                from services.text_task_service import TextTaskService
                original_id = r["_retry_unsent_of"]
                row = db.execute("SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?",
                                 (context.owner, original_id)).fetchone()
                try:
                    if not row or r.get("_retry_unsent_input_hash") != row[0]:
                        raise ValueError
                    original = json.loads(row[1])
                    retained = self.store.load_input(original["_input_ref"])
                    stored = self.store.load_input(r["_input_ref"])
                    envelope = {key: stored.get(key) for key in TextTaskService.RETRY_UNSENT_FIELDS}
                    expected = TextTaskService._unsent_retry_body_at(
                        context.owner, original_id, row[0], original, retained, envelope,
                        context.request_id, now=now,
                    )
                    if stored != expected:
                        raise ValueError
                except (OSError, ValueError, TypeError, KeyError, ConversationBindingError):
                    raise AdmissionLost("unsent retry original changed before send") from None
            requested = r.get("_requested_account_identity")
            if (r.get("_requested_account_ref") and not requested
                    or requested and r.get("provider_account_identity") != requested):
                raise AdmissionLost("requested Chat account changed before send")
            marker = r.get("_temporary_chat_second_slot")
            if marker is not None:
                settings = self._settings()
                trial = settings.get("temporary_chat_second_slot")
                if (not isinstance(marker, dict) or not isinstance(trial, dict)
                        or type(marker.get("expires_at")) is not int
                        or settings.get("revision", 0) != marker.get("revision")
                        or any(trial.get(key) != marker.get(key)
                               for key in ("account_identity", "request_id", "expires_at"))
                        or r.get("provider_account_identity") != marker.get("account_identity")
                        or context.request_id != marker.get("request_id")
                        or now >= marker.get("expires_at", 0)):
                    raise AdmissionLost("temporary second Chat turn was stopped or expired before send")
            firstsend_marker = r.get("_temporary_firstsend_model_trial")
            if firstsend_trial is not None:
                settings = self._settings()
                trial = settings.get("temporary_firstsend_model_trial")
                target = _firstsend_model_trial_target(trial, context.request_id)
                if (not isinstance(firstsend_marker, dict) or not isinstance(trial, dict)
                        or type(firstsend_marker.get("expires_at")) is not int
                        or settings.get("revision", 0) != firstsend_marker.get("revision")
                        or not isinstance(target, dict) or any(target.get(key) != firstsend_marker.get(key)
                               for key in ("account_identity", "request_id", "expires_at"))
                        or r.get("provider_account_identity") != firstsend_marker.get("account_identity")
                        or context.request_id != firstsend_marker.get("request_id")
                        or now >= firstsend_marker.get("expires_at", 0)
                        or r.get("model") != _COMBINED_COPY_MODEL
                        or r.get("_submission_started") is not False
                        or _combined_copy_send_evidence(r)):
                    raise AdmissionLost("temporary first-send model trial was stopped or expired before send")
            if context.kind == "image" and r.get("_image_thread"):
                # Only these exact same-owner originals participate in the
                # predecessor check. Keep full source bytes for its fingerprint,
                # without decoding every unrelated historical image payload.
                thread = r["_image_thread"]
                owned = {}
                for task_id in (thread.get("previous_task_id"), thread.get("edit_source_task_id")):
                    if task_id and task_id not in owned:
                        task = self.store.read_receipt(db, "image", context.owner, task_id)
                        if task:
                            owned[task_id] = task
                source = owned.get(thread.get("edit_source_task_id")) or {}
                selected_id = (source.get("_completion") or {}).get("selected_id")
                if selected_id and selected_id not in owned:
                    selected = self.store.read_receipt(db, "image", context.owner, selected_id)
                    if selected:
                        owned[selected_id] = selected
                binding, blocked = predecessor_state(r, owned)
                if blocked or any(r.get(k) != v for k, v in binding.items()):
                    raise AdmissionLost("original image thread predecessor changed before send")
            missing_node_image_id = None
            missing_node_completed_predecessors = set()
            if context.kind == "text" and r.get("_supersedes_request_id"):
                from services.text_task_service import TextTaskService
                TextTaskService._validate_supersede(self.store, db, context.owner, r, now)
                if r.get(TextTaskService.MISSING_NODE_RETRY_FIELD):
                    if not TextTaskService._missing_node_model_rejection_retry(
                            self.store, db, context.owner, r, now):
                        raise AdmissionLost("missing-node retry ordering changed before send")
                    missing_node_image_id = r[TextTaskService.MISSING_NODE_RETRY_FIELD]
                missing_attempt = TextTaskService._missing_successor_attempt(r)
                if "_missing_successor_resume" in r and missing_attempt is None:
                    raise AdmissionLost("missing-successor resend marker is invalid")
                if missing_attempt:
                    if missing_attempt["state"] != "reserved":
                        raise AdmissionLost("missing-successor resend was already consumed")
                    r["_missing_successor_resume"] = {
                        **missing_attempt, "state": "submitted", "submitted_at": now,
                    }
            if context.kind == "text" and (r.get("_codex_rejected_retry_of")
                                           or getattr(context, "expected_codex_rejected_retry", None)):
                from services.codex_text_fallback import validate_rejected_retry_before_send
                # This is evidence invalidation, not a transient capacity
                # race: durable_forward records it as failed/not_sent.  Do
                # not convert the binding error into AdmissionLost, which
                # would requeue E forever after D has a late result.
                validate_rejected_retry_before_send(self.store, db, context.owner, r)
            elif context.kind == "image":
                # A4 may send only after its exact C completed successfully.
                # The structural proof identifies the two historical text
                # barriers R/P that this one image is permitted to pass; no
                # unrelated UNKNOWN receipt is released.
                missing_node_link = _missing_node_retry_image_parent(
                    self.store, db, context.owner, context.request_id, now,
                )
                if missing_node_link:
                    parent_message_id, parent_id, root_id = missing_node_link
                    missing_node_completed_predecessors.update({parent_id, root_id})
                    saved_parent = r.get("_missing_node_retry_request_parent_message_id")
                    if saved_parent is not None and saved_parent != parent_message_id:
                        raise AdmissionLost("missing-node image continuation parent changed before send")
                    r.update(_missing_node_retry_request_parent_message_id=parent_message_id,
                             request_parent_message_id=parent_message_id)
            if context.kind == "text" and r.get("_public_session_ref") and r.get("_previous_request_id") and not r.get("_terminal_empty_correction_of") and not r.get("_same_session_retry_of"):
                previous = self.store.read_receipt(db, "text", context.owner, r["_previous_request_id"]) or {}
                if (r.get("_public_previous_waiting") or previous.get("status") != "succeeded"
                        or previous.get("upstream_outcome") == "unknown"
                        or any(r.get(key) != previous.get(key) for key in ("provider_binding_id", "provider_account_identity", "conversation_id"))
                        or r.get("_submission_parent_message_id") != previous.get("parent_message_id")):
                    raise AdmissionLost("original Chat predecessor is not confirmed before send")
            physical = physical_conversation_key(r)
            needs_image_capacity = (r.get("_route") != "codex"
                                    and (context.kind == "image" or r.get("_operation") == "image"))
            occupancy = list(self.store.receipts(db,
                statuses=("queued", "not_started", "running", "unknown", "failed", "error"),
                include_upstream_unfinished=True)) if physical or needs_image_capacity else ()
            if physical:
                for other_kind, other_owner, other_id, other in occupancy:
                    if (context.kind == "text" and other_kind == "image"
                            and other.get("id") == missing_node_image_id):
                        continue
                    if (context.kind == "image" and other_kind == "text" and other_owner == context.owner
                            and other_id in missing_node_completed_predecessors):
                        continue
                    if (other_kind != context.kind and physical_conversation_key(other) == physical
                            and unfinished(other_kind, other)
                            and (other.get("status") == "running" or unresolved_result(other_kind, other)
                                 or int(other.get("_sequence") or 0) < int(r.get("_sequence") or 0))):
                        raise AdmissionLost("original physical conversation is occupied before send")
            sequence = int(r.get("_send_sequence") or 0)
            if r.get("_submission_started") and sequence <= int(r.get("_last_sent_sequence") or 0):
                raise AdmissionLost("original model request was already submitted")
            selected_rows = [a for a in self._rows() if a.get("provider_account_identity") == r.get("provider_account_identity")]
            selected = selected_rows[0] if len(selected_rows) == 1 else None
            if requested and selected is not None and context.kind == "text" and r.get("_operation", "text") != "image" and r.get("_route", "chat") == "chat":
                if (selected_route is None
                        or self.accounts._normalize_account_type(selected.get("type")) not in getattr(selected_route, "account_types", selected_route)
                        or not self._model_account_matches(selected, selected_route)
                        or self.accounts.get_bound_account_identity(r.get("provider_binding_id")) != requested):
                    raise AdmissionLost("selected Chat model or binding changed before send")
            if selected is None or selected.get("managed_disabled") or selected.get("status") in {"禁用", "异常", "限流"}:
                raise AdmissionLost("original account is unavailable before send")
            if context.kind == "image" or r.get("_operation") == "image":
                if image_capacity(selected, self._settings(), r.get("model", "gpt-image-2")) <= 0:
                    raise AdmissionLost("original image capability is unavailable before send")
                if r.get("provider_binding_id"):
                    try:
                        bound_identity = self.accounts.get_bound_account_identity(r["provider_binding_id"])
                    except RuntimeError:
                        raise AdmissionLost("original image binding changed before send") from None
                    if bound_identity != r.get("provider_account_identity"):
                        raise AdmissionLost("original image binding changed before send")
            send_observation = {}
            if r.get("_route") == "codex":
                if self.codex is None or self.codex._eligible_account(selected, "" if context.kind == "image" or r.get("_operation") == "image" else r.get("model", ""), allow_probe=False) is None:
                    raise AdmissionLost("original Codex model is unavailable before send")
                expected = getattr(context, "expected_codex_quota_bucket", None)
                if expected is not None and self.codex.quota_decision(selected, r.get("model", ""))["quota_bucket"] != expected:
                    raise AdmissionLost("original Codex quota changed before send")
            else:
                if context.kind == "image" or r.get("_operation") == "image":
                    capacity = image_capacity(selected, self._settings(), r.get("model", "gpt-image-2"))
                    occupied = sum(1 for kind, _, _, other in occupancy
                                   if (kind == "image" or other.get("_operation") == "image")
                                   and other.get("_account_resource") == r.get("_account_resource")
                                   and image_generation_active(kind, other, other.get("status") == "running" or unresolved_result(kind, other)))
                    if capacity < occupied:
                        raise AdmissionLost("original image capacity decreased before send")
                    send_observation = {"image_capacity_at_send": capacity,
                                        "image_occupied_at_send_including_current": occupied}
                if any(isinstance(limit, dict) and limit.get("feature_name") == execution_model and limit.get("remaining") == 0
                       for limit in selected.get("limits_progress") or []):
                    raise AdmissionLost("original model quota is unavailable before send")
            if r.get("_route", "chat") == "chat":
                # A claimed independent request can wait while a stalled
                # original's next exact-ID read makes verified progress. Reuse
                # the normal snapshot inside this send transaction so that
                # progress re-holds its account slot before either a Chat text
                # or Chat image request posts. Image-specific capacity remains
                # checked independently above; this is the shared account-turn
                # fence only.
                # Do not decode completed image payloads merely to recheck a
                # Chat turn. The status/unfinished filter is the scheduler's
                # existing active-occupancy set; completed text continuations
                # still join it when they can transfer an original turn.
                capacity_receipts = list(self.store.receipts(
                    db, statuses=("queued", "not_started", "running", "unknown", "failed", "error"),
                    include_pending_completion=True, include_upstream_unfinished=True,
                ))
                seen = {(kind, owner, request_id) for kind, owner, request_id, _ in capacity_receipts}
                for owner, request_id, raw in db.execute(
                        "SELECT owner,id,receipt FROM requests WHERE json_extract(receipt,'$.status')='succeeded' "
                        "AND (json_type(receipt,'$._same_session_retry_of') IS NOT NULL "
                        "OR json_type(receipt,'$._terminal_empty_correction_of') IS NOT NULL)"):
                    key = ("text", owner, request_id)
                    if key not in seen:
                        capacity_receipts.append(("text", owner, request_id, json.loads(raw)))
                        seen.add(key)
                snapshot = self._snapshot(
                    self._rows(), capacity_receipts, self._settings(), {}, now,
                    self.store.runtime(db, "fairness", {}),
                    workflow_scheduling.state_snapshot(self.store, db),
                    product_work.state_snapshot(self.store, db), db=db,
                )
                resources = {resource.key: resource for resource in snapshot.resources}
                turn = resources.get("chat_turn:" + account_clock_key(selected))
                capacity = turn.capacity if turn is not None else 0
                # The marker was fully revalidated above. _snapshot sees this
                # request already claimed, so preserve its existing scoped
                # second-slot allowance while still rejecting a newly re-held
                # original that would make the account exceed that allowance.
                if marker is not None:
                    capacity = max(capacity, 2)
                if turn is None or turn.occupied is None or turn.occupied > capacity:
                    raise AdmissionLost("original Chat account capacity changed before send")
            # A parent always needs its durable product guard. A legacy
            # receipt can conflict with a product parent only when it has a
            # proven physical conversation identity; do not decode unrelated
            # saved image payloads for a receipt that cannot have such a match.
            product_guard_needed = bool(r.get("_product_work_key")) or bool(
                r.get("conversation_id") and (r.get("provider_account_identity") or r.get("provider_binding_id")))
            if product_guard_needed:
                from services.work_lifecycle import WorkLifecycleError
                try:
                    product_work.before_send(self.store, db, r)
                except WorkLifecycleError as exc:
                    raise AdmissionLost(exc.code) from None
            workflow_scheduling.before_send(self.store, db, r, now)
            if r.get("_supersedes_request_id"):
                r.update(upstream_outcome="unknown", error_code=None, waiting=None)
            elif (context.kind == "text" and r.get("_route", "chat") == "chat"
                  and r.get("_operation", "text") == "text" and not r.get("_forward_protocol")
                  and r.get("upstream_outcome") == "not_sent"):
                # All send guards passed. A prior unsent attempt no longer
                # describes this attempt once its durable send boundary opens.
                r["upstream_outcome"] = "unknown"
            timeline = list(r.get("_execution_timeline") or [])
            if firstsend_trial is not None:
                # Timeline is bounded. Preserve the exact send-edge decision on
                # the existing per-request marker as well, so later result
                # reads never infer an effective model from an active control.
                r["_temporary_firstsend_model_trial"] = {
                    **firstsend_marker,
                    "original_model": _COMBINED_COPY_MODEL,
                    "original_thinking_effort": _COMBINED_COPY_EFFORT,
                    "effective_model": _FIRSTSEND_TRIAL_MODEL,
                    "effective_thinking_effort": _FIRSTSEND_TRIAL_EFFORT,
                    "sent_sequence": sequence,
                    "sent_at": now,
                }
                timeline.append({"stage": "first_send_model_trial_selected", "at": now,
                                 "original_model": _COMBINED_COPY_MODEL,
                                 "original_thinking_effort": _COMBINED_COPY_EFFORT,
                                 "effective_model": _FIRSTSEND_TRIAL_MODEL,
                                 "effective_thinking_effort": _FIRSTSEND_TRIAL_EFFORT})
            timeline.append({"stage": "send_guard_passed", "at": now, **send_observation})
            r.update(_submission_started=True, _last_sent_sequence=sequence, _turn_reserved=True,
                     _claim_until=now + self.CLAIM_SECONDS, _execution_timeline=timeline[-32:])
            self.store.write_receipt(db, context.kind, context.owner, context.request_id, r)

    def _start_claim_heartbeat(self, context, done, *, name):
        """Renew a fenced claim without letting a busy writer consume its lease."""
        def heartbeat():
            interval = self.CLAIM_SECONDS / 3
            retry_delay = min(self.CLAIM_HEARTBEAT_RETRY_SECS, interval)
            while not done.wait(interval):
                while not done.is_set():
                    try:
                        self.renew_claim(context)
                        break
                    except Exception as exc:
                        # No exception message, input, credential or raw identity.
                        # A short SQLite writer collision is not a lost claim:
                        # retry only the exact BUSY/LOCKED codes. Other storage
                        # failures remain visible and cannot be spun forever.
                        logger.warning({"event": "pool_execution_stage", "stage": "claim_heartbeat_failed",
                                        "at": float(self.clock()),
                                        "request_ref": hashlib.sha256((context.owner + ":" + context.request_id).encode()).hexdigest()[:24],
                                        "error_type": ("AdmissionLost" if isinstance(exc, AdmissionLost)
                                                       else type(exc).__name__ if type(exc).__name__ in
                                                       {"OperationalError", "DatabaseError", "TimeoutError", "OSError"}
                                                       else "ClaimRefreshError")})
                        if self._sqlite_lock_contention(exc):
                            if done.wait(retry_delay):
                                return
                            continue
                        return
        heart = threading.Thread(target=heartbeat, name=name, daemon=True)
        heart.start()
        return heart

    def execute(self, context):
        done = threading.Event()
        self._start_claim_heartbeat(context, done, name="original-task-heartbeat")
        try:
            with self.store.connect() as db:
                r = self.store.read_receipt(db, context.kind, context.owner, context.request_id)
                expected_hash = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?", (context.owner, context.request_id)).fetchone()[0] if context.kind == "text" else r.get("request_hash")
            try:
                body = self.store.load_input(r["_input_ref"])
            except (OSError, ValueError, TypeError):
                self.update_claim(context, status="failed" if context.kind == "text" else "error",
                                  error_code="TASK_INPUT_UNAVAILABLE", _turn_reserved=False, upstream_unfinished=False)
                return
            if context.kind == "text":
                from services.text_task_service import TextTaskService
                _, actual_hash = TextTaskService._submission_identity(context.owner, body)
            else:
                from services.image_task_service import _request_hash
                actual_hash = _request_hash(body["mode"], body["payload"])
            if actual_hash != expected_hash:
                self.update_claim(context, status="failed" if context.kind == "text" else "error",
                                  error_code="TASK_INPUT_UNAVAILABLE", _turn_reserved=False, upstream_unfinished=False)
                return
            if context.kind == "text":
                # Keep the retained original and its request hash intact.  The
                # closed compactor returns an execution-only copy, after the
                # immutable-input check and before the existing send guard.
                if _legacy_combined_copy_scope(context, r, body) and _combined_copy_send_evidence(r):
                    # A historically inconsistent queued receipt must not send
                    # again merely because its old submission flag was lost.
                    # The exact original remains recoverable as UNKNOWN.
                    with self.store.transaction() as db:
                        current = self.store.read_receipt(db, context.kind, context.owner, context.request_id)
                        if (current and current.get("_claim_id") == context.claim and current.get("status") == "running"
                                and _legacy_combined_copy_scope(context, current, body)
                                and _combined_copy_send_evidence(current)):
                            current.update(status="unknown", error_code="CONVERSATION_OUTCOME_UNKNOWN",
                                           upstream_outcome="unknown", _submission_started=True,
                                           _turn_reserved=True, _claim_until=0)
                            self.store.write_receipt(db, context.kind, context.owner, context.request_id, current)
                    return
                frozen_body = body
                try:
                    body = _compact_queued_combined_copy(context, r, body, self.store, self.clock)
                except Exception as exc:
                    from services.conversation_binding_service import ConversationBindingError
                    if not isinstance(exc, ConversationBindingError):
                        raise
                    # A recognised legacy shape that cannot be proven safe is
                    # a deterministic pre-send failure.  Do not requeue it
                    # into an endless claim/compaction loop, and never turn a
                    # lost-claim or storage failure into a terminal result.
                    with self.store.transaction() as db:
                        current = self.store.read_receipt(db, context.kind, context.owner, context.request_id)
                        if current and current.get("_claim_id") == context.claim and current.get("status") == "running":
                            current.update(status="failed", error_code=getattr(exc, "code", None) or "CHAT_DERIVED_INPUT_INVALID",
                                           upstream_outcome="not_sent", upstream_unfinished=False,
                                           _turn_reserved=False, _claim_until=0)
                            record_unsent_failure(context, current, float(self.clock()), type(exc).__name__)
                            self.store.write_receipt(db, context.kind, context.owner, context.request_id, current)
                    return
                # A single, explicitly armed original may select the proven
                # instant execution model.  This happens only after the
                # immutable frozen-input/hash check and remains an in-memory
                # body copy; before_send revalidates the marker and prepared
                # exact-account proof immediately before the model POST.
                body = _firstsend_model_execution_body(
                    context, r, body, self.store, self._settings(), self.clock,
                    scope_body=frozen_body,
                )
                # The exact bound catalog read must complete before the
                # handler acquires its per-account HTTP send clock.  The
                # final guard consumes only the prepared in-memory route and
                # rejects it if that proof, binding or control has changed.
                self._prepare_firstsend_model_capability(context)
            missing_node_retry_parent = None
            if context.kind == "image":
                # Resolve C's output parent for the in-memory A4 execution
                # copy. The durable input remains unchanged and was hashed
                # above; this receipt metadata is rechecked by before_send at
                # the real network edge before the bound-image protocol reads
                # the conversation.
                with self.store.transaction() as db:
                    current = self.store.read_receipt(db, "image", context.owner, context.request_id)
                    if not current or current.get("_claim_id") != context.claim:
                        raise AdmissionLost("original task claim expired")
                    link = _missing_node_retry_image_parent(
                        self.store, db, context.owner, context.request_id, float(self.clock()),
                    )
                    if link:
                        missing_node_retry_parent = link[0]
                        saved_parent = current.get("_missing_node_retry_request_parent_message_id")
                        if saved_parent is not None and saved_parent != missing_node_retry_parent:
                            raise AdmissionLost("missing-node image continuation parent changed before execution")
                        current.update(_missing_node_retry_request_parent_message_id=missing_node_retry_parent,
                                       request_parent_message_id=missing_node_retry_parent)
                        self.store.write_receipt(db, "image", context.owner, context.request_id, current)
                        r.update(_missing_node_retry_request_parent_message_id=missing_node_retry_parent,
                                 request_parent_message_id=missing_node_retry_parent)
            payload = body["payload"] if context.kind == "image" else body
            if context.kind == "image" and r.get("_same_session_retry_of"):
                from services.generation_completion import same_session_retry
                with self.store.connect() as db:
                    original = self.store.read_receipt(db, "image", context.owner, r["_same_session_retry_of"])
                    previous_id = ((original or {}).get("_image_thread") or {}).get("previous_task_id")
                    previous = self.store.read_receipt(db, "image", context.owner, previous_id) if previous_id else None
                if (not same_session_retry(original, r) or original.get("result_file_ids")
                        or original.get("result_sediment_ids") or original.get("data")
                        or original.get("recovery_phase") == "download_image_result"):
                    self.update_claim(context, status="error", error_code="COMPLETION_ORIGINAL_CURSOR_UNCONFIRMED",
                                      upstream_outcome="not_sent", _turn_reserved=False, upstream_unfinished=False)
                    return
                payload["_failed_retry_original"] = original
                payload["_failed_retry_predecessor"] = previous
            payload.update({k: r[k] for k in ("provider_binding_id", "provider_account_identity", "client_conversation_id", "_requested_account_ref", "_requested_account_identity") if r.get(k)})
            if context.kind == "image" and r.get("_image_thread"):
                payload.update({k: r[k] for k in ("_image_thread", "_image_thread_predecessor_message",
                    "_image_thread_predecessor_result_ids", "conversation_id", "parent_message_id") if r.get(k)})
                self.update_claim(context, _image_thread_request_parent=r.get("parent_message_id") or None)
            if missing_node_retry_parent:
                payload["parent_message_id"] = missing_node_retry_parent
            payload["_admission_claim"] = context.claim
            payload["_request_message_id"] = r.get("request_message_id")
            with executing(context):
                self.handlers[context.kind](context, body)
        except Exception as exc:
            with self.store.transaction() as db:
                r = self.store.read_receipt(db, context.kind, context.owner, context.request_id)
                if r and r.get("_claim_id") == context.claim and r.get("status") == "running":
                    if r.get("_submission_started"):
                        r.update(status="unknown" if context.kind == "text" else "error", error_code="CONVERSATION_OUTCOME_UNKNOWN")
                    else:
                        record_unsent_failure(context, r, float(self.clock()), type(exc).__name__)
                        r.update(status="queued", _claim_id=None, _claim_until=0, _executing=False,
                                 _turn_reserved=False, upstream_unfinished=False,
                                 _ready_at=float(self.clock()) + 1)
                        reset_unsent_image_attempt(context.kind, r)
                    self.store.write_receipt(db, context.kind, context.owner, context.request_id, r)
        finally:
            done.set()
            with self.store.transaction() as db:
                r = self.store.read_receipt(db, context.kind, context.owner, context.request_id)
                if r and r.get("_claim_id") == context.claim:
                    r["_executing"] = False
                    if context.kind == "text":
                        from services.text_task_service import TextTaskService
                        TextTaskService.retire_recovered_final_after_sender_exit(r, context.claim)
                    if not r.get("_submission_started") and r.get("error_code") in {"CONVERSATION_OUTCOME_UNKNOWN", "CONVERSATION_BINDING_UNAVAILABLE", "IMAGE_RESOURCE_UNAVAILABLE"}:
                        record_unsent_failure(context, r, float(self.clock()))
                        r.update(status="queued", _claim_id=None, _claim_until=0, _executing=False,
                                 _turn_reserved=False, upstream_unfinished=False,
                                 _ready_at=float(self.clock()) + 1)
                        reset_unsent_image_attempt(context.kind, r)
                    self.store.write_receipt(db, context.kind, context.owner, context.request_id, r)
            try:
                context.record_outcome()
            except Exception:
                logger.warning({"event": "pool_outcome_unavailable", "layer": "provider"})
            self.wake()


def configure_original_task_admission():
    from services.account_service import account_service
    from services.text_task_service import text_task_service
    from services.image_task_service import image_task_service
    from services.codex_service import codex_service
    from utils.helper import is_codex_image_model
    admission = PoolAdmission(text_task_service.store, account_service, codex=codex_service)
    admission.register("text", lambda context, body: text_task_service._run(context.owner, context.request_id, body))
    def image(context, body):
        payload = body["payload"]
        payload["retain_conversation"] = not is_codex_image_model(str(payload.get("model") or "gpt-image-2"))
        image_task_service._run_task(context.owner + ":" + context.request_id, body["mode"], payload,
                                     body["identity"], str(payload.get("model") or "gpt-image-2"))
    admission.register("image", image)
    admission.recoveries["text"] = lambda owner, request_id: text_task_service.read(owner, request_id)
    admission.recoveries["image"] = lambda owner, request_id: image_task_service.resume_poll({"id": owner, "role": "user"}, request_id)
    text_task_service.admission = admission
    image_task_service.admission = admission
    from services.work_lifecycle import WorkLifecycleService
    admission.work_lifecycle = WorkLifecycleService(text_task_service, image_task_service)
    from services.generation_completion import GenerationCompletionService
    admission.generation_completion = GenerationCompletionService(text_task_service, image_task_service, admission.work_lifecycle)
    return admission
