"""Proof helpers for the one explicit image-quota failover attempt.

This is deliberately not a scheduler: it only recognizes the durable receipt
relationship created by ``GenerationCompletionService``.  In particular, an
ordinary ``NO_IMAGE_GENERATED`` response, an UNKNOWN result, or a caller-made
child is never a failover candidate.
"""
from __future__ import annotations

import copy


FAILURE_REASON = "image_generation_quota_exhausted"
PROTOCOL = "image-quota-failover-v1"


def _id(receipt):
    return (receipt or {}).get("request_id") or (receipt or {}).get("id")


def eligible_original(receipt):
    """Return whether this is the exact, already-observed quota terminal."""
    return bool(
        isinstance(receipt, dict)
        and receipt.get("status") == "error"
        and receipt.get("error_code") == "NO_IMAGE_GENERATED"
        and receipt.get("upstream_outcome") == "failed"
        and receipt.get("upstream_unfinished") is False
        and receipt.get("_authoritative_failure_reason") == FAILURE_REASON
        and receipt.get("_submission_started") is True
        and not any(receipt.get(key) for key in (
            "data", "result_file_ids", "result_sediment_ids", "_pending_image_result_ids",
            "_pending_image_output", "recovery_claim_id", "_recovery_claim_id",
            "_executing", "_turn_reserved", "_recovery_paused", "_recovery_suppressed",
        ))
        and all(isinstance(receipt.get(key), str) and receipt[key] for key in (
            "provider_binding_id", "provider_account_identity",
            "client_conversation_id", "conversation_id", "_product_work_key",
        ))
    )


def prepared_payload(root_payload, root, replacement_id):
    """Derive the one unbound, fresh-conversation input from the saved root."""
    if not isinstance(root_payload, dict) or not eligible_original(root):
        raise ValueError("quota failover original is not eligible")
    payload = copy.deepcopy(root_payload)
    for key in (
        "provider_binding_id", "provider_account_identity", "client_conversation_id",
        "conversation_id", "parent_message_id", "image_thread_id", "_requested_account_ref",
    ):
        payload.pop(key, None)
    # ProductWork owns the new session until it is explicitly closed.  This is
    # normally added by ImageTaskService; keeping it in the frozen input makes
    # the completion hand-off byte-for-byte verifiable before that service runs.
    payload["retain_conversation"] = True
    payload.update(
        _completion_of=_id(root),
        _image_quota_failover={
            "protocol": PROTOCOL,
            "original_request_id": _id(root),
            "original_request_hash": root["request_hash"],
            "excluded_account_identity": root["provider_account_identity"],
            "replacement_id": replacement_id,
        },
    )
    return payload


def marker(root, child):
    """Read the private marker only after checking the exact parent relation."""
    value = (child or {}).get("_image_quota_failover")
    if not isinstance(value, dict) or set(value) != {
        "protocol", "original_request_id", "original_request_hash",
        "excluded_account_identity", "replacement_id",
    }:
        return None
    if not eligible_original(root):
        return None
    if (value.get("protocol") != PROTOCOL
            or value.get("original_request_id") != _id(root)
            or value.get("original_request_hash") != root.get("request_hash")
            or value.get("excluded_account_identity") != root.get("provider_account_identity")
            or value.get("replacement_id") != _id(child)):
        return None
    return value


def is_successor(root, child, *, after_send=False):
    """Validate the one child without treating a new session as same-session."""
    if not registration_successor(root, child):
        return False
    if any(child.get(key) != root.get(key) for key in ("owner_id", "_product_work_key", "_source")):
        return False
    if child.get("_product_work") != root.get("_product_work"):
        return False
    if after_send:
        value = marker(root, child)
        return bool(
            child.get("_submission_started") is True
            and child.get("provider_account_identity")
            and child.get("provider_account_identity") != value["excluded_account_identity"]
            and child.get("provider_binding_id")
            and child.get("client_conversation_id")
            and child.get("client_conversation_id") != root.get("client_conversation_id")
            and child.get("conversation_id")
            and child.get("conversation_id") != root.get("conversation_id")
        )
    return True


def registration_successor(root, child):
    """The pre-admission relation accepted while the child is still unbound."""
    value = marker(root, child)
    if not value:
        return False
    state = root.get("_completion") or {}
    if (child.get("_completion_of") != _id(root)
            or state.get("quota_failover") is not True
            or state.get("replacement_id") != _id(child)
            or state.get("selected_id") not in {None, _id(child)}
            or state.get("max_extra_requests") != 1):
        return False
    return True


def excluded_account(receipt):
    """The scheduler-facing exclusion, valid only for an unsent trusted child."""
    value = (receipt or {}).get("_image_quota_failover")
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL:
        return None
    excluded = value.get("excluded_account_identity")
    return excluded if isinstance(excluded, str) and excluded else None
