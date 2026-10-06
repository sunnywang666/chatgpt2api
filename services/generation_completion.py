"""Bounded completion of a pure generation step, without rewriting its receipt.

An ended or freshly verified empty original branch is retried in its original
account and conversation. Its reserved ID,
original input and selected result live on the original receipt in the existing
SQLite transaction. Ordinary result reads never authorize another model send.
"""
from __future__ import annotations

import copy
import math
import time
import uuid

from services.work_lifecycle import WorkLifecycleError


class CompletionError(WorkLifecycleError):
    pass


def successful(kind, receipt):
    return bool(receipt and receipt.get("status") == ("succeeded" if kind == "text" else "success")
                and (receipt.get("content") if kind == "text" else receipt.get("data")))


def unresolved(receipt):
    return bool(receipt.get("status") == "unknown" or receipt.get("upstream_outcome") == "unknown"
                or receipt.get("upstream_unfinished") or receipt.get("error_code") == "CONVERSATION_OUTCOME_UNKNOWN")


def ended_image_edit_recheck(receipt):
    """Explicit completion recheck only; does not reopen ordinary polling."""
    state, thread = receipt.get("_completion") or {}, receipt.get("_image_thread") or {}
    return bool(receipt.get("_attempt_finished_at") and receipt.get("mode") == "edit"
                and thread.get("protocol") == "image-thread-v1" and thread.get("previous_task_id")
                and thread.get("previous_task_id") == thread.get("edit_source_task_id")
                and receipt.get("status") == "error" and receipt.get("upstream_outcome") == "unknown"
                and receipt.get("error_code") in {"RESULT_UNRECOVERABLE", "CONVERSATION_OUTCOME_UNKNOWN"}
                and state.get("allow_unconfirmed_retry") is True
                and not any(receipt.get(k) for k in ("_completion_of", "data", "result_file_ids", "result_sediment_ids",
                    "_pending_image_result_ids", "_executing", "recovery_claim_id", "_recovery_paused", "_recovery_suppressed"))
                and not state.get("replacement_id") and not state.get("selected_id"))


def retry_cursor(document, receipt, *, kind="text", now=None, predecessor=None):
    """A fresh, empty original branch; never an arbitrary latest conversation reply.

    This proves where a bounded retry can continue, NOT that an old upstream
    attempt was cancelled. Any result, later user, branch or missing history
    prevents automatic continuation.
    """
    conversation, message = receipt.get("conversation_id"), receipt.get("request_message_id")
    if (not conversation or not message or not isinstance(document, dict)
            or document.get("conversation_id", conversation) != conversation
            or document.get("is_archived") is not False):
        return None
    mapping, head = document.get("mapping"), document.get("current_node")
    if not isinstance(mapping, dict) or not isinstance(head, str) or head not in mapping:
        return None
    original = mapping.get(message)
    if kind == "image" and message not in mapping:
        from services.image_thread import absent_request_parent
        parent = absent_request_parent(document, receipt, predecessor)
        if parent:
            return {"conversation_id": conversation, "request_message_id": message,
                    "retry_parent_message_id": parent, "source": "absent_image_thread_request",
                    "observed_at": time.time() if now is None else now}
        return None
    if (not isinstance(original, dict) or not isinstance(original.get("message"), dict)
            or not isinstance(original["message"].get("author"), dict)
            or original["message"]["author"].get("role") != "user"):
        return None
    expected = receipt.get("request_parent_message_id") or receipt.get("_image_thread_request_parent")
    if expected and original.get("parent") != expected:
        return None
    children = {}
    for node_id, node in mapping.items():
        if isinstance(node, dict) and isinstance(node.get("parent"), (str, type(None))):
            children.setdefault(node.get("parent"), []).append(node_id)
    pending, seen = [message], set()
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            return None
        seen.add(node_id)
        node = mapping[node_id]
        msg = node.get("message") or {}
        if not isinstance(msg, dict) or not isinstance(msg.get("author"), dict):
            return None
        role = (msg.get("author") or {}).get("role")
        if node_id != message:
            if role not in {"assistant", "tool"}:
                return None
            content = msg.get("content") or {}
            parts = content.get("parts") if isinstance(content, dict) else None
            # Keep partial final output as evidence, never replace it as empty.
            if kind == "text" and msg.get("channel") in {None, "final"} and parts and any(parts):
                return None
            from services.openai_backend_api import OpenAIBackendAPI
            if OpenAIBackendAPI._has_image_asset_pointer({"content": content, "metadata": msg.get("metadata")}):
                return None
        descendants = children.get(node_id, [])
        if len(descendants) > 1:
            return None
        pending.extend(descendants)
    if head not in seen or children.get(head):
        return None
    proof = {"conversation_id": conversation, "request_message_id": message,
             "retry_parent_message_id": head, "observed_at": time.time() if now is None else now}
    if kind == "image":
        from services.image_task_service import _authoritative_image_failure
        if _authoritative_image_failure(document, message):
            proof["source"] = "terminal_image_failure"
    return proof


def retry_evidence(receipt):
    proof = receipt.get("_retry_cursor")
    fields = {"conversation_id", "request_message_id", "retry_parent_message_id", "observed_at"}
    if isinstance(proof, dict) and proof.get("source") in {"absent_image_thread_request", "terminal_image_failure"}:
        fields.add("source")
    if (not isinstance(proof, dict) or set(proof) != fields
            or any(not isinstance(proof.get(k), str) or not 1 <= len(proof[k]) <= 200 for k in (
                "conversation_id", "request_message_id", "retry_parent_message_id"))
            or proof["conversation_id"] != receipt.get("conversation_id")
            or proof["request_message_id"] != receipt.get("request_message_id")
            or type(proof["observed_at"]) not in {int, float} or not math.isfinite(proof["observed_at"])
            or proof["observed_at"] <= 0
            or any(not receipt.get(k) for k in ("provider_binding_id", "provider_account_identity", "client_conversation_id"))):
        return None
    return proof


def verified_image_failure(receipt, now, maximum_age):
    """A fresh strict no-image terminal read, not an error-string guess."""
    proof = retry_evidence(receipt)
    return bool(proof and proof.get("source") == "terminal_image_failure"
                and 0 <= now - proof["observed_at"] <= maximum_age
                and receipt.get("status") == "error" and receipt.get("error_code") == "NO_IMAGE_GENERATED"
                and receipt.get("upstream_unfinished") is False
                and not any(receipt.get(k) for k in ("data", "result_file_ids", "result_sediment_ids",
                    "_pending_image_result_ids", "_pending_image_output", "_recovery_paused", "_recovery_suppressed")))


def same_session_retry(root, child):
    proof = retry_evidence(root or {})
    root_id = (root or {}).get("request_id") or (root or {}).get("id")
    return bool(proof and child.get("_same_session_retry_of") == root_id
                and child.get("_completion_of") == root_id
                and (root.get("_completion") or {}).get("replacement_id") == (child.get("request_id") or child.get("id"))
                and child.get("_submission_parent_message_id") == proof["retry_parent_message_id"]
                and all(child.get(k) == root.get(k) for k in (
                    "provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id", "_work_key")))


def attach_replacement(store, db, kind, owner, request_id, payload, receipt):
    root_id = payload.get("_completion_of")
    if not root_id:
        return
    root = store.read_receipt(db, kind, owner, root_id)
    state = (root or {}).get("_completion") or {}
    if (not root or root.get("_completion_of") or state.get("replacement_id") != request_id
            or state.get("selected_id") or not state.get("prepared_input")):
        raise CompletionError("COMPLETION_REPLACEMENT_NOT_AUTHORIZED")
    prepared = store.load_input(state["prepared_input"])
    expected = prepared if kind == "text" else prepared["payload"]
    if payload != expected or kind == "image" and prepared["mode"] != receipt.get("mode"):
        raise CompletionError("COMPLETION_REPLACEMENT_INPUT_CHANGED")
    receipt["_completion_of"] = root_id
    if payload.get("_continue_after_failed_attempt"):
        proof = retry_evidence(root)
        if kind == "image" and proof and receipt.get("_image_thread"):
            # Public thread envelopes forbid caller-supplied cursors. Only this
            # exact stored server-prepared child inherits the original binding.
            if receipt.get("client_conversation_id") != root.get("client_conversation_id"):
                raise CompletionError("COMPLETION_ORIGINAL_CURSOR_UNCONFIRMED")
            receipt.update({k: root[k] for k in ("provider_binding_id", "provider_account_identity", "conversation_id")})
            receipt["parent_message_id"] = proof["retry_parent_message_id"]
        if not proof or any(receipt.get(k) != root.get(k) for k in (
                "provider_binding_id", "provider_account_identity", "conversation_id", "client_conversation_id")):
            raise CompletionError("COMPLETION_ORIGINAL_CURSOR_UNCONFIRMED")
        receipt.update(_same_session_retry_of=root_id,
                       _submission_parent_message_id=proof["retry_parent_message_id"])
        root.update(_attempt_finished_at=time.time(), _attempt_reason="ATTEMPT_REPLACED_SAME_CONVERSATION",
                    _turn_reserved=False)
        store.write_receipt(db, kind, owner, root_id, root)
    elif kind == "text" and payload.get("_continue_after_terminal_empty"):
        root.update(_attempt_finished_at=time.time(), _attempt_reason="ATTEMPT_REPLACED_SAME_CONVERSATION",
                    _turn_reserved=False)
        store.write_receipt(db, kind, owner, root_id, root)


def replacement_send_allowed(store, db, kind, owner, request_id, receipt):
    """Called in admission's send transaction; no network or additional POST."""
    root_id = receipt.get("_completion_of")
    if not root_id:
        return True
    root = store.read_receipt(db, kind, owner, root_id)
    state = (root or {}).get("_completion") or {}
    if receipt.get("_same_session_retry_of") and not same_session_retry(root, receipt):
        return False
    if kind == "image" and root and (root.get("result_file_ids") or root.get("result_sediment_ids")
                                    or root.get("data") or root.get("recovery_phase") == "download_image_result"):
        return False
    if state.get("conversation_mode") == "reconstructed":
        return False
    if kind == "text" and receipt.get("_terminal_empty_correction_of"):
        from services.text_task_service import TextTaskService
        proof = TextTaskService._verified_retryable_empty(root or {})
        if (not proof or receipt.get("_terminal_empty_correction_of") != root_id
                or receipt.get("_previous_request_id") != root_id
                or receipt.get("_submission_parent_message_id") != proof["retry_parent_message_id"]
                or any(receipt.get(k) != (root or {}).get(k) for k in (
                    "provider_account_identity", "provider_binding_id", "conversation_id", "_work_key"))):
            return False
    return bool(root and root.get("_recovery_paused") is not True
                and state.get("replacement_id") == request_id
                and not state.get("selected_id") and not successful(kind, root)
                and state.get("state") != "completed")


class GenerationCompletionService:
    STALL_SECONDS = 900
    INVESTIGATION_SECONDS = 300
    RECHECK_SECONDS = 30

    def __init__(self, text_service, image_service, lifecycle, *, clock=time.time):
        self.text, self.images, self.lifecycle = text_service, image_service, lifecycle
        self.store, self.clock = text_service.store, clock

    def _root(self, db, kind, owner, request_id):
        row = self.store.read_receipt(db, kind, owner, request_id)
        if not row:
            raise CompletionError("COMPLETION_REQUEST_NOT_FOUND", 404)
        if row.get("_completion_of"):
            raise CompletionError("COMPLETION_ATTEMPT_LIMIT")
        if kind == "text":
            pure = (row.get("route") == "chat" and row.get("_route", "chat") == "chat"
                    and row.get("_operation", "text") == "text" and not row.get("_forward_protocol"))
        else:
            pure = (row.get("retain_receipt") and row.get("model") == "gpt-image-2"
                    and row.get("mode") in {"generate", "edit"} and row.get("_route", "chat") != "codex")
        if not pure:
            raise CompletionError("COMPLETION_PURE_GENERATION_REQUIRED")
        return row

    def start(self, kind, identity, request_id, *, allow_unconfirmed_retry=False,
              retry_not_sent_failure_at=None, original_only=False):
        owner, now = str(identity["id"]), float(self.clock())
        if getattr(self.text, "admission", None) is None:
            raise CompletionError("COMPLETION_SCHEDULER_REQUIRED", 503)
        with self.store.transaction() as db:
            root = self._root(db, kind, owner, request_id)
            state = root.get("_completion")
            if original_only and (kind != "image" or allow_unconfirmed_retry
                                  or state and state.get("replacement_id")):
                raise CompletionError("COMPLETION_POLICY_CONFLICT")
            if state and state["allow_unconfirmed_retry"] != allow_unconfirmed_retry:
                if allow_unconfirmed_retry and not state.get("replacement_id") and not state.get("selected_id"):
                    state.update(allow_unconfirmed_retry=True, authorized_at=now, next_at=now)
                    self.store.write_receipt(db, kind, owner, request_id, root)
                else:
                    raise CompletionError("COMPLETION_POLICY_CONFLICT")
            if not state:
                root["_completion"] = {
                    "state": "checking_original", "started_at": now,
                    "allow_unconfirmed_retry": allow_unconfirmed_retry,
                    "max_extra_requests": 1, "next_at": now,
                }
                self.store.write_receipt(db, kind, owner, request_id, root)
            elif kind == "image" and allow_unconfirmed_retry and ended_image_edit_recheck(root):
                # An explicit recover may gather fresh evidence after a repair.
                # Keep the ended marker/history; no automatic restart on upgrade.
                state.update(state="checking_original", next_at=now)
                self.store.write_receipt(db, kind, owner, request_id, root)
            if original_only:
                # The company ingress can recover the original receipt or a
                # proven unsent attempt, but cannot authorize a successor.
                # Persist the existing request budget so background/restart
                # recovery observes the same boundary as this HTTP call.
                root["_completion"]["max_extra_requests"] = 0
                self.store.write_receipt(db, kind, owner, request_id, root)
            if retry_not_sent_failure_at is not None:
                state = root["_completion"]
                stamp = retry_not_sent_failure_at
                if (kind != "image" or not isinstance(stamp, (int, float))
                        or isinstance(stamp, bool) or not math.isfinite(stamp) or stamp <= 0):
                    raise CompletionError("COMPLETION_NOT_SENT_RETRY_INVALID")
                # A lost POST response can be retried without granting another
                # send. Explicit repair is tied to the observed failed attempt.
                if state.get("retried_not_sent_failure_at") != stamp:
                    work = self.store.runtime(db, root.get("_work_key")) if root.get("_work_key") else None
                    if (root.get("_recovery_paused") or root.get("_recovery_suppressed")
                            or work and work.get("state") != "active"):
                        raise CompletionError("COMPLETION_WORK_NOT_ACTIVE")
                    if (root.get("status") != "error" or root.get("_executing") or root.get("recovery_claim_id")
                            or root.get("upstream_outcome") not in {"not_sent", "not_submitted"}
                            or root.get("_submission_started") is not False
                            or root.get("upstream_submission_started") is not False
                            or root.get("recovery_retryable") is not True
                            or root.get("upstream_unfinished")
                            or any(root.get(k) for k in ("data", "result_file_ids", "result_sediment_ids"))
                            or any(e.get("stage") == "send_call_started" for e in root.get("_execution_timeline", []))
                            or state.get("replacement_id") or state.get("selected_id")
                            or not state.get("same_request_retry")
                            or state.get("state") != "needs_attention"
                            or (root.get("last_recovery_failure") or {}).get("at") != stamp):
                        raise CompletionError("COMPLETION_NOT_SENT_RETRY_CONFLICT")
                    self._load_verified_input(db, kind, owner, request_id, root)
                    self._queue_not_sent(root, state, now, kind)
                    state["retried_not_sent_failure_at"] = stamp
                    self.store.write_receipt(db, kind, owner, request_id, root)
        self.advance(kind, owner, request_id)
        self.text.admission.wake()
        return self.read(kind, identity, request_id)

    def _queue_not_sent(self, root, state, now, kind):
        root.setdefault("_execution_timeline", []).append({
            "stage": "not_submitted_retry_queued", "at": now,
            "previous_active_started_at": root.get("active_attempt_started_at"),
            "previous_active_deadline_at": root.get("active_attempt_deadline_at"),
            "failure": copy.deepcopy(root.get("last_recovery_failure")),
        })
        root.update(status="queued", _claim_id=None, _claim_until=0, _executing=False,
                    _turn_reserved=False, _ready_at=now, error_code=None,
                    _attempt_finished_at=None, _attempt_reason=None,
                    boot=self.text.boot if kind == "text" else root.get("boot"))
        if kind == "image":
            root.update(active_attempt_started_at=None, active_attempt_deadline_at=None)
        state.update(state="checking_original", same_request_retry=True, next_at=now)
        state.pop("reason", None)

    def _load_verified_input(self, db, kind, owner, request_id, receipt):
        try:
            body = self.store.load_input(receipt["_input_ref"])
            if kind == "text":
                expected = db.execute("SELECT request_hash FROM requests WHERE owner=? AND id=?",
                                      (owner, request_id)).fetchone()[0]
                if self.text._submission_identity(owner, body) != (request_id, expected):
                    raise ValueError()
            else:
                from services.image_task_service import _request_hash
                if _request_hash(body["mode"], body["payload"]) != receipt["request_hash"]:
                    raise ValueError()
            return body
        except (OSError, KeyError, TypeError, ValueError):
            raise CompletionError("COMPLETION_ORIGINAL_INPUT_UNAVAILABLE") from None

    def _prepare(self, db, kind, owner, request_id, root, replacement_id):
        body = copy.deepcopy(self._load_verified_input(db, kind, owner, request_id, root))
        if kind == "text":
            if self.text._verified_retryable_empty(root):
                # Reuse the proven session/physical binding through the existing
                # continuation validator. Do not reconstruct its prior context
                # or allocate a second conversation work slot.
                payload = {k: body[k] for k in ("model", "reasoning_effort", "messages", "_requested_account_ref", "_scheduling") if k in body}
                for key in ("_requested_account_ref", "_scheduling"):
                    if root.get(key) is not None:
                        payload[key] = copy.deepcopy(root[key])
                payload.update(client_request_id=replacement_id,
                               client_conversation_id=root["client_conversation_id"],
                               _public_session_ref=root["_public_session_ref"], _public_route="chat",
                               _previous_request_id=request_id, _continue_after_terminal_empty=True,
                               _text_only_binding=True, _completion_of=request_id)
                return payload
            proof = retry_evidence(root)
            if not proof or not root.get("_public_session_ref"):
                raise CompletionError("COMPLETION_ORIGINAL_CURSOR_UNCONFIRMED")
            payload = {k: body[k] for k in ("model", "reasoning_effort", "messages", "_requested_account_ref", "_scheduling") if k in body}
            payload.update(client_request_id=replacement_id,
                           client_conversation_id=root["client_conversation_id"],
                           _public_session_ref=root["_public_session_ref"], _public_route="chat",
                           _previous_request_id=request_id, _continue_after_failed_attempt=True,
                           _text_only_binding=True, _completion_of=request_id)
            return payload
        payload = body["payload"]
        if root.get("recovery_phase") == "download_image_result" or root.get("result_file_ids") or root.get("result_sediment_ids") or root.get("data"):
            raise CompletionError("COMPLETION_DOWNLOAD_ORIGINAL_RESULT")
        proof = retry_evidence(root)
        if not proof or root.get("recovery_requires_new_conversation"):
            raise CompletionError("COMPLETION_ORIGINAL_CURSOR_UNCONFIRMED")
        if proof.get("source") == "absent_image_thread_request":
            from services.image_task_service import UNRECOVERABLE_QUALIFIED_READS
            if (root.get("status") != "error" or root.get("error_code") != "RESULT_UNRECOVERABLE"
                    or root.get("upstream_outcome") != "unknown" or root.get("upstream_unfinished") is not False
                    or int(root.get("recovery_no_result_reads") or 0) < UNRECOVERABLE_QUALIFIED_READS
                    or root.get("_pending_image_result_ids")):
                raise CompletionError("COMPLETION_ORIGINAL_CURSOR_UNCONFIRMED")
        if int(payload.get("n") or 1) != 1 or payload.get("upstream_model"):
            raise CompletionError("COMPLETION_SINGLE_OUTPUT_REQUIRED")
        for key in ("provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id"):
            payload[key] = root[key]
        for key in ("_requested_account_ref", "_scheduling"):
            if root.get(key) is not None:
                payload[key] = copy.deepcopy(root[key])
        if root.get("_image_thread"):
            payload["image_thread_id"] = root["_image_thread"]["id"]
        payload.update(parent_message_id=proof["retry_parent_message_id"],
                       _continue_after_failed_attempt=True, _completion_of=request_id)
        if root.get("_image_thread"):
            for key in ("provider_binding_id", "provider_account_identity", "client_conversation_id", "conversation_id", "parent_message_id", "retain_conversation"):
                payload.pop(key, None)
        return body

    @staticmethod
    def _age_anchor(root):
        # Upgrades and recovery reads cannot reset the original request's age.
        sent = [item.get("at") for item in root.get("_execution_timeline", [])
                if isinstance(item, dict) and item.get("stage") == "send_call_started"
                and type(item.get("at")) in {int, float}]
        created = root.get("created_ts", root.get("created_at"))
        anchor = min(sent) if sent else created if type(created) in {int, float} else None
        progress = root.get("_result_last_progress_at")
        return max(anchor, progress) if anchor is not None and type(progress) in {int, float} else anchor

    def _select(self, db, kind, owner, request_id, root):
        state = root["_completion"]
        child_id = state.get("replacement_id")
        child = self.store.read_receipt(db, kind, owner, child_id) if child_id else None
        if not state.get("selected_id"):
            # Selection is write-once. A late original after this point stays
            # readable at its own ID and cannot cause another downstream save.
            # Once the replacement crossed the send boundary it owns this
            # completion. A late root cannot strand that in-flight request.
            original_selectable = not child or not child.get("_submission_started")
            chosen = (request_id if original_selectable and successful(kind, root)
                      else child_id if successful(kind, child) else None)
            if chosen:
                state.update(selected_id=chosen, selected_at=float(self.clock()), state="result_ready", next_at=None)
                state.pop("reason", None)
        if (child and state.get("selected_id") == request_id and not child.get("_submission_started")
                and (child.get("status") in {"queued", "running"}
                     or child.get("status") in {"failed", "error"} and child.get("upstream_outcome") == "not_sent")):
            child.update(status="failed" if kind == "text" else "error", error_code="COMPLETION_ORIGINAL_RECOVERED",
                         upstream_outcome="not_sent", upstream_unfinished=False, _turn_reserved=False,
                         _claim_id=None, _claim_until=0, _executing=False)
            self.store.write_receipt(db, kind, owner, child_id, child)
            from services.workflow_scheduling import release_provisional_slot
            release_provisional_slot(self.store, db, child)
        if (child and state.get("selected_id") == request_id
                and child.get("error_code") == "COMPLETION_ORIGINAL_RECOVERED"
                and child.get("upstream_outcome") == "not_sent" and not child.get("_submission_started")):
            if child.get("_work_key") and child.get("_work_key") == root.get("_work_key"):
                work = self.store.runtime(db, root["_work_key"])
                if work and work["last_request_id"] == child_id:
                    work["last_request_id"] = request_id
                    self.store.set_runtime(db, work["key"], work)
        return child

    def _release_ended_work(self, db, receipt):
        work_key = receipt.get("_work_key")
        if not work_key:
            return
        members = [r for _, _, _, r in self.store.receipts(db, work_key=work_key) if r.get("_work_key") == work_key]
        if any(r.get("_executing") or r.get("recovery_claim_id")
               or not r.get("_attempt_finished_at") and (unresolved(r) or r.get("status") in {"queued", "running"}) for r in members):
            return
        work = self.store.runtime(db, work_key)
        if work:
            work["slot_held"] = False
            self.store.set_runtime(db, work_key, work)

    def advance(self, kind, owner, request_id, *, wait_for_image_recovery=False):
        now = float(self.clock())
        with self.store.transaction() as db:
            root = self._root(db, kind, owner, request_id)
            state = root.get("_completion")
            if not state or state.get("state") == "completed" or root.get("_recovery_paused") is True or root.get("_recovery_suppressed"):
                return
            child = self._select(db, kind, owner, request_id, root)
            if state.get("selected_id"):
                self.store.write_receipt(db, kind, owner, request_id, root)
                return
            # Lease the orchestration attempt; actual model admission still uses
            # the normal durable claim. Repeated API calls cannot bypass cooldown.
            if float(state.get("next_at") or 0) > now:
                return
            if state.get("state") == "needs_attention" and state.get("next_at") is None:
                return
            state["next_at"] = now + self.RECHECK_SECONDS
            self.store.write_receipt(db, kind, owner, request_id, root)
        # Existing exact-original readers preserve Retry-After/backoff and errors.
        if kind == "text":
            self.text.read(owner, request_id)
        elif (root.get("status") == "error" and unresolved(root)
              and (root.get("conversation_id") or self.images.can_locate_original_cursor(root))
              and now >= float(root.get("next_poll_at") or 0)
              and now - float(root.get("_completion_read_at") or 0) > self.INVESTIGATION_SECONDS):
            self.images.resume_poll({"id": owner}, request_id, extra_timeout_secs=5,
                                    allow_unrecoverable_retry=True, completion_recheck=True,
                                    **({"wait_for_completion": True} if wait_for_image_recovery else {}))
        with self.store.transaction() as db:
            root = self._root(db, kind, owner, request_id)
            state = root["_completion"]
            child = self._select(db, kind, owner, request_id, root)
            if not state.get("selected_id") and not child:
                try:
                    active = root.get("_executing") or root.get("recovery_claim_id") or root.get("status") in {"queued", "running", "not_started"}
                    if root.get("_recovery_paused") is True:
                        raise CompletionError("COMPLETION_ORIGINAL_RECOVERY_PAUSED")
                    if active:
                        raise CompletionError("COMPLETION_ORIGINAL_ACTIVE")
                    work = self.store.runtime(db, root.get("_work_key")) if root.get("_work_key") else None
                    if work and work.get("state") != "active":
                        raise CompletionError("COMPLETION_WORK_NOT_ACTIVE")
                    not_sent = (root.get("upstream_outcome") in {"not_sent", "not_submitted"}
                                and root.get("_submission_started") is not True
                                and root.get("upstream_submission_started") is not True)
                    if not_sent:
                        if (state.get("same_request_retry") or not (root.get("error_code") in {
                                "TEXT_TASK_CAPACITY_EXCEEDED", "IMAGE_GENERATION_NOT_SUBMITTED",
                                "CONVERSATION_BINDING_UNAVAILABLE", "CHAT_ARCHIVE_RESTORE_UNCONFIRMED"}
                                or kind == "image" and root.get("error_code") == "RESULT_UNRECOVERABLE"
                                and root.get("recovery_retryable") is True)):
                            raise CompletionError("COMPLETION_ORIGINAL_NOT_RETRYABLE")
                        self._queue_not_sent(root, state, now, kind)
                        self.store.write_receipt(db, kind, owner, request_id, root)
                        self.text.admission.wake()
                        return
                    if kind == "image" and unresolved(root) and not root.get("conversation_id"):
                        if root.get("recovery_error_code") == "RECOVERY_AUTH_REQUIRED":
                            raise CompletionError("COMPLETION_ORIGINAL_READ_UNAVAILABLE")
                        if self.images.can_locate_original_cursor(root):
                            # The bounded scan may need several windows. A
                            # missing final cursor during its own cooldown is
                            # not permission to stop scanning or generate again.
                            state.update(state="checking_original", reason="COMPLETION_INVESTIGATING_ORIGINAL",
                                         next_at=max(now + self.RECHECK_SECONDS, float(root.get("next_poll_at") or 0)))
                            self.store.write_receipt(db, kind, owner, request_id, root)
                            return
                        # A send can disconnect before its first cursor arrives.
                        # resume_poll cannot query that original without a cursor;
                        # repeating its local ValueError is not an investigation.
                        raise CompletionError("COMPLETION_ORIGINAL_CURSOR_UNAVAILABLE")
                    ended = (self.text._verified_retryable_empty(root) if kind == "text"
                             else verified_image_failure(root, now, self.INVESTIGATION_SECONDS))
                    retry_authorized = state["allow_unconfirmed_retry"] or state.get("automatic_failure_retry")
                    if kind == "text" and root.get("conversation_id") and not ended and not retry_authorized:
                        raise CompletionError("COMPLETION_ORIGINAL_END_UNCONFIRMED")
                    unknown = unresolved(root) and not ended
                    if unknown:
                        if not retry_authorized:
                            raise CompletionError("COMPLETION_ORIGINAL_END_UNCONFIRMED")
                        anchor = self._age_anchor(root)
                        if anchor is None or now < anchor + self.STALL_SECONDS + self.INVESTIGATION_SECONDS:
                            raise CompletionError("COMPLETION_INVESTIGATING_ORIGINAL")
                        if root.get("_recovery_suppressed"):
                            raise CompletionError("COMPLETION_ORIGINAL_RECOVERY_PAUSED")
                        if root.get("recovery_error_code") in {"RECOVERY_RATE_LIMITED", "RECOVERY_AUTH_REQUIRED"}:
                            raise CompletionError("COMPLETION_ORIGINAL_READ_UNAVAILABLE")
                        if kind == "text" and (root.get("recovery_error_code") != "UPSTREAM_OUTCOME_UNKNOWN"
                                or now - float(root.get("_result_last_checked_at") or 0) > self.INVESTIGATION_SECONDS):
                            raise CompletionError("COMPLETION_ORIGINAL_READ_UNAVAILABLE")
                        if kind == "image" and now - float(root.get("_completion_read_at") or 0) > self.INVESTIGATION_SECONDS:
                            raise CompletionError("COMPLETION_ORIGINAL_READ_UNAVAILABLE")
                    elif not ended and root.get("status") not in {"failed", "error"}:
                        raise CompletionError("COMPLETION_ORIGINAL_NOT_TERMINAL")
                    if not state.get("replacement_id"):
                        if state.get("max_extra_requests") == 0:
                            raise CompletionError("COMPLETION_ORIGINAL_ONLY")
                        replacement_id = "completion-" + uuid.uuid4().hex
                        prepared = self._prepare(db, kind, owner, request_id, root, replacement_id)
                        state.update(replacement_id=replacement_id, prepared_input=self.store.save_input(prepared),
                                     conversation_mode="original", state="replacement_pending")
                    state.pop("reason", None)
                except CompletionError as exc:
                    state.update(state="needs_attention" if exc.code not in {
                        "COMPLETION_ORIGINAL_ACTIVE", "COMPLETION_INVESTIGATING_ORIGINAL"} else "checking_original", reason=exc.code)
                    if state["state"] == "needs_attention":
                        paused = exc.code in {"COMPLETION_WORK_NOT_ACTIVE", "COMPLETION_ORIGINAL_RECOVERY_PAUSED"}
                        state["next_at"] = now + self.RECHECK_SECONDS if paused else None
                        # End automatic investigation, not the upstream fact.
                        definitely_unsent = root.get("upstream_outcome") in {"not_sent", "not_submitted"} and root.get("_submission_started") is not True
                        if not active and (definitely_unsent or (state.get("allow_unconfirmed_retry") or state.get("automatic_failure_retry")) and unresolved(root)) and exc.code not in {
                                "COMPLETION_WORK_NOT_ACTIVE", "COMPLETION_ORIGINAL_RECOVERY_PAUSED",
                                "COMPLETION_DOWNLOAD_ORIGINAL_RESULT"}:
                            root.update(_attempt_finished_at=now, _attempt_reason=exc.code, _turn_reserved=False)
            elif child and not state.get("selected_id"):
                child_anchor = self._age_anchor(child)
                child_pending = (child.get("status") in {"queued", "running"}
                    or unresolved(child) and child_anchor is not None
                    and now < child_anchor + self.STALL_SECONDS + self.INVESTIGATION_SECONDS
                    and not (kind == "text" and self.text._verified_retryable_empty(child)))
                state.update(state="replacement_pending" if child_pending else "needs_attention",
                             reason="COMPLETION_ATTEMPT_PENDING" if child_pending else "COMPLETION_ATTEMPT_EXHAUSTED")
                if (kind == "text" and child.get("_completion_of") == request_id
                        and child.get("_terminal_empty_correction_of") == request_id
                        and all(child.get(k) == root.get(k) for k in (
                            "provider_account_identity", "provider_binding_id", "conversation_id", "_work_key"))
                        and self.text._verified_retryable_empty(child)):
                    # The bounded empty attempt is over locally. Keep UNKNOWN
                    # and conversation ordering, but release its physical slot.
                    child["_turn_reserved"] = False
                    self.store.write_receipt(db, kind, owner, state["replacement_id"], child)
                original_work = self.store.runtime(db, root.get("_work_key", "")) or {}
                if root.get("_recovery_suppressed") or original_work.get("state", "active") != "active":
                    state.update(state="needs_attention", reason="COMPLETION_WORK_NOT_ACTIVE")
                if state["state"] == "needs_attention":
                    paused = state.get("reason") == "COMPLETION_WORK_NOT_ACTIVE"
                    state["next_at"] = now + self.RECHECK_SECONDS if paused else None
                    if not paused and not child.get("_executing") and not child.get("recovery_claim_id"):
                        child.update(_attempt_finished_at=now, _attempt_reason="COMPLETION_ATTEMPT_EXHAUSTED", _turn_reserved=False)
                        root.update(_attempt_finished_at=now, _attempt_reason="COMPLETION_ATTEMPT_EXHAUSTED", _turn_reserved=False)
                        self.store.write_receipt(db, kind, owner, state["replacement_id"], child)
            self.store.write_receipt(db, kind, owner, request_id, root)
            if state.get("state") == "needs_attention" and root.get("_attempt_finished_at"):
                self._release_ended_work(db, root)
            submit = bool(state.get("replacement_id") and not child and not state.get("selected_id")
                          and root.get("_recovery_paused") is not True)
            prepared_ref, replacement_id = state.get("prepared_input"), state.get("replacement_id")
        if submit:
            body = self.store.load_input(prepared_ref)
            # Crash between reservation and submit resumes this same ID/input.
            if kind == "text":
                self.text.submit(owner, body, source=root.get("_source"))
            else:
                self.images._submit(body["identity"], client_task_id=replacement_id, mode=body["mode"], payload=body["payload"])

    def process_one(self, *, dispatch=None):
        # Only newly accepted pure-generation requests opt into the automatic
        # policy. Upgrading never silently replays historical UNKNOWN receipts.
        with self.store.transaction() as db:
            for kind, owner, rid, row in self.store.receipts(
                    db, statuses=("unknown", "failed", "error"), include_pending_completion=True):
                if (row.get("_automatic_generation_recovery") and not row.get("_completion")
                        and not row.get("_completion_of") and row.get("status") in {"unknown", "failed", "error"}
                        and not row.get("_recovery_paused") and not row.get("_recovery_suppressed")):
                    row["_completion"] = {"state": "checking_original", "started_at": float(self.clock()),
                        "allow_unconfirmed_retry": False, "automatic_failure_retry": True,
                        "max_extra_requests": 1, "next_at": float(self.clock())}
                    self.store.write_receipt(db, kind, owner, rid, row)
                state = row.get("_completion") or {}
                if (state.get("state") not in {None, "completed", "result_ready"}
                        and not row.get("_recovery_paused") and not row.get("_recovery_suppressed")):
                    # Original-result recovery can finish after automatic
                    # investigation has ended or while its retry timer is still
                    # pending. Settle saved evidence locally without waiting;
                    # do not reopen reads/retries or starve due work with a late
                    # original whose already-sent replacement still owns it.
                    self._select(db, kind, owner, rid, row)
                    if state.get("selected_id"):
                        self.store.write_receipt(db, kind, owner, rid, row)
        with self.store.connect() as db:
            candidates = [(kind, owner, rid, row) for kind, owner, rid, row in self.store.receipts(
                              db, statuses=(), include_pending_completion=True)
                          if row.get("_completion", {}).get("state") not in {None, "completed", "result_ready"}
                          and not row.get("_recovery_paused") and not row.get("_recovery_suppressed")
                          and row.get("_completion", {}).get("next_at") is not None
                          and float(row["_completion"].get("next_at") or 0) <= float(self.clock())]
        if candidates:
            if dispatch is None:
                self.advance(*candidates[0][:3])
            else:
                for kind, owner, rid, row in candidates:
                    dispatch(kind, owner, rid, lambda o, r, k=kind: self.advance(
                        k, o, r, wait_for_image_recovery=True), row)

    def read(self, kind, identity, request_id):
        owner = str(identity["id"])
        with self.store.transaction() as db:
            root = self._root(db, kind, owner, request_id)
            state = root.get("_completion")
            if not state:
                raise CompletionError("COMPLETION_NOT_REQUESTED", 404)
            child = self._select(db, kind, owner, request_id, root)
            self.store.write_receipt(db, kind, owner, request_id, root)
            selected = root if state.get("selected_id") == request_id else child if state.get("selected_id") else None
            ended = kind == "text" and bool(self.text._verified_terminal_empty(root))
            result = {"protocol": "generation-completion-v1", "kind": kind, "original_id": request_id,
                      **{k: state[k] for k in ("state", "reason", "started_at", "next_at", "replacement_id", "selected_id",
                                              "selected_at", "max_extra_requests", "conversation_mode", "results_saved", "work") if k in state},
                      "original_status": root.get("status"), "replacement_status": (child or {}).get("status"),
                      "stop": {"capability": "unsupported", "confirmed": False},
                      "original_turn_ended": bool(ended or successful(kind, root)),
                      "original_attempt_state": "ended" if root.get("_attempt_finished_at") else "active",
                      "empty_response_confirmed": bool(kind == "text" and self.text._verified_retryable_empty(root)),
                      "original_cleanup": "pending" if unresolved(root) and not ended else "not_required"}
            if state.get("state") in {"result_ready", "completed"}:
                result.pop("reason", None)
            same_retry = bool(child and kind == "text" and child.get("_terminal_empty_correction_of") == request_id
                              and child.get("_work_key") == root.get("_work_key")
                              and self.text._verified_retryable_empty(root))
            same_retry = same_retry or bool(child and same_session_retry(root, child))
            local_receipt = child or root
            empty_released = (kind == "text" and local_receipt.get("_turn_reserved") is False
                              and self.text._verified_retryable_empty(local_receipt))
            result["local_reservation"] = ("released" if local_receipt.get("_attempt_finished_at") or empty_released or same_retry and successful(kind, child)
                                            else "transferred" if same_retry and child.get("status") in {"queued", "running", "unknown"}
                                            else "held" if unresolved(root) and not ended else "released")
            if same_retry and not ended and successful(kind, child) and state.get("state") == "completed":
                result["original_cleanup"] = "completed"
            if child and state.get("selected_id") == request_id:
                result["replacement_cleanup"] = "pending" if unresolved(child) or child.get("status") == "running" else "not_required"
            if child and child.get("waiting"):
                result["waiting"] = {key: child["waiting"][key] for key in ("reasons", "next_check_at") if key in child["waiting"]}
            if root.get("_work_key"):
                from services.work_lifecycle import _projection
                original_work = self.store.runtime(db, root["_work_key"])
                if original_work:
                    result["original_work"] = _projection(original_work)
                    if original_work.get("cleanup_pending"):
                        result["original_cleanup"] = "pending"
            if selected and selected.get("_work_key"):
                from services.work_lifecycle import _projection
                work = self.store.runtime(db, selected["_work_key"])
                if work:
                    result["work"] = _projection(work)
            if selected and kind == "text":
                from services.public_chat_service import project_public_chat_receipt
                result["result"] = project_public_chat_receipt(self.text._public(selected))
            elif selected:
                # The caller downloads from the selected ID through the existing
                # authenticated image route; private URLs/bindings stay private.
                result["result"] = {"id": state["selected_id"], "status": selected["status"], "image_count": len(selected.get("data") or [])}
            return result

    def rework(self, kind, identity, request_id, selected_id):
        current = self.read(kind, identity, request_id)
        if not selected_id or current.get("selected_id") != selected_id or current["state"] != "completed":
            raise CompletionError("COMPLETION_SELECTED_RESULT_MISMATCH")
        work = self.lifecycle.update(kind, identity, selected_id, "active")
        return {**current, "work": work}

    def complete(self, kind, identity, request_id, selected_id):
        current = self.read(kind, identity, request_id)
        if not selected_id or current.get("selected_id") != selected_id:
            raise CompletionError("COMPLETION_SELECTED_RESULT_MISMATCH")
        # This is the caller's explicit post-save/review acknowledgement. The
        # unknown original remains protected; only the chosen conversation closes.
        work = self.lifecycle.update(kind, identity, selected_id, "completed", results_saved=True)
        with self.store.transaction() as db:
            root = self._root(db, kind, str(identity["id"]), request_id)
            unused_id = request_id if selected_id != request_id else root["_completion"].get("replacement_id")
            unused = self.store.read_receipt(db, kind, str(identity["id"]), unused_id) if unused_id else None
            chosen = self.store.read_receipt(db, kind, str(identity["id"]), selected_id)
            if unused and unused.get("_work_key") and unused.get("_work_key") != (chosen or {}).get("_work_key"):
                # Business completion is linked to the selected saved result.
                # Keep the old physical work's slot until its own cleanup is
                # confirmed; archive processing already refuses UNKNOWN turns.
                from services.work_lifecycle import save_work
                original_work = self.store.runtime(db, unused["_work_key"])
                if original_work and not original_work.get("completion_result_id"):
                    pending = unresolved(unused) or unused.get("status") == "running"
                    original_work.update(state="completed", results_saved=True, cleanup_pending=pending,
                                         completion_result_id=selected_id, version=original_work["version"] + 1)
                    if not pending:
                        original_work["slot_held"] = False
                    self.lifecycle._intent(original_work, unused, desired=True, now=float(self.clock()),
                                           never_sent=not unused.get("_submission_started") and not unused.get("conversation_id"))
                    save_work(self.store, db, original_work)
            root["_completion"].update(state="completed", results_saved=True, work=work, next_at=None)
            root["_completion"].pop("reason", None)
            self.store.write_receipt(db, kind, str(identity["id"]), request_id, root)
        return self.read(kind, identity, request_id)


def get_generation_completion_service():
    from services.text_task_service import text_task_service
    from services.image_task_service import image_task_service
    from services.work_lifecycle import get_work_lifecycle_service
    return GenerationCompletionService(text_task_service, image_task_service, get_work_lifecycle_service())
