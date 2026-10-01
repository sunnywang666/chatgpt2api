"""Bounded completion of a pure generation step, without rewriting its receipt.

An ended original turn is retried in its original conversation first. An
explicitly authorized unresolved replacement may use a new conversation. Its reserved ID,
original input and selected result live on the original receipt in the existing
SQLite transaction. Ordinary result reads never authorize another model send.
"""
from __future__ import annotations

import copy
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


def replacement_send_allowed(store, db, kind, owner, request_id, receipt):
    """Called in admission's send transaction; no network or additional POST."""
    root_id = receipt.get("_completion_of")
    if not root_id:
        return True
    root = store.read_receipt(db, kind, owner, root_id)
    state = (root or {}).get("_completion") or {}
    if kind == "text" and receipt.get("_terminal_empty_correction_of"):
        from services.text_task_service import TextTaskService
        proof = TextTaskService._verified_retryable_empty(root or {})
        if (not proof or receipt.get("_terminal_empty_correction_of") != root_id
                or receipt.get("_previous_request_id") != root_id
                or receipt.get("_submission_parent_message_id") != proof["retry_parent_message_id"]
                or any(receipt.get(k) != (root or {}).get(k) for k in (
                    "provider_account_identity", "provider_binding_id", "conversation_id", "_work_key"))):
            return False
    return bool(root and state.get("replacement_id") == request_id
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

    def start(self, kind, identity, request_id, *, allow_unconfirmed_retry=False):
        owner, now = str(identity["id"]), float(self.clock())
        if getattr(self.text, "admission", None) is None:
            raise CompletionError("COMPLETION_SCHEDULER_REQUIRED", 503)
        with self.store.transaction() as db:
            root = self._root(db, kind, owner, request_id)
            state = root.get("_completion")
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
        self.advance(kind, owner, request_id)
        self.text.admission.wake()
        return self.read(kind, identity, request_id)

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
            messages, current, seen = copy.deepcopy(body["messages"]), root, {request_id}
            while current.get("_previous_request_id"):
                previous_id = current["_previous_request_id"]
                if previous_id in seen or len(seen) >= 100:
                    raise CompletionError("COMPLETION_CONTEXT_UNAVAILABLE")
                seen.add(previous_id)
                previous = self.store.read_receipt(db, kind, owner, previous_id)
                if not successful(kind, previous):
                    raise CompletionError("COMPLETION_CONTEXT_UNAVAILABLE")
                prior = self._load_verified_input(db, kind, owner, previous_id, previous)
                messages = prior["messages"] + [{"role": "assistant", "content": previous["content"]}] + messages
                current = previous
            if current.get("_legacy_session_anchor") or len(messages) > 100:
                raise CompletionError("COMPLETION_CONTEXT_UNAVAILABLE")
            # Public text requests have no caller-provided external tools. Keep
            # model, reasoning, account_ref and scheduling; discard old cursors.
            payload = {k: body[k] for k in ("model", "reasoning_effort", "_requested_account_ref", "_scheduling") if k in body}
            for key in ("_requested_account_ref", "_scheduling"):
                if root.get(key) is not None:
                    payload[key] = copy.deepcopy(root[key])
            payload.update(client_request_id=replacement_id, messages=messages,
                           client_conversation_id="public-" + replacement_id,
                           _public_session_ref=replacement_id, _public_route="chat",
                           _text_only_binding=True, _completion_of=request_id)
            return payload
        payload = body["payload"]
        for key in ("_requested_account_ref", "_scheduling"):
            if root.get(key) is not None:
                payload[key] = copy.deepcopy(root[key])
        # Existing images/download evidence is recovered, never regenerated.
        if root.get("recovery_phase") == "download_image_result" or root.get("result_file_ids") or root.get("result_sediment_ids") or root.get("data"):
            raise CompletionError("COMPLETION_DOWNLOAD_ORIGINAL_RESULT")
        # A thread's saved edit image may be an exact complete input. A generate
        # continuation, however, depends on upstream context absent from input.
        if root.get("_image_thread") and (body["mode"] != "edit" or not payload.get("images")):
            if (root["_image_thread"].get("previous_task_id") or payload.get("conversation_id")):
                raise CompletionError("COMPLETION_CONTEXT_UNAVAILABLE")
        if body["mode"] == "edit" and not payload.get("images"):
            raise CompletionError("COMPLETION_CONTEXT_UNAVAILABLE")
        if int(payload.get("n") or 1) != 1:
            raise CompletionError("COMPLETION_SINGLE_OUTPUT_REQUIRED")
        for key in ("provider_binding_id", "provider_account_identity", "conversation_id", "parent_message_id",
                    "client_conversation_id", "retain_conversation", "image_thread_id", "edit_source_task_id", "edit_source_index", "_image_thread"):
            payload.pop(key, None)
        if payload.get("upstream_model"):
            raise CompletionError("COMPLETION_CONTEXT_UNAVAILABLE")
        payload.update(image_thread_id=replacement_id, _completion_of=request_id)
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
        if (child and state.get("selected_id") == request_id and not child.get("_submission_started")
                and (child.get("status") in {"queued", "running"}
                     or child.get("status") == "failed" and child.get("upstream_outcome") == "not_sent")):
            child.update(status="failed" if kind == "text" else "error", error_code="COMPLETION_ORIGINAL_RECOVERED",
                         upstream_outcome="not_sent", upstream_unfinished=False, _turn_reserved=False,
                         _claim_id=None, _claim_until=0, _executing=False)
            self.store.write_receipt(db, kind, owner, child_id, child)
            from services.workflow_scheduling import release_provisional_slot
            release_provisional_slot(self.store, db, child)
        if kind == "text" and child and self.text._cancelled_completion_child(child, root):
            if child.get("_work_key") and child.get("_work_key") == root.get("_work_key"):
                work = self.store.runtime(db, root["_work_key"])
                if work and work["last_request_id"] == child_id:
                    work["last_request_id"] = request_id
                    self.store.set_runtime(db, work["key"], work)
        return child

    def advance(self, kind, owner, request_id):
        now = float(self.clock())
        with self.store.transaction() as db:
            root = self._root(db, kind, owner, request_id)
            state = root.get("_completion")
            if not state or state.get("state") == "completed":
                return
            child = self._select(db, kind, owner, request_id, root)
            if state.get("selected_id"):
                self.store.write_receipt(db, kind, owner, request_id, root)
                return
            # Lease the orchestration attempt; actual model admission still uses
            # the normal durable claim. Repeated API calls cannot bypass cooldown.
            if float(state.get("next_at") or 0) > now:
                return
            state["next_at"] = now + self.RECHECK_SECONDS
            self.store.write_receipt(db, kind, owner, request_id, root)
        # Existing exact-original readers preserve Retry-After/backoff and errors.
        if kind == "text":
            self.text.read(owner, request_id)
        elif (root.get("status") == "error" and unresolved(root)
              and now - float(root.get("_completion_read_at") or 0) > self.INVESTIGATION_SECONDS):
            self.images.resume_poll({"id": owner}, request_id, extra_timeout_secs=5,
                                    allow_unrecoverable_retry=True, completion_recheck=True)
        with self.store.transaction() as db:
            root = self._root(db, kind, owner, request_id)
            state = root["_completion"]
            child = self._select(db, kind, owner, request_id, root)
            if not state.get("selected_id") and not child:
                try:
                    active = root.get("_executing") or root.get("recovery_claim_id") or root.get("status") in {"queued", "running", "not_started"}
                    if active:
                        raise CompletionError("COMPLETION_ORIGINAL_ACTIVE")
                    work = self.store.runtime(db, root.get("_work_key")) if root.get("_work_key") else None
                    if work and work.get("state") != "active":
                        raise CompletionError("COMPLETION_WORK_NOT_ACTIVE")
                    ended = kind == "text" and self.text._verified_retryable_empty(root)
                    if kind == "text" and root.get("conversation_id") and not ended and not state["allow_unconfirmed_retry"]:
                        raise CompletionError("COMPLETION_ORIGINAL_END_UNCONFIRMED")
                    unknown = unresolved(root) and not ended
                    if unknown:
                        if not state["allow_unconfirmed_retry"]:
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
                        replacement_id = "completion-" + uuid.uuid4().hex
                        prepared = self._prepare(db, kind, owner, request_id, root, replacement_id)
                        state.update(replacement_id=replacement_id, prepared_input=self.store.save_input(prepared),
                                     conversation_mode="original" if ended else "reconstructed", state="replacement_pending")
                    state.pop("reason", None)
                except CompletionError as exc:
                    state.update(state="needs_attention" if exc.code not in {
                        "COMPLETION_ORIGINAL_ACTIVE", "COMPLETION_INVESTIGATING_ORIGINAL"} else "checking_original", reason=exc.code)
            elif child and not state.get("selected_id"):
                state.update(state="replacement_pending" if child.get("status") in {"queued", "running"} else "needs_attention",
                             reason="COMPLETION_ATTEMPT_PENDING" if child.get("status") in {"queued", "running"} else "COMPLETION_ATTEMPT_EXHAUSTED")
                original_work = self.store.runtime(db, root.get("_work_key", "")) or {}
                if root.get("_recovery_suppressed") or original_work.get("state", "active") != "active":
                    state.update(state="needs_attention", reason="COMPLETION_WORK_NOT_ACTIVE")
            self.store.write_receipt(db, kind, owner, request_id, root)
            submit = bool(state.get("replacement_id") and not child and not state.get("selected_id"))
            prepared_ref, replacement_id = state.get("prepared_input"), state.get("replacement_id")
        if submit:
            body = self.store.load_input(prepared_ref)
            # Crash between reservation and submit resumes this same ID/input.
            if kind == "text":
                self.text.submit(owner, body, source=root.get("_source"))
            else:
                self.images._submit(body["identity"], client_task_id=replacement_id, mode=body["mode"], payload=body["payload"])

    def process_one(self):
        with self.store.connect() as db:
            candidates = [(kind, owner, rid) for kind, owner, rid, row in self.store.receipts(db)
                          if row.get("_completion", {}).get("state") not in {None, "completed", "result_ready"}
                          and float(row["_completion"].get("next_at") or 0) <= float(self.clock())]
        if candidates:
            self.advance(*candidates[0])

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
                      "empty_response_confirmed": bool(kind == "text" and self.text._verified_retryable_empty(root)),
                      "original_cleanup": "pending" if unresolved(root) and not ended else "not_required"}
            same_retry = bool(child and kind == "text" and child.get("_terminal_empty_correction_of") == request_id
                              and child.get("_work_key") == root.get("_work_key")
                              and self.text._verified_retryable_empty(root))
            result["local_reservation"] = ("released" if same_retry and successful(kind, child)
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
            self.store.write_receipt(db, kind, str(identity["id"]), request_id, root)
        return self.read(kind, identity, request_id)


def get_generation_completion_service():
    from services.text_task_service import text_task_service
    from services.image_task_service import image_task_service
    from services.work_lifecycle import get_work_lifecycle_service
    return GenerationCompletionService(text_task_service, image_task_service, get_work_lifecycle_service())
