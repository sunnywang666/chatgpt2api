"""Dispatch the existing durable receipts against one account occupancy view.

No waiting request owns an executor thread. Selection, occupancy, the original
account binding and fairness cursor commit in the existing receipt database.
Expired claims are fenced at the model-send edge, never inferred to be unsent.
"""
from __future__ import annotations

from services.image_thread import bind_waiting_threads, predecessor_state

from contextlib import nullcontext
import hashlib
import threading
import time
import uuid

from services.admission_planner import Need, Resource, Offer, RequestRef, WaitingRequest, Snapshot, choose_next
from services.request_context import AdmissionLost, executing, safe_account_ref
from services.task_store import TaskStore
from utils.log import logger


def account_clock_key(account):
    # Use the identity already owned by AccountRequestClock. Duplicate imports
    # of that upstream identity share both its turn and image constraints.
    identity = str(account.get("account_id") or account.get("provider_account_identity") or account.get("access_token") or "")
    return hashlib.sha256(identity.encode()).hexdigest()


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


def unfinished(kind, receipt):
    if recovery_suppressed(receipt):
        return False
    if kind == "image":
        return receipt.get("upstream_unfinished") is True or receipt.get("status") in {"queued", "running"}
    return receipt.get("status") in {"queued", "running", "not_started"} or unknown_text_result(receipt)


def original_turn_ended(kind, receipt):
    if receipt.get("_upstream_terminal") is True:
        return True
    # Older public Chat code recorded a definitive validation rejection as
    # UNKNOWN. Only its exact model endpoint / pre-SSE HTTP evidence proves
    # that turn ended. Missing stages, timeouts and result age prove nothing.
    return (kind == "text" and unknown_text_result(receipt)
            and receipt.get("_route", "chat") == "chat" and not receipt.get("_forward_protocol")
            and receipt.get("original_failure_phase") == "stream_open"
            and receipt.get("original_exception_category") == "http"
            and receipt.get("original_upstream_request_stage") == "conversation"
            and type(receipt.get("original_http_status")) is int and receipt["original_http_status"] == 422)


def image_capacity(account, settings):
    capacity = min(int(settings["image_account_concurrency"]), max(0, int(account.get("quota") or 0)))
    for limit in account.get("limits_progress") or []:
        if isinstance(limit, dict) and limit.get("feature_name") == "image_gen":
            remaining = limit.get("remaining")
            if type(remaining) in (int, float) and remaining >= 0:
                capacity = min(capacity, int(remaining))
    return capacity


def image_generation_active(kind, receipt, active):
    """Count generation only; confirmed output download/save has its own work.

    An EOF, old request or suppressed recovery is not proof of generated output.
    Native image receipts persist original result IDs before downloading; wire
    image receipts persist the original turn's explicit terminal evidence.
    """
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

    def release_turn(self):
        if int(self.receipt().get("_expected_sends") or 1) > 1:
            return  # Bounded legacy multi-image request runs its slots serially.
        self.admission.update_claim(self, _turn_reserved=False)

    def image_slot(self, index):
        r = self.receipt()
        if index >= int(r.get("_expected_sends") or 1) or (index and r.get("_completed_slot") != index - 1):
            raise AdmissionLost("original image sequence is not complete")
        self.admission.update_claim(self, _send_sequence=index)

    def image_slot_complete(self, index):
        self.admission.update_claim(self, _completed_slot=index)

    def record_limit(self, evidence):
        self.admission.update_claim(self, rate_limit=evidence)

    def record_stage(self, stage, **extra):
        """Persist and log a safe stage in the original request timeline."""
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
               if key in {"status_code", "upstream_request_id", "output_ref", "known", "task_status", "image_count"}},
        }
        timeline = list(receipt.get("_execution_timeline") or [])
        timeline.append(entry)
        try:
            self.admission.update_claim(self, _execution_timeline=timeline[-32:])
        except AdmissionLost:
            # Telemetry must never turn an expired original into a retry.
            pass
        logger.info({"event": "pool_execution_stage", **entry})
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
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._next_codex_probe = 0.0

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
                while not self._stop.is_set():
                    claim = self.claim_next()
                    if claim is None:
                        break
                    threading.Thread(target=self.execute, args=(claim,), name="original-task-execute", daemon=True).start()
                if self._recovery_thread is None or not self._recovery_thread.is_alive():
                    self._recovery_thread = threading.Thread(target=self.recover_one, name="original-result-read", daemon=True)
                    self._recovery_thread.start()
            except Exception:
                # Invalid snapshots/storage are not permission to bypass
                # admission. The existing receipt remains queryable.
                from utils.log import logger
                logger.warning({"event": "task_admission_unavailable", "layer": "provider", "phase": "claim"})
            # Account/config changes in another worker are seen without a
            # client retry. Timed waits never occupy execution workers.
            self._event.wait(1.0)

    def recover_one(self):
        now = float(self.clock())
        with self.store.connect() as db:
            rows = list(self.store.receipts(db))
        # Use the existing persisted retry timestamps. A short-backoff legacy
        # scan must not monopolize recovery ahead of other overdue originals.
        # Missing/zero retry times are immediately due. Image created_at is a
        # formatted string, so it must not enter this numeric retry ordering.
        rows.sort(key=lambda row: float(row[3].get("recovery_next_at" if row[0] == "text" else "next_poll_at")
                                        or 0))
        for kind, owner, request_id, r in rows:
            if recovery_suppressed(r):
                continue
            if kind not in self.recoveries:
                continue
            if r.get("_forward_protocol"):
                from services.durable_forward import chat_recovery_supported
                if not chat_recovery_supported(r):
                    continue
            if kind == "text":
                due = unknown_text_result(r) and r.get("provider_binding_id") and float(r.get("recovery_next_at") or 0) <= now
            else:
                due = (r.get("status") == "error" and r.get("error_code") == "CONVERSATION_OUTCOME_UNKNOWN"
                       and r.get("conversation_id") and r.get("request_message_id") and float(r.get("next_poll_at") or 0) <= now)
            if due:
                try:
                    self.recoveries[kind](owner, request_id)
                except Exception:
                    pass  # Original read handlers retain bounded retry evidence.
                return

    def run_recovery(self, context, function, args):
        done = threading.Event()
        def heartbeat():
            while not done.wait(self.CLAIM_SECONDS / 3):
                try:
                    self.update_claim(context, _claim_until=float(self.clock()) + self.CLAIM_SECONDS)
                except Exception:
                    return
        threading.Thread(target=heartbeat, name="original-read-heartbeat", daemon=True).start()
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
        return model_catalog_service.route_for_model(model).account_types

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

    def _snapshot(self, rows, receipts, settings, types, now, cursor):
        # A verified empty upstream turn may have one explicitly accepted
        # correction in the same public Chat session. Its UNKNOWN receipt must
        # remain intact for recovery, but it cannot remain the session's order
        # head forever. This is a planner-only projection, never a status edit.
        text_receipts = {(owner, request_id): r for kind, owner, request_id, r in receipts
                         if kind == "text"}
        corrections = {}
        for kind, owner, request_id, r in receipts:
            previous_id = r.get("_terminal_empty_correction_of")
            if kind == "text" and isinstance(previous_id, str) and previous_id:
                corrections.setdefault((owner, previous_id), []).append((request_id, r))
        released_order_heads = set()
        if corrections:
            from services.text_task_service import TextTaskService
            for key, candidates in corrections.items():
                if len(candidates) != 1:
                    continue
                previous = text_receipts.get(key)
                evidence = TextTaskService._verified_terminal_empty(previous or {})
                request_id, correction = candidates[0]
                if (not evidence or not previous or request_id == key[1]
                        or correction.get("status") not in {"queued", "running", "unknown", "succeeded"}
                        or not correction.get("_input_ref")
                        or correction.get("route") != "chat"
                        or correction.get("_route", "chat") != "chat"
                        or correction.get("_previous_request_id") != key[1]
                        or correction.get("_public_session_ref") != previous.get("_public_session_ref")
                        or correction.get("client_conversation_id") != previous.get("client_conversation_id")
                        or correction.get("model") != previous.get("model")
                        or correction.get("parent_message_id") != evidence["final_message_id"]
                        or correction.get("request_message_id") == previous.get("request_message_id")
                        or type(correction.get("_sequence")) is not int
                        or type(previous.get("_sequence")) is not int
                        or correction["_sequence"] <= previous["_sequence"]
                        or any(correction.get(field) != previous.get(field) for field in (
                            "provider_binding_id", "provider_account_identity", "conversation_id"))):
                    continue
                released_order_heads.add(key)
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
        for kind, owner, request_id, r in receipts:
            status = r.get("status")
            pending = unfinished(kind, r)
            unknown = unresolved_result(kind, r)
            account = by_identity.get(str(r.get("provider_account_identity") or ""))
            resource = account_clock_key(account) if account else r.get("_account_resource")
            active = status == "running" or unknown
            # An unresolved result and an executing model turn are distinct.
            # Only positive original-turn terminal evidence can clear UNKNOWN
            # occupancy; age, a closed socket or a missing result cannot.
            turn_active = active and not original_turn_ended(kind, r)
            if status == "running" and r.get("_executing") and float(r.get("_claim_until") or 0) > now:
                active_bytes += int(r.get("_input_bytes") or 0)
                if kind == "image" or r.get("_operation") == "image":
                    active_images += 1
                if kind == "text" and r.get("_route", "chat") == "chat" and r.get("_operation") != "image":
                    active_text += 1
            if turn_active and not resource:
                unknown_unbound = True
            route = r.get("_route", "chat")
            if resource and turn_active and (unknown or r.get("_turn_reserved", True)):
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
            state = "unknown" if unknown else "queued" if status in {"queued", "not_started"} else "active" if active else "succeeded"
            thread_needs = ()
            if kind == "image" and r.get("_image_thread"):
                _, blocked = predecessor_state(r, image_receipts.get(owner, {}))
                dependency = "image_thread:" + hashlib.sha256((owner + "\0" + request_id).encode()).hexdigest()
                thread_resources.append(Resource(dependency, 0 if blocked else 1, 0, now))
                thread_needs = (Need(dependency),)
            # Legacy pending rows still protect their conversation order, but
            # cannot execute without a durable input reference.
            requests.append(WaitingRequest(
                RequestRef(owner, kind, request_id), r.get("_source") or "key:" + owner,
                int(r.get("_sequence") or 0), route, str(r.get("model") or "auto"),
                "image" if kind == "image" else r.get("_operation", "text"),
                bool(r.get("_input_ref")), state=state,
                bound_account=str(r.get("provider_account_identity") or "") or None,
                order_group=(None if (owner, request_id) in released_order_heads
                             else str(r.get("client_conversation_id") or "") or None),
                ready_at=float(r.get("_ready_at") or 0),
                needs=thread_needs + (Need("execution_input_bytes", max(1, int(r.get("_input_bytes") or 1))),)
                      + ((Need("chat_executor"),) if kind == "text" and route == "chat" and r.get("_operation") != "image" else ())
                      + ((Need("image_executor"),) if kind == "image" or r.get("_operation") == "image" else ()),
            ))
        codex_used = sum(value for key, value in occupied.items() if key.startswith("codex_turn:"))
        resources = {"execution_input_bytes": Resource("execution_input_bytes", self.MAX_ACTIVE_INPUT_BYTES, active_bytes, now),
                     "chat_executor": Resource("chat_executor", self.text_workers or 0, active_text, now),
                     "codex_server": Resource("codex_server", int(settings.get("codex_max_concurrency", 4)), codex_used, now)}
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
            new = Resource(turn_key, chat_capacity, None if unknown_unbound else occupied.get(turn_key, 0), pace.get("next_at"))
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
                    eligible = self.codex is not None and self.codex._eligible_account(account, model, allow_probe=False) is not None
                    if request.operation == "image":
                        eligible = eligible and account.get("source_type") == "codex"
                    decision = self.codex.quota_decision(account, model) if eligible and hasattr(self.codex, "quota_decision") else {}
                    offers.append(Offer(identity, "codex", model, request.operation,
                                        (Need(codex_key), Need("codex_server")), enabled=bool(eligible),
                                        preference=0 if decision.get("quota_bucket") == "gpt-reserve" else 1))
                    continue
                enabled = not disabled and chat_saved and paid
                if request.operation == "image":
                    from utils.helper import is_codex_image_model, split_image_model
                    required_plan, _ = split_image_model(model)
                    enabled = enabled and image_slots > 0 and not is_codex_image_model(model)
                    if required_plan:
                        enabled = enabled and plan == self.accounts._normalize_account_type(required_plan)
                    needs = (Need(turn_key), Need(image_key))
                else:
                    enabled = enabled and plan in types.get(model, set())
                    needs = (Need(turn_key),)
                # Explicit per-model exhausted observations apply only to that
                # model. Unknown limits are not invented as an extra balance.
                for limit in account.get("limits_progress") or []:
                    if isinstance(limit, dict) and limit.get("feature_name") == model and limit.get("remaining") == 0:
                        enabled = False
                offers.append(Offer(identity, "chat", model, request.operation, needs, enabled=bool(enabled)))
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
        return Snapshot(uuid.uuid4().hex, now, now + 1, tuple(resources.values()), tuple(unique.values()), tuple(requests),
                        last_source=cursor.get("source"), last_account=cursor.get("account"),
                        last_owner_by_source=tuple((cursor.get("owners") or {}).items()),
                        scoped_extra_turn=scoped_extra_turn)

    def _recover_claims(self, db, receipts, now):
        for kind, owner, request_id, r in receipts:
            if not r.get("_claim_id") or r.get("status") != "running" or float(r.get("_claim_until") or 0) > now:
                continue
            if r.get("_submission_started"):
                r.update(status="unknown" if kind == "text" else "error", error_code="CONVERSATION_OUTCOME_UNKNOWN")
                if kind == "image":
                    r["upstream_unfinished"] = not bool(r.get("result_file_ids") or r.get("result_sediment_ids"))
                    if not r["upstream_unfinished"]:
                        r.update(upstream_outcome="generated", recovery_error_code="RECOVERY_DOWNLOAD_FAILED", next_poll_at=0)
            else:
                r.update(status="queued", _turn_reserved=False, _claim_id=None, upstream_unfinished=False)
            self.store.write_receipt(db, kind, owner, request_id, r)

    def resource_snapshot(self):
        """Read-only projection of the same persisted occupancy used to claim.

        Null means unresolved occupancy; zero is a known empty/closed resource.
        Session counts are intentionally absent from execution capacity.
        """
        now = float(self.clock())
        with self.store.connect() as db, self._account_guard():
            rows = self._rows()
            receipts = list(self.store.receipts(db))
            settings = self._settings()
            snapshot = self._snapshot(rows, receipts, settings, {}, now, {})
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
        recovery = sum(1 for kind, _, _, r in receipts if not recovery_suppressed(r) and
                       (unknown_text_result(r) if kind == "text" else r.get("upstream_unfinished") is True
                        and r.get("status") in {"error", "unknown"}))
        saving = sum(1 for kind, _, _, r in receipts if r.get("status") == "running"
                     and (kind == "image" or r.get("_operation") == "image")
                     and not image_generation_active(kind, r, True))
        return {"accounts": result, "settings": settings, "chat_turn": aggregate("chat_turn"), "image": image_summary, "codex": native,
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
        with self.store.connect() as db, self._account_guard():
            rows = self._rows()
            snapshot = self._snapshot(rows, list(self.store.receipts(db)), self._settings(), {}, now, {})
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

    def claim_next(self):
        # Catalog lookup may perform a metadata read; never do that under the
        # database/account transaction or occupy an execution worker waiting.
        with self.store.connect() as db:
            models = {str(r.get("model") or "auto") for kind, _, _, r in self.store.receipts(db)
                      if kind == "text" and unfinished(kind, r) and r.get("_route", "chat") == "chat"}
        types = {}
        for model in models:
            try:
                types[model] = self._types(model)
            except Exception:
                types[model] = set()
        if self.codex is not None:
            with self.store.connect() as db:
                native_models = {r.get("model") for _, _, _, r in self.store.receipts(db)
                                 if r.get("status") == "queued" and r.get("_route") == "codex"}
            # Reuse the existing bounded metadata refresh, outside the claim
            # transaction. This never sends a model request.
            if native_models and float(self.clock()) >= self._next_codex_probe:
                self._next_codex_probe = float(self.clock()) + 30
                probes = 0
                for account in self._rows():
                    projection = self.codex.account_projection(account)
                    if projection.get("authorization_status") != "saved" or account.get("managed_disabled"):
                        continue
                    if self.codex._observation_fresh(projection) and projection.get("state") != "unknown":
                        continue
                    if float((account.get("codex_rate_limit") or {}).get("cooldown_until") or 0) > time.time():
                        continue
                    self.codex._eligible_account(account, allow_probe=True)
                    probes += 1
                    if probes >= 3:
                        break
        with self._settings_guard(), self.store.transaction() as db, self._account_guard():
            now = float(self.clock())
            receipts = list(self.store.receipts(db))
            self._recover_claims(db, receipts, now)
            bind_waiting_threads(self.store, db, receipts)
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
                        r["provider_account_identity"] = bound[0]["provider_account_identity"]
                    self.store.write_receipt(db, kind, owner, request_id, r)
            settings = self._settings()
            snapshot = self._snapshot(self._rows(), receipts, settings, types, now, self.store.runtime(db, "fairness", {}))
            selection = choose_next(snapshot, now)
            for deferred in selection.deferred:
                r = self.store.read_receipt(db, deferred.ref.kind, deferred.ref.owner, deferred.ref.request_id)
                r["waiting"] = {"reasons": ([r["_image_thread_waiting_reason"]] if r.get("_image_thread_waiting_reason") else list(deferred.reasons)), "next_check_at": deferred.next_at}
                previous_id = (r.get("_image_thread") or {}).get("previous_task_id")
                if previous_id and r.get("_image_thread_waiting_reason"):
                    # Same-owner original receipt is the recovery entry. This is
                    # diagnostic only: never replace or resend its failed turn.
                    r["waiting"].update(previous_task_id=previous_id, action="read_original_predecessor")
                self.store.write_receipt(db, deferred.ref.kind, deferred.ref.owner, deferred.ref.request_id, r)
            if selection.dispatch is None:
                return None
            pick = selection.dispatch
            ref = pick.ref
            r = self.store.read_receipt(db, ref.kind, ref.owner, ref.request_id)
            if r.get("status") != "queued" or not r.get("_input_ref"):
                return None
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
            r.update(status="running", _claim_id=claim, _claim_until=now + self.CLAIM_SECONDS,
                     _turn_reserved=True, _submission_started=False, _executing=True,
                     _account_resource=account_clock_key(rows[pick.account]),
                     _temporary_chat_second_slot=trial_marker,
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
            self.store.write_receipt(db, ref.kind, ref.owner, ref.request_id, r)
            owners = {source: owner for source, owner in snapshot.last_owner_by_source
                      if any(request.source == source and request.state == "queued" for request in snapshot.requests)}
            owners[pick.source] = ref.owner
            self.store.set_runtime(db, "fairness", {"source": pick.source, "account": pick.account, "owners": owners})
            return ExecutionContext(self, ref.kind, ref.owner, ref.request_id, claim)

    def update_claim(self, context, **changes):
        with self.store.transaction() as db:
            r = self.store.read_receipt(db, context.kind, context.owner, context.request_id)
            if r is None or r.get("_claim_id") != context.claim:
                raise AdmissionLost("original task claim changed")
            r.update(changes)
            self.store.write_receipt(db, context.kind, context.owner, context.request_id, r)
        self.wake()

    def before_send(self, context):
        with self._settings_guard(), self.store.transaction() as db, self._account_guard():
            now = float(self.clock())
            r = self.store.read_receipt(db, context.kind, context.owner, context.request_id)
            if (r is None or r.get("_claim_id") != context.claim or r.get("status") != "running"
                    or float(r.get("_claim_until") or 0) <= now):
                raise AdmissionLost("original task claim expired")
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
            if context.kind == "image" and r.get("_image_thread"):
                owned = {task_id: task for kind, owner, task_id, task in self.store.receipts(db) if kind == "image" and owner == context.owner}
                binding, blocked = predecessor_state(r, owned)
                if blocked or any(r.get(k) != v for k, v in binding.items()):
                    raise AdmissionLost("original image thread predecessor changed before send")
            sequence = int(r.get("_send_sequence") or 0)
            if r.get("_submission_started") and sequence <= int(r.get("_last_sent_sequence") or 0):
                raise AdmissionLost("original model request was already submitted")
            selected = next((a for a in self._rows() if a.get("provider_account_identity") == r.get("provider_account_identity")), None)
            if selected is None or selected.get("managed_disabled") or selected.get("status") in {"禁用", "异常", "限流"}:
                raise AdmissionLost("original account is unavailable before send")
            if r.get("_route") == "codex":
                if self.codex is None or self.codex._eligible_account(selected, r.get("model", ""), allow_probe=False) is None:
                    raise AdmissionLost("original Codex model is unavailable before send")
                expected = getattr(context, "expected_codex_quota_bucket", None)
                if expected is not None and self.codex.quota_decision(selected, r.get("model", ""))["quota_bucket"] != expected:
                    raise AdmissionLost("original Codex quota changed before send")
            else:
                if context.kind == "image" or r.get("_operation") == "image":
                    capacity = image_capacity(selected, self._settings())
                    occupied = sum(1 for kind, _, _, other in self.store.receipts(db)
                                   if (kind == "image" or other.get("_operation") == "image")
                                   and other.get("_account_resource") == r.get("_account_resource")
                                   and image_generation_active(kind, other, other.get("status") == "running" or unresolved_result(kind, other)))
                    if capacity < occupied:
                        raise AdmissionLost("original image capacity decreased before send")
                if any(isinstance(limit, dict) and limit.get("feature_name") == r.get("model") and limit.get("remaining") == 0
                       for limit in selected.get("limits_progress") or []):
                    raise AdmissionLost("original model quota is unavailable before send")
            timeline = list(r.get("_execution_timeline") or [])
            timeline.append({"stage": "send_guard_passed", "at": now})
            r.update(_submission_started=True, _last_sent_sequence=sequence, _turn_reserved=True,
                     _claim_until=now + self.CLAIM_SECONDS, _execution_timeline=timeline[-32:])
            self.store.write_receipt(db, context.kind, context.owner, context.request_id, r)

    def execute(self, context):
        done = threading.Event()
        def heartbeat():
            while not done.wait(self.CLAIM_SECONDS / 3):
                try:
                    self.update_claim(context, _claim_until=float(self.clock()) + self.CLAIM_SECONDS)
                except Exception:
                    return
        heart = threading.Thread(target=heartbeat, name="original-task-heartbeat", daemon=True)
        heart.start()
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
            payload = body["payload"] if context.kind == "image" else body
            payload.update({k: r[k] for k in ("provider_binding_id", "provider_account_identity", "client_conversation_id") if r.get(k)})
            if context.kind == "image" and r.get("_image_thread"):
                payload.update({k: r[k] for k in ("_image_thread", "_image_thread_predecessor_message",
                    "_image_thread_predecessor_result_ids", "conversation_id", "parent_message_id") if r.get(k)})
                self.update_claim(context, _image_thread_request_parent=r.get("parent_message_id") or None)
            payload["_admission_claim"] = context.claim
            payload["_request_message_id"] = r.get("request_message_id")
            with executing(context):
                self.handlers[context.kind](context, body)
        except Exception:
            with self.store.transaction() as db:
                r = self.store.read_receipt(db, context.kind, context.owner, context.request_id)
                if r and r.get("_claim_id") == context.claim and r.get("status") == "running":
                    if r.get("_submission_started"):
                        r.update(status="unknown" if context.kind == "text" else "error", error_code="CONVERSATION_OUTCOME_UNKNOWN")
                    else:
                        r.update(status="queued", _claim_id=None, _turn_reserved=False, upstream_unfinished=False,
                                 _ready_at=float(self.clock()) + 1)
                    self.store.write_receipt(db, context.kind, context.owner, context.request_id, r)
        finally:
            done.set()
            with self.store.transaction() as db:
                r = self.store.read_receipt(db, context.kind, context.owner, context.request_id)
                if r and r.get("_claim_id") == context.claim:
                    r["_executing"] = False
                    if not r.get("_submission_started") and r.get("error_code") in {"CONVERSATION_OUTCOME_UNKNOWN", "CONVERSATION_BINDING_UNAVAILABLE", "IMAGE_RESOURCE_UNAVAILABLE"}:
                        r.update(status="queued", _claim_id=None, _turn_reserved=False, upstream_unfinished=False,
                                 _ready_at=float(self.clock()) + 1)
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
    admission = PoolAdmission(text_task_service.store, account_service, codex=codex_service)
    admission.register("text", lambda context, body: text_task_service._run(context.owner, context.request_id, body))
    def image(context, body):
        payload = body["payload"]
        payload["retain_conversation"] = True
        image_task_service._run_task(context.owner + ":" + context.request_id, body["mode"], payload,
                                     body["identity"], str(payload.get("model") or "gpt-image-2"))
    admission.register("image", image)
    admission.recoveries["text"] = lambda owner, request_id: text_task_service.read(owner, request_id)
    admission.recoveries["image"] = lambda owner, request_id: image_task_service.resume_poll({"id": owner, "role": "user"}, request_id)
    text_task_service.admission = admission
    image_task_service.admission = admission
    return admission
