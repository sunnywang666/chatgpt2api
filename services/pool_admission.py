"""Dispatch the existing durable receipts against one account occupancy view.

No waiting request owns an executor thread. Selection, occupancy, the original
account binding and fairness cursor commit in the existing receipt database.
Expired claims are fenced at the model-send edge, never inferred to be unsent.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import threading
import time
import uuid

from services.admission_planner import Need, Resource, Offer, RequestRef, WaitingRequest, Snapshot, choose_next
from services.request_context import AdmissionLost, executing
from services.task_store import TaskStore


def account_clock_key(account):
    # Use the identity already owned by AccountRequestClock. Duplicate imports
    # of that upstream identity share both its turn and image constraints.
    identity = str(account.get("account_id") or account.get("provider_account_identity") or account.get("access_token") or "")
    return hashlib.sha256(identity.encode()).hexdigest()


def unknown_text_result(receipt):
    return (receipt.get("status") == "unknown"
            or receipt.get("status") == "failed" and receipt.get("error_code") == "RESULT_UNRECOVERABLE"
            and receipt.get("upstream_outcome") == "unknown")


def unfinished(kind, receipt):
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

    def log_fields(self):
        r = self.receipt()
        return {"model": r.get("model"), "operation": r.get("_operation", "image" if self.kind == "image" else "text"),
                "retained_input_bytes": r.get("_input_bytes"), "send_sequence": r.get("_send_sequence", 0),
                "route": r.get("_route", "chat")}

    def receipt(self):
        with self.admission.store.connect() as db:
            return self.admission.store.read_receipt(db, self.kind, self.owner, self.request_id)

    def selected_account(self):
        identity = self.receipt().get("provider_account_identity")
        return next((a for a in self.admission._rows() if a.get("provider_account_identity") == identity), None)

    def terminal(self, known):
        self.admission.update_claim(self, _upstream_terminal=bool(known), _turn_reserved=not known)


class PoolAdmission:
    CLAIM_SECONDS = 30.0
    # Existing text executor's retained-input protection; waiting inputs live
    # on disk. This is an execution memory ceiling, not an account/user budget.
    MAX_ACTIVE_INPUT_BYTES = 256 * 1024 * 1024

    def __init__(self, store: TaskStore, accounts, *, clock=time.time,
                 settings=None, model_types=None, pacing=None, codex=None, text_workers=4):
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
                self.update_claim(context, _executing=False)
            except AdmissionLost:
                pass
            self.wake()

    def _settings(self):
        if self.settings:
            return self.settings()
        from services.config import config
        return config.resource_settings()

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
        by_identity = {str(a.get("provider_account_identity") or ""): a for a in rows}
        occupied = {}
        unknown_unbound = False
        active_bytes = 0
        active_text = 0
        requests = []
        for kind, owner, request_id, r in receipts:
            status = r.get("status")
            pending = unfinished(kind, r)
            unknown = (unknown_text_result(r) if kind == "text" else status == "unknown"
                       or status == "error" and r.get("upstream_unfinished"))
            account = by_identity.get(str(r.get("provider_account_identity") or ""))
            resource = account_clock_key(account) if account else r.get("_account_resource")
            active = status == "running" or unknown
            # An unresolved result and an executing model turn are distinct.
            # Only positive original-turn terminal evidence can clear UNKNOWN
            # occupancy; age, a closed socket or a missing result cannot.
            turn_active = active and not original_turn_ended(kind, r)
            if status == "running" and r.get("_executing") and float(r.get("_claim_until") or 0) > now:
                active_bytes += int(r.get("_input_bytes") or 0)
                if kind == "text" and r.get("_route", "chat") == "chat":
                    active_text += 1
            if turn_active and not resource:
                unknown_unbound = True
            route = r.get("_route", "chat")
            if resource and turn_active and (unknown or r.get("_turn_reserved", True)):
                key = route + "_turn:" + resource
                occupied[key] = occupied.get(key, 0) + 1
            if resource and (kind == "image" or r.get("_operation") == "image") and (r.get("upstream_unfinished") or active):
                key = "image:" + resource
                occupied[key] = occupied.get(key, 0) + 1
            state = "unknown" if unknown else "queued" if status in {"queued", "not_started"} else "active" if active else "succeeded"
            # Legacy pending rows still protect their conversation order, but
            # cannot execute without a durable input reference.
            requests.append(WaitingRequest(
                RequestRef(owner, kind, request_id), r.get("_source") or "key:" + owner,
                int(r.get("_sequence") or 0), route, str(r.get("model") or "auto"),
                "image" if kind == "image" else r.get("_operation", "text"),
                bool(r.get("_input_ref")), state=state,
                bound_account=str(r.get("provider_account_identity") or "") or None,
                order_group=str(r.get("client_conversation_id") or "") or None,
                ready_at=float(r.get("_ready_at") or 0),
                needs=(Need("execution_input_bytes", max(1, int(r.get("_input_bytes") or 1))),)
                      + ((Need("chat_executor"),) if kind == "text" and route == "chat" else ()),
            ))
        codex_used = sum(value for key, value in occupied.items() if key.startswith("codex_turn:"))
        resources = {"execution_input_bytes": Resource("execution_input_bytes", self.MAX_ACTIVE_INPUT_BYTES, active_bytes, now),
                     "chat_executor": Resource("chat_executor", self.text_workers, active_text, now),
                     "codex_server": Resource("codex_server", int(settings.get("codex_max_concurrency", 4)), codex_used, now)}
        offers = []
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
            new = Resource(turn_key, 1, None if unknown_unbound else occupied.get(turn_key, 0), pace.get("next_at"))
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
        # All sources are server-authenticated; cursor moves only on a claim.
        unique = {offer.key: offer for offer in offers}
        return Snapshot(uuid.uuid4().hex, now, now + 1, tuple(resources.values()), tuple(unique.values()), tuple(requests),
                        last_source=cursor.get("source"), last_account=cursor.get("account"))

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
            item = {"provider_account_identity": identity, "enabled": bool(enabled)}
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
        native = aggregate("codex")
        native["server_limit"] = settings["codex_max_concurrency"]
        native["slots_total"] = min(native["slots_total"], native["server_limit"])
        if native["slots_free"] is not None:
            native["slots_free"] = min(native["slots_free"], max(0, native["server_limit"] - native["inflight"]))
            native["dispatchable_now"] = min(native["dispatchable_now"], native["slots_free"])
        queued = [(kind, owner, r) for kind, owner, _, r in receipts if r.get("status") == "queued"]
        sources = {}
        for _, owner, r in queued:
            source = r.get("_source") or "key:" + owner
            sources[source] = sources.get(source, 0) + 1
        return {"accounts": result, "settings": settings, "chat_turn": aggregate("chat_turn"), "image": aggregate("image"), "codex": native,
                "queue": {"mode": "durable_original_receipts", "queued": len(queued), "by_source": sources},
                "execution": {"active_input_bytes": resources["execution_input_bytes"].occupied, "max_input_bytes": self.MAX_ACTIVE_INPUT_BYTES,
                              "chat_workers_active": resources["chat_executor"].occupied, "chat_workers_limit": self.text_workers}}

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
        now = float(self.clock())
        with self.store.transaction() as db, self._account_guard():
            receipts = list(self.store.receipts(db))
            self._recover_claims(db, receipts, now)
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
            snapshot = self._snapshot(self._rows(), receipts, self._settings(), types, now, self.store.runtime(db, "fairness", {}))
            selection = choose_next(snapshot, now)
            for deferred in selection.deferred:
                r = self.store.read_receipt(db, deferred.ref.kind, deferred.ref.owner, deferred.ref.request_id)
                r["waiting"] = {"reasons": list(deferred.reasons), "next_check_at": deferred.next_at}
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
            r.update(status="running", _claim_id=claim, _claim_until=now + self.CLAIM_SECONDS,
                     _turn_reserved=True, _submission_started=False, _executing=True,
                     _account_resource=account_clock_key(rows[pick.account]),
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
            self.store.set_runtime(db, "fairness", {"source": pick.source, "account": pick.account})
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
        now = float(self.clock())
        with self.store.transaction() as db, self._account_guard():
            r = self.store.read_receipt(db, context.kind, context.owner, context.request_id)
            if (r is None or r.get("_claim_id") != context.claim or r.get("status") != "running"
                    or float(r.get("_claim_until") or 0) <= now):
                raise AdmissionLost("original task claim expired")
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
                                   and (other.get("upstream_unfinished") or other.get("status") in {"running", "unknown"}))
                    if capacity < occupied:
                        raise AdmissionLost("original image capacity decreased before send")
                if any(isinstance(limit, dict) and limit.get("feature_name") == r.get("model") and limit.get("remaining") == 0
                       for limit in selected.get("limits_progress") or []):
                    raise AdmissionLost("original model quota is unavailable before send")
            r.update(_submission_started=True, _last_sent_sequence=sequence, _turn_reserved=True, _claim_until=now + self.CLAIM_SECONDS)
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
            self.wake()


def configure_original_task_admission():
    from services.account_service import account_service
    from services.text_task_service import text_task_service
    from services.image_task_service import image_task_service
    from services.codex_service import codex_service
    admission = PoolAdmission(text_task_service.store, account_service, codex=codex_service, text_workers=text_task_service.executor.limit)
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
