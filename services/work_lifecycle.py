"""Work completion in the existing receipt database, distinct from one model turn.

The work row holds a workflow slot and a recoverable archive/restore intent.
It never generates, owns credentials, or replaces an original receipt.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from contextlib import nullcontext


class WorkLifecycleError(ValueError):
    def __init__(self, code, status=409):
        super().__init__(code)
        self.code, self.status = code, status


def _reference(kind, owner, receipt):
    if kind == "image" and receipt.get("_image_thread"):
        return str(receipt["_image_thread"]["id"])
    return str(receipt.get("_public_session_ref") or receipt.get("client_conversation_id")
               or receipt.get("request_id") or receipt.get("id") or "")


def _same_conversation(left, right):
    if not left.get("conversation_id") or left.get("conversation_id") != right.get("conversation_id"):
        return False
    if left.get("provider_account_identity") and right.get("provider_account_identity"):
        return left["provider_account_identity"] == right["provider_account_identity"]
    return bool(left.get("provider_binding_id") and left.get("provider_binding_id") == right.get("provider_binding_id"))


def work_key(kind, owner, receipt):
    if receipt.get("_work_key"):
        return receipt["_work_key"]
    ref = _reference(kind, owner, receipt)
    if not ref:
        return None
    # The stable caller scopes workflow capacity, while receipt ownership still
    # scopes application session references. Only a proven predecessor inherits
    # a work row; equal client strings across keys do not grant result access.
    binding = receipt.get("provider_binding_id")
    conversation = receipt.get("conversation_id")
    account = receipt.get("provider_account_identity") or binding
    canonical = ("upstream", account, conversation) if account and conversation else ("client", owner, ref)
    source = receipt.get("_scheduling_owner") or receipt.get("_source") or "key:" + owner
    raw = json.dumps([kind, source, canonical], separators=(",", ":"))
    return "work:" + hashlib.sha256(raw.encode()).hexdigest()


def read_work(store, db, kind, owner, receipt):
    key = work_key(kind, owner, receipt)
    return store.runtime(db, key) if key else None


def save_work(store, db, work):
    store.set_runtime(db, work["key"], work)


def ensure_work(store, db, kind, owner, request_id, receipt, *, source=None, scheduling=None):
    public = bool(receipt.get("_public_session_ref") or receipt.get("_image_thread"))
    scheduling = scheduling if scheduling is not None else receipt.get("_scheduling")
    previous_id = receipt.get("_previous_request_id") or (receipt.get("_image_thread") or {}).get("previous_task_id")
    if previous_id:
        previous = store.read_receipt(db, kind, owner, previous_id)
        if previous and previous.get("_work_key"):
            receipt["_work_key"] = previous["_work_key"]
    if (receipt.get("provider_binding_id") or receipt.get("provider_account_identity")) and receipt.get("conversation_id"):
        known = {r["_work_key"] for k, _, _, r in store.receipts(
                    db, conversation_id=receipt["conversation_id"])
                 if r.get("_work_key") and _same_conversation(r, receipt)}
        if len(known) > 1:
            raise WorkLifecycleError("WORK_BINDING_CONFLICT")
        if known:
            canonical = known.pop()
            if receipt.get("_work_key") not in {None, canonical}:
                raise WorkLifecycleError("WORK_BINDING_CONFLICT")
            receipt["_work_key"] = canonical
    key = work_key(kind, owner, receipt)
    row = store.runtime(db, key)
    if not row and not public and not scheduling and not receipt.get("_work_key"):
        return None
    if row:
        if row["owner"] != owner or row["kind"] != kind:
            raise WorkLifecycleError("WORK_OWNER_CONFLICT")
        if row["last_request_id"] != request_id and row["state"] != "active":
            raise WorkLifecycleError("WORK_NOT_ACTIVE")
        inherited = dict(row.get("scheduling") or {})
        scheduling = {**inherited, **(scheduling or {})}
        if scheduling and (row.get("workflow_id"), row.get("workflow_concurrency")) != (
                scheduling.get("workflow_id"), scheduling.get("workflow_concurrency")):
            raise WorkLifecycleError("WORK_SCHEDULING_CONFLICT")
        row["last_request_id"] = request_id
    else:
        row = {"key": key, "kind": kind, "owner": owner, "work_ref": _reference(kind, owner, receipt),
               "last_request_id": request_id, "source": receipt.get("_scheduling_owner") or source or receipt.get("_source") or "key:" + owner,
               "workflow_id": (scheduling or {}).get("workflow_id"),
               "workflow_concurrency": (scheduling or {}).get("workflow_concurrency"),
               "state": "active", "slot_held": False, "version": 0,
               "created_at": receipt.get("created_at") or time.time(), "archive": None}
    row["scheduling"] = {key: value for key, value in (scheduling or {}).items()
                         if key in {"workflow_id", "workflow_concurrency", "min_send_interval_seconds"}}
    if scheduling:
        receipt["_scheduling"] = scheduling
    row["updated_at"] = time.time()
    receipt["_work_key"], receipt["_work_ref"] = key, row["work_ref"]
    save_work(store, db, row)
    return row


def record_slot(store, db, kind, owner, receipt, held=True):
    row = read_work(store, db, kind, owner, receipt)
    if row:
        row["slot_held"] = bool(held)
        save_work(store, db, row)
    return row


def _blocked(receipt, *, allow_queued=False, members=()):
    # Operator suppression or local worker release never proves remote end.
    if receipt.get("_executing") or receipt.get("recovery_claim_id"):
        return True
    if (receipt.get("status") == "unknown" or receipt.get("upstream_outcome") == "unknown"
            or receipt.get("upstream_unfinished") is True):
        from services.generation_completion import same_session_retry, successful
        restored = [r for r in members if same_session_retry(receipt, r)
                    and successful("image" if r.get("id") else "text", r)
                    and (receipt.get("_completion") or {}).get("selected_id") == (r.get("id") or r.get("request_id"))]
        if len(restored) == 1:
            return False
        from services.text_task_service import TextTaskService
        evidence = TextTaskService._verified_retryable_empty(receipt)
        corrections = [r for r in members if evidence and r.get("status") == "succeeded"
                       and r.get("_terminal_empty_correction_of") == receipt.get("request_id")
                       and r.get("_previous_request_id") == receipt.get("request_id")
                       and r.get("_submission_parent_message_id") == evidence["retry_parent_message_id"]
                       and r.get("_work_key") == receipt.get("_work_key")
                       and r.get("provider_binding_id") == receipt.get("provider_binding_id")
                       and _same_conversation(r, receipt)]
        # Only the completed, proven correction can close the original work.
        # An empty terminal by itself is still not a completed business result.
        if len(corrections) != 1 or (receipt.get("_completion") and
                receipt["_completion"].get("selected_id") != corrections[0].get("request_id")):
            return True
    if receipt.get("status") in {"running", "not_started"}:
        return True
    if receipt.get("status") == "queued":
        return not allow_queued or bool(receipt.get("_submission_started"))
    return False


def _projection(work):
    archive = work.get("archive") or {"status": "not_requested", "desired": None}
    safe_archive = {key: archive[key] for key in (
        "status", "desired", "archived", "next_at", "error_code", "attempts", "scope",
        "requested_at", "attempt_started_at", "confirmed_at") if key in archive}
    return {"protocol": "work-v1", "kind": work["kind"], "work_ref": work["work_ref"],
            "request_id": work["last_request_id"], "state": work["state"],
            "slot_held": work["slot_held"], "version": work["version"],
            "results_saved": bool(work.get("results_saved")), "archive": safe_archive,
            **({"results_saved_at": work["results_saved_at"]} if work.get("results_saved_at") is not None else {}),
            **({"completion_result_id": work["completion_result_id"],
                "cleanup_pending": bool(work.get("cleanup_pending"))} if work.get("completion_result_id") else {})}


def _archive_read_owner(work):
    return hashlib.sha256(f"{work['key']}:{work['archive']['version']}".encode()).hexdigest()


class WorkLifecycleService:
    def __init__(self, text_service, image_service, *, clock=time.time):
        self.text, self.images, self.store = text_service, image_service, text_service.store
        self.clock = clock
        self._operation_lock = threading.Lock()
        self._operation_slots = threading.BoundedSemaphore(4)

    def _load(self, db, kind, identity, request_id):
        owner = str(identity["id"])
        receipt = self.store.read_receipt(db, kind, owner, request_id)
        if not receipt:
            raise WorkLifecycleError("WORK_REQUEST_NOT_FOUND", 404)
        work = read_work(self.store, db, kind, owner, receipt)
        if not work:
            work = ensure_work(self.store, db, kind, owner, request_id, receipt)
            if work:
                self.store.write_receipt(db, kind, owner, request_id, receipt)
        if not work:
            raise WorkLifecycleError("WORK_SESSION_REQUIRED", 400)
        if work["last_request_id"] != request_id:
            raise WorkLifecycleError("WORK_SUPERSEDED")
        # Legacy sessions did not have work rows. Include their proven native
        # session members so a prior UNKNOWN is never lost during adoption.
        rows = list(self.store.work_receipts(db, kind, owner, work["key"], receipt.get("conversation_id")))
        members = [r for k, o, _, r in rows if _same_conversation(r, receipt) or (k == kind and o == owner and (
            r.get("_work_key") == work["key"] or (
                not r.get("_work_key") and _reference(k, o, r) == work["work_ref"]))) ]
        own_members = [r for k, o, _, r in rows if k == kind and o == owner and r.get("_work_key") == work["key"]]
        from services.text_task_service import TextTaskService
        if any(int(r.get("_sequence") or 0) > int(receipt.get("_sequence") or 0)
               and not TextTaskService._cancelled_completion_child(r, receipt) for r in own_members):
            raise WorkLifecycleError("WORK_SUPERSEDED")
        return receipt, work, members

    def get(self, kind, identity, request_id):
        # An existing work is an observation, not a writer. Keep its receipt,
        # work version and member checks in one committed SQLite snapshot so a
        # concurrent archive/update does not make the read compete for its lock.
        with self.store.connect() as db:
            db.execute("BEGIN")
            receipt = self.store.read_receipt(db, kind, str(identity["id"]), request_id)
            if receipt and read_work(self.store, db, kind, str(identity["id"]), receipt):
                _, work, _ = self._load(db, kind, identity, request_id)
                return _projection(work)
        # Legacy adoption still needs the original write transaction and full
        # recheck; never upgrade a potentially stale read snapshot to a writer.
        with self.store.transaction() as db:
            _, work, _ = self._load(db, kind, identity, request_id)
            return _projection(work)

    def update(self, kind, identity, request_id, state, results_saved=False):
        if state not in {"active", "paused", "completed"}:
            raise WorkLifecycleError("WORK_STATE_INVALID", 400)
        with self.store.transaction() as db:
            receipt, work, members = self._load(db, kind, identity, request_id)
            if state == "completed" and results_saved is not True:
                raise WorkLifecycleError("WORK_RESULTS_SAVE_REQUIRED", 400)
            allow_queued = state == "paused" or state == "active" and work["state"] == "paused"
            from services.text_task_service import TextTaskService
            pause_empty = lambda r: (state == "paused" and kind == "text"
                                    and r.get("_work_key") == work["key"]
                                    and TextTaskService._verified_retryable_empty(r))
            if any(_blocked(r, allow_queued=allow_queued, members=members) and not pause_empty(r) for r in members):
                raise WorkLifecycleError("WORK_TURN_UNFINISHED")
            for member in members:
                if pause_empty(member):
                    member["_turn_reserved"] = False
                    self.store.write_receipt(db, kind, str(identity["id"]), member["request_id"], member)
            legacy_restore = state == work["state"] == "active" and not work.get("archive")
            if state == work["state"] and not legacy_restore:
                return _projection(work)
            if work["state"] == "restoring":
                raise WorkLifecycleError("WORK_RESTORE_PENDING")
            if work["state"] == "completed" and state == "paused":
                raise WorkLifecycleError("WORK_COMPLETED_REQUIRES_RESTORE")
            archive = work.get("archive") or {}
            if work["state"] == "completed" and archive.get("status") not in {"confirmed", "not_applicable"}:
                raise WorkLifecycleError("WORK_ARCHIVE_PENDING")
            now = float(self.clock())
            work.update(version=work["version"] + 1, updated_at=now)
            if state == "completed":
                # The immutable target and archive intent are committed in the
                # same transaction as releasing this work's workflow slot.
                # This is the caller's saved-result attestation, not a timestamp
                # claiming that we observed the caller's filesystem write.
                work.update(state="completed", slot_held=False, results_saved=True, results_saved_at=now)
                never_sent = all(r.get("upstream_outcome") == "not_sent"
                                 and not r.get("_submission_started") and not r.get("conversation_id")
                                 for r in members)
                self._intent(work, receipt, desired=True, now=now, never_sent=never_sent)
            elif state == "paused":
                work.update(state="paused", slot_held=False)
            elif work["state"] == "completed" or legacy_restore:
                work.update(state="restoring", slot_held=False)
                self._intent(work, receipt, desired=False, now=now,
                             never_sent=archive.get("error_code") == "UPSTREAM_CONVERSATION_NOT_CREATED")
                if work["archive"]["status"] == "not_applicable":
                    work["state"] = "active"
            else:
                work["state"] = "active"
            save_work(self.store, db, work)
            result = _projection(work)
        admission = getattr(self.text, "admission", None)
        if admission:
            admission.wake()
        return result

    @staticmethod
    def _intent(work, receipt, *, desired, now, never_sent=False):
        supported = bool(receipt.get("_public_session_ref") if work["kind"] == "text" else receipt.get("_image_thread"))
        supported = supported and not never_sent
        work["archive"] = {"status": "pending" if supported else "not_applicable", "desired": desired,
                           "scope": "upstream_conversation" if supported else "provider_work",
                           "next_at": now if supported else None, "requested_at": now, "attempts": 0,
                           "request_id": work["last_request_id"], "version": work["version"]}
        if not supported:
            # Native Codex / legacy single-request protocols expose no verified
            # upstream archive operation. Close the local work, saying so.
            work["archive"]["error_code"] = ("UPSTREAM_CONVERSATION_NOT_CREATED" if never_sent
                                              else "UPSTREAM_ARCHIVE_UNSUPPORTED")

    def set_archived(self, kind, identity, request_id, archived):
        """Preserve the legacy synchronous shape, using the same durable intent."""
        self.update(kind, identity, request_id, "completed" if archived else "active", results_saved=archived)
        with self.store.transaction() as db:
            _, work, _ = self._load(db, kind, identity, request_id)
            key = work["key"]
        self.process_one(target_key=key, defer_reads=True)
        result = self.get(kind, identity, request_id)
        if result["archive"].get("status") != "confirmed" or result["archive"].get("archived") is not archived:
            raise WorkLifecycleError("WORK_ARCHIVE_UNCONFIRMED", 503)
        if kind == "text":
            return {"request_id": request_id, "archived": archived,
                    "conversation": {"protocol": "sequential-v1", "client_conversation_id": result["work_ref"]}}
        with self.store.connect() as db:
            receipt = self.store.read_receipt(db, kind, str(identity["id"]), request_id)
        from services.image_thread import public_thread
        return {"task_id": request_id, "archived": archived, "image_thread": public_thread(receipt)}

    def process_one(self, *, target_key=None, background=False, defer_reads=False):
        # Only ready I/O occupies the bounded workers. Independent original
        # conversations may overlap; a read-credit wait returns its intent to
        # pending instead of holding a whole-account pipeline lock.
        if not self._operation_slots.acquire(blocking=False):
            return False
        if not self._operation_lock.acquire(blocking=False):
            self._operation_slots.release()
            return False
        work = None
        account = None
        reservation = candidate_account = None
        releases = []
        from services.account_request_pacing import (account_pacing_snapshot,
            reserve_account_archive_read, release_account_archive_read)
        try:
            now = float(self.clock())
            # Read account identities before the receipt transaction, following
            # admission's account -> store lock order. Never hold the recovery
            # worker waiting on a known account cooldown or HTTP pacing slot.
            accounts = getattr(getattr(self.text, "admission", None), "accounts", None)
            account_rows = {}
            if accounts is not None:
                with getattr(accounts, "admission_transaction", nullcontext)():
                    read = getattr(accounts, "admission_accounts", None) or accounts.list_accounts
                    account_rows = {a["provider_account_identity"]: a for a in read() if a.get("provider_account_identity")}
            with self.store.connect() as db:
                candidates = [json.loads(raw) for (raw,) in db.execute(
                    "SELECT value FROM task_runtime WHERE name LIKE 'work:%' "
                    "AND json_extract(value,'$.archive.status') IN ('pending','unknown','running')")]
                candidates.sort(key=lambda w: ((w.get("archive") or {}).get("requested_at", w.get("updated_at", 0)), w["key"]))
            for candidate in candidates:
                reservation = None
                candidate_account = None
                with self.store.transaction() as db:
                    candidate = self.store.runtime(db, candidate["key"])
                    if target_key is not None and candidate["key"] != target_key:
                        continue
                    a = candidate.get("archive") or {}
                    if (a.get("status") in {"pending", "unknown", "running"}
                            and float(a.get("next_at") or 0) <= now
                            and float(a.get("claim_until") or 0) <= now):
                        receipt = self.store.read_receipt(db, candidate["kind"], candidate["owner"], a["request_id"]) or {}
                        candidate_account = account_rows.get(receipt.get("provider_account_identity"))
                        try:
                            receipt, current, members = self._load(db, candidate["kind"], {"id": candidate["owner"]}, a["request_id"])
                            if current["version"] != a["version"] or any(_blocked(r, members=members) for r in members):
                                raise WorkLifecycleError("WORK_TURN_UNFINISHED")
                        except WorkLifecycleError as exc:
                            a.update(status="unknown", error_code=exc.code, next_at=now + 60)
                            save_work(self.store, db, candidate)
                            if candidate_account is not None:
                                releases.append((candidate_account, _archive_read_owner(candidate)))
                            continue
                        candidate_account = account_rows.get(receipt.get("provider_account_identity"))
                        if candidate_account is not None:
                            refresh_error = str(candidate_account.get("last_token_refresh_error") or "").lower()
                            if (candidate_account.get("status") == "异常" and (
                                    str(candidate_account.get("last_refresh_error") or "").lower().startswith("token invalidated")
                                    or "refresh_token_invalidated" in refresh_error
                                    or "app_session_terminated" in refresh_error)):
                                # The original account needs reauthorization;
                                # retrying archive cannot repair its credentials.
                                # Keep the intent and allow other accounts to run.
                                a.update(status="unknown", error_code="RECOVERY_AUTH_REQUIRED", next_at=now + 60)
                                save_work(self.store, db, candidate)
                                releases.append((candidate_account, _archive_read_owner(candidate)))
                                continue
                            ready_at = account_pacing_snapshot(candidate_account, now, include_turn=False, include_conversation_read=True)["next_at"]
                            if ready_at is None or ready_at > now:
                                a["next_at"] = now + 60 if ready_at is None else ready_at
                                if ready_at is None:
                                    a.update(status="unknown", error_code="ACCOUNT_PACING_UNAVAILABLE")
                                elif float(account_pacing_snapshot(candidate_account, now, include_turn=False).get("cooldown_until") or 0) <= now:
                                    # Book after committing: HTTP guards acquire
                                    # SQLite while holding the clock, never invert it.
                                    reservation = _archive_read_owner(candidate)
                                save_work(self.store, db, candidate)
                            else:
                                reservation = _archive_read_owner(candidate)
                        else:
                            ready_at = now
                    else:
                        continue
                # Clock -> SQLite is the HTTP guard lock order. Never acquire
                # a clock/FIFO reservation while holding a receipt transaction.
                ready = True
                if reservation is not None:
                    ready = reserve_account_archive_read(candidate_account, reservation)
                if ready_at is None or ready_at > now or ((background or defer_reads) and not ready):
                    continue
                with self.store.transaction() as db:
                    receipt, current, members = self._load(db, candidate["kind"], {"id": candidate["owner"]}, a["request_id"])
                    active = current.get("archive") or {}
                    if (current["version"] != a["version"] or active.get("version") != a["version"]
                            or active.get("status") not in {"pending", "unknown", "running"}
                            or float(active.get("claim_until") or 0) > now
                            or any(_blocked(r, members=members) for r in members)):
                        work = None
                    else:
                        work = current
                    if work is not None:
                        archive = work["archive"]
                        claim = uuid.uuid4().hex
                        archive.update(status="running", claim=claim, claim_until=now + 300,
                                       attempts=int(archive.get("attempts") or 0) + 1, attempt_started_at=now)
                        save_work(self.store, db, work)
                if work is None:
                    if reservation is not None:
                        release_account_archive_read(candidate_account, reservation)
                else:
                    account = candidate_account
                    break
        except Exception:
            if reservation is not None and candidate_account is not None:
                releases.append((candidate_account, reservation))
            self._operation_slots.release()
            raise
        finally:
            self._operation_lock.release()
            for release_account, owner in releases:
                release_account_archive_read(release_account, owner)
        if work is None:
            self._operation_slots.release()
            return False
        # The claim is durable before dispatch. A failed thread start must not
        # strand that original intent for the full 300-second lease.
        if background:
            try:
                threading.Thread(target=self._run_claim, args=(work, claim, account, True),
                                 name="original-work-archive", daemon=True).start()
            except Exception:
                try:
                    with self.store.transaction() as db:
                        current = self.store.runtime(db, work["key"])
                        active = (current or {}).get("archive") or {}
                        if current and current["version"] == work["version"] and active.get("claim") == claim:
                            active.update(status="pending", claim=None, claim_until=None,
                                          error_code=active.get("error_code") or "WORK_ARCHIVE_DISPATCH_FAILED",
                                          next_at=float(self.clock()) + 1)
                            save_work(self.store, db, current)
                finally:
                    if account is not None:
                        release_account_archive_read(account, _archive_read_owner(work))
                    self._release_operation()
                raise
            return True
        return self._run_claim(work, claim, account, defer_reads)

    def _release_operation(self):
        self._operation_slots.release()
        admission = getattr(self.text, "admission", None)
        if admission is not None:
            admission.wake()

    def _run_claim(self, work, claim, account, defer_reads):
        archive = work["archive"]
        request_id = archive["request_id"]
        deferred = None
        from services.account_request_pacing import ArchiveReadDeferred, release_account_archive_read
        try:
            error_code = None
            def check_and_renew_archive():
                with self.store.transaction() as db:
                    current = self.store.runtime(db, work["key"])
                    active = (current or {}).get("archive") or {}
                    now = float(self.clock())
                    if (not current or current["version"] != work["version"]
                            or active.get("claim") != claim or active.get("desired") is not archive["desired"]
                            or float(active.get("claim_until") or 0) <= now):
                        raise WorkLifecycleError("WORK_ARCHIVE_CLAIM_LOST")
                    active["claim_until"] = now + 300
                    save_work(self.store, db, current)
            try:
                from services.request_context import guarding_archive
                with guarding_archive(check_and_renew_archive, read_owner=_archive_read_owner(work),
                                      request_key=work["owner"] + ":" + request_id, work_key=work["key"],
                                      defer_reads=defer_reads):
                    if work["kind"] == "text":
                        result = self.text.set_public_session_archived(work["owner"], request_id, archive["desired"])
                        valid = (result.get("request_id") == request_id
                                 and result.get("conversation", {}).get("client_conversation_id") == work["work_ref"])
                    else:
                        result = self.images.set_thread_archived({"id": work["owner"], "role": "user"}, request_id, archive["desired"])
                        valid = (result.get("task_id") == request_id
                                 and result.get("image_thread", {}).get("id") == work["work_ref"])
                if not valid or result.get("archived") is not archive["desired"]:
                    raise WorkLifecycleError("WORK_ARCHIVE_UNCONFIRMED")
            except ArchiveReadDeferred as exc:
                deferred = exc
            except Exception as exc:
                code = str(getattr(exc, "code", "WORK_ARCHIVE_UNCONFIRMED"))
                error_code = code if re.fullmatch(r"[A-Za-z0-9_]{1,80}", code) else "WORK_ARCHIVE_UNCONFIRMED"
            with self.store.transaction() as db:
                current = self.store.runtime(db, work["key"])
                if (not current or current["version"] != work["version"]
                        or (current.get("archive") or {}).get("claim") != claim):
                    deferred = None  # A superseded intent must release its FIFO lease.
                    return True
                updated = current["archive"]
                updated.update(claim=None, claim_until=None, error_code=error_code)
                if deferred is not None:
                    updated.update(status="pending", next_at=deferred.next_at,
                                   attempts=max(0, updated["attempts"] - 1))
                elif error_code:
                    updated.update(status="unknown", next_at=float(self.clock()) + min(300, 2 ** min(updated["attempts"], 8)))
                else:
                    updated.update(status="confirmed", archived=updated["desired"], next_at=None,
                                   confirmed_at=float(self.clock()))
                    if current.get("cleanup_pending") and updated["desired"]:
                        current.update(cleanup_pending=False, slot_held=False)
                    if current["state"] == "restoring":
                        current["state"] = "active"
                        current["results_saved"] = False
                        current.pop("results_saved_at", None)
                save_work(self.store, db, current)
            return True
        finally:
            if deferred is None and account is not None:
                release_account_archive_read(account, _archive_read_owner(work))
            self._release_operation()


def get_work_lifecycle_service():
    from services.text_task_service import text_task_service
    from services.image_task_service import image_task_service
    admission = text_task_service.admission
    if admission is None:
        raise WorkLifecycleError("WORK_REQUIRES_DURABLE_ADMISSION", 503)
    service = getattr(admission, "work_lifecycle", None)
    if service is None:
        service = WorkLifecycleService(text_task_service, image_task_service)
        admission.work_lifecycle = service
    return service
