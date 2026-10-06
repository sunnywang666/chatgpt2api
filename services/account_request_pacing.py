"""One upstream request clock per account, shared by text, images and polling.

This is pacing, not a promise of an upstream quota. Never retry a request here:
the owning task still decides whether an operation may safely be submitted.
"""
from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
import threading
import time
import uuid
from contextvars import copy_context
from pathlib import Path
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from curl_cffi import CurlInfo

from services.config import DATA_DIR, config
from utils.log import logger
from services.request_context import (current_request, current_archive_guard, current_archive_read_owner,
                                      current_archive_observation, current_archive_step)


# libcurl times are cumulative from request start, not individual phase durations.
# Never collect URL/IP/header/body fields in this diagnostic projection.
_TRANSPORT_INFO_FIELDS = {
    CurlInfo.NAMELOOKUP_TIME: "namelookup_secs",
    CurlInfo.CONNECT_TIME: "connect_secs",
    CurlInfo.APPCONNECT_TIME: "appconnect_secs",
    CurlInfo.PRETRANSFER_TIME: "pretransfer_secs",
    CurlInfo.STARTTRANSFER_TIME: "starttransfer_secs",
    CurlInfo.TOTAL_TIME: "total_secs",
    CurlInfo.NUM_CONNECTS: "num_connects",
    CurlInfo.OS_ERRNO: "os_errno",
    CurlInfo.HTTP_CONNECTCODE: "http_connect_code",
    CurlInfo.SIZE_DOWNLOAD_T: "downloaded_bytes",
}


def _transport_snapshot(response):
    """Use the library's pre-reset snapshot even when perform raised."""
    result = {}
    try:
        status = getattr(response, "status_code", None)
        if type(status) is int and 100 <= status <= 599:
            result["transport_response_status"] = status
        infos = getattr(response, "infos", None)
        if isinstance(infos, dict):
            values = {}
            for info, name in _TRANSPORT_INFO_FIELDS.items():
                value = infos.get(info)
                if name.endswith("_secs"):
                    if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 86400:
                        values[name] = round(float(value), 6)
                elif type(value) is int and 0 <= value <= 2 ** 53:
                    values[name] = value
            if values:
                result["transport_details"] = values
    except Exception:
        # A diagnostic accessor must not replace the original transport error.
        pass
    return result


def _rate_limit_response_features(response):
    """Classify a 429 without retaining body, URL or raw header values.

    Markers are evidence, not an attribution to the application or its edge.
    This projection must never change retry/cooldown decisions.
    """
    result = {"content_type_class": "missing", "retry_after_kind": "absent",
              "request_id_header": "none", "cf_ray_present": False, "via_present": False,
              "response_origin_evidence": "no_marker"}
    try:
        allowed = {"content-type", "retry-after", "x-request-id", "openai-request-id", "cf-ray", "via"}
        headers = {k.lower(): v for k, v in (getattr(response, "headers", {}) or {}).items()
                   if isinstance(k, str) and k.lower() in allowed}
        if "content-type" in headers:
            value = headers["content-type"]
            mime = value.split(";", 1)[0].strip().lower() if isinstance(value, str) else ""
            result["content_type_class"] = {"application/json": "application_json",
                "text/html": "text_html", "text/plain": "text_plain"}.get(mime, "other")
        if "retry-after" in headers:
            result["retry_after_kind"] = "invalid"
            value = headers["retry-after"]
            try:
                seconds = float(value)
                if math.isfinite(seconds) and seconds >= 0:
                    result["retry_after_kind"] = "seconds"
            except (TypeError, ValueError, OverflowError):
                if isinstance(value, str):
                    try:
                        parsedate_to_datetime(value)
                        result["retry_after_kind"] = "http_date"
                    except (TypeError, ValueError, OverflowError):
                        pass
        for name in ("x-request-id", "openai-request-id"):
            if name in headers:
                result["request_id_header"] = name.replace("-", "_")
                break
        result.update(cf_ray_present="cf-ray" in headers, via_present="via" in headers)
        request_marker = result["request_id_header"] != "none"
        edge_marker = result["cf_ray_present"] or result["via_present"]
        result["response_origin_evidence"] = ("mixed_markers" if request_marker and edge_marker
            else "request_id_marker" if request_marker else "edge_marker" if edge_marker else "no_marker")
    except Exception:
        pass  # Diagnostics must not hide the original 429.
    return result


class ProcessMutex:
    """The existing pacing clock/turn lock, shared by workers on this data root."""
    def __init__(self, path=None, on_acquire=None):
        self.path, self.on_acquire = path, on_acquire
        self.local = threading.Lock()
        self.fd = None

    def acquire(self, blocking=True, timeout=-1):
        deadline = time.monotonic() + timeout if timeout >= 0 else None
        if not self.local.acquire(blocking, timeout):
            return False
        try:
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                while True:
                    try:
                        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if not blocking or (deadline is not None and time.monotonic() >= deadline):
                            self.release()
                            return False
                        time.sleep(min(0.05, max(0, deadline - time.monotonic())) if deadline is not None else 0.05)
            if self.on_acquire:
                self.on_acquire()
            return True
        except BaseException:
            self.release()
            raise

    def release(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.local.release()

    def locked(self):
        return self.local.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


class AccountRequestDeadlineExceeded(TimeoutError):
    """A caller's finite request budget elapsed before an upstream send began."""
    pass


class AccountReadRetryBudgetInsufficient(AccountRequestDeadlineExceeded):
    """An original GET attempt no longer has its minimum connection window."""


def _backoff_seconds(failures):
    return min(900.0, 60.0 * (2 ** min(max(0, failures - 1), 4)))


def _read_policy(now, rate_failures, last_limit, read_failures, last_read_limit):
    failures = max(rate_failures if now - last_limit < 900 else 0,
                   read_failures if now - last_read_limit < 900 else 0)
    interval = min(300.0, getattr(config, "account_conversation_read_interval_secs", 0.0)
                   * 2 ** min(failures, 4))
    # A real limit disables bursts until the existing backoff history expires.
    capacity = 1 if failures or not interval else getattr(config, "account_conversation_read_burst", 1)
    return interval, capacity


def _read_credit(bucket, now, interval, capacity, floor):
    """Project the existing account clock without mutating/reserving a read.

    A policy change never grants new credits: retain at most one old credit,
    then accrue elapsed time at the current interval. R=0 records no credit.
    Legacy clocks retain their entire floor and earn credit only while idle.
    """
    if not interval:
        return 0.0, now, max(now, floor)
    if bucket is None:
        credit, at = 1.0, floor
    else:
        credit, at = bucket["credit"], bucket["at"]
    credit = min(float(capacity), credit + max(0.0, now - at) / interval)
    if bucket is not None and (bucket["interval"], bucket["capacity"]) != (interval, capacity):
        # Clamp AFTER accrual: old idle time cannot fill a newly enlarged burst.
        # Keep the old timestamp for the read-only projection; resetting it to
        # now on every snapshot would move the deadline forever. The first
        # admitted read persists the new policy and its fresh refill timestamp.
        credit = min(credit, 1.0)
    at = max(now, at)
    return credit, at, max(floor, at + max(0.0, 1.0 - credit) * interval)


def _checked_read_bucket(bucket, offset=0.0):
    if bucket is None:
        return None
    if (not isinstance(bucket, dict) or set(bucket) != {"credit", "at", "interval", "capacity"}
            or any(type(bucket[k]) not in (int, float) or not math.isfinite(bucket[k])
                   for k in ("credit", "at", "interval"))
            or type(bucket["capacity"]) is not int or not 1 <= bucket["capacity"] <= 100
            or not 0 <= bucket["credit"] <= bucket["capacity"] or not 0 <= bucket["interval"] <= 300):
        raise ValueError("Invalid saved conversation read credits")
    return {**bucket, "at": bucket["at"] - offset}


class AccountRequestClock:
    def __init__(self, account_key="", state_path: Path | None = None) -> None:
        self.account_key = account_key
        self.state_path = state_path
        self.lock = ProcessMutex(state_path.with_suffix(".lock") if state_path else None, self._load)
        self.turn_lock = ProcessMutex(state_path.with_suffix(".turn.lock") if state_path else None)
        self.next_request = 0.0
        self.next_turn = 0.0
        self.next_conversation_read = 0.0
        self.conversation_read_bucket = None
        self.archive_read_owner = None
        self.archive_read_until = 0.0
        self.ordinary_read_wait_until = 0.0
        self.ordinary_read_queue = []
        self.last_read_was_archive = False
        self.cooldown_until = 0.0
        self.rate_failures = 0
        self.last_rate_limit = 0.0
        self.conversation_read_rate_failures = 0
        self.last_conversation_read_rate_limit = 0.0
        self.last_rate_limit_evidence = None
        self.last_turn_started = None
        self._load()

    def _load(self):
        if self.state_path is None or not self.state_path.exists():
            return
        saved = json.loads(self.state_path.read_text())
        offset = time.time() - time.monotonic()
        for field in ("next_request", "next_turn", "cooldown_until", "last_rate_limit"):
            value = float(saved[field])
            if not math.isfinite(value):
                raise ValueError("Invalid saved account request clock")
            setattr(self, field, value - offset)
        self.rate_failures = max(0, int(saved["rate_failures"]))
        # Old clocks may contain mixed limit history. Keep their global backoff;
        # the last evidence alone cannot reconstruct all previous limit phases.
        self.conversation_read_rate_failures = max(0, int(saved.get("conversation_read_rate_failures", 0)))
        read_limit_at = float(saved.get("last_conversation_read_rate_limit", 0.0))
        if not math.isfinite(read_limit_at):
            raise ValueError("Invalid saved conversation read limit")
        self.last_conversation_read_rate_limit = read_limit_at - offset
        self.last_rate_limit_evidence = saved.get("last_rate_limit_evidence")
        read_at = float(saved.get("next_conversation_read", 0.0))
        if not math.isfinite(read_at):
            raise ValueError("Invalid saved conversation read clock")
        self.next_conversation_read = read_at - offset
        self.conversation_read_bucket = _checked_read_bucket(saved.get("conversation_read_bucket"), offset)
        for field in ("archive_read_until", "ordinary_read_wait_until"):
            value = float(saved.get(field, 0.0))
            if not math.isfinite(value):
                raise ValueError("Invalid saved read reservation")
            setattr(self, field, value - offset)
        self.archive_read_owner = saved.get("archive_read_owner")
        self.ordinary_read_queue = []
        for entry in saved.get("ordinary_read_queue", []):
            until = float(entry["until"])
            if not isinstance(entry.get("owner"), str) or not math.isfinite(until):
                raise ValueError("Invalid saved result read reservation")
            self.ordinary_read_queue.append({"owner": entry["owner"], "until": until - offset})
        self.last_read_was_archive = saved.get("last_read_was_archive") is True
        started = saved.get("last_turn_started")
        if started is not None:
            self.last_turn_started = float(started) - offset

    def _save(self):
        if self.state_path is None:
            return
        offset = time.time() - time.monotonic()
        saved = {field: getattr(self, field) + offset for field in
                 ("next_request", "next_turn", "next_conversation_read", "cooldown_until", "last_rate_limit")}
        saved["rate_failures"] = self.rate_failures
        saved["conversation_read_rate_failures"] = self.conversation_read_rate_failures
        saved["last_conversation_read_rate_limit"] = self.last_conversation_read_rate_limit + offset
        saved["last_rate_limit_evidence"] = self.last_rate_limit_evidence
        saved.update(archive_read_owner=self.archive_read_owner,
                     archive_read_until=self.archive_read_until + offset,
                     ordinary_read_wait_until=self.ordinary_read_wait_until + offset,
                     last_read_was_archive=self.last_read_was_archive)
        saved["ordinary_read_queue"] = [
            {"owner": entry["owner"], "until": entry["until"] + offset}
            for entry in self.ordinary_read_queue]
        saved["last_turn_started"] = None if self.last_turn_started is None else self.last_turn_started + offset
        if self.conversation_read_bucket is not None:
            saved["conversation_read_bucket"] = {**self.conversation_read_bucket,
                                                  "at": self.conversation_read_bucket["at"] + offset}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            json.dump(saved, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.state_path)

    def _ordinary_read_turn(self, owner, now, delay):
        # Sleeping outside the send lock is essential, but waking readers must
        # not race for every read edge: a busy reader could starve other results.
        # Keep FIFO reservations in the existing cross-process clock. Renew at
        # each wake; a dead worker's reservation expires without clearing tasks.
        queue = [entry for entry in self.ordinary_read_queue if entry["until"] > now]
        changed = len(queue) != len(self.ordinary_read_queue)
        entry = next((entry for entry in queue if entry["owner"] == owner), None)
        if entry is None:
            entry = {"owner": owner, "until": now + max(0, delay) + 30}
            queue.append(entry)
            changed = True
        elif entry["until"] < now + max(0, delay) + 5:
            entry["until"] = now + max(0, delay) + 30
            changed = True
        self.ordinary_read_queue = queue
        self.ordinary_read_wait_until = max(entry["until"] for entry in queue)
        if changed:
            self._save()
        return queue[0]["owner"] == owner

    def _release_ordinary_read(self, owner):
        queue = [entry for entry in self.ordinary_read_queue if entry["owner"] != owner]
        if len(queue) == len(self.ordinary_read_queue):
            return
        self.ordinary_read_queue = queue
        self.ordinary_read_wait_until = max((entry["until"] for entry in queue), default=0.0)
        self._save()

    def _reserve_archive_read(self, owner, now):
        """Called with the existing clock lock; reserve one GET, never a POST."""
        if not getattr(config, "account_conversation_read_interval_secs", 0.0):
            return True
        ready = max(now, self.next_request, self.cooldown_until, self._read_ready(now))
        if ready - now >= 235:
            return False  # An archive HTTP step has a 240 second finite budget.
        if self.archive_read_owner and self.archive_read_until > now:
            return self.archive_read_owner == owner
        if self.last_read_was_archive and self.ordinary_read_wait_until > now:
            return False
        self.archive_read_owner = owner
        self.archive_read_until = ready + 5
        self._save()
        return True

    def _expire_backoff(self):
        now = time.monotonic()
        if self.rate_failures and now - self.last_rate_limit >= 900:
            self.rate_failures = 0
        if self.conversation_read_rate_failures and now - self.last_conversation_read_rate_limit >= 900:
            self.conversation_read_rate_failures = 0

    def _read_cooldown_until(self):
        if not self.conversation_read_rate_failures:
            return 0.0
        return self.last_conversation_read_rate_limit + _backoff_seconds(self.conversation_read_rate_failures)

    def _read_ready(self, now):
        interval, capacity = _read_policy(now, self.rate_failures, self.last_rate_limit,
                                         self.conversation_read_rate_failures, self.last_conversation_read_rate_limit)
        return max(self._read_cooldown_until(),
                   _read_credit(self.conversation_read_bucket, now, interval, capacity,
                                self.next_conversation_read)[2])

    def _consume_read(self, now):
        interval, capacity = _read_policy(now, self.rate_failures, self.last_rate_limit,
                                         self.conversation_read_rate_failures, self.last_conversation_read_rate_limit)
        credit, _, ready = _read_credit(self.conversation_read_bucket, now, interval, capacity,
                                       self.next_conversation_read)
        if ready > now + 0.000001 or self._read_cooldown_until() > now:
            raise AccountRequestDeadlineExceeded("conversation read credit is not ready")
        remaining = max(0.0, credit - 1.0) if interval else 0.0
        self.conversation_read_bucket = {"credit": remaining, "at": now,
                                         "interval": interval, "capacity": capacity}
        self.next_conversation_read = now + max(0.0, 1.0 - remaining) * interval

    def limited(self, retry_after=0.0, *, evidence=None, retry_after_present=False, read_sent_at=None):
        # A conversation GET limit without Retry-After backs off that read lane.
        # Explicit provider waits and limits from other/unknown phases retain
        # the account-wide protection. Successful unrelated HTTP never resets it.
        self._expire_backoff()
        read_only = ((evidence or {}).get("phase") == "conversation_read"
                     and not retry_after_present and retry_after <= 0)
        now = time.monotonic()
        same_read_incident = (read_only and read_sent_at is not None
                              and self.conversation_read_rate_failures > 0
                              and read_sent_at <= self.last_conversation_read_rate_limit)
        if read_only:
            # Reads already in flight when a limit was observed belong to that
            # incident. Count every response, but do not turn one parallel burst
            # into several exponential retries or extend its persisted deadline.
            if not same_read_incident:
                self.conversation_read_rate_failures += 1
                self.last_conversation_read_rate_limit = now
                self.next_conversation_read = max(
                    self.next_conversation_read, now + _backoff_seconds(self.conversation_read_rate_failures))
            failures = self.conversation_read_rate_failures
            wait = max(0.0, self._read_cooldown_until() - now)
        else:
            self.rate_failures += 1
            self.last_rate_limit = now
            failures = self.rate_failures
            wait = max(_backoff_seconds(failures), retry_after)
            self.cooldown_until = max(self.cooldown_until, now + wait)
        context = current_request.get()
        observed = {"layer": "upstream_chatgpt", "phase": "unknown", "origin": "http_429",
                    **(evidence or {}), "retry_after_seconds": retry_after,
                    "scope": "conversation_read" if read_only else "account",
                    "same_read_incident": same_read_incident,
                    "cooldown_seconds": wait,
                    "cooldown_until": time.time() + wait,
                    "observed_at": time.time(), "account": self.account_key}
        if context is not None:
            observed["request_ref"] = hashlib.sha256((context.owner + ":" + context.request_id).encode()).hexdigest()[:24]
        # Metadata/archiving calls have no generation context. Keep their
        # bounded, sanitized evidence with the cooldown across restarts too.
        self.last_rate_limit_evidence = observed
        self._save()
        if context is not None:
            context.record_limit(observed)
        logger.warning({"event": "account_rate_limited", "account": self.account_key,
                        "consecutive_limits": failures,
                        "retry_after_secs": retry_after, "cooldown_secs": wait, **observed})

    def request(self, send, method, url, **kwargs):
        io_cleanup = kwargs.pop("_account_request_io_cleanup", None)
        deadline_at = kwargs.pop("_account_request_deadline_monotonic", None)
        minimum_budget = kwargs.pop("_account_request_minimum_budget_secs", None)
        if type(minimum_budget) not in (int, float) or not math.isfinite(minimum_budget) or minimum_budget <= 0:
            minimum_budget = None
        local_wait = kwargs.pop("_account_request_local_wait", None)
        before_send = kwargs.pop("_account_request_before_send", None)
        preflight = kwargs.pop("_account_request_preflight", None)
        if not isinstance(deadline_at, (int, float)) or isinstance(deadline_at, bool):
            deadline_at = None

        def remaining_budget() -> float | None:
            if deadline_at is None:
                return None
            return float(deadline_at) - time.monotonic()

        def acquire_with_budget(lock) -> None:
            remaining = remaining_budget()
            if remaining is None:
                lock.acquire()
                return
            if remaining <= 0 or not lock.acquire(timeout=remaining):
                raise AccountRequestDeadlineExceeded("account request deadline elapsed before upstream send")

        def cap_timeout_before_send() -> None:
            remaining = remaining_budget()
            if remaining is None:
                return
            if minimum_budget is not None and remaining < minimum_budget:
                raise AccountReadRetryBudgetInsufficient("original read connection budget unavailable")
            if remaining <= 0:
                raise AccountRequestDeadlineExceeded("account request deadline elapsed before upstream send")
            timeout = kwargs.get("timeout")
            if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
                kwargs["timeout"] = max(0.001, min(float(timeout), remaining))

        read_wait_seconds = {}
        read_http_attempts = 0
        read_queue_position_max = 0

        def wait_for_pace(delay: float, message: str, reason="account_pace") -> None:
            nonlocal deadline_at
            remaining = remaining_budget()
            # Credit only our configured pacing wait. A provider cooldown is
            # still bounded by the caller's deadline and is never shortened.
            provider_wait = max(self.cooldown_until, self._read_cooldown_until() if is_conversation_read else 0)
            credit = callable(local_wait) and provider_wait <= time.monotonic()
            if remaining is not None and (remaining <= 0 or (not credit and delay >= remaining)):
                raise AccountRequestDeadlineExceeded(message)
            if (remaining is not None and minimum_budget is not None and not credit
                    and remaining - delay < minimum_budget):
                # No upstream attempt can fit after this known pacing wait.
                # Return the existing not-sent deferral now, preserving the
                # clock and leaving the caller to retry the same original read.
                raise AccountReadRetryBudgetInsufficient("original read connection budget unavailable")
            started_wait = time.monotonic()
            time.sleep(delay)
            elapsed = max(0.0, time.monotonic() - started_wait)
            if is_conversation_read:
                read_wait_seconds[reason] = read_wait_seconds.get(reason, 0.0) + elapsed
            if credit and elapsed:
                local_wait(elapsed)
                if deadline_at is not None:
                    deadline_at += elapsed

        path = urlparse(str(url)).path.rstrip("/")
        is_turn = str(method).upper() == "POST" and (path.endswith("/conversation") or path.endswith("/responses"))
        context = current_request.get()
        phase = "conversation" if is_turn else "prepare" if path.endswith("/conversation/prepare") else "account_read"
        if "/conversation/" in path and str(method).upper() == "PATCH":
            phase = "conversation_update"
            body = kwargs.get("json")
            if isinstance(body, dict) and isinstance(body.get("is_archived"), bool):
                phase = "conversation_archive" if body["is_archived"] else "conversation_restore"
        elif "/conversation/" in path and "/attachment/" not in path and str(method).upper() == "GET":
            phase = "conversation_read"
        is_conversation_read = phase == "conversation_read"
        # Only the three read-only account observations used by get_user_info
        # may receive outside the pacing lock. Other account_read operations
        # include writes and authentication; the phase alone is not sufficient.
        metadata_kind = {
            ("GET", "/backend-api/me"): "account_profile",
            ("GET", "/backend-api/accounts/check/v4-2023-04-27"): "account_subscription",
        }.get((str(method).upper(), path))
        body = kwargs.get("json")
        if (str(method).upper() == "POST" and path == "/backend-api/conversation/init"
                and isinstance(body, dict)
                and set(body) == {"gizmo_id", "requested_default_model", "conversation_id", "timezone_offset_min"}
                and all(body[k] is None for k in ("gizmo_id", "requested_default_model", "conversation_id"))
                and isinstance(body["timezone_offset_min"], (int, float))
                and not isinstance(body["timezone_offset_min"], bool)):
            metadata_kind = "account_limits"
        concurrent_io = is_conversation_read or metadata_kind is not None
        archive_guard = current_archive_guard.get() if phase in {
            "conversation_read", "conversation_archive", "conversation_restore"} else None
        read_owner = current_archive_read_owner.get() if archive_guard is not None else None
        ordinary_owner = None
        if is_conversation_read and not read_owner:
            # Due image recovery may already hold a FIFO place before its
            # worker starts. Reuse it at the actual GET; do not queue behind
            # our own reservation. The original image claim prevents overlap.
            ordinary_owner = (_image_read_owner(context.owner, context.request_id)
                              if getattr(context, "kind", None) == "image" else uuid.uuid4().hex)
        if archive_guard is not None:
            archive_guard()
            # Each HTTP step must finish inside the renewed 300s work claim.
            # Revalidate again after pacing, immediately before the send edge.
            archive_deadline = time.monotonic() + 240
            deadline_at = min(deadline_at, archive_deadline) if deadline_at is not None else archive_deadline
        raw_model = (kwargs.get("json") or {}).get("model") if isinstance(kwargs.get("json"), dict) else None
        model = raw_model if isinstance(raw_model, str) and len(raw_model) <= 160 else None
        request_ref = hashlib.sha256((context.owner + ":" + context.request_id).encode()).hexdigest()[:24] if context else None
        archive_observation = (current_archive_observation.get() or {}) if archive_guard is not None else {}
        request_ref = request_ref or archive_observation.get("request_ref")

        def observed_send(send_method, send_url, send_phase, *, read_started=None, **send_kwargs):
            nonlocal read_http_attempts
            # Count actual transport attempts, including metadata/preflight GETs.
            # Never log URLs, request bodies/headers, response bodies or exception
            # messages: conversation URLs and signed downloads can contain secrets.
            started_at, started = time.time(), time.monotonic()
            endpoint = urlparse(str(send_url)).path.rstrip("/")
            endpoint_kind = metadata_kind or send_phase
            if "/attachment/" in endpoint:
                endpoint_kind = "attachment"
            elif "/files/" in endpoint or endpoint.endswith("/files"):
                endpoint_kind = "files"
            elif endpoint.endswith("/tasks"):
                endpoint_kind = "tasks"
            elif endpoint.endswith("/models"):
                endpoint_kind = "models"
            elif "/sentinel/" in endpoint:
                endpoint_kind = "requirements"
            elif endpoint.endswith("/conversations"):
                endpoint_kind = "conversation_list"
            elif endpoint == "/backend-api/celsius/ws/user":
                endpoint_kind = "completion_subscription"
            response = None
            transport_error = None
            transport_code = None
            transport_snapshot = {}
            try:
                if send_phase == "conversation_read":
                    read_http_attempts += 1
                if read_started is not None:
                    read_started(started)
                response = send(send_method, send_url, **send_kwargs)
                return response
            except Exception as exc:
                # Exception messages/URLs may contain credentials. Retain only
                # known class names and a numeric transport code for diagnosis.
                names = {"TimeoutError", "Timeout", "ReadTimeout", "ConnectTimeout", "ConnectionError",
                         "ConnectionResetError", "SSLError", "ProxyError", "DNSError", "OSError",
                         "CurlError", "RequestsError", "RequestException", "CertificateVerifyError"}
                transport_error = next((cls.__name__ for cls in type(exc).__mro__
                                        if cls.__name__ in names), "other")
                try:
                    code = getattr(exc, "code", None)
                except Exception:
                    code = None
                if isinstance(code, int) and not isinstance(code, bool) and 0 <= code <= 999:
                    transport_code = int(code)
                try:
                    transport_snapshot = _transport_snapshot(getattr(exc, "response", None))
                except Exception:
                    pass
                raise
            finally:
                verb = str(send_method).upper()
                status = getattr(response, "status_code", None)
                try:
                    timeout = send_kwargs.get("timeout")
                    timeout = (float(timeout) if isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
                               and math.isfinite(timeout) and timeout > 0 else None)
                except (ValueError, TypeError, OverflowError):
                    timeout = None
                logger.info({"event": "account_http_attempt", "account": self.account_key,
                             "request_ref": request_ref, "layer": "upstream_chatgpt",
                             "work_ref": archive_observation.get("work_ref"),
                             "archive_step": current_archive_step.get() if archive_guard is not None else None,
                             "method": verb if verb in {"GET", "POST", "PATCH", "PUT", "DELETE", "HEAD", "OPTIONS"} else "OTHER",
                             "phase": send_phase, "endpoint_kind": endpoint_kind, "started_at": started_at,
                             "headers_elapsed_secs": round(time.monotonic() - started, 6),
                             "status_code": status if isinstance(status, int) and not isinstance(status, bool) else None,
                             "outcome": "response" if response is not None else "transport_error",
                             "transport_error_type": transport_error, "transport_error_code": transport_code,
                             **(transport_snapshot if response is None else _transport_snapshot(response)),
                             **({"response_features": _rate_limit_response_features(response)} if status == 429 else {}),
                             "request_timeout_secs": timeout,
                             "stream": bool(send_kwargs.get("stream"))})
        # Serialize only the send edge. The account activity reservation lives
        # in PoolAdmission until the response stream is terminal; holding this
        # file lock for the whole stream would silently force capacity back to 1.
        if is_turn:
            acquire_with_budget(self.turn_lock)
        pacing_held = is_turn
        turn_held = is_turn
        release_lock = threading.Lock()

        def release_pacing():
            nonlocal pacing_held
            with release_lock:
                if pacing_held:
                    pacing_held = False
                    self.turn_lock.release()

        def release_turn():
            nonlocal turn_held
            with release_lock:
                if turn_held:
                    turn_held = False
                    if context is not None:
                        context.release_turn()

        try:
            while True:
                acquire_with_budget(self.lock)
                try:
                    now = time.monotonic()
                    read_delay, wait_reason = max((
                        (max(0, self.next_request - now), "account_pace"),
                        (max(0, self._read_ready(now) - now) if is_conversation_read else 0, "read_rate"),
                        (max(0, self.cooldown_until - now,
                             self._read_cooldown_until() - now if is_conversation_read else 0), "upstream_cooldown")),
                        key=lambda item: (item[0], item[1] == "upstream_cooldown"))
                    if is_conversation_read and read_owner:
                        if not self._reserve_archive_read(read_owner, now):
                            other = self.archive_read_until if self.archive_read_until > now else 0.0
                            ordinary = self.ordinary_read_wait_until if self.last_read_was_archive else 0.0
                            # These leases bound abandoned readers, not the
                            # time a live reader needs. Recheck released claims
                            # promptly; the real HTTP/read/cooldown edge above
                            # still determines when an upstream send is legal.
                            retry_check = min(1.0, max(0.1, config.account_request_interval_secs))
                            reservation_delay = min(retry_check, max(0.0, other - now, ordinary - now))
                            if reservation_delay > read_delay:
                                wait_reason = "archive_reservation"
                            read_delay = max(read_delay, reservation_delay)
                    elif is_conversation_read:
                        if self.archive_read_owner and self.archive_read_until > now:
                            # A live archive may release this lease immediately.
                            # Recheck it promptly; retain the actual rate/cooldown floor.
                            retry_check = min(1.0, max(0.1, config.account_request_interval_secs))
                            reservation_delay = min(retry_check, self.archive_read_until - now)
                            if reservation_delay > read_delay:
                                wait_reason = "archive_reservation"
                            read_delay = max(read_delay, reservation_delay)
                        if not self._ordinary_read_turn(ordinary_owner, now, read_delay):
                            # Give the reserved reader time to wake. Do not
                            # consume another full upstream interval locally,
                            # or add a whole second to fractional HTTP pacing.
                            retry_check = min(1.0, max(0.1, config.account_request_interval_secs))
                            if retry_check > read_delay:
                                wait_reason = "result_fifo"
                            read_delay = max(read_delay, retry_check)
                        read_queue_position_max = max(read_queue_position_max,
                            next((i + 1 for i, entry in enumerate(self.ordinary_read_queue)
                                  if entry["owner"] == ordinary_owner), 0))
                except BaseException:
                    self.lock.release()
                    raise
                if not concurrent_io or read_delay <= 0:
                    break
                # Waiting for a safe read must not occupy the shared send-edge lock
                # and delay a generation POST that is otherwise ready. Reload
                # all deadlines under the cross-process lock after waking.
                self.lock.release()
                wait_for_pace(read_delay, "account request deadline elapsed during read wait", wait_reason)
            clock_held = True
            response = None
            try:
                self._expire_backoff()
                ready = max(self.next_request, self.cooldown_until,
                            self.next_turn if is_turn else 0.0)
                delay = ready - time.monotonic()
                if delay > 0:
                    wait_for_pace(delay, "account request deadline elapsed during cooldown wait")
                if preflight is not None:
                    # A superseded original may arrive during the cooldown.
                    # This GET shares the held pacing lock and raw transport;
                    # recursively calling the paced session would deadlock.
                    def read_original(read_method, read_url, **read_kwargs):
                        if str(read_method).upper() != "GET":
                            raise ValueError("submission preflight must be read-only")
                        remaining = remaining_budget()
                        if remaining is not None:
                            if remaining <= 0:
                                raise AccountRequestDeadlineExceeded("preflight deadline elapsed")
                            read_kwargs["timeout"] = min(float(read_kwargs.get("timeout", 60)), remaining)
                        self.next_request = time.monotonic() + min(60.0, config.account_request_interval_secs * 2 ** min(self.rate_failures, 4))
                        self._save()
                        sent_at = time.monotonic()
                        try:
                            response = observed_send(read_method, read_url, "conversation_preflight", **read_kwargs)
                        finally:
                            # Persisting the reservation can itself take time.
                            # Correct its floor from the actual transport edge
                            # before another caller can acquire this clock.
                            self.next_request = max(self.next_request, sent_at + min(60.0, config.account_request_interval_secs * 2 ** min(self.rate_failures, 4)))
                            self._save()
                        if response.status_code == 429:
                            request_id = (response.headers.get("x-request-id") or response.headers.get("openai-request-id"))
                            safe_id = request_id if isinstance(request_id, str) and len(request_id) <= 160 and request_id.isascii() and not any(c.isspace() for c in request_id) else None
                            self.limited(retry_after_seconds(response.headers.get("Retry-After")),
                                         retry_after_present="Retry-After" in response.headers,
                                         evidence={"phase": "conversation_preflight", "model": model, "origin": "http_429", "upstream_request_id": safe_id,
                                                   "response_features": _rate_limit_response_features(response)})
                        return response
                    preflight(read_original)
                    # The metadata GET does not bypass the account request pace.
                    delay = max(self.next_request, self.cooldown_until, self.next_turn if is_turn else 0.0) - time.monotonic()
                    if delay > 0:
                        wait_for_pace(delay, "deadline elapsed after preflight")
                if context is not None and is_turn:
                    context.before_send()
                if callable(before_send):
                    before_send()
                if archive_guard is not None:
                    archive_guard()
                # Receipt persistence and final fences must not consume the
                # interval reserved for the following upstream request.
                now = time.monotonic()
                cap_timeout_before_send()
                factor = 2 ** min(self.rate_failures, 4)
                self.next_request = now + min(60.0, config.account_request_interval_secs * factor)
                read_correction_applied = False
                def correct_read_start(sent_at):
                    nonlocal read_correction_applied
                    if is_conversation_read and not read_correction_applied:
                        # Still under the send mutex: no later reader has consumed
                        # credit yet. Never repeat this after reloading late replies.
                        delay = max(0.0, sent_at - now)
                        self.conversation_read_bucket["at"] += delay
                        self.next_conversation_read += delay
                        read_correction_applied = True
                if is_conversation_read:
                    self._consume_read(now)
                    self.last_read_was_archive = archive_guard is not None
                    if read_owner and self.archive_read_owner == read_owner:
                        self.archive_read_owner = None
                        self.archive_read_until = 0.0
                    if ordinary_owner:
                        self._release_ordinary_read(ordinary_owner)
                        ordinary_owner = None
                if is_turn:
                    self.next_turn = now + min(300.0, config.account_message_interval_secs * factor)
                    logger.info({"event": "account_message_start", "account": self.account_key,
                                 "since_previous_secs": None if self.last_turn_started is None else round(now - self.last_turn_started, 3),
                                 "minimum_interval_secs": min(300.0, config.account_message_interval_secs * factor),
                                 "layer": "upstream_chatgpt", "phase": phase, "model": model,
                                 "request_ref": request_ref, **(context.log_fields() if context else {})})
                    self.last_turn_started = now
                # Reserve the interval before sending. Restarting after an
                # unknown response must not erase the account's wait period.
                self._save()
                # Saving pacing state and the submission receipt can consume
                # part of the declared budget; cap once more at the send edge.
                cap_timeout_before_send()
                if context is not None and is_turn and hasattr(context, "record_stage"):
                    context.record_stage("send_call_started")
                sent_at = time.monotonic()
                try:
                    if concurrent_io:
                        # Keep the durable clock through the local transport-call
                        # edge, not through the network response. Other original
                        # conversations may read once their own start floor is due.
                        # The caller still joins this one timeout-limited request;
                        # no retry, detached task or new scheduling queue is added.
                        entered = threading.Event()
                        outcome = {}
                        def read_started(at):
                            outcome["started_at"] = at
                            entered.set()
                        def read_io():
                            try:
                                # Thread scheduling consumes the same pre-send
                                # budget; an expired caller must not send later.
                                cap_timeout_before_send()
                                outcome["response"] = observed_send(method, url, phase,
                                    read_started=read_started, **kwargs)
                            except BaseException as exc:
                                outcome["error"] = exc
                            finally:
                                if callable(io_cleanup):
                                    try:
                                        io_cleanup()
                                    except Exception:
                                        logger.warning({"event": "account_read_transport_cleanup_failed",
                                                        "account": self.account_key})
                                entered.set()
                        worker_context = copy_context()
                        worker = threading.Thread(target=worker_context.run, args=(read_io,),
                                                  name="account-metadata-read" if metadata_kind else "original-conversation-read")
                        worker.start()
                        reservation_error = None
                        floor_durable = False
                        try:
                            entered.wait()
                            if "started_at" not in outcome:
                                raise outcome["error"]
                            sent_at = outcome["started_at"]
                            self.next_request = max(self.next_request, sent_at + min(60.0, config.account_request_interval_secs * factor))
                            correct_read_start(sent_at)
                            self._save()
                            floor_durable = True
                        except BaseException as exc:
                            reservation_error = exc
                        # Reacquisition reloads the newest cross-process state.
                        # A late 200 cannot erase another read's newer 429/queue.
                        # On a failed correction retain the lock until the
                        # joined response/final save; the old floor is too early.
                        if floor_durable:
                            self.lock.release()
                            clock_held = False
                        try:
                            # Reap the transport even on caller interruption;
                            # never leave a late GET running after this returns.
                            while worker.is_alive():
                                try:
                                    worker.join()
                                except BaseException as exc:
                                    reservation_error = reservation_error or exc
                        finally:
                            response = outcome.get("response")
                            # This is result/cooldown reconciliation, not a new
                            # send admission. It can outlast the send budget;
                            # discarding a received 429 would lose its cooldown.
                            if not clock_held:
                                self.lock.acquire()
                                clock_held = True
                        if "error" in outcome:
                            if reservation_error is not None:
                                raise reservation_error from outcome["error"]
                            raise outcome["error"]
                    else:
                        response = observed_send(method, url, phase, **kwargs)
                finally:
                    # Keep the pre-send durable reservation for crash safety,
                    # then account for its I/O delay even if transport fails.
                    if clock_held:
                        self.next_request = max(self.next_request, sent_at + min(60.0, config.account_request_interval_secs * factor))
                        correct_read_start(sent_at)
                        if is_turn:
                            self.next_turn = max(self.next_turn, sent_at + min(300.0, config.account_message_interval_secs * factor))
                            self.last_turn_started = sent_at
                        self._save()
                release_pacing()
                response_headers = getattr(response, "headers", {}) or {}
                upstream_id = response_headers.get("x-request-id") or response_headers.get("openai-request-id")
                safe_id = upstream_id if isinstance(upstream_id, str) and len(upstream_id) <= 160 and upstream_id.isascii() and not any(c.isspace() for c in upstream_id) else None
                if response.status_code == 429:
                    self.limited(retry_after_seconds(response_headers.get("Retry-After")),
                                 retry_after_present="Retry-After" in response_headers,
                                 read_sent_at=sent_at if is_conversation_read else None,
                                 evidence={"phase": phase, "model": model, "origin": "http_429", "upstream_request_id": safe_id,
                                           "response_features": _rate_limit_response_features(response)})
                if concurrent_io and reservation_error is not None:
                    response.close()
                    raise reservation_error
                if context is not None and is_turn and hasattr(context, "record_stage"):
                    context.record_stage("response_headers_received", status_code=response.status_code,
                                         upstream_request_id=safe_id)
            finally:
                if not clock_held and response is not None:
                    response.close()
                if clock_held:
                    self.lock.release()
            if is_turn and kwargs.get("stream") and 200 <= response.status_code < 300:
                close = response.close
                def close_turn():
                    try:
                        return close()
                    finally:
                        release_turn()
                response.close = close_turn
                # Some upstream limits arrive inside HTTP-200 SSE. Observe only
                # explicit error envelopes, never the assistant's text content.
                lines = response.iter_lines
                def paced_lines(*args, **line_kwargs):
                    limited = False
                    first_output = False
                    try:
                        for line in lines(*args, **line_kwargs):
                            if not first_output and line:
                                first_output = True
                                if context is not None and hasattr(context, "record_stage"):
                                    context.record_stage("first_output")
                            if not limited and rate_limited_event(line):
                                with self.lock:
                                    self.limited(rate_limit_retry_after(line), evidence={"phase": "conversation_stream", "model": model, "origin": "sse_rate_limit", "upstream_request_id": safe_id})
                                limited = True
                            yield line
                    finally:
                        close_turn()
                response.iter_lines = paced_lines
            else:
                release_turn()
            return response
        except BaseException:
            release_pacing()
            release_turn()
            raise
        finally:
            if is_conversation_read:
                logger.info({"event": "account_read_wait_finished", "account": self.account_key,
                    "request_ref": request_ref, "work_ref": archive_observation.get("work_ref"),
                    "archive_step": current_archive_step.get() if archive_guard is not None else None,
                    "observed_at": time.time(), "http_attempts": read_http_attempts,
                    "queue_position_max": read_queue_position_max,
                    "wait_seconds_by_controlling_reason": {k: round(v, 6) for k, v in read_wait_seconds.items()}})
            if ordinary_owner:
                # Deadline/preflight/transport failure must not leave a live
                # caller's place blocking the following result reader.
                try:
                    with self.lock:
                        self._release_ordinary_read(ordinary_owner)
                except Exception as exc:
                    logger.warning({"event": "result_read_reservation_cleanup_failed",
                                    "account": self.account_key, "error_type": type(exc).__name__})


def rate_limited_event(line) -> bool:
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    if not isinstance(line, str) or not line.startswith("data:"):
        return False
    try:
        event = json.loads(line[5:].strip())
    except (ValueError, TypeError):
        return False
    if not isinstance(event, dict):
        return False
    error = event.get("error")
    if not error and event.get("type") == "error":
        error = event
    if not isinstance(error, dict):
        return False
    code = str(error.get("code") or error.get("type") or "").lower()
    message = str(error.get("message") or "").lower()
    return code in {"rate_limit_exceeded", "rate_limit_error", "too_many_requests"} or "too many requests" in message


def retry_after_seconds(value) -> float:
    if not isinstance(value, str):
        return 0.0
    try:
        seconds = float(value)
        return seconds if math.isfinite(seconds) and seconds > 0 else 0.0
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0.0


def rate_limit_retry_after(line) -> float:
    if not rate_limited_event(line):
        return 0.0
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    event = json.loads(line[5:].strip())
    error = event.get("error") or event
    value = error.get("retry_after_seconds", error.get("retry_after"))
    return retry_after_seconds(str(value)) if value is not None else 0.0


def account_pacing_snapshot(account, now=None, *, include_turn=True, include_conversation_read=False):
    """Read the original clock without booking or advancing a send interval."""
    now = time.time() if now is None else now
    identity = str(account.get("account_id") or account.get("provider_account_identity") or account.get("access_token") or "")
    key = hashlib.sha256(identity.encode()).hexdigest()
    path = DATA_DIR / "account_request_clocks" / f"{key}.json"
    try:
        saved = json.loads(path.read_text())
    except FileNotFoundError:
        return {"next_at": now, "cooldown_until": None}
    except (OSError, ValueError):
        return {"next_at": None, "cooldown_until": None}
    fields = ("next_request", "next_turn", "cooldown_until") if include_turn else ("next_request", "cooldown_until")
    values = [saved.get(field) for field in fields]
    if include_conversation_read:
        values.append(saved.get("next_conversation_read", 0.0))
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        return {"next_at": None, "cooldown_until": None}
    cooldown = saved["cooldown_until"]
    if include_conversation_read:
        try:
            failures = max(0, int(saved.get("conversation_read_rate_failures", 0)))
            limited_at = float(saved.get("last_conversation_read_rate_limit", 0.0))
            if not math.isfinite(limited_at):
                raise ValueError("Invalid saved conversation read limit")
        except (TypeError, ValueError, OverflowError):
            return {"next_at": None, "cooldown_until": None}
        if failures:
            cooldown = max(cooldown, limited_at + _backoff_seconds(failures))
        try:
            last_global_limit = float(saved.get("last_rate_limit", 0.0))
            if not math.isfinite(last_global_limit):
                raise ValueError("Invalid saved account rate limit")
            interval, capacity = _read_policy(now, max(0, int(saved.get("rate_failures", 0))),
                                             last_global_limit, failures, limited_at)
            bucket = _checked_read_bucket(saved.get("conversation_read_bucket"))
            values.append(_read_credit(bucket, now, interval, capacity,
                                       saved.get("next_conversation_read", 0.0))[2])
        except (TypeError, ValueError, OverflowError, KeyError):
            return {"next_at": None, "cooldown_until": None}
    return {"next_at": max(now, cooldown, *values), "cooldown_until": cooldown}


_clocks: dict[str, AccountRequestClock] = {}
_clocks_lock = threading.Lock()


def _image_read_owner(owner, request_id):
    return "image:" + hashlib.sha256(json.dumps([owner, request_id]).encode()).hexdigest()


def reserve_account_image_recovery_read(account, owner, request_id):
    """Join the existing read FIFO without an I/O worker or consuming credit.

    Called only for due original image recovery, outside account/SQLite locks.
    Each scan renews the existing short lease; a stopped/dead task expires by
    the same rule as other readers. The transport still enforces every clock.
    """
    return _image_recovery_read_reservation(account, owner, request_id, cancel=False)


def release_account_image_recovery_read(account, owner, request_id):
    return _image_recovery_read_reservation(account, owner, request_id, cancel=True)


def _image_recovery_read_reservation(account, owner, request_id, *, cancel):
    identity = str(account.get("account_id") or account.get("provider_account_identity") or account.get("access_token") or "")
    if not identity:
        return False
    key = hashlib.sha256(identity.encode()).hexdigest()
    try:
        with _clocks_lock:
            clock = _clocks.get(key)
            if clock is None:
                clock = AccountRequestClock(key[:12], DATA_DIR / "account_request_clocks" / f"{key}.json")
                _clocks[key] = clock
        if not clock.lock.acquire(blocking=False):
            return False
        try:
            read_owner = _image_read_owner(owner, request_id)
            if cancel:
                clock._release_ordinary_read(read_owner)
                return True
            now = time.monotonic()
            if max(clock.cooldown_until, clock._read_cooldown_until()) > now:
                clock._release_ordinary_read(read_owner)
                return False
            delay = max(0, clock.next_request - now, clock._read_ready(now) - now)
            if not getattr(config, "account_conversation_read_interval_secs", 0.0):
                clock._release_ordinary_read(read_owner)
                return delay <= 0
            first = clock._ordinary_read_turn(read_owner, now, delay)
            return first and delay <= 0 and not (clock.archive_read_owner and clock.archive_read_until > now)
        finally:
            clock.lock.release()
    except (OSError, ValueError):
        return False


def reserve_account_archive_read(account, owner):
    """Best-effort booking outside SQLite transactions; never wait on HTTP."""
    identity = str(account.get("account_id") or account.get("provider_account_identity") or account.get("access_token") or "")
    if not identity or not getattr(config, "account_conversation_read_interval_secs", 0.0):
        return False
    key = hashlib.sha256(identity.encode()).hexdigest()
    try:
        with _clocks_lock:
            clock = _clocks.get(key)
            if clock is None:
                clock = AccountRequestClock(key[:12], DATA_DIR / "account_request_clocks" / f"{key}.json")
                _clocks[key] = clock
        if not clock.lock.acquire(blocking=False):
            return False
        try:
            return clock._reserve_archive_read(owner, time.monotonic())
        finally:
            clock.lock.release()
    except (OSError, ValueError):
        return False


def _send_with_bounded_stream_close(session, send, method, url, **kwargs):
    """Bound native streaming I/O before perform; close without waiting for a tail.

    curl_cffi's synchronous close waits on its streaming Future. A confirmed
    result must not wait for a silent SSE connection, but its native handle
    must only be freed after perform exits. The stream clone gets a hard total
    timeout and the original finalizer runs from the completed Future.
    """
    from curl_cffi import CurlOpt
    from curl_cffi.requests import Session
    from curl_cffi.requests.models import STREAM_END

    timeout = kwargs.get("timeout")
    if (not kwargs.get("stream") or not isinstance(session, Session)
            or not getattr(session, "_use_thread_local_curl", False)
            or type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
            or any(option in session.curl_options for option in (CurlOpt.TIMEOUT, CurlOpt.TIMEOUT_MS))):
        return send(method, url, **kwargs)
    # Same sending thread, before the library duplicates this idle handle.
    # Do not mutate a live stream handle or shared session options.
    source = session.curl
    source.setopt(CurlOpt.TIMEOUT_MS, max(1, int(timeout * 1000)))
    try:
        response = send(method, url, **kwargs)
    finally:
        source.setopt(CurlOpt.TIMEOUT_MS, 0)
    future, quit_now, queue = response.stream_task, response.quit_now, response.queue
    finalize = response._finalize_stream
    lock, requested = threading.Lock(), False

    def cleanup(_future):
        try:
            finalize()
        except Exception as exc:
            logger.warning({"event": "stream_cleanup_failed", "error_type": type(exc).__name__})
        finally:
            # A failed Future can make the library finalizer raise before close.
            # perform has ended before this callback; releasing here is safe.
            response.curl.close()

    def request_close():
        nonlocal requested
        with lock:
            if requested:
                return
            requested = True
        quit_now.set()
        queue.put_nowait(STREAM_END)
        future.add_done_callback(cleanup)

    # iter_content calls _finalize_stream directly, not through close().
    response._finalize_stream = request_close
    return response


def pace_account_session(session, account: dict, access_token: str) -> None:
    if not access_token:
        return
    from curl_cffi.requests import Session
    if isinstance(session, Session):
        # Both supported curl_cffi versions snapshot these before reset, on
        # success and on exceptions. This does not issue additional requests.
        session.curl_infos = list(dict.fromkeys([*session.curl_infos, *_TRANSPORT_INFO_FIELDS]))
    # Account identity survives access-token refresh. No raw credential is
    # retained in the clock registry, logs or exceptions.
    identity = str(account.get("account_id") or account.get("provider_account_identity") or access_token)
    key = hashlib.sha256(identity.encode()).hexdigest()
    with _clocks_lock:
        clock = _clocks.get(key)
        if clock is None:
            clock = AccountRequestClock(key[:12], DATA_DIR / "account_request_clocks" / f"{key}.json")
            _clocks[key] = clock
    raw_send = session.request

    def close_read_transport():
        # Buffered GETs run on a joined, short-lived I/O thread. Session.close
        # on the caller cannot close this thread-local handle; retained errors
        # can otherwise keep it alive through a traceback cycle until GC.
        # Never close the Session or any other thread's stream/handle here.
        if isinstance(session, Session) and session._use_thread_local_curl:
            local = session._local
            curl = getattr(local, "curl", None)
            if curl is not None:
                try:
                    curl.close()
                finally:
                    del local.curl

    def send(method, url, **kwargs):
        connect = kwargs.pop("_account_request_connect_timeout_secs", None)
        timeout = kwargs.get("timeout")
        if (str(method).upper() == "GET" and not kwargs.get("stream")
                and type(connect) in (int, float) and math.isfinite(connect) and connect > 0
                and type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0):
            # Convert only AFTER the clock has capped the remaining budget.
            # curl_cffi sums this pair for TIMEOUT_MS; never add connect time
            # to the caller's existing deadline or mutate shared curl options.
            connect = min(connect, timeout)
            kwargs["timeout"] = (connect, timeout - connect)
        return _send_with_bounded_stream_close(session, raw_send, method, url, **kwargs)

    def paced_request(method, url, **kwargs):
        # Object-storage downloads are not ChatGPT account API calls.
        if urlparse(str(url)).hostname != "chatgpt.com":
            # Image upload/download still has a caller deadline, but no
            # account-clock wait. Consume our private option before requests.
            deadline = kwargs.pop("_account_request_deadline_monotonic", None)
            kwargs.pop("_account_request_local_wait", None)
            if isinstance(deadline, (int, float)) and not isinstance(deadline, bool):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AccountRequestDeadlineExceeded("image transfer deadline elapsed before send")
                timeout = kwargs.get("timeout")
                if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
                    kwargs["timeout"] = min(float(timeout), remaining)
            return send(method, url, **kwargs)
        path = urlparse(str(url)).path.rstrip("/")
        if (str(method).upper() == "POST"
                and (path.endswith("/conversation") or path.endswith("/conversation/prepare") or path.endswith("/responses"))
                and str(account.get("type") or "").strip().lower() == "free"):
            raise RuntimeError("Free account messages are disabled")
        return clock.request(send, method, url,
                             _account_request_io_cleanup=close_read_transport, **kwargs)

    session.request = paced_request
