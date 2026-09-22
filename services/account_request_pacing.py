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
from pathlib import Path
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from services.config import DATA_DIR, config
from utils.log import logger
from services.request_context import current_request


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


class AccountRequestClock:
    def __init__(self, account_key="", state_path: Path | None = None) -> None:
        self.account_key = account_key
        self.state_path = state_path
        self.lock = ProcessMutex(state_path.with_suffix(".lock") if state_path else None, self._load)
        self.turn_lock = ProcessMutex(state_path.with_suffix(".turn.lock") if state_path else None)
        self.next_request = 0.0
        self.next_turn = 0.0
        self.cooldown_until = 0.0
        self.rate_failures = 0
        self.last_rate_limit = 0.0
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
        started = saved.get("last_turn_started")
        if started is not None:
            self.last_turn_started = float(started) - offset

    def _save(self):
        if self.state_path is None:
            return
        offset = time.time() - time.monotonic()
        saved = {field: getattr(self, field) + offset for field in
                 ("next_request", "next_turn", "cooldown_until", "last_rate_limit")}
        saved["rate_failures"] = self.rate_failures
        saved["last_turn_started"] = None if self.last_turn_started is None else self.last_turn_started + offset
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            json.dump(saved, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.state_path)

    def limited(self, retry_after=0.0, *, evidence=None):
        # A successful metadata GET does not prove that generation capacity has
        # recovered. Keep the shared backoff for a quiet 15-minute window.
        self.rate_failures += 1
        self.last_rate_limit = time.monotonic()
        fallback = min(900.0, 60.0 * (2 ** min(self.rate_failures - 1, 4)))
        self.cooldown_until = self.last_rate_limit + max(fallback, retry_after)
        self._save()
        context = current_request.get()
        observed = {"layer": "upstream_chatgpt", "phase": "unknown", "origin": "http_429",
                    **(evidence or {}), "retry_after_seconds": retry_after,
                    "cooldown_seconds": max(fallback, retry_after),
                    "cooldown_until": time.time() + max(fallback, retry_after),
                    "observed_at": time.time(), "account": self.account_key}
        if context is not None:
            observed["request_ref"] = hashlib.sha256((context.owner + ":" + context.request_id).encode()).hexdigest()[:24]
            context.record_limit(observed)
        logger.warning({"event": "account_rate_limited", "account": self.account_key,
                        "consecutive_limits": self.rate_failures,
                        "retry_after_secs": retry_after, "cooldown_secs": max(fallback, retry_after), **observed})

    def request(self, send, method, url, **kwargs):
        deadline_at = kwargs.pop("_account_request_deadline_monotonic", None)
        before_send = kwargs.pop("_account_request_before_send", None)
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
            if remaining <= 0:
                raise AccountRequestDeadlineExceeded("account request deadline elapsed before upstream send")
            timeout = kwargs.get("timeout")
            if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
                kwargs["timeout"] = max(0.001, min(float(timeout), remaining))

        path = urlparse(str(url)).path.rstrip("/")
        is_turn = str(method).upper() == "POST" and (path.endswith("/conversation") or path.endswith("/responses"))
        context = current_request.get()
        phase = "conversation" if is_turn else "prepare" if path.endswith("/conversation/prepare") else "account_read"
        raw_model = (kwargs.get("json") or {}).get("model") if isinstance(kwargs.get("json"), dict) else None
        model = raw_model if isinstance(raw_model, str) and len(raw_model) <= 160 else None
        request_ref = hashlib.sha256((context.owner + ":" + context.request_id).encode()).hexdigest()[:24] if context else None
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
            acquire_with_budget(self.lock)
            try:
                if self.rate_failures and time.monotonic() - self.last_rate_limit >= 900:
                    self.rate_failures = 0
                ready = max(self.next_request, self.cooldown_until,
                            self.next_turn if is_turn else 0.0)
                delay = ready - time.monotonic()
                if delay > 0:
                    remaining = remaining_budget()
                    if remaining is not None and delay >= remaining:
                        raise AccountRequestDeadlineExceeded("account request deadline elapsed during cooldown wait")
                    time.sleep(delay)
                now = time.monotonic()
                cap_timeout_before_send()
                factor = 2 ** min(self.rate_failures, 4)
                self.next_request = now + min(60.0, config.account_request_interval_secs * factor)
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
                remaining = remaining_budget()
                if remaining is not None and remaining <= 0:
                    raise AccountRequestDeadlineExceeded("account request deadline elapsed before upstream send")
                if context is not None and is_turn:
                    context.before_send()
                    if hasattr(context, "record_stage"):
                        context.record_stage("send_call_started")
                if callable(before_send):
                    before_send()
                # Saving pacing state and the submission receipt can consume
                # part of the declared budget; cap once more at the send edge.
                cap_timeout_before_send()
                response = send(method, url, **kwargs)
                release_pacing()
                response_headers = getattr(response, "headers", {}) or {}
                upstream_id = response_headers.get("x-request-id") or response_headers.get("openai-request-id")
                safe_id = upstream_id if isinstance(upstream_id, str) and len(upstream_id) <= 160 and upstream_id.isascii() and not any(c.isspace() for c in upstream_id) else None
                if response.status_code == 429:
                    self.limited(retry_after_seconds(response.headers.get("Retry-After")),
                                 evidence={"phase": phase, "model": model, "origin": "http_429", "upstream_request_id": safe_id})
                if context is not None and is_turn and hasattr(context, "record_stage"):
                    context.record_stage("response_headers_received", status_code=response.status_code,
                                         upstream_request_id=safe_id)
            finally:
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


def account_pacing_snapshot(account, now=None):
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
    values = [saved.get(field) for field in ("next_request", "next_turn", "cooldown_until")]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        return {"next_at": None, "cooldown_until": None}
    return {"next_at": max(now, *values), "cooldown_until": saved["cooldown_until"]}


_clocks: dict[str, AccountRequestClock] = {}
_clocks_lock = threading.Lock()


def pace_account_session(session, account: dict, access_token: str) -> None:
    if not access_token:
        return
    # Account identity survives access-token refresh. No raw credential is
    # retained in the clock registry, logs or exceptions.
    identity = str(account.get("account_id") or account.get("provider_account_identity") or access_token)
    key = hashlib.sha256(identity.encode()).hexdigest()
    with _clocks_lock:
        clock = _clocks.get(key)
        if clock is None:
            clock = AccountRequestClock(key[:12], DATA_DIR / "account_request_clocks" / f"{key}.json")
            _clocks[key] = clock
    send = session.request

    def paced_request(method, url, **kwargs):
        # Object-storage downloads are not ChatGPT account API calls.
        if urlparse(str(url)).hostname != "chatgpt.com":
            return send(method, url, **kwargs)
        path = urlparse(str(url)).path.rstrip("/")
        if (str(method).upper() == "POST"
                and (path.endswith("/conversation") or path.endswith("/conversation/prepare") or path.endswith("/responses"))
                and str(account.get("type") or "").strip().lower() == "free"):
            raise RuntimeError("Free account messages are disabled")
        return clock.request(send, method, url, **kwargs)

    session.request = paced_request
