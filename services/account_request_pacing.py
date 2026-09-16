"""One upstream request clock per account, shared by text, images and polling.

This is pacing, not a promise of an upstream quota. Never retry a request here:
the owning task still decides whether an operation may safely be submitted.
"""
from __future__ import annotations

import hashlib
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


class AccountRequestClock:
    def __init__(self, account_key="", state_path: Path | None = None) -> None:
        self.account_key = account_key
        self.state_path = state_path
        self.lock = threading.Lock()
        self.turn_lock = threading.Lock()
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

    def limited(self, retry_after=0.0):
        # A successful metadata GET does not prove that generation capacity has
        # recovered. Keep the shared backoff for a quiet 15-minute window.
        self.rate_failures += 1
        self.last_rate_limit = time.monotonic()
        fallback = min(900.0, 60.0 * (2 ** min(self.rate_failures - 1, 4)))
        self.cooldown_until = self.last_rate_limit + max(fallback, retry_after)
        self._save()
        logger.warning({"event": "account_rate_limited", "account": self.account_key,
                        "consecutive_limits": self.rate_failures,
                        "retry_after_secs": retry_after, "cooldown_secs": max(fallback, retry_after)})

    def request(self, send, method, url, **kwargs):
        path = urlparse(str(url)).path.rstrip("/")
        is_turn = str(method).upper() == "POST" and path.endswith("/conversation")
        # Serialize complete message streams, while allowing paced readback for
        # the active turn. Never hold the request lock while waiting for a turn.
        if is_turn:
            self.turn_lock.acquire()
        held = is_turn
        release_lock = threading.Lock()

        def release():
            nonlocal held
            with release_lock:
                if held:
                    held = False
                    self.turn_lock.release()

        try:
            with self.lock:
                if self.rate_failures and time.monotonic() - self.last_rate_limit >= 900:
                    self.rate_failures = 0
                ready = max(self.next_request, self.cooldown_until,
                            self.next_turn if is_turn else 0.0)
                delay = ready - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                now = time.monotonic()
                factor = 2 ** min(self.rate_failures, 4)
                self.next_request = now + min(60.0, config.account_request_interval_secs * factor)
                if is_turn:
                    self.next_turn = now + min(300.0, config.account_message_interval_secs * factor)
                    logger.info({"event": "account_message_start", "account": self.account_key,
                                 "since_previous_secs": None if self.last_turn_started is None else round(now - self.last_turn_started, 3),
                                 "minimum_interval_secs": min(300.0, config.account_message_interval_secs * factor)})
                    self.last_turn_started = now
                # Reserve the interval before sending. Restarting after an
                # unknown response must not erase the account's wait period.
                self._save()
                response = send(method, url, **kwargs)
                if response.status_code == 429:
                    self.limited(retry_after_seconds(response.headers.get("Retry-After")))
            if is_turn and kwargs.get("stream") and 200 <= response.status_code < 300:
                close = response.close
                def close_turn():
                    try:
                        return close()
                    finally:
                        release()
                response.close = close_turn
                # Some upstream limits arrive inside HTTP-200 SSE. Observe only
                # explicit error envelopes, never the assistant's text content.
                lines = response.iter_lines
                def paced_lines(*args, **line_kwargs):
                    limited = False
                    try:
                        for line in lines(*args, **line_kwargs):
                            if not limited and rate_limited_event(line):
                                with self.lock:
                                    self.limited()
                                limited = True
                            yield line
                    finally:
                        close_turn()
                response.iter_lines = paced_lines
            else:
                release()
            return response
        except BaseException:
            release()
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
