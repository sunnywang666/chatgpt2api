"""One upstream request clock per account, shared by text, images and polling.

This is pacing, not a promise of an upstream quota. Never retry a request here:
the owning task still decides whether an operation may safely be submitted.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from services.config import config


class AccountRequestClock:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.next_turn = 0.0
        self.cooldown_until = 0.0
        self.rate_failures = 0

    def request(self, send, method, url, **kwargs):
        path = urlparse(str(url)).path.rstrip("/")
        is_turn = str(method).upper() == "POST" and path.endswith("/conversation")
        # Hold through response headers so a 429 is visible before another
        # request on this account can start. Streaming content is not locked.
        with self.lock:
            ready = max(self.next_request, self.cooldown_until,
                        self.next_turn if is_turn else 0.0)
            delay = ready - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            now = time.monotonic()
            self.next_request = now + config.account_request_interval_secs
            if is_turn:
                self.next_turn = now + config.account_message_interval_secs
            response = send(method, url, **kwargs)
            if response.status_code == 429:
                self.rate_failures += 1
                fallback = min(900.0, 60.0 * (2 ** min(self.rate_failures - 1, 4)))
                delay = max(fallback, retry_after_seconds(response.headers.get("Retry-After")))
                self.cooldown_until = time.monotonic() + delay
            elif 200 <= response.status_code < 300:
                self.rate_failures = 0
            return response


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
        clock = _clocks.setdefault(key, AccountRequestClock())
    send = session.request

    def paced_request(method, url, **kwargs):
        # Object-storage downloads are not ChatGPT account API calls.
        if urlparse(str(url)).hostname != "chatgpt.com":
            return send(method, url, **kwargs)
        return clock.request(send, method, url, **kwargs)

    session.request = paced_request
