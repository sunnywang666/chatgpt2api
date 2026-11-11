"""Serialize the existing JSON account authority across local workers."""
import fcntl
import os
import sys
import threading
import time

from utils.log import logger


class AccountStateLock:
    _SLOW_SECONDS = 1.0
    _CALLER_DEPTH = 8
    _ACCOUNT_MUTATIONS = frozenset({
        "_record_token_refresh_error", "_apply_refreshed_tokens", "_record_codex_token_refresh_error",
        "refresh_access_token", "refresh_codex_access_token", "_conversation_binding_for_token_locked",
        "_provider_account_identity_for_token_locked", "mark_text_used", "set_pool_account_enabled",
        "set_pool_account_label", "_refresh_pool_account_once", "_attach_codex_locked",
        "refresh_submitted_codex_observation", "import_owned_codex_authorization",
        "_import_verified_chat_account", "import_owned_account", "set_owned_account_enabled",
        "_add_account_payloads", "delete_accounts", "update_account", "_record_invalid_token_seen",
        "mark_image_result",
    })

    def __init__(self, owner, path=None):
        self.owner, self.path = owner, path
        self.lock = threading.RLock()
        self.local = threading.local()

    @staticmethod
    def _safe_now():
        try:
            return time.monotonic()
        except BaseException:
            return None

    @staticmethod
    def _milliseconds(start, end):
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return None
        return max(0.0, (end - start) * 1000)

    @classmethod
    def _caller_category(cls):
        frame = None
        try:
            frame = sys._getframe(1)
            for _ in range(cls._CALLER_DEPTH):
                if frame is None:
                    break
                name = frame.f_code.co_name
                if name == "claim_next":
                    return "claim_next"
                if name == "resource_snapshot":
                    return "resource_snapshot"
                if name in {"observe_product_capacity_limit", "product_capacity_limit"}:
                    return "capacity_read"
                if name in cls._ACCOUNT_MUTATIONS:
                    return "account_mutation"
                frame = frame.f_back
        except BaseException:
            pass
        finally:
            del frame
        return "other"

    @classmethod
    def _log_slow_outer_release(cls, timing, released_at):
        try:
            held_ms = cls._milliseconds(timing.get("acquired_at"), released_at)
            event = {
                "event": "account_state_lock_timing",
                "caller": timing.get("caller", "other"),
                "local_rlock_wait_ms": cls._milliseconds(timing.get("requested_at"), timing.get("acquired_at")),
                "flock_wait_ms": timing.get("flock_wait_ms"),
                "reload_ms": timing.get("reload_ms"),
                "held_ms": held_ms,
            }
            values = (event["local_rlock_wait_ms"], event["flock_wait_ms"], event["reload_ms"], held_ms)
            if any(isinstance(value, (int, float)) and value >= cls._SLOW_SECONDS * 1000 for value in values):
                logger.info(event)
        except BaseException:
            pass

    def acquire(self, blocking=True, timeout=-1):
        requested_at = self._safe_now()
        deadline = time.monotonic() + timeout if timeout >= 0 else None
        acquired = self.lock.acquire(blocking, timeout)
        if not acquired:
            return False
        acquired_at = self._safe_now()
        depth = getattr(self.local, "depth", 0)
        try:
            if depth == 0 and self.path is not None:
                fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                flock_requested_at = self._safe_now()
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if not blocking or (deadline is not None and time.monotonic() >= deadline):
                            os.close(fd)
                            self.lock.release()
                            return False
                        time.sleep(.01)
                self.local.fd = fd
                flock_acquired_at = self._safe_now()
                flock_wait_ms = self._milliseconds(flock_requested_at, flock_acquired_at)
                reload_ms = 0.0
                if hasattr(self.owner, "_accounts"):
                    reload_started_at = flock_acquired_at
                    previous = self.owner._accounts
                    fresh = self.owner._load_accounts()
                    by_id = {self.owner._stable_account_identity(a): token for token, a in fresh.items()}
                    for token, account in previous.items():
                        next_token = by_id.get(self.owner._stable_account_identity(account))
                        if next_token and next_token != token:
                            self.owner._token_aliases[token] = next_token
                            count = self.owner._image_inflight.pop(token, 0)
                            if count:
                                self.owner._image_inflight[next_token] = self.owner._image_inflight.get(next_token, 0) + count
                    self.owner._accounts = fresh
                    reload_ms = self._milliseconds(reload_started_at, self._safe_now())
            else:
                flock_wait_ms = reload_ms = 0.0
            if depth == 0:
                self.local.timing = {
                    "requested_at": requested_at,
                    "acquired_at": acquired_at,
                    "flock_wait_ms": flock_wait_ms,
                    "reload_ms": reload_ms,
                    "caller": self._caller_category(),
                }
            self.local.depth = depth + 1
            return True
        except BaseException:
            if depth == 0 and hasattr(self.local, "fd"):
                os.close(self.local.fd)
                del self.local.fd
            self.lock.release()
            raise

    def release(self):
        self.local.depth -= 1
        timing = None
        if self.local.depth == 0 and hasattr(self.local, "fd"):
            os.close(self.local.fd)
            del self.local.fd
        if self.local.depth == 0 and hasattr(self.local, "timing"):
            timing = self.local.timing
            del self.local.timing
        released_at = self._safe_now() if timing is not None else None
        self.lock.release()
        if timing is not None:
            self._log_slow_outer_release(timing, released_at)

    def _is_owned(self):
        return self.lock._is_owned()

    def _release_save(self):
        depth = self.local.depth
        for _ in range(depth):
            self.release()
        return depth

    def _acquire_restore(self, depth):
        for _ in range(depth):
            self.acquire()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()
