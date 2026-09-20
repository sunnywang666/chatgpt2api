"""Serialize the existing JSON account authority across local workers."""
import fcntl
import os
import threading
import time


class AccountStateLock:
    def __init__(self, owner, path=None):
        self.owner, self.path = owner, path
        self.lock = threading.RLock()
        self.local = threading.local()

    def acquire(self, blocking=True, timeout=-1):
        deadline = time.monotonic() + timeout if timeout >= 0 else None
        acquired = self.lock.acquire(blocking, timeout)
        if not acquired:
            return False
        depth = getattr(self.local, "depth", 0)
        try:
            if depth == 0 and self.path is not None:
                fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
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
                if hasattr(self.owner, "_accounts"):
                    previous = self.owner._accounts
                    fresh = self.owner._load_accounts()
                    by_id = {a.get("provider_account_identity"): token for token, a in fresh.items() if a.get("provider_account_identity")}
                    for token, account in previous.items():
                        next_token = by_id.get(account.get("provider_account_identity"))
                        if next_token and next_token != token:
                            self.owner._token_aliases[token] = next_token
                            count = self.owner._image_inflight.pop(token, 0)
                            if count:
                                self.owner._image_inflight[next_token] = self.owner._image_inflight.get(next_token, 0) + count
                    self.owner._accounts = fresh
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
        if self.local.depth == 0 and hasattr(self.local, "fd"):
            os.close(self.local.fd)
            del self.local.fd
        self.lock.release()

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
