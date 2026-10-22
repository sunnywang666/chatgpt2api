"""The current original receipt at the upstream send boundary."""
from contextvars import ContextVar
from contextlib import contextmanager, nullcontext
from functools import wraps
import math
import time
import hashlib
import re

current_operation_deadline = ContextVar("provider_operation_deadline", default=None)
current_request = ContextVar("provider_original_request", default=None)
current_archive_guard = ContextVar("provider_archive_guard", default=None)
current_archive_read_owner = ContextVar("provider_archive_read_owner", default=None)
current_archive_observation = ContextVar("provider_archive_observation", default=None)
current_archive_step = ContextVar("provider_archive_step", default=None)
current_original_read_owner = ContextVar("provider_original_read_owner", default=None)
_FAIR_SOURCE = re.compile(r"^(?:user|company):[0-9a-f]{64}$")


class AdmissionLost(RuntimeError):
    """The original claim is no longer allowed to send."""


@contextmanager
def operation_deadline(deadline):
    """One transient monotonic budget; nested stages may only shorten it."""
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("finite operation deadline required")
    inherited = current_operation_deadline.get()
    token = current_operation_deadline.set(min(deadline, inherited) if inherited is not None else deadline)
    try:
        yield
    finally:
        current_operation_deadline.reset(token)


def operation_budget(seconds):
    """Bound a synchronous operation without abandoning a background writer."""
    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            budget = seconds(args[0]) if callable(seconds) else seconds
            with operation_deadline(time.monotonic() + budget):
                return function(*args, **kwargs)
        return run
    return decorate


def operation_remaining(maximum):
    deadline = current_operation_deadline.get()
    if deadline is None:
        return maximum
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        from services.account_request_pacing import AccountRequestDeadlineExceeded
        raise AccountRequestDeadlineExceeded("original operation deadline exhausted")
    return min(maximum, remaining)


def operation_request_options(maximum):
    options = {"timeout": operation_remaining(maximum)}
    deadline = current_operation_deadline.get()
    if deadline is not None:
        options["_account_request_deadline_monotonic"] = deadline
    return options


@contextmanager
def bounded_lock(lock, deadline_monotonic=None):
    """Wait only within the caller's budget; never release another owner's lock."""
    inherited = current_operation_deadline.get()
    deadline = (inherited if deadline_monotonic is None else deadline_monotonic
                if inherited is None else min(inherited, deadline_monotonic))
    if deadline is None or isinstance(lock, nullcontext):
        with lock:
            yield
        return
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not lock.acquire(timeout=remaining):
        from services.account_request_pacing import AccountRequestDeadlineExceeded
        raise AccountRequestDeadlineExceeded("original session lock deadline exhausted")
    try:
        yield
    finally:
        lock.release()


def safe_account_ref(identity):
    value = str(identity or "")
    return hashlib.sha256(value.encode()).hexdigest()[:24] if value else None


@contextmanager
def executing(context):
    token = current_request.set(context)
    try:
        yield
    finally:
        current_request.reset(token)


@contextmanager
def guarding_archive(check_and_renew, *, read_owner=None, request_key=None, work_key=None):
    token = current_archive_guard.set(check_and_renew)
    read_token = current_archive_read_owner.set(read_owner)
    observation_token = current_archive_observation.set({
        "request_ref": safe_account_ref(request_key), "work_ref": safe_account_ref(work_key),
    })
    try:
        yield
    finally:
        current_archive_observation.reset(observation_token)
        current_archive_read_owner.reset(read_token)
        current_archive_guard.reset(token)


@contextmanager
def reserving_original_read(owner):
    """Carry a pre-booked ordinary-read FIFO owner to the exact result GET."""
    token = current_original_read_owner.set(owner)
    try:
        yield
    finally:
        current_original_read_owner.reset(token)


@contextmanager
def observing_archive_step(step):
    """Label existing HTTP calls without changing their pacing or authority."""
    if step not in {"terminal_check", "precheck", "patch", "readback"}:
        raise ValueError("invalid archive observation step")
    token = current_archive_step.set(step)
    try:
        yield
    finally:
        current_archive_step.reset(token)


def trusted_source(identity, request=None):
    # Ownership is supplied by authentication. No priority or source field in
    # a JSON body participates in scheduling. Private internal callers already
    # holding the service credential may name their existing consumer lane.
    source = identity.get("_fair_source")
    if request is not None and getattr(request.state, "company_identity", None):
        if isinstance(source, str) and _FAIR_SOURCE.fullmatch(source):
            return source
        return "company:" + str(identity["id"])
    if (identity.get("_authenticated_fair_source") is True
            and isinstance(source, str) and _FAIR_SOURCE.fullmatch(source)):
        return source
    if identity.get("role") == "admin" and request is not None:
        lane = request.headers.get("x-workbench-consumer", "")
        if lane in {"happy", "wb-to-ozon", "ozon-to-wb", "listing", "content"}:
            return "internal:" + lane
    return "key:" + str(identity.get("id") or "anonymous")
