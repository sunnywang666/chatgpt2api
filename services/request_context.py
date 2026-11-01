"""The current original receipt at the upstream send boundary."""
from contextvars import ContextVar
from contextlib import contextmanager, nullcontext
from functools import wraps
import math
import time
import hashlib
import re

current_operation_deadline = ContextVar("provider_operation_deadline", default=None)
current_manual_conversation_read_budget = ContextVar("provider_manual_conversation_read_budget", default=None)
current_request = ContextVar("provider_original_request", default=None)
current_archive_guard = ContextVar("provider_archive_guard", default=None)
current_archive_read_owner = ContextVar("provider_archive_read_owner", default=None)
current_archive_observation = ContextVar("provider_archive_observation", default=None)
current_archive_step = ContextVar("provider_archive_step", default=None)
current_original_read_owner = ContextVar("provider_original_read_owner", default=None)
_FAIR_SOURCE = re.compile(r"^(?:user|company):[0-9a-f]{64}$")

# Archiving a conversation needs a fresh precheck and, after a mutation, a
# fresh readback.  Keep its finite chain budget aligned with the account
# pacer's existing 240-second archive-read ceiling.
ARCHIVE_OPERATION_BUDGET_SECONDS = 240.0


class AdmissionLost(RuntimeError):
    """The original claim is no longer allowed to send."""


class ManualConversationReadBudget:
    """A manual source read's finite local-pacing extension.

    The normal operation budget still pays for locks, parsing and transport.
    Only the account pacer's known local FIFO/pacing waits may advance this
    deadline, and never beyond the single hard deadline.
    """

    def __init__(self, initial_deadline: float, hard_deadline: float, local_wait_credit: float):
        self.initial_deadline = initial_deadline
        self.deadline = initial_deadline
        self.hard_deadline = hard_deadline
        self.local_wait_credit_remaining = max(0.0, local_wait_credit)
        self._admitted_local_wait_credit = 0.0
        self._admitted_local_wait_ceiling = hard_deadline

    def reserve_local_wait_credit(self) -> float:
        # A read that has already exhausted its ordinary processing/network
        # budget cannot turn a later lock/CPU delay into pacing credit.
        if time.monotonic() >= self.deadline:
            self._admitted_local_wait_credit = 0.0
            return 0.0
        nested_ceiling = current_operation_deadline.get()
        ceiling = min(self.hard_deadline, nested_ceiling) if nested_ceiling is not None else self.hard_deadline
        self._admitted_local_wait_ceiling = ceiling
        self._admitted_local_wait_credit = max(
            0.0, min(self.local_wait_credit_remaining, ceiling - self.deadline),
        )
        return self._admitted_local_wait_credit

    def record_local_pacing_wait(self, elapsed: float) -> float:
        """Consume and return only the eligible part of a pacer wait."""
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
            return 0.0
        # The pacer called reserve_local_wait_credit before it slept.  Do not
        # re-evaluate against time.monotonic here: a valid long FIFO wait can
        # naturally finish after the initial 60-second processing deadline.
        credited = min(float(elapsed), self._admitted_local_wait_credit,
                       self.local_wait_credit_remaining,
                       self._admitted_local_wait_ceiling - self.deadline)
        credited = max(0.0, credited)
        self._admitted_local_wait_credit = 0.0
        self.local_wait_credit_remaining -= credited
        self.deadline = min(self.hard_deadline, self.deadline + credited)
        return credited

    def reservation_deadline(self) -> float:
        """Bound a durable FIFO ticket to the same hard/nested ceiling."""
        nested = current_operation_deadline.get()
        return min(self.hard_deadline, nested) if nested is not None else self.hard_deadline


def operation_effective_deadline():
    """Return the active deadline, including the narrow manual-read credit."""
    budget = current_manual_conversation_read_budget.get()
    if budget is not None:
        nested = current_operation_deadline.get()
        return min(budget.deadline, nested) if nested is not None else budget.deadline
    return current_operation_deadline.get()


@contextmanager
def manual_conversation_read_budget(processing_seconds=60.0, local_wait_credit_seconds=180.0):
    """Give one explicit manual two-GET read a bounded local-pacing allowance."""
    if (type(processing_seconds) not in (int, float) or not math.isfinite(processing_seconds)
            or processing_seconds <= 0 or type(local_wait_credit_seconds) not in (int, float)
            or not math.isfinite(local_wait_credit_seconds) or local_wait_credit_seconds < 0):
        raise ValueError("finite manual conversation read budget required")
    now = time.monotonic()
    inherited = current_operation_deadline.get()
    hard_deadline = now + float(processing_seconds) + float(local_wait_credit_seconds)
    if inherited is not None:
        hard_deadline = min(hard_deadline, inherited)
    initial_deadline = min(now + float(processing_seconds), hard_deadline)
    budget = ManualConversationReadBudget(
        initial_deadline, hard_deadline, min(float(local_wait_credit_seconds), hard_deadline - initial_deadline),
    )
    # Keep the context ceiling at the hard deadline so a nested operation
    # deadline can still shorten this scope. operation_effective_deadline()
    # supplies the initial 60-second processing limit to ordinary callers.
    deadline_token = current_operation_deadline.set(hard_deadline)
    budget_token = current_manual_conversation_read_budget.set(budget)
    try:
        yield budget
    finally:
        current_manual_conversation_read_budget.reset(budget_token)
        current_operation_deadline.reset(deadline_token)


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
    deadline = operation_effective_deadline()
    if deadline is None:
        return maximum
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        from services.account_request_pacing import AccountRequestDeadlineExceeded
        raise AccountRequestDeadlineExceeded("original operation deadline exhausted")
    return min(maximum, remaining)


def operation_request_options(maximum):
    options = {"timeout": operation_remaining(maximum)}
    deadline = operation_effective_deadline()
    if deadline is not None:
        options["_account_request_deadline_monotonic"] = deadline
    return options


@contextmanager
def bounded_lock(lock, deadline_monotonic=None):
    """Wait only within the caller's budget; never release another owner's lock."""
    inherited = operation_effective_deadline()
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
