"""The current original receipt at the upstream send boundary."""
from contextvars import ContextVar
from contextlib import contextmanager
import hashlib
import re

current_request = ContextVar("provider_original_request", default=None)
current_archive_guard = ContextVar("provider_archive_guard", default=None)
current_archive_read_owner = ContextVar("provider_archive_read_owner", default=None)
current_archive_observation = ContextVar("provider_archive_observation", default=None)
current_archive_step = ContextVar("provider_archive_step", default=None)
_FAIR_SOURCE = re.compile(r"^(?:user|company):[0-9a-f]{64}$")


class AdmissionLost(RuntimeError):
    """The original claim is no longer allowed to send."""


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
