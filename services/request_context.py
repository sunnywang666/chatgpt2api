"""The current original receipt at the upstream send boundary."""
from contextvars import ContextVar
from contextlib import contextmanager
import hashlib
import re

current_request = ContextVar("provider_original_request", default=None)
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
