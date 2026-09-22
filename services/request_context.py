"""The current original receipt at the upstream send boundary."""
from contextvars import ContextVar
from contextlib import contextmanager
import hashlib

current_request = ContextVar("provider_original_request", default=None)


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
    if request is not None and getattr(request.state, "company_identity", None):
        return "company:" + str(identity["id"])
    if identity.get("role") == "admin" and request is not None:
        lane = request.headers.get("x-workbench-consumer", "")
        if lane in {"happy", "wb-to-ozon", "ozon-to-wb", "listing", "content"}:
            return "internal:" + lane
    return "key:" + str(identity.get("id") or "anonymous")
