"""Bounded error diagnostics, without retaining the upstream response body."""
import re
import unicodedata


_UPSTREAM_RESPONSE_ID_KEYS = frozenset({"request_id", "cf_ray"})
_UPSTREAM_REQUEST_ID_RE = re.compile(
    r"(?:req_[A-Za-z0-9_-]{8,96}|[0-9A-Fa-f]{16,64}|[0-9A-Fa-f]{8}"
    r"-(?:[0-9A-Fa-f]{4}-){3}[0-9A-Fa-f]{12})"
)
_UPSTREAM_CF_RAY_RE = re.compile(r"[0-9A-Fa-f]{16,32}(?:-[A-Za-z]{3})?")


def safe_upstream_response_ids(value: object) -> dict[str, str]:
    """Keep only recognizable correlation IDs from a failed HTTP response."""
    if not isinstance(value, dict) or set(value) - _UPSTREAM_RESPONSE_ID_KEYS:
        return {}
    result: dict[str, str] = {}
    request_id = value.get("request_id")
    if isinstance(request_id, str) and _UPSTREAM_REQUEST_ID_RE.fullmatch(request_id):
        result["request_id"] = request_id
    cf_ray = value.get("cf_ray")
    if isinstance(cf_ray, str) and _UPSTREAM_CF_RAY_RE.fullmatch(cf_ray):
        result["cf_ray"] = cf_ray
    return result


def upstream_response_ids_from_headers(headers: object) -> dict[str, str]:
    """Read the two allowlisted response headers without retaining headers."""
    items = getattr(headers, "items", None)
    if not callable(items):
        return {}
    values = {"request_id": [], "cf_ray": []}
    names = {
        "x-request-id": "request_id",
        "openai-request-id": "request_id",
        "cf-ray": "cf_ray",
    }
    try:
        pairs = items()
    except Exception:
        return {}
    for name, candidate in pairs:
        if not isinstance(name, str):
            continue
        target = names.get(name.lower())
        if target is not None:
            values[target].append(candidate)
    captured = {}
    for key, candidates in values.items():
        # Both recognized request-ID headers are useful only when they agree.
        # A malformed or conflicting value is no evidence.
        if candidates and all(isinstance(item, str) for item in candidates) and len(set(candidates)) == 1:
            captured[key] = candidates[0]
    return safe_upstream_response_ids(captured)


def _safe_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    # Redact before truncating: cutting a credential first can hide its shape.
    value = value[:8192]
    value = re.sub(r"```[\s\S]*?(?:```|$)", "[redacted]", value)
    value = re.sub(r"(?i)(?:https?://|data:)[^\s<>\"']+", "[redacted]", value)
    value = re.sub(r"(?i)\b(?:authorization|cookie|set-cookie|access[_ -]?token|refresh[_ -]?token|token|api[_ -]?key|password|secret)\b[\"']?\s*[:=]\s*[^\n,;]+", "[redacted]", value)
    value = re.sub(r"(?i)\b(?:bearer|basic)\s+\S+", "[redacted]", value)
    value = re.sub(r"\b(?:sk-|eyJ)[A-Za-z0-9_.-]+", "[redacted]", value)
    value = re.sub(r"\b[A-Za-z0-9_.+/-]{32,}\b", "[redacted]", value)
    value = re.sub(r"\b[^\s@]+@[^\s@]+\.[^\s@]+", "[redacted]", value)
    # Error messages can quote user input. Keep the diagnostic sentence, not
    # quoted payloads or labelled request/prompt dumps.
    value = re.sub(r'''"[^"\n]*"|(?<!\w)'[^'\n]*'(?!\w)|\u201c[^\u201d]*\u201d|\u2018[^\u2019]*\u2019''', "[redacted]", value)
    value = re.sub(r"(?i)\b(?:prompt|messages?|request[_. -]?body|input(?:[_. -]?text)?|payload|body|content)\b[^\s:=]*\s*[:=][\s\S]*", "[redacted]", value)
    value = "".join(c if not unicodedata.category(c).startswith("C") else " " for c in value)
    return " ".join(value.split())[:limit]


def safe_upstream_error(value: object) -> dict[str, str]:
    """Revalidate persisted values at the public boundary as well as capture."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ("code", "type"):
        text = value.get(key)
        if isinstance(text, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,95}", text):
            safe = _safe_text(text, 96)
            if safe == text:
                result[key] = text
    message = _safe_text(value.get("message"), 512)
    if message:
        result["message"] = message
    return result


def sse_error_detail(event: dict, *, named_error: bool = False) -> dict[str, str]:
    error = event.get("error")
    # Explicit protocol path only; never search arbitrary payload/message trees.
    if (not isinstance(error, (dict, str)) or not error) and event.get("type") == "response.failed" and isinstance(event.get("response"), dict):
        error = event["response"].get("error")
    if isinstance(error, dict):
        return safe_upstream_error(error)
    if isinstance(error, str) and error:
        return safe_upstream_error({"message": error})
    if named_error or event.get("type") == "error":
        return safe_upstream_error(event)
    return {}
