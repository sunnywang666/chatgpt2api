"""Bounded error diagnostics, without retaining the upstream response body."""
import re
import unicodedata


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
