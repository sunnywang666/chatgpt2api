"""SSE wrapper failures keep a bounded public error without logging its body."""
import json

import pytest

from utils.helper import anthropic_sse_stream, sse_json_stream


SAFE_REASON = "upstream rejected the request; retry later"
PRIVATE_TOKEN = "TOKEN_PRIVATE_VALUE"
PRIVATE_AUTHORIZATION = "AUTHORIZATION_PRIVATE_VALUE"
PRIVATE_URL = "https://private.example.test/stream?token=URL_PRIVATE_VALUE"
PRIVATE_EMAIL = "private.person@example.test"
PRIVATE_PROMPT = "PROMPT_PRIVATE_VALUE"
PRIVATE_VALUES = (PRIVATE_TOKEN, PRIVATE_AUTHORIZATION, PRIVATE_URL, PRIVATE_EMAIL, PRIVATE_PROMPT)


def _private_message():
    return (
        SAFE_REASON + "; token=" + PRIVATE_TOKEN
        + "; authorization=Bearer " + PRIVATE_AUTHORIZATION
        + "; url=" + PRIVATE_URL
        + "; email=" + PRIVATE_EMAIL
        + "; prompt=" + PRIVATE_PROMPT
    )


class PlainStreamFailure(RuntimeError):
    pass


class OpenAIStreamFailure(RuntimeError):
    def to_openai_error(self):
        return {"error": {
            "message": str(self),
            "type": "provider_stream_failure",
            "code": "provider_stream_code",
            "param": PRIVATE_PROMPT,
            "account_email": PRIVATE_EMAIL,
        }}


def _failing_items(error):
    yield {"type": "message_delta", "delta": {"text": "first safe item"}}
    raise error


def _error_payload(chunks):
    payloads = [json.loads(chunk[6:].strip()) for chunk in chunks
                if chunk.startswith("data:") and chunk.strip() != "data: [DONE]"]
    return payloads[-1]["error"]


@pytest.mark.parametrize(("wrapper", "error_type", "log_event", "is_anthropic"), [
    (sse_json_stream, PlainStreamFailure, "sse_stream_error", False),
    (sse_json_stream, OpenAIStreamFailure, "sse_stream_error", False),
    (anthropic_sse_stream, PlainStreamFailure, "anthropic_sse_stream_error", True),
    (anthropic_sse_stream, OpenAIStreamFailure, "anthropic_sse_stream_error", True),
])
def test_stream_wrappers_redact_error_bodies_but_preserve_safe_error_shape(
        monkeypatch, wrapper, error_type, log_event, is_anthropic):
    logged = []
    monkeypatch.setattr("utils.helper.logger.warning", lambda event: logged.append(event))
    chunks = list(wrapper(_failing_items(error_type(_private_message()))))
    error = _error_payload(chunks)
    serialized = json.dumps(chunks)

    assert SAFE_REASON in error["message"]
    assert error["type"] == ("provider_stream_failure" if error_type is OpenAIStreamFailure else "PlainStreamFailure")
    if error_type is OpenAIStreamFailure:
        assert error["code"] == "provider_stream_code"
    else:
        assert "code" not in error
    assert set(error) <= {"message", "type", "code"}
    for private in PRIVATE_VALUES:
        assert private not in serialized
        assert private not in json.dumps(logged)
    assert logged == [{"event": log_event, "error_type": error_type.__name__}]

    if is_anthropic:
        assert "event: error\n" in chunks
    else:
        assert chunks[0] == ": stream-open\n\n"
        assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.parametrize(("wrapper", "is_anthropic"), [
    (sse_json_stream, False),
    (anthropic_sse_stream, True),
])
def test_call_log_stream_records_only_exception_type_before_wrapper_redacts(
        monkeypatch, wrapper, is_anthropic):
    from services import log_service as log_service_module

    recorded = []
    monkeypatch.setattr(log_service_module.log_service, "add", lambda *args, **kwargs: recorded.append((args, kwargs)))
    call = log_service_module.LoggedCall(
        identity={"id": "safe-key", "name": "safe-name", "role": "user"},
        endpoint="/v1/chat/completions",
        model="fixture-text",
        summary="fixture call",
    )
    chunks = list(wrapper(call.stream(_failing_items(PlainStreamFailure(_private_message())))))
    error = _error_payload(chunks)

    assert SAFE_REASON in error["message"]
    assert error["type"] == "PlainStreamFailure"
    assert len(recorded) == 1
    args, kwargs = recorded[0]
    assert args[:2] == (log_service_module.LOG_TYPE_CALL, "fixture call流式调用失败")
    assert kwargs == {}
    detail = args[2]
    assert detail["error"] == "PlainStreamFailure"
    for private in (*PRIVATE_VALUES, SAFE_REASON):
        assert private not in json.dumps(detail)

    if is_anthropic:
        assert "event: error\n" in chunks
    else:
        assert chunks[-1] == "data: [DONE]\n\n"
