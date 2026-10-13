"""Regression coverage for bounded upstream SSE error diagnostics."""
import json

import pytest

from services.conversation_binding_service import ConversationBindingError
from services.openai_backend_api import OpenAIBackendAPI
from services.public_chat_service import project_public_chat_receipt, safe_public_execution
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_reliable_pool_routes import runtime


class Response:
    def __init__(self, lines):
        self.lines = lines
        self.closed = False

    def iter_lines(self):
        yield from self.lines

    def close(self):
        self.closed = True


def _unknown_receipt_after_restart(runtime, response, request_id):
    def run(_body):
        current_request.get().before_send()
        list(OpenAIBackendAPI._iter_sse_payloads_capped(None, response, 60, observe_text=True))
        raise ConversationBindingError("original result is unconfirmed", code="CONVERSATION_OUTCOME_UNKNOWN")

    runtime.admission.register("text", lambda _ctx, body: run(body))
    runtime.service.submit("owner", {
        "client_request_id": request_id,
        "client_conversation_id": "diagnostic-session",
        "model": "fixture-text",
        "messages": [{"role": "user", "content": "fixture"}],
    })
    runtime.admission.execute(runtime.admission.claim_next())
    restarted = TextTaskService(runtime.store.path, admission=runtime.admission, clock=runtime.clock)
    with restarted.store.connect() as db:
        return restarted.store.read_receipt(db, "text", "owner", request_id)


@pytest.mark.parametrize(("lines", "expected"), [
    (
        [b'data: {"error":{"code":"provider_new_code","type":"provider_problem","message":"retry in one minute"}}', b'data: [DONE]'],
        {"code": "provider_new_code", "type": "provider_problem", "message": "retry in one minute"},
    ),
    (
        [b'data: {"error":"retry after a specific upstream refusal"}', b'data: [DONE]'],
        {"message": "retry after a specific upstream refusal"},
    ),
    (
        [b'event: error', b'data: {"code":"unrecognised_named_code","type":"error","message":"named event says retry later"}', b'data: [DONE]'],
        {"code": "unrecognised_named_code", "type": "error", "message": "named event says retry later"},
    ),
])
def test_sse_error_detail_survives_restart_and_owner_projection(runtime, lines, expected):
    row = _unknown_receipt_after_restart(runtime, Response(lines), "error-detail-" + expected.get("code", "string"))
    stream = next(item for item in row["_execution_timeline"] if item["stage"] == "stream_finished")

    assert row["status"] == "unknown"
    assert sum(item["stage"] == "send_guard_passed" for item in row["_execution_timeline"]) == 1
    assert stream["stream_end"] == "done"
    assert stream["upstream_error"] == expected

    # The ordinary owner response is reconstructed from the persisted receipt,
    # not copied from transient SSE observation state.
    public = project_public_chat_receipt(TextTaskService._public(row))
    assert public["execution"]["upstream_error"] == expected
    assert public["execution"]["sse_error_event"] is True


@pytest.mark.parametrize(("payload", "expected", "request_id"), [
    (
        b'data: upstream is temporarily unavailable',
        {"message": "upstream is temporarily unavailable"},
        "named-text-error",
    ),
    (
        b'data: "upstream sent a JSON string failure"',
        {"message": "upstream sent a JSON string failure"},
        "named-json-string-error",
    ),
])
def test_named_sse_error_captures_text_and_json_string_after_restart(runtime, payload, expected, request_id):
    row = _unknown_receipt_after_restart(runtime, Response([
        b'event: error', payload, b'data: [DONE]',
    ]), request_id)
    stream = next(item for item in row["_execution_timeline"] if item["stage"] == "stream_finished")

    assert stream["upstream_error"] == expected
    assert stream["sse_error_event"] is True
    assert project_public_chat_receipt(TextTaskService._public(row))["execution"]["upstream_error"] == expected


def test_response_failed_nested_error_is_persisted_and_projected(runtime):
    response = Response([
        ("data: " + json.dumps({
            "type": "response.failed",
            "response": {"error": {
                "code": "provider_response_new_code",
                "type": "provider_response_problem",
                "message": "response failure has a useful reason",
            }},
        })).encode(),
        b'data: [DONE]',
    ])
    expected = {
        "code": "provider_response_new_code",
        "type": "provider_response_problem",
        "message": "response failure has a useful reason",
    }
    row = _unknown_receipt_after_restart(runtime, response, "response-failed-error")
    stream = next(item for item in row["_execution_timeline"] if item["stage"] == "stream_finished")

    assert stream["upstream_error"] == expected
    assert stream["sse_error_event"] is True
    assert project_public_chat_receipt(TextTaskService._public(row))["execution"]["upstream_error"] == expected


@pytest.mark.parametrize("top_error", [{"message": "actual original reason"}, "actual original reason"])
def test_response_failed_preserves_top_level_reason(runtime, top_error):
    event = {"type": "response.failed", "error": top_error, "response": {"id": "r1"}}
    row = _unknown_receipt_after_restart(runtime, Response([
        ("data: " + json.dumps(event)).encode(), b'data: [DONE]',
    ]), "top-level-error")
    assert project_public_chat_receipt(TextTaskService._public(row))["execution"]["upstream_error"] == {
        "message": "actual original reason"}


@pytest.mark.parametrize("field", ["messages[0].content", "input_text", "request.body", "body"])
def test_request_content_inside_error_message_is_redacted(runtime, field):
    event = {"error": {"message": "invalid request: " + field + "=USER_PRIVATE_PROMPT"}}
    row = _unknown_receipt_after_restart(runtime, Response([
        ("data: " + json.dumps(event)).encode(), b'data: [DONE]',
    ]), "inline-input-redaction")
    public = project_public_chat_receipt(TextTaskService._public(row))
    assert "USER_PRIVATE_PROMPT" not in json.dumps(row)
    assert "USER_PRIVATE_PROMPT" not in json.dumps(public)
    assert public["execution"]["upstream_error"]["message"] == "invalid request: [redacted]"


def test_later_sse_message_completes_code_only_detail_without_replacing_first_reason(runtime):
    response = Response([
        b'data: {"error":{"code":"initial_provider_code"}}',
        b'data: {"error":{"message":"first useful reason"}}',
        b'data: {"error":{"code":"replacement_provider_code","message":"replacement reason"}}',
        b'data: [DONE]',
    ])
    row = _unknown_receipt_after_restart(runtime, response, "merge-error-detail")
    stream = next(item for item in row["_execution_timeline"] if item["stage"] == "stream_finished")
    expected = {"code": "initial_provider_code", "message": "first useful reason"}

    assert stream["upstream_error"] == expected
    assert project_public_chat_receipt(TextTaskService._public(row))["execution"]["upstream_error"] == expected


def test_sse_error_detail_redacts_bodies_and_public_projection_revalidates(runtime):
    authorization = "Bearer LIVE_AUTHORIZATION_SECRET"
    cookie = "session=COOKIE_PRIVATE"
    token = "short-private-token"
    url = "https://private.example.test/path?access_token=URL_PRIVATE"
    email = "person.private@example.test"
    opaque = "A" * 160
    body = "ENTIRE_PRIVATE_REQUEST_BODY"
    message = (
        "provider refused request; authorization: " + authorization + "; cookie=" + cookie
        + "; token=" + token + "; url=" + url + "; email=" + email
        + "; opaque=" + opaque + "; request body: " + body
    )
    response = Response([
        ("data: " + json.dumps({"error": {
            "code": "provider_policy_new", "type": "provider_problem", "message": message,
        }})).encode(),
        b'data: [DONE]',
    ])
    row = _unknown_receipt_after_restart(runtime, response, "redacted-error-detail")
    stream = next(item for item in row["_execution_timeline"] if item["stage"] == "stream_finished")
    stored = json.dumps(stream)

    for private in (authorization, cookie, token, url, email, opaque, body):
        assert private not in stored
    assert len(stream["upstream_error"]["message"]) <= 512

    # Treat persisted fields as hostile on the public deserialization path too.
    replayed = json.loads(json.dumps(row))
    replayed_stream = next(item for item in replayed["_execution_timeline"] if item["stage"] == "stream_finished")
    replayed_stream["upstream_error"] = {"code": "provider_policy_new", "message": message}
    public = project_public_chat_receipt(TextTaskService._public(replayed))
    serialized_public = json.dumps(public)
    for private in (authorization, cookie, token, url, email, opaque, body):
        assert private not in serialized_public
    assert public["execution"]["upstream_error"]["code"] == "provider_policy_new"

    # The same public edge does not admit arbitrary execution extensions.
    assert safe_public_execution({"upstream_error": {"message": message}, "raw_body": body}) == {
        "upstream_error": safe_public_execution({"upstream_error": {"message": message}})["upstream_error"]
    }


def test_sse_error_detail_never_retains_nested_input_or_single_quoted_content(runtime):
    prompt = "USER_PRIVATE_PROMPT"
    input_text = "SECRET_INPUT_TEXT"
    request_body = "SECRET_REQUEST_BODY"
    body = "SECRET_BODY"
    quoted = "SINGLE_QUOTED_PRIVATE_CONTENT"
    event = {
        "error": {
            "code": "provider_bad_request",
            "message": "provider rejected '" + quoted + "' before retry",
            "messages": [{"content": prompt}],
            "input_text": input_text,
            "request": {"body": request_body},
            "body": body,
        },
    }
    row = _unknown_receipt_after_restart(runtime, Response([
        ("data: " + json.dumps(event)).encode(), b'data: [DONE]',
    ]), "nested-input-redaction")
    stream = next(item for item in row["_execution_timeline"] if item["stage"] == "stream_finished")
    public = project_public_chat_receipt(TextTaskService._public(row))

    for private in (prompt, input_text, request_body, body, quoted):
        assert private not in json.dumps(row)
        assert private not in json.dumps(public)
    assert stream["upstream_error"]["code"] == "provider_bad_request"


def test_first_nonempty_sse_error_is_retained_after_done_or_empty_followup(runtime):
    first = {"code": "first_provider_error", "type": "provider_problem", "message": "first useful reason"}
    response = Response([
        b'event: error', ("data: " + json.dumps(first)).encode(), b'',
        b'data: [DONE]', b'event: error', b'data: {}',
    ])
    row = _unknown_receipt_after_restart(runtime, response, "first-error-wins")
    stream = next(item for item in row["_execution_timeline"] if item["stage"] == "stream_finished")

    assert stream["upstream_error"] == first
    assert stream["stream_end"] == "done"
    assert project_public_chat_receipt(TextTaskService._public(row))["execution"]["upstream_error"] == first


def test_success_projection_hides_historical_sse_error_without_rewriting_receipt(runtime):
    row = _unknown_receipt_after_restart(runtime, Response([
        b'data: {"error":{"code":"historical_provider_error","message":"original diagnostic"}}', b'data: [DONE]',
    ]), "historical-error")
    stream = next(item for item in row["_execution_timeline"] if item["stage"] == "stream_finished")
    succeeded = {**row, "status": "succeeded", "upstream_outcome": "completed", "content": "recovered answer"}

    public = project_public_chat_receipt(TextTaskService._public(succeeded))
    assert stream["upstream_error"]["message"] == "original diagnostic"
    assert "upstream_error" not in public["execution"]
    assert "error_code" not in public
