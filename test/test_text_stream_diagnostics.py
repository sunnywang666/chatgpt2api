"""Persist structural stream evidence, never raw upstream error bodies."""
import json
from types import SimpleNamespace

import pytest

from services.conversation_binding_service import ConversationBindingError
from services.openai_backend_api import ChatRequirements, OpenAIBackendAPI, StreamHardTimeoutError
from services.public_chat_service import project_text_execution, safe_public_execution
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_reliable_pool_routes import runtime


class Response:
    def __init__(self, lines, error=None):
        self.lines, self.error, self.closed = lines, error, False

    def iter_lines(self):
        yield from self.lines
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


@pytest.mark.parametrize("end", ["done", "eof", "consumer_closed", "transport_error", "hard_timeout"])
def test_stream_end_persists_across_restart_with_only_safe_structural_evidence(runtime, end):
    lines = [b'event: error', b'data: {"error":"PRIVATE_TOKEN_AND_BODY"}', b'data: PRIVATE_BAD_JSON']
    if end == "done":
        lines.append(b'data: [DONE]')
    error = ConnectionError("PRIVATE_TRANSPORT") if end == "transport_error" else (
        StreamHardTimeoutError("PRIVATE_TIMEOUT") if end == "hard_timeout" else None)
    response = Response(lines, error)

    def run(body):
        current_request.get().before_send()
        iterator = OpenAIBackendAPI._iter_sse_payloads_capped(None, response, 60, observe_text=True)
        if end == "consumer_closed":
            next(iterator)
            iterator.close()
        else:
            list(iterator)
        raise ConversationBindingError("result remains unconfirmed", code="CONVERSATION_OUTCOME_UNKNOWN")

    runtime.admission.register("text", lambda ctx, body: run(body))
    runtime.service.submit("owner", {"client_request_id": "stream-original", "client_conversation_id": "stream-session", "model": "fixture-text",
                                     "messages": [{"role": "user", "content": "fixture input"}]})
    runtime.admission.execute(runtime.admission.claim_next())
    restarted = TextTaskService(runtime.store.path, admission=runtime.admission, clock=runtime.clock)
    with restarted.store.connect() as db:
        row = restarted.store.read_receipt(db, "text", "owner", "stream-original")
    stages = [s for s in row["_execution_timeline"] if s["stage"] == "stream_finished"]
    assert len(stages) == 1
    assert stages[0]["stream_end"] == end
    assert stages[0]["sse_error_event"] is True
    assert stages[0]["sse_data_count"] == (1 if end == "consumer_closed" else 3 if end == "done" else 2)
    assert stages[0]["sse_parse_errors"] == (0 if end == "consumer_closed" else 1)
    public = project_text_execution(row)
    assert public["stream_end"] == end and public["stream_ended_at"] > 0
    assert public["sse_error_event"] is True
    assert response.closed
    assert "PRIVATE" not in json.dumps(stages)
    assert "PRIVATE" not in json.dumps(public)


def test_failed_diagnostic_write_does_not_replace_original_transport_exception():
    def unavailable(*args, **kwargs):
        raise OSError("diagnostic storage unavailable")
    token = current_request.set(SimpleNamespace(record_stage=unavailable))
    try:
        response = Response([], ConnectionError("original transport failure"))
        with pytest.raises(ConnectionError, match="original transport failure"):
            list(OpenAIBackendAPI._iter_sse_payloads_capped(None, response, 60, observe_text=True))
    finally:
        current_request.reset(token)


def test_public_stream_diagnostics_reject_arbitrary_values():
    assert safe_public_execution({"stream_end": "PRIVATE", "stream_ended_at": float("inf"),
                                  "sse_data_count": True, "sse_parse_errors": -1,
                                  "sse_error_event": "PRIVATE", "sse_error_category": "PRIVATE", "error_body": "PRIVATE"}) == {}


@pytest.mark.parametrize("code,category", [
    ("rate_limit_exceeded", "rate_limit"), ("insufficient_quota", "quota"),
    ("authentication_error", "auth"), ("server_error", "upstream"),
    ("PRIVATE_UNKNOWN_CODE", "unknown"), (None, "unknown"),
])
@pytest.mark.parametrize("named_event", [False, True])
def test_sse_error_retains_only_structured_category_across_restart(runtime, code, category, named_event):
    error = {"type": code, "message": "PRIVATE_BODY rate_limit_exceeded"}
    response = Response(([b'event: error'] if named_event else []) + [
        ("data: " + json.dumps(error if named_event else {"error": error})).encode(), b'', b'data: [DONE]'])

    def run(body):
        current_request.get().before_send()
        list(OpenAIBackendAPI._iter_sse_payloads_capped(None, response, 60, observe_text=True))
        raise ConversationBindingError("unconfirmed", code="CONVERSATION_OUTCOME_UNKNOWN")

    runtime.admission.register("text", lambda ctx, body: run(body))
    runtime.service.submit("owner", {"client_request_id": "category-original", "client_conversation_id": "category-session", "model": "fixture-text",
                                    "messages": [{"role": "user", "content": "fixture"}]})
    runtime.admission.execute(runtime.admission.claim_next())
    restarted = TextTaskService(runtime.store.path, admission=runtime.admission, clock=runtime.clock)
    with restarted.store.connect() as db:
        row = restarted.store.read_receipt(db, "text", "owner", "category-original")
    stage = next(e for e in row["_execution_timeline"] if e["stage"] == "stream_finished")
    public = project_text_execution(row)
    assert stage["sse_error_category"] == public["sse_error_category"] == category
    assert public["sse_error_event"] is True
    assert "PRIVATE" not in json.dumps(stage) and "PRIVATE" not in json.dumps(public)


@pytest.mark.parametrize("end", ["done", "eof", "transport_error"])
def test_image_stream_records_its_own_end_before_result_collection(monkeypatch, end):
    monkeypatch.setattr("services.openai_backend_api.account_service.get_account", lambda token: {})
    backend = OpenAIBackendAPI(access_token="fixture-token")
    monkeypatch.setattr(backend, "_bootstrap", lambda: None)
    monkeypatch.setattr(backend, "_get_chat_requirements", lambda: ChatRequirements(token="fixture"))
    monkeypatch.setattr(backend, "_prepare_image_conversation", lambda *a, **kw: "conduit")
    lines = [b'data: {"private_image_body":"PRIVATE_IMAGE_CONTENT"}']
    if end == "done":
        lines.append(b'data: [DONE]')
    response = Response(lines, ConnectionError("PRIVATE_TRANSPORT") if end == "transport_error" else None)
    monkeypatch.setattr(backend, "_start_image_generation", lambda *a, **kw: response)
    stages = []
    token = current_request.set(SimpleNamespace(record_stage=lambda stage, **fields: stages.append((stage, fields))))
    try:
        if end == "transport_error":
            with pytest.raises(ConnectionError, match="PRIVATE_TRANSPORT"):
                list(backend._stream_picture_conversation("draw", "gpt-image-2", []))
        else:
            list(backend._stream_picture_conversation("draw", "gpt-image-2", []))
    finally:
        current_request.reset(token)
    assert len(stages) == 1
    stage, evidence = stages[0]
    assert stage == "stream_finished" and evidence["stream_end"] == end
    assert evidence["sse_parse_errors"] == 0 and evidence["sse_error_event"] is False
    assert response.closed
    assert "PRIVATE" not in json.dumps(stages)


def test_sse_terminal_observations_preserve_unknown_and_never_store_nodes_or_content(runtime):
    request_node = "PRIVATE_ORIGINAL_USER_NODE"
    message = {"id": "PRIVATE_ASSISTANT_NODE", "author": {"role": "assistant"},
               "status": "finished_successfully", "end_turn": True, "channel": "final",
               "metadata": {"parent_id": request_node},
               "content": {"parts": ["PRIVATE_TEXT https://private.test/?token=PRIVATE_TOKEN"]}}
    events = [{"v": {"message": message}}, {"message": message},
              {"message": {**message, "parent_id": "PRIVATE_DIFFERENT_PARENT"}},
              {"message": {**message, "id": ""}},
              {"message": {**message, "end_turn": "true"}},
              {"message": {**message, "status": "in_progress"}},
              {"message": {**message, "author": {"role": "tool"}}},
              {"p": "/message/end_turn", "v": True}]
    payloads = [json.dumps(event) for event in events] + ["[DONE]"]
    response = Response([("data: " + payload).encode() for payload in payloads])

    def run(body):
        current_request.get().before_send()
        assert list(OpenAIBackendAPI._iter_sse_payloads_capped(
            None, response, 60, observe_text=True, request_message_id=request_node)) == payloads
        raise ConversationBindingError("still needs original branch verification", code="CONVERSATION_OUTCOME_UNKNOWN")

    runtime.admission.register("text", lambda ctx, body: run(body))
    runtime.service.submit("owner", {"client_request_id": "terminal-evidence", "client_conversation_id": "terminal-session",
                                     "model": "fixture-text", "messages": [{"role": "user", "content": "fixture"}]})
    runtime.admission.execute(runtime.admission.claim_next())
    restarted = TextTaskService(runtime.store.path, admission=runtime.admission, clock=runtime.clock)
    with restarted.store.connect() as db:
        row = restarted.store.read_receipt(db, "text", "owner", "terminal-evidence")
    evidence = next(e for e in row["_execution_timeline"] if e["stage"] == "stream_finished")
    assert evidence["sse_message_snapshot_events"] == 7
    assert evidence["sse_terminal_assistant_events"] == 4
    assert evidence["sse_terminal_id_events"] == 3
    assert evidence["sse_terminal_final_channel_events"] == 4
    assert evidence["sse_terminal_parent_events"] == 4
    assert evidence["sse_terminal_direct_parent_match_events"] == 2
    assert row["status"] != "succeeded"
    assert sum(e["stage"] == "send_guard_passed" for e in row["_execution_timeline"]) == 1
    assert "PRIVATE" not in json.dumps(evidence)


def test_sse_done_or_patch_alone_is_not_a_terminal_message_snapshot():
    from utils.helper import iter_sse_payloads
    observation = {}
    response = Response([b'data: {"p":"/message/end_turn","v":true}', b'data: [DONE]'])
    list(iter_sse_payloads(response, observation=observation, request_message_id="original"))
    assert observation["stream_end"] == "done"
    assert observation["sse_terminal_assistant_events"] == 0
    assert observation["sse_terminal_direct_parent_match_events"] == 0


def test_image_stream_passes_original_node_to_safe_observation(monkeypatch):
    monkeypatch.setattr("services.openai_backend_api.account_service.get_account", lambda token: {})
    backend = OpenAIBackendAPI(access_token="fixture-token")
    backend.image_request_message_id = "PRIVATE_IMAGE_USER_NODE"
    monkeypatch.setattr(backend, "_bootstrap", lambda: None)
    monkeypatch.setattr(backend, "_get_chat_requirements", lambda: ChatRequirements(token="fixture"))
    monkeypatch.setattr(backend, "_prepare_image_conversation", lambda *a, **kw: "conduit")
    response = Response([b'data: {"message":{"id":"PRIVATE_ASSISTANT","author":{"role":"assistant"},"status":"finished_successfully","end_turn":true,"parent_id":"PRIVATE_IMAGE_USER_NODE"}}', b'data: [DONE]'])
    monkeypatch.setattr(backend, "_start_image_generation", lambda *a, **kw: response)
    stages = []
    token = current_request.set(SimpleNamespace(record_stage=lambda stage, **fields: stages.append((stage, fields))))
    try:
        list(backend._stream_picture_conversation("draw", "gpt-image-2", []))
    finally:
        current_request.reset(token)
    assert stages[0][1]["sse_terminal_direct_parent_match_events"] == 1
    assert "PRIVATE" not in json.dumps(stages)
