"""A stream handoff may only consume the exact original turn once."""
import json
import time
from types import SimpleNamespace

import pytest
from curl_cffi import CurlECode, CurlError, CurlWsFlag
from services.account_request_pacing import AccountRequestDeadlineExceeded
from services import openai_backend_api as backend_module
from services.openai_backend_api import ChatRequirements, OpenAIBackendAPI, _StreamHandoff
from services.pool_admission import _safe_stream_handoff_event_metadata
from services.protocol.conversation import conversation_events, is_visible_assistant_message
from services.request_context import current_request


class Response:
    status_code = 200
    headers = {}

    def __init__(self, lines=(), body=None):
        self.lines = list(lines)
        self.body = body if body is not None else {}
        self.closed = False

    def iter_lines(self):
        yield from self.lines

    def json(self):
        return self.body

    def close(self):
        self.closed = True


class RaisingResponse(Response):
    """Yield bounded SSE evidence, then fail while reading its body."""
    def __init__(self, lines=(), error=None, body=None):
        super().__init__(lines, body=body)
        self.error = error or ConnectionError("stream body closed")

    def iter_lines(self):
        yield from self.lines
        raise self.error


class StageContext:
    def __init__(self):
        self.stages = []

    def record_stage(self, stage, **fields):
        self.stages.append((stage, fields))


class WebSocket:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.sent = []
        self.closed = False
        self.terminated = False

    def send_str(self, value):
        self.sent.append(value)

    def recv_str(self):
        return next(self.frames)

    def close(self):
        self.closed = True

    def terminate(self):
        self.terminated = True
        self.closed = True


class FragmentWebSocket(WebSocket):
    """A curl-cffi-like socket that exposes individually flagged fragments."""
    def __init__(self, frames):
        super().__init__(frames)
        self.curl = SimpleNamespace(getinfo=lambda _info: 9)

    def recv_fragment(self):
        item = next(self.frames)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            return item
        return item.encode(), SimpleNamespace(flags=CurlWsFlag.TEXT, bytesleft=0)


def _fragment(chunk, flags, *, bytesleft=0):
    return chunk, SimpleNamespace(flags=flags, bytesleft=bytesleft)


class Session:
    def __init__(self, initial, ws_url, frames=()):
        self.initial = initial
        self.ws_url = ws_url
        self.frames = frames
        self.post_calls = []
        self.get_calls = []
        self.ws_calls = []
        self.ws = None
        self.ws_type = WebSocket
        self.verify = True
        self.impersonate = "chrome110"
        self.proxies = {"all": "socks5h://bound-proxy.example:1080"}
        self.cookies = {"session": "bound"}
        self.closed = False

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self.initial

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return Response(body={"websocket_url": self.ws_url})

    def ws_connect(self, url, **kwargs):
        self.ws_calls.append((url, kwargs))
        self.ws = self.ws_type(self.frames)
        return self.ws

    def close(self):
        self.closed = True


class FailingHandoffSession(Session):
    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        raise RuntimeError("PRIVATE_WS_TRANSPORT_DETAIL")


def _line(event):
    return ("data: " + json.dumps(event, separators=(",", ":"))).encode()


def _handoff(conversation="conversation-1", turn="turn-1", topic=None):
    topic = topic or f"conversation-turn-{turn}"
    return {
        "type": "stream_handoff",
        "conversation_id": conversation,
        "turn_exchange_id": turn,
        "options": [
            {"type": "resume_sse_endpoint", "url": "https://ignored.invalid/private", "topic_id": topic},
            {"type": "subscribe_ws_topic", "topic_id": topic},
        ],
    }


def _ste(conversation="conversation-1", turn="turn-1", *, top_level=False, conflict=False):
    event = {
        "type": "server_ste_metadata",
        "conversation_id": conversation,
        "metadata": {"turn_exchange_id": turn},
    }
    if top_level:
        event["turn_exchange_id"] = "different-turn" if conflict else turn
    return event


def _backend(initial, ws_url="wss://ws.chatgpt.com:443/p4/ws/user/user?verify=ephemeral", frames=()):
    backend = object.__new__(OpenAIBackendAPI)
    backend.base_url = "https://chatgpt.com"
    backend._bootstrap = lambda: None
    backend._get_chat_requirements = lambda: ChatRequirements(token="fixture")
    backend._chat_target = lambda: ("/backend-api/conversation", "UTC")
    backend._conversation_headers = lambda *_args: {"content-type": "application/json"}
    backend._conversation_payload = lambda *_args, **_kwargs: {"action": "next"}
    backend._headers = lambda _path, extra=None: {"Authorization": "Bearer bound", **dict(extra or {})}
    backend.text_request_message_id = "original-user"
    backend.retain_bound_conversation = True
    backend.text_stream_handoff_enabled = True
    backend.session = Session(initial, ws_url, frames)
    return backend


def _initial_events(*, conversation="conversation-1", handoff=True, ste=False, terminal=False, terminal_parent="original-user"):
    events = [
        {"type": "resume_conversation_token", "kind": "topic", "token": "PRIVATE", "conversation_id": conversation},
    ]
    if handoff:
        events.append(_handoff(conversation=conversation))
    if ste:
        events.append(_ste(conversation=conversation))
    if terminal:
        events.append({"v": {"message": {
            "id": "thought-parent", "author": {"role": "assistant"}, "parent": "original-user",
        }}})
        events.append({"v": {"message": {
            "id": "thinking-leaf", "author": {"role": "assistant"},
            "status": "finished_successfully", "end_turn": True, "parent": terminal_parent,
        }}})
    return Response([*map(_line, events), b"data: [DONE]"])


def _ws_frame(topic="conversation-turn-turn-1"):
    encoded = "\n".join((
        'data: {"v":{"conversation_id":"conversation-1","message":{"id":"thought-leaf","author":{"role":"assistant"},"parent":"original-user"}}}',
        'data: {"v":{"conversation_id":"conversation-1","message":{"id":"final-leaf","author":{"role":"assistant"},"status":"finished_successfully","end_turn":true,"parent":"thought-leaf"}}}',
        "data: [DONE]",
    ))
    return json.dumps({"type": "reply", "reply": {"topic_id": topic, "catchups": [
        {"payload": {"payload": {"encoded_item": encoded}}},
    ]}})


def test_handoff_records_only_fixed_safe_progress_and_one_transport_each():
    backend = _backend(_initial_events(), frames=[_ws_frame()])
    context = StageContext()
    token = current_request.set(context)
    try:
        assert list(backend.stream_conversation(prompt="fixture"))[-1] == "[DONE]"
    finally:
        current_request.reset(token)

    fields = [data for stage, data in context.stages if stage == "stream_handoff"]
    assert [data["handoff_stage"] for data in fields] == [
        "candidate_bound", "ws_endpoint_get_entered", "ws_endpoint_get_completed",
        "ws_connected", "ws_subscribed", "ws_terminal",
    ]
    assert fields[-1]["handoff_terminal_seen"] is True
    assert len(backend.session.get_calls) == len(backend.session.ws_calls) == 1
    serialized = json.dumps(fields)
    assert "PRIVATE" not in serialized and "topic" not in serialized and "url" not in serialized


def test_handoff_transport_error_stays_unknown_and_records_no_exception_text():
    backend = _backend(_initial_events(), frames=[_ws_frame()])
    backend.session = FailingHandoffSession(backend.session.initial, backend.session.ws_url)
    context = StageContext()
    token = current_request.set(context)
    try:
        assert list(backend.stream_conversation(prompt="fixture"))[-1] == "[DONE]"
    finally:
        current_request.reset(token)

    fields = [data for stage, data in context.stages if stage == "stream_handoff"]
    assert [data["handoff_stage"] for data in fields] == ["candidate_bound", "ws_endpoint_get_entered", "ws_transport_error"]
    assert len(backend.session.get_calls) == 1 and backend.session.ws_calls == []
    assert "PRIVATE" not in json.dumps(fields)


def test_eligible_handoff_continues_after_sse_body_transport_error_without_second_post():
    initial = RaisingResponse([
        _line({"type": "resume_conversation_token", "kind": "topic", "token": "PRIVATE",
               "conversation_id": "conversation-1"}),
        _line(_ste()),
    ])
    backend = _backend(initial, frames=[_ws_frame()])
    context = StageContext()
    token = current_request.set(context)
    try:
        payloads = list(backend.stream_conversation(prompt="fixture"))
    finally:
        current_request.reset(token)

    assert payloads[-1] == "[DONE]"
    assert any("final-leaf" in payload for payload in payloads)
    assert len(backend.session.post_calls) == 1
    assert len(backend.session.get_calls) == len(backend.session.ws_calls) == 1
    stages = [fields["handoff_stage"] for stage, fields in context.stages if stage == "stream_handoff"]
    assert stages == [
        "candidate_bound", "ws_endpoint_get_entered", "ws_endpoint_get_completed", "ws_connected",
        "ws_subscribed", "ws_terminal",
    ]


@pytest.mark.parametrize("candidate", [
    [],
    [_line(_ste(top_level=True, conflict=True))],
])
def test_sse_body_transport_error_without_an_eligible_handoff_preserves_original_error(candidate):
    initial = RaisingResponse([
        _line({"type": "resume_conversation_token", "kind": "topic", "token": "PRIVATE",
               "conversation_id": "conversation-1"}),
        *candidate,
    ])
    backend = _backend(initial, frames=[_ws_frame()])

    with pytest.raises(ConnectionError, match="stream body closed"):
        list(backend.stream_conversation(prompt="fixture"))

    assert len(backend.session.post_calls) == 1
    assert backend.session.get_calls == backend.session.ws_calls == []


def test_explicit_upstream_error_never_opens_handoff_or_becomes_a_terminal():
    from utils.helper import UpstreamHTTPError

    initial = RaisingResponse([
        _line({"type": "resume_conversation_token", "kind": "topic", "token": "PRIVATE",
               "conversation_id": "conversation-1"}),
        _line(_ste()),
    ], error=UpstreamHTTPError("/backend-api/conversation", 500, {"error": "opaque"}))
    backend = _backend(initial, frames=[_ws_frame()])

    with pytest.raises(UpstreamHTTPError):
        list(backend.stream_conversation(prompt="fixture"))

    assert len(backend.session.post_calls) == 1
    assert backend.session.get_calls == backend.session.ws_calls == []


@pytest.mark.parametrize("failure", ["endpoint", "empty", "foreign_branch"])
def test_failed_handoff_after_sse_transport_error_preserves_the_original_exception(failure):
    original_error = ConnectionError("original body failure")
    initial = RaisingResponse(_initial_events(handoff=False, ste=True).lines[:-1], error=original_error)
    frames = [] if failure == "empty" else [_ws_frame().replace("original-user", "foreign-user")]
    backend = _backend(initial, frames=frames)
    if failure == "endpoint":
        backend.session = FailingHandoffSession(initial, backend.session.ws_url)

    with pytest.raises(ConnectionError) as caught:
        list(backend.stream_conversation(prompt="fixture"))

    assert caught.value is original_error
    assert len(backend.session.post_calls) == 1
    assert backend.text_stream_terminal is None


def test_handoff_timeline_whitelist_drops_arbitrary_fields_and_values():
    fields = _safe_stream_handoff_event_metadata({
        "handoff_stage": "ws_terminal", "handoff_frame_count": 2,
        "handoff_byte_count": 17, "handoff_terminal_seen": True,
        "topic_id": "PRIVATE_TOPIC", "exception": "PRIVATE_EXCEPTION",
    })
    assert fields == {"handoff_stage": "ws_terminal", "handoff_frame_count": 2,
                      "handoff_byte_count": 17, "handoff_terminal_seen": True}
    assert _safe_stream_handoff_event_metadata({"handoff_stage": "PRIVATE"}) == {}


def test_handoff_parent_edges_are_bounded_and_conflicts_cannot_prove_original_branch():
    bounded = _StreamHandoff(original_message_id="original-user")
    for index in range(256):
        bounded.observe(json.dumps({"v": {"message": {
            "id": f"thought-{index}", "author": {"role": "assistant"}, "parent": "original-user",
        }}}))
    assert len(bounded._parents) == 128

    handoff = _StreamHandoff(original_message_id="original-user")
    handoff.observe(json.dumps({"v": {"message": {
        "id": "conflicted", "author": {"role": "assistant"}, "parent": "original-user",
    }}}))
    handoff.observe(json.dumps({"v": {"message": {
        "id": "conflicted", "author": {"role": "assistant"}, "parent": "old-user",
    }}}))
    final = {"id": "final", "author": {"role": "assistant"}, "status": "finished_successfully",
             "end_turn": True, "parent": "conflicted"}
    handoff.observe(json.dumps({"v": {"message": final}}))
    assert handoff._conflicted_parent_ids == {"conflicted"}
    assert handoff.terminal_seen is False


def test_handoff_accepts_metadata_parent_chain_and_keeps_a_final_text_cursor():
    handoff = _StreamHandoff(expected_conversation_id="conversation-1", original_message_id="original-user")
    handoff.observe(json.dumps({"type": "resume_conversation_token", "kind": "topic",
                                "conversation_id": "conversation-1"}))
    handoff.observe(json.dumps({"v": {"conversation_id": "conversation-1", "message": {
        "id": "thought", "author": {"role": "assistant"},
        "metadata": {"parent_id": "original-user", "channel": "analysis"},
    }}}))
    handoff.observe(json.dumps({"v": {"conversation_id": "conversation-1", "message": {
        "id": "final", "author": {"role": "assistant"}, "status": "finished_successfully",
        "end_turn": True, "metadata": {"parent_id": "thought", "channel": "final"},
        "content": {"content_type": "text", "parts": ["complete answer"]},
    }}}))

    assert handoff.terminal_seen is True
    assert handoff.terminal_cursor() == {
        "conversation_id": "conversation-1", "message_id": "final",
        "text": "complete answer", "original_message_id": "original-user",
    }


def test_handoff_rejects_conflicting_parent_or_metadata_reasoning_phase():
    handoff = _StreamHandoff(expected_conversation_id="conversation-1", original_message_id="original-user")
    handoff.observe(json.dumps({"v": {"conversation_id": "conversation-1", "message": {
        "id": "conflicted", "author": {"role": "assistant"}, "parent": "original-user",
        "metadata": {"parent_id": "other-user"},
    }}}))
    handoff.observe(json.dumps({"v": {"conversation_id": "conversation-1", "message": {
        "id": "final", "author": {"role": "assistant"}, "parent": "conflicted",
        "status": "finished_successfully", "end_turn": True, "channel": "final",
        "metadata": {"channel": "analysis"},
        "content": {"content_type": "text", "parts": ["must not accept"]},
    }}}))

    assert handoff.terminal_seen is False
    assert handoff.terminal_cursor() is None


@pytest.mark.parametrize(("message", "containers", "expected"), [
    ({"channel": " FINAL "}, (), True),
    ({}, (), None),
    ({}, ({"metadata": {"channel": "analysis"}},), False),
    ({"channel": "final", "metadata": {"channel": "analysis"}}, (), False),
    ({"channel": "analysis", "metadata": {"channel": "final"}}, (), False),
    ({"channel": 1}, (), False),
])
def test_final_phase_requires_normalized_nonconflicting_message_and_metadata(message, containers, expected):
    assert backend_module.explicit_final_channel(message, *containers) is expected


def test_visible_assistant_message_does_not_ignore_conflicting_frame_metadata_phase():
    message = {"author": {"role": "assistant"}, "channel": " FINAL "}
    assert is_visible_assistant_message(message)
    assert not is_visible_assistant_message(message, {"metadata": {"channel": "analysis"}})


def test_handoff_rejects_final_text_when_wrapped_frame_metadata_conflicts():
    handoff = _StreamHandoff(expected_conversation_id="conversation-1", original_message_id="original-user")
    handoff.observe(json.dumps({"v": {
        "conversation_id": "conversation-1", "metadata": {"channel": "analysis"}, "message": {
            "id": "final", "parent": "original-user", "author": {"role": "assistant"},
            "status": "finished_successfully", "end_turn": True, "channel": " FINAL ",
            "content": {"content_type": "text", "parts": ["must not certify"]},
        },
    }}))

    assert handoff.terminal_seen is False
    assert handoff.terminal_cursor() is None


def test_handoff_never_certifies_an_idless_final_after_a_valid_parent_chain():
    handoff = _StreamHandoff(expected_conversation_id="conversation-1", original_message_id="original-user")
    handoff.observe(json.dumps({"v": {"conversation_id": "conversation-1", "message": {
        "id": "thought", "author": {"role": "assistant"}, "parent_id": "original-user",
    }}}))
    handoff.observe(json.dumps({"v": {"conversation_id": "conversation-1", "message": {
        "id": "", "author": {"role": "assistant"}, "parent_id": "thought",
        "status": "finished_successfully", "end_turn": True, "metadata": {"channel": "final"},
        "content": {"content_type": "text", "parts": ["must not certify"]},
    }}}))

    assert handoff.terminal_cursor() is None


def test_handoff_certificate_is_permanently_invalidated_by_later_conflicts_or_finals():
    def observe(handoff, message):
        handoff.observe(json.dumps({"v": {"conversation_id": "conversation-1", "message": message}}))

    def certified():
        handoff = _StreamHandoff(expected_conversation_id="conversation-1", original_message_id="original-user")
        observe(handoff, {"id": "thought", "author": {"role": "assistant"}, "parent": "original-user"})
        observe(handoff, {"id": "final-a", "author": {"role": "assistant"}, "parent": "thought",
                          "status": "finished_successfully", "end_turn": True, "channel": "final",
                          "content": {"content_type": "text", "parts": ["answer"]}})
        assert handoff.terminal_cursor() is not None
        return handoff

    parent_conflict = certified()
    observe(parent_conflict, {"id": "thought", "author": {"role": "assistant"}, "parent": "other-user"})
    assert parent_conflict.terminal_cursor() is None

    phase_conflict = certified()
    observe(phase_conflict, {"id": "final-a", "author": {"role": "assistant"}, "parent": "thought",
                             "status": "finished_successfully", "end_turn": True,
                             "metadata": {"channel": "analysis"},
                             "content": {"content_type": "thoughts", "parts": ["hidden"]}})
    assert phase_conflict.terminal_cursor() is None

    body_conflict = certified()
    observe(body_conflict, {"id": "final-a", "author": {"role": "assistant"}, "parent": "thought",
                            "status": "finished_successfully", "end_turn": True, "channel": "final",
                            "content": {"content_type": "text", "parts": ["changed"]}})
    assert body_conflict.terminal_cursor() is None

    multiple_finals = certified()
    observe(multiple_finals, {"id": "final-b", "author": {"role": "assistant"}, "parent": "thought",
                              "status": "finished_successfully", "end_turn": True, "channel": "final",
                              "content": {"content_type": "text", "parts": ["second"]}})
    observe(multiple_finals, {"id": "final-c", "author": {"role": "assistant"}, "parent": "thought",
                              "status": "finished_successfully", "end_turn": True, "channel": "final",
                              "content": {"content_type": "text", "parts": ["third"]}})
    assert multiple_finals.terminal_cursor() is None


def test_backend_clears_cached_terminal_after_later_foreign_conversation_event():
    initial = Response([
        _line({"type": "resume_conversation_token", "kind": "topic", "conversation_id": "conversation-1"}),
        _line({"v": {"conversation_id": "conversation-1", "message": {
            "id": "thought", "author": {"role": "assistant"}, "parent": "original-user",
        }}}),
        _line({"v": {"conversation_id": "conversation-1", "message": {
            "id": "final", "author": {"role": "assistant"}, "parent": "thought",
            "status": "finished_successfully", "end_turn": True, "channel": "final",
            "content": {"content_type": "text", "parts": ["answer"]},
        }}}),
        _line({"v": {"conversation_id": "conversation-2", "message": {
            "id": "foreign", "author": {"role": "assistant"}, "parent": "original-user",
        }}}),
        b"data: [DONE]",
    ])
    backend = _backend(initial)

    assert list(backend.stream_conversation(prompt="fixture", conversation_id="conversation-1"))[-1] == "[DONE]"
    assert backend.text_stream_terminal is None


@pytest.mark.parametrize("second_content", [{}, {"content": {"content_type": "text", "parts": []}}])
def test_later_original_terminal_without_visible_text_revokes_cursor_and_uses_get(monkeypatch, second_content):
    from contextlib import nullcontext
    from services import conversation_binding_service as binding_module

    initial = Response([
        _line({"type": "resume_conversation_token", "kind": "topic", "conversation_id": "conversation-1"}),
        _line({"v": {"conversation_id": "conversation-1", "message": {
            "id": "final-a", "parent": "original-user", "author": {"role": "assistant"},
            "channel": "final", "status": "finished_successfully", "end_turn": True,
            "content": {"content_type": "text", "parts": ["answer"]},
        }}}),
        _line({"v": {"conversation_id": "conversation-1", "message": {
            "id": "final-b", "parent": "original-user", "author": {"role": "assistant"},
            "channel": "final", "status": "finished_successfully", "end_turn": True,
            **second_content,
        }}}),
        b"data: [DONE]",
    ])
    backend = _backend(initial)
    parent_reads = []
    backend.get_conversation_parent_message_id = lambda conversation_id: (
        parent_reads.append(conversation_id) or "read-parent")
    backend.close = lambda: None
    monkeypatch.setattr(binding_module, "OpenAIBackendAPI", lambda access_token: backend)
    monkeypatch.setattr(binding_module.account_service, "get_bound_account_identity", lambda *_args, **_kwargs: "account-a")
    monkeypatch.setattr(binding_module.account_service, "get_bound_text_access_token", lambda *_args, **_kwargs: "token-a")
    monkeypatch.setattr(binding_module.account_service, "conversation_binding_lock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(binding_module.account_service, "mark_text_used", lambda *_args, **_kwargs: None)

    result = binding_module.ConversationBindingService().complete_text({
        "provider_binding_id": "binding-a", "provider_account_identity": "account-a",
        "client_conversation_id": "chain-a", "_request_message_id": "original-user",
        "messages": [{"role": "user", "content": "write"}],
    })

    assert backend.text_stream_terminal is None
    assert parent_reads == ["conversation-1"]
    assert result["parent_message_id"] == "read-parent"


@pytest.mark.parametrize("exact_complete", [False, True])
def test_transport_handoff_terminal_still_uses_binding_exact_reader(monkeypatch, exact_complete):
    """WS terminal ends transport only; the saved request branch owns success."""
    from contextlib import nullcontext
    from services import conversation_binding_service as binding_module

    initial = RaisingResponse([
        _line({"type": "resume_conversation_token", "kind": "topic", "token": "PRIVATE",
               "conversation_id": "conversation-1"}),
        _line(_ste()),
        _line({"v": {"conversation_id": "conversation-1", "message": {
            "id": "thought-leaf", "author": {"role": "assistant"}, "parent": "original-user",
        }}}),
    ])
    backend = _backend(initial, frames=[_ws_frame()])
    backend.close = lambda: None
    exact_reads = []

    def exact_reader(_backend, receipt, **_kwargs):
        exact_reads.append(dict(receipt))
        if not exact_complete:
            return {"status": "running", "recovery_reason": "REQUEST_RESULT_INCOMPLETE"}
        return {
            "status": "succeeded", "binding_status": "bound", "content": "saved exact result",
            "provider_binding_id": "binding-a", "provider_account_identity": "account-a",
            "client_conversation_id": "chain-a", "conversation_id": "conversation-1",
            "parent_message_id": "saved-final",
        }

    monkeypatch.setattr(binding_module, "OpenAIBackendAPI", lambda access_token: backend)
    monkeypatch.setattr(binding_module.ConversationBindingService, "_read_text_request_result", staticmethod(exact_reader))
    monkeypatch.setattr(binding_module.account_service, "get_bound_account_identity", lambda *_args, **_kwargs: "account-a")
    monkeypatch.setattr(binding_module.account_service, "get_bound_text_access_token", lambda *_args, **_kwargs: "token-a")
    monkeypatch.setattr(binding_module.account_service, "conversation_binding_lock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(binding_module.account_service, "mark_text_used", lambda *_args, **_kwargs: None)

    body = {
        "provider_binding_id": "binding-a", "provider_account_identity": "account-a",
        "client_conversation_id": "chain-a", "_request_message_id": "original-user",
        "_public_session_ref": "public-session", "_route": "chat", "_operation": "text",
        "messages": [{"role": "user", "content": "write"}],
    }

    service = binding_module.ConversationBindingService()
    if exact_complete:
        result = service.complete_text(body)
        assert result["content"] == "saved exact result"
        assert result["_upstream_terminal"] is True
    else:
        with pytest.raises(binding_module.ConversationBindingError) as caught:
            service.complete_text(body)
        assert caught.value.code == "CONVERSATION_OUTCOME_UNKNOWN"
        assert caught.value.recovery_reason == "REQUEST_RESULT_INCOMPLETE"
    assert len(exact_reads) == 1
    assert exact_reads[0]["request_message_id"] == "original-user"
    assert len(backend.session.post_calls) == len(backend.session.get_calls) == len(backend.session.ws_calls) == 1


def test_handoff_rejects_foreign_parent_edges_before_an_idless_conversation_final():
    handoff = _StreamHandoff(expected_conversation_id="conversation-1", original_message_id="original-user")
    handoff.observe(json.dumps({"v": {"conversation_id": "conversation-2", "message": {
        "id": "foreign-thought", "author": {"role": "assistant"}, "parent": "original-user",
    }}}))
    handoff.observe(json.dumps({"v": {"message": {
        "id": "final", "author": {"role": "assistant"}, "parent": "foreign-thought",
        "status": "finished_successfully", "end_turn": True, "channel": "final",
        "content": {"content_type": "text", "parts": ["must not accept"]},
    }}}))

    assert handoff.invalid is True
    assert handoff._parents == {}
    assert handoff.terminal_seen is False
    assert handoff.terminal_cursor() is None


def test_handoff_rejects_conflicting_or_malformed_conversation_fields():
    for outer, inner in (("conversation-1", "conversation-2"), ("conversation-1", {"id": "conversation-1"})):
        handoff = _StreamHandoff(expected_conversation_id="conversation-1", original_message_id="original-user")
        handoff.observe(json.dumps({"conversation_id": outer, "v": {
            "conversation_id": inner, "message": {
                "id": "final", "parent": "original-user", "author": {"role": "assistant"},
                "channel": "final", "status": "finished_successfully", "end_turn": True,
                "content": {"content_type": "text", "parts": ["answer"]},
            },
        }}))
        assert handoff.invalid is True
        assert handoff.terminal_cursor() is None


def test_handoff_never_certifies_the_request_user_as_its_own_answer():
    handoff = _StreamHandoff(expected_conversation_id="conversation-1", original_message_id="original-user")
    handoff.observe(json.dumps({"conversation_id": "conversation-1", "message": {
        "id": "original-user", "parent": "root", "author": {"role": "assistant"},
        "channel": "final", "status": "finished_successfully", "end_turn": True,
        "content": {"content_type": "text", "parts": ["answer"]},
    }}))
    assert handoff.terminal_cursor() is None


def test_handoff_subscribes_once_and_closes_without_a_second_conversation_post():
    backend = _backend(_initial_events(), frames=[_ws_frame()])

    payloads = list(backend.stream_conversation(prompt="fixture"))

    assert payloads[-2:] == [
        '{"v":{"conversation_id":"conversation-1","message":{"id":"final-leaf","author":{"role":"assistant"},"status":"finished_successfully","end_turn":true,"parent":"thought-leaf"}}}',
        "[DONE]",
    ]
    assert payloads.count("[DONE]") == 1
    assert len(backend.session.post_calls) == 1
    assert len(backend.session.get_calls) == 1
    assert len(backend.session.ws_calls) == 1
    assert json.loads(backend.session.ws.sent[0]) == [
        {"id": 1, "command": {
            "type": "connect", "presence": {"type": "presence", "state": "foreground"},
        }},
        {"id": 2, "command": {
            "type": "subscribe", "topic_id": "conversation-turn-turn-1", "offset": "0",
        }},
    ]
    assert backend.session.ws_calls[0][1]["allow_redirects"] is False
    assert backend.session.ws_calls[0][1]["headers"]["Authorization"] == "Bearer bound"
    assert backend.session.ws_calls[0][1]["verify"] is True
    assert backend.session.ws_calls[0][1]["impersonate"] == "chrome110"
    assert backend.session.ws_calls[0][1]["proxies"] == {"all": "socks5h://bound-proxy.example:1080"}
    assert backend.session.ws_calls[0][1]["cookies"] == {"session": "bound"}
    assert backend.session.ws.closed is True


def test_protocol_consumer_closure_preserves_terminal_stage_before_closing_handoff():
    """The DONE adapter closes upstream, but must not erase terminal evidence."""
    backend = _backend(_initial_events(), frames=[_ws_frame()])
    context = StageContext()
    token = current_request.set(context)
    try:
        events = list(conversation_events(backend, prompt="fixture"))
    finally:
        current_request.reset(token)

    assert events[-1]["type"] == "conversation.done"
    stages = [data["handoff_stage"] for stage, data in context.stages if stage == "stream_handoff"]
    assert stages == [
        "candidate_bound", "ws_endpoint_get_entered", "ws_endpoint_get_completed",
        "ws_connected", "ws_subscribed", "ws_terminal",
    ]
    assert len(backend.session.post_calls) == len(backend.session.get_calls) == len(backend.session.ws_calls) == 1
    assert backend.session.ws.closed is True


def test_handoff_consumes_curl_fragments_with_the_shared_deadline():
    backend = _backend(_initial_events(), frames=[_ws_frame()])
    backend.session.ws_type = FragmentWebSocket

    assert list(backend.stream_conversation(prompt="fixture"))[-1] == "[DONE]"
    assert len(backend.session.post_calls) == len(backend.session.get_calls) == len(backend.session.ws_calls) == 1
    assert backend.session.ws.closed is True


def test_handoff_fragment_receive_respects_the_existing_shared_deadline():
    with pytest.raises(AccountRequestDeadlineExceeded):
        OpenAIBackendAPI._handoff_ws_text(
            FragmentWebSocket([_ws_frame()]), time.monotonic() - 0.01, 1024)


def test_handoff_fragment_receive_collects_text_continuations_and_ignores_ping():
    ws = FragmentWebSocket([
        _fragment(b'{"reply":', CurlWsFlag.TEXT | CurlWsFlag.CONT, bytesleft=12),
        _fragment(b"", CurlWsFlag.PING, bytesleft=0),
        _fragment(b'"ok"', CurlWsFlag.CONT, bytesleft=2),
        _fragment(b"}", 0, bytesleft=0),
    ])

    assert OpenAIBackendAPI._handoff_ws_text(ws, time.monotonic() + 1, 1024) == '{"reply":"ok"}'


def test_handoff_fragment_close_has_no_terminal_payload():
    ws = FragmentWebSocket([_fragment(b"", CurlWsFlag.CLOSE, bytesleft=0)])

    assert OpenAIBackendAPI._handoff_ws_text(ws, time.monotonic() + 1, 1024) == ""


def test_handoff_fragment_again_waits_only_until_shared_deadline(monkeypatch):
    ws = FragmentWebSocket([CurlError("again", CurlECode.AGAIN)])
    ticks = iter((10.0, 11.0))
    waits = []
    monkeypatch.setattr(backend_module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(backend_module.select, "select",
                        lambda _r, _w, _x, timeout: waits.append(timeout) or ([], [], []))

    with pytest.raises(AccountRequestDeadlineExceeded):
        OpenAIBackendAPI._handoff_ws_text(ws, 11.0, 1024)

    assert waits == [0.5]


def test_handoff_fragment_rejects_oversized_text_before_decode():
    ws = FragmentWebSocket([_fragment(b"too-large", CurlWsFlag.TEXT, bytesleft=0)])

    with pytest.raises(AccountRequestDeadlineExceeded):
        OpenAIBackendAPI._handoff_ws_text(ws, time.monotonic() + 1, 3)


def test_handoff_budget_terminates_the_socket_without_a_graceful_close(monkeypatch):
    backend = _backend(_initial_events(), frames=[_ws_frame()])

    def deadline(*_args, **_kwargs):
        raise AccountRequestDeadlineExceeded("fixture deadline")

    monkeypatch.setattr(OpenAIBackendAPI, "_handoff_ws_text", staticmethod(deadline))
    handoff = SimpleNamespace(topic_id="conversation-turn-turn-1")

    assert list(backend._consume_stream_handoff(handoff, time.monotonic() + 5)) == []
    assert backend.session.ws.terminated is True


def test_unbound_stream_does_not_open_a_handoff_transport():
    backend = _backend(_initial_events(), frames=[_ws_frame()])
    backend.retain_bound_conversation = False

    assert list(backend.stream_conversation(prompt="fixture"))[-1] == "[DONE]"
    assert backend.session.get_calls == [] and backend.session.ws_calls == []


def test_bound_image_model_with_ste_does_not_open_text_handoff_transport():
    backend = _backend(_initial_events(handoff=False, ste=True), frames=[_ws_frame()])
    # Even an accidental text marker cannot override the image-model guard.
    backend.text_stream_handoff_enabled = True

    assert list(backend.stream_conversation(prompt="fixture", model="gpt-image-2"))[-1] == "[DONE]"
    assert len(backend.session.post_calls) == 1
    assert backend.session.get_calls == [] and backend.session.ws_calls == []


def test_handoff_rejects_cross_conversation_before_any_ws_endpoint_read():
    backend = _backend(_initial_events(conversation="conversation-a"))
    backend.session.initial = Response([
        _line({"type": "resume_conversation_token", "kind": "topic", "token": "PRIVATE", "conversation_id": "conversation-a"}),
        _line(_handoff(conversation="conversation-b")), b"data: [DONE]",
    ])

    assert list(backend.stream_conversation(prompt="fixture"))[-1] == "[DONE]"
    assert len(backend.session.post_calls) == 1
    assert backend.session.get_calls == [] and backend.session.ws_calls == []


def test_handoff_rejects_topic_that_does_not_bind_its_turn_before_any_ws_endpoint_read():
    backend = _backend(_initial_events())
    backend.session.initial = Response([
        _line({"type": "resume_conversation_token", "kind": "topic", "token": "PRIVATE", "conversation_id": "conversation-1"}),
        _line(_handoff(topic="conversation-turn-a-different-turn")), b"data: [DONE]",
    ])

    assert list(backend.stream_conversation(prompt="fixture"))[-1] == "[DONE]"
    assert backend.session.get_calls == [] and backend.session.ws_calls == []


def test_handoff_rejects_untrusted_ws_url_and_never_connects():
    backend = _backend(_initial_events(), ws_url="wss://ws.chatgpt.com.evil.example:443/private")

    assert list(backend.stream_conversation(prompt="fixture"))[-1] == "[DONE]"
    assert len(backend.session.get_calls) == 1
    assert backend.session.ws_calls == []


def test_initial_terminal_skips_handoff_even_when_its_parent_is_a_thought_node():
    backend = _backend(_initial_events(terminal=True), frames=[_ws_frame()])

    assert list(backend.stream_conversation(prompt="fixture"))[-1] == "[DONE]"
    assert backend.session.get_calls == [] and backend.session.ws_calls == []


@pytest.mark.parametrize("channel,content_type", [
    ("analysis", "text"), ("commentary", "text"), (None, "thoughts"),
    (None, "reasoning_recap"), ("final", "thoughts"),
])
@pytest.mark.parametrize("phase", ["initial_sse", "ws_catchup"])
@pytest.mark.parametrize("terminal_shape", ["full_message", "sparse_patch"])
def test_finished_reasoning_phase_keeps_reading_until_same_turn_final(channel, content_type, phase, terminal_shape):
    thought = {"v": {"conversation_id": "conversation-1", "message": {
        "id": "phase-end", "parent": "original-user", "author": {"role": "assistant"},
        "status": "finished_successfully", "end_turn": True, "channel": channel,
        "content": {"content_type": content_type, "parts": ["intermediate phase"]},
    }}}
    final = {"v": {"conversation_id": "conversation-1", "message": {
        "id": "actual-final", "parent": "phase-end", "author": {"role": "assistant"},
        "status": "finished_successfully", "end_turn": True, "channel": "final",
        "content": {"content_type": "text", "parts": ["complete answer"]},
    }}}

    def frame(event):
        return json.dumps({"type": "message", "topic_id": "conversation-turn-turn-1",
            "payload": {"payload": {"encoded_item": "data: " + json.dumps(event) + "\n" + "data: [DONE]"}}})

    phase_events = [thought]
    if terminal_shape == "sparse_patch":
        thought["v"]["message"].update(status="in_progress", end_turn=False)
        phase_events.append({"v": {"conversation_id": "conversation-1", "message": {
            "id": "phase-end", "author": {"role": "assistant"},
            "status": "finished_successfully", "end_turn": True,
        }}})
    initial = _initial_events()
    if phase == "initial_sse":
        initial.lines[-1:-1] = list(map(_line, phase_events))
        frames = [frame(final)]
    else:
        frames = [*map(frame, phase_events), frame(final)]
    backend = _backend(initial, frames=frames)

    payloads = list(backend.stream_conversation(prompt="fixture"))

    final_index = next(index for index, payload in enumerate(payloads) if "actual-final" in payload)
    assert "[DONE]" not in payloads[:final_index]
    assert payloads.count("[DONE]") == 1
    assert len(backend.session.post_calls) == 1
    assert len(backend.session.get_calls) == len(backend.session.ws_calls) == 1
    assert len(backend.session.ws.sent) == 1
    assert backend.session.ws.closed is True


def test_later_user_branch_never_turns_a_nonterminal_sse_parent_chain_into_original_final():
    backend = _backend(Response([
        _line({"type": "resume_conversation_token", "kind": "topic", "token": "PRIVATE", "conversation_id": "conversation-1"}),
        _line(_handoff()),
        _line({"v": {"message": {"id": "later-user", "author": {"role": "user"}, "parent": "original-user"}}}),
        _line({"v": {"message": {"id": "old-thought", "author": {"role": "assistant"}, "parent": "later-user"}}}),
        _line({"v": {"message": {"id": "old-final", "author": {"role": "assistant"}, "status": "finished_successfully", "end_turn": True, "parent": "old-thought"}}}),
        b"data: [DONE]",
    ]), frames=[_ws_frame()])

    assert list(backend.stream_conversation(prompt="fixture", conversation_id="conversation-1"))[-1] == "[DONE]"
    assert len(backend.session.get_calls) == len(backend.session.ws_calls) == 1


def test_explicit_final_text_update_replaces_same_message_reasoning_phase():
    handoff = _StreamHandoff(expected_conversation_id="conversation-1", original_message_id="original-user")
    message = {"id": "phase-node", "parent": "original-user", "author": {"role": "assistant"},
        "status": "in_progress", "end_turn": False, "channel": "analysis",
        "content": {"content_type": "thoughts"}}
    handoff.observe(json.dumps({"message": message}))
    assert handoff.terminal_seen is False
    message.update(status="finished_successfully", end_turn=True, channel="final",
        content={"content_type": "text", "parts": ["complete answer"]})
    handoff.observe(json.dumps({"message": message}))
    assert handoff.terminal_seen is True
    assert len(handoff._message_phases) <= len(handoff._parents)


def test_terminal_from_an_old_branch_does_not_suppress_this_turns_handoff():
    backend = _backend(_initial_events(terminal=True, terminal_parent="old-user"), frames=[_ws_frame()])

    assert list(backend.stream_conversation(prompt="fixture"))[-1] == "[DONE]"
    assert len(backend.session.get_calls) == 1
    assert len(backend.session.ws_calls) == 1


def test_cross_conversation_ws_payload_is_discarded_and_stays_unknown():
    backend = _backend(_initial_events(), frames=[_ws_frame().replace('conversation-1', 'conversation-other')])

    assert list(backend.stream_conversation(prompt="fixture")) == [
        json.dumps(event, separators=(",", ":")) for event in [
            {"type": "resume_conversation_token", "kind": "topic", "token": "PRIVATE", "conversation_id": "conversation-1"},
            _handoff(),
        ]
    ] + ["[DONE]"]
    assert len(backend.session.ws_calls) == 1


def test_same_conversation_old_branch_final_is_not_yielded_or_settled():
    old = json.dumps({"type": "reply", "reply": {"topic_id": "conversation-turn-turn-1", "catchups": [
        {"payload": {"payload": {"encoded_item": "\n".join((
            'data: {"v":{"conversation_id":"conversation-1","message":{"id":"old-final","author":{"role":"assistant"},"status":"finished_successfully","end_turn":true,"parent":"old-user"}}}',
            "data: [DONE]",
        ))}}},
    ]}})
    backend = _backend(_initial_events(), frames=[old])

    payloads = list(backend.stream_conversation(prompt="fixture"))

    assert all("old-final" not in payload for payload in payloads)
    assert payloads[-1] == "[DONE]"


def test_server_ste_metadata_before_resume_binds_once_after_same_conversation_arrives():
    backend = _backend(Response([
        _line(_ste(conversation="conversation-1")),
        _line({"type": "resume_conversation_token", "kind": "topic", "token": "PRIVATE", "conversation_id": "conversation-1"}),
        b"data: [DONE]",
    ]), frames=[_ws_frame()])

    payloads = list(backend.stream_conversation(prompt="fixture", conversation_id="conversation-1"))

    assert payloads[-1] == "[DONE]"
    assert len(backend.session.post_calls) == 1
    assert len(backend.session.get_calls) == len(backend.session.ws_calls) == 1


def test_server_ste_before_resume_with_a_different_conversation_stays_unknown():
    backend = _backend(Response([
        _line(_ste(conversation="conversation-1")),
        _line({"type": "resume_conversation_token", "kind": "topic", "token": "PRIVATE", "conversation_id": "conversation-other"}),
        b"data: [DONE]",
    ]), frames=[_ws_frame()])

    assert list(backend.stream_conversation(prompt="fixture", conversation_id="conversation-1"))[-1] == "[DONE]"
    assert backend.session.get_calls == backend.session.ws_calls == []


def test_duplicate_ste_candidates_stay_unknown_before_any_ws_endpoint_read():
    backend = _backend(Response([
        _line(_ste(conversation="conversation-1")),
        _line(_ste(conversation="conversation-1")),
        _line({"type": "resume_conversation_token", "kind": "topic", "token": "PRIVATE", "conversation_id": "conversation-1"}),
        b"data: [DONE]",
    ]), frames=[_ws_frame()])

    assert list(backend.stream_conversation(prompt="fixture", conversation_id="conversation-1"))[-1] == "[DONE]"
    assert backend.session.get_calls == backend.session.ws_calls == []


def test_server_ste_metadata_turn_opens_one_bound_ws_handoff():
    backend = _backend(_initial_events(handoff=False, ste=True), frames=[_ws_frame()])

    payloads = list(backend.stream_conversation(prompt="fixture"))

    assert payloads[-1] == "[DONE]"
    assert len(backend.session.post_calls) == 1
    assert len(backend.session.get_calls) == 1
    assert len(backend.session.ws_calls) == 1
    assert json.loads(backend.session.ws.sent[0])[1]["command"]["topic_id"] == "conversation-turn-turn-1"
    assert backend.session.ws.closed is True


def test_server_ste_metadata_without_turn_or_with_conflicting_turn_stays_unknown():
    missing = _backend(_initial_events(handoff=False, ste=False), frames=[_ws_frame()])
    missing.session.initial = Response([
        _line({"type": "resume_conversation_token", "kind": "topic", "token": "PRIVATE", "conversation_id": "conversation-1"}),
        _line({"type": "server_ste_metadata", "conversation_id": "conversation-1", "metadata": {}}), b"data: [DONE]",
    ])
    conflict = _backend(_initial_events(handoff=False, ste=True), frames=[_ws_frame()])
    conflict.session.initial = Response([
        _line({"type": "resume_conversation_token", "kind": "topic", "token": "PRIVATE", "conversation_id": "conversation-1"}),
        _line(_ste(top_level=True, conflict=True)), b"data: [DONE]",
    ])

    assert list(missing.stream_conversation(prompt="fixture"))[-1] == "[DONE]"
    assert list(conflict.stream_conversation(prompt="fixture"))[-1] == "[DONE]"
    assert missing.session.get_calls == missing.session.ws_calls == []
    assert conflict.session.get_calls == conflict.session.ws_calls == []


def test_unconfirmed_ws_done_in_one_catchup_does_not_end_the_turn_before_late_final():
    late = json.dumps({"type": "reply", "reply": {"topic_id": "conversation-turn-turn-1", "catchups": [
        {"payload": {"payload": {"encoded_item": "\n".join((
            'data: {"v":{"conversation_id":"conversation-1","message":{"id":"thought-leaf","author":{"role":"assistant"},"parent":"original-user"}}}',
            "data: [DONE]",
            'data: {"v":{"conversation_id":"conversation-1","message":{"id":"final-leaf","author":{"role":"assistant"},"status":"finished_successfully","end_turn":true,"parent":"thought-leaf"}}}',
        ))}}},
    ]}})
    backend = _backend(_initial_events(), frames=[late])

    payloads = list(backend.stream_conversation(prompt="fixture"))

    final_index = next(index for index, payload in enumerate(payloads) if "final-leaf" in payload)
    assert "[DONE]" not in payloads[:final_index]
    assert payloads[-1] == "[DONE]"


def test_unconfirmed_catchup_done_keeps_same_subscription_for_later_live_terminal():
    catchup = json.dumps({"type": "reply", "reply": {"topic_id": "conversation-turn-turn-1", "catchups": [
        {"payload": {"payload": {"encoded_item": "\n".join((
            'data: {"v":{"conversation_id":"conversation-1","message":{"id":"thought-leaf","author":{"role":"assistant"},"parent":"original-user"}}}',
            "data: [DONE]",
        ))}}},
    ]}})
    live = json.dumps({"type": "message", "topic_id": "conversation-turn-turn-1", "payload": {"payload": {
        "encoded_item": "\n".join((
            'data: {"v":{"conversation_id":"conversation-1","message":{"id":"final-leaf","author":{"role":"assistant"},"status":"finished_successfully","end_turn":true,"parent":"thought-leaf"}}}',
            "data: [DONE]",
        )),
    }}})
    backend = _backend(_initial_events(), frames=[catchup, live])

    payloads = list(backend.stream_conversation(prompt="fixture"))

    assert any("thought-leaf" in payload for payload in payloads)
    final_index = next(index for index, payload in enumerate(payloads) if "final-leaf" in payload)
    assert "[DONE]" not in payloads[:final_index]
    assert payloads.count("[DONE]") == 1
    assert len(backend.session.post_calls) == 1
    assert len(backend.session.get_calls) == len(backend.session.ws_calls) == 1
    assert backend.session.ws.closed is True


def _handoff_with_elapsed_setup(monkeypatch, *, connect_seconds=39.5, subscribe_seconds=0.0):
    backend = _backend(_initial_events(), frames=[_ws_frame()])
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(backend_module.time, "monotonic", lambda: clock.now)
    original_get, original_connect = backend.session.get, backend.session.ws_connect
    original_send = backend._handoff_ws_send

    def get(*args, **kwargs):
        result = original_get(*args, **kwargs)
        clock.now += 20.0
        return result

    def connect(*args, **kwargs):
        result = original_connect(*args, **kwargs)
        clock.now += connect_seconds
        return result

    def subscribe(ws, data):
        original_send(ws, data)
        clock.now += subscribe_seconds

    monkeypatch.setattr(backend.session, "get", get)
    monkeypatch.setattr(backend.session, "ws_connect", connect)
    monkeypatch.setattr(backend, "_handoff_ws_send", subscribe)
    deadlines = []
    def receive(ws, deadline, max_bytes):
        deadlines.append(deadline)
        return ws.recv_str()
    monkeypatch.setattr(backend, "_handoff_ws_text", receive)
    state = backend_module._StreamHandoff(
        expected_conversation_id="conversation-1", original_message_id="original-user",
        resume_conversation_id="conversation-1", handoff_conversation_id="conversation-1",
        turn_exchange_id="turn-1", topic_id="conversation-turn-turn-1", handoff_count=1,
    )
    return backend, state, deadlines


@pytest.mark.parametrize("request_deadline,expected_receive_deadline", [(1000.0, 219.5), (180.0, 180.0)])
def test_handoff_setup_does_not_consume_receive_window_or_extend_original_deadline(
        monkeypatch, request_deadline, expected_receive_deadline):
    backend, state, deadlines = _handoff_with_elapsed_setup(monkeypatch)

    assert list(backend._consume_stream_handoff(state, request_deadline))[-1] == "[DONE]"
    assert backend.session.get_calls[0][1]["timeout"] == 60.0
    assert backend.session.ws_calls[0][1]["timeout"] == 40.0
    assert deadlines == [expected_receive_deadline]


@pytest.mark.parametrize("connect_seconds,subscribe_seconds,subscribed", [
    (41.0, 0.0, False),
    (39.5, 1.0, True),
])
def test_late_setup_cannot_restart_receive_budget(monkeypatch, connect_seconds, subscribe_seconds, subscribed):
    backend, state, deadlines = _handoff_with_elapsed_setup(
        monkeypatch, connect_seconds=connect_seconds, subscribe_seconds=subscribe_seconds,
    )
    assert list(backend._consume_stream_handoff(state, 1000.0)) == []
    assert bool(backend.session.ws.sent) is subscribed
    assert deadlines == []
    assert backend.session.ws.terminated is True


def test_handoff_refuses_an_exhausted_global_deadline_before_setup():
    backend = _backend(_initial_events(), frames=[_ws_frame()])
    handoff = SimpleNamespace(topic_id="conversation-turn-turn-1")

    assert list(backend._consume_stream_handoff(handoff, time.monotonic() - 0.01)) == []
    assert backend.session.post_calls == backend.session.get_calls == backend.session.ws_calls == []
