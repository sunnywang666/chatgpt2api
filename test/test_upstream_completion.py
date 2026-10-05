"""Completion notifications only wake the existing original-image readback."""
import json
import time
import threading
from types import SimpleNamespace

import pytest
from curl_cffi import CurlWsFlag

from services import upstream_completion as completion
from services.openai_backend_api import OpenAIBackendAPI, ImageActiveDeadlineExceeded


TOPIC = "conv-turn-low-ttl-original"
CID = "original-conversation"
URL = "wss://ws.chatgpt.com/channel?private=never-log"
HANDOFF = {"type": "stream_handoff", "turn_exchange_id": "original-exchange",
           "options": [{"type": "subscribe_ws_topic", "topic_id": TOPIC}]}


def event(kind="done", cid=CID, turn="original-turn", topic=TOPIC):
    return {"type": "message", "topic_id": topic, "payload": {
        "type": "conversation-turn-stream", "payload": {
            "type": kind, "conversation_id": cid, "turn_id": turn,
            **({"encoded_item": "data: {}", "stream_item_id": "item", "parent_stream_item_id": None}
               if kind == "stream-item" else {})}}}


def ack(**extra):
    return {"id": 2, "type": "reply", "reply": {
        "type": "subscribe", "topic_id": TOPIC, "recovered": True, **extra}}


class Socket:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.sent = []
        self.closed = False

    def send(self, body, **kwargs):
        self.sent.extend(json.loads(body))

    def recv_fragment(self):
        return json.dumps(next(self.frames)).encode(), SimpleNamespace(flags=CurlWsFlag.TEXT, bytesleft=0)

    def close(self):
        self.closed = True


class Session:
    def __init__(self, frames):
        self.ws = Socket(frames)
        self.connected = []

    def ws_connect(self, url, **kwargs):
        self.connected.append(url)
        return self.ws


def wait(session, **kwargs):
    return completion.wait_for_turn_done(session, URL, TOPIC, CID, time.monotonic() + 60,
                                         remaining=lambda: 60, **kwargs)


@pytest.mark.parametrize("typed_reply", [False, True])
def test_catchup_done_accepts_both_live_reply_shapes(typed_reply):
    reply = ack(catchups=[event("heartbeat"), event("stream-item"), event()])
    if not typed_reply:
        reply.pop("type")
    session = Session([[reply]])
    assert wait(session) == "done"
    assert session.ws.sent[1]["command"] == {"type": "subscribe", "topic_id": TOPIC, "offset": "0"}
    assert session.ws.closed


@pytest.mark.parametrize("frames,reason", [
    ([[ack(recovered=False)]], "history_unavailable"),
    ([[event()]], "unconfirmed_subscription"),
    ([[ack(), event(cid="another")]], "wrong_conversation"),
    ([[ack(), event(topic="conversation-another")]], "wrong_topic"),
    ([[ack(), event("heartbeat"), event(turn="late-turn")]], "wrong_turn"),
    ([[ack(), event("unexpected")]], "invalid_event"),
    ([{}], "invalid_frame"),
])
def test_failed_notification_never_means_result_success(frames, reason):
    session = Session(frames)
    assert wait(session) == reason
    assert session.ws.closed


def test_heartbeat_and_items_do_not_finish_wait():
    session = Session([[ack()], [event("heartbeat", cid=None)], [event("stream-item")], [event()]])
    assert wait(session) == "done"
    assert list(session.ws.frames) == []


@pytest.mark.parametrize("url", ["https://ws.chatgpt.com/x", "wss://evil.example/x",
    "wss://ws.chatgpt.com.evil.example/x", "wss://user:pass@ws.chatgpt.com/x", "wss://ws.chatgpt.com:888/x"])
def test_no_credentials_sent_to_untrusted_websocket(url):
    session = Session([])
    assert completion.wait_for_turn_done(session, url, TOPIC, CID, time.monotonic()+60,
                                         remaining=lambda: 60) == "invalid_url"
    assert not session.connected


def test_total_deadline_is_not_extended_by_notifications():
    session = Session([])
    assert completion.wait_for_turn_done(session, URL, TOPIC, CID, time.monotonic()-1,
                                         remaining=lambda: 60) == "deadline"
    assert not session.connected


def test_active_attempt_deadline_propagates_and_closes_socket():
    session = Session([[ack()]])
    calls = []
    def remaining():
        calls.append(True)
        if len(calls) > 1:
            raise ImageActiveDeadlineExceeded("expired")
        return 60
    with pytest.raises(ImageActiveDeadlineExceeded):
        completion.wait_for_turn_done(session, URL, TOPIC, CID, time.monotonic()+60, remaining=remaining)
    assert session.ws.closed


def backend(monkeypatch, outcome):
    instance = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
    instance.base_url = "https://chatgpt.com"
    instance.progress_callback = None
    instance.account = {}
    instance.fp = {"impersonate": "chrome110"}
    instance._headers = lambda path: {}
    response = SimpleNamespace(status_code=200, json=lambda: {"websocket_url": URL}, close=lambda: None)
    gets = []
    instance.session = SimpleNamespace(get=lambda url, **kw: gets.append(url) or response)
    calls = []
    def receive(session, url, topic, cid, deadline, **kwargs):
        calls.append((topic, cid))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(completion, "wait_for_turn_done", receive)
    return instance, gets, calls


def test_new_conversation_id_from_original_sse_and_done_wakes_existing_reader(monkeypatch):
    instance, gets, calls = backend(monkeypatch, "done")
    closed = []
    def payloads():
        try:
            yield json.dumps({"conversation_id": CID})
            yield json.dumps(HANDOFF)
            yield "[DONE]"
            raise AssertionError("must not keep waiting after done hint")
        finally:
            closed.append(True)
    result = list(instance._iter_image_completion_payloads(payloads(), "", time.monotonic()+60))
    assert result[-1] == "[DONE]"
    # No output/asset is synthesized; protocol must still perform original GET.
    assert not any("file-service" in value for value in result)
    assert calls == [(TOPIC, CID)]
    assert len(gets) == 1 and closed == [True]


@pytest.mark.parametrize("outcome", ["timeout", "history_unavailable", RuntimeError(URL)])
def test_fallback_retains_original_sse_without_generation_replay(monkeypatch, outcome):
    instance, gets, calls = backend(monkeypatch, outcome)
    original = [json.dumps(HANDOFF), json.dumps(HANDOFF), "original-payload", "[DONE]"]
    assert list(instance._iter_image_completion_payloads(iter(original), CID, time.monotonic()+60)) == original
    assert calls == [(TOPIC, CID)] and len(gets) == 1


def test_no_handoff_or_no_original_conversation_keeps_old_path(monkeypatch):
    instance, gets, calls = backend(monkeypatch, "done")
    original = [json.dumps(HANDOFF), "[DONE]"]
    assert list(instance._iter_image_completion_payloads(iter(original), "", time.monotonic()+60)) == original
    assert not gets and not calls


def test_buffered_sse_terminal_takes_precedence_over_websocket(monkeypatch):
    instance, gets, calls = backend(monkeypatch, "timeout")
    started = threading.Event()
    exited = threading.Event()
    def listening(*args, remaining, **kwargs):
        started.set()
        until = time.monotonic() + 2
        while remaining() > 0 and time.monotonic() < until:
            time.sleep(.005)
        exited.set()
        return "timeout"
    monkeypatch.setattr(completion, "wait_for_turn_done", listening)
    original = [json.dumps(HANDOFF), json.dumps({"message": {
        "end_turn": True, "status": "finished_successfully", "author": {"role": "assistant"}}}), "[DONE]"]
    def payloads():
        yield original[0]
        assert started.wait(2)
        yield from original[1:]
    before = time.monotonic()
    assert list(instance._iter_image_completion_payloads(payloads(), CID, time.monotonic()+60)) == original
    assert time.monotonic() - before < 1
    assert exited.is_set()


def test_websocket_done_wakes_handoff_sse_that_has_not_closed(monkeypatch):
    instance, gets, calls = backend(monkeypatch, "done")
    closed = threading.Event()
    def payloads():
        yield json.dumps(HANDOFF)
        assert closed.wait(2), "SSE must be woken by the done signal"
    result = list(instance._iter_image_completion_payloads(payloads(), CID, time.monotonic()+60, closed.set))
    assert result[-1] == "[DONE]" and calls == [(TOPIC, CID)]


def test_malformed_item_cannot_keep_connection_alive():
    item = event("stream-item")
    del item["payload"]["payload"]["encoded_item"]
    assert wait(Session([[ack(), item]])) == "invalid_event"


@pytest.mark.parametrize("prefix", ["conversation-", "conv-turn-low-ttl-"])
def test_handoff_requires_nonempty_topic(prefix):
    assert completion.handoff_topic({**HANDOFF, "options": [{"type": "subscribe_ws_topic", "topic_id": prefix}]}) is None
