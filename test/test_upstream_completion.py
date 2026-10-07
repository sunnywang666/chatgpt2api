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


def global_image_event(cid=CID, **updates):
    message = {"author": {"role": "tool"}, "status": "finished_successfully", "channel": "final",
               "content": {"content_type": "multimodal_text"}, "metadata": {"ghostrider": {"status": "final"}}}
    message.update(updates)
    return {"type": "conversation-update", "payload": {"conversation_id": cid,
        "update_type": "add-messages", "update_content": {"messages": [None, message]}}}


@pytest.mark.parametrize("update", [
    {"status": "in_progress"}, {"metadata": {"ghostrider": {"status": "intermediate"}}},
    {"author": {"role": "assistant"}, "end_turn": True}, {"content": {"content_type": "text"}},
    {"channel": "commentary"}, {"metadata": None},
])
def test_global_hint_requires_final_image_tool_not_transport_or_assistant_end(update):
    assert completion.completed_conversation_hint(global_image_event(**update)) is None
    assert completion.completed_conversation_hint({"type": "conversation-turn-complete", "payload": {"conversation_id": CID}}) is None
    assert completion.completed_conversation_hint(global_image_event()) == CID


@pytest.mark.parametrize("wrong", ["none", "cid", "topic", "before-ack"])
def test_global_socket_routes_only_current_conversation_and_closes(wrong):
    reply = {"id": 2, "reply": {"type": "subscribe", "topic_id": "conversations", "recovered": False}}
    message = {"type": "message", "topic_id": "foreign" if wrong == "topic" else "conversations",
               "payload": global_image_event("foreign" if wrong == "cid" else CID)}
    session = Session([[message, reply] if wrong == "before-ack" else [reply, message]])
    hub = completion._ConversationHints()
    signal = threading.Event(); hub.signals[CID] = {signal}
    cleanup = []
    hub.run(lambda: (session, URL, {}, lambda: cleanup.append(True), time.monotonic() + 10))
    assert signal.is_set() is (wrong == "none")
    assert session.ws.sent[1]["command"] == {"type": "subscribe", "topic_id": "conversations"}
    assert session.ws.closed and cleanup == [True] and hub.stopped.is_set()


def test_active_account_shares_one_socket_and_last_release_stops_it(monkeypatch):
    started, runs = threading.Event(), []
    def run(hub, factory):
        runs.append(factory); started.set(); hub.stopped.wait(2)
    monkeypatch.setattr(completion._ConversationHints, "run", run)
    opener = lambda: None
    with completion.image_completion_hints("fixture", CID, opener) as first:
        assert started.wait(1)
        hub = completion._hubs["fixture"]
        with completion.image_completion_hints("fixture", "second", opener) as second:
            assert len(runs) == 1
            with hub.lock:
                for signal in hub.signals[CID]: signal.set()
            assert first(.01) is True and second(.01) is False
        assert not hub.stopped.is_set()
    assert hub.stopped.is_set() and "fixture" not in completion._hubs and not hub.thread.is_alive()


def test_global_listener_observations_distinguish_subscription_and_routing_without_private_data(monkeypatch):
    records = []
    monkeypatch.setattr(completion.logger, "info", records.append)
    reply = {"id": 2, "reply": {"type": "subscribe", "topic_id": "conversations"}}
    session = Session([[global_image_event(), reply, global_image_event("private-foreign"),
                        global_image_event(status="in_progress"), global_image_event(status="in_progress"),
                        global_image_event("private-second", status="in_progress"), global_image_event()]])
    hub = completion._ConversationHints("private-account")
    signal = threading.Event(); hub.signals[CID] = {signal}
    second = threading.Event(); hub.signals["private-second"] = {second}
    hub.run(lambda: (session, URL, {}, lambda: None, time.monotonic() + 10))
    assert signal.is_set()
    assert [r["stage"] for r in records][:4] == [
        "opening_transport", "socket_connected", "subscription_requested", "subscribed"]
    matched = next(r for r in records if r["stage"] == "hint_matched")
    assert matched["conversation_ref"] == completion.safe_account_ref(CID)
    assert matched["waiting_contexts"] == 1
    nonfinal = [r for r in records if r["stage"] == "hint_nonfinal"]
    assert len(nonfinal) == 2  # Repeated updates on one CID are coalesced.
    assert {r["conversation_ref"] for r in nonfinal} == {
        completion.safe_account_ref(CID), completion.safe_account_ref("private-second")}
    assert all(r["reason"] == "no_final_image_tool" for r in nonfinal)
    assert not second.is_set()
    stopped = records[-1]
    assert stopped["subscribed"] is True
    assert stopped["counts"] == dict(before_subscription=1, other_topic=0,
                                     unregistered_conversation=1, active_nonfinal=3, matched_hints=1)
    serialized = json.dumps(records)
    for private in (URL, CID, "private-account", "private-foreign", "private-second", "ghostrider", "update_content"):
        assert private not in serialized


def test_listener_connection_exception_logs_only_fixed_class_and_preserves_fallback(monkeypatch):
    records = []
    monkeypatch.setattr(completion.logger, "info", records.append)
    hub = completion._ConversationHints("private-account")
    def fail():
        raise RuntimeError("secret cookie and signed URL: " + URL)
    hub.run(fail)
    assert hub.stopped.is_set()
    assert records[-1]["reason"] == "transport_exception"
    assert records[-1]["subscribed"] is False
    assert next(r for r in records if r["stage"] == "transport_error")["error_type"] == "RuntimeError"
    assert "secret cookie" not in json.dumps(records) and URL not in json.dumps(records)


def test_listener_logging_failure_cannot_drop_completion_hint(monkeypatch):
    def broken_log(_record):
        raise RuntimeError("log unavailable")
    monkeypatch.setattr(completion.logger, "info", broken_log)
    reply = {"id": 2, "reply": {"type": "subscribe", "topic_id": "conversations"}}
    session = Session([[reply, global_image_event()]])
    hub = completion._ConversationHints()
    signal = threading.Event(); hub.signals[CID] = {signal}
    hub.run(lambda: (session, URL, {}, lambda: None, time.monotonic() + 10))
    assert signal.is_set() and session.ws.closed and hub.stopped.is_set()


def test_active_shared_listener_survives_first_callers_five_minute_window(monkeypatch):
    from curl_cffi import CurlECode, CurlError
    now = [0.0]
    monkeypatch.setattr(completion.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(completion.select, "select", lambda *_args: ([], [], []))
    hub = completion._ConversationHints()
    original, later = threading.Event(), threading.Event()
    hub.signals[CID] = {original}
    reply = {"id": 2, "reply": {"type": "subscribe", "topic_id": "conversations"}}
    session = Session([[reply], [global_image_event("later")]])
    recv = session.ws.recv_fragment
    calls = [0]
    def receive():
        calls[0] += 1
        if calls[0] == 2:
            now[0] = 301
            hub.signals["later"] = {later}
            raise CurlError("try again", CurlECode.AGAIN)
        return recv()
    session.ws.curl = SimpleNamespace(getinfo=lambda _info: 0)
    session.ws.recv_fragment = receive
    hub.run(lambda: (session, URL, {}, lambda: None, 10))
    assert later.is_set() and not original.is_set()
    assert len(session.connected) == 1 and session.ws.closed


def test_disconnected_shared_listener_preserves_fallback_interval(monkeypatch):
    def disconnect(hub, _factory):
        hub.stopped.set()
    monkeypatch.setattr(completion._ConversationHints, "run", disconnect)
    sleeps = []
    monkeypatch.setattr(completion.time, "sleep", sleeps.append)
    with completion.image_completion_hints("disconnected-fixture", CID, lambda: None) as wait:
        completion._hubs["disconnected-fixture"].thread.join(1)
        assert wait(10, before_first_read=True) is False
        assert wait(10) is False
    assert sleeps == [10] and "disconnected-fixture" not in completion._hubs


def test_new_conversation_replaces_dead_listener_without_old_release_stopping_it(monkeypatch):
    runs = []
    def run(hub, _factory):
        runs.append(hub)
        if len(runs) == 1:
            hub.stopped.set()
        else:
            hub.stopped.wait(2)
    monkeypatch.setattr(completion._ConversationHints, "run", run)
    account = "reconnect-fixture"
    old_context = completion.image_completion_hints(account, CID, lambda: None)
    old_wait = old_context.__enter__()
    old_hub = completion._hubs[account]; old_hub.thread.join(1)
    old_released = False
    try:
        assert old_hub.stopped.is_set()
        with completion.image_completion_hints(account, "new-conversation", lambda: None) as new_wait:
            new_hub = completion._hubs[account]
            assert new_hub is not old_hub and not new_hub.stopped.is_set()
            assert old_wait(.001) is False
            old_context.__exit__(None, None, None); old_released = True
            assert completion._hubs[account] is new_hub and not new_hub.stopped.is_set()
            with new_hub.lock:
                for signal in new_hub.signals["new-conversation"]: signal.set()
            assert new_wait(.01) is True
        assert new_hub.stopped.is_set() and account not in completion._hubs
        assert len(runs) == 2  # No automatic retry loop or generation replay.
    finally:
        if not old_released:
            old_context.__exit__(None, None, None)


def test_last_real_context_closes_idle_socket(monkeypatch):
    from curl_cffi import CurlECode, CurlError
    idle, cleanup = threading.Event(), []
    session = Session([])
    session.ws.curl = SimpleNamespace(getinfo=lambda _info: 0)
    def receive():
        raise CurlError("try again", CurlECode.AGAIN)
    def wait_readable(*_args):
        idle.set()
        completion._hubs["idle-fixture"].stopped.wait(.01)
        return [], [], []
    session.ws.recv_fragment = receive
    monkeypatch.setattr(completion.select, "select", wait_readable)
    opener = lambda: (session, URL, {}, lambda: cleanup.append(True), time.monotonic() + 10)
    with completion.image_completion_hints("idle-fixture", CID, opener):
        hub = completion._hubs["idle-fixture"]
        assert idle.wait(1)
    assert session.ws.closed and cleanup == [True] and not hub.thread.is_alive()
    assert "idle-fixture" not in completion._hubs


def test_cancel_during_handshake_does_not_subscribe_after_last_release():
    connecting, finish_connect, cleanup = threading.Event(), threading.Event(), []
    session = Session([])
    def connect(*_args, **_kwargs):
        connecting.set()
        assert finish_connect.wait(2)
        return session.ws
    session.ws_connect = connect
    opener = lambda: (session, URL, {}, lambda: cleanup.append(True), time.monotonic() + 10)
    with completion.image_completion_hints("connecting-fixture", CID, opener):
        hub = completion._hubs["connecting-fixture"]
        assert connecting.wait(1)
    finish_connect.set()
    hub.thread.join(1)
    assert not hub.thread.is_alive() and session.ws.sent == []
    assert session.ws.closed and cleanup == [True]


def test_expired_image_never_starts_notification_transport():
    backend = object.__new__(OpenAIBackendAPI)
    backend.account = {"provider_account_identity": "expired-fixture"}
    backend.progress_callback = SimpleNamespace(active_deadline_at=time.time() - 1)
    opened = []
    backend._open_image_notification_transport = lambda deadline: opened.append(deadline)
    with pytest.raises(ImageActiveDeadlineExceeded):
        backend._poll_image_results(CID, 5, request_message_id="request")
    assert not opened and "expired-fixture" not in completion._hubs


@pytest.mark.parametrize("queue_seconds,active_seconds,expected_send", [
    (9.13, 120, True), (11, 120, False), (9.13, 10, False),
])
def test_notification_setup_keeps_a_full_http_budget_after_account_wait(
        monkeypatch, queue_seconds, active_seconds, expected_send):
    from services import account_request_pacing as pacing
    from services import openai_backend_api as backend_module
    now = [100.0]
    clock_time = SimpleNamespace(monotonic=lambda: now[0], time=lambda: 1700000000 + now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(pacing, "time", clock_time)
    monkeypatch.setattr(backend_module, "time", clock_time)
    monkeypatch.setattr(pacing, "config", SimpleNamespace(account_request_interval_secs=0.1,
        account_conversation_read_interval_secs=60, account_message_interval_secs=0))
    clock = pacing.AccountRequestClock("notification-fixture")
    clock.next_request = now[0] + queue_seconds
    sent, closed, response_closed = [], [], []
    response = SimpleNamespace(status_code=200, headers={}, json=lambda: {"websocket_url": URL},
                               close=lambda: response_closed.append(True))
    def raw(method, url, **options):
        sent.append((now[0], options))
        return response
    session = SimpleNamespace(get=lambda url, **options: clock.request(raw, "GET", url, **options))
    client = SimpleNamespace(session=session, base_url="https://chatgpt.com", _headers=lambda path: {},
        account={}, fp={"impersonate": "fixture"}, close=lambda: closed.append(True))
    backend = object.__new__(OpenAIBackendAPI)
    backend.access_token = "fixture-token"
    monkeypatch.setattr(backend_module, "OpenAIBackendAPI", lambda **options: client)
    monkeypatch.setattr(backend_module.proxy_settings, "build_session_kwargs", lambda **options: {})
    deadline = 100 + active_seconds
    if expected_send:
        transport = backend._open_image_notification_transport(deadline)
        assert len(sent) == 1 and sent[0][0] == pytest.approx(109.13)
        assert sent[0][1]["timeout"] == 10
        assert transport[-1] == deadline and response_closed == [True]
        assert not closed
        transport[3]()
    else:
        with pytest.raises(pacing.AccountReadRetryBudgetInsufficient):
            backend._open_image_notification_transport(deadline)
        assert not sent and not response_closed
    assert closed == [True] and not clock.lock.locked()


@pytest.mark.parametrize("stopped", [True, False])
def test_notification_does_not_connect_after_last_release_or_transport_deadline(stopped):
    hub = completion._ConversationHints()
    session, cleanup = Session([]), []
    if stopped: hub.stopped.set()
    hub.run(lambda: (session, URL, {}, lambda: cleanup.append(True), time.monotonic() + (10 if stopped else -1)))
    assert session.connected == [] and cleanup == ([] if stopped else [True])


def test_image_notification_wakes_original_poll_without_result_or_retry_authority(monkeypatch):
    from services.config import config
    from services.request_context import current_request
    backend = object.__new__(OpenAIBackendAPI)
    reads, waits, observed = [], [], []
    asset = "file_000000001234567890abcdef12345678"
    empty = {"conversation_id": CID, "current_node": "request", "mapping": {"request": {
        "parent": None, "message": {"id": "request", "author": {"role": "user"}}}}}
    full = {**empty, "current_node": "image", "mapping": {**empty["mapping"], "image": {
        "parent": "request", "message": {"id": "image", "author": {"role": "tool"},
            "content": {"parts": [{"asset_pointer": "sediment://" + asset}]}}}}}
    backend._get_conversation = lambda cid: reads.append(cid) or full
    backend._query_backend_tasks = lambda **kw: []
    def wake(seconds, *, before_first_read=False, max_wait_seconds=None):
        waits.append(seconds)
        assert before_first_read and not reads, "wait for the hint before reading an empty turn"
        assert 19 <= max_wait_seconds <= 20, "preserve the fallback network budget"
        return True
    monkeypatch.setitem(config.data, "image_poll_interval_secs", 10)
    monkeypatch.setitem(config.data, "image_check_before_hit_enabled", False)
    context = current_request.set(SimpleNamespace(record_stage=lambda stage: observed.append(stage)))
    try:
        files, sediments = backend._poll_image_results_inner(CID, 30, request_message_id="request", _completion_wait=wake)
    finally:
        current_request.reset(context)
    assert reads == [CID] and len(waits) == 1 and files == sediments == [asset]
    assert observed == ["upstream_image_completion_signal"]


@pytest.mark.parametrize("status,retry_after,wakes", [
    (None, None, True), (503, None, True), (429, None, False),
    (503, 3, False), (503, 0, False),
])
def test_completion_hint_interrupts_only_optional_error_backoff(monkeypatch, status, retry_after, wakes):
    from services.config import config
    from services import openai_backend_api as module
    from utils.helper import UpstreamHTTPError
    backend = object.__new__(OpenAIBackendAPI)
    reads, waits, sleeps = [], [], []
    asset = "file_000000001234567890abcdef12345678"
    def read(cid):
        reads.append(cid)
        if len(reads) == 1:
            if status is None:
                raise module.requests.exceptions.RequestException("controlled connection failure")
            raise UpstreamHTTPError("controlled read", status, {}, retry_after=retry_after)
        return {"fixture": "original"}
    backend._get_conversation = read
    backend._extract_image_tool_records = lambda doc, rid: [{"file_ids": [asset], "sediment_ids": []}]
    backend._query_backend_tasks = lambda **kw: pytest.fail("completed result needs no task diagnosis")
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    monkeypatch.setitem(config.data, "image_check_before_hit_enabled", False)
    result = backend._poll_image_results_inner(CID, 20, request_message_id="request",
                                               initial_file_ids=[asset],
                                               _completion_wait=lambda seconds: waits.append(seconds) or True)
    assert reads == [CID, CID] and result == ([asset], [])
    assert bool(waits) is wakes and bool(sleeps) is not wakes


def test_notification_poll_defers_task_diagnosis_to_final_budget_window(monkeypatch):
    from services.config import config
    from services import openai_backend_api as module
    backend = object.__new__(OpenAIBackendAPI)
    now, reads, diagnostics = [0.0], [], []
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    monkeypatch.setitem(config.data, "image_poll_interval_secs", 10)
    backend._get_conversation = lambda cid: reads.append((cid, now[0])) or {}
    backend._extract_image_tool_records = lambda *_: []
    backend._find_content_policy_error_in_conversation = lambda *_: None
    backend._query_backend_tasks = lambda **kw: diagnostics.append(now[0]) or []
    def no_hint(seconds, **kwargs):
        now[0] += seconds
        return False
    with pytest.raises(module.ImagePollTimeoutError):
        backend._poll_image_results_inner(CID, 35, request_message_id="request", _completion_wait=no_hint)
    assert [at for _, at in reads] == [10, 20, 30]
    assert diagnostics == [30], "tasks is diagnostic, not a second generation poll"


def test_first_notification_wait_wakes_on_disconnect_without_full_interval(monkeypatch):
    started, disconnect = threading.Event(), threading.Event()
    def run(hub, _factory):
        started.set()
        disconnect.wait(1)
        hub.stopped.set()
        with hub.changed:
            hub.changed.notify_all()
    monkeypatch.setattr(completion._ConversationHints, "run", run)
    with completion.image_completion_hints("first-read-disconnect", CID, lambda: None) as wait:
        assert started.wait(1)
        returned = threading.Event()
        worker = threading.Thread(target=lambda: (wait(30, before_first_read=True), returned.set()))
        worker.start()
        disconnect.set()
        assert returned.wait(1), "a disconnected listener must fall back immediately"
        worker.join(1)


def test_short_active_budget_preserves_first_read_instead_of_only_waiting(monkeypatch):
    from services.config import config
    backend = object.__new__(OpenAIBackendAPI)
    reads = []
    asset = "file_000000001234567890abcdef12345678"
    backend._get_conversation = lambda cid: reads.append(cid) or {}
    backend._extract_image_tool_records = lambda *_: [{"file_ids": [asset], "sediment_ids": []}]
    monkeypatch.setitem(config.data, "image_poll_interval_secs", 30)
    monkeypatch.setitem(config.data, "image_check_before_hit_enabled", False)
    result = backend._poll_image_results_inner(CID, 5, request_message_id="request",
        _completion_wait=lambda *_args, **_kw: pytest.fail("must retain first read budget"))
    assert reads == [CID] and result == ([asset], [])


@pytest.mark.parametrize("mode,ended,notified", [
    ("same_conversation", 33, True),
    ("foreign_conversation", 30, False),
    ("continuous_progress", 40, False),
])
def test_first_read_watchdog_tracks_local_progress_but_cannot_extend_active_budget(monkeypatch, mode, ended, notified):
    # Reproduce a 30-second watchdog expiring three seconds before the final
    # hint, despite a real progress message five seconds before that expiry.
    now, waits = [0.0], []
    monkeypatch.setattr(completion.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(completion._ConversationHints, "run", lambda *_: None)
    with completion.image_completion_hints("progress-fixture", CID, lambda: None) as wait:
        hub = completion._hubs["progress-fixture"]
        hub.subscribed = True
        signal = next(iter(hub.signals[CID]))
        def tick(seconds):
            waits.append(seconds)
            if len(waits) == 1:
                now[0] = 25
                hub.activity["foreign" if mode == "foreign_conversation" else CID] = 25
            elif mode == "same_conversation":
                now[0] = 33
                signal.set()
            elif mode == "foreign_conversation":
                now[0] += seconds
            elif len(waits) == 2:
                now[0] = 37
                hub.activity[CID] = 37
            else:
                now[0] += seconds
        monkeypatch.setattr(hub.changed, "wait", tick)
        assert wait(30, before_first_read=True, max_wait_seconds=40) is notified
        assert now[0] == ended
        assert waits[0] == 30
    assert CID not in hub.activity, "released contexts must not retain conversation activity"


@pytest.mark.parametrize("role,update_type,recorded", [
    ("tool", "add-messages", True), ("assistant", "add-messages", True),
    ("user", "add-messages", False), ("tool", "title", False),
])
def test_only_active_conversation_generation_messages_extend_first_read_watchdog(monkeypatch, role, update_type, recorded):
    monkeypatch.setattr(completion.time, "monotonic", lambda: 25.0)
    reply = {"id": 2, "reply": {"type": "subscribe", "topic_id": "conversations"}}
    progress = global_image_event(status="in_progress", author={"role": role})
    progress["payload"]["update_type"] = update_type
    session = Session([[reply, global_image_event("foreign", status="in_progress"), progress]])
    hub = completion._ConversationHints()
    signal = threading.Event(); hub.signals[CID] = {signal}
    hub.run(lambda: (session, URL, {}, lambda: None, 35))
    assert hub.activity == ({CID: 25.0} if recorded else {})
    assert not signal.is_set(), "progress is not a completion hint or a result"


def test_progress_frame_cannot_restore_activity_after_concurrent_context_release():
    reply = {"id": 2, "reply": {"type": "subscribe", "topic_id": "conversations"}}
    session = Session([[reply, global_image_event(status="in_progress"),
                        global_image_event("later", status="in_progress"), global_image_event("later")]])
    hub = completion._ConversationHints()
    later = threading.Event()
    hub.signals = {CID: {threading.Event()}, "later": {later}}
    lock = hub.lock
    class ReleaseBetweenRoutingAndProgress:
        entries = 0
        def __enter__(self):
            lock.acquire()
            self.entries += 1
            if self.entries == 2:
                hub.signals.pop(CID)
                hub.activity.pop(CID, None)
                hub.nonfinal_reasons.pop(CID, None)
        def __exit__(self, *_):
            lock.release()
    hub.lock = ReleaseBetweenRoutingAndProgress()
    hub.run(lambda: (session, URL, {}, lambda: None, time.monotonic() + 10))
    assert CID not in hub.activity and CID not in hub.nonfinal_reasons
    assert "later" in hub.activity and later.is_set()
    assert hub.counts["unregistered_conversation"] == 1
    assert hub.counts["active_nonfinal"] == 1


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


def test_asset_tail_requires_a_real_stream_close(monkeypatch):
    instance, _, _ = backend(monkeypatch, "timeout")
    stream = instance._iter_image_completion_payloads(iter(["{}", "[DONE]"]), CID, time.monotonic()+60)
    assert next(stream) == "{}"
    assert not hasattr(instance, "_arm_image_asset_tail")
    assert list(stream) == ["[DONE]"]


def test_asset_tail_refresh_and_cleanup_invalidate_old_callbacks(monkeypatch):
    instance, _, _ = backend(monkeypatch, "timeout")
    timers, closed = [], []
    class Timer:
        def __init__(self, seconds, fn, args=()):
            self.fn, self.args, self.cancelled = fn, args, False
            timers.append(self)
        def start(self): pass
        def cancel(self): self.cancelled = True
        def fire(self): self.fn(*self.args)
    monkeypatch.setattr(threading, "Timer", Timer)
    stream = instance._iter_image_completion_payloads(iter(["{}", "{}", "[DONE]"]), CID,
                                                       time.monotonic()+60, lambda: closed.append(True))
    next(stream)
    arm = instance._arm_image_asset_tail
    assert arm() is True
    next(stream)  # Fresh SSE activity resets the quiet period.
    assert timers[0].cancelled and len(timers) == 2
    timers[0].fire()  # Model a callback already racing with cancel().
    assert not closed
    assert list(stream) == ["[DONE]"]
    for timer in timers: timer.fire()
    assert not closed and arm() is False
    assert not hasattr(instance, "_arm_image_asset_tail")


@pytest.mark.parametrize("error", [RuntimeError("logic failure"), ImageActiveDeadlineExceeded("deadline")])
def test_asset_tail_does_not_hide_nontransport_errors(monkeypatch, error):
    instance, _, _ = backend(monkeypatch, "timeout")
    from services.config import config
    monkeypatch.setitem(config.data, "image_poll_interval_secs", .01)
    closed = threading.Event()
    def payloads():
        yield "{}"
        assert closed.wait(1)
        raise error
    stream = instance._iter_image_completion_payloads(payloads(), CID, time.monotonic()+60, closed.set)
    next(stream)
    assert instance._arm_image_asset_tail()
    with pytest.raises(type(error), match=str(error)):
        next(stream)
    assert not hasattr(instance, "_arm_image_asset_tail")


@pytest.mark.parametrize("collection_fails", [False, True])
def test_quiet_handoff_transfers_to_collection_without_waiting_for_listener(monkeypatch, collection_fails):
    instance, gets, calls = backend(monkeypatch, "timeout")
    instance.image_request_message_id = "request"
    instance.account = {"provider_account_identity": "handoff-fixture"}
    listening, exited = threading.Event(), threading.Event()

    def receive(*args, remaining, **kwargs):
        listening.set()
        try:
            until = time.monotonic() + 3
            while remaining() > 0 and time.monotonic() < until:
                time.sleep(.002)
            return "timeout"
        finally:
            exited.set()

    monkeypatch.setattr(completion, "wait_for_turn_done", receive)
    monkeypatch.setattr(completion, "image_completion_hints",
                        lambda *a: pytest.fail("must reuse the exact topic, not open a second socket"))

    def payloads():
        yield json.dumps(HANDOFF)
        assert listening.wait(1)
        yield "[DONE]"

    before = time.monotonic()
    try:
        assert list(instance._iter_image_completion_payloads(payloads(), CID, before + 60))[-1] == "[DONE]"
        assert time.monotonic() - before < .5 and not exited.is_set()

        def collect(*args, _completion_wait, **kwargs):
            assert not _completion_wait(.02, before_first_read=True, max_wait_seconds=1)
            if collection_fails:
                raise RuntimeError("controlled original read failure")
            return "strict-original-read"

        instance._poll_image_results_inner = collect
        if collection_fails:
            with pytest.raises(RuntimeError, match="controlled original read failure"):
                instance._poll_image_results(CID, 20, request_message_id="request")
        else:
            assert instance._poll_image_results(CID, 20, request_message_id="request") == "strict-original-read"
        assert time.monotonic() - before < .5
        assert exited.is_set() and instance._image_completion_listener is None
        assert len(gets) == 1
    finally:
        instance.close()


@pytest.mark.parametrize("notification", ["done", "disconnected"])
def test_handoff_done_or_disconnect_wakes_strict_collection_without_another_socket(monkeypatch, notification):
    from services.config import config
    instance, gets, calls = backend(monkeypatch, notification)
    instance.image_request_message_id = "request"
    instance.account = {"provider_account_identity": "handoff-fixture"}
    monkeypatch.setattr(completion, "image_completion_hints",
                        lambda *a: pytest.fail("must reuse exact-topic listener"))
    monkeypatch.setitem(config.data, "image_check_before_hit_enabled", False)
    reads = []
    asset = "file_000000001234567890abcdef12345678"
    document = {"conversation_id": CID, "current_node": "image", "mapping": {
        "request": {"parent": None, "message": {"id": "request", "author": {"role": "user"}}},
        "image": {"parent": "request", "message": {"id": "image", "author": {"role": "tool"},
            "content": {"parts": [{"asset_pointer": "sediment://" + asset}]}}}}}
    instance._get_conversation = lambda cid: reads.append(cid) or document
    instance._query_backend_tasks = lambda **kw: pytest.fail("saved result needs no task diagnosis")
    try:
        list(instance._iter_image_completion_payloads(iter([json.dumps(HANDOFF), "[DONE]"]), CID, time.monotonic()+60))
        assert not reads
        assert instance._poll_image_results(CID, 20, request_message_id="request") == ([asset], [asset])
        assert reads == [CID] and len(gets) == 1 and calls == [(TOPIC, CID)]
        assert instance._image_completion_listener is None
    finally:
        instance.close()


@pytest.mark.parametrize("changed", ["conversation", "request", "snapshot", "close"])
def test_handoff_cannot_escape_original_attempt_and_unused_listener_is_closed(monkeypatch, changed):
    instance, gets, calls = backend(monkeypatch, "done")
    instance.image_request_message_id = "request"
    list(instance._iter_image_completion_payloads(iter([json.dumps(HANDOFF), "[DONE]"]), CID, time.monotonic()+60))
    listener = instance._image_completion_listener
    assert listener is not None
    instance._poll_image_results_inner = lambda *a, **kw: (
        pytest.fail("mismatched hint must not wake another original") if "_completion_wait" in kw else "original-only")
    try:
        if changed != "close":
            assert instance._poll_image_results(
                "other-conversation" if changed == "conversation" else CID, 20,
                request_message_id="other-request" if changed == "request" else "request",
                **({"initial_document": {}} if changed == "snapshot" else {})) == "original-only"
        else:
            instance.close()
        assert listener["stopped"].is_set() and not listener["thread"].is_alive()
        assert instance._image_completion_listener is None
    finally:
        instance.close()


def test_malformed_item_cannot_keep_connection_alive():
    item = event("stream-item")
    del item["payload"]["payload"]["encoded_item"]
    assert wait(Session([[ack(), item]])) == "invalid_event"


@pytest.mark.parametrize("prefix", ["conversation-", "conv-turn-low-ttl-"])
def test_handoff_requires_nonempty_topic(prefix):
    assert completion.handoff_topic({**HANDOFF, "options": [{"type": "subscribe_ws_topic", "topic_id": prefix}]}) is None
