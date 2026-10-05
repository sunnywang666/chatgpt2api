"""Consume a ChatGPT turn notification, never an image result or send permission.

Consume a supplied turn topic or account conversation updates. A matching
hint wakes original-message readback; it cannot complete a task.
"""
import json
import select
import threading
import time
from contextlib import contextmanager
from contextvars import copy_context
from urllib.parse import urlparse

from curl_cffi import CurlECode, CurlError, CurlInfo, CurlWsFlag


def completed_conversation_hint(envelope):
    """Return only the conversation to read, never a result or send authority."""
    if not isinstance(envelope, dict) or envelope.get("type") != "conversation-update":
        return None
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or payload.get("update_type") != "add-messages":
        return None
    cid = payload.get("conversation_id")
    update = payload.get("update_content")
    messages = update.get("messages") if isinstance(update, dict) else None
    if not isinstance(cid, str) or not cid or not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict) or message.get("status") != "finished_successfully":
            continue
        author, metadata = message.get("author"), message.get("metadata")
        if not isinstance(author, dict):
            continue
        ghost = metadata.get("ghostrider") if isinstance(metadata, dict) else None
        content = message.get("content")
        if (author.get("role") == "tool" and message.get("channel") == "final"
                and isinstance(ghost, dict) and ghost.get("status") == "final"
                and isinstance(content, dict) and content.get("content_type") == "multimodal_text"):
            return cid
    return None


_hubs = {}
_hubs_lock = threading.RLock()


class _ConversationHints:
    """One short-lived socket per active account, with exact local CID routing."""
    def __init__(self):
        self.stopped = threading.Event()
        self.signals = {}
        self.lock = threading.Lock()
        self.thread = None

    def run(self, open_transport):
        ws = cleanup = None
        try:
            if self.stopped.is_set():
                return
            session, url, options, cleanup, transport_deadline = open_transport()
            parsed = urlparse(url)
            if (self.stopped.is_set() or transport_deadline <= time.monotonic()
                    or parsed.scheme != "wss" or parsed.hostname != "ws.chatgpt.com"
                    or parsed.username or parsed.password or parsed.port not in (None, 443) or parsed.fragment):
                return
            ws = session.ws_connect(url, timeout=min(10, transport_deadline - time.monotonic()), headers={"Origin": "https://chatgpt.com"},
                                    allow_redirects=False, **options)
            ws.send(json.dumps([
                {"id": 1, "command": {"type": "connect", "presence": {"type": "presence", "state": "background"}}},
                # No history replay: these are wakeups for currently running
                # originals, not a feed of past account activity.
                {"id": 2, "command": {"type": "subscribe", "topic_id": "conversations"}},
            ]), flags=CurlWsFlag.TEXT)
            fragments, subscribed = bytearray(), False
            deadline = time.monotonic() + 300
            while not self.stopped.is_set() and time.monotonic() < deadline:
                try:
                    chunk, frame = ws.recv_fragment()
                except CurlError as exc:
                    if exc.code != CurlECode.AGAIN:
                        return
                    select.select([ws.curl.getinfo(CurlInfo.ACTIVESOCKET)], [], [], .25)
                    continue
                if frame.flags & CurlWsFlag.CLOSE:
                    return
                if frame.flags & (CurlWsFlag.PING | CurlWsFlag.PONG):
                    continue
                fragments.extend(chunk)
                if len(fragments) > 2 * 1024 * 1024:
                    return
                if frame.bytesleft or frame.flags & CurlWsFlag.CONT:
                    continue
                frame_data = json.loads(fragments)
                fragments.clear()
                for message in frame_data if isinstance(frame_data, list) else [frame_data]:
                    if not isinstance(message, dict):
                        return
                    if "reply" in message:
                        if message.get("id") != 2:
                            continue
                        reply = message["reply"]
                        if (not isinstance(reply, dict) or reply.get("type") != "subscribe"
                                or reply.get("topic_id") != "conversations"):
                            return
                        subscribed = True  # recovered=false is normal without an offset.
                        continue
                    if not subscribed:
                        continue
                    if message.get("type") == "message":
                        if message.get("topic_id") != "conversations":
                            continue
                        message = message.get("payload")
                    payload = message.get("payload") if isinstance(message, dict) else None
                    cid = payload.get("conversation_id") if isinstance(payload, dict) else None
                    with self.lock:
                        if not isinstance(cid, str) or cid not in self.signals:
                            continue
                    cid = completed_conversation_hint(message)
                    if cid:
                        with self.lock:
                            for signal in self.signals.get(cid, ()):
                                signal.set()
        except Exception:
            # Signed URLs and unrelated messages must never enter logs.
            pass
        finally:
            self.stopped.set()
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    try:
                        ws.terminate()
                    except Exception:
                        pass
            if cleanup is not None:
                cleanup()


@contextmanager
def image_completion_hints(account_identity, conversation_id, open_transport):
    """Deduplicate sockets; closure/disconnect preserves ordinary polling."""
    signal = threading.Event()
    with _hubs_lock:
        hub = _hubs.get(account_identity)
        if hub is None:
            hub = _hubs[account_identity] = _ConversationHints()
        with hub.lock:
            hub.signals.setdefault(conversation_id, set()).add(signal)
        if hub.thread is None:
            hub.thread = threading.Thread(target=copy_context().run, args=(hub.run, open_transport), daemon=True)
            hub.thread.start()
    try:
        def wait(seconds):
            if signal.is_set():
                signal.clear()
                return True
            if hub.stopped.is_set():
                time.sleep(seconds)
                return False
            notified = signal.wait(seconds)
            signal.clear()  # Coalesce duplicate updates into one original GET.
            return notified
        yield wait
    finally:
        with _hubs_lock:
            with hub.lock:
                hub.signals[conversation_id].discard(signal)
                if not hub.signals[conversation_id]:
                    del hub.signals[conversation_id]
                empty = not hub.signals
            if empty:
                hub.stopped.set()
                if _hubs.get(account_identity) is hub:
                    del _hubs[account_identity]
        if empty:
            hub.thread.join(timeout=.5)


def handoff_topic(event):
    if not isinstance(event, dict) or event.get("type") != "stream_handoff":
        return None
    if not isinstance(event.get("turn_exchange_id"), str):
        return None
    options = event.get("options")
    if not isinstance(options, list):
        return None
    for option in options:
        if not isinstance(option, dict) or option.get("type") != "subscribe_ws_topic":
            continue
        topic = option.get("topic_id")
        if (isinstance(topic, str) and len(topic) <= 1024
                and any(topic.startswith(prefix) and len(topic) > len(prefix)
                        for prefix in ("conversation-", "conv-turn-low-ttl-"))):
            return topic
    return None


def wait_for_turn_done(session, url, topic, conversation_id, deadline, *, remaining, connect_options=None):
    """Bounded synchronous wait. Return a fixed reason; never retain URL/body.

    The supplied topic is already tied to one original generation POST. Its
    history can be replayed, so even a matching done is only a readback hint.
    No stream content is promoted to a result here; the existing image branch
    validator still checks the original request message and actual image files.
    """
    parsed = urlparse(url)
    if (parsed.scheme != "wss" or parsed.hostname != "ws.chatgpt.com"
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.fragment):
        return "invalid_url"
    if not conversation_id:
        return "missing_conversation"
    ws = None
    try:
        timeout = min(10.0, deadline - time.monotonic(), remaining())
        if timeout <= 0:
            return "deadline"
        ws = session.ws_connect(url, timeout=timeout, headers={"Origin": "https://chatgpt.com"},
                                allow_redirects=False, **(connect_options or {}))
        ws.send(json.dumps([
            {"id": 1, "command": {"type": "connect", "presence": {"type": "presence", "state": "foreground"}}},
            {"id": 2, "command": {"type": "subscribe", "topic_id": topic, "offset": "0"}},
        ]), flags=CurlWsFlag.TEXT)
        idle_deadline = time.monotonic() + 5.0
        subscribed = False
        turn_id = None
        fragments = bytearray()
        while True:
            budget = min(deadline - time.monotonic(), idle_deadline - time.monotonic(), remaining())
            if budget <= 0:
                return "timeout"
            try:
                chunk, frame = ws.recv_fragment()
            except CurlError as exc:
                if exc.code != CurlECode.AGAIN:
                    return "disconnected"
                select.select([ws.curl.getinfo(CurlInfo.ACTIVESOCKET)], [], [], min(0.5, budget))
                continue
            if frame.flags & CurlWsFlag.CLOSE:
                return "disconnected"
            if frame.flags & (CurlWsFlag.PING | CurlWsFlag.PONG):
                continue
            fragments.extend(chunk)
            if len(fragments) > 2 * 1024 * 1024:
                return "invalid_frame"
            if frame.bytesleft or frame.flags & CurlWsFlag.CONT:
                continue
            messages = json.loads(fragments)
            fragments.clear()
            if not isinstance(messages, list):
                return "invalid_frame"
            pending = list(messages)
            while pending:
                message = pending.pop(0)
                if not isinstance(message, dict):
                    return "invalid_frame"
                # Some servers include type=reply; the browser schema strips
                # that extra key. Accept both wire shapes explicitly.
                if "reply" in message:
                    if message.get("id") != 2:
                        continue
                    reply = message["reply"]
                    if (not isinstance(reply, dict) or reply.get("type") != "subscribe"
                            or reply.get("topic_id") != topic or reply.get("recovered") is not True):
                        return "history_unavailable"
                    catchups = reply.get("catchups", [])
                    if not isinstance(catchups, list):
                        return "invalid_frame"
                    subscribed = True
                    idle_deadline = time.monotonic() + 5.0
                    pending[0:0] = catchups
                    continue
                if message.get("type") != "message" or message.get("topic_id") != topic:
                    return "wrong_topic"
                if not subscribed:
                    return "unconfirmed_subscription"
                envelope = message.get("payload")
                if not isinstance(envelope, dict) or envelope.get("type") != "conversation-turn-stream":
                    return "invalid_event"
                item = envelope.get("payload")
                if not isinstance(item, dict) or item.get("type") not in {"done", "heartbeat", "stream-item"}:
                    return "invalid_event"
                kind = item["type"]
                if kind == "stream-item" and (
                        not isinstance(item.get("encoded_item"), str)
                        or any(key not in item or (item[key] is not None and not isinstance(item[key], str))
                               for key in ("stream_item_id", "parent_stream_item_id"))):
                    return "invalid_event"
                cid = item.get("conversation_id")
                if cid != conversation_id and not (kind == "heartbeat" and cid is None):
                    return "wrong_conversation"
                current_turn = item.get("turn_id")
                if not isinstance(current_turn, str) or not current_turn:
                    return "invalid_event"
                if turn_id is not None and current_turn != turn_id:
                    return "wrong_turn"
                turn_id = current_turn
                idle_deadline = time.monotonic() + 30.0
                if kind == "done":
                    return "done"
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                # close() can fail while sending its closing frame. Cleanup
                # must still release the curl handle in that disconnected case.
                try:
                    ws.terminate()
                except Exception:
                    pass
