"""Consume a ChatGPT turn notification, never an image result or send permission.

Only subscribe to the topic supplied by the current POST's SSE handoff. A done
event wakes the existing original-message readback; it cannot complete a task.
"""
import json
import select
import time
from urllib.parse import urlparse

from curl_cffi import CurlECode, CurlError, CurlInfo, CurlWsFlag


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
