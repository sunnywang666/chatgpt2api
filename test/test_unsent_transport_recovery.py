"""Local socket failures and durable recovery; never contact a real account."""
import socket
import threading
import json
from datetime import datetime, timezone
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from curl_cffi import CurlInfo
from curl_cffi.requests import Session
from curl_cffi.requests.exceptions import RequestException

import services.account_request_pacing as pacing
from services.image_task_service import _failure_details, _public_failure_details
from services.openai_backend_api import OpenAIBackendAPI, ChatRequirements
from services.protocol.conversation import ConversationRequest, ImageGenerationError, _generate_bound_single_image
from test.test_generation_completion import setup, row, patch_row, IDENTITY


def connection_error(**changes):
    infos = {key: 0 for key in (CurlInfo.REQUEST_SIZE, CurlInfo.SIZE_UPLOAD_T, CurlInfo.EARLYDATA_SENT_T,
        CurlInfo.REDIRECT_COUNT, CurlInfo.RESPONSE_CODE, CurlInfo.HTTP_VERSION, CurlInfo.SIZE_DOWNLOAD_T,
        CurlInfo.APPCONNECT_TIME)}
    infos[CurlInfo.NUM_CONNECTS] = 1
    infos.update(changes.pop("infos", {}))
    return RequestException("private diagnostic text", code=changes.pop("code", 35),
        response=SimpleNamespace(status_code=changes.pop("status", 0), infos=infos))


def test_broken_exception_evidence_cannot_mask_original_failure():
    class Broken(Exception):
        @property
        def _account_request_not_submitted(self):
            raise RuntimeError("private accessor failure")
    original = Broken("private original failure")
    assert pacing.unsent_transport_failure(original) == {}
    details = _failure_details(original, "start_image_generation")
    assert details["type"] == "Broken" and "private" not in repr(details)
    for code in (True, "35", 28, None):
        original = RuntimeError("private")
        original._account_request_not_submitted = True
        original.code = code
        assert pacing.unsent_transport_failure(original) == {}


@pytest.mark.parametrize("change", ["none", "timeout", "send", "receive", "missing", "written",
    "uploaded", "early", "early_rejected", "redirect", "response", "version", "tls", "reconnect", "retry", "status"])
def test_only_complete_connect_stage_evidence_is_retryable(change):
    error = connection_error()
    fields = {"written": CurlInfo.REQUEST_SIZE, "uploaded": CurlInfo.SIZE_UPLOAD_T,
        "early": CurlInfo.EARLYDATA_SENT_T, "redirect": CurlInfo.REDIRECT_COUNT,
        "response": CurlInfo.RESPONSE_CODE, "version": CurlInfo.HTTP_VERSION, "tls": CurlInfo.APPCONNECT_TIME}
    if change in fields: error.response.infos[fields[change]] = 1
    if change == "early_rejected": error.response.infos[CurlInfo.EARLYDATA_SENT_T] = -1
    if change == "missing": del error.response.infos[CurlInfo.REQUEST_SIZE]
    if change == "reconnect": error.response.infos[CurlInfo.NUM_CONNECTS] = 2
    if change == "status": error.response.status_code = 200
    if change in {"timeout", "send", "receive"}: error.code = {"timeout": 28, "send": 55, "receive": 56}[change]
    with Session(retry=1 if change == "retry" else 0) as session:
        assert pacing._connect_failed_before_request(session, error) is (change == "none")


@pytest.mark.parametrize("mode", ["tls", "proxy_tls", "post_received"])
def test_native_stream_failure_distinguishes_tls_from_lost_response(tmp_path, monkeypatch, mode):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(3)
    port = listener.getsockname()[1]
    received = []
    failures = []

    def serve():
        try:
            with listener.accept()[0] as conn:
                conn.settimeout(3)
                data = conn.recv(8192)
                if mode == "proxy_tls":
                    assert data.startswith(b"CONNECT ")
                    conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                    data = conn.recv(8192)
                if mode == "post_received":
                    while b"\r\n\r\n" not in data:
                        data += conn.recv(8192)
                    head, body = data.split(b"\r\n\r\n", 1)
                    length = int(next(line.split(b":", 1)[1] for line in head.split(b"\r\n")
                                      if line.lower().startswith(b"content-length:")))
                    while len(body) < length:
                        body += conn.recv(8192)
                    received.append(body)
                else:
                    assert data.startswith(b"\x16\x03")  # TLS ClientHello, no HTTP POST.
        except Exception as exc:
            failures.append(exc)
        finally:
            listener.close()

    worker = threading.Thread(target=serve)
    worker.start()
    monkeypatch.setattr(pacing, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pacing, "_clocks", {})
    with Session(trust_env=False, proxy=f"http://127.0.0.1:{port}" if mode == "proxy_tls" else None) as session:
        # Keep the real native Session while directing every byte to loopback.
        raw = session.request
        local_url = f"{'http' if mode == 'post_received' else 'https'}://127.0.0.1:{port}/backend-api/f/conversation"
        session.request = lambda method, url, **kw: raw(method, url if mode == "proxy_tls" else local_url, **kw)
        pacing.pace_account_session(session, {"account_id": "local-fixture"}, "local-fixture")
        with pytest.raises(RequestException) as raised:
            session.post("https://chatgpt.com/backend-api/f/conversation", json={"prompt": "fixture"}, timeout=2, stream=True)
        error = raised.value
        if mode == "post_received":
            assert received and error.response.infos[CurlInfo.REQUEST_SIZE] > 0
            assert not pacing.unsent_transport_failure(error)
        else:
            assert not received and error.code == 35
            assert pacing.unsent_transport_failure(error)["transport_error_code"] == 35
            assert error.response.infos[CurlInfo.HTTP_CONNECTCODE] == (200 if mode == "proxy_tls" else 0)
    worker.join(4)
    assert not worker.is_alive() and not failures


@pytest.mark.parametrize("total", [1, 2])
@pytest.mark.parametrize("fallback", [False, True])
def test_bound_transport_failure_reaches_receipt_without_charge_or_partial_set_replay(total, fallback):
    backend = object.__new__(OpenAIBackendAPI)
    backend.access_token = "fixture"
    backend.base_url = "https://fixture.invalid"
    backend.image_request_message_id = "original-message"
    backend.image_submission_started = False
    backend._image_model_settings = lambda _: ("gpt-image", "")
    backend._image_headers = lambda *_: {}
    backend.close = Mock()
    error = connection_error()
    error._account_request_not_submitted = True  # Classification independently exercised above.
    sends = []
    def send(url, **kw):
        kw["_account_request_before_send"]()
        sends.append(url)
        if fallback and len(sends) == 1:
            return Mock(status_code=404)
        raise error
    backend.session = SimpleNamespace(post=send)
    request = ConversationRequest(prompt="retained input", model="gpt-image-2", n=1,
        response_format="url", provider_binding_id="original-binding", provider_account_identity="original-account",
        client_conversation_id="original-client", retain_conversation=True)
    with patch("services.protocol.conversation.account_service.get_bound_account_identity", return_value="original-account"), \
         patch("services.protocol.conversation.account_service.acquire_bound_image_access_token", return_value="fixture"), \
         patch("services.protocol.conversation.account_service.get_account", return_value={}), \
         patch("services.protocol.conversation.account_service.conversation_binding_lock", return_value=nullcontext()), \
         patch("services.protocol.conversation.account_service.mark_image_result") as mark, \
         patch("services.protocol.conversation.account_service.release_image_slot") as release, \
         patch("services.protocol.conversation.OpenAIBackendAPI", return_value=backend), \
         patch("services.openai_backend_api.account_service.require_image_account"), \
         patch("services.protocol.conversation.stream_image_outputs", side_effect=lambda *_:
               backend._start_image_generation("retained input", ChatRequirements(token="fixture"), "fixture", "gpt-image-2")):
        with pytest.raises(ImageGenerationError) as raised:
            _generate_bound_single_image(request, 1, total)
        assert raised.value.upstream_submitted is (total != 1)
        assert raised.value.request_message_id == "original-message"
        assert raised.value.provider_account_identity == "original-account"
        if total == 1:
            assert raised.value.code == "IMAGE_GENERATION_NOT_SUBMITTED"
            mark.assert_not_called()
            release.assert_called_once_with("fixture")
        else:
            assert raised.value.code == "CONVERSATION_BINDING_UNAVAILABLE"
            mark.assert_called_once()
        details = _public_failure_details(_failure_details(raised.value, "start_image_generation"))
        assert details["submission_evidence"] == "connect_failed_before_request"
        assert details["transport_error_code"] == 35
        assert "private" not in repr(details)
        assert len(sends) == (2 if fallback else 1)


def test_connect_failure_retries_original_once_across_restart(setup):
    service, admission, _ = setup
    rid = "connect-failed-original"
    service.images.submit_generation(IDENTITY, client_task_id=rid, prompt="original input", model="gpt-image-2", size=None)
    failure = {"at": 2900.0, "submission_evidence": "connect_failed_before_request", "transport_error_code": 35}
    failed = dict(status="error", upstream_outcome="not_submitted", error_code="RESULT_UNRECOVERABLE",
        _submission_started=False, upstream_submission_started=False, upstream_unfinished=False,
        recovery_retryable=True, last_recovery_failure=failure,
        provider_binding_id="original-binding", provider_account_identity="original-account",
        conversation_id="original-conversation", parent_message_id="original-parent", request_message_id="original-message",
        _execution_timeline=[{"stage": "send_call_started", "at": 2890}])
    patch_row(service, "image", rid, **failed)
    original = row(service, "image", rid)
    service.start("image", IDENTITY, rid)
    retried = row(service, "image", rid)
    assert retried["status"] == "queued" and retried["_completion"]["same_request_retry"]
    for key in ("_input_ref", "request_hash", "provider_binding_id", "provider_account_identity", "conversation_id",
                "parent_message_id", "request_message_id"):
        assert retried[key] == original[key]
    assert retried["_execution_timeline"][-1]["failure"] == failure
    assert not retried.get("_executing") and not retried.get("_turn_reserved")
    patch_row(service, "image", rid, **failed)
    restarted = type(service)(service.text, service.images, service.lifecycle, clock=admission.clock)
    result = restarted.start("image", IDENTITY, rid)
    assert result["state"] == "needs_attention" and "replacement_id" not in result
    assert row(service, "image", rid)["status"] == "error"


@pytest.mark.parametrize("mode", ["unsent", "unmarked", "timeout", "stream_event", "cursor_read"])
def test_text_transport_proof_survives_wrapper_only_before_response(tmp_path, mode):
    from services.conversation_binding_service import ConversationBindingService
    from services.text_task_service import TextTaskService
    from test.test_text_task_service import QueuedExecutor
    queue = QueuedExecutor()
    service = TextTaskService(tmp_path / "text.sqlite3", ConversationBindingService().complete_text,
                              queue, recovery_reader=lambda _: {"status": "running"})
    body = {"client_request_id": "original", "client_conversation_id": "original-client",
            "provider_binding_id": "binding", "provider_account_identity": "account",
            "messages": [{"role": "user", "content": "retained original input"}]}
    error = connection_error(code=28 if mode == "timeout" else 35)
    error._account_request_not_submitted = mode != "unmarked"
    backend = Mock()
    backend.get_conversation_parent_message_id.side_effect = error

    def events(*args, **kwargs):
        # The send guard ran; only native no-request evidence can undo this flag.
        service._update("owner", "original", _submission_started=True, _turn_reserved=True)
        if mode in {"stream_event", "cursor_read"}:
            yield {"type": "conversation.delta", "conversation_id": "original-conversation", "delta": "answer"}
        if mode != "cursor_read":
            raise error

    with patch("services.conversation_binding_service.account_service.get_bound_account_identity", return_value="account"), \
         patch("services.conversation_binding_service.account_service.get_bound_text_access_token", return_value="fixture"), \
         patch("services.conversation_binding_service.account_service.conversation_binding_lock", return_value=nullcontext()), \
         patch("services.conversation_binding_service.account_service.mark_text_used"), \
         patch("services.conversation_binding_service.OpenAIBackendAPI", return_value=backend), \
         patch("services.conversation_binding_service.conversation_events", side_effect=events):
        service.submit("owner", body)
        queue.run()
    with service.store.connect() as db:
        receipt = service.store.read_receipt(db, "text", "owner", "original")
    assert receipt["provider_binding_id"] == "binding" and receipt["provider_account_identity"] == "account"
    assert receipt["_input_ref"]
    if mode == "unsent":
        assert receipt["status"] == "failed" and receipt["upstream_outcome"] == "not_sent"
        assert receipt["error_code"] == "TEXT_GENERATION_NOT_SUBMITTED"
        assert receipt["_submission_started"] is False and receipt["_turn_reserved"] is False
        assert receipt.get("_claim_id") is None and not receipt.get("_claim_until")
        assert receipt["last_recovery_failure"]["submission_evidence"] == "connect_failed_before_request"
        assert "private" not in json.dumps(receipt)
        assert service._public(receipt)["upstream_submission_started"] is False
        for change in ({"_submission_started": True}, {"_executing": True}, {"status": "running"},
                       {"upstream_outcome": "unknown"}, {"last_recovery_failure": {}},
                       {"last_recovery_failure": {"submission_evidence": "connect_failed_before_request",
                                                  "transport_error_code": 28}}):
            assert "upstream_submission_started" not in service._public({**receipt, **change})
    else:
        assert receipt["status"] == "unknown" and receipt["_submission_started"] is True
        assert not receipt.get("last_recovery_failure")


def test_text_confirmed_unsent_retries_original_once_across_restart(setup):
    service, admission, _ = setup
    failure = {"at": 2900.0, "submission_evidence": "connect_failed_before_request", "transport_error_code": 35}
    failed = dict(status="failed", upstream_outcome="not_sent", error_code="TEXT_GENERATION_NOT_SUBMITTED",
                  _submission_started=False, upstream_submission_started=False, _turn_reserved=False,
                  _executing=False, last_recovery_failure=failure)
    patch_row(service, **failed)
    original = row(service)
    service.start("text", IDENTITY, "old-0")
    retried = row(service)
    assert retried["status"] == "queued" and retried["_completion"]["same_request_retry"]
    for key in ("_input_ref", "provider_binding_id", "provider_account_identity", "conversation_id",
                "parent_message_id", "request_message_id"):
        assert retried.get(key) == original.get(key)
    assert retried["_execution_timeline"][-1]["failure"] == failure
    patch_row(service, **failed)
    restarted = type(service)(service.text, service.images, service.lifecycle, clock=admission.clock)
    result = restarted.start("text", IDENTITY, "old-0")
    assert result["state"] == "needs_attention" and "replacement_id" not in result
    assert row(service)["status"] == "failed"


def test_text_unsent_error_code_without_native_cause_cannot_clear_send(tmp_path):
    from services.conversation_binding_service import ConversationBindingError
    from services.text_task_service import TextTaskService
    from test.test_text_task_service import QueuedExecutor
    queue = QueuedExecutor()
    def runner(body, on_cursor):
        service._update("owner", "original", _submission_started=True, _turn_reserved=True)
        raise ConversationBindingError("unproven", code="TEXT_GENERATION_NOT_SUBMITTED",
            original_failure_phase="stream_open", original_exception_category="transport")
    service = TextTaskService(tmp_path / "text.sqlite3", runner, queue)
    service.submit("owner", {"client_request_id": "original", "client_conversation_id": "client",
                             "messages": [{"role": "user", "content": "retained input"}]})
    queue.run()
    with service.store.connect() as db:
        receipt = service.store.read_receipt(db, "text", "owner", "original")
    assert receipt["status"] == "unknown" and receipt["_submission_started"] is True
    assert receipt["error_code"] == "CONVERSATION_OUTCOME_UNKNOWN"


@pytest.mark.parametrize("second_fails", [False, True])
def test_text_unsent_handler_uses_existing_completion_and_admission(setup, second_fails):
    from services.conversation_binding_service import ConversationBindingError
    from services.request_context import current_request
    service, admission, _ = setup
    patch_row(service, status="succeeded", _turn_reserved=False)
    rid = "fresh-text"
    service.text.submit("owner", {"client_request_id": rid, "client_conversation_id": "fresh-client",
        "_public_session_ref": "fresh-session", "_public_route": "chat", "_route": "chat", "_operation": "text", "model": "fixture-text",
        "provider_binding_id": "binding-0", "provider_account_identity": "account-0",
        "messages": [{"role": "user", "content": "retained original input"}]})
    original = row(service, request_id=rid)
    calls = []
    def runner(body, on_cursor):
        ctx = current_request.get()
        ctx.before_send()
        ctx.release_turn()
        calls.append((ctx.request_id, body["messages"], body["provider_binding_id"],
                      body["provider_account_identity"], body.get("_request_message_id"), body.get("conversation_id")))
        if len(calls) == 1 or second_fails:
            error = connection_error()
            error._account_request_not_submitted = True
            raise ConversationBindingError("connection setup failed", code="TEXT_GENERATION_NOT_SUBMITTED",
                original_failure_phase="stream_open", original_exception_category="transport") from error
        return {"content": "saved original answer", "conversation_id": "conv-0", "parent_message_id": "final"}
    service.text.runner = runner
    for attempt in range(2):
        ctx = admission.claim_next()
        assert ctx is not None and ctx.request_id == rid
        admission.execute(ctx)
        receipt = row(service, request_id=rid)
        assert not receipt.get("_executing") and not receipt.get("_turn_reserved")
        if attempt == 0:
            assert receipt["status"] == "failed" and receipt["upstream_outcome"] == "not_sent"
            assert receipt.get("_claim_id") is None and not receipt.get("_claim_until")
        service.process_one()
        admission.clock.now += 2
    final = row(service, request_id=rid)
    assert calls[0] == calls[1]
    assert final["_input_ref"] == original["_input_ref"]
    assert "replacement_id" not in final["_completion"]
    if second_fails:
        assert final["status"] == "failed" and final["_completion"]["state"] == "needs_attention"
        assert admission.claim_next() is None
    else:
        assert final["status"] == "succeeded" and final["content"] == "saved original answer"
        assert final["_completion"]["state"] == "result_ready"


@pytest.mark.parametrize("second_fails", [False, True])
def test_handler_to_scheduler_retries_once_and_releases_execution(setup, tmp_path, second_fails):
    from services.request_context import current_request
    service, admission, _ = setup
    account_path = tmp_path / "accounts.json"
    accounts = json.loads(account_path.read_text())
    for account in accounts:
        account.update(limits_progress=[{"feature_name": "image_gen", "remaining": 99}],
                       capacity_observed_at=datetime.now(timezone.utc).isoformat())
    account_path.write_text(json.dumps(accounts))
    calls = []
    def generate(payload):
        ctx = current_request.get()
        ctx.before_send()
        callback = payload["progress_callback"]
        callback.record_submission_started()
        ctx.record_stage("send_call_started")
        # The production pacing wrapper releases the HTTP turn on both a
        # completed response and a transport exception (tested independently).
        ctx.release_turn()
        calls.append((ctx.request_id, payload["prompt"], payload["provider_binding_id"],
                      payload["provider_account_identity"], payload.get("_request_message_id")))
        if len(calls) == 1 or second_fails:
            error = connection_error()
            error._account_request_not_submitted = True
            try:
                raise error
            except RequestException as exc:
                raise ImageGenerationError("connection setup failed", code="IMAGE_GENERATION_NOT_SUBMITTED",
                    provider_binding_id=payload["provider_binding_id"],
                    provider_account_identity=payload["provider_account_identity"],
                    request_message_id="original-message", upstream_submitted=False) from exc
        return {"data": [{"b64_json": "cG5n"}], "_provider_binding_id": payload["provider_binding_id"],
                "_provider_account_identity": payload["provider_account_identity"],
                "_conversation_id": "original-conversation", "_parent_message_id": "original-result",
                "_image_thread_terminal": True}
    service.images.generation_handler = generate
    admission.register("image", lambda ctx, body: service.images._run_task(
        ctx.owner+":"+ctx.request_id, body["mode"], body["payload"], body["identity"], "gpt-image-2"))
    rid = "original-send-failure"
    service.images.submit_generation(IDENTITY, client_task_id=rid, prompt="original input", model="gpt-image-2",
                                     size=None, image_thread_id="original-thread")
    patch_row(service, "image", rid, request_message_id="original-message")
    original = row(service, "image", rid)
    for attempt in range(2):
        ctx = admission.claim_next()
        assert ctx is not None and ctx.request_id == rid
        admission.execute(ctx)
        receipt = row(service, "image", rid)
        assert not receipt.get("_executing") and not receipt.get("_turn_reserved")
        if attempt == 0:
            assert receipt["status"] == "error" and receipt["upstream_outcome"] == "not_submitted"
            assert receipt["last_recovery_failure"]["submission_evidence"] == "connect_failed_before_request"
        # Use the ordinary automatic recovery worker, not a client re-POST.
        service.process_one()
        admission.clock.now += 2
    final = row(service, "image", rid)
    assert calls[0] == calls[1]
    assert final["_input_ref"] == original["_input_ref"] and final["request_hash"] == original["request_hash"]
    assert "replacement_id" not in final["_completion"]
    if second_fails:
        assert final["status"] == "error" and final["_completion"]["state"] == "needs_attention"
        assert admission.claim_next() is None
        with service.store.connect() as db:
            assert not service.store.runtime(db, final["_work_key"])["slot_held"]
    else:
        assert final["status"] == "success" and final["data"] == [{"b64_json": "cG5n"}]
        assert final["_completion"]["state"] == "result_ready"
