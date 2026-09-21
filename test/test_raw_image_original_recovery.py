"""Legacy image formatters on the original receipt, fake sends/GETs/downloads."""
import asyncio
import base64
import hashlib
from io import BytesIO
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from PIL import Image
import pytest

from services import durable_forward
from services.openai_backend_api import OpenAIBackendAPI
from services.protocol.conversation import ImageOutput
from services.request_context import current_request
from services.text_task_service import TextTaskService
from test.test_reliable_pool_routes import runtime, request


WHO = {"id": "image-user", "role": "user"}
PROTOCOLS = ["openai_v1_image_generations", "openai_v1_image_edit", "openai_v1_chat_complete", "openai_v1_response"]


def image_bytes(index):
    output = BytesIO()
    Image.new("RGB", (2, 2), (index * 30, 20, 10)).save(output, format="PNG")
    return output.getvalue()


def image_url(data, base_url=None):
    return "https://image.example/" + hashlib.sha256(data).hexdigest()


def original(runtime):
    return durable_forward.raw_receipt(runtime.service, WHO["id"], "original-wire")


def fixture_upstream(runtime, monkeypatch, *, fail_slot=1, saved_ids=False, cursor=True):
    from services import openai_backend_api
    from services.protocol import conversation
    runtime.admission.recoveries["text"] = lambda owner, rid: runtime.service.read(owner, rid)
    observed = SimpleNamespace(sends=[], reads=[], downloads=[], resolves=[], lists=[], active=False,
                               download_failure=False, before_download=None, before_read=None, before_resolve=None)
    def generate(req, index, total):
        ctx = current_request.get()
        assert ctx.selected_account()["provider_account_identity"] == "account-0"
        callback = req.progress_callback
        message_id = callback.request_message_id
        assert original(runtime)["_image_slots"][str(index - 1)]["request_message_id"] == message_id
        ctx.before_send()
        observed.sends.append((index, message_id))
        cid = "image-conversation-" + str(index)
        if cursor:
            callback.record_conversation_id(cid)
        if saved_ids:
            callback.record_result_ids(["original-file-" + str(index)], [])
        if index == fail_slot:
            raise ConnectionError("synthetic interrupted original image")
        data = image_bytes(index)
        return [ImageOutput(kind="result", model=req.model, index=index, total=total, conversation_id=cid,
                            account_email="PRIVATE_ACCOUNT_EMAIL", data=[{"b64_json": base64.b64encode(data).decode(),
                            "url": image_url(data), "revised_prompt": req.prompt}])]
    class Backend(OpenAIBackendAPI):
        def __init__(self, access_token):
            assert access_token == runtime.account["access_token"]
        def _get_conversation(self, cid, **kwargs):
            observed.reads.append(cid)
            if observed.before_read:
                observed.before_read()
            index = int(cid.rsplit("-", 1)[1])
            user = dict(observed.sends)[index]
            def message(mid, role, content, **extra):
                return {"id": mid, "author": {"role": role}, "content": content, **extra}
            return {"conversation_id": cid, "current_node": "unrelated-answer", "mapping": {
                user: {"parent": "root", "message": message(user, "user", {"content_type": "text", "parts": ["original prompt"]})},
                "original-image": {"parent": user, "message": message("original-image", "tool", {
                    "content_type": "multimodal_text", "parts": [{"content_type": "image_asset_pointer",
                    "asset_pointer": "file-service://original-file-" + str(index)}]}, status="finished_successfully", create_time=1)},
                "original-final": {"parent": "original-image", "message": message("original-final", "assistant",
                    {"content_type": "text", "parts": [""]}, channel="final",
                    status="in_progress" if observed.active else "finished_successfully", end_turn=not observed.active)},
                "unrelated-user": {"parent": "root", "message": message("unrelated-user", "user", {"content_type": "text", "parts": ["other"]})},
                "unrelated-answer": {"parent": "unrelated-user", "message": message("unrelated-answer", "assistant",
                    {"content_type": "text", "parts": ["unrelated answer"]}, status="finished_successfully", end_turn=True)},
            }}
        def _list_recent_conversations(self, **kwargs):
            observed.lists.append(kwargs)
            return [{"id": "image-conversation-" + str(index), "update_time": runtime.clock.now} for index, _ in reversed(observed.sends)]
        def _get_file_download_url(self, file_id):
            assert file_id.startswith("original-file-")
            index = int(file_id.rsplit("-", 1)[1])
            observed.resolves.append(index)
            if observed.before_resolve:
                observed.before_resolve(index)
            return "https://download.example/" + str(index)
        def _poll_image_results(self, *args, **kwargs):
            raise AssertionError("saved original IDs must not poll or regenerate")
        def download_image_bytes(self, urls):
            index = int(urls[0].rsplit("/", 1)[1])
            observed.downloads.append(index)
            if observed.before_download:
                observed.before_download()
            if observed.download_failure:
                raise ConnectionError("synthetic download interruption")
            return [image_bytes(index)]
        def close(self):
            pass
    monkeypatch.setattr(conversation, "_generate_single_image", generate)
    monkeypatch.setattr(conversation, "save_image_bytes", image_url)
    monkeypatch.setattr(openai_backend_api, "OpenAIBackendAPI", Backend)
    monkeypatch.setattr("services.account_service.account_service", runtime.accounts)
    return observed


def execute(runtime, protocol, *, stream=False, n=1):
    data = {"model": "gpt-image-2", "stream": stream, "n": n, "prompt": "original prompt"}
    if protocol == "openai_v1_image_edit":
        data["images"] = [(image_bytes(0), "reference.png", "image/png")]
    if protocol == "openai_v1_chat_complete":
        data["messages"] = [{"role": "user", "content": "original prompt"}]
    if protocol == "openai_v1_response":
        data["input"] = "original prompt"
        data["tools"] = [{"type": "image_generation"}]
    runtime.service.submit(WHO["id"], durable_forward.envelope(WHO, data, request(), protocol, operation="image"))
    context = runtime.admission.claim_next()
    assert context is not None, original(runtime)
    runtime.admission.execute(context)
    return data


def recover(runtime):
    runtime.clock.now = max(runtime.clock.now, original(runtime).get("recovery_next_at") or 0) + 1
    return runtime.service.read(WHO["id"], "original-wire")


async def response_bytes(runtime, protocol, data):
    response = await durable_forward.respond(WHO, data, request(), protocol, operation="image", service=runtime.service)
    return response.body if hasattr(response, "body") else b"".join([part async for part in response.body_iterator])


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("saved_ids", [False, True])
def test_restart_recovers_original_image_without_a_second_send(runtime, monkeypatch, protocol, stream, saved_ids):
    observed = fixture_upstream(runtime, monkeypatch, saved_ids=saved_ids)
    data = execute(runtime, protocol, stream=stream)
    before = original(runtime)
    assert before["status"] == "unknown"
    with runtime.store.output_file(before["_wire_output"]) as handle:
        prefix = handle.read()
    assert runtime.admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    runtime.service = TextTaskService(runtime.store.path, admission=runtime.admission, clock=runtime.clock)
    assert recover(runtime)["status"] == "succeeded", original(runtime)
    wire = asyncio.run(response_bytes(runtime, protocol, data))
    assert image_url(image_bytes(1)).encode() in wire or base64.b64encode(image_bytes(1)) in wire
    assert b"PRIVATE_ACCOUNT_EMAIL" not in wire
    assert original(runtime)["provider_account_identity"] == before["provider_account_identity"]
    assert original(runtime)["request_message_id"] == before["request_message_id"]
    if before.get("_wire_identity", {}).get("id"):
        assert before["_wire_identity"]["id"].encode() in wire
    with runtime.store.output_file(before["_wire_output"]) as handle:
        assert handle.read() == prefix
    assert len(observed.sends) == len(observed.downloads) == len(runtime.logs.list(type="call")) == 1
    assert runtime.admission.resource_snapshot()["chat_turn"]["inflight"] == 0


def test_missing_cursor_recovers_by_original_message_not_latest_image(runtime, monkeypatch):
    observed = fixture_upstream(runtime, monkeypatch, cursor=False)
    execute(runtime, "openai_v1_image_generations")
    assert recover(runtime)["status"] == "unknown"
    assert recover(runtime)["status"] == "succeeded", original(runtime)
    assert len(observed.sends) == len(observed.downloads) == len(observed.lists) == 1


def test_download_failure_and_active_branch_keep_saved_result_without_redrawing(runtime, monkeypatch):
    observed = fixture_upstream(runtime, monkeypatch, saved_ids=True)
    execute(runtime, "openai_v1_image_generations")
    observed.download_failure = True
    assert recover(runtime)["status"] == "unknown"
    assert observed.reads == []  # Known file IDs go straight to download.
    observed.download_failure, observed.active = False, True
    assert recover(runtime)["status"] == "unknown"
    saved = original(runtime)["_image_slots"]["0"]["output_ref"]
    assert runtime.admission.resource_snapshot()["chat_turn"]["inflight"] == 1
    observed.active = False
    assert recover(runtime)["status"] == "succeeded"
    assert original(runtime)["_image_slots"]["0"]["output_ref"] == saved
    assert len(observed.sends) == 1 and len(observed.downloads) == 2


@pytest.mark.parametrize("protocol", ["openai_v1_image_generations", "openai_v1_chat_complete"])
def test_recovery_requeues_only_unsent_slots_and_keeps_original_account(runtime, monkeypatch, protocol):
    observed = fixture_upstream(runtime, monkeypatch, fail_slot=2)
    data = execute(runtime, protocol, n=3, stream=True)
    before = original(runtime)
    assert [index for index, _ in observed.sends] == [1, 2]
    first_ref = before["_image_slots"]["0"]["output_ref"]
    assert recover(runtime)["status"] == "queued"
    runtime.accounts.update_account(runtime.account["access_token"], {"managed_disabled": True})
    assert runtime.admission.claim_next() is None
    runtime.accounts.update_account(runtime.account["access_token"], {"managed_disabled": False})
    assert runtime.admission.claim_next() is None  # Re-enable still needs a valid refresh.
    runtime.accounts.update_account(runtime.account["access_token"], {"status": "正常"})
    context = runtime.admission.claim_next()
    assert context is not None, original(runtime).get("waiting")
    runtime.admission.execute(context)
    after = original(runtime)
    assert after["status"] == "succeeded", after
    assert after["_image_slots"]["0"]["output_ref"] == first_ref
    assert [index for index, _ in observed.sends] == [1, 2, 3]
    assert observed.downloads == [2]
    assert len(runtime.logs.list(type="call")) == 1
    wire = asyncio.run(response_bytes(runtime, protocol, data))
    for index in [1, 2, 3]:
        assert image_url(image_bytes(index)).encode() in wire or base64.b64encode(image_bytes(index)) in wire
    if before.get("_wire_identity", {}).get("id"):
        assert before["_wire_identity"]["id"].encode() in wire


def test_expired_recovery_worker_cannot_overwrite_new_claim_or_requeue_twice(runtime, monkeypatch):
    observed = fixture_upstream(runtime, monkeypatch, saved_ids=True)
    execute(runtime, "openai_v1_image_generations", n=2)
    entered, release = threading.Event(), threading.Event()
    first = []
    def block_first():
        if not first:
            first.append(True)
            entered.set()
            assert release.wait(10)
    observed.before_download = block_first
    with ThreadPoolExecutor() as executor:
        reading = executor.submit(runtime.service.read, WHO["id"], "original-wire")
        assert entered.wait(2)
        second = TextTaskService(runtime.store.path, admission=runtime.admission, clock=runtime.clock)
        assert second.read(WHO["id"], "original-wire")["status"] == "unknown"
        assert len(observed.downloads) == 1
        # A second process has its own conversation locks; SQLite is the
        # shared lease fence even after the first worker stops heartbeating.
        from services.account_service import AccountService
        from services.storage.json_storage import JSONStorageBackend
        monkeypatch.setattr("services.account_service.account_service", AccountService(
            JSONStorageBackend(runtime.store.path.parent / "accounts.json")))
        runtime.clock.now += 61
        assert second.read(WHO["id"], "original-wire")["status"] == "queued"
        new_ref = original(runtime)["_image_slots"]["0"]["output_ref"]
        release.set()
        assert reading.result(timeout=2)["status"] == "queued"
    assert original(runtime)["_image_slots"]["0"]["output_ref"] == new_ref
    runtime.admission.execute(runtime.admission.claim_next())
    assert original(runtime)["status"] == "succeeded"
    assert [index for index, _ in observed.sends] == [1, 2]


def test_saved_slots_rebuild_incomplete_wire_without_upstream_or_duplicate_log(runtime, monkeypatch):
    from services.protocol import openai_v1_chat_complete as chat
    observed = fixture_upstream(runtime, monkeypatch, fail_slot=0)
    formatter = chat.stream_image_chat_completion
    def disconnected(outputs, model, **kwargs):
        yield from formatter(outputs, model, **kwargs)
        raise ConnectionError("synthetic wire completion failure")
    monkeypatch.setattr(chat, "stream_image_chat_completion", disconnected)
    data = execute(runtime, "openai_v1_chat_complete", stream=True, n=2)
    before = original(runtime)
    assert before["status"] == "unknown" and before["_completed_slot"] == 1
    monkeypatch.setattr(chat, "stream_image_chat_completion", formatter)
    runtime.admission.recover_one()  # No subscriber or explicit recover request.
    assert original(runtime)["status"] == "succeeded"
    wire = asyncio.run(response_bytes(runtime, "openai_v1_chat_complete", data))
    assert before["_wire_identity"]["id"].encode() in wire
    assert [index for index, _ in observed.sends] == [1, 2]
    assert observed.reads == observed.downloads == []
    assert original(runtime)["_image_slots"] == before["_image_slots"]
    assert len(runtime.logs.list(type="call")) == 1


def test_old_raw_image_without_recovery_metadata_is_not_replayed(runtime, monkeypatch):
    observed = fixture_upstream(runtime, monkeypatch)
    execute(runtime, "openai_v1_image_generations")
    runtime.service._update(WHO["id"], "original-wire", _image_recovery=None)
    runtime.admission.recover_one()
    assert recover(runtime)["status"] == "unknown"
    assert len(observed.sends) == 1 and observed.reads == observed.downloads == []


@pytest.mark.parametrize("stage", ["read_image_request", "resolve_image_download", "download_image"])
def test_recovery_429_keeps_original_request_and_distinguishes_account_from_download(runtime, monkeypatch, stage):
    from services.account_request_pacing import AccountRequestClock
    from services.request_context import AdmissionLost
    from utils.helper import UpstreamHTTPError
    observed = fixture_upstream(runtime, monkeypatch, saved_ids=stage != "read_image_request")
    execute(runtime, "openai_v1_image_generations")
    clock = AccountRequestClock("synthetic-account", runtime.store.path.parent / "test-pacing.json")
    def limited(*args):
        with pytest.raises(AdmissionLost):
            current_request.get().before_send()
        if stage != "download_image":
            clock.limited(123, evidence={"phase": "account_read", "upstream_request_id": "original-upstream-get"})
        raise UpstreamHTTPError(stage, 429, "PRIVATE_ERROR_BODY", retry_after=123)
    if stage == "read_image_request":
        observed.before_read = limited
    elif stage == "resolve_image_download":
        observed.before_resolve = limited
    else:
        observed.before_download = limited
    assert recover(runtime)["status"] == "unknown"
    receipt = original(runtime)
    assert receipt["recovery_error_code"] == "RECOVERY_RATE_LIMITED"
    assert receipt["recovery_phase"] == stage and receipt["recovery_retry_after_seconds"] == 123
    assert receipt["recovery_next_at"] == runtime.clock.now + 123
    if stage != "download_image":
        assert receipt["rate_limit"]["layer"] == "upstream_chatgpt"
        assert receipt["rate_limit"]["recovery_phase"] == stage
        assert receipt["rate_limit"]["upstream_request_id"] == "original-upstream-get"
        assert receipt["rate_limit"]["request_ref"] == hashlib.sha256((WHO["id"] + ":original-wire").encode()).hexdigest()[:24]
        assert clock.cooldown_until > 0
    else:
        assert "rate_limit" not in receipt and clock.cooldown_until == 0
    assert "PRIVATE_ERROR_BODY" not in json.dumps(receipt)
    assert len(observed.sends) == 1
    observed.before_read = observed.before_resolve = observed.before_download = None
    assert recover(runtime)["status"] == "succeeded"
    assert len(observed.sends) == 1


def test_partial_original_url_failure_does_not_publish_partial_success(runtime, monkeypatch):
    from services import durable_image_forward
    observed = fixture_upstream(runtime, monkeypatch, saved_ids=True)
    execute(runtime, "openai_v1_image_generations")
    runtime.service._update(WHO["id"], "original-wire", **durable_image_forward._slot_changes(
        original(runtime), 0, file_ids=["original-file-1", "original-file-2"]))
    def fail_second(index):
        if index == 2:
            raise ConnectionError("synthetic second original URL unavailable")
    observed.before_resolve = fail_second
    assert recover(runtime)["status"] == "unknown"
    assert original(runtime)["_image_slots"]["0"]["file_ids"] == ["original-file-1", "original-file-2"]
    assert not original(runtime)["_image_slots"]["0"].get("completed")
    assert observed.resolves == [1, 2] and observed.downloads == []
    assert len(observed.sends) == 1


def test_verified_turn_end_releases_model_during_download_failure(runtime, monkeypatch):
    observed = fixture_upstream(runtime, monkeypatch)
    execute(runtime, "openai_v1_image_generations")
    observed.download_failure = True
    assert recover(runtime)["status"] == "unknown"
    receipt = original(runtime)
    assert receipt["_image_slots"]["0"]["turn_end"]["request_message_id"] == receipt["request_message_id"]
    assert runtime.admission.resource_snapshot()["chat_turn"]["inflight"] == 0
    assert receipt["recovery_phase"] == "download_image"
    assert not receipt["_image_slots"]["0"].get("completed")
    assert len(observed.sends) == 1
