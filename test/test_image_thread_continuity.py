"""Actual public task store/admission/bound-image path, controlled upstream only."""
import base64
import copy
import hashlib
from datetime import datetime, timezone
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from services.program_key_policy import make_policy
from services.image_thread import ImageThreadError, PROTOCOL, finished_parent, input_fields
from services.image_task_service import ImageTaskService, _request_hash
from services.task_store import TaskStore
from services.pool_admission import PoolAdmission
from services.account_service import AccountService
from services.storage.json_storage import JSONStorageBackend
from services.request_context import current_request
from services.protocol import conversation, openai_v1_image_edit, openai_v1_image_generations
from services.openai_backend_api import OpenAIBackendAPI as RealOpenAIBackendAPI

REAL_IMAGE_STREAM = conversation.stream_image_outputs


def png(color):
    out = BytesIO()
    Image.new("RGB", (3, 4), color).save(out, "PNG")
    return out.getvalue()


SOURCE = png("red")
OUTPUT = png("green")
WHO = {"id": "happy", "role": "user", "external_image_client": True, "policy": make_policy(["chat"], revision=1).to_record()}


def node(mid, role, parent, *, end=False):
    return {"parent": parent, "message": {"id": mid, "author": {"role": role},
            "status": "finished_successfully", "end_turn": end,
            "content": {"content_type": "text", "parts": ["fixture"]}}}


def document(cid="conversation-a", rid="request-a", parent="root"):
    return {"conversation_id": cid, "current_node": rid + "-final", "mapping": {
        rid: node(rid, "user", parent), rid + "-image": node(rid + "-image", "tool", rid),
        rid + "-final": node(rid + "-final", "assistant", rid + "-image", end=True)}}


def tool_document(cid="conversation-a", rid="request-a", parent="root"):
    result_id = "file_00000000" + hashlib.sha256(rid.encode()).hexdigest()[:24]
    tool = node(rid + "-image", "tool", rid + "-code")
    tool["message"]["content"] = {"content_type": "multimodal_text", "parts": [
        {"content_type": "image_asset_pointer", "asset_pointer": "file-service://" + result_id}]}
    return {"conversation_id": cid, "current_node": rid + "-image", "mapping": {
        rid: node(rid, "user", parent), rid + "-code": node(rid + "-code", "assistant", rid),
        rid + "-image": tool}}, result_id


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    rows = [{"access_token": "fixture-token", "account_id": "fixture-upstream",
             "provider_account_identity": "account-0", "type": "Plus", "status": "正常",
             "quota": 999, "source_type": "web", "conversation_binding_ids": ["binding-0"],
             "limits_progress": [{"feature_name": "image_gen", "remaining": 999}],
             "capacity_observed_at": datetime.now(timezone.utc).isoformat()}]
    (tmp_path / "accounts.json").write_text(json.dumps(rows))
    accounts = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    monkeypatch.setattr(accounts, "refresh_image_capability", lambda _: None)
    store = TaskStore(tmp_path / "text_tasks.sqlite3")
    admission = PoolAdmission(store, accounts, clock=lambda: 1000.0,
        settings=lambda: {"image_account_concurrency": 4, "chat_account_concurrency": 4, "codex_max_concurrency": 4},
        model_types=lambda _: {"Plus"}, pacing=lambda _a, now: {"next_at": now})
    service = ImageTaskService(tmp_path / "images.json", store=store, admission=admission)
    admission.register("image", lambda ctx, body: service._run_task(ctx.owner+":"+ctx.request_id,
        body["mode"], {**body["payload"], "retain_conversation": True}, body["identity"], body["payload"]["model"]))
    state = SimpleNamespace(sends=[], documents={}, fail_after_send=False, fail_after_result=False,
        tool_leaf=False, pruned_recap=False, drift=False, final_pending=False, reads=[], polls=[], naive_reads=0, archive_actions=[])
    account_stub = SimpleNamespace(get_bound_account_identity=lambda _b: "account-0",
        acquire_bound_image_access_token=lambda *a, **k: "fixture-token", get_account=lambda _t: rows[0],
        conversation_binding_lock=lambda *a: nullcontext(), mark_image_result=lambda *a, **kw: None,
        mark_image_capacity_consumed=lambda *a: None,
        release_image_slot=lambda *a: None, get_bound_text_access_token=lambda *a, **k: "fixture-token")
    monkeypatch.setattr(conversation, "account_service", account_stub)
    import services.account_service as account_module
    monkeypatch.setattr(account_module, "account_service", account_stub)
    for mod in (openai_v1_image_edit, openai_v1_image_generations):
        monkeypatch.setattr(mod, "count_text_tokens", lambda *a, **k: 0)
    class Backend:
        _has_image_asset_pointer = RealOpenAIBackendAPI._has_image_asset_pointer
        def __init__(self, access_token):
            assert access_token == "fixture-token"
            self.image_submission_started = False
        def _get_conversation(self, cid, **kwargs):
            state.reads.append(cid)
            doc = copy.deepcopy(state.documents[cid])
            if state.drift:
                doc["mapping"]["foreign"] = node("foreign", "user", doc["current_node"])
                doc["current_node"] = "foreign"
            return doc
        def get_conversation_parent_message_id(self, cid):
            state.naive_reads += 1
            raise AssertionError("new thread may not accept arbitrary current_node")
        def set_conversation_archived(self, cid, parent, archived, *, validate_document=None):
            doc = self._get_conversation(cid)
            if validate_document is not None:
                parent = validate_document(doc) or parent
            assert doc['current_node'] == parent
            assert parent in state.documents[cid]["mapping"]
            state.documents[cid]["is_archived"] = archived
            state.archive_actions.append((cid, archived))
            return {"archived": archived}
        def archive_conversation(self, cid, parent):
            return self.set_conversation_archived(cid, parent, True)
        def resolve_conversation_image_urls(self, cid, files, sediment, **kwargs):
            assert kwargs["request_message_id"] in state.documents[cid]["mapping"]
            return ["https://fixture.invalid/original.png"]
        def _poll_image_results(self, cid, timeout, *, request_message_id, initial_document=None):
            assert request_message_id in state.documents[cid]["mapping"]
            state.polls.append((cid, request_message_id))
            _, result_id = tool_document(cid, request_message_id)
            return [result_id], [result_id]
        def download_image_bytes(self, urls):
            assert urls == ["https://fixture.invalid/original.png"]
            return [OUTPUT]
        def close(self):
            pass
    monkeypatch.setattr(conversation, "OpenAIBackendAPI", Backend)
    import services.openai_backend_api as backend_module
    monkeypatch.setattr(backend_module, "OpenAIBackendAPI", Backend)
    def stream(backend, req, index, total):
        callback = req.progress_callback
        rid = callback.request_message_id
        cid = req.conversation_id or "conversation-" + str(len(state.documents))
        backend.image_request_message_id = rid
        context = current_request.get()
        if callable(getattr(backend, "image_pre_send_check", None)):
            backend.image_pre_send_check()
        context.before_send()
        backend.image_submission_started = True
        callback.record_submission_started()
        callback.record_conversation_id(cid)
        state.sends.append({"task": context.request_id, "account": req.provider_account_identity,
                            "conversation": cid, "parent": req.parent_message_id, "message": rid,
                            "images": list(req.images or []), "thread": callback.image_thread})
        old = state.documents.get(cid)
        if old:
            assert old["current_node"] == req.parent_message_id
        if state.tool_leaf or state.pruned_recap:
            doc, result_id = tool_document(cid, rid, req.parent_message_id or "root")
            if state.pruned_recap:
                doc["mapping"].pop(rid + "-code")
                doc["mapping"][rid + "-image"]["parent"] = rid
                recap = node(rid + "-recap", "assistant", doc["current_node"])
                recap["message"]["recipient"] = "all"
                recap["message"]["content"] = {"content_type": "reasoning_recap", "content": "fixture"}
                doc["mapping"][rid + "-recap"] = recap
                doc["mapping"][rid + "-final"] = node(rid + "-final", "assistant", rid + "-recap", end=True)
                doc["current_node"] = rid + "-final"
        else:
            doc = document(cid, rid, req.parent_message_id or "root")
            result_id = "file-" + rid
        if old:
            doc["mapping"] = {**old["mapping"], **doc["mapping"]}
            if state.pruned_recap:
                prior = doc["mapping"].pop(req.parent_message_id)
                doc["mapping"][rid]["parent"] = prior["parent"]
        state.documents[cid] = doc
        if state.fail_after_send:
            raise ConnectionError("fixture reply lost after sending")
        callback.record_result_ids([result_id], [result_id] if state.tool_leaf else [])
        if state.fail_after_result:
            raise ConnectionError("fixture result reference saved before reply loss")
        if state.final_pending:
            doc["mapping"][rid + "-final"]["message"].update(end_turn=False, status="in_progress")
        yield conversation.ImageOutput(kind="result", model=req.model, index=index, total=total,
            conversation_id=cid, data=[{"b64_json": base64.b64encode(OUTPUT).decode()}])
    monkeypatch.setattr(conversation, "stream_image_outputs", stream)
    def submit(tid, thread="product-a", *, who=WHO, source=None, images=None):
        args = dict(client_task_id=tid, prompt="fixture image", model="gpt-image-2", size=None)
        if thread is not None:
            args["image_thread_id"] = thread
        if source:
            args["edit_source_task_id"] = source
        return service.submit_edit(who, **args, images=images or [(OUTPUT if source else SOURCE, "input.png", "image/png")])
    def read(tid, who=WHO):
        with store.connect() as db:
            return store.read_receipt(db, "image", who["id"], tid)
    return SimpleNamespace(service=service, store=store, admission=admission, state=state,
                           account_stub=account_stub,
                           submit=submit, read=read, root=tmp_path)


def run_next(r, expected):
    ctx = r.admission.claim_next()
    assert ctx and ctx.request_id == expected
    r.admission.execute(ctx)
    result = r.read(expected)
    assert result["status"] == "success", result
    return result


@pytest.mark.parametrize("path", ["direct", "text_retry", "fallback"])
@pytest.mark.parametrize("fail_confirmation", [False, True])
def test_first_download_stays_private_until_confirmed_and_survives_restart(runtime, monkeypatch, path, fail_confirmation):
    r = runtime
    backend_class = conversation.OpenAIBackendAPI
    fail = [fail_confirmation]
    calls = {"download": 0, "resolve": 0, "published": 0}

    def events(backend, **kwargs):
        callback = backend.progress_callback
        rid = callback.request_message_id
        cid = "conversation-first-download"
        current_request.get().before_send()
        backend.image_request_message_id = rid
        backend.image_submission_started = True
        callback.record_submission_started()
        callback.record_conversation_id(cid)
        doc, asset = tool_document(cid, rid)
        r.state.documents[cid] = doc
        r.state.sends.append({"conversation": cid, "message": rid})
        yield {"conversation_id": cid, "file_ids": [asset] if path == "direct" else [],
               "text": '{"referenced_image_ids": ["input"]}' if path == "text_retry" else "",
               "turn_use_case": "image gen"}

    def resolve(_backend, cid, files, sediments, **_kwargs):
        calls["resolve"] += 1
        return ["https://fixture.invalid/original.png"] if files or sediments else []

    def poll(_backend, cid, *_args, request_message_id, **_kwargs):
        _, asset = tool_document(cid, request_message_id)
        return [asset], []

    def download(_backend, _urls):
        calls["download"] += 1
        return [OUTPUT]

    def read(_backend, cid):
        if fail[0]:
            raise TimeoutError("original final read unavailable")
        return copy.deepcopy(r.state.documents[cid])

    def publish(items, *_args, **_kwargs):
        assert not fail[0], "no public image storage before exact final confirmation"
        calls["published"] += 1
        return {"data": [{"b64_json": items[0]["b64_json"], "url": "https://fixture.invalid/confirmed.png"}]}

    monkeypatch.setattr(conversation, "conversation_events", events)
    monkeypatch.setattr(conversation, "stream_image_outputs", REAL_IMAGE_STREAM)
    monkeypatch.setattr(conversation, "_get_detailed_error_from_tasks", lambda *_a, **_k: "")
    monkeypatch.setattr(backend_class, "resolve_conversation_image_urls", resolve)
    monkeypatch.setattr(backend_class, "_poll_image_results", poll)
    monkeypatch.setattr(backend_class, "download_image_bytes", download)
    monkeypatch.setattr(backend_class, "_get_conversation", read)
    monkeypatch.setattr(conversation, "format_image_result", publish)
    r.submit("original")
    r.admission.execute(r.admission.claim_next())
    original = r.read("original")
    assert calls["download"] == 1
    if fail_confirmation:
        assert original["status"] == "error" and not original.get("data")
        assert r.service.list_tasks(WHO, ["original"])["items"][0]["result_stage"] == "downloaded_waiting_original_confirmation"
        assert calls["published"] == 0
        cached = original["_pending_image_output"]
        with r.store.output_file(cached["output_ref"]) as handle:
            import os
            assert os.fstat(handle.fileno()).st_mode & 0o077 == 0
            assert json.loads(handle.read())[0]["b64_json"] == base64.b64encode(OUTPUT).decode()
        restarted = ImageTaskService(r.root / "images.json", store=TaskStore(r.store.path), admission=r.admission)
        fail[0] = False
        resolves_before = calls["resolve"]
        restarted._run_resume_poll("happy:original", original["conversation_id"], 5, "", WHO,
                                   "edit", "gpt-image-2", False, False)
        assert calls["resolve"] == resolves_before
    result = r.read("original")
    assert result["status"] == "success" and result["_image_thread_terminal"] is True
    assert not result.get("_pending_image_output")
    assert calls["download"] == calls["published"] == len(r.state.sends) == 1
    assert result["conversation_id"] == original["conversation_id"]
    assert r.service.list_tasks(WHO, ["original"])["items"][0]["result_stage"] == "result_ready"


def test_private_image_output_cannot_escape_through_chunk_or_collection():
    private = conversation.ImageOutput(kind="result", model="gpt-image-2", index=1, total=1,
                                      _pending_image_items=[{"b64_json": "cHJpdmF0ZQ=="}])
    assert private.to_chunk()["data"] == []
    assert conversation.collect_image_outputs([private])["data"] == []
    assert "cHJpdmF0ZQ==" not in repr(private)


@pytest.mark.parametrize("change", ["none", "unfinished", "sibling", "successor", "missing-original", "namespace", "namespace-added", "nonfinal", "download-failure"])
def test_recovery_expands_partial_assets_only_on_complete_original_branch(runtime, monkeypatch, change):
    r = runtime
    r.state.tool_leaf = r.state.fail_after_result = True
    r.submit("original"); r.admission.execute(r.admission.claim_next())
    original = r.read("original")
    assert original["status"] == "error"
    cid, rid = original["conversation_id"], original["request_message_id"]
    asset = original["result_file_ids"][0]
    if change == "namespace-added":
        r.service._update_task("happy:original", result_sediment_ids=[])
        original = r.read("original")
    coverage = {"conversation_id": cid, "request_message_id": rid,
                "file_ids": original["result_file_ids"], "sediment_ids": original["result_sediment_ids"]}
    r.service._store_pending_image_output("happy:original", coverage,
        [{"b64_json": base64.b64encode(OUTPUT).decode()}])
    cached = r.read("original")["_pending_image_output"]
    doc = r.state.documents[cid]
    doc["mapping"][rid + "-image"]["message"]["content"]["parts"][0]["asset_pointer"] = "sediment://" + asset
    extra = "file_00000000" + hashlib.sha256((rid + "-extra").encode()).hexdigest()[:24]
    if change == "namespace-added": extra = asset
    tool = copy.deepcopy(doc["mapping"][rid + "-image"])
    tool["parent"] = rid + "-image"
    tool["message"]["id"] = rid + "-extra"
    tool["message"]["content"]["parts"][0]["asset_pointer"] = "sediment://" + extra
    doc["mapping"][rid + "-extra"] = tool
    doc["mapping"][rid + "-final"] = node(rid + "-final", "assistant", rid + "-extra", end=True)
    doc["current_node"] = rid + "-final"
    if change == "unfinished": tool["message"]["status"] = "in_progress"
    if change == "nonfinal": doc["mapping"][rid + "-final"]["message"]["channel"] = "commentary"
    if change == "sibling": doc["mapping"]["sibling"] = node("sibling", "assistant", rid)
    if change == "successor":
        doc["mapping"]["foreign"] = node("foreign", "user", doc["current_node"])
        doc["current_node"] = "foreign"
    if change == "missing-original":
        doc["mapping"][rid + "-image"]["message"]["content"]["parts"] = []
    Backend = conversation.OpenAIBackendAPI
    for name, value in {
        "_current_message_branch_ids": staticmethod(RealOpenAIBackendAPI._current_message_branch_ids),
        "_extract_image_reference_ids": staticmethod(RealOpenAIBackendAPI._extract_image_reference_ids),
        "_extract_image_tool_records": RealOpenAIBackendAPI._extract_image_tool_records,
    }.items(): monkeypatch.setattr(Backend, name, value, raising=False)
    if change == "namespace":
        def records(self, document, request_id):
            results = RealOpenAIBackendAPI._extract_image_tool_records(self, document, request_id)
            for record in results: record["sediment_ids"] = []
            return results
        monkeypatch.setattr(Backend, "_extract_image_tool_records", records)
    downloads = []
    monkeypatch.setattr(Backend, "resolve_conversation_image_urls",
        lambda _self, _cid, files, _sediments, **_kw: files)
    def download(_self, urls):
        downloads.append(urls)
        if change == "download-failure": raise TimeoutError("download unavailable")
        return [OUTPUT] if change == "namespace-added" else [OUTPUT, SOURCE]
    monkeypatch.setattr(Backend, "download_image_bytes", download)
    monkeypatch.setattr(conversation, "format_image_result", lambda items, *_a, **_kw: {"data": items})
    restarted = ImageTaskService(r.root / "images.json", store=TaskStore(r.store.path), admission=r.admission)
    restarted._run_resume_poll("happy:original", cid, 5, "", WHO, "edit", "gpt-image-2", False, False)
    result = r.read("original")
    assert len(r.state.sends) == 1 and result["conversation_id"] == cid and result["request_message_id"] == rid
    with r.store.output_file(cached["output_ref"]) as handle:
        assert json.loads(handle.read())[0]["b64_json"] == base64.b64encode(OUTPUT).decode()
    if change in {"none", "namespace-added"}:
        assert result["status"] == "success" and len(result["data"]) == len({asset, extra})
        assert set(result["result_file_ids"]) == {asset, extra}
        assert set(result["result_sediment_ids"]) == {asset, extra}
        assert result["parent_message_id"] == rid + "-final" and result["_image_thread_terminal"]
        assert downloads == [list(dict.fromkeys([asset, extra]))]
    else:
        assert result["status"] == "error" and not result.get("data")
        assert result["_pending_image_output"] == cached
        if change != "download-failure":
            assert not downloads and result["result_file_ids"] == original["result_file_ids"]
        else:
            change = "none"
            restored = ImageTaskService(r.root / "images.json", store=TaskStore(r.store.path), admission=r.admission)
            restored._run_resume_poll("happy:original", cid, 5, "", WHO, "edit", "gpt-image-2", False, False)
            result = r.read("original")
            assert result["status"] == "success" and len(result["data"]) == 2
            assert set(result["result_file_ids"]) == {asset, extra} and len(r.state.sends) == 1


@pytest.mark.parametrize("change", ["none", "unfinished", "wrong-parent", "sibling", "current", "late-drift"])
def test_strict_terminal_poll_saves_settle_read_but_keeps_post_download_fence(runtime, monkeypatch, change):
    from services.config import config
    r = runtime
    r.submit("prior"); run_next(r, "prior")
    backend_class = conversation.OpenAIBackendAPI
    monkeypatch.setattr(backend_class, "_current_message_branch_ids",
                        staticmethod(RealOpenAIBackendAPI._current_message_branch_ids), raising=False)
    calls = {"read": 0, "download": 0, "publish": 0}

    def events(backend, **kwargs):
        callback = backend.progress_callback
        rid = callback.request_message_id
        cid = kwargs["conversation_id"]
        current_request.get().before_send()
        backend.image_request_message_id = rid
        backend.image_submission_started = True
        callback.record_submission_started(); callback.record_conversation_id(cid)
        doc, _ = tool_document(cid, rid, kwargs["parent_message_id"])
        doc["mapping"] = {**r.state.documents[cid]["mapping"], **doc["mapping"]}
        r.state.documents[cid] = doc
        r.state.sends.append({"conversation": cid, "message": rid})
        yield {"conversation_id": cid, "file_ids": [], "turn_use_case": "image gen"}

    def read(backend, cid):
        calls["read"] += 1
        doc = copy.deepcopy(r.state.documents[cid])
        rid = backend.image_request_message_id
        if calls["read"] == 1:
            if change == "unfinished": doc["mapping"][rid + "-image"]["message"]["status"] = "in_progress"
            if change == "wrong-parent": doc["mapping"][rid]["parent"] = "foreign"
            if change == "sibling": doc["mapping"]["sibling"] = node("sibling", "assistant", rid)
            if change == "current": doc["current_node"] = rid
        if change == "late-drift" and calls["download"]:
            doc["mapping"]["foreign"] = node("foreign", "user", doc["current_node"])
            doc["current_node"] = "foreign"
        return doc

    def poll(backend, cid, timeout, *args, **kwargs):
        # Run the real polling algorithm with the callback installed by the
        # real bound wrapper; transport alone is controlled.
        probe = RealOpenAIBackendAPI.__new__(RealOpenAIBackendAPI)
        probe.progress_callback = backend.progress_callback
        if hasattr(backend, "image_poll_terminal_check"):
            probe.image_poll_terminal_check = backend.image_poll_terminal_check
        probe._get_conversation = lambda c: read(backend, c)
        probe._query_backend_tasks = lambda **_kw: []
        return probe._poll_image_results(cid, timeout, *args, **kwargs)

    def resolve(_backend, _cid, files, sediments, **_kwargs):
        return ["https://fixture.invalid/original.png"] if files or sediments else []

    def download(_backend, _urls):
        calls["download"] += 1
        return [OUTPUT]

    def publish(items, *_args, **_kwargs):
        calls["publish"] += 1
        assert calls["read"] >= 2, "fresh post-download confirmation must precede publication"
        return {"data": items}

    for name, value in {"image_poll_initial_wait_secs": 0, "image_poll_interval_secs": .01,
                        "image_check_before_hit_enabled": True, "image_settle_enabled": True,
                        "image_settle_secs": .01}.items(): monkeypatch.setitem(config.data, name, value)
    monkeypatch.setattr(conversation, "conversation_events", events)
    monkeypatch.setattr(conversation, "stream_image_outputs", REAL_IMAGE_STREAM)
    monkeypatch.setattr(conversation, "_get_detailed_error_from_tasks", lambda *_a, **_k: "")
    monkeypatch.setattr(conversation, "format_image_result", publish)
    for name, value in {"_poll_image_results": poll, "_get_conversation": read,
                        "resolve_conversation_image_urls": resolve, "download_image_bytes": download}.items():
        monkeypatch.setattr(backend_class, name, value)
    # The continuation precheck is outside the measured result polling reads.
    original_read = read
    def with_precheck(backend, cid):
        if not hasattr(backend, "image_request_message_id"):
            return copy.deepcopy(r.state.documents[cid])
        return original_read(backend, cid)
    monkeypatch.setattr(backend_class, "_get_conversation", with_precheck)
    r.submit("original"); r.admission.execute(r.admission.claim_next())
    result = r.read("original")
    assert calls["read"] == (2 if change in {"none", "late-drift"} else 3)
    assert calls["download"] == 1 and len(r.state.sends) == 2
    if change == "late-drift":
        assert result["status"] == "error" and not result.get("data") and calls["publish"] == 0
        assert result.get("_pending_image_output"), "retain private original on late branch change"
    else:
        assert result["status"] == "success" and calls["publish"] == 1
        assert result["conversation_id"] == r.read("prior")["conversation_id"]


@pytest.mark.parametrize('change', ['none', 'external_successor', 'download_failure', 'partial_download'])
def test_complete_original_final_publishes_without_second_get_but_mutations_recheck(runtime, monkeypatch, change):
    r = runtime
    Backend = conversation.OpenAIBackendAPI
    calls = {'read': 0, 'download': 0, 'publish': 0}
    for name, value in {
        '_current_message_branch_ids': staticmethod(RealOpenAIBackendAPI._current_message_branch_ids),
        '_extract_image_reference_ids': staticmethod(RealOpenAIBackendAPI._extract_image_reference_ids),
        '_extract_image_tool_records': RealOpenAIBackendAPI._extract_image_tool_records,
    }.items():
        monkeypatch.setattr(Backend, name, value, raising=False)

    def events(backend, **kwargs):
        callback = backend.progress_callback
        rid, cid = callback.request_message_id, 'complete-original'
        current_request.get().before_send()
        backend.image_request_message_id = rid
        backend.image_submission_started = True
        callback.record_submission_started(); callback.record_conversation_id(cid)
        doc, asset = tool_document(cid, rid)
        assets = [asset]
        if change == 'partial_download':
            assets.append(asset + 'b')
            doc['mapping'][rid + '-image']['message']['content']['parts'].append(
                {'content_type': 'image_asset_pointer', 'asset_pointer': 'file-service://' + assets[-1]})
        doc['mapping'][rid + '-final'] = node(rid + '-final', 'assistant', rid + '-image', end=True)
        doc['current_node'] = rid + '-final'
        r.state.documents[cid] = doc
        r.state.sends.append({'conversation': cid, 'message': rid})
        yield {'conversation_id': cid, 'file_ids': assets, 'sediment_ids': [], 'turn_use_case': 'image gen'}
        raise AssertionError('complete original signal must close this stream')

    def read(_backend, cid, **kwargs):
        calls['read'] += 1
        return copy.deepcopy(r.state.documents[cid])

    def download(_backend, urls):
        calls['download'] += 1
        assert len(urls) == 1
        if change == 'download_failure': raise TimeoutError('attachment unavailable')
        if change == 'external_successor':
            doc = r.state.documents['complete-original']
            doc['mapping']['external'] = node('external', 'user', doc['current_node'])
            doc['current_node'] = 'external'
        return [OUTPUT]

    def publish(items, *_args, **_kwargs):
        calls['publish'] += 1
        assert calls['read'] == 1
        assert [base64.b64decode(i['b64_json']) for i in items] == [OUTPUT]
        return {'data': items}

    monkeypatch.setattr(conversation, 'conversation_events', events)
    monkeypatch.setattr(conversation, 'stream_image_outputs', REAL_IMAGE_STREAM)
    monkeypatch.setattr(conversation, 'format_image_result', publish)
    monkeypatch.setattr(Backend, '_get_conversation', read)
    monkeypatch.setattr(Backend, 'download_image_bytes', download)
    monkeypatch.setattr(Backend, 'resolve_conversation_image_urls',
        lambda _self, _cid, files, _sediments, **_kw: ['https://fixture.invalid/' + f for f in files[:1]])
    r.submit('original'); r.admission.execute(r.admission.claim_next())
    original = r.read('original')
    assert len(r.state.sends) == 1 and calls['download'] == calls['read'] == 1
    if change in {'download_failure', 'partial_download'}:
        assert original['status'] == 'error' and not original.get('data') and calls['publish'] == 0
        if change == 'partial_download':
            assert len(original['result_file_ids']) == 2 and original.get('_pending_image_output')
        return
    assert original['status'] == 'success' and calls['publish'] == 1
    assert len(original['data']) == len(original['result_file_ids']) == 1
    assert original['parent_message_id'] == original['request_message_id'] + '-final'
    if change == 'external_successor':
        # Saving immutable original results grants no authority to mutate a
        # conversation whose current cursor has subsequently changed.
        with pytest.raises(ImageThreadError): r.service.archive_thread(WHO, 'original')
        assert calls['read'] == 2 and not r.state.archive_actions
        r.submit('next', source='original'); r.admission.execute(r.admission.claim_next())
        assert calls['read'] == 3 and len(r.state.sends) == 1
        assert r.read('next').get('upstream_outcome') in ('not_sent', 'not_submitted')


def test_complete_image_proof_rejects_tool_leaf_missing_assets_and_nonfinal_channel():
    doc, asset = tool_document()
    with pytest.raises(ImageThreadError):
        finished_parent(doc, 'conversation-a', 'request-a', expected_result_ids=[asset], require_final=True)
    doc['mapping']['request-a-final'] = node('request-a-final', 'assistant', 'request-a-image', end=True)
    doc['current_node'] = 'request-a-final'
    assert finished_parent(doc, 'conversation-a', 'request-a', expected_result_ids=[asset], require_final=True) == 'request-a-final'
    for expected in ([], [asset, 'missing-asset']):
        with pytest.raises(ImageThreadError):
            finished_parent(doc, 'conversation-a', 'request-a', expected_result_ids=expected, require_final=True)
    doc['mapping']['request-a-final']['message']['channel'] = 'analysis'
    with pytest.raises(ImageThreadError):
        finished_parent(doc, 'conversation-a', 'request-a', expected_result_ids=[asset], require_final=True)


def test_advanced_selector_reaches_actual_bound_protocol_and_thread(runtime):
    r = runtime
    selected = r.admission.accounts.list_accounts()[0]
    ref = r.admission.accounts.pool_account_ref(selected)
    # Another fully eligible physical account must not change the explicit choice.
    other = {**selected, "access_token": "other-fixture", "account_id": "other-upstream",
             "provider_account_identity": "other-account", "conversation_binding_ids": []}
    other.pop("managed_pool_account_ref", None)
    with r.admission.accounts._lock:
        r.admission.accounts._accounts[other["access_token"]] = other
        r.admission.accounts._save_accounts()
    r.service.submit_generation(WHO, client_task_id="chosen", prompt="fixture", model="gpt-image-2",
                                size=None, account_ref=ref, image_thread_id="product-a")
    first = run_next(r, "chosen")
    r.submit("inherited")
    second = run_next(r, "inherited")
    assert [send["account"] for send in r.state.sends] == ["account-0", "account-0"]
    assert first["_requested_account_ref"] == second["_requested_account_ref"] == ref
    assert first["conversation_id"] == second["conversation_id"]


def test_same_product_images_and_edit_of_earlier_image_use_original_real_conversation(runtime):
    r = runtime
    r.submit("main-v1"); r.submit("selling-v1")
    first = run_next(r, "main-v1")
    second = run_next(r, "selling-v1")
    r.submit("main-v2", source="main-v1")
    third = run_next(r, "main-v2")
    assert len(r.state.documents) == 1
    assert len({s["account"] for s in r.state.sends}) == 1
    assert len({s["message"] for s in r.state.sends}) == 3
    assert r.state.sends[1]["parent"] == first["parent_message_id"]
    assert r.state.sends[2]["parent"] == second["parent_message_id"], "editing main must append after latest completed image, not fork back"
    assert r.state.sends[2]["images"][0] == base64.b64encode(OUTPUT).decode()
    assert third["_image_thread"]["edit_source_task_id"] == "main-v1"
    assert r.read("main-v1")["data"] == first["data"]
    assert r.state.naive_reads == 0


def test_review_approval_archives_exact_image_thread_and_later_edit_restores_it(runtime):
    r = runtime
    r.submit("main-v1"); first = run_next(r, "main-v1")
    result = r.service.archive_thread(WHO, "main-v1")
    assert result["archived"] is True
    assert r.state.archive_actions == [(first["conversation_id"], True)]
    assert r.service.archive_thread(WHO, "main-v1")["archived"] is True
    r.submit("main-v2", source="main-v1")
    run_next(r, "main-v2")
    assert r.state.archive_actions[-1] == (first["conversation_id"], False)


def test_unarchive_timeout_requeues_original_image_request_before_any_new_send(runtime, monkeypatch):
    r = runtime
    r.submit("main-v1"); first = run_next(r, "main-v1")
    r.service.archive_thread(WHO, "main-v1")
    original = conversation.OpenAIBackendAPI.set_conversation_archived
    attempts = 0
    def unarchive_once(self, cid, parent, archived):
        nonlocal attempts
        if not archived and attempts == 0:
            attempts += 1
            raise TimeoutError("fixture unarchive response lost")
        return original(self, cid, parent, archived)
    monkeypatch.setattr(conversation.OpenAIBackendAPI, "set_conversation_archived", unarchive_once)
    r.submit("main-v2", source="main-v1")
    before = r.read("main-v2")
    claim = r.admission.claim_next(); assert claim and claim.request_id == "main-v2"
    r.admission.execute(claim)
    waiting = r.read("main-v2")
    assert waiting["status"] == "queued" and waiting["upstream_outcome"] == "not_submitted"
    assert waiting["request_hash"] == before["request_hash"]
    assert len(r.state.sends) == 1
    assert r.admission.claim_next() is None
    r.admission.clock = lambda: 1100.0
    recovered = run_next(r, "main-v2")
    assert not recovered.get("error_code") and not recovered.get("waiting")
    assert len(r.state.sends) == 2
    assert r.state.sends[-1]["conversation"] == first["conversation_id"]
    assert r.state.archive_actions == [(first["conversation_id"], True), (first["conversation_id"], False)]


def test_cursor_drift_during_unarchive_never_submits_new_image(runtime, monkeypatch):
    r = runtime
    r.submit("main-v1"); first = run_next(r, "main-v1")
    r.service.archive_thread(WHO, "main-v1")
    original_get = conversation.OpenAIBackendAPI._get_conversation
    reads = 0
    def drift_after_first_read(self, cid):
        nonlocal reads
        reads += 1
        document = original_get(self, cid)
        if reads >= 2:
            document["mapping"]["newer"] = node("newer", "user", document["current_node"])
            document["current_node"] = "newer"
        return document
    monkeypatch.setattr(conversation.OpenAIBackendAPI, "_get_conversation", drift_after_first_read)
    monkeypatch.setattr(conversation.OpenAIBackendAPI, "set_conversation_archived",
                        RealOpenAIBackendAPI.set_conversation_archived)
    r.submit("main-v2", source="main-v1")
    before = r.read("main-v2")
    claim = r.admission.claim_next(); assert claim and claim.request_id == "main-v2"
    r.admission.execute(claim)
    waiting = r.read("main-v2")
    assert reads >= 2
    assert waiting["status"] == "queued" and waiting["upstream_outcome"] == "not_submitted"
    assert waiting["request_hash"] == before["request_hash"]
    assert len(r.state.sends) == 1
    assert r.state.archive_actions == [(first["conversation_id"], True)]


def test_old_or_unfinished_image_task_cannot_archive_newer_work(runtime):
    r = runtime
    r.submit("main-v1"); run_next(r, "main-v1")
    r.submit("selling-v1")
    with pytest.raises(ImageThreadError, match="IMAGE_THREAD_NOT_TERMINAL"):
        r.service.archive_thread(WHO, "main-v1")
    assert not r.state.archive_actions


def test_image_archive_route_uses_original_owner_task_and_no_upstream_cursor(runtime, monkeypatch):
    r = runtime
    r.submit("main-v1"); run_next(r, "main-v1")
    import api.image_tasks as routes
    monkeypatch.setattr(routes, "image_task_service", r.service)
    monkeypatch.setattr(routes, "require_identity", lambda *a, **k: WHO)
    app = FastAPI(); app.include_router(routes.create_router())
    with TestClient(app) as client:
        assert client.post("/api/image-tasks/main-v1/archive-thread", json={"conversation_id": "forged"}).status_code == 422
        response = client.post("/api/image-tasks/main-v1/archive-thread", json={})
        assert response.status_code == 200, response.text
        assert response.json() == {"image_thread": {"protocol": PROTOCOL, "id": "product-a",
            "previous_task_id": None, "edit_source_task_id": None}, "archived": True, "task_id": "main-v1"}
        assert client.post("/api/image-tasks/main-v1/restore-thread", json={"conversation_id":"forged"}).status_code == 422
        restored = client.post("/api/image-tasks/main-v1/restore-thread", json={})
        assert restored.status_code == 200 and restored.json()["archived"] is False
        assert client.post("/api/image-tasks/unknown/archive-thread", json={}).status_code == 404
    assert [archived for _,archived in r.state.archive_actions] == [True, False]


def test_same_thread_waits_but_another_product_keeps_its_independent_capacity(runtime):
    r=runtime
    r.submit("a1"); r.submit("a2"); r.submit("b1", "product-b")
    first=r.admission.claim_next(); assert first.request_id == "a1"
    next_=r.admission.claim_next(); assert next_.request_id == "b1"
    assert r.admission.claim_next() is None
    r.admission.execute(first); r.admission.execute(next_)
    run_next(r, "a2")
    assert r.state.sends[0]["conversation"] != r.state.sends[1]["conversation"]
    assert r.state.sends[0]["conversation"] == r.state.sends[2]["conversation"]


def test_concurrent_acceptance_is_a_single_transactional_predecessor_chain(runtime):
    r=runtime
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda i:r.submit("image-"+str(i)),range(6)))
    with r.store.connect() as db:
        tasks=sorted([row[3] for row in r.store.receipts(db)],key=lambda v:v["_sequence"])
    assert [v["_image_thread"]["previous_task_id"] for v in tasks] == [None]+[v["id"] for v in tasks[:-1]]
    for task in tasks: run_next(r,task["id"])
    assert len(r.state.documents)==1


def test_restart_and_duplicate_submission_keep_original_inputs_and_do_not_regenerate(runtime):
    r=runtime
    r.submit("a1"); first=run_next(r,"a1")
    before=(first["request_hash"],first["_input_ref"],first["data"])
    restarted=ImageTaskService(r.root/"images.json",store=TaskStore(r.store.path),admission=r.admission)
    result=restarted.submit_edit(WHO,client_task_id="a1",prompt="fixture image",model="gpt-image-2",size=None,
                                 image_thread_id="product-a",images=[(SOURCE,"input.png","image/png")])
    assert result["status"]=="success"
    assert (r.read("a1")["request_hash"],r.read("a1")["_input_ref"],r.read("a1")["data"])==before
    r.submit("a2");run_next(r,"a2"); assert len(r.state.sends)==2


@pytest.mark.parametrize("state",["unknown","failed","suppressed","missing","unconfirmed"])
def test_unresolved_or_missing_predecessor_never_dispatches_replacement(runtime,state):
    r=runtime;r.submit("a1");run_next(r,"a1");r.submit("a2")
    with r.store.transaction() as db:
        prior=r.store.read_receipt(db,"image","happy","a1")
        if state=="missing":db.execute("DELETE FROM image_requests WHERE task_key=?",("happy:a1",))
        else:
            if state=="suppressed":prior["_recovery_suppressed"]={"reason":"operator"}
            elif state=="unconfirmed":prior.pop("_image_thread_terminal",None)
            else:prior.update(status="error",error_code="CONVERSATION_OUTCOME_UNKNOWN" if state=="unknown" else "content_policy_violation",upstream_unfinished=state=="unknown")
            r.store.write_receipt(db,"image","happy","a1",prior)
    assert r.admission.claim_next() is None
    assert len(r.state.sends)==1; assert r.read("a2")["status"]=="queued"


def test_late_manual_message_rejects_continuation_without_new_chat_or_new_send(runtime):
    r=runtime;r.submit("a1");run_next(r,"a1");r.submit("a2")
    r.state.drift=True
    r.admission.execute(r.admission.claim_next())
    task=r.read("a2")
    assert task["status"]=="error" and task["upstream_submission_started"] is False
    assert len(r.state.sends)==1 and len(r.state.documents)==1


def test_reply_loss_blocks_successors_and_original_request_is_not_resubmitted(runtime):
    r=runtime;r.submit("a1");r.submit("a2");r.state.fail_after_send=True
    r.admission.execute(r.admission.claim_next())
    assert r.read("a1")["error_code"]=="CONVERSATION_OUTCOME_UNKNOWN"
    r.submit("a1");assert r.admission.claim_next() is None;assert len(r.state.sends)==1


def test_image_without_final_terminal_proof_does_not_advance_thread(runtime):
    r=runtime;r.submit("a1");r.submit("a2");r.state.final_pending=True
    r.admission.execute(r.admission.claim_next())
    assert r.read("a1")["status"]=="error";assert not r.read("a1").get("_image_thread_terminal")
    assert r.admission.claim_next() is None


@pytest.mark.parametrize("change",["owner","thread","bytes","index","self"])
def test_edit_must_use_original_owner_thread_and_selected_image_bytes(runtime,change):
    r=runtime;r.submit("a1");run_next(r,"a1")
    args=dict(client_task_id="edit",prompt="edit",model="gpt-image-2",size=None,image_thread_id="product-a",
              edit_source_task_id="a1",images=[(OUTPUT,"source.png","image/png")])
    who=WHO
    if change=="owner":who={**WHO,"id":"foreign"}
    elif change=="thread":args["image_thread_id"]="other"
    elif change=="bytes":args["images"]=[(SOURCE,"wrong.png","image/png")]
    elif change=="index":args["edit_source_index"]=3
    else:args["edit_source_task_id"]="edit"
    count=len(list(r.store.input_dir.iterdir()))
    with pytest.raises(ImageThreadError):r.service.submit_edit(who,**args)
    assert r.read("edit",who) is None
    assert len(list(r.store.input_dir.iterdir()))==count,"invalid source cannot leave a newly accepted private input"
    assert len(r.state.sends)==1


def test_legacy_edit_reuses_exact_original_conversation_without_rewriting_legacy_receipt(runtime):
    r=runtime
    r.submit("legacy",None)
    with r.store.transaction() as db:
        old=r.store.read_receipt(db,"image","happy","legacy")
        old.update(status="success",provider_binding_id="binding-0",provider_account_identity="account-0",
                   client_conversation_id="original-legacy-client",conversation_id="legacy-conversation",
                   request_message_id="legacy-request",parent_message_id="legacy-request-final",data=[{"b64_json":base64.b64encode(OUTPUT).decode()}],
                   upstream_unfinished=False)
        r.store.write_receipt(db,"image","happy","legacy",old)
    r.state.documents["legacy-conversation"]=document("legacy-conversation","legacy-request")
    r.submit("edit","legacy-followup",source="legacy");run_next(r,"edit")
    assert r.read("legacy")==old
    assert r.state.sends[0]["conversation"]=="legacy-conversation"
    with pytest.raises(ImageThreadError,match="ALREADY_LINKED"):
        r.submit("competing","other-legacy-thread",source="legacy")


def test_old_hash_ignores_absent_new_fields_but_new_thread_and_edit_identity_are_immutable(runtime):
    a={"prompt":"x","model":"gpt-image-2","images":[]}
    assert _request_hash("edit",a)==_request_hash("edit",{**a,"image_thread_id":"","edit_source_task_id":"","edit_source_index":0})
    assert _request_hash("edit",a)!=_request_hash("edit",{**a,"image_thread_id":"product"})
    runtime.submit("a1")
    with pytest.raises(ValueError,match="immutable"):runtime.submit("a1","other")


@pytest.mark.parametrize("change",["current","child","sibling","unfinished","missing","wrong-role","wrong-parent","cycle","analysis"])
def test_terminal_parent_is_exact_and_rejects_ambiguous_or_unfinished_turn(change):
    d=document()
    if change=="current":d["current_node"]="other"
    elif change=="child":d["mapping"]["later"]=node("later","user","request-a-final")
    elif change=="sibling":d["mapping"]["parallel"]=node("parallel","tool","request-a")
    elif change=="unfinished":d["mapping"]["request-a-image"]["message"]["status"]="in_progress"
    elif change=="missing":del d["mapping"]["request-a-image"]
    elif change=="wrong-role":d["mapping"]["request-a-final"]["message"]["author"]["role"]="user"
    elif change=="wrong-parent":d["mapping"]["request-a"]["parent"]="different"
    elif change=="cycle":d["mapping"]["request-a-image"]["parent"]="request-a-final"
    else:d["mapping"]["request-a-final"]["message"]["channel"]="analysis"
    with pytest.raises(ImageThreadError):finished_parent(d,"conversation-a","request-a",expected_parent="root")


def test_public_generation_and_multipart_edit_forward_safe_fields_and_reject_raw_cursors(runtime,monkeypatch):
    import api.image_tasks as tasks
    r=runtime
    monkeypatch.setattr(tasks,"image_task_service",r.service)
    monkeypatch.setattr(tasks,"require_identity",lambda *a,**k: WHO)
    app=FastAPI();app.include_router(tasks.create_router());client=TestClient(app)
    headers={"x-workbench-image-client":"1"}
    result=client.post("/api/image-tasks/generations",headers=headers,json={"client_task_id":"first","prompt":"fixture image","image_thread_id":"product-a"})
    assert result.status_code==200,result.text
    assert result.json()["image_thread"]=={"protocol":PROTOCOL,"id":"product-a","previous_task_id":None,"edit_source_task_id":None}
    run_next(r,"first")
    result=client.post("/api/image-tasks/edits",headers=headers,data={"client_task_id":"edit","prompt":"edit","image_thread_id":"product-a","edit_source_task_id":"first","edit_source_index":"0"},files={"image":("base.png",OUTPUT,"image/png")})
    assert result.status_code==200,result.text
    assert result.json()["image_thread"]["previous_task_id"]=="first"
    assert not any(k in result.json() for k in ["provider_binding_id","conversation_id","_image_thread"])
    for field in ["provider_binding_id","provider_account_identity","client_conversation_id","conversation_id","parent_message_id"]:
        bad=client.post("/api/image-tasks/generations",headers=headers,json={"client_task_id":"forged","prompt":"x","image_thread_id":"product-a",field:"forged"})
        assert bad.status_code==400 and r.read("forged") is None


def test_generated_image_recovery_proves_original_final_then_releases_waiting_successor(runtime):
    r=runtime;r.submit("a1");r.submit("a2");r.state.final_pending=True
    r.admission.execute(r.admission.claim_next())
    original=r.read("a1");assert original["status"]=="error" and original["result_file_ids"]
    assert r.admission.claim_next() is None
    doc=r.state.documents[original["conversation_id"]]
    doc["mapping"][original["request_message_id"]+"-final"]["message"].update(status="finished_successfully",end_turn=True)
    r.state.final_pending=False
    r.service._run_resume_poll("happy:a1",original["conversation_id"],5,"",WHO,"edit","gpt-image-2",False,False)
    recovered=r.read("a1")
    assert recovered["status"]=="success",recovered
    assert recovered["_image_thread_terminal"] is True
    assert recovered["request_hash"]==original["request_hash"]
    assert len(r.state.sends)==1,"recovery must only read/download"
    run_next(r,"a2")
    assert r.state.sends[1]["conversation"]==original["conversation_id"]
    assert r.state.sends[1]["parent"]==recovered["parent_message_id"]


@pytest.mark.parametrize("retired", [False, True, "legacy-unrecoverable"])
def test_generated_tool_leaf_recovery_downloads_original_and_releases_successor(runtime, retired):
    r = runtime
    r.state.tool_leaf = True
    r.state.fail_after_result = True
    r.submit("a1"); r.submit("a2")
    r.admission.execute(r.admission.claim_next())
    original = r.read("a1")
    assert original["status"] == "error" and original["result_file_ids"]
    assert original["upstream_outcome"] == "generated"
    assert r.admission.claim_next() is None
    r.state.fail_after_result = False
    if retired:
        with r.store.transaction() as db:
            saved = r.store.read_receipt(db, "image", "happy", "a1")
            saved.update(_attempt_finished_at=1000, _turn_reserved=False, next_poll_at=0)
            if retired == "legacy-unrecoverable":
                # Old exhausted reads discarded phase/outcome even when IDs
                # were retained. Their exact-original downloader still applies.
                saved.update(error_code="RESULT_UNRECOVERABLE", upstream_outcome="unknown", recovery_phase="")
            r.store.write_receipt(db, "image", "happy", "a1", saved)
    restarted = ImageTaskService(r.root / "images.json", store=TaskStore(r.store.path), admission=r.admission)
    if retired:
        # Exercise the public recovery guard too: ending an empty attempt must
        # never disable download of an already discovered original image.
        import time
        if retired == "legacy-unrecoverable":
            r.admission.recoveries["image"] = lambda owner, rid: restarted.resume_poll(WHO, rid, 5)
            r.admission.recover_one()
        else:
            restarted.resume_poll(WHO, "a1", 5)
        deadline = time.monotonic() + 3
        while r.read("a1")["status"] == "running" and time.monotonic() < deadline:
            time.sleep(.01)
    else:
        restarted._run_resume_poll("happy:a1", original["conversation_id"], 5, "", WHO,
            "edit", "gpt-image-2", False, False)
    recovered = r.read("a1")
    assert recovered["status"] == "success", (recovered.get("error_code"), recovered.get("recovery_error_code"), recovered.get("error"))
    assert recovered["_image_thread_terminal"] is True
    assert recovered["result_file_ids"] == original["result_file_ids"]
    assert recovered["request_hash"] == original["request_hash"]
    assert recovered["parent_message_id"] == original["request_message_id"] + "-image"
    assert len(r.state.sends) == 1, "only original result lookup and download may recover"
    run_next(r, "a2")
    assert len(r.state.sends) == 2
    assert r.state.sends[1]["conversation"] == original["conversation_id"]
    assert r.state.sends[1]["parent"] == recovered["parent_message_id"]


def test_first_recovery_read_discovers_result_ids_and_finishes_without_second_poll(runtime):
    r = runtime
    r.state.tool_leaf = True
    r.state.fail_after_send = True
    r.submit("a1"); r.submit("a2")
    r.admission.execute(r.admission.claim_next())
    original = r.read("a1")
    assert original["status"] == "error" and not original.get("result_file_ids")
    assert r.admission.claim_next() is None
    r.state.fail_after_send = False
    restarted = ImageTaskService(r.root / "images.json", store=TaskStore(r.store.path), admission=r.admission)
    restarted._run_resume_poll("happy:a1", original["conversation_id"], 5, "", WHO,
        "edit", "gpt-image-2", False, False)
    recovered = r.read("a1")
    assert recovered["status"] == "success", (recovered.get("recovery_error_code"), recovered.get("error"))
    assert recovered["result_file_ids"] and recovered["_image_thread_terminal"] is True
    assert recovered["request_hash"] == original["request_hash"]
    assert r.state.polls == [(original["conversation_id"], original["request_message_id"])]
    assert len(r.state.sends) == 1, "original result read/download cannot send another generation"
    run_next(r, "a2")
    assert r.state.sends[1]["parent"] == recovered["parent_message_id"]


def test_first_recovery_read_rejects_changed_bound_account_before_poll(runtime):
    r = runtime
    r.state.tool_leaf = True
    r.state.fail_after_send = True
    r.submit("a1"); r.submit("a2")
    r.admission.execute(r.admission.claim_next())
    original = r.read("a1")
    r.account_stub.get_bound_account_identity = lambda _binding: "another-account"
    restarted = ImageTaskService(r.root / "images.json", store=TaskStore(r.store.path), admission=r.admission)
    restarted._run_resume_poll("happy:a1", original["conversation_id"], 5, "", WHO,
        "edit", "gpt-image-2", False, False)
    after = r.read("a1")
    assert after["status"] == "error" and after["recovery_error_code"] == "RECOVERY_AUTH_REQUIRED"
    assert after["request_hash"] == original["request_hash"]
    assert not after.get("result_file_ids") and not r.state.polls
    assert len(r.state.sends) == 1 and r.admission.claim_next() is None


def test_completed_tool_leaf_continues_and_archives_same_image_thread(runtime):
    r = runtime
    r.state.tool_leaf = True
    r.submit("a1"); r.submit("a2")
    first = run_next(r, "a1")
    second = run_next(r, "a2")
    assert r.state.sends[1]["parent"] == first["parent_message_id"]
    assert second["conversation_id"] == first["conversation_id"]
    assert r.service.archive_thread(WHO, "a2")["archived"] is True
    assert r.state.archive_actions == [(first["conversation_id"], True)]


def test_late_image_terminal_tail_archives_and_reworks_without_changing_source(runtime):
    r = runtime
    r.state.tool_leaf = True
    r.submit("a1")
    first = run_next(r, "a1")
    doc = r.state.documents[first['conversation_id']]
    old = first['parent_message_id']
    # This fixture's stream reports the same asset in both namespaces.
    doc['mapping'][old]['message']['content']['parts'].append({
        'content_type':'image_asset_pointer', 'asset_pointer':'sediment://'+first['result_sediment_ids'][0]})
    doc['mapping']['late-note'] = node('late-note', 'tool', old)
    doc['mapping']['late-final'] = node('late-final', 'assistant', 'late-note', end=True)
    doc['current_node'] = 'late-final'
    assert r.service.archive_thread(WHO, 'a1')['archived'] is True
    after = r.read('a1')
    assert after['parent_message_id'] == old
    assert after['data'] == first['data'] and after['request_hash'] == first['request_hash']
    restarted = ImageTaskService(r.root / 'images.json', store=TaskStore(r.store.path), admission=r.admission)
    assert restarted.restore_thread(WHO, 'a1')['archived'] is False
    r.submit('a2', source='a1')
    second = run_next(r, 'a2')
    assert r.state.sends[1]['parent'] == 'late-final'
    assert second['conversation_id'] == first['conversation_id']
    assert len(r.state.sends) == 2
    assert r.read('a1')['parent_message_id'] == old


@pytest.mark.parametrize('change', ['new-user', 'sibling', 'new-asset', 'repeated-asset-tail', 'missing-old', 'unfinished', 'wrong-cursor', 'non-final', 'request-parent', 'namespace'])
def test_archive_tail_advance_rejects_changed_original(change):
    from services.image_thread import archive_parent
    doc, asset = tool_document()
    old = doc['current_node']
    doc['mapping']['late-final'] = node('late-final', 'assistant', old, end=True)
    doc['current_node'] = 'late-final'
    assert archive_parent(doc, 'conversation-a', 'request-a', old, [asset]) == 'late-final'
    if change == 'new-user':
        doc['mapping']['late-final']['message']['author']['role'] = 'user'
    elif change == 'sibling':
        doc['mapping']['sibling'] = node('sibling', 'assistant', old, end=True)
    elif change == 'new-asset':
        doc['mapping']['late-final']['message']['content'] = {'content_type':'multimodal_text', 'parts':[
            {'content_type':'image_asset_pointer', 'asset_pointer':'file-service://file_00000000'+'f'*24}]}
    elif change == 'repeated-asset-tail':
        doc['mapping']['late-final']['message']['content'] = copy.deepcopy(doc['mapping'][old]['message']['content'])
    elif change == 'missing-old':
        del doc['mapping'][old]
    elif change == 'unfinished':
        doc['mapping']['late-final']['message']['status'] = 'in_progress'
    elif change == 'wrong-cursor':
        old = 'request-a-code'
    elif change == 'request-parent':
        doc['mapping']['request-a']['parent'] = 'foreign-parent'
    elif change == 'namespace':
        doc['mapping'][old]['message']['content']['parts'][0]['asset_pointer'] = 'sediment://'+asset
    else:
        doc['mapping']['late-final']['message']['end_turn'] = False
    with pytest.raises(ImageThreadError):
        archive_parent(doc, 'conversation-a', 'request-a', old, [asset], expected_parent='root',
                       expected_file_ids=[asset], expected_sediment_ids=[])


def test_archive_late_tail_does_not_invalidate_edit_accepted_during_readback(runtime, monkeypatch):
    from services.image_thread import source_fingerprint
    r = runtime
    r.state.tool_leaf = True
    r.submit('a1')
    first = run_next(r, 'a1')
    doc = r.state.documents[first['conversation_id']]
    old = first['parent_message_id']
    doc['mapping'][old]['message']['content']['parts'].append({
        'content_type':'image_asset_pointer', 'asset_pointer':'sediment://'+first['result_sediment_ids'][0]})
    doc['mapping']['late-final'] = node('late-final', 'assistant', old, end=True)
    doc['current_node'] = 'late-final'
    original = conversation.OpenAIBackendAPI.set_conversation_archived
    def accept_during_readback(self, cid, parent, archived, **kwargs):
        result = original(self, cid, parent, archived, **kwargs)
        if archived:
            r.submit('a2', source='a1')
        return result
    monkeypatch.setattr(conversation.OpenAIBackendAPI, 'set_conversation_archived', accept_during_readback)
    r.service.archive_thread(WHO, 'a1')
    assert source_fingerprint(r.read('a1')) == source_fingerprint(first)
    assert run_next(r, 'a2')['status'] == 'success'
    assert r.state.sends[1]['parent'] == 'late-final'


def test_mismatched_tool_leaf_keeps_original_generated_result_unclaimed(runtime):
    r = runtime
    r.state.tool_leaf = True
    r.state.fail_after_result = True
    r.submit("a1"); r.submit("a2")
    r.admission.execute(r.admission.claim_next())
    original = r.read("a1")
    leaf = r.state.documents[original["conversation_id"]]["mapping"][original["request_message_id"] + "-image"]["message"]
    leaf["content"]["parts"][0]["asset_pointer"] = "file-service://file_00000000" + "f" * 24
    r.service._run_resume_poll("happy:a1", original["conversation_id"], 5, "", WHO,
        "edit", "gpt-image-2", False, False)
    after = r.read("a1")
    assert after["status"] == "error"
    assert after["recovery_error_code"] == "RECOVERY_THREAD_UNCONFIRMED"
    assert after["result_file_ids"] == original["result_file_ids"]
    assert len(r.state.sends) == 1
    assert r.admission.claim_next() is None


@pytest.mark.parametrize("change", ["missing-ids", "wrong-ids", "unfinished", "drift", "sibling", "wrong-role", "no-pointer"])
def test_tool_leaf_requires_exact_saved_asset_and_current_completed_branch(change):
    doc, result_id = tool_document()
    ids = [result_id]
    leaf = doc["mapping"]["request-a-image"]["message"]
    if change == "missing-ids": ids = []
    elif change == "wrong-ids": ids = ["file_00000000" + "f" * 24]
    elif change == "unfinished": leaf["status"] = "in_progress"
    elif change == "drift": doc["current_node"] = "request-a-code"
    elif change == "sibling": doc["mapping"]["other"] = node("other", "tool", "request-a-code")
    elif change == "wrong-role": doc["mapping"]["request-a-code"]["message"]["author"]["role"] = "tool"
    else: leaf["content"] = {"content_type": "text", "parts": [result_id]}
    with pytest.raises(ImageThreadError):
        finished_parent(doc, "conversation-a", "request-a", expected_parent="root", expected_result_ids=ids)


def test_tool_leaf_accepts_exact_saved_asset():
    doc, result_id = tool_document()
    assert finished_parent(doc, "conversation-a", "request-a", expected_parent="root",
        expected_result_ids=[result_id, result_id]) == "request-a-image"


@pytest.mark.parametrize("change", [None, "missing-prior", "missing-assets", "wrong-assets", "extra-assets",
    "parent-still-present", "sibling", "later-user", "unfinished", "not-recap", "terminal-recap",
    "recap-channel", "recap-recipient", "middle-assistant", "recap-sibling", "recap-same-asset", "no-tool",
    "current-assets", "current-drift", "current-unfinished"])
def test_pruned_previous_final_requires_exact_completed_predecessor_and_recap(change):
    doc, asset = tool_document()
    doc["mapping"].pop("request-a-code")
    doc["mapping"]["request-a-image"]["parent"] = "request-a"
    recap = node("recap", "assistant", "request-a-image")
    recap["message"]["recipient"] = "all"
    recap["message"]["content"] = {"content_type": "reasoning_recap", "content": "fixture"}
    doc["mapping"]["recap"] = recap
    child = document("conversation-a", "edit", "recap")
    doc["mapping"].update(child["mapping"])
    doc["current_node"] = child["current_node"]
    prior, assets = "request-a", [asset]
    if change == "missing-prior": prior = None
    elif change == "missing-assets": assets = []
    elif change == "wrong-assets": assets = ["file_00000000" + "f" * 24]
    elif change == "extra-assets": assets += ["file_00000000" + "f" * 24]
    elif change == "parent-still-present": doc["mapping"]["removed-final"] = node("removed-final", "assistant", "recap", end=True)
    elif change == "sibling": doc["mapping"]["sibling"] = node("sibling", "assistant", "request-a")
    elif change == "later-user": recap["message"]["author"]["role"] = "user"
    elif change == "unfinished": doc["mapping"]["request-a-image"]["message"]["status"] = "in_progress"
    elif change == "not-recap": recap["message"]["content"]["content_type"] = "text"
    elif change == "terminal-recap": recap["message"]["end_turn"] = True
    elif change == "recap-channel": recap["message"]["channel"] = "final"
    elif change == "recap-recipient": recap["message"]["recipient"] = "image_gen"
    elif change == "middle-assistant": doc["mapping"]["request-a-image"]["message"]["author"]["role"] = "assistant"
    elif change == "recap-sibling": doc["mapping"]["sibling"] = node("sibling", "user", "recap")
    elif change == "recap-same-asset": recap["message"]["metadata"] = {"asset_pointer": "file-service://" + asset}
    elif change == "no-tool":
        doc["mapping"].pop("request-a-image")
        recap["parent"] = "request-a"
    elif change == "current-assets": doc["mapping"]["edit-image"]["message"]["content"] = {
        "content_type": "multimodal_text", "parts": [{"content_type": "image_asset_pointer",
        "asset_pointer": "file-service://file_00000000" + "f" * 24}]}
    elif change == "current-drift": doc["current_node"] = "recap"
    elif change == "current-unfinished": doc["mapping"]["edit-final"]["message"]["status"] = "in_progress"
    def confirm():
        return finished_parent(doc, "conversation-a", "edit", expected_parent="removed-final",
            expected_result_ids=["file_00000000" + "e" * 24],
            predecessor_request_message_id=prior, predecessor_result_ids=assets)
    if change is None:
        assert confirm() == "edit-final"
    else:
        with pytest.raises(ImageThreadError): confirm()


@pytest.mark.parametrize("recover", [False, True])
def test_image_edit_keeps_original_turn_when_upstream_prunes_previous_final(runtime, recover):
    r = runtime
    r.state.pruned_recap = True
    r.submit("source")
    first = run_next(r, "source")
    r.state.fail_after_result = recover
    r.submit("edited", source="source")
    ctx = r.admission.claim_next()
    r.admission.execute(ctx)
    original = r.read("edited")
    if recover:
        assert original["status"] == "error"
        restored = ImageTaskService(r.root / "images.json", store=TaskStore(r.store.path), admission=r.admission)
        restored._run_resume_poll("happy:edited", original["conversation_id"], 5, "", WHO,
            "edit", "gpt-image-2", False, False)
    final = r.read("edited")
    assert final["status"] == "success", final
    assert final["conversation_id"] == first["conversation_id"]
    assert final["_image_thread_request_parent"] == first["parent_message_id"]
    assert final["request_message_id"] == original["request_message_id"]
    assert final["data"] and final["_image_thread_terminal"]
    assert len(r.state.sends) == 2


@pytest.mark.parametrize("change", [None, "missing-asset", "extra-asset", "unfinished", "sibling", "drift"])
def test_consecutive_image_tool_results_require_exact_original_branch_and_all_saved_assets(change):
    doc, first_id = tool_document()
    second_id = "file_00000000" + "b" * 24
    second = node("request-a-image-2", "tool", "request-a-image")
    second["message"]["content"] = {"content_type": "multimodal_text", "parts": [
        {"content_type": "image_asset_pointer", "asset_pointer": "file-service://" + second_id}]}
    doc["mapping"]["request-a-image-2"] = second
    doc["current_node"] = "request-a-image-2"
    expected = [first_id, first_id, second_id, second_id]
    if change == "missing-asset": expected = [first_id]
    elif change == "extra-asset": expected.append("file_00000000" + "c" * 24)
    elif change == "unfinished": second["message"]["status"] = "in_progress"
    elif change == "sibling": doc["mapping"]["other"] = node("other", "tool", "request-a-image")
    elif change == "drift": doc["current_node"] = "request-a-image"
    if change is None:
        assert finished_parent(doc, "conversation-a", "request-a", expected_parent="root",
            expected_result_ids=expected) == "request-a-image-2"
    else:
        with pytest.raises(ImageThreadError):
            finished_parent(doc, "conversation-a", "request-a", expected_parent="root",
                expected_result_ids=expected)


def test_accepted_edit_does_not_dispatch_if_source_record_is_later_replaced(runtime):
    r=runtime;r.submit("a1");run_next(r,"a1");r.submit("edit",source="a1")
    with r.store.transaction() as db:
        original=r.store.read_receipt(db,"image","happy","a1")
        original["data"]=[{"b64_json":base64.b64encode(SOURCE).decode()}]
        r.store.write_receipt(db,"image","happy","a1",original)
    assert r.admission.claim_next() is None
    assert len(r.state.sends)==1


def test_send_guard_rechecks_source_after_claim_and_before_the_network(runtime):
    r=runtime;r.submit("a1");run_next(r,"a1");r.submit("edit",source="a1")
    ctx=r.admission.claim_next();assert ctx.request_id=="edit"
    with r.store.transaction() as db:
        original=r.store.read_receipt(db,"image","happy","a1");original["_recovery_suppressed"]={"reason":"explicit stop"}
        r.store.write_receipt(db,"image","happy","a1",original)
    from services.request_context import AdmissionLost
    with pytest.raises(AdmissionLost):ctx.before_send()
    assert len(r.state.sends)==1

@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("ingress", ["public", "company", "internal"])
def test_thread_capability_matches_ingress_without_changing_private_catalog(runtime, monkeypatch, enabled, ingress):
    from api import ai, company_requests
    import services.image_task_service as image_module
    import services.public_chat_service as public_chat_module
    r = runtime
    catalog = {"object": "list", "data": []}
    monkeypatch.setattr(ai, "require_identity", lambda *a, **k: WHO)
    monkeypatch.setattr(ai.openai_v1_models, "list_models", lambda: catalog)
    # An intentionally empty known catalog is different from discovery being
    # unavailable. Keep this contract test independent of global cache state.
    monkeypatch.setattr(public_chat_module.model_catalog_service, "catalog_is_unknown", lambda: False)
    monkeypatch.setattr(image_module, "image_task_service", r.service)
    if not enabled:
        r.service.admission = None
    app = FastAPI()
    app.include_router(ai.create_router())
    headers, route = {}, "/v1/models"
    if ingress == "public":
        headers["x-workbench-image-client"] = "1"
    elif ingress == "company":
        monkeypatch.setattr(company_requests, "require_admin", lambda *_: None)
        app.middleware("http")(company_requests.company_request_boundary)
        headers = {"x-workbench-company-org": "fixture", "x-workbench-company-user": "owner",
                   "x-workbench-expected-user": "owner",
                   "x-workbench-company-connector": "1f084f01-d4b2-4080-8bce-b926f31cc454"}
        route = company_requests.PREFIX + route
    with TestClient(app) as client:
        response = client.get(route, headers=headers)
    assert response.status_code == 200, response.text
    if ingress == "internal":
        assert response.json() == catalog
    else:
        assert response.json()["service_capabilities"] == {"image_thread": PROTOCOL if enabled else None}
    assert catalog == {"object": "list", "data": []}, "discovery may not mutate cached catalog"


@pytest.mark.parametrize("previous,reason", [
    (None, "IMAGE_THREAD_PREVIOUS_MISSING"),
    ({"status": "error", "upstream_outcome": "rejected"}, "IMAGE_THREAD_PREVIOUS_FAILED"),
    ({"status": "unknown"}, "IMAGE_THREAD_PREVIOUS_UNKNOWN"),
    ({"status": "error", "upstream_unfinished": True}, "IMAGE_THREAD_PREVIOUS_UNKNOWN"),
    ({"status": "success", "_recovery_suppressed": True}, "IMAGE_THREAD_PREVIOUS_RECOVERY_STOPPED"),
    ({"status": "queued"}, "IMAGE_THREAD_PREVIOUS_PENDING"),
    ({"status": "success"}, "IMAGE_THREAD_PREVIOUS_UNCONFIRMED"),
])
def test_predecessor_diagnostics_are_specific_and_read_only(previous, reason):
    from services.image_thread import predecessor_state
    task = {"_image_thread": {"protocol": PROTOCOL, "id": "same-product", "previous_task_id": "original"}}
    owned = {} if previous is None else {"original": previous}
    before = copy.deepcopy(owned)
    assert predecessor_state(task, owned) == ({}, reason)
    assert owned == before

@pytest.mark.parametrize('case', ['success', 'late_original_success', 'selected_source_changed', 'drift', 'late_result', 'missing_root'])
def test_bounded_image_retry_retains_original_account_thread_and_send_edge(runtime, case):
    import time
    from services.generation_completion import GenerationCompletionService, retry_cursor
    from services.text_task_service import TextTaskService
    from services.work_lifecycle import WorkLifecycleService
    r = runtime
    r.state.fail_after_send = True
    r.submit('empty-original')
    r.admission.execute(r.admission.claim_next())
    original = r.read('empty-original')
    assert original['status'] == 'error' and len(r.state.sends) == 1
    cid, mid = original['conversation_id'], original['request_message_id']
    doc = r.state.documents[cid]
    doc['is_archived'] = False
    for key, value in doc['mapping'].items():
        if key != mid:
            value['message']['content']['parts'] = []
            value['message'].update(status='in_progress', end_turn=None)
    now = time.time()
    r.admission.clock = lambda: now
    with r.store.transaction() as db:
        saved = r.store.read_receipt(db, 'image', WHO['id'], 'empty-original')
        saved.update(_completion_read_at=now, _retry_cursor=retry_cursor(doc, original, kind='image'),
                     _execution_timeline=[{'stage': 'send_call_started', 'at': now-1300}],
                     _executing=False, _claim_id=None, _claim_until=0)
        r.store.write_receipt(db, 'image', WHO['id'], 'empty-original', saved)
    text = TextTaskService(r.store.path, admission=r.admission, clock=r.admission.clock)
    lifecycle = WorkLifecycleService(text, r.service, clock=r.admission.clock)
    completion = GenerationCompletionService(text, r.service, lifecycle, clock=r.admission.clock)
    r.admission.generation_completion = completion
    result = completion.start('image', WHO, 'empty-original', allow_unconfirmed_retry=True)
    assert result.get('replacement_id'), result
    child_id = result['replacement_id']; child = r.read(child_id)
    for key in ('provider_account_identity','provider_binding_id','conversation_id','client_conversation_id','_work_key'):
        assert child[key] == original[key]
    assert child['_image_thread']['id'] == original['_image_thread']['id']
    assert child['_image_thread']['previous_task_id'] == 'empty-original'
    assert r.read('empty-original')['_attempt_finished_at']
    assert not r.read('empty-original')['_turn_reserved']
    ctx = r.admission.claim_next()
    assert ctx and ctx.request_id == child_id
    r.state.fail_after_send = False
    if case == 'drift': r.state.drift = True
    if case in {'late_result', 'missing_root'}:
        with r.store.transaction() as db:
            root = r.store.read_receipt(db, 'image', WHO['id'], 'empty-original')
            if case == 'late_result':
                root.update(result_file_ids=['already-generated'], upstream_outcome='generated', recovery_phase='download_image_result')
            else:
                # Identity loss has the same fail-closed effect as a missing row,
                # without bypassing the store's retention policy in the fixture.
                root['_completion'] = {}
            r.store.write_receipt(db, 'image', WHO['id'], 'empty-original', root)
    r.admission.execute(ctx)
    if case not in {'success', 'late_original_success', 'selected_source_changed'}:
        assert len(r.state.sends) == 1
        assert r.read(child_id)['upstream_outcome'] in {'not_sent', 'not_submitted'}
        return
    assert len(r.state.sends) == 2
    assert r.state.sends[-1]['account'] == r.state.sends[0]['account']
    assert r.state.sends[-1]['conversation'] == cid
    assert r.state.sends[-1]['images'] == r.state.sends[0]['images']
    assert r.read(child_id)['status'] == 'success'
    from services.image_thread import saved_image_bytes
    assert saved_image_bytes(r.read(child_id)) == OUTPUT
    result = completion.read('image', WHO, 'empty-original')
    assert result['selected_id'] == child_id
    assert completion.complete('image', WHO, 'empty-original', child_id)['state'] == 'completed'
    assert r.read('empty-original')['status'] == 'error'
    if case == 'late_original_success':
        with r.store.transaction() as db:
            late = r.store.read_receipt(db, 'image', WHO['id'], 'empty-original')
            late.update(status='success', data=[{'b64_json': base64.b64encode(SOURCE).decode()}])
            r.store.write_receipt(db, 'image', WHO['id'], 'empty-original', late)
    # The lifecycle worker archives the selected physical request, while the
    # public caller may still name its original logical task. Both use the
    # successful child's exact turn; the old UNKNOWN is retained.
    assert r.service.archive_thread(WHO, child_id)['archived'] is True
    assert r.service.restore_thread(WHO, 'empty-original')['archived'] is False
    assert r.state.archive_actions[-2:] == [(cid, True), (cid, False)]
    assert lifecycle.process_one()
    completion.rework('image', WHO, 'empty-original', child_id)
    assert lifecycle.process_one()
    # Next edit accepts bytes selected for the original ID without requiring
    # callers to replace their business IDs with internal retry IDs.
    r.submit('after-recovery', source='empty-original')
    with pytest.raises(ImageThreadError, match='IMAGE_THREAD_NOT_TERMINAL'):
        r.service.archive_thread(WHO, child_id)
    next_context = r.admission.claim_next()
    assert next_context.request_id == 'after-recovery'
    if case == 'selected_source_changed':
        with r.store.transaction() as db:
            changed = r.store.read_receipt(db, 'image', WHO['id'], child_id)
            changed['data'] = [{'b64_json': base64.b64encode(SOURCE).decode()}]
            r.store.write_receipt(db, 'image', WHO['id'], child_id, changed)
        from services.request_context import AdmissionLost
        with pytest.raises(AdmissionLost, match='predecessor changed'):
            next_context.before_send()
        assert len(r.state.sends) == 2
        assert r.read('after-recovery')['_submission_started'] is False
        return
    r.admission.execute(next_context)
    assert r.read('after-recovery')['status'] == 'success'
    assert r.read('after-recovery')['conversation_id'] == cid
    assert r.read('empty-original')['status'] == ('success' if case == 'late_original_success' else 'error')


@pytest.mark.parametrize('pruned', [False, True])
@pytest.mark.parametrize('change', [None, 'archive', 'branch', 'later_user', 'assets', 'binding',
                                   'source', 'active', 'head', 'original_present'])
def test_absent_edit_cursor_requires_exact_saved_completed_predecessor(runtime, pruned, change):
    from services.generation_completion import retry_cursor
    r = runtime
    r.state.pruned_recap = True
    r.submit('source'); previous = run_next(r, 'source')
    r.submit('absent-edit', source='source')
    ctx = r.admission.claim_next()
    task = r.read('absent-edit')
    task['request_message_id'] = 'original-absent-message'
    task['_image_thread_request_parent'] = previous['parent_message_id']
    doc = copy.deepcopy(r.state.documents[previous['conversation_id']])
    doc['is_archived'] = False
    if pruned:
        terminal = doc['mapping'].pop(doc['current_node'])
        doc['current_node'] = terminal['parent']
    if change == 'archive': doc['is_archived'] = True
    elif change == 'branch': doc['mapping']['sibling'] = node('sibling', 'user', previous['request_message_id'])
    elif change == 'later_user':
        doc['mapping']['later'] = node('later', 'user', doc['current_node']); doc['current_node'] = 'later'
    elif change == 'assets': task['_image_thread_predecessor_result_ids'] = ['file-wrong']
    elif change == 'binding': previous['provider_binding_id'] = 'other'
    elif change == 'source': previous['data'] = []
    elif change == 'active': doc['mapping'][doc['current_node']]['message']['status'] = 'in_progress'
    elif change == 'head': doc['current_node'] = previous['request_message_id']
    elif change == 'original_present': doc['mapping'][task['request_message_id']] = node(task['request_message_id'], 'user', doc['current_node'])
    proof = retry_cursor(doc, task, kind='image', predecessor=previous)
    if change is None:
        assert proof['source'] == 'absent_image_thread_request'
        assert proof['retry_parent_message_id'] == doc['current_node']
    else:
        assert proof is None


@pytest.mark.parametrize('case', ['success', 'active_tasks', 'read_failure', 'two_reads', 'presend_drift', 'ended_recheck', 'ended_before_reads'])
def test_absent_edit_qualified_reads_then_one_same_session_completion(runtime, monkeypatch, case):
    import time
    from services.generation_completion import GenerationCompletionService
    from services.text_task_service import TextTaskService
    from services.work_lifecycle import WorkLifecycleService
    r = runtime
    r.state.pruned_recap = True
    r.submit('source'); previous = run_next(r, 'source')
    r.submit('absent-edit', source='source')
    stream = conversation.stream_image_outputs
    def false_start(backend, req, *args):
        current_request.get().before_send()
        backend.image_submission_started = True
        req.progress_callback.record_submission_started()
        raise TimeoutError('old pacing deadline before actual transport')
        yield
    monkeypatch.setattr(conversation, 'stream_image_outputs', false_start)
    r.admission.execute(r.admission.claim_next())
    original = r.read('absent-edit')
    assert original['status'] == 'error' and len(r.state.sends) == 1
    doc = r.state.documents[previous['conversation_id']]
    doc['is_archived'] = False
    assert original['request_message_id'] not in doc['mapping']
    now = time.time()
    r.admission.clock = lambda: now
    with r.store.transaction() as db:
        saved = r.store.read_receipt(db, 'image', WHO['id'], 'absent-edit')
        saved.update(_execution_timeline=[{'stage': 'send_call_started', 'at': now-1300}],
                     active_attempt_deadline_at=now-1000, _executing=False, _claim_until=0)
        r.store.write_receipt(db, 'image', WHO['id'], 'absent-edit', saved)
    Backend = conversation.OpenAIBackendAPI
    def tasks(self, **kwargs):
        assert kwargs['strict_schema'] is True
        if case == 'read_failure': raise ConnectionError('task query unavailable')
        return [{'status': 'running'}] if case == 'active_tasks' else []
    monkeypatch.setattr(Backend, '_query_backend_tasks', tasks, raising=False)
    monkeypatch.setattr(Backend, '_poll_image_results', lambda *a, **k: ([], []))
    for _ in range(0 if case == 'ended_before_reads' else 2 if case == 'two_reads' else 3):
        r.service._run_resume_poll('happy:absent-edit', previous['conversation_id'], 5, '', WHO,
                                  'edit', 'gpt-image-2', True, True)
    root = r.read('absent-edit')
    if case in {'active_tasks', 'read_failure'}:
        assert not root.get('_retry_cursor') and root.get('recovery_no_result_reads', 0) == 0
        return
    text = TextTaskService(r.store.path, admission=r.admission, clock=r.admission.clock)
    lifecycle = WorkLifecycleService(text, r.service, clock=r.admission.clock)
    completion = GenerationCompletionService(text, r.service, lifecycle, clock=r.admission.clock)
    r.admission.generation_completion = completion
    if case == 'two_reads':
        from services.generation_completion import CompletionError
        with r.store.connect() as db, pytest.raises(CompletionError, match='COMPLETION_ORIGINAL_CURSOR_UNCONFIRMED'):
            completion._prepare(db, 'image', WHO['id'], 'absent-edit', root, 'never-send')
        return
    if case != 'ended_before_reads':
        assert root['error_code'] == 'RESULT_UNRECOVERABLE' and root['recovery_no_result_reads'] == 3
        assert root['_retry_cursor']['source'] == 'absent_image_thread_request'
    if case in {'ended_recheck', 'ended_before_reads'}:
        import threading
        ended_at = now-10
        monkeypatch.setattr(time, 'time', lambda: now)
        with r.store.transaction() as db:
            ended = r.store.read_receipt(db, 'image', WHO['id'], 'absent-edit')
            ended.update(_attempt_finished_at=ended_at, _attempt_reason='COMPLETION_ORIGINAL_CURSOR_UNCONFIRMED',
                         _retry_cursor=None, _completion_read_at=now-400,
                         _completion={'state': 'needs_attention', 'next_at': None, 'started_at': now-1300,
                                      'allow_unconfirmed_retry': True, 'max_extra_requests': 1})
            r.store.write_receipt(db, 'image', WHO['id'], 'absent-edit', ended)
        before = len(r.state.reads)
        r.service.resume_poll(WHO, 'absent-edit', allow_unrecoverable_retry=True)
        assert len(r.state.reads) == before, 'ordinary polling cannot reopen an ended attempt'
        original_thread, threads = threading.Thread, []
        def recording_thread(*args, **kwargs):
            thread = original_thread(*args, **kwargs)
            if kwargs.get('name', '').startswith('image-resume-'):
                threads.append(thread)
                start_thread = thread.start
                def start_and_join():
                    start_thread(); thread.join(5)
                    assert not thread.is_alive()
                thread.start = start_and_join
            return thread
        monkeypatch.setattr(threading, 'Thread', recording_thread)
        for _ in range(3 if case == 'ended_before_reads' else 1):
            completion.start('image', WHO, 'absent-edit', allow_unconfirmed_retry=True)
            now += 31
        assert len(threads) == (3 if case == 'ended_before_reads' else 1)
        reread = r.read('absent-edit')
        assert reread['_retry_cursor']['source'] == 'absent_image_thread_request'
        assert reread['_attempt_reason'] in {'COMPLETION_ORIGINAL_CURSOR_UNCONFIRMED', 'ATTEMPT_REPLACED_SAME_CONVERSATION'}
        completion.advance('image', WHO['id'], 'absent-edit')
    result = completion.start('image', WHO, 'absent-edit', allow_unconfirmed_retry=True)
    child_id = result['replacement_id']
    assert completion.start('image', WHO, 'absent-edit', allow_unconfirmed_retry=True)['replacement_id'] == child_id
    child = r.read(child_id)
    for key in ('provider_account_identity', 'provider_binding_id', 'conversation_id', 'client_conversation_id', '_work_key'):
        assert child[key] == root[key]
    monkeypatch.setattr(conversation, 'stream_image_outputs', stream)
    if case == 'presend_drift': r.state.drift = True
    r.admission.execute(r.admission.claim_next())
    if case == 'presend_drift':
        assert len(r.state.sends) == 1 and r.read(child_id)['upstream_outcome'] in {'not_sent', 'not_submitted'}
    else:
        assert len(r.state.sends) == 2 and r.read(child_id)['status'] == 'success'
        assert completion.read('image', WHO, 'absent-edit')['selected_id'] == child_id
        assert completion.complete('image', WHO, 'absent-edit', child_id)['state'] == 'completed'
        assert r.read('absent-edit')['status'] == 'error'


def test_archive_uses_same_fresh_precheck_for_validator_and_keeps_readback():
    from types import SimpleNamespace
    backend = object.__new__(RealOpenAIBackendAPI)
    backend.base_url = 'https://fixture.invalid'
    backend._headers = lambda *a, **k: {}
    reads, patches, validated = [], [], []
    document = {'current_node': 'parent', 'mapping': {'parent': {}}, 'is_archived': False}
    def get(cid):
        reads.append(cid)
        return {**document}
    def patch(*a, **k):
        patches.append(k)
        document['is_archived'] = True
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {})
    backend._get_conversation = get
    backend.session = SimpleNamespace(patch=patch)
    def reject(doc):
        assert doc['is_archived'] is False
        raise ValueError('terminal result changed')
    with pytest.raises(ValueError, match='terminal result changed'):
        backend.set_conversation_archived('original', 'parent', True, validate_document=reject)
    assert len(reads) == 1 and not patches
    reads.clear()
    backend.set_conversation_archived('original', 'parent', True, validate_document=lambda doc: validated.append(doc))
    assert len(reads) == 2 and len(patches) == 1 and len(validated) == 1
    assert validated[0]['is_archived'] is False


def test_archive_resolved_cursor_still_requires_exact_post_patch_readback():
    from services.openai_backend_api import ConversationArchiveCursorMismatch
    backend = object.__new__(RealOpenAIBackendAPI)
    backend.base_url = 'https://fixture.invalid'
    backend._headers = lambda *a, **k: {}
    doc = {'current_node':'new-final','mapping':{'old-tool':{},'new-final':{}},'is_archived':False}
    patches = []
    backend._get_conversation = lambda cid: dict(doc)
    def patch(*a, **kwargs):
        patches.append(kwargs)
        doc.update(is_archived=True, current_node='foreign')
        return SimpleNamespace(status_code=200, raise_for_status=lambda:None)
    backend.session = SimpleNamespace(patch=patch)
    with pytest.raises(ConversationArchiveCursorMismatch):
        backend.set_conversation_archived('original','old-tool',True)
    assert not patches
    with pytest.raises(RuntimeError, match='cursor changed during'):
        backend.set_conversation_archived('original','old-tool',True,validate_document=lambda d:'new-final')
    assert len(patches) == 1
