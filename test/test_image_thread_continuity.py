"""Actual public task store/admission/bound-image path, controlled upstream only."""
import base64
import copy
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


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    rows = [{"access_token": "fixture-token", "account_id": "fixture-upstream",
             "provider_account_identity": "account-0", "type": "Plus", "status": "正常",
             "quota": 999, "source_type": "web", "conversation_binding_ids": ["binding-0"]}]
    (tmp_path / "accounts.json").write_text(json.dumps(rows))
    accounts = AccountService(JSONStorageBackend(tmp_path / "accounts.json"))
    store = TaskStore(tmp_path / "text_tasks.sqlite3")
    admission = PoolAdmission(store, accounts, clock=lambda: 1000.0,
        settings=lambda: {"image_account_concurrency": 4, "chat_account_concurrency": 4, "codex_max_concurrency": 4},
        model_types=lambda _: {"Plus"}, pacing=lambda _a, now: {"next_at": now})
    service = ImageTaskService(tmp_path / "images.json", store=store, admission=admission)
    admission.register("image", lambda ctx, body: service._run_task(ctx.owner+":"+ctx.request_id,
        body["mode"], {**body["payload"], "retain_conversation": True}, body["identity"], body["payload"]["model"]))
    state = SimpleNamespace(sends=[], documents={}, fail_after_send=False, drift=False, final_pending=False, reads=[], naive_reads=0, archive_actions=[])
    account_stub = SimpleNamespace(get_bound_account_identity=lambda _b: "account-0",
        acquire_bound_image_access_token=lambda *a, **k: "fixture-token", get_account=lambda _t: rows[0],
        conversation_binding_lock=lambda *a: nullcontext(), mark_image_result=lambda *a: None,
        release_image_slot=lambda *a: None, get_bound_text_access_token=lambda *a, **k: "fixture-token")
    monkeypatch.setattr(conversation, "account_service", account_stub)
    import services.account_service as account_module
    monkeypatch.setattr(account_module, "account_service", account_stub)
    for mod in (openai_v1_image_edit, openai_v1_image_generations):
        monkeypatch.setattr(mod, "count_text_tokens", lambda *a, **k: 0)
    class Backend:
        def __init__(self, access_token):
            assert access_token == "fixture-token"
            self.image_submission_started = False
        def _get_conversation(self, cid):
            state.reads.append(cid)
            doc = copy.deepcopy(state.documents[cid])
            if state.drift:
                doc["mapping"]["foreign"] = node("foreign", "user", doc["current_node"])
                doc["current_node"] = "foreign"
            return doc
        def get_conversation_parent_message_id(self, cid):
            state.naive_reads += 1
            raise AssertionError("new thread may not accept arbitrary current_node")
        def set_conversation_archived(self, cid, parent, archived):
            assert parent in state.documents[cid]["mapping"]
            state.documents[cid]["is_archived"] = archived
            state.archive_actions.append((cid, archived))
            return {"archived": archived}
        def archive_conversation(self, cid, parent):
            return self.set_conversation_archived(cid, parent, True)
        def resolve_conversation_image_urls(self, cid, files, sediment, **kwargs):
            assert kwargs["request_message_id"] in state.documents[cid]["mapping"]
            return ["https://fixture.invalid/original.png"]
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
        doc = document(cid, rid, req.parent_message_id or "root")
        if old:
            doc["mapping"] = {**old["mapping"], **doc["mapping"]}
        state.documents[cid] = doc
        if state.fail_after_send:
            raise ConnectionError("fixture reply lost after sending")
        callback.record_result_ids(["file-" + rid], [])
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
                           submit=submit, read=read, root=tmp_path)


def run_next(r, expected):
    ctx = r.admission.claim_next()
    assert ctx and ctx.request_id == expected
    r.admission.execute(ctx)
    result = r.read(expected)
    assert result["status"] == "success", result
    return result


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
    r = runtime
    catalog = {"object": "list", "data": []}
    monkeypatch.setattr(ai, "require_identity", lambda *a, **k: WHO)
    monkeypatch.setattr(ai.openai_v1_models, "list_models", lambda: catalog)
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
