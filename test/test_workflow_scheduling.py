"""Durable caller scheduling constraints, controlled accounts and no upstream."""
from datetime import datetime, timezone
from types import SimpleNamespace
import json

import pytest

from services.image_task_service import ImageTaskService
from services.request_context import AdmissionLost, trusted_source
from services.auth_service import AuthService
from services.storage.json_storage import JSONStorageBackend
from services.workflow_scheduling import normalize_scheduling, group
from test.test_image_account_selection import runtime


def at(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def submit(rt, name, *, owner='owner', source='user:one', session='', scheduling=None):
    return rt.tasks.submit_generation({'id':owner, '_trusted_source':source}, client_task_id=name,
        prompt='synthetic', model='gpt-image-2', size=None, client_conversation_id=session,
        scheduling=scheduling)


def row(rt, name, owner='owner'):
    with rt.store.connect() as db:return rt.store.read_receipt(db,'image',owner,name)


def work(rt, saved):
    with rt.store.connect() as db:return rt.store.runtime(db,saved['_work_key'])


@pytest.mark.parametrize('value',[{'unknown':1},{'workflow_concurrency':2},{'workflow_id':'w','workflow_concurrency':True},
    {'workflow_id':'w','workflow_concurrency':0},{'min_send_interval_seconds':float('nan')},
    {'min_send_interval_seconds':-1},{'not_before':'2026-09-30T12:00:00'},
    {'not_before':at(2000),'wait_deadline':at(1000)}])
def test_invalid_scheduling_is_rejected(value):
    with pytest.raises(ValueError,match='SCHEDULING_INVALID'):normalize_scheduling(value)


def test_not_before_is_saved_in_hash_receipt_and_restart(runtime):
    rt=runtime; options={'not_before':at(1050),'wait_deadline':at(1200)}
    response=submit(rt,'original',scheduling=options)
    assert response['scheduling']==options
    assert rt.admission.claim_next() is None
    saved=row(rt,'original')
    assert rt.store.load_input(saved['_input_ref'])['payload']['_scheduling']==options
    with pytest.raises(ValueError,match='immutable'):submit(rt,'original',scheduling={'not_before':at(1040)})
    with pytest.raises(ValueError,match='immutable'):submit(rt,'original')
    rt.tasks=ImageTaskService(rt.root/'images.json',store=rt.store,admission=rt.admission)
    assert submit(rt,'original',scheduling=options)['id']=='original'
    rt.admission.clock=lambda:1051
    assert rt.admission.claim_next().request_id=='original'


def test_message_success_holds_work_slot_until_explicit_release(runtime):
    rt=runtime; options={'workflow_id':'catalog','workflow_concurrency':1}
    submit(rt,'first',scheduling=options);submit(rt,'second',scheduling=options)
    first=rt.admission.claim_next();assert first.request_id=='first'
    rt.admission.execute(first)
    assert row(rt,'first')['status']=='success'
    assert work(rt,row(rt,'first'))['slot_held']
    assert rt.admission.claim_next() is None
    with rt.store.transaction() as db:
        saved=rt.store.runtime(db,row(rt,'first')['_work_key'])
        saved.update(state='completed',slot_held=False)
        rt.store.set_runtime(db,saved['key'],saved)
    assert rt.admission.claim_next().request_id=='second'


def test_two_keys_of_same_user_share_workflow_limit(runtime):
    rt=runtime; options={'workflow_id':'catalog','workflow_concurrency':1}
    submit(rt,'first',owner='key-a',scheduling=options)
    submit(rt,'second',owner='key-b',scheduling=options)
    submit(rt,'other-user',owner='key-c',source='user:two',scheduling=options)
    first=rt.admission.claim_next();assert first.owner=='key-a'
    second=rt.admission.claim_next();assert second.owner=='key-c'
    assert rt.admission.claim_next() is None
    assert row(rt,'second','key-b')['status']=='queued'


def test_authenticated_owner_principal_controls_workflow_fairness(runtime):
    rt = runtime
    auth = AuthService(JSONStorageBackend(rt.root / "self-service-keys.json"))
    first, first_secret = auth.create_key(
        role="user", name="first", owner_subject={"organization": "shop", "user": "same"}, routes=["chat"])
    second, second_secret = auth.create_key(
        role="user", name="second", owner_subject={"user": "same", "organization": "shop"}, routes=["chat"])
    third, third_secret = auth.create_key(
        role="user", name="third", owner_subject={"organization": "shop", "user": "other"}, routes=["chat"])
    first_identity, second_identity, third_identity = (
        auth.authenticate(first_secret), auth.authenticate(second_secret), auth.authenticate(third_secret))
    assert first_identity and second_identity and third_identity
    assert first_identity["id"] == first["id"] and second_identity["id"] == second["id"]
    assert first_identity["_fair_source"] == second_identity["_fair_source"]
    assert first_identity["_fair_source"] != third_identity["_fair_source"]
    assert "owner_subject" not in first_identity
    assert all("_fair_source" not in item for item in auth.list_keys())

    spoofed_request = SimpleNamespace(
        state=SimpleNamespace(company_identity=None),
        headers={"x-workbench-consumer": "happy", "x-fair-source": third_identity["_fair_source"]},
    )
    first_source = trusted_source(first_identity, spoofed_request)
    assert first_source == first_identity["_fair_source"]
    assert trusted_source({"id": first["id"], "role": "user", "_fair_source": third_identity["_fair_source"]}, spoofed_request) == "key:" + first["id"]

    options = {"workflow_id": "catalog", "workflow_concurrency": 1}
    for identity, request_id in ((first_identity, "first"), (second_identity, "second"), (third_identity, "third")):
        rt.tasks.submit_generation(
            {**identity, "_trusted_source": trusted_source(identity, spoofed_request)},
            client_task_id=request_id, prompt="synthetic", model="gpt-image-2", size=None, scheduling=options,
        )
    with rt.store.connect() as db:
        stored_sources = {
            owner: rt.store.read_receipt(db, "image", owner, request_id).get("_scheduling_owner")
            for owner, request_id in ((first["id"], "first"), (second["id"], "second"), (third["id"], "third"))
        }
        works = [value for key, value in db.execute("SELECT name,value FROM task_runtime WHERE name LIKE 'work:%'")]
    assert stored_sources == {
        first["id"]: first_source,
        second["id"]: first_source,
        third["id"]: third_identity["_fair_source"],
    }
    assert [json.loads(value)["source"] for value in works].count(first_source) == 2
    claimed = [rt.admission.claim_next(), rt.admission.claim_next()]
    with rt.store.connect() as db:
        claimed_sources = [rt.store.read_receipt(db, "image", context.owner, context.request_id).get("_source")
                           for context in claimed if context]
    claimed_owners = {context.owner for context in claimed if context}
    assert third["id"] in claimed_owners
    assert len(claimed_owners & {first["id"], second["id"]}) == 1
    assert set(claimed_sources) == {first_source, third_identity["_fair_source"]}
    assert rt.admission.claim_next() is None

    legacy, legacy_secret = auth.create_key(role="user", routes=["chat"])
    legacy_identity = auth.authenticate(legacy_secret)
    assert legacy_identity and "_fair_source" not in legacy_identity
    assert trusted_source(legacy_identity, spoofed_request) == "key:" + legacy["id"]


def test_authenticated_source_ignores_untrusted_header_and_body(tmp_path, monkeypatch):
    from fastapi import FastAPI, Header, Request
    from fastapi.testclient import TestClient
    from api.support import require_identity
    import api.support

    auth = AuthService(JSONStorageBackend(tmp_path / "self-service-keys.json"))
    _, first_secret = auth.create_key(role="user", owner_subject="workbench:org:first", routes=["chat"])
    _, other_secret = auth.create_key(role="user", owner_subject="workbench:org:other", routes=["chat"])
    expected = auth.authenticate(first_secret)["_fair_source"]
    forged = auth.authenticate(other_secret)["_fair_source"]
    app = FastAPI()

    @app.post("/source")
    async def source_probe(request: Request, authorization: str | None = Header(default=None)):
        return {"source": trusted_source(require_identity(authorization, request=request), request)}

    monkeypatch.setattr(api.support, "auth_service", auth)
    response = TestClient(app).post(
        "/source", headers={"Authorization": "Bearer " + first_secret, "X-Fair-Source": forged},
        json={"_fair_source": forged, "_authenticated_fair_source": True},
    )
    assert response.status_code == 200
    assert response.json() == {"source": expected}


def test_paused_work_cannot_be_bypassed_by_new_model_submission(runtime):
    from services.work_lifecycle import WorkLifecycleError
    rt=runtime;options={'workflow_id':'w','workflow_concurrency':1}
    submit(rt,'first',session='same',scheduling=options)
    with rt.store.transaction() as db:
        saved=rt.store.runtime(db,row(rt,'first')['_work_key']);saved['state']='paused'
        rt.store.set_runtime(db,saved['key'],saved)
    assert rt.admission.claim_next() is None
    with pytest.raises(WorkLifecycleError,match='WORK_NOT_ACTIVE'):
        submit(rt,'next',session='same',scheduling=options)


def test_wait_deadline_after_claim_is_not_sent_and_releases_provisional_slot(runtime):
    rt=runtime;submit(rt,'first',scheduling={'workflow_id':'w','workflow_concurrency':1,'wait_deadline':at(1010)})
    ctx=rt.admission.claim_next();assert ctx
    assert row(rt,'first')['upstream_unfinished']
    rt.admission.clock=lambda:1011
    with pytest.raises(AdmissionLost):ctx.before_send()
    saved=row(rt,'first')
    assert saved['status']=='error' and saved['error_code']=='WAIT_DEADLINE_EXCEEDED'
    assert saved['upstream_outcome']=='not_sent' and not saved['upstream_unfinished']
    assert not work(rt,saved)['slot_held']
    assert not rt.calls


def test_deadline_does_not_terminate_sent_unknown_or_release_work(runtime):
    rt=runtime;submit(rt,'first',scheduling={'workflow_id':'w','workflow_concurrency':1,'wait_deadline':at(1010)})
    ctx=rt.admission.claim_next();ctx.before_send()
    with rt.store.transaction() as db:
        saved=rt.store.read_receipt(db,'image','owner','first')
        saved.update(status='error',upstream_outcome='unknown',upstream_unfinished=True)
        rt.store.write_receipt(db,'image','owner','first',saved)
    rt.admission.clock=lambda:2000
    assert rt.admission.claim_next() is None
    assert row(rt,'first')==saved and work(rt,saved)['slot_held']


def test_send_interval_is_finally_rechecked_after_concurrent_claim(runtime):
    rt=runtime;options={'workflow_id':'w','workflow_concurrency':2,'min_send_interval_seconds':30}
    submit(rt,'one',scheduling=options);submit(rt,'two',scheduling=options)
    first=rt.admission.claim_next();second=rt.admission.claim_next();assert first and second
    first.before_send()
    with pytest.raises(AdmissionLost,match='SCHEDULING_SEND_INTERVAL'):second.before_send()
    saved=row(rt,'two')
    assert saved['status']=='queued' and saved['upstream_outcome']=='not_sent'
    assert saved['waiting']['reasons']==['send_interval']
    assert not saved['_submission_started'] and not saved['upstream_unfinished']
    rt.admission.clock=lambda:1031
    again=rt.admission.claim_next();assert again and again.request_id=='two'
    again.before_send()


def test_min_interval_applies_to_actual_slots_of_one_partial_multi_image_request(runtime,monkeypatch):
    rt=runtime
    submit(rt,'original',scheduling={'min_send_interval_seconds':3,'wait_deadline':at(1001)})
    ctx=rt.admission.claim_next()
    rt.admission.update_claim(ctx,_expected_sends=2)
    ctx.before_send();ctx.image_slot_complete(0)
    clock=[1000]
    rt.admission.clock=lambda:clock[0]
    waits=[]
    monkeypatch.setattr('services.pool_admission.time.sleep',lambda seconds:(waits.append(seconds),clock.__setitem__(0,clock[0]+seconds)))
    ctx.image_slot(1)
    assert sum(waits)==3
    ctx.before_send()
    assert row(rt,'original')['_last_sent_sequence']==1
    assert row(rt,'original')['_submission_started']


def test_omitted_options_in_same_work_inherit_limits_without_old_deadline(runtime):
    rt=runtime
    submit(rt,'one',session='same',scheduling={'workflow_id':'w','workflow_concurrency':1,
                                           'min_send_interval_seconds':30,'wait_deadline':at(1050)})
    first=rt.admission.claim_next();rt.admission.execute(first)
    submit(rt,'two',session='same')
    second=row(rt,'two')
    assert second['_work_key']==row(rt,'one')['_work_key']
    assert second['_scheduling']=={'workflow_id':'w','workflow_concurrency':1,'min_send_interval_seconds':30}
    assert rt.store.load_input(second['_input_ref'])['payload'].get('_scheduling') is None
    assert rt.admission.claim_next() is None
    rt.admission.clock=lambda:1031
    assert rt.admission.claim_next().request_id=='two'


def test_same_actual_conversation_cannot_gain_second_slot_through_another_key(runtime):
    from services.work_lifecycle import WorkLifecycleError
    rt=runtime
    kwargs=dict(prompt='synthetic',model='gpt-image-2',size=None,provider_binding_id='binding-A',
                provider_account_identity='account-A',conversation_id='same-upstream',
                scheduling={'workflow_id':'w','workflow_concurrency':2})
    rt.tasks.submit_generation({'id':'key-a','_trusted_source':'user:one'},client_task_id='one',**kwargs)
    with pytest.raises(WorkLifecycleError,match='WORK_OWNER_CONFLICT'):
        rt.tasks.submit_generation({'id':'key-b','_trusted_source':'user:one'},client_task_id='two',**kwargs)
    assert row(rt,'two','key-b') is None


def test_resource_projection_separates_work_slots_from_finished_messages(runtime):
    rt=runtime
    submit(rt,'one',scheduling={'workflow_id':'w','workflow_concurrency':1})
    ctx=rt.admission.claim_next();rt.admission.execute(ctx)
    resources=rt.admission.resource_snapshot()
    assert resources['workflows']['slots_held']==1
    assert resources['workflows']['active_work']==1
    assert resources['execution']['image_workers_active']==0


def test_before_send_rechecks_not_before_after_claim_and_returns_original_to_queue(runtime):
    rt=runtime
    submit(rt,'one',scheduling={'not_before':at(990)})
    ctx=rt.admission.claim_next();assert ctx
    # A backwards-moving clock must not let the acquired claim bypass not_before.
    rt.admission.clock=lambda:980
    with pytest.raises(AdmissionLost,match='SCHEDULING_NOT_BEFORE'):ctx.before_send()
    saved=row(rt,'one')
    assert saved['status']=='queued' and saved['upstream_outcome']=='not_sent'
    assert saved['waiting']['reasons']==['not_before'] and not work(rt,saved)['slot_held']
    rt.admission.clock=lambda:1001
    assert rt.admission.claim_next().request_id=='one'


@pytest.mark.parametrize('unknown',[False,True])
def test_text_and_image_cannot_interleave_same_physical_conversation(runtime,unknown):
    from services.text_task_service import TextTaskService
    rt=runtime
    rt.admission.settings=lambda:{'image_account_concurrency':4,'chat_account_concurrency':4,'codex_max_concurrency':4}
    text=TextTaskService(rt.store.path,admission=rt.admission)
    body={'client_request_id':'text-first','client_conversation_id':'text-alias','model':'fixture-text',
          'messages':[{'role':'user','content':'synthetic'}], 'provider_binding_id':'binding-A',
          'provider_account_identity':'account-A','conversation_id':'physical-chat','parent_message_id':'parent'}
    text.submit('owner',body)
    rt.tasks.submit_generation({'id':'owner'},client_task_id='image-next',prompt='synthetic',model='gpt-image-2',size=None,
                              provider_binding_id='binding-A',provider_account_identity='account-A',
                              conversation_id='physical-chat',parent_message_id='parent',client_conversation_id='image-alias')
    first=rt.admission.claim_next();assert first.kind=='text'
    if unknown:
        with rt.store.transaction() as db:
            saved=rt.store.read_receipt(db,'text','owner','text-first')
            saved.update(status='unknown',upstream_outcome='unknown',_submission_started=True)
            rt.store.write_receipt(db,'text','owner','text-first',saved)
    assert rt.admission.claim_next() is None
    with rt.store.transaction() as db:
        saved=rt.store.read_receipt(db,'text','owner','text-first')
        saved.update(status='succeeded',upstream_outcome='completed',_turn_reserved=False,_executing=False)
        rt.store.write_receipt(db,'text','owner','text-first',saved)
    assert rt.admission.claim_next().request_id=='image-next'


def test_cross_kind_final_send_defense_and_independent_conversation_parallel(runtime):
    from services.text_task_service import TextTaskService
    rt=runtime
    rt.admission.settings=lambda:{'image_account_concurrency':4,'chat_account_concurrency':4,'codex_max_concurrency':4}
    rt.tasks.submit_generation({'id':'owner'},client_task_id='image',prompt='synthetic',model='gpt-image-2',size=None,
                              provider_binding_id='binding-A',provider_account_identity='account-A',
                              conversation_id='chat-a',parent_message_id='parent')
    image=rt.admission.claim_next();assert image
    text=TextTaskService(rt.store.path,admission=rt.admission)
    text.submit('owner',{'client_request_id':'text','client_conversation_id':'different-client', 'model':'fixture-text',
                        'messages':[{'role':'user','content':'synthetic'}], 'provider_binding_id':'binding-A',
                        'provider_account_identity':'account-A','conversation_id':'chat-b','parent_message_id':'parent'})
    other=rt.admission.claim_next();assert other and other.kind=='text'
    with rt.store.transaction() as db:
        saved=rt.store.read_receipt(db,'text','owner','text')
        saved['conversation_id']='chat-a'  # Simulate an existing legacy executor publishing its actual cursor.
        rt.store.write_receipt(db,'text','owner','text',saved)
    with pytest.raises(AdmissionLost,match='physical conversation'):image.before_send()
    assert row(rt,'image')['status']=='queued'
    assert row(rt,'image')['upstream_outcome']=='not_sent' and not rt.calls


def test_multi_image_final_interval_race_waits_same_next_slot_without_false_unknown(runtime,monkeypatch):
    from services.protocol.conversation import ConversationRequest, ImageOutput, stream_image_outputs_with_pool
    from services.request_context import current_request
    rt=runtime;options={'workflow_id':'shared','workflow_concurrency':2,'min_send_interval_seconds':3}
    submit(rt,'multi',scheduling=options);submit(rt,'competitor',scheduling=options)
    original=rt.admission.claim_next();competitor=rt.admission.claim_next()
    assert original and competitor
    assert original.selected_account()['provider_account_identity'] != competitor.selected_account()['provider_account_identity']
    rt.admission.update_claim(original,_expected_sends=2)
    clock=[1000.0];rt.admission.clock=lambda:clock[0]
    waits=[];sent=[];generated=[]
    def sleep(seconds):
        waits.append(seconds);clock[0]+=seconds
    monkeypatch.setattr('services.pool_admission.time.sleep',sleep)
    def generate(request,index,total):
        assert current_request.get() is original
        generated.append(index)
        if index==2:
            # The pre-slot wait finished. Another account wins the actual send
            # transaction before this slot reaches its paced POST callback.
            assert clock[0]==1003
            competitor.before_send()
            sent.append(('competitor',clock[0]))
        original.before_send()
        sent.append((index,clock[0]))
        with pytest.raises(AdmissionLost):original.before_send()
        return [ImageOutput(kind='result',model=request.model,index=index,total=total,data=[{'url':'https://fixture.invalid/'+str(index)}])]
    monkeypatch.setattr('services.protocol.conversation._generate_single_image',generate)
    outputs=[]
    def execute(context,body):
        outputs.extend(stream_image_outputs_with_pool(ConversationRequest(prompt='synthetic',model='gpt-image-2',n=2)))
        rt.admission.update_claim(context,status='success',upstream_unfinished=False)
    rt.admission.register('image',execute)
    rt.admission.execute(original)
    saved=row(rt,'multi')
    assert saved['status']=='success' and saved.get('error_code') is None
    assert saved['_completed_slot']==1 and saved['_last_sent_sequence']==1
    assert generated==[1,2] and [output.index for output in outputs]==[1,2]
    assert sent==[(1,1000.0),('competitor',1003.0),(2,1006.0)]
    assert sum(waits)==6
