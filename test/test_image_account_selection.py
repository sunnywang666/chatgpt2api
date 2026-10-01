"""Image capability and exact selection through durable admission, no upstream."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from services.account_service import AccountService
from services.image_task_service import ImageTaskService, _request_hash
from services.image_thread import ImageThreadError, accept_thread, predecessor_state
from services.owned_accounts import image_dispatch_capacity, image_capability_projection
from services.request_context import AdmissionLost, current_request
from test.test_pool_admission import build


def fresh(remaining=2):
    return {'limits_progress':[{'feature_name':'image_gen','remaining':remaining}],
            'capacity_observed_at':datetime.now(timezone.utc).isoformat(),
            'capacity_read_failed_at':None,'capacity_used_since_observation':False}


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    rows=[{'access_token':'fixture-'+x, 'provider_account_identity':'account-'+x,
           'managed_pool_account_ref':'car_'+x*43,'type':'Plus','status':'正常','source_type':'web',
           'quota':999, 'conversation_binding_ids':['binding-'+x], **fresh()} for x in 'AB']
    (tmp_path/'accounts.json').write_text(json.dumps(rows))
    accounts,store,admission=build(tmp_path)
    monkeypatch.setattr(accounts,'refresh_image_capability',Mock())
    monkeypatch.setattr(accounts,'fetch_remote_info',lambda token,*a,**k:accounts.get_account(token))
    monkeypatch.setattr(accounts,'refresh_access_token',lambda token,**k:token)
    monkeypatch.setattr('services.account_service.account_service',accounts)
    calls=[]
    def handler(body):
        current_request.get().before_send()
        calls.append(dict(body))
        return {'created':1,'data':[{'b64_json':'aW1hZ2U='}],
                '_provider_binding_id':body.get('provider_binding_id'),
                '_provider_account_identity':body.get('provider_account_identity'),
                '_conversation_id':'synthetic-chat','_parent_message_id':'synthetic-parent'}
    tasks=ImageTaskService(tmp_path/'images.json',store=store,admission=admission,
                           generation_handler=handler,edit_handler=handler)
    admission.register('image',lambda ctx,body:tasks._run_task(ctx.owner+':'+ctx.request_id,
                       body['mode'],body['payload'],body['identity'],body['payload']['model']))
    def update(letter,**changes):
        accounts.update_account('fixture-'+letter,changes,quiet=True)
    return SimpleNamespace(accounts=accounts,store=store,admission=admission,tasks=tasks,calls=calls,update=update,root=tmp_path)


def submit(rt,name='original',letter='B',mode='generate',**kw):
    method=rt.tasks.submit_generation if mode=='generate' else rt.tasks.submit_edit
    return method({'id':'owner','role':'user','external_image_client':True},client_task_id=name,
                  prompt='synthetic input',model='gpt-image-2',size=None,
                  **({'account_ref':'car_'+letter*43} if letter else {}),**kw)


def row(rt,name='original'):
    with rt.store.connect() as db:return rt.store.read_receipt(db,'image','owner',name)


@pytest.mark.parametrize('mode',['generate','edit'])
def test_selected_b_is_persisted_and_dispatched_when_a_also_eligible(runtime,mode):
    rt=runtime
    response=submit(rt,mode=mode)
    assert response['account_ref']=='car_'+'B'*43
    saved=row(rt)
    assert saved['_requested_account_identity']=='account-B'
    assert rt.store.load_input(saved['_input_ref'])['payload']['_requested_account_ref']=='car_'+'B'*43
    ctx=rt.admission.claim_next()
    assert ctx.selected_account()['provider_account_identity']=='account-B'
    rt.admission.execute(ctx)
    assert row(rt)['status']=='success'
    assert rt.calls[0]['provider_account_identity']=='account-B'


@pytest.mark.parametrize('change',[{'quota':999,'limits_progress':[]},fresh(0),
    {**fresh(),'capacity_read_failed_at':datetime.now(timezone.utc).isoformat()},
    {**fresh(),'capacity_observed_at':(datetime.now(timezone.utc)-timedelta(minutes=8)).isoformat()},
    {**fresh(),'capacity_used_since_observation':True}, {'managed_disabled':True}, {'status':'限流'}])
def test_original_waits_without_fallback_and_recovers(runtime,change):
    rt=runtime;rt.update('B',**change)
    submit(rt)
    original=row(rt)
    assert rt.admission.claim_next() is None
    assert row(rt)['status']=='queued' and not rt.calls
    rt.update('B',**fresh(),status='正常',managed_disabled=False)
    # A remains eligible; storage/restart retains original request and choice.
    restarted=ImageTaskService(rt.root/'images.json',store=rt.store,admission=rt.admission)
    assert restarted.submit_generation({'id':'owner','role':'user','external_image_client':True},client_task_id='original',
        prompt='synthetic input',model='gpt-image-2',size=None,account_ref='car_'+'B'*43)['id']=='original'
    ctx=rt.admission.claim_next();assert ctx is not None
    rt.admission.execute(ctx)
    assert rt.calls[0]['provider_account_identity']=='account-B'
    assert row(rt)['_input_ref']==original['_input_ref']


def test_automatic_selection_uses_only_image_evidence(runtime):
    rt=runtime;rt.update('A',quota=999,limits_progress=[])
    submit(rt,letter=None)
    ctx=rt.admission.claim_next()
    assert ctx.selected_account()['provider_account_identity']=='account-B'


@pytest.mark.parametrize('value,code',[('bad','IMAGE_ACCOUNT_REF_INVALID'),('car_'+'C'*43,'IMAGE_ACCOUNT_NOT_FOUND')])
def test_bad_selector_has_no_receipt(runtime,value,code):
    rt=runtime
    with pytest.raises(ImageThreadError) as error:
        rt.tasks.submit_generation({'id':'owner'},client_task_id='original',prompt='x',model='gpt-image-2',size=None,account_ref=value)
    assert error.value.code==code and row(rt) is None


def test_ambiguous_ref_and_existing_binding_conflict(runtime):
    rt=runtime
    with pytest.raises(ImageThreadError,match='IMAGE_ACCOUNT_SELECTION_CONFLICT'):
        submit(rt,provider_binding_id='binding-A',provider_account_identity='account-A')
    with rt.accounts._lock:
        rt.accounts._accounts['duplicate']={**rt.accounts._accounts['fixture-B'],'access_token':'duplicate'}
        rt.accounts._save_accounts()
    with pytest.raises(ImageThreadError,match='IMAGE_ACCOUNT_AMBIGUOUS'):submit(rt)


def test_same_id_reads_original_without_discovery_and_selection_drift_conflicts(runtime,monkeypatch):
    rt=runtime;submit(rt)
    monkeypatch.setattr(rt.accounts,'resolve_image_account',Mock(side_effect=RuntimeError('catalog down')))
    assert submit(rt)['id']=='original'
    for letter in ['A',None]:
        with pytest.raises(ValueError,match='immutable'):submit(rt,letter=letter)
    assert not rt.calls


@pytest.mark.parametrize('change',[{'limits_progress':[]},fresh(0),{'managed_disabled':True}])
def test_final_send_guard_refuses_lost_capability(runtime,change):
    rt=runtime;submit(rt);ctx=rt.admission.claim_next();rt.update('B',**change)
    with pytest.raises(AdmissionLost):ctx.before_send()
    assert not row(rt)['_submission_started']


def test_initial_and_bound_binding_agree_on_image_metadata(runtime):
    rt=runtime;rt.update('B',quota=999,limits_progress=[])
    with pytest.raises(RuntimeError):
        rt.accounts.create_conversation_binding(image_model='gpt-image-2',requested_account_identity='account-B')
    rt.update('B',**fresh())
    binding,identity,token=rt.accounts.create_conversation_binding(image_model='gpt-image-2',requested_account_identity='account-B')
    assert identity=='account-B' and token=='fixture-B'
    rt.accounts.release_image_slot(token)
    rt.update('B',**fresh(0))
    with pytest.raises(RuntimeError):rt.accounts.acquire_bound_image_access_token(binding,image_model='gpt-image-2')


def test_selected_thread_inherits_and_conflicts_without_changing_predecessor():
    previous={'id':'old','owner_id':'owner','_sequence':1,'_requested_account_identity':'account-B',
              '_requested_account_ref':'car_'+'B'*43,'client_conversation_id':'session',
              '_image_thread':{'id':'work','protocol':'image-thread-v1'}}
    task={'id':'next','owner_id':'owner'}
    accept_thread(task,[previous],{'image_thread_id':'work'},'generate')
    assert task['_requested_account_identity']=='account-B'
    assert previous['id']=='old'
    with pytest.raises(ImageThreadError,match='IMAGE_ACCOUNT_SELECTION_CONFLICT'):
        accept_thread({'id':'other','owner_id':'owner','_requested_account_identity':'account-A'},[previous],{'image_thread_id':'work'},'generate')


def test_codex_text_observation_and_chat_image_count_do_not_fabricate_image_slots(runtime):
    rt=runtime;rt.update('B',source_type='codex',**fresh(),codex_observation={'state':'observed','models':[{'id':'gpt-5.5'}]})
    result=rt.tasks.submit_generation({'id':'owner'},client_task_id='original',prompt='x',model='codex-gpt-image-2',size=None,account_ref='car_'+'B'*43)
    assert result['status']=='queued' and row(rt)['_route']=='codex'
    assert rt.admission.claim_next() is None
    assert image_dispatch_capacity(rt.accounts.get_account('fixture-B'),'codex-gpt-image-2')==0
    assert not rt.calls


@pytest.mark.parametrize('observed,disabled,expected',[(2,False,'正常'),(0,False,'限流'),(2,True,'禁用'),(None,False,'限流')])
def test_metadata_refresh_releases_only_verified_positive_limit(runtime,monkeypatch,observed,disabled,expected):
    rt=runtime;rt.update('B',status='限流',managed_disabled=disabled,**fresh(0))
    submit(rt)
    assert rt.admission.claim_next() is None
    if observed is None:
        monkeypatch.setattr(rt.accounts,'_verified_chat_info',Mock(side_effect=TimeoutError('synthetic metadata')))
    else:
        monkeypatch.setattr(rt.accounts,'_verified_chat_info',lambda token:(('fixture-user',''),{
            **fresh(observed),'quota':observed,'status':'正常' if observed else '限流'}))
    rt.accounts._refresh_pool_chat('car_'+'B'*43)
    account=rt.accounts.get_account('fixture-B')
    assert account['status']==expected
    ctx=rt.admission.claim_next()
    assert (ctx is not None)==(observed==2 and not disabled)
    if ctx:assert ctx.selected_account()['provider_account_identity']=='account-B'
    assert not rt.calls


def test_used_or_missing_timestamp_is_not_current_capacity():
    account={'access_token':'fixture','source_type':'web','type':'Plus','status':'正常','quota':500,
             'limits_progress':[{'feature_name':'image_gen','remaining':50}]}
    assert not image_capability_projection(account)['capable']
    assert image_dispatch_capacity(account)==0


def test_metadata_read_recovers_same_selected_queue_without_resubmit(runtime,monkeypatch):
    rt=runtime;rt.update('B',status='限流',**fresh(0));submit(rt)
    original=row(rt)
    metadata=Mock(return_value=(('fixture-user',''),{**fresh(3),'quota':3,'status':'正常'}))
    monkeypatch.setattr(rt.accounts,'_verified_chat_info',metadata)
    monkeypatch.setattr(rt.accounts,'refresh_image_capability',rt.accounts._refresh_pool_chat)
    ctx=rt.admission.claim_next()
    assert ctx is not None and ctx.request_id=='original'
    assert ctx.selected_account()['provider_account_identity']=='account-B'
    assert metadata.call_args.args[0]=='fixture-B'
    assert row(rt)['_input_ref']==original['_input_ref']
    assert not rt.calls


def test_unknown_image_never_probes_or_resubmits(runtime,monkeypatch):
    rt=runtime;submit(rt)
    with rt.store.transaction() as db:
        original=rt.store.read_receipt(db,'image','owner','original')
        original.update(status='error',upstream_outcome='unknown',upstream_unfinished=True,_submission_started=True)
        rt.store.write_receipt(db,'image','owner','original',original)
    assert rt.admission.claim_next() is None
    rt.accounts.refresh_image_capability.assert_not_called()
    assert row(rt)==original and not rt.calls


def test_text_and_image_binding_requires_same_physical_support(runtime,monkeypatch):
    rt=runtime
    monkeypatch.setattr('services.model_service.model_catalog_service.route_for_model',lambda _:SimpleNamespace(
        account_types=frozenset({'Plus'}),account_identities=frozenset({'account-B'})))
    binding,identity,token=rt.accounts.create_conversation_binding(image_model='gpt-image-2',text_model='text-b')
    assert identity=='account-B';rt.accounts.release_image_slot(token)
    with pytest.raises(RuntimeError):
        rt.accounts.create_conversation_binding(image_model='gpt-image-2',text_model='text-b',requested_account_identity='account-A')


def test_selected_execution_copy_and_final_identity_guard(runtime):
    rt=runtime;submit(rt);ctx=rt.admission.claim_next()
    with rt.store.transaction() as db:
        task=rt.store.read_receipt(db,'image','owner','original')
        task.update(provider_account_identity='account-A',provider_binding_id='binding-A')
        rt.store.write_receipt(db,'image','owner','original',task)
    with pytest.raises(AdmissionLost):ctx.before_send()
    assert not row(rt)['_submission_started'] and not rt.calls


def test_codex_image_eligibility_does_not_treat_image_alias_as_text_model(runtime):
    rt=runtime
    rt.update('B',source_type='codex')
    eligible=Mock(return_value=rt.accounts.get_account('fixture-B'))
    rt.admission.codex=SimpleNamespace(_eligible_account=eligible)
    rt.admission._next_codex_probe=2000
    rt.tasks.submit_generation({'id':'owner'},client_task_id='original',prompt='x',model='codex-gpt-image-2',size=None,account_ref='car_'+'B'*43)
    assert rt.admission.claim_next() is None
    assert eligible.call_args_list
    assert all(len(call.args)<2 or call.args[1]=='' for call in eligible.call_args_list)
    assert row(rt)['status']=='queued' and not rt.calls


def test_codex_existing_affinity_cannot_override_explicit_selection(runtime):
    rt=runtime
    rt.update('A',source_type='codex',codex_affinities={'same-session':{'state':'idle'}})
    rt.update('B',source_type='codex')
    rt.admission.codex=SimpleNamespace(_eligible_account=Mock(return_value=None))
    rt.admission._next_codex_probe=2000
    rt.tasks.submit_generation({'id':'owner'},client_task_id='original',prompt='x',model='codex-gpt-image-2',size=None,account_ref='car_'+'B'*43)
    with rt.store.transaction() as db:
        saved=rt.store.read_receipt(db,'image','owner','original')
        saved['client_conversation_id']='same-session'
        rt.store.write_receipt(db,'image','owner','original',saved)
    assert rt.admission.claim_next() is None
    assert row(rt)['error_code']=='IMAGE_ACCOUNT_SELECTION_CONFLICT'
    assert row(rt)['_requested_account_identity']=='account-B' and not rt.calls
