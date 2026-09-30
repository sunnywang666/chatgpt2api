"""Ordinary Chat account selection uses real receipts/admission; upstream stays isolated."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from services.account_service import AccountService
from services.conversation_binding_service import ConversationBindingError, ConversationBindingService
from services.request_context import AdmissionLost, current_request
from services.storage.json_storage import JSONStorageBackend
from services.text_task_service import TextTaskService
from test.test_pool_admission import build
from test.test_public_chat_api import public_chat, request_body, png_data_url


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    rows = [{"access_token": f"fixture-{x}", "provider_account_identity": f"account-{x}",
             "managed_pool_account_ref": "car_" + x * 43, "type": "Plus", "status": "正常",
             "source_type": "web", "conversation_binding_ids": [f"binding-{x}"]} for x in "AB"]
    (tmp_path / 'accounts.json').write_text(json.dumps(rows))
    service = AccountService(JSONStorageBackend(tmp_path / 'accounts.json'))
    monkeypatch.setattr('services.account_service.account_service', service)
    monkeypatch.setattr('services.conversation_binding_service.account_service', service)
    monkeypatch.setattr(service, 'refresh_access_token', lambda token, **kw: token)
    monkeypatch.setattr('services.conversation_binding_service.conversation_events', Mock(side_effect=AssertionError('unexpected upstream attempt')))
    monkeypatch.setattr('services.conversation_binding_service.OpenAIBackendAPI', Mock())
    return service


def post(h, body):
    return h.client.post('/api/chat-requests', headers=h.headers(), json=body)


def selected(name='chat-1', letter='B'):
    return {**request_body(name), 'account_ref': 'car_' + letter * 43}


def receipt(h, name='chat-1'):
    with h.tasks.store.connect() as db:
        return h.tasks.store.read_receipt(db, 'text', h.key_a['id'], name)


@pytest.mark.parametrize('value', [None, '', 'account-B', 'car_short', 'car_'+'B'*42, 'car_'+'!'+ 'B'*42, 123])
def test_invalid_selector_rejected(public_chat, value):
    h=public_chat
    assert post(h, {**request_body(), 'account_ref': value}).status_code == 422
    assert not h.queue.calls


def test_unknown_and_ambiguous_selectors_fail_without_queue(public_chat, accounts):
    h=public_chat
    assert post(h, selected(letter='C')).json()['detail']['code']=='CHAT_ACCOUNT_NOT_FOUND'
    with accounts._lock:
        accounts._accounts['duplicate'] = {**accounts._accounts['fixture-B'], 'access_token':'duplicate'}
        accounts._save_accounts()
    assert post(h, selected()).json()['detail']['code']=='CHAT_ACCOUNT_AMBIGUOUS'
    assert not h.queue.calls


def test_same_id_selection_is_immutable_and_receipt_survives_catalog_failure(public_chat, accounts, monkeypatch):
    h=public_chat
    assert post(h, selected()).status_code==202
    original=receipt(h)
    assert original['_requested_account_identity']=='account-B'
    assert original['_requested_account_ref']=='car_'+'B'*43
    assert '_requested_account_identity' not in post(h, selected()).text
    monkeypatch.setattr(accounts, 'resolve_public_chat_account', Mock(side_effect=RuntimeError('offline')))
    monkeypatch.setattr('api.chat_requests.require_public_text_model', Mock(side_effect=RuntimeError('offline')))
    assert post(h, selected()).status_code==202
    assert post(h, selected(letter='A')).status_code==409
    assert post(h, request_body()).status_code==409
    restarted=TextTaskService(h.tasks.store.path, executor=h.queue)
    assert restarted.validate_submission(h.key_a['id'], h.tasks.store.load_input(original['_input_ref'])) is not None
    assert len(h.queue.calls)==1


@pytest.mark.parametrize('letter', ['B', None, 'A'])
def test_continuation_inherits_or_matches_original_only(public_chat, accounts, letter):
    h=public_chat
    h.upstream.return_value.update(provider_account_identity='account-B', provider_binding_id='binding-B')
    first={**selected(), 'client_conversation_id':'work'}
    assert post(h,first).status_code==202
    h.queue.run()
    nxt={**request_body('next'), 'client_conversation_id':'work', 'previous_request_id':'chat-1'}
    if letter:
        nxt['account_ref']='car_'+letter*43
    response=post(h,nxt)
    if letter=='A':
        assert response.status_code==409 and response.json()['detail']['code']=='CHAT_ACCOUNT_SELECTION_CONFLICT'
        assert not h.queue.calls
    else:
        assert response.status_code==202
        assert receipt(h,'next')['provider_account_identity']=='account-B'
        h.queue.run()
        assert h.upstream.call_args.args[0]['provider_account_identity']=='account-B'


def admitted(tmp_path, accounts):
    _, store, admission=build(tmp_path)
    admission.accounts=accounts
    calls=[]
    def runner(body,on_cursor):
        current_request.get().before_send()
        calls.append(body)
        return {'content':'done'}
    tasks=TextTaskService(store.path, runner=runner, admission=admission)
    admission.register('text', lambda ctx, body: tasks._run(ctx.owner,ctx.request_id,body))
    return tasks, admission, calls


def internal(name='selected', letter='B'):
    return {'client_request_id':name, 'client_conversation_id':'session-'+name, 'model':'gpt-text',
            'messages':[{'role':'user','content':'hello'}], '_public_route':'chat', '_text_only_binding':True,
            **({'_requested_account_ref':'car_'+letter*43} if letter else {})}


@pytest.mark.parametrize('blocked', ['disabled','model','quota'])
def test_selected_account_waits_without_fallback_and_recovers_after_restart(tmp_path, accounts, blocked):
    tasks, admission, calls=admitted(tmp_path,accounts)
    with accounts._lock:
        b=accounts._accounts['fixture-B']
        if blocked=='disabled': b['managed_disabled']=True
        elif blocked=='quota': b['limits_progress']=[{'feature_name':'gpt-text','remaining':0}]
        else: admission.model_types=lambda _:SimpleNamespace(account_types={'Plus'},account_identities={'account-A'})
        accounts._save_accounts()
    tasks.submit('owner',internal())
    assert admission.claim_next() is None
    with tasks.store.connect() as db:
        saved=tasks.store.read_receipt(db,'text','owner','selected')
    assert saved['status']=='queued' and saved['_requested_account_identity']=='account-B'
    # Recreate services with durable storage. A is still eligible throughout.
    with accounts._lock:
        b=accounts._accounts['fixture-B']
        b['managed_disabled']=False; b['status']='正常'; b.pop('limits_progress',None)
        accounts._save_accounts()
    restarted, recovered, after=admitted(tmp_path,accounts)
    ctx=recovered.claim_next()
    assert ctx is not None
    recovered.execute(ctx)
    assert after[0]['provider_account_identity']=='account-B'
    assert after[0]['_requested_account_identity']=='account-B'
    assert restarted.read('owner','selected')['status']=='succeeded'
    assert not calls


def test_selector_and_automatic_requests_use_existing_account_scheduler(tmp_path, accounts):
    tasks, admission, calls=admitted(tmp_path,accounts)
    tasks.submit('owner',internal())
    first=admission.claim_next(); admission.execute(first)
    assert calls[-1]['provider_account_identity']=='account-B'
    tasks.submit('owner',internal('auto',None))
    admission.execute(admission.claim_next())
    assert calls[-1]['provider_account_identity']=='account-A'


@pytest.mark.parametrize('change', ['identity','binding','model','duplicate'])
def test_before_send_refuses_selected_identity_or_capability_drift(tmp_path, accounts, change):
    tasks, admission, calls=admitted(tmp_path,accounts)
    tasks.submit('owner',internal())
    ctx=admission.claim_next()
    if change in ('identity','binding'):
        with tasks.store.transaction() as db:
            row=tasks.store.read_receipt(db,'text','owner','selected')
            row['provider_account_identity' if change=='identity' else 'provider_binding_id']='account-A' if change=='identity' else 'binding-A'
            tasks.store.write_receipt(db,'text','owner','selected',row)
    elif change=='model':
        admission.model_types=lambda _:SimpleNamespace(account_types={'Plus'},account_identities={'account-A'})
    else:
        with accounts._lock:
            accounts._accounts['duplicate']={**accounts._accounts['fixture-B'],'access_token':'duplicate'}
            accounts._save_accounts()
    with pytest.raises(AdmissionLost): ctx.before_send()
    assert not calls


def test_nonadmission_actual_binding_selects_only_b(public_chat, accounts, monkeypatch):
    h=public_chat
    h.tasks.runner=ConversationBindingService().complete_text
    backend=Mock()
    backend.get_conversation_parent_message_id.return_value='final'
    monkeypatch.setattr('services.conversation_binding_service.OpenAIBackendAPI',Mock(return_value=backend))
    monkeypatch.setattr('services.conversation_binding_service.conversation_events',lambda *a,**k: iter([
        {'type':'conversation.delta','conversation_id':'new-chat','delta':'answer'}]))
    monkeypatch.setattr(ConversationBindingService,'_read_text_request_result',lambda *a,**k:{
        'status':'succeeded','content':'answer','conversation_id':'new-chat','parent_message_id':'final'})
    assert post(h,selected()).status_code==202
    h.queue.run()
    row=receipt(h)
    assert row['provider_account_identity']=='account-B'
    assert accounts.get_bound_account_identity(row['provider_binding_id'])=='account-B'
    assert row['status']=='succeeded'


@pytest.mark.parametrize("unavailable", ["disabled", "quota", "model"])
def test_nonadmission_unavailable_selection_never_binds_a(public_chat, accounts, monkeypatch, unavailable):
    h=public_chat
    with accounts._lock:
        if unavailable=='disabled':
            accounts._accounts['fixture-B']['managed_disabled']=True
        elif unavailable=='quota':
            accounts._accounts['fixture-B']['limits_progress']=[{'feature_name':'gpt-text','remaining':0}]
        accounts._save_accounts()
    if unavailable=='model':
        monkeypatch.setattr('services.model_service.model_catalog_service.route_for_model', lambda _:SimpleNamespace(
            account_types=frozenset({'Plus'}), account_identities=frozenset({'account-A'})))
    h.tasks.runner=ConversationBindingService().complete_text
    assert post(h,selected()).status_code==202
    h.queue.run()
    row=receipt(h)
    assert row['status']=='failed' and row.get('provider_account_identity') is None


@pytest.mark.parametrize("observed_image", [True, False])
def test_image_evidence_keeps_partial_public_directory(public_chat, monkeypatch, observed_image):
    from api import ai
    from services.model_service import ModelRoute
    catalog = SimpleNamespace(
        catalog_is_unknown=lambda: True,
        known_account_types_for_model=lambda _: frozenset(),
        route_for_model=lambda _: ModelRoute(frozenset(), False, frozenset()),
        public_accounts_for_model=lambda model, capabilities: ([{
            "account_ref": "car_" + "B" * 43, "capabilities": capabilities,
            "state": "unavailable", "reason": "quota_exhausted", "observation_state": "observed",
        }] if observed_image else []),
    )
    monkeypatch.setattr("services.public_chat_service.model_catalog_service", catalog)
    monkeypatch.setattr(ai.openai_v1_models, "list_models", lambda: {
        "object": "list", "data": [{"id": "gpt-image-2"}],
    })
    response = public_chat.client.get("/v1/models", headers=public_chat.headers())
    if observed_image:
        assert response.status_code == 200
        assert response.json()["model_catalog"] == {"state": "partial"}
        assert [row["id"] for row in response.json()["data"]] == ["gpt-image-2"]
    else:
        assert response.status_code == 502
        assert response.json()["detail"]["code"] == "MODEL_DISCOVERY_UNAVAILABLE"
    unknown = post(public_chat, request_body(model="new-text-model"))
    assert unknown.status_code == 503
    assert unknown.json()["detail"]["code"] == "MODEL_DISCOVERY_UNAVAILABLE"


def test_image_input_preserves_selected_account_in_original_envelope(public_chat, accounts):
    h=public_chat
    body=selected()
    body['messages']=[{'role':'user','content':[
        {'type':'text','text':'Describe this product'},
        {'type':'image_url','image_url':{'url':png_data_url()}},
    ]}]
    assert post(h,body).status_code==202
    saved=receipt(h)
    envelope=h.tasks.store.load_input(saved['_input_ref'])
    assert envelope['_requested_account_ref']==body['account_ref']
    assert saved['_requested_account_identity']=='account-B'
    h.queue.run()
    assert h.upstream.call_args.args[0]['_requested_account_identity']=='account-B'


def test_independent_selected_accounts_can_claim_in_parallel(tmp_path, accounts):
    tasks, admission, calls=admitted(tmp_path,accounts)
    tasks.submit('owner',internal('first','A'))
    tasks.submit('owner',internal('second','B'))
    first=admission.claim_next()
    second=admission.claim_next()
    assert first is not None and second is not None
    assert first.request_id != second.request_id
    admission.execute(first); admission.execute(second)
    assert {body['provider_account_identity'] for body in calls}=={'account-A','account-B'}


def test_selected_binding_ambiguity_keeps_distinct_error(accounts, monkeypatch):
    monkeypatch.setattr('services.model_service.model_catalog_service.route_for_model', lambda _:SimpleNamespace(
        account_types=frozenset({'Plus'}), account_identities=frozenset({'account-B'})))
    with accounts._lock:
        accounts._accounts['duplicate']={**accounts._accounts['fixture-B'],'access_token':'duplicate'}
        accounts._save_accounts()
    with pytest.raises(RuntimeError, match='no unique paid account supports text model'):
        accounts.create_text_conversation_binding(text_model='gpt-text', requested_account_identity='account-B')
