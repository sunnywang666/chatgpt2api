"""Sequential work-session contract on the real public handler and SQLite store.

Upstream is isolated. These checks are not real ChatGPT/Production acceptance.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from services.text_task_service import TextTaskService
from services.conversation_binding_service import ConversationBindingError, ConversationBindingService
from test.test_public_chat_api import public_chat, request_body, QueuedExecutor


def turn(name, previous=None, session='dsh-session'):
    body = request_body(name)
    body['client_conversation_id'] = session
    if previous is not None:
        body['previous_request_id'] = previous
    return body


def post(h, body):
    return h.client.post('/api/chat-requests', headers=h.headers(), json=body)


def test_three_turns_keep_one_upstream_account_and_conversation(public_chat):
    h = public_chat
    calls = []
    def upstream(body, on_cursor):
        calls.append(dict(body))
        n = len(calls)
        if n > 1:
            assert body['provider_account_identity'] == 'original-account'
            assert body['provider_binding_id'] == 'original-binding'
            assert body['conversation_id'] == 'original-conversation'
            assert body['parent_message_id'] == f'answer-{n-1}'
        return {'content': f'answer{n}', 'provider_binding_id': 'original-binding',
                'provider_account_identity': 'original-account', 'conversation_id': 'original-conversation',
                'parent_message_id': f'answer-{n}', 'binding_status': 'bound'}
    h.tasks.runner = upstream
    for n in range(1,4):
        response = post(h, turn(f'r{n}', f'r{n-1}' if n > 1 else None))
        assert response.status_code == 202
        assert response.json()['conversation'] == {'client_conversation_id':'dsh-session',
            'previous_request_id': f'r{n-1}' if n>1 else None, 'protocol':'sequential-v1'}
        h.queue.run()
    assert len(calls) == 3
    assert len({b['client_conversation_id'] for b in calls}) == 1
    assert len({b['_request_message_id'] for b in calls}) == 3
    # Earlier same-ID reads/retries must still work after newer turns, unchanged.
    assert post(h, turn('r1')).status_code == 200
    assert post(h, turn('r2','r1')).status_code == 200
    assert not h.queue.calls
    assert post(h, {**turn('r1'), 'messages':[{'role':'user','content':'drift'}]}).status_code == 409


def test_archive_route_uses_only_the_latest_original_owner_request(public_chat):
    h=public_chat
    post(h,turn('r1'));h.queue.run()
    post(h,turn('r2','r1'));h.queue.run()
    from services.text_task_service import conversation_binding_service
    with patch.object(conversation_binding_service,'archive',return_value={'archived':True}) as archive:
        early=h.client.post('/api/chat-requests/r1/archive-conversation',headers=h.headers(),json={})
        assert early.status_code==409,early.text
        assert h.client.post('/api/chat-requests/r2/archive-conversation',headers=h.headers(),json={'conversation_id':'forged'}).status_code==422
        assert h.client.post('/api/chat-requests/r2/archive-conversation',headers=h.headers(h.secret_b),json={}).status_code==404
        response=h.client.post('/api/chat-requests/r2/archive-conversation',headers=h.headers(),json={})
    assert response.status_code==200,response.text
    assert response.json()=={'request_id':'r2','archived':True,'conversation':{'client_conversation_id':'dsh-session','protocol':'sequential-v1'}}
    assert archive.call_args.args[0]['parent_message_id']=='parent-secret'


def test_archived_sequential_text_chat_is_restored_before_continuation():
    service=ConversationBindingService();backend=Mock()
    backend._get_conversation.return_value={'current_node':'previous-answer','is_archived':True}
    body={'provider_binding_id':'binding','provider_account_identity':'account',
          'client_conversation_id':'work','conversation_id':'chat','parent_message_id':'previous-answer',
          '_public_session_ref':'s','_request_message_id':'our-user','model':'auto',
          'messages':[{'role':'user','content':'do work'}]}
    from contextlib import nullcontext
    with patch('services.conversation_binding_service.account_service') as accounts, \
         patch('services.conversation_binding_service.OpenAIBackendAPI',return_value=backend), \
         patch('services.conversation_binding_service.conversation_events',return_value=iter([
             {'type':'conversation.delta','conversation_id':'chat','delta':'original answer'}])), \
         patch.object(service,'_read_text_request_result',return_value={'status':'succeeded','content':'original answer','conversation_id':'chat','parent_message_id':'new-final'}):
        accounts.get_bound_account_identity.return_value='account'
        accounts.get_bound_text_access_token.return_value='token'
        accounts.conversation_binding_lock.return_value=nullcontext()
        service.complete_text(body,on_cursor=Mock())
    backend.set_conversation_archived.assert_called_once_with('chat','previous-answer',False)


def test_uncertain_archive_restore_never_sends_a_new_text_turn(public_chat):
    service=ConversationBindingService();backend=Mock()
    backend._get_conversation.return_value={'current_node':'previous-answer','is_archived':True}
    backend.set_conversation_archived.side_effect=TimeoutError('fixture timeout')
    body={'provider_binding_id':'binding','provider_account_identity':'account',
          'client_conversation_id':'work','conversation_id':'chat','parent_message_id':'previous-answer',
          '_public_session_ref':'s','_request_message_id':'our-user','model':'auto',
          'messages':[{'role':'user','content':'do work'}]}
    from contextlib import nullcontext
    with patch('services.conversation_binding_service.account_service') as accounts, \
         patch('services.conversation_binding_service.OpenAIBackendAPI',return_value=backend), \
         patch('services.conversation_binding_service.conversation_events') as send:
        accounts.get_bound_account_identity.return_value='account'
        accounts.get_bound_text_access_token.return_value='token'
        accounts.conversation_binding_lock.return_value=nullcontext()
        with pytest.raises(ConversationBindingError) as failure:
            service.complete_text(body,on_cursor=Mock())
    assert failure.value.code=='CHAT_ARCHIVE_RESTORE_UNCONFIRMED'
    send.assert_not_called()
    h=public_chat
    h.tasks.runner=Mock(side_effect=ConversationBindingError('restore unavailable',code='CHAT_ARCHIVE_RESTORE_UNCONFIRMED'))
    post(h,turn('same-request'));h.queue.run()
    receipt=h.tasks.read(h.key_a['id'],'same-request')
    assert receipt['status']=='failed' and receipt['upstream_outcome']=='not_sent'


@pytest.mark.parametrize('status', ['queued','running','unknown','not_started','failed'])
def test_previous_not_completed_never_advances_or_creates_a_request(public_chat, status):
    h=public_chat
    assert post(h,turn('r1')).status_code==202
    h.tasks._update(h.key_a['id'],'r1',status=status,error_code='CONVERSATION_OUTCOME_UNKNOWN')
    result=post(h,turn('r2','r1'))
    assert result.status_code==409
    assert result.json()['detail']['code']=='CHAT_PREVIOUS_REQUEST_PENDING'
    assert h.tasks.read(h.key_a['id'],'r2')['status']=='not_found'
    assert len(h.queue.calls)==1


def test_missing_predecessor_stale_predecessor_and_foreign_owner_fail_closed(public_chat):
    h=public_chat
    post(h,turn('r1')); h.queue.run()
    assert post(h,turn('r2')).json()['detail']['code']=='CHAT_PREVIOUS_REQUEST_REQUIRED'
    assert post(h,turn('r2','missing')).status_code==409
    other=h.client.post('/api/chat-requests',headers=h.headers(h.secret_b),json=turn('r2','r1'))
    assert other.status_code==409 and other.json()['detail']['code']=='CHAT_PREVIOUS_REQUEST_NOT_FOUND'
    assert post(h,turn('r2','r1')).status_code==202
    assert post(h,turn('r3','r1')).json()['detail']['code']=='CHAT_CONVERSATION_CONFLICT'
    h.queue.run()
    assert post(h,turn('r3','r2',session='other')).status_code==409
    # Different product/work sessions stay independent, even in the same owner.
    assert post(h,turn('other1',session='other')).status_code==202


def test_completed_legacy_request_can_anchor_without_modifying_old_receipt(public_chat):
    h=public_chat
    assert post(h,request_body('legacy')).status_code==202;h.queue.run()
    with h.tasks._db() as db:
        before=db.execute('SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?',(h.key_a['id'],'legacy')).fetchone()
    h.tasks.recovery_reader=lambda previous:{**previous,'status':'succeeded'}
    assert post(h,turn('first-continuation','legacy')).status_code==202
    h.queue.run()
    sent=h.upstream.call_args.args[0]
    assert sent['conversation_id']=='conversation-secret'
    with h.tasks._db() as db:
        after=db.execute('SELECT request_hash,receipt FROM requests WHERE owner=? AND id=?',(h.key_a['id'],'legacy')).fetchone()
    assert before==after


def test_two_store_instances_cannot_fork_one_predecessor_and_restart_keeps_cursor(public_chat):
    h=public_chat
    post(h,turn('r1'));h.queue.run()
    from api.chat_requests import PublicChatRequest,_payload
    owner=h.key_a['id']
    queues=[QueuedExecutor(),QueuedExecutor()]
    services=[TextTaskService(h.tasks.path, runner=h.upstream,executor=q) for q in queues]
    def submit(n):
        body=turn(f'competing-{n}','r1');obj=PublicChatRequest(**body)
        payload=_payload(owner,obj,[{'role':'user','content':f'next-{n}'}])
        try:
            return services[n].submit(owner,payload)
        except ConversationBindingError as exc:
            return {'error':exc.code}
    with ThreadPoolExecutor(max_workers=2) as executor:
        results=list(executor.map(submit,range(2)))
    assert sum(r.get('status')=='queued' for r in results)==1
    assert sum(r.get('error')=='CHAT_CONVERSATION_CONFLICT' for r in results)==1
    winner=next(r for r in results if r.get('status')=='queued')
    # Real SQLite + fsynced input: recreate service as a second process would.
    restarted=TextTaskService(h.tasks.path,runner=h.upstream,executor=QueuedExecutor())
    with restarted._db() as db:
        receipt=restarted.store.read_receipt(db,'text',owner,winner['request_id'])
        body=restarted.store.load_input(receipt['_input_ref'])
        receipt.update(status='queued',boot=restarted.boot)
        restarted.store.write_receipt(db,'text',owner,winner['request_id'],receipt)
    assert 'provider_binding_id' not in body, 'private cursor is not mutable submitted input'
    restarted._run(owner,winner['request_id'],{**body,'_request_message_id':receipt['request_message_id']})
    assert h.upstream.call_args.args[0]['conversation_id']=='conversation-secret'
    assert h.upstream.call_args.args[0]['parent_message_id']=='parent-secret'


@pytest.mark.parametrize('extra',[
    {'previous_request_id':'previous'}, {'client_conversation_id':None},
    {'client_conversation_id':'..'}, {'client_conversation_id':'x','previous_request_id':None},
    {'client_conversation_id':'x','previous_request_id':'r1'},
    {'client_conversation_id':'x','conversation_id':'foreign-upstream'},
    {'client_conversation_id':'x','provider_binding_id':'admin'},
])
def test_session_reference_validation_does_not_open_arbitrary_cursors(public_chat,extra):
    assert post(public_chat,{**request_body('r1'),**extra}).status_code==422


def test_session_final_user_parent_is_not_batch_submission_root():
    from services.openai_backend_api import OpenAIBackendAPI
    backend=object.__new__(OpenAIBackendAPI)
    backend.access_token='test-only'
    backend.text_request_message_id='final-user'
    calls=[]
    backend.text_cursor_callback=calls.append
    payload=backend._conversation_payload([
        {'role':'user','content':'work'}, {'role':'user','content':'runtime'},
        {'role':'user','content':'controller'},
    ],'auto','UTC',conversation_id='same-chat',parent_message_id='previous-answer')
    assert len(payload['messages'])==3
    assert payload['messages'][-1]['id']=='final-user'
    assert calls==[{'request_parent_message_id':payload['messages'][-2]['id'],
                   '_submission_parent_message_id':'previous-answer'}]


def test_strict_session_uses_exact_original_final_not_latest_chat_answer():
    service=ConversationBindingService()
    backend=Mock()
    backend.text_request_parent_message_id='batch-user-parent'
    backend._get_conversation.return_value={'current_node':'previous-answer','is_archived':False}
    backend.get_conversation_parent_message_id.return_value='unrelated-latest'
    body={'provider_binding_id':'binding','provider_account_identity':'account',
          'client_conversation_id':'work','conversation_id':'chat','parent_message_id':'previous-answer',
          '_public_session_ref':'s','_request_message_id':'our-user','model':'auto',
          'messages':[{'role':'user','content':'do work'}]}
    events=[{'type':'conversation.delta','conversation_id':'chat','delta':'stream prefix'}]
    original={'status':'succeeded','content':'complete original','parent_message_id':'our-final',
              'conversation_id':'chat','provider_binding_id':'binding','provider_account_identity':'account'}
    from contextlib import nullcontext
    with patch('services.conversation_binding_service.account_service') as accounts, \
         patch('services.conversation_binding_service.OpenAIBackendAPI',return_value=backend), \
         patch('services.conversation_binding_service.conversation_events',return_value=iter(events)), \
         patch.object(service,'_read_text_request_result',return_value=original) as read:
        accounts.get_bound_account_identity.return_value='account'
        accounts.get_bound_text_access_token.return_value='token'
        accounts.conversation_binding_lock.return_value=nullcontext()
        result=service.complete_text(body,on_cursor=Mock())
    assert result['parent_message_id']=='our-final' and result['_upstream_terminal'] is True
    assert read.call_args.args[1]['request_message_id']=='our-user'
    backend.get_conversation_parent_message_id.assert_not_called()


def test_legacy_unproven_anchor_does_not_send_a_replacement(public_chat):
    h=public_chat
    post(h,request_body('legacy'));h.queue.run()
    h.tasks.recovery_reader=Mock(return_value={'status':'running'})
    post(h,turn('continuation','legacy'));h.queue.run()
    assert h.upstream.call_count==1
    current=h.tasks.read(h.key_a['id'],'continuation')
    assert current['error_code']=='CHAT_LEGACY_ANCHOR_UNVERIFIED'
    assert current['upstream_outcome']=='not_sent'


@pytest.mark.parametrize('state', ['running', 'unknown'])
def test_sequential_partial_stream_never_becomes_success(state):
    service=ConversationBindingService()
    backend=Mock()
    backend.text_request_parent_message_id='batch-parent'
    backend._get_conversation.return_value={'current_node':'previous-answer','is_archived':False}
    body={'provider_binding_id':'binding','provider_account_identity':'account',
          'client_conversation_id':'work','conversation_id':'chat','parent_message_id':'previous-answer',
          '_public_session_ref':'s','_request_message_id':'our-user','model':'auto',
          'messages':[{'role':'user','content':'do work'}]}
    from contextlib import nullcontext
    with patch('services.conversation_binding_service.account_service') as accounts, \
         patch('services.conversation_binding_service.OpenAIBackendAPI',return_value=backend), \
         patch('services.conversation_binding_service.conversation_events',return_value=iter([
             {'type':'conversation.delta','conversation_id':'chat','delta':'not a complete answer'}])), \
         patch.object(service,'_read_text_request_result',return_value={'status':state}):
        accounts.get_bound_account_identity.return_value='account'
        accounts.get_bound_text_access_token.return_value='token'
        accounts.conversation_binding_lock.return_value=nullcontext()
        with pytest.raises(ConversationBindingError) as error:
            service.complete_text(body,on_cursor=Mock())
    assert error.value.code=='CONVERSATION_OUTCOME_UNKNOWN'
    accounts.mark_text_used.assert_not_called()
    backend.get_conversation_parent_message_id.assert_not_called()


from test.test_company_requests import company, PREFIX, OTHER_CONNECTOR


def test_company_ingress_continuation_keeps_real_owner_and_cursor(company):
    h=company
    h.runner.return_value={'content':'answer','provider_binding_id':'private-binding',
                           'provider_account_identity':'private-account','conversation_id':'private-conversation',
                           'parent_message_id':'private-answer'}
    body=turn('first')
    assert h.client.post(PREFIX+'/api/chat-requests',headers=h.headers(),json=body).status_code==202
    h.queue.run()
    for headers in (h.headers(user='other'),h.headers(org='other'),h.headers(connector=OTHER_CONNECTOR)):
        denied=h.client.post(PREFIX+'/api/chat-requests',headers=headers,json=turn('second','first'))
        assert denied.status_code==409 and denied.json()['detail']['code']=='CHAT_PREVIOUS_REQUEST_NOT_FOUND'
    result=h.client.post(PREFIX+'/api/chat-requests',headers=h.headers(),json=turn('second','first'))
    assert result.status_code==202
    assert not any(secret in result.text for secret in ['private-binding','private-account','private-conversation','private-answer'])
    h.queue.run()
    sent=h.runner.call_args.args[0]
    assert sent['conversation_id']=='private-conversation' and sent['parent_message_id']=='private-answer'


@pytest.mark.parametrize('readback', [
    'missing_request', 'invalid_mapping', 'ambiguous', 'foreign_conversation', 'foreign_parent',
])
def test_sequential_readback_errors_keep_original_unknown_until_get_recovers(tmp_path, readback):
    from services.openai_backend_api import OpenAIBackendAPI
    from services.request_context import current_request
    from test.test_pool_admission import build

    (tmp_path / 'accounts.json').write_text(json.dumps([{
        'access_token': 'fixture-token', 'account_id': 'fixture-upstream',
        'provider_account_identity': 'original-account', 'type': 'Plus', 'status': '正常',
        'quota': 999, 'source_type': 'web', 'conversation_binding_ids': ['original-binding'],
    }]))
    accounts, store, admission = build(tmp_path)
    service = ConversationBindingService()
    tasks = TextTaskService(store.path, runner=service.complete_text,
                            recovery_reader=service.read_text_request, admission=admission,
                            clock=lambda: 1000.0)
    admission.register('text', lambda ctx, body: tasks._run(ctx.owner, ctx.request_id, body))
    backend = object.__new__(OpenAIBackendAPI)
    backend.access_token = 'fixture-token'
    backend.close = Mock()
    backend.get_conversation_parent_message_id = Mock()
    valid_readback = False

    def document(_conversation_id):
        user = backend.text_request_message_id
        answer = {'parent': user, 'message': {
            'id': 'original-final', 'author': {'role': 'assistant'},
            'status': 'finished_successfully', 'end_turn': True, 'channel': 'final',
            'content': {'content_type': 'text', 'parts': ['complete answer']},
        }}
        result = {'conversation_id': 'original-chat', 'mapping': {
            user: {'parent': backend.text_request_parent_message_id,
                   'message': {'id': user, 'author': {'role': 'user'}}},
            'original-final': answer,
        }}
        if not valid_readback:
            if readback == 'missing_request':
                result['mapping'] = {}
            elif readback == 'invalid_mapping':
                result['mapping'] = None
            elif readback == 'ambiguous':
                result['mapping']['other-final'] = {
                    **answer, 'message': {**answer['message'], 'id': 'other-final'},
                }
            elif readback == 'foreign_conversation':
                result['conversation_id'] = 'foreign-chat'
            elif readback == 'foreign_parent':
                result['mapping'][user]['parent'] = 'foreign-parent'
        return result

    backend._get_conversation = Mock(side_effect=document)

    def stream(actual_backend, **kwargs):
        actual_backend._conversation_payload(timezone='UTC', **kwargs)
        current_request.get().before_send()
        yield {'type': 'conversation.delta', 'conversation_id': 'original-chat', 'delta': 'answer'}

    body = {'client_request_id': 'original-request', 'client_conversation_id': 'original-session',
            '_public_session_ref': 'work-session', '_public_route': 'chat', '_text_only_binding': True,
            'model': 'auto', 'messages': [{'role': 'user', 'content': 'original input'}]}
    with patch('services.conversation_binding_service.account_service', accounts), \
         patch.object(accounts, 'get_bound_text_access_token', return_value='fixture-token'), \
         patch('services.conversation_binding_service.OpenAIBackendAPI', return_value=backend), \
         patch('services.conversation_binding_service.conversation_events', side_effect=stream) as sends:
        tasks.submit('owner', body)
        admission.execute(admission.claim_next())
        with store.connect() as db:
            original = store.read_receipt(db, 'text', 'owner', 'original-request')
        assert original['status'] == 'unknown'
        assert original['error_code'] == 'CONVERSATION_OUTCOME_UNKNOWN'
        assert original['original_failure_phase'] == 'cursor_read'
        assert original['original_exception_category'] == 'provider_error'
        assert original['_submission_started'] is True
        assert original['provider_binding_id'] == 'original-binding'
        assert original['provider_account_identity'] == 'original-account'
        assert original['conversation_id'] == 'original-chat'
        assert original['request_parent_message_id'] == backend.text_request_parent_message_id
        assert original.get('parent_message_id') != 'foreign-parent'
        assert admission.resource_snapshot()['chat_turn']['inflight'] == 1

        valid_readback = True
        recovered = tasks.recover('owner', 'original-request')
        assert recovered['status'] == 'succeeded'
        assert recovered['content'] == 'complete answer'
        assert recovered['parent_message_id'] == 'original-final'
        for key in ('request_message_id', 'provider_binding_id', 'provider_account_identity',
                    'client_conversation_id', 'conversation_id', 'request_parent_message_id'):
            assert recovered[key] == original[key]
        assert admission.resource_snapshot()['chat_turn']['inflight'] == 0
        assert tasks.submit('owner', body)['status'] == 'succeeded'
        assert admission.claim_next() is None
        assert sends.call_count == 1
        assert backend._get_conversation.call_count == 2
        backend.get_conversation_parent_message_id.assert_not_called()
