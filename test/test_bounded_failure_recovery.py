"""Controlled send/query failures: no real upstream traffic or quota use."""
import copy
import io
import json
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

from services.generation_completion import GenerationCompletionService, retry_cursor
from services.conversation_binding_service import ConversationBindingError, ConversationBindingService
from test.test_generation_completion import setup, row, patch_row, start, IDENTITY
from test.test_unknown_turn_recovery import document
from test.test_external_image_client import image_client as cli


@pytest.mark.parametrize('envelope', ['recovery', 'detail'])
def test_client_recognizes_both_explicit_not_sent_envelopes(envelope):
    exc = HTTPError('http://localhost/api/chat-requests', 429, 'busy', {'Retry-After': '2'},
                    io.BytesIO(json.dumps({envelope: {'code': 'CAPACITY', 'upstream_outcome': 'not_sent'}}).encode()))
    cli._http_error_detail(exc)
    assert exc.pool_not_sent and exc.pool_retry_after == 2
    unproven = HTTPError('http://localhost/api/chat-requests', 429, 'busy', {}, io.BytesIO(b'{"detail":"busy"}'))
    cli._http_error_detail(unproven)
    assert not unproven.pool_not_sent


def test_client_preserves_http_date_cooldown(monkeypatch):
    from email.utils import formatdate
    monkeypatch.setattr(cli.time, 'time', lambda: 1700000000)
    exc = HTTPError('http://localhost/api/chat-requests', 429, 'busy',
                    {'Retry-After': formatdate(1700000120, usegmt=True)},
                    io.BytesIO(b'{"detail":{"upstream_outcome":"not_sent"}}'))
    cli._http_error_detail(exc)
    assert exc.pool_not_sent and exc.pool_retry_after == 120


@pytest.mark.parametrize('kind', ['text', 'image'])
def test_client_retries_definite_transport_rejection_same_id_only_once(tmp_path, monkeypatch, kind):
    path = tmp_path / 'state.json'
    args = cli._parser().parse_args([
        'chat-submit' if kind == 'text' else 'submit', '--state', str(path),
        '--request-id' if kind == 'text' else '--client-task-id', 'original', '--model', 'fixture', '--prompt', 'short sample'])
    posts = []
    def call(method, endpoint, **kwargs):
        if method == 'GET':
            if kind == 'text':
                raise cli.HttpFailure(404, 'CHAT_REQUEST_NOT_FOUND')
            return {'items': [], 'missing_ids': ['original']}
        assert json.loads(path.read_text())['phase'] == 'unknown'
        posts.append(copy.deepcopy(kwargs))
        if len(posts) <= 2:
            raise cli.HttpFailure(429, 'busy', not_sent=True, retry_after=0)
        return {'request_id' if kind == 'text' else 'id': 'original', 'status': 'queued', 'route': 'chat'}
    api = Mock(); api.json.side_effect = call
    submit = cli._command_chat_submit if kind == 'text' else cli._command_submit
    with pytest.raises(cli.HttpFailure):
        submit(api, args)
    assert len(posts) == 2 and posts[0] == posts[1]
    assert json.loads(path.read_text())['phase'] == 'not_sent'
    submit(api, args)
    assert len(posts) == 3 and posts[2] == posts[0]
    assert json.loads(path.read_text())['phase'] == 'accepted'


@pytest.mark.parametrize('failure', ['timeout', 'unproven429', 'crash'])
def test_transport_ambiguity_and_404_never_authorize_resend(tmp_path, failure):
    path = tmp_path / 'state.json'
    args = cli._parser().parse_args(['chat-submit', '--state', str(path), '--request-id', 'original',
                                    '--model', 'fixture', '--prompt', 'short sample'])
    failure_error = {'timeout': cli.ClientError('timeout'), 'unproven429': cli.HttpFailure(429, 'busy'),
                     'crash': KeyboardInterrupt()}[failure]
    api = Mock(); api.json.side_effect = failure_error
    with pytest.raises((cli.ClientError, KeyboardInterrupt)):
        cli._command_chat_submit(api, args)
    assert json.loads(path.read_text())['phase'] == 'unknown'
    api.json.side_effect = cli.HttpFailure(404, 'CHAT_REQUEST_NOT_FOUND')
    with pytest.raises(cli.HttpFailure):
        cli._command_chat_submit(api, args)
    assert [c.args[0] for c in api.json.call_args_list] == ['POST', 'GET']


def test_persisted_transport_cooldown_blocks_early_retry(tmp_path):
    api = Mock(); api.json.side_effect = cli.HttpFailure(429, 'busy', not_sent=True, retry_after=120)
    path = tmp_path / 'state.json'; state = {'phase': 'prepared'}
    with pytest.raises(cli.ClientError, match='cooldown'):
        cli._submit_original(api, path, state, '/api/chat-requests', payload={'client_request_id': 'same'})
    assert api.json.call_count == 1 and state['phase'] == 'not_sent'
    with pytest.raises(cli.ClientError, match='cooldown'):
        cli._submit_original(api, path, state, '/api/chat-requests', payload={'client_request_id': 'same'})
    assert api.json.call_count == 1


@pytest.mark.parametrize('case', ['good', 'partial', 'later_user', 'branch', 'archived', 'missing', 'malformed', 'image'])
def test_retry_cursor_requires_empty_exact_unbranched_original(case):
    receipt = {'conversation_id': 'conversation', 'request_message_id': 'user', 'request_parent_message_id': 'parent'}
    doc = document(receipt); doc['is_archived'] = False
    msg = doc['mapping']['final-user']['message']
    msg.update(status='in_progress', end_turn=None)
    if case == 'partial': msg['content']['parts'] = ['retained partial answer']
    if case == 'later_user': msg['author']['role'] = 'user'
    if case == 'branch': doc['mapping']['sibling'] = copy.deepcopy(doc['mapping']['final-user'])
    if case == 'archived': doc['is_archived'] = True
    if case == 'missing': del doc['mapping']['user']
    if case == 'malformed': msg['author'] = 'invalid'
    if case == 'image': msg['content']['parts'] = [{'asset_pointer': 'file-service://result'}]
    assert bool(retry_cursor(doc, receipt)) == (case == 'good')


def test_new_failure_automatically_retries_once_and_exhaustion_releases_only_its_work(setup):
    service, admission, calls = setup
    original = row(service)
    patch_row(service, _automatic_generation_recovery=True)
    service.process_one()
    child_id = service.read('text', IDENTITY, 'old-0')['replacement_id']
    assert row(service)['_attempt_finished_at']
    child = row(service, request_id=child_id)
    assert child['_work_key'] == original['_work_key']
    # Model accepts the child but produces no result. A later worker restarts.
    def fail(body, on_cursor):
        from services.request_context import current_request
        current_request.get().before_send(); calls.append(body)
        raise ConnectionError('controlled response loss')
    service.text.runner = fail
    admission.execute(admission.claim_next())
    patch_row(service, request_id=child_id, _execution_timeline=[{'stage': 'send_call_started', 'at': 10}])
    admission.clock.now += 31
    restarted = GenerationCompletionService(service.text, service.images, service.lifecycle, clock=admission.clock)
    restarted.process_one()
    result = restarted.read('text', IDENTITY, 'old-0')
    assert result['reason'] == 'COMPLETION_ATTEMPT_EXHAUSTED' and result['next_at'] is None
    assert result['local_reservation'] == 'released'
    assert row(service)['status'] == row(service, request_id=child_id)['status'] == 'unknown'
    with service.store.connect() as db:
        assert not service.store.runtime(db, original['_work_key'])['slot_held']
    for _ in range(3):
        admission.clock.now += 31; restarted.process_one()
    assert len(calls) == 1
    assert admission.resource_snapshot()['chat_turn']['inflight'] == 0
    before = service.text.recovery_reader
    service.text.recovery_reader = Mock(side_effect=AssertionError('retired attempts must not poll'))
    service.text.read('owner', 'old-0'); service.text.read('owner', child_id)
    assert service.text.recovery_reader.call_count == 0


def test_failed_query_ends_bounded_investigation_without_fabricating_not_sent(setup):
    service, admission, calls = setup
    patch_row(service, _automatic_generation_recovery=True)
    service.text.recovery_reader = Mock(side_effect=ConnectionError('controlled read failure'))
    service.process_one()
    result = service.read('text', IDENTITY, 'old-0')
    assert result['reason'] == 'COMPLETION_ORIGINAL_READ_UNAVAILABLE'
    assert result['original_attempt_state'] == 'ended' and result['local_reservation'] == 'released'
    assert row(service)['upstream_outcome'] == 'unknown' and not calls
    with service.store.connect() as db:
        assert not service.store.runtime(db, row(service)['_work_key'])['slot_held']


def test_pause_neither_retires_child_nor_consumes_its_retry(setup):
    service, admission, calls = setup
    child_id = start(service)['replacement_id']
    original = row(service)
    with service.store.transaction() as db:
        work = service.store.runtime(db, original['_work_key']); work['state'] = 'paused'
        service.store.set_runtime(db, work['key'], work)
    admission.clock.now += 31; service.process_one()
    assert not row(service, request_id=child_id).get('_attempt_finished_at')
    assert not calls and admission.claim_next() is None
    with service.store.transaction() as db:
        work = service.store.runtime(db, original['_work_key']); work['state'] = 'active'
        service.store.set_runtime(db, work['key'], work)
    admission.clock.now += 31; service.process_one()
    admission.execute(admission.claim_next())
    assert service.read('text', IDENTITY, 'old-0')['selected_id'] == child_id


@pytest.mark.parametrize('sent', [False, True])
def test_binding_error_classification_rechecks_send_marker_in_update_transaction(setup, sent):
    service, admission, calls = setup
    patch_row(service, _submission_started=sent, _execution_wait_ended_at=None)
    service.text._update('owner', 'old-0', classify_before_send=True, status='failed', error_code='BINDING_UNAVAILABLE')
    saved = row(service)
    assert saved['upstream_outcome'] == ('unknown' if sent else 'not_sent')


def test_paused_completion_does_not_starve_another_due_root(setup):
    service, admission, _ = setup
    first = row(service)
    state = {'state': 'checking_original', 'allow_unconfirmed_retry': True, 'max_extra_requests': 1, 'next_at': 0}
    patch_row(service, _completion=state, _recovery_paused=True)
    with service.store.transaction() as db:
        second = {**first, 'request_id': 'second', '_completion': copy.deepcopy(state)}
        db.execute('INSERT INTO requests VALUES(?,?,?,?)', ('owner', 'second', 'fixture', json.dumps(second)))
    service.advance = Mock()
    service.process_one()
    service.advance.assert_called_once_with('text', 'owner', 'second')
    patch_row(service, _recovery_paused=False)
    service.advance.reset_mock(); service.process_one()
    service.advance.assert_called_once_with('text', 'owner', 'old-0')


def test_known_unsent_exhaustion_releases_local_work_without_clearing_receipt(setup):
    service, admission, _ = setup
    old = row(service)
    patch_row(service, status='failed', upstream_outcome='not_sent', _submission_started=False,
              error_code='INVALID_INPUT', _automatic_generation_recovery=True)
    with service.store.transaction() as db:
        service.store.set_runtime(db, 'unrelated-work', {'slot_held': True, 'state': 'paused'})
    service.process_one()
    result = service.read('text', IDENTITY, 'old-0')
    assert result['reason'] == 'COMPLETION_ORIGINAL_NOT_RETRYABLE'
    assert result['local_reservation'] == 'released'
    assert row(service)['_input_ref'] == old['_input_ref']
    with service.store.connect() as db:
        assert service.store.runtime(db, 'unrelated-work') == {'slot_held': True, 'state': 'paused'}
        assert not service.store.runtime(db, old['_work_key'])['slot_held']


def test_verified_late_result_at_final_retry_read_is_saved_even_after_local_retirement(setup):
    service, admission, calls = setup
    child_id = start(service)['replacement_id']
    def late(receipt):
        doc = document(receipt); doc['is_archived'] = False
        doc['mapping'][doc['current_node']]['message']['content']['parts'] = ['actual late original result']
        return ConversationBindingService._read_text_request_result(None, receipt, document=doc)
    service.text.recovery_reader = late
    admission.execute(admission.claim_next())
    result = service.read('text', IDENTITY, 'old-0')
    assert result['selected_id'] == 'old-0'
    assert result['result']['content'] == 'actual late original result'
    assert row(service, request_id=child_id)['upstream_outcome'] == 'not_sent' and not calls
    assert result['work']['request_id'] == 'old-0'
