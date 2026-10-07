"""Owner-scoped, committed-only notifications without an upstream dependency."""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api import task_events as events
from services.program_key_policy import make_policy
from services.task_store import TaskStore


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    store = TaskStore(tmp_path / 'events.sqlite3')
    store.initialize()
    identity = {'id': 'one', 'role': 'user', 'policy': make_policy(['chat'], revision=1).to_record()}
    monkeypatch.setattr(events, 'require_identity', lambda *a, **k: identity.copy())
    monkeypatch.setattr(events, 'CHECK_SECONDS', 0.001)
    app = FastAPI()
    service = SimpleNamespace(store=store, admission=None)
    app.state.event_service = service
    for kind in ('image', 'text'):
        app.include_router(events.create_router(kind, lambda: service))
    def put(kind='image', rid='original', owner='one', **fields):
        row = {'model': 'gpt-image-2' if kind == 'image' else 'gpt-5-6-thinking', **fields}
        with store.transaction() as db:
            if kind == 'text':
                # The ordinary acceptance service owns insertion of text rows.
                cols = [r[1] for r in db.execute('PRAGMA table_info(requests)')]
                values = {'owner': owner, 'id': rid, 'receipt': json.dumps(row),
                          'request_hash': 'fixture', 'input_path': '', 'created_at': 1}
                names = [name for name in cols if name in values]
                db.execute('INSERT OR REPLACE INTO requests (' + ','.join(names) + ') VALUES (' + ','.join('?' for _ in names) + ')', [values[name] for name in names])
            else:
                store.write_receipt(db, kind, owner, rid, row)
    with TestClient(app) as client:
        yield client, put, identity, store


def test_current_saved_result_and_reconnect_are_read_only_and_redacted(runtime):
    client, put, _, store = runtime
    put(status='success', data=[{'url': 'private-signed-url'}], access_token='secret')
    for _ in range(2):
        response = client.get('/api/image-tasks/original/events')
        assert response.status_code == 200
        assert 'event: result_ready' in response.text
        assert 'private' not in response.text and 'secret' not in response.text
        assert response.headers['x-accel-buffering'] == 'no'
    with store.connect() as db:
        assert store.read_receipt(db, 'image', 'one', 'original')['status'] == 'success'
    assert client.get('/api/image-tasks/other/events').status_code == 404
    put(owner='other', rid='someone-else', status='success', data=[{}])
    assert client.get('/api/image-tasks/someone-else/events').status_code == 404


def test_running_to_committed_result_and_revocation(runtime, monkeypatch):
    client, put, identity, _ = runtime
    put(status='running', data=[])
    calls = 0
    def authenticate(*a, **k):
        nonlocal calls
        calls += 1
        if calls == 2:
            put(status='success', data=[{'url': 'saved'}])
        return identity.copy()
    monkeypatch.setattr(events, 'require_identity', authenticate)
    result = client.get('/api/image-tasks/original/events')
    assert result.text.index('event: state') < result.text.index('event: result_ready')
    put(status='running', data=[])
    calls = 0
    def revoked(*a, **k):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise HTTPException(401)
        return identity.copy()
    monkeypatch.setattr(events, 'require_identity', revoked)
    result = client.get('/api/image-tasks/original/events')
    assert 'event: access_lost' in result.text and 'event: result_ready' not in result.text


def test_image_stages_emit_state_changes_without_exposing_assets(runtime, monkeypatch):
    client, put, identity, _ = runtime
    put(status='running', upstream_outcome='unknown', conversation_id='original')
    changes = iter([
        {'status': 'running', 'upstream_submission_started': False},
        {'status': 'running', 'upstream_submission_started': True},
        {'status': 'running', 'result_file_ids': ['private-asset']},
        {'status': 'running', 'result_file_ids': ['private-asset'],
         '_execution_timeline': [{'stage': 'upstream_terminal', 'known': True}]},
        {'status': 'running', '_pending_image_output': {'output_ref': 'private-ref', 'coverage': {}}},
        {'status': 'success', 'data': [{'url': 'private-result'}]},
    ])
    calls = 0
    def authenticate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            put(**next(changes))
        return identity.copy()
    monkeypatch.setattr(events, 'require_identity', authenticate)
    response = client.get('/api/image-tasks/original/events')
    payloads = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert [p['result_stage'] for p in payloads] == ['submission_unconfirmed', 'preparing', 'submitted', 'assets_discovered',
        'upstream_finished', 'downloaded_waiting_original_confirmation', 'result_ready']
    assert all(p['result_ready'] is False for p in payloads[:-1])
    assert response.text.count('event: state\n') == 6
    assert response.text.count('event: result_ready\n') == 1
    assert 'private-' not in response.text


def test_policy_change_and_bounded_wait(runtime, monkeypatch):
    client, put, identity, _ = runtime
    put(status='running', data=[])
    monkeypatch.setattr(events, 'STREAM_SECONDS', 0)
    result = client.get('/api/image-tasks/original/events')
    assert 'event: reconnect' in result.text and 'event: result_ready' not in result.text
    identity['policy'] = make_policy(['codex'], revision=2).to_record()
    assert client.get('/api/image-tasks/original/events').status_code == 403


@pytest.mark.parametrize('fields,expected', [
    ({}, 'upstream_finished'),
    ({'_send_sequence': 2}, 'assets_discovered'),
    ({'_expected_sends': 2}, 'assets_discovered'),
    ({'_recovery_paused': True}, 'needs_attention'),
    ({'_recovery_suppressed': True}, 'needs_attention'),
    ({'_execution_timeline': [{'stage': 'upstream_terminal', 'known': False}]}, 'assets_discovered'),
])
def test_terminal_observation_is_current_progress_only(runtime, fields, expected):
    _, put, identity, store = runtime
    row = {'status': 'running', 'result_file_ids': ['private-asset'],
           '_execution_timeline': [{'stage': 'upstream_terminal', 'known': True}], **fields}
    put(**row)
    snapshot = events._snapshot('image', SimpleNamespace(store=store), identity, 'original')
    assert snapshot['result_stage'] == expected
    assert snapshot['result_ready'] is False and snapshot['result_count'] == 0
    with store.connect() as db:
        unchanged = store.read_receipt(db, 'image', identity['id'], 'original')
    assert unchanged == {'model': 'gpt-image-2', **row}


@pytest.mark.parametrize('kind,status,body', [('image', 'success', {'data': []}),
    ('text', 'succeeded', {'content': '  '}), ('image', 'unknown', {})])
def test_empty_or_unknown_is_attention_not_success(runtime, kind, status, body):
    client, put, _, _ = runtime
    put(kind=kind, status=status, **body)
    prefix = 'image-tasks' if kind == 'image' else 'chat-requests'
    result = client.get('/api/' + prefix + '/original/events')
    assert 'event: needs_attention' in result.text and 'event: result_ready' not in result.text


def test_text_saved_result_is_not_exposed_in_notification(runtime):
    client, put, _, _ = runtime
    put(kind='text', status='succeeded', content='confidential result')
    result = client.get('/api/chat-requests/original/events')
    assert 'event: result_ready' in result.text and 'confidential' not in result.text


def test_policy_reduction_during_stream_never_releases_ready_result(runtime, monkeypatch):
    client, put, identity, _ = runtime
    put(status='running', data=[])
    calls = 0
    def authenticate(*a, **k):
        nonlocal calls
        calls += 1
        if calls > 1:
            put(status='success', data=[{'url': 'saved'}])
            identity['policy'] = make_policy(['codex'], revision=2).to_record()
        return identity.copy()
    monkeypatch.setattr(events, 'require_identity', authenticate)
    result = client.get('/api/image-tasks/original/events')
    assert 'event: access_lost' in result.text and 'event: result_ready' not in result.text


@pytest.mark.parametrize('extra', [
    {'upstream_unfinished': True},
    {'upstream_unfinished': False, 'result_file_ids': ['original-file'], '_attempt_finished_at': 1},
    {'upstream_unfinished': False, 'result_file_ids': ['original-file'], '_attempt_finished_at': 1, 'error_code': 'RESULT_UNRECOVERABLE'},
    {'upstream_unfinished': False, '_pending_image_result_ids': {'file_ids': ['pending-file'], 'sediment_ids': []}},
])
def test_original_recovery_error_keeps_waiting_until_result_is_saved(runtime, monkeypatch, extra):
    client, put, identity, _ = runtime
    put(**{'status': 'error', 'error_code': 'CONVERSATION_OUTCOME_UNKNOWN', 'conversation_id': 'conversation',
           'request_message_id': 'message', 'next_poll_at': 9999999999, **extra})
    calls = 0
    def authenticate(*a, **k):
        nonlocal calls
        calls += 1
        if calls == 2:
            put(status='success', data=[{'url': 'private-result'}])
        return identity.copy()
    monkeypatch.setattr(events, 'require_identity', authenticate)
    response = client.get('/api/image-tasks/original/events')
    assert 'event: needs_attention' not in response.text
    assert '"recovering_original":true' in response.text
    assert 'event: result_ready' in response.text
    assert 'private-result' not in response.text and 'original-file' not in response.text


@pytest.mark.parametrize('extra', [
    {'_recovery_paused': True}, {'_recovery_suppressed': True},
    {'_attempt_finished_at': 1}, {'request_message_id': ''},
    {'error_code': 'RESULT_UNRECOVERABLE'},
])
def test_stopped_or_ineligible_original_recovery_still_needs_attention(runtime, extra):
    client, put, _, _ = runtime
    fields = {'status': 'error', 'error_code': 'CONVERSATION_OUTCOME_UNKNOWN',
              'conversation_id': 'conversation', 'request_message_id': 'message',
              'upstream_unfinished': True, **extra}
    put(**fields)
    response = client.get('/api/image-tasks/original/events')
    assert 'event: needs_attention' in response.text
    assert 'event: result_ready' not in response.text


@pytest.mark.parametrize("kind,status,code", [("image", "error", "IMAGE_RESOURCE_UNAVAILABLE"),
    ("text", "failed", "CONVERSATION_BINDING_UNAVAILABLE"),
    ("text", "unknown", "CONVERSATION_OUTCOME_UNKNOWN")])
def test_claimed_unsent_resource_failure_waits_for_original_requeue(runtime, monkeypatch, kind, status, code):
    client, put, identity, _ = runtime
    put(kind=kind, status=status, error_code=code,
        _submission_started=False, _executing=True, _claim_id="original-claim",
        _claim_until=9999999999)
    calls = 0
    def authenticate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            put(kind=kind, status="queued", _submission_started=False)
        elif calls == 3:
            put(kind=kind, **({"status": "success", "data": [{"url": "private"}]}
                             if kind == "image" else {"status": "succeeded", "content": "private"}))
        return identity.copy()
    monkeypatch.setattr(events, "require_identity", authenticate)
    prefix = "image-tasks" if kind == "image" else "chat-requests"
    response = client.get("/api/" + prefix + "/original/events")
    assert "event: needs_attention" not in response.text
    assert '"retrying_unsent":true' in response.text
    assert "event: result_ready" in response.text
    assert "original-claim" not in response.text and "private" not in response.text


@pytest.mark.parametrize("extra", [
    {"_submission_started": True}, {"_executing": False}, {"_claim_id": None},
    {"_claim_until": 1}, {"_recovery_paused": True}, {"_recovery_suppressed": True},
    {"_attempt_finished_at": 1}, {"_submission_started": None},
    {"error_code": "TASK_INPUT_UNAVAILABLE"},
])
def test_unsent_wait_never_hides_finished_or_paused_failure(runtime, extra):
    client, put, _, _ = runtime
    fields = {"status": "error", "error_code": "CONVERSATION_BINDING_UNAVAILABLE",
              "_submission_started": False, "_executing": True,
              "_claim_id": "claim", "_claim_until": 9999999999, **extra}
    put(**fields)
    response = client.get("/api/image-tasks/original/events")
    assert "event: needs_attention" in response.text
    assert "event: result_ready" not in response.text


def unsent_automatic_image():
    return {'status': 'error', 'error_code': 'RESULT_UNRECOVERABLE',
        '_automatic_generation_recovery': True, 'recovery_retryable': True,
        '_submission_started': False, 'upstream_submission_started': False,
        'upstream_unfinished': False, 'upstream_outcome': 'not_submitted'}


@pytest.mark.parametrize('initial', [{}, {'_executing': True, '_claim_id': 'claim', '_claim_until': 9999999999},
    {'_completion': {'state': 'checking_original', 'automatic_failure_retry': True, 'next_at': 1}}])
def test_automatic_unsent_notification_survives_all_same_id_handoff_windows(runtime, monkeypatch, initial):
    client, put, identity, _ = runtime
    client.app.state.event_service.admission = SimpleNamespace(generation_completion=object())
    original = unsent_automatic_image()
    put(**{**original, **initial})
    changes = iter([
        original,
        {**original, '_completion': {'state': 'checking_original', 'automatic_failure_retry': True, 'next_at': 1}},
        {'status': 'queued', '_completion': {'same_request_retry': True}},
        {'status': 'success', 'data': [{'url': 'private-result'}]},
    ])
    calls = 0
    def authenticate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1: put(**next(changes))
        return identity.copy()
    monkeypatch.setattr(events, 'require_identity', authenticate)
    result = client.get('/api/image-tasks/original/events')
    assert 'event: needs_attention' not in result.text
    assert '"retrying_unsent":true' in result.text
    assert 'event: result_ready' in result.text
    assert 'private-result' not in result.text
    payloads = [json.loads(line[6:]) for line in result.text.splitlines() if line.startswith('data: ')]
    assert all(p['result_stage'] == 'preparing' for p in payloads if p['retrying_unsent'])


@pytest.mark.parametrize('extra', [
    {'_automatic_generation_recovery': None}, {'recovery_retryable': False},
    {'_submission_started': True}, {'_submission_started': None},
    {'upstream_submission_started': True}, {'upstream_submission_started': None},
    {'upstream_outcome': 'unknown'}, {'upstream_unfinished': True},
    {'_recovery_paused': True}, {'_recovery_suppressed': True}, {'_attempt_finished_at': 1},
    {'_completion_of': 'original-parent'}, {'result_file_ids': ['private-asset']},
    {'_pending_image_result_ids': {'file_ids': ['private-asset']}},
    {'_completion': {'state': 'needs_attention'}},
    {'_completion': {'state': 'checking_original', 'automatic_failure_retry': True, 'next_at': 1, 'same_request_retry': True}},
    {'_completion': {'state': 'checking_original', 'automatic_failure_retry': False, 'next_at': 1}},
    {'_completion': {'state': 'checking_original', 'automatic_failure_retry': True}},
    {'_completion': {'state': 'checking_original', 'automatic_failure_retry': True, 'next_at': 1, 'replacement_id': 'other'}},
    {'_work_key': 'missing-work'},
])
def test_automatic_unsent_wait_does_not_hide_terminal_or_ambiguous_result(runtime, extra):
    client, put, _, _ = runtime
    client.app.state.event_service.admission = SimpleNamespace(generation_completion=object())
    put(**{**unsent_automatic_image(), **extra})
    result = client.get('/api/image-tasks/original/events')
    assert 'event: needs_attention' in result.text
    assert '"retrying_unsent":true' not in result.text
    assert 'event: result_ready' not in result.text
    assert '"result_stage":"needs_attention"' in result.text


def test_no_completion_scheduler_does_not_promise_automatic_unsent_retry(runtime):
    client, put, _, _ = runtime
    put(**unsent_automatic_image())
    result = client.get('/api/image-tasks/original/events')
    assert 'event: needs_attention' in result.text
    assert '"retrying_unsent":true' not in result.text


@pytest.mark.parametrize('state,waiting', [('active', True), ('paused', False), ('completed', False)])
def test_automatic_unsent_wait_respects_original_work_state(runtime, monkeypatch, state, waiting):
    client, put, _, store = runtime
    client.app.state.event_service.admission = SimpleNamespace(generation_completion=object())
    put(**{**unsent_automatic_image(), '_work_key': 'original-work'})
    with store.transaction() as db:
        store.set_runtime(db, 'original-work', {'state': state})
    monkeypatch.setattr(events, 'STREAM_SECONDS', 0)
    result = client.get('/api/image-tasks/original/events')
    assert ('event: needs_attention' in result.text) is not waiting
    assert ('"retrying_unsent":true' in result.text) is waiting
    assert 'event: result_ready' not in result.text


def completion_rows():
    root = {'id': 'original', 'status': 'error', 'error_code': 'NO_IMAGE_GENERATED',
            'upstream_outcome': 'unknown', 'upstream_unfinished': False,
            'provider_binding_id': 'binding', 'provider_account_identity': 'account',
            'client_conversation_id': 'client', 'conversation_id': 'conversation',
            'request_message_id': 'message', '_work_key': 'work',
            '_image_thread': {'protocol': 'image-thread-v1', 'id': 'thread'},
            '_retry_cursor': {'conversation_id': 'conversation', 'request_message_id': 'message',
                              'retry_parent_message_id': 'parent', 'observed_at': events.time.time(),
                              'source': 'terminal_image_failure'},
            '_completion': {'state': 'checking_original', 'automatic_failure_retry': True, 'next_at': 9999999999}}
    child = {k: root[k] for k in ('provider_binding_id', 'provider_account_identity',
             'client_conversation_id', 'conversation_id', '_work_key', '_image_thread')}
    child.update(id='child', status='queued', _completion_of='original', _same_session_retry_of='original',
                 _submission_parent_message_id='parent')
    return root, child


def test_automatic_completion_notifies_selected_result_without_ending_wait_early(runtime, monkeypatch):
    client, put, identity, store = runtime
    client.app.state.event_service.admission = SimpleNamespace(generation_completion=object())
    root, child = completion_rows()
    put(**root)
    with store.transaction() as db:
        store.set_runtime(db, 'work', {'state': 'active'})
    calls = 0
    def authenticate(*a, **k):
        nonlocal calls
        calls += 1
        if calls == 2:
            root['_completion'].update(state='replacement_pending', replacement_id='child')
            put(**root); put(rid='child', **child)
        elif calls == 3:
            root['_completion'].update(state='result_ready', selected_id='child', next_at=None)
            child.update(status='success', data=[{'url': 'private-result'}], _image_thread_terminal=True)
            put(**root); put(rid='child', **child)
        return identity.copy()
    monkeypatch.setattr(events, 'require_identity', authenticate)
    response = client.get('/api/image-tasks/original/events')
    assert 'event: needs_attention' not in response.text
    assert response.text.count('event: result_ready\n') == 1
    payloads = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert all(p['request_id'] == 'original' and p['status'] == 'error' for p in payloads)
    assert [p['result_ready'] for p in payloads] == [False, False, True]
    assert payloads[-1]['selected_result_id'] == 'child' and payloads[-1]['result_count'] == 1
    assert 'private-result' not in response.text and 'binding' not in response.text


def test_completion_notification_waits_across_reserved_child_submit(runtime, monkeypatch):
    client, put, identity, store = runtime
    client.app.state.event_service.admission = SimpleNamespace(generation_completion=object())
    root, child = completion_rows()
    root['_completion'].update(state='replacement_pending', replacement_id='child',
                               prepared_input='private-input-path')
    put(**root)
    with store.transaction() as db:
        store.set_runtime(db, 'work', {'state': 'active'})
    calls = 0
    def authenticate(*a, **k):
        nonlocal calls
        calls += 1
        if calls == 2:
            put(rid='child', **child)
        elif calls == 3:
            root['_completion'].update(state='result_ready', selected_id='child', next_at=None)
            child.update(status='success', data=[{'url': 'private-result'}], _image_thread_terminal=True)
            put(**root); put(rid='child', **child)
        return identity.copy()
    monkeypatch.setattr(events, 'require_identity', authenticate)
    response = client.get('/api/image-tasks/original/events')
    assert calls == 3
    assert 'event: needs_attention' not in response.text
    assert '"recovering_original":true' in response.text
    assert response.text.count('event: result_ready\n') == 1
    assert '"selected_result_id":"child"' in response.text
    assert 'private-' not in response.text


@pytest.mark.parametrize('change', ['proof_missing_source', 'proof_stale', 'proof_wrong_binding',
                                    'result_asset', 'upstream_unfinished'])
def test_scheduled_original_investigation_waits_until_completion_stops(runtime, monkeypatch, change):
    client, put, identity, store = runtime
    client.app.state.event_service.admission = SimpleNamespace(generation_completion=object())
    root, _ = completion_rows()
    if change == 'proof_missing_source': root['_retry_cursor'].pop('source')
    if change == 'proof_stale': root['_retry_cursor']['observed_at'] -= 301
    if change == 'proof_wrong_binding': root['_retry_cursor']['conversation_id'] = 'other'
    if change == 'result_asset': root['result_file_ids'] = ['existing-asset']
    if change == 'upstream_unfinished': root['upstream_unfinished'] = True
    with store.transaction() as db:
        store.set_runtime(db, 'work', {'state': 'active'})
    # Without a scheduled completion, invalid proof must not invent one.
    unscheduled = {k: v for k, v in root.items() if k != '_completion'}
    unscheduled['_automatic_generation_recovery'] = True
    put(**unscheduled)
    assert events._snapshot('image', client.app.state.event_service, identity, 'original').get('completion_state') is None
    put(**root)
    calls = 0
    def authenticate(*a, **k):
        nonlocal calls
        calls += 1
        if calls == 2:
            root['_completion'].update(state='needs_attention', next_at=None)
            put(**root)
        return identity.copy()
    monkeypatch.setattr(events, 'require_identity', authenticate)
    response = client.get('/api/image-tasks/original/events')
    assert calls == 2
    assert '"completion_state":"checking_original"' in response.text
    assert 'event: needs_attention' in response.text
    assert 'event: result_ready' not in response.text


@pytest.mark.parametrize('change', ['no_prepared_input', 'wrong_binding', 'wrong_thread', 'wrong_lineage', 'child_paused'])
def test_reserved_completion_does_not_hide_invalid_child(runtime, monkeypatch, change):
    client, put, _, store = runtime
    client.app.state.event_service.admission = SimpleNamespace(generation_completion=object())
    root, child = completion_rows()
    root['_completion'].update(state='replacement_pending', replacement_id='child', prepared_input='private-input')
    if change == 'no_prepared_input': root['_completion'].pop('prepared_input')
    else:
        if change == 'wrong_binding': child['provider_binding_id'] = 'other'
        if change == 'wrong_thread': child['_image_thread'] = {'protocol': 'image-thread-v1', 'id': 'other'}
        if change == 'wrong_lineage': child['_completion_of'] = 'other'
        if change == 'child_paused': child['_recovery_paused'] = True
        put(rid='child', **child)
    put(**root)
    with store.transaction() as db:
        store.set_runtime(db, 'work', {'state': 'active'})
    monkeypatch.setattr(events, 'STREAM_SECONDS', 0)
    response = client.get('/api/image-tasks/original/events')
    assert 'event: needs_attention' in response.text
    assert 'event: result_ready' not in response.text


def test_verified_no_image_failure_waits_across_completion_creation_transaction(runtime, monkeypatch):
    client, put, _, store = runtime
    client.app.state.event_service.admission = SimpleNamespace(generation_completion=object())
    root, _ = completion_rows()
    root.pop('_completion')
    root['_automatic_generation_recovery'] = True
    root['_retry_cursor'].update(source='terminal_image_failure', observed_at=events.time.time())
    put(**root)
    with store.transaction() as db:
        store.set_runtime(db, 'work', {'state': 'active'})
    monkeypatch.setattr(events, 'STREAM_SECONDS', 0)
    response = client.get('/api/image-tasks/original/events')
    assert 'event: needs_attention' not in response.text
    assert 'event: reconnect' in response.text
    assert '"recovering_original":true' in response.text
    assert 'event: result_ready' not in response.text


@pytest.mark.parametrize('change', ['paused', 'work_paused', 'scheduler_missing', 'no_next_at', 'nan_next_at',
                                    'bool_next_at', 'attempt_finished', 'no_auto_retry',
                                    'selected_wrong_binding', 'selected_wrong_thread', 'selected_not_terminal'])
def test_invalid_or_stopped_completion_does_not_promise_a_result(runtime, monkeypatch, change):
    client, put, _, store = runtime
    client.app.state.event_service.admission = SimpleNamespace(generation_completion=object())
    root, child = completion_rows()
    if change == 'paused': root['_recovery_paused'] = True
    if change == 'scheduler_missing': client.app.state.event_service.admission = None
    if change == 'no_next_at': root['_completion']['next_at'] = None
    if change == 'nan_next_at': root['_completion']['next_at'] = float('nan')
    if change == 'bool_next_at': root['_completion']['next_at'] = True
    if change == 'attempt_finished': root['_attempt_finished_at'] = events.time.time()
    if change == 'no_auto_retry': root['_completion']['automatic_failure_retry'] = False
    if change.startswith('selected_'):
        root['_completion'].update(state='result_ready', selected_id='child', replacement_id='child', next_at=None)
        child.update(status='success', data=[{'url': 'private-result'}], _image_thread_terminal=True)
        if change == 'selected_wrong_binding': child['provider_binding_id'] = 'different'
        if change == 'selected_wrong_thread': child['_image_thread'] = {'protocol': 'image-thread-v1', 'id': 'different'}
        if change == 'selected_not_terminal': child['_image_thread_terminal'] = False
        put(rid='child', **child)
    with store.transaction() as db:
        store.set_runtime(db, 'work', {'state': 'paused' if change == 'work_paused' else 'active'})
    put(**root)
    monkeypatch.setattr(events, 'STREAM_SECONDS', 0)
    response = client.get('/api/image-tasks/original/events')
    assert 'event: needs_attention' in response.text
    assert 'event: result_ready' not in response.text
