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
    for kind in ('image', 'text'):
        app.include_router(events.create_router(kind, lambda: SimpleNamespace(store=store)))
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


def test_policy_change_and_bounded_wait(runtime, monkeypatch):
    client, put, identity, _ = runtime
    put(status='running', data=[])
    monkeypatch.setattr(events, 'STREAM_SECONDS', 0)
    result = client.get('/api/image-tasks/original/events')
    assert 'event: reconnect' in result.text and 'event: result_ready' not in result.text
    identity['policy'] = make_policy(['codex'], revision=2).to_record()
    assert client.get('/api/image-tasks/original/events').status_code == 403


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
