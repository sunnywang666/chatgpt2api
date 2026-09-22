import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from scripts.audit_recovery_scan import audit, summarize


def database(path, row):
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE requests(owner TEXT, id TEXT, receipt TEXT)')
        db.execute('INSERT INTO requests VALUES(?,?,?)', ('owner', 'original', json.dumps(row)))


def test_audit_keeps_receipt_bytes_and_suppression_and_filters_secrets(tmp_path):
    path = tmp_path / 'original.sqlite3'
    row = {'status': 'unknown', '_recovery_suppressed': True, 'content': 'DO_NOT_PRINT',
           'access_token': 'DO_NOT_PRINT', 'request_message_id': 'PRIVATE_MESSAGE',
           'provider_account_identity': 'PRIVATE_ACCOUNT',
           '_recovery_conversation_scan': {'conversation_ids': ['PRIVATE_CHAT'], 'next_index': 0,
                'next_offset': 20, 'coverage_complete': True, 'matches': [],
                'failed_reads': {'PRIVATE_CHAT': {'attempts': 3, 'next_at': 1200,
                     'error': {'category': 'http', 'http_status': 404, 'body': 'DO_NOT_PRINT'}}}}}
    database(path, row)
    before = path.read_bytes()
    report = audit(path, 'owner', 'original')
    assert report['recovery_suppressed'] is True and report['status'] == 'unknown'
    assert report['scan']['unread_in_window'] == 1
    assert report['scan']['candidates'][0]['http_status'] == 404
    assert report['scan']['candidates'][0]['candidate_ref'] == hashlib.sha256(b'PRIVATE_CHAT').hexdigest()[:16]
    assert path.read_bytes() == before
    assert all(v not in json.dumps(report) for v in ('DO_NOT_PRINT', 'PRIVATE_CHAT', 'PRIVATE_ACCOUNT', 'PRIVATE_MESSAGE'))


def test_audit_missing_receipt_and_parameterized_identity(tmp_path):
    path = tmp_path / 'original.sqlite3'
    database(path, {'status': 'succeeded'})
    assert audit(path, 'owner', "' OR 1=1 --")['found'] is False
    assert audit(path, 'other-owner', 'original')['found'] is False
    assert audit(path, 'owner', 'original')['status'] == 'succeeded'


def test_audit_uses_wal_instead_of_ignoring_latest_committed_state(tmp_path):
    path = tmp_path / 'original.sqlite3'
    database(path, {'status': 'unknown'})
    with sqlite3.connect(path) as writer:
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('UPDATE requests SET receipt=?', (json.dumps({'status': 'succeeded'}),))
        writer.commit()
        assert audit(path, 'owner', 'original')['status'] == 'succeeded'


def test_audit_does_not_create_missing_database_or_expose_path(tmp_path):
    path = tmp_path / 'PRIVATE_PATH.sqlite3'
    result = subprocess.run([sys.executable, 'scripts/audit_recovery_scan.py', '--database', str(path),
                             '--owner', 'owner', '--request-id', 'original'], capture_output=True, text=True)
    assert result.returncode == 2 and not path.exists()
    assert json.loads(result.stdout) == {'error': 'READ_ONLY_AUDIT_UNAVAILABLE'}
    assert 'PRIVATE_PATH' not in result.stdout + result.stderr


def test_audit_invalid_scan_does_not_invent_progress():
    assert summarize({'_recovery_conversation_scan': {'next_index': -1}})['scan']['invalid'] is True
    assert summarize({'_recovery_conversation_scan': {}})['scan']['invalid'] is True
