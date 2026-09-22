"""Actual locator + SQLite receipts; upstream reads are controlled, never model POSTs."""
import copy
import hashlib
import json
from unittest import mock

import pytest

from services.conversation_binding_service import (
    ConversationBindingError, ConversationBindingService as Binding,
    RECOVERY_CONVERSATION_SCAN_FIELD as SCAN, TextRecoveryReason,
)
from services.public_chat_service import project_public_chat_receipt
from services.text_task_service import TextTaskService


class Clock:
    def __init__(self):
        self.wall, self.tick = 1000.0, 100.0
    def advance(self, value=31):
        self.wall += value
        self.tick += value


@pytest.fixture
def clock():
    c = Clock()
    with mock.patch('services.conversation_binding_service.time.time', side_effect=lambda: c.wall), \
         mock.patch('services.conversation_binding_service.time.monotonic', side_effect=lambda: c.tick):
        yield c


def receipt():
    return {'request_id': 'original', 'model': 'fixture-text', 'created_at': 1.0,
            'provider_binding_id': 'original-binding', 'provider_account_identity': 'original-account',
            'client_conversation_id': 'original-session', 'request_message_id': 'original-user',
            'request_parent_message_id': 'original-parent'}


def document(cid, match=False):
    mapping = {}
    if match:
        mapping = {
            'original-user': {'parent': 'original-parent', 'message': {
                'id': 'original-user', 'author': {'role': 'user'}}},
            'original-answer': {'parent': 'original-user', 'message': {
                'id': 'original-answer', 'author': {'role': 'assistant'},
                'status': 'finished_successfully', 'end_turn': True, 'channel': 'final',
                'content': {'content_type': 'text', 'parts': ['the original answer']}}},
        }
    return {'conversation_id': cid, 'mapping': mapping, 'current_node': 'original-answer'}


def http_error(status, retry=None):
    from utils.helper import UpstreamHTTPError
    return UpstreamHTTPError('/private?DO_NOT_STORE_URL', status, {'secret': 'DO_NOT_STORE_BODY'}, retry)


class Backend:
    def __init__(self, rows):
        self.rows = dict(rows)
        self.calls, self.pages = [], []
    def _list_recent_conversations(self, *, limit, offset, **_):
        self.pages.append(offset)
        return [{'id': k} for k in list(self.rows)[offset:offset+limit]]
    def _get_conversation(self, cid, *, timeout_secs):
        self.calls.append((cid, timeout_secs))
        value = self.rows[cid]
        if callable(value):
            value = value(timeout_secs)
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


def scan_error(backend, row):
    with pytest.raises(ConversationBindingError) as info:
        Binding._locate_text_request_conversation(backend, row)
    error = info.value
    if isinstance(error.recovery_scan, dict):
        row[SCAN] = copy.deepcopy(error.recovery_scan)
    return error


def test_bad_candidate_does_not_hide_later_match_or_count_as_absence(clock):
    row = receipt()
    backend = Backend([('bad', http_error(404)), ('match', document('match', True))])
    error = scan_error(backend, row)
    assert [c[0] for c in backend.calls] == ['bad', 'match']
    assert error.code == 'RECOVERY_READ_FAILED'
    assert row[SCAN]['matches'][0]['conversation_id'] == 'match'
    assert row[SCAN]['next_index'] == 1
    assert 'bad' in row[SCAN]['failed_reads']
    assert not error.conversation_id  # provisional match is not adopted
    assert error.recovery_reason != TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value


def test_retry_wait_does_not_revisit_completed_candidates(clock):
    row = receipt()
    backend = Backend([('bad', http_error(404)), ('match', document('match', True))])
    scan_error(backend, row)
    scan_error(backend, row)
    assert len(backend.calls) == 2
    backend.rows['bad'] = document('bad')
    clock.advance()
    scan_error(backend, row)  # persist the completed scan before exact final read
    located, doc = Binding._locate_text_request_conversation(backend, row)
    answer = Binding._read_text_request_result(backend, located, document=doc)
    assert answer['content'] == 'the original answer'
    assert located['provider_account_identity'] == 'original-account'
    assert [c[0] for c in backend.calls] == ['bad', 'match', 'bad', 'match']
    assert backend.pages == [0]


def test_timeout_consuming_window_rotates_before_persisting(clock):
    row = receipt()
    def slow(timeout):
        clock.tick += timeout
        raise TimeoutError('DO_NOT_STORE_TIMEOUT')
    backend = Backend([('slow', slow), ('later', document('later', True))])
    scan_error(backend, row)
    assert row[SCAN]['conversation_ids'] == ['later', 'slow']
    assert 'slow' in row[SCAN]['failed_reads']
    # No sleep is needed to examine the other candidate in a new short window.
    scan_error(backend, row)
    assert [c[0] for c in backend.calls] == ['slow', 'later']
    assert len(row[SCAN]['matches']) == 1


def test_each_failed_candidate_read_at_most_once_per_window(clock):
    row = receipt()
    backend = Backend([('a', http_error(404)), ('b', OSError('DO_NOT_STORE_TRANSPORT')),
                       ('c', document('c'))])
    scan_error(backend, row)
    assert [c[0] for c in backend.calls] == ['a', 'b', 'c']
    assert row[SCAN]['next_index'] == 1
    assert set(row[SCAN]['failed_reads']) == {'a', 'b'}
    scan_error(backend, row)
    assert len(backend.calls) == 3


@pytest.mark.parametrize('status,expected', [(401, 'RECOVERY_AUTH_REQUIRED'),
    (403, 'RECOVERY_AUTH_REQUIRED'), (429, 'RECOVERY_RATE_LIMITED')])
@pytest.mark.parametrize('phase', ['conversation_list', 'conversation_detail'])
def test_account_limits_and_auth_stop_scan_and_keep_typed_evidence(clock, status, expected, phase):
    row = receipt()
    backend = Backend([('bad', http_error(status, 120)), ('later', document('later', True))])
    if phase == 'conversation_list':
        backend._list_recent_conversations = mock.Mock(side_effect=http_error(status, 120))
    error = scan_error(backend, row)
    code, delay = TextTaskService._recovery_failure(error)
    assert code == expected
    assert delay == (120 if status == 429 else None)
    assert error.recovery_read_error['phase'] == phase
    assert error.recovery_read_error['http_status'] == status
    assert len(backend.calls) == (0 if phase == 'conversation_list' else 1)
    assert 'DO_NOT_STORE' not in json.dumps(error.recovery_read_error)
    assert 'DO_NOT_STORE' not in str(error)


def test_failed_candidate_can_reveal_a_second_match_not_false_unique(clock):
    row = receipt()
    backend = Backend([('temporarily-bad', http_error(503)), ('match', document('match', True))])
    scan_error(backend, row)
    backend.rows['temporarily-bad'] = document('temporarily-bad', True)
    clock.advance()
    error = scan_error(backend, row)
    assert error.code == 'CONVERSATION_BINDING_MISMATCH'
    assert not error.conversation_id


def test_schema_failure_kept_while_other_documents_are_examined(clock):
    row = receipt()
    backend = Backend([('bad', {'conversation_id': 'bad', 'mapping': 'wrong'}),
                       ('later', document('later', True))])
    error = scan_error(backend, row)
    assert [c[0] for c in backend.calls] == ['bad', 'later']
    assert error.code == 'CONVERSATION_BINDING_CONTRACT_INVALID'
    assert row[SCAN]['failed_reads']['bad']['error']['category'] == 'parse'
    assert len(row[SCAN]['matches']) == 1


def test_window_rollover_retains_failed_candidates_and_matches(clock):
    row = receipt()
    rows = [('bad', http_error(404))] + [(f'other-{i}', document(f'other-{i}')) for i in range(99)]
    rows += [('later', document('later', True))]
    backend = Backend(rows)
    scan_error(backend, row)
    assert row[SCAN]['next_offset'] == 100
    assert row[SCAN]['conversation_ids'] == ['bad']
    assert row[SCAN]['next_index'] == 0
    scan_error(backend, row)
    assert backend.calls[-1][0] == 'later'
    assert row[SCAN]['failed_reads']['bad']['attempts'] == 1
    assert len(row[SCAN]['matches']) == 1
    assert backend.pages == [0, 20, 40, 60, 80, 100]


def test_all_failed_windows_stay_bounded_and_never_report_no_match(clock):
    row = receipt()
    backend = Backend([(f'bad-{i}', http_error(404)) for i in range(121)])
    scan_error(backend, row)
    scan_error(backend, row)
    assert len(row[SCAN]['conversation_ids']) == 100
    assert len(row[SCAN]['failed_reads']) == 100
    assert row[SCAN]['coverage_complete'] is False
    assert len(backend.calls) == 100


def test_deferred_retry_honors_longer_retry_after(clock):
    row = receipt()
    backend = Backend([('bad', http_error(503, 200))])
    error = scan_error(backend, row)
    assert row[SCAN]['failed_reads']['bad']['next_at'] == 1200
    assert TextTaskService._recovery_failure(error)[1] == 200
    clock.advance(199)
    scan_error(backend, row)
    assert len(backend.calls) == 1
    clock.advance(1)
    scan_error(backend, row)
    assert len(backend.calls) == 2
    assert row[SCAN]['failed_reads']['bad']['attempts'] == 2


def test_complete_empty_search_still_reports_original_unattributable(clock):
    error = scan_error(Backend([]), receipt())
    assert error.recovery_reason == TextRecoveryReason.REQUEST_CONVERSATION_UNATTRIBUTABLE.value
    assert error.recovery_scan == {}


def test_failed_reads_cannot_inject_fields_or_claim_inspected_prefix(clock):
    row = receipt()
    backend = Backend([('bad', http_error(404)), ('checked', document('checked'))])
    scan_error(backend, row)
    state = row[SCAN]
    assert TextTaskService._safe_recovery_scan(state) == state
    for change in ['body', 'wrong_candidate', 'nonfinite', 'unchecked_ref', 'wrong_type']:
        tampered = copy.deepcopy(state)
        failure = tampered['failed_reads']['bad']
        if change == 'body':
            failure['error']['body'] = 'DO_NOT_STORE'
        elif change == 'wrong_candidate':
            tampered['failed_reads']['checked'] = tampered['failed_reads'].pop('bad')
        elif change == 'nonfinite':
            failure['next_at'] = float('nan')
        elif change == 'unchecked_ref':
            failure['error']['candidate_ref'] = 'attacker-url'
        else:
            failure['error']['category'] = []
        assert TextTaskService._safe_recovery_scan(tampered) is None
        assert Binding._validated_recovery_scan({**row, SCAN: tampered},
                                                Binding._recovery_scan_identity(row)) is None


class Queue:
    def __init__(self):
        self.calls = []
    def submit(self, function, *args):
        self.calls.append((function, args))


def stored_service(tmp_path, clock, backend):
    queue = Queue()
    def reader(row):
        located, doc = Binding._locate_text_request_conversation(backend, row)
        return Binding._read_text_request_result(backend, located, document=doc)
    service = TextTaskService(tmp_path / 'tasks.sqlite3', executor=queue,
                              recovery_reader=reader, clock=lambda: clock.wall)
    body = {'client_request_id': 'original', 'client_conversation_id': 'original-session',
            'model': 'fixture-text', 'messages': [{'role': 'user', 'content': 'immutable original input'}]}
    service.submit('owner', body)
    service._update('owner', 'original', **{k: v for k, v in receipt().items() if k != 'request_id'}, status='unknown',
                    error_code='CONVERSATION_OUTCOME_UNKNOWN', upstream_outcome='unknown')
    return service, queue, reader


def test_sqlite_restart_retains_provisional_match_errors_and_original_input_identity(tmp_path, clock):
    backend = Backend([('bad', http_error(404)), ('match', document('match', True))])
    service, queue, reader = stored_service(tmp_path, clock, backend)
    with service._db() as db:
        original_hash = db.execute('SELECT request_hash FROM requests').fetchone()[0]
    first = service.recover('owner', 'original', True)
    assert first['status'] == 'unknown' and first['recovery_no_result_reads'] == 0
    assert first['recovery_last_read_error']['http_status'] == 404
    progress = first['recovery_scan_progress']
    assert progress['list_complete'] is True and progress['window_checked'] == 1
    assert progress['window_pending'] == 1 and progress['matches'] == 1
    public = project_public_chat_receipt(first)
    assert public['recovery']['scan'] == progress
    assert public['recovery']['last_read_error']['category'] == 'http'
    for secret in ['original-account', 'original-binding', 'original-user', 'DO_NOT_STORE',
                   '"bad"', '"match"', 'immutable original input']:
        assert secret not in json.dumps(public)
    restarted = TextTaskService(service.path, executor=queue, recovery_reader=reader, clock=lambda: clock.wall)
    clock.advance()
    backend.rows['bad'] = document('bad')
    midway = restarted.recover('owner', 'original', True)
    assert midway['status'] == 'unknown'
    clock.advance()
    final = restarted.recover('owner', 'original', True)
    assert final['status'] == 'succeeded' and final['content'] == 'the original answer'
    assert final['provider_account_identity'] == 'original-account'
    assert final.get('recovery_last_read_error') is None
    with service._db() as db:
        assert db.execute('SELECT request_hash FROM requests').fetchone()[0] == original_hash
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 1
    assert len(queue.calls) == 1  # the original queued placeholder, never executed or resubmitted


def test_wrapped_429_persists_rate_limit_and_original_id_without_more_gets(tmp_path, clock):
    backend = Backend([('bad', http_error(429, 120)), ('later', document('later', True))])
    service, queue, _ = stored_service(tmp_path, clock, backend)
    result = service.read('owner', 'original')
    assert result['recovery_error_code'] == 'RECOVERY_RATE_LIMITED'
    assert result['recovery_next_at'] == clock.wall + 120
    assert result['recovery_last_read_error']['http_status'] == 429
    service.read('owner', 'original')
    assert len(backend.calls) == 1 and len(queue.calls) == 1


def test_suppressed_recovery_is_not_reenabled_by_scan_fix(tmp_path, clock):
    backend = Backend([('match', document('match', True))])
    service, queue, _ = stored_service(tmp_path, clock, backend)
    service._update('owner', 'original', _recovery_suppressed=True)
    for _ in range(3):
        result = service.recover('owner', 'original')
        clock.advance(3600)
        assert result['status'] == 'unknown'
    assert backend.calls == [] and backend.pages == []
    with service._db() as db:
        row = json.loads(db.execute('SELECT receipt FROM requests').fetchone()[0])
        assert row['_recovery_suppressed'] is True
    assert len(queue.calls) == 1


def test_public_scan_metadata_is_derived_and_allowlisted():
    row = receipt()
    row['recovery_last_read_error'] = {'http_status': 500, 'body': 'DO_NOT_STORE'}
    row['recovery_scan_progress'] = {'raw_conversation': 'DO_NOT_STORE'}
    projected = TextTaskService._public(row)
    assert 'recovery_last_read_error' not in projected
    assert 'recovery_scan_progress' not in projected
    assert 'DO_NOT_STORE' not in json.dumps(projected)
