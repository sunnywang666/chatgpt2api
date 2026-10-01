"""Caller scheduling constraints over original receipts and work lifecycle rows."""
from datetime import datetime, timezone
import hashlib
import json
import math

from services.admission_planner import Need, Resource
from services.request_context import AdmissionLost


def normalize_scheduling(value):
    if value is None:
        return None
    allowed = {'workflow_id', 'workflow_concurrency', 'min_send_interval_seconds', 'not_before', 'wait_deadline'}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError('SCHEDULING_INVALID: unsupported scheduling fields')
    result = dict(value)
    workflow = result.get('workflow_id')
    if workflow is not None and (not isinstance(workflow, str) or not workflow.strip() or len(workflow) > 128
                                 or any(ord(c) < 32 for c in workflow)):
        raise ValueError('SCHEDULING_INVALID: workflow_id')
    if 'workflow_concurrency' in result:
        count = result['workflow_concurrency']
        if not workflow or type(count) is not int or not 1 <= count <= 64:
            raise ValueError('SCHEDULING_INVALID: workflow_concurrency requires workflow_id and 1..64')
    elif workflow:
        result['workflow_concurrency'] = 1
    if 'min_send_interval_seconds' in result:
        seconds = result['min_send_interval_seconds']
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 86400:
            raise ValueError('SCHEDULING_INVALID: min_send_interval_seconds')
    for field in ('not_before', 'wait_deadline'):
        if field in result:
            timestamp(result[field])
    if result.get('not_before') and result.get('wait_deadline') and timestamp(result['wait_deadline']) <= timestamp(result['not_before']):
        raise ValueError('SCHEDULING_INVALID: wait_deadline must follow not_before')
    return result


def timestamp(value):
    try:
        if not isinstance(value, str):
            raise ValueError()
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError()
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        raise ValueError('SCHEDULING_INVALID: timestamps require UTC ISO8601') from None


def group(receipt):
    scheduling = receipt.get('_scheduling') or {}
    subject = receipt.get('_scheduling_owner') or receipt.get('_source') or ''
    workflow = scheduling.get('workflow_id') or receipt.get('_work_key') or receipt.get('request_id') or receipt.get('id')
    return hashlib.sha256((str(subject) + '\0' + str(workflow)).encode()).hexdigest()


def prepare_receipt(receipt, scheduling, source):
    receipt['_scheduling_owner'] = source
    if scheduling is not None:
        receipt['_scheduling'] = normalize_scheduling(scheduling)


def state_snapshot(store, db):
    works = {}
    for key, value in db.execute("SELECT name,value FROM task_runtime WHERE name LIKE 'work:%'"):
        works[key] = json.loads(value)
    clocks = {key: json.loads(value) for key, value in db.execute(
        "SELECT name,value FROM task_runtime WHERE name LIKE 'workflow_send:%'")}
    return works, clocks


def constraints(receipt, works, clocks, receipts, now):
    """Return additional resources/needs and caller earliest-send timestamp."""
    scheduling = receipt.get('_scheduling') or {}
    work = works.get(receipt.get('_work_key'))
    resources, needs = [], []
    ready = timestamp(scheduling['not_before']) if scheduling.get('not_before') else 0
    interval = scheduling.get('min_send_interval_seconds', 0)
    ready = max(ready, float(clocks.get('workflow_send:' + group(receipt), 0)) + interval if interval else 0)
    if not work:
        return resources, needs, ready
    work_id = work['key']
    enabled = work.get('state', 'active') == 'active'
    active = sum(1 for _, _, _, r in receipts if r.get('_work_key') == work_id and (
        r.get('status') == 'running' or r.get('upstream_unfinished') is True or r.get('upstream_outcome') == 'unknown'
        or r.get('status') == 'unknown'))
    turn = 'work_turn:' + work_id.removeprefix('work:')
    resources.append(Resource(turn, int(enabled), active, now))
    needs.append(Need(turn))
    workflow = scheduling.get('workflow_id') or work.get('workflow_id')
    concurrency = scheduling.get('workflow_concurrency') or work.get('workflow_concurrency')
    if workflow and concurrency and not work.get('slot_held'):
        subject = receipt.get('_scheduling_owner') or receipt.get('_source') or work.get('source')
        related = [w for w in works.values() if w.get('source') == subject and w.get('workflow_id') == workflow]
        capacity = min([concurrency] + [w['workflow_concurrency'] for w in related
                       if w.get('state') == 'active' and w.get('workflow_concurrency')])
        occupied = sum(bool(w.get('slot_held')) for w in related)
        key = 'workflow_slots:' + group({**receipt, '_scheduling': {'workflow_id': workflow}})
        resources.append(Resource(key, capacity, occupied, now))
        needs.append(Need(key))
    return resources, needs, ready


def expire_unsent(store, db, receipts, now):
    for kind, owner, request_id, receipt in receipts:
        deadline = (receipt.get('_scheduling') or {}).get('wait_deadline')
        if (not deadline or now < timestamp(deadline) or receipt.get('_submission_started')
                or receipt.get('status') not in {'queued', 'not_started', 'running'}
                or receipt.get('upstream_outcome') == 'unknown'):
            continue
        receipt.update(status='error' if kind == 'image' else 'failed', error_code='WAIT_DEADLINE_EXCEEDED',
                       upstream_outcome='not_sent', upstream_unfinished=False, _turn_reserved=False, _executing=False,
                       _claim_id=None, _claim_until=0, finished_at=now,
                       waiting={'reasons': ['wait_deadline_exceeded'], 'next_check_at': None})
        store.write_receipt(db, kind, owner, request_id, receipt)
        release_provisional_slot(store, db, receipt)


def release_provisional_slot(store, db, receipt):
    key = receipt.get("_work_key")
    if not key:
        return
    members = [r for _, _, _, r in store.receipts(db) if r.get("_work_key") == key]
    if any(r.get("_submission_started") or r.get("upstream_outcome") == "unknown" for r in members):
        return
    work = store.runtime(db, key)
    if work:
        work["slot_held"] = False
        store.set_runtime(db, key, work)


def send_delay(store, db, receipt, now):
    interval = (receipt.get('_scheduling') or {}).get('min_send_interval_seconds', 0)
    last = float(store.runtime(db, 'workflow_send:' + group(receipt), 0))
    return max(0.0, last + interval - now) if interval else 0.0


def before_send(store, db, receipt, now):
    scheduling = receipt.get('_scheduling') or {}
    if scheduling.get('wait_deadline') and not receipt.get('_submission_started') and now >= timestamp(scheduling['wait_deadline']):
        raise AdmissionLost('WAIT_DEADLINE_EXCEEDED')
    if scheduling.get('not_before') and now < timestamp(scheduling['not_before']):
        raise AdmissionLost('SCHEDULING_NOT_BEFORE')
    work = store.runtime(db, receipt.get('_work_key', ''), {})
    if work and (work.get('state', 'active') != 'active' or not work.get('slot_held')):
        raise AdmissionLost('WORK_NOT_ACTIVE')
    key = 'workflow_send:' + group(receipt)
    last = float(store.runtime(db, key, 0))
    if now < last + scheduling.get('min_send_interval_seconds', 0):
        raise AdmissionLost('SCHEDULING_SEND_INTERVAL')
    if scheduling:
        store.set_runtime(db, key, now)


def waiting_reasons(receipt, works, clocks, receipts, now, reasons):
    resources, _, ready = constraints(receipt, works, clocks, receipts, now)
    result = list(reasons)
    scheduling = receipt.get('_scheduling') or {}
    if scheduling.get('not_before') and timestamp(scheduling['not_before']) > now:
        result.append('not_before')
    elif ready > now:
        result.append('send_interval')
    for resource in resources:
        if resource.occupied >= resource.capacity:
            if resource.key.startswith('workflow_slots:'):
                result.append('workflow_concurrency')
            elif not resource.capacity:
                result.append('work_not_active')
            else:
                result.append('same_work_turn')
    return list(dict.fromkeys(result))
