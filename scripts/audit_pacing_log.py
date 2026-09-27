#!/usr/bin/env python3
"""Read-only local extractor for Provider clock and execution timeline events.

Pipe a bounded `docker logs --timestamps --since ...` export on the owning host.
No application import, database access, HTTP request, credentials, configuration
write or automatic parameter change. Preserve current request/model attribution
while keeping old events explicitly incomplete. Neither format proves active
occupancy or a safe N. Only allowlisted fields reach stdout; source lines and
arbitrary exception text are never printed.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import sys

EVENTS = frozenset({'account_message_start', 'account_rate_limited', 'pool_execution_stage'})
ACCOUNT = re.compile(r'^[0-9a-f]{12}$')
REQUEST_REF = re.compile(r'^[0-9a-f]{24}$')
MODEL = re.compile(r'^(?:gpt-(?:image-)?[0-9][a-z0-9.-]{0,100}|o[1-9][a-z0-9.-]{0,100}|auto)$')
MAX_SAMPLES = 2000
PREFIX = re.compile(r'^(?:(?P<time>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))\s+)?\[(?:INFO|WARNING)\]\s+')
MAX_LINE = 64 * 1024


def numeric(value):
    try:
        return value if (not isinstance(value, bool) and isinstance(value, (int, float))
                         and math.isfinite(value) and value >= 0) else None
    except (OverflowError, TypeError, ValueError):
        return None


def decode_event(line):
    if len(line) > MAX_LINE:
        return None
    match = PREFIX.match(line)
    text = line[match.end():] if match else line.strip()
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if not isinstance(obj, dict) or obj.get('event') not in EVENTS:
        return None
    account = obj.get('account')
    account_ref = obj.get('account_ref')
    if obj.get('event') == 'pool_execution_stage':
        account = account_ref
        valid_account = isinstance(account, str) and REQUEST_REF.fullmatch(account)
    else:
        valid_account = isinstance(account, str) and ACCOUNT.fullmatch(account)
    if not valid_account:
        return None
    # Keep only the two fixed event names, the provider's opaque hash and numbers.
    clean = {'event': obj['event'], 'account': account}
    fields = (('since_previous_secs', 'minimum_interval_secs') if obj['event'] == 'account_message_start'
              else ('consecutive_limits', 'retry_after_secs', 'cooldown_secs') if obj['event'] == 'account_rate_limited'
              else ('status_code', 'input_bytes', 'config_revision'))
    for name in fields:
        clean[name] = numeric(obj.get(name))
    # These fields are emitted by the existing durable request context. Missing
    # values in old logs stay missing; do not infer them from neighbouring lines.
    clean['request_ref'] = obj.get('request_ref') if isinstance(obj.get('request_ref'), str) and REQUEST_REF.fullmatch(obj['request_ref']) else None
    clean['model'] = obj.get('model') if isinstance(obj.get('model'), str) and MODEL.fullmatch(obj['model']) else None
    for name, allowed in {
        'layer': {'upstream_chatgpt', 'upstream_codex', 'provider_capacity', 'company_transport'},
        'phase': {'conversation', 'conversation_stream', 'prepare', 'account_read', 'sse', 'stream', 'unknown'},
        'origin': {'http_429', 'sse_rate_limit'},
        'operation': {'text', 'image', 'search'},
        'route': {'chat', 'codex'},
    }.items():
        value = obj.get(name)
        clean[name] = value if isinstance(value, str) and value in allowed else None
    for name in ('retained_input_bytes', 'send_sequence', 'observed_at', 'cooldown_until'):
        clean[name] = numeric(obj.get(name))
    # Upstream IDs can be arbitrary strings. Keep a correlatable digest instead
    # of copying a potentially sensitive header value into the shared report.
    upstream = obj.get('upstream_request_id')
    clean['upstream_request_ref'] = hashlib.sha256(upstream.encode()).hexdigest()[:24] if isinstance(upstream, str) and 0 < len(upstream) <= 160 else None
    clean['at'] = numeric(obj.get('at'))
    clean['queue_wait_seconds'] = numeric(obj.get('queue_wait_seconds'))
    clean['image_count'] = numeric(obj.get('image_count'))
    clean['task_status'] = obj.get('task_status') if obj.get('task_status') in {'succeeded', 'failed', 'unknown'} else None
    clean['account_ref'] = account
    clean['stage'] = obj.get('stage') if obj['event'] == 'pool_execution_stage' and isinstance(obj.get('stage'), str) else None
    source = obj.get('source')
    clean['source'] = source if isinstance(source, str) and len(source) <= 160 and all(ord(c) >= 32 for c in source) else None
    if match and match.group('time'):
        clean['timestamp'] = match.group('time')
    return clean


def _range(values):
    return {'min': min(values), 'max': max(values), 'samples': len(values)} if values else {'min': None, 'max': None, 'samples': 0}


def analyze(lines, *, input_complete=True):
    rows = defaultdict(lambda: {'starts': 0, 'limits': 0, 'stages': 0, 'interval': [], 'local_minimum': [], 'retry_after': [], 'cooldown': [], 'times': []})
    ignored = total = recognized = attributed = 0
    samples = []
    outcomes = {}
    for line in lines:
        total += 1
        event = decode_event(line)
        if event is None:
            ignored += 1
            continue
        recognized += 1
        if (event['stage'] == 'task_finished' and event['request_ref'] and event['task_status'] and event['at'] is not None):
            key = event['request_ref']
            if key not in outcomes or event['at'] > outcomes[key]['at']:
                outcomes[key] = event
        if event['request_ref'] is not None:
            attributed += 1
        if len(samples) < MAX_SAMPLES:
            samples.append(event)
        row = rows[event['account']]
        if 'timestamp' in event:
            row['times'].append(event['timestamp'])
        if event['event'] == 'account_message_start':
            row['starts'] += 1
            for field, dest in [('since_previous_secs', 'interval'), ('minimum_interval_secs', 'local_minimum')]:
                if event[field] is not None:
                    row[dest].append(event[field])
        elif event['event'] == 'account_rate_limited':
            row['limits'] += 1
            for field, dest in [('retry_after_secs', 'retry_after'), ('cooldown_secs', 'cooldown')]:
                if event[field] is not None:
                    row[dest].append(event[field])
        else:
            row['stages'] += 1
    accounts = []
    for account, row in sorted(rows.items()):
        accounts.append({
            'account_ref': account, 'start_events': row['starts'], 'rate_limit_events': row['limits'],
            'execution_stage_events': row['stages'],
            'start_spacing_seconds': _range(row['interval']),
            'configured_start_spacing_seconds': _range(row['local_minimum']),
            'retry_after_seconds': _range(row['retry_after']), 'cooldown_seconds': _range(row['cooldown']),
            'timestamped_events': len(row['times']),
            'model': None, 'operation': None, 'safe_concurrency': None, 'recommended_interval_seconds': None,
        })
    hours = {}
    for event in outcomes.values():
        try:
            hour = datetime.fromtimestamp(event['at'], timezone.utc).strftime('%Y-%m-%dT%H:00:00Z')
        except (ValueError, OverflowError, OSError):
            continue
        row = hours.setdefault(hour, {'succeeded_requests': 0, 'failed_requests': 0, 'unknown_requests': 0,
                                      'succeeded_image_requests': 0, 'known_saved_images': 0, 'image_count_unknown_requests': 0, 'wait_seconds': []})
        row[event['task_status'] + '_requests'] += 1
        if event['queue_wait_seconds'] is not None:
            row['wait_seconds'].append(event['queue_wait_seconds'])
        if event['operation'] == 'image' and event['task_status'] == 'succeeded':
            row['succeeded_image_requests'] += 1
            if event['image_count'] is None:
                row['image_count_unknown_requests'] += 1
            else:
                row['known_saved_images'] += event['image_count']
    hourly = []
    for hour, row in sorted(hours.items()):
        waits = sorted(row.pop('wait_seconds'))
        completed = row['succeeded_requests'] + row['failed_requests']
        hourly.append({'hour_utc': hour, **row, 'failure_rate_known_outcomes': row['failed_requests'] / completed if completed else None,
                       'queue_wait_seconds': {**_range(waits), 'p95': waits[max(0, math.ceil(len(waits) * .95) - 1)] if waits else None}})
    return {
        'schema': 1, 'mode': 'read_only_clock_and_execution_events',
        'coverage': {'input_complete': input_complete, 'lines_read': total, 'recognized_events': recognized, 'ignored_lines': ignored,
                     'request_attributed_events': attributed, 'sampled_events': len(samples),
                     'omitted_samples': recognized - len(samples)},
        'accounts': accounts,
        'hourly_outcomes': hourly,
        'real_24h_acceptance': 'unverified_requires_complete_window_runtime_and_original_receipt_readback',
        'samples': samples,
        'limitations': [
            'Clock counts are observed events. Hourly outcomes use the latest original task_finished event per request, not all accepted requests or billable generations; UNKNOWN remains separate and image counts may be missing.',
            'Old clock events lack per-event attribution; current clock and execution-stage events may include model, operation, persisted input bytes and the owner/request hash.',
            'Execution-stage events provide send/result boundaries; they still require the original receipt and an observed pool snapshot to establish occupancy.',
            'A message-start event precedes final send guards and is not proof that the model received the request. Confirm the original receipt/result.',
            'Request refs use the existing first 24 hex characters of SHA256(owner + colon + request ID); upstream header IDs are separately hashed for safe correlation.',
            'Samples are bounded and omitted_samples reports truncation; account totals still cover every recognized event in the input.',
            'Duplicate or overlapping log exports cannot be deduplicated without event IDs.',
            'No events is missing evidence, not zero rate limits. The first spacing sample can precede the selected time window.',
            'Company ingress and Provider admission errors require their own evidence; these events do not classify a transport-layer 429 as an upstream limit.',
            'No safe concurrency or production setting is inferred or changed.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-bytes', type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()
    if not 1 <= args.max_bytes <= 1024 * 1024 * 1024:
        parser.error('max-bytes must be between 1 and 1073741824')
    # This CLI requires a bounded export; it never tails a live stream forever.
    data = sys.stdin.buffer.read(args.max_bytes + 1)
    complete = len(data) <= args.max_bytes
    if not complete:
        data = data[:args.max_bytes]
        data = data.rsplit(b'\n', 1)[0] if b'\n' in data else b''
    report = analyze(data.decode('utf8', errors='replace').splitlines(), input_complete=complete)
    report['source_sha256'] = hashlib.sha256(data).hexdigest()
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.write('\n')
    return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
