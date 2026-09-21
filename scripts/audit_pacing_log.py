#!/usr/bin/env python3
"""Read-only local extractor for the two existing Provider clock log events.

Pipe a bounded `docker logs --timestamps --since ...` export on the owning host.
No application import, database access, HTTP request, credentials, configuration
write or automatic parameter change. It deliberately cannot derive a safe N
from event counts: old events do not identify model, operation or active work.
Only explicitly allowlisted numeric fields and existing opaque account references
reach stdout; source lines and arbitrary exception text are never printed.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import re
import sys

EVENTS = frozenset({'account_message_start', 'account_rate_limited'})
ACCOUNT = re.compile(r'^[0-9a-f]{12}$')
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
    if not isinstance(account, str) or not ACCOUNT.fullmatch(account):
        return None
    # Keep only the two fixed event names, the provider's opaque hash and numbers.
    clean = {'event': obj['event'], 'account': account}
    fields = ('since_previous_secs', 'minimum_interval_secs') if obj['event'] == 'account_message_start' else ('consecutive_limits', 'retry_after_secs', 'cooldown_secs')
    for name in fields:
        clean[name] = numeric(obj.get(name))
    if match and match.group('time'):
        clean['timestamp'] = match.group('time')
    return clean


def _range(values):
    return {'min': min(values), 'max': max(values), 'samples': len(values)} if values else {'min': None, 'max': None, 'samples': 0}


def analyze(lines, *, input_complete=True):
    rows = defaultdict(lambda: {'starts': 0, 'limits': 0, 'interval': [], 'local_minimum': [], 'retry_after': [], 'cooldown': [], 'times': []})
    ignored = total = recognized = 0
    for line in lines:
        total += 1
        event = decode_event(line)
        if event is None:
            ignored += 1
            continue
        recognized += 1
        row = rows[event['account']]
        if 'timestamp' in event:
            row['times'].append(event['timestamp'])
        if event['event'] == 'account_message_start':
            row['starts'] += 1
            for field, dest in [('since_previous_secs', 'interval'), ('minimum_interval_secs', 'local_minimum')]:
                if event[field] is not None:
                    row[dest].append(event[field])
        else:
            row['limits'] += 1
            for field, dest in [('retry_after_secs', 'retry_after'), ('cooldown_secs', 'cooldown')]:
                if event[field] is not None:
                    row[dest].append(event[field])
    accounts = []
    for account, row in sorted(rows.items()):
        accounts.append({
            'account_ref': account, 'start_events': row['starts'], 'rate_limit_events': row['limits'],
            'start_spacing_seconds': _range(row['interval']),
            'configured_start_spacing_seconds': _range(row['local_minimum']),
            'retry_after_seconds': _range(row['retry_after']), 'cooldown_seconds': _range(row['cooldown']),
            'timestamped_events': len(row['times']),
            'model': None, 'operation': None, 'safe_concurrency': None, 'recommended_interval_seconds': None,
        })
    return {
        'schema': 1, 'mode': 'read_only_existing_clock_events',
        'coverage': {'input_complete': input_complete, 'lines_read': total, 'recognized_events': recognized, 'ignored_lines': ignored},
        'accounts': accounts,
        'limitations': [
            'Counts are observed log events, not all requests, success rates, model usage or billable generations.',
            'Old clock events lack per-event model, endpoint, original request ID and concurrent occupancy.',
            'Duplicate or overlapping log exports cannot be deduplicated without event IDs.',
            'No events is missing evidence, not zero rate limits. The first spacing sample can precede the selected time window.',
            'Company ingress and Provider admission errors require their own evidence; these two event types only describe the account clock.',
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
