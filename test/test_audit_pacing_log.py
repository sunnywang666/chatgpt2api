import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.audit_pacing_log import analyze, decode_event, MAX_LINE
REF='0123456789ab'

def line(event='account_message_start', **kw):
    return '[INFO] '+json.dumps({'event':event,'account':REF,**kw})

class AuditTests(unittest.TestCase):
    def test_existing_json_log_format(self):
        result=analyze([line(since_previous_secs=61,minimum_interval_secs=60)])
        self.assertEqual(result['accounts'][0]['start_spacing_seconds']['min'],61)
        self.assertIsNone(result['accounts'][0]['safe_concurrency'])

    def test_docker_timestamps_supported(self):
        item=decode_event('2026-09-20T00:00:00.123456Z '+line())
        self.assertEqual(item['timestamp'],'2026-09-20T00:00:00.123456Z')

    def test_empty_input_is_not_zero_rate_evidence(self):
        result=analyze([])
        self.assertEqual(result['accounts'],[])
        self.assertTrue(any('missing evidence' in s for s in result['limitations']))

    def test_arbitrary_secrets_never_leave_allowlist(self):
        result=analyze([line(token='PRIVATE_TOKEN_VALUE',password='PASSWORD_VALUE',prompt='PROMPT_SECRET',since_previous_secs=4),
                        '[ERROR] secret=SECRET_URL'])
        text=json.dumps(result)
        for secret in ['PRIVATE_TOKEN_VALUE','PASSWORD_VALUE','PROMPT_SECRET','SECRET_URL']:
            self.assertNotIn(secret,text)

    def test_invalid_account_reference_is_discarded(self):
        self.assertIsNone(decode_event('[INFO] '+json.dumps({'event':'account_message_start','account':'private@example.com'})))

    def test_plain_text_mention_of_429_is_ignored(self):
        result=analyze(['[INFO] model answered: 429 too many requests', '[INFO] '+json.dumps({'event':'model_output','content':'429'})])
        self.assertEqual(result['coverage']['recognized_events'],0)

    def test_rate_event_and_numeric_ranges(self):
        result=analyze([line('account_rate_limited',retry_after_secs=120,cooldown_secs=180,consecutive_limits=2)])
        self.assertEqual(result['accounts'][0]['rate_limit_events'],1)
        self.assertEqual(result['accounts'][0]['cooldown_seconds']['max'],180)

    def test_bad_numbers_are_not_coerced(self):
        result=analyze([line(since_previous_secs='SECRET',minimum_interval_secs=True),line('account_rate_limited',retry_after_secs=float('nan'),cooldown_secs=-1)])
        self.assertIsNone(result['accounts'][0]['start_spacing_seconds']['min'])
        self.assertIsNone(result['accounts'][0]['retry_after_seconds']['max'])

    def test_oversized_numbers_do_not_crash_or_become_limits(self):
        result=analyze([line(since_previous_secs=10**400)])
        self.assertIsNone(result['accounts'][0]['start_spacing_seconds']['min'])

    def test_malformed_or_oversized_lines_are_discarded(self):
        for raw in ['{}','[]','not-json',line()+'x'*MAX_LINE]:
            self.assertIsNone(decode_event(raw))

    def test_cli_reads_only_stdin_and_outputs_safe_summary(self):
        script=Path(__file__).resolve().parents[1]/'scripts'/'audit_pacing_log.py'
        p=subprocess.run([sys.executable,str(script)],input=line(since_previous_secs=61)+'\n',text=True,capture_output=True,timeout=5)
        self.assertEqual(p.returncode,0)
        self.assertEqual(json.loads(p.stdout)['coverage']['recognized_events'],1)

    def test_cli_truncation_is_visible(self):
        script=Path(__file__).resolve().parents[1]/'scripts'/'audit_pacing_log.py'
        p=subprocess.run([sys.executable,str(script),'--max-bytes','20'],input='x'*30,text=True,capture_output=True,timeout=5)
        self.assertEqual(p.returncode,2)
        self.assertFalse(json.loads(p.stdout)['coverage']['input_complete'])

if __name__=='__main__': unittest.main(verbosity=2)
