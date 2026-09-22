import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.audit_pacing_log import analyze, decode_event, MAX_LINE, MAX_SAMPLES
REF='0123456789ab'

def line(event='account_message_start', **kw):
    return '[INFO] '+json.dumps({'event':event,'account':REF,**kw})

class AuditTests(unittest.TestCase):
    def test_current_events_retain_request_model_input_and_rate_evidence(self):
        result=analyze([
            line(request_ref='a'*24,model='gpt-5-6-thinking',operation='text',route='chat',
                 retained_input_bytes=2048,send_sequence=0,layer='upstream_chatgpt',phase='conversation'),
            line('account_rate_limited',request_ref='a'*24,model='gpt-5-6-thinking',
                 layer='upstream_chatgpt',phase='conversation',origin='http_429',
                 upstream_request_id='request-from-upstream',retry_after_secs=120,cooldown_secs=180),
        ])
        first,limited=result['samples']
        self.assertEqual(first['request_ref'],limited['request_ref'])
        self.assertEqual(first['model'],'gpt-5-6-thinking')
        self.assertEqual(first['retained_input_bytes'],2048)
        self.assertEqual(first['send_sequence'],0)
        self.assertEqual(limited['origin'],'http_429')
        self.assertEqual(limited['retry_after_secs'],120)
        self.assertEqual(len(limited['upstream_request_ref']),24)
        self.assertNotIn('request-from-upstream',json.dumps(result))
        self.assertEqual(result['coverage']['request_attributed_events'],2)
        self.assertIsNone(result['accounts'][0]['safe_concurrency'])

    def test_missing_or_untrusted_attribution_is_not_invented_or_exposed(self):
        result=analyze([line(),line(model='PRIVATE_TOKEN_VALUE',request_ref='PRIVATE_REQUEST',
            layer='PRIVATE_LAYER',phase='PRIVATE_PHASE',operation=['PRIVATE_OPERATION'],
            retained_input_bytes='PRIVATE_INPUT',upstream_request_id='PRIVATE_HEADER')])
        for field in ('model','request_ref','layer','phase','operation','retained_input_bytes'):
            self.assertIsNone(result['samples'][1][field])
        self.assertEqual(result['coverage']['request_attributed_events'],0)
        self.assertNotIn('PRIVATE_',json.dumps(result))

    def test_sse_limit_preserves_the_exact_phase_emitted_by_account_clock(self):
        result=analyze([line('account_rate_limited',request_ref='b'*24,layer='upstream_chatgpt',
                            phase='conversation_stream',origin='sse_rate_limit',retry_after_secs=60)])
        self.assertEqual(result['samples'][0]['phase'],'conversation_stream')
        self.assertEqual(result['samples'][0]['origin'],'sse_rate_limit')

    def test_sample_bound_does_not_hide_aggregate_counts_or_truncation(self):
        result=analyze(line(request_ref='a'*24) for _ in range(MAX_SAMPLES+1))
        self.assertEqual(len(result['samples']),MAX_SAMPLES)
        self.assertEqual(result['coverage']['omitted_samples'],1)
        self.assertEqual(result['accounts'][0]['start_events'],MAX_SAMPLES+1)

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
