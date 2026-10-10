"""Batch snapshots, failure isolation, audit retention, and success publication."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import discover
import ingest_theme
import research_job as jobs
from tests.test_engine import annual_history, valuation
from tests.test_readiness import policy


class ResearchJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = self.root / 'job.json'
        self.output = self.root / 'batch-output'
        self.companies = self.root / 'companies.json'
        self.documents = self.root / 'corpus.jsonl'
        self.policy_path = self.root / 'policy.json'
        self.companies.write_text(json.dumps([dict(ticker='600001', name='Example', market='A',
            sector='Industrials', scope='manufacturing', universe_as_of='2026-05-01')]), encoding='utf-8')
        (self.root / 'source.txt').write_text('本公司生产机器人产品并对外销售。', encoding='utf-8')
        manifest = self.root / 'ingest.json'
        manifest.write_text(json.dumps([dict(ticker='600001', available_at='2026-04-01',
            source_url='https://example.org/report.txt', source_type='annual_report',
            local_path='source.txt', date_basis='official_release')]), encoding='utf-8')
        ingest_theme.ingest_manifest(manifest, self.documents)
        self.policy_path.write_text(json.dumps(policy()), encoding='utf-8')
        for name, rows in (('statements.csv', annual_history()), ('valuations.csv', [valuation()])):
            with (self.root / name).open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        self.config = dict(schema_version=1, as_of='2026-05-01', companies='companies.json',
                           documents='corpus.jsonl', statements='statements.csv', valuations='valuations.csv',
                           production_policy='policy.json', output_root='batch-output',
                           queries=[dict(name='robots', query='机器人'), dict(name='industrial', query='工业机器人')])
        self.save()

    def save(self, config=None):
        self.config_path.write_text(json.dumps(self.config if config is None else config, ensure_ascii=False), encoding='utf-8')

    def test_successful_offline_batch_saves_auditable_snapshots_and_pointer(self):
        with mock.patch.object(ingest_theme, '_download', side_effect=AssertionError('network forbidden')):
            job = jobs.run_job(self.config_path)
        self.assertEqual(job['status'], 'succeeded')
        self.assertFalse(job['production_approved'])
        self.assertEqual([query['status'] for query in job['queries']], ['succeeded', 'succeeded'])
        run_dir = Path(job['run_dir'])
        self.assertEqual(jobs.verify_job(run_dir), job)
        for role, item in job['input_snapshots'].items():
            self.assertEqual(jobs._hash_file(run_dir / item['snapshot_path']), item['sha256'], role)
        state = jobs.status(self.output)
        self.assertEqual(state['last_success']['run_id'], job['run_id'])
        self.assertFalse(state['last_success']['production_approved'])
        self.assertIn('research_job.py', job['code_sha256'])
        events = [json.loads(line) for line in (run_dir / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual([row['event'] for row in events].count('query_started'), 2)
        self.assertEqual([row['event'] for row in events].count('query_finished'), 2)

    def test_all_queries_read_same_snapshot_when_original_inputs_change(self):
        execute = discover.run_discovery
        seen = []
        def change_original(**kwargs):
            seen.append((kwargs['companies_path'], kwargs['companies_path'].read_bytes(),
                         kwargs['documents_path'], kwargs['documents_path'].read_bytes()))
            result = execute(**kwargs)
            self.companies.write_bytes(b'original input replaced after snapshot')
            self.documents.write_bytes(b'original documents replaced after snapshot')
            return result
        with mock.patch.object(discover, 'run_discovery', side_effect=change_original):
            job = jobs.run_job(self.config_path)
        self.assertEqual(job['status'], 'succeeded')
        self.assertEqual(seen[0], seen[1])
        self.assertNotEqual(seen[0][0], self.companies)
        self.assertEqual(jobs.verify_job(job['run_dir'])['status'], 'succeeded')

    def test_one_query_failure_does_not_discard_other_query_results(self):
        execute = discover.run_discovery
        def fail_one(**kwargs):
            if kwargs['output'].name == 'robots':
                raise RuntimeError('simulated per-query failure')
            return execute(**kwargs)
        with mock.patch.object(discover, 'run_discovery', side_effect=fail_one):
            job = jobs.run_job(self.config_path)
        self.assertEqual(job['status'], 'failed')
        self.assertEqual([query['status'] for query in job['queries']], ['failed', 'succeeded'])
        self.assertIn('simulated per-query failure', job['queries'][0]['error'])
        self.assertEqual(discover.verify_discovery(Path(job['run_dir']) / 'queries/industrial')['status'], 'verified')
        self.assertEqual(jobs.verify_job(job['run_dir'])['status'], 'failed')
        self.assertFalse((self.output / 'last_success.json').exists())

    def test_blocked_new_run_preserves_previous_success_and_all_history(self):
        first = jobs.run_job(self.config_path)
        pointer = (self.output / 'last_success.json').read_bytes()
        prior_marker = (Path(first['run_dir']) / 'completion.json').read_bytes()
        failing = policy()
        failing['expected_tickers'].append('600002')
        self.policy_path.write_text(json.dumps(failing), encoding='utf-8')
        second = jobs.run_job(self.config_path)
        self.assertEqual(second['status'], 'blocked')
        self.assertTrue(all(query['status'] == 'blocked' for query in second['queries']))
        self.assertEqual((self.output / 'last_success.json').read_bytes(), pointer)
        self.assertEqual((Path(first['run_dir']) / 'completion.json').read_bytes(), prior_marker)
        self.assertEqual(jobs.verify_job(first['run_dir'])['status'], 'succeeded')
        self.assertEqual(jobs.status(self.output)['total_runs'], 2)

    def test_repeated_success_uses_new_directory_and_keeps_old_run_unchanged(self):
        first = jobs.run_job(self.config_path)
        old = {p.relative_to(first['run_dir']).as_posix(): p.read_bytes()
               for p in Path(first['run_dir']).rglob('*') if p.is_file()}
        second = jobs.run_job(self.config_path)
        self.assertNotEqual(first['run_id'], second['run_id'])
        for name, content in old.items():
            self.assertEqual((Path(first['run_dir']) / name).read_bytes(), content)
        self.assertEqual(jobs.status(self.output)['last_success']['run_id'], second['run_id'])

    def test_older_cutoff_success_cannot_replace_more_recent_success(self):
        self.config['as_of'] = '2026-05-02'
        self.save()
        first = jobs.run_job(self.config_path)
        pointer = (self.output / 'last_success.json').read_bytes()
        self.config['as_of'] = '2026-05-01'
        self.save()
        older = jobs.run_job(self.config_path)
        self.assertEqual(older['status'], 'succeeded')
        self.assertEqual(older['promotion']['reason'], 'older_as_of')
        self.assertEqual((self.output / 'last_success.json').read_bytes(), pointer)
        state = jobs.status(self.output)
        self.assertEqual(state['last_success_as_of'], '2026-05-02')
        self.assertEqual(state['latest_attempt_as_of'], '2026-05-01')
        self.assertEqual(state['latest_attempt_status'], 'succeeded')
        self.assertFalse(state['is_last_success_latest'])

    def test_success_pointer_cutoff_must_match_sealed_job(self):
        jobs.run_job(self.config_path)
        pointer_path = self.output / 'last_success.json'
        pointer = json.loads(pointer_path.read_text(encoding='utf-8'))
        pointer['as_of'] = '2026-05-02'
        pointer_path.write_text(json.dumps(pointer), encoding='utf-8')
        state = jobs.status(self.output)
        self.assertIsNone(state['last_success'])
        self.assertIn('matching cutoff', state['last_success_error'])

    def test_missing_ingestion_completion_is_audited_failure(self):
        Path(str(self.documents) + '.complete.json').unlink()
        job = jobs.run_job(self.config_path)
        self.assertEqual(job['status'], 'failed')
        self.assertIn('FileNotFoundError', job['error'])
        self.assertIn('job_config', job['input_snapshots'])
        self.assertTrue(job['code_sha256'])
        self.assertFalse((self.output / 'last_success.json').exists())
        self.assertEqual(jobs.verify_job(job['run_dir'])['status'], 'failed')

    def test_changed_corpus_is_rejected_before_any_query(self):
        self.documents.write_bytes(self.documents.read_bytes() + b' ')
        with mock.patch.object(discover, 'run_discovery') as query:
            job = jobs.run_job(self.config_path)
        query.assert_not_called()
        self.assertEqual(job['status'], 'failed')
        self.assertIn('SHA256 mismatch', job['error'])

    def test_snapshot_tampering_between_queries_prevents_promotion(self):
        execute = discover.run_discovery
        def tamper_snapshot(**kwargs):
            result = execute(**kwargs)
            kwargs['companies_path'].write_bytes(b'changed frozen snapshot')
            return result
        with mock.patch.object(discover, 'run_discovery', side_effect=tamper_snapshot):
            job = jobs.run_job(self.config_path)
        self.assertEqual(job['status'], 'failed')
        self.assertTrue(all(query['status'] == 'failed' for query in job['queries']))
        self.assertFalse((self.output / 'last_success.json').exists())

    def test_input_change_while_snapshot_set_is_built_is_detected(self):
        snapshot = jobs._snapshot_file
        def change_original(source, target, run_dir):
            result = snapshot(source, target, run_dir)
            if ingest_theme.filesystem_path(source) == ingest_theme.filesystem_path(self.companies):
                self.companies.write_bytes(b'changed during snapshot copy')
            return result
        with mock.patch.object(jobs, '_snapshot_file', side_effect=change_original):
            job = jobs.run_job(self.config_path)
        self.assertEqual(job['status'], 'failed')
        self.assertIn('changed before snapshot set completed', job['error'])

    def test_publication_failure_is_logged_and_preserves_prior_pointer(self):
        jobs.run_job(self.config_path)
        original = (self.output / 'last_success.json').read_bytes()
        write = jobs._atomic_json
        def fail_pointer(path, value):
            if ingest_theme.filesystem_path(path) == ingest_theme.filesystem_path(self.output / 'last_success.json'):
                raise OSError('simulated pointer publication failure')
            return write(path, value)
        with mock.patch.object(jobs, '_atomic_json', side_effect=fail_pointer):
            job = jobs.run_job(self.config_path)
        self.assertEqual(job['status'], 'failed')
        self.assertIn('publication failed', job['error'])
        self.assertEqual((self.output / 'last_success.json').read_bytes(), original)
        self.assertEqual(jobs.verify_job(job['run_dir'])['status'], 'failed')

    def test_unknown_fields_invalid_paths_and_duplicate_names_are_rejected(self):
        invalid = [dict(self.config, unexpected=True), dict(self.config, as_of='2026-02-30'),
                   dict(self.config, schema_version=True), dict(self.config, limit=True),
                   dict(self.config, limit=0), dict(self.config, queries=[])]
        for name in ('../escape', 'a/b', 'a\\b', '.', 'NUL', 'CON', 'a.', 'a b'):
            invalid.append(dict(self.config, queries=[dict(name=name, query='机器人')]))
        invalid.append(dict(self.config, queries=[dict(name='same', query='a'), dict(name='SAME', query='b')]))
        invalid.append(dict(self.config, queries=[dict(name='name', query='x', typo=True)]))
        for config in invalid:
            with self.subTest(config=config):
                self.save(config)
                with self.assertRaises(ValueError):
                    jobs.run_job(self.config_path)
        self.assertFalse(self.output.exists())

    def test_missing_explicit_data_and_policy_are_not_replaced_with_demo(self):
        for field in ('companies', 'documents', 'production_policy', 'as_of'):
            config = copy.deepcopy(self.config)
            config.pop(field)
            self.save(config)
            with self.subTest(field=field), self.assertRaises(ValueError):
                jobs.run_job(self.config_path)

    def test_output_root_cannot_contain_inputs(self):
        original = self.companies.read_bytes()
        self.save(dict(self.config, output_root='.'))
        with self.assertRaisesRegex(ValueError, 'contain or overwrite'):
            jobs.run_job(self.config_path)
        self.assertEqual(self.companies.read_bytes(), original)

    def test_whole_job_lock_rejects_concurrent_process(self):
        code = ('import sys\nfrom research_job import run_job\n'
                'try:\n run_job(sys.argv[1])\nexcept ValueError as e:\n'
                ' print(str(e));sys.exit(23)\n')
        with ingest_theme.output_lock(self.output):
            result = subprocess.run([sys.executable, '-c', code, str(self.config_path)],
                                    cwd=jobs.ROOT, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 23, result.stderr.decode())
        self.assertIn(b'lock', result.stdout)
        self.assertFalse((self.output / 'runs').exists())

    def test_status_detects_tampered_success_instead_of_trusting_pointer(self):
        job = jobs.run_job(self.config_path)
        (Path(job['run_dir']) / 'queries/robots/results.json').write_bytes(b'tampered')
        status = jobs.status(self.output)
        self.assertIsNone(status['last_success'])
        self.assertIn('SHA256 mismatch', status['last_success_error'])
        self.assertEqual(status['runs'][0]['status'], 'incomplete_or_changed')

    def test_malformed_or_traversing_success_pointer_is_reported_not_followed(self):
        self.output.mkdir()
        for pointer in ([], None, {'run_id': '../../private', 'relative_path': '../../private'},
                        {'run_id': 123, 'production_approved': True}):
            (self.output / 'last_success.json').write_text(json.dumps(pointer), encoding='utf-8')
            with self.subTest(pointer=pointer):
                state = jobs.status(self.output)
                self.assertIsNone(state['last_success'])
                self.assertIn('Invalid last-success pointer', state['last_success_error'])

    def test_incomplete_audit_is_visible_and_never_becomes_last_success(self):
        run_id = '20260501T120000000000Z-' + 'a' * 32
        directory = self.output / 'runs' / run_id
        directory.mkdir(parents=True)
        (directory / 'job.json').write_text('{"status":"started"}', encoding='utf-8')
        state = jobs.status(self.output)
        self.assertIsNone(state['last_success'])
        self.assertEqual(state['runs'][0]['status'], 'incomplete_or_changed')

    def test_cli_exit_codes_and_read_only_status(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(jobs.main(['--config', str(self.config_path)]), 0)
            before = (self.output / 'last_success.json').read_bytes()
            self.assertEqual(jobs.main(['--status', str(self.output)]), 0)
            self.assertEqual((self.output / 'last_success.json').read_bytes(), before)
            self.policy_path.write_text(json.dumps(dict(policy(), expected_tickers=['600002'])), encoding='utf-8')
            self.assertEqual(jobs.main(['--config', str(self.config_path)]), 3)
            self.save(dict(self.config, typo=1))
            self.assertEqual(jobs.main(['--config', str(self.config_path)]), 2)


if __name__ == '__main__':
    unittest.main()
