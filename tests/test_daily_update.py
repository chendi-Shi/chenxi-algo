"""Daily refresh boundaries with mocked public transport and real research jobs."""
import contextlib
import copy
import csv
from datetime import datetime, timezone
import io
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import daily_update as daily
import ingest_theme
import research_job as jobs
from tests.test_engine import annual_history, valuation
from tests.test_readiness import policy


class DailyUpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_path = self.root / 'daily.json'
        self.output = self.root / 'daily-output'
        self.companies = self.root / 'companies.json'
        self.company_rows = [dict(ticker='600001', name='Reviewed name', market='A', sector='Industrials',
                                 scope='manufacturing', universe_as_of='2026-04-01')]
        self.companies.write_text(json.dumps(self.company_rows), encoding='utf-8')
        (self.root / 'policy.json').write_text(json.dumps(policy()), encoding='utf-8')
        for filename, rows in (('statements.csv', annual_history()), ('valuations.csv', [valuation()])):
            with (self.root / filename).open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        self.config = dict(schema_version=1, start_date='2026-04-01', end_date='2026-05-01',
            companies='companies.json', production_policy='policy.json', statements='statements.csv',
            valuations='valuations.csv', queries=[dict(name='robots', query='机器人')],
            output_root='daily-output', source_selection=dict(universe_sources=['sse'], categories=['all'],
            max_pages=20, min_interval_seconds=1.0))
        self.save()
        self.universe_result = {'status': 'completed', 'requested_sources': ['sse'], 'securities': [dict(ticker='600001', name='Official name',
            market='A', exchange='SSE', security_type='equity', universe_as_of='2026-05-01', source_effective_at=None)]}
        self.disclosure_result = {'status': 'completed', 'records': [], 'failures': [], 'audit': [],
            'coverage': {}, 'ingest_manifest': [dict(ticker='600001', available_at='2026-04-01',
                source_url='https://example.org/annual.txt', source_type='annual_report', format='txt',
                date_basis='official_release')]}
        self.body = '本公司生产机器人产品并对外销售。'.encode('utf-8')
        self.real_acquire = daily._acquire_universe
        for name, function in (('_acquire_universe', self.universe), ('_disclosures', self.disclosures)):
            patcher = mock.patch.object(daily, name, side_effect=function)
            setattr(self, name + '_mock', patcher.start())
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(ingest_theme, '_download', side_effect=lambda url, **kwargs: (self.body, url))
        self.download_mock = patcher.start()
        self.addCleanup(patcher.stop)

    def save(self, config=None):
        self.config_path.write_text(json.dumps(self.config if config is None else config), encoding='utf-8')

    def universe(self, directory, *, download, sources):
        if download:
            directory.mkdir(parents=True, exist_ok=False)
            (directory / 'fixture.json').write_text(json.dumps(self.universe_result), encoding='utf-8')
            return copy.deepcopy(self.universe_result)
        return json.loads((directory / 'fixture.json').read_text(encoding='utf-8'))

    def disclosures(self, securities, start_date, end_date, *, download, snapshot_dir, **kwargs):
        if download:
            snapshot_dir.mkdir(parents=True, exist_ok=False)
            (snapshot_dir / 'fixture.json').write_text(json.dumps(self.disclosure_result), encoding='utf-8')
            return copy.deepcopy(self.disclosure_result)
        return json.loads((snapshot_dir / 'fixture.json').read_text(encoding='utf-8'))

    def run_daily(self, **kwargs):
        return daily.run_daily(self.config_path, download=True, **kwargs)

    def test_complete_refresh_research_and_success_pointer_preserve_classification(self):
        original = self.companies.read_bytes()
        job = self.run_daily()
        self.assertEqual(job['status'], 'succeeded', job.get('error'))
        self.assertFalse(job['production_approved'])
        refreshed = json.loads((Path(job['run_dir']) / 'refreshed_companies.json').read_text(encoding='utf-8'))
        self.assertEqual(refreshed[0]['universe_as_of'], '2026-05-01')
        self.assertEqual(refreshed[0]['name'], 'Official name')
        self.assertIn('Reviewed name', refreshed[0]['aliases'])
        self.assertEqual(refreshed[0]['scope'], 'manufacturing')
        self.assertEqual(refreshed[0]['sector'], 'Industrials')
        self.assertEqual(self.companies.read_bytes(), original)
        self.assertEqual(len(job['stages']), 6)
        self.assertTrue(all(stage['duration_seconds'] >= 0 for stage in job['stages']))
        self.assertEqual(jobs.verify_job(job['run_dir'])['status'], 'succeeded')
        self.assertEqual(jobs.status(self.output)['last_success']['run_id'], job['run_id'])

    def test_real_filesystem_pipeline_and_cache_work_beyond_260_characters(self):
        first = self.root / ('long-' + 'a' * 60)
        long_root = first / ('nested-' + 'b' * 60) / ('nested-' + 'c' * 60) / ('nested-' + 'd' * 60)
        self.assertTrue(ingest_theme.filesystem_path(first).is_relative_to(ingest_theme.filesystem_path(self.root)))
        self.addCleanup(lambda: shutil.rmtree(ingest_theme.filesystem_path(first)) if first.exists() else None)
        self.config['output_root'] = str(long_root)
        self.save()
        job = self.run_daily()
        self.assertEqual(job['status'], 'succeeded', job.get('error'))
        self.assertGreater(len(job['run_dir']), 260)
        run_dir = Path(job['run_dir'])
        self.assertEqual(ingest_theme.verify_ingestion_output(run_dir / 'corpus.jsonl')['status'], 'completed')
        self.assertEqual(jobs.verify_job(run_dir)['status'], 'succeeded')
        self.assertEqual(jobs.status(long_root)['last_success']['run_id'], job['run_id'])
        self.assertTrue(list((run_dir / 'source_cache' / 'entries').glob('*.json')))

    def test_official_name_change_preserves_prior_alias_without_duplicates(self):
        company = dict(self.company_rows[0], aliases=['Reviewed name', 'Prior Chinese Name'])
        refreshed, _, failures = daily._refresh_companies([company], self.universe_result, '2026-05-01')
        self.assertFalse(failures)
        self.assertEqual(refreshed[0]['name'], 'Official name')
        self.assertEqual(refreshed[0]['aliases'], ['Reviewed name', 'Prior Chinese Name'])
        self.assertEqual(company['name'], 'Reviewed name')

    def test_output_inside_replay_directory_is_rejected_before_writes(self):
        for role in ('universe_snapshot', 'disclosure_snapshot', 'document_cache'):
            with self.subTest(role=role):
                source = self.root / role
                source.mkdir()
                seed = source / 'seed.txt'
                seed.write_text('immutable input', encoding='utf-8')
                config = dict(self.config, universe_snapshot='universe_snapshot',
                              disclosure_snapshot='disclosure_snapshot', document_cache='document_cache',
                              output_root=str(source / 'output'))
                self.save(config)
                with self.assertRaisesRegex(ValueError, 'inside a replay input'):
                    daily.run_daily(self.config_path)
                self.assertEqual(list(source.iterdir()), [seed])
                self.assertEqual(seed.read_text(encoding='utf-8'), 'immutable input')
        self._acquire_universe_mock.assert_not_called()
        self.download_mock.assert_not_called()

    def test_directory_copy_rejects_recursive_destination(self):
        source = self.root / 'source'
        source.mkdir()
        for destination in (source, source / 'output' / 'copy', self.root):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(ValueError, 'must not overlap'):
                    daily._copy_directory(source, destination, self.root)
        self.assertEqual(list(source.iterdir()), [])

    def test_failed_refresh_does_not_advance_success_and_recovery_keeps_history(self):
        first = self.run_daily()
        old_pointer = (self.output / 'last_success.json').read_bytes()
        old_marker = (Path(first['run_dir']) / 'completion.json').read_bytes()
        self.disclosure_result['status'] = 'partial'
        failed = self.run_daily()
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual((self.output / 'last_success.json').read_bytes(), old_pointer)
        self.disclosure_result['status'] = 'completed'
        recovered = self.run_daily()
        self.assertEqual(recovered['status'], 'succeeded')
        self.assertEqual((Path(first['run_dir']) / 'completion.json').read_bytes(), old_marker)
        self.assertEqual(jobs.status(self.output)['total_runs'], 3)
        self.assertEqual(jobs.status(self.output)['last_success']['run_id'], recovered['run_id'])

    def test_future_official_directory_blocks_before_disclosure_or_pdf_download(self):
        self.universe_result['status'] = 'blocked_future_effective_date'
        self.universe_result['securities'][0]['universe_as_of'] = '2026-05-03'
        job = self.run_daily()
        self.assertEqual(job['status'], 'blocked')
        self._disclosures_mock.assert_not_called()
        self.download_mock.assert_not_called()
        self.assertFalse((self.output / 'last_success.json').exists())
        self.assertEqual(job['stages'][-1]['status'], 'blocked')

    def test_missing_or_future_membership_cannot_be_refreshed_to_today(self):
        self.universe_result['securities'] = []
        job = self.run_daily()
        self.assertEqual(job['status'], 'blocked')
        self.assertFalse((Path(job['run_dir']) / 'refreshed_companies.json').exists())
        membership = json.loads((Path(job['run_dir']) / 'membership.json').read_text(encoding='utf-8'))
        self.assertEqual(membership['failures'][0]['reason'], 'missing_from_official_directory')

    def test_depositary_receipt_is_not_treated_as_an_ordinary_equity(self):
        self.universe_result['securities'][0]['security_type'] = 'depositary_receipt'
        job = self.run_daily()
        self.assertEqual(job['status'], 'blocked')
        membership = json.loads((Path(job['run_dir']) / 'membership.json').read_text(encoding='utf-8'))
        self.assertEqual(membership['failures'][0]['reason'], 'security_is_not_equity')
        self._disclosures_mock.assert_not_called()

    def test_old_source_date_is_preserved_and_cannot_pass_freshness_by_clock(self):
        self.universe_result['securities'][0]['universe_as_of'] = '2026-04-01'
        job = self.run_daily()
        self.assertEqual(job['status'], 'blocked')
        refreshed = json.loads((Path(job['run_dir']) / 'refreshed_companies.json').read_text(encoding='utf-8'))
        self.assertEqual(refreshed[0]['universe_as_of'], '2026-04-01')
        self.assertIn('universe_freshness', job['queries'][0]['reasons'])

    def test_missing_financials_block_but_preserve_discovered_evidence(self):
        self.config.pop('statements')
        self.config.pop('valuations')
        self.save()
        job = self.run_daily()
        self.assertEqual(job['status'], 'blocked')
        self.assertTrue((Path(job['run_dir']) / 'corpus.jsonl').exists())
        self.assertEqual(job['stages'][-1]['name'], 'research_batch')
        self.assertFalse((self.output / 'last_success.json').exists())

    def test_empty_disclosure_window_is_blocked_without_promoting_old_corpus(self):
        self.disclosure_result['ingest_manifest'] = []
        job = self.run_daily()
        self.assertEqual(job['status'], 'blocked')
        self.assertIn('No importable disclosures', job['error'])
        self.download_mock.assert_not_called()

    def test_out_of_scope_disclosure_is_rejected(self):
        self.disclosure_result['ingest_manifest'][0]['available_at'] = '2026-05-02'
        job = self.run_daily()
        self.assertEqual(job['status'], 'failed')
        self.assertIn('outside the requested', job['error'])
        self.download_mock.assert_not_called()

    def test_offline_replay_uses_copied_raw_sources_and_verified_document_cache(self):
        first = self.run_daily()
        self.assertEqual(first['status'], 'succeeded', first.get('error'))
        run = Path(first['run_dir'])
        self.config.update(output_root='offline-output', universe_snapshot=str(run / 'source_snapshots/universe'),
                           disclosure_snapshot=str(run / 'source_snapshots/disclosures'), document_cache=str(run / 'source_cache'))
        self.save()
        self.download_mock.reset_mock()
        self.download_mock.side_effect = AssertionError('offline run must not access network')
        replayed = daily.run_daily(self.config_path)
        self.assertEqual(replayed['status'], 'succeeded', replayed.get('error'))
        self.download_mock.assert_not_called()
        self.assertEqual((Path(replayed['run_dir']) / 'corpus.jsonl').read_bytes(), (run / 'corpus.jsonl').read_bytes())

    def test_rolling_window_uses_asia_shanghai_calendar_and_inclusive_endpoints(self):
        self.config.pop('start_date')
        self.config.pop('end_date')
        self.config['lookback_days'] = 7
        self.save()
        with mock.patch.object(daily, '_clock', return_value=datetime(2026, 4, 30, 17, 0, tzinfo=timezone.utc)):
            _, _, _, _, window = daily._configuration(self.config_path, None, True)
        self.assertEqual(window['end_date'], '2026-05-01')
        self.assertEqual(window['start_date'], '2026-04-24')
        self.assertEqual(window['inclusive_days'], 8)
        _, _, _, _, overridden = daily._configuration(self.config_path, '2026-06-01', True)
        self.assertEqual(overridden['start_date'], '2026-05-25')
        self.assertEqual(overridden['timezone'], 'Asia/Shanghai')

    def test_live_future_cutoff_cannot_bypass_future_effective_directory(self):
        self.config['end_date'] = '2099-01-01'
        self.save()
        with self.assertRaisesRegex(ValueError, 'current Asia/Shanghai date'):
            self.run_daily()
        self._acquire_universe_mock.assert_not_called()

    def test_offline_directory_must_match_requested_sources_and_interval_is_prevalidated(self):
        import universe_sources
        with mock.patch.object(universe_sources, 'replay_snapshot', return_value={'requested_sources': ['hkex']}):
            with self.assertRaisesRegex(ValueError, 'exact requested sources'):
                self.real_acquire(self.root, download=False, sources=['sse'])
        self.config['source_selection']['min_interval_seconds'] = 0.1
        self.save()
        with self.assertRaisesRegex(ValueError, 'between 0.5 and 60'):
            self.run_daily()

    def test_backdated_success_is_retained_without_replacing_newer_current_pointer(self):
        self.config['end_date'] = '2026-05-02'
        self.save()
        first = self.run_daily()
        self.assertEqual(first['status'], 'succeeded', first.get('error'))
        pointer = (self.output / 'last_success.json').read_bytes()
        self.config['end_date'] = '2026-05-01'
        self.save()
        older = self.run_daily()
        self.assertEqual(older['status'], 'succeeded', older.get('error'))
        self.assertEqual(older['promotion']['status'], 'not_promoted')
        self.assertEqual(older['promotion']['reason'], 'older_as_of')
        self.assertEqual((self.output / 'last_success.json').read_bytes(), pointer)
        state = jobs.status(self.output)
        self.assertEqual(state['latest_attempt_as_of'], '2026-05-01')
        self.assertEqual(state['last_success_as_of'], '2026-05-02')
        self.assertFalse(state['is_last_success_latest'])

    def test_invalid_or_ambiguous_config_cannot_start_downloads(self):
        configs = [dict(self.config, typo=True), dict(self.config, lookback_days=7),
                   dict(self.config, end_date='2026-03-01'), dict(self.config, output_root='.'),
                   dict(self.config, queries=[{'name': '../escape', 'query': '机器人'}])]
        for config in configs:
            self.save(config)
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.run_daily()
        self.save()
        with self.assertRaisesRegex(ValueError, 'Offline runs require'):
            daily.run_daily(self.config_path)
        self._acquire_universe_mock.assert_not_called()

    def test_concurrent_same_daily_target_is_rejected_by_os_lock(self):
        code = ('import sys\nfrom daily_update import run_daily\n'
                'try:\n run_daily(sys.argv[1],download=True)\nexcept ValueError as e:\n'
                ' print(str(e));sys.exit(23)\n')
        with ingest_theme.output_lock(self.output):
            process = subprocess.run([sys.executable, '-c', code, str(self.config_path)], cwd=jobs.ROOT,
                                     capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 23, process.stderr.decode())
        self.assertIn(b'lock', process.stdout)
        self.assertFalse((self.output / 'runs').exists())

    def test_cli_does_not_install_scheduler_and_reports_blocked_nonzero(self):
        self.universe_result['status'] = 'blocked_future_effective_date'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(daily.main(['--config', str(self.config_path), '--download']), 3)
            self.assertEqual(daily.main(['--status', str(self.output)]), 0)
            self.assertEqual(daily.main(['--status', str(self.output), '--download']), 2)


if __name__ == '__main__':
    unittest.main()
