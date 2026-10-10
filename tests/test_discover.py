import copy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import io
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

import discover
from ingest_theme import output_lock
from tests.test_engine import annual_history, valuation


class DiscoveryWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.companies = self.root / 'companies.json'
        self.docs = self.root / 'docs.jsonl'
        self.output = self.root / 'output'
        self.companies.write_text(json.dumps([{'ticker': '600001', 'name': '示例制造', 'market': 'A',
                'sector': 'Industrials', 'scope': 'manufacturing', 'universe_as_of': '2026-01-01'}]), encoding='utf-8')
        self.docs.write_text(json.dumps({'document_id': 'one', 'ticker': '600001', 'available_at': '2026-04-01',
            'source_url': 'https://example.org/report.pdf', 'source_type': 'annual_report', 'page': 1,
            'text': '本公司生产机器人产品并对外销售。'}, ensure_ascii=False) + '\n', encoding='utf-8')

    def run_it(self, **updates):
        kwargs = dict(query='机器人', as_of='2026-05-01', companies_path=self.companies,
                      documents_path=self.docs, output=self.output)
        kwargs.update(updates)
        return discover.run_discovery(**kwargs)

    def test_complete_offline_workflow_and_export_verification(self):
        result = self.run_it()
        self.assertEqual(result['style_lists']['unclassified'], ['600001'])
        self.assertEqual(discover.verify_discovery(self.output, check_inputs=True)['status'], 'verified')
        self.assertEqual(result['companies'][0]['evidence'][0]['page'], 1)

    def long_input_directory(self):
        first = self.root / ('long-' + 'a' * 60)
        directory = first / ('nested-' + 'b' * 60) / ('nested-' + 'c' * 60) / ('nested-' + 'd' * 60)
        self.assertTrue(discover.filesystem_path(first).is_relative_to(discover.filesystem_path(self.root)))
        self.addCleanup(lambda: shutil.rmtree(discover.filesystem_path(first)) if first.exists() else None)
        discover.filesystem_path(directory).mkdir(parents=True)
        discover.filesystem_path(directory / 'companies.json').write_bytes(self.companies.read_bytes())
        discover.filesystem_path(directory / 'docs.jsonl').write_bytes(self.docs.read_bytes())
        self.assertGreater(len(str(directory)), 260)
        self.assertFalse(str(directory).startswith('\\\\?\\'))
        return directory

    def test_direct_api_accepts_unprefixed_long_inputs_output_and_verification(self):
        directory = self.long_input_directory()
        output = directory / 'api-output'
        result = self.run_it(companies_path=directory / 'companies.json',
                             documents_path=directory / 'docs.jsonl', output=output)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(discover.verify_discovery(str(output), check_inputs=True)['status'], 'verified')

    def test_direct_cli_accepts_unprefixed_long_inputs_and_output(self):
        directory = self.long_input_directory()
        output = directory / 'cli-output'
        result = subprocess.run([sys.executable, discover.__file__, '--query', '机器人',
            '--as-of', '2026-05-01', '--companies', str(directory / 'companies.json'),
            '--documents', str(directory / 'docs.jsonl'), '--output', str(output)],
            capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(discover.verify_discovery(output, check_inputs=True)['status'], 'verified')

    def test_long_path_aliases_preserve_input_and_lock_collision_guards(self):
        directory = self.long_input_directory()
        output = directory / 'guard-output'
        discover.filesystem_path(output).mkdir()
        collision = output / 'results.json'
        original = self.companies.read_bytes()
        discover.filesystem_path(collision).write_bytes(original)
        for actual_output, source in ((output, discover.filesystem_path(collision)),
                                      (discover.filesystem_path(output), collision)):
            with self.subTest(output=str(actual_output)), self.assertRaisesRegex(ValueError, 'overwrite'):
                self.run_it(companies_path=source, output=actual_output)
        self.assertEqual(discover.filesystem_path(collision).read_bytes(), original)
        lock = directory / 'lock-output.lock'
        discover.filesystem_path(lock).write_bytes(b'')
        for actual_output, source in ((directory / 'lock-output', discover.filesystem_path(lock)),
                                      (discover.filesystem_path(directory / 'lock-output'), lock)):
            with self.subTest(output=str(actual_output)), self.assertRaisesRegex(ValueError, 'output lock'):
                self.run_it(companies_path=source, output=actual_output)
        self.assertEqual(discover.filesystem_path(lock).read_bytes(), b'')

    def test_financial_csv_integration_preserves_business_rank(self):
        statements, prices = self.root / 'statements.csv', self.root / 'values.csv'
        for path, rows in ((statements, annual_history()), (prices, [valuation()])):
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        result = self.run_it(statements_path=statements, valuations_path=prices)
        self.assertEqual(result['style_lists']['quality_growth'], ['600001'])
        self.assertEqual(result['companies'][0]['ticker'], '600001')

    def test_tampered_export_fails_hash_verification(self):
        self.run_it()
        with (self.output / 'matches.csv').open('a', encoding='utf-8') as handle:
            handle.write('tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            discover.verify_discovery(self.output)

    def test_changed_input_is_detected(self):
        self.run_it()
        self.docs.write_text('', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'input hash mismatch'):
            discover.verify_discovery(self.output, check_inputs=True)

    def test_mutation_during_run_does_not_produce_completion(self):
        real = discover.attach_fundamentals
        def change(*args, **kwargs):
            result = real(*args, **kwargs)
            self.docs.write_text('', encoding='utf-8')
            return result
        with patch.object(discover, 'attach_fundamentals', side_effect=change):
            with self.assertRaisesRegex(ValueError, 'input changed'):
                self.run_it()
        self.assertFalse((self.output / 'completion.json').exists())

    def test_reproducible_run_id(self):
        a, b = self.run_it(), self.run_it(output=self.root / 'other')
        self.assertEqual(a['run_id'], b['run_id'])
        self.assertEqual((self.output / 'results.json').read_bytes(), (self.root / 'other/results.json').read_bytes())

    def test_no_match_and_market_filter_have_explicit_status(self):
        self.assertEqual(self.run_it(query='不存在的产品')['status'], 'no_matches')
        self.assertEqual(self.run_it(market='HK')['status'], 'no_matches')

    def test_future_document_is_not_visible(self):
        self.assertEqual(self.run_it(as_of='2026-02-01')['status'], 'no_eligible_documents')

    def test_invalid_json_and_config_fail_explicitly(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                discover.strict_json(text)
        config = self.root / 'config.json'
        config.write_text('{"typo":1}', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.run_it(config_path=config)

    def test_output_cannot_overwrite_input(self):
        collision = self.root / 'results.json'
        collision.write_bytes(self.companies.read_bytes())
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            self.run_it(companies_path=collision, output=self.root)
        self.assertEqual(collision.read_bytes(), self.companies.read_bytes())

    def test_csv_semantic_mismatch_rejected_even_when_file_hash_is_consistent(self):
        self.run_it()
        path = self.output / 'matches.csv'
        content = path.read_text(encoding='utf-8-sig').replace('unclassified', 'quality_growth')
        path.write_text(content, encoding='utf-8-sig')
        completion = json.loads((self.output / 'completion.json').read_text(encoding='utf-8'))
        completion['files']['matches.csv'] = discover._hash(path.read_bytes())
        (self.output / 'completion.json').write_text(json.dumps(completion), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'reconciliation'):
            discover.verify_discovery(self.output)

    def test_partial_or_changed_ingestion_output_is_rejected(self):
        audit = self.docs.with_name(self.docs.name + '.manifest.json')
        for status, sha in [('partial', discover._hash(self.docs.read_bytes())), ('completed', 'bad')]:
            audit.write_text(json.dumps({'status': status, 'output_sha256': sha}), encoding='utf-8')
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, 'ingestion'):
                self.run_it()
        audit.write_text(json.dumps({'status': 'completed', 'output_sha256': discover._hash(self.docs.read_bytes())}), encoding='utf-8')
        self.assertEqual(self.run_it()['status'], 'completed')

    def test_fictional_demonstration_exercises_all_five_lists(self):
        from demo_theme import create_theme_demo
        directory = create_theme_demo(self.root / 'fictional')
        result = self.run_it(as_of='2026-10-10', companies_path=directory / 'companies.json',
                             documents_path=directory / 'documents.jsonl', statements_path=directory / 'statements.csv',
                             valuations_path=directory / 'valuations.csv')
        self.assertTrue(all(result['style_lists'].values()))
        self.assertEqual(result['style_lists']['loss_watchlist'], ['DEMO007'])

    def test_concurrent_same_target_is_rejected(self):
        with output_lock(self.output):
            with self.assertRaisesRegex(ValueError, 'lock'):
                self.run_it()
        self.assertEqual(self.run_it()['status'], 'completed')

    def test_cli_requires_explicit_data_or_demo(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            discover.main(['--query', '机器人', '--as-of', '2026-05-01'])
        self.assertEqual(raised.exception.code, 2)

    def test_production_coverage_failure_has_nonzero_exit_and_persisted_reasons(self):
        from tests.test_readiness import policy
        cfg = self.root / 'policy.json'
        settings = policy()
        settings['expected_tickers'].append('600002')
        cfg.write_text(json.dumps(settings), encoding='utf-8')
        with redirect_stdout(io.StringIO()):
            code = discover.main(['--query', '机器人', '--as-of', '2026-05-01',
                                  '--companies', str(self.companies), '--documents', str(self.docs),
                                  '--production-policy', str(cfg), '--output', str(self.output)])
        self.assertEqual(code, 3)
        report = json.loads((self.output / 'results.json').read_text(encoding='utf-8'))
        self.assertEqual(report['production_readiness']['status'], 'blocked')
        self.assertFalse(discover.verify_discovery(self.output)['production_approved'])

    def test_financial_input_uses_fingerprinted_snapshot_despite_temporary_file_change(self):
        statements = self.root / 'statements.csv'
        rows = annual_history()
        with statements.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        original = statements.read_bytes()
        real_discovery, real_attach = discover.discover_companies, discover.attach_fundamentals
        def swap_before_financial_parse(*args, **kwargs):
            result = real_discovery(*args, **kwargs)
            statements.write_bytes(original.replace(b'20,20,18', b'-20,-20,-18'))
            return result
        def restore_after_financial_parse(*args, **kwargs):
            result = real_attach(*args, **kwargs)
            statements.write_bytes(original)
            return result
        with patch.object(discover, 'discover_companies', side_effect=swap_before_financial_parse), \
                patch.object(discover, 'attach_fundamentals', side_effect=restore_after_financial_parse):
            result = self.run_it(statements_path=statements)
        self.assertFalse(result['companies'][0]['financial']['loss_making'])
        self.assertIn('quality_growth', result['companies'][0]['financial']['styles'])

    def test_strict_csv_snapshot_rejects_duplicate_headers_extra_cells_and_unclosed_quote(self):
        for content in (b'x,x\n1,2\n', b'x\n1,2\n', b'x\n"unclosed', b'"unterminated'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                discover._load_csv_snapshot(content)

    def ingestion_bundle(self):
        audit = self.docs.with_name(self.docs.name + '.manifest.json')
        marker = self.docs.with_name(self.docs.name + '.complete.json')
        report = {'schema_version': 2, 'status': 'completed', 'run_id': 'fixture-ingestion',
                  'documents_written': 1, 'output_sha256': discover._hash(self.docs.read_bytes())}
        audit.write_text(json.dumps(report), encoding='utf-8')
        commit = {'schema_version': 1, 'state': 'complete', 'run_id': 'fixture-ingestion',
                  'documents_written': 1, 'files': {self.docs.name: discover._hash(self.docs.read_bytes()),
                                                    audit.name: discover._hash(audit.read_bytes())}}
        marker.write_text(json.dumps(commit), encoding='utf-8')
        return audit, marker, commit

    def test_captured_ingestion_transaction_is_verified_as_a_whole(self):
        audit, marker, commit = self.ingestion_bundle()
        self.assertEqual(self.run_it()['status'], 'completed')
        cases = [dict(commit, state='writing'), dict(commit, files={}),
                 dict(commit, run_id='different'), dict(commit, documents_written=2),
                 dict(commit, documents_written=True)]
        for invalid in cases:
            with self.subTest(invalid=invalid):
                marker.write_text(json.dumps(invalid), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'ingestion'):
                    self.run_it()
        marker.unlink()
        with self.assertRaisesRegex(ValueError, 'completion marker is missing'):
            self.run_it()

    def test_ingestion_rewrite_between_snapshot_reads_cannot_capture_writing_marker(self):
        audit, marker, commit = self.ingestion_bundle()
        original_read = Path.read_bytes
        def begin_rewrite_after_audit_read(path):
            content = original_read(path)
            if discover.filesystem_path(path) == discover.filesystem_path(audit):
                marker.write_text(json.dumps(dict(commit, state='writing')), encoding='utf-8')
            return content
        with patch.object(Path, 'read_bytes', begin_rewrite_after_audit_read):
            with self.assertRaisesRegex(ValueError, 'snapshot transaction is not complete'):
                self.run_it()
        self.assertFalse((self.output / 'completion.json').exists())

    def test_ingestion_marker_without_audit_is_rejected(self):
        audit, marker, commit = self.ingestion_bundle()
        audit.unlink()
        with self.assertRaisesRegex(ValueError, 'no audit report'):
            self.run_it()

    def test_new_ingestion_sidecar_during_run_cannot_escape_input_snapshot(self):
        original = discover.attach_fundamentals
        def create_marker(*args, **kwargs):
            report = original(*args, **kwargs)
            self.docs.with_name(self.docs.name + '.complete.json').write_text('{"state":"writing"}', encoding='utf-8')
            return report
        with patch.object(discover, 'attach_fundamentals', side_effect=create_marker):
            with self.assertRaisesRegex(ValueError, 'sidecar set changed'):
                self.run_it()

    def test_lock_path_cannot_open_an_input_even_when_input_is_empty(self):
        collision = self.output.with_name(self.output.name + '.lock')
        collision.write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'output lock'):
            self.run_it(companies_path=collision)
        self.assertEqual(collision.read_bytes(), b'')

    def test_display_limit_preserves_all_candidate_evidence_and_style_checks(self):
        from demo_theme import create_theme_demo
        directory = create_theme_demo(self.root / 'all-evidence')
        result = self.run_it(as_of='2026-10-10', companies_path=directory / 'companies.json',
                             documents_path=directory / 'documents.jsonl', statements_path=directory / 'statements.csv',
                             valuations_path=directory / 'valuations.csv', limit=1)
        self.assertEqual(len(result['companies']), 1)
        self.assertGreater(len(result['all_companies']), 1)
        self.assertEqual(result['companies'], result['all_companies'][:1])
        self.assertTrue(all(result['all_matched_style_lists'].values()))
        for company in result['all_companies']:
            self.assertTrue(company['evidence'])
            self.assertIn('checks', company['financial'])
        verified = discover.verify_discovery(self.output, check_inputs=True)
        self.assertEqual(verified['all_matched_companies'], len(result['all_companies']))
        with (self.output / 'matches.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]['business_status'], result['companies'][0]['business_status'])

    def test_full_style_lists_are_reconciled_with_full_evidence_records(self):
        self.run_it()
        report_path = self.output / 'results.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        report['all_matched_style_lists']['quality_growth'].append('600001')
        report_path.write_text(json.dumps(report), encoding='utf-8')
        marker_path = self.output / 'completion.json'
        commit = json.loads(marker_path.read_text(encoding='utf-8'))
        commit['files']['results.json'] = discover._hash(report_path.read_bytes())
        marker_path.write_text(json.dumps(commit), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'all_matched_style_lists does not reconcile'):
            discover.verify_discovery(self.output)


if __name__ == '__main__':
    unittest.main()
