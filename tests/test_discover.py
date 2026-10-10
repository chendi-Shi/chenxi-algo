import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import discover
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


if __name__ == '__main__':
    unittest.main()
