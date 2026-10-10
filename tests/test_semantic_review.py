"""Independent adversarial review of semantic discovery's publication boundary."""
import copy
import csv
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import semantic_discover as cli
from semantic_retrieval import RetrievalIndex
from tests.test_engine import annual_history, valuation
from tests.test_readiness import policy


class ReviewModel:
    def __init__(self):
        self.calls = []

    def fingerprint(self):
        return {'independent_review_fixture': True, 'version': 1}

    def encode(self, texts, kind='passage'):
        self.calls.append((kind, list(texts)))
        return np.array([[1., 1.] for _ in texts], dtype=np.float32)

    def rerank(self, query, texts):
        return [1.] * len(texts)

    def statistics(self):
        return {'fixture_only': True}


class SemanticIndependentReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.companies_path = self.root / 'companies.json'
        self.documents_path = self.root / 'documents.jsonl'
        self.output = self.root / 'export'
        self.companies = [dict(ticker='600001', name='Synthetic issuer', market='A',
            sector='Industrials', scope='manufacturing', universe_as_of='2026-05-01')]
        self.documents = [self.doc('support', '本公司生产机器人产品并对外销售。')]
        self.save_inputs()

    def doc(self, identity, text, **changes):
        result = dict(document_id=identity, ticker='600001', source_url='https://example.org/'+identity,
            source_type='announcement', available_at='2026-04-01', page=1, text=text,
            date_basis='official_release', source_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
        result.update(changes)
        return result

    def save_inputs(self):
        self.companies_path.write_text(json.dumps(self.companies), encoding='utf-8')
        self.documents_path.write_text(''.join(json.dumps(doc, ensure_ascii=False)+'\n' for doc in self.documents), encoding='utf-8')

    def run_it(self, **changes):
        args = dict(companies_path=self.companies_path, documents_path=self.documents_path,
            query='机器人', as_of='2026-05-01', output=self.output, method='fixed_bm25', candidate_k=10, rerank_k=10)
        args.update(changes)
        return cli.run(**args)

    def financial_files(self, tickers=('600001',)):
        statements, valuations = self.root/'statements.csv', self.root/'valuations.csv'
        for path, rows in [(statements, [row for ticker in tickers for row in annual_history(ticker)]),
                           (valuations, [valuation(ticker) for ticker in tickers])]:
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        config = policy();config['expected_tickers'] = list(tickers)
        policy_path = self.root/'policy.json';policy_path.write_text(json.dumps(config), encoding='utf-8')
        return dict(statements_path=statements, valuations_path=valuations, policy_path=policy_path)

    def rehash_export(self, filename):
        marker = self.output/'complete.json'
        value = json.loads(marker.read_text(encoding='utf-8'))
        value['files'][filename] = hashlib.sha256((self.output/filename).read_bytes()).hexdigest()
        marker.write_text(json.dumps(value), encoding='utf-8')

    def test_all_product_exit_blocks_old_800g_and_retains_unqualified_counterproof(self):
        docs = [self.doc('old800', '本公司生产800G光模块产品。', available_at='2025-04-01'),
                self.doc('all-exit', '本公司已停止生产光模块产品。')]
        result = RetrievalIndex(self.companies, docs, '2026-05-01', model=ReviewModel()).company_report(
            '800G 光模块', method='dense', candidate_k=1, rerank_k=1)
        row = result['companies'][0]
        self.assertEqual(row['business_status'], 'historical_or_disputed')
        self.assertEqual([x['document_id'] for x in row['counter_evidence']], ['all-exit'])
        self.assertNotIn('800G', row['counter_evidence'][0]['exact_excerpt'])

    def test_unrelated_model_denial_does_not_override_800g(self):
        docs = [self.doc('old800', '本公司生产800G光模块产品。', available_at='2025-04-01'),
                self.doc('other', '本公司不生产100G光模块产品。')]
        row = RetrievalIndex(self.companies, docs, '2026-05-01').company_report(
            '800G 光模块', method='fixed_bm25')['companies'][0]
        self.assertEqual(row['business_status'], 'current_business')

    def test_large_and_tiny_finite_vectors_stay_normalized(self):
        index = RetrievalIndex([], [], '2026-05-01')
        vectors = index._validate_vectors([[1e30, -1e30], [1e-30, 1e-30]], 2)
        self.assertTrue(np.isfinite(vectors).all())
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1., 1.], atol=1e-6)

    def test_ineligible_metadata_is_filtered_before_encoding_and_top_one(self):
        companies = self.companies + [dict(self.companies[0], ticker='FOREIGN', market='US'),
                                      dict(self.companies[0], ticker='FUTURE', universe_as_of='2099-01-01')]
        docs = [*self.documents,
                self.doc('future', 'FUTURE POISON', available_at='2099-01-01'),
                self.doc('foreign', 'FOREIGN POISON', ticker='FOREIGN'),
                self.doc('invisible', 'INVISIBLE POISON', ticker='FUTURE'),
                self.doc('invalid', 'INVALID POISON', source_type='untrusted_social_post')]
        model = ReviewModel()
        result = RetrievalIndex(companies, docs, '2026-05-01', model=model).search(
            'factory assistants', method='dense', candidate_k=1, rerank_k=1)
        self.assertEqual([x['document_id'] for x in result['hits']], ['support'])
        encoded = [text for kind, texts in model.calls if kind == 'passage' for text in texts]
        self.assertEqual(encoded, [self.documents[0]['text']])

    def test_old_theme_support_cannot_be_refreshed_by_unrelated_new_filing(self):
        self.documents[0]['available_at'] = '2023-04-01'
        self.documents.append(self.doc('fresh-unrelated', '公司召开董事会年度会议。'))
        self.save_inputs()
        result = self.run_it(**self.financial_files())
        self.assertIn('theme_business_freshness', result['production_readiness']['reasons'])
        self.assertNotIn('document_coverage', result['production_readiness']['reasons'])
        row = result['companies'][0]
        self.assertEqual(row['financial']['styles'], ['unclassified'])
        self.assertIn('quality_growth', row['financial']['financial_style_candidates'])

    def test_semantic_only_positive_similarity_does_not_unlock_financial_styles(self):
        with patch('neural_models.ModelBundle', return_value=ReviewModel()):
            result = self.run_it(method='dense', model_dir=self.root/'models', query='autonomous factory assistants',
                                 **self.financial_files())
        row = result['companies'][0]
        self.assertTrue(row['semantic_only'])
        self.assertEqual(row['business_status'], 'uncertain')
        self.assertEqual(row['financial']['styles'], ['unclassified'])
        self.assertIn('quality_growth', row['financial']['financial_style_candidates'])
        self.assertFalse(result['production_approved'])

    def test_input_hardlink_alias_cannot_be_opened_as_output_lock(self):
        original = self.companies_path.read_bytes()
        os.link(self.companies_path, self.output.with_name(self.output.name+'.lock'))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.run_it()
        self.assertEqual(self.companies_path.read_bytes(), original)

    def test_semantic_csv_reconciles_all_exported_values_not_only_status(self):
        self.run_it()
        path = self.output/'matches.csv'
        original = path.read_text(encoding='utf-8')
        for field, value in [('relevance_score', '999999'), ('semantic_only', 'True')]:
            rows = list(csv.DictReader(io.StringIO(original)))
            rows[0][field] = value
            buffer = io.StringIO(newline='');writer=csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
            path.write_text(buffer.getvalue(), encoding='utf-8')
            self.rehash_export('matches.csv')
            with self.subTest(field=field), self.assertRaises(ValueError):
                cli.verify_output(self.output)

    def test_semantic_csv_does_not_emit_spreadsheet_formulas(self):
        self.companies[0]['ticker'] = '=1+1'
        self.documents[0]['ticker'] = '=1+1'
        self.save_inputs();self.run_it()
        rows = list(csv.DictReader(io.StringIO((self.output/'matches.csv').read_text(encoding='utf-8'))))
        self.assertEqual(rows[0]['ticker'], "'=1+1")
        self.assertEqual(cli.verify_output(self.output)['status'], 'verified')

    def test_complete_company_evidence_reconciles_with_displayed_records(self):
        self.run_it()
        path = self.output/'results.json'
        value = json.loads(path.read_text(encoding='utf-8'));value['all_companies'] = []
        path.write_text(json.dumps(value), encoding='utf-8');self.rehash_export('results.json')
        with self.assertRaises(ValueError):
            cli.verify_output(self.output)

    def test_new_ingestion_sidecar_during_run_invalidates_snapshot_set(self):
        original = cli.attach_fundamentals
        def create_writing_marker(*args, **kwargs):
            result = original(*args, **kwargs)
            self.documents_path.with_name(self.documents_path.name+'.complete.json').write_text(
                '{"state":"writing"}', encoding='utf-8')
            return result
        with patch.object(cli, 'attach_fundamentals', side_effect=create_writing_marker):
            with self.assertRaisesRegex(ValueError, 'sidecar|input|ingestion'):
                self.run_it()
        self.assertFalse((self.output/'complete.json').exists())

    def test_removed_ingestion_sidecar_invalidates_snapshot_set(self):
        audit = self.documents_path.with_name(self.documents_path.name+'.manifest.json')
        audit.write_text(json.dumps({'schema_version': 1, 'status': 'completed',
            'output_sha256': hashlib.sha256(self.documents_path.read_bytes()).hexdigest()}), encoding='utf-8')
        original = cli.attach_fundamentals
        def remove_audit(*args, **kwargs):
            result = original(*args, **kwargs);audit.unlink();return result
        with patch.object(cli, 'attach_fundamentals', side_effect=remove_audit):
            with self.assertRaisesRegex(ValueError, 'sidecar'):
                self.run_it()

    def test_input_change_while_exporting_leaves_incomplete_marker(self):
        atomic = cli._atomic_write
        def change_after_first_export(path, content):
            atomic(path, content)
            if path.name == 'results.json':
                self.documents_path.write_bytes(self.documents_path.read_bytes()+b' ')
        with patch.object(cli, '_atomic_write', side_effect=change_after_first_export):
            with self.assertRaisesRegex(ValueError, 'input files changed'):
                self.run_it()
        with self.assertRaisesRegex(ValueError, 'completion marker'):
            cli.verify_output(self.output)

    def test_complete_and_displayed_style_lists_are_both_reconciled(self):
        self.run_it()
        path = self.output/'results.json'
        original = json.loads(path.read_text(encoding='utf-8'))
        for key in ('style_lists', 'all_matched_style_lists'):
            report = copy.deepcopy(original);report[key]['quality_growth'].append('600001')
            path.write_text(json.dumps(report), encoding='utf-8');self.rehash_export('results.json')
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                cli.verify_output(self.output)

    def test_csv_schema_and_malformed_quote_are_rejected_with_empty_matches(self):
        self.run_it(query='unrelated vocabulary')
        path = self.output/'matches.csv'
        for invalid in ('ticker,ticker\n', 'ticker,business_status\n', '"unterminated'):
            path.write_text(invalid, encoding='utf-8');self.rehash_export('matches.csv')
            with self.subTest(csv=invalid), self.assertRaises(ValueError):
                cli.verify_output(self.output)

    def test_display_limit_keeps_all_evidence_and_style_lists(self):
        self.companies.append(dict(self.companies[0], ticker='600002'))
        self.documents.append(self.doc('second', '本公司主要生产机器人产品。', ticker='600002'))
        self.save_inputs()
        report = self.run_it(limit=1)
        self.assertEqual(len(report['companies']), 1)
        self.assertEqual(len(report['all_companies']), 2)
        self.assertEqual(report['companies'], report['all_companies'][:1])
        self.assertEqual(len(report['all_matched_style_lists']['unclassified']), 2)
        self.assertEqual(cli.verify_output(self.output)['all_companies'], 2)

    def test_cli_normal_execution_and_verify_return_structured_output(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli.main(['--method', 'fixed_bm25', '--companies', str(self.companies_path),
                '--documents', str(self.documents_path), '--query', '机器人', '--as-of', '2026-05-01',
                '--output', str(self.output), '--cache', str(self.root/'vectors')])
        self.assertEqual(result, 0)
        self.assertFalse(json.loads(stdout.getvalue())['production_approved'])
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = cli.main(['--verify', str(self.output)])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue())['status'], 'verified')

    def test_cli_missing_required_inputs_exits_two_without_creating_export(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            cli.main([])
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_cli_missing_neural_runtime_returns_actionable_failure(self):
        message = 'Local semantic dependencies are missing; install requirements-semantic.txt into .venv-semantic'
        stdout = io.StringIO()
        with patch('neural_models.ModelBundle._check_files'), \
             patch('neural_models._dependencies', side_effect=RuntimeError(message)) as dependencies, \
             redirect_stdout(stdout):
            result = cli.main(['--method', 'dense', '--companies', str(self.companies_path),
                '--documents', str(self.documents_path), '--query', '机器人', '--as-of', '2026-05-01',
                '--models', str(self.root/'models'), '--output', str(self.output),
                '--cache', str(self.root/'vectors')])
        dependencies.assert_called_once_with()
        self.assertEqual(result, 2)
        self.assertEqual(json.loads(stdout.getvalue()), {'status': 'failed', 'error': message})
        self.assertFalse((self.output/'complete.json').exists())

    def test_cli_malformed_completion_returns_failure_instead_of_traceback(self):
        self.output.mkdir()
        for invalid in ([], None, {'schema_version': 1, 'files': []}, {'schema_version': 1, 'status': 'writing'}):
            (self.output/'complete.json').write_text(json.dumps(invalid), encoding='utf-8')
            stdout = io.StringIO()
            with self.subTest(marker=invalid), redirect_stdout(stdout):
                self.assertEqual(cli.main(['--verify', str(self.output)]), 2)
            self.assertEqual(json.loads(stdout.getvalue())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
