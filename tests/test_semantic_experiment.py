"""Experiment aggregation and pre-inference input checks; no real models."""
import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ingest_theme import filesystem_path, output_lock
from validation import run_semantic_experiment as experiment


def hit(ticker, score, identity):
    return dict(ticker=ticker, score=score, document_id=identity, chunk_id=identity,
                page=1, char_start=0, char_end=12)


class SemanticExperimentTests(unittest.TestCase):
    def test_company_ranking_uses_best_passage_not_duplicate_sum(self):
        hits = [hit('A', 2.0, 'a1'), hit('B', 3.0, 'b1'), hit('A', 2.0, 'a2'),
                hit('A', 4.0, 'a3')]
        result = experiment._ranking(hits)
        self.assertEqual([(row['ticker'], row['score']) for row in result], [('A', 4.0), ('B', 3.0)])
        self.assertEqual(result[0]['evidence'][0]['document_id'], 'a3')

    def test_company_aggregation_retains_negative_logits_and_breaks_ties_by_ticker(self):
        result = experiment._ranking([hit('B', -2.0, 'b'), hit('A', -2.0, 'a')])
        self.assertEqual([row['ticker'] for row in result], ['A', 'B'])
        self.assertEqual([row['score'] for row in result], [-2.0, -2.0])

    def test_passage_budget_is_not_a_company_budget(self):
        hits = [hit('A', 3.0, 'a1'), hit('A', 2.5, 'a2'), hit('B', 2.0, 'b')]
        self.assertEqual(len(experiment._ranking(hits[:2])), 1)
        self.assertEqual(len(experiment._ranking(hits)), 2)

    def test_frozen_corpus_mismatch_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'source.jsonl').write_text('changed corpus\n', encoding='utf-8')
            dataset = root / 'dataset.json'
            dataset.write_text(json.dumps({'corpus_inputs': [
                {'path': 'source.jsonl', 'sha256': '0' * 64}]}), encoding='utf-8')
            with mock.patch.object(experiment, 'ROOT', root), mock.patch.object(experiment, 'ModelBundle') as model:
                with self.assertRaisesRegex(ValueError, 'frozen corpus hash mismatch'):
                    experiment.run(dataset_path=dataset, model_dir=root / 'models',
                                   cache_dir=root / 'cache', output=root / 'report.json')
                model.assert_not_called()
            self.assertFalse((root / 'report.json').exists())


class SemanticExperimentOutputProtectionTests(unittest.TestCase):
    """Guard regressions use a valid tiny corpus and stop before any model loads."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = filesystem_path(self.tmp.name)
        self.dataset = self.root / 'labels.json'
        self.output = self.root / 'exports' / 'report.json'
        self.output.parent.mkdir()
        company = dict(ticker='A', name='Fixture issuer', market='A', sector='Technology',
                       scope='technology', universe_as_of='2026-01-01')
        doc = dict(document_id='source-a', ticker='A', available_at='2026-01-01',
                   source_url='https://example.org/annual.pdf', source_type='annual_report',
                   source_sha256='a' * 64, date_basis='official_release', page=1,
                   text='本公司生产机器人产品。')
        contents = {'data/theme_companies.json': json.dumps([company]),
                    'output/theme_full_documents.jsonl': json.dumps(doc) + '\n',
                    'data/theme_documents.jsonl': '', 'data/theme_sources.json': '[]'}
        for name in ('semantic_retrieval.py', 'neural_models.py', 'theme_search.py', 'ingest_theme.py',
                     'validation/semantic_eval.py', 'validation/run_semantic_experiment.py'):
            contents[name] = '# Frozen fixture code bytes; this file is never executed.\n'
        for name, text in contents.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
        dataset = dict(candidate_tickers=['A'], as_of='2026-05-01', maximum_grade=2,
            positive_grade_threshold=1, acceptance_status='synthetic_test_fixture',
            source_documents=[{key: value for key, value in doc.items() if key != 'text'}],
            queries=[dict(query_id='q1', group_id='g1', text='机器人', judgments=[
                dict(ticker='A', grade=2, reason='Synthetic fixture evidence.', source_document_ids=['source-a'])])],
            corpus_inputs=[dict(path=name, sha256=hashlib.sha256((self.root / name).read_bytes()).hexdigest())
                for name in ('output/theme_full_documents.jsonl', 'data/theme_documents.jsonl', 'data/theme_sources.json')])
        self.dataset.write_text(json.dumps(dataset), encoding='utf-8')
        self.protected = {'dataset': self.dataset, 'companies': self.root / 'data/theme_companies.json',
                          'corpus': self.root / 'output/theme_full_documents.jsonl',
                          'corpus_manifest': self.root / 'data/theme_sources.json',
                          'code': self.root / 'semantic_retrieval.py'}
        self.before = {path: path.read_bytes() for path in self.protected.values()}

    def invoke(self, output, *, dataset=None):
        return experiment.run(dataset_path=self.dataset if dataset is None else dataset,
            model_dir=self.root / 'models', cache_dir=self.root / 'vectors', output=output)

    def assert_sources_unchanged(self):
        for path, original in self.before.items():
            self.assertEqual(path.read_bytes(), original, str(path))

    def test_output_cannot_equal_any_protected_input_before_model_load(self):
        for role, target in self.protected.items():
            with self.subTest(role=role), mock.patch.object(experiment, 'ROOT', self.root), \
                    mock.patch.object(experiment, 'ModelBundle', side_effect=AssertionError('model must not load')) as model:
                with self.assertRaisesRegex(ValueError, 'overlap|collid|conflict|overwrite'):
                    self.invoke(target)
                model.assert_not_called()
                self.assert_sources_unchanged()

    def test_partial_and_lock_cannot_equal_dataset_path_before_model_load(self):
        for suffix in ('.partial', '.lock'):
            source = self.root / ('alternate-labels.json' + suffix)
            original = self.dataset.read_bytes()
            source.write_bytes(original)
            target = self.root / 'alternate-labels.json'
            with self.subTest(suffix=suffix), mock.patch.object(experiment, 'ROOT', self.root), \
                    mock.patch.object(experiment, 'ModelBundle', side_effect=AssertionError('model must not load')) as model:
                with self.assertRaisesRegex(ValueError, 'overlap|collid|conflict|overwrite'):
                    self.invoke(target, dataset=source)
                model.assert_not_called()
                self.assertEqual(source.read_bytes(), original)

    def test_output_partial_and_lock_hardlinks_to_inputs_are_rejected(self):
        for role, source in self.protected.items():
            for suffix in ('', '.partial', '.lock'):
                target = self.output.with_name('report-' + role + '-' + (suffix[1:] or 'main') + '.json')
                alias = Path(str(target) + suffix)
                os.link(source, alias)
                with self.subTest(role=role, suffix=suffix), mock.patch.object(experiment, 'ROOT', self.root), \
                        mock.patch.object(experiment, 'ModelBundle', side_effect=AssertionError('model must not load')) as model:
                    with self.assertRaisesRegex(ValueError, 'overlap|collid|conflict|overwrite'):
                        self.invoke(target)
                    model.assert_not_called()
                    self.assert_sources_unchanged()

    def test_existing_run_lock_blocks_same_target_before_model_load(self):
        with output_lock(self.output), mock.patch.object(experiment, 'ROOT', self.root), \
                mock.patch.object(experiment, 'ModelBundle', side_effect=AssertionError('model must not load')) as model:
            with self.assertRaisesRegex(ValueError, '[Ll]ock|[Aa]nother run'):
                self.invoke(self.output)
            model.assert_not_called()
        self.assert_sources_unchanged()

    def test_existing_run_lock_does_not_block_a_different_target(self):
        with output_lock(self.output), mock.patch.object(experiment, 'ROOT', self.root), \
                mock.patch.object(experiment, 'ModelBundle', side_effect=AssertionError('inference sentinel')) as model:
            with self.assertRaisesRegex(AssertionError, 'inference sentinel'):
                self.invoke(self.output.with_name('independent.json'))
            model.assert_called_once()
        self.assert_sources_unchanged()


if __name__ == '__main__':
    unittest.main()
