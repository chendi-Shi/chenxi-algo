"""Check the benchmark's accounting, provenance and honest result separation."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from validation import business_claim_eval as evaluation


class BusinessClaimEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = evaluation.validate_dataset(evaluation._strict_json(evaluation.DEFAULT_DATASET.read_bytes()))

    def case(self, name):
        return next(case for case in self.dataset['cases'] if case['id'] == name)

    def test_confusion_and_macro_f1_count_missed_predictions(self):
        result = evaluation.classification_metrics([
            ('direct_business', 'direct_business'), ('direct_business', 'planned'),
            ('planned', 'planned'), ('planned', 'not_retrieved')], evaluation.CLAIM_LABELS)
        self.assertEqual(result['cases'], 4)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['macro_f1'], .388889)
        self.assertEqual(result['per_class']['direct_business']['f1'], .666667)
        self.assertEqual(result['confusion_matrix']['planned']['not_retrieved'], 1)
        self.assertIn('not_retrieved', result['macro_classes'])
        self.assertIsNone(result['per_class']['negated']['f1'])

    def test_empty_task_has_no_accuracy_estimate(self):
        result = evaluation.classification_metrics([], evaluation.BUSINESS_LABELS)
        self.assertIsNone(result['macro_f1'])
        self.assertIsNone(result['accuracy'])
        self.assertEqual(result['cases'], 0)
        with self.assertRaises(ValueError):
            evaluation.classification_metrics([('current_business', 'arbitrary')], evaluation.BUSINESS_LABELS)

    def test_public_quotes_and_original_source_metadata_match_existing_examples(self):
        originals = {row['document_id']: row for row in [json.loads(line) for line in
            (evaluation.ROOT / 'data/theme_documents.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]}
        public = [case for case in self.dataset['cases'] if case['kind'] == 'public_excerpt']
        self.assertEqual(len(public), 9)
        for case in public:
            original = originals[case['public_reference']['document_id']]
            document = case['documents'][0]
            for field in ('text', 'source_url', 'source_sha256', 'page', 'available_at', 'date_basis'):
                self.assertEqual(document[field], original[field])
            self.assertNotIn('fixture_kind', document)

    def test_dataset_labels_and_fixture_provenance_cannot_silently_change_shape(self):
        mutations = [lambda data: data['cases'].append(copy.deepcopy(data['cases'][0])),
                     lambda data: data['cases'][0].update(kind='real_market'),
                     lambda data: data['cases'][0]['documents'][0].update(text='changed quote'),
                     lambda data: data.update(independently_annotated=True),
                     lambda data: data.update(schema_version=True),
                     lambda data: data['cases'][0].update(expected={'claim_status': 'invented_label'})]
        for mutate in mutations:
            data = copy.deepcopy(self.dataset)
            mutate(data)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                evaluation.validate_dataset(data)
        with self.assertRaises(ValueError):
            evaluation._strict_json(b'{"schema_version":1,"schema_version":2}')

    def test_pure_denials_are_evaluated_even_when_company_is_not_returned(self):
        result = evaluation.predict_case(self.case('synthetic-denial'))
        self.assertEqual(result['claim_status'], 'negated')
        self.assertEqual(result['business_status'], 'no_match')
        self.assertTrue(result['assertions'])
        self.assertTrue(result['negated_only'])

    def test_future_and_model_mismatch_remain_explicit_missed_retrievals(self):
        for name in ('synthetic-future-assertion', 'synthetic-model-mismatch', 'synthetic-separate-window-model'):
            with self.subTest(name=name):
                result = evaluation.predict_case(self.case(name))
                self.assertEqual(result['claim_status'], 'not_retrieved')
                self.assertEqual(result['business_status'], 'no_match')
                self.assertEqual(result['assertions'], [])

    def test_complete_report_separates_public_and_synthetic_without_aggregate_score(self):
        first = evaluation.evaluate()
        self.assertEqual(first, evaluation.evaluate())
        self.assertEqual(first['status'], 'evaluated')
        self.assertFalse(first['independently_annotated'])
        self.assertFalse(first['held_out'])
        self.assertFalse(first['production_approved'])
        self.assertNotIn('accuracy', first)
        self.assertNotIn('macro_f1', first)
        self.assertEqual(set(first['metrics_by_data_kind']), set(evaluation.KINDS))
        self.assertEqual(sum(group['cases'] for group in first['metrics_by_data_kind'].values()), len(self.dataset['cases']))
        for kind, group in first['metrics_by_data_kind'].items():
            self.assertEqual(group['business_status']['cases'], group['cases'])
            self.assertEqual(group['cases'], sum(case['kind'] == kind for case in self.dataset['cases']))
        for row in first['cases']:
            self.assertEqual(row['id'] in first['misclassified_case_ids'], not all(row['correct'].values()))

    def test_cli_writes_json_without_local_source_paths_and_does_not_overwrite_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'evaluation.json'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(evaluation.main(['--output', str(path)]), 0)
            report = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(report['dataset_sha256'], evaluation._sha(evaluation.DEFAULT_DATASET.read_bytes()))
            self.assertNotIn(str(evaluation.ROOT), path.read_text(encoding='utf-8'))
            before = evaluation.DEFAULT_DATASET.read_bytes()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(evaluation.main(['--output', str(evaluation.DEFAULT_DATASET)]), 2)
            self.assertEqual(evaluation.DEFAULT_DATASET.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
