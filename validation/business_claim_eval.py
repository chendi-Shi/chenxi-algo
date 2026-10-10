"""Evaluate the existing rule classifier on separately labelled finite corpora.

This is an offline development benchmark, not an independent market study or a
trained classifier. Public excerpts and invented adversarial cases never share
an aggregate accuracy. A missed assertion is scored, not dropped.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import theme_search as theme
from ingest_theme import filesystem_path

DEFAULT_DATASET = ROOT / 'validation/business_claim_dataset.json'
CLAIM_LABELS = ('direct_business', 'upstream_or_downstream', 'planned', 'negated',
                'uncertain', 'not_retrieved')
BUSINESS_LABELS = ('current_business', 'planned_business', 'historical_or_disputed',
                   'uncertain', 'no_match')
KINDS = ('public_excerpt', 'synthetic_adversarial')


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f'Non-finite JSON value: {value}')
    return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=reject)


def validate_dataset(dataset):
    if not isinstance(dataset, dict) or type(dataset.get('schema_version')) is not int or dataset['schema_version'] != 1:
        raise ValueError('Dataset schema_version must be 1')
    if not isinstance(dataset.get('dataset_id'), str) or not dataset['dataset_id']:
        raise ValueError('Dataset id is required')
    if dataset.get('independently_annotated') is not False or dataset.get('held_out') is not False:
        raise ValueError('This dataset must retain its development/non-independent annotation disclosure')
    cases = dataset.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('Dataset must contain cases')
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError('Each case must be an object')
        name = case.get('id')
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError('Case ids must be unique nonempty strings')
        seen.add(name)
        if case.get('kind') not in KINDS:
            raise ValueError(f'{name}: unknown data kind')
        if not isinstance(case.get('query'), str) or not case['query'].strip():
            raise ValueError(f'{name}: query is required')
        cutoff = theme._date(case.get('as_of'), 'as_of')
        if not isinstance(case.get('tags'), list) or not case['tags'] or any(
                not isinstance(tag, str) or not tag for tag in case['tags']):
            raise ValueError(f'{name}: explicit coverage tags are required')
        if not isinstance(case.get('annotation_reason'), str) or not case['annotation_reason']:
            raise ValueError(f'{name}: annotation rationale is required')
        expected = case.get('expected')
        if (not isinstance(expected, dict) or not expected or set(expected) - {'claim_status', 'business_status'}
                or expected.get('claim_status', CLAIM_LABELS[0]) not in CLAIM_LABELS
                or expected.get('business_status', BUSINESS_LABELS[0]) not in BUSINESS_LABELS):
            raise ValueError(f'{name}: expected labels are invalid')
        documents = case.get('documents')
        if not isinstance(documents, list) or not documents:
            raise ValueError(f'{name}: documents are required')
        if 'claim_status' in expected and len(documents) != 1:
            raise ValueError(f'{name}: assertion-level labels require a single source excerpt')
        audit = {'rejected_companies': [], 'rejected_documents': [], 'conflicts': [],
                 'duplicates': {'companies': 0, 'documents': 0, 'passages': 0}}
        universe = theme._validate_companies([case.get('company')], cutoff, audit)
        theme._validate_documents(documents, universe, cutoff, audit)
        if (audit['rejected_companies'] or audit['conflicts'] or any(
                row['reason'] != 'available_after_as_of' for row in audit['rejected_documents'])):
            raise ValueError(f'{name}: invalid source fixture or company metadata')
        for document in documents:
            if not isinstance(document.get('text'), str) or document.get('excerpt_sha256') != _sha(document['text'].encode('utf-8')):
                raise ValueError(f'{name}: excerpt SHA256 mismatch')
            if case['kind'] == 'public_excerpt':
                if not isinstance(case.get('public_reference'), dict) or not document.get('source_sha256'):
                    raise ValueError(f'{name}: public source reference and original source hash required')
            elif document.get('fixture_kind') != 'synthetic_adversarial':
                raise ValueError(f'{name}: synthetic document must be explicitly marked')
    return dataset


def classification_metrics(pairs, labels):
    """Macro F1 uses observed truth/prediction labels, including missed retrieval."""
    pairs = list(pairs)
    if any(actual not in labels or predicted not in labels for actual, predicted in pairs):
        raise ValueError('Metric labels are outside the declared taxonomy')
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for actual, predicted in pairs:
        matrix[actual][predicted] += 1
    per_class = {}
    active = []
    f1_values = []
    for label in labels:
        support = sum(matrix[label].values())
        predicted_count = sum(matrix[actual][label] for actual in labels)
        tp = matrix[label][label]
        fp, fn = predicted_count - tp, support - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        per_class[label] = {'support': support, 'predicted_count': predicted_count,
            'precision': round(tp / predicted_count, 6) if predicted_count else None,
            'recall': round(tp / support, 6) if support else None,
            'f1': round(f1, 6) if f1 is not None else None}
        if support or predicted_count:
            active.append(label)
            f1_values.append(f1)
    return {'cases': len(pairs), 'correct': sum(a == p for a, p in pairs),
        'accuracy': round(sum(a == p for a, p in pairs) / len(pairs), 6) if pairs else None,
        'macro_f1': round(sum(f1_values) / len(f1_values), 6) if f1_values else None,
        'macro_classes': active, 'classes_without_reference_examples': [x for x in labels if not per_class[x]['support']],
        'macro_definition': 'Unweighted mean over labels present in truth or predictions; no-retrieval/no-match errors remain in the confusion matrix.',
        'confusion_orientation': 'rows=expected, columns=predicted', 'confusion_matrix': matrix,
        'per_class': per_class}


def predict_case(case):
    """Exercise public company discovery and its unchanged assertion rules.

    Pure denials are deliberately omitted by discovery. Inspecting the existing
    validated passage pipeline allows evaluating their actual negated label,
    rather than incorrectly reclassifying them as a retrieval failure.
    """
    company, documents = case['company'], case['documents']
    report = theme.discover_companies([company], documents, case['query'], case['as_of'], limit=1)
    matched = report['companies']
    business = matched[0]['business_status'] if matched else 'no_match'
    settings = theme._configuration(None)
    terms, _ = theme._query_terms(case['query'], settings['theme_dictionary'])
    audit = {'rejected_companies': [], 'rejected_documents': [], 'conflicts': [],
             'duplicates': {'companies': 0, 'documents': 0, 'passages': 0}, 'qualifier_filtered_passages': 0}
    cutoff = theme._date(case['as_of'], 'as_of')
    universe = theme._validate_companies([company], cutoff, audit)
    visible = theme._validate_documents(documents, universe, cutoff, audit)
    passages = theme._passages(visible, [(term, theme._pattern(theme._norm(term))) for term in terms],
        settings['excerpt_chars'], audit, universe, case['query'], theme._required_qualifiers(case['query']))
    assertions = []
    for indexed in passages:
        for version in indexed['versions']:
            if not version['claims']:
                continue
            label = max((label for _, label in version['claims']), key=lambda value: theme.RELATION_WEIGHTS[value])
            document = version['document']
            assertions.append({'document_id': document['document_id'], 'claim_status': label,
                'company_level_denial': version['company_level_denial'], 'available_at': document['available_at'],
                'date_basis': document['date_basis'], 'source_url': document['source_url'], 'page': document['page'],
                'char_start': version['start'], 'char_end': version['end'],
                'quote_sha256': _sha(version['exact_excerpt'].encode('utf-8'))})
    claim = max((x['claim_status'] for x in assertions), key=lambda x: theme.RELATION_WEIGHTS[x], default='not_retrieved')
    return {'claim_status': claim, 'business_status': business, 'assertions': assertions,
            'discovery_status': report['status'], 'negated_only': report['audit']['negated_only_tickers'],
            'rejected_documents': report['audit']['rejected_documents']}


def evaluate(dataset_path=DEFAULT_DATASET):
    path = filesystem_path(dataset_path)
    raw = path.read_bytes()
    dataset = validate_dataset(_strict_json(raw))
    code = {name: _sha(filesystem_path(ROOT / name).read_bytes()) for name in
            ('theme_search.py', 'validation/business_claim_eval.py')}
    results = []
    for case in dataset['cases']:
        prediction = predict_case(case)
        results.append({'id': case['id'], 'kind': case['kind'], 'tags': case['tags'],
            'expected': case['expected'], 'predicted': prediction,
            'correct': {task: prediction[task] == expected for task, expected in case['expected'].items()}})
    groups = {}
    for kind in KINDS:
        subset = [row for row in results if row['kind'] == kind]
        groups[kind] = {'cases': len(subset), 'tags': dict(sorted(Counter(tag for row in subset for tag in row['tags']).items())),
            'claim_status': classification_metrics(((row['expected']['claim_status'], row['predicted']['claim_status'])
                for row in subset if 'claim_status' in row['expected']), CLAIM_LABELS),
            'business_status': classification_metrics(((row['expected']['business_status'], row['predicted']['business_status'])
                for row in subset if 'business_status' in row['expected']), BUSINESS_LABELS)}
    if raw != path.read_bytes() or any(_sha(filesystem_path(ROOT / name).read_bytes()) != digest for name, digest in code.items()):
        raise ValueError('Dataset or evaluator code changed during evaluation')
    return {'schema_version': 1, 'status': 'evaluated', 'dataset_id': dataset['dataset_id'],
        'dataset_sha256': _sha(raw), 'code_sha256': code, 'classifier': 'existing_theme_search_rules_no_training',
        'independently_annotated': False, 'held_out': False, 'production_approved': False,
        'metrics_by_data_kind': groups, 'cases': results,
        'misclassified_case_ids': [row['id'] for row in results if not all(row['correct'].values())],
        'limitations': ['Public labels concern the selected excerpt and supplied context, not complete current company operations.',
            'Labels were curated during development and have not received independent double annotation.',
            'Synthetic adversarial cases are invented text and never count as real-market accuracy.',
            'The two data kinds deliberately have no combined accuracy or F1.',
            'Unevaluated classes and zero-support classes are reported explicitly; this is not a held-out performance estimate.',
            'Temporal company cases and single-source assertion cases are separate tasks; one company can have conflicting assertions.',
            'No model is trained, no dataset is sampled randomly, and no acceptance threshold is implied by status=evaluated.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.dataset)
        content = json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2) + '\n'
        if args.output:
            output = filesystem_path(args.output)
            protected = [filesystem_path(path) for path in (args.dataset, __file__, theme.__file__)]
            if any(output == path or (output.exists() and path.exists() and output.samefile(path)) for path in protected):
                raise ValueError('Evaluation output must not overwrite dataset or evaluator code')
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding='utf-8')
        print(content, end='')
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc)}, ensure_ascii=True))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
