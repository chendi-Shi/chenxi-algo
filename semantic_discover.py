"""Explicit experimental neural retrieval CLI; existing daily entry stays lexical.

Run model download separately. Inference never fetches weights automatically.
Output remains a review queue until analyst-labelled relevance gates exist.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time

from discover import strict_json, _load_csv_snapshot, _validate_ingestion_snapshot, _csv_safe
from ingest_theme import filesystem_path, output_lock, _atomic_write
from readiness import apply_theme_business_policy, assess_readiness
from semantic_retrieval import METHODS, RetrievalIndex
from theme_financials import STYLE_NAMES, attach_fundamentals


ROOT = Path(__file__).resolve().parent
CODE_FILES = ('semantic_discover.py', 'semantic_retrieval.py', 'neural_models.py',
              'theme_search.py', 'theme_financials.py', 'engine.py', 'readiness.py',
              'discover.py', 'ingest_theme.py')
CSV_COLUMNS = ('ticker', 'business_status', 'semantic_only', 'relevance_score', 'styles')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def _csv_rows(companies):
    rows = []
    for company in companies:
        row = {key: company[key] for key in CSV_COLUMNS if key != 'styles'}
        row['styles'] = '|'.join(company['financial']['styles'])
        rows.append({key: '' if value is None else str(_csv_safe(value)) for key, value in row.items()})
    return rows


def verify_output(output):
    directory = filesystem_path(output)
    marker = strict_json((directory / 'complete.json').read_text(encoding='utf-8'))
    if (not isinstance(marker, dict) or type(marker.get('schema_version')) is not int
            or marker['schema_version'] != 1 or marker.get('status', 'complete') != 'complete'
            or not isinstance(marker.get('files'), dict)
            or set(marker['files']) != {'results.json', 'matches.csv', 'manifest.json'}):
        raise ValueError('invalid semantic export completion marker')
    raw = {name: (directory / name).read_bytes() for name in marker['files']}
    for name, expected in marker['files'].items():
        if _sha(raw[name]) != expected:
            raise ValueError(f'semantic export hash mismatch: {name}')
    report = strict_json(raw['results.json'].decode('utf-8'))
    manifest = strict_json(raw['manifest.json'].decode('utf-8'))
    if not isinstance(report, dict) or not isinstance(manifest, dict):
        raise ValueError('semantic report and manifest must be objects')
    if (not isinstance(marker.get('run_id'), str) or not marker['run_id']
            or not report.get('run_id') == manifest.get('run_id') == marker['run_id']):
        raise ValueError('mixed semantic export run ids')
    parameters = manifest.get('parameters')
    if not isinstance(parameters, dict) or type(parameters.get('limit')) is not int or parameters['limit'] < 1:
        raise ValueError('semantic export has an invalid display limit')
    all_companies, companies = report.get('all_companies'), report.get('companies')
    if not isinstance(all_companies, list) or not isinstance(companies, list):
        raise ValueError('semantic export is missing complete company evidence')
    tickers = []
    for company in all_companies:
        if (not isinstance(company, dict) or not isinstance(company.get('ticker'), str)
                or not company['ticker'] or not isinstance(company.get('financial'), dict)
                or not isinstance(company['financial'].get('styles'), list)
                or not company['financial']['styles']
                or any(style not in STYLE_NAMES for style in company['financial']['styles'])
                or len(set(company['financial']['styles'])) != len(company['financial']['styles'])
                or company.get('business_status') not in ('current_business', 'planned_business', 'historical_or_disputed', 'uncertain')
                or type(company.get('semantic_only')) is not bool
                or type(company.get('relevance_score')) not in (int, float)):
            raise ValueError('semantic company assessment is invalid')
        tickers.append(company['ticker'])
    if len(set(tickers)) != len(tickers) or companies != all_companies[:parameters['limit']]:
        raise ValueError('displayed companies do not reconcile with complete company evidence')
    for key, rows in (('style_lists', companies), ('all_matched_style_lists', all_companies)):
        expected = {style: [row['ticker'] for row in rows if style in row['financial']['styles']] for style in STYLE_NAMES}
        if report.get(key) != expected:
            raise ValueError(f'{key} does not reconcile with complete company evidence')
    csv_rows = _load_csv_snapshot(raw['matches.csv'])
    header = next(csv.reader(io.StringIO(raw['matches.csv'].decode('utf-8-sig')), strict=True), None)
    if header != list(CSV_COLUMNS) or csv_rows != _csv_rows(companies):
        raise ValueError('semantic CSV assessment mismatch')
    return {'status': 'verified', 'run_id': marker['run_id'], 'companies': len(csv_rows),
            'all_companies': len(all_companies),
            'production_approved': False}


def run(*, companies_path, documents_path, query, as_of, output,
        method='hybrid_rerank', model_dir=None, cache_dir=None, statements_path=None,
        valuations_path=None, policy_path=None, candidate_k=100, rerank_k=50,
        limit=20, threads=2, chunk_chars=480, overlap_chars=80):
    if method not in METHODS:
        raise ValueError('unsupported retrieval method')
    if valuations_path is not None and statements_path is None:
        raise ValueError('valuations require statements')
    paths = {'companies': filesystem_path(companies_path), 'documents': filesystem_path(documents_path)}
    for key, path in [('statements', statements_path), ('valuations', valuations_path), ('policy', policy_path)]:
        if path is not None:
            paths[key] = filesystem_path(path)
    sidecars = {key: paths['documents'].with_name(paths['documents'].name + suffix)
                for key, suffix in [('ingestion_audit', '.manifest.json'), ('ingestion_completion', '.complete.json')]}
    for key, sidecar in sidecars.items():
        if sidecar.exists():
            paths[key] = sidecar
    directory = filesystem_path(output)
    protected = [*paths.values(), *sidecars.values(), *(filesystem_path(ROOT / name) for name in CODE_FILES)]
    lock = directory.with_name(directory.name + '.lock')
    for path in protected:
        if directory == path or directory in path.parents or lock == path or (lock.exists() and path.exists() and lock.samefile(path)):
            raise ValueError('output or its lock would overlap an input/program file')
    for storage in [model_dir, cache_dir]:
        if storage is not None:
            store = filesystem_path(storage)
            if store == directory or store in directory.parents or directory in store.parents:
                raise ValueError('output, model, and vector storage must use separate directories')
    with output_lock(directory):
        raw = {k: path.read_bytes() for k, path in paths.items()}
        _validate_ingestion_snapshot(raw, paths)
        companies = strict_json(raw['companies'].decode('utf-8-sig'))
        documents = [strict_json(line) for line in raw['documents'].decode('utf-8-sig').splitlines() if line.strip()]
        code = {name: _sha((ROOT / name).read_bytes()) for name in CODE_FILES}
        began = time.perf_counter()
        model = None
        if method != 'fixed_bm25':
            if model_dir is None:
                raise ValueError('neural methods require --models pointing to previously downloaded weights')
            from neural_models import ModelBundle
            model = ModelBundle(model_dir, threads=threads)
        index = RetrievalIndex(companies, documents, as_of, model=model, cache_dir=cache_dir,
                               chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        report = index.company_report(query, method=method, limit=max(1, len(companies)),
                                      candidate_k=candidate_k, rerank_k=rerank_k)
        report['companies'] = report.pop('all_companies')
        statements = _load_csv_snapshot(raw['statements']) if 'statements' in raw else []
        valuations = _load_csv_snapshot(raw['valuations']) if 'valuations' in raw else []
        report = attach_fundamentals(report, statements, valuations)
        policy = strict_json(raw['policy'].decode('utf-8-sig')) if 'policy' in raw else None
        apply_theme_business_policy(report, policy)
        report['production_readiness'] = assess_readiness(report, companies, documents, policy)
        report['all_companies'], report['all_matched_style_lists'] = report['companies'], report['style_lists']
        if type(limit) is not int or limit < 1:
            raise ValueError('limit must be a positive integer')
        report['companies'] = report['companies'][:limit]
        shown = {c['ticker'] for c in report['companies']}
        report['style_lists'] = {key: [t for t in values if t in shown] for key, values in report['style_lists'].items()}
        report['runtime'] = {'seconds': time.perf_counter() - began, 'index_build_seconds': index.build_seconds,
                             'model_statistics': model.statistics() if model is not None else None}
        report['release_status'] = 'experimental_review_queue'
        report['production_approved'] = False
        def check_unchanged():
            if any(path.exists() != (key in raw) for key, path in sidecars.items()):
                raise ValueError('ingestion sidecar set changed during neural run')
            if any(paths[k].read_bytes() != content for k, content in raw.items()):
                raise ValueError('input files changed during neural run')
            if any(_sha((ROOT / name).read_bytes()) != value for name, value in code.items()):
                raise ValueError('code changed during neural run')
        check_unchanged()
        parameters = dict(query=query, as_of=as_of, method=method, candidate_k=candidate_k,
                          rerank_k=rerank_k, limit=limit, threads=threads,
                          chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        manifest = {'schema_version': 1, 'inputs': {k: {'name': p.name, 'sha256': _sha(raw[k])} for k,p in paths.items()},
                    'code_sha256': code, 'parameters': parameters, 'model': report['model'],
                    'corpus_sha256': report['corpus_sha256'], 'production_approved': False}
        run_id = _sha(_json(manifest))[:24]
        report['run_id'] = manifest['run_id'] = run_id
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS);writer.writeheader()
        writer.writerows(_csv_rows(report['companies']))
        outputs = {'results.json': _json(report), 'manifest.json': _json(manifest),
                   'matches.csv': buffer.getvalue().encode('utf-8')}
        directory.mkdir(parents=True, exist_ok=True)
        for name in (*outputs, 'complete.json'):
            if (directory / name).is_symlink():
                raise ValueError('output files must not be symlinks')
        _atomic_write(directory / 'complete.json', _json({'schema_version': 1, 'status': 'writing'}))
        for name, content in outputs.items():
            _atomic_write(directory / name, content)
        check_unchanged()
        _atomic_write(directory / 'complete.json', _json({'schema_version': 1, 'status': 'complete', 'run_id': run_id,
                                                        'files': {k: _sha(v) for k,v in outputs.items()}}))
        verify_output(directory)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--companies', type=Path)
    parser.add_argument('--documents', type=Path)
    parser.add_argument('--query')
    parser.add_argument('--as-of')
    parser.add_argument('--method', choices=METHODS, default='hybrid_rerank')
    parser.add_argument('--models', type=Path)
    parser.add_argument('--cache', type=Path, default=Path('output/semantic_vectors'))
    parser.add_argument('--output', type=Path, default=Path('output/semantic_query'))
    parser.add_argument('--statements', type=Path)
    parser.add_argument('--valuations', type=Path)
    parser.add_argument('--production-policy', type=Path)
    parser.add_argument('--candidate-k', type=int, default=100)
    parser.add_argument('--rerank-k', type=int, default=50)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.verify:
            print(json.dumps(verify_output(args.verify), ensure_ascii=True));return 0
        if not all([args.companies, args.documents, args.query, args.as_of]):
            parser.error('--companies, --documents, --query, --as-of are required')
        report = run(companies_path=args.companies, documents_path=args.documents,
                     query=args.query, as_of=args.as_of, output=args.output, method=args.method,
                     model_dir=args.models, cache_dir=args.cache, statements_path=args.statements,
                     valuations_path=args.valuations, policy_path=args.production_policy,
                     candidate_k=args.candidate_k, rerank_k=args.rerank_k, limit=args.limit, threads=args.threads)
        print(json.dumps({'status': report['status'], 'release_status': report['release_status'],
                          'companies': len(report['companies']), 'production_approved': False,
                          'data_gates': report['production_readiness']['status']}, ensure_ascii=True))
        return 3 if report['production_readiness']['status'] == 'blocked' else 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc)}, ensure_ascii=True));return 2


if __name__ == '__main__':
    raise SystemExit(main())
