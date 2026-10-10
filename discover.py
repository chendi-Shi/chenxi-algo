"""Theme -> evidence-backed companies -> auditable fundamental research lists."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
from pathlib import Path
import tempfile
from functools import wraps

from engine import iso
from theme_financials import STYLE_NAMES, attach_fundamentals
from theme_search import discover_companies
from ingest_theme import filesystem_path, output_lock
from readiness import apply_theme_business_policy, assess_readiness

ROOT = filesystem_path(__file__).parent
OUTPUT_FILES = ('results.json', 'matches.csv', 'manifest.json')
CODE_FILES = ('discover.py', 'theme_search.py', 'theme_financials.py', 'engine.py', 'ingest_theme.py', 'readiness.py')


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + '\n'


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    def bad(value):
        raise ValueError(f'nonfinite JSON number: {value}')
    def numeric(value):
        result = float(value)
        return result if math.isfinite(result) else bad(value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad, parse_float=numeric)


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def _load_csv_snapshot(content):
    """Parse exactly the bytes fingerprinted for this run, never re-open a file."""
    reader = csv.DictReader(io.StringIO(content.decode('utf-8-sig'), newline=''), strict=True)
    rows = []
    try:
        fields = reader.fieldnames
        if (not fields or any(not field or field != field.strip() for field in fields)
                or len(set(fields)) != len(fields)):
            raise ValueError('CSV column names must be nonempty, unique and have no surrounding whitespace')
        for row in reader:
            if None in row:
                raise ValueError(f'CSV row {reader.line_num} has more values than column names')
            rows.append(row)
    except csv.Error as exc:
        raise ValueError(f'invalid CSV: {exc}') from exc
    return rows


def _validate_ingestion_snapshot(raw, paths):
    """Validate precisely the captured corpus/audit/marker bytes used by this run."""
    if 'ingestion_audit' not in raw:
        if 'ingestion_completion' in raw:
            raise ValueError('ingestion completion marker has no audit report')
        return
    audit = strict_json(raw['ingestion_audit'].decode('utf-8-sig'))
    if (not isinstance(audit, dict) or audit.get('status') != 'completed'
            or audit.get('output_sha256') != _hash(raw['documents'])):
        raise ValueError('document ingestion is incomplete or its output hash does not match')
    if audit.get('schema_version') not in (None, 1, 2):
        raise ValueError('unsupported ingestion audit schema')
    if 'ingestion_completion' not in raw:
        if audit.get('schema_version') == 2:
            raise ValueError('ingestion completion marker is missing')
        return  # Compatibility with the explicitly untransactional v1 corpus format.
    marker = strict_json(raw['ingestion_completion'].decode('utf-8-sig'))
    if (not isinstance(marker, dict) or marker.get('schema_version') != 1
            or marker.get('state') != 'complete'):
        raise ValueError('ingestion snapshot transaction is not complete')
    expected = {paths['documents'].name: _hash(raw['documents']),
                paths['ingestion_audit'].name: _hash(raw['ingestion_audit'])}
    if marker.get('files') != expected:
        raise ValueError('ingestion snapshot file hashes do not match completion marker')
    count = marker.get('documents_written')
    actual_count = sum(bool(line.strip()) for line in raw['documents'].decode('utf-8-sig').splitlines())
    if (not isinstance(marker.get('run_id'), str) or not marker['run_id']
            or marker['run_id'] != audit.get('run_id')
            or type(count) is not int or count < 0 or count != actual_count
            or type(audit.get('documents_written')) is not int or audit['documents_written'] != count):
        raise ValueError('ingestion snapshot run identity or document count does not reconcile')


def _csv_safe(value):
    return "'" + value if isinstance(value, str) and value[:1] in ('=', '+', '-', '@', '\t', '\r') else value


def _same(a, b):
    a, b = filesystem_path(a), filesystem_path(b)
    return a == b or (a.exists() and b.exists() and a.samefile(b))


def _atomic(path, content):
    path = filesystem_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_discovery(output, *, check_inputs=False):
    output = filesystem_path(output)
    completion = strict_json((output / 'completion.json').read_text(encoding='utf-8'))
    if (not isinstance(completion, dict) or completion.get('status') != 'complete'
            or not isinstance(completion.get('files'), dict) or set(completion['files']) != set(OUTPUT_FILES)):
        raise ValueError('discovery export is incomplete')
    artifacts = {name: (output / name).read_bytes() for name in OUTPUT_FILES}
    for name in OUTPUT_FILES:
        if _hash(artifacts[name]) != completion['files'][name]:
            raise ValueError(f'export hash mismatch: {name}')
    report = strict_json(artifacts['results.json'].decode('utf-8'))
    manifest = strict_json(artifacts['manifest.json'].decode('utf-8'))
    if not isinstance(report, dict) or not isinstance(manifest, dict):
        raise ValueError('discovery report and manifest must be objects')
    if not report.get('run_id') == manifest.get('run_id') == completion.get('run_id'):
        raise ValueError('run identity mismatch')
    rows = _load_csv_snapshot(artifacts['matches.csv'])
    if report.get('schema_version') == 2:
        all_companies = report.get('all_companies')
        if not isinstance(all_companies, list):
            raise ValueError('complete evidence records are missing')
        all_tickers = [company['ticker'] for company in all_companies]
        if len(set(all_tickers)) != len(all_tickers):
            raise ValueError('complete evidence records contain duplicate tickers')
        if report['companies'] != all_companies[:manifest['arguments']['limit']]:
            raise ValueError('displayed companies do not reconcile with complete evidence records')
        for key, companies in (('style_lists', report['companies']), ('all_matched_style_lists', all_companies)):
            expected_lists = {style: [company['ticker'] for company in companies
                                     if style in company['financial']['styles']] for style in STYLE_NAMES}
            if report.get(key) != expected_lists:
                raise ValueError(f'{key} does not reconcile with complete evidence records')
    expected_rows = []
    for company in report['companies']:
        expected = {k: company.get(k) for k in ('ticker', 'name', 'market', 'sector', 'relevance_score',
                                               'evidence_count', 'requires_review')}
        if report.get('schema_version') == 2:
            expected['business_status'] = company['business_status']
            expected['business_freshness_status'] = company['theme_business_freshness']['status']
        expected.update(relations=' | '.join(company['relations']), financial_status=company['financial']['status'],
                        styles=' | '.join(company['financial']['styles']), loss_making=company['financial']['loss_making'])
        expected_rows.append({key: '' if value is None else str(_csv_safe(value)) for key, value in expected.items()})
    if rows != expected_rows:
        raise ValueError('CSV/company reconciliation failed')
    if check_inputs:
        for item in manifest['inputs'].values():
            if _hash(filesystem_path(item['path']).read_bytes()) != item['sha256']:
                raise ValueError(f'input hash mismatch: {item["path"]}')
    return {'status': 'verified', 'run_id': completion['run_id'], 'companies': len(rows),
            'all_matched_companies': len(report.get('all_companies', report['companies'])),
            'data_gate_status': report.get('production_readiness', {}).get('status', 'not_assessed'),
            'production_approved': False}


def _serialized_output(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        output = filesystem_path(kwargs['output'])
        lock_path = output.with_name(output.name + '.lock')
        protected = [filesystem_path(value) for key, value in kwargs.items()
                     if key.endswith('_path') and value is not None]
        if kwargs.get('documents_path') is not None:
            documents = filesystem_path(kwargs['documents_path'])
            protected.extend(documents.with_name(documents.name + suffix)
                             for suffix in ('.manifest.json', '.complete.json'))
        protected.extend(ROOT / name for name in CODE_FILES)
        if lock_path.is_symlink() or any(_same(lock_path, path) for path in protected):
            raise ValueError('output lock must not overwrite or open an input or program file')
        with output_lock(output):
            return function(*args, **kwargs)
    return wrapped


@_serialized_output
def run_discovery(*, query, as_of, companies_path, documents_path, output,
                  config_path=None, statements_path=None, valuations_path=None,
                  financial_config_path=None, production_policy_path=None, limit=20, market=None):
    iso(as_of)
    if valuations_path is not None and statements_path is None:
        raise ValueError('--valuations requires --statements')
    if market not in (None, 'A', 'HK'):
        raise ValueError('market must be A or HK')
    paths = {'companies': filesystem_path(companies_path), 'documents': filesystem_path(documents_path)}
    ingestion_audit = paths['documents'].with_name(paths['documents'].name + '.manifest.json')
    completion = paths['documents'].with_name(paths['documents'].name + '.complete.json')
    for name, path in (('ingestion_audit', ingestion_audit), ('ingestion_completion', completion)):
        if path.exists():
            paths[name] = path
    for name, path in (('config', config_path), ('statements', statements_path),
                       ('valuations', valuations_path), ('financial_config', financial_config_path),
                       ('production_policy', production_policy_path)):
        if path is not None:
            paths[name] = filesystem_path(path)
    raw = {name: path.read_bytes() for name, path in paths.items()}
    _validate_ingestion_snapshot(raw, paths)
    inputs = {name: {'path': str(path), 'sha256': _hash(raw[name])} for name, path in paths.items()}
    code = {name: _hash((ROOT / name).read_bytes()) for name in CODE_FILES}
    output = filesystem_path(output)
    for name in (*OUTPUT_FILES, 'completion.json'):
        target = output / name
        if any(_same(target, path) for path in [*paths.values(), *(ROOT / x for x in CODE_FILES)]):
            raise ValueError('output must not overwrite an input or program file')
        # Avoid following a pre-existing symlink into an unrelated file.
        if target.is_symlink():
            raise ValueError('output files must not be symlinks')
    companies = strict_json(raw['companies'].decode('utf-8-sig'))
    documents = []
    for number, line in enumerate(raw['documents'].decode('utf-8-sig').splitlines(), 1):
        if line.strip():
            try:
                documents.append(strict_json(line))
            except ValueError as exc:
                raise ValueError(f'documents line {number}: {exc}') from exc
    if not isinstance(companies, list):
        raise ValueError('company directory must be a JSON list')
    cfg = strict_json(raw['config'].decode('utf-8-sig')) if 'config' in raw else {}
    if not isinstance(cfg, dict) or set(cfg) - {'retrieval', 'styles'}:
        raise ValueError('theme config permits only retrieval and styles objects')
    financial_cfg = strict_json(raw['financial_config'].decode('utf-8-sig')) if 'financial_config' in raw else None
    # Screen before applying a display market filter so validation still audits
    # all directory rows. Financial peers likewise use all supplied statements.
    found = discover_companies(companies, documents, query, as_of,
                               config=cfg.get('retrieval'), limit=max(len(companies), 1))
    if type(limit) is not int or limit < 1:
        raise ValueError('limit must be a positive integer')
    if market:
        found['companies'] = [c for c in found['companies'] if c['market'] == market]
    found['coverage']['matches_after_market_filter'] = len(found['companies'])
    if not found['companies'] and found['status'] == 'completed':
        found['status'] = 'no_matches'
    statements = _load_csv_snapshot(raw['statements']) if 'statements' in raw else []
    valuations = _load_csv_snapshot(raw['valuations']) if 'valuations' in raw else []
    report = attach_fundamentals(found, statements, valuations, config=cfg.get('styles'), screen_config=financial_cfg)
    policy = strict_json(raw['production_policy'].decode('utf-8-sig')) if 'production_policy' in raw else None
    apply_theme_business_policy(report, policy)
    report['production_readiness'] = assess_readiness(report, companies, documents, policy)
    # Calculate financial lists across all matches. The display limit must not
    # silently remove a financially qualified company before classification.
    report['all_matched_style_lists'] = report['style_lists']
    report['all_companies'] = report['companies']
    report['companies'] = report['companies'][:limit]
    shown = {c['ticker'] for c in report['companies']}
    report['style_lists'] = {key: [ticker for ticker in tickers if ticker in shown]
                             for key, tickers in report['style_lists'].items()}
    report['coverage']['returned_companies'] = len(report['companies'])
    report['coverage']['display_truncated'] = report['coverage']['matches_after_market_filter'] > len(report['companies'])
    arguments = dict(query=query, as_of=as_of, limit=limit, market=market)
    environment = {'python': platform.python_version(), 'platform': platform.system()}
    run_id = _hash(dump({'inputs': {k: v['sha256'] for k, v in inputs.items()},
                         'code': code, 'arguments': arguments, 'environment': environment}).encode('utf-8'))[:24]
    report['run_id'] = run_id
    report['schema_version'] = 2
    report['coverage']['corpus_note'] = 'Supplied corpus only; this is not an exhaustive A/H market scan.'
    manifest = {'schema_version': 2, 'run_id': run_id, 'inputs': inputs,
                'code_sha256': code, 'arguments': arguments, 'environment': environment,
                'resolved_retrieval_config': report['config'], 'resolved_style_config': report['style_config']}
    buffer = io.StringIO(newline='')
    fields = ('ticker', 'name', 'market', 'sector', 'relevance_score', 'relations', 'business_status', 'business_freshness_status',
              'financial_status', 'styles', 'loss_making', 'evidence_count', 'requires_review')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for company in report['companies']:
        row = {k: company.get(k) for k in fields}
        row.update(relations=' | '.join(company['relations']), financial_status=company['financial']['status'],
                   styles=' | '.join(company['financial']['styles']), loss_making=company['financial']['loss_making'],
                   business_freshness_status=company['theme_business_freshness']['status'])
        # CSV is convenient to inspect in Excel: escape formula-like strings.
        writer.writerow({k: _csv_safe(value) for k, value in row.items()})
    artifacts = {'results.json': dump(report).encode('utf-8'),
                 'manifest.json': dump(manifest).encode('utf-8'),
                 'matches.csv': buffer.getvalue().encode('utf-8-sig')}
    def unchanged():
        if any(path.exists() != (name in paths) for name, path in
               (('ingestion_audit', ingestion_audit), ('ingestion_completion', completion))):
            raise ValueError('ingestion sidecar set changed while building discovery output; rerun required')
        if any(_hash(path.read_bytes()) != inputs[name]['sha256'] for name, path in paths.items()):
            raise ValueError('input changed while building discovery output; rerun required')
        if any(_hash((ROOT / name).read_bytes()) != sha for name, sha in code.items()):
            raise ValueError('program changed while building discovery output; rerun required')
    unchanged()
    _atomic(output / 'completion.json', dump({'status': 'writing', 'run_id': run_id}).encode('utf-8'))
    for name, content in artifacts.items():
        _atomic(output / name, content)
    unchanged()
    _atomic(output / 'completion.json', dump({'status': 'complete', 'run_id': run_id,
                                             'files': {k: _hash(v) for k, v in artifacts.items()}}).encode('utf-8'))
    verify_discovery(output, check_inputs=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', help='Product or theme, e.g. 光模块 / 机器人 / AI服务器')
    parser.add_argument('--as-of', help='Information availability cutoff YYYY-MM-DD')
    parser.add_argument('--companies', type=Path)
    parser.add_argument('--documents', type=Path)
    parser.add_argument('--demo-corpus', action='store_true', help='Explicitly use the nine-company development sample')
    parser.add_argument('--config', type=Path, default=ROOT / 'theme_config.json')
    parser.add_argument('--statements', type=Path)
    parser.add_argument('--valuations', type=Path)
    parser.add_argument('--financial-config', type=Path)
    parser.add_argument('--production-policy', type=Path, help='Explicit declared-universe coverage and freshness gates')
    parser.add_argument('--market', choices=('A', 'HK'))
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/discovery')
    parser.add_argument('--verify', type=Path, help='Verify an existing discovery export')
    args = parser.parse_args(argv)
    try:
        if args.verify:
            print(dump(verify_discovery(args.verify, check_inputs=True)), end='')
            return 0
        if not args.query or not args.as_of:
            parser.error('--query and --as-of are required unless using --verify')
        if args.demo_corpus:
            if args.companies or args.documents or args.production_policy:
                parser.error('--demo-corpus cannot be combined with custom data or production policy')
            args.companies, args.documents = ROOT / 'data/theme_companies.json', ROOT / 'data/theme_documents.jsonl'
        elif not args.companies or not args.documents:
            parser.error('Provide --companies and --documents; use --demo-corpus only for the development sample')
        result = run_discovery(query=args.query, as_of=args.as_of,
                               companies_path=args.companies, documents_path=args.documents,
                               config_path=args.config, statements_path=args.statements,
                               valuations_path=args.valuations, financial_config_path=args.financial_config,
                               production_policy_path=args.production_policy,
                               market=args.market, limit=args.limit, output=args.output)
        # ASCII-safe stdout also works in legacy Windows consoles; JSON files are UTF-8.
        print(json.dumps({'status': result['status'], 'companies': len(result['companies']),
                          'style_lists': result['style_lists'], 'output': str(filesystem_path(args.output)),
                          'coverage': result['coverage'],
                          'data_gate_status': result['production_readiness']['status'],
                          'production_approved': False}, ensure_ascii=True, indent=2))
        return 3 if result['production_readiness']['status'] == 'blocked' else 0
    except (ValueError, OSError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc)}, ensure_ascii=True))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
