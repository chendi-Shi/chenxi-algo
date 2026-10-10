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

from engine import iso, load_csv
from theme_financials import attach_fundamentals
from theme_search import discover_companies

ROOT = Path(__file__).resolve().parent
OUTPUT_FILES = ('results.json', 'matches.csv', 'manifest.json')
CODE_FILES = ('discover.py', 'theme_search.py', 'theme_financials.py', 'engine.py')


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


def _csv_safe(value):
    return "'" + value if isinstance(value, str) and value[:1] in ('=', '+', '-', '@', '\t', '\r') else value


def _same(a, b):
    return a.resolve() == b.resolve() or (a.exists() and b.exists() and a.samefile(b))


def _atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_discovery(output, *, check_inputs=False):
    output = Path(output).resolve()
    completion = strict_json((output / 'completion.json').read_text(encoding='utf-8'))
    if completion.get('status') != 'complete' or set(completion.get('files', {})) != set(OUTPUT_FILES):
        raise ValueError('discovery export is incomplete')
    for name in OUTPUT_FILES:
        if _hash((output / name).read_bytes()) != completion['files'][name]:
            raise ValueError(f'export hash mismatch: {name}')
    report = strict_json((output / 'results.json').read_text(encoding='utf-8'))
    manifest = strict_json((output / 'manifest.json').read_text(encoding='utf-8'))
    if not report.get('run_id') == manifest.get('run_id') == completion.get('run_id'):
        raise ValueError('run identity mismatch')
    with (output / 'matches.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = []
    for company in report['companies']:
        expected = {k: company.get(k) for k in ('ticker', 'name', 'market', 'sector', 'relevance_score',
                                               'evidence_count', 'requires_review')}
        expected.update(relations=' | '.join(company['relations']), financial_status=company['financial']['status'],
                        styles=' | '.join(company['financial']['styles']), loss_making=company['financial']['loss_making'])
        expected_rows.append({key: '' if value is None else str(_csv_safe(value)) for key, value in expected.items()})
    if rows != expected_rows:
        raise ValueError('CSV/company reconciliation failed')
    if check_inputs:
        for item in manifest['inputs'].values():
            if _hash(Path(item['path']).read_bytes()) != item['sha256']:
                raise ValueError(f'input hash mismatch: {item["path"]}')
    return {'status': 'verified', 'run_id': completion['run_id'], 'companies': len(rows)}


def run_discovery(*, query, as_of, companies_path, documents_path, output,
                  config_path=None, statements_path=None, valuations_path=None,
                  financial_config_path=None, limit=20, market=None):
    iso(as_of)
    if valuations_path is not None and statements_path is None:
        raise ValueError('--valuations requires --statements')
    if market not in (None, 'A', 'HK'):
        raise ValueError('market must be A or HK')
    paths = {'companies': Path(companies_path).resolve(), 'documents': Path(documents_path).resolve()}
    ingestion_audit = paths['documents'].with_name(paths['documents'].name + '.manifest.json')
    if ingestion_audit.exists():
        paths['ingestion_audit'] = ingestion_audit
    for name, path in (('config', config_path), ('statements', statements_path),
                       ('valuations', valuations_path), ('financial_config', financial_config_path)):
        if path is not None:
            paths[name] = Path(path).resolve()
    raw = {name: path.read_bytes() for name, path in paths.items()}
    if 'ingestion_audit' in raw:
        ingestion = strict_json(raw['ingestion_audit'].decode('utf-8-sig'))
        if (not isinstance(ingestion, dict) or ingestion.get('status') != 'completed'
                or ingestion.get('output_sha256') != _hash(raw['documents'])):
            raise ValueError('document ingestion is incomplete or its output hash does not match')
    inputs = {name: {'path': str(path), 'sha256': _hash(raw[name])} for name, path in paths.items()}
    code = {name: _hash((ROOT / name).read_bytes()) for name in CODE_FILES}
    output = Path(output).resolve()
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
    found['companies'] = found['companies'][:limit]
    found['coverage']['returned_companies'] = len(found['companies'])
    if not found['companies'] and found['status'] == 'completed':
        found['status'] = 'no_matches'
    statements = load_csv(paths['statements']) if 'statements' in paths else []
    valuations = load_csv(paths['valuations']) if 'valuations' in paths else []
    report = attach_fundamentals(found, statements, valuations, config=cfg.get('styles'), screen_config=financial_cfg)
    arguments = dict(query=query, as_of=as_of, limit=limit, market=market)
    environment = {'python': platform.python_version(), 'platform': platform.system()}
    run_id = _hash(dump({'inputs': {k: v['sha256'] for k, v in inputs.items()},
                         'code': code, 'arguments': arguments, 'environment': environment}).encode('utf-8'))[:24]
    report['run_id'] = run_id
    report['schema_version'] = 1
    report['coverage']['corpus_note'] = 'Supplied corpus only; this is not an exhaustive A/H market scan.'
    manifest = {'schema_version': 1, 'run_id': run_id, 'inputs': inputs,
                'code_sha256': code, 'arguments': arguments, 'environment': environment,
                'resolved_retrieval_config': report['config'], 'resolved_style_config': report['style_config']}
    buffer = io.StringIO(newline='')
    fields = ('ticker', 'name', 'market', 'sector', 'relevance_score', 'relations',
              'financial_status', 'styles', 'loss_making', 'evidence_count', 'requires_review')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for company in report['companies']:
        row = {k: company.get(k) for k in fields}
        row.update(relations=' | '.join(company['relations']), financial_status=company['financial']['status'],
                   styles=' | '.join(company['financial']['styles']), loss_making=company['financial']['loss_making'])
        # CSV is convenient to inspect in Excel: escape formula-like strings.
        writer.writerow({k: _csv_safe(value) for k, value in row.items()})
    artifacts = {'results.json': dump(report).encode('utf-8'),
                 'manifest.json': dump(manifest).encode('utf-8'),
                 'matches.csv': buffer.getvalue().encode('utf-8-sig')}
    def unchanged():
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
    parser.add_argument('--companies', type=Path, default=ROOT / 'data/theme_companies.json')
    parser.add_argument('--documents', type=Path, default=ROOT / 'data/theme_documents.jsonl')
    parser.add_argument('--config', type=Path, default=ROOT / 'theme_config.json')
    parser.add_argument('--statements', type=Path)
    parser.add_argument('--valuations', type=Path)
    parser.add_argument('--financial-config', type=Path)
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
        result = run_discovery(query=args.query, as_of=args.as_of,
                               companies_path=args.companies, documents_path=args.documents,
                               config_path=args.config, statements_path=args.statements,
                               valuations_path=args.valuations, financial_config_path=args.financial_config,
                               market=args.market, limit=args.limit, output=args.output)
        # ASCII-safe stdout also works in legacy Windows consoles; JSON files are UTF-8.
        print(json.dumps({'status': result['status'], 'companies': len(result['companies']),
                          'style_lists': result['style_lists'], 'output': str(args.output.resolve()),
                          'coverage': result['coverage']}, ensure_ascii=True, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc)}, ensure_ascii=True))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
