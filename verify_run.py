"""Verify a completed export before an internal downstream process consumes it.

This verifies byte integrity and internal consistency, not source authenticity
or investment suitability. A completion record is not a digital signature.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
from contextlib import closing

from run import CSV_FIELDS, OUTPUT_FILES, csv_row, dump_json

IDENTITY_KEYS = ('as_of', 'config', 'inputs', 'code', 'environment', 'synthetic_demo',
                 'ml_cleaning_requested', 'query', 'configuration_search_request')


def _check_sqlite_table(con, run_id, table, columns, key_columns, expected_rows, errors):
    """Reconcile both queryable columns and complete payloads against the JSON."""
    # Table/column identifiers come only from the fixed internal declarations below.
    records = con.execute(f'SELECT {",".join(columns)},payload_json FROM {table} WHERE run_id=?',
                          (run_id,)).fetchall()
    expected = {tuple(row[k] for k in key_columns): row for row in expected_rows}
    keys = [tuple(record[columns.index(k)] for k in key_columns) for record in records]
    if (len(expected) != len(expected_rows) or len(keys) != len(set(keys))
            or set(keys) != set(expected)):
        errors.append(f'SQLite {table} record set differs from JSON')
    for record, key in zip(records, keys):
        row = expected.get(key)
        if row is None:
            continue
        if tuple(record[:-1]) != tuple(row[column] for column in columns):
            errors.append(f'SQLite {table} columns differ from JSON: {key}')
        if json.loads(record[-1]) != row:
            errors.append(f'SQLite {table} payload differs from JSON: {key}')


def verify_output(directory, *, check_inputs=False):
    directory = Path(directory)
    errors = []
    try:
        marker = json.loads((directory / 'completion.json').read_text(encoding='utf-8'))
        if marker.get('status') != 'complete':
            raise ValueError('export does not have a complete marker')
        if set(marker.get('sha256', {})) != set(OUTPUT_FILES):
            raise ValueError('completion marker does not cover the expected output files')
        for name in OUTPUT_FILES:
            digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if digest != marker['sha256'][name]:
                errors.append(f'output hash mismatch: {name}')
        if errors:
            return {'valid': False, 'errors': errors}
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        results = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
        identity = {key: manifest[key] for key in IDENTITY_KEYS}
        expected = hashlib.sha256(dump_json(identity).encode('utf-8')).hexdigest()[:20]
        if not expected == marker['run_id'] == manifest['run_id'] == results['run_id']:
            errors.append('run identity mismatch')
        if manifest['as_of'] != results['as_of'] or manifest['config'] != results['config']:
            errors.append('result cutoff/config differs from manifest')
        companies = {c['ticker']: c for c in results['companies']}
        if len(companies) != len(results['companies']):
            errors.append('duplicate company ticker in results')
        with (directory / 'screen.csv').open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames != list(CSV_FIELDS):
                errors.append('CSV columns differ from expected export schema')
            exported = list(reader)
        expected_rows = [csv_row(company) for company in results['companies']]
        expected_csv = {row['ticker']: row for row in expected_rows}
        exported_tickers = [row['ticker'] for row in exported]
        if len(exported_tickers) != len(set(exported_tickers)):
            errors.append('duplicate company ticker in CSV')
        if len(expected_csv) != len(companies):
            errors.append('CSV escaping creates ambiguous company tickers')
        if set(exported_tickers) != set(expected_csv):
            errors.append('CSV company set differs from JSON')
        for row in exported:
            ticker = row['ticker']
            if row != expected_csv.get(ticker):
                errors.append(f'CSV company fields differ from JSON: {ticker}')
        uri = (directory / 'research.sqlite').resolve().as_uri() + '?mode=ro'
        with closing(sqlite3.connect(uri, uri=True)) as con:
            if con.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                errors.append('SQLite integrity check failed')
            for table, columns, key_columns, expected_rows in (
                ('companies', ('ticker', 'market', 'sector', 'status', 'score'),
                 ('ticker',), results['companies']),
                ('statements', ('ticker', 'period_end', 'available_at', 'filing_id'),
                 ('ticker', 'period_end'), results['normalized_statements']),
                ('evidence', ('document_id', 'ticker', 'source_url', 'available_at'),
                 ('document_id',), results['documents']['evidence']),
            ):
                _check_sqlite_table(con, expected, table, columns, key_columns, expected_rows, errors)
            record = con.execute('SELECT as_of,manifest_json FROM runs WHERE run_id=?', (expected,)).fetchone()
            if record is None or record[0] != results['as_of'] or json.loads(record[1]) != manifest:
                errors.append('SQLite manifest mismatch')
        search = json.loads((directory / 'search.json').read_text(encoding='utf-8'))
        if search != results.get('configuration_search', {'status': 'not_requested', 'run_id': expected}):
            errors.append('search export mismatch')
        if check_inputs:
            for name, digest in manifest['inputs'].items():
                actual = hashlib.sha256(Path(manifest['input_paths'][name]).read_bytes()).hexdigest()
                if actual != digest:
                    errors.append(f'original input changed: {name}')
        return {'valid': not errors, 'run_id': expected, 'company_count': len(companies),
                'checked_input_files': check_inputs, 'errors': errors}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, csv.Error, sqlite3.Error) as exc:
        return {'valid': False, 'errors': [str(exc)]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args(argv)
    result = verify_output(args.directory, check_inputs=args.check_inputs)
    print(dump_json(result))
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
