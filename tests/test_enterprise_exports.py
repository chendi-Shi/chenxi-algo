"""Real filesystem/SQLite tests for complete, consistent research exports."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

import run
from verify_run import verify_output
from test_engine import AS_OF, annual_history, valuation


class EnterpriseExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.rows = [row for i in range(7) for row in annual_history(f'A{i}')]
        self.values = [valuation(f'A{i}', market_cap=140 + i * 35) for i in range(7)]
        for name, records in (('statements', self.rows), ('valuations', self.values)):
            with (self.folder / f'{name}.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        self.output = self.folder / 'export'
        self.args = ['--statements', str(self.folder / 'statements.csv'),
                     '--valuations', str(self.folder / 'valuations.csv'),
                     '--as-of', AS_OF.isoformat(), '--output', str(self.output)]

    def execute(self, extra=()):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run.main(self.args + list(extra))

    def refresh_hash(self, name):
        marker_path = self.output / 'completion.json'
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
        marker['sha256'][name] = hashlib.sha256((self.output / name).read_bytes()).hexdigest()
        marker_path.write_text(run.dump_json(marker), encoding='utf-8')

    def write_csv_rows(self, rows, fields=run.CSV_FIELDS):
        with (self.output / 'screen.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        self.refresh_hash('screen.csv')

    def read_csv_rows(self):
        with (self.output / 'screen.csv').open(encoding='utf-8-sig', newline='') as handle:
            return list(csv.DictReader(handle))

    def test_real_exports_and_original_inputs_reconcile(self):
        self.assertEqual(self.execute(['--ml-cleaning', '--search-method', 'grid',
            '--search-market', 'A', '--search-sector', 'Industrials',
            '--search-target-size', '2', '--search-max-evaluations', '12']), 0)
        verified = verify_output(self.output, check_inputs=True)
        self.assertTrue(verified['valid'], verified)
        self.assertEqual(verified['company_count'], 7)

    def test_byte_change_is_rejected(self):
        self.assertEqual(self.execute(), 0)
        with (self.output / 'report.html').open('a', encoding='utf-8') as handle:
            handle.write('corrupted')
        check = verify_output(self.output)
        self.assertFalse(check['valid'])
        self.assertIn('output hash mismatch: report.html', check['errors'])

    def test_export_failure_invalidates_previous_completion(self):
        self.assertEqual(self.execute(), 0)
        with mock.patch.object(run, 'save_database', side_effect=OSError('disk failure')):
            self.assertEqual(self.execute(), 2)
        self.assertFalse(verify_output(self.output)['valid'])
        marker = json.loads((self.output / 'completion.json').read_text(encoding='utf-8'))
        self.assertEqual(marker['status'], 'writing')

    def test_semantic_disagreement_fails_even_if_file_hashes_are_refreshed(self):
        self.assertEqual(self.execute(), 0)
        path = self.output / 'results.json'
        results = json.loads(path.read_text(encoding='utf-8'))
        results['companies'][0]['status'] = 'excluded'
        path.write_text(run.dump_json(results), encoding='utf-8')
        marker_path = self.output / 'completion.json'
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
        marker['sha256']['results.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
        marker_path.write_text(run.dump_json(marker), encoding='utf-8')
        self.assertFalse(verify_output(self.output)['valid'])

    def test_input_change_during_calculation_prevents_publication(self):
        original = run.load_csv
        def changed(path):
            result = original(path)
            if path.name == 'valuations.csv':
                with path.open('a', encoding='utf-8') as handle:
                    handle.write('\n')
            return result
        with mock.patch.object(run, 'load_csv', side_effect=changed):
            self.assertEqual(self.execute(), 2)
        self.assertFalse((self.output / 'completion.json').exists())

    def test_archived_export_stays_valid_but_changed_original_is_reported(self):
        self.assertEqual(self.execute(), 0)
        with (self.folder / 'statements.csv').open('a', encoding='utf-8') as handle:
            handle.write('\n')
        self.assertTrue(verify_output(self.output)['valid'])
        self.assertFalse(verify_output(self.output, check_inputs=True)['valid'])

    def test_duplicate_csv_company_cannot_replace_a_missing_company(self):
        self.assertEqual(self.execute(), 0)
        rows = self.read_csv_rows()
        rows[1] = dict(rows[0])
        self.write_csv_rows(rows)
        checked = verify_output(self.output)
        self.assertFalse(checked['valid'])
        self.assertIn('duplicate company ticker in CSV', checked['errors'])
        self.assertIn('CSV company set differs from JSON', checked['errors'])

    def test_every_csv_field_reconciles_even_with_refreshed_hash(self):
        self.assertEqual(self.execute(), 0)
        original = self.read_csv_rows()
        for field in run.CSV_FIELDS:
            with self.subTest(field=field):
                rows = [dict(row) for row in original]
                rows[0][field] = '9999' if field == 'roe' else '__incorrect__'
                self.write_csv_rows(rows)
                checked = verify_output(self.output)
                self.assertFalse(checked['valid'], field)
                self.assertTrue(any('CSV company' in e for e in checked['errors']))

    def test_sqlite_queryable_company_columns_reconcile_with_payload(self):
        self.assertEqual(self.execute(), 0)
        with contextlib.closing(sqlite3.connect(self.output / 'research.sqlite')) as con, con:
            con.execute("UPDATE companies SET status='candidate',score=99,market='HK',sector='Other' WHERE ticker='A0'")
        self.refresh_hash('research.sqlite')
        checked = verify_output(self.output)
        self.assertFalse(checked['valid'])
        self.assertTrue(any('SQLite companies columns differ' in e for e in checked['errors']))

    def test_sqlite_statement_fields_and_missing_records_are_rejected(self):
        self.assertEqual(self.execute(), 0)
        with contextlib.closing(sqlite3.connect(self.output / 'research.sqlite')) as con, con:
            con.execute("UPDATE statements SET available_at='2020-01-01' WHERE ticker='A0'")
            con.execute("DELETE FROM statements WHERE ticker='A1'")
        self.refresh_hash('research.sqlite')
        checked = verify_output(self.output)
        self.assertFalse(checked['valid'])
        self.assertIn('SQLite statements record set differs from JSON', checked['errors'])
        self.assertTrue(any('SQLite statements columns differ' in e for e in checked['errors']))

    def test_sqlite_statement_payload_is_reconciled(self):
        self.assertEqual(self.execute(), 0)
        with contextlib.closing(sqlite3.connect(self.output / 'research.sqlite')) as con, con:
            con.execute("UPDATE statements SET payload_json='{}' WHERE ticker='A0'")
        self.refresh_hash('research.sqlite')
        checked = verify_output(self.output)
        self.assertFalse(checked['valid'])
        self.assertTrue(any('SQLite statements payload differs' in e for e in checked['errors']))

    def test_sqlite_evidence_fields_and_payload_are_reconciled(self):
        document = {'document_id': 'doc-1', 'ticker': 'A0', 'available_at': AS_OF.isoformat(),
                    'source_url': 'https://example.invalid/report', 'source_type': 'annual_report',
                    'page': '1', 'text': 'Operating cash generation increased.'}
        path = self.folder / 'documents.jsonl'
        path.write_text(json.dumps(document) + '\n', encoding='utf-8')
        self.assertEqual(self.execute(['--documents', str(path)]), 0)
        self.assertTrue(verify_output(self.output)['valid'])
        with contextlib.closing(sqlite3.connect(self.output / 'research.sqlite')) as con, con:
            con.execute("UPDATE evidence SET source_url='https://example.invalid/wrong',payload_json='{}'")
        self.refresh_hash('research.sqlite')
        checked = verify_output(self.output)
        self.assertFalse(checked['valid'])
        self.assertTrue(any('SQLite evidence columns differ' in e for e in checked['errors']))
        self.assertTrue(any('SQLite evidence payload differs' in e for e in checked['errors']))

    def test_duplicate_csv_header_is_rejected(self):
        self.assertEqual(self.execute(), 0)
        self.write_csv_rows(self.read_csv_rows(), (*run.CSV_FIELDS, 'roe'))
        checked = verify_output(self.output)
        self.assertFalse(checked['valid'])
        self.assertIn('CSV columns differ from expected export schema', checked['errors'])

    def test_csv_escaping_round_trips_without_changing_real_numbers(self):
        for name in ('statements', 'valuations'):
            path = self.folder / f'{name}.csv'
            text = path.read_text(encoding='utf-8').replace('A0', '=FORMULA()')
            path.write_text(text, encoding='utf-8')
        self.assertEqual(self.execute(), 0)
        self.assertTrue(verify_output(self.output)['valid'])
        row = next(r for r in self.read_csv_rows() if r['ticker'] == "'=FORMULA()")
        self.assertGreater(float(row['roe']), 0)


if __name__ == '__main__':
    unittest.main()
