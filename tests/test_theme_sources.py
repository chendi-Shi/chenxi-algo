"""Source-verifier regressions using tiny local fixtures and no PDF dependency."""
import hashlib
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from validation.validate_theme_sources import main, validate


class ThemeSourceValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.cache = self.root / 'validation/raw/theme/fixture.html'
        self.cache.parent.mkdir(parents=True)
        # Entirely synthetic source bytes; no factual claim about the ticker.
        self.quote = 'Synthetic optical transceiver fixture.'
        self.cache.write_text('<h1>Fixture section</h1><p>' + self.quote + '</p>', encoding='utf-8')
        digest = hashlib.sha256(self.cache.read_bytes()).hexdigest()
        self.company = dict(ticker='300308.SZ', name='Synthetic test issuer', market='A',
                            sector='Fixture sector', scope='technology', universe_as_of='2026-10-10')
        self.document = dict(document_id='fixture', ticker='300308.SZ', available_at='2026-10-10',
                             source_url='https://www.zj-innolight.com/synthetic-test-fixture',
                             source_type='company_website', page='Fixture section', text=self.quote,
                             title='Synthetic fixture', source_sha256=digest, source_id='fixture',
                             date_basis='observed_at', scope='technology', universe_as_of='2026-10-10')
        self.source = {
            key: self.document[key]
            for key in ('ticker', 'available_at', 'source_url', 'source_type', 'title',
                        'source_sha256', 'source_id', 'date_basis')
        }
        self.source.update(document_ids=['fixture'], observed_at='2026-10-10',
                           format='html', cache_path='validation/raw/theme/fixture.html',
                           summary_basis='analyst_paraphrase',
                           quote_sha256=hashlib.sha256(self.quote.encode('utf-8')).hexdigest(),
                           quote_page='Fixture section')
        self.write_inputs()

    def write_inputs(self):
        (self.data / 'theme_companies.json').write_text(json.dumps([self.company]), encoding='utf-8')
        (self.data / 'theme_documents.jsonl').write_text(json.dumps(self.document) + '\n', encoding='utf-8')
        metadata = dict(as_of='2026-10-10', sources=[self.source], evaluation_cases=[
            dict(query='fixture', direct_tickers=['300308.SZ'], development_tickers=[], non_direct_tickers=[])
        ])
        (self.data / 'theme_sources.json').write_text(json.dumps(metadata), encoding='utf-8')
        (self.data / 'theme_ingest_manifest.json').write_text('[]', encoding='utf-8')

    def assert_rejected(self, result, message):
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['original_sources_reverified'])
        self.assertTrue(any(message in error for error in result['errors']), result['errors'])

    def test_metadata_mode_does_not_claim_original_verification(self):
        with patch('socket.create_connection', side_effect=AssertionError('network forbidden')):
            offline = validate(self.root)
            self.assertEqual(offline['status'], 'passed')
            self.assertEqual(offline['mode'], 'offline_metadata_only')
            self.assertFalse(offline['original_sources_reverified'])
            self.assertEqual(offline['source_checks'], [])
            # Positive control: the tiny original is sufficient only when asked.
            checked = validate(self.root, verify_sources=True)
            self.assertEqual(checked['status'], 'passed')
            self.assertTrue(checked['original_sources_reverified'])
            self.assertTrue(checked['source_checks'][0]['quote_found'])

    def test_changed_quote_is_rejected(self):
        self.document['text'] = 'An invented quotation unrelated to the cached document.'
        self.write_inputs()
        self.assert_rejected(validate(self.root), 'quote/page mismatch')

    def test_future_availability_date_is_rejected(self):
        self.source['available_at'] = self.document['available_at'] = '2030-01-01'
        self.write_inputs()
        self.assert_rejected(validate(self.root), 'future source')

    def test_modified_original_bytes_are_rejected(self):
        self.cache.write_bytes(b'<h1>Fixture section</h1><p>Tampered source.</p>')
        self.assert_rejected(validate(self.root, verify_sources=True), 'cached source hash mismatch')

    def test_cli_output_saves_the_actual_verification_result(self):
        output = self.root / 'reports/source-results.json'
        stdout = io.StringIO()
        argv = ['validate_theme_sources.py', '--root', str(self.root), '--verify-sources', '--output', str(output)]
        with patch('sys.argv', argv), redirect_stdout(stdout):
            status = main()
        self.assertEqual(status, 0)
        saved = json.loads(output.read_text(encoding='utf-8'))
        self.assertTrue(saved['original_sources_reverified'])
        self.assertEqual(saved, json.loads(stdout.getvalue()))
        self.assertTrue(stdout.getvalue().isascii())

    def test_cli_parse_failure_replaces_a_previous_pass_report(self):
        output = self.root / 'source-results.json'
        output.write_text('{"status":"passed"}', encoding='utf-8')
        (self.data / 'theme_documents.jsonl').write_text('{broken', encoding='utf-8')
        argv = ['validate_theme_sources.py', '--root', str(self.root), '--output', str(output)]
        with patch('sys.argv', argv), redirect_stdout(io.StringIO()):
            status = main()
        self.assertEqual(status, 1)
        saved = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(saved['status'], 'failed')
        self.assertFalse(saved['original_sources_reverified'])


if __name__ == '__main__':
    unittest.main()
