"""Document ingestion contracts, integrity checks, and mocked network boundaries."""
import contextlib
import hashlib
import io
import json
import logging
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import ingest_theme as ingest


class FakeResponse:
    def __init__(self, body=b'', status=200, headers=None):
        self.body = io.BytesIO(body)
        self.status = status
        self.headers = headers or {}

    def getheader(self, key):
        return self.headers.get(key)

    def read(self, size):
        return self.body.read(size)


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.manifest = self.directory / 'input.json'
        self.output = self.directory / 'corpus.jsonl'
        self.source = self.directory / 'report.txt'
        self.source.write_text('主营光模块。\nOptical modules: revenue 100.\n', encoding='utf-8')
        self.entry = {'ticker': '300308.SZ', 'available_at': '2025-04-01',
            'source_url': 'https://example.com/report.txt', 'source_type': 'annual_report',
            'local_path': 'report.txt', 'title': 'Annual report'}

    def write_manifest(self, entries=None):
        entries = [self.entry] if entries is None else entries
        self.manifest.write_text(json.dumps(entries, ensure_ascii=False), encoding='utf-8')

    def invoke(self, entries=None, **kwargs):
        self.write_manifest(entries)
        return ingest.ingest_manifest(self.manifest, self.output, **kwargs)

    def rows(self):
        return [json.loads(line) for line in self.output.read_text(encoding='utf-8').splitlines()]

    def network(self, responses, addresses=None):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        address = '93.184.216.34'
        dns = addresses if addresses is not None else [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 443))]
        stack.enter_context(mock.patch.object(ingest.socket, 'getaddrinfo', return_value=dns))
        connections = []
        for response in responses:
            connection = mock.MagicMock()
            connection.getresponse.return_value = response
            connections.append(connection)
        factory = stack.enter_context(mock.patch.object(ingest, '_PinnedHTTPSConnection', side_effect=connections))
        return factory, connections

    def remote_entry(self):
        return {key: value for key, value in self.entry.items() if key != 'local_path'}

    def test_local_complete_text_and_provenance_are_deterministic(self):
        with mock.patch.object(ingest, '_download', side_effect=AssertionError('network forbidden')):
            report = self.invoke()
            first = self.output.read_bytes()
            repeat = ingest.ingest_manifest(self.manifest, self.output)
        self.assertEqual(report, repeat)
        self.assertEqual(first, self.output.read_bytes())
        row, = self.rows()
        self.assertEqual(row['text'], self.source.read_bytes().decode('utf-8'))
        self.assertEqual(row['page'], 1)
        self.assertEqual(row['source_sha256'], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(report['input_sha256'], hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        self.assertEqual(report['output_sha256'], hashlib.sha256(first).hexdigest())
        self.assertEqual(report['status'], 'completed')
        saved = json.loads(Path(str(self.output) + '.manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(saved, report)

    def test_different_company_date_and_source_change_document_identity(self):
        entries = [dict(self.entry), dict(self.entry, ticker='000001.SZ'),
                   dict(self.entry, available_at='2025-04-02'),
                   dict(self.entry, source_url='https://example.com/other.txt')]
        report = self.invoke(entries)
        self.assertEqual(report['documents_written'], 4)
        self.assertEqual(len({row['document_id'] for row in self.rows()}), 4)

    def test_duplicate_entry_rejected_without_duplicate_evidence(self):
        report = self.invoke([self.entry, self.entry])
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['documents_written'], 1)
        self.assertIn('Duplicate', report['rejected'][0]['error'])

    def test_no_implicit_download(self):
        with mock.patch.object(ingest, '_download') as download:
            report = self.invoke([self.remote_entry()])
        download.assert_not_called()
        self.assertEqual(report['status'], 'failed')
        self.assertIn('--download', report['rejected'][0]['error'])

    def test_optional_source_identity_and_date_basis_are_preserved(self):
        report = self.invoke([dict(self.entry, document_id='official-source', date_basis='official_release')])
        row, = self.rows()
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(row['source_document_id'], 'official-source')
        self.assertTrue(row['document_id'].startswith('official-source-p1-'))
        self.assertEqual(row['date_basis'], 'official_release')
        for change in ({'document_id': 7}, {'date_basis': 'guessed'}, {'date_basis': None}):
            self.assertEqual(self.invoke([dict(self.entry, **change)])['status'], 'failed')

    def test_missing_local_can_explicitly_fallback_only_with_expected_hash(self):
        data = b'public text'
        entry = dict(self.entry, local_path='not-downloaded.txt', source_sha256=hashlib.sha256(data).hexdigest())
        with mock.patch.object(ingest, '_download', return_value=(data, entry['source_url'])) as download:
            report = self.invoke([entry], download=True)
        download.assert_called_once()
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(Path(report['accepted'][0]['cache_path']).read_bytes(), data)
        with mock.patch.object(ingest, '_download') as download:
            report = self.invoke([dict(self.entry, local_path='missing.txt')], download=True)
        download.assert_not_called()
        self.assertIn('requires expected source_sha256', report['rejected'][0]['error'])

    def test_fallback_hash_drift_is_failure_without_cache_or_evidence(self):
        entry = dict(self.entry, local_path='missing.txt', source_sha256='0' * 64)
        with mock.patch.object(ingest, '_download', return_value=(b'drifted text', entry['source_url'])):
            report = self.invoke([entry], download=True)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['documents_written'], 0)
        self.assertFalse((self.directory / 'source-cache').exists())

    def test_pdf_warnings_are_deduplicated_counted_and_logging_restored(self):
        logger = logging.getLogger('pypdf')
        original = (logger.handlers[:], logger.level, logger.propagate)
        warnings = []
        with ingest._pdf_warnings(warnings):
            child = logging.getLogger('pypdf.generic')
            child.warning('Repeated PDF defect')
            child.warning('Repeated PDF defect')
            child.error('Different PDF defect')
        self.assertEqual((logger.handlers, logger.level, logger.propagate), original)
        self.assertEqual(sum(row['count'] for row in warnings), 3)
        self.assertEqual(len(warnings), 2)
        self.assertEqual(next(row['count'] for row in warnings if row['message'] == 'Repeated PDF defect'), 2)

    def test_64_mib_limit_supports_large_official_public_pdfs(self):
        self.assertEqual(ingest.MAX_SOURCE_BYTES, 64 * 1024 * 1024)
        report = self.invoke()
        self.assertEqual(report['source_size_limit_bytes'], 64 * 1024 * 1024)

    def test_invalid_schema_and_types_are_rejected(self):
        bad = [None, [], dict(self.entry, ticker=123), dict(self.entry, available_at=True),
               dict(self.entry, available_at='2025-02-30'), dict(self.entry, available_at='20250401'),
               dict(self.entry, source_type='guess'), dict(self.entry, local_path=False),
               dict(self.entry, format='docx'), dict(self.entry, title=[]),
               dict(self.entry, source_sha256='wrong'), dict(self.entry, surprise='field'),
               dict(self.entry, ticker=' 300308.SZ'), dict(self.entry, ticker='bad\nname')]
        report = self.invoke(bad)
        self.assertEqual(report['entries_rejected'], len(bad))
        self.assertEqual(self.rows(), [])

    def test_invalid_json_is_fatal_and_preserves_previous_output(self):
        self.output.write_bytes(b'previous output')
        for content in ('[]', '{}', '[NaN]', '[Infinity]', '[1e9999]',
                        '[{"ticker":"a","ticker":"b"}]'):
            with self.subTest(content=content):
                self.manifest.write_text(content, encoding='utf-8')
                with self.assertRaises(ValueError):
                    ingest.ingest_manifest(self.manifest, self.output)
                self.assertEqual(self.output.read_bytes(), b'previous output')

    def test_utf8_decode_is_strict(self):
        self.source.write_bytes(b'bad\xfftext')
        report = self.invoke()
        self.assertEqual(report['status'], 'failed')
        self.assertIn('UnicodeDecodeError', report['rejected'][0]['error'])
        self.manifest.write_bytes(b'\xff')
        with self.assertRaises(UnicodeDecodeError):
            ingest.ingest_manifest(self.manifest, self.output)

    def test_explicit_source_hash_checked_before_extraction(self):
        report = self.invoke([dict(self.entry, source_sha256='0' * 64)])
        self.assertEqual(report['status'], 'failed')
        self.assertIn('SHA256', report['rejected'][0]['error'])
        expected = hashlib.sha256(self.source.read_bytes()).hexdigest().upper()
        self.assertEqual(self.invoke([dict(self.entry, source_sha256=expected)])['status'], 'completed')

    def test_input_and_manifest_cannot_be_overwritten(self):
        self.write_manifest()
        source_bytes = self.source.read_bytes()
        manifest_bytes = self.manifest.read_bytes()
        for output in (self.source, self.manifest):
            with self.assertRaises(ValueError):
                ingest.ingest_manifest(self.manifest, output)
        self.assertEqual(self.source.read_bytes(), source_bytes)
        self.assertEqual(self.manifest.read_bytes(), manifest_bytes)
        collision = Path(str(self.output) + '.manifest.json')
        collision.write_bytes(source_bytes)
        self.write_manifest([dict(self.entry, local_path=collision.name)])
        with self.assertRaises(ValueError):
            ingest.ingest_manifest(self.manifest, self.output)
        self.assertEqual(collision.read_bytes(), source_bytes)

    def test_missing_source_and_source_size_limits_are_visible(self):
        report = self.invoke([dict(self.entry, local_path='missing.txt')])
        self.assertIn('FileNotFoundError', report['rejected'][0]['error'])
        with mock.patch.object(ingest, 'MAX_SOURCE_BYTES', 2):
            report = self.invoke()
        self.assertIn('size limit', report['rejected'][0]['error'])
        self.write_manifest()
        with mock.patch.object(ingest, 'MAX_MANIFEST_BYTES', 2), self.assertRaises(ValueError):
            ingest.ingest_manifest(self.manifest, self.output)

    def test_empty_or_too_long_text_is_not_evidence(self):
        self.source.write_text(' \n ', encoding='utf-8')
        self.assertEqual(self.invoke()['status'], 'failed')
        self.source.write_text('12345', encoding='utf-8')
        with mock.patch.object(ingest, 'MAX_TEXT_CHARS', 3):
            self.assertEqual(self.invoke()['status'], 'failed')

    def test_pdf_pages_preserve_original_numbers_and_report_unreadable_pages(self):
        self.source = self.directory / 'report.pdf'
        self.source.write_bytes(b'%PDF-mocked-fixture')
        self.entry['local_path'] = self.source.name
        reader = types.SimpleNamespace(is_encrypted=False, pages=[
            types.SimpleNamespace(extract_text=lambda: 'page one'),
            types.SimpleNamespace(extract_text=lambda: ''),
            types.SimpleNamespace(extract_text=lambda: 'page three')])
        module = types.SimpleNamespace(PdfReader=mock.Mock(return_value=reader))
        with mock.patch.dict('sys.modules', {'pypdf': module}):
            report = self.invoke()
        self.assertEqual([row['page'] for row in self.rows()], [1, 3])
        self.assertEqual(report['accepted'][0]['pages_without_text'], [2])
        self.assertEqual(report['accepted'][0]['pages_total'], 3)

    def test_pdf_dependency_and_encryption_fail_explicitly(self):
        self.entry['format'] = 'pdf'
        with mock.patch.dict('sys.modules', {'pypdf': None}):
            report = self.invoke()
        self.assertIn('pypdf', report['rejected'][0]['error'])
        module = types.SimpleNamespace(PdfReader=lambda *a, **k: types.SimpleNamespace(is_encrypted=True))
        with mock.patch.dict('sys.modules', {'pypdf': module}):
            report = self.invoke()
        self.assertIn('Encrypted', report['rejected'][0]['error'])

    def test_https_download_is_bounded_cached_and_hashed(self):
        data = '光模块'.encode('utf-8')
        factory, connections = self.network([FakeResponse(data, headers={'Content-Length': str(len(data))})])
        report = self.invoke([self.remote_entry()], download=True, cache_dir=self.directory / 'cache')
        self.assertEqual(report['status'], 'completed')
        cached = Path(report['accepted'][0]['cache_path'])
        self.assertEqual(cached.read_bytes(), data)
        self.assertEqual(cached.name, hashlib.sha256(data).hexdigest() + '.txt')
        factory.assert_called_once_with('example.com', '93.184.216.34')
        connections[0].close.assert_called_once()
        self.assertEqual(self.rows()[0]['text'], '光模块')

    def test_direct_unsafe_urls_rejected_even_without_download(self):
        bad = ['http://example.com/a.txt', 'file:///a.txt', 'https://localhost/a.txt',
               'https://foo.localhost/a.txt', 'https://127.0.0.1/a.txt', 'https://10.0.0.1/a.txt',
               'https://169.254.169.254/a.txt', 'https://[::1]/a.txt', 'https://[fc00::1]/a.txt',
               'https://user:password@example.com/a.txt', 'https://example.com:8080/a.txt',
               'https://224.0.0.1/a.txt', 'https://example.com/a.txt#fragment',
               'https://example.com/ bad.txt', 'https://example.com\\local/a.txt']
        report = self.invoke([dict(self.entry, source_url=url) for url in bad])
        self.assertEqual(report['entries_rejected'], len(bad))

    def test_private_dns_and_mixed_public_private_dns_are_rejected(self):
        for addresses in ([('10.0.0.1', 443)], [('93.184.216.34', 443), ('127.0.0.1', 443)]):
            dns = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', address) for address in addresses]
            with mock.patch.object(ingest.socket, 'getaddrinfo', return_value=dns), \
                 mock.patch.object(ingest, '_PinnedHTTPSConnection') as connection:
                report = self.invoke([self.remote_entry()], download=True)
            connection.assert_not_called()
            self.assertEqual(report['status'], 'failed')
            self.assertIn('DNS', report['rejected'][0]['error'])

    def test_redirects_revalidate_destination_and_record_final_url(self):
        self.network([FakeResponse(status=302, headers={'Location': '/moved.txt'}), FakeResponse(b'good')])
        report = self.invoke([self.remote_entry()], download=True)
        self.assertEqual(report['accepted'][0]['final_source_url'], 'https://example.com/moved.txt')
        self.assertEqual(self.rows()[0]['source_url'], self.entry['source_url'])

    def test_private_and_http_redirects_are_rejected_before_connect(self):
        for destination in ('https://127.0.0.1/secret.txt', 'http://example.com/report.txt'):
            with mock.patch.object(ingest, '_public_addresses', return_value=['93.184.216.34']), \
                 mock.patch.object(ingest, '_PinnedHTTPSConnection') as factory:
                factory.return_value.getresponse.return_value = FakeResponse(status=302, headers={'Location': destination})
                report = self.invoke([self.remote_entry()], download=True)
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(factory.call_count, 1)

    def test_http_errors_truncation_oversize_and_compression_rejected(self):
        cases = [FakeResponse(status=404), FakeResponse(b'123', headers={'Content-Length': '4'}),
                 FakeResponse(b'123', headers={'Content-Length': '999999999'}),
                 FakeResponse(b'123', headers={'Content-Length': '-1'}),
                 FakeResponse(b'123', headers={'Content-Encoding': 'gzip'}), FakeResponse(b'123456')]
        for response in cases:
            with mock.patch.object(ingest, '_public_addresses', return_value=['93.184.216.34']), \
                 mock.patch.object(ingest, '_PinnedHTTPSConnection') as factory, \
                 mock.patch.object(ingest, 'MAX_SOURCE_BYTES', 5):
                factory.return_value.getresponse.return_value = response
                report = self.invoke([self.remote_entry()], download=True)
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['documents_written'], 0)

    def test_timeout_is_rejection_and_does_not_mask_good_local_entry(self):
        with mock.patch.object(ingest, '_download', side_effect=TimeoutError('timed out')):
            report = self.invoke([self.remote_entry(), self.entry], download=True)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['documents_written'], 1)
        self.assertIn('TimeoutError', report['rejected'][0]['error'])

    def test_tls_connect_pins_ip_and_verifies_original_hostname(self):
        connection = ingest._PinnedHTTPSConnection('example.com', '93.184.216.34')
        context = mock.Mock()
        connection._context = context
        with mock.patch.object(ingest.socket, 'create_connection') as create:
            connection.connect()
        create.assert_called_once_with(('93.184.216.34', 443), ingest.DOWNLOAD_TIMEOUT)
        context.wrap_socket.assert_called_once_with(create.return_value, server_hostname='example.com')

    def test_failed_atomic_replace_preserves_output_and_removes_tempfile(self):
        self.output.write_bytes(b'previous')
        self.write_manifest()
        with mock.patch.object(ingest.os, 'replace', side_effect=OSError('write failed')):
            with self.assertRaises(OSError):
                ingest.ingest_manifest(self.manifest, self.output)
        self.assertEqual(self.output.read_bytes(), b'previous')
        self.assertEqual(list(self.directory.glob('.*.tmp')), [])

    def test_cli_exit_codes_distinguish_success_rejections_and_fatal_errors(self):
        args = ['--manifest', str(self.manifest), '--output', str(self.output)]
        self.write_manifest()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ingest.main(args), 0)
            self.write_manifest([self.remote_entry()])
            self.assertEqual(ingest.main(args), 1)
            self.manifest.write_text('{}', encoding='utf-8')
            self.assertEqual(ingest.main(args), 2)

    def test_downloaded_source_is_reused_offline_after_digest_validation(self):
        entry = self.remote_entry()
        data = b'public original business'
        with mock.patch.object(ingest, '_download', return_value=(data, entry['source_url'])):
            first = self.invoke([entry], download=True)
        with mock.patch.object(ingest, '_download', side_effect=AssertionError('offline')):
            second = ingest.ingest_manifest(self.manifest, self.output)
        self.assertEqual(second['status'], 'completed')
        self.assertEqual(second['accepted'][0]['acquisition'], 'verified_cache')
        self.assertEqual(second['accepted'][0]['fetched_at'], first['accepted'][0]['fetched_at'])
        self.assertEqual(second['accepted'][0]['source_sha256'], hashlib.sha256(data).hexdigest())
        self.assertEqual(second['accepted'][0]['requests'], [])
        self.assertEqual(ingest.verify_ingestion_output(self.output), second)

    def test_manifest_metadata_and_expected_hash_changes_invalidate_cache_index(self):
        entry = self.remote_entry()
        data = b'original business'
        with mock.patch.object(ingest, '_download', return_value=(data, entry['source_url'])):
            self.invoke([entry], download=True)
        changes = [{'title': 'changed title'}, {'available_at': '2025-04-02'},
                   {'source_url': 'https://example.com/new.txt'},
                   {'source_sha256': '0' * 64}, {'ticker': 'other-company'}]
        for change in changes:
            with self.subTest(change=change), mock.patch.object(ingest, '_download') as download:
                result = self.invoke([dict(entry, **change)])
            download.assert_not_called()
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['documents_written'], 0)

    def test_cache_corruption_is_not_silently_redownloaded_or_used(self):
        entry = self.remote_entry()
        with mock.patch.object(ingest, '_download', return_value=(b'good', entry['source_url'])):
            first = self.invoke([entry], download=True)
        Path(first['accepted'][0]['cache_path']).write_bytes(b'tampered')
        with mock.patch.object(ingest, '_download') as download:
            result = ingest.ingest_manifest(self.manifest, self.output, download=True)
        download.assert_not_called()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('SHA256 verification', result['rejected'][0]['error'])

    def test_refresh_is_explicit_and_new_download_can_replace_cache_index(self):
        entry = self.remote_entry()
        with mock.patch.object(ingest, '_download', return_value=(b'old', entry['source_url'])):
            self.invoke([entry], download=True)
        with self.assertRaises(ValueError):
            ingest.ingest_manifest(self.manifest, self.output, refresh=True)
        with mock.patch.object(ingest, '_download', return_value=(b'new', entry['source_url'])) as download:
            result = ingest.ingest_manifest(self.manifest, self.output, download=True, refresh=True)
        download.assert_called_once()
        self.assertEqual(result['accepted'][0]['acquisition'], 'https_download')
        self.assertEqual(self.rows()[0]['text'], 'new')
        self.assertEqual(ingest.ingest_manifest(self.manifest, self.output)['accepted'][0]['acquisition'], 'verified_cache')
        self.assertEqual(self.rows()[0]['text'], 'new')

    def test_changed_expected_hash_fetches_new_bytes_without_old_cache_reuse(self):
        entry = self.remote_entry()
        old = b'old public report'
        new = b'restated public report'
        entry['source_sha256'] = hashlib.sha256(old).hexdigest()
        with mock.patch.object(ingest, '_download', return_value=(old, entry['source_url'])):
            first = self.invoke([entry], download=True)
        entry['source_sha256'] = hashlib.sha256(new).hexdigest()
        with mock.patch.object(ingest, '_download', return_value=(new, entry['source_url'])) as download:
            second = self.invoke([entry], download=True)
        download.assert_called_once()
        self.assertEqual(second['status'], 'completed')
        self.assertEqual(self.rows()[0]['text'], new.decode())
        self.assertEqual(Path(first['accepted'][0]['cache_path']).read_bytes(), old)
        self.assertEqual(ingest.ingest_manifest(self.manifest, self.output)['accepted'][0]['acquisition'], 'verified_cache')
        self.assertEqual(self.rows()[0]['text'], new.decode())

    def test_transient_http_failure_retries_and_audits_each_request(self):
        factory, _ = self.network([FakeResponse(status=503, headers={'Retry-After': '120'}),
                                   FakeResponse(status=429), FakeResponse(b'recovered')])
        with mock.patch.object(ingest.time, 'sleep') as sleep:
            report = self.invoke([self.remote_entry()], download=True)
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(factory.call_count, 3)
        requests = report['accepted'][0]['requests']
        self.assertEqual([request['status'] for request in requests], [503, 429, 200])
        self.assertEqual([request['attempt'] for request in requests], [1, 2, 3])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5.0, 1.0])

    def test_network_timeout_exhaustion_is_bounded_and_audited(self):
        with mock.patch.object(ingest, '_public_addresses', return_value=['93.184.216.34']), \
             mock.patch.object(ingest, '_PinnedHTTPSConnection') as factory, \
             mock.patch.object(ingest.time, 'sleep') as sleep:
            factory.return_value.getresponse.side_effect = TimeoutError('fixture timeout')
            report = self.invoke([self.remote_entry()], download=True, max_attempts=2)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(factory.call_count, 2)
        sleep.assert_called_once_with(0.5)
        self.assertEqual(len(report['rejected'][0]['requests']), 2)

    def test_nonretryable_errors_and_hash_mismatch_do_not_retry(self):
        for status in (400, 401, 403, 404):
            factory, _ = self.network([FakeResponse(status=status)])
            with mock.patch.object(ingest.time, 'sleep') as sleep:
                report = self.invoke([self.remote_entry()], download=True)
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(factory.call_count, 1)
            sleep.assert_not_called()
        factory, _ = self.network([FakeResponse(b'wrong hash')])
        with mock.patch.object(ingest.time, 'sleep') as sleep:
            report = self.invoke([dict(self.remote_entry(), source_sha256='0' * 64)], download=True)
        self.assertEqual(factory.call_count, 1)
        sleep.assert_not_called()
        self.assertIn('SHA256', report['rejected'][0]['error'])

    def test_retry_setting_is_strict(self):
        self.write_manifest()
        for attempts in (True, 0, -1, 6, 1.5, '3'):
            with self.subTest(attempts=attempts), self.assertRaises(ValueError):
                ingest.ingest_manifest(self.manifest, self.output, max_attempts=attempts)

    def test_tls_verification_failure_is_not_retried(self):
        with mock.patch.object(ingest, '_public_addresses', return_value=['93.184.216.34']), \
             mock.patch.object(ingest, '_PinnedHTTPSConnection') as factory, \
             mock.patch.object(ingest.time, 'sleep') as sleep:
            factory.return_value.request.side_effect = ssl.SSLCertVerificationError('fixture certificate error')
            report = self.invoke([self.remote_entry()], download=True)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(factory.call_count, 1)
        sleep.assert_not_called()

    def test_manifest_or_local_source_mutation_aborts_without_replacing_success(self):
        first = self.invoke()
        previous = self.output.read_bytes()
        parse = ingest._pages
        for target in (self.manifest, self.source):
            original = target.read_bytes()
            def mutate(content, document_format, **kwargs):
                target.write_bytes(original + b' ')
                return parse(content, document_format, **kwargs)
            with self.subTest(target=target), mock.patch.object(ingest, '_pages', side_effect=mutate):
                with self.assertRaisesRegex(ValueError, 'Input changed during ingestion'):
                    ingest.ingest_manifest(self.manifest, self.output)
            self.assertEqual(self.output.read_bytes(), previous)
            self.assertEqual(ingest.verify_ingestion_output(self.output), first)
            target.write_bytes(original)

    def test_source_mutation_while_reading_cannot_become_evidence(self):
        read = ingest._bounded_read
        def mutate(path, limit):
            data = read(path, limit)
            if ingest.filesystem_path(path) == ingest.filesystem_path(self.source):
                self.source.write_bytes(data + b'changed')
            return data
        with mock.patch.object(ingest, '_bounded_read', side_effect=mutate):
            report = self.invoke()
        self.assertEqual(report['status'], 'failed')
        self.assertIn('changed while being read', report['rejected'][0]['error'])
        self.assertEqual(report['documents_written'], 0)

    def test_external_output_writer_is_detected_before_commit(self):
        self.invoke()
        parse = ingest._pages
        def mutate(content, document_format, **kwargs):
            self.output.write_bytes(b'external run')
            return parse(content, document_format, **kwargs)
        with mock.patch.object(ingest, '_pages', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'Output changed during ingestion'):
                ingest.ingest_manifest(self.manifest, self.output)
        self.assertEqual(self.output.read_bytes(), b'external run')

    def test_interrupted_commit_cannot_pass_verification_and_next_run_recovers(self):
        self.invoke()
        self.source.write_bytes(b'new corpus')
        write = ingest._atomic_write
        report_path = Path(str(self.output) + '.manifest.json')
        def fail_report(path, content):
            if ingest.filesystem_path(path) == ingest.filesystem_path(report_path):
                raise OSError('simulated report write failure')
            return write(path, content)
        with mock.patch.object(ingest, '_atomic_write', side_effect=fail_report):
            with self.assertRaises(OSError):
                ingest.ingest_manifest(self.manifest, self.output)
        with self.assertRaisesRegex(ValueError, 'not complete'):
            ingest.verify_ingestion_output(self.output)
        recovered = ingest.ingest_manifest(self.manifest, self.output)
        self.assertEqual(ingest.verify_ingestion_output(self.output), recovered)
        self.assertEqual(self.rows()[0]['text'], 'new corpus')

    def test_verifier_rejects_tampered_corpus_and_report(self):
        for target in (self.output, Path(str(self.output) + '.manifest.json')):
            self.invoke()
            target.write_bytes(target.read_bytes() + b' ')
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                ingest.verify_ingestion_output(self.output)

    def test_same_output_has_process_lock_but_other_output_is_independent(self):
        code = ('import sys\nfrom ingest_theme import output_lock\n'
                'try:\n with output_lock(sys.argv[1]): print("acquired")\n'
                'except ValueError:\n sys.exit(23)\n')
        repo = Path(ingest.__file__).parent
        with ingest.output_lock(self.output) as lock_path:
            self.assertEqual(lock_path, Path(str(self.output) + '.lock'))
            same = subprocess.run([sys.executable, '-c', code, str(self.output)],
                                  cwd=repo, capture_output=True, timeout=20)
            other = subprocess.run([sys.executable, '-c', code, str(self.directory / 'other.jsonl')],
                                   cwd=repo, capture_output=True, timeout=20)
        self.assertEqual(same.returncode, 23, same.stderr.decode())
        self.assertEqual(other.returncode, 0, other.stderr.decode())
        self.assertTrue(lock_path.exists())
        with ingest.output_lock(self.output):
            pass

    def test_sidecar_and_lock_cannot_overwrite_sources(self):
        for suffix in ('.complete.json', '.lock'):
            source_path = Path(str(self.output) + suffix)
            source_path.write_bytes(b'protected source')
            self.write_manifest([dict(self.entry, local_path=source_path.name, format='txt')])
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                ingest.ingest_manifest(self.manifest, self.output)
            self.assertEqual(source_path.read_bytes(), b'protected source')

    def test_crashed_process_releases_os_lock_without_removing_lock_file(self):
        code = ('import os,sys\nfrom ingest_theme import output_lock\n'
                'with output_lock(sys.argv[1]): os._exit(17)\n')
        result = subprocess.run([sys.executable, '-c', code, str(self.output)],
                                cwd=Path(ingest.__file__).parent, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 17, result.stderr.decode())
        self.assertTrue(Path(str(self.output) + '.lock').exists())
        with ingest.output_lock(self.output):
            pass

    def test_report_records_parser_and_ingestion_code_version(self):
        report = self.invoke()
        self.assertEqual(report['ingestion_code_sha256'], hashlib.sha256(Path(ingest.__file__).read_bytes()).hexdigest())
        self.assertEqual(report['accepted'][0]['parser']['name'], 'python-utf8')
        self.assertEqual(report['accepted'][0]['parser']['version'], sys.version.split()[0])
        self.assertEqual(report['accepted'][0]['acquisition'], 'local')


if __name__ == '__main__':
    unittest.main()
