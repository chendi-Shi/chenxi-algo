"""Import explicitly identified public documents into a traceable theme corpus.

Local UTF-8 TXT files need only the standard library. PDF extraction requires
the optional ``pypdf`` package. No publication date, company, or URL is inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import date
import hashlib
import http.client
import ipaddress
import json
import logging
import os
from pathlib import Path
import re
import socket
import tempfile
from urllib.parse import urljoin, urlsplit


MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 5 * 1024 * 1024
MAX_TEXT_CHARS = 8_000_000
MAX_PAGES = 10_000
DOWNLOAD_TIMEOUT = 20
MAX_REDIRECTS = 3
SOURCE_TYPES = {'annual_report', 'announcement', 'company_website'}
ENTRY_FIELDS = {'ticker', 'available_at', 'source_url', 'source_type',
                'local_path', 'title', 'format', 'source_sha256', 'document_id', 'date_basis'}
DATE_BASES = {'official_release', 'observed_at', 'operator_supplied'}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _reject_constant(value):
    raise ValueError(f'Non-finite JSON number is not allowed: {value}')


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _bounded_read(path: Path, limit: int) -> bytes:
    with path.open('rb') as handle:
        content = handle.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f'File exceeds the {limit}-byte size limit')
    return content


def _required_text(entry, name):
    value = entry.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    if value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f'{name} contains whitespace padding or control characters')
    return value


def _is_public_address(value):
    address = ipaddress.ip_address(value)
    return address.is_global and not address.is_multicast and not address.is_reserved


def _url_parts(url: str):
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise ValueError('Invalid source URL') from exc
    if parts.scheme.lower() != 'https' or not parts.hostname:
        raise ValueError('source_url must use HTTPS and include a hostname')
    if parts.username is not None or parts.password is not None:
        raise ValueError('URL credentials are not permitted')
    if parts.fragment or any(ord(char) < 33 for char in url) or '\\' in url:
        raise ValueError('URL fragments, whitespace, and backslashes are not permitted')
    host = parts.hostname.rstrip('.').lower()
    if host == 'localhost' or host.endswith('.localhost') or '%' in host:
        raise ValueError('Local or scoped hosts are not permitted')
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not _is_public_address(address):
        raise ValueError('Non-public IP addresses are not permitted')
    if port not in (None, 443):
        raise ValueError('Only HTTPS port 443 is permitted')
    return parts


def _public_addresses(host: str) -> list[str]:
    addresses = sorted({answer[4][0] for answer in
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)})
    if not addresses or any(not _is_public_address(address)
                            for address in addresses):
        raise ValueError('DNS must resolve exclusively to public IP addresses')
    return addresses


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Keep TLS hostname verification while connecting to a checked DNS address."""
    def __init__(self, host, address):
        super().__init__(host, port=443, timeout=DOWNLOAD_TIMEOUT)
        self._address = address

    def connect(self):
        raw_socket = socket.create_connection((self._address, 443), self.timeout)
        try:
            self.sock = self._context.wrap_socket(raw_socket, server_hostname=self.host)
        except BaseException:
            raw_socket.close()
            raise


def _download(url: str) -> tuple[bytes, str]:
    """Use no environment proxy; validate and pin each redirect's destination."""
    for redirect in range(MAX_REDIRECTS + 1):
        parts = _url_parts(url)
        host = parts.hostname.encode('idna').decode('ascii')
        addresses = _public_addresses(host)
        connection = _PinnedHTTPSConnection(host, addresses[0])
        try:
            target = parts.path or '/'
            if parts.query:
                target += '?' + parts.query
            connection.request('GET', target, headers={
                'User-Agent': 'chenxi-algo-document-ingestion/1.0',
                'Accept-Encoding': 'identity'})
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader('Location')
                if not location or redirect == MAX_REDIRECTS:
                    raise ValueError('Missing redirect location or too many redirects')
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError(f'Download returned HTTP {response.status}')
            encoding = response.getheader('Content-Encoding')
            if encoding and encoding.lower() != 'identity':
                raise ValueError('Compressed HTTP responses are not supported')
            length = response.getheader('Content-Length')
            if length is not None:
                if not length.isdigit() or int(length) > MAX_SOURCE_BYTES:
                    raise ValueError('Invalid or oversized HTTP Content-Length')
            chunks = []
            total = 0
            while True:
                chunk = response.read(min(65536, MAX_SOURCE_BYTES + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_SOURCE_BYTES:
                    raise ValueError('Downloaded document exceeds size limit')
            content = b''.join(chunks)
            if length is not None and len(content) != int(length):
                raise ValueError('Downloaded byte count differs from Content-Length')
            return content, url
        finally:
            connection.close()
    raise ValueError('Too many redirects')


def _entry(entry, manifest_dir):
    if not isinstance(entry, dict):
        raise ValueError('Manifest entries must be objects')
    if set(entry) - ENTRY_FIELDS:
        raise ValueError('Unknown manifest fields: ' + ', '.join(sorted(set(entry) - ENTRY_FIELDS)))
    for field in ('ticker', 'available_at', 'source_url', 'source_type'):
        _required_text(entry, field)
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', entry['available_at']):
        raise ValueError('available_at must be an explicit ISO date (YYYY-MM-DD)')
    date.fromisoformat(entry['available_at'])
    _url_parts(entry['source_url'])
    if entry['source_type'] not in SOURCE_TYPES:
        raise ValueError('Unsupported source_type')
    for field in ('local_path', 'title', 'format', 'source_sha256', 'document_id', 'date_basis'):
        if field in entry:
            _required_text(entry, field)
    if 'source_sha256' in entry and not re.fullmatch(r'[0-9a-fA-F]{64}', entry['source_sha256']):
        raise ValueError('source_sha256 must contain 64 hexadecimal characters')
    if 'date_basis' in entry and entry['date_basis'] not in DATE_BASES:
        raise ValueError('date_basis must be official_release, observed_at, or operator_supplied')
    local = (manifest_dir / entry['local_path']).resolve() if 'local_path' in entry else None
    suffix = local.suffix if local else Path(urlsplit(entry['source_url']).path).suffix
    document_format = entry.get('format', suffix.lstrip('.').lower())
    if document_format not in ('txt', 'pdf'):
        raise ValueError('Document format must be txt or pdf; set format explicitly if needed')
    return local, document_format


@contextmanager
def _pdf_warnings(destination):
    """Retain warning counts without flooding CLI logs for repeated PDF defects."""
    counts = Counter()
    class Capture(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                message = record.getMessage()
                key = (record.levelname, message)
                if key not in counts and len(counts) >= 50:
                    key = ('WARNING', 'Additional distinct PDF warnings omitted; count retained.')
                counts[key] += 1
    logger = logging.getLogger('pypdf')
    handlers, propagate, level = logger.handlers[:], logger.propagate, logger.level
    logger.handlers, logger.propagate, logger.level = [Capture()], False, logging.WARNING
    try:
        yield
    finally:
        logger.handlers, logger.propagate, logger.level = handlers, propagate, level
        destination.extend({'level': level, 'message': message, 'count': count}
                           for (level, message), count in sorted(counts.items()))


def _pages(content: bytes, document_format: str, *, warnings=None) -> list[tuple[int, str]]:
    if document_format == 'txt':
        pages = [(1, content.decode('utf-8', errors='strict'))]
    else:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError('PDF input requires optional dependency pypdf') from exc
        import io
        pages = []
        with _pdf_warnings(warnings if warnings is not None else []):
            reader = PdfReader(io.BytesIO(content), strict=True)
            if reader.is_encrypted:
                raise ValueError('Encrypted PDFs are not supported')
            if len(reader.pages) > MAX_PAGES:
                raise ValueError('PDF exceeds page limit')
            total = 0
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ''
                total += len(text)
                if total > MAX_TEXT_CHARS:
                    raise ValueError('Extracted text exceeds character limit')
                pages.append((number, text))
    if not any(text.strip() for _, text in pages):
        raise ValueError('No extractable text; image-only PDFs require a separate OCR step')
    if sum(len(text) for _, text in pages) > MAX_TEXT_CHARS:
        raise ValueError('Extracted text exceeds character limit')
    return pages


def _atomic_write(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _same_file(first: Path, second: Path) -> bool:
    if first == second:
        return True
    return first.exists() and second.exists() and first.samefile(second)


def ingest_manifest(manifest_path, output_path, *, download=False, cache_dir=None) -> dict:
    """Write documents plus ``<output>.manifest.json``; report every rejected entry.

    A rejected entry does not become evidence. ``partial`` or ``failed`` status
    requires caller attention even when the JSONL contains usable other sources.
    """
    if not isinstance(download, bool):
        raise ValueError('download must be a boolean')
    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()
    report_path = output_path.with_name(output_path.name + '.manifest.json')
    cache = Path(cache_dir).resolve() if cache_dir is not None else output_path.parent / 'source-cache'
    raw = _bounded_read(manifest_path, MAX_MANIFEST_BYTES)
    entries = json.loads(raw.decode('utf-8', errors='strict'),
        parse_constant=_reject_constant, parse_float=_reject_constant, object_pairs_hook=_object)
    if not isinstance(entries, list) or not entries:
        raise ValueError('Manifest must be a nonempty JSON list')
    protected = [manifest_path]
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get('local_path'), str):
            protected.append((manifest_path.parent / entry['local_path']).resolve())
    for target in (output_path, report_path):
        if any(_same_file(target, source) for source in protected):
            raise ValueError('Output paths must not overwrite the manifest or source files')
    documents, accepted, rejected = [], [], []
    seen_ids = set()
    for index, entry in enumerate(entries):
        extraction_warnings = []
        try:
            local, document_format = _entry(entry, manifest_path.parent)
            final_url = entry['source_url']
            downloaded = False
            if local is not None and (local.exists() or not download):
                content = _bounded_read(local, MAX_SOURCE_BYTES)
            elif not download:
                raise ValueError('No local_path provided; URL download requires --download')
            else:
                if local is not None and 'source_sha256' not in entry:
                    raise ValueError('Missing local file: download fallback requires expected source_sha256')
                content, final_url = _download(entry['source_url'])
                downloaded = True
            source_hash = _sha256(content)
            if 'source_sha256' in entry and source_hash != entry['source_sha256'].lower():
                raise ValueError('Document SHA256 does not match source_sha256')
            pages = _pages(content, document_format, warnings=extraction_warnings)
            imported = []
            for page, text in pages:
                if not text.strip():
                    continue
                identity = json.dumps([entry['ticker'], entry['available_at'], entry['source_url'],
                    entry['source_type'], source_hash, page], ensure_ascii=True, separators=(',', ':'))
                document_id = entry.get('document_id', 'doc') + '-p' + str(page) + '-' + _sha256(identity.encode('ascii'))
                if document_id in seen_ids:
                    raise ValueError('Duplicate document/page in manifest')
                record = {key: entry[key] for key in ('ticker', 'available_at', 'source_url', 'source_type')}
                record.update(document_id=document_id, page=page, text=text, source_sha256=source_hash)
                if 'title' in entry:
                    record['title'] = entry['title']
                if 'date_basis' in entry:
                    record['date_basis'] = entry['date_basis']
                if 'document_id' in entry:
                    record['source_document_id'] = entry['document_id']
                imported.append(record)
            cache_path = None
            if downloaded:
                cache_path = cache / (source_hash + '.' + document_format)
                if any(_same_file(cache_path, path) for path in protected + [output_path, report_path]):
                    raise ValueError('Cache path collides with an input or output path')
                if cache_path.exists():
                    if _sha256(_bounded_read(cache_path, MAX_SOURCE_BYTES)) != source_hash:
                        raise ValueError('Existing cache file failed SHA256 verification')
                else:
                    _atomic_write(cache_path, content)
            documents.extend(imported)
            seen_ids.update(record['document_id'] for record in imported)
            accepted.append({'entry_index': index, 'ticker': entry['ticker'],
                'source_url': entry['source_url'], 'final_source_url': final_url,
                'source_sha256': source_hash, 'bytes': len(content),
                'pages_imported': len(imported), 'pages_total': len(pages),
                'pages_without_text': [page for page, text in pages if not text.strip()],
                'extraction_warnings': extraction_warnings,
                'cache_path': str(cache_path) if cache_path else None})
        except Exception as exc:
            rejected.append({'entry_index': index,
                'ticker': entry.get('ticker') if isinstance(entry, dict) else None,
                'error': f'{type(exc).__name__}: {exc}', 'extraction_warnings': extraction_warnings})
    corpus = ''.join(json.dumps(document, ensure_ascii=False, allow_nan=False) + '\n'
                     for document in documents).encode('utf-8')
    report = {'schema_version': 1,
        'status': 'completed' if not rejected else ('partial' if accepted else 'failed'),
        'input_sha256': _sha256(raw), 'output_sha256': _sha256(corpus),
        'download_enabled': download, 'source_size_limit_bytes': MAX_SOURCE_BYTES,
        'download_timeout_seconds': DOWNLOAD_TIMEOUT, 'entries_total': len(entries),
        'entries_accepted': len(accepted), 'entries_rejected': len(rejected),
        'documents_written': len(documents), 'accepted': accepted, 'rejected': rejected,
        'limitations': ['Company identity, source attribution, and available_at are supplied by the operator.',
            'Text extraction does not establish the factual accuracy of a source.',
            'Image-only PDF pages need separate OCR and are not imported as evidence.']}
    _atomic_write(output_path, corpus)
    _atomic_write(report_path, (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--download', action='store_true', help='Allow explicit manifest HTTPS downloads')
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        report = ingest_manifest(args.manifest, args.output, download=args.download, cache_dir=args.cache_dir)
    except (OSError, ValueError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
