"""Import explicitly identified public documents into a traceable theme corpus.

Local UTF-8 TXT files need only the standard library. PDF extraction requires
the optional ``pypdf`` package. No publication date, company, or URL is inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
import hashlib
import http.client
import importlib.metadata
import ipaddress
import json
import logging
import os
from pathlib import Path
import re
import socket
import ssl
import sys
import tempfile
import time
from urllib.parse import urljoin, urlsplit


MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 5 * 1024 * 1024
MAX_TEXT_CHARS = 8_000_000
MAX_PAGES = 10_000
DOWNLOAD_TIMEOUT = 20
MAX_REDIRECTS = 3
MAX_DOWNLOAD_ATTEMPTS = 3
RETRY_BASE_SECONDS = 0.5
MAX_RETRY_DELAY_SECONDS = 5.0
INGESTION_VERSION = '2.0'
SOURCE_TYPES = {'annual_report', 'announcement', 'company_website'}
ENTRY_FIELDS = {'ticker', 'available_at', 'source_url', 'source_type',
                'local_path', 'title', 'format', 'source_sha256', 'document_id', 'date_basis'}
DATE_BASES = {'official_release', 'observed_at', 'operator_supplied'}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def filesystem_path(path):
    """Use Windows extended absolute paths without changing machine policy.

    Nested retained runs and SHA256 filenames can exceed legacy MAX_PATH even
    when their individual filenames are short. POSIX paths are unchanged.
    """
    resolved = Path(path).resolve()
    value = str(resolved)
    if os.name != 'nt' or value.startswith('\\\\?\\'):
        return resolved
    if value.startswith('\\\\'):
        return Path('\\\\?\\UNC\\' + value[2:])
    return Path('\\\\?\\' + value)


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
    with filesystem_path(path).open('rb') as handle:
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


class _RetryableHTTPError(OSError):
    def __init__(self, status, retry_after=None):
        super().__init__(f'Download returned HTTP {status}')
        self.retry_after = retry_after


def _download_once(url, audit, attempt):
    """Use no environment proxy; validate and pin each redirect's destination."""
    deadline = time.monotonic() + DOWNLOAD_TIMEOUT
    for redirect in range(MAX_REDIRECTS + 1):
        event = {'attempt': attempt, 'redirect': redirect, 'url': url,
                 'outcome': 'started'}
        audit.append(event)
        connection = None
        try:
            parts = _url_parts(url)
            host = parts.hostname.encode('idna').decode('ascii')
            addresses = _public_addresses(host)
            address = addresses[(attempt - 1) % len(addresses)]
            event['address'] = address
            connection = _PinnedHTTPSConnection(host, address)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Download attempt exceeded its time budget')
            connection.timeout = remaining
            target = parts.path or '/'
            if parts.query:
                target += '?' + parts.query
            connection.request('GET', target, headers={
                'User-Agent': 'chenxi-algo-document-ingestion/1.0',
                'Accept-Encoding': 'identity'})
            response = connection.getresponse()
            event['status'] = response.status
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader('Location')
                if not location or redirect == MAX_REDIRECTS:
                    raise ValueError('Missing redirect location or too many redirects')
                url = urljoin(url, location)
                event['outcome'] = 'redirect'
                continue
            if response.status in (429, 500, 502, 503, 504):
                raise _RetryableHTTPError(response.status, response.getheader('Retry-After'))
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
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Download attempt exceeded its time budget')
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = response.read(min(65536, MAX_SOURCE_BYTES + 1 - total))
                if time.monotonic() > deadline:
                    raise TimeoutError('Download attempt exceeded its time budget')
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_SOURCE_BYTES:
                    raise ValueError('Downloaded document exceeds size limit')
            content = b''.join(chunks)
            if length is not None and len(content) != int(length):
                raise ValueError('Downloaded byte count differs from Content-Length')
            event.update(outcome='completed', bytes=len(content))
            return content, url
        except Exception as exc:
            event.update(outcome='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if connection is not None:
                connection.close()
    raise ValueError('Too many redirects')


def _download(url: str, *, audit=None, max_attempts=MAX_DOWNLOAD_ATTEMPTS) -> tuple[bytes, str]:
    """Retry transient failures only, with bounded attempts and delays."""
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 5:
        raise ValueError('max_attempts must be an integer between 1 and 5')
    audit = audit if audit is not None else []
    for attempt in range(1, max_attempts + 1):
        try:
            return _download_once(url, audit, attempt)
        except (OSError, http.client.HTTPException) as exc:
            if isinstance(exc, ssl.SSLCertVerificationError) or attempt == max_attempts:
                raise
            delay = min(MAX_RETRY_DELAY_SECONDS, RETRY_BASE_SECONDS * 2 ** (attempt - 1))
            retry_after = getattr(exc, 'retry_after', None)
            if isinstance(retry_after, str) and retry_after.isdigit():
                delay = min(MAX_RETRY_DELAY_SECONDS, max(delay, int(retry_after)))
            audit[-1]['retry_delay_seconds'] = delay
            time.sleep(delay)
    raise AssertionError('Unreachable retry state')


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
    local = filesystem_path(manifest_dir / entry['local_path']) if 'local_path' in entry else None
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
    path = filesystem_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.ingest-',
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
    first, second = filesystem_path(first), filesystem_path(second)
    if first == second:
        return True
    return first.exists() and second.exists() and first.samefile(second)


@contextmanager
def output_lock(target_path):
    """Hold a nonblocking process lock on one output target; never infer stale PIDs.

    The adjacent ``<target>.lock`` file remains on disk. The OS releases the lock
    on process exit, including abnormal exit. Different targets remain independent.
    """
    target = Path(target_path).resolve()
    lock_path = target.with_name(target.name + '.lock')
    filesystem_path(lock_path.parent).mkdir(parents=True, exist_ok=True)
    with filesystem_path(lock_path).open('a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'\0')
            handle.flush()
        handle.seek(0)
        acquired = False
        try:
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as exc:
                raise ValueError(f'Another run holds the output lock: {lock_path}') from exc
            yield lock_path
        finally:
            if acquired:
                handle.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _stat_identity(path):
    stat = filesystem_path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _read_snapshot(path, limit):
    before = _stat_identity(path)
    content = _bounded_read(path, limit)
    after = _stat_identity(path)
    if before != after:
        raise ValueError(f'Input changed while being read: {path}')
    return content, {'path': path, 'stat': after, 'sha256': _sha256(content), 'limit': limit}


def _check_snapshot(snapshot):
    try:
        content, current = _read_snapshot(snapshot['path'], snapshot['limit'])
    except (OSError, ValueError) as exc:
        raise ValueError(f'Input changed during ingestion: {snapshot["path"]}') from exc
    if current['stat'] != snapshot['stat'] or current['sha256'] != snapshot['sha256']:
        raise ValueError(f'Input changed during ingestion: {snapshot["path"]}')


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _parser_metadata(document_format):
    version = None
    if document_format == 'pdf':
        try:
            version = importlib.metadata.version('pypdf')
        except importlib.metadata.PackageNotFoundError:
            version = 'unavailable'
    return {'name': 'pypdf' if document_format == 'pdf' else 'python-utf8',
            'version': version if document_format == 'pdf' else sys.version.split()[0],
            'format': document_format, 'ingestion_version': INGESTION_VERSION}


def _cache_key(entry, document_format, local):
    # Every operator-supplied metadata field participates, including the expected
    # digest and publication date. A changed entry cannot pick up its old index.
    value = {'entry': entry, 'format': document_format,
             'local_path': str(local) if local is not None else None}
    return _sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=True, allow_nan=False).encode('ascii'))


def _safe_cache_path(path, protected):
    if any(_same_file(path, source) for source in protected):
        raise ValueError('Cache path collides with an input or output path')


def _cached_source(cache, key, document_format, entry, protected):
    index_path = filesystem_path(cache / 'entries' / (key + '.json'))
    _safe_cache_path(index_path, protected)
    if not index_path.exists():
        return None
    raw = _bounded_read(index_path, 65536)
    index = json.loads(raw.decode('utf-8'), parse_constant=_reject_constant,
                       parse_float=_reject_constant, object_pairs_hook=_object)
    if not isinstance(index, dict) or index.get('schema_version') != 1 or index.get('entry_key') != key:
        raise ValueError('Invalid source cache index')
    digest = index.get('source_sha256')
    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise ValueError('Invalid source cache SHA256')
    if entry.get('source_sha256', digest).lower() != digest:
        raise ValueError('Cached SHA256 does not match expected source_sha256')
    if index.get('source_url') != entry['source_url'] or index.get('format') != document_format:
        raise ValueError('Source cache metadata does not match manifest entry')
    _url_parts(index.get('final_source_url', ''))
    source_path = filesystem_path(cache / (digest + '.' + document_format))
    _safe_cache_path(source_path, protected)
    content, snapshot = _read_snapshot(source_path, MAX_SOURCE_BYTES)
    if _sha256(content) != digest:
        raise ValueError('Existing cache file failed SHA256 verification')
    return content, source_path, index, snapshot


def verify_ingestion_output(output_path) -> dict:
    """Require a committed corpus/report pair whose hashes match its marker."""
    output = filesystem_path(output_path)
    report_path = output.with_name(output.name + '.manifest.json')
    marker_path = output.with_name(output.name + '.complete.json')
    marker = json.loads(_bounded_read(marker_path, 65536).decode('utf-8'),
                        parse_constant=_reject_constant, object_pairs_hook=_object)
    if not isinstance(marker, dict) or marker.get('schema_version') != 1 or marker.get('state') != 'complete':
        raise ValueError('Ingestion output transaction is not complete')
    expected = marker.get('files')
    if not isinstance(expected, dict) or set(expected) != {output.name, report_path.name}:
        raise ValueError('Ingestion completion marker has an invalid file set')
    if any(not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)
           for digest in expected.values()):
        raise ValueError('Ingestion completion marker has invalid SHA256 values')
    count = marker.get('documents_written')
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('Ingestion completion marker has an invalid document count')
    corpus_hash = hashlib.sha256()
    with output.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            corpus_hash.update(chunk)
    report_bytes = _bounded_read(report_path, 64 * 1024 * 1024)
    for path, digest in ((output, corpus_hash.hexdigest()), (report_path, _sha256(report_bytes))):
        if digest != expected[path.name]:
            raise ValueError(f'Ingestion output SHA256 mismatch: {path.name}')
    report = json.loads(report_bytes.decode('utf-8'), parse_constant=_reject_constant,
                        object_pairs_hook=_object)
    if (not isinstance(report, dict) or report.get('output_sha256') != corpus_hash.hexdigest()
            or report.get('run_id') != marker.get('run_id') or report.get('documents_written') != count):
        raise ValueError('Ingestion report and completion marker disagree')
    return report


def ingest_manifest(manifest_path, output_path, *, download=False, cache_dir=None,
                    refresh=False, max_attempts=MAX_DOWNLOAD_ATTEMPTS) -> dict:
    """Write documents plus ``<output>.manifest.json``; report every rejected entry.

    A rejected entry does not become evidence. ``partial`` or ``failed`` status
    requires caller attention even when the JSONL contains usable other sources.
    """
    if not isinstance(download, bool):
        raise ValueError('download must be a boolean')
    if not isinstance(refresh, bool) or (refresh and not download):
        raise ValueError('refresh must be boolean and requires download=True')
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 5:
        raise ValueError('max_attempts must be an integer between 1 and 5')
    manifest_path = filesystem_path(manifest_path)
    output_path = filesystem_path(output_path)
    report_path = output_path.with_name(output_path.name + '.manifest.json')
    marker_path = output_path.with_name(output_path.name + '.complete.json')
    lock_path = output_path.with_name(output_path.name + '.lock')
    cache = filesystem_path(cache_dir) if cache_dir is not None else output_path.parent / 'source-cache'
    raw, manifest_snapshot = _read_snapshot(manifest_path, MAX_MANIFEST_BYTES)
    entries = json.loads(raw.decode('utf-8', errors='strict'),
        parse_constant=_reject_constant, parse_float=_reject_constant, object_pairs_hook=_object)
    if not isinstance(entries, list) or not entries:
        raise ValueError('Manifest must be a nonempty JSON list')
    protected = [manifest_path]
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get('local_path'), str):
            protected.append((manifest_path.parent / entry['local_path']).resolve())
    targets = [output_path, report_path, marker_path, lock_path]
    for target in targets:
        if any(_same_file(target, source) for source in protected):
            raise ValueError('Output paths must not overwrite the manifest or source files')
        if any(_same_file(target, other) for other in targets if target != other):
            raise ValueError('Output paths must not alias each other')
    with output_lock(output_path):
        _check_snapshot(manifest_snapshot)
        return _ingest_entries(entries, manifest_path, manifest_snapshot, output_path,
                               report_path, marker_path, cache, protected + targets,
                               download, refresh, max_attempts)


def _ingest_entries(entries, manifest_path, manifest_snapshot, output_path,
                    report_path, marker_path, cache, protected, download,
                    refresh, max_attempts):
    _, code_snapshot = _read_snapshot(Path(__file__).resolve(), 5 * 1024 * 1024)
    snapshots = [manifest_snapshot, code_snapshot]
    # Detect an external writer that ignores our lock before committing results.
    output_state = {path: _stat_identity(path) if path.exists() else None
                    for path in (output_path, report_path, marker_path)}
    documents, accepted, rejected = [], [], []
    seen_ids = set()
    for index, entry in enumerate(entries):
        extraction_warnings = []
        request_audit = []
        try:
            local, document_format = _entry(entry, manifest_path.parent)
            final_url = entry['source_url']
            downloaded = False
            cache_path = None
            cached = None
            fetched_at = None
            acquisition = 'local'
            key = _cache_key(entry, document_format, local)
            use_local = local is not None and local.exists()
            if use_local:
                content, snapshot = _read_snapshot(local, MAX_SOURCE_BYTES)
                snapshots.append(snapshot)
            else:
                if not refresh:
                    cached = _cached_source(cache, key, document_format, entry, protected)
                if cached is not None:
                    content, cache_path, index_record, snapshot = cached
                    snapshots.append(snapshot)
                    final_url = index_record['final_source_url']
                    fetched_at = index_record.get('fetched_at')
                    acquisition = 'verified_cache'
                elif not download:
                    if local is not None:
                        raise FileNotFoundError(f'Local source and verified cache unavailable: {local}; URL download requires --download')
                    raise ValueError('No verified cache or local_path provided; URL download requires --download')
            if not use_local and cached is None:
                if local is not None and 'source_sha256' not in entry:
                    raise ValueError('Missing local file: download fallback requires expected source_sha256')
                content, final_url = _download(entry['source_url'], audit=request_audit,
                                               max_attempts=max_attempts)
                downloaded = True
                acquisition = 'https_download'
                fetched_at = datetime.now(timezone.utc).isoformat()
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
            if downloaded:
                cache_path = filesystem_path(cache / (source_hash + '.' + document_format))
                _safe_cache_path(cache_path, protected)
                if cache_path.exists():
                    if _sha256(_bounded_read(cache_path, MAX_SOURCE_BYTES)) != source_hash:
                        raise ValueError('Existing cache file failed SHA256 verification')
                else:
                    _atomic_write(cache_path, content)
                index_path = filesystem_path(cache / 'entries' / (key + '.json'))
                _safe_cache_path(index_path, protected)
                _atomic_write(index_path, _json_bytes({'schema_version': 1,
                    'entry_key': key, 'source_sha256': source_hash, 'format': document_format,
                    'source_url': entry['source_url'], 'final_source_url': final_url,
                    'fetched_at': fetched_at}))
            documents.extend(imported)
            seen_ids.update(record['document_id'] for record in imported)
            accepted.append({'entry_index': index, 'ticker': entry['ticker'],
                'source_url': entry['source_url'], 'final_source_url': final_url,
                'source_sha256': source_hash, 'bytes': len(content),
                'pages_imported': len(imported), 'pages_total': len(pages),
                'pages_without_text': [page for page, text in pages if not text.strip()],
                'extraction_warnings': extraction_warnings,
                'acquisition': acquisition, 'fetched_at': fetched_at,
                'parser': _parser_metadata(document_format), 'requests': request_audit,
                'cache_path': str(cache_path) if cache_path else None})
        except Exception as exc:
            rejected.append({'entry_index': index,
                'ticker': entry.get('ticker') if isinstance(entry, dict) else None,
                'error': f'{type(exc).__name__}: {exc}', 'extraction_warnings': extraction_warnings,
                'requests': request_audit})
    corpus = ''.join(json.dumps(document, ensure_ascii=False, allow_nan=False) + '\n'
                     for document in documents).encode('utf-8')
    code_hash = code_snapshot['sha256']
    run_id = _sha256((manifest_snapshot['sha256'] + _sha256(corpus) + code_hash).encode('ascii'))
    report = {'schema_version': 2, 'run_id': run_id,
        'status': 'completed' if not rejected else ('partial' if accepted else 'failed'),
        'input_sha256': manifest_snapshot['sha256'], 'output_sha256': _sha256(corpus),
        'ingestion_version': INGESTION_VERSION, 'ingestion_code_sha256': code_hash,
        'python_version': sys.version.split()[0], 'max_download_attempts': max_attempts,
        'download_enabled': download, 'source_size_limit_bytes': MAX_SOURCE_BYTES,
        'download_timeout_seconds': DOWNLOAD_TIMEOUT, 'entries_total': len(entries),
        'entries_accepted': len(accepted), 'entries_rejected': len(rejected),
        'documents_written': len(documents), 'accepted': accepted, 'rejected': rejected,
        'limitations': ['Company identity, source attribution, and available_at are supplied by the operator.',
            'Text extraction does not establish the factual accuracy of a source.',
            'Image-only PDF pages need separate OCR and are not imported as evidence.',
            'Verified cached bytes are a frozen source snapshot; use refresh=True with download=True to check for changed content.']}
    for snapshot in snapshots:
        _check_snapshot(snapshot)
    for path, before in output_state.items():
        after = _stat_identity(path) if path.exists() else None
        if before != after:
            raise ValueError(f'Output changed during ingestion: {path}')
    report_bytes = _json_bytes(report)
    marker = {'schema_version': 1, 'state': 'writing', 'run_id': run_id,
              'documents_written': len(documents), 'files': {
                  output_path.name: _sha256(corpus), report_path.name: _sha256(report_bytes)}}
    # A failed multi-file commit stays visibly incomplete; it cannot retain a
    # previous success marker for a mixture of old and new files.
    _atomic_write(marker_path, _json_bytes(marker))
    _atomic_write(output_path, corpus)
    _atomic_write(report_path, report_bytes)
    marker['state'] = 'complete'
    _atomic_write(marker_path, _json_bytes(marker))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--download', action='store_true', help='Allow explicit manifest HTTPS downloads')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--refresh', action='store_true', help='Re-fetch a remote source; requires --download')
    parser.add_argument('--max-attempts', type=int, default=MAX_DOWNLOAD_ATTEMPTS)
    args = parser.parse_args(argv)
    try:
        report = ingest_manifest(args.manifest, args.output, download=args.download, cache_dir=args.cache_dir,
                                 refresh=args.refresh, max_attempts=args.max_attempts)
    except (OSError, ValueError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
