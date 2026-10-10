"""Run one offline research batch against preserved input snapshots.

This program neither schedules itself nor grants deployment approval. Each
invocation preserves a new audit directory; it never reuses or deletes old runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import uuid

import discover
from engine import iso
from ingest_theme import filesystem_path, output_lock, verify_ingestion_output


ROOT = Path(__file__).resolve().parent
MAX_CONFIG_BYTES = 1024 * 1024
MAX_QUERIES = 100
CONFIG_FIELDS = {'schema_version', 'as_of', 'companies', 'documents', 'statements',
                 'valuations', 'production_policy', 'queries', 'output_root',
                 'theme_config', 'financial_config', 'limit'}
REQUIRED_FIELDS = {'schema_version', 'as_of', 'companies', 'documents',
                   'production_policy', 'queries', 'output_root'}
PATH_FIELDS = ('companies', 'documents', 'statements', 'valuations',
               'production_policy', 'theme_config', 'financial_config')
RESERVED_NAMES = {'CON', 'PRN', 'AUX', 'NUL', *('COM' + str(i) for i in range(1, 10)),
                  *('LPT' + str(i) for i in range(1, 10))}
RUN_PATTERN = r'\d{8}T\d{12}Z-[0-9a-f]{32}'


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash_bytes(content):
    return hashlib.sha256(content).hexdigest()


def _hash_file(path):
    digest = hashlib.sha256()
    with filesystem_path(path).open('rb') as handle:
        for content in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(content)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def _atomic_json(path, value):
    path = filesystem_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json(path, *, limit=MAX_CONFIG_BYTES):
    with filesystem_path(path).open('rb') as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f'JSON file exceeds {limit} bytes: {path}')
    return discover.strict_json(raw.decode('utf-8-sig')), raw


def _stat(path):
    value = filesystem_path(path).stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _path(value, base, label):
    if (not isinstance(value, str) or not value.strip() or value != value.strip()
            or any(ord(char) < 32 for char in value)):
        raise ValueError(f'{label} must be a nonempty path string without padding or control characters')
    return filesystem_path(base / value)


def _load_config(config_path):
    config_path = filesystem_path(config_path)
    config, raw = _read_json(config_path)
    if not isinstance(config, dict) or set(config) - CONFIG_FIELDS or REQUIRED_FIELDS - set(config):
        raise ValueError('Job config has missing or unknown fields; required: ' + ', '.join(sorted(REQUIRED_FIELDS)))
    if type(config['schema_version']) is not int or config['schema_version'] != 1:
        raise ValueError('schema_version must be 1')
    iso(config['as_of'])
    limit = config.get('limit', 20)
    if type(limit) is not int or not 1 <= limit <= 10000:
        raise ValueError('limit must be an integer between 1 and 10000')
    queries = config['queries']
    if not isinstance(queries, list) or not 1 <= len(queries) <= MAX_QUERIES:
        raise ValueError(f'queries must contain between 1 and {MAX_QUERIES} entries')
    names = set()
    for query in queries:
        if (not isinstance(query, dict) or set(query) - {'name', 'query', 'market'}
                or {'name', 'query'} - set(query)):
            raise ValueError('Each query permits name, query, and optional market only')
        name = query['name']
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name)
                or name.upper() in RESERVED_NAMES or name.lower() in names):
            raise ValueError('Query names must be unique portable directory names, up to 64 ASCII characters')
        names.add(name.lower())
        text = query['query']
        if (not isinstance(text, str) or not text.strip() or len(text) > 512
                or any(ord(char) < 32 for char in text)):
            raise ValueError('Query text must contain 1 to 512 characters without controls')
        if query.get('market') not in (None, 'A', 'HK'):
            raise ValueError('Query market must be A or HK')
    if 'valuations' in config and 'statements' not in config:
        raise ValueError('valuations requires statements')
    paths = {field: _path(config[field], config_path.parent, field)
             for field in PATH_FIELDS if field in config}
    paths = {'job_config': config_path, **paths}
    documents = paths['documents']
    paths['ingestion_audit'] = documents.with_name(documents.name + '.manifest.json')
    paths['ingestion_completion'] = documents.with_name(documents.name + '.complete.json')
    output = _path(config['output_root'], config_path.parent, 'output_root')
    protected = [*paths.values(), *(filesystem_path(ROOT / name) for name in discover.CODE_FILES), filesystem_path(__file__)]
    lock = output.with_name(output.name + '.lock')
    for source in protected:
        if source == output or source.is_relative_to(output):
            raise ValueError('output_root must not contain or overwrite an input or program file')
        if source == lock or (lock.exists() and source.exists() and source.samefile(lock)):
            raise ValueError('Job lock must not overwrite an input or program file')
    if output.exists() and not output.is_dir():
        raise ValueError('output_root must be a directory')
    return config, paths, output, raw


def _snapshot_file(source, target, run_dir):
    source, target, run_dir = filesystem_path(source), filesystem_path(target), filesystem_path(run_dir)
    before = _stat(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open('rb') as incoming, target.open('xb') as outgoing:
        for content in iter(lambda: incoming.read(1024 * 1024), b''):
            digest.update(content)
            size += len(content)
            outgoing.write(content)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if _stat(source) != before:
        raise ValueError(f'Input changed while snapshotting: {source}')
    return {'source_path': str(source), 'snapshot_path': target.relative_to(run_dir).as_posix(),
            'sha256': digest.hexdigest(), 'bytes': size}


def _check_snapshots(run_dir, snapshots):
    for item in snapshots.values():
        if _hash_file(run_dir / item['snapshot_path']) != item['sha256']:
            raise ValueError(f'Input snapshot changed: {item["snapshot_path"]}')


def _check_code(code_hashes):
    for name, digest in code_hashes.items():
        if _hash_file(ROOT / name) != digest:
            raise ValueError(f'Program changed during job: {name}')


def _event(run_dir, event, **fields):
    with (run_dir / 'events.jsonl').open('ab') as handle:
        handle.write((json.dumps({'at': _now(), 'event': event, **fields}, ensure_ascii=False,
                                sort_keys=True, allow_nan=False) + '\n').encode('utf-8'))
        handle.flush()
        os.fsync(handle.fileno())


def _seal(run_dir, job):
    run_dir = filesystem_path(run_dir)
    _atomic_json(run_dir / 'job.json', job)
    files = {}
    for path in sorted(run_dir.rglob('*')):
        if path.is_symlink():
            raise ValueError('Job artifacts must not be symlinks')
        if path.is_file() and path != run_dir / 'completion.json':
            files[path.relative_to(run_dir).as_posix()] = _hash_file(path)
    marker = {'schema_version': 1, 'state': 'complete', 'run_id': job['run_id'],
              'status': job['status'], 'production_approved': False, 'files': files}
    _atomic_json(run_dir / 'completion.json', marker)


def verify_job(run_dir):
    """Validate the complete immutable audit, not merely a successful JSON flag."""
    run_dir = filesystem_path(run_dir)
    marker, _ = _read_json(run_dir / 'completion.json', limit=16 * 1024 * 1024)
    if (not isinstance(marker, dict) or marker.get('state') != 'complete'
            or marker.get('production_approved') is not False):
        raise ValueError('Job audit is incomplete')
    files = marker.get('files')
    if not isinstance(files, dict) or 'job.json' not in files:
        raise ValueError('Job completion file list is invalid')
    actual = set()
    for path in run_dir.rglob('*'):
        if path.is_symlink():
            raise ValueError('Job artifacts must not be symlinks')
        if path.is_file() and path != run_dir / 'completion.json':
            actual.add(path.relative_to(run_dir).as_posix())
    if set(files) != actual:
        raise ValueError('Job artifact file set changed')
    for name, digest in files.items():
        if _hash_file(run_dir / name) != digest:
            raise ValueError(f'Job artifact SHA256 mismatch: {name}')
    job, _ = _read_json(run_dir / 'job.json', limit=16 * 1024 * 1024)
    if (not isinstance(job, dict) or job.get('run_id') != marker.get('run_id') or job.get('status') != marker.get('status')
            or job.get('production_approved') is not False):
        raise ValueError('Job summary and completion disagree')
    return job


def _last_success(output):
    pointer_path = output / 'last_success.json'
    if not pointer_path.exists():
        return None
    pointer, _ = _read_json(pointer_path)
    if not isinstance(pointer, dict):
        raise ValueError('Invalid last-success pointer')
    run_id = pointer.get('run_id')
    if (not isinstance(run_id, str) or not re.fullmatch(RUN_PATTERN, run_id)
            or pointer.get('relative_path') != 'runs/' + run_id or pointer.get('production_approved') is not False):
        raise ValueError('Invalid last-success pointer')
    iso(pointer.get('as_of'))
    directory = output / 'runs' / run_id
    if directory.is_symlink() or _hash_file(directory / 'completion.json') != pointer.get('completion_sha256'):
        raise ValueError('Last-success completion hash mismatch')
    job = verify_job(directory)
    if (job['status'] != 'succeeded' or job.get('as_of') != pointer['as_of'] or not job.get('queries')
            or any(query['status'] != 'succeeded' or query.get('data_gate_status') != 'data_gates_passed'
                   for query in job['queries'])):
        raise ValueError('Last-success target is not a successful data-gated job with matching cutoff')
    return pointer


def _promotion_decision(output, as_of):
    previous = _last_success(output)
    if previous is not None and iso(previous['as_of']) > iso(as_of):
        return {'status': 'not_promoted', 'reason': 'older_as_of',
                'last_success_as_of': previous['as_of'], 'last_success_run_id': previous['run_id']}
    return {'status': 'promoted', 'reason': 'latest_as_of'}


def _success_pointer(run_dir, job):
    return {'schema_version': 1, 'run_id': job['run_id'], 'relative_path': 'runs/' + job['run_id'],
            'as_of': job['as_of'], 'completed_at': job['finished_at'], 'production_approved': False,
            'completion_sha256': _hash_file(run_dir / 'completion.json')}


def run_job(config_path):
    """Execute one bounded batch; return succeeded, blocked, or failed."""
    config, paths, output_root, config_raw = _load_config(config_path)
    with output_lock(output_root):
        output_root.mkdir(parents=True, exist_ok=True)
        for path in (output_root / 'runs', output_root / 'last_success.json'):
            if path.is_symlink():
                raise ValueError('Job control paths must not be symlinks')
        runs = output_root / 'runs'
        runs.mkdir(exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ-') + uuid.uuid4().hex
        run_dir = runs / run_id
        run_dir.mkdir(exist_ok=False)
        began = time.monotonic()
        job = {'schema_version': 1, 'run_id': run_id, 'run_dir': str(run_dir),
               'as_of': config['as_of'], 'status': 'started', 'started_at': _now(),
               'production_approved': False, 'input_snapshots': {}, 'code_sha256': {},
               'queries': [{**query, 'status': 'pending'} for query in config['queries']]}
        _atomic_json(run_dir / 'job.json', job)
        _event(run_dir, 'job_started', run_id=run_id)
        try:
            for name in sorted(set(discover.CODE_FILES) | {Path(__file__).name}):
                item = _snapshot_file(ROOT / name, run_dir / 'code' / name, run_dir)
                job['code_sha256'][name] = item['sha256']
            snapshot_paths = {}
            for role, source in paths.items():
                if role in ('documents', 'ingestion_audit', 'ingestion_completion'):
                    target = run_dir / 'inputs' / 'documents' / source.name
                else:
                    target = run_dir / 'inputs' / role / source.name
                item = _snapshot_file(source, target, run_dir)
                if role == 'job_config' and item['sha256'] != _hash_bytes(config_raw):
                    raise ValueError('Job config changed before input snapshot')
                job['input_snapshots'][role] = item
                snapshot_paths[role] = target
            for role, source in paths.items():
                if _hash_file(source) != job['input_snapshots'][role]['sha256']:
                    raise ValueError(f'Input changed before snapshot set completed: {source}')
            copied_ingestion = verify_ingestion_output(snapshot_paths['documents'])
            if copied_ingestion.get('status') != 'completed':
                raise ValueError('Document ingestion snapshot is not completed')
            _check_code(job['code_sha256'])
            _event(run_dir, 'snapshots_ready', inputs={role: item['sha256']
                   for role, item in job['input_snapshots'].items()})
            _atomic_json(run_dir / 'job.json', job)
            for query in job['queries']:
                query_started = time.monotonic()
                query.update(status='started', started_at=_now())
                _event(run_dir, 'query_started', name=query['name'])
                _atomic_json(run_dir / 'job.json', job)
                try:
                    _check_snapshots(run_dir, job['input_snapshots'])
                    _check_code(job['code_sha256'])
                    destination = run_dir / 'queries' / query['name']
                    report = discover.run_discovery(query=query['query'], as_of=config['as_of'],
                        companies_path=snapshot_paths['companies'], documents_path=snapshot_paths['documents'],
                        statements_path=snapshot_paths.get('statements'), valuations_path=snapshot_paths.get('valuations'),
                        config_path=snapshot_paths.get('theme_config'), financial_config_path=snapshot_paths.get('financial_config'),
                        production_policy_path=snapshot_paths['production_policy'], market=query.get('market'),
                        limit=config.get('limit', 20), output=destination)
                    verification = discover.verify_discovery(destination, check_inputs=True)
                    _check_snapshots(run_dir, job['input_snapshots'])
                    _check_code(job['code_sha256'])
                    gate = report.get('production_readiness', {}).get('status', 'not_assessed')
                    if verification.get('data_gate_status') != gate:
                        raise ValueError('Discovery data gate status differs from verified export')
                    query.update(status='succeeded' if gate == 'data_gates_passed' else 'blocked',
                                 data_gate_status=gate, discovery_status=report.get('status'),
                                 discovery_run_id=verification['run_id'],
                                 reasons=report.get('production_readiness', {}).get('reasons', []),
                                 output_path=destination.relative_to(run_dir).as_posix())
                except Exception as exc:
                    query.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                query.update(finished_at=_now(), duration_seconds=round(time.monotonic() - query_started, 6))
                _event(run_dir, 'query_finished', name=query['name'], status=query['status'],
                       error=query.get('error'), data_gate_status=query.get('data_gate_status'))
                _atomic_json(run_dir / 'job.json', job)
            _check_snapshots(run_dir, job['input_snapshots'])
            _check_code(job['code_sha256'])
            statuses = {query['status'] for query in job['queries']}
            job['status'] = 'failed' if 'failed' in statuses else ('blocked' if 'blocked' in statuses else 'succeeded')
        except Exception as exc:
            job.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            for query in job['queries']:
                if query['status'] in ('pending', 'started'):
                    query.update(status='failed', error='Job input/setup failure prevented completion')
            _event(run_dir, 'job_failed', error=job['error'])
        if job['status'] == 'succeeded':
            try:
                job['promotion'] = _promotion_decision(output_root, job['as_of'])
            except Exception as exc:
                job.update(status='failed', error=f'Success-pointer validation failed: {type(exc).__name__}: {exc}')
        job.setdefault('promotion', {'status': 'not_promoted', 'reason': 'job_not_succeeded'})
        job.update(finished_at=_now(), duration_seconds=round(time.monotonic() - began, 6))
        _event(run_dir, 'job_finished', status=job['status'])
        _seal(run_dir, job)
        verify_job(run_dir)
        if job['status'] == 'succeeded' and job['promotion']['status'] == 'promoted':
            pointer = _success_pointer(run_dir, job)
            try:
                _atomic_json(output_root / 'last_success.json', pointer)
            except Exception as exc:
                job.update(status='failed', error=f'Last-success publication failed: {type(exc).__name__}: {exc}')
                job['promotion'] = {'status': 'not_promoted', 'reason': 'publication_failed'}
                _event(run_dir, 'publication_failed', error=job['error'])
                _seal(run_dir, job)
        return job


def status(output_root):
    """Inspect retained runs without launching work or mutating any output."""
    output = filesystem_path(output_root)
    result = {'output_root': str(output), 'production_approved': False, 'runs': [], 'last_success': None}
    runs = output / 'runs'
    directories = sorted((path for path in runs.iterdir() if path.is_dir()), reverse=True) if runs.exists() else []
    result.update(total_runs=len(directories), truncated=len(directories) > 100)
    for directory in directories[:100]:
        if directory.is_symlink() or not re.fullmatch(RUN_PATTERN, directory.name):
            result['runs'].append({'run_id': directory.name, 'status': 'unverifiable'})
            continue
        try:
            job = verify_job(directory)
            result['runs'].append({key: job.get(key) for key in ('run_id', 'status', 'as_of', 'finished_at', 'promotion')})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result['runs'].append({'run_id': directory.name, 'status': 'incomplete_or_changed', 'error': str(exc)})
    try:
        result['last_success'] = _last_success(output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['last_success_error'] = str(exc)
    latest = result['runs'][0] if result['runs'] else {}
    successful = result['last_success'] or {}
    result.update(latest_attempt_status=latest.get('status'), latest_attempt_as_of=latest.get('as_of'),
                  last_success_as_of=successful.get('as_of'),
                  is_last_success_latest=bool(successful) and latest.get('run_id') == successful.get('run_id'))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=Path)
    group.add_argument('--status', type=Path, metavar='OUTPUT_ROOT')
    args = parser.parse_args(argv)
    try:
        report = status(args.status) if args.status is not None else run_job(args.config)
        print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
        if args.status is not None:
            return 2 if report.get('last_success_error') else 0
        return {'succeeded': 0, 'blocked': 3, 'failed': 2}[report['status']]
    except (OSError, ValueError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc), 'production_approved': False}, ensure_ascii=True))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
