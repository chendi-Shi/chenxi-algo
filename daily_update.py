"""One explicit official-source refresh followed by a preserved research batch.

The fixed date window is refreshed in full. This is not an incremental historical
database, and the program does not install a scheduler or run a background loop.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import time
import uuid

import discover
from engine import iso
from ingest_theme import filesystem_path, ingest_manifest, output_lock, verify_ingestion_output
import research_job as jobs


MARKET_TIMEZONE = timezone(timedelta(hours=8), name='Asia/Shanghai')
FIELDS = {'schema_version', 'start_date', 'end_date', 'lookback_days', 'companies',
          'production_policy', 'queries', 'output_root', 'source_selection', 'statements',
          'valuations', 'theme_config', 'financial_config', 'limit',
          'universe_snapshot', 'disclosure_snapshot', 'document_cache'}
REQUIRED = {'schema_version', 'companies', 'production_policy', 'queries', 'output_root', 'source_selection'}
INPUT_FILES = ('companies', 'production_policy', 'statements', 'valuations', 'theme_config', 'financial_config')
SOURCE_FIELDS = {'universe_sources', 'categories', 'max_pages', 'min_interval_seconds'}


def _clock():
    return datetime.now(timezone.utc)


def _acquire_universe(directory, *, download, sources):
    from universe_sources import acquire_snapshot, replay_snapshot
    result = acquire_snapshot(directory, download=True, sources=tuple(sources)) if download else replay_snapshot(directory)
    if result.get('requested_sources') != list(sources):
        raise ValueError('Official directory snapshot does not match the exact requested sources')
    return result


def _disclosures(securities, start_date, end_date, **kwargs):
    from disclosure_sources import discover_disclosures
    return discover_disclosures(securities, start_date, end_date, **kwargs)


def _configuration(path, as_of, download):
    path = filesystem_path(path)
    cfg, raw = jobs._read_json(path)
    if not isinstance(cfg, dict) or set(cfg) - FIELDS or REQUIRED - set(cfg):
        raise ValueError('Daily config has missing or unknown fields; required: ' + ', '.join(sorted(REQUIRED)))
    if type(cfg['schema_version']) is not int or cfg['schema_version'] != 1:
        raise ValueError('schema_version must be 1')
    if type(download) is not bool:
        raise ValueError('download must be boolean')
    now = _clock()
    if 'lookback_days' in cfg:
        if 'start_date' in cfg or 'end_date' in cfg:
            raise ValueError('lookback_days is mutually exclusive with start_date/end_date')
        if type(cfg['lookback_days']) is not int or cfg['lookback_days'] < 1:
            raise ValueError('lookback_days must be a positive integer')
        end = iso(as_of) if as_of is not None else now.astimezone(MARKET_TIMEZONE).date()
        try:
            start = end - timedelta(days=cfg['lookback_days'])
        except (OverflowError, ValueError) as exc:
            raise ValueError('lookback_days exceeds supported calendar range') from exc
        mode = 'rolling_window'
    else:
        if as_of is not None:
            raise ValueError('--as-of applies only to lookback_days mode')
        if {'start_date', 'end_date'} - set(cfg):
            raise ValueError('Provide start_date and end_date, or lookback_days')
        start, end = iso(cfg['start_date']), iso(cfg['end_date'])
        mode = 'explicit_window'
    if end < start:
        raise ValueError('end_date must not precede start_date')
    if download and end > now.astimezone(MARKET_TIMEZONE).date():
        raise ValueError('Live end_date cannot be after the current Asia/Shanghai date')
    sources = cfg['source_selection']
    if not isinstance(sources, dict) or set(sources) != SOURCE_FIELDS:
        raise ValueError('source_selection must explicitly specify ' + ', '.join(sorted(SOURCE_FIELDS)))
    selected = sources['universe_sources']
    if (not isinstance(selected, list) or not selected or any(value not in ('sse', 'szse', 'hkex') for value in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError('universe_sources must be a unique nonempty list of sse, szse, hkex')
    categories = sources['categories']
    if (not isinstance(categories, list) or not categories or any(not isinstance(value, str) or not value.strip()
            for value in categories) or len(set(categories)) != len(categories)):
        raise ValueError('categories must be a unique nonempty list')
    if any(value not in ('all', 'annual_report', 'announcement') for value in categories) or ('all' in categories and len(categories) != 1):
        raise ValueError('categories must be all, or a subset of annual_report/announcement')
    if type(sources['max_pages']) is not int or not 1 <= sources['max_pages'] <= 100:
        raise ValueError('max_pages must be an integer from 1 to 100')
    interval = sources['min_interval_seconds']
    try:
        valid_interval = type(interval) in (int, float) and math.isfinite(interval) and 0.5 <= interval <= 60
    except OverflowError:
        valid_interval = False
    if not valid_interval:
        raise ValueError('min_interval_seconds must be between 0.5 and 60')
    queries = cfg['queries']
    if not isinstance(queries, list) or not 1 <= len(queries) <= jobs.MAX_QUERIES:
        raise ValueError('queries must contain between 1 and 100 entries')
    names = set()
    for query in queries:
        if not isinstance(query, dict) or set(query) - {'name', 'query', 'market'} or {'name', 'query'} - set(query):
            raise ValueError('Each query permits name, query, and optional market only')
        name = query['name']
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name)
                or name.upper() in jobs.RESERVED_NAMES or name.lower() in names):
            raise ValueError('Query names must be unique portable directory names')
        names.add(name.lower())
        if (not isinstance(query['query'], str) or not query['query'].strip() or len(query['query']) > 512
                or any(ord(char) < 32 for char in query['query']) or query.get('market') not in (None, 'A', 'HK')):
            raise ValueError('Invalid query text or market')
    if type(cfg.get('limit', 20)) is not int or not 1 <= cfg.get('limit', 20) <= 10000:
        raise ValueError('limit must be between 1 and 10000')
    if 'valuations' in cfg and 'statements' not in cfg:
        raise ValueError('valuations requires statements')
    paths = {'daily_config': path, **{key: jobs._path(cfg[key], path.parent, key)
             for key in (*INPUT_FILES, 'universe_snapshot', 'disclosure_snapshot', 'document_cache') if key in cfg}}
    if not download and {'universe_snapshot', 'disclosure_snapshot', 'document_cache'} - paths.keys():
        raise ValueError('Offline runs require universe_snapshot, disclosure_snapshot, and document_cache; explicit --download permits fresh acquisition')
    output = jobs._path(cfg['output_root'], path.parent, 'output_root')
    lock = output.with_name(output.name + '.lock')
    for source in [*paths.values(), filesystem_path(jobs.ROOT)]:
        if source == output or source.is_relative_to(output):
            raise ValueError('output_root must not contain an input or program directory')
        if source == lock or (lock.exists() and source.exists() and source.samefile(lock)):
            raise ValueError('Daily lock must not overwrite an input')
    for role in ('universe_snapshot', 'disclosure_snapshot', 'document_cache'):
        if role in paths and output.is_relative_to(paths[role]):
            raise ValueError('output_root must not be inside a replay input directory')
    window = {'mode': mode, 'start_date': start.isoformat(), 'end_date': end.isoformat(),
              'inclusive_days': (end - start).days + 1, 'timezone': 'Asia/Shanghai',
              'requested_at_utc': now.isoformat(), 'requested_at_local': now.astimezone(MARKET_TIMEZONE).isoformat()}
    return cfg, paths, output, raw, window


def _copy_directory(source, destination, run_dir):
    if source.is_symlink() or not source.is_dir():
        raise ValueError('Replay source must be a directory without symlinks')
    source, destination, run_dir = map(filesystem_path, (source, destination, run_dir))
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Replay source and destination directories must not overlap')
    destination.mkdir(parents=True, exist_ok=False)
    for item in sorted(source.rglob('*')):
        if item.is_symlink():
            raise ValueError('Replay snapshots must not contain symlinks')
        if item.is_file():
            jobs._snapshot_file(item, destination / item.relative_to(source), run_dir)


def _refresh_companies(companies, universe, end_date):
    if not isinstance(companies, list) or not companies:
        raise ValueError('companies must be a nonempty explicitly classified JSON list')
    official = {}
    for row in universe.get('securities', []):
        ticker = row.get('ticker')
        if ticker in official:
            raise ValueError('Official directory contains duplicate security identities')
        official[ticker] = row
    cutoff = iso(end_date)
    updated, selected, failures, seen = [], [], [], set()
    for row in companies:
        if not isinstance(row, dict) or not isinstance(row.get('ticker'), str):
            raise ValueError('Each declared company must have an explicit canonical ticker')
        ticker = row['ticker']
        if ticker != ticker.strip().upper() or ticker in seen:
            raise ValueError('Company tickers must be unique and canonical')
        seen.add(ticker)
        if (row.get('scope') not in ('technology', 'manufacturing')
                or not isinstance(row.get('sector'), str) or not row['sector'].strip()):
            raise ValueError('Company sector and technology/manufacturing scope require prior explicit classification')
        source = official.get(ticker)
        if source is None:
            failures.append({'ticker': ticker, 'reason': 'missing_from_official_directory'})
            continue
        if source.get('market') != row.get('market'):
            failures.append({'ticker': ticker, 'reason': 'official_market_mismatch'})
            continue
        if source.get('security_type') != 'equity':
            failures.append({'ticker': ticker, 'reason': 'security_is_not_equity'})
            continue
        observed = iso(source['universe_as_of'])
        existing = iso(row['universe_as_of'])
        effective = iso(source['source_effective_at']) if source.get('source_effective_at') else observed
        if observed > cutoff or effective > cutoff or existing > cutoff:
            failures.append({'ticker': ticker, 'reason': 'directory_not_available_at_cutoff'})
            continue
        if observed < existing:
            failures.append({'ticker': ticker, 'reason': 'official_snapshot_predates_existing_membership'})
            continue
        # Keep the previously reviewed issuer name as evidence for matching old
        # disclosures when an official directory changes the display name.
        old_name = row.get('name')
        aliases = row.get('aliases', [])
        if (not isinstance(old_name, str) or not old_name.strip() or not isinstance(aliases, list)
                or any(not isinstance(alias, str) or not alias.strip() for alias in aliases)):
            raise ValueError('Company name and aliases must contain explicit nonempty strings')
        aliases = list(aliases)
        official_name = source.get('name') or old_name
        if official_name != old_name and old_name.casefold() not in {alias.casefold() for alias in aliases}:
            aliases.append(old_name)
        # Dates come from the verified official snapshot, never from the job clock.
        refreshed = dict(row, name=official_name, universe_as_of=source['universe_as_of'])
        if aliases or 'aliases' in row:
            refreshed['aliases'] = aliases
        updated.append(refreshed)
        selected.append(source)
    return updated, selected, failures


class _Blocked(ValueError):
    pass


def run_daily(config_path, *, download=False, as_of=None):
    cfg, paths, output_root, config_raw, window = _configuration(config_path, as_of, download)
    with output_lock(output_root):
        output_root.mkdir(parents=True, exist_ok=True)
        for target in (output_root / 'runs', output_root / 'last_success.json'):
            if target.is_symlink():
                raise ValueError('Daily output control paths must not be symlinks')
        (output_root / 'runs').mkdir(exist_ok=True)
        run_id = _clock().strftime('%Y%m%dT%H%M%S%fZ-') + uuid.uuid4().hex
        run_dir = output_root / 'runs' / run_id
        run_dir.mkdir(exist_ok=False)
        began = time.monotonic()
        job = {'schema_version': 1, 'kind': 'daily_refresh', 'run_id': run_id, 'run_dir': str(run_dir),
               'as_of': window['end_date'], 'window': window, 'download_enabled': download,
               'status': 'started', 'started_at': jobs._now(), 'production_approved': False,
               'input_snapshots': {}, 'code_sha256': {}, 'stages': [], 'queries': [],
               'limitations': ['The complete fixed window is re-fetched; this is not a complete historical or incremental database.',
                   'Financial CSVs are supplied snapshots; this refresh does not manufacture missing financial data.',
                   'Official directory membership does not establish industry classification or investment suitability.']}
        jobs._atomic_json(run_dir / 'job.json', job)
        jobs._event(run_dir, 'daily_started', window=window)
        def stage(name, operation):
            record = {'name': name, 'status': 'started', 'started_at': jobs._now()}
            job['stages'].append(record)
            stage_started = time.monotonic()
            jobs._event(run_dir, 'stage_started', name=name)
            jobs._atomic_json(run_dir / 'job.json', job)
            try:
                value = operation()
                record['status'] = 'succeeded'
                return value
            except Exception as exc:
                record.update(status='blocked' if isinstance(exc, _Blocked) else 'failed',
                              error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                record.update(finished_at=jobs._now(), duration_seconds=round(time.monotonic() - stage_started, 6))
                jobs._event(run_dir, 'stage_finished', name=name, status=record['status'], error=record.get('error'))
                jobs._atomic_json(run_dir / 'job.json', job)
        try:
            snapshot_paths = {}
            def snapshots():
                for name in sorted(set(discover.CODE_FILES) | {'research_job.py', 'daily_update.py', 'universe_sources.py', 'disclosure_sources.py'}):
                    item = jobs._snapshot_file(jobs.ROOT / name, run_dir / 'code' / name, run_dir)
                    job['code_sha256'][name] = item['sha256']
                for role in ('daily_config', *INPUT_FILES):
                    if role not in paths:
                        continue
                    target = run_dir / 'inputs' / role / paths[role].name
                    item = jobs._snapshot_file(paths[role], target, run_dir)
                    if role == 'daily_config' and item['sha256'] != jobs._hash_bytes(config_raw):
                        raise ValueError('Daily config changed before snapshot')
                    job['input_snapshots'][role] = item
                    snapshot_paths[role] = target
                for role, item in job['input_snapshots'].items():
                    if jobs._hash_file(paths[role]) != item['sha256']:
                        raise ValueError('Input changed before snapshot set completed')
                jobs._check_code(job['code_sha256'])
            stage('input_snapshots', snapshots)
            universe_dir = run_dir / 'source_snapshots' / 'universe'
            disclosure_dir = run_dir / 'source_snapshots' / 'disclosures'
            def universe_step():
                if not download:
                    _copy_directory(paths['universe_snapshot'], universe_dir, run_dir)
                result = _acquire_universe(universe_dir, download=download,
                                            sources=cfg['source_selection']['universe_sources'])
                jobs._atomic_json(run_dir / 'universe_report.json', result)
                if result.get('status') == 'blocked_future_effective_date':
                    raise _Blocked('Official directory has a future effective date')
                if result.get('status') != 'completed':
                    raise ValueError('Official directory acquisition was not complete')
                return result
            universe = stage('official_directory', universe_step)
            companies, _ = jobs._read_json(snapshot_paths['companies'], limit=32 * 1024 * 1024)
            def membership():
                updated, securities, failures = _refresh_companies(companies, universe, window['end_date'])
                jobs._atomic_json(run_dir / 'membership.json', {'companies_total': len(companies),
                    'companies_verified': len(updated), 'failures': failures})
                if failures:
                    raise _Blocked('Declared companies are missing or not yet available in the official directory')
                jobs._atomic_json(run_dir / 'refreshed_companies.json', updated)
                return securities
            selected = stage('company_membership', membership)
            def disclosure_step():
                if not download:
                    _copy_directory(paths['disclosure_snapshot'], disclosure_dir, run_dir)
                result = _disclosures(selected, window['start_date'], window['end_date'], download=download,
                    snapshot_dir=disclosure_dir, categories=tuple(cfg['source_selection']['categories']),
                    max_pages=cfg['source_selection']['max_pages'],
                    min_interval_seconds=cfg['source_selection']['min_interval_seconds'])
                jobs._atomic_json(run_dir / 'disclosure_report.json', result)
                if result.get('status') != 'completed':
                    raise ValueError('Disclosure discovery was not complete')
                entries = result.get('ingest_manifest')
                if not isinstance(entries, list) or not entries:
                    raise _Blocked('No importable disclosures in the selected fixed window')
                for entry in entries:
                    if (entry.get('ticker') not in {row['ticker'] for row in selected}
                            or not window['start_date'] <= entry.get('available_at', '') <= window['end_date']):
                        raise ValueError('Disclosure manifest falls outside the requested company/date scope')
                jobs._atomic_json(run_dir / 'ingest_manifest.json', entries)
                return entries
            stage('disclosure_discovery', disclosure_step)
            corpus = run_dir / 'corpus.jsonl'
            def ingestion_step():
                if not download:
                    _copy_directory(paths['document_cache'], run_dir / 'source_cache', run_dir)
                result = ingest_manifest(run_dir / 'ingest_manifest.json', corpus, download=download,
                                         cache_dir=run_dir / 'source_cache')
                verify_ingestion_output(corpus)
                if result['status'] != 'completed':
                    raise ValueError('Document ingestion contains rejected sources')
                return result
            stage('document_ingestion', ingestion_step)
            def research_step():
                batch = {'schema_version': 1, 'as_of': window['end_date'],
                    'companies': str(run_dir / 'refreshed_companies.json'), 'documents': str(corpus),
                    'production_policy': str(snapshot_paths['production_policy']),
                    'queries': cfg['queries'], 'limit': cfg.get('limit', 20),
                    'output_root': str(run_dir / 'research')}
                for role in ('statements', 'valuations', 'theme_config', 'financial_config'):
                    if role in snapshot_paths:
                        batch[role] = str(snapshot_paths[role])
                jobs._atomic_json(run_dir / 'research_config.json', batch)
                result = jobs.run_job(run_dir / 'research_config.json')
                job['research_run_id'] = result['run_id']
                job['queries'] = result['queries']
                if result['status'] == 'blocked':
                    raise _Blocked('Research data gates were not passed')
                if result['status'] != 'succeeded':
                    details = result.get('error') or '; '.join(query.get('error', '')
                        for query in result.get('queries', []) if query.get('status') == 'failed')
                    raise ValueError('Research batch failed: ' + details[:2000])
                return result
            stage('research_batch', research_step)
            jobs._check_snapshots(run_dir, job['input_snapshots'])
            jobs._check_code(job['code_sha256'])
            job['status'] = 'succeeded'
        except Exception as exc:
            job.update(status='blocked' if isinstance(exc, _Blocked) else 'failed',
                       error=f'{type(exc).__name__}: {exc}')
        if job['status'] == 'succeeded':
            try:
                job['promotion'] = jobs._promotion_decision(output_root, job['as_of'])
            except Exception as exc:
                job.update(status='failed', error=f'Success-pointer validation failed: {type(exc).__name__}: {exc}')
        job.setdefault('promotion', {'status': 'not_promoted', 'reason': 'job_not_succeeded'})
        job.update(finished_at=jobs._now(), duration_seconds=round(time.monotonic() - began, 6))
        jobs._event(run_dir, 'daily_finished', status=job['status'], error=job.get('error'))
        jobs._seal(run_dir, job)
        jobs.verify_job(run_dir)
        if job['status'] == 'succeeded' and job['promotion']['status'] == 'promoted':
            pointer = jobs._success_pointer(run_dir, job)
            try:
                jobs._atomic_json(output_root / 'last_success.json', pointer)
            except Exception as exc:
                job.update(status='failed', error=f'Last-success publication failed: {type(exc).__name__}: {exc}')
                job['promotion'] = {'status': 'not_promoted', 'reason': 'publication_failed'}
                jobs._event(run_dir, 'publication_failed', error=job['error'])
                jobs._seal(run_dir, job)
        return job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=Path)
    group.add_argument('--status', type=Path, metavar='OUTPUT_ROOT')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--as-of', help='Explicit Asia/Shanghai cutoff in lookback_days mode')
    args = parser.parse_args(argv)
    try:
        if args.status is not None:
            if args.download or args.as_of:
                raise ValueError('--status cannot be combined with --download or --as-of')
            result = jobs.status(args.status)
            print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
            return 2 if result.get('last_success_error') else 0
        result = run_daily(args.config, download=args.download, as_of=args.as_of)
        print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
        return {'succeeded': 0, 'blocked': 3, 'failed': 2}[result['status']]
    except (OSError, ValueError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc), 'production_approved': False}, ensure_ascii=True))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
