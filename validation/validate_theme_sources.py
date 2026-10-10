"""Validate the small public theme corpus; network access is never implicit.

Default: schema, identity, dates, source metadata and manifest consistency only.
--verify-sources: additionally hash LOCAL cached originals and find each short
quote on its specified physical PDF page or in parsed HTML. It does not fetch.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_HOSTS = {
    'www.zj-innolight.com', 'static.cninfo.com.cn', 'www.accelink.com',
    'www.ubtrobot.com', 'lenovopress.lenovo.com',
}


class VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {'script', 'style'} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def normalize(text):
    """Allow only Unicode compatibility and layout whitespace differences."""
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def iso_date(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError(f'not an ISO date: {value!r}')
    return date.fromisoformat(value)


def validate(root=ROOT, verify_sources=False):
    root = Path(root).resolve()
    data = root / 'data'
    companies = json.loads((data / 'theme_companies.json').read_text(encoding='utf-8'))
    documents = [json.loads(line) for line in (data / 'theme_documents.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    metadata = json.loads((data / 'theme_sources.json').read_text(encoding='utf-8'))
    manifest = json.loads((data / 'theme_ingest_manifest.json').read_text(encoding='utf-8'))
    errors = []
    checks = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    require(isinstance(companies, list) and bool(companies), 'companies must be a nonempty list')
    require(isinstance(manifest, list), 'manifest must be a list')
    cutoff = iso_date(metadata['as_of'])
    company_map = {c['ticker']: c for c in companies}
    source_map = {s['source_id']: s for s in metadata['sources']}
    doc_map = {d['document_id']: d for d in documents}
    require(len(company_map) == len(companies), 'duplicate company ticker')
    require(len(source_map) == len(metadata['sources']), 'duplicate source id')
    require(len(doc_map) == len(documents), 'duplicate document id')
    require(len({m['document_id'] for m in manifest}) == len(manifest), 'duplicate manifest document id')
    for c in companies:
        tag = c['ticker']
        require(c.get('market') in {'A', 'HK'}, f'{tag}: unsupported market')
        require(c.get('scope') in {'technology', 'manufacturing'}, f'{tag}: unsupported scope')
        require(bool(c.get('name')) and bool(c.get('sector')), f'{tag}: missing identity')
        require(iso_date(c['universe_as_of']) == cutoff, f'{tag}: snapshot mismatch')
        require(bool(re.fullmatch(r'\d{5}\.HK' if c['market'] == 'HK' else r'\d{6}\.(SZ|SH)', tag)), f'{tag}: ticker/market mismatch')
    for s in source_map.values():
        sid = s['source_id']
        parsed = urlparse(s['source_url'])
        require(parsed.scheme == 'https' and parsed.hostname in OFFICIAL_HOSTS, f'{sid}: unapproved official source URL')
        require(bool(re.fullmatch('[0-9a-f]{64}', s['source_sha256'])), f'{sid}: invalid SHA256')
        require(s['ticker'] in company_map, f'{sid}: unknown ticker')
        require(s['date_basis'] in {'official_release', 'observed_at'}, f'{sid}: unsupported date basis')
        require(iso_date(s['available_at']) <= cutoff, f'{sid}: future source')
        require(iso_date(s['observed_at']) == cutoff, f'{sid}: observed cutoff mismatch')
        if s['date_basis'] == 'observed_at':
            require(s['available_at'] == s['observed_at'], f'{sid}: observed source was backdated')
        elif parsed.hostname == 'static.cninfo.com.cn':
            require('/' + s['available_at'] + '/' in parsed.path, f'{sid}: release-date/URL mismatch')
        require(s['summary_basis'] == 'analyst_paraphrase', f'{sid}: summary must be distinguished from original quote')
        require(s['format'] in {'pdf', 'html'}, f'{sid}: unsupported format')
        require(set(s['document_ids']) <= set(doc_map), f'{sid}: missing referenced document')
    for d in documents:
        did = d['document_id']
        s = source_map[d['source_id']]
        c = company_map[d['ticker']]
        for key in ('ticker', 'available_at', 'source_url', 'source_type', 'title', 'source_sha256', 'date_basis'):
            require(d[key] == s[key], f'{did}: source {key} mismatch')
        require(did in s['document_ids'], f'{did}: missing reverse source reference')
        require(d['scope'] == c['scope'] and d['universe_as_of'] == c['universe_as_of'], f'{did}: company scope/snapshot mismatch')
        require(hashlib.sha256(d['text'].encode('utf-8')).hexdigest() == s['quote_sha256'] and d['page'] == s['quote_page'], f'{did}: quote/page mismatch')
        require(bool(d['text'].strip()), f'{did}: empty quote')
        if s['format'] == 'pdf':
            require(type(d['page']) is int and d['page'] > 0, f'{did}: PDF physical page must be a positive integer')
        else:
            require(isinstance(d['page'], str) and bool(d['page'].strip()), f'{did}: HTML section missing')
        if not verify_sources:
            continue
        cache = (root / s['cache_path']).resolve()
        cache_root = (root / 'validation/raw/theme').resolve()
        require(cache.is_relative_to(cache_root), f'{did}: cache outside allowed source directory')
        if not cache.is_relative_to(cache_root):
            continue
        if not cache.is_file():
            errors.append(f'{did}: original not cached; source was not reverified')
            continue
        body = cache.read_bytes()
        actual_hash = hashlib.sha256(body).hexdigest()
        require(actual_hash == s['source_sha256'], f'{did}: cached source hash mismatch')
        if actual_hash != s['source_sha256']:
            continue
        if s['format'] == 'pdf':
            from pypdf import PdfReader
            reader = PdfReader(cache)
            require(d['page'] <= len(reader.pages), f'{did}: page outside PDF')
            if d['page'] > len(reader.pages):
                continue
            original = reader.pages[d['page'] - 1].extract_text() or ''
        else:
            parser = VisibleHTML()
            parser.feed(body.decode('utf-8'))
            original = ' '.join(parser.parts)
            require(normalize(d['page']) in normalize(original), f'{did}: HTML section missing from source')
        found = normalize(d['text']) in normalize(original)
        require(found, f'{did}: exact quote not found on the specified page/HTML')
        checks.append(dict(document_id=did,sha256_match=True,quote_found=found,page=d['page']))
    for m in manifest:
        d = doc_map[m['document_id']]
        s = source_map[d['source_id']]
        for key in ('ticker', 'available_at', 'source_url', 'source_type', 'title', 'source_sha256', 'date_basis'):
            require(m[key] == d[key], f'{m["document_id"]}: manifest {key} mismatch')
        require(m['format'] == s['format'] == 'pdf', f'{m["document_id"]}: manifest format mismatch')
        require((data / m['local_path']).resolve() == (root / s['cache_path']).resolve(), f'{m["document_id"]}: manifest path mismatch')
    expected_pdfs = {d['document_id'] for d in documents if source_map[d['source_id']]['format'] == 'pdf'}
    require({m['document_id'] for m in manifest} == expected_pdfs, 'full PDF manifest coverage mismatch')
    for case in metadata['evaluation_cases']:
        sets = [set(case[k]) for k in ('direct_tickers', 'development_tickers', 'non_direct_tickers')]
        require(all(x <= company_map.keys() for x in sets), f'{case["query"]}: case contains unknown ticker')
        require(not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), f'{case["query"]}: labels overlap')
        require(set.union(*sets) == company_map.keys(), f'{case["query"]}: case does not cover selected corpus')
    return dict(status='passed' if not errors else 'failed',as_of=metadata['as_of'],companies=len(companies),documents=len(documents),sources=len(source_map),full_pdf_sources=len(manifest),mode='cached_originals' if verify_sources else 'offline_metadata_only',original_sources_reverified=verify_sources and len(checks)==len(documents) and not errors,source_checks=checks,errors=errors,limitations=['Small curated smoke-test corpus; not market-wide recall, precision or investment validation.','Offline metadata mode never claims original-source verification.','Cached-source verification does not establish current website contents or undisclosed later business changes.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-sources', action='store_true', help='Verify local original hashes and exact quote pages; no network access.')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, help='Also write the validation result as UTF-8 JSON.')
    args = parser.parse_args()
    try:
        result = validate(args.root, args.verify_sources)
    except (KeyError, TypeError, ValueError, OSError, ImportError) as exc:
        result = {'status':'failed','original_sources_reverified':False,'errors':[str(exc)]}
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        except OSError as exc:
            result = {'status':'failed','original_sources_reverified':False,'errors':[f'Cannot write validation result: {exc}']}
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
