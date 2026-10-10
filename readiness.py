"""Explicit input-coverage gates. Passing these is not a deployment acceptance."""
from __future__ import annotations

import math
import re

from engine import iso

POLICY_KEYS = {
    'expected_tickers', 'max_universe_age_days', 'max_document_age_days',
    'min_document_coverage', 'min_operating_coverage', 'min_valuation_coverage',
    'max_invalid_records', 'max_identity_conflicts', 'require_document_hashes',
    'require_known_date_basis',
}


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != POLICY_KEYS:
        raise ValueError('Production policy must explicitly specify: ' + ', '.join(sorted(POLICY_KEYS)))
    expected = policy['expected_tickers']
    if (not isinstance(expected, list) or not expected or
            any(not isinstance(t, str) or not t.strip() or t != t.strip().upper() for t in expected)
            or len(set(expected)) != len(expected)):
        raise ValueError('expected_tickers must be a nonempty unique list of canonical security identifiers')
    for key in ('max_universe_age_days', 'max_document_age_days', 'max_invalid_records', 'max_identity_conflicts'):
        if type(policy[key]) is not int or policy[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('min_document_coverage', 'min_operating_coverage', 'min_valuation_coverage'):
        value = policy[key]
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and 0 < value <= 1
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f'{key} must be >0 and <=1')
    for key in ('require_document_hashes', 'require_known_date_basis'):
        if type(policy[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    return dict(policy, expected_tickers=sorted(expected))


def _theme_evidence_assessment(company, cutoff, policy):
    """Check topic-specific operating proof, never an unrelated recent filing."""
    if policy is None:
        return {'status': 'not_assessed', 'reasons': ['No explicit freshness policy supplied.'],
                'qualifying_document_ids': []}
    if company.get('business_status') != 'current_business':
        return {'status': 'not_current_business', 'reasons': ['Operating business is not currently established.'],
                'qualifying_document_ids': []}
    supports = company.get('business_support_evidence', [])
    qualified, rejected = [], []
    for item in supports:
        reasons = []
        if item.get('claim_status') not in ('direct_business', 'upstream_or_downstream'):
            reasons.append('not_operating_support')
        try:
            age = (cutoff - iso(item.get('available_at'))).days
            if not 0 <= age <= policy['max_document_age_days']:
                reasons.append('stale_or_future_business_support')
        except (ValueError, TypeError):
            reasons.append('invalid_support_date')
        if (policy['require_document_hashes'] and
                (not isinstance(item.get('source_sha256'), str) or
                 not re.fullmatch('[0-9a-fA-F]{64}', item['source_sha256']))):
            reasons.append('missing_source_hash')
        if policy['require_known_date_basis'] and item.get('date_basis') not in ('official_release', 'observed_at'):
            reasons.append('unknown_date_basis')
        if (not isinstance(item.get('exact_excerpt'), str) or not item['exact_excerpt'].strip()
                or type(item.get('char_start')) is not int or type(item.get('char_end')) is not int
                or item['char_start'] < 0 or item['char_end'] - item['char_start'] != len(item['exact_excerpt'])
                or not isinstance(item.get('source_url'), str) or not item['source_url']
                or not item.get('document_id') or not item.get('page')):
            reasons.append('missing_auditable_support')
        if reasons:
            rejected.append({'document_id': item.get('document_id'), 'reasons': reasons})
        else:
            qualified.append(item['document_id'])
    return {'status': 'passed' if qualified else 'blocked',
            'reasons': [] if qualified else ['No current-theme operating evidence meets the explicit age/provenance policy.'],
            'max_document_age_days': policy['max_document_age_days'],
            'require_document_hashes': policy['require_document_hashes'],
            'require_known_date_basis': policy['require_known_date_basis'],
            'qualifying_document_ids': sorted(set(qualified)), 'rejected_support': rejected}


def apply_theme_business_policy(report, policy=None):
    """Withhold profitable labels without fresh, auditable current-theme proof.

    The financial calculations and financial_style_candidates remain intact.
    Without an explicit policy, no enterprise freshness threshold is invented.
    Call before applying a display limit.
    """
    policy = validate_policy(policy) if policy is not None else None
    cutoff = iso(report['as_of'])
    for company in report['companies']:
        assessment = _theme_evidence_assessment(company, cutoff, policy)
        company['theme_business_freshness'] = assessment
        if assessment['status'] != 'blocked':
            continue
        financial = company['financial']
        withheld = [style for style in financial['styles']
                    if style not in ('loss_watchlist', 'unclassified')]
        financial['styles'] = [style for style in financial['styles'] if style == 'loss_watchlist'] or ['unclassified']
        financial['business_review_required'] = True
        if withheld:
            financial['reasons'].append('Financial thresholds pass, but topic-specific operating evidence fails the explicit freshness/provenance policy; profitable styles withheld.')
    report['style_lists'] = {style: [company['ticker'] for company in report['companies']
                                    if style in company['financial']['styles']]
                             for style in report['style_lists']}
    return report


def assess_readiness(report, companies, documents, policy=None):
    """Check the declared research universe, never only a top-N result subset.

    The caller must specify the actual intended pool. No sample-count default or
    successful computation can silently become a production acceptance.
    """
    if policy is None:
        return {'status': 'not_assessed', 'production_approved': False,
                'reasons': ['No explicit coverage/freshness policy supplied.'], 'checks': []}
    policy = validate_policy(policy)
    cutoff = iso(report['as_of'])
    expected = set(policy['expected_tickers'])
    audit = report['audit']
    rejected_companies = {r['index'] for r in audit['rejected_companies']}
    rejected_documents = {r['index'] for r in audit['rejected_documents']}
    bad_tickers = {r['ticker'] for r in audit['conflicts'] if r.get('type') == 'company_identity'}
    bad_document_ids = {r['document_id'] for r in audit['conflicts'] if r.get('type') == 'document_id'}
    accepted_companies = {c['ticker'].strip().upper(): c for index, c in enumerate(companies)
                          if index not in rejected_companies and c['ticker'].strip().upper() not in bad_tickers}
    present = expected & accepted_companies.keys()
    stale_directory = sorted(ticker for ticker in present if
                             (cutoff - iso(accepted_companies[ticker]['universe_as_of'])).days > policy['max_universe_age_days'])
    docs_by_company = {}
    for index, doc in enumerate(documents):
        if index in rejected_documents or doc['document_id'] in bad_document_ids:
            continue
        ticker = doc['ticker'].strip().upper()
        if ticker not in present:
            continue
        if (cutoff - iso(doc['available_at'])).days > policy['max_document_age_days']:
            continue
        if policy['require_document_hashes'] and not re.fullmatch('[0-9a-fA-F]{64}', doc.get('source_sha256', '')):
            continue
        if policy['require_known_date_basis'] and doc.get('date_basis') not in ('official_release', 'observed_at'):
            continue
        docs_by_company.setdefault(ticker, []).append(doc['document_id'])
    financial = report['financial_audit']['company_statuses']
    valid_status = {'candidate', 'watchlist', 'excluded'}
    def financially_covered(status_key):
        return {ticker for ticker in present if ticker in financial
                and financial[ticker][status_key] in valid_status
                and all(financial[ticker][key] == accepted_companies[ticker][key] for key in ('market', 'sector'))}
    operating = financially_covered('operating_status')
    valuation = financially_covered('valuation_status')
    expected_exclusions = {'universe_after_as_of', 'available_after_as_of', 'outside_scope', 'ticker_not_in_visible_universe'}
    invalid = [row for key in ('rejected_companies', 'rejected_documents') for row in audit[key]
               if row['reason'] not in expected_exclusions]
    invalid.extend(row for key in ('statement_versions', 'valuation_versions')
                   for row in report['financial_audit'][key] if row['reason'] != 'not_yet_available')
    identity_conflicts = [row for row in audit['conflicts'] if row['type'] in ('company_identity', 'document_id')]
    theme_checks = {company['ticker']: _theme_evidence_assessment(company, cutoff, policy)
                    for company in report.get('all_companies', report['companies'])}
    stale_theme_business = {ticker for ticker, assessment in theme_checks.items() if assessment['status'] == 'blocked'}
    checks = []
    def check(name, actual, threshold, passed, missing=None):
        item = {'name': name, 'actual': actual, 'threshold': threshold, 'passed': bool(passed)}
        if missing is not None:
            item['affected_tickers'] = sorted(missing)
        checks.append(item)
    check('declared_universe_present', len(present), len(expected), present == expected, expected - present)
    check('universe_freshness', len(stale_directory), 0, not stale_directory, stale_directory)
    for name, covered, minimum in (
            ('document_coverage', set(docs_by_company), policy['min_document_coverage']),
            ('operating_coverage', operating, policy['min_operating_coverage']),
            ('valuation_coverage', valuation, policy['min_valuation_coverage'])):
        coverage = len(covered) / len(expected)
        check(name, coverage, minimum, coverage >= minimum, expected - covered)
    check('invalid_records', len(invalid), policy['max_invalid_records'], len(invalid) <= policy['max_invalid_records'])
    check('identity_conflicts', len(identity_conflicts), policy['max_identity_conflicts'], len(identity_conflicts) <= policy['max_identity_conflicts'])
    check('theme_business_freshness', len(stale_theme_business), 0, not stale_theme_business, stale_theme_business)
    failures = [c['name'] for c in checks if not c['passed']]
    return {'status': 'blocked' if failures else 'data_gates_passed', 'production_approved': False,
            'checks': checks, 'reasons': failures, 'policy': policy,
            'theme_business_checks': theme_checks,
            'scope': {'expected_companies': len(expected), 'present_companies': len(present),
                      'document_companies': len(docs_by_company), 'operating_companies': len(operating),
                      'valuation_companies': len(valuation)},
            'limitations': ['This checks the explicitly declared research pool, not all listed companies.',
                            'Hashes check integrity; they do not certify factual accuracy.',
                            'Deployment, data-source permission, research relevance and operational acceptance remain separate.']}
