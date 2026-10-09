"""Public-data feasibility study; historical reconstruction is NOT a PIT backtest.

Run collect_public.py first. Raw provider payloads are excluded from Git. Only
selected financial facts, source fingerprints, and descriptive results are saved.
No missing borrowing/issuance/adjusted-earnings fact is inferred as zero.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'validation'
sys.path.insert(0, str(ROOT))
import engine

CODES = ('600519', '000858', '000568', '600809', '002304', '603369')
AS_OF = date(2025, 6, 30)
MAPS = {
    'balance': {'total_assets': 'TOTAL_ASSETS', 'equity_parent': 'TOTAL_PARENT_EQUITY',
                'current_assets': 'TOTAL_CURRENT_ASSETS', 'current_liabilities': 'TOTAL_CURRENT_LIAB'},
    'income': {'revenue': 'OPERATE_INCOME', 'cogs': 'OPERATE_COST', 'net_income': 'NETPROFIT',
               'net_income_parent': 'PARENT_NETPROFIT', 'core_income_parent': 'DEDUCT_PARENT_NETPROFIT'},
    'cashflow': {'operating_cash_flow': 'NETCASH_OPERATE', 'capex': 'CONSTRUCT_LONG_ASSET', 'cash': 'END_CCE'},
}


def load(code, kind):
    path = HERE / 'raw' / f'{code}_{kind}.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    response = payload['response']
    rows = response.get('data') if isinstance(response, dict) else None
    if rows is None and isinstance(response.get('result'), dict):
        rows = response['result'].get('data')
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{code}/{kind}: no source records')
    return rows, {'file': path.name, 'url': payload['url'],
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'metadata': payload.get('metadata', {})}


def compact(report):
    return {'counts': dict(Counter(c['status'] for c in report['companies'])),
            'scored': sum(c['score'] is not None for c in report['companies']),
            'companies': [{k: c.get(k) for k in ('ticker', 'name', 'score', 'status', 'f_score',
                'f_score_known', 'metric_coverage', 'metrics', 'blocks', 'warnings', 'reasons')} for c in report['companies']],
            'audit': report['audit']}


def ranks(values):
    return [sum(x < v for x in values) + (sum(x == v for x in values) + 1) / 2 for v in values]


def spearman(pairs):
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    a, b = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    am, bm = statistics.mean(a), statistics.mean(b)
    denominator = sum((v-am)**2 for v in a) * sum((v-bm)**2 for v in b)
    return None if denominator == 0 else sum((x-am)*(y-bm) for x,y in zip(a,b)) / denominator**.5


def build_dataset():
    statements, valuations, source_files, facts = [], [], [], []
    for code in CODES:
        by_kind = {}
        for kind in MAPS:
            records, meta = load(code, kind)
            source_files.append(meta)
            by_kind[kind] = {r['REPORT_DATE'][:10]: r for r in records}
        for year in (2022, 2023, 2024, 2025):
            end = f'{year}-12-31'
            originals = [by_kind[kind][end] for kind in MAPS]
            b = originals[0]
            notices = [r['NOTICE_DATE'][:10] for r in originals]
            updates = [(r.get('UPDATE_DATE') or r['NOTICE_DATE'])[:10] for r in originals]
            row = {'ticker': b['SECUCODE'], 'name': b['SECURITY_NAME_ABBR'], 'market': 'A',
                'sector': 'Liquor', 'currency': b['CURRENCY'], 'scope': 'consolidated', 'unit_scale': 1,
                'period_start': f'{year}-01-01', 'period_end': end,
                'available_at': max(notices + updates), 'original_notice_date': max(notices),
                'provider_update_date': max(updates), 'revision_id': 0,
                'filing_id': f'eastmoney-current-vintage-{code}-{year}',
                'source_url': f'https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/Index?type=web&code={b["SECUCODE"].split(".")[-1].lower()}{code}',
                'audit_opinion': 'unqualified' if b.get('OPINION_TYPE') == '标准无保留意见' else '',
                # These require notes/event validation. Provider nulls are unknown.
                'equity_issued': '', 'total_debt': None, 'long_term_debt': None,
                'ebit': None, 'interest_expense': None,
            }
            for kind, mapping in MAPS.items():
                source = by_kind[kind][end]
                row.update({target: source.get(key) for target, key in mapping.items()})
            if row['currency'] != 'CNY':
                raise ValueError('unexpected unit/currency')
            facts.append(row)
            if year <= 2024:
                statements.append(row)
        value_rows, meta = load(code, 'value')
        source_files.append(meta)
        v = next(r for r in value_rows if r['TRADE_DATE'][:10] == AS_OF.isoformat())
        if abs(v['CLOSE_PRICE'] * v['TOTAL_SHARES'] - v['TOTAL_MARKET_CAP']) > max(1, v['TOTAL_MARKET_CAP'] * 1e-8):
            raise ValueError(f'{code}: cap units/share-price identity mismatch')
        valuations.append({'ticker': b['SECUCODE'], 'snapshot_date': AS_OF.isoformat(),
            'available_at': AS_OF.isoformat(), 'revision_id': 0, 'market_cap': v['TOTAL_MARKET_CAP'],
            'unit_scale': 1, 'currency': 'CNY', 'cap_scope': 'total_company', 'source_url': meta['url']})
    return {'notice': 'Selected current-vintage public financial facts; no verified historical snapshot.',
        'statements': statements, 'valuations': valuations, 'all_years': facts, 'source_files': source_files}


def official_check(dataset):
    check_path = HERE / 'official_checks.json'
    if not check_path.exists():
        raise ValueError('official_checks.json missing: independent annual-report reconciliation required')
    checks = json.loads(check_path.read_text(encoding='utf-8'))
    # The six independently checked Moutai figures come from annual report p5.
    official = checks['companies']['600519']['values']['2024']
    expected = {target: official[source] for target, source in {
        'revenue': 'revenue', 'net_income_parent': 'net_profit',
        'core_income_parent': 'net_profit_core', 'operating_cash_flow': 'operating_cash_flow',
        'equity_parent': 'total_equity', 'total_assets': 'total_assets'}.items()}
    current = next(r for r in dataset['all_years'] if r['ticker'] == '600519.SH' and r['period_end'] == '2024-12-31')
    comparisons = [{'ticker': '600519.SH', 'year': 2024, 'field': field, 'official': value,
        'provider': current[field], 'match': current[field] is not None and abs(current[field]-value) <= .01}
        for field, value in expected.items()]
    if not all(r['match'] for r in comparisons):
        raise ValueError('official financial reconciliation failed')
    return {'matched': len(comparisons), 'total': len(comparisons), 'comparisons': comparisons,
        'source': 'https://static.cninfo.com.cn/finalpage/2025-04-03/1222993920.PDF',
        'independent_checks_file_sha256': hashlib.sha256(check_path.read_bytes()).hexdigest()}


def evaluate(dataset):
    config = engine.config_from(None)
    strict = engine.screen(dataset['statements'], dataset['valuations'], AS_OF, config)
    reconstructed = copy.deepcopy(dataset['statements'])
    for row in reconstructed:
        row['available_at'] = row['original_notice_date']
        row['filing_id'] = row['filing_id'].replace('current-vintage', 'UNVERIFIED-notice-reconstruction')
    default = engine.screen(reconstructed, dataset['valuations'], AS_OF, config)
    # Predefined three-block diagnostic: debt notes are unverified, so status gates
    # remain data_review. A score is a diagnostic, never permission to bypass gates.
    profiles = {'balanced': {'quality': .5, 'value': .3, 'growth': .2, 'balance': 0},
        'equal': {'quality': 1/3, 'value': 1/3, 'growth': 1/3, 'balance': 0},
        'quality': {'quality': .7, 'value': .2, 'growth': .1, 'balance': 0},
        'value': {'quality': .2, 'value': .6, 'growth': .2, 'balance': 0},
        'growth': {'quality': .2, 'value': .2, 'growth': .6, 'balance': 0}}
    sensitivity, reports = {}, {}
    for name, weights in profiles.items():
        alternative = copy.deepcopy(config)
        alternative['weights'] = weights
        report = engine.screen(reconstructed, dataset['valuations'], AS_OF, alternative)
        reports[name] = report
        scored = [c for c in report['companies'] if c['score'] is not None]
        sensitivity[name] = {'weights': weights, 'scored': len(scored),
            'top_two': [c['ticker'] for c in scored[:2]],
            'scores': {c['ticker']: c['score'] for c in scored}}
    base = reports['balanced']
    by_ticker = {c['ticker']: c for c in base['companies']}
    years = {(r['ticker'], r['period_end']): r for r in dataset['all_years']}
    outcomes = []
    for ticker, company in by_ticker.items():
        previous, current = years[ticker, '2024-12-31'], years[ticker, '2025-12-31']
        outcomes.append({'ticker': ticker, 'name': company['name'], 'diagnostic_score_2024': company['score'],
            'roe_2024': company['metrics'].get('roe'),
            'revenue_growth_2025': engine.growth(current['revenue'], previous['revenue']),
            'core_profit_growth_2025': engine.growth(current['core_income_parent'], previous['core_income_parent']),
            'cash_conversion_2025': engine.divide(current['operating_cash_flow'], current['net_income'])})
    correlations = {name: {'diagnostic_score': spearman([(r['diagnostic_score_2024'], r[name]) for r in outcomes]),
        'roe_only_baseline': spearman([(r['roe_2024'], r[name]) for r in outcomes])}
        for name in ('revenue_growth_2025', 'core_profit_growth_2025', 'cash_conversion_2025')}
    base_top = set(sensitivity['balanced']['top_two'])
    for profile in sensitivity.values():
        chosen = set(profile['top_two'])
        profile['top_two_jaccard_vs_balanced'] = len(base_top & chosen) / len(base_top | chosen) if (base_top | chosen) else None
    leave_one_out = {}
    for ticker in by_ticker:
        reduced = [r for r in reconstructed if r['ticker'] != ticker]
        alternative = copy.deepcopy(config)
        alternative['weights'] = profiles['balanced']
        replay = engine.screen(reduced, dataset['valuations'], AS_OF, alternative)
        scored = [c for c in replay['companies'] if c['score'] is not None]
        leave_one_out[ticker] = {'scored': len(scored), 'top_two': [c['ticker'] for c in scored[:2]]}
    return {'strict_provider_update_date': compact(strict), 'unverified_original_notice_reconstruction': compact(default),
        'three_block_diagnostic_reconstruction': compact(base), 'preference_sensitivity': sensitivity,
        'leave_one_peer_out': leave_one_out, '2025_descriptive_outcomes': outcomes,
        'descriptive_spearman_n6': correlations,
        'limitations': ['Current-vintage history is not verified point-in-time data; these are not backtest results.',
            'Six purposefully selected liquor peers and one outcome year cannot establish generalization or investment efficacy.',
            'Missing borrowing/lease/current-debt notes and issuance events remain unknown; no automated candidate approval.',
            'Moutai consolidated CFO includes a finance subsidiary; require an operating-business cash-flow reconciliation.',
            'Hong Kong core/adjusted earnings need separate accounting harmonization; no universal A/H profit definition.',
            'Three-block diagnostic intentionally disables balance scoring, while critical debt checks remain active.',
            'No return, alpha, win-rate, or trading recommendation is calculated.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-cache', action='store_true', help='Rebuild facts from locally downloaded raw cache; default replays committed dataset.json offline.')
    args = parser.parse_args()
    dataset = build_dataset() if args.from_cache else json.loads((HERE/'dataset.json').read_text(encoding='utf-8'))
    report = {'as_of_for_exploration': AS_OF.isoformat(), 'retrieval_date': '2026-10-09',
        'engine_sha256': hashlib.sha256((ROOT/'engine.py').read_bytes()).hexdigest(),
        'sample': {'A_share_issuers': len(CODES), 'annual_company_years': len(dataset['all_years']),
            'industry': 'Liquor', 'selection': 'purposive, no survivorship-complete historical universe'},
        'official_reconciliation': official_check(dataset), 'evaluation': evaluate(dataset)}
    hk = json.loads((HERE/'hk_coverage.json').read_text(encoding='utf-8'))
    official_hk = json.loads((HERE/'official_checks.json').read_text(encoding='utf-8'))['companies']['02331']['values']['2024']
    hk_check = {target: {'provider_yuan': hk['years']['2024'][target]['value'],
        'official_yuan': official_hk[source],
        'matched': hk['years']['2024'][target]['value'] == official_hk[source]}
        for target, source in {'revenue': 'revenue', 'net_income_parent': 'net_profit',
            'total_assets': 'total_assets', 'equity_parent': 'total_equity',
            'operating_cash_flow': 'operating_cash_flow'}.items()}
    if not all(v['matched'] for v in hk_check.values()) or any(hk['years'][y]['core_income_parent']['value'] is not None for y in ('2022','2023','2024')):
        raise ValueError('HK financial reconciliation or unknown-core-profit check failed')
    report['hk_reconciliation'] = {'ticker': '02331.HK', 'matched': len(hk_check), 'comparisons': hk_check,
        'source': hk['official_check_source'], 'screening_status': 'coverage study only; no harmonized core-profit input or complete valuation/debt lineage'}
    for filename, content in [('dataset.json', dataset), ('real_results.json', report)]:
        (HERE/filename).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'sample': report['sample'], 'official_matches': report['official_reconciliation']['matched'],
        'hk_matches': report['hk_reconciliation']['matched'],
        'strict': report['evaluation']['strict_provider_update_date']['counts'],
        'default_reconstruction': report['evaluation']['unverified_original_notice_reconstruction']['counts'],
        'diagnostic_scored': report['evaluation']['three_block_diagnostic_reconstruction']['scored'],
        'sensitivity': report['evaluation']['preference_sensitivity'],
        'correlations': report['evaluation']['descriptive_spearman_n6']}, ensure_ascii=True, allow_nan=False))

if __name__ == '__main__':
    main()
