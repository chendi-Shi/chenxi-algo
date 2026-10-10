"""Deterministic independent arithmetic, search and scale acceptance checks.

Synthetic fixtures test implementation, not financial predictive power. The
reference arithmetic uses Decimal and ordinal sorting, never engine helpers.
"""
from __future__ import annotations

import argparse
import copy
from decimal import Decimal, localcontext
import hashlib
import importlib.metadata
import io
import json
import platform
from pathlib import Path
import random
import statistics
import sys
import time
import unittest
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine
from search import search_configs

AS_OF = date(2026, 5, 1)
MONEY = ('net_income', 'net_income_parent', 'core_income_parent', 'operating_cash_flow',
         'total_assets', 'equity_parent', 'current_assets', 'current_liabilities',
         'long_term_debt', 'total_debt', 'cash', 'revenue', 'cogs', 'capex', 'ebit', 'interest_expense')


def universe(count, seed):
    randomizer = random.Random(seed)
    rows, values = [], []
    for i in range(count):
        ticker = f'SYNTHETIC-{i:05d}'
        revenue = randomizer.randint(200, 500)
        margin = randomizer.uniform(.05, .25)
        growth = randomizer.uniform(.7, 1.5)
        cash_ratio = randomizer.uniform(.4, 2.5)
        for j, year in enumerate((2023, 2024, 2025)):
            sales = round(revenue * growth ** j, 4)
            profit = round(sales * margin, 4)
            rows.append(dict(ticker=ticker, name='Synthetic validation', market='A' if i % 40 < 20 else 'HK',
                sector=f'Sector-{i // 40}', currency='CNY', scope='consolidated',
                period_start=f'{year}-01-01', period_end=f'{year}-12-31', available_at=f'{year+1}-03-31',
                revision_id=0, source_url=f'https://example.invalid/{ticker}/{year}',
                filing_id=f'{ticker}-{year}', unit_scale=1, audit_opinion='unqualified', equity_issued=0,
                net_income=profit, net_income_parent=profit * .95, core_income_parent=profit * .9,
                operating_cash_flow=round(profit * cash_ratio, 4), total_assets=500 + 25 * j,
                equity_parent=200 + 15 * j, current_assets=150 + 10 * j, current_liabilities=100,
                long_term_debt=20, total_debt=40, cash=60, revenue=sales, cogs=sales * .6,
                capex=8, ebit=profit + 10, interest_expense=randomizer.randint(1, 6)))
        values.append(dict(ticker=ticker, snapshot_date=AS_OF.isoformat(), available_at=AS_OF.isoformat(),
            revision_id=0, market_cap=randomizer.randint(150, 1200), unit_scale=1,
            currency='CNY', cap_scope='total_company', source_url=f'https://example.invalid/cap/{ticker}'))
    return rows, values


def independent_metrics(history, cap):
    d = lambda value: Decimal(str(value))
    old, prior, current = [{k: d(row[k]) for k in MONEY} for row in history]
    with localcontext() as ctx:
        ctx.prec = 40
        return {k: float(v) for k, v in {
            'roe': current['net_income_parent'] / ((prior['equity_parent'] + current['equity_parent']) / 2),
            'cash_conversion': current['operating_cash_flow'] / current['net_income'],
            'core_profit_share': current['core_income_parent'] / current['net_income_parent'],
            'core_earnings_yield': current['core_income_parent'] / d(cap),
            'fcf_yield': (current['operating_cash_flow'] - current['capex']) / d(cap),
            'revenue_cagr_2y': (current['revenue'] / old['revenue']).sqrt() - 1,
            'core_profit_cagr_2y': (current['core_income_parent'] / old['core_income_parent']).sqrt() - 1,
            'net_debt_to_cfo': (current['total_debt'] - current['cash']) / current['operating_cash_flow'],
            'interest_cover': current['ebit'] / current['interest_expense'],
        }.items()}


def oracle_check():
    rows, values = universe(200, 1701)
    companies = engine.screen(rows, values, AS_OF)['companies']
    reference = {v['ticker']: independent_metrics(rows[i*3:i*3+3], v['market_cap']) for i, v in enumerate(values)}
    errors = [abs(c['metrics'][k] - value) for c in companies for k, value in reference[c['ticker']].items()]
    # Explicit financial definitions and tie ranking independent of engine code.
    blocks = [(('roe', 'cash_conversion', 'core_profit_share'), .4),
              (('core_earnings_yield', 'fcf_yield'), .25),
              (('revenue_cagr_2y', 'core_profit_cagr_2y'), .2),
              (('net_debt_to_cfo', 'interest_cover'), .15)]
    scores = []
    for c in companies:
        peers = [x for x in companies if (x['market'], x['sector']) == (c['market'], c['sector'])]
        total = 0
        for fields, weight in blocks:
            ranks = []
            for field in fields:
                cap = {'cash_conversion': 2, 'core_profit_share': 1}.get(field, float('inf'))
                # Avoid fabricated ties from floating point in near-constant inputs:
                # the scoring oracle uses the production raw finite value solely
                # to identify exact ties; the financial ratios above are independent.
                vector = sorted(float(format(min(p['metrics'][field], cap), '.12g')) for p in peers)
                own = float(format(min(c['metrics'][field], cap), '.12g'))
                positions = [i for i, number in enumerate(vector) if number == own]
                score = statistics.mean(positions) / (len(vector) - 1) * 100
                ranks.append(100 - score if field == 'net_debt_to_cfo' else score)
            total += statistics.mean(ranks) * weight
        scores.append(abs(total - c['score']))
    if max(errors) > 1e-10 or max(scores) > 1e-10:
        raise AssertionError('independent arithmetic/ranking mismatch')
    scaled = copy.deepcopy(rows)
    scaled_values = copy.deepcopy(values)
    for row in scaled:
        row['unit_scale'] = 10000
        for field in MONEY:
            row[field] /= 10000
    for row in scaled_values:
        row['unit_scale'] = 10000
        row['market_cap'] /= 10000
    replay = {c['ticker']: c for c in engine.screen(scaled, scaled_values, AS_OF)['companies']}
    # A denomination change must not turn roundoff into different rankings.
    unit_error = max(abs(c['metrics'][k] - replay[c['ticker']]['metrics'][k]) for c in companies for k in reference[c['ticker']])
    unit_rank_drift = max(abs(c['score'] - replay[c['ticker']]['score']) for c in companies)
    if unit_error > 1e-10:
        raise AssertionError('currency unit normalization changes financial ratios')
    if unit_rank_drift > 1e-9:
        raise AssertionError(f'currency unit normalization changes score by {unit_rank_drift}')
    return {'companies': len(companies), 'financial_metric_comparisons': len(errors),
            'max_metric_error': max(errors), 'weighted_score_comparisons': len(scores),
            'max_score_error': max(scores), 'unit_conversion_metric_error': unit_error,
            'unit_conversion_max_score_drift': unit_rank_drift}


def search_check():
    comparisons = []
    for seed in range(20):
        rows, values = universe(20, seed)
        kwargs = dict(market='A', sector='Sector-0', target_count=4,
            space={'weights': [dict(quality=.4, value=.25, growth=.2, balance=.15),
                               dict(quality=.2, value=.5, growth=.15, balance=.15)],
                   'candidate_score': [65, 70, 75], 'min_roe': [.08, .12, .16],
                   'min_cash_conversion': [.8, 1.2], 'max_net_debt_to_cfo': [4, 0]})
        grid = search_configs(rows, values, AS_OF, {}, method='grid', **kwargs)
        beam = search_configs(rows, values, AS_OF, {}, method='beam', beam_width=100, **kwargs)
        narrow = search_configs(rows, values, AS_OF, {}, method='beam', beam_width=1, **kwargs)
        if grid['best'] != beam['best']:
            raise AssertionError(f'wide beam differs from exhaustive grid at seed {seed}')
        comparisons.append({'seed': seed, 'wide_beam_matches_grid': True,
            'narrow_beam_matches_grid': grid['best'] == narrow['best'],
            'grid_evaluations': grid['evaluations'], 'narrow_beam_evaluations': narrow['evaluations']})
    return {'independent_random_cohorts': len(comparisons), 'wide_beam_matches': len(comparisons),
            'narrow_beam_matches': sum(c['narrow_beam_matches_grid'] for c in comparisons), 'cases': comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'validation'/'enterprise_results.json')
    parser.add_argument('--companies', type=int, default=1000)
    args = parser.parse_args(argv)
    if args.companies < 40:
        parser.error('--companies must be >=40')
    test_log = io.StringIO()
    sys.path.insert(0, str(ROOT/'tests'))
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName('validation.enterprise_a_share_tests'))
    tests = unittest.TextTestRunner(stream=test_log).run(suite)
    report = {'validation_date': '2026-10-10', 'python': platform.python_version(),
        'platform': platform.system(), 'numpy': importlib.metadata.version('numpy'),
        'tests': {'run': tests.testsRun, 'failures': len(tests.failures), 'errors': len(tests.errors),
                  'skipped': len(tests.skipped)}, 'checks': {}, 'failures': []}
    for name, checker in (('independent_arithmetic', oracle_check), ('search_cross_check', search_check)):
        try:
            report['checks'][name] = checker()
        except (AssertionError, ValueError) as exc:
            report['failures'].append(f'{name}: {exc}')
    rows, values = universe(args.companies, 1776)
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        result = engine.screen(rows, values, AS_OF)
        timings.append(time.perf_counter() - start)
    report['checks']['scale'] = {'synthetic_companies': len(result['companies']), 'statement_rows': len(rows),
        'repetitions': 3, 'seconds': timings, 'median_seconds': statistics.median(timings)}
    report['code_sha256'] = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted([p for p in ROOT.glob('*.py') if p.name != 'package_delivery.py']
                        + list((ROOT/'tests').glob('*.py')) + list((ROOT/'validation').glob('enterprise*.py')))}
    report['software_checks_passed'] = tests.wasSuccessful() and not tests.skipped and not report['failures']
    report['investment_efficacy_validated'] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True))
    if not tests.wasSuccessful():
        print(test_log.getvalue(), file=sys.stderr)
    return 0 if report['software_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
