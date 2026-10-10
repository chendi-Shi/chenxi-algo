"""Production regressions: unverified inputs cannot alter verified rankings."""
from __future__ import annotations

import copy
from datetime import datetime
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine
from test_engine import AS_OF, annual_history, valuation


def cohort(count=5):
    rows, values = [], []
    for i in range(count):
        ticker = f'PEER-{i}'
        history = annual_history(ticker)
        history[-1]['net_income_parent'] += i
        rows.extend(history)
        values.append(valuation(ticker, market_cap=150 + 30 * i))
    return rows, values


def companies(rows, values, config=None):
    return {c['ticker']: c for c in engine.screen(rows, values, AS_OF, config)['companies']}


class PeerIsolationTests(unittest.TestCase):
    def test_data_review_values_cannot_change_verified_percentiles(self):
        baseline_rows, baseline_values = cohort()
        baseline = companies(baseline_rows, baseline_values)
        for mode in ('audit', 'valuation', 'missing_cash', 'opening_equity'):
            with self.subTest(mode=mode):
                extra = annual_history('UNVERIFIED')
                extra[-1].update(net_income_parent=2000, core_income_parent=1800,
                                 revenue=10000, operating_cash_flow=4000)
                values = baseline_values + [valuation('UNVERIFIED', market_cap=2)]
                if mode == 'audit':
                    extra[-1]['audit_opinion'] = 'qualified'
                elif mode == 'valuation':
                    values.pop()
                elif mode == 'missing_cash':
                    extra[-1]['cash'] = ''
                else:
                    extra[-2]['equity_parent'] = -1
                result = companies(baseline_rows + extra, values)
                rejected = result.pop('UNVERIFIED')
                self.assertEqual(rejected['status'], 'data_review')
                self.assertTrue(rejected['metrics'])
                self.assertIsNone(rejected['score'])
                self.assertFalse(rejected['peer_eligible'])
                self.assertEqual(rejected['peer_count'], 0)
                for ticker in baseline:
                    for key in ('score', 'blocks', 'peer_count', 'status'):
                        self.assertEqual(result[ticker][key], baseline[ticker][key],
                                         f'{mode} changed verified {ticker} {key}')

    def test_unverified_company_cannot_supply_fifth_peer(self):
        rows, values = cohort(4)
        extra = annual_history('UNVERIFIED')
        extra[-1]['audit_opinion'] = ''
        result = companies(rows + extra, values + [valuation('UNVERIFIED')])
        for ticker, company in result.items():
            self.assertIsNone(company['score'])
            self.assertEqual(company['peer_count'], 0 if ticker == 'UNVERIFIED' else 4)

    def test_economic_exclusion_remains_a_valid_peer(self):
        rows, values = cohort(4)
        extra = annual_history('HIGH-DEBT')
        extra[-1]['total_debt'] = 1000
        result = companies(rows + extra, values + [valuation('HIGH-DEBT')])
        self.assertEqual(result['HIGH-DEBT']['status'], 'excluded')
        self.assertTrue(result['HIGH-DEBT']['peer_eligible'])
        self.assertTrue(all(c['peer_count'] == 5 and c['score'] is not None for c in result.values()))

    def test_data_missing_alongside_economic_failure_has_review_priority(self):
        for field in ('total_debt', 'cash'):
            with self.subTest(field=field):
                rows = annual_history('BAD')
                rows[-1].update(net_income=-1)
                rows[-1][field] = ''
                company = companies(rows, [valuation('BAD')])['BAD']
                self.assertEqual(company['status'], 'data_review')
                self.assertFalse(company['peer_eligible'])

    def test_each_historical_audit_is_required_for_multiyear_metrics(self):
        for year in (0, 1, 2):
            for audit in ('qualified', ''):
                with self.subTest(year=year, audit=audit):
                    rows = annual_history('BAD')
                    rows[year]['audit_opinion'] = audit
                    company = companies(rows, [valuation('BAD')])['BAD']
                    self.assertEqual(company['status'], 'data_review')
                    self.assertTrue(company['metrics'])
                    self.assertIsNone(company['score'])
                    self.assertTrue(any('three-year history' in w for w in company['warnings']))


class StrictInputTests(unittest.TestCase):
    def test_financial_boolean_or_overflow_is_rejected_without_poisoning_others(self):
        for value in (True, False, 10 ** 400):
            with self.subTest(value_type=type(value).__name__):
                bad = annual_history('BAD')
                bad[-1]['net_income'] = value
                result = companies(annual_history('GOOD') + bad,
                                   [valuation('GOOD'), valuation('BAD')])
                self.assertEqual(result['BAD']['status'], 'data_review')
                self.assertEqual(result['GOOD']['status'], 'watchlist')

    def test_boolean_or_overflow_market_cap_is_never_usable(self):
        for value in (True, False, 10 ** 400):
            with self.subTest(value_type=type(value).__name__):
                company = companies(annual_history(), [valuation(market_cap=value)])['600001']
                self.assertEqual(company['status'], 'data_review')
                self.assertIsNone(company['metrics']['core_earnings_yield'])

    def test_revision_ids_are_integers_not_coerced_floats_or_booleans(self):
        for value in (True, 1.0, 1.5, '1.5', -1, '1e3'):
            with self.subTest(revision=value):
                rows = annual_history()
                rows[-1]['revision_id'] = value
                report = engine.screen(rows, [valuation()], AS_OF)
                self.assertEqual(report['companies'][0]['status'], 'data_review')
                self.assertTrue(any('nonnegative integer' in r['reason'] for r in report['audit']))
                report = engine.screen(annual_history(), [valuation(revision_id=value)], AS_OF)
                self.assertEqual(report['companies'][0]['status'], 'data_review')

    def test_future_invalid_revisions_are_ignored_before_validation(self):
        rows, values = cohort()
        baseline = companies(rows, values)
        future = dict(rows[-1], available_at='2026-05-02', revision_id=True,
                      net_income=False, audit_opinion='qualified')
        self.assertEqual(companies(rows + [future], values), baseline)

    def test_malformed_csv_headers_or_extra_cells_are_rejected(self):
        for raw in ('ticker,total_debt,total_debt\nA,100,0\n',
                    'ticker,\nA,0\n', 'ticker, total_debt\nA,0\n',
                    'ticker,total_debt\nA,100,0\n', ''):
            with self.subTest(raw=raw), mock.patch('builtins.open', return_value=io.StringIO(raw)):
                with self.assertRaisesRegex(ValueError, 'CSV'):
                    engine.load_csv('unused.csv')

    def test_csv_leading_zero_ticker_and_quoted_text_are_preserved(self):
        raw = 'ticker,name,cash\n00700.HK,"Company, Ltd",\n'
        with mock.patch('builtins.open', return_value=io.StringIO(raw)):
            self.assertEqual(engine.load_csv('unused.csv'),
                             [{'ticker': '00700.HK', 'name': 'Company, Ltd', 'cash': ''}])


class ConfigurationBoundaryTests(unittest.TestCase):
    def test_null_or_duplicate_config_keys_cannot_silently_change_rules(self):
        for raw in ('null', '{"candidate_score": 65, "candidate_score": 0}',
                    '{"weights": {"quality": 1, "quality": 0, "value": 1, "growth": 0, "balance": 0}}'):
            with self.subTest(raw=raw), mock.patch.object(Path, 'read_text', return_value=raw):
                with self.assertRaises(ValueError):
                    engine.config_from('unused.json')

    def test_screen_and_config_file_share_strict_validation(self):
        invalid = (
            {'weights': None}, {'weights': []},
            {'weights': dict.fromkeys(engine.BLOCK_FIELDS, 0)},
            {'weights': {'quality': True, 'value': 0, 'growth': 0, 'balance': 0}},
            {'candidate_score': True}, {'candidate_score': float('nan')},
            {'min_roe': 10 ** 400}, {'min_cash_conversion': '0.8'},
            {'min_peer_count': True}, {'max_fx_age_days': False},
            {'min_f_score': True}, {'unknown': 3}, {'excluded_sectors': None},
            [], True, 'invalid',
        )
        for config in invalid:
            with self.subTest(config_type=type(config).__name__):
                with self.assertRaises(ValueError):
                    engine.screen([], [], AS_OF, config)
                with mock.patch.object(Path, 'read_text', return_value=json.dumps(config)):
                    with self.assertRaises(ValueError):
                        engine.config_from('unused.json')

    def test_partial_config_is_completed_and_outputs_do_not_alias_caller(self):
        supplied = {'candidate_score': 70}
        report = engine.screen([], [], AS_OF, supplied)
        self.assertEqual(report['config']['candidate_score'], 70)
        self.assertEqual(report['config']['weights'], engine.DEFAULT_CONFIG['weights'])
        supplied['candidate_score'] = 100
        report['config']['weights']['quality'] = 0
        self.assertEqual(engine.validate_config()['weights']['quality'], .4)
        self.assertEqual(report['config']['candidate_score'], 70)

    def test_cutoff_requires_calendar_date(self):
        for invalid in ('2026-05-01', datetime(2026, 5, 1), None):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, 'calendar date'):
                engine.screen([], [], invalid)


class AccountingOracleTests(unittest.TestCase):
    def test_metrics_match_independent_hand_calculation(self):
        c = companies(annual_history(), [valuation()])['600001']
        expected = {
            'roe': 20 / 65, 'roa': 20 / 110, 'gross_margin': 75 / 160,
            'cash_conversion': 25 / 20, 'core_profit_share': 18 / 20,
            'fcf': 20, 'core_earnings_yield': 18 / 200, 'fcf_yield': 20 / 200,
            'revenue_cagr_2y': (160 / 100) ** .5 - 1,
            'core_profit_cagr_2y': (18 / 9) ** .5 - 1,
            'net_debt_to_cfo': 20 / 25, 'interest_cover': 30 / 2,
            'current_ratio': 55 / 25, 'revenue_growth_yoy': 160 / 125 - 1,
            'core_profit_growth_yoy': 18 / 11 - 1,
        }
        for metric, value in expected.items():
            with self.subTest(metric=metric):
                self.assertAlmostEqual(c['metrics'][metric], value)

    def test_unit_rescaling_preserves_ratios_scores_and_status(self):
        rows, values = cohort()
        baseline = companies(rows, values)
        scaled_rows, scaled_values = copy.deepcopy(rows), copy.deepcopy(values)
        for row in scaled_rows:
            row['unit_scale'] = 1_000_000
        for row in scaled_values:
            row['unit_scale'] = 1_000_000
        scaled = companies(scaled_rows, scaled_values)
        for ticker, original in baseline.items():
            self.assertEqual(original['status'], scaled[ticker]['status'])
            self.assertEqual(original['score'], scaled[ticker]['score'])
            for metric, value in original['metrics'].items():
                expected = value * 1_000_000 if metric == 'fcf' else value
                self.assertAlmostEqual(scaled[ticker]['metrics'][metric], expected)


if __name__ == '__main__':
    unittest.main()
