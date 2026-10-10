import copy
import json
import unittest

from tests.test_engine import annual_history, valuation
from theme_financials import attach_fundamentals


def discovery(tickers=('600001',), **changes):
    defaults = dict(name='Fixture', market='A', sector='Industrials', relevance_score=1.,
                    relations=['direct_business'], business_status='current_business')
    defaults.update(changes)
    return {'as_of': '2026-05-01', 'companies': [dict(ticker=t, **defaults) for t in tickers]}


class ThemeFinancialTests(unittest.TestCase):
    def test_no_financial_input_retains_business_match(self):
        result = attach_fundamentals(discovery())
        self.assertEqual(result['style_lists']['unclassified'], ['600001'])
        self.assertIsNone(result['companies'][0]['financial']['loss_making'])

    def test_disputed_planned_and_uncertain_business_cannot_enter_profitable_styles(self):
        for status in ('historical_or_disputed', 'planned_business', 'uncertain', 'not_assessed'):
            result = attach_fundamentals(discovery(business_status=status), annual_history(), [valuation()])
            financial = result['companies'][0]['financial']
            self.assertEqual(financial['styles'], ['unclassified'])
            self.assertIn('quality_growth', financial['financial_style_candidates'])

    def test_loss_watchlist_survives_disputed_business_classification(self):
        rows = annual_history()
        rows[-1]['net_income_parent'] = -1
        result = attach_fundamentals(discovery(business_status='historical_or_disputed'), rows)
        self.assertEqual(result['style_lists']['loss_watchlist'], ['600001'])

    def test_quality_and_improvement_do_not_require_a_share_price(self):
        result = attach_fundamentals(discovery(), annual_history())
        self.assertEqual(result['style_lists']['quality_growth'], ['600001'])
        self.assertEqual(result['style_lists']['operating_improvement'], ['600001'])
        self.assertEqual(result['style_lists']['relative_value'], [])

    def test_loss_narrowing_has_no_growth_or_value_label(self):
        rows = annual_history()
        for row, profit in zip(rows, [-30, -20, -10]):
            row.update(net_income_parent=profit, net_income=profit, core_income_parent=profit)
        result = attach_fundamentals(discovery(), rows, [valuation()])
        financial = result['companies'][0]['financial']
        self.assertEqual(financial['styles'], ['loss_watchlist'])
        self.assertGreaterEqual(financial['improvement_signal_count'], 2)
        self.assertIsNone(financial['metrics']['core_profit_cagr_2y'])

    def test_only_one_sourced_year_still_exposes_loss_without_classification(self):
        rows = annual_history()[-1:]
        rows[0]['net_income_parent'] = -1
        result = attach_fundamentals(discovery(), rows)
        financial = result['companies'][0]['financial']
        self.assertEqual(financial['status'], 'data_review')
        self.assertEqual(financial['styles'], ['loss_watchlist'])
        self.assertEqual(financial['checks'], {})

    def test_future_loss_does_not_replace_visible_profit(self):
        rows = annual_history()
        future = copy.deepcopy(rows[-1])
        future.update(available_at='2026-06-01', revision_id='1', net_income_parent=-100)
        result = attach_fundamentals(discovery(), rows + [future])
        self.assertFalse(result['companies'][0]['financial']['loss_making'])

    def test_unknown_audit_prevents_all_positive_style_lists(self):
        rows = annual_history()
        rows[0]['audit_opinion'] = 'qualified'
        result = attach_fundamentals(discovery(), rows, [valuation()])
        self.assertEqual(result['companies'][0]['financial']['styles'], ['unclassified'])

    def test_market_or_sector_mismatch_never_joins_financials(self):
        for key, value in [('market', 'HK'), ('sector', 'Technology')]:
            query = discovery()
            query['companies'][0][key] = value
            result = attach_fundamentals(query, annual_history(), [valuation()])
            self.assertEqual(result['companies'][0]['financial']['status'], 'data_review')
            self.assertEqual(result['companies'][0]['financial']['metrics'], {})

    def test_relative_value_uses_unmatched_same_sector_peers(self):
        rows, prices = [], []
        for index in range(6):
            ticker = f'{600001 + index}'
            rows.extend(annual_history(ticker))
            prices.append(valuation(ticker, market_cap=100 + index * 100))
        result = attach_fundamentals(discovery(), rows, prices)
        financial = result['companies'][0]['financial']
        self.assertEqual(financial['peer_count'], 6)
        self.assertIn('relative_value', financial['styles'])
        result = attach_fundamentals(discovery(), rows[:3], prices[:1])
        self.assertNotIn('relative_value', result['companies'][0]['financial']['styles'])

    def test_missing_debt_unknown_cash_or_stale_data_not_quality(self):
        for field, value in [('total_debt', None), ('cash', None)]:
            rows = annual_history()
            rows[-1][field] = value
            result = attach_fundamentals(discovery(), rows)
            self.assertEqual(result['companies'][0]['financial']['styles'], ['unclassified'])
        query = discovery()
        query['as_of'] = '2028-05-01'
        result = attach_fundamentals(query, annual_history())
        self.assertEqual(result['companies'][0]['financial']['status'], 'data_review')

    def test_independent_growth_and_exact_quality_threshold(self):
        result = attach_fundamentals(discovery(), annual_history(), config={'quality_min_roe': 20 / 65})
        self.assertIn('quality_growth', result['companies'][0]['financial']['styles'])
        result = attach_fundamentals(discovery(), annual_history(), config={'quality_min_roe': 20 / 65 + .001})
        self.assertNotIn('quality_growth', result['companies'][0]['financial']['styles'])

    def test_overflow_change_is_unknown_not_improvement(self):
        rows = annual_history()
        for row, profit in zip(rows[-2:], [-1e308, 1e308]):
            row.update(net_income=profit, net_income_parent=profit, core_income_parent=profit, operating_cash_flow=profit)
        result = attach_fundamentals(discovery(), rows)
        json.dumps(result, allow_nan=False)
        checks = result['companies'][0]['financial']['checks']['operating_improvement']
        self.assertIsNone(checks[0]['actual'])
        self.assertIsNone(checks[2]['passed'])

    def test_arguments_not_mutated_and_bad_settings_rejected(self):
        query, rows = discovery(), annual_history()
        before = copy.deepcopy((query, rows))
        attach_fundamentals(query, rows)
        self.assertEqual((query, rows), before)
        for cfg in ({'unknown': 1}, {'value_min_peers': 1}, {'value_min_percentile': 101},
                    {'quality_min_roe': float('nan')}, {'improvement_min_signals': True}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                attach_fundamentals(query, rows, config=cfg)


if __name__ == '__main__':
    unittest.main()
