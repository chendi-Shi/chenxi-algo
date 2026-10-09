"""Search correctness, bounded work, and financial eligibility regressions."""
import copy
import unittest

import engine
from search import search_configs
from test_engine import AS_OF, annual_history, valuation


def fixture():
    rows, values = [], []
    for index in range(6):
        ticker = f'SEARCH-{index}'
        rows.extend(annual_history(ticker))
        values.append(valuation(ticker, market_cap=150 + 25 * index))
    config = engine.config_from(None)
    config['candidate_score'] = 50
    return rows, values, config


class SearchTests(unittest.TestCase):
    def run_search(self, rows=None, values=None, config=None, **kwargs):
        original_rows, original_values, original_config = fixture()
        args = {'market': 'A', 'sector': 'Industrials', 'target_count': 2,
                'space': {'min_roe': [.08, .12], 'min_cash_conversion': [.8, 1],
                          'candidate_score': [50, 55]}}
        args.update(kwargs)
        return search_configs(original_rows if rows is None else rows,
            original_values if values is None else values, AS_OF,
            original_config if config is None else config, **args)

    def test_grid_checks_all_unique_configurations_and_matches_target(self):
        result = self.run_search()
        self.assertEqual(result['evaluations'], 8)
        self.assertEqual(result['grid_combinations'], 8)
        self.assertEqual(result['baseline']['candidate_count'], 3)
        self.assertEqual(result['best']['candidate_count'], 2)
        self.assertTrue(result['best']['target_met'])
        self.assertTrue(result['global_optimum_verified'])
        self.assertEqual(result['best']['config']['min_roe'], .08)
        self.assertEqual(result['best']['config']['candidate_score'], 55)

    def test_repeated_axis_values_are_cached_as_one_configuration(self):
        result = self.run_search(space={'candidate_score': [50, 50.0, 55, 55.0]})
        self.assertEqual(result['evaluations'], 2)

    def test_evaluation_budget_is_a_hard_limit(self):
        result = self.run_search(max_evaluations=2)
        self.assertEqual(result['evaluations'], 2)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertFalse(result['exhaustive'])
        self.assertIsNotNone(result['best'])

    def test_wide_beam_matches_exhaustive_grid_on_small_space(self):
        grid = self.run_search()
        beam = self.run_search(method='beam', beam_width=100)
        self.assertEqual(beam['best'], grid['best'])
        self.assertTrue(beam['exhaustive'])
        self.assertFalse(beam['heuristic'])

    def test_narrow_beam_reports_pruning(self):
        result = self.run_search(method='beam', beam_width=1)
        self.assertTrue(result['heuristic'])
        self.assertFalse(result['global_optimum_verified'])

    def test_beam_retains_overfull_branch_that_can_reach_target(self):
        result = self.run_search(method='beam', beam_width=1, target_count=1,
            space={'min_roe': [.08, .35], 'candidate_score': [50, 60]})
        self.assertTrue(result['best']['target_met'])
        self.assertEqual(result['best']['config']['min_roe'], .08)
        self.assertEqual(result['best']['config']['candidate_score'], 60)

    def test_partial_assignments_cannot_be_final_recommendations(self):
        result = self.run_search(method='beam', target_count=1,
            space={'min_roe': [.35], 'candidate_score': [55]})
        self.assertEqual(result['best']['config']['min_roe'], .35)
        self.assertEqual(result['best']['config']['candidate_score'], 55)

    def test_search_cannot_relax_baseline_or_disable_scoring_blocks(self):
        bad_spaces = [
            {'min_roe': [.07]}, {'min_cash_conversion': [.7]},
            {'max_net_debt_to_cfo': [5]}, {'candidate_score': [49]},
            {'weights': [{'quality': .5, 'value': .3, 'growth': .2, 'balance': 0}]},
        ]
        for space in bad_spaces:
            with self.subTest(space=space), self.assertRaises(ValueError):
                self.run_search(space=space)

    def test_invalid_search_inputs_are_rejected(self):
        for space in ({}, {'unknown': [1]}, {'min_roe': []}, {'min_roe': [float('nan')]},
                      {'min_roe': [True]}, {'min_roe': [10 ** 1000]},
                      {'weights': [{'quality': 1}]}):
            with self.subTest(space=str(space)[:100]), self.assertRaises(ValueError):
                self.run_search(space=space)
        for kwargs in ({'target_count': 0}, {'max_evaluations': 0}, {'beam_width': True},
                       {'sector': ''}, {'market': 'US'}, {'method': 'unknown'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_search(**kwargs)

    def test_baseline_review_exclusion_and_unscored_rows_never_become_candidates(self):
        rows, values, config = fixture()
        for ticker, changes in (
            ('REVIEW', {'audit_opinion': ''}),
            ('EXCLUDED', {'total_debt': 400}),
            ('UNSCORED', {'interest_expense': ''}),
        ):
            annual = annual_history(ticker)
            annual[-1].update(changes)
            rows.extend(annual)
            values.append(valuation(ticker, market_cap=1))
        rows.extend(annual_history('BANK', sector='Banks'))
        values.append(valuation('BANK'))
        result = self.run_search(rows, values, config)
        self.assertEqual(result['baseline']['eligible_count'], 6)
        forbidden = {'REVIEW', 'EXCLUDED', 'UNSCORED', 'BANK'}
        for choice in result['alternatives']:
            self.assertFalse(forbidden.intersection(choice['candidates']))

    def test_all_missing_debt_returns_no_search_candidate(self):
        rows, values, config = fixture()
        for row in rows:
            row['total_debt'] = ''
        result = self.run_search(rows, values, config)
        self.assertEqual(result['status'], 'insufficient_verified_data')
        self.assertEqual(result['evaluations'], 1)
        self.assertIsNone(result['best'])
        self.assertEqual(result['alternatives'], [])

    def test_future_financial_and_valuation_versions_cannot_change_search(self):
        rows, values, config = fixture()
        base = self.run_search(rows, values, config)
        rows.append(dict(rows[-1], available_at='2026-05-02', revision_id='invalid',
                         period_end='invalid', net_income='NaN'))
        values.append(valuation('SEARCH-5', available_at='2026-05-02',
                                snapshot_date='invalid', revision_id='invalid', market_cap=1))
        self.assertEqual(base, self.run_search(rows, values, config))

    def test_other_markets_do_not_change_same_cohort_search(self):
        rows, values, config = fixture()
        base = self.run_search(rows, values, config)
        for i in range(6):
            rows.extend(annual_history(f'HK-{i}', market='HK'))
            values.append(valuation(f'HK-{i}', market_cap=1))
        self.assertEqual(base, self.run_search(rows, values, config))

    def test_search_is_deterministic_and_does_not_mutate_inputs(self):
        rows, values, config = fixture()
        original = copy.deepcopy((rows, values, config))
        first = self.run_search(rows, values, config)
        second = self.run_search(rows, values, config)
        self.assertEqual(first, second)
        self.assertEqual((rows, values, config), original)

    def test_unreachable_target_is_reported_without_inventing_companies(self):
        result = self.run_search(target_count=100)
        self.assertFalse(result['target_feasible_in_eligible_pool'])
        self.assertFalse(result['best']['target_met'])
        self.assertLessEqual(result['best']['candidate_count'], result['baseline']['eligible_count'])


if __name__ == '__main__':
    unittest.main()
