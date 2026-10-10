import copy
import unittest

from readiness import apply_theme_business_policy, assess_readiness, validate_policy
from theme_search import discover_companies
from theme_financials import attach_fundamentals
from tests.test_engine import annual_history, valuation


def policy():
    return dict(expected_tickers=['600001'], max_universe_age_days=7,
                max_document_age_days=365, min_document_coverage=1., min_operating_coverage=1.,
                min_valuation_coverage=1., max_invalid_records=0, max_identity_conflicts=0,
                require_document_hashes=True, require_known_date_basis=True)


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.companies = [dict(ticker='600001', name='示例', market='A', sector='Industrials',
                               scope='manufacturing', universe_as_of='2026-05-01')]
        self.docs = [dict(document_id='one', ticker='600001', source_url='https://example.org/a.pdf',
                          source_type='annual_report', available_at='2026-04-01', page=1,
                          text='本公司生产机器人产品。', date_basis='official_release', source_sha256='a' * 64)]

    def report(self, rows=None, values=None):
        found = discover_companies(self.companies, self.docs, '机器人', '2026-05-01')
        return attach_fundamentals(found, annual_history() if rows is None else rows,
                                    [valuation()] if values is None else values)

    def test_passing_data_checks_does_not_claim_deployment_approval(self):
        result = assess_readiness(self.report(), self.companies, self.docs, policy())
        self.assertEqual(result['status'], 'data_gates_passed')
        self.assertFalse(result['production_approved'])

    def test_no_policy_is_not_assessed_not_approved(self):
        self.assertEqual(assess_readiness(self.report(), self.companies, self.docs)['status'], 'not_assessed')

    def test_missing_expected_company_cannot_hide_behind_good_matches(self):
        cfg = policy()
        cfg['expected_tickers'].append('600002')
        result = assess_readiness(self.report(), self.companies, self.docs, cfg)
        self.assertEqual(result['scope']['expected_companies'], 2)
        self.assertIn('declared_universe_present', result['reasons'])
        self.assertIn('document_coverage', result['reasons'])

    def test_unknown_financials_and_missing_valuations_block(self):
        result = assess_readiness(self.report(rows=[], values=[]), self.companies, self.docs, policy())
        self.assertIn('operating_coverage', result['reasons'])
        self.assertIn('valuation_coverage', result['reasons'])
        result = assess_readiness(self.report(values=[]), self.companies, self.docs, policy())
        self.assertNotIn('operating_coverage', result['reasons'])
        self.assertIn('valuation_coverage', result['reasons'])

    def test_stale_directory_and_stale_evidence_block(self):
        self.companies[0]['universe_as_of'] = '2026-01-01'
        self.docs[0]['available_at'] = '2024-01-01'
        result = assess_readiness(self.report(), self.companies, self.docs, policy())
        self.assertIn('universe_freshness', result['reasons'])
        self.assertIn('document_coverage', result['reasons'])

    def test_bad_source_and_ambiguous_identity_cannot_count_toward_coverage(self):
        duplicate = dict(self.docs[0], text='另一段冲突原文')
        self.docs.append(duplicate)
        result = assess_readiness(self.report(), self.companies, self.docs, policy())
        self.assertIn('identity_conflicts', result['reasons'])
        self.assertIn('document_coverage', result['reasons'])

    def test_future_records_excluded_without_invalid_record_penalty(self):
        self.docs.append(dict(self.docs[0], document_id='future', available_at='2099-01-01'))
        result = assess_readiness(self.report(), self.companies, self.docs, policy())
        self.assertEqual(result['status'], 'data_gates_passed')

    def test_hash_and_date_basis_requirement_is_explicit(self):
        self.docs[0].pop('source_sha256')
        self.docs[0].pop('date_basis')
        result = assess_readiness(self.report(), self.companies, self.docs, policy())
        self.assertIn('document_coverage', result['reasons'])

    def test_unrelated_fresh_filing_cannot_refresh_old_theme_business(self):
        self.docs[0]['available_at'] = '2023-04-01'
        self.docs.append(dict(self.docs[0], document_id='fresh-unrelated', available_at='2026-04-01',
                              text='公司董事会召开年度会议。'))
        report = self.report()
        before = copy.deepcopy(report['companies'][0]['financial'])
        apply_theme_business_policy(report, policy())
        financial = report['companies'][0]['financial']
        self.assertEqual(financial['styles'], ['unclassified'])
        self.assertEqual(financial['financial_style_candidates'], before['financial_style_candidates'])
        self.assertEqual(financial['checks'], before['checks'])
        result = assess_readiness(report, self.companies, self.docs, policy())
        self.assertEqual(result['status'], 'blocked')
        self.assertNotIn('document_coverage', result['reasons'])
        self.assertIn('theme_business_freshness', result['reasons'])
        check = next(c for c in result['checks'] if c['name'] == 'theme_business_freshness')
        self.assertEqual(check['affected_tickers'], ['600001'])

    def test_theme_support_must_meet_its_own_provenance_requirements(self):
        for missing in ('source_sha256', 'date_basis'):
            with self.subTest(missing=missing):
                original = copy.deepcopy(self.docs)
                self.docs.append(dict(self.docs[0], document_id='fresh-unrelated', text='公司召开董事会会议。'))
                self.docs[0].pop(missing)
                report = self.report()
                apply_theme_business_policy(report, policy())
                result = assess_readiness(report, self.companies, self.docs, policy())
                self.assertNotIn('document_coverage', result['reasons'])
                self.assertIn('theme_business_freshness', result['reasons'])
                self.assertEqual(report['companies'][0]['financial']['styles'], ['unclassified'])
                self.docs = original

    def test_no_policy_does_not_invent_enterprise_freshness_threshold(self):
        self.docs[0]['available_at'] = '2023-04-01'
        report = self.report()
        styles = list(report['companies'][0]['financial']['styles'])
        apply_theme_business_policy(report)
        self.assertEqual(report['companies'][0]['financial']['styles'], styles)
        self.assertEqual(report['companies'][0]['theme_business_freshness']['status'], 'not_assessed')

    def test_stale_theme_does_not_remove_loss_observation(self):
        self.docs[0]['available_at'] = '2023-04-01'
        rows = annual_history()
        rows[-1]['net_income_parent'] = -10
        report = self.report(rows=rows)
        apply_theme_business_policy(report, policy())
        self.assertEqual(report['companies'][0]['financial']['styles'], ['loss_watchlist'])
        self.assertEqual(report['style_lists']['loss_watchlist'], ['600001'])

    def test_new_plan_cannot_refresh_old_operating_support(self):
        self.docs[0]['available_at'] = '2023-04-01'
        self.docs.append(dict(self.docs[0], document_id='plan', available_at='2026-04-01',
                              text='本公司计划开发新一代机器人产品。'))
        report = self.report()
        apply_theme_business_policy(report, policy())
        self.assertEqual(report['companies'][0]['theme_business_freshness']['status'], 'blocked')

    def test_low_scoring_fresh_support_survives_evidence_display_budget(self):
        self.docs[0].update(available_at='2023-04-01', text='本公司生产机器人、机器人、机器人产品。')
        self.docs.append(dict(self.docs[0], document_id='fresh-low-score', available_at='2026-04-01',
                              text='本公司生产机器人产品并交付' + '，经营情况良好' * 35 + '。'))
        found = discover_companies(self.companies, self.docs, '机器人', '2026-05-01',
                                   config={'max_evidence_per_company': 1})
        self.assertEqual(found['companies'][0]['evidence'][0]['document_id'], 'one')
        support = found['companies'][0]['business_support_evidence'][0]
        self.assertEqual(support['document_id'], 'fresh-low-score')
        self.assertEqual(support['exact_excerpt'], self.docs[1]['text'][support['char_start']:support['char_end']])
        report = attach_fundamentals(found, annual_history(), [valuation()])
        apply_theme_business_policy(report, policy())
        self.assertIn('quality_growth', report['companies'][0]['financial']['styles'])
        self.assertEqual(assess_readiness(report, self.companies, self.docs, policy())['status'], 'data_gates_passed')

    def test_new_unhashed_support_does_not_hide_still_fresh_verified_support(self):
        self.docs.append(dict(self.docs[0], document_id='newer-unhashed', available_at='2026-04-20'))
        self.docs[-1].pop('source_sha256')
        report = self.report()
        apply_theme_business_policy(report, policy())
        self.assertEqual(report['companies'][0]['theme_business_freshness']['qualifying_document_ids'], ['one'])
        self.assertEqual(assess_readiness(report, self.companies, self.docs, policy())['status'], 'data_gates_passed')

    def test_empty_invalid_or_unknown_policy_is_rejected(self):
        configs = [{}, dict(policy(), expected_tickers=[]), dict(policy(), min_document_coverage=True),
                   dict(policy(), min_document_coverage=0), dict(policy(), max_invalid_records=-1),
                   dict(policy(), typo=1), dict(policy(), min_operating_coverage=float('inf'))]
        for cfg in configs:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                validate_policy(cfg)


if __name__ == '__main__':
    unittest.main()
