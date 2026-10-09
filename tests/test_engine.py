"""Small in-memory regression fixtures for the annual research screener.

These verify accounting, availability, and comparability rules. They do not
claim that a score predicts stock returns.
"""
from __future__ import annotations

import copy
import math
from datetime import date
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine


AS_OF = date(2026, 5, 1)


def annual_history(ticker="600001", market="A", sector="Industrials", currency="CNY"):
    rows = []
    for i, year in enumerate((2023, 2024, 2025)):
        rows.append({
            "ticker": ticker, "name": "Synthetic fixture", "market": market,
            "sector": sector, "currency": currency, "scope": "consolidated",
            "period_start": f"{year}-01-01", "period_end": f"{year}-12-31",
            "available_at": f"{year + 1}-03-31", "revision_id": "0",
            "source_url": f"https://example.invalid/{ticker}/{year}",
            "filing_id": f"{ticker}-{year}", "unit_scale": "1",
            "net_income": (10, 12, 20)[i],
            "net_income_parent": (10, 12, 20)[i],
            "core_income_parent": (9, 11, 18)[i],
            "operating_cash_flow": (12, 16, 25)[i],
            "total_assets": (100, 110, 120)[i],
            "equity_parent": (50, 60, 70)[i],
            "current_assets": (30, 40, 55)[i],
            "current_liabilities": (20, 22, 25)[i],
            "long_term_debt": (20, 18, 16)[i],
            "total_debt": 30, "cash": 10,
            "revenue": (100, 125, 160)[i], "cogs": (60, 70, 85)[i],
            "capex": 5, "ebit": (15, 20, 30)[i], "interest_expense": 2,
            "equity_issued": "0", "audit_opinion": "unqualified",
        })
    return rows


def valuation(ticker="600001", currency="CNY", **changes):
    row = {
        "ticker": ticker, "snapshot_date": AS_OF.isoformat(),
        "available_at": AS_OF.isoformat(), "revision_id": "0",
        "market_cap": 200, "unit_scale": 1, "currency": currency,
        "cap_scope": "total_company",
        "source_url": f"https://example.invalid/cap/{ticker}",
    }
    row.update(changes)
    return row


def result(rows=None, values=None, as_of=AS_OF):
    report = engine.screen(
        annual_history() if rows is None else rows,
        [valuation()] if values is None else values, as_of,
    )
    return report, {r["ticker"]: r for r in report["companies"]}


class StatementValidationTests(unittest.TestCase):
    def test_unit_conversion_retains_missing_values(self):
        row = annual_history()[-1]
        row.update(unit_scale=1_000_000, capex="", equity_issued="")
        parsed = engine.parse_statement(row)
        self.assertEqual(parsed["net_income"], 20_000_000)
        self.assertEqual(parsed["unit_scale"], 1)
        self.assertIsNone(parsed["capex"])
        self.assertIsNone(parsed["equity_issued"])
        self.assertEqual(row["net_income"], 20, "normalization must not mutate input")

    def test_nonfinite_inputs_are_rejected(self):
        for invalid in ("NaN", "inf", "-inf", float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                row = annual_history()[-1]
                row["net_income"] = invalid
                with self.assertRaisesRegex(ValueError, "NaN/inf"):
                    engine.parse_statement(row)

    def test_invalid_signs_are_rejected_but_losses_are_allowed(self):
        for field in ("total_assets", "revenue", "cash", "total_debt", "capex", "interest_expense"):
            with self.subTest(field=field):
                row = annual_history()[-1]
                row[field] = -1
                with self.assertRaises(ValueError):
                    engine.parse_statement(row)
        row = annual_history()[-1]
        row["net_income"] = -10
        self.assertEqual(engine.parse_statement(row)["net_income"], -10)

    def test_invalid_units_and_conversion_overflow_are_rejected(self):
        for scale in (0, -1, "", "NaN"):
            with self.subTest(scale=scale):
                row = annual_history()[-1]
                row["unit_scale"] = scale
                with self.assertRaises(ValueError):
                    engine.parse_statement(row)
        row = annual_history()[-1]
        row.update(unit_scale=1e308, total_assets=1e308)
        with self.assertRaisesRegex(ValueError, "overflow"):
            engine.parse_statement(row)

    def test_quarterly_or_unconsolidated_statements_are_rejected(self):
        row = annual_history()[-1]
        row["period_start"] = "2025-10-01"
        with self.assertRaisesRegex(ValueError, "full annual"):
            engine.parse_statement(row)
        row = annual_history()[-1]
        row["scope"] = "parent_only"
        with self.assertRaisesRegex(ValueError, "consolidated"):
            engine.parse_statement(row)


class AccountingMetricTests(unittest.TestCase):
    def test_missing_core_profit_is_data_review_not_bad_company(self):
        rows = annual_history()
        rows[-1]['core_income_parent'] = ''
        _, companies = result(rows)
        self.assertEqual(companies['600001']['status'], 'data_review')
        self.assertIsNone(companies['600001']['metrics']['core_earnings_yield'])

    def test_known_loss_or_negative_cfo_is_excluded_not_missing_data(self):
        for field in ('net_income', 'net_income_parent', 'core_income_parent', 'operating_cash_flow', 'equity_parent'):
            with self.subTest(field=field):
                rows = annual_history()
                rows[-1][field] = -1
                _, companies = result(rows)
                self.assertEqual(companies['600001']['status'], 'excluded')
                self.assertTrue(any(field in reason for reason in companies['600001']['reasons']))

    def test_known_metrics_and_nine_signals(self):
        old, prior, current = map(engine.parse_statement, annual_history())
        metrics, signals = engine.compute_metrics(old, prior, current, {"market_cap_reporting": 200})
        self.assertAlmostEqual(metrics["roe"], 20 / 65)
        self.assertAlmostEqual(metrics["roa"], 20 / 110)
        self.assertAlmostEqual(metrics["cash_conversion"], 25 / 20)
        self.assertAlmostEqual(metrics["net_debt_to_cfo"], 20 / 25)
        self.assertAlmostEqual(metrics["core_earnings_yield"], 18 / 200)
        self.assertAlmostEqual(metrics["fcf_yield"], 20 / 200)
        self.assertAlmostEqual(metrics["revenue_cagr_2y"], math.sqrt(1.6) - 1)
        self.assertEqual(set(signals), set(engine.SIGNALS))
        self.assertTrue(all(signal is True for signal in signals.values()))

    def test_missing_f_score_input_is_unknown_and_requires_review(self):
        rows = annual_history()
        rows[-1]["equity_issued"] = ""
        _, companies = result(rows)
        company = companies["600001"]
        self.assertIsNone(company["signals"]["no_equity_issuance"])
        self.assertIsNone(company["f_score"])
        self.assertEqual(company["f_score_known"], 8)
        self.assertEqual(company["f_score_lower_bound"], 8)
        self.assertEqual(company["status"], "data_review")

    def test_missing_financial_item_is_not_zero(self):
        rows = annual_history()
        rows[-1]["operating_cash_flow"] = ""
        _, companies = result(rows)
        company = companies["600001"]
        self.assertIsNone(company["metrics"]["cash_conversion"])
        self.assertIsNone(company["metrics"]["fcf"])
        self.assertIsNone(company["metrics"]["net_debt_to_cfo"])
        self.assertIsNone(company["signals"]["positive_cfo"])
        self.assertEqual(company["status"], "data_review")

    def test_zero_interest_is_missing_not_infinite(self):
        rows = annual_history()
        rows[-1]["interest_expense"] = 0
        _, companies = result(rows)
        self.assertIsNone(companies["600001"]["metrics"]["interest_cover"])
        self.assertIsNone(companies["600001"]["score"])

    def test_negative_growth_base_is_not_manufactured_cagr(self):
        rows = annual_history()
        rows[0]["core_income_parent"] = -1
        _, companies = result(rows)
        self.assertIsNone(companies["600001"]["metrics"]["core_profit_cagr_2y"])

    def test_derived_metrics_cannot_be_infinite(self):
        rows = annual_history()
        rows[-1].update(ebit=1e308, interest_expense=1e-308)
        _, companies = result(rows)
        for name, value in companies["600001"]["metrics"].items():
            with self.subTest(metric=name):
                self.assertTrue(value is None or math.isfinite(value),
                                "finite source values must not produce an infinite scored metric")


class AvailabilityAndComparabilityTests(unittest.TestCase):
    def test_unordered_years_are_sorted_deterministically(self):
        _, baseline = result()
        _, reordered = result(list(reversed(annual_history())))
        self.assertEqual(reordered["600001"]["metrics"], baseline["600001"]["metrics"])
        self.assertEqual(reordered["600001"]["sources"], baseline["600001"]["sources"])

    def test_future_restatement_cannot_rewrite_current_metrics(self):
        rows = annual_history()
        future = dict(rows[-1], revision_id="1", available_at="2026-06-01", net_income=999)
        report, companies = result(rows + [future])
        self.assertAlmostEqual(companies["600001"]["metrics"]["roa"], 20 / 110)
        self.assertTrue(any(a["reason"] == "not_yet_available" for a in report["audit"]))
        future["available_at"] = "2026-04-01"
        _, revised = result(rows + [future])
        self.assertAlmostEqual(revised["600001"]["metrics"]["roa"], 999 / 110)

    def test_undisclosed_current_year_requires_review(self):
        rows = annual_history()
        rows[-1]["available_at"] = "2026-05-02"
        report, companies = result(rows)
        self.assertEqual(companies["600001"]["status"], "data_review")
        self.assertEqual(companies["600001"]["metrics"], {})
        self.assertEqual(report["audit"][0]["reason"], "not_yet_available")

    def test_same_revision_conflict_is_audited_and_not_scored(self):
        rows = annual_history()
        conflict = dict(rows[-1], net_income=21)
        report, companies = result(rows + [conflict])
        self.assertEqual(companies["600001"]["status"], "data_review")
        self.assertEqual(companies["600001"]["metrics"], {})
        self.assertTrue(any("conflicting" in a["reason"] for a in report["audit"]))

    def test_duplicate_identical_rows_are_idempotent(self):
        _, baseline = result()
        rows = annual_history()
        report, duplicated = result(rows + [dict(rows[-1])])
        self.assertEqual(report["audit"], [])
        self.assertEqual(duplicated["600001"]["metrics"], baseline["600001"]["metrics"])

    def test_nonconsecutive_annual_periods_require_review(self):
        rows = annual_history()
        rows[1].update(period_start="2022-01-01", period_end="2022-12-31", available_at="2023-03-31")
        _, companies = result(rows)
        self.assertEqual(companies["600001"]["status"], "data_review")
        self.assertTrue(any("not consecutive" in reason for reason in companies["600001"]["reasons"]))

    def test_reporting_currency_change_requires_comparable_history(self):
        rows = annual_history()
        rows[0]["currency"] = "HKD"
        _, companies = result(rows)
        self.assertEqual(companies["600001"]["status"], "data_review")
        self.assertTrue(any("currency changed" in reason for reason in companies["600001"]["reasons"]))

    def test_specialist_sectors_exit_without_generic_score(self):
        for sector in ("Banks", "Insurance", "REITs", "银行"):
            with self.subTest(sector=sector):
                _, companies = result(annual_history(sector=sector))
                self.assertEqual(companies["600001"]["status"], "specialist_review")
                self.assertEqual(companies["600001"]["metrics"], {})
                self.assertIsNone(companies["600001"]["score"])

    def test_stale_financial_history_requires_review(self):
        _, companies = result(as_of=date(2027, 7, 5))
        self.assertEqual(companies["600001"]["status"], "data_review")
        self.assertTrue(any("financial data is stale" in reason for reason in companies["600001"]["reasons"]))


class ValuationAndPeersTests(unittest.TestCase):
    def test_market_cap_must_cover_all_share_classes(self):
        _, companies = result(values=[valuation(cap_scope="listed_class")])
        company = companies["600001"]
        self.assertEqual(company["status"], "data_review")
        self.assertIsNone(company["metrics"]["core_earnings_yield"])
        self.assertTrue(any("all share classes" in warning for warning in company["warnings"]))

    def test_dated_sourced_fx_and_unit_scale_convert_cap(self):
        values = [valuation(currency="HKD", market_cap=100, unit_scale=2,
                            fx_to_reporting=.9, fx_date=AS_OF.isoformat(),
                            fx_source_url="https://example.invalid/fx")]
        _, companies = result(values=values)
        self.assertAlmostEqual(companies["600001"]["metrics"]["core_earnings_yield"], 18 / 180)

    def test_missing_future_stale_or_unsourced_fx_requires_review(self):
        base = valuation(currency="HKD", fx_to_reporting=.9, fx_date=AS_OF.isoformat(),
                         fx_source_url="https://example.invalid/fx")
        for change in ({"fx_to_reporting": ""}, {"fx_date": "2026-05-02"},
                       {"fx_date": "2026-04-01"}, {"fx_source_url": ""},
                       {"fx_to_reporting": -1}, {"fx_to_reporting": "inf"}):
            with self.subTest(change=change):
                _, companies = result(values=[dict(base, **change)])
                self.assertEqual(companies["600001"]["status"], "data_review")
                self.assertIsNone(companies["600001"]["metrics"]["core_earnings_yield"])

    def test_future_valuation_and_conflicting_valuation_are_rejected(self):
        future = valuation(available_at="2026-05-02")
        report, companies = result(values=[future])
        self.assertEqual(companies["600001"]["status"], "data_review")
        self.assertEqual(report["valuation_audit"][0]["reason"], "not_yet_available")
        report, companies = result(values=[valuation(), valuation(market_cap=201)])
        self.assertEqual(companies["600001"]["status"], "data_review")
        self.assertTrue(any("conflicting" in a["reason"] for a in report["valuation_audit"]))

    def test_stale_and_invalid_market_caps_do_not_enter_value_scores(self):
        for changes in ({"snapshot_date": "2026-04-01"}, {"market_cap": 0},
                        {"market_cap": -1}, {"market_cap": "NaN"}, {"unit_scale": 0}):
            with self.subTest(changes=changes):
                _, companies = result(values=[valuation(**changes)])
                self.assertEqual(companies["600001"]["status"], "data_review")
                self.assertIsNone(companies["600001"]["metrics"]["fcf_yield"])

    def test_five_equal_peers_are_neutral_and_not_automatic_candidates(self):
        rows, values = [], []
        for ticker in ("A1", "A2", "A3", "A4", "A5"):
            rows.extend(annual_history(ticker))
            values.append(valuation(ticker))
        _, companies = result(rows, values)
        for company in companies.values():
            self.assertEqual(company["peer_count"], 5)
            self.assertEqual(company["score"], 50)
            self.assertEqual(company["status"], "watchlist")

    def test_markets_and_sectors_are_separate_peer_cohorts(self):
        rows, values = [], []
        for market, sector in (("A", "Industrials"), ("HK", "Industrials"), ("A", "Consumer")):
            for i in range(4):
                ticker = f"{market}-{sector}-{i}"
                rows.extend(annual_history(ticker, market, sector))
                values.append(valuation(ticker))
        _, companies = result(rows, values)
        for company in companies.values():
            self.assertEqual(company["peer_count"], 4)
            self.assertIsNone(company["score"])
            self.assertEqual(company["status"], "watchlist")

    def test_each_metric_requires_five_observed_peer_values(self):
        rows, values = [], []
        for i in range(5):
            ticker = f"A{i}"
            history = annual_history(ticker)
            if i == 0:
                history[-1]["interest_expense"] = ""
            rows.extend(history)
            values.append(valuation(ticker))
        _, companies = result(rows, values)
        for company in companies.values():
            self.assertEqual(company["peer_count"], 5)
            self.assertIsNone(company["blocks"]["balance"]["score"])
            self.assertIsNone(company["score"])


class RankingAndConfigurationTests(unittest.TestCase):
    def test_midrank_ties_and_direction(self):
        self.assertEqual(engine.percentile(2, [1, 2, 2, 4]), 50)
        self.assertEqual(engine.percentile(4, [1, 2, 2, 4]), 100)
        self.assertEqual(engine.percentile(4, [1, 2, 2, 4], higher=False), 0)
        self.assertEqual(engine.percentile(9, [9, 9, 9]), 50)
        self.assertEqual(engine.percentile(9, [9]), 50)

    def test_default_configuration_is_valid(self):
        config = engine.config_from(None)
        self.assertEqual(set(config["weights"]), set(engine.BLOCK_FIELDS))
        self.assertAlmostEqual(sum(config["weights"].values()), 1)
        self.assertGreaterEqual(config["min_peer_count"], 5)

    def test_default_configuration_returns_independent_nested_values(self):
        # Preserve defaults even if an older implementation exposes shared objects.
        original = copy.deepcopy(engine.DEFAULT_CONFIG)
        try:
            config = engine.config_from(None)
            config["weights"]["quality"] = 0
            config["excluded_sectors"].append("fixture-only")
            fresh = engine.config_from(None)
            self.assertEqual(fresh, original,
                             "one caller's configuration edits must not pollute other runs")
        finally:
            engine.DEFAULT_CONFIG.clear()
            engine.DEFAULT_CONFIG.update(original)


if __name__ == "__main__":
    unittest.main()
