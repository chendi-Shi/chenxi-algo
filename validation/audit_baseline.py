"""Economic boundary probes and preference sensitivity; synthetic, not efficacy proof.

This does not change engine.py or generate fixture financial files. Run with the
project's Python runtime. The JSON includes the audited engine content hash.
"""
from __future__ import annotations

import copy
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine

AS_OF = date(2026, 5, 1)


def history(ticker="PROBE-1", market="A"):
    rows = []
    for i, year in enumerate((2023, 2024, 2025)):
        rows.append({
            "ticker": ticker, "name": "SYNTHETIC economic probe", "market": market,
            "sector": "Consumer", "currency": "CNY", "scope": "consolidated",
            "period_start": f"{year}-01-01", "period_end": f"{year}-12-31",
            "available_at": f"{year + 1}-03-31", "revision_id": 0,
            "source_url": f"https://example.invalid/probe/{ticker}/{year}",
            "filing_id": f"probe-{ticker}-{year}", "unit_scale": 1,
            "net_income": (10, 12, 20)[i], "net_income_parent": (10, 12, 20)[i],
            "core_income_parent": (9, 11, 18)[i], "operating_cash_flow": (12, 16, 25)[i],
            "total_assets": (100, 110, 120)[i], "equity_parent": (50, 60, 70)[i],
            "current_assets": (30, 40, 55)[i], "current_liabilities": (20, 22, 25)[i],
            "long_term_debt": (20, 18, 16)[i], "total_debt": 30, "cash": 10,
            "revenue": (100, 125, 160)[i], "cogs": (60, 70, 85)[i],
            "capex": 5, "ebit": (15, 20, 30)[i], "interest_expense": 2,
            "equity_issued": 0, "audit_opinion": "unqualified",
        })
    return rows


def valuation(ticker):
    return {"ticker": ticker, "snapshot_date": AS_OF.isoformat(),
            "available_at": AS_OF.isoformat(), "revision_id": 0,
            "market_cap": 200, "unit_scale": 1, "currency": "CNY",
            "cap_scope": "total_company", "source_url": "https://example.invalid/probe/cap"}


def group(transform=None, count=5, market="A", config=None):
    rows, values = [], []
    for index in range(count):
        ticker = f"PROBE-{index}"
        annual = history(ticker, market)
        if transform:
            transform(annual, index)
        rows.extend(annual)
        values.append(valuation(ticker))
    return engine.screen(rows, values, AS_OF, config)


def compact(report):
    return {"counts": dict(Counter(c["status"] for c in report["companies"])),
            "scored_count": sum(c["score"] is not None for c in report["companies"]),
            "fscore_distribution": dict(Counter(str(c["f_score"]) for c in report["companies"])),
            "companies": [{k: c.get(k) for k in (
                "ticker", "status", "score", "f_score", "f_score_known", "peer_count",
                "metrics", "reasons", "warnings")} for c in report["companies"][:1]]}


def priority_summary(companies):
    """Only a defined score can establish a research-priority ordering."""
    scored = [c for c in companies if c["score"] is not None]
    return {"scored_count": len(scored),
            "top_two": [c["ticker"] for c in scored[:2]],
            "scores": {c["ticker"]: c["score"] for c in companies},
            "candidates": [c["ticker"] for c in companies if c["status"] == "candidate"]}


def stress_probes():
    def mature(rows, index):
        constant = {"net_income": 20, "net_income_parent": 20, "core_income_parent": 18,
                    "operating_cash_flow": 25, "total_assets": 120, "equity_parent": 70,
                    "current_assets": 55, "current_liabilities": 25, "long_term_debt": 0,
                    "total_debt": 0, "cash": 30, "revenue": 160, "cogs": 85,
                    "capex": 5, "ebit": 30, "interest_expense": 2}
        for row in rows:
            row.update(constant)

    def zero_interest(rows, index):
        for row in rows:
            row.update(long_term_debt=0, total_debt=0, interest_expense=0)

    def one_zero_interest(rows, index):
        if index == 0:
            rows[-1]["interest_expense"] = 0

    def zero_current_liabilities(rows, index):
        for row in rows:
            row["current_liabilities"] = 0

    def negative_opening_equity(rows, index):
        rows[-2]["equity_parent"] = -60

    def high_conversion(rows, index):
        rows[-1]["operating_cash_flow"] = 2000

    def core_profit_above_total(rows, index):
        rows[-1]["core_income_parent"] = 100

    def missing_hk_adjusted_profit(rows, index):
        for row in rows:
            row["core_income_parent"] = ""

    def previous_year_only(rows, index):
        for row in rows:
            for field in ("period_start", "period_end", "available_at"):
                row[field] = str(int(row[field][:4]) - 1) + row[field][4:]

    transforms = {"healthy_baseline": None,
                  "mature_stable_net_cash": mature,
                  "all_debt_free_zero_interest": zero_interest,
                  "one_zero_interest_in_five_peers": one_zero_interest,
                  "zero_current_liabilities": zero_current_liabilities,
                  "negative_opening_equity_positive_current": negative_opening_equity,
                  "extreme_cash_conversion": high_conversion,
                  "core_profit_five_times_total": core_profit_above_total,
                  "latest_annual_missing_after_filing_window": previous_year_only}
    probes = {name: compact(group(transform)) for name, transform in transforms.items()}
    zero_balance_weight = engine.config_from(None)
    zero_balance_weight["weights"] = {"quality": .5, "value": .3, "growth": .2, "balance": 0}
    probes["zero_interest_despite_disabled_balance_weight"] = compact(group(zero_interest, config=zero_balance_weight))
    probes["hong_kong_no_harmonized_core_profit"] = compact(group(missing_hk_adjusted_profit, market="HK"))
    return probes


def preferences():
    rows = engine.load_csv(ROOT / "examples" / "statements.csv")
    values = engine.load_csv(ROOT / "examples" / "valuations.csv")
    profiles = {
        "default": {"quality": .4, "value": .25, "growth": .2, "balance": .15},
        "equal": {"quality": .25, "value": .25, "growth": .25, "balance": .25},
        "quality": {"quality": .6, "value": .15, "growth": .1, "balance": .15},
        "value": {"quality": .2, "value": .5, "growth": .15, "balance": .15},
        "growth": {"quality": .2, "value": .15, "growth": .5, "balance": .15},
    }
    results = {}
    candidate_sets = {}
    for name, weights in profiles.items():
        config = engine.config_from(None)
        config["weights"] = weights
        result = engine.screen(rows, values, date(2026, 10, 9), config)
        candidate_sets[name] = {c["ticker"] for c in result["companies"] if c["status"] == "candidate"}
        results[name] = {"weights": weights, "counts": dict(Counter(c["status"] for c in result["companies"])),
                         "candidates": sorted(candidate_sets[name])}
    baseline = candidate_sets["default"]
    for name, chosen in candidate_sets.items():
        union = baseline | chosen
        results[name]["candidate_jaccard_vs_default"] = len(baseline & chosen) / len(union) if union else None
    gates = {}
    for threshold in (0, 4, 6, 8):
        config = engine.config_from(None)
        config["min_f_score"] = threshold
        result = engine.screen(rows, values, date(2026, 10, 9), config)
        gates[str(threshold)] = dict(Counter(c["status"] for c in result["companies"]))
    tradeoff_rows, tradeoff_values = [], []
    for index in range(6):
        ticker = f"TRADEOFF-{index}"
        annual = history(ticker)
        growth_rate = .3 - index * .04
        for offset, row in enumerate(annual):
            row["revenue"] = 100 * (1 + growth_rate) ** offset
            row["cogs"] = row["revenue"] * (.6 - .02 * offset)
            row["net_income"] = 10 * (1 + growth_rate) ** offset
            row["net_income_parent"] = row["net_income"]
            row["core_income_parent"] = row["net_income"] * (.7 + .05 * index)
            row["equity_parent"] = row["net_income"] / (.15 + .04 * index)
            row["operating_cash_flow"] = row["net_income"] * (.85 + .15 * index)
            row["total_debt"] = 25 - 3 * index
            # The illustrative debt scopes must be internally consistent in
            # every year, otherwise a missing cohort is not weight sensitivity.
            row["long_term_debt"] = row["total_debt"] * .5
            row["ebit"] = row["net_income"] * 1.3
        v = valuation(ticker)
        v["market_cap"] = annual[-1]["core_income_parent"] / (.1 - .012 * index)
        tradeoff_rows.extend(annual)
        tradeoff_values.append(v)
    tradeoff = {}
    for name, weights in profiles.items():
        config = engine.config_from(None)
        config.update(weights=weights, min_f_score=None)
        result = engine.screen(tradeoff_rows, tradeoff_values, AS_OF, config)
        tradeoff[name] = priority_summary(result["companies"])
    control_config = engine.config_from(None)
    control_config["min_f_score"] = None
    control = priority_summary(engine.screen(tradeoff_rows[:3], tradeoff_values[:1],
                                            AS_OF, control_config)["companies"])
    if control["scored_count"] != 0 or control["top_two"]:
        raise AssertionError("insufficient-peer control must not manufacture a top-two ordering")
    return {"synthetic_demo_weights": results, "synthetic_demo_fscore_gate": gates,
            "synthetic_conflicting_factors": {"note": "quality rises while valuation yields and growth fall; F-score gate disabled to isolate preference sensitivity", "profiles": tradeoff,
                                               "insufficient_peer_control": control}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Write the UTF-8 audit JSON to this file instead of stdout.")
    args = parser.parse_args()
    output = {"data": "synthetic boundary probes and existing synthetic demo; no investment efficacy claim",
              "engine_sha256": hashlib.sha256((ROOT / "engine.py").read_bytes()).hexdigest(),
              "economic_probes": stress_probes(), "preference_sensitivity": preferences()}
    rendered = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
