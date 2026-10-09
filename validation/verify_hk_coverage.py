"""Reconcile selected Li Ning public raw fields, without inventing core earnings."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
MAPPING = {
    "revenue": ("income", "营业额"),
    "net_income_parent": ("income", "股东应占溢利"),
    "total_assets": ("balance", "总资产"),
    "equity_parent": ("balance", "股东权益"),
    "operating_cash_flow": ("cashflow", "经营业务现金净额"),
}
# Original annual-report consolidated figures; its displayed RMB'000 values
# multiplied by 1,000, as independently extracted by the main validation task.
EXPECTED_2024 = {
    "revenue": 28675643000, "net_income_parent": 3012918000,
    "total_assets": 35708406000, "equity_parent": 26103689000,
    "operating_cash_flow": 5267935000,
}
OFFICIAL_URL = "https://www.hkexnews.hk/listedco/listconews/sehk/2025/0429/2025042901719_c.pdf"


def run(from_cache=False):
    if not from_cache:
        frozen = json.loads((ROOT/'hk_coverage.json').read_text(encoding='utf-8'))
        official = json.loads((ROOT/'official_checks.json').read_text(encoding='utf-8'))['companies']['02331']['values']['2024']
        mapping = {'revenue': 'revenue', 'net_income_parent': 'net_profit', 'total_assets': 'total_assets',
                   'equity_parent': 'total_equity', 'operating_cash_flow': 'operating_cash_flow'}
        matches = {field: frozen['years']['2024'][field]['value'] == official[key] for field, key in mapping.items()}
        print(json.dumps({'offline_frozen_fact_checks': matches},ensure_ascii=True))
        return 0 if all(matches.values()) else 1
    envelopes = {kind: json.loads((RAW / f"02331_hk_{kind}.json").read_text(encoding="utf-8"))
                 for kind in ("balance", "income", "cashflow", "summary")}
    out = {
        "ticker": "02331.HK", "name": "Li Ning Company Limited",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_check_source": OFFICIAL_URL,
        "official_check_scope": "Five 2024 consolidated financial values only, converted from report RMB thousands to yuan. Historical raw values remain latest aggregator versions.",
        "currency": "CNY", "raw_amount_unit_scale": 1,
        "unit_check_method": "Compare AMOUNT directly with five independent annual-report RMB'000 values multiplied by 1000; no scaling factor fitted to ranking results.",
        "sources": {}, "years": {}, "checks_2024": {},
        "unverified_engine_fields": {
            "core_income_parent": "No consistent A-share-style non-recurring-adjusted parent-profit mapping established; remains null.",
            "equity_issued": "Issuance events not reviewed for all years.",
            "audit_opinion": "Audit report opinion not mapped in this lightweight raw-field check.",
            "total_company_market_cap": "No verified 2025-06-30 all-share-classes HKD market capitalization plus contemporaneous CNY FX input in this sample.",
            "debt_and_interest_semantics": "Raw line items require lease/debt/finance-cost accounting reconciliation; not inferred or null-filled here.",
            "available_at": "The HK raw long-table cache does not establish original publication and restatement availability dates.",
        },
    }
    for kind, env in envelopes.items():
        path = RAW / f"02331_hk_{kind}.json"
        out["sources"][kind] = {
            "url": env["url"], "cache_path": "raw/" + path.name,
            "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **env.get("metadata", {}),
        }
    for year in ("2022", "2023", "2024"):
        date = year + "-12-31"
        annual = {}
        for field, (kind, name) in MAPPING.items():
            candidates = [row for row in envelopes[kind]["response"]["result"]["data"]
                          if str(row.get("REPORT_DATE", ""))[:10] == date
                          and row.get("STD_ITEM_NAME") == name]
            distinct = {(row.get("STD_ITEM_CODE"), row.get("AMOUNT")) for row in candidates}
            if len(candidates) != 1 or len(distinct) != 1:
                annual[field] = {"value": None, "reason": "Missing or ambiguous exact-name mapping.",
                                 "candidates": candidates}
            else:
                row = candidates[0]
                annual[field] = {"value": row.get("AMOUNT"),
                                 "STD_ITEM_CODE": row.get("STD_ITEM_CODE"),
                                 "STD_ITEM_NAME": row.get("STD_ITEM_NAME"),
                                 "DATE_TYPE_CODE": row.get("DATE_TYPE_CODE"),
                                 "START_DATE": row.get("START_DATE"),
                                 "raw_kind": kind}
        annual["core_income_parent"] = {"value": None, "reason": out["unverified_engine_fields"]["core_income_parent"]}
        out["years"][year] = annual
    for field, expected in EXPECTED_2024.items():
        observed = out["years"]["2024"][field]["value"]
        out["checks_2024"][field] = {
            "raw_yuan": observed, "annual_report_yuan": expected,
            "matched": observed == expected,
            "difference_yuan": observed - expected if observed is not None else None,
        }
    out["all_five_2024_values_match"] = all(row["matched"] for row in out["checks_2024"].values())
    path = ROOT / "hk_coverage.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_five_2024_values_match": out["all_five_2024_values_match"],
                      "checks_2024": out["checks_2024"], "years": out["years"]}, ensure_ascii=True), flush=True)
    return 0 if out["all_five_2024_values_match"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-cache', action='store_true', help='Rebuild from local raw downloads; default checks the committed frozen financial facts offline.')
    raise SystemExit(run(parser.parse_args().from_cache))
