"""Cache public Eastmoney reports for a small disclosed feasibility sample.

No null-to-zero conversion, debt inference, or accounting normalization occurs here.
Run with Python's standard library; requests are limited to three company workers.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
RAW.mkdir(exist_ok=True)
YEARS = ("2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31")
COMPANIES = {
    "600519": "SH", "000858": "SZ", "000568": "SZ",
    "600809": "SH", "002304": "SZ", "603369": "SH",
}
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://emweb.securities.eastmoney.com/"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def request(url, params=None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as response:
                payload = response.read()
            return payload, url
        except Exception as exc:
            last = exc
            if attempt == 0:
                time.sleep(1)
    raise RuntimeError(f"request failed after two attempts: {url}: {last}")


def cached_json(name, url, params):
    path = RAW / f"{name}.json"
    if path.exists():
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if envelope.get("response", {}).get("success") is False:
            # Retain the unsuccessful upstream response before a corrected retry.
            failed_path = RAW / f"{name}.failed.json"
            if not failed_path.exists():
                failed_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
                body = RAW / f"{name}.response.json"
                if body.exists():
                    (RAW / f"{name}.failed.response.json").write_bytes(body.read_bytes())
        elif "response" in envelope and "url" in envelope:
            # The earlier probe did not save a download timestamp or raw body.
            # Preserve this limitation, rather than inventing metadata.
            if "metadata" not in envelope:
                envelope["metadata"] = {
                    "fetched_at_utc": None,
                    "cache_file_mtime_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc).isoformat(),
                    "response_body_sha256": None,
                    "canonical_response_sha256": canonical_digest(envelope["response"]),
                    "cache_reused": True,
                    "metadata_note": "Legacy probe cache: download time and original HTTP bytes unavailable.",
                }
                path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
            return envelope
    payload, complete_url = request(url, params)
    response = json.loads(payload)
    body_path = RAW / f"{name}.response.json"
    body_path.write_bytes(payload)
    envelope = {
        "url": complete_url,
        "response": response,
        "metadata": {
            "fetched_at_utc": utc_now(),
            "response_body_sha256": hashlib.sha256(payload).hexdigest(),
            "canonical_response_sha256": canonical_digest(response),
            "response_body_file": body_path.name,
            "cache_reused": False,
            "source_kind": "public_aggregator_latest_reconstructed_history",
        },
    }
    path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    if response.get("success") is False:
        raise RuntimeError(f"upstream rejected JSON request: code={response.get('code')} message={response.get('message')}")
    return envelope


def records(envelope):
    response = envelope.get("response", {})
    result = response.get("result")
    rows = result.get("data") if isinstance(result, dict) else response.get("data")
    return rows if isinstance(rows, list) else []


def company_type(code, exchange):
    name = f"{code}_company_type"
    path = RAW / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["company_type"]
    payload, url = request(
        "https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/Index",
        {"type": "web", "code": exchange.lower() + code})
    text = payload.decode("utf-8")
    match = re.search(r'id="hidctype"[^>]*value="([^"]+)"', text)
    if not match:
        raise ValueError(f"company type not found for {code}")
    ctype = match.group(1)
    path.write_text(json.dumps({
        "url": url, "company_type": ctype, "fetched_at_utc": utc_now(),
        "response_body_sha256": hashlib.sha256(payload).hexdigest(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return ctype


def collect_company(code, exchange):
    # The successful probe established this directly. Reuse its company type
    # instead of repeating the original 600519 homepage/report requests.
    ctype = "4" if code == "600519" else company_type(code, exchange)
    output = {"code": code, "company_type": ctype, "counts": {}, "errors": {}}
    for kind, endpoint in (("balance", "zcfzbAjaxNew"), ("income", "lrbAjaxNew"),
                           ("cashflow", "xjllbAjaxNew")):
        try:
            result = cached_json(code + "_" + kind,
                "https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/" + endpoint,
                {"companyType": ctype, "reportDateType": "1", "reportType": "1",
                 "dates": ",".join(reversed(YEARS)), "code": exchange + code})
            output["counts"][kind] = len(records(result))
        except Exception as exc:
            output["errors"][kind] = str(exc)
    try:
        result = cached_json(code + "_metrics",
            "https://datacenter.eastmoney.com/securities/api/data/get",
            {"type": "RPT_F10_FINANCE_MAINFINADATA", "sty": "APP_F10_MAINFINADATA",
             "quoteColumns": "", "filter": f'(SECUCODE="{code}.{exchange}")',
             "p": "1", "ps": "200", "sr": "-1", "st": "REPORT_DATE",
             "source": "HSF10", "client": "PC"})
        output["counts"]["metrics"] = len(records(result))
    except Exception as exc:
        output["errors"]["metrics"] = str(exc)
    try:
        result = cached_json(code + "_value",
            "https://datacenter-web.eastmoney.com/api/data/v1/get",
            {"sortColumns": "TRADE_DATE", "sortTypes": "-1", "pageSize": "20", "pageNumber": "1",
             "reportName": "RPT_VALUEANALYSIS_DET", "columns": "ALL", "source": "WEB", "client": "WEB",
             "filter": f'(SECURITY_CODE="{code}")(TRADE_DATE>=\'2025-06-24\')(TRADE_DATE<=\'2025-06-30\')'})
        output["counts"]["value"] = len(records(result))
    except Exception as exc:
        output["errors"]["value"] = str(exc)
    return output


def collect_hk():
    code = "02331"
    output = {"code": code, "counts": {}, "errors": {}}
    for kind, report in (("balance", "RPT_HKF10_FN_BALANCE_PC"),
                         ("income", "RPT_HKF10_FN_INCOME_PC"),
                         ("cashflow", "RPT_HKF10_FN_CASHFLOW_PC"),
                         ("summary", "RPT_CUSTOM_HKSK_APPFN_CASHFLOW_SUMMARY")):
        try:
            filter_value = f'(SECUCODE="{code}.HK")'
            if kind != "summary":
                filter_value += "(REPORT_DATE in ('2022-12-31','2023-12-31','2024-12-31'))"
            columns = ("SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,START_DATE,REPORT_DATE,FISCAL_YEAR,"
                       "CURRENCY,ACCOUNT_STANDARD,REPORT_TYPE") if kind == "summary" else "ALL"
            result = cached_json(code + "_hk_" + kind,
                "https://datacenter.eastmoney.com/securities/api/data/v1/get",
                {"reportName": report, "columns": columns, "quoteColumns": "", "filter": filter_value,
                 "pageNumber": "1", "pageSize": "500", "source": "F10", "client": "PC"})
            output["counts"][kind] = len(records(result))
        except Exception as exc:
            output["errors"][kind] = str(exc)
    return output


def field_checks():
    summary = {
        "generated_at_utc": utc_now(),
        "semantics": "Observed raw fields only; null remains null; no inference of missing debt, interest, core earnings, share events, or audit opinions.",
        "companies": {}, "hk_items": {},
    }
    fields_by_kind = {
        "balance": ["NOTICE_DATE", "UPDATE_DATE", "REPORT_DATE", "TOTAL_ASSETS", "TOTAL_PARENT_EQUITY",
                    "TOTAL_CURRENT_ASSETS", "TOTAL_CURRENT_LIAB", "LONG_LOAN", "BOND_PAYABLE",
                    "LEASE_LIAB", "SHORT_LOAN", "NONCURRENT_LIAB_1YEAR", "MONETARYFUNDS"],
        "income": ["NOTICE_DATE", "UPDATE_DATE", "REPORT_DATE", "NETPROFIT", "PARENT_NETPROFIT",
                   "DEDUCT_PARENT_NETPROFIT", "OPERATE_INCOME", "TOTAL_OPERATE_INCOME", "OPERATE_COST",
                   "TOTAL_PROFIT", "FE_INTEREST_EXPENSE", "INTEREST_EXPENSE", "OPINION_TYPE"],
        "cashflow": ["NOTICE_DATE", "UPDATE_DATE", "REPORT_DATE", "NETCASH_OPERATE", "END_CCE", "CONSTRUCT_LONG_ASSET",
                     "ACCEPT_INVEST_CASH", "OTHER_RECEIVE_RELATED_INVEST", "ASSIGN_DIVIDEND_PORFIT"],
        "metrics": ["NOTICE_DATE", "UPDATE_DATE", "REPORT_DATE", "CURRENCY", "KCFJCXSYJLR", "DEDUCT_PARENT_NETPROFIT", "DEDUCT_PARENTNETPROFIT",
                    "PARENTNETPROFIT", "TOTALOPERATEREVE", "ROEJQ", "ROEKCJQ", "ROEJQ_DILUTED"],
        "value": ["TRADE_DATE", "TOTAL_MARKET_CAP", "MARKET_CAP", "PE_TTM", "PB_MRQ", "TOTAL_SHARES"],
    }
    for code in COMPANIES:
        company = {}
        for kind, wanted in fields_by_kind.items():
            path = RAW / f"{code}_{kind}.json"
            if not path.exists():
                company[kind] = {"missing_cache": True}
                continue
            rows = records(json.loads(path.read_text(encoding="utf-8")))
            selected = [r for r in rows if str(r.get("REPORT_DATE", ""))[:10] in YEARS] if kind != "value" else rows
            company[kind] = {
                "rows_total": len(rows), "rows_selected": len(selected),
                "field_coverage": {field: {
                    "present": sum(field in row for row in selected),
                    "non_null": sum(row.get(field) is not None for row in selected),
                    "explicit_zero": sum(row.get(field) == 0 for row in selected),
                } for field in wanted},
                "candidate_fields": sorted({k for row in selected for k in row
                    if any(part in k for part in ("DEDUCT", "KCFJCXSYJLR", "CORE", "INTEREST", "LOAN", "DEBT", "BOND", "LEASE", "CCE", "MARKET_CAP", "SHARES", "OPINION"))}),
                "annual_values": {
                    str(r.get("REPORT_DATE", r.get("TRADE_DATE", "")))[:10]:
                    {field: {"present": field in r, "value": r.get(field)} for field in wanted}
                    for r in selected
                },
            }
        summary["companies"][code] = company
    for kind in ("balance", "income", "cashflow", "summary"):
        path = RAW / f"02331_hk_{kind}.json"
        if path.exists():
            rows = records(json.loads(path.read_text(encoding="utf-8")))
            summary["hk_items"][kind] = {
                "rows": len(rows), "columns": sorted({k for r in rows for k in r}),
                "item_names": sorted({str(r.get("STD_ITEM_NAME")) for r in rows if r.get("STD_ITEM_NAME")}),
            }
    path = ROOT / "public_field_checks.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--skip-hk", action="store_true")
    args = parser.parse_args()
    failures = []
    if not args.inspect_only:
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = {pool.submit(collect_company, code, exchange): code for code, exchange in COMPANIES.items()}
            for future in as_completed(jobs):
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"code": jobs[future], "errors": {"company": str(exc)}}
                print(json.dumps(result, ensure_ascii=True), flush=True)
                if result.get("errors"):
                    failures.append(result)
        if not args.skip_hk:
            result = collect_hk()
            print(json.dumps(result, ensure_ascii=True), flush=True)
            if result.get("errors"):
                failures.append(result)
        (ROOT / "public_fetch_errors.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = field_checks()
    print("Field checks: " + str(ROOT / "public_field_checks.json"), flush=True)
    for code, company in summary["companies"].items():
        print(code + ": " + json.dumps({kind: {"rows": block.get("rows_selected"),
            "field_coverage": {field: count for field, count in block.get("field_coverage", {}).items()
                if field in {"DEDUCT_PARENT_NETPROFIT", "END_CCE", "LONG_LOAN", "BOND_PAYABLE", "FE_INTEREST_EXPENSE", "INTEREST_EXPENSE", "TOTAL_MARKET_CAP"}}}
            for kind, block in company.items()}, ensure_ascii=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
