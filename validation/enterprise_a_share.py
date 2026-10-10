"""Reconcile the curated A-share source facts without changing screening inputs.

Offline arithmetic and dataset comparisons: python validation/enterprise_a_share.py
Verify original PDFs too: python validation/enterprise_a_share.py --verify-pdfs
Fetch missing PDFs: python validation/enterprise_a_share.py --download --verify-pdfs

PDF verification needs pypdf. Downloads must match the pinned SHA256 before use.
The reference JSON is manually curated; text matches are supporting checks, not an
independent audit opinion or proof of historical archive custody.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
FACTS = ROOT / "validation" / "enterprise_a_share_facts.json"
OUTPUT = ROOT / "validation" / "enterprise_a_share_results.json"


def normalize(value: str) -> str:
    return re.sub(r"\s+", "", value)


def monetary(value: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("Money must be an exact decimal string")
    number = Decimal(value)
    if not number.is_finite():
        raise ValueError("Non-finite monetary fact")
    return number


def validate(document: dict, *, verify_pdfs: bool = False,
             download: bool = False, as_of: str = "2025-06-30") -> dict:
    cutoff = date.fromisoformat(as_of)
    sources = document["sources"]
    facts = document["facts"]
    ids = [fact["id"] for fact in facts]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source fact identifier")
    checked_sources = []
    pages: dict[str, dict[int, str]] = {}
    for fact in facts:
        page = fact["pdf_page"]
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError(f"Invalid PDF page: {fact['id']}")
        if fact["report_page"] != page or not fact["source_label"].strip():
            raise ValueError(f"Invalid page/label basis: {fact['id']}")
        if fact["source_id"] not in sources:
            raise ValueError(f"Unknown source: {fact['id']}")
    for source_id, source in sources.items():
        date.fromisoformat(source["publication_date"])
        if not re.fullmatch(r"[a-f0-9]{64}", source["sha256"]):
            raise ValueError(f"Invalid source hash: {source_id}")
        path = (ROOT / source["local_path"]).resolve()
        if not path.is_relative_to((ROOT / "validation" / "raw").resolve()):
            raise ValueError("Source cache path outside validation/raw")
        if download and not path.exists():
            request = Request(source["url"], headers={"User-Agent": "Chenxi-source-verification/1.0"})
            data = urlopen(request, timeout=90).read()
            if hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise ValueError(f"Downloaded document changed: {source_id}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        hash_match = None
        if verify_pdfs:
            from pypdf import PdfReader
            hash_match = hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
            if not hash_match:
                raise ValueError(f"Source SHA256 mismatch: {source_id}")
            reader = PdfReader(path)
            used_pages = {fact["pdf_page"] for fact in facts if fact["source_id"] == source_id}
            pages[source_id] = {
                page: normalize(reader.pages[page - 1].extract_text() or "")
                for page in sorted(used_pages)
            }
        checked_sources.append({"source_id": source_id, "sha256": source["sha256"],
                                "pdf_hash_match": hash_match,
                                "stated_publication_before_cutoff": date.fromisoformat(source["publication_date"]) <= cutoff,
                                "contemporaneous_archive_capture_verified": False})
    dataset = json.loads((ROOT / "validation" / "dataset.json").read_text(encoding="utf8"))
    baseline = {(row["ticker"], int(row["period_end"][:4])): row for row in dataset["statements"]}
    source_checks, comparisons = [], []
    for fact in facts:
        value = monetary(fact["value"])
        source = sources[fact["source_id"]]
        if fact["unit"] != "CNY yuan" or fact["scale_to_yuan"] != 1 or fact["scope"] != "consolidated":
            raise ValueError(f"Unexpected fact basis: {fact['id']}")
        if fact["source_value_token"] != f"{value:,.2f}":
            raise ValueError(f"Source token differs from value: {fact['id']}")
        amount_found = label_found = None
        if verify_pdfs:
            text = pages[fact["source_id"]][fact["pdf_page"]]
            amount_found = normalize(fact["source_value_token"]) in text
            label_found = normalize(fact["source_label"]) in text
            if not amount_found or not label_found:
                raise ValueError(f"PDF token check failed: {fact['id']}; amount={amount_found}, label={label_found}")
        source_checks.append({"id": fact["id"], "pdf_amount_found": amount_found,
                              "pdf_label_found": label_found,
                              "available_from_dated_source": source["publication_date"],
                              "source_eligible_at_cutoff": date.fromisoformat(source["publication_date"]) <= cutoff,
                              "selected_page_visually_reviewed": fact["visual_reviewed"]})
        field = fact["dataset_field"]
        if field:
            provider_value = baseline[(fact["ticker"], fact["fiscal_year"])].get(field)
            delta = None if provider_value is None else Decimal(str(provider_value)) - value
            comparisons.append({"id": fact["id"], "dataset_field": field,
                                "source_value": fact["value"], "dataset_value": provider_value,
                                "difference_yuan": None if delta is None else str(delta),
                                "match_to_cent": delta is not None and abs(delta) < Decimal("0.005")})
    by_key = {(f["ticker"], f["fiscal_year"], f["field"]): monetary(f["value"]) for f in facts}
    derived = []
    for ticker, year in sorted({(f["ticker"], f["fiscal_year"]) for f in facts}):
        get = lambda field: by_key[(ticker, year, field)]
        lease = get("lease_current") + get("lease_noncurrent")
        item = {"ticker": ticker, "fiscal_year": year,
                "verified_lease_liabilities_including_current": str(lease),
                "lease_calculation": "lease_current + lease_noncurrent",
                "verified_finance_cost_gross_interest": str(get("gross_interest_expense_finance_cost")),
                "verified_cash_and_cash_equivalents": str(get("cash")),
                "total_debt": None, "net_debt": None, "interest_coverage": None,
                "candidate_approval": False}
        if ticker == "000858.SZ":
            cash_calculated = get("monetary_funds") - get("restricted_monetary_funds") - get("accrued_deposit_interest_excluded_from_cash")
            item.update(cash_reconciliation="monetary_funds - restricted_monetary_funds - accrued_deposit_interest_excluded_from_cash",
                        reconciled_cash=str(cash_calculated),
                        cash_reconciliation_difference=str(cash_calculated - get("cash")),
                        cash_reconciliation_match=cash_calculated == get("cash"))
            if cash_calculated != get("cash"):
                raise ValueError(f"Cash reconciliation mismatch: {ticker} {year}")
        else:
            item.update(financial_subsidiary_customer_deposits=str(get("financial_subsidiary_customer_deposits")),
                        financial_subsidiary_interest_cost=str(get("financial_subsidiary_interest_cost")),
                        additional_review="Separate finance-subsidiary customer funding/cash/CFO from industrial cash. Consolidated interest policy remains unapproved.")
        derived.append(item)
    for change in document["restatements"]:
        if monetary(change["after"]) - monetary(change["before"]) != monetary(change["delta"]):
            raise ValueError("Restatement arithmetic mismatch")
    failures = [item for item in comparisons if not item["match_to_cent"]]
    return {
        "validation_date": "2026-10-10", "historical_cutoff": as_of,
        "status": "source_checks_passed_with_scope_limitations" if verify_pdfs and not failures else "offline_arithmetic_only" if not failures else "dataset_mismatch",
        "facts_sha256": hashlib.sha256(FACTS.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": {"sources": len(sources), "companies": len({f["ticker"] for f in facts}), "company_years": len(derived),
                    "monetary_source_facts": len(facts), "pdf_facts_checked": len(facts) if verify_pdfs else 0,
                    "dataset_comparisons": len(comparisons), "dataset_matches": len(comparisons) - len(failures),
                    "restatement_fields": len(document["restatements"]),
                    "cash_reconciliations": sum("cash_reconciliation_match" in row for row in derived),
                    "sources_after_cutoff": sum(not row["stated_publication_before_cutoff"] for row in checked_sources),
                    "candidate_approvals": 0},
        "source_documents": checked_sources, "source_fact_checks": source_checks,
        "dataset_comparisons": comparisons, "derived_reconciliations": derived,
        "restatements": document["restatements"], "unresolved": document["unknowns"],
        "limitations": document["limitations"],
        "review_status": "independent_human_preparer_reviewer_signoff_required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-pdfs", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--as-of", default="2025-06-30")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    document = json.loads(FACTS.read_text(encoding="utf8"))
    result = validate(document, verify_pdfs=args.verify_pdfs, download=args.download, as_of=args.as_of)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": result["status"], **result["summary"]}))
    return 1 if result["status"] == "dataset_mismatch" else 0


if __name__ == "__main__":
    raise SystemExit(main())
