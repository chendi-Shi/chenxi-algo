"""Measure public-source theme retrieval without treating snippets as full truth.

The selected-passage labels are a development smoke test, not a holdout. Full PDF
retrieval checks known positives and reports new associations for human review;
it does not manufacture market-wide precision from incomplete snippet labels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theme_search import discover_companies


ALIASES = {"光模块": ["optical transceiver", "光模組"],
           "机器人": ["robotics", "機器人"],
           "AI服务器": ["AI servers", "ＡＩ服务器"]}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio(top, bottom):
    return round(top / bottom, 6) if bottom else None


def _classification(predicted, positive, labelled):
    negative = labelled - positive
    tp, fp = predicted & positive, predicted & negative
    fn, tn = positive - predicted, negative - predicted
    return {"true_positive": len(tp), "false_positive": len(fp),
            "false_negative": len(fn), "true_negative": len(tn),
            "precision": _ratio(len(tp), len(tp) + len(fp)),
            "recall": _ratio(len(tp), len(positive)),
            "false_positive_tickers": sorted(fp), "false_negative_tickers": sorted(fn)}


def _summary(report):
    return [{"ticker": row["ticker"], "relations": row["relations"],
             "relevance_score": row["relevance_score"],
             "evidence": [{"document_id": item["document_id"], "page": item["page"],
                           "claim_status": item["claim_status"],
                           "char_start": item["char_start"], "char_end": item["char_end"],
                           "evidence_age_days": item["evidence_age_days"]}
                          for item in row["evidence"]]}
            for row in report["companies"]]


def _totals(cases, metric):
    counts = {name: sum(case[metric][name] for case in cases)
              for name in ("true_positive", "false_positive", "false_negative", "true_negative")}
    counts["precision"] = _ratio(counts["true_positive"], counts["true_positive"] + counts["false_positive"])
    counts["recall"] = _ratio(counts["true_positive"], counts["true_positive"] + counts["false_negative"])
    return counts


def validate(*, companies_path, sources_path, excerpts_path, full_documents_path=None):
    companies, sources = _json(companies_path), _json(sources_path)
    excerpts = _jsonl(excerpts_path)
    cutoff = sources["as_of"]
    paths = {"companies": companies_path, "sources_and_labels": sources_path,
             "selected_passages": excerpts_path, "retrieval_code": ROOT / "theme_search.py",
             "evaluation_code": Path(__file__)}
    errors, curated_cases, aliases, full_cases, full_aliases = [], [], [], [], []
    baseline_reports = {}
    full_documents = None
    if full_documents_path is not None:
        extracted = _jsonl(full_documents_path)
        extracted_tickers = {row["ticker"] for row in extracted}
        # Website HTML import is not supported. Clearly identify this hybrid set.
        supplements = [row for row in excerpts if row["ticker"] not in extracted_tickers]
        full_documents = extracted + supplements
        paths["full_pdf_pages"] = full_documents_path
    for case in sources["evaluation_cases"]:
        query = case["query"]
        positive = set(case["direct_tickers"])
        development = set(case["development_tickers"])
        labelled = positive | development | set(case["non_direct_tickers"])
        related = positive | development
        result = discover_companies(companies, excerpts, query, cutoff, limit=len(companies))
        baseline_reports[("selected_passages", query)] = result
        found = {row["ticker"] for row in result["companies"]}
        direct = {row["ticker"] for row in result["companies"] if "direct_business" in row["relations"]}
        curated_cases.append({"query": query, "expected_related": sorted(related),
                              "matches": _summary(result),
                              "thematic_retrieval": _classification(found, related, labelled),
                              "established_business_heuristic": _classification(direct, positive, labelled),
                              "development_mislabelled_established": sorted(development & direct)})
        if related - found:
            errors.append({"check": "curated_known_positive_recall", "query": query, "tickers": sorted(related - found)})
        if development & direct:
            errors.append({"check": "curated_development_claim_safety", "query": query, "tickers": sorted(development & direct)})
        for alias in ALIASES.get(query, []):
            alternate = discover_companies(companies, excerpts, alias, cutoff, limit=len(companies))
            actual = {row["ticker"] for row in alternate["companies"]}
            same_relations = {row["ticker"]: row["relations"] for row in result["companies"]} == {
                row["ticker"]: row["relations"] for row in alternate["companies"]}
            aliases.append({"canonical_query": query, "alias": alias,
                            "same_company_set": actual == found, "same_relations": same_relations,
                            "found_tickers": sorted(actual)})
            if actual != found or not same_relations:
                errors.append({"check": "alias_consistency", "query": alias})
        if full_documents is not None:
            complete = discover_companies(companies, full_documents, query, cutoff, limit=len(companies))
            baseline_reports[("full_pdf_plus_selected_websites", query)] = complete
            complete_found = {row["ticker"] for row in complete["companies"]}
            complete_direct = {row["ticker"] for row in complete["companies"] if "direct_business" in row["relations"]}
            full_cases.append({"query": query, "known_related_tickers": sorted(related),
                               "known_positive_recall": _ratio(len(complete_found & related), len(related)),
                               "missing_known_positive_tickers": sorted(related - complete_found),
                               "additional_matches_requiring_full_source_review": sorted(complete_found - related),
                               "development_mislabelled_established": sorted(development & complete_direct),
                               "coverage": complete["coverage"], "matches": _summary(complete)})
            if related - complete_found:
                errors.append({"check": "fulltext_known_positive_recall", "query": query,
                               "tickers": sorted(related - complete_found)})
            if development & complete_direct:
                errors.append({"check": "fulltext_development_claim_safety", "query": query,
                               "tickers": sorted(development & complete_direct)})
            for alias in ALIASES.get(query, []):
                alternate = discover_companies(companies, full_documents, alias, cutoff, limit=len(companies))
                actual = {row["ticker"] for row in alternate["companies"]}
                same_relations = {row["ticker"]: row["relations"] for row in complete["companies"]} == {
                    row["ticker"]: row["relations"] for row in alternate["companies"]}
                full_aliases.append({"canonical_query": query, "alias": alias,
                                     "same_company_set": actual == complete_found,
                                     "same_relations": same_relations, "found_tickers": sorted(actual)})
                if actual != complete_found or not same_relations:
                    errors.append({"check": "fulltext_alias_consistency", "query": alias})
    robustness = []
    document_sets = [("selected_passages", excerpts)]
    if full_documents is not None:
        document_sets.append(("full_pdf_plus_selected_websites", full_documents))
    for name, corpus in document_sets:
        for case in sources["evaluation_cases"]:
            query = case["query"]
            baseline = baseline_reports[(name, query)]
            reversed_result = discover_companies(list(reversed(companies)), list(reversed(corpus)), query, cutoff, limit=len(companies))
            future = {"document_id": "future-poison", "ticker": companies[0]["ticker"],
                      "available_at": "2099-01-01", "source_url": None, "text": query * 1000}
            future_result = discover_companies(companies, corpus + [future], query, cutoff, limit=len(companies))
            duplicate = copy.deepcopy(corpus[0])
            duplicate["document_id"] = "zz-duplicate-document"
            duplicate_result = discover_companies(companies, corpus + [duplicate], query, cutoff, limit=len(companies))
            checks = {"input_order_invariant": baseline["companies"] == reversed_result["companies"],
                      "future_document_isolated": baseline["companies"] == future_result["companies"],
                      "duplicate_does_not_change_ranking": baseline["companies"] == duplicate_result["companies"]}
            robustness.append({"corpus": name, "query": query, **checks})
            errors.extend({"check": check, "query": query, "corpus": name} for check, passed in checks.items() if not passed)
    return {"schema_version": 1, "as_of": cutoff,
            "status": "passed_with_documented_limitations" if not errors else "failed",
            "dataset_kind": "curated_development_smoke_test_not_holdout",
            "input_hashes": {name: _sha(path) for name, path in paths.items()},
            "selected_passage_metrics": {"canonical_queries": len(curated_cases),
                                         "thematic_retrieval": _totals(curated_cases, "thematic_retrieval"),
                                         "established_business_heuristic": _totals(curated_cases, "established_business_heuristic")},
            "selected_passage_cases": curated_cases, "alias_cases": aliases,
            "fulltext_corpus": ({"extracted_pdf_pages": len(extracted),
                                 "pdf_companies": len(extracted_tickers),
                                 "selected_website_supplements": len(supplements),
                                 "supplement_tickers": sorted({row["ticker"] for row in supplements})}
                                if full_documents is not None else {"status": "not_supplied"}),
            "fulltext_cases": full_cases, "fulltext_alias_cases": full_aliases,
            "robustness_checks": robustness, "errors": errors,
            "limitations": ["Nine selected companies and three themes are development cases, not a held-out or market-wide validation universe.",
                            "Selected-passage labels describe those passages and dates, not complete current company operations.",
                            "An established-business heuristic means a direct_business label; all such labels still require human verification.",
                            "Full-PDF recall is measured against known positives only. New fulltext matches lack independent whole-document truth and are not counted as false positives.",
                            "Six aliases repeat canonical concepts and are not independent observations in aggregate metrics.",
                            "This report evaluates retrieval and evidence interpretation, not stock returns, valuation accuracy, or investment efficacy."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", type=Path, default=ROOT / "data/theme_companies.json")
    parser.add_argument("--sources", type=Path, default=ROOT / "data/theme_sources.json")
    parser.add_argument("--excerpts", type=Path, default=ROOT / "data/theme_documents.jsonl")
    parser.add_argument("--full-documents", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/theme_retrieval_results.json")
    args = parser.parse_args(argv)
    result = validate(companies_path=args.companies, sources_path=args.sources,
                      excerpts_path=args.excerpts, full_documents_path=args.full_documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "selected_passage_metrics": result["selected_passage_metrics"],
                      "fulltext_corpus": result["fulltext_corpus"], "errors": result["errors"]}, ensure_ascii=True))
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
