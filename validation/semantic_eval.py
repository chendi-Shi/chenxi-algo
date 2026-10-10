"""Evaluate a frozen, incompletely judged company-discovery development set.

Standard library only. No model inference, network calls or automatic labels.
``runs`` is {method: {query_id: [{"ticker": str, "score": float, ...}]}}.
Rank order is the submitted order; scores are checked but never re-sorted.

Unjudged candidates keep their rank. Precision, recall, reciprocal rank and
NDCG have conservative bounds over all unjudged grades in [0, maximum_grade].
A point value is emitted only when the bounds agree. Recall's denominator is
the whole declared candidate universe, not just the judged pool. A separate
judged_positive_recall is a development diagnostic, not population recall.
NDCG bounds deliberately allow numerator/denominator extremes independently;
they are valid but may be loose. Empty relevance sets have undefined recall,
NDCG and MRR. With no known positive, bounds are conditional on some relevance.

Short rankings have empty, non-retrieved slots: precision divides by requested
k, while judged coverage divides by actual returned slots. All variants first
receive equal weight within an intent group; intent groups then receive equal
weight. Unknown point values are never silently omitted from macro averages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any


METHODS = ("legacy_bm25", "fixed_bm25", "dense", "hybrid", "hybrid_rerank")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _interval(lower: float | None, upper: float | None) -> dict[str, Any]:
    if lower is None or upper is None:
        return {"value": None, "lower": None, "upper": None}
    lower, upper = max(0.0, lower), min(1.0, upper)
    _require(lower <= upper + 1e-12, "invalid metric bounds")
    value = (lower + upper) / 2 if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-12) else None
    return {"value": value, "lower": lower, "upper": upper}


def _point(value: float | None) -> dict[str, Any]:
    return _interval(value, value)


def _average(intervals: list[dict[str, Any]]) -> dict[str, Any]:
    """Propagate undefined values instead of dropping difficult queries."""
    result: dict[str, Any] = {"items": len(intervals)}
    for field in ("value", "lower", "upper"):
        values = [item[field] for item in intervals]
        result[field] = (sum(values) / len(values)
                         if values and all(value is not None for value in values) else None)
    result["point_defined_items"] = sum(item["value"] is not None for item in intervals)
    result["bounds_defined_items"] = sum(item["lower"] is not None for item in intervals)
    return result


def _dcg(grades: list[int], k: int) -> float:
    return sum((2 ** grade - 1) / math.log2(rank + 2)
               for rank, grade in enumerate(grades[:k]))


def _validate_dataset(dataset: dict[str, Any]) -> tuple[dict, dict, list, int, int]:
    _require(isinstance(dataset, dict), "dataset must be an object")
    universe = dataset.get("candidate_tickers")
    _require(isinstance(universe, list) and bool(universe), "candidate_tickers must be a nonempty list")
    _require(all(isinstance(ticker, str) and ticker.strip() == ticker and ticker
                 for ticker in universe), "invalid candidate ticker")
    _require(len(set(universe)) == len(universe), "duplicate candidate ticker")
    maximum = dataset.get("maximum_grade", 2)
    threshold = dataset.get("positive_grade_threshold", 1)
    _require(type(maximum) is int and maximum >= 1, "maximum_grade must be a positive integer")
    _require(type(threshold) is int and 1 <= threshold <= maximum, "invalid positive grade threshold")
    # NDCG uses all positive grades, so binary relevance must use that same set.
    _require(threshold == 1, "only positive_grade_threshold=1 is supported")
    records = dataset.get("queries")
    _require(isinstance(records, list) and bool(records), "queries must be a nonempty list")
    source_ids = None
    source_tickers = {}
    if "source_documents" in dataset:
        sources = dataset["source_documents"]
        _require(isinstance(sources, list), "source_documents must be a list")
        _require(all(isinstance(s, dict) and isinstance(s.get("document_id"), str) for s in sources),
                 "invalid source document")
        source_ids = {s["document_id"] for s in sources}
        source_tickers = {s["document_id"]: s.get("ticker") for s in sources}
        _require(len(source_ids) == len(sources), "duplicate source document id")
    queries: dict[str, dict] = {}
    groups: dict[str, list[str]] = {}
    group_grades: dict[str, dict[str, int]] = {}
    for query in records:
        _require(isinstance(query, dict), "query must be an object")
        query_id, group_id = query.get("query_id"), query.get("group_id")
        _require(isinstance(query_id, str) and bool(query_id), "invalid query_id")
        _require(query_id not in queries, f"duplicate query_id: {query_id}")
        _require(isinstance(group_id, str) and bool(group_id), f"invalid group_id: {query_id}")
        _require(isinstance(query.get("text"), str) and bool(query["text"].strip()), f"empty query text: {query_id}")
        judgments = query.get("judgments")
        _require(isinstance(judgments, list) and bool(judgments), f"missing judgments: {query_id}")
        grades: dict[str, int] = {}
        for judgment in judgments:
            _require(isinstance(judgment, dict), f"invalid judgment: {query_id}")
            ticker, grade = judgment.get("ticker"), judgment.get("grade")
            _require(isinstance(ticker, str) and ticker in universe, f"judgment ticker outside universe: {ticker}")
            _require(ticker not in grades, f"duplicate judgment ticker: {query_id}/{ticker}")
            _require(type(grade) is int and 0 <= grade <= maximum, f"invalid grade: {query_id}/{ticker}")
            if source_ids is not None:
                refs = judgment.get("source_document_ids")
                _require(isinstance(refs, list) and bool(refs) and all(isinstance(r, str) and r in source_ids for r in refs),
                         f"missing or unknown judgment evidence: {query_id}/{ticker}")
                _require(all(source_tickers[r] == ticker for r in refs),
                         f"judgment evidence belongs to a different security: {query_id}/{ticker}")
                _require(isinstance(judgment.get("reason"), str) and bool(judgment["reason"].strip()),
                         f"missing judgment rationale: {query_id}/{ticker}")
            grades[ticker] = grade
        if group_id in group_grades:
            _require(grades == group_grades[group_id], f"inconsistent labels within intent group: {group_id}")
        group_grades[group_id] = grades
        groups.setdefault(group_id, []).append(query_id)
        queries[query_id] = {"group_id": group_id, "grades": grades}
    if "groups" in dataset:
        declared = dataset["groups"]
        _require(isinstance(declared, list) and all(isinstance(g, dict) for g in declared), "invalid groups")
        declared_ids = [g.get("group_id") for g in declared]
        _require(all(isinstance(g, str) for g in declared_ids), "invalid declared group_id")
        _require(len(set(declared_ids)) == len(declared_ids) and set(declared_ids) == set(groups), "group declarations do not match queries")
        for group in declared:
            _require(group.get("query_ids") == groups[group["group_id"]], f"group query_ids mismatch: {group['group_id']}")
    return queries, groups, universe, maximum, threshold


def _query_metrics(grades: dict[str, int], ranking: list[dict], universe: list[str],
                   maximum: int, threshold: int, k_values: tuple[int, ...]) -> dict[str, Any]:
    relevant = sum(g >= threshold for g in grades.values())
    unknown_total = len(universe) - len(grades)
    ranks = [row["ticker"] for row in ranking]
    result: dict[str, Any] = {
        "judged_candidates": len(grades), "unjudged_candidates": unknown_total,
        "judged_positive_candidates": relevant, "returned_candidates": len(ranks),
        "relevance_existence_confirmed": relevant > 0,
        "bounds_conditional_on_any_relevant": relevant == 0 and unknown_total > 0,
        "at_k": {},
    }
    first_known = next((i for i, t in enumerate(ranks, 1) if grades.get(t, 0) >= threshold), None)
    first_possible = next((i for i, t in enumerate(ranks, 1) if t not in grades or grades[t] >= threshold), None)
    if relevant + unknown_total == 0:
        result["mrr"] = _interval(None, None)
    else:
        result["mrr"] = _interval(1 / first_known if first_known else 0.0,
                                  1 / first_possible if first_possible else 0.0)
    for k in k_values:
        top = ranks[:k]
        positive_top = sum(grades.get(t, 0) >= threshold for t in top)
        unknown_top = sum(t not in grades for t in top)
        actual = len(top)
        precision = _interval(positive_top / k, (positive_top + unknown_top) / k)
        if relevant + unknown_total == 0:
            recall = _interval(None, None)
        else:
            minimum_denominator = relevant + unknown_total - unknown_top
            maximum_denominator = relevant + unknown_top
            # If all possible positives are returned, nonempty relevance implies recall=1.
            low = positive_top / minimum_denominator if minimum_denominator else 1.0
            high = ((positive_top + unknown_top) / maximum_denominator
                    if maximum_denominator else 0.0)
            recall = _interval(low, high)
        dcg_low = _dcg([grades.get(t, 0) for t in top], k)
        dcg_high = _dcg([grades.get(t, maximum) for t in top], k)
        idcg_low = _dcg(sorted(grades.values(), reverse=True), k)
        idcg_high = _dcg(sorted([*grades.values(), *([maximum] * unknown_total)], reverse=True), k)
        if idcg_high == 0:
            ndcg = _interval(None, None)
        else:
            ndcg = _interval(dcg_low / idcg_high,
                             min(1.0, dcg_high / idcg_low) if idcg_low else (1.0 if dcg_high else 0.0))
        result["at_k"][str(k)] = {
            "precision": precision, "recall": recall, "ndcg": ndcg,
            "judged_positive_recall": _point(positive_top / relevant if relevant else None),
            "judged_coverage": _point((actual - unknown_top) / actual if actual else None),
            "returned_coverage": _point(actual / k),
            "returned_count": actual, "judged_count": actual - unknown_top,
            "unjudged_count": unknown_top, "judged_positive_count": positive_top,
        }
    return result


def validate_corpus_alignment(dataset: dict[str, Any], documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve label references against actual indexed documents without relabeling.

    A repository short excerpt and its full PDF page can have different IDs.
    They refer to the same evidence only when ticker, source SHA256, page and
    source URL agree. An excerpt quote, if present, must occur in the matched
    page after whitespace removal. This checks source provenance, not whether a
    model-selected passage answers a query. File-byte hashes in corpus_inputs
    must additionally be checked by the experiment driver before inference.
    """
    _validate_dataset(dataset)
    _require(isinstance(documents, list) and bool(documents), "corpus documents must be a nonempty list")
    _require(isinstance(dataset.get("source_documents"), list) and bool(dataset["source_documents"]),
             "dataset requires source_documents for corpus alignment")
    document_ids = set()
    by_source: dict[tuple, list[dict]] = {}
    for doc in documents:
        _require(isinstance(doc, dict), "invalid corpus document")
        doc_id = doc.get("document_id")
        _require(isinstance(doc_id, str) and bool(doc_id) and doc_id not in document_ids,
                 f"missing or duplicate corpus document_id: {doc_id}")
        _require(isinstance(doc.get("text"), str) and bool(doc["text"].strip()), f"missing corpus text: {doc_id}")
        document_ids.add(doc_id)
        _require(doc.get("page") is not None, f"missing corpus page: {doc_id}")
        key = (doc.get("ticker"), doc.get("source_sha256"), str(doc.get("page")), doc.get("source_url"))
        _require(all(isinstance(value, str) and bool(value) for value in key), f"missing corpus source identity: {doc_id}")
        by_source.setdefault(key, []).append(doc)
    mappings = []
    for source in dataset["source_documents"]:
        key = (source.get("ticker"), source.get("source_sha256"), str(source.get("page")), source.get("source_url"))
        _require(source.get("page") is not None and all(isinstance(value, str) and bool(value) for value in key),
                 f"missing label source identity: {source['document_id']}")
        candidates = by_source.get(key, [])
        _require(bool(candidates), f"label source absent or changed in corpus: {source['document_id']}")
        quote = source.get("existing_repository_excerpt")
        if quote:
            _require(isinstance(quote, str), f"invalid label excerpt: {source['document_id']}")
            compact_quote = "".join(quote.split())
            candidates = [doc for doc in candidates if compact_quote in "".join(doc["text"].split())]
            _require(bool(candidates), f"label excerpt absent from matched corpus page: {source['document_id']}")
        mappings.append({"label_document_id": source["document_id"],
                         "corpus_document_ids": sorted(doc["document_id"] for doc in candidates),
                         "source_sha256": source["source_sha256"], "page": source["page"],
                         "ticker": source["ticker"]})
    return {"status": "aligned", "corpus_document_count": len(documents),
            "label_reference_count": len(mappings), "mappings": mappings}


def _macro(items: list[dict], k_values: tuple[int, ...]) -> dict:
    metrics = ("precision", "recall", "ndcg", "judged_positive_recall", "judged_coverage", "returned_coverage")
    return {
        "items": len(items),
        "mrr": _average([item["mrr"] for item in items]),
        "at_k": {str(k): {metric: _average([item["at_k"][str(k)][metric] for item in items])
                           for metric in metrics} for k in k_values},
    }


def evaluate(dataset: dict[str, Any], runs: dict[str, Any],
             k_values: tuple[int, ...] = (1, 3, 5)) -> dict[str, Any]:
    """Validate supplied rankings, then compute query and intent-group metrics.

    Nonempty subsets of the five declared methods are allowed and reported.
    Every supplied method MUST contain every dataset query, even for an empty
    ranking. Unknown queries, duplicate tickers and non-finite scores fail.
    """
    queries, groups, universe, maximum, threshold = _validate_dataset(dataset)
    _require(isinstance(k_values, (list, tuple)) and bool(k_values), "k_values must be nonempty")
    _require(all(type(k) is int and k > 0 for k in k_values) and len(set(k_values)) == len(k_values),
             "k_values must contain distinct positive integers")
    k_values = tuple(k_values)
    _require(isinstance(runs, dict) and bool(runs), "runs must be a nonempty method mapping")
    _require(all(method in METHODS for method in runs), "unknown retrieval method")
    result: dict[str, Any] = {
        "schema_version": 1, "dataset_id": dataset.get("dataset_id"),
        "dataset_canonical_sha256": hashlib.sha256(json.dumps(dataset, sort_keys=True, ensure_ascii=False,
                                                                separators=(",", ":")).encode("utf-8")).hexdigest(),
        "annotation_status": dataset.get("annotation_status", "unspecified"),
        "acceptance_status": dataset.get("acceptance_status", "unspecified"),
        "blind_holdout": dataset.get("blind_holdout", False),
        "query_count": len(queries), "intent_group_count": len(groups),
        "candidate_count": len(universe), "k_values": list(k_values),
        "evaluated_methods": list(runs), "missing_methods": [method for method in METHODS if method not in runs],
        "metric_policy": {
            "unjudged": "retain rank; never assume negative; output bounds and null uncertain point values",
            "recall": "all relevant candidates in the declared finite universe; judged_positive_recall is a separate incomplete-label diagnostic",
            "ndcg": "gain=2**grade-1, discount=log2(rank+1); conservative bounds over unjudged grades",
            "mrr": "first relevant rank in the entire submitted ranking, not truncated at k",
            "precision": "denominator=requested k; absent result slots receive zero retrieval credit",
            "judged_coverage": "judged returned candidates / actual returned candidates; null for empty ranking",
            "aggregation": "equal query weight within group, then equal group weight; undefined values propagate",
            "no_known_positive": "recall/NDCG/MRR bounds conditional on existence of any relevant candidate; undefined if all candidates judged zero",
        },
        "methods": {},
    }
    for method, query_runs in runs.items():
        _require(isinstance(query_runs, dict), f"method results must be a query mapping: {method}")
        unknown = set(query_runs) - set(queries)
        missing = set(queries) - set(query_runs)
        _require(not unknown, f"unknown query results for {method}: {sorted(unknown, key=str)}")
        _require(not missing, f"missing query results for {method}: {sorted(missing)}")
        per_query = {}
        for query_id, query in queries.items():
            ranking = query_runs[query_id]
            _require(isinstance(ranking, list), f"ranking must be a list: {method}/{query_id}")
            seen = set()
            for row in ranking:
                _require(isinstance(row, dict), f"ranking row must be an object: {method}/{query_id}")
                ticker, score = row.get("ticker"), row.get("score")
                _require(isinstance(ticker, str) and ticker in universe, f"ranking ticker outside candidate universe: {ticker}")
                _require(ticker not in seen, f"duplicate ranking ticker: {method}/{query_id}/{ticker}")
                _require(isinstance(score, Real) and not isinstance(score, bool) and math.isfinite(score),
                         f"score must be finite numeric: {method}/{query_id}/{ticker}")
                seen.add(ticker)
            per_query[query_id] = {"group_id": query["group_id"],
                                   **_query_metrics(query["grades"], ranking, universe, maximum, threshold, k_values)}
        per_group = {group_id: _macro([per_query[q] for q in query_ids], k_values)
                     for group_id, query_ids in groups.items()}
        result["methods"][method] = {"queries": per_query, "groups": per_group,
                                    "group_macro": _macro(list(per_group.values()), k_values)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()
    dataset_bytes = args.dataset.read_bytes()
    run_bytes = args.runs.read_bytes()
    report = evaluate(json.loads(dataset_bytes), json.loads(run_bytes), tuple(args.k))
    report["dataset_file_sha256"] = hashlib.sha256(dataset_bytes).hexdigest()
    report["runs_file_sha256"] = hashlib.sha256(run_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "methods": report["evaluated_methods"],
                      "query_count": report["query_count"], "intent_group_count": report["intent_group_count"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
