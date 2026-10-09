"""Traceable document triage for fundamental research, using only stdlib.

Keyword hits are review prompts, not verified facts or investment conclusions.
Negated statements are intentionally retained: "no related party transactions"
must remain available for a human to interpret. Text is data and is never run.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


SOURCE_TYPES = frozenset({"annual_report", "announcement", "transcript", "alternative_data"})
NEAR_DUPLICATE_THRESHOLD = 0.90

# These categories flag potentially relevant passages, including statements that
# explicitly deny the issue. They do not assert that an issue actually exists.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cash_flow": (
        "现金流", "現金流", "经营活动", "經營活動", "cash flow", "cashflow",
        "operating cash", "free cash flow",
    ),
    "debt": (
        "负债", "負債", "债务", "債務", "借款", "偿债", "償債", "债券", "債券",
        "debt", "borrowing", "borrowings", "leverage", "covenant", "covenants",
    ),
    "customer_concentration": (
        "客户集中", "客戶集中", "主要客户", "主要客戶", "大客户", "大客戶",
        "前五大客户", "前五大客戶", "customer concentration", "major customer",
        "major customers", "largest customer", "top five customers",
    ),
    "related_party": (
        "关联方", "關聯方", "关联交易", "關聯交易", "关连交易", "關連交易",
        "related party", "related-party", "connected transaction", "connected transactions",
    ),
    "audit": (
        "审计", "審計", "核数", "核數", "保留意见", "保留意見", "持续经营", "持續經營",
        "audit", "auditor", "auditors", "going concern", "qualified opinion",
    ),
    "capacity": (
        "产能", "產能", "利用率", "扩产", "擴產", "在建工程", "资本开支", "資本開支",
        "capacity", "utilization", "utilisation", "capital expenditure", "capex",
    ),
}

_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_ENGLISH_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_REQUIRED_TEXT_FIELDS = ("document_id", "ticker", "available_at", "source_url", "text")


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _as_date(value: str | date, field: str) -> date:
    if isinstance(value, date):
        # Reject datetime objects: the schema uses explicit calendar dates.
        if type(value) is not date:
            raise ValueError(f"{field} must be a calendar date (YYYY-MM-DD)")
        return value
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{field} must be a calendar date (YYYY-MM-DD)")
    return date.fromisoformat(value)


def _terms(text: str) -> Counter[str]:
    normalized = _normalize_text(text)
    terms = Counter("en:" + word for word in _ENGLISH_TOKEN.findall(normalized))
    for run in _CJK_RUN.findall(normalized):
        if len(run) == 1:
            terms["zh:" + run] += 1
        else:
            terms.update("zh:" + run[i : i + 2] for i in range(len(run) - 1))
    return terms


def _tfidf(texts: Iterable[str]) -> tuple[list[dict[str, float]], dict[str, float]]:
    counts = [_terms(text) for text in texts]
    document_frequency: Counter[str] = Counter()
    for count in counts:
        document_frequency.update(count.keys())
    total = len(counts)
    idf = {term: math.log((1 + total) / (1 + frequency)) + 1
           for term, frequency in document_frequency.items()}
    return [_vector(count, idf) for count in counts], idf


def _vector(counts: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    weights = {term: (1 + math.log(count)) * idf[term]
               for term, count in counts.items() if term in idf}
    magnitude = math.sqrt(sum(weight * weight for weight in weights.values()))
    return {term: weight / magnitude for term, weight in weights.items()} if magnitude else {}


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return min(1.0, max(0.0, sum(weight * right.get(term, 0.0) for term, weight in left.items())))


def _topic_hits(text: str) -> tuple[list[str], int | None]:
    # Keep offsets into original text so the quoted excerpt remains verbatim.
    hits: list[str] = []
    first_offset: int | None = None
    for topic, keywords in TOPIC_KEYWORDS.items():
        offsets: list[int] = []
        for keyword in keywords:
            pattern = re.escape(keyword)
            if keyword.isascii():
                pattern = r"(?<![A-Za-z0-9])" + pattern + r"(?![A-Za-z0-9])"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                offsets.append(match.start())
        if offsets:
            hits.append(topic)
            offset = min(offsets)
            first_offset = offset if first_offset is None else min(first_offset, offset)
    return hits, first_offset


def _excerpt(text: str, first_hit: int | None, max_chars: int = 1000) -> str:
    if len(text) <= max_chars:
        return text
    # Include preceding context; never delete a denial or label a hit as a fact.
    start = max(0, (first_hit or 0) - 250)
    end = min(len(text), start + max_chars)
    if end == len(text):
        start = max(0, end - max_chars)
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


def process_documents(path: str | Path, as_of: str | date) -> dict[str, Any]:
    """Validate and triage JSONL evidence available by ``as_of`` (inclusive).

    Every nonblank line must be a JSON object. ``available_at`` is the date when
    a document became available, not its fiscal-period end. HTTP(S) source URLs
    and provenance fields are required. Source content is not authenticated by
    this module. Exact text duplicates are removed only within the same ticker.
    Near duplicates are suggestions based on text overlap, requiring review.
    """
    cutoff = _as_date(as_of, "as_of")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    normalized_seen: dict[tuple[str, str], str] = {}
    id_seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                def reject_constant(value):
                    raise ValueError(f'nonstandard JSON constant: {value}')
                def finite_float(value):
                    parsed = float(value)
                    if not math.isfinite(parsed):
                        raise ValueError('nonfinite JSON number')
                    return parsed
                record = json.loads(line, parse_constant=reject_constant, parse_float=finite_float)
            except ValueError:
                rejected.append({"line": line_number, "reason": "invalid_json"})
                continue
            if not isinstance(record, dict):
                rejected.append({"line": line_number, "reason": "record_not_object"})
                continue
            identity = {"line": line_number, "document_id": record.get("document_id"),
                        "ticker": record.get("ticker")}
            missing = [field for field in _REQUIRED_TEXT_FIELDS
                       if not isinstance(record.get(field), str) or not record[field].strip()]
            if missing:
                rejected.append({**identity, "reason": "missing_required_fields", "fields": missing})
                continue
            try:
                available = _as_date(record["available_at"], "available_at")
            except ValueError:
                rejected.append({**identity, "reason": "invalid_available_at"})
                continue
            if available > cutoff:
                rejected.append({**identity, "reason": "available_after_as_of"})
                continue
            try:
                url = urlparse(record["source_url"].strip())
                valid_url = url.scheme in {"http", "https"} and bool(url.netloc)
            except ValueError:
                valid_url = False
            if not valid_url:
                rejected.append({**identity, "reason": "invalid_source_url"})
                continue
            if not isinstance(record.get("source_type"), str) or record["source_type"] not in SOURCE_TYPES:
                rejected.append({**identity, "reason": "invalid_source_type"})
                continue
            ticker = record["ticker"].strip().upper()
            document_id = record["document_id"].strip()
            if document_id in id_seen:
                rejected.append({**identity, "reason": "duplicate_document_id"})
                continue
            key = (ticker, _normalize_text(record["text"]))
            if key in normalized_seen:
                rejected.append({**identity, "reason": "exact_duplicate_text",
                                 "duplicate_of": normalized_seen[key]})
                continue
            normalized_seen[key] = document_id
            id_seen.add(document_id)
            accepted.append({**record, "ticker": ticker, "document_id": document_id,
                             "available_at": available.isoformat(),
                             "source_url": record["source_url"].strip()})

    vectors, _ = _tfidf(record["text"] for record in accepted)
    evidence: list[dict[str, Any]] = []
    by_ticker: dict[str, list[int]] = {}
    for index, record in enumerate(accepted):
        topics, first_hit = _topic_hits(record["text"])
        best_index: int | None = None
        similarity = 0.0
        for prior in by_ticker.get(record["ticker"], []):
            score = _cosine(vectors[index], vectors[prior])
            if score > similarity:
                best_index, similarity = prior, score
        duplicate_of = (accepted[best_index]["document_id"]
                        if best_index is not None and similarity >= NEAR_DUPLICATE_THRESHOLD
                        else None)
        evidence.append({
            "ticker": record["ticker"], "document_id": record["document_id"],
            "available_at": record["available_at"], "source_url": record["source_url"],
            "source_type": record["source_type"], "page": record.get("page"),
            "topics": topics, "excerpt": _excerpt(record["text"], first_hit),
            "possible_duplicate_of": duplicate_of, "similarity": round(similarity, 6),
        })
        by_ticker.setdefault(record["ticker"], []).append(index)
    return {
        "evidence": evidence,
        "rejected": rejected,
        "method": (
            "Point-in-time JSONL validation; NFKC/casefold/whitespace exact deduplication "
            "within ticker; English tokens and Chinese character bigrams with smoothed "
            "TF-IDF cosine similarity (near-duplicate review threshold 0.90). Keywords "
            "flag human-review topics, not factual conclusions; negation is retained. "
            "Excerpts and source/page references are preserved. No financial score is changed. "
            "URLs are provenance references; source authenticity is not verified."
        ),
    }


def search_evidence(evidence: Iterable[dict[str, Any]], query: str,
                    limit: int = 5) -> list[dict[str, Any]]:
    """Rank stored excerpts by lexical TF-IDF similarity, without fact inference.

    Retrieval is over excerpts, not full source documents. It uses no labels,
    predictive model, investment score, or claimed accuracy metric.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    records = list(evidence)
    if not records or not query.strip() or limit == 0:
        return []
    vectors, idf = _tfidf(str(record.get("excerpt", "")) for record in records)
    query_vector = _vector(_terms(query), idf)
    ranked = [(index, _cosine(query_vector, vector)) for index, vector in enumerate(vectors)]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [{**records[index], "relevance": round(score, 6)}
            for index, score in ranked[:limit] if score > 0]
