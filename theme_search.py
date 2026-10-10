"""Point-in-time company discovery with BM25 and auditable text evidence.

Scores measure lexical relevance, never investment merit or probability. Relation
labels are conservative heuristics requiring human review, not verified business
or customer relationships. This module does not fetch or authenticate sources.
"""

from __future__ import annotations

import copy
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from urllib.parse import urlparse


DEFAULT_THEME_DICTIONARY = {
    "光模块": ["光模块", "光模組", "光模块产品", "optical transceiver", "optical transceivers", "optical module", "optical modules"],
    "机器人": ["机器人", "機器人", "robot", "robots", "robotics", "humanoid", "人形机器人", "人形機器人"],
    "AI服务器": ["AI服务器", "AI 服务器", "AI通用服务器", "AI伺服器", "AI 伺服器", "人工智能服务器", "人工智慧伺服器", "AI server", "AI servers", "artificial intelligence server", "artificial intelligence servers"],
}
DEFAULT_CONFIG = {"k1": 1.5, "b": 0.75, "excerpt_chars": 480,
                  "max_evidence_per_company": 5, "min_score": 0.0,
                  "theme_dictionary": DEFAULT_THEME_DICTIONARY}
SOURCE_TYPES = frozenset({"annual_report", "announcement", "company_website"})
DATE_BASES = frozenset({"official_release", "observed_at", "operator_supplied"})
RELATION_WEIGHTS = {"direct_business": 1.0, "upstream_or_downstream": 0.7,
                    "planned": 0.35, "uncertain": 0.3, "negated": 0.0}
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_WORDS = re.compile(r"[a-z0-9]+")
_NEGATION = re.compile(
    r"(?:不涉及|不从事|不從事|未涉及|未开展|未開展|未从事|未從事|没有|沒有|并无|並無|尚无|尚無|尚未|不生产|不生產|不销售|不銷售|未生产|未生產|未销售|未銷售|不提供|未提供|未布局|不是|並非|并非|不存在|未有|无相关|無相關|不属于|不屬於|not\s+(?:currently\s+)?(?:engage|involve|manufactur|produc|sell|supply)|(?:does|do|did|is|are|has|have)\s+not|(?:doesn|don|didn|isn|aren|hasn|haven)['’]t|\bno\b[^.!?;。！？；]{0,80}\b(?:business|revenue|sales|involvement|operations)\b|neither)", re.I)
_PLAN = re.compile(r"(?:计划|計劃|拟|擬|将开展|將開展|探索|筹备|籌備|有望|未来|未來|plan(?:s|ned)?\s+to|intend(?:s)?\s+to|explor(?:e|es|ing)|may\s+(?:enter|develop)|potential)", re.I)
_SELF = re.compile(r"(?:本公司|本集团|本集團|我们|我們|本企业|本企業|(?<![\u3400-\u9fffA-Za-z0-9])(?:公司|集团|集團)|\bwe\b|\bour\b|\bthe\s+(?:company|group)\b)", re.I)
_OPERATING = re.compile(r"(?:主要从事|主要從事|主营|主營|业务|業務|产品|產品|生产|生產|销售|銷售|制造|製造|研发|研發|收入|营收|營收|交付|发布|發佈|推出|manufactur|produc|develop|design|sell|sales|revenue|business|deliver|\boffers?\b|\bportfolio\b|\blaunch|\bintroduc)", re.I)
_CHAIN = re.compile(r"(?:上游|下游|零部件|零组件|零組件|供应|供應|配套|供货|供貨|供给|供給|供应商|供應商|用于|用於|upstream|downstream|component|suppl(?:y|ies|ier)|used\s+in)", re.I)
_INDUSTRY = re.compile(r"(?:行业|行業|市场|市場|趋势|趨勢|行业规模|行業規模|industry|market\s+(?:size|growth|trend)|global\s+demand)", re.I)
_CUSTOMER = re.compile(r"(?:客户|客戶|客户名称|客戶名稱|customer|client)", re.I)
_THIRD_PARTY = re.compile(r"(?:竞争对手|競爭對手|竞争者|競爭者|其他公司|其它公司|其他企业|其他企業|友商|同行企业|同行企業|该公司|該公司|competitor|competing\s+compan|other\s+compan|another\s+compan)", re.I)
_CEASED = re.compile(r"(?:已停止|已经停止|已經停止|已终止|已終止|已退出|退出了|不再|停产|停產|停售|停止(?:生产|生產|销售|銷售|经营|經營)|\bdiscontinu(?:e|ed|es|ing)\b|\bceas(?:e|ed|es|ing)\b|\bno\s+longer\b|\bexited\b|\bwithdrawn\s+from\b)", re.I)
_DEVELOPMENT = re.compile(r"(?:(?:仍|尚|目前|当前|當前)[\s\S]{0,14}(?:开发|開發|研发|研發|准备|準備|探索)[\s\S]{0,14}(?:阶段|階段)|尚未(?:实现|實現)?(?:量产|量產|商业化|商業化)|(?:still|currently|early)[\s\S]{0,25}(?:development|preparation)\s+(?:stage|phase)|not\s+yet[\s\S]{0,20}(?:mass\s+production|commerciali[sz]))", re.I)
_REGISTERED_SCOPE = re.compile(r"经营范围|經營範圍|营业范围|營業範圍|registered\s+(?:business\s+)?scope", re.I)
_RD_TEAM = re.compile(r"(?:组建|組建|成立|建立)[\s\S]{0,30}(?:研发|研發)[\s\S]{0,8}(?:团队|團隊)|(?:assembled|formed)[\s\S]{0,35}(?:research|development)\s+team", re.I)
_LOCAL_VARIANT = re.compile(r"新型号|新型號|新一代|新产品|新產品|部分|子业务|子業務|下一代|全新|新型|new\s+(?:model|generation|product)|next\s+generation", re.I)
_ESTABLISHED_PRODUCTION = re.compile(r"(?:已|已经|已經)[\s\S]{0,10}(?:量产|量產|交付)|(?:in|entered)\s+mass\s+production|(?:already\s+)?delivered", re.I)


def _analysis_text(text):
    """Join PDF line breaks inside Chinese words without altering quote offsets."""
    return re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", text)


def _matching_text(text):
    """Normalize matching text and retain an exact map to original characters.

    PDF line wrapping can split a Chinese product name. Removing that spacing
    only in the matching copy prevents false negatives without forging quotes.
    NFKC/casefold may expand one character into several, all mapped to its source.
    """
    removed = set()
    for match in re.finditer(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", text):
        removed.update(range(match.start(), match.end()))
    chars, offsets = [], []
    for index, char in enumerate(text):
        if index in removed:
            continue
        normalized = unicodedata.normalize("NFKC", char).casefold()
        chars.extend(normalized)
        offsets.extend([index] * len(normalized))
    return "".join(chars), offsets


def _source_key(document):
    return (document["ticker"], document["source_url"], document["available_at"], document.get("source_sha256"))


def _norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _date(value, field):
    if type(value) is date:
        return value
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{field} must be YYYY-MM-DD")
    return date.fromisoformat(value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _number(value, field, low, high=None, inclusive_low=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(value) or value < low or (not inclusive_low and value == low) or (high is not None and value > high):
        raise ValueError(f"{field} is outside the supported range")
    return value


def _integer(value, field, low, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"{field} must be an integer in range")
    return value


def _configuration(config):
    result = copy.deepcopy(DEFAULT_CONFIG)
    if config is None:
        return result
    if not isinstance(config, dict) or set(config) - set(DEFAULT_CONFIG):
        raise ValueError("config must contain only supported keys")
    for key, value in config.items():
        if key != "theme_dictionary":
            result[key] = value
    result["k1"] = _number(result["k1"], "k1", 0, 100, inclusive_low=False)
    result["b"] = _number(result["b"], "b", 0, 1)
    result["min_score"] = _number(result["min_score"], "min_score", 0)
    _integer(result["excerpt_chars"], "excerpt_chars", 80, 4000)
    _integer(result["max_evidence_per_company"], "max_evidence_per_company", 1, 100)
    if "theme_dictionary" in config:
        custom = config["theme_dictionary"]
        if not isinstance(custom, dict):
            raise ValueError("theme_dictionary must be an object of theme -> nonempty term list")
        normalized_keys = set()
        for theme, aliases in custom.items():
            if not _text(theme) or _norm(theme) in normalized_keys:
                raise ValueError("theme names must be nonempty and unique after normalization")
            normalized_keys.add(_norm(theme))
            if not isinstance(aliases, list) or not aliases or any(not _text(term) for term in aliases):
                raise ValueError("each theme requires a nonempty string list")
            if len({_norm(term) for term in aliases}) != len(aliases):
                raise ValueError("duplicate normalized theme terms")
            old_key = next((key for key in result["theme_dictionary"] if _norm(key) == _norm(theme)), None)
            if old_key is not None:
                del result["theme_dictionary"][old_key]
            result["theme_dictionary"][theme] = list(aliases)
    return result


def _pattern(term):
    body = r"\s+".join(re.escape(part) for part in term.split())
    if term and term[0].isascii() and term[0].isalnum():
        body = r"(?<![A-Za-z0-9])" + body
    if term and term[-1].isascii() and term[-1].isalnum():
        body += r"(?![A-Za-z0-9])"
    return re.compile(body, re.I)


def _query_terms(query, dictionary):
    query = unicodedata.normalize("NFKC", query).strip()
    terms = [query]
    themes = []
    for theme in sorted(dictionary, key=_norm):
        aliases = [theme] + dictionary[theme]
        if any(_pattern(alias).search(query) for alias in aliases):
            themes.append(theme)
            terms.extend(aliases)
    unique = {}
    for term in terms:
        unique.setdefault(_norm(term), term)
    return sorted(unique.values(), key=lambda item: (_norm(item), item)), themes


def _required_qualifiers(query):
    """Keep literal alphanumeric model/specification constraints during expansion."""
    words = re.findall(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*", unicodedata.normalize("NFKC", query))
    return sorted({_norm(word) for word in words
                   if re.search(r"[A-Za-z]", word) and re.search(r"[0-9]", word)})


def _tokens(text):
    normalized = _norm(text)
    counts = Counter("en:" + word for word in _WORDS.findall(normalized))
    for run in _CJK.findall(normalized):
        if len(run) == 1:
            counts["zh:" + run] += 1
        else:
            counts.update("zh:" + run[i:i + 2] for i in range(len(run) - 1))
    return counts


def _validate_companies(companies, cutoff, audit):
    grouped = defaultdict(list)
    for index, row in enumerate(companies):
        reason = None
        if not isinstance(row, dict):
            reason = "invalid_record"
        else:
            try:
                visible = _date(row.get("universe_as_of"), "universe_as_of")
                if visible > cutoff:
                    reason = "universe_after_as_of"
            except (ValueError, TypeError):
                reason = "invalid_universe_as_of"
            if reason is None:
                if any(not _text(row.get(field)) for field in ("ticker", "name", "sector")):
                    reason = "missing_identity"
                elif row.get("market") not in ("A", "HK"):
                    reason = "unsupported_market"
                elif row.get("scope") not in ("technology", "manufacturing"):
                    reason = "outside_scope"
                elif "aliases" in row and (not isinstance(row["aliases"], list) or
                     any(not _text(alias) for alias in row["aliases"]) or
                     len({_norm(alias) for alias in row["aliases"]}) != len(row["aliases"])):
                    reason = "invalid_company_aliases"
        if reason:
            audit["rejected_companies"].append({"index": index, "ticker": row.get("ticker") if isinstance(row, dict) else None, "reason": reason})
            continue
        record = {key: row[key] for key in ("ticker", "name", "market", "sector", "scope", "universe_as_of")}
        record["ticker"] = row["ticker"].strip().upper()
        record["aliases"] = sorted(row.get("aliases", []), key=_norm)
        record["universe_as_of"] = visible.isoformat()
        grouped[record["ticker"]].append(record)
    accepted = {}
    for ticker, rows in sorted(grouped.items()):
        if any(row != rows[0] for row in rows):
            audit["conflicts"].append({"type": "company_identity", "ticker": ticker})
            continue
        accepted[ticker] = rows[0]
        audit["duplicates"]["companies"] += len(rows) - 1
    return accepted


def _validate_documents(documents, companies, cutoff, audit):
    grouped = defaultdict(list)
    for index, row in enumerate(documents):
        reason = None
        if not isinstance(row, dict):
            reason = "invalid_record"
        else:
            try:
                visible = _date(row.get("available_at"), "available_at")
                if visible > cutoff:
                    reason = "available_after_as_of"
            except (ValueError, TypeError):
                reason = "invalid_available_at"
            if reason is None:
                if any(not _text(row.get(key)) for key in ("document_id", "ticker", "source_url", "text")):
                    reason = "missing_provenance_or_text"
                elif row["ticker"].strip().upper() not in companies:
                    reason = "ticker_not_in_visible_universe"
                elif not isinstance(row.get("source_type"), str) or row["source_type"] not in SOURCE_TYPES:
                    reason = "invalid_source_type"
                elif not ((type(row.get("page")) is int and row["page"] > 0) or _text(row.get("page"))):
                    reason = "invalid_page_locator"
                else:
                    try:
                        url = urlparse(row["source_url"])
                        if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password:
                            reason = "invalid_source_url"
                    except ValueError:
                        reason = "invalid_source_url"
                if reason is None and "source_sha256" in row and (not isinstance(row["source_sha256"], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", row["source_sha256"])):
                    reason = "invalid_source_sha256"
                if reason is None and "title" in row and not isinstance(row["title"], str):
                    reason = "invalid_title"
                if reason is None and "date_basis" in row and (not isinstance(row["date_basis"], str) or row["date_basis"] not in DATE_BASES):
                    reason = "invalid_date_basis"
        if reason:
            audit["rejected_documents"].append({"index": index, "document_id": row.get("document_id") if isinstance(row, dict) else None, "reason": reason})
            continue
        record = {key: row[key] for key in ("document_id", "ticker", "available_at", "source_url", "source_type", "page", "text")}
        record["ticker"] = row["ticker"].strip().upper()
        record["available_at"] = visible.isoformat()
        record["date_basis"] = row.get("date_basis", "operator_supplied")
        for optional in ("title", "source_sha256"):
            if optional in row:
                record[optional] = row[optional]
        grouped[record["document_id"]].append(record)
    accepted = []
    text_seen = set()
    candidates = []
    for doc_id, rows in sorted(grouped.items()):
        if any(row != rows[0] for row in rows):
            audit["conflicts"].append({"type": "document_id", "document_id": doc_id})
            continue
        candidates.append(rows[0])
        audit["duplicates"]["documents"] += len(rows) - 1
    # Same-source duplicates do not add information. A distinct dated release
    # must survive deduplication: a reissued assertion may resolve an old denial.
    for row in sorted(candidates, key=lambda item: (item["available_at"], item["document_id"])):
        key = (row["ticker"], _norm(row["text"]), row["source_url"],
               row["available_at"], row["date_basis"], row.get("source_sha256"))
        if key in text_seen:
            audit["duplicates"]["documents"] += 1
        else:
            text_seen.add(key)
            accepted.append(row)
    return accepted


def _sentences(text):
    """Offsets preserve original text, including line endings and punctuation."""
    start = 0
    for match in re.finditer(r"[。！？!?；;]+|\n\s*\n|(?<=[A-Za-z0-9])\.(?=\s|$)", text):
        end = match.end()
        if text[start:end].strip():
            yield start, end
        start = end
    if text[start:].strip():
        yield start, len(text)


def _claim_status(sentence, hit_start, hit_end, issuer_patterns=()):
    # Local comma clauses stop a denial about a different product leaking across.
    before = list(re.finditer(r"[,，]", sentence[:hit_start]))
    start = before[-1].end() if before else 0
    after = re.search(r"[,，]", sentence[hit_end:])
    end = hit_end + after.start() if after else len(sentence)
    clause = _analysis_text(sentence[start:end])
    # 'not only' and '不仅' are additive claims, not denials.
    normalized_clause = re.sub(r"\bnot\s+only\b|不仅|不僅", "", clause, flags=re.I)
    # 'Not yet mass produced' describes development, not absence of relevance.
    # Keep this bound to the matched comma clause, never a different product.
    if _DEVELOPMENT.search(clause):
        return "planned"
    if _NEGATION.search(normalized_clause) or _CEASED.search(normalized_clause):
        return "negated"
    if _PLAN.search(clause) or _RD_TEAM.search(clause):
        return "planned"
    if _THIRD_PARTY.search(clause) or _REGISTERED_SCOPE.search(clause):
        return "uncertain"
    own = bool(_SELF.search(clause)) or any(pattern.search(clause) for pattern in issuer_patterns)
    if not own and start:
        # A company subject may carry over from the preceding comma clause.
        own = bool(_SELF.search(sentence[:start])) and not _INDUSTRY.search(sentence[:start])
    if _CUSTOMER.search(clause) and not (_CHAIN.search(clause) and own and _OPERATING.search(clause)):
        return "uncertain"
    if own and _CHAIN.search(clause) and _OPERATING.search(clause):
        return "upstream_or_downstream"
    if own and _OPERATING.search(clause) and not _INDUSTRY.search(clause):
        return "direct_business"
    return "uncertain"


def _company_level_denial(sentence, hit_start, hit_end, term, query, issuer_patterns):
    """Recognize only explicit issuer-wide denial, not a product variant's status."""
    before = list(re.finditer(r"[,，]", sentence[:hit_start]))
    start = before[-1].end() if before else 0
    after = re.search(r"[,，]", sentence[hit_end:])
    end = hit_end + after.start() if after else len(sentence)
    clause = _analysis_text(sentence[start:end])
    prefix = _analysis_text(sentence[start:hit_start])
    suffix = _analysis_text(sentence[hit_end:end])
    own = bool(_SELF.search(clause)) or any(pattern.search(clause) for pattern in issuer_patterns)
    if not own or any(pattern.search(clause) for pattern in
                      (_LOCAL_VARIANT, _CUSTOMER, _THIRD_PARTY, _INDUSTRY, _DEVELOPMENT)):
        return False
    # Expansion may include narrower related products. A denial of humanoids
    # cannot erase an issuer's established industrial robot business.
    normalized_term, normalized_query = _norm(term), _norm(query)
    if normalized_term in {"humanoid", "人形机器人", "人形機器人"} and normalized_term not in normalized_query:
        return False
    if normalized_query in normalized_term and normalized_term != normalized_query:
        return False
    # Requiring the denial predicate adjacent to the matched product avoids
    # treating 'does not make 800G optical modules' as denial of ALL modules.
    predicate_before = re.search(
        r"(?:不涉及|不从事|不從事|未涉及|未开展|未開展|未从事|未從事|不生产|不生產|不销售|不銷售|未生产|未生產|未销售|未銷售|不提供|未提供|未布局|尚无|尚無|没有|沒有|并无|並無|已退出|已停止|已经停止|已經停止|不再(?:生产|生產|销售|銷售|经营|經營)?|停止(?:生产|生產|销售|銷售|经营|經營))\s*$|"
        r"(?:(?:does|do|did|is|are|has|have)\s+not\s+(?:(?:currently|manufacture|manufacturing|produce|producing|sell|selling|offer|supply|develop)\s+)*|(?:doesn|don|didn)['’]t\s+(?:(?:manufacture|produce|sell|offer|supply|develop)\s+)*|(?:has|have)\s+no\s+|no\s+longer\s+(?:manufactures?|produces?|sells?|offers?|supplies)\s+|(?:discontinued|ceased)\s+(?:manufacturing|producing|selling|offering|supplying)\s+)\s*$",
        prefix, re.I)
    predicate_after = re.match(
        r"(?:业务|業務|产品|產品)(?:已经|已經|已)?(?:停止|终止|終止|退出|停产|停產)|"
        r"\s+(?:business|operations?|production)\s+(?:has|have|was|were)\s+(?:been\s+)?(?:discontinued|ceased)",
        suffix, re.I)
    # In subject-product-predicate form, reject an unrecognized subtype before
    # the product ('company humanoid robot business has ceased').
    if predicate_after and not re.search(r"(?:本公司|公司|本集团|本集團)(?:的)?$|\b(?:our|the company(?:'s)?)\s*$", prefix, re.I):
        predicate_after = None
    return bool(predicate_before or predicate_after)


def _business_status(evidence):
    support = [item for item in evidence if item["claim_status"] in ("direct_business", "upstream_or_downstream")]
    denials = [item for item in evidence if item["company_level_denial"]]
    if support and denials:
        if any(item["date_basis"] != "official_release" for item in support + denials):
            return "historical_or_disputed", ["Issuer-wide positive and negative assertions have non-comparable publication dates; verify current business."]
        newest_support = max(item["available_at"] for item in support)
        newest_denial = max(item["available_at"] for item in denials)
        if newest_denial >= newest_support:
            return "historical_or_disputed", ["An issuer-wide denial is as recent as or newer than all operating support; historical evidence cannot establish current business."]
        return "current_business", ["Operating support was officially released after the issuer-wide denial; verify any change in business scope."]
    if support:
        return "current_business", ["Source text supports operating business; this heuristic still requires human review."]
    if any(item["claim_status"] == "planned" for item in evidence):
        return "planned_business", ["Only planned or development-stage business is supported."]
    return "uncertain", ["Related text does not establish that the issuer operates this business."]


def _passages(documents, patterns, maximum, audit, companies, query, qualifiers):
    passages = []
    seen = {}
    source_stages = {}
    sources_with_established_production = set()
    for document in documents:
        text = document["text"]
        issuer = companies[document["ticker"]]
        issuer_patterns = [_pattern(name) for name in [issuer["name"]] + issuer["aliases"]]
        for start, end in _sentences(text):
            sentence = text[start:end]
            matching, offsets = _matching_text(sentence)
            hits = [(offsets[match.start()], offsets[match.end() - 1] + 1, term)
                    for term, pattern in patterns for match in pattern.finditer(matching)]
            analysis_sentence = _analysis_text(sentence)
            if _ESTABLISHED_PRODUCTION.search(analysis_sentence) and any(
                    _claim_status(sentence, a, b, issuer_patterns) == "direct_business" for a, b, _ in hits):
                sources_with_established_production.add(_source_key(document))
            # Cross-page stage propagation is restricted to company-wide business
            # declarations in one announcement. A new model in an annual report
            # cannot downgrade an existing product line elsewhere in that report.
            if (document["source_type"] == "announcement" and hits and
                    _DEVELOPMENT.search(analysis_sentence) and not _LOCAL_VARIANT.search(analysis_sentence)):
                before_phase = re.split(r"[,，]", analysis_sentence)[0]
                whole_business = any(re.search(
                    r"(?:本公司|公司|本集团|本集團)[\s\S]{0,25}" + pattern.pattern +
                    r"[\s\S]{0,8}(?:业务|業務)[\s\S]{0,30}(?:仍|尚|目前|当前|當前)",
                    before_phase, flags=re.I) for _, pattern in patterns)
                if whole_business and _DEVELOPMENT.search(before_phase) and not _THIRD_PARTY.search(before_phase):
                    source_stages.setdefault(_source_key(document), {
                        "document_id": document["document_id"], "page": document["page"],
                        "source_url": document["source_url"], "available_at": document["available_at"],
                        "char_start": start, "char_end": min(end, start + maximum),
                        "exact_excerpt": text[start:min(end, start + maximum)]})
            # Index every sentence, retaining only matching windows in huge ones.
            windows = [(start, end)] if end - start <= maximum else []
            if not windows:
                anchors = [hit[0] for hit in hits] or list(range(0, len(sentence), maximum))
                for anchor in anchors:
                    left = max(start, min(start + anchor - maximum // 3, end - maximum))
                    windows.append((left, min(end, left + maximum)))
            for left, right in sorted(set(windows)):
                excerpt = text[left:right]
                local_hits = [(a, b, term) for a, b, term in hits if start + a >= left and start + b <= right]
                normalized_excerpt, _ = _matching_text(excerpt)
                if local_hits and not all(_pattern(term).search(normalized_excerpt) for term in qualifiers):
                    audit["qualifier_filtered_passages"] += 1
                    # A denial of the issuer's entire product business also
                    # applies to a requested model (e.g. 800G). Keep that proof
                    # even when the broad withdrawal does not repeat the model.
                    # A different model's denial fails _company_level_denial.
                    local_hits = [(a, b, term) for a, b, term in local_hits
                                  if _claim_status(sentence, a, b, issuer_patterns) == "negated"
                                  and _company_level_denial(sentence, a, b, term, query, issuer_patterns)]
                claims = sorted({(term, _claim_status(sentence, a, b, issuer_patterns)) for a, b, term in local_hits})
                issuer_denial = any(_claim_status(sentence, a, b, issuer_patterns) == "negated" and
                                    _company_level_denial(sentence, a, b, term, query, issuer_patterns)
                                    for a, b, term in local_hits)
                version = {"document": document, "start": left, "end": right,
                           "exact_excerpt": excerpt, "claims": claims,
                           "company_level_denial": issuer_denial,
                           "claim_context_truncated": left > start or right < end}
                key = (document["ticker"], _norm(excerpt), tuple(claims), issuer_denial)
                version_key = (*_source_key(document), document["page"], document["date_basis"])
                if key in seen:
                    audit["duplicates"]["passages"] += 1
                    existing = seen[key]
                    if version_key not in existing["version_keys"]:
                        existing["versions"].append(version)
                        existing["version_keys"].add(version_key)
                    continue
                passage = {"tokens": _tokens(normalized_excerpt), "versions": [version],
                           "version_keys": {version_key}}
                seen[key] = passage
                passages.append(passage)
    for indexed in passages:
        for passage in indexed["versions"]:
            stage = source_stages.get(_source_key(passage["document"]))
            if stage and _source_key(passage["document"]) not in sources_with_established_production and any(status in ("direct_business", "upstream_or_downstream") for _, status in passage["claims"]):
                passage["claims"] = [(term, "planned" if status in ("direct_business", "upstream_or_downstream") else status)
                                     for term, status in passage["claims"]]
                passage["source_development_disclosure"] = stage
    return passages


def discover_companies(companies, documents, query, as_of, *, config=None, limit=20):
    """Discover A/H technology/manufacturing companies from dated source text.

    Inputs are lists of explicit company identity and document dictionaries.
    Invalid records are quarantined in ``audit``; invalid API configuration raises
    ValueError. Matching uses full input text, not an earlier excerpt. Duplicate
    evidence cannot increase company score (maximum passage score aggregation).
    """
    if not isinstance(companies, list) or not isinstance(documents, list):
        raise ValueError("companies and documents must be lists")
    if not _text(query) or not _tokens(query):
        raise ValueError("query must contain Chinese characters or alphanumeric terms")
    cutoff = _date(as_of, "as_of")
    _integer(limit, "limit", 1)
    settings = _configuration(config)
    terms, themes = _query_terms(query, settings["theme_dictionary"])
    qualifiers = _required_qualifiers(query)
    query_tokens = set().union(*(_tokens(term).keys() for term in terms))
    audit = {"rejected_companies": [], "rejected_documents": [],
             "duplicates": {"companies": 0, "documents": 0, "passages": 0}, "conflicts": [],
             "qualifier_filtered_passages": 0}
    universe = _validate_companies(companies, cutoff, audit)
    visible_docs = _validate_documents(documents, universe, cutoff, audit)
    passages = _passages(visible_docs, [(term, _pattern(_norm(term))) for term in terms],
                         settings["excerpt_chars"], audit, universe, query, qualifiers)
    postings = defaultdict(list)
    lengths = []
    for index, passage in enumerate(passages):
        lengths.append(sum(passage["tokens"].values()))
        for token, frequency in passage["tokens"].items():
            postings[token].append((index, frequency))
    average_length = sum(lengths) / len(lengths) if lengths else 1.0
    average_length = average_length or 1.0
    scores = defaultdict(float)
    for token in sorted(query_tokens):
        token_postings = postings.get(token, [])
        idf = math.log(1 + (len(passages) - len(token_postings) + 0.5) / (len(token_postings) + 0.5))
        for index, frequency in token_postings:
            denominator = frequency + settings["k1"] * (1 - settings["b"] + settings["b"] * lengths[index] / average_length)
            scores[index] += idf * frequency * (settings["k1"] + 1) / denominator
    by_company = defaultdict(list)
    # Each unique lexical passage is scored once; separate dated source versions
    # retain their own quotes and assertion dates without multiplying BM25 input.
    for index, passage in ((i, version) for i, indexed in enumerate(passages)
                           for version in indexed["versions"]):
        if not passage["claims"]:
            continue  # Bigram/word overlap alone is not a theme assertion.
        statuses = {status for _, status in passage["claims"]}
        best_status = min(statuses, key=lambda status: (-RELATION_WEIGHTS[status], status))
        score = scores[index] * RELATION_WEIGHTS[best_status]
        doc = passage["document"]
        evidence = {key: doc[key] for key in ("document_id", "source_url", "source_type", "page", "available_at", "date_basis")}
        evidence_age = (cutoff - _date(doc["available_at"], "available_at")).days
        for optional in ("title", "source_sha256"):
            if optional in doc:
                evidence[optional] = doc[optional]
        if "source_development_disclosure" in passage:
            evidence["source_development_disclosure"] = passage["source_development_disclosure"]
        evidence.update({"exact_excerpt": passage["exact_excerpt"], "char_start": passage["start"], "char_end": passage["end"],
                         "claim_context_truncated": passage["claim_context_truncated"],
                         "evidence_age_days": evidence_age,
                         "warnings": ["historical_evidence_over_730_days"] if evidence_age > 730 else [],
                         "matched_terms": sorted({term for term, _ in passage["claims"]}, key=_norm),
                         "claim_status": best_status,
                         "company_level_denial": passage["company_level_denial"],
                         "term_claims": [{"term": term, "status": status} for term, status in passage["claims"]],
                         "relevance_score": round(score, 8), "requires_review": True})
        by_company[doc["ticker"]].append(evidence)
    results = []
    negated_only = []
    for ticker, evidence in sorted(by_company.items()):
        positive = [item for item in evidence if item["claim_status"] != "negated" and item["relevance_score"] > settings["min_score"]]
        if not positive:
            if all(item["claim_status"] == "negated" for item in evidence):
                negated_only.append(ticker)
            continue
        evidence.sort(key=lambda item: (-item["relevance_score"], -_date(item["available_at"], "available_at").toordinal(), item["document_id"], item["char_start"]))
        statuses = sorted({claim["status"] for item in evidence for claim in item["term_claims"]})
        contradictory = "negated" in statuses and "direct_business" in statuses
        if contradictory:
            audit["conflicts"].append({"type": "positive_and_negative_assertions", "ticker": ticker})
        selected = evidence[:settings["max_evidence_per_company"]]
        # Include denials in the evidence budget where possible; never hide them.
        denials = [item for item in evidence if any(claim["status"] == "negated" for claim in item["term_claims"])]
        if denials and all(item not in selected for item in denials):
            if len(selected) > 1:
                selected[-1] = denials[0]
            else:
                # Contradictory evidence is mandatory even with a one-item
                # display budget. A source-backed warning cannot hide its proof.
                selected.append(denials[0])
        business_status, business_reasons = _business_status(evidence)
        # The display budget must not hide the newest actual operating support.
        # Keep its full quote/offset/provenance, separately for each date/hash
        # assurance tier, so a stricter caller can select qualifying evidence.
        support_by_assurance = {}
        for item in sorted(evidence, key=lambda item: (-_date(item["available_at"], "available_at").toordinal(),
                                                        item["document_id"], item["char_start"])):
            if item["claim_status"] not in ("direct_business", "upstream_or_downstream"):
                continue
            key = (item["date_basis"], "source_sha256" in item)
            support_by_assurance.setdefault(key, item)
        business_support = list(support_by_assurance.values())
        row = {key: universe[ticker][key] for key in ("ticker", "name", "market", "sector")}
        row.update({"relevance_score": max(item["relevance_score"] for item in positive),
                    "matched_terms": sorted({term for item in positive for term in item["matched_terms"]}, key=_norm),
                    "relations": statuses, "evidence": selected, "requires_review": True,
                    "business_status": business_status, "business_status_reasons": business_reasons,
                    "business_support_evidence": business_support,
                    "has_conflicting_assertions": contradictory,
                    "has_mixed_business_stages": "planned" in statuses and "direct_business" in statuses,
                    "warnings": (["all_supporting_evidence_over_730_days"] if all(item["evidence_age_days"] > 730 for item in positive)
                                 else ["includes_historical_evidence_over_730_days"] if any(item["evidence_age_days"] > 730 for item in evidence) else []),
                    "evidence_count": len(evidence), "evidence_truncated": len(evidence) > len(selected)})
        results.append(row)
    results.sort(key=lambda item: (-item["relevance_score"], item["ticker"]))
    audit["negated_only_tickers"] = negated_only
    return {"status": "completed" if results else ("no_matches" if visible_docs else "no_eligible_documents"),
            "query": query, "as_of": cutoff.isoformat(), "method": "bm25_theme_expansion",
            "expanded_terms": terms, "matched_themes": themes, "required_qualifiers": qualifiers, "config": settings,
            "companies": results[:limit], "audit": audit,
            "coverage": {"input_companies": len(companies), "visible_companies": len(universe),
                         "input_documents": len(documents), "visible_unique_documents": len(visible_docs),
                         "companies_with_documents": len({doc["ticker"] for doc in visible_docs}),
                         "indexed_passages": len(passages), "matched_companies": len(results),
                         "returned_companies": min(limit, len(results)),
                         "universe_without_documents": sorted(set(universe) - {doc["ticker"] for doc in visible_docs})},
            "limitations": ["Lexical relevance is not an investment score or probability.",
                            "All relation labels require human verification; customer mentions do not verify customer relationships.",
                            "Source content, identity mappings and available dates are supplied by the caller and are not authenticated.",
                            "Coverage is limited to the supplied point-in-time universe and documents; absence is not evidence of no business.",
                            "Evidence older than 730 days is flagged as historical; this does not establish whether the business currently exists.",
                            "Model/specification tokens containing both letters and digits are required literally in the same evidence window; other natural-language modifiers are not semantically interpreted.",
                            "Business status uses conservative issuer-wide denial rules and official release dates; observed or operator-supplied dates cannot prove that conflicting business assertions supersede each other.",
                            "Company relevance is the maximum weighted passage BM25 score, not a sum of repeated mentions."]}
