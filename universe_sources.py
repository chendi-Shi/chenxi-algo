"""Observed official security directories with immutable snapshots and offline replay.

This is a security master acquisition layer, not a technology/manufacturing
classification. BSE is not supported yet. No source is a historical membership
database; ``universe_as_of`` never predates observation or a source effective date.
Run with explicit --download, or --replay to verify an existing snapshot offline.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
import time
import zipfile
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from xml.etree import ElementTree as ET


SSE_BASE = "https://query.sse.com.cn/sseQuery/commonQuery.do"
SZSE_URL = "https://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=1110&TABKEY=tab1"
HKEX_URL = "https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx"
DEFAULT_SOURCES = ("sse", "szse", "hkex")
MAX_BYTES = 20 * 1024 * 1024
MAX_UNCOMPRESSED = 100 * 1024 * 1024
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SOURCE_PAGES = {
    "sse": "https://www.sse.com.cn/assortment/stock/list/share/",
    "szse": "https://www.szse.cn/market/product/stock/list/index.html",
    "hkex": "https://www.hkex.com.hk/Services/Trading/Securities/Securities-Lists?sc_lang=en",
}


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("date must be YYYY-MM-DD")
    return date.fromisoformat(value)


def _observed(value):
    if not isinstance(value, str):
        raise ValueError("observed_at must be a timezone-aware timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("observed_at must have a timezone")
    return parsed.astimezone(timezone.utc)


def _market_date(value):
    return _observed(value).astimezone(timezone(timedelta(hours=8))).date().isoformat()


def _sse_url(page):
    return SSE_BASE + "?" + urlencode({
        "STOCK_TYPE": "1,8", "REG_PROVINCE": "", "CSRC_CODE": "", "STOCK_CODE": "",
        "sqlId": "COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L", "COMPANY_STATUS": "2,4,5,7,8",
        "type": "inParams", "isPagination": "true", "pageHelp.cacheSize": "1",
        "pageHelp.beginPage": str(page), "pageHelp.pageSize": "2000",
        "pageHelp.pageNo": str(page), "pageHelp.endPage": str(page),
    })


def _allowed_url(source, url):
    if source == "sse":
        return any(url == _sse_url(page) for page in range(1, 51))
    return url == {"szse": SZSE_URL, "hkex": HKEX_URL}.get(source)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("official source redirected; endpoint requires review")


def _download(source, url, *, timeout=20.0, retries=2, max_bytes=MAX_BYTES):
    if not _allowed_url(source, url):
        raise ValueError("only the reviewed official source endpoints are allowed")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("timeout must be between 0 and 60 seconds")
    if type(retries) is not int or not 0 <= retries <= 3:
        raise ValueError("retries must be an integer between 0 and 3")
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES:
        raise ValueError("invalid byte limit")
    opener = build_opener(_NoRedirect())
    request = Request(url, headers={"User-Agent": "chenxi-algo/0.2 official-directory-research",
                                   "Referer": SOURCE_PAGES[source], "Accept-Encoding": "identity"})
    for attempt in range(retries + 1):
        started = time.monotonic()
        try:
            with opener.open(request, timeout=timeout) as response:
                if response.status != 200 or response.geturl() != url:
                    raise ValueError("unexpected HTTP status or URL")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise ValueError("unexpected compressed HTTP response")
                size = response.headers.get("Content-Length")
                if size is not None and (not size.isdigit() or int(size) > max_bytes):
                    raise ValueError("invalid or excessive HTTP content length")
                body = bytearray()
                while True:
                    # A server cannot keep a run alive indefinitely by trickling bytes.
                    if time.monotonic() - started > timeout:
                        raise TimeoutError("download exceeded elapsed time limit")
                    block = response.read(min(65536, max_bytes + 1 - len(body)))
                    if not block:
                        break
                    body.extend(block)
                    if len(body) > max_bytes:
                        raise ValueError("download exceeded byte limit")
                if not body or (size is not None and len(body) != int(size)):
                    raise ValueError("empty or truncated download")
                return bytes(body), {"content_type": response.headers.get("Content-Type"),
                                     "last_modified": response.headers.get("Last-Modified"),
                                     "attempts": attempt + 1}
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}:
                raise
            # Certificate verification failures are not bypassed or retried.
            if "CERTIFICATE_VERIFY_FAILED" in str(exc) or attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 4))
    raise RuntimeError("unreachable")


def _xml(data):
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("XML declarations with entities are unsupported")
    return ET.fromstring(data)


def _xlsx_rows(data):
    """Read only plain values from a bounded, single-sheet official workbook."""
    if not isinstance(data, bytes) or len(data) > MAX_BYTES:
        raise ValueError("invalid workbook size")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) > 100 or sum(item.file_size for item in infos) > MAX_UNCOMPRESSED:
            raise ValueError("workbook decompression limit exceeded")
        if len({item.filename for item in infos}) != len(infos):
            raise ValueError("duplicate ZIP members")
        names = archive.namelist()
        sheets = [name for name in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)]
        if sheets != ["xl/worksheets/sheet1.xml"]:
            raise ValueError("expected one official worksheet")
        shared = []
        if "xl/sharedStrings.xml" in names:
            shared = ["".join(t.text or "" for t in item.findall(".//s:t", NS))
                      for item in _xml(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        rows = []
        for row in _xml(archive.read(sheets[0])).findall(".//s:sheetData/s:row", NS):
            cells = {}
            for cell in row.findall("s:c", NS):
                coordinate = cell.get("r", "")
                if not re.fullmatch(r"[A-Z]{1,3}[1-9]\d*", coordinate):
                    raise ValueError("invalid worksheet cell coordinate")
                column = re.match(r"[A-Z]+", coordinate).group()
                if column in cells or cell.find("s:f", NS) is not None:
                    raise ValueError("duplicate cell or formula in security directory")
                value = cell.findtext("s:v", "", NS) or ""
                kind = cell.get("t")
                if kind == "s":
                    if not value.isdigit() or int(value) >= len(shared):
                        raise ValueError("invalid shared-string reference")
                    value = shared[int(value)]
                elif kind == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall("s:is//s:t", NS))
                elif kind not in {None, "n", "str"}:
                    raise ValueError("unsupported worksheet cell type")
                cells[column] = value.strip()
            if any(cells.values()):
                rows.append(cells)
            if len(rows) > 100000:
                raise ValueError("worksheet row limit exceeded")
        return rows


def _table(rows, headers):
    for index, row in enumerate(rows[:10]):
        if headers <= set(row.values()):
            nonempty = [value for value in row.values() if value]
            if len(set(nonempty)) != len(nonempty):
                raise ValueError("duplicate directory column labels")
            return [{label: item.get(column, "") for column, label in row.items() if label}
                    for item in rows[index + 1:]]
    raise ValueError("security directory headers changed or are missing")


def _security(code, name, source, observed_at, source_url, raw_sha256, *, effective_at=None,
              source_sector=None, industry_code=None, **fields):
    code = str(code).strip()
    expected = r"(?:60|68)\d{4}" if source == "sse" else r"(?:00|30)\d{4}" if source == "szse" else r"\d{5}"
    if not re.fullmatch(expected, code) or not isinstance(name, str) or not name.strip():
        raise ValueError("invalid stock code or empty company name")
    observed_date = _market_date(observed_at)
    if effective_at is not None:
        _date(effective_at)
    return {
        "ticker": code + {"sse": ".SH", "szse": ".SZ", "hkex": ".HK"}[source],
        "name": name.strip(), "market": "HK" if source == "hkex" else "A",
        "exchange": {"sse": "SSE", "szse": "SZSE", "hkex": "HKEX"}[source],
        "security_type": "depositary_receipt" if source == "sse" and code.startswith("689") else "equity",
        "sector": None, "scope": None, "classification_status": "unverified",
        "source_sector": source_sector or None,
        "official_industry_code": industry_code or None,
        "official_industry_name": source_sector or None,
        "classification_source": source_url if industry_code else None,
        "scope_suggestion": {"C": "manufacturing", "I": "technology"}.get(industry_code),
        "scope_suggestion_basis": "official broad industry code mapping C/I; not reviewed business classification" if industry_code in {"C", "I"} else None,
        "universe_as_of": max(observed_date, effective_at or observed_date), "observed_at": observed_at,
        "source_effective_at": effective_at, "source_url": source_url,
        "raw_sha256": raw_sha256, "listing_status": "present_in_official_directory",
        **fields,
    }


def _unique(rows):
    if not rows:
        raise ValueError("empty equity directory")
    counts = Counter(row["ticker"] for row in rows)
    if any(count != 1 for count in counts.values()):
        raise ValueError("duplicate ticker in official directory or pagination drift")
    return sorted(rows, key=lambda row: row["ticker"])


def parse_sse_page(data):
    obj = _json(data)
    if not isinstance(obj, dict) or obj.get("actionErrors") or obj.get("fieldErrors"):
        raise ValueError("SSE error or unexpected payload")
    page = obj.get("pageHelp")
    records = obj.get("result")
    if not isinstance(page, dict) or not isinstance(records, list) or not records:
        raise ValueError("SSE missing nonempty result and pageHelp")
    for field in ("pageNo", "pageCount", "total", "pageSize"):
        if type(page.get(field)) is not int or page[field] <= 0:
            raise ValueError("invalid SSE pagination: " + field)
    if page["pageCount"] > 50 or page["total"] > 50000:
        raise ValueError("SSE pagination exceeds expected bound")
    if page["pageCount"] != math.ceil(page["total"] / page["pageSize"]) or page["pageNo"] > page["pageCount"]:
        raise ValueError("inconsistent SSE pagination")
    expected = min(page["pageSize"], page["total"] - (page["pageNo"] - 1) * page["pageSize"])
    if len(records) != expected or ("data" in page and page["data"] != records):
        raise ValueError("truncated or inconsistent SSE page")
    return records, {key: page[key] for key in ("pageNo", "pageCount", "total", "pageSize")}


def parse_source(source, pages, observed_at):
    """Parse [(raw_bytes, source_url)] from one source; no network access."""
    _observed(observed_at)
    if source not in DEFAULT_SOURCES or not pages:
        raise ValueError("unsupported source or empty snapshot")
    securities = []
    effective = None
    excluded = Counter()
    if source == "sse":
        expected_pagination = None
        for number, (data, url) in enumerate(pages, 1):
            if url != _sse_url(number):
                raise ValueError("unexpected SSE page URL/order")
            rows, page = parse_sse_page(data)
            identity = (page["pageCount"], page["total"], page["pageSize"])
            if page["pageNo"] != number or (expected_pagination is not None and identity != expected_pagination):
                raise ValueError("SSE pagination changed during collection")
            expected_pagination = identity
            raw_sha256 = _hash(data)
            for row in rows:
                if not isinstance(row, dict) or row.get("STOCK_TYPE") not in {"1", "8"} or row.get("STATE_CODE") not in {"2", "4", "5", "7", "8"}:
                    raise ValueError("unexpected SSE stock type/state")
                securities.append(_security(row.get("A_STOCK_CODE"), row.get("SEC_NAME_CN"), source,
                    observed_at, url, raw_sha256, source_sector=row.get("CSRC_CODE_DESC"), industry_code=row.get("CSRC_CODE"),
                    source_state=row["STATE_CODE"], board=row["STOCK_TYPE"], listing_date=row.get("LIST_DATE")))
        if len(pages) != expected_pagination[0] or len(securities) != expected_pagination[1]:
            raise ValueError("SSE snapshot is missing pages")
    else:
        if len(pages) != 1 or not _allowed_url(source, pages[0][1]):
            raise ValueError("unexpected directory source URL/page count")
        data, url = pages[0]
        raw_sha256 = _hash(data)
        workbook = _xlsx_rows(data)
        if source == "szse":
            rows = _table(workbook, {"A股代码", "A股简称", "A股上市日期", "所属行业", "板块"})
            for row in rows:
                industry = re.fullmatch(r"([A-S])\s+(.+)", row["所属行业"])
                if not industry:
                    raise ValueError("SZSE official industry format changed")
                securities.append(_security(row["A股代码"], row["A股简称"], source,
                    observed_at, url, raw_sha256, source_sector=industry[2], industry_code=industry[1],
                    board=row["板块"], listing_date=row["A股上市日期"]))
        else:
            updates = [value for row in workbook[:3] for value in row.values() if value.startswith("Updated as at ")]
            if len(updates) != 1 or not re.fullmatch(r"Updated as at \d{2}/\d{2}/\d{4}", updates[0]):
                raise ValueError("HKEX effective date absent or changed")
            effective = datetime.strptime(updates[0][14:], "%d/%m/%Y").date().isoformat()
            rows = _table(workbook, {"Stock Code", "Name of Securities", "Category", "Sub-Category", "ISIN", "Trading Currency"})
            for row in rows:
                category = row["Category"]
                if category != "Equity":
                    excluded["non_equity"] += 1
                    continue
                if row["Sub-Category"] not in {"Equity Securities (Main Board)", "Equity Securities (GEM)"}:
                    excluded["other_equity_subcategory"] += 1
                    continue
                if row["Trading Currency"] != "HKD":
                    excluded["non_HKD_counter"] += 1
                    continue
                securities.append(_security(row["Stock Code"], row["Name of Securities"], source,
                    observed_at, url, raw_sha256, effective_at=effective,
                    board=row["Sub-Category"], isin=row["ISIN"], trading_currency="HKD"))
    securities = _unique(securities)
    return {"source": source, "securities": securities, "count": len(securities),
            "effective_at": effective, "excluded": dict(excluded),
            "usable_on_observation_date": effective is None or effective <= _market_date(observed_at)}


def _report(results, failures, observed_at, requested):
    securities = _unique([row for result in results for row in result["securities"]]) if results else []
    future_sources = [result["source"] for result in results if not result["usable_on_observation_date"]]
    included = {result["source"] for result in results}
    return {"schema_version": 1, "status": "partial" if failures else "blocked_future_effective_date" if future_sources else "completed",
            "observed_at": observed_at, "requested_sources": list(requested),
            "sources": [{key: value for key, value in result.items() if key != "securities"} for result in results],
            "failures": failures, "future_effective_sources": future_sources, "securities": securities,
            "coverage": {"security_count": len(securities), "by_market": dict(Counter(row["market"] for row in securities)),
                         "by_security_type": dict(Counter(row["security_type"] for row in securities)),
                         "eligible_on_observation_date": sum(row["universe_as_of"] <= _market_date(observed_at) for row in securities),
                         "all_a_shares": False,
                         "missing_exchanges": [exchange for source, exchange in (("sse", "SSE"), ("szse", "SZSE"), ("bse", "BSE"), ("hkex", "HKEX")) if source not in included],
                         "technology_manufacturing_classified": 0,
                         "official_industry_scope_suggestions": dict(Counter(row["scope_suggestion"] for row in securities if row["scope_suggestion"]))},
            "limitations": ["BSE current directory is not supported; A-share coverage is incomplete.",
                            "SSE STAR directory includes depositary receipts (689 prefix), retained with an explicit security_type.",
                            "HKEX includes HKD Main Board/GEM equity categories only; other products/counters are excluded.",
                            "These are observed security directories, not verified historical membership or tradability.",
                            "sector/scope remain unverified; source_sector is a provider classification only.",
                            "Public access does not itself establish commercial redistribution rights."]}


def _check_staging_paths(staging, destination, expected_parent):
    if (staging.resolve().parent != expected_parent or destination.resolve().parent != expected_parent
            or not staging.name.startswith(".universe-") or staging.is_symlink()):
        raise ValueError("snapshot staging/destination escaped the checked parent directory")


def acquire_snapshot(destination, *, download=False, sources=DEFAULT_SOURCES, timeout=20.0, retries=2):
    """Collect into a NEW snapshot directory. Failure never reuses old/demo data."""
    if not download:
        raise ValueError("network access requires explicit download=True / --download")
    if not sources or len(set(sources)) != len(sources) or any(source not in DEFAULT_SOURCES for source in sources):
        raise ValueError("sources must be a nonempty unique subset of sse, szse, hkex")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("snapshot destination already exists; use a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_parent = destination.parent
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    observed_at = started_at
    staging = Path(tempfile.mkdtemp(prefix=".universe-", dir=destination.parent))
    results, failures, entries = [], [], []
    try:
        for source in sources:
            pages, records = [], []
            entry = {"source": source, "pages": records}
            entries.append(entry)
            try:
                number, total_pages = 1, 1
                while number <= total_pages:
                    url = _sse_url(number) if source == "sse" else {"szse": SZSE_URL, "hkex": HKEX_URL}[source]
                    data, transport = _download(source, url, timeout=timeout, retries=retries)
                    raw_name = f"{source}-{number}.raw"
                    (staging / raw_name).write_bytes(data)
                    records.append({"path": raw_name, "source_url": url, "sha256": _hash(data),
                                    "bytes": len(data), **transport})
                    pages.append((data, url))
                    if source == "sse":
                        _, pagination = parse_sse_page(data)
                        total_pages = pagination["pageCount"]
                    number += 1
                result = parse_source(source, pages, observed_at)
                # Low counts require human source review rather than silently accepting a partial file.
                minimum = {"sse": 1000, "szse": 1500, "hkex": 1500}[source]
                if result["count"] < minimum:
                    raise ValueError(f"directory count {result['count']} below review threshold {minimum}")
                entry["status"] = "completed"
                results.append(result)
            except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile, ET.ParseError) as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                failures.append({"source": source, "error": entry["error"]})
        # Availability is the completed collection time, never the request start.
        observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for result in results:
            for row in result["securities"]:
                row["observed_at"] = observed_at
                row["universe_as_of"] = max(_market_date(observed_at), row["source_effective_at"] or _market_date(observed_at))
            result["usable_on_observation_date"] = result["effective_at"] is None or result["effective_at"] <= _market_date(observed_at)
        report = _report(results, failures, observed_at, sources)
        report_bytes = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        (staging / "universe.json").write_bytes(report_bytes)
        manifest = {"schema_version": 1, "observed_at": observed_at, "started_at": started_at, "requested_sources": list(sources),
                    "entries": entries, "universe_sha256": _hash(report_bytes)}
        (staging / "snapshot.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Directory rename on the same filesystem publishes the entire immutable snapshot.
        _check_staging_paths(staging, destination, expected_parent)
        if destination.exists():
            raise FileExistsError("snapshot destination was created concurrently")
        staging.rename(destination)
        return report
    finally:
        if staging.exists():
            _check_staging_paths(staging, destination, expected_parent)
            shutil.rmtree(staging)


def replay_snapshot(snapshot_dir):
    """Verify bytes and reconstruct normalized directory entirely offline."""
    root = Path(snapshot_dir).resolve()
    manifest = _json((root / "snapshot.json").read_bytes())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported snapshot manifest")
    observed_at = manifest.get("observed_at")
    _observed(observed_at)
    requested = manifest.get("requested_sources")
    entries = manifest.get("entries")
    if not isinstance(requested, list) or not requested or len(set(requested)) != len(requested) or any(s not in DEFAULT_SOURCES for s in requested):
        raise ValueError("invalid requested sources in snapshot")
    if not isinstance(entries, list) or [item.get("source") for item in entries] != requested:
        raise ValueError("snapshot source inventory mismatch")
    results, failures = [], []
    for entry in entries:
        source = entry["source"]
        pages = []
        for number, page in enumerate(entry.get("pages", []), 1):
            expected_name = f"{source}-{number}.raw"
            if page.get("path") != expected_name:
                raise ValueError("unexpected raw path in snapshot")
            path = root / expected_name
            if path.resolve().parent != root or path.is_symlink():
                raise ValueError("raw snapshot path escapes snapshot directory")
            if not 0 < path.stat().st_size <= MAX_BYTES:
                raise ValueError("invalid raw snapshot size")
            data = path.read_bytes()
            if len(data) != page.get("bytes") or _hash(data) != page.get("sha256"):
                raise ValueError("raw snapshot integrity mismatch")
            pages.append((data, page.get("source_url")))
        if entry.get("status") == "completed":
            results.append(parse_source(source, pages, observed_at))
        elif entry.get("status") == "failed" and isinstance(entry.get("error"), str):
            failures.append({"source": source, "error": entry["error"]})
        else:
            raise ValueError("invalid snapshot source status")
    report = _report(results, failures, observed_at, requested)
    saved = (root / "universe.json").read_bytes()
    if _hash(saved) != manifest.get("universe_sha256") or _json(saved) != report:
        raise ValueError("normalized universe differs from raw snapshot replay")
    return report


def export_scope_candidates(snapshot_dir, output_path, *, as_of, scope_mapping=None):
    """Export observed ordinary-equity candidates, retaining unverified scope/sector.

    The result is a classification work queue, not a ready theme_search directory.
    Exclusions and the source snapshot status remain visible even when some sources
    succeeded. The snapshot's own integrity replay must pass before any export.
    """
    _date(as_of)
    mapping = {"C": "manufacturing", "I": "technology"} if scope_mapping is None else scope_mapping
    if not isinstance(mapping, dict) or not mapping or any(
        not isinstance(code, str) or not re.fullmatch(r"[A-S]", code) or scope not in {"manufacturing", "technology"}
        for code, scope in mapping.items()
    ):
        raise ValueError("scope_mapping must map official A-S industry codes to technology/manufacturing")
    root = Path(snapshot_dir).resolve()
    output = Path(output_path).resolve()
    if output == root or root in output.parents:
        raise ValueError("candidate export must not modify the immutable snapshot")
    if output.exists():
        raise FileExistsError("candidate export already exists")
    report = replay_snapshot(root)
    candidates, excluded = [], Counter()
    for row in report["securities"]:
        if row["universe_as_of"] > as_of:
            excluded["unavailable_as_of"] += 1
        elif row["security_type"] != "equity":
            excluded["non_ordinary_equity"] += 1
        elif row["official_industry_code"] not in mapping:
            excluded["industry_missing_or_outside_mapping"] += 1
        else:
            candidates.append({**row, "scope_suggestion": mapping[row["official_industry_code"]],
                               "scope": None, "sector": None, "classification_status": "unverified",
                               "scope_suggestion_basis": "explicit export mapping of official broad industry code; business review required"})
    result = {"schema_version": 1, "kind": "scope_candidate_review_queue", "status": "requires_classification_review",
              "as_of": as_of, "source_snapshot_status": report["status"],
              "source_snapshot_manifest_sha256": _hash((root / "snapshot.json").read_bytes()),
              "source_coverage": report["coverage"], "scope_mapping": mapping,
              "count": len(candidates), "excluded": dict(excluded), "candidates": candidates,
              "limitations": ["Candidates have official broad-industry suggestions; scope and peer sector have not been reviewed.",
                              "Future-effective and non-equity records are excluded. Missing exchanges remain missing.",
                              "Business documents and financial histories must be collected separately."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation makes an operator-selected existing file impossible to overwrite.
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with output.open("xb") as handle:
        handle.write(data)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true", help="explicitly allow official HTTPS downloads")
    mode.add_argument("--replay", metavar="SNAPSHOT_DIR", help="verify an existing snapshot offline")
    parser.add_argument("--output", default="output/universe/latest", help="new snapshot directory; never overwrites")
    parser.add_argument("--sources", nargs="+", choices=DEFAULT_SOURCES, default=list(DEFAULT_SOURCES))
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        result = replay_snapshot(args.replay) if args.replay else acquire_snapshot(
            args.output, download=True, sources=args.sources, timeout=args.timeout, retries=args.retries)
        print(json.dumps({key: value for key, value in result.items() if key != "securities"}, ensure_ascii=True, indent=2))
        return 2 if result["status"] != "completed" else 0
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile, ET.ParseError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
