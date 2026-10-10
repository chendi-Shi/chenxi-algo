"""Discover official CNINFO/HKEX filing links, with bounded requests and replay.

No PDF bodies are downloaded. HTTP response snapshots are private local working
files; production use still needs disclosure completeness/availability monitoring.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener

from universe_sources import _NoRedirect, _check_staging_paths, _date, _hash, _json


CN_STOCKS = "https://www.cninfo.com.cn/new/data/szse_stock.json"
CN_SEARCH = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
HK_STOCKS = "https://www1.hkexnews.hk/ncms/script/eds/activestock_sehk_e.json"
HK_SEARCH = "https://www1.hkexnews.hk/search/titleSearchServlet.do"
HK_TAXONOMY = "https://www1.hkexnews.hk/ncms/script/eds/tiertwo_e.json"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
TZ = timezone(timedelta(hours=8))


class SourceError(ValueError):
    """A source cannot substantiate a complete answer for this request."""


def _clean(value):
    if not isinstance(value, str):
        raise SourceError("expected text field")
    return " ".join(html.unescape(re.sub(r"<[^>]*>", " ", value)).split())


def _transport(url, form, *, timeout=20.0, retries=2):
    if not (url in {CN_STOCKS, CN_SEARCH, HK_STOCKS, HK_TAXONOMY} or url.startswith(HK_SEARCH + "?")):
        raise SourceError("unreviewed disclosure endpoint")
    if not math.isfinite(timeout) or not 0 < timeout <= 60 or type(retries) is not int or not 0 <= retries <= 3:
        raise ValueError("invalid timeout/retries")
    payload = urlencode(form).encode("utf-8") if form is not None else None
    provider_cn = url.startswith("https://www.cninfo.com.cn/")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; chenxi-algo disclosure research)", "Accept-Encoding": "identity",
               "Referer": "https://www.cninfo.com.cn/" if provider_cn else "https://www1.hkexnews.hk/search/titlesearch.xhtml?lang=en"}
    if provider_cn:
        headers.update({"Origin": "https://www.cninfo.com.cn", "X-Requested-With": "XMLHttpRequest",
                        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"})
    request = Request(url, data=payload, headers=headers)
    opener = build_opener(_NoRedirect())
    for attempt in range(retries + 1):
        started = time.monotonic()
        try:
            with opener.open(request, timeout=timeout) as response:
                if response.status != 200 or response.geturl() != url:
                    raise SourceError("unexpected source status or redirect")
                length = response.headers.get("Content-Length")
                if length is not None and (not length.isdigit() or int(length) > MAX_RESPONSE_BYTES):
                    raise SourceError("excessive or invalid Content-Length")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise SourceError("unexpected compressed HTTP response")
                body = bytearray()
                while True:
                    if time.monotonic() - started > timeout:
                        raise TimeoutError("source elapsed time limit exceeded")
                    block = response.read(min(65536, MAX_RESPONSE_BYTES + 1 - len(body)))
                    if not block:
                        break
                    body.extend(block)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise SourceError("source response size limit exceeded")
                if not body or (length is not None and len(body) != int(length)):
                    raise SourceError("empty or truncated source response")
                return bytes(body), {"attempts": attempt + 1, "content_type": response.headers.get("Content-Type")}
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            if (isinstance(exc, HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}) or "CERTIFICATE_VERIFY_FAILED" in str(exc) or attempt == retries:
                raise SourceError(f"{type(exc).__name__}: {exc}") from exc
            time.sleep(min(2 ** attempt, 4))
    raise RuntimeError("unreachable")


class _Client:
    def __init__(self, root, *, events=None, interval=1.0, timeout=20.0, retries=2, max_requests=10000):
        self.root = root
        self.replay = events is not None
        self.events = events if events is not None else []
        self.position = 0
        self.last_request = None
        self.interval, self.timeout, self.retries, self.max_requests = interval, timeout, retries, max_requests

    def get(self, url, form=None):
        if self.position >= self.max_requests:
            raise SourceError("global request budget exceeded")
        if self.replay:
            if self.position >= len(self.events):
                raise SourceError("snapshot does not contain the required request")
            event = self.events[self.position]
            self.position += 1
            if event.get("url") != url or event.get("form") != form:
                raise SourceError("offline request differs from captured request")
            if event.get("status") == "failed":
                raise SourceError(event["error"])
            filename = f"response-{self.position:05d}.json"
            if event.get("path") != filename:
                raise SourceError("unexpected raw response path")
            path = self.root / filename
            if path.resolve().parent != self.root or path.is_symlink() or not 0 < path.stat().st_size <= MAX_RESPONSE_BYTES:
                raise SourceError("invalid snapshot response path/size")
            data = path.read_bytes()
            if len(data) != event.get("bytes") or _hash(data) != event.get("sha256"):
                raise SourceError("raw disclosure response integrity mismatch")
        else:
            if self.last_request is not None:
                time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            self.position += 1
            event = {"url": url, "form": form, "requested_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            self.events.append(event)
            try:
                data, meta = _transport(url, form, timeout=self.timeout, retries=self.retries)
                filename = f"response-{self.position:05d}.json"
                (self.root / filename).write_bytes(data)
                event.update({"status": "completed", "path": filename, "sha256": _hash(data), "bytes": len(data), **meta})
            except (ValueError, OSError) as exc:
                event.update({"status": "failed", "error": str(exc)})
                raise SourceError(str(exc)) from exc
        try:
            return _json(data)
        except (ValueError, UnicodeError) as exc:
            raise SourceError(f"invalid source JSON: {exc}") from exc


def _request_definition(securities, start_date, end_date, categories, max_pages, page_size, max_requests):
    first, last = _date(start_date), _date(end_date)
    if first > last or (last - first).days > 366:
        raise ValueError("date window must be ordered and at most 367 calendar days")
    if last > datetime.now(TZ).date():
        raise ValueError("end_date cannot be later than the current Hong Kong/China date")
    if not isinstance(securities, list) or not 1 <= len(securities) <= 5000:
        raise ValueError("securities must contain 1-5000 explicitly selected companies")
    selected = []
    for row in securities:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"].strip():
            raise ValueError("every security requires name/ticker/market")
        ticker = row.get("ticker", "")
        if re.fullmatch(r"(?:60|68)\d{4}\.SH|(?:00|30)\d{4}\.SZ", ticker):
            expected_market = "A"
        elif re.fullmatch(r"\d{5}\.HK", ticker):
            expected_market = "HK"
        else:
            raise ValueError("unsupported ticker; only reviewed SH/SZ/HK adapters are available")
        if row.get("market") != expected_market:
            raise ValueError("ticker and market disagree")
        selected.append({"ticker": ticker, "name": row["name"].strip(), "market": expected_market})
    if len({row["ticker"] for row in selected}) != len(selected):
        raise ValueError("duplicate selected ticker")
    categories = list(categories)
    if not categories or len(set(categories)) != len(categories) or any(x not in {"all", "annual_report", "announcement"} for x in categories) or ("all" in categories and len(categories) != 1):
        raise ValueError("categories must be ('all',) or a subset of annual_report/announcement")
    if type(max_pages) is not int or not 1 <= max_pages <= 100 or type(page_size) is not int or not 1 <= page_size <= 1000:
        raise ValueError("max_pages must be 1-100 and HK page_size 1-1000")
    if type(max_requests) is not int or not 1 <= max_requests <= 10000:
        raise ValueError("max_requests must be 1-10000")
    return {"securities": sorted(selected, key=lambda r: r["ticker"]), "start_date": start_date, "end_date": end_date,
            "categories": sorted(categories), "max_pages": max_pages, "page_size": page_size, "max_requests": max_requests}


def _stock_ids(provider, payload):
    rows = payload.get("stockList") if provider == "cninfo" and isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise SourceError("empty or changed official stock lookup")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SourceError("malformed stock lookup row")
        if provider == "cninfo":
            if row.get("category") not in {"A股", "CDR", "B股"}:
                raise SourceError("unknown CNINFO category in stock lookup")
            if row["category"] == "B股":
                continue
            code, identifier = row.get("code"), row.get("orgId")
            valid = isinstance(code, str) and re.fullmatch(r"\d{6}", code) and isinstance(identifier, str) and re.fullmatch(r"[A-Za-z0-9]+", identifier)
        else:
            code, identifier = row.get("c"), row.get("i")
            valid = isinstance(code, str) and re.fullmatch(r"\d{5}", code) and type(identifier) is int and identifier > 0
        if not valid or code in result:
            raise SourceError("invalid or duplicate official stock identifier")
        result[code] = str(identifier)
    return result


def _official_url(provider, path):
    if not isinstance(path, str):
        raise SourceError("missing filing path")
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or "\\" in path or "%" in path or ".." in path.split("/"):
        raise SourceError("unsafe or unexpected filing path")
    if provider == "cninfo":
        if not re.fullmatch(r"finalpage/\d{4}-\d{2}-\d{2}/[A-Za-z0-9_.-]+", path):
            raise SourceError("unexpected CNINFO filing path")
        return "https://static.cninfo.com.cn/" + path
    if not re.fullmatch(r"/listedco/listconews/(?:sehk|gem)/\d{4}/\d{4}/[A-Za-z0-9_.-]+", path):
        raise SourceError("unexpected HKEX filing path")
    return "https://www1.hkexnews.hk" + path


def _record(provider, ticker, row, start_date, end_date):
    if not isinstance(row, dict):
        raise SourceError("malformed filing row")
    code = ticker.split(".")[0]
    if provider == "cninfo":
        if row.get("secCode") != code:
            raise SourceError("CNINFO returned a different security")
        identifier = row.get("announcementId")
        stamp = row.get("announcementTime")
        if not isinstance(identifier, str) or not identifier.isdigit() or type(stamp) is not int or not 0 < stamp < 4102444800000:
            raise SourceError("invalid CNINFO filing identity/date")
        available = datetime.fromtimestamp(stamp / 1000, TZ).date().isoformat()
        title = _clean(row.get("announcementTitle"))
        url = _official_url(provider, row.get("adjunctUrl"))
        annual = bool(re.fullmatch(r"\d{4}年年度报告(?:[（(](?:修订|更新|更正|全文)[^）)]*[）)])?", title) or re.fullmatch(r"\d{4}\s+annual\s+report", title, re.I))
        file_type = row.get("adjunctType")
        released, precision = None, "day"
        source_category = row.get("announcementType")
    else:
        codes = _clean(row.get("STOCK_CODE")).split()
        if not all(re.fullmatch(r"\d{5}", item) for item in codes) or code not in codes:
            raise SourceError("HKEX returned a different security")
        identifier = row.get("NEWS_ID")
        if not isinstance(identifier, str) or not identifier.isdigit():
            raise SourceError("invalid HKEX filing identifier")
        timestamp = _clean(row.get("DATE_TIME"))
        if not re.fullmatch(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}", timestamp):
            raise SourceError("HKEX release timestamp format changed")
        moment = datetime.strptime(timestamp, "%d/%m/%Y %H:%M").replace(tzinfo=TZ)
        available, released, precision = moment.date().isoformat(), moment.isoformat(timespec="minutes"), "minute"
        title = _clean(row.get("TITLE"))
        url = _official_url(provider, row.get("FILE_LINK"))
        source_category = _clean(row.get("LONG_TEXT", ""))
        labels = re.search(r"\[([^\]]+)\]$", source_category)
        annual = bool(labels and "Annual Report" in [item.strip() for item in labels[1].split("/")])
        file_type = row.get("FILE_TYPE")
    if not title or not start_date <= available <= end_date:
        raise SourceError("empty title or filing outside requested date window")
    is_pdf = isinstance(file_type, str) and file_type.upper() == "PDF" and url.lower().endswith(".pdf")
    if isinstance(file_type, str) and file_type.upper() == "PDF" and not url.lower().endswith(".pdf"):
        raise SourceError("PDF filing type disagrees with source URL")
    return {"document_id": f"{provider}-{code}-{identifier}", "ticker": ticker, "title": title,
            "source_url": url, "available_at": available, "released_at": released, "date_precision": precision,
            "provider": provider, "provider_record_id": identifier, "provider_category": source_category,
            "source_type": "annual_report" if annual else "announcement", "is_pdf": is_pdf,
            "date_basis": "official_release"}


def _cninfo(client, ticker, identifier, request):
    records, expected_total, seen, pages = [], None, set(), 0
    for number in range(1, request["max_pages"] + 1):
        form = {"stock": ticker[:6] + "," + identifier, "tabName": "fulltext", "pageSize": "30", "pageNum": str(number),
                "column": "sse" if ticker.endswith(".SH") else "szse",
                "category": "category_ndbg_szsh" if request["categories"] == ["annual_report"] else "",
                "seDate": request["start_date"] + "~" + request["end_date"], "isHLtitle": "false", "sortName": "time", "sortType": "desc"}
        obj = client.get(CN_SEARCH, form)
        if not isinstance(obj, dict) or type(obj.get("totalAnnouncement")) is not int or type(obj.get("totalRecordNum")) is not int or type(obj.get("hasMore")) is not bool:
            raise SourceError("CNINFO response pagination schema changed")
        total = obj["totalAnnouncement"]
        if total < 0 or total != obj["totalRecordNum"] or (expected_total is not None and total != expected_total):
            raise SourceError("CNINFO total count changed or disagrees")
        expected_total = total
        rows = obj.get("announcements")
        if total == 0 and rows is None:
            rows = []
        expected_length = min(30, max(0, total - (number - 1) * 30))
        if not isinstance(rows, list) or len(rows) != expected_length or obj["hasMore"] != (number * 30 < total):
            raise SourceError("CNINFO truncated page or inconsistent hasMore")
        pages += 1
        for row in rows:
            record = _record("cninfo", ticker, row, request["start_date"], request["end_date"])
            if record["document_id"] in seen:
                raise SourceError("CNINFO duplicate filing across pages; possible pagination drift")
            seen.add(record["document_id"])
            records.append(record)
        if not obj["hasMore"]:
            return records, {"pages": pages, "provider_total": total, "totalpages_field_ignored": "observed provider field is 0 even for nonempty results"}
    raise SourceError("CNINFO pagination budget exceeded; narrow the date window")


def _hkex(client, ticker, identifier, request):
    pending = [(request["start_date"], request["end_date"])]
    records, seen, pages, splits, leaf_totals = [], set(), 0, 0, 0
    while pending:
        if pages >= request["max_pages"]:
            raise SourceError("HKEX search-window budget exceeded; narrow the date window")
        start, end = pending.pop(0)
        fields = {"sortDir": "0", "sortByOptions": "DateTime", "category": "0", "market": "SEHK", "stockId": identifier,
                  "documentType": "-1", "fromDate": start.replace("-", ""), "toDate": end.replace("-", ""), "title": "", "searchType": "0",
                  "t1code": "40000" if request["categories"] == ["annual_report"] else "-2",
                  "t2Gcode": "-2", "t2code": "40100" if request["categories"] == ["annual_report"] else "-2",
                  "rowRange": str(request["page_size"]), "lang": "E"}
        obj = client.get(HK_SEARCH + "?" + urlencode(fields))
        pages += 1
        if not isinstance(obj, dict) or not isinstance(obj.get("result"), str) or type(obj.get("recordCnt")) is not int or type(obj.get("loadedRecord")) is not int or type(obj.get("hasNextRow")) is not bool:
            raise SourceError("HKEX pagination schema changed")
        rows = _json(obj["result"])
        total = obj["recordCnt"]
        if not isinstance(rows, list) or total < 0 or obj["loadedRecord"] != len(rows) or len(rows) > total or obj.get("rowRange") != request["page_size"]:
            raise SourceError("HKEX count fields disagree")
        for row in rows:
            if not isinstance(row, dict) or str(row.get("TOTAL_COUNT")) != str(total):
                raise SourceError("HKEX per-row total disagrees")
            _record("hkex", ticker, row, start, end)
        truncated = total > len(rows) or obj["hasNextRow"]
        if truncated:
            if start == end:
                raise SourceError("HKEX same-day result exceeds row cap; completeness cannot be established")
            middle = _date(start) + (_date(end) - _date(start)) // 2
            pending[0:0] = [(start, middle.isoformat()), ((middle + timedelta(days=1)).isoformat(), end)]
            splits += 1
            continue
        leaf_totals += total
        for row in rows:
            record = _record("hkex", ticker, row, start, end)
            if record["document_id"] in seen:
                raise SourceError("duplicate HKEX filing across disjoint windows")
            seen.add(record["document_id"])
            records.append(record)
    if leaf_totals != len(records):
        raise SourceError("HKEX leaf totals do not reconcile")
    return records, {"pages": pages, "date_window_splits": splits, "provider_total": leaf_totals}


def _run(client, request):
    lookups, lookup_errors = {}, {}
    providers = sorted({"cninfo" if row["market"] == "A" else "hkex" for row in request["securities"]})
    for provider in providers:
        try:
            lookups[provider] = _stock_ids(provider, client.get(CN_STOCKS if provider == "cninfo" else HK_STOCKS))
            if provider == "hkex" and request["categories"] == ["annual_report"]:
                taxonomy = client.get(HK_TAXONOMY)
                if not isinstance(taxonomy, list) or not any(isinstance(row, dict) and row.get("code") == "40100" and row.get("t1code") == "40000" and row.get("name") == "Annual Report" for row in taxonomy):
                    raise SourceError("HKEX annual-report taxonomy changed")
        except (ValueError, OSError, TypeError, KeyError, OverflowError) as exc:
            lookup_errors[provider] = str(exc)
    records, audit, failures = [], [], []
    for company in request["securities"]:
        ticker = company["ticker"]
        provider = "cninfo" if company["market"] == "A" else "hkex"
        before = client.position
        try:
            if provider in lookup_errors:
                raise SourceError(lookup_errors[provider])
            code = ticker.split(".")[0]
            identifier = lookups[provider].get(code)
            if identifier is None:
                raise SourceError("selected security missing from official identifier lookup")
            found, detail = (_cninfo if provider == "cninfo" else _hkex)(client, ticker, identifier, request)
            selected = [row for row in found if "all" in request["categories"] or row["source_type"] in request["categories"]]
            records.extend(selected)
            audit.append({"ticker": ticker, "provider": provider, "status": "completed", "selected": len(selected),
                          "category_filtered": len(found) - len(selected), "requests": client.position - before, **detail})
        except (ValueError, OSError, TypeError, KeyError, OverflowError) as exc:
            failures.append({"ticker": ticker, "provider": provider, "error": str(exc)})
            audit.append({"ticker": ticker, "provider": provider, "status": "failed", "requests": client.position - before})
    records.sort(key=lambda row: (row["ticker"], row["available_at"], row["document_id"]))
    manifest = [{key: row[key] for key in ("document_id", "ticker", "title", "source_url", "available_at", "source_type", "date_basis")} | {"format": "pdf"}
                for row in records if row["is_pdf"]]
    return {"schema_version": 1, "status": "partial" if failures else "completed", "request": request,
            "securities": request["securities"], "start_date": request["start_date"], "end_date": request["end_date"],
            "records": records, "ingest_manifest": manifest, "failures": failures, "audit": audit,
            "coverage": {"requested_companies": len(request["securities"]), "completed_companies": len(request["securities"]) - len(failures),
                         "failed_tickers": [row["ticker"] for row in failures], "records": len(records), "pdf_links": len(manifest),
                         "non_pdf_records": len(records) - len(manifest), "http_requests": client.position},
            "limitations": ["Counts reconcile with returned provider pages/windows; no independent proof of issuer disclosure completeness or uptime.",
                            "CNINFO release precision is a calendar day; no actual midnight publication time is asserted.",
                            "HKEX supports currently resolved securities; historic/delisted identifier coverage is not implemented.",
                            "Failed company results are discarded; partial status must block a complete daily update.",
                            "PDF contents, OCR, financial facts, and commercial data rights are separate checks."]}


def discover_disclosures(securities, start_date, end_date, *, download=False, snapshot_dir,
                         categories=("all",), max_pages=20, min_interval_seconds=1.0,
                         page_size=100, timeout=20.0, retries=2, max_requests=10000):
    request = _request_definition(securities, start_date, end_date, categories, max_pages, page_size, max_requests)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 60 or type(retries) is not int or not 0 <= retries <= 3:
        raise ValueError("timeout must be 0-60 seconds and retries an integer 0-3")
    if isinstance(min_interval_seconds, bool) or not isinstance(min_interval_seconds, (int, float)) or not math.isfinite(min_interval_seconds) or not 0.5 <= min_interval_seconds <= 60:
        raise ValueError("request interval must be 0.5-60 seconds")
    root = Path(snapshot_dir).resolve()
    if not download:
        metadata = _json((root / "snapshot.json").read_bytes())
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1 or metadata.get("request") != request or not isinstance(metadata.get("events"), list):
            raise ValueError("offline disclosure snapshot does not match the exact requested securities/window/categories/budgets")
        client = _Client(root, events=metadata["events"], max_requests=max_requests)
        report = _run(client, request)
        if client.position != len(client.events):
            raise ValueError("offline snapshot contains unused source responses")
        saved = (root / "disclosures.json").read_bytes()
        if _hash(saved) != metadata.get("report_sha256") or _json(saved) != report:
            raise ValueError("disclosure response replay differs from normalized result")
        return report
    if root.exists():
        raise FileExistsError("disclosure snapshot destination already exists")
    root.parent.mkdir(parents=True, exist_ok=True)
    parent = root.parent
    staging = Path(tempfile.mkdtemp(prefix=".universe-disclosures-", dir=parent))
    try:
        client = _Client(staging, interval=min_interval_seconds, timeout=timeout, retries=retries, max_requests=max_requests)
        report = _run(client, request)
        data = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        (staging / "disclosures.json").write_bytes(data)
        metadata = {"schema_version": 1, "request": request, "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "events": client.events, "report_sha256": _hash(data)}
        (staging / "snapshot.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _check_staging_paths(staging, root, parent)
        if root.exists():
            raise FileExistsError("disclosure destination was created concurrently")
        staging.rename(root)
        return report
    finally:
        if staging.exists():
            _check_staging_paths(staging, root, parent)
            shutil.rmtree(staging)


def replay_disclosures(snapshot_dir):
    """Replay the captured request; caller must inspect returned request/pool/window.

    For an expected pool/window use discover_disclosures(..., download=False),
    which additionally rejects any mismatch with the explicit caller request.
    """
    root = Path(snapshot_dir).resolve()
    metadata = _json((root / "snapshot.json").read_bytes())
    if not isinstance(metadata, dict) or not isinstance(metadata.get("request"), dict):
        raise ValueError("invalid disclosure snapshot request metadata")
    request = metadata["request"]
    return discover_disclosures(request["securities"], request["start_date"], request["end_date"],
                                snapshot_dir=root, categories=request["categories"], max_pages=request["max_pages"],
                                page_size=request["page_size"], max_requests=request["max_requests"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", required=True, help="JSON array of ticker/name/market")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--categories", nargs="+", default=["all"])
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--page-size", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        report = discover_disclosures(_json(Path(args.companies).read_bytes()), args.start, args.end,
                                      download=args.download, snapshot_dir=args.snapshot, categories=args.categories,
                                      max_pages=args.max_pages, page_size=args.page_size)
        print(json.dumps({key: value for key, value in report.items() if key not in {"records", "ingest_manifest"}}, ensure_ascii=True, indent=2))
        return 0 if report["status"] == "completed" else 2
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
