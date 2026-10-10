"""Offline protocol/replay tests; fixtures are synthetic, never live market claims."""
import copy
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from xml.sax.saxutils import escape

import universe_sources as u


OBSERVED = "2026-10-10T03:00:00+00:00"


def workbook(rows, *, shared=False):
    strings = []
    xml_rows = []
    for number, row in enumerate(rows, 1):
        cells = []
        for index, value in enumerate(row):
            coordinate = chr(65 + index) + str(number)
            value = str(value)
            if shared:
                strings.append(value)
                cells.append(f'<c r="{coordinate}" t="s"><v>{len(strings)-1}</v></c>')
            else:
                cells.append(f'<c r="{coordinate}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
        xml_rows.append('<row r="' + str(number) + '">' + ''.join(cells) + '</row>')
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as z:
        ns = u.NS["s"]
        z.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{ns}"><sheetData>{"".join(xml_rows)}</sheetData></worksheet>')
        if shared:
            z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{ns}">' + ''.join('<si><t>'+escape(v)+'</t></si>' for v in strings) + '</sst>')
    return data.getvalue()


def sse_payload(number=1, total=3, size=2, *, wrong_page=False):
    rows = [{"A_STOCK_CODE": f"{600000+i:06d}", "SEC_NAME_CN": f"Company {i}",
             "STOCK_TYPE": "1", "STATE_CODE": "2", "CSRC_CODE_DESC": "Manufacturing"}
            for i in range((number-1)*size, min(number*size, total))]
    return json.dumps({"result": rows, "actionErrors": [], "fieldErrors": {},
                       "pageHelp": {"pageNo": number + int(wrong_page), "pageCount": (total+size-1)//size,
                                    "pageSize": size, "total": total, "data": rows}}).encode()


def szse_payload(count=1):
    return workbook([["A股代码", "A股简称", "A股上市日期", "所属行业", "板块"]] +
                    [[f"{i+1:06d}", f"Test {i}", "2000-01-01", "C 制造业", "主板"] for i in range(count)])


def hk_payload(*, date="12/10/2026", shared=True):
    return workbook([
        ["List of Securities"], ["Updated as at " + date],
        ["Stock Code", "Name of Securities", "Category", "Sub-Category", "ISIN", "Trading Currency"],
        ["00001", "TEST HOLDINGS", "Equity", "Equity Securities (Main Board)", "HK123", "HKD"],
        ["80001", "TEST RMB", "Equity", "Equity Securities (Main Board)", "HK123", "RMB"],
        ["08001", "TEST GEM", "Equity", "Equity Securities (GEM)", "HK456", "HKD"],
        ["01001", "WARRANT", "Derivative Warrants", "", "HK789", "HKD"],
        ["02001", "FUND", "Exchange Traded Products", "Exchange Traded Funds", "HK000", "HKD"],
        ["03001", "INV COMPANY", "Equity", "Investment Companies", "HK555", "HKD"],
    ], shared=shared)


class UniverseParsingTests(unittest.TestCase):
    def test_sse_requires_every_page_and_preserves_provider_sector(self):
        pages = [(sse_payload(i), u._sse_url(i)) for i in (1, 2)]
        report = u.parse_source("sse", pages, OBSERVED)
        self.assertEqual(report["count"], 3)
        row = report["securities"][0]
        self.assertIsNone(row["scope"])
        self.assertIsNone(row["sector"])
        self.assertEqual(row["source_sector"], "Manufacturing")
        self.assertEqual(row["classification_status"], "unverified")
        self.assertEqual(row["universe_as_of"], "2026-10-10")
        with self.assertRaisesRegex(ValueError, "missing pages"):
            u.parse_source("sse", pages[:1], OBSERVED)

    def test_sse_checks_server_pagination_not_requested_page_size(self):
        _, page = u.parse_sse_page(sse_payload())
        self.assertEqual(page["pageSize"], 2)
        self.assertEqual(page["pageCount"], 2)
        obj = json.loads(sse_payload())
        obj["result"] = obj["result"][:1]
        with self.assertRaisesRegex(ValueError, "truncated"):
            u.parse_sse_page(json.dumps(obj).encode())

    def test_sse_rejects_total_drift_wrong_page_and_duplicate_ticker(self):
        for second in [sse_payload(2, 4), sse_payload(2, wrong_page=True)]:
            with self.assertRaises(ValueError):
                u.parse_source("sse", [(sse_payload(), u._sse_url(1)), (second, u._sse_url(2))], OBSERVED)
        second = json.loads(sse_payload(2))
        second["result"][0]["A_STOCK_CODE"] = "600000"
        second["pageHelp"]["data"] = second["result"]
        with self.assertRaisesRegex(ValueError, "duplicate ticker"):
            u.parse_source("sse", [(sse_payload(), u._sse_url(1)), (json.dumps(second).encode(), u._sse_url(2))], OBSERVED)

    def test_sse_rejects_error_envelope_duplicate_keys_and_nonfinite(self):
        for raw in [b'{"result":[],"result":[]}', b'{"x":NaN}', b'{"actionErrors":["bad"]}']:
            with self.assertRaises(ValueError):
                u.parse_sse_page(raw)

    def test_szse_inline_strings_preserve_leading_zero(self):
        result = u.parse_source("szse", [(szse_payload(), u.SZSE_URL)], OBSERVED)
        self.assertEqual(result["securities"][0]["ticker"], "000001.SZ")
        self.assertIsNone(result["securities"][0]["scope"])

    def test_szse_schema_drift_empty_or_wrong_market_is_failure(self):
        for data in [workbook([["code", "name"]]), szse_payload(0),
                     workbook([["A股代码", "A股简称", "A股上市日期", "所属行业", "板块"],
                               ["600001", "Not Shenzhen", "2000-01-01", "C", "main"]])]:
            with self.assertRaises(ValueError):
                u.parse_source("szse", [(data, u.SZSE_URL)], OBSERVED)

    def test_hkex_filters_derivatives_funds_and_dual_counters(self):
        result = u.parse_source("hkex", [(hk_payload(), u.HKEX_URL)], OBSERVED)
        self.assertEqual([r["ticker"] for r in result["securities"]], ["00001.HK", "08001.HK"])
        self.assertEqual(result["excluded"], {"non_HKD_counter": 1, "non_equity": 2, "other_equity_subcategory": 1})

    def test_hkex_future_effective_date_never_backdates_membership(self):
        result = u.parse_source("hkex", [(hk_payload(), u.HKEX_URL)], OBSERVED)
        self.assertFalse(result["usable_on_observation_date"])
        self.assertEqual(result["securities"][0]["universe_as_of"], "2026-10-12")
        older = u.parse_source("hkex", [(hk_payload(date="09/10/2026"), u.HKEX_URL)], OBSERVED)
        self.assertEqual(older["securities"][0]["universe_as_of"], "2026-10-10")

    def test_future_source_blocks_report_and_market_timezone_is_used(self):
        result = u.parse_source("hkex", [(hk_payload(), u.HKEX_URL)], OBSERVED)
        report = u._report([result], [], OBSERVED, ["hkex"])
        self.assertEqual(report["status"], "blocked_future_effective_date")
        self.assertEqual(report["coverage"]["eligible_on_observation_date"], 0)
        self.assertEqual(report["future_effective_sources"], ["hkex"])
        late = u.parse_source("szse", [(szse_payload(), u.SZSE_URL)], "2026-10-10T17:00:00+00:00")
        self.assertEqual(late["securities"][0]["universe_as_of"], "2026-10-11")

    def test_broad_official_industry_is_only_a_scope_suggestion(self):
        row = u.parse_source("szse", [(szse_payload(), u.SZSE_URL)], OBSERVED)["securities"][0]
        self.assertEqual(row["official_industry_code"], "C")
        self.assertEqual(row["scope_suggestion"], "manufacturing")
        self.assertEqual(row["classification_status"], "unverified")
        self.assertIsNone(row["scope"])

    def test_unknown_or_empty_sources_and_missing_timezone_fail(self):
        for source, pages, stamp in [("bse", [], OBSERVED), ("hkex", [], OBSERVED),
                                     ("szse", [(szse_payload(), u.SZSE_URL)], "2026-10-10")]:
            with self.assertRaises(ValueError):
                u.parse_source(source, pages, stamp)

    def test_xlsx_duplicate_member_formula_and_xml_entities_fail(self):
        for xml in ['<worksheet xmlns="'+u.NS['s']+'"><sheetData><row><c r="A1"><f>1+1</f><v>2</v></c></row></sheetData></worksheet>',
                    '<!DOCTYPE a [<!ENTITY x "a">]><worksheet/>']:
            out = io.BytesIO()
            with zipfile.ZipFile(out, "w") as z:
                z.writestr("xl/worksheets/sheet1.xml", xml)
            with self.assertRaises(ValueError):
                u._xlsx_rows(out.getvalue())


class UniverseSnapshotTests(unittest.TestCase):
    def test_network_requires_opt_in_and_existing_snapshot_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d, patch.object(u, "_download") as download:
            with self.assertRaises(ValueError):
                u.acquire_snapshot(Path(d)/"new")
            with self.assertRaises(FileExistsError):
                u.acquire_snapshot(d, download=True)
            download.assert_not_called()

    def test_cleanup_guard_rejects_unchecked_paths(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d).resolve()
            u._check_staging_paths(base/".universe-ok", base/"final", base)
            for staging, destination in [(base/"not-temp", base/"final"),
                                         (base.parent/".universe-escape", base/"final"),
                                         (base/".universe-ok", base.parent/"outside")]:
                with self.assertRaisesRegex(ValueError, "escaped"):
                    u._check_staging_paths(staging, destination, base)

    def test_scope_export_filters_future_and_cdr_without_verified_scope(self):
        rows = [u._security("600001", "Manufacturing", "sse", OBSERVED, u._sse_url(1), "a"*64, industry_code="C"),
                u._security("689009", "CDR", "sse", OBSERVED, u._sse_url(1), "a"*64, industry_code="C"),
                u._security("00001", "Future", "hkex", OBSERVED, u.HKEX_URL, "b"*64, effective_at="2026-10-12", industry_code="C"),
                u._security("600002", "Finance", "sse", OBSERVED, u._sse_url(1), "a"*64, industry_code="J")]
        source = {"securities": rows, "status": "blocked_future_effective_date", "coverage": {"all_a_shares": False}}
        with tempfile.TemporaryDirectory() as d, patch.object(u, "replay_snapshot", return_value=source):
            root = Path(d)/"snapshot"
            root.mkdir()
            (root/"snapshot.json").write_text("{}")
            result = u.export_scope_candidates(root, Path(d)/"candidates.json", as_of="2026-10-10")
            self.assertEqual(result["count"], 1)
            self.assertEqual(result["excluded"], {"non_ordinary_equity": 1, "unavailable_as_of": 1, "industry_missing_or_outside_mapping": 1})
            self.assertIsNone(result["candidates"][0]["scope"])
            self.assertEqual(result["candidates"][0]["scope_suggestion"], "manufacturing")
            with self.assertRaises(FileExistsError):
                u.export_scope_candidates(root, Path(d)/"candidates.json", as_of="2026-10-10")
            with self.assertRaises(ValueError):
                u.export_scope_candidates(root, root/"modify.json", as_of="2026-10-10")

    def test_actual_bytes_replay_and_tamper_detection(self):
        payload = szse_payload(1500)
        with tempfile.TemporaryDirectory() as d, patch.object(u, "_download", return_value=(payload, {"attempts": 1})):
            dest = Path(d)/"snapshot"
            original = u.acquire_snapshot(dest, download=True, sources=("szse",))
            with patch.object(u, "_download", side_effect=AssertionError("offline only")):
                replay = u.replay_snapshot(dest)
            self.assertEqual(replay, original)
            self.assertFalse(replay["coverage"]["all_a_shares"])
            (dest/"szse-1.raw").write_bytes(payload + b"tamper")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                u.replay_snapshot(dest)

    def test_partial_fetch_never_falls_back_to_other_or_demo_data(self):
        with tempfile.TemporaryDirectory() as d, patch.object(u, "_download", side_effect=TimeoutError("unavailable")):
            dest = Path(d)/"snapshot"
            result = u.acquire_snapshot(dest, download=True, sources=("sse", "hkex"))
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["securities"], [])
            self.assertEqual(len(result["failures"]), 2)
            self.assertEqual(u.replay_snapshot(dest), result)

    def test_low_count_is_review_failure_not_success(self):
        with tempfile.TemporaryDirectory() as d, patch.object(u, "_download", return_value=(szse_payload(), {})):
            result = u.acquire_snapshot(Path(d)/"new", download=True, sources=("szse",))
            self.assertEqual(result["status"], "partial")
            self.assertIn("below review threshold", result["failures"][0]["error"])

    def test_replay_rejects_path_traversal_and_normalized_tampering(self):
        with tempfile.TemporaryDirectory() as d, patch.object(u, "_download", return_value=(szse_payload(1500), {})):
            dest = Path(d)/"snapshot"
            u.acquire_snapshot(dest, download=True, sources=("szse",))
            manifest_path = dest/"snapshot.json"
            manifest = json.loads(manifest_path.read_bytes())
            bad = copy.deepcopy(manifest)
            bad["entries"][0]["pages"][0]["path"] = "../outside.raw"
            manifest_path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected raw path"):
                u.replay_snapshot(dest)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (dest/"universe.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                u.replay_snapshot(dest)

    def test_http_only_reviewed_endpoints_and_finite_limits(self):
        for source, url, kwargs in [("hkex", "http://127.0.0.1/private", {}),
                                    ("hkex", u.HKEX_URL, {"timeout": float("nan")}),
                                    ("hkex", u.HKEX_URL, {"retries": True}),
                                    ("hkex", u.HKEX_URL, {"max_bytes": -1})]:
            with self.assertRaises(ValueError):
                u._download(source, url, **kwargs)

    def test_http_transient_failure_retries_but_404_does_not(self):
        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Length": "2"}
            def geturl(self):
                return u.HKEX_URL
        with patch.object(u, "build_opener") as opener, patch.object(u.time, "sleep") as sleep:
            opener.return_value.open.side_effect = [HTTPError(u.HKEX_URL, 503, "unavailable", {}, None), Response(b"ok")]
            body, info = u._download("hkex", u.HKEX_URL)
            self.assertEqual(body, b"ok")
            self.assertEqual(info["attempts"], 2)
            sleep.assert_called_once()
        with patch.object(u, "build_opener") as opener:
            opener.return_value.open.side_effect = HTTPError(u.HKEX_URL, 404, "absent", {}, None)
            with self.assertRaises(HTTPError):
                u._download("hkex", u.HKEX_URL)
            self.assertEqual(opener.return_value.open.call_count, 1)


if __name__ == "__main__":
    unittest.main()
