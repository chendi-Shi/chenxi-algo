"""Offline disclosure protocol fixtures exercise failure and completeness gates."""
import copy
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import disclosure_sources as d


COMPANY = {"ticker": "300308.SZ", "name": "Example", "market": "A"}
HK_COMPANY = {"ticker": "00992.HK", "name": "Example HK", "market": "HK"}


def cn_row(identifier="12345", *, title="2025年年度报告", day="2026-03-31", code="300308"):
    return {"secCode": code, "announcementId": identifier, "announcementTime": int(datetime.fromisoformat(day).replace(tzinfo=d.TZ).timestamp()*1000),
            "announcementTitle": title, "adjunctUrl": f"finalpage/{day}/{identifier}.PDF", "adjunctType": "PDF", "announcementType": "010301"}


def cn_page(rows, total=None, more=False):
    return {"totalAnnouncement": len(rows) if total is None else total, "totalRecordNum": len(rows) if total is None else total,
            "hasMore": more, "announcements": rows, "totalpages": 0}


def hk_row(identifier="67890", *, day="01/01/2026", code="00992<br/>80992", total=1):
    return {"STOCK_CODE": code, "NEWS_ID": identifier, "DATE_TIME": day + " 17:04", "TITLE": "Annual Report 2025",
            "FILE_TYPE": "PDF", "FILE_LINK": "/listedco/listconews/sehk/2026/0101/" + identifier + ".pdf",
            "LONG_TEXT": "Financial Statements/ESG Information - [Annual Report]", "TOTAL_COUNT": str(total)}


def hk_page(rows, total=None, more=False, size=100):
    return {"result": json.dumps(rows), "recordCnt": len(rows) if total is None else total,
            "loadedRecord": len(rows), "hasNextRow": more, "rowRange": size}


class FakeClient:
    def __init__(self, pages):
        self.pages, self.calls = list(pages), []
    def get(self, url, form=None):
        self.calls.append((url, form))
        return self.pages.pop(0)


def request(companies=None, *, start="2026-01-01", end="2026-10-10", max_pages=20, size=100):
    return d._request_definition(companies or [COMPANY], start, end, ["all"], max_pages, size, 10000)


class DisclosureProtocolTests(unittest.TestCase):
    def test_cninfo_does_not_invent_midnight_release_time(self):
        row = d._record("cninfo", COMPANY["ticker"], cn_row(), "2026-01-01", "2026-10-10")
        self.assertEqual(row["available_at"], "2026-03-31")
        self.assertEqual(row["date_precision"], "day")
        self.assertIsNone(row["released_at"])
        self.assertEqual(row["source_type"], "annual_report")
        summary = d._record("cninfo", COMPANY["ticker"], cn_row(title="2025年年度报告摘要"), "2026-01-01", "2026-10-10")
        self.assertEqual(summary["source_type"], "announcement")

    def test_hk_dual_counter_identity_and_release_precision(self):
        record = d._record("hkex", "00992.HK", hk_row(), "2026-01-01", "2026-10-10")
        self.assertEqual(record["released_at"], "2026-01-01T17:04+08:00")
        self.assertEqual(record["source_type"], "annual_report")
        with self.assertRaisesRegex(d.SourceError, "different security"):
            d._record("hkex", "00700.HK", hk_row(), "2026-01-01", "2026-10-10")

    def test_hk_annual_report_combined_with_esg_tag_is_retained(self):
        row = hk_row()
        row["LONG_TEXT"] = "Financial Statements&#x2f;ESG Information - [Annual Report &#x2f; Environmental, Social and Governance Information&#x2f;Report]"
        record = d._record("hkex", "00992.HK", row, "2026-01-01", "2026-10-10")
        self.assertEqual(record["source_type"], "annual_report")
        row["LONG_TEXT"] = "Announcements and Notices - [Notice of Publication of Annual Report]"
        record = d._record("hkex", "00992.HK", row, "2026-01-01", "2026-10-10")
        self.assertEqual(record["source_type"], "announcement")

    def test_identity_date_and_unsafe_pdf_paths_are_rejected(self):
        for mutated in [cn_row(code="300309"), cn_row(day="2025-12-31"), {**cn_row(), "announcementTime": True},
                        {**cn_row(), "adjunctUrl": "https://evil.example/file.pdf"},
                        {**cn_row(), "adjunctUrl": "finalpage/2026-03-31/../../secret.pdf"},
                        {**cn_row(), "adjunctUrl": "finalpage/2026-03-31/x.html"}]:
            with self.assertRaises(d.SourceError):
                d._record("cninfo", COMPANY["ticker"], mutated, "2026-01-01", "2026-10-10")

    def test_cninfo_paginates_and_ignores_unreliable_totalpages_field(self):
        pages = [cn_page([cn_row(str(i)) for i in range(30)], total=31, more=True), cn_page([cn_row("30")], total=31)]
        client = FakeClient(pages)
        rows, audit = d._cninfo(client, COMPANY["ticker"], "990001", request())
        self.assertEqual(len(rows), 31)
        self.assertEqual(audit["pages"], 2)
        self.assertEqual(client.calls[1][1]["pageNum"], "2")

    def test_cninfo_count_drift_duplicate_and_wrong_more_fail(self):
        first = cn_page([cn_row(str(i)) for i in range(30)], total=31, more=True)
        for second in [cn_page([cn_row("31")], total=32), cn_page([cn_row("0")], total=31), cn_page([cn_row("30")], total=31, more=True)]:
            with self.assertRaises(d.SourceError):
                d._cninfo(FakeClient([first, second]), COMPANY["ticker"], "id", request())

    def test_cninfo_budget_failure_and_valid_zero_are_distinct(self):
        first = cn_page([cn_row(str(i)) for i in range(30)], total=31, more=True)
        with self.assertRaisesRegex(d.SourceError, "budget exceeded"):
            d._cninfo(FakeClient([first]), COMPANY["ticker"], "id", request(max_pages=1))
        zero = cn_page([], total=0)
        zero["announcements"] = None
        rows, audit = d._cninfo(FakeClient([zero]), COMPANY["ticker"], "id", request())
        self.assertEqual(rows, [])
        self.assertEqual(audit["provider_total"], 0)

    def test_hkex_splits_capped_date_windows_and_reconciles_leaves(self):
        broad = hk_page([hk_row("1", total=3), hk_row("2", day="02/01/2026", total=3)], total=3, more=True, size=2)
        first = hk_page([hk_row("1", total=1)], size=2)
        second = hk_page([hk_row("2", day="02/01/2026", total=2), hk_row("3", day="02/01/2026", total=2)], size=2)
        client = FakeClient([broad, first, second])
        rows, audit = d._hkex(client, HK_COMPANY["ticker"], "2325", request([HK_COMPANY], end="2026-01-02", size=2))
        self.assertEqual(len(rows), 3)
        self.assertEqual(audit, {"pages": 3, "date_window_splits": 1, "provider_total": 3})
        self.assertIn("fromDate=20260102", client.calls[2][0])

    def test_hkex_same_day_truncation_is_not_success(self):
        capped = hk_page([hk_row(total=2)], total=2, more=True, size=1)
        with self.assertRaisesRegex(d.SourceError, "same-day"):
            d._hkex(FakeClient([capped]), HK_COMPANY["ticker"], "2325", request([HK_COMPANY], end="2026-01-01", size=1))

    def test_hkex_total_disagreement_or_wrong_rowrange_fails(self):
        for obj in [hk_page([hk_row(total=2)], total=1), hk_page([hk_row()], size=20),
                    {**hk_page([hk_row()]), "loadedRecord": 2}]:
            with self.assertRaises(d.SourceError):
                d._hkex(FakeClient([obj]), HK_COMPANY["ticker"], "2325", request([HK_COMPANY]))

    def test_provider_lookup_requires_unique_exact_identifiers(self):
        valid = {"stockList": [{"code": "300308", "orgId": "9900022016", "category": "A股"}]}
        self.assertEqual(d._stock_ids("cninfo", valid)["300308"], "9900022016")
        duplicate = copy.deepcopy(valid)
        duplicate["stockList"] *= 2
        with self.assertRaises(d.SourceError):
            d._stock_ids("cninfo", duplicate)
        self.assertEqual(d._stock_ids("hkex", [{"i": 2325, "c": "00992", "s": 999999}])["00992"], "2325")

    def test_request_rejects_bad_dates_ticker_market_duplicates_and_budgets(self):
        variants = [([COMPANY, COMPANY], {}), ([{**COMPANY, "market": "HK"}], {}),
                    ([{**COMPANY, "ticker": "920001.BJ"}], {}), ([COMPANY], {"end": "2025-01-01"}),
                    ([COMPANY], {"max_pages": True}), ([COMPANY], {"size": 1001})]
        for companies, kwargs in variants:
            with self.assertRaises(ValueError):
                request(companies, **kwargs)


class DisclosureSnapshotTests(unittest.TestCase):
    @staticmethod
    def transport(url, form, **kwargs):
        if url == d.CN_STOCKS:
            payload = {"stockList": [{"code": "300308", "orgId": "9900022016", "category": "A股"}]}
        elif url == d.CN_SEARCH:
            payload = cn_page([cn_row(), cn_row("12346", title="2025年年度报告摘要")])
        else:
            raise AssertionError("unexpected request " + url)
        return json.dumps(payload, ensure_ascii=False).encode(), {"attempts": 1}

    def test_network_capture_and_exact_offline_replay(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(d, "_transport", side_effect=self.transport), patch.object(d.time, "sleep"):
            root = Path(tmp)/"snapshot"
            original = d.discover_disclosures([COMPANY], "2026-01-01", "2026-10-10", download=True, snapshot_dir=root)
            with patch.object(d, "_transport", side_effect=AssertionError("offline should not request network")):
                replay = d.replay_disclosures(root)
            self.assertEqual(replay, original)
            self.assertEqual(replay["coverage"]["pdf_links"], 2)
            self.assertEqual(set(replay["ingest_manifest"][0]), {"document_id", "ticker", "title", "source_url", "available_at", "source_type", "date_basis", "format"})

    def test_offline_window_pool_categories_or_budget_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(d, "_transport", side_effect=self.transport), patch.object(d.time, "sleep"):
            root = Path(tmp)/"snapshot"
            d.discover_disclosures([COMPANY], "2026-01-01", "2026-10-10", download=True, snapshot_dir=root)
            variants = [([COMPANY], "2026-01-02", {}), ([{**COMPANY, "name": "Different pool identity"}], "2026-01-01", {}),
                        ([COMPANY], "2026-01-01", {"categories": ["annual_report"]}), ([COMPANY], "2026-01-01", {"max_pages": 21})]
            for companies, start, kwargs in variants:
                with self.assertRaisesRegex(ValueError, "exact requested"):
                    d.discover_disclosures(companies, start, "2026-10-10", snapshot_dir=root, **kwargs)

    def test_response_tamper_is_not_replayed_as_success(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(d, "_transport", side_effect=self.transport), patch.object(d.time, "sleep"):
            root = Path(tmp)/"snapshot"
            d.discover_disclosures([COMPANY], "2026-01-01", "2026-10-10", download=True, snapshot_dir=root)
            (root/"response-00002.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "differs"):
                d.replay_disclosures(root)

    def test_source_failure_snapshot_remains_failed_offline(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(d, "_transport", side_effect=d.SourceError("HTTP 503 unavailable")):
            root = Path(tmp)/"snapshot"
            original = d.discover_disclosures([COMPANY], "2026-01-01", "2026-10-10", download=True, snapshot_dir=root)
            self.assertEqual(original["status"], "partial")
            self.assertEqual(original["records"], [])
            self.assertEqual(original["coverage"]["failed_tickers"], ["300308.SZ"])
            self.assertEqual(d.replay_disclosures(root), original)

    def test_annual_report_category_does_not_include_summary(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(d, "_transport", side_effect=self.transport), patch.object(d.time, "sleep"):
            report = d.discover_disclosures([COMPANY], "2026-01-01", "2026-10-10", download=True, snapshot_dir=Path(tmp)/"snapshot", categories=["annual_report"])
            self.assertEqual(len(report["records"]), 1)
            self.assertEqual(report["audit"][0]["category_filtered"], 1)

    def test_empty_or_existing_snapshot_not_silently_created_or_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(d, "_transport") as transport:
            with self.assertRaises(FileNotFoundError):
                d.discover_disclosures([COMPANY], "2026-01-01", "2026-10-10", snapshot_dir=Path(tmp)/"absent")
            with self.assertRaises(FileExistsError):
                d.discover_disclosures([COMPANY], "2026-01-01", "2026-10-10", download=True, snapshot_dir=tmp)
            transport.assert_not_called()

    def test_nonfinite_timeout_and_invalid_interval_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(d, "_transport") as transport:
            for kwargs in [{"timeout": float("nan")}, {"min_interval_seconds": 0}, {"retries": True}]:
                with self.assertRaises(ValueError):
                    d.discover_disclosures([COMPANY], "2026-01-01", "2026-10-10", download=True, snapshot_dir=Path(tmp)/"new", **kwargs)
            transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
