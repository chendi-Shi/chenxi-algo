"""Integration checks for the review workflow, with no exported test files.

CLI output paths are mocked and SQLite uses an in-memory connection. Fixtures
exercise user-facing audit and source-handling properties, not stock returns.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine
import evidence
import run
from test_engine import AS_OF, annual_history, valuation


def document(**changes):
    row = {
        "document_id": "visible-extract", "ticker": " a1 ",
        "available_at": AS_OF.isoformat(), "source_type": "annual_report",
        "source_url": "https://example.invalid/report", "page": 88,
        "text": "No related party transactions. Cash flow improved. <script>alert(1)</script>",
    }
    row.update(changes)
    return row


def screened_result(ticker="A1"):
    value = engine.screen(annual_history(ticker), [valuation(ticker)], AS_OF)
    value["documents"] = {"evidence": [], "rejected": [], "method": "Fixture text review"}
    return value


class ExportAndStorageTests(unittest.TestCase):
    def test_html_escapes_source_text_and_displays_actual_weights(self):
        results = screened_result()
        results["companies"][0]["name"] = "<img src=x onerror=alert(1)>"
        results["companies"][0]["reasons"].append("<script>evil()</script>")
        results["documents"]["evidence"] = [{
            **document(ticker="A1"), "topics": ["related_party"],
            "excerpt": document()["text"],
        }]
        results["config"]["weights"] = {
            "quality": .1, "value": .2, "growth": .3, "balance": .4,
        }
        rendered = run.report_html(results)
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<img src=x", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("No related party transactions.", rendered)
        self.assertIn("10%", rendered)
        self.assertIn("30%", rendered)
        self.assertIn("未知", rendered, "a missing composite score must remain visibly unknown")

    def test_csv_keeps_missing_values_blank_and_neutralizes_formula_text(self):
        class ReadableAfterClose(io.StringIO):
            def close(self):
                pass

        buffer = ReadableAfterClose()
        destination = mock.Mock()
        destination.open.return_value = buffer
        company = screened_result()["companies"][0]
        company["name"] = "\t=1+1"
        company["reasons"] = ["=HYPERLINK(\"https://example.invalid\")"]
        company["metrics"]["core_earnings_yield"] = None
        company["metrics"]["revenue_cagr_2y"] = -.25
        run.export_csv(destination, [company])
        row = next(csv.DictReader(io.StringIO(buffer.getvalue())))
        self.assertEqual(row["core_earnings_yield"], "")
        self.assertEqual(row["score"], "")
        self.assertEqual(float(row["revenue_cagr_2y"]), -.25,
                         "negative numeric values must remain numeric")
        self.assertTrue(row["name"].startswith("'"))
        self.assertTrue(row["reasons"].startswith("'"))

    def test_sqlite_preserves_sources_and_uses_parameterized_records(self):
        ticker = "A1'; DROP TABLE COMPANIES; --"
        results = screened_result(ticker)
        doc = document(ticker=ticker, document_id="D'; --", excerpt="Research source", topics=[])
        results["documents"]["evidence"] = [doc]
        inputs = {"statements": annual_history(ticker), "valuations": [valuation(ticker)]}
        original = copy.deepcopy((results, inputs))
        connection = sqlite3.connect(":memory:")
        try:
            with mock.patch.object(run.sqlite3, "connect", return_value=connection):
                run.save_database("unused", "fixture-run", {"input_hash": "hash"}, results, inputs)
                run.save_database("unused", "fixture-run", {"input_hash": "hash"}, results, inputs)
            self.assertEqual(connection.execute("SELECT count(*) FROM companies").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT count(*) FROM statements").fetchone()[0], 3)
            self.assertEqual(connection.execute("SELECT count(*) FROM raw_records").fetchone()[0], 4)
            self.assertEqual(connection.execute("SELECT count(*) FROM evidence").fetchone()[0], 1)
            stored = json.loads(connection.execute("SELECT payload_json FROM companies").fetchone()[0])
            self.assertEqual(stored["ticker"], ticker)
            self.assertEqual(stored["sources"], results["companies"][0]["sources"])
            self.assertEqual((results, inputs), original, "persistence must not rewrite research inputs")
        finally:
            connection.close()


class InputIsolationRegressionTests(unittest.TestCase):
    def test_future_malformed_versions_cannot_poison_visible_company(self):
        rows = annual_history("A1")
        future = dict(rows[-1], available_at="2026-05-02", revision_id="bad",
                      period_end="malformed", net_income="NaN")
        value_future = valuation("A1", available_at="2026-05-02", revision_id="bad",
                                 snapshot_date="malformed", market_cap="NaN")
        report = engine.screen(rows + [future], [valuation("A1"), value_future], AS_OF)
        company = report["companies"][0]
        self.assertEqual(company["status"], "watchlist")
        self.assertAlmostEqual(company["metrics"]["roa"], 20 / 110)
        self.assertEqual(report["audit"][0]["reason"], "not_yet_available")
        self.assertEqual(report["valuation_audit"][0]["reason"], "not_yet_available")

    def test_visible_revision_conflicts_are_order_independent_even_when_superseded(self):
        history = annual_history("A1")
        conflict = dict(history[-1], net_income=21)
        latest = dict(history[-1], revision_id="1", available_at="2026-04-01", net_income=22)
        for versions in ((history[-1], conflict, latest), (latest, history[-1], conflict)):
            with self.subTest(versions=[v["revision_id"] for v in versions]):
                report = engine.screen(history[:-1] + list(versions), [valuation("A1")], AS_OF)
                self.assertEqual(report["companies"][0]["status"], "data_review")
                self.assertTrue(any("conflicting" in item["reason"] for item in report["audit"]))
        original = valuation("A1")
        conflict = valuation("A1", market_cap=201)
        latest = valuation("A1", revision_id="1", market_cap=202)
        for versions in ((original, conflict, latest), (latest, original, conflict)):
            with self.subTest(valuations=[v["revision_id"] for v in versions]):
                report = engine.screen(history, list(versions), AS_OF)
                self.assertEqual(report["companies"][0]["status"], "data_review")
                self.assertTrue(any("conflicting" in item["reason"] for item in report["valuation_audit"]))

    def test_bad_company_classification_does_not_interrupt_other_companies(self):
        good = annual_history("GOOD")
        bad = annual_history("BAD")
        bad[-1]["sector"] = None
        report = engine.screen(good + bad, [valuation("GOOD"), valuation("BAD")], AS_OF)
        companies = {c["ticker"]: c for c in report["companies"]}
        self.assertEqual(companies["GOOD"]["status"], "watchlist")
        self.assertEqual(companies["BAD"]["status"], "data_review")
        self.assertEqual(companies["BAD"]["metrics"], {})

    def test_dates_and_provenance_require_the_documented_schema(self):
        for invalid in ("20260501", "2026-W18-5", None):
            with self.subTest(date=invalid), self.assertRaises(ValueError):
                engine.iso(invalid)
        for field, invalid in (("sector", None), ("source_url", "javascript:alert(1)"),
                               ("source_url", "not-a-url"), ("name", 123)):
            row = annual_history()[-1]
            row[field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                engine.parse_statement(row)

    def test_nonstandard_json_constants_are_rejected_without_stopping_valid_documents(self):
        valid = document()
        invalid = dict(valid, document_id="nonfinite-page", page=float("nan"))
        overflow = json.dumps(dict(valid, document_id="overflow-page", page=123)).replace('"page": 123', '"page": 1e999')
        lines = json.dumps(invalid) + "\n" + overflow + "\n" + json.dumps(valid) + "\n"
        with mock.patch.object(Path, "open", return_value=io.StringIO(lines)):
            triage = evidence.process_documents(Path("unused.jsonl"), AS_OF)
        self.assertEqual(len(triage["evidence"]), 1)
        self.assertEqual([item["reason"] for item in triage["rejected"]], ["invalid_json", "invalid_json"])
        self.assertEqual(triage["evidence"][0]["ticker"], "A1")
        run.dump_json(triage)

    def test_import_whitespace_does_not_bypass_specialist_sector_exit(self):
        for sector in ("Banks ", " 银行 "):
            with self.subTest(sector=sector):
                report = engine.screen(annual_history(sector=sector), [valuation()], AS_OF)
                self.assertEqual(report["companies"][0]["status"], "specialist_review")
                self.assertIsNone(report["companies"][0]["score"])


class CLIWorkflowTests(unittest.TestCase):
    def run_in_memory(self, ml_version=None, changed_input=False):
        rows, values = [], []
        for ticker in (" a1 ", "A2", "A3", "A4", "A5"):
            rows.extend(annual_history(ticker))
            values.append(valuation(ticker.strip().upper()))
        documents = [document(), document(document_id="future", available_at="2026-05-02")]
        lines = "\n".join(json.dumps(row) for row in documents)
        written = {}
        args = ["--as-of", AS_OF.isoformat(), "--statements", "statements.csv",
                "--valuations", "valuations.csv", "--documents", "documents.jsonl",
                "--output", "mock-output", "--query", "related party"]
        if ml_version is not None:
            args.append("--ml-cleaning")

        def read_input(path):
            return copy.deepcopy(rows if Path(path).name == "statements.csv" else values)

        def write_output(path, value, **kwargs):
            written[path.name] = value
            return len(value)

        def read_bytes(path):
            suffix = b"-changed" if changed_input and path.name == "statements.csv" else b""
            return path.name.encode("utf-8") + suffix

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(run, "config_from", return_value=engine.config_from(None)))
            stack.enter_context(mock.patch.object(run, "load_csv", side_effect=read_input))
            stack.enter_context(mock.patch.object(Path, "open", side_effect=lambda *a, **k: io.StringIO(lines)))
            stack.enter_context(mock.patch.object(Path, "read_bytes", autospec=True, side_effect=read_bytes))
            stack.enter_context(mock.patch.object(Path, "write_text", autospec=True, side_effect=write_output))
            stack.enter_context(mock.patch.object(Path, "mkdir"))
            csv_export = stack.enter_context(mock.patch.object(run, "export_csv"))
            database = stack.enter_context(mock.patch.object(run, "save_database"))
            stack.enter_context(mock.patch("sys.stdout", new_callable=io.StringIO))
            if ml_version is not None:
                stack.enter_context(mock.patch.object(run.importlib.metadata, "version", return_value=ml_version))
                stack.enter_context(mock.patch("anomaly.annotate_anomalies", return_value={
                    "summary": {"numpy_available": True, "status": "fixture"}, "annotations": {},
                }))
            status = run.main(args)
        self.assertEqual(status, 0)
        self.assertEqual(csv_export.call_count, 1)
        self.assertEqual(database.call_count, 1)
        return json.loads(written["results.json"]), json.loads(written["manifest.json"]), written, database

    def test_workflow_matches_text_to_normalized_ticker_and_excludes_future_evidence(self):
        results, manifest, written, database = self.run_in_memory()
        self.assertEqual(len(results["companies"]), 5)
        self.assertTrue(all(c["score"] == 50 for c in results["companies"]))
        self.assertEqual(results["unmatched_document_tickers"], [])
        self.assertEqual([d["document_id"] for d in results["documents"]["evidence"]], ["visible-extract"])
        self.assertEqual(results["documents"]["rejected"][0]["reason"], "available_after_as_of")
        self.assertEqual(results["document_search"][0]["ticker"], "A1")
        self.assertNotIn("<script>", written["report.html"])
        self.assertIn("&lt;script&gt;", written["report.html"])
        self.assertEqual(database.call_args.args[3]["run_id"], manifest["run_id"])
        self.assertEqual(set(manifest["inputs"]), {"statements", "valuations", "documents"})

    def test_repeatable_inputs_have_same_identity_and_changed_inputs_do_not(self):
        first, manifest, _, _ = self.run_in_memory()
        repeat, repeated_manifest, _, _ = self.run_in_memory()
        changed, _, _, _ = self.run_in_memory(changed_input=True)
        self.assertEqual(first, repeat)
        self.assertEqual(manifest, repeated_manifest)
        self.assertNotEqual(first["run_id"], changed["run_id"])

    def test_dependency_version_is_part_of_ml_run_identity(self):
        first, first_manifest, _, _ = self.run_in_memory(ml_version="1.0-fixture")
        second, second_manifest, _, _ = self.run_in_memory(ml_version="2.0-fixture")
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertNotEqual(first_manifest["environment"], second_manifest["environment"])
        self.assertIn("python_version", first_manifest["environment"])
        self.assertTrue(first_manifest["environment"]["numpy_available"])


if __name__ == "__main__":
    unittest.main()
