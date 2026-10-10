"""Failure-mode checks for the source-fact audit (no network/PDF dependency)."""
import copy
import json
import unittest

from validation.enterprise_a_share import FACTS, monetary, validate


class EnterpriseAShareSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(FACTS.read_text(encoding="utf8"))

    def test_all_existing_mapped_facts_match_and_unknown_debt_is_preserved(self):
        report = validate(self.document)
        self.assertEqual(report["summary"]["dataset_matches"], 54)
        self.assertEqual(report["summary"]["pdf_facts_checked"], 0)
        self.assertEqual(report["status"], "offline_arithmetic_only")
        self.assertTrue(all(row["total_debt"] is None for row in report["derived_reconciliations"]))
        self.assertTrue(all(not row["candidate_approval"] for row in report["derived_reconciliations"]))

    def test_duplicate_facts_rejected(self):
        data = copy.deepcopy(self.document)
        data["facts"].append(data["facts"][0])
        with self.assertRaises(ValueError):
            validate(data)

    def test_amount_token_disagreement_rejected(self):
        data = copy.deepcopy(self.document)
        data["facts"][0]["value"] = "0"
        with self.assertRaises(ValueError):
            validate(data)

    def test_source_amount_change_is_reported_as_dataset_mismatch(self):
        data = copy.deepcopy(self.document)
        data["facts"][0]["value"] = "1.00"
        data["facts"][0]["source_value_token"] = "1.00"
        report = validate(data)
        self.assertEqual(report["status"], "dataset_mismatch")
        self.assertEqual(report["summary"]["dataset_matches"], 53)

    def test_late_reports_are_not_claimed_available(self):
        report = validate(self.document, as_of="2023-06-30")
        self.assertEqual(report["summary"]["sources_after_cutoff"], 4)
        self.assertFalse(any(row["source_eligible_at_cutoff"] for row in report["source_fact_checks"]))

    def test_restricted_cash_cannot_be_subtracted_twice(self):
        data = copy.deepcopy(self.document)
        fact = next(f for f in data["facts"] if f["ticker"] == "000858.SZ" and f["fiscal_year"] == 2024 and f["field"] == "cash")
        fact["value"] = "124644427415.11"
        fact["source_value_token"] = "124,644,427,415.11"
        with self.assertRaisesRegex(ValueError, "Cash reconciliation mismatch"):
            validate(data)

    def test_nonfinite_money_and_invalid_page_rejected(self):
        for value in ["NaN", "Infinity", "-Infinity", 1.2, None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                monetary(value)
        for page in [0, -1, True, "5"]:
            data = copy.deepcopy(self.document)
            data["facts"][0]["pdf_page"] = page
            with self.subTest(page=page), self.assertRaises(ValueError):
                validate(data)

    def test_restatement_delta_tampering_rejected(self):
        data = copy.deepcopy(self.document)
        data["restatements"][0]["delta"] = "0"
        with self.assertRaisesRegex(ValueError, "Restatement arithmetic mismatch"):
            validate(data)


if __name__ == "__main__":
    unittest.main()
