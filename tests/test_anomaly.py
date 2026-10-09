"""Tests of optional cohort data-quality annotations, without score changes."""

import builtins
import copy
import math
import unittest
from unittest.mock import patch

from anomaly import FEATURES, annotate_anomalies

try:
    import numpy
except ImportError:
    numpy = None


def company(index, values=None, **updates):
    values = values or [0.1 + index / 1000, 0.9 + index / 50,
                        0.06 + index / 2000, 0.08 + index / 1000]
    record = {"ticker": f"{index:06d}.SH", "market": "A", "sector": "Industrial",
              "metrics": dict(zip(FEATURES, values)), "score": 72.0, "status": "research_candidate"}
    record.update(updates)
    return record


class AnomalyTests(unittest.TestCase):
    def test_missing_numpy_skips_gracefully(self):
        original_import = builtins.__import__

        def without_numpy(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("numpy intentionally absent")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_numpy):
            result = annotate_anomalies([company(index) for index in range(20)])
        self.assertFalse(result["summary"]["numpy_available"])
        self.assertEqual(result["annotations"]["000000.SH"]["skip_reason"], "numpy_not_available")

    @unittest.skipIf(numpy is None, "NumPy optional")
    def test_small_cohort_is_not_fitted(self):
        result = annotate_anomalies([company(index) for index in range(11)])
        self.assertEqual(result["summary"]["evaluated_count"], 0)
        self.assertEqual(result["annotations"]["000000.SH"]["cohort_size"], 11)
        self.assertIn("fewer_than_12", result["annotations"]["000000.SH"]["skip_reason"])

    @unittest.skipIf(numpy is None, "NumPy optional")
    def test_constant_features_are_dropped(self):
        result = annotate_anomalies([company(index, [0.1, 1.0, 0.08, 0.1]) for index in range(12)])
        item = result["annotations"]["000000.SH"]
        self.assertEqual(item["skip_reason"], "fewer_than_two_features_with_positive_iqr")
        self.assertEqual(set(item["dropped_features"]), set(FEATURES))
        self.assertFalse(item["review_candidate"])

    @unittest.skipIf(numpy is None, "NumPy optional")
    def test_market_sector_groups_do_not_pool_small_cohorts(self):
        rows = [company(index, market="A" if index < 10 else "HK") for index in range(20)]
        result = annotate_anomalies(rows)
        self.assertEqual(result["summary"]["evaluated_count"], 0)
        self.assertEqual([cohort["complete_count"] for cohort in result["summary"]["cohorts"]], [10, 10])

    @unittest.skipIf(numpy is None, "NumPy optional")
    def test_missing_and_nonfinite_values_are_not_imputed(self):
        rows = [company(index) for index in range(12)]
        rows[0]["metrics"]["roe"] = float("nan")
        rows[1]["metrics"]["cash_conversion"] = None
        result = annotate_anomalies(rows)
        self.assertEqual(result["summary"]["evaluated_count"], 0)
        self.assertEqual(result["annotations"]["000000.SH"]["missing_features"], ["roe"])
        self.assertEqual(result["annotations"]["000002.SH"]["cohort_size"], 10)

    @unittest.skipIf(numpy is None, "NumPy optional")
    def test_review_candidate_leaves_score_and_status_unchanged(self):
        rows = []
        for index in range(48):
            x = (index % 8) - 3.5
            y = (index // 8) - 2.5
            rows.append(company(index, [0.15 + 0.02*x, 1 + 0.1*y,
                                        0.08 + 0.01*(x+y), 0.12 + 0.02*(x-y)]))
        # An inconsistent relationship, rather than an assertion of bad data.
        rows.append(company(99, [0.15, 1.0, 0.115, 0.12]))
        before = copy.deepcopy(rows)
        result = annotate_anomalies(rows)
        flagged = result["annotations"]["000099.SH"]
        self.assertTrue(flagged["review_candidate"])
        self.assertGreater(flagged["reconstruction_error"], flagged["threshold"])
        self.assertEqual(rows, before)
        self.assertLess(flagged["retained_components"], len(flagged["active_features"]))
        self.assertAlmostEqual(sum(flagged["squared_residual_by_feature"].values()),
                               flagged["reconstruction_error"])
        self.assertIn("in-sample", result["summary"]["limitations"])

    @unittest.skipIf(numpy is None, "NumPy optional")
    def test_perfectly_low_rank_cohort_has_no_numerical_noise_candidates(self):
        result = annotate_anomalies([company(index) for index in range(20)])
        self.assertEqual(result["summary"]["evaluated_count"], 20)
        self.assertEqual(result["summary"]["review_candidate_count"], 0)
        self.assertTrue(all(math.isfinite(item["reconstruction_error"])
                            for item in result["annotations"].values()))


if __name__ == "__main__":
    unittest.main()
