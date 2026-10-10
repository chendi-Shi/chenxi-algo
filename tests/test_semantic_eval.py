"""Synthetic metric oracles; no model inference and no label tuning."""

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import unittest

from validation.semantic_eval import evaluate, validate_corpus_alignment


def dataset(grades=None, universe=None):
    grades = {"A": 2, "B": 1, "C": 0} if grades is None else grades
    return {
        "candidate_tickers": universe or ["A", "B", "C"],
        "maximum_grade": 2,
        "positive_grade_threshold": 1,
        "queries": [{"query_id": "q1", "group_id": "intent1", "text": "synthetic oracle",
                     "judgments": [{"ticker": t, "grade": g} for t, g in grades.items()]}],
    }


def ranking(tickers):
    return [{"ticker": ticker, "score": float(len(tickers) - i)} for i, ticker in enumerate(tickers)]


def run(data, tickers, k=(1, 2, 3)):
    return evaluate(data, {"fixed_bm25": {"q1": ranking(tickers)}}, k)["methods"]["fixed_bm25"]


class SemanticMetricTests(unittest.TestCase):
    def test_perfect_rank_has_expected_manual_values(self):
        result = run(dataset(), ["A", "B", "C"])["queries"]["q1"]
        self.assertEqual(result["mrr"]["value"], 1)
        self.assertEqual(result["at_k"]["1"]["recall"]["value"], 0.5)
        self.assertEqual(result["at_k"]["2"]["precision"]["value"], 1)
        self.assertEqual(result["at_k"]["2"]["recall"]["value"], 1)
        self.assertAlmostEqual(result["at_k"]["3"]["precision"]["value"], 2 / 3)
        for values in result["at_k"].values():
            self.assertEqual(values["ndcg"]["value"], 1)
            self.assertEqual(values["judged_coverage"]["value"], 1)

    def test_bad_rank_matches_hand_computed_dcg(self):
        result = run(dataset(), ["C", "B", "A"])["queries"]["q1"]
        self.assertEqual(result["mrr"]["value"], 0.5)
        self.assertEqual(result["at_k"]["2"]["recall"]["value"], 0.5)
        expected = (1 / math.log2(3)) / (3 + 1 / math.log2(3))
        self.assertAlmostEqual(result["at_k"]["2"]["ndcg"]["value"], expected)

    def test_unjudged_results_keep_rank_and_do_not_become_negatives(self):
        result = run(dataset({"A": 2, "B": 0}), ["C", "B", "A"])["queries"]["q1"]
        top = result["at_k"]["1"]
        self.assertEqual(top["precision"], {"value": None, "lower": 0, "upper": 1})
        self.assertEqual(top["recall"], {"value": None, "lower": 0, "upper": 0.5})
        self.assertEqual(result["mrr"], {"value": None, "lower": 1 / 3, "upper": 1})
        self.assertEqual(top["judged_coverage"]["value"], 0)
        self.assertEqual(result["at_k"]["3"]["recall"]["value"], 1)

    def test_unknown_outside_results_makes_full_universe_recall_uncertain(self):
        result = run(dataset({"A": 2, "B": 0}), ["A"])["queries"]["q1"]
        top = result["at_k"]["1"]
        self.assertEqual(top["precision"]["value"], 1)
        self.assertEqual(top["recall"], {"value": None, "lower": 0.5, "upper": 1})
        self.assertEqual(top["judged_positive_recall"]["value"], 1)
        self.assertEqual(top["judged_coverage"]["value"], 1)

    def test_bounds_contain_every_completed_label_assignment(self):
        # Independent oracle enumerates every possible truth for three unknowns.
        universe = ["A", "B", "C", "D", "E"]
        known = {"A": 2, "B": 0}
        ranked = ["C", "B", "A", "D"]
        result = run(dataset(known, universe), ranked, (1, 3, 5))["queries"]["q1"]
        for values in itertools.product((0, 1, 2), repeat=3):
            truth = {**known, **dict(zip(["C", "D", "E"], values))}
            first = next(i for i, t in enumerate(ranked, 1) if truth[t] > 0)
            rr = 1 / first
            self.assertLessEqual(result["mrr"]["lower"], rr + 1e-12)
            self.assertGreaterEqual(result["mrr"]["upper"], rr - 1e-12)
            for k in (1, 3, 5):
                top = ranked[:k]
                positives = sum(truth[t] > 0 for t in top)
                dcg = sum((2 ** truth[t] - 1) / math.log2(i + 1) for i, t in enumerate(top, 1))
                ideal = sorted(truth.values(), reverse=True)[:k]
                idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
                actual = {"precision": positives / k,
                          "recall": positives / sum(g > 0 for g in truth.values()),
                          "ndcg": dcg / idcg}
                for metric, value in actual.items():
                    interval = result["at_k"][str(k)][metric]
                    self.assertLessEqual(interval["lower"], value + 1e-12)
                    self.assertGreaterEqual(interval["upper"], value - 1e-12)

    def test_intent_group_macro_does_not_overweight_variants(self):
        data = dataset()
        for i in (2, 3, 4):
            query = copy.deepcopy(data["queries"][0])
            query["query_id"] = f"q{i}"
            query["group_id"] = "intent2" if i == 4 else "intent1"
            data["queries"].append(query)
        runs = {"fixed_bm25": {f"q{i}": ranking(["A"] if i < 4 else ["C"]) for i in (1, 2, 3, 4)}}
        result = evaluate(data, runs, (1,))["methods"]["fixed_bm25"]
        self.assertEqual(result["group_macro"]["at_k"]["1"]["precision"]["value"], 0.5)
        self.assertEqual(result["group_macro"]["items"], 2)

    def test_short_and_empty_results_are_not_padded_with_fake_judgments(self):
        result = run(dataset(), ["A"])["queries"]["q1"]["at_k"]["3"]
        self.assertEqual(result["precision"]["value"], 1 / 3)
        self.assertEqual(result["judged_coverage"]["value"], 1)
        self.assertEqual(result["returned_coverage"]["value"], 1 / 3)
        empty = run(dataset(), [])
        self.assertEqual(empty["queries"]["q1"]["mrr"]["value"], 0)
        self.assertIsNone(empty["group_macro"]["at_k"]["1"]["judged_coverage"]["value"])
        self.assertEqual(empty["group_macro"]["at_k"]["1"]["precision"]["value"], 0)

    def test_undefined_group_points_propagate(self):
        data = dataset({"A": 2, "B": 0})
        second = copy.deepcopy(data["queries"][0])
        second.update(query_id="q2", group_id="intent2")
        data["queries"].append(second)
        result = evaluate(data, {"fixed_bm25": {"q1": ranking(["A"]), "q2": ranking(["C"])}}, (1,))
        metric = result["methods"]["fixed_bm25"]["group_macro"]["at_k"]["1"]["precision"]
        self.assertIsNone(metric["value"])
        self.assertEqual(metric["lower"], 0.5)
        self.assertEqual(metric["upper"], 1)
        self.assertEqual(metric["point_defined_items"], 1)

    def test_zero_relevance_and_only_unknown_relevance_are_explicit(self):
        known_zero = run(dataset({"A": 0, "B": 0, "C": 0}), ["A", "B", "C"])["queries"]["q1"]
        self.assertIsNone(known_zero["mrr"]["value"])
        self.assertIsNone(known_zero["at_k"]["1"]["recall"]["lower"])
        self.assertIsNone(known_zero["at_k"]["1"]["ndcg"]["value"])
        uncertain = run(dataset({"A": 0, "B": 0}), ["C"])["queries"]["q1"]
        self.assertTrue(uncertain["bounds_conditional_on_any_relevant"])
        self.assertEqual(uncertain["at_k"]["1"]["recall"]["value"], 1)
        self.assertIsNone(uncertain["at_k"]["1"]["precision"]["value"])

    def test_ranking_order_is_preserved_including_score_ties(self):
        ranks = [{"ticker": "C", "score": 1}, {"ticker": "A", "score": 1}]
        result = evaluate(dataset(), {"dense": {"q1": ranks}}, (1,))
        self.assertEqual(result["methods"]["dense"]["queries"]["q1"]["mrr"]["value"], 0.5)
        self.assertIn("legacy_bm25", result["missing_methods"])

    def test_duplicate_ranking_ticker_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate ranking ticker"):
            run(dataset(), ["A", "A"])

    def test_missing_and_unknown_queries_rejected(self):
        for results, message in [({}, "missing query"), ({"q1": [], "not_a_query": []}, "unknown query")]:
            with self.subTest(results=results), self.assertRaisesRegex(ValueError, message):
                evaluate(dataset(), {"fixed_bm25": results})

    def test_unknown_candidate_or_method_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside candidate universe"):
            run(dataset(), ["OTHER"])
        with self.assertRaisesRegex(ValueError, "unknown retrieval method"):
            evaluate(dataset(), {"magic": {"q1": []}})

    def test_bad_scores_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), True, "1", None):
            with self.subTest(score=value), self.assertRaisesRegex(ValueError, "finite numeric"):
                evaluate(dataset(), {"dense": {"q1": [{"ticker": "A", "score": value}]}})

    def test_duplicate_or_invalid_labels_rejected(self):
        data = dataset()
        data["queries"][0]["judgments"].append({"ticker": "A", "grade": 2})
        with self.assertRaisesRegex(ValueError, "duplicate judgment"):
            run(data, [])
        data = dataset({"A": True})
        with self.assertRaisesRegex(ValueError, "invalid grade"):
            run(data, [])

    def test_variant_label_changes_rejected(self):
        data = dataset()
        duplicate = copy.deepcopy(data["queries"][0])
        duplicate["query_id"] = "q2"
        duplicate["judgments"][0]["grade"] = 0
        data["queries"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "inconsistent labels"):
            evaluate(data, {"dense": {"q1": [], "q2": []}})

    def test_bad_k_values_and_empty_runs_rejected(self):
        for k in ((), (0,), (True,), (1, 1), (-1,), (1.5,)):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "k_values"):
                run(dataset(), [], k)
        with self.assertRaisesRegex(ValueError, "nonempty method mapping"):
            evaluate(dataset(), {})

    def test_frozen_development_dataset_has_traceable_labels(self):
        path = Path(__file__).resolve().parents[1] / "validation" / "semantic_dataset.json"
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), "1231a1c29dd61304e52e67057a780e69826348cd23cb1fd5cfc8bdcadadb85b4")
        data = json.loads(raw)
        self.assertEqual(data["annotation_status"], "assistant_annotated")
        self.assertEqual(data["acceptance_status"], "development_not_human_accepted")
        self.assertFalse(data["blind_holdout"])
        self.assertEqual(len(data["queries"]), 30)
        self.assertEqual(len(data["groups"]), 10)
        # Empty synthetic results validate schema/provenance, not model quality.
        results = evaluate(data, {"fixed_bm25": {q["query_id"]: [] for q in data["queries"]}})
        self.assertEqual(results["intent_group_count"], 10)
        self.assertEqual(results["candidate_count"], 9)
        refs = {d["document_id"] for d in data["source_documents"]}
        for query in data["queries"]:
            for judgment in query["judgments"]:
                self.assertTrue(set(judgment["source_document_ids"]) <= refs)
                self.assertTrue(judgment["reason"])

    def test_excerpt_alias_resolves_to_same_hashed_pdf_page(self):
        data = dataset({"A": 2})
        data["source_documents"] = [{"document_id": "short-id", "ticker": "A", "source_sha256": "sha",
                                      "page": 10, "source_url": "https://example.org/report.pdf",
                                      "existing_repository_excerpt": "A supported phrase"}]
        for judgment in data["queries"][0]["judgments"]:
            judgment.update(source_document_ids=["short-id"], reason="synthetic provenance test")
        doc = {"document_id": "full-page-id", "ticker": "A", "source_sha256": "sha", "page": 10,
               "source_url": "https://example.org/report.pdf", "text": "Page text: A supported\nphrase and context."}
        report = validate_corpus_alignment(data, [doc])
        self.assertEqual(report["mappings"][0]["corpus_document_ids"], ["full-page-id"])
        for field, bad_value in (("page", 11), ("source_sha256", "different"),
                                 ("ticker", "B"), ("source_url", "https://elsewhere.org/file")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "absent or changed"):
                validate_corpus_alignment(data, [{**doc, field: bad_value}])
        with self.assertRaisesRegex(ValueError, "excerpt absent"):
            validate_corpus_alignment(data, [{**doc, "text": "Unrelated body."}])
        data["source_documents"][0]["ticker"] = "B"
        with self.assertRaisesRegex(ValueError, "different security"):
            evaluate(data, {"fixed_bm25": {"q1": []}})


if __name__ == "__main__":
    unittest.main()
