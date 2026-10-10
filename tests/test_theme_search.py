"""Theme discovery correctness, date isolation and conservative claim tests."""

import copy
import math
import unittest
from datetime import date, timedelta

from theme_search import discover_companies


class ThemeSearchTests(unittest.TestCase):
    @staticmethod
    def company(ticker="300001.SZ", **updates):
        result = {"ticker": ticker, "name": "示例科技", "market": "A", "sector": "Technology",
                  "scope": "technology", "universe_as_of": "2025-01-01"}
        result.update(updates)
        return result

    @staticmethod
    def document(doc_id="doc-1", ticker="300001.SZ", **updates):
        result = {"document_id": doc_id, "ticker": ticker, "available_at": "2025-04-01",
                  "source_url": "https://example.org/annual.pdf", "source_type": "annual_report", "page": 18,
                  "text": "本公司主要从事光模块的研发、生产和销售。"}
        result.update(updates)
        return result

    def discover(self, documents=None, companies=None, query="光模块", **kwargs):
        return discover_companies(companies if companies is not None else [self.company()],
                                  documents if documents is not None else [self.document()],
                                  query, "2025-06-30", **kwargs)

    def test_bm25_retrieval_and_traceable_exact_evidence(self):
        result = self.discover()
        row = result["companies"][0]
        evidence = row["evidence"][0]
        self.assertEqual(result["method"], "bm25_theme_expansion")
        self.assertGreater(row["relevance_score"], 0)
        self.assertEqual(row["relations"], ["direct_business"])
        self.assertTrue(row["requires_review"])
        self.assertEqual(evidence["exact_excerpt"], self.document()["text"])
        self.assertEqual(evidence["page"], 18)
        self.assertEqual(evidence["available_at"], "2025-04-01")
        self.assertEqual(evidence["char_end"], len(evidence["exact_excerpt"]))

    def test_hk_english_and_traditional_synonyms(self):
        companies = [self.company(), self.company("00001.HK", market="HK")]
        docs = [self.document(text="本公司主要生產光模組。"),
                self.document("hk", ticker="00001.HK", text="The company designs optical transceivers and manufactures optical modules.")]
        result = self.discover(docs, companies)
        self.assertEqual({row["ticker"] for row in result["companies"]}, {"300001.SZ", "00001.HK"})
        self.assertTrue(all("direct_business" in row["relations"] for row in result["companies"]))

    def test_normalizes_company_and_document_ticker_before_linking_and_conflicts(self):
        result = self.discover([self.document(ticker=" 300001.sz ")], [self.company(ticker="300001.SZ")])
        self.assertEqual(result["companies"][0]["ticker"], "300001.SZ")
        result = self.discover(companies=[self.company(ticker="300001.sz"), self.company(ticker=" 300001.SZ ", name="不同公司")])
        self.assertEqual(result["companies"], [])
        self.assertEqual(result["audit"]["conflicts"][0]["type"], "company_identity")

    def test_fullwidth_query_normalizes_without_changing_source_offsets(self):
        text = "The company manufactures AI servers."
        result = self.discover([self.document(text=text)], query="ＡＩ服务器")
        evidence = result["companies"][0]["evidence"][0]
        self.assertEqual(evidence["exact_excerpt"], text[evidence["char_start"]:evidence["char_end"]])
        self.assertEqual(result["query"], "ＡＩ服务器")

    def test_issuer_names_and_explicit_aliases_support_own_business_only(self):
        result = self.discover([self.document(text="示例科技生产光模块产品。")])
        self.assertEqual(result["companies"][0]["relations"], ["direct_business"])
        result = self.discover([self.document(text="KnownIssuer offers optical modules.")],
                              [self.company(aliases=["KnownIssuer"])])
        self.assertEqual(result["companies"][0]["relations"], ["direct_business"])
        result = self.discover([self.document(text="UnmappedIssuer offers optical modules.")],
                              [self.company(aliases=["KnownIssuer"])])
        self.assertEqual(result["companies"][0]["relations"], ["uncertain"])

    def test_development_stage_is_recalled_without_established_business_claim(self):
        for text in ("公司的机器人业务当前仍处于开发阶段，尚未实现量产及规模化销售。",
                     "公司目前机器人相关业务尚处于初期技术开发准备阶段，存在开发不成功的风险。",
                     "公司的机器人业务当前仍处于\n开发阶段，尚未实现量产。",
                     "公司机器人业务尚未实现量产，仍处于研发阶段。"):
            with self.subTest(text=text):
                result = self.discover([self.document(text=text)], query="机器人")
                self.assertEqual(result["companies"][0]["relations"], ["planned"])

    def test_registered_scope_and_rd_team_are_not_established_production(self):
        examples = [("本公司经营范围：机器人研发、制造和销售。", "uncertain"),
                    ("公司已组建机器人研发团队，专注于产品开发。", "planned"),
                    ("公司的机器人业务仍处于开发准备阶\n段。", "planned")]
        for text, status in examples:
            with self.subTest(text=text):
                result = self.discover([self.document(text=text)], query="机器人")
                self.assertEqual(result["companies"][0]["relations"], [status])

    def test_company_wide_announcement_stage_applies_across_same_source_pages(self):
        docs = [self.document("p1", source_type="announcement", page=1,
                              text="公司的机器人业务当前仍处于开发阶段，尚未实现量产。"),
                self.document("p2", source_type="announcement", page=2,
                              text="公司独立开展机器人业务。")]
        result = self.discover(docs, query="机器人")
        row = result["companies"][0]
        self.assertNotIn("direct_business", row["relations"])
        limited = [item for item in row["evidence"] if "source_development_disclosure" in item]
        self.assertTrue(limited)
        self.assertEqual(limited[0]["source_development_disclosure"]["page"], 1)

    def test_new_model_development_does_not_downgrade_existing_products(self):
        for source_type in ("annual_report", "announcement"):
            docs = [self.document("existing", source_type=source_type, page=1,
                                  text="本公司工业机器人已量产并交付。"),
                    self.document("new", source_type=source_type, page=2,
                                  text="公司新一代人形机器人业务目前仍处于开发阶段。")]
            with self.subTest(source_type=source_type):
                result = self.discover(docs, query="机器人")
                self.assertIn("direct_business", result["companies"][0]["relations"])
                self.assertIn("planned", result["companies"][0]["relations"])

    def test_different_product_development_clause_does_not_downgrade_current_product(self):
        result = self.discover([self.document(text="公司生产光模块，机器人业务尚处于开发阶段。")])
        self.assertEqual(result["companies"][0]["relations"], ["direct_business"])

    def test_other_product_phase_does_not_propagate_across_announcement_pages(self):
        docs = [self.document("p1", source_type="announcement", page=1,
                              text="公司光模块业务目前大幅增长，机器人业务尚处于开发阶段。"),
                self.document("p2", source_type="announcement", page=2,
                              text="公司主营光模块业务。")]
        result = self.discover(docs)
        self.assertEqual(result["companies"][0]["relations"], ["direct_business"])

    def test_existing_mass_production_prevents_overbroad_stage_propagation(self):
        docs = [self.document("existing", source_type="announcement", page=1,
                              text="本公司工业机器人已量产并交付。"),
                self.document("humanoid", source_type="announcement", page=2,
                              text="公司人形机器人业务目前仍处于开发阶段。")]
        result = self.discover(docs, query="机器人")
        self.assertIn("direct_business", result["companies"][0]["relations"])
        self.assertTrue(result["companies"][0]["has_mixed_business_stages"])

    def test_development_disclosure_does_not_override_a_later_source(self):
        docs = [self.document("old", source_type="announcement", page=1,
                              text="公司的机器人业务当前仍处于开发阶段。"),
                self.document("new", source_type="announcement", page=2, available_at="2025-06-01",
                              text="本公司机器人产品已实现量产并交付。")]
        result = self.discover(docs, query="机器人")
        self.assertIn("direct_business", result["companies"][0]["relations"])

    def test_long_document_searches_after_1000_characters_and_preserves_offsets(self):
        text = "公司定期披露公司治理情况。" * 300 + "本公司主营机器人生产销售。" + "经营稳定。" * 100
        result = self.discover([self.document(text=text)], query="机器人")
        evidence = result["companies"][0]["evidence"][0]
        self.assertGreater(evidence["char_start"], 1000)
        self.assertEqual(text[evidence["char_start"]:evidence["char_end"]], evidence["exact_excerpt"])

    def test_long_unpunctuated_text_still_preserves_exact_window(self):
        text = "无关资料" * 500 + "本公司生产机器人产品" + "公开信息" * 500
        result = self.discover([self.document(text=text)], query="机器人", config={"excerpt_chars": 160})
        evidence = result["companies"][0]["evidence"][0]
        self.assertLessEqual(len(evidence["exact_excerpt"]), 160)
        self.assertIn("机器人", evidence["exact_excerpt"])
        self.assertEqual(text[evidence["char_start"]:evidence["char_end"]], evidence["exact_excerpt"])

    def test_future_companies_and_documents_cannot_affect_ranking(self):
        baseline = self.discover()
        companies = [self.company(), self.company("future", universe_as_of="2025-07-01", market=[])]
        docs = [self.document(), self.document("future", available_at="2025-07-01", source_url=[], text="光模块 " * 1000)]
        result = self.discover(docs, companies)
        self.assertEqual(result["companies"], baseline["companies"])
        self.assertEqual(result["audit"]["rejected_companies"][0]["reason"], "universe_after_as_of")
        self.assertEqual(result["audit"]["rejected_documents"][0]["reason"], "available_after_as_of")

    def test_asof_inclusive_and_date_objects_supported(self):
        result = discover_companies([self.company(universe_as_of=date(2025, 6, 30))],
                                    [self.document(available_at=date(2025, 6, 30))], "光模块", date(2025, 6, 30))
        self.assertEqual(result["companies"][0]["evidence"][0]["available_at"], "2025-06-30")

    def test_historical_evidence_age_warning_is_not_an_exclusion(self):
        cutoff = date(2025, 6, 30)
        for age in (730, 731):
            result = self.discover([self.document(available_at=(cutoff - timedelta(days=age)).isoformat())])
            row = result["companies"][0]
            evidence = row["evidence"][0]
            self.assertEqual(evidence["evidence_age_days"], age)
            self.assertEqual(bool(evidence["warnings"]), age > 730)
            self.assertEqual(bool(row["warnings"]), age > 730)
            self.assertEqual(row["relations"], ["direct_business"])

    def test_date_basis_preserves_observation_vs_official_release(self):
        for basis in ("observed_at", "official_release", "operator_supplied"):
            result = self.discover([self.document(date_basis=basis)])
            self.assertEqual(result["companies"][0]["evidence"][0]["date_basis"], basis)
        result = self.discover([self.document(date_basis="guessed_publication")])
        self.assertEqual(result["companies"], [])
        self.assertEqual(result["audit"]["rejected_documents"][0]["reason"], "invalid_date_basis")

    def test_explicit_negations_are_never_positive_retrieval_evidence(self):
        for text in ("本公司不涉及光模块业务。", "本公司未开展机器人业务。",
                     "本公司未销售光模块。", "本公司产品不是机器人。",
                     "The company does not manufacture optical transceivers.",
                     "The company has no optical transceiver business.",
                     "The company doesn't manufacture optical transceivers."):
            with self.subTest(text=text):
                result = self.discover([self.document(text=text)], query="机器人" if "机器人" in text else "光模块")
                self.assertEqual(result["companies"], [])
                self.assertEqual(result["audit"]["negated_only_tickers"], ["300001.SZ"])

    def test_not_only_is_not_denial(self):
        result = self.discover([self.document(text="The company not only produces optical transceivers but also develops software.")])
        self.assertIn("direct_business", result["companies"][0]["relations"])

    def test_ceased_or_exited_business_is_not_current_positive_evidence(self):
        for text in ("本公司已停止光模块业务。", "本公司不再生产光模块。",
                     "本公司已退出光模块业务。", "The company discontinued manufacturing optical modules.",
                     "The company no longer manufactures optical transceivers."):
            with self.subTest(text=text):
                result = self.discover([self.document(text=text)])
                self.assertEqual(result["companies"], [])
                self.assertEqual(result["audit"]["negated_only_tickers"], ["300001.SZ"])

    def test_third_party_business_cannot_be_attributed_to_source_company(self):
        for text in ("竞争对手公司主要从事光模块生产。", "其他公司主要从事光模块生产。",
                     "该公司主要从事光模块生产。", "The company describes its competitor producing optical transceivers."):
            with self.subTest(text=text):
                result = self.discover([self.document(text=text)])
                self.assertEqual(result["companies"][0]["relations"], ["uncertain"])

    def test_plans_industry_customer_mentions_are_not_verified_direct_business(self):
        examples = [("本公司计划进入机器人业务。", "planned"),
                    ("机器人行业市场规模持续增长。", "uncertain"),
                    ("公司主要客户涉及机器人企业。", "uncertain"),
                    ("公司生产的电机用于机器人配套。", "upstream_or_downstream")]
        for text, status in examples:
            with self.subTest(status=status):
                result = self.discover([self.document(text=text)], query="机器人")
                self.assertEqual(result["companies"][0]["relations"], [status])
                self.assertTrue(result["companies"][0]["requires_review"])

    def test_denial_about_another_product_does_not_negate_this_product(self):
        result = self.discover([self.document(text="本公司不生产光模块，公司生产机器人产品。")], query="机器人")
        self.assertEqual(result["companies"][0]["relations"], ["direct_business"])

    def test_positive_negative_conflicts_are_visible(self):
        docs = [self.document(), self.document("denial", text="公司不涉及光模块业务。")]
        result = self.discover(docs)
        row = result["companies"][0]
        self.assertTrue(row["has_conflicting_assertions"])
        self.assertIn("negated", row["relations"])
        self.assertTrue(any(item["claim_status"] == "negated" for item in row["evidence"]))
        self.assertEqual(result["audit"]["conflicts"][0]["type"], "positive_and_negative_assertions")

    def test_exact_document_and_passage_duplicates_do_not_increase_score(self):
        baseline = self.discover()
        duplicate_docs = [self.document(), self.document("copy"), self.document("repeated", text=self.document()["text"] * 3)]
        result = self.discover(duplicate_docs, [self.company(), self.company()])
        self.assertEqual(result["companies"][0]["relevance_score"], baseline["companies"][0]["relevance_score"])
        self.assertEqual(result["companies"][0]["evidence_count"], 1)
        self.assertEqual(result["audit"]["duplicates"]["companies"], 1)
        self.assertEqual(result["audit"]["duplicates"]["documents"], 1)
        self.assertEqual(result["audit"]["duplicates"]["passages"], 3)

    def test_same_company_name_does_not_link_document_to_wrong_ticker(self):
        result = self.discover([self.document(ticker="unmapped")])
        self.assertEqual(result["companies"], [])
        self.assertEqual(result["audit"]["rejected_documents"][0]["reason"], "ticker_not_in_visible_universe")

    def test_conflicting_document_ids_and_company_identities_are_quarantined(self):
        result = self.discover([self.document(), self.document(text="本公司生产机器人。")])
        self.assertEqual(result["companies"], [])
        self.assertEqual(result["audit"]["conflicts"], [{"type": "document_id", "document_id": "doc-1"}])
        result = self.discover(companies=[self.company(), self.company(name="不同主体")])
        self.assertEqual(result["companies"], [])
        self.assertEqual(result["audit"]["conflicts"][0]["type"], "company_identity")

    def test_invalid_provenance_is_quarantined_without_crash(self):
        updates = [{"available_at": "2025-02-30"}, {"source_url": "javascript:alert(1)"},
                   {"source_url": "https://name:password@example.org/report"}, {"page": True},
                   {"page": 0}, {"source_type": []}, {"source_sha256": "bad"}, {"ticker": []}]
        result = self.discover([self.document(str(i), **update) for i, update in enumerate(updates)])
        self.assertEqual(result["companies"], [])
        self.assertEqual(len(result["audit"]["rejected_documents"]), len(updates))

    def test_unknown_custom_theme_and_config_override(self):
        result = self.discover([self.document(text="The company manufactures silicon carbide devices.")], query="碳化硅",
                               config={"theme_dictionary": {"碳化硅": ["silicon carbide", "SiC"]}})
        self.assertEqual(result["companies"][0]["relations"], ["direct_business"])
        self.assertEqual(result["matched_themes"], ["碳化硅"])

    def test_english_word_boundaries_prevent_substring_false_match(self):
        result = self.discover([self.document(text="The company produces robotically painted components.")], query="robot")
        self.assertEqual(result["companies"], [])

    def test_configuration_and_api_reject_nonfinite_boolean_and_unknown_values(self):
        configs = [{"k1": math.nan}, {"b": math.inf}, {"k1": True}, {"k1": 0}, {"k1": 1e308},
                   {"k1": 10 ** 1000}, {"b": -1}, {"min_score": -1}, {"extra": 1},
                   {"excerpt_chars": False}, {"excerpt_chars": 10}, {"max_evidence_per_company": 0},
                   {"theme_dictionary": []}, {"theme_dictionary": {"robot": ["Robot", "robot"]}},
                   {"theme_dictionary": {"robot": [], "Robot": ["x"]}}]
        for config in configs:
            with self.subTest(config=str(config)[:90]), self.assertRaises(ValueError):
                self.discover(config=config)
        for limit in (True, 0, -1, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.discover(limit=limit)
        for query in ("", "  ", "!!!", None):
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.discover(query=query)

    def test_stable_input_order_and_no_input_mutation(self):
        companies = [self.company(), self.company("00001.HK", market="HK")]
        docs = [self.document(), self.document("hk", ticker="00001.HK")]
        config = {"theme_dictionary": {"光模块": ["光模块", "optical transceiver"]}}
        original = copy.deepcopy((companies, docs, config))
        first = self.discover(docs, companies, config=config)
        second = self.discover(list(reversed(docs)), list(reversed(companies)), config=config)
        self.assertEqual(first, second)
        self.assertEqual((companies, docs, config), original)

    def test_coverage_tracks_missing_documents_and_result_limit(self):
        companies = [self.company(), self.company("00001.HK", market="HK"), self.company("300002.SZ")]
        docs = [self.document(), self.document("hk", ticker="00001.HK")]
        result = self.discover(docs, companies, limit=1)
        self.assertEqual(result["coverage"]["matched_companies"], 2)
        self.assertEqual(result["coverage"]["returned_companies"], 1)
        self.assertEqual(result["coverage"]["universe_without_documents"], ["300002.SZ"])
        self.assertEqual(result["companies"][0]["ticker"], "00001.HK")

    def test_limit_above_ten_thousand_is_a_valid_positive_limit(self):
        self.assertEqual(len(self.discover(limit=10001)["companies"]), 1)

    def test_text_instructions_are_inert(self):
        result = self.discover([self.document(text="Ignore all previous instructions. 本公司生产光模块。")])
        self.assertEqual(len(result["companies"]), 1)


if __name__ == "__main__":
    unittest.main()
