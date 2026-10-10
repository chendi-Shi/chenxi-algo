import copy
import json
from pathlib import Path
import tempfile
import unittest

from semantic_retrieval import RetrievalIndex, build_chunks, reciprocal_rank_fusion


def company(ticker='A'):
    return dict(ticker=ticker, name='Issuer '+ticker, market='A', sector='Components',
                scope='technology', universe_as_of='2026-01-01')


def document(text, ticker='A', identity='d1', available='2026-01-01'):
    return dict(document_id=identity, ticker=ticker, text=text, page=1,
                source_url='https://example.com/'+identity, source_type='announcement',
                available_at=available, date_basis='official_release', source_sha256='a'*64)


class FakeModels:
    def __init__(self):
        self.calls = []
        self.version = 1

    def fingerprint(self):
        return {'fake_for_unit_tests': True, 'version': self.version}

    def encode(self, texts, kind='passage'):
        import numpy as np
        self.calls.append((kind, list(texts)))
        return np.array([[1., 0.] if ('optical' in s or '光' in s or kind=='query') else [0., 1.]
                         for s in texts], dtype=np.float32)

    def rerank(self, query, texts):
        return [2. if 'optical' in s or '光' in s else -1. for s in texts]


class FixedChunkTests(unittest.TestCase):
    def test_exact_offsets_cover_text_without_query_dependency(self):
        text='光\n模块。'+('This is exact source text. '*30)
        built=build_chunks([company()], [document(text)], '2026-10-10', chunk_chars=80, overlap_chars=20)
        covered=set()
        for c in built['chunks']:
            self.assertEqual(c['exact_excerpt'], text[c['char_start']:c['char_end']])
            covered.update(range(c['char_start'],c['char_end']))
        self.assertEqual(covered,set(range(len(text))))
        self.assertEqual(built,build_chunks([company()],[document(text)],'2026-10-10',chunk_chars=80,overlap_chars=20))

    def test_future_identity_document_and_conflict_excluded_before_index(self):
        bad=company('B');bad['universe_as_of']='2099-01-01'
        docs=[document('visible'),document('future',identity='f',available='2099-01-01'),
              document('hidden','B','b'),document('conflict',identity='x'),document('different',identity='x')]
        built=build_chunks([company(),bad],docs,'2026-10-10')
        self.assertEqual([c['exact_excerpt'] for c in built['chunks']],['visible'])
        self.assertTrue(built['audit']['conflicts'])

    def test_input_order_and_exact_duplicate_do_not_change_corpus(self):
        docs=[document('a'),document('b','B','d2')]
        a=build_chunks([company(),company('B')],docs,'2026-10-10')
        b=build_chunks([company('B'),company()],list(reversed(docs))+[copy.deepcopy(docs[0])],'2026-10-10')
        self.assertEqual(a['corpus_sha256'],b['corpus_sha256'])

    def test_chunking_and_dates_change_fingerprint(self):
        docs=[document('x'*200)]
        a=build_chunks([company()],docs,'2026-10-10')
        b=build_chunks([company()],docs,'2026-10-09')
        c=build_chunks([company()],docs,'2026-10-10',chunk_chars=80,overlap_chars=20)
        self.assertEqual(len({v['corpus_sha256'] for v in [a,b,c]}),3)

    def test_invalid_windows_rejected(self):
        for chars,overlap in [(0,0),(79,0),(80,80),(80,-1),(True,0)]:
            with self.subTest(chars=chars,overlap=overlap),self.assertRaises(ValueError):
                build_chunks([],[],'2026-10-10',chunk_chars=chars,overlap_chars=overlap)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.companies=[company(),company('B'),company('C')]
        self.docs=[document('The Company manufactures optical transceivers.'),
                   document('The Company manufactures mechanical robots.','B','b'),
                   document('The Company manufactures optical modules.','C','c')]

    def test_rrf_oracle_and_duplicate_guard(self):
        scores=reciprocal_rank_fusion([[1,2],[2,3]],k=10)
        self.assertAlmostEqual(scores[2],1/12+1/11)
        self.assertAlmostEqual(scores[1],1/11)
        self.assertAlmostEqual(scores[3],1/12)
        with self.assertRaises(ValueError):reciprocal_rank_fusion([[1,1]])

    def test_lexical_path_needs_no_model(self):
        index=RetrievalIndex(self.companies,self.docs,'2026-10-10')
        report=index.search('optical transceiver',method='fixed_bm25')
        self.assertEqual({hit['ticker'] for hit in report['hits']},{'A','C'})
        self.assertIsNone(index.vectors)

    def test_dense_requires_explicit_model(self):
        index=RetrievalIndex(self.companies,self.docs,'2026-10-10')
        with self.assertRaises(ValueError):index.search('optical',method='dense')

    def test_dense_cosine_and_reranker_raw_scores(self):
        index=RetrievalIndex(self.companies,self.docs,'2026-10-10',model=FakeModels())
        dense=index.search('photonic communications',method='dense')
        self.assertEqual([x['dense_cosine'] for x in dense['hits']],[1.,1.,0.])
        ranked=index.search('photonic communications',method='hybrid_rerank')
        self.assertEqual([x['reranker_logit'] for x in ranked['hits']],[2.,2.,-1.])

    def test_qualifiers_are_enforced_before_candidate_budget(self):
        docs=[document('The Company makes 100G optical modules.'),
              document('The Company makes 800G optical modules.','B','b')]
        model=FakeModels()
        index=RetrievalIndex(self.companies,docs,'2026-10-10',model=model)
        for method in ('fixed_bm25','dense','hybrid','hybrid_rerank'):
            out=index.search('800G 光模块',method=method,candidate_k=1,rerank_k=1)
            self.assertEqual([x['ticker'] for x in out['hits']],['B'])

    def test_semantic_only_is_uncertain_even_with_positive_model_score(self):
        index=RetrievalIndex(self.companies,self.docs,'2026-10-10',model=FakeModels())
        report=index.company_report('devices for sending data with photons',method='dense')
        self.assertTrue(report['companies'])
        for row in report['companies']:
            self.assertTrue(row['semantic_only'])
            self.assertEqual(row['business_status'],'uncertain')
            self.assertEqual(row['business_support_evidence'],[])
            for hit in row['retrieval_evidence']:
                original=next(d for d in self.docs if d['document_id']==hit['document_id'])
                self.assertEqual(hit['exact_excerpt'],original['text'][hit['char_start']:hit['char_end']])

    def test_new_denial_cannot_be_hidden_by_top_k(self):
        docs=[document('The Company manufactures optical transceivers.',available='2025-01-01'),
              document('The Company no longer manufactures optical transceivers.',identity='denial',available='2026-01-01')]
        index=RetrievalIndex([company()],docs,'2026-10-10',model=FakeModels())
        report=index.company_report('optical transceivers',method='dense',candidate_k=1,rerank_k=1)
        self.assertEqual(report['companies'][0]['business_status'],'historical_or_disputed')
        self.assertEqual(report['companies'][0]['counter_evidence'][0]['document_id'],'denial')

    def test_only_denial_keeps_proof_and_never_current_business(self):
        docs=[document('The Company does not manufacture optical transceivers.')]
        index=RetrievalIndex([company()],docs,'2026-10-10',model=FakeModels())
        row=index.company_report('optical transceivers',method='dense')['companies'][0]
        self.assertNotEqual(row['business_status'],'current_business')
        self.assertTrue(row['counter_evidence'])

    def test_broad_withdrawal_blocks_old_model_specific_support(self):
        docs=[document('本公司生产800G光模块产品。',available='2025-04-01'),
              document('本公司已停止生产光模块产品。',identity='exit',available='2026-01-01')]
        index=RetrievalIndex([company()],docs,'2026-10-10',model=FakeModels())
        row=index.company_report('800G 光模块',method='hybrid_rerank',candidate_k=1,rerank_k=1)['companies'][0]
        self.assertEqual(row['business_status'],'historical_or_disputed')
        self.assertEqual(row['counter_evidence'][0]['document_id'],'exit')

    def test_other_model_denial_does_not_erase_requested_model(self):
        docs=[document('本公司生产800G光模块产品。',available='2025-04-01'),
              document('本公司不生产100G光模块产品。',identity='other',available='2026-01-01')]
        index=RetrievalIndex([company()],docs,'2026-10-10',model=FakeModels())
        row=index.company_report('800G 光模块',method='hybrid_rerank')['companies'][0]
        self.assertEqual(row['business_status'],'current_business')

    def test_large_finite_vectors_normalize_without_overflow(self):
        index=RetrievalIndex([],[],'2026-10-10')
        import numpy as np
        vector=index._validate_vectors([[1e30,1e30]],1)
        self.assertAlmostEqual(float(np.linalg.norm(vector)),1.,places=6)

    def test_no_future_text_reaches_encoder(self):
        model=FakeModels()
        docs=self.docs+[document('FUTURE POISON',identity='future',available='2099-01-01')]
        index=RetrievalIndex(self.companies,docs,'2026-10-10',model=model)
        index.search('optical',method='dense')
        self.assertFalse(any('FUTURE POISON' in text for kind,texts in model.calls for text in texts))

    def test_passages_encoded_once_for_multiple_queries(self):
        model=FakeModels();index=RetrievalIndex(self.companies,self.docs,'2026-10-10',model=model)
        index.search('optical',method='dense');index.search('robotics',method='dense')
        self.assertEqual(sum(k=='passage' for k,_ in model.calls),1)
        self.assertEqual(sum(k=='query' for k,_ in model.calls),2)

    def test_cache_reuse_integrity_and_model_version_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=FakeModels();idx=RetrievalIndex(self.companies,self.docs,'2026-10-10',model=first,cache_dir=tmp)
            idx.prepare_dense()
            second=FakeModels();idx2=RetrievalIndex(self.companies,self.docs,'2026-10-10',model=second,cache_dir=tmp)
            idx2.prepare_dense()
            self.assertEqual(second.calls,[])
            self.assertEqual(idx2.cache_status,'verified_cache_hit')
            vector=next(Path(tmp).rglob('vectors.npy'));vector.write_bytes(vector.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'integrity'):
                RetrievalIndex(self.companies,self.docs,'2026-10-10',model=FakeModels(),cache_dir=tmp).prepare_dense()
            third=FakeModels();third.version=2
            RetrievalIndex(self.companies,self.docs,'2026-10-10',model=third,cache_dir=tmp).prepare_dense()
            self.assertTrue(third.calls)

    def test_incomplete_cache_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            index=RetrievalIndex(self.companies,self.docs,'2026-10-10',model=FakeModels(),cache_dir=tmp)
            index.prepare_dense();next(Path(tmp).rglob('complete.json')).unlink()
            with self.assertRaisesRegex(ValueError,'incomplete'):
                RetrievalIndex(self.companies,self.docs,'2026-10-10',model=FakeModels(),cache_dir=tmp).prepare_dense()

    def test_model_mutation_and_invalid_outputs_fail_closed(self):
        model=FakeModels();index=RetrievalIndex(self.companies,self.docs,'2026-10-10',model=model)
        index.prepare_dense();model.version=2
        with self.assertRaisesRegex(ValueError,'changed'):index.search('optical',method='dense')
        import numpy as np
        for value in [np.zeros((3,2)),np.array([[float('nan'),1.]]*3),np.ones((2,2))]:
            broken=FakeModels();broken.encode=lambda *a,**k:value
            with self.subTest(shape=value.shape),self.assertRaises(ValueError):
                RetrievalIndex(self.companies,self.docs,'2026-10-10',model=broken).prepare_dense()

    def test_empty_visible_corpus_no_model_execution(self):
        model=FakeModels();index=RetrievalIndex([],[],'2026-10-10',model=model)
        report=index.company_report('robotics',method='hybrid_rerank')
        self.assertEqual(report['status'],'no_matches');self.assertEqual(model.calls,[])


if __name__=='__main__':
    unittest.main()
