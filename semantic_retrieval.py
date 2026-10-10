"""Fixed-chunk lexical/dense retrieval with auditable, conservative company joins.

Neural similarity retrieves research leads; it never establishes issuer business
or changes financial rules. The legacy lexical classifier supplies any positive
business assertion, and semantic-only matches remain explicitly uncertain.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from collections import Counter, defaultdict

import theme_search as lexical
from ingest_theme import filesystem_path, output_lock, _atomic_write


METHODS = ('fixed_bm25', 'dense', 'hybrid', 'hybrid_rerank')
CHUNK_VERSION = 'fixed-character-window-v1'


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       separators=(',', ':')) + '\n').encode('utf-8')


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _positive_int(value, name, maximum=None):
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f'{name} must be a positive integer within bounds')
    return value


def build_chunks(companies, documents, as_of, *, chunk_chars=480, overlap_chars=80):
    """Validate point-in-time data BEFORE indexing; offsets are source characters.

No query-dependent boundaries or generated text. Page boundaries are retained.
Repeated source records are removed by the existing provenance validator.
"""
    _positive_int(chunk_chars, 'chunk_chars', 4000)
    if chunk_chars < 80 or type(overlap_chars) is not int or not 0 <= overlap_chars < chunk_chars:
        raise ValueError('require chunk_chars >= 80 and 0 <= overlap_chars < chunk_chars')
    if not isinstance(companies, list) or not isinstance(documents, list):
        raise ValueError('companies and documents must be lists')
    cutoff = lexical._date(as_of, 'as_of')
    audit = {'rejected_companies': [], 'rejected_documents': [], 'conflicts': [],
             'duplicates': {'companies': 0, 'documents': 0, 'passages': 0}}
    universe = lexical._validate_companies(companies, cutoff, audit)
    visible = lexical._validate_documents(documents, universe, cutoff, audit)
    chunks = []
    for doc in visible:
        text = doc['text']
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_chars)
            excerpt = text[start:end]
            if excerpt.strip():
                chunk = {k: doc[k] for k in ('document_id', 'ticker', 'source_url', 'source_type',
                                             'page', 'available_at', 'date_basis')}
                for key in ('source_sha256', 'title'):
                    if key in doc:
                        chunk[key] = doc[key]
                chunk.update(char_start=start, char_end=end, exact_excerpt=excerpt)
                chunk['chunk_id'] = _hash(_bytes(chunk))
                chunks.append(chunk)
            if end == len(text):
                break
            start = end - overlap_chars
    chunks.sort(key=lambda x: x['chunk_id'])
    config = {'version': CHUNK_VERSION, 'chunk_chars': chunk_chars, 'overlap_chars': overlap_chars}
    fingerprint = _hash(_bytes({'config': config, 'chunks': chunks,
                               'companies': universe, 'as_of': cutoff.isoformat()}))
    return {'companies': universe, 'documents': visible, 'chunks': chunks, 'audit': audit,
            'as_of': cutoff.isoformat(), 'chunking': config, 'corpus_sha256': fingerprint}


def reciprocal_rank_fusion(rankings, *, k=60):
    _positive_int(k, 'rrf_k')
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, 1):
            if item in seen:
                raise ValueError('duplicate item within a fusion ranking')
            seen.add(item)
            scores[item] += 1.0 / (k + rank)
    return dict(scores)


class RetrievalIndex:
    """Reusable in-memory postings + persisted exact dense vector index.

Exact normalized dot product is the ANN-quality reference, not an approximate
index. Optional model bundle: encode(texts, kind), rerank(query,texts), fingerprint().
"""
    def __init__(self, companies, documents, as_of, *, model=None, cache_dir=None,
                 chunk_chars=480, overlap_chars=80):
        began = time.perf_counter()
        self.corpus = build_chunks(companies, documents, as_of, chunk_chars=chunk_chars,
                                   overlap_chars=overlap_chars)
        self.chunks = self.corpus['chunks']
        self.model, self.cache_dir = model, cache_dir
        self.vectors = None
        self.cache_status = 'not_requested'
        self._model_fingerprint = None
        self.postings = defaultdict(list)
        self.lengths = []
        # Lexical scoring deduplicates identical issuer text across source dates;
        # dated evidence itself remains available for business-state resolution.
        self.lexical_units, self.unit_chunks, keys = [], [], {}
        for index, chunk in enumerate(self.chunks):
            normalized, _ = lexical._matching_text(chunk['exact_excerpt'])
            key = (chunk['ticker'], normalized)
            if key not in keys:
                keys[key] = len(self.lexical_units)
                self.lexical_units.append(lexical._tokens(normalized))
                self.unit_chunks.append([])
            self.unit_chunks[keys[key]].append(index)
        for index, tokens in enumerate(self.lexical_units):
            self.lengths.append(sum(tokens.values()))
            for token, count in tokens.items():
                self.postings[token].append((index, count))
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        self.average_length = self.average_length or 1.0
        self.build_seconds = time.perf_counter() - began
        self._contexts = {}

    def _fingerprint(self):
        if self.model is None:
            raise ValueError('this method requires an explicit local model bundle')
        actual = self.model.fingerprint()
        if not isinstance(actual, dict) or not actual:
            raise ValueError('model fingerprint must be a nonempty dictionary')
        if self._model_fingerprint is not None and actual != self._model_fingerprint:
            raise ValueError('model artifacts changed during index lifetime')
        self._model_fingerprint = copy.deepcopy(actual)
        return actual

    def _validate_vectors(self, array, count):
        import numpy as np
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != count or array.shape[1] < 1:
            raise ValueError('encoder returned an invalid vector shape')
        if not np.isfinite(array).all():
            raise ValueError('encoder returned nonfinite vectors')
        norms = np.linalg.norm(array.astype(np.float64), axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError('encoder returned zero vectors')
        normalized = np.ascontiguousarray(array / norms, dtype=np.float32)
        if not np.isfinite(normalized).all() or np.any(np.linalg.norm(normalized, axis=1) <= 0):
            raise ValueError('encoder normalization failed')
        return normalized

    def prepare_dense(self):
        import numpy as np
        fingerprint = self._fingerprint()
        if self.vectors is not None:
            return
        if not self.chunks:
            self.vectors = np.empty((0, 0), dtype=np.float32)
            self.cache_status = 'empty_corpus'
            return
        cache_key = _hash(_bytes({'corpus': self.corpus['corpus_sha256'], 'model': fingerprint,
                                 'implementation_sha256': _hash(Path(__file__).read_bytes())}))
        def compute():
            # Identical text is encoded once, even if it is repeated in sources.
            texts = list(dict.fromkeys(c['exact_excerpt'] for c in self.chunks))
            encoded = self._validate_vectors(self.model.encode(texts, kind='passage'), len(texts))
            if self._fingerprint() != fingerprint:
                raise ValueError('model changed during encoding')
            offsets = {text: i for i, text in enumerate(texts)}
            return encoded[[offsets[c['exact_excerpt']] for c in self.chunks]]
        if self.cache_dir is None:
            self.vectors = compute()
            self.cache_status = 'computed_without_persistence'
            return
        directory = filesystem_path(self.cache_dir) / cache_key
        with output_lock(directory):
            directory.mkdir(parents=True, exist_ok=True)
            vectors_path, marker = directory / 'vectors.npy', directory / 'complete.json'
            for path in (directory, vectors_path, marker):
                if path.is_symlink():
                    raise ValueError('vector cache must not follow symlinks')
            if marker.exists():
                report = json.loads(marker.read_text(encoding='utf-8'))
                raw = vectors_path.read_bytes()
                if report.get('key') != cache_key or report.get('vectors_sha256') != _hash(raw):
                    raise ValueError('vector cache integrity mismatch')
                candidate = np.load(io.BytesIO(raw), allow_pickle=False)
                if candidate.dtype != np.float32:
                    raise ValueError('vector cache dtype mismatch')
                normalized = self._validate_vectors(candidate, len(self.chunks))
                if not np.allclose(candidate, normalized, rtol=1e-5, atol=1e-6):
                    raise ValueError('vector cache contains unnormalized vectors')
                self.vectors = candidate
                self.cache_status = 'verified_cache_hit'
            else:
                if vectors_path.exists():
                    raise ValueError('incomplete vector cache; preserve and rebuild in a new cache directory')
                candidate = compute()
                buffer = io.BytesIO()
                np.save(buffer, candidate, allow_pickle=False)
                raw = buffer.getvalue()
                _atomic_write(vectors_path, raw)
                _atomic_write(marker, _bytes({'schema_version': 1, 'key': cache_key,
                                             'corpus_sha256': self.corpus['corpus_sha256'],
                                             'model': fingerprint, 'shape': list(candidate.shape),
                                             'vectors_sha256': _hash(raw)}))
                self.vectors = candidate
                self.cache_status = 'computed_and_persisted'

    def _eligible(self, query):
        qualifiers = lexical._required_qualifiers(query)
        return {i for i, chunk in enumerate(self.chunks)
                if all(lexical._pattern(term).search(lexical._matching_text(chunk['exact_excerpt'])[0])
                       for term in qualifiers)}

    def _bm25(self, query, eligible):
        terms, _ = lexical._query_terms(query, lexical.DEFAULT_THEME_DICTIONARY)
        tokens = set().union(*(lexical._tokens(term) for term in terms))
        units = defaultdict(float)
        for token in sorted(tokens):
            posting = self.postings.get(token, [])
            idf = math.log(1 + (len(self.lexical_units) - len(posting) + .5) / (len(posting) + .5))
            for index, tf in posting:
                denominator = tf + 1.5 * (.25 + .75 * self.lengths[index] / self.average_length)
                units[index] += idf * tf * 2.5 / denominator
        return {index: value for unit, value in units.items() for index in self.unit_chunks[unit]
                if index in eligible and value > 0}

    def _ordered(self, scores, count):
        return sorted(scores, key=lambda i: (-scores[i], self.chunks[i]['chunk_id']))[:count]

    def search(self, query, *, method='hybrid_rerank', candidate_k=100, rerank_k=50, rrf_k=60):
        if method not in METHODS or not isinstance(query, str) or not lexical._tokens(query):
            raise ValueError('invalid method or empty query')
        _positive_int(candidate_k, 'candidate_k', 10000)
        _positive_int(rerank_k, 'rerank_k', 10000)
        _positive_int(rrf_k, 'rrf_k', 10000)
        if rerank_k > candidate_k:
            raise ValueError('rerank_k cannot exceed candidate_k')
        began = time.perf_counter()
        eligible = self._eligible(query)
        bm25, dense, fusion, rerank = self._bm25(query, eligible), {}, {}, {}
        if method != 'fixed_bm25' and eligible:
            self.prepare_dense()
            vector = self._validate_vectors(self.model.encode([query], kind='query'), 1)[0]
            if vector.shape[0] != self.vectors.shape[1]:
                raise ValueError('query and passage embedding dimensions differ')
            similarities = self.vectors @ vector
            dense = {i: float(similarities[i]) for i in sorted(eligible)}
        if method == 'fixed_bm25':
            selected, final = self._ordered(bm25, candidate_k), bm25
        elif method == 'dense':
            selected, final = self._ordered(dense, candidate_k), dense
        else:
            fusion = reciprocal_rank_fusion([self._ordered(bm25, candidate_k),
                                              self._ordered(dense, candidate_k)], k=rrf_k)
            selected, final = self._ordered(fusion, candidate_k), fusion
            if method == 'hybrid_rerank' and selected:
                selected = selected[:rerank_k]
                raw = list(self.model.rerank(query, [self.chunks[i]['exact_excerpt'] for i in selected]))
                if len(raw) != len(selected) or any(not math.isfinite(float(x)) for x in raw):
                    raise ValueError('reranker returned invalid scores')
                rerank = {i: float(score) for i, score in zip(selected, raw)}
                selected, final = self._ordered(rerank, rerank_k), rerank
        if method != 'fixed_bm25' and eligible:
            self._fingerprint()
        hits = [{**self.chunks[i], 'score': float(final[i]),
                 'bm25_score': bm25.get(i), 'dense_cosine': dense.get(i),
                 'rrf_score': fusion.get(i), 'reranker_logit': rerank.get(i)} for i in selected]
        return {'method': method, 'query': query, 'hits': hits,
                'latency_seconds': time.perf_counter() - began,
                'required_qualifiers': lexical._required_qualifiers(query),
                'eligible_chunks': len(eligible), 'total_chunks': len(self.chunks),
                'candidate_k': candidate_k, 'rerank_k': rerank_k, 'rrf_k': rrf_k,
                'cache_status': self.cache_status,
                'score_semantics': 'retrieval_order_only_not_probability_or_business_verification'}

    def _context(self, query):
        if query not in self._contexts:
            companies = list(self.corpus['companies'].values())
            report = lexical.discover_companies(companies, self.corpus['documents'], query,
                                                self.corpus['as_of'], limit=max(1, len(companies)))
            terms, _ = lexical._query_terms(query, lexical.DEFAULT_THEME_DICTIONARY)
            evidence = lexical._passages(self.corpus['documents'], [(t, lexical._pattern(lexical._norm(t))) for t in terms],
                                         480, {'duplicates': {'passages': 0}, 'qualifier_filtered_passages': 0},
                                         self.corpus['companies'], query, lexical._required_qualifiers(query))
            denials = defaultdict(list)
            for indexed in evidence:
                for item in indexed['versions']:
                    if not any(status == 'negated' for _, status in item['claims']):
                        continue
                    doc = item['document']
                    denial = {k: doc[k] for k in ('document_id', 'ticker', 'source_url', 'source_type',
                                                   'page', 'available_at', 'date_basis')}
                    if 'source_sha256' in doc:
                        denial['source_sha256'] = doc['source_sha256']
                    denial.update(char_start=item['start'], char_end=item['end'],
                                  exact_excerpt=item['exact_excerpt'], claim_status='negated',
                                  company_level_denial=item['company_level_denial'])
                    denials[doc['ticker']].append(denial)
            self._contexts[query] = report, dict(denials)
        return self._contexts[query]

    def company_report(self, query, *, method='hybrid_rerank', limit=20, candidate_k=100, rerank_k=50):
        _positive_int(limit, 'limit', 10000)
        search = self.search(query, method=method, candidate_k=candidate_k, rerank_k=rerank_k)
        baseline, denials = self._context(query)
        verified = {c['ticker']: c for c in baseline['companies']}
        grouped = defaultdict(list)
        for hit in search['hits']:
            grouped[hit['ticker']].append(hit)
        rows = []
        for ticker, hits in grouped.items():
            if ticker in verified:
                row = copy.deepcopy(verified[ticker])
                row['lexical_business_relevance_score'] = row['relevance_score']
            else:
                row = {k: self.corpus['companies'][ticker][k] for k in ('ticker', 'name', 'market', 'sector')}
                row.update(business_status='uncertain', business_status_reasons=[
                    'Semantic retrieval is a lead only; no lexical current-business proof for this query.'],
                    relations=['semantic_candidate'], matched_terms=[], evidence=[],
                    business_support_evidence=[], requires_review=True, warnings=[],
                    has_conflicting_assertions=False, has_mixed_business_stages=False,
                    evidence_count=0, evidence_truncated=False)
                if any(d['company_level_denial'] for d in denials.get(ticker, [])):
                    row['business_status'] = 'historical_or_disputed'
                    row['business_status_reasons'] = ['Issuer-wide denial found in visible source evidence.']
            row.update(relevance_score=max(h['score'] for h in hits),
                       retrieval_evidence=hits[:5], retrieval_evidence_count=len(hits),
                       retrieval_evidence_truncated=len(hits) > 5,
                       counter_evidence=copy.deepcopy(denials.get(ticker, [])),
                       semantic_only=ticker not in verified)
            rows.append(row)
        rows.sort(key=lambda c: (-c['relevance_score'], c['ticker']))
        search.pop('hits')
        return {'schema_version': 1, 'status': 'completed' if rows else 'no_matches',
                'query': query, 'as_of': self.corpus['as_of'], 'method': method,
                'companies': rows[:limit], 'all_companies': rows, 'audit': copy.deepcopy(self.corpus['audit']),
                'retrieval': search, 'model': copy.deepcopy(self._model_fingerprint),
                'corpus_sha256': self.corpus['corpus_sha256'], 'chunking': self.corpus['chunking'],
                'production_approved': False,
                'limitations': ['All semantic-only companies require business verification.',
                                'Scores are not probabilities, revenue exposure, or investment merit.',
                                'Top-k passage retrieval can omit companies; the supplied corpus is not the entire market.',
                                'No default relevance threshold has been calibrated on analyst-approved labels.']}
