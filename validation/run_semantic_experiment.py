"""Run frozen, source-backed retrieval ablations without changing their labels.

Models are local, inputs are hash checked, unjudged companies remain unknown.
Full evidence outputs and source text are intentionally excluded from this report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest_theme import filesystem_path, output_lock, _atomic_write
from semantic_retrieval import RetrievalIndex, METHODS
from theme_search import discover_companies
from neural_models import ModelBundle
from validation.semantic_eval import evaluate, validate_corpus_alignment


CODE_FILES = ('semantic_retrieval.py', 'neural_models.py', 'theme_search.py', 'ingest_theme.py',
              'validation/semantic_eval.py', 'validation/run_semantic_experiment.py')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _dump(value):
    return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')


def _jsonl(raw):
    return [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]


def _ranking(hits):
    scores={}
    for hit in hits:
        ticker=hit['ticker']
        if ticker not in scores or hit['score']>scores[ticker]['score']:
            scores[ticker]={'ticker':ticker,'score':hit['score'],
                           'evidence':[{'document_id':hit['document_id'],'page':hit['page'],
                                        'char_start':hit['char_start'],'char_end':hit['char_end'],
                                        'chunk_id':hit['chunk_id']}]}
    return sorted(scores.values(),key=lambda x:(-x['score'],x['ticker']))


def _protect_outputs(output, protected, model_dir, cache_dir):
    """Reject output aliases before opening the lock or replacing a result."""
    targets = [filesystem_path(str(output) + suffix) for suffix in ('', '.partial', '.lock')]
    sources = [filesystem_path(path) for path in protected]
    for index, target in enumerate(targets):
        for source in (*sources, *targets[:index]):
            if target == source or (target.exists() and source.exists() and target.samefile(source)):
                raise ValueError('Experiment output overlaps an input or another output: ' + str(target))
        if any(target.is_relative_to(filesystem_path(directory)) for directory in (model_dir, cache_dir)):
            raise ValueError('Experiment output overlaps the model or vector cache directory: ' + str(target))


def run(*, dataset_path, model_dir, cache_dir, output, threads=2, candidate_k=100, rerank_k=50):
    began=time.perf_counter()
    dataset_path=filesystem_path(dataset_path)
    output=filesystem_path(output)
    model_dir=filesystem_path(model_dir)
    cache_dir=filesystem_path(cache_dir)
    label_raw=dataset_path.read_bytes();dataset=json.loads(label_raw)
    protected=[dataset_path, ROOT/'data/theme_companies.json',
               *(ROOT/item['path'] for item in dataset['corpus_inputs']),
               *(ROOT/name for name in CODE_FILES)]
    # Check before output_lock: even its adjacent file must never alias an input.
    _protect_outputs(output,protected,model_dir,cache_dir)
    with output_lock(output):
        _protect_outputs(output,protected,model_dir,cache_dir)
        return _run_locked(dataset_path=dataset_path,model_dir=model_dir,cache_dir=cache_dir,
                           output=output,label_raw=label_raw,dataset=dataset,protected=protected,
                           began=began,threads=threads,candidate_k=candidate_k,rerank_k=rerank_k)


def _run_locked(*, dataset_path, model_dir, cache_dir, output, label_raw, dataset, protected,
                began, threads, candidate_k, rerank_k):
    snapshot={item['path']:(ROOT/item['path']).read_bytes() for item in dataset['corpus_inputs']}
    for item in dataset['corpus_inputs']:
        if _sha(snapshot[item['path']])!=item['sha256']:
            raise ValueError('frozen corpus hash mismatch: '+item['path'])
    company_path=ROOT/'data/theme_companies.json';company_raw=company_path.read_bytes()
    companies=json.loads(company_raw)
    if sorted(c['ticker'] for c in companies)!=sorted(dataset['candidate_tickers']):
        raise ValueError('candidate company universe differs from frozen benchmark')
    full=_jsonl(snapshot['output/theme_full_documents.jsonl'])
    extracted={d['ticker'] for d in full}
    supplements=[d for d in _jsonl(snapshot['data/theme_documents.jsonl']) if d['ticker'] not in extracted]
    documents=full+supplements
    alignment=validate_corpus_alignment(dataset,documents)
    code={name:_sha((ROOT/name).read_bytes()) for name in CODE_FILES}
    model=ModelBundle(model_dir,threads=threads)
    fingerprint=model.fingerprint()
    index=RetrievalIndex(companies,documents,dataset['as_of'],model=model,cache_dir=cache_dir)
    print(json.dumps({'stage':'index_prepared','chunks':len(index.chunks),'documents':len(documents),
                      'label_sha256':_sha(label_raw)}),flush=True)
    encoding_start=time.perf_counter();index.prepare_dense();encoding_seconds=time.perf_counter()-encoding_start
    print(json.dumps({'stage':'passages_encoded','seconds':encoding_seconds,'cache':index.cache_status}),flush=True)
    runs={method:{} for method in ('legacy_bm25',*METHODS)}
    latencies={method:[] for method in runs}
    query_audits=[]
    for number,case in enumerate(dataset['queries'],1):
        query,query_id=case['text'],case['query_id']
        start=time.perf_counter()
        baseline=discover_companies(companies,documents,query,dataset['as_of'],limit=len(companies))
        latencies['legacy_bm25'].append(time.perf_counter()-start)
        runs['legacy_bm25'][query_id]=[{'ticker':c['ticker'],'score':c['relevance_score'],
                                       'evidence':[{'document_id':e['document_id'],'page':e['page'],
                                                    'char_start':e['char_start'],'char_end':e['char_end']}
                                                   for e in c['evidence']]} for c in baseline['companies']]
        for method in METHODS:
            search=index.search(query,method=method,candidate_k=candidate_k,rerank_k=rerank_k)
            runs[method][query_id]=_ranking(search['hits'])
            latencies[method].append(search['latency_seconds'])
            query_audits.append({'query_id':query_id,'method':method,
                                 'eligible_chunks':search['eligible_chunks'],
                                 'required_qualifiers':search['required_qualifiers'],
                                 'retrieved_chunks':len(search['hits'])})
        print(json.dumps({'stage':'query_completed','number':number,'total':len(dataset['queries']),
                          'query_id':query_id}),flush=True)
        # A crash leaves a visibly partial record, never a completed result.
        _protect_outputs(output,protected,model_dir,cache_dir)
        _atomic_write(filesystem_path(str(output)+'.partial'),_dump({'status':'partial','completed_queries':number,
                                                   'dataset_sha256':_sha(label_raw),'rankings':runs}))
    if dataset_path.read_bytes()!=label_raw or company_path.read_bytes()!=company_raw:
        raise ValueError('labels or companies changed during evaluation')
    if any((ROOT/name).read_bytes()!=raw for name,raw in snapshot.items()):
        raise ValueError('corpus changed during evaluation')
    if any(_sha((ROOT/name).read_bytes())!=value for name,value in code.items()):
        raise ValueError('code changed during evaluation')
    if model.fingerprint()!=fingerprint:
        raise ValueError('model artifacts changed during evaluation')
    metrics=evaluate(dataset,runs,k_values=(1,3,5))
    def timings(values):
        ordered=sorted(values)
        return {'observations':len(values),'median_seconds':statistics.median(values),
                'p95_seconds':ordered[min(len(ordered)-1, int(.95*len(ordered)))],
                'total_seconds':sum(values)}
    report={'schema_version':1,'status':'completed','production_approved':False,
            'dataset_sha256':_sha(label_raw),'company_sha256':_sha(company_raw),
            'dataset_kind':dataset['acceptance_status'],'blind_holdout':False,
            'corpus':{'input_hashes':{name:_sha(raw) for name,raw in snapshot.items()},
                      'pdf_pages':len(full),'website_excerpts':len(supplements),
                      'chunks':len(index.chunks),'corpus_sha256':index.corpus['corpus_sha256']},
            'code_sha256':code,'model':fingerprint,'model_statistics':model.statistics(),
            'parameters':{'candidate_k':candidate_k,'rerank_k':rerank_k,'rrf_k':60,
                          'chunking':index.corpus['chunking'],'threads':threads},
            'environment':{'python':platform.python_version(),'platform':platform.system()},
            'timing':{'index_build_seconds':index.build_seconds,'encoding_seconds':encoding_seconds,
                      'encoding_cache':index.cache_status,'per_method':{k:timings(v) for k,v in latencies.items()},
                      'total_seconds':time.perf_counter()-began},
            'corpus_alignment':alignment,'metrics':metrics,'rankings':runs,'query_audits':query_audits,
            'limitations':['Assistant-annotated development set, not analyst acceptance or a blind holdout.',
                           'Intent variants share labels and are not independent observations.',
                           'Unknown companies are unjudged, not implicit negatives.',
                           'Company relevance does not validate current business, financial quality, or returns.',
                           'Legacy baseline includes rule weighting; fixed BM25 is the controlled lexical ablation.',
                           'Exact dense cosine is used; no approximate-nearest-neighbor speed claims.',
                           'Latency includes local model/session work on one machine; no production SLA.']}
    _protect_outputs(output,protected,model_dir,cache_dir)
    _atomic_write(filesystem_path(output),_dump(report))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'validation/semantic_dataset.json')
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=ROOT/'output/semantic_vectors')
    parser.add_argument('--output',type=Path,default=ROOT/'validation/output/semantic_experiment.json')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--candidate-k',type=int,default=100)
    parser.add_argument('--rerank-k',type=int,default=50)
    args=parser.parse_args(argv)
    report=run(dataset_path=args.dataset,model_dir=args.models,cache_dir=args.cache,output=args.output,
               threads=args.threads,candidate_k=args.candidate_k,rerank_k=args.rerank_k)
    print(json.dumps({'status':report['status'],'seconds':report['timing']['total_seconds'],
                      'dataset_sha256':report['dataset_sha256']}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
