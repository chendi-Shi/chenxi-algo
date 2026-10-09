"""Compare grid/beam search and verify that real-data gaps stay blocked."""
import copy
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import config_from, load_csv
from search import search_configs


def summary(result):
    return {key: result[key] for key in ('status', 'method', 'evaluations', 'grid_combinations',
        'exhaustive', 'heuristic', 'global_optimum_verified', 'baseline', 'best')}


def main():
    statements = load_csv(ROOT/'examples'/'statements.csv')
    valuations = load_csv(ROOT/'examples'/'valuations.csv')
    config = config_from(None)
    runs = {}
    for method in ('grid', 'beam'):
        result = search_configs(statements, valuations, date(2026, 10, 9), config,
            market='A', sector='Consumer', target_count=3, method=method, beam_width=5)
        runs[method] = summary(result)
    real = json.loads((ROOT/'validation'/'dataset.json').read_text(encoding='utf-8'))
    strict = search_configs(real['statements'], real['valuations'], date(2025, 6, 30), config,
        market='A', sector='Liquor', target_count=3)
    reconstructed = copy.deepcopy(real['statements'])
    for row in reconstructed:
        row['available_at'] = row['original_notice_date']
    reconstruction = search_configs(reconstructed, real['valuations'], date(2025, 6, 30), config,
        market='A', sector='Liquor', target_count=3)
    for result in (strict, reconstruction):
        assert result['status'] == 'insufficient_verified_data'
        assert result['best'] is None and result['baseline']['eligible_count'] == 0
    output = {'date': '2026-10-09', 'synthetic_comparison': runs,
        'real_current_version_block': summary(strict),
        'real_unverified_notice_reconstruction_block': summary(reconstruction),
        'code_sha256': {file: hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
                        for file in ('engine.py', 'search.py')},
        'note': 'Algorithmic search comparison on synthetic data; real incomplete data remains blocked. No investment efficacy test.'}
    path = ROOT/'validation'/'search_results.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({method: {'evaluations': run['evaluations'], 'status': run['status'],
        'baseline_count': run['baseline']['candidate_count'],
        'best_count': run['best']['candidate_count'] if run['best'] else None,
        'target_met': run['best']['target_met'] if run['best'] else None,
        'global_optimum_verified': run['global_optimum_verified']}
        for method, run in runs.items()}, ensure_ascii=True))
    print('Real-data blockers preserved in strict and unverified-notice reconstructions.')


if __name__ == '__main__':
    main()
