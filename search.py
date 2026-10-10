"""Bounded configuration search for a same-cohort research list.

The objective is research capacity, not investment performance: first minimize
distance from the requested candidate count, then changes from the baseline.
Only stricter thresholds and the same enabled score blocks are permitted.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import statistics
from collections import Counter

from engine import BLOCK_FIELDS, DEFAULT_CONFIG, screen, validate_config

AXES = ('weights', 'min_roe', 'min_cash_conversion',
        'max_net_debt_to_cfo', 'candidate_score')
MINIMUM_AXES = {'min_roe', 'min_cash_conversion', 'candidate_score'}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number')
    try:
        converted = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(converted):
        raise ValueError(f'{name} must be a finite number')
    return converted


def _weights(value):
    if not isinstance(value, dict) or set(value) != set(BLOCK_FIELDS):
        raise ValueError('weights must contain all four scoring blocks')
    weights = {k: _number(value[k], 'weight') for k in BLOCK_FIELDS}
    if any(v < 0 for v in weights.values()) or not math.isclose(sum(weights.values()), 1, abs_tol=1e-9):
        raise ValueError('weights must be nonnegative and sum to one')
    return weights


def _key(config):
    # Numeric 1 and 1.0 are the same configuration, including the baseline.
    values = {k: (_weights(config[k]) if k == 'weights' else float(config[k])) for k in AXES}
    return json.dumps(values, sort_keys=True, separators=(',', ':'), allow_nan=False)


def default_space(base):
    """405 configurations for the normal four-block baseline, including itself."""
    profiles = [base['weights'],
        {'quality': .25, 'value': .25, 'growth': .25, 'balance': .25},
        {'quality': .60, 'value': .15, 'growth': .10, 'balance': .15},
        {'quality': .20, 'value': .50, 'growth': .15, 'balance': .15},
        {'quality': .20, 'value': .15, 'growth': .50, 'balance': .15}]
    active = {k for k, v in base['weights'].items() if v > 0}
    projected = []
    for profile in profiles:
        total = sum(profile[k] for k in active)
        projected.append({k: profile[k] / total if k in active else 0 for k in BLOCK_FIELDS})
    return {
        'weights': projected,
        'min_roe': [base['min_roe'], max(base['min_roe'], .12), max(base['min_roe'], .16)],
        'min_cash_conversion': [base['min_cash_conversion'], max(base['min_cash_conversion'], 1), max(base['min_cash_conversion'], 1.2)],
        'max_net_debt_to_cfo': [base['max_net_debt_to_cfo'], min(base['max_net_debt_to_cfo'], 2), min(base['max_net_debt_to_cfo'], 0)],
        'candidate_score': [base['candidate_score'], max(base['candidate_score'], 70), max(base['candidate_score'], 75)],
    }


def _space(supplied, base):
    if not isinstance(supplied, dict) or not supplied or set(supplied) - set(AXES):
        raise ValueError('search space must be a nonempty object with supported axes')
    active = {k for k, v in base['weights'].items() if v > 0}
    result = {}
    for axis in AXES:
        if axis not in supplied:
            continue
        choices = supplied[axis]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f'{axis} search axis must be a nonempty list')
        validated, seen = [], set()
        for raw in choices:
            if axis == 'weights':
                value = _weights(raw)
                if {k for k, v in value.items() if v > 0} != active:
                    raise ValueError('search may not enable or disable baseline scoring blocks')
            else:
                value = _number(raw, axis)
                if axis in MINIMUM_AXES and value < base[axis]:
                    raise ValueError(f'search may not relax baseline {axis}')
                if axis == 'max_net_debt_to_cfo' and value > base[axis]:
                    raise ValueError('search may not relax baseline max_net_debt_to_cfo')
                if axis == 'candidate_score' and not 0 <= value <= 100:
                    raise ValueError('candidate_score must be between zero and 100')
            identity = json.dumps(value, sort_keys=True, allow_nan=False)
            if identity not in seen:
                validated.append(value)
                seen.add(identity)
        result[axis] = validated
    return result


def _distance(config, baseline):
    cost = sum(abs(config['weights'][k] - baseline['weights'][k]) for k in BLOCK_FIELDS)
    for axis, scale in (('min_roe', .01), ('min_cash_conversion', .1),
                        ('max_net_debt_to_cfo', 1), ('candidate_score', 10)):
        cost += abs(config[axis] - baseline[axis]) / max(abs(baseline[axis]), scale)
    return cost


def _order(record):
    objective = record['objective']
    mean = objective['mean_baseline_score']
    return (objective['target_distance'], objective['configuration_distance'],
            -(mean if mean is not None else -1), _key(record['config']))


def _beam_order(record, target_count):
    # Weights are assigned first. Remaining thresholds can only remove names:
    # an overfull partial list may still reach the target, an underfull one cannot.
    # This optimistic count bound avoids pruning a useful overfull branch simply
    # because a different branch has already removed too many companies.
    order = _order(record)
    return (max(0, target_count - record['candidate_count']), *order[1:])


class _BudgetReached(Exception):
    pass


def search_configs(statement_rows, valuation_rows, as_of, base_config, *,
                   market, sector, target_count=5, method='grid', beam_width=5,
                   max_evaluations=1000, top_k=5, space=None):
    """Search a bounded configuration space without changing the main screen.

    Eligibility is frozen from baseline watchlist/candidate rows that have a
    complete composite score. The search cannot rehabilitate rejected or
    incomplete companies. All comparisons stay inside one market/sector cohort.
    Grid checks every combination if budget permits. Beam evaluates completed
    default-filled partial assignments, retaining only beam_width per depth;
    only full assignments and the baseline can become final alternatives.
    """
    if market not in ('A', 'HK') or not isinstance(sector, str) or not sector.strip():
        raise ValueError('search needs an explicit market A/HK and nonempty sector')
    sector = sector.strip()
    if method not in ('grid', 'beam'):
        raise ValueError('search method must be grid or beam')
    for name, value in (('target_count', target_count), ('beam_width', beam_width),
                        ('max_evaluations', max_evaluations), ('top_k', top_k)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not isinstance(base_config, dict) or set(base_config) - set(DEFAULT_CONFIG):
        raise ValueError('base_config has unsupported configuration keys')
    base = validate_config(base_config)
    base['weights'] = _weights(base['weights'])
    for axis in AXES[1:]:
        base[axis] = _number(base[axis], axis)
    if not 0 <= base['candidate_score'] <= 100:
        raise ValueError('baseline candidate_score must be between zero and 100')
    choices = _space(default_space(base) if space is None else space, base)
    baseline_report = screen(statement_rows, valuation_rows, as_of, base)
    cohort = [c for c in baseline_report['companies'] if c['market'] == market and c['sector'] == sector]
    eligible = {c['ticker']: c for c in cohort if c['status'] in ('candidate', 'watchlist')
                and c['score'] is not None and math.isfinite(c['score'])}
    base_candidates = sorted(c['ticker'] for c in cohort if c['status'] == 'candidate' and c['ticker'] in eligible)
    result = {
        'status': 'completed', 'method': method, 'market': market, 'sector': sector,
        'as_of': as_of.isoformat(), 'target_count': target_count,
        'space': choices, 'grid_combinations': math.prod(len(v) for v in choices.values()),
        'max_evaluations': max_evaluations, 'beam_width': beam_width, 'evaluations': 1,
        'exhaustive': False, 'heuristic': method == 'beam', 'global_optimum_verified': False,
        'baseline': {'config': copy.deepcopy(base), 'candidate_count': len(base_candidates),
                     'candidates': base_candidates, 'eligible_count': len(eligible),
                     'eligible_tickers': sorted(eligible),
                     'cohort_status_counts': dict(Counter(c['status'] for c in cohort))},
        'best': None, 'alternatives': [],
        'target_feasible_in_eligible_pool': target_count <= len(eligible),
        'objective': ['minimize absolute candidate-count distance from target',
                      'minimize normalized threshold changes plus weight L1 distance from baseline',
                      'prefer higher mean fixed-baseline score only as a further tie-break'],
        'limitations': [
            'Research-list capacity and preference search, not investment-performance optimization.',
            'No future financial outcomes, stock returns or labels enter the objective.',
            'Baseline data_review, specialist_review, excluded and unscored companies remain ineligible.',
            'Thresholds can only tighten and enabled scoring blocks stay unchanged.',
            'A numerical search optimum does not establish a superior investment strategy.',
            'Beam pruning can miss the best grid configuration; evaluation budgets may stop either method.',
        ],
    }
    if not eligible:
        result['status'] = 'insufficient_verified_data'
        return result

    cache, final_records = {}, {}

    def evaluate(config, report=None):
        key = _key(config)
        if key in cache:
            return cache[key]
        if len(cache) >= max_evaluations:
            raise _BudgetReached
        report = report or screen(statement_rows, valuation_rows, as_of, config)
        selected = [c for c in report['companies'] if c['market'] == market and c['sector'] == sector
                    and c['status'] == 'candidate']
        candidates = sorted(c['ticker'] for c in selected if c['ticker'] in eligible)
        candidate_set, baseline_set = set(candidates), set(base_candidates)
        union = candidate_set | baseline_set
        record = {
            'config': copy.deepcopy(config), 'candidate_count': len(candidates),
            'candidates': candidates, 'target_met': len(candidates) == target_count,
            'objective': {'target_distance': abs(len(candidates) - target_count),
                          'configuration_distance': _distance(config, base),
                          'mean_baseline_score': statistics.mean(eligible[t]['score'] for t in candidates) if candidates else None},
            'overlap_with_baseline': len(candidate_set & baseline_set) / len(union) if union else None,
            'suppressed_candidates': sorted(c['ticker'] for c in selected if c['ticker'] not in eligible),
        }
        cache[key] = record
        return record

    initial = evaluate(base, baseline_report)
    final_records[_key(base)] = initial
    dimensions = list(choices)
    exhausted, pruned = False, False
    try:
        if method == 'grid':
            for combination in itertools.product(*(choices[k] for k in dimensions)):
                config = copy.deepcopy(base)
                config.update(zip(dimensions, combination))
                record = evaluate(config)
                final_records[_key(config)] = record
        else:
            frontier = [copy.deepcopy(base)]
            for depth, axis in enumerate(dimensions):
                expanded = {}
                for parent in frontier:
                    for value in choices[axis]:
                        config = copy.deepcopy(parent)
                        config[axis] = copy.deepcopy(value)
                        record = evaluate(config)
                        expanded[_key(config)] = record
                        if depth == len(dimensions) - 1:
                            final_records[_key(config)] = record
                ordered = sorted(expanded.values(), key=lambda r: _beam_order(r, target_count))
                if depth < len(dimensions) - 1 and len(ordered) > beam_width:
                    pruned = True
                frontier = [copy.deepcopy(r['config']) for r in ordered[:beam_width]]
    except _BudgetReached:
        exhausted = True
    ordered = sorted(final_records.values(), key=_order)
    result.update(status='budget_exhausted' if exhausted else 'completed',
                  evaluations=len(cache), complete_alternatives_evaluated=len(final_records),
                  exhaustive=not exhausted and not pruned,
                  heuristic=method == 'beam' and (exhausted or pruned),
                  global_optimum_verified=not exhausted and not pruned,
                  best=copy.deepcopy(ordered[0]), alternatives=copy.deepcopy(ordered[:top_k]))
    return result
