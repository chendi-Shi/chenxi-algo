"""Auditable annual fundamental screening for non-financial A/H companies.

F-score signal structure adapted from Ayon Dey's MIT-licensed implementation.
See reference/LICENSE and RESEARCH.md. No price prediction or order generation.
"""
from __future__ import annotations

import csv
import copy
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

MONEY_FIELDS = (
    'net_income', 'net_income_parent', 'core_income_parent',
    'operating_cash_flow', 'total_assets', 'equity_parent', 'current_assets',
    'current_liabilities', 'long_term_debt', 'total_debt', 'cash',
    'revenue', 'cogs', 'capex', 'ebit', 'interest_expense',
)
SIGNALS = (
    'positive_roa', 'positive_cfo', 'roa_improved', 'accruals_quality',
    'lower_leverage', 'higher_current_ratio', 'no_equity_issuance',
    'higher_gross_margin', 'higher_asset_turnover',
)
BLOCK_FIELDS = {
    'quality': {'roe': True, 'cash_conversion': True, 'core_profit_share': True},
    'value': {'core_earnings_yield': True, 'fcf_yield': True},
    'growth': {'revenue_cagr_2y': True, 'core_profit_cagr_2y': True},
    'balance': {'net_debt_to_cfo': False, 'interest_cover': True},
}
DEFAULT_CONFIG = {
    'weights': {'quality': .40, 'value': .25, 'growth': .20, 'balance': .15},
    'min_peer_count': 5,
    'max_filing_age_days': 550,
    'max_valuation_age_days': 7,
    'max_fx_age_days': 7,
    'min_f_score': 6,
    'candidate_score': 65,
    'min_roe': .08,
    'min_cash_conversion': .8,
    'max_net_debt_to_cfo': 4,
    'excluded_sectors': ['Banks', 'Insurance', 'Financials', 'REITs', '银行', '保险', '金融', '房地产信托'],
}


def iso(value: str) -> date:
    # Only date precision is supported; cutoff means end of that calendar day.
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('date must be YYYY-MM-DD')
    return date.fromisoformat(value)


def source_url_valid(value):
    try:
        parsed = urlparse(value)
        return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
    except (TypeError, ValueError):
        return False


def number(value, field: str):
    if value is None or str(value).strip() in ('', 'NA', 'N/A', 'null'):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{field}: invalid numeric value') from exc
    if not math.isfinite(result):
        raise ValueError(f'{field}: NaN/inf is invalid')
    return result


def divide(a, b, positive_denominator=True):
    if a is None or b is None or b == 0 or (positive_denominator and b <= 0):
        return None
    result = a / b
    return result if math.isfinite(result) else None


def compare(a, b, op):
    return None if a is None or b is None else op(a, b)


def average(a, b):
    return None if a is None or b is None else a / 2 + b / 2


def difference(a, b):
    if a is None or b is None:
        return None
    result = a - b
    return result if math.isfinite(result) else None


def growth(current, prior):
    return difference(divide(current, prior), 1)


def cagr(current, old, years=2):
    if current is None or old is None or current <= 0 or old <= 0:
        return None
    # Log-space avoids overflow when both raw inputs are finite.
    try:
        result = math.expm1((math.log(current) - math.log(old)) / years)
        return result if math.isfinite(result) else None
    except (OverflowError, ValueError):
        return None


def parse_statement(raw: dict) -> dict:
    r = dict(raw)
    for key in ('ticker', 'name', 'market', 'sector', 'currency', 'scope',
                'period_start', 'period_end', 'available_at', 'source_url', 'filing_id'):
        if not isinstance(r.get(key), str) or not r[key].strip():
            raise ValueError(f'missing {key}')
        r[key] = r[key].strip()
    r['ticker'] = r['ticker'].upper()
    if r['market'] not in ('A', 'HK'):
        raise ValueError('market must be A or HK')
    if r['scope'] != 'consolidated':
        raise ValueError('scope must be consolidated')
    if not source_url_valid(r['source_url']):
        raise ValueError('source_url must be an HTTP(S) provenance reference')
    start, end, available = map(iso, (r['period_start'], r['period_end'], r['available_at']))
    if not 350 <= (end - start).days + 1 <= 380:
        raise ValueError('requires a full annual period (350-380 days)')
    if available < end:
        raise ValueError('available_at is before period_end')
    scale = number(r.get('unit_scale'), 'unit_scale')
    if scale is None or scale <= 0:
        raise ValueError('positive unit_scale is required')
    for field in MONEY_FIELDS:
        r[field] = number(r.get(field), field)
        if r[field] is not None:
            r[field] *= scale
            if not math.isfinite(r[field]):
                raise ValueError(f'{field}: unit conversion overflow')
    for field in ('total_assets', 'revenue'):
        if r[field] is not None and r[field] <= 0:
            raise ValueError(f'{field} must be positive')
    for field in ('current_assets', 'current_liabilities', 'long_term_debt',
                  'total_debt', 'cash', 'cogs', 'capex', 'interest_expense'):
        if r[field] is not None and r[field] < 0:
            raise ValueError(f'{field} must be nonnegative; verify sign convention')
    issued = str(r.get('equity_issued', '')).strip()
    if issued not in ('', '0', '1'):
        raise ValueError('equity_issued must be 0, 1, or blank')
    r['equity_issued'] = None if issued == '' else int(issued)
    r['unit_scale'] = 1.0
    return r


def visible_versions(rows: list[dict], as_of: date, end_field: str):
    """Resolve latest visible revision deterministically, without future rewrites."""
    selected, rejected, seen_versions = {}, [], {}
    for line, raw in enumerate(rows, 2):
        raw = dict(raw)
        ticker = raw.get('ticker', '').strip().upper() if isinstance(raw.get('ticker'), str) else ''
        raw['ticker'] = ticker
        try:
            if not ticker:
                raise ValueError('missing ticker')
            available = iso(raw.get('available_at', ''))
            if available > as_of:
                rejected.append({'line': line, 'ticker': ticker, 'reason': 'not_yet_available'})
                continue
            end = iso(raw.get(end_field, ''))
            if end > as_of:
                rejected.append({'line': line, 'ticker': ticker, 'reason': 'not_yet_available'})
                continue
            revision = int(raw.get('revision_id', '0') or '0')
            if revision < 0:
                raise ValueError('revision_id must be nonnegative')
            key, version = (ticker, end), (available, revision)
            seen_key = (*key, *version)
            if seen_key in seen_versions and seen_versions[seen_key] != raw:
                raise ValueError('conflicting rows at identical date/revision')
            seen_versions[seen_key] = raw
            prior = selected.get(key)
            if not prior or version > prior[0]:
                selected[key] = version, raw
        except (ValueError, TypeError) as exc:
            rejected.append({'line': line, 'ticker': ticker, 'reason': str(exc)})
    return [v[1] for v in selected.values()], rejected


def compute_metrics(old: dict, prior: dict, current: dict, valuation: dict | None):
    """Three annual balance sheets provide the two opening asset denominators."""
    o, p, c = old, prior, current
    roa = divide(c['net_income'], p['total_assets'])
    prior_roa = divide(p['net_income'], o['total_assets'])
    cfo_assets = divide(c['operating_cash_flow'], p['total_assets'])
    lev = divide(c['long_term_debt'], average(c['total_assets'], p['total_assets']))
    prior_lev = divide(p['long_term_debt'], average(p['total_assets'], o['total_assets']))
    liquid = divide(c['current_assets'], c['current_liabilities'])
    prior_liquid = divide(p['current_assets'], p['current_liabilities'])
    gm = divide(difference(c['revenue'], c['cogs']), c['revenue'])
    prior_gm = divide(difference(p['revenue'], p['cogs']), p['revenue'])
    turnover, prior_turnover = divide(c['revenue'], p['total_assets']), divide(p['revenue'], o['total_assets'])
    signals = {
        'positive_roa': compare(roa, 0, lambda a, b: a > b),
        'positive_cfo': compare(cfo_assets, 0, lambda a, b: a > b),
        'roa_improved': compare(roa, prior_roa, lambda a, b: a > b),
        'accruals_quality': compare(cfo_assets, roa, lambda a, b: a > b),
        'lower_leverage': compare(lev, prior_lev, lambda a, b: a < b),
        'higher_current_ratio': compare(liquid, prior_liquid, lambda a, b: a > b),
        'no_equity_issuance': None if c['equity_issued'] is None else c['equity_issued'] == 0,
        'higher_gross_margin': compare(gm, prior_gm, lambda a, b: a > b),
        'higher_asset_turnover': compare(turnover, prior_turnover, lambda a, b: a > b),
    }
    net_debt = difference(c['total_debt'], c['cash'])
    fcf = difference(c['operating_cash_flow'], c['capex'])
    cap = None if valuation is None else valuation['market_cap_reporting']
    metrics = {
        'roa': roa, 'roe': divide(c['net_income_parent'], average(c['equity_parent'], p['equity_parent'])),
        'gross_margin': gm,
        'cash_conversion': divide(c['operating_cash_flow'], c['net_income']),
        'core_profit_share': divide(c['core_income_parent'], c['net_income_parent']),
        'fcf': fcf, 'core_earnings_yield': divide(c['core_income_parent'], cap),
        'fcf_yield': divide(fcf, cap),
        'revenue_cagr_2y': cagr(c['revenue'], o['revenue']),
        'core_profit_cagr_2y': cagr(c['core_income_parent'], o['core_income_parent']),
        'net_debt_to_cfo': divide(net_debt, c['operating_cash_flow']),
        # Debt-free/interest-free firms use missing, not an artificial infinity.
        'interest_cover': divide(c['ebit'], c['interest_expense']),
        'current_ratio': liquid,
        'revenue_growth_yoy': growth(c['revenue'], p['revenue']),
        'core_profit_growth_yoy': growth(c['core_income_parent'], p['core_income_parent']),
    }
    return metrics, signals


def percentile(value: float, values: list[float], higher=True):
    """Ties share their midrank. A constant cohort is neutral (50), never 100."""
    if len(values) < 2 or min(values) == max(values):
        return 50.0
    lower = sum(x < value for x in values)
    equal = sum(x == value for x in values)
    rank = (lower + (equal - 1) / 2) / (len(values) - 1) * 100
    return rank if higher else 100 - rank


def config_from(path: str | None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        supplied = json.loads(Path(path).read_text(encoding='utf-8'))
        unknown = set(supplied) - set(config)
        if unknown:
            raise ValueError(f'unknown configuration keys: {sorted(unknown)}')
        config.update(supplied)
    weights = config['weights']
    if set(weights) != set(BLOCK_FIELDS) or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in weights.values()) or not math.isclose(sum(weights.values()), 1):
        raise ValueError('four finite nonnegative weights must sum to 1')
    for key in ('min_peer_count', 'max_filing_age_days', 'max_valuation_age_days', 'max_fx_age_days'):
        if not isinstance(config[key], int) or config[key] < (2 if key == 'min_peer_count' else 0):
            raise ValueError(f'invalid {key}')
    for key in ('min_f_score', 'candidate_score', 'min_roe', 'min_cash_conversion', 'max_net_debt_to_cfo'):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]):
            raise ValueError(f'invalid {key}')
    if not 0 <= config['min_f_score'] <= 9 or not 0 <= config['candidate_score'] <= 100:
        raise ValueError('score threshold is outside its range')
    if not isinstance(config['excluded_sectors'], list) or not all(isinstance(x, str) for x in config['excluded_sectors']):
        raise ValueError('excluded_sectors must be a list of strings')
    return config


def load_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def screen(statement_rows, valuation_rows, as_of: date, config=None):
    config = config or config_from(None)
    visible, audit = visible_versions(statement_rows, as_of, 'period_end')
    valuations, value_audit = visible_versions(valuation_rows, as_of, 'snapshot_date')
    history = defaultdict(list)
    for raw in visible:
        history[raw['ticker']].append(raw)
    latest_values = {}
    for raw in valuations:
        if raw['ticker'] not in latest_values or raw['snapshot_date'] > latest_values[raw['ticker']]['snapshot_date']:
            latest_values[raw['ticker']] = raw
    invalid_tickers = {r['ticker'] for r in audit if r['reason'] != 'not_yet_available'}
    invalid_values = {r['ticker'] for r in value_audit if r['reason'] != 'not_yet_available'}
    results, clean_rows = [], []
    all_tickers = sorted({r['ticker'].strip().upper() for r in statement_rows if isinstance(r.get('ticker'), str)} - {''})
    for ticker in all_tickers:
        rows = sorted(history[ticker], key=lambda r: r['period_end'])[-3:]
        display_metadata = {key: str(rows[-1].get(key) or '') if rows else '' for key in ('name', 'market', 'sector')}
        r = {'ticker': ticker, **display_metadata,
             'status': 'data_review', 'score': None, 'blocks': {}, 'metrics': {},
             'f_score': None, 'f_score_lower_bound': 0, 'f_score_known': 0,
             'signals': {}, 'warnings': [], 'reasons': [], 'sources': [], 'peer_count': 0}
        results.append(r)
        try:
            if ticker in invalid_tickers:
                raise ValueError('ambiguous or invalid statement version; see audit log')
            if len(rows) < 3:
                raise ValueError('requires three visible consecutive annual statements')
            parsed = [parse_statement(x) for x in rows]
            clean_rows.extend(parsed)
            o, p, c = parsed
            r.update({key: c[key] for key in ('name', 'market', 'sector')})
            r['period_end'], r['currency'] = c['period_end'], c['currency']
            r['sources'] = [{'period_end': x['period_end'], 'available_at': x['available_at'], 'filing_id': x['filing_id'], 'source_url': x['source_url']} for x in parsed]
            if len({x['currency'] for x in parsed}) != 1:
                raise ValueError('reporting currency changed; needs restated comparable history')
            if len({x['sector'] for x in parsed}) != 1 or len({x['market'] for x in parsed}) != 1:
                raise ValueError('sector/market classification changed; normalize before comparison')
            for before, after in zip(parsed, parsed[1:]):
                if iso(after['period_start']) != iso(before['period_end']) + timedelta(days=1):
                    raise ValueError('annual periods are not consecutive; no quarterly-as-YoY comparison')
            if (as_of - iso(c['period_end'])).days > config['max_filing_age_days']:
                raise ValueError('annual financial data is stale')
            if c['sector'].casefold() in {x.casefold() for x in config['excluded_sectors']}:
                r['status'] = 'specialist_review'
                r['reasons'].append('bank/insurance/financial/REIT requires sector-specific model')
                continue
            valuation = None
            vraw = latest_values.get(ticker)
            try:
                if ticker in invalid_values:
                    raise ValueError('ambiguous or invalid valuation version')
                if vraw is None:
                    raise ValueError('missing valuation snapshot')
                if not source_url_valid(vraw.get('source_url')) or vraw.get('cap_scope') != 'total_company':
                    raise ValueError('valuation needs source_url and cap_scope=total_company (all share classes)')
                if iso(vraw['available_at']) < iso(vraw['snapshot_date']):
                    raise ValueError('valuation available_at precedes snapshot_date')
                if (as_of - iso(vraw['snapshot_date'])).days > config['max_valuation_age_days']:
                    raise ValueError('valuation snapshot is stale')
                cap = number(vraw.get('market_cap'), 'market_cap')
                scale = number(vraw.get('unit_scale'), 'unit_scale')
                if cap is None or cap <= 0 or scale is None or scale <= 0:
                    raise ValueError('valuation requires positive market_cap and unit_scale')
                fx = 1.0
                if not vraw.get('currency'):
                    raise ValueError('missing market-cap currency')
                if vraw['currency'] != c['currency']:
                    fx = number(vraw.get('fx_to_reporting'), 'fx_to_reporting')
                    fx_date = iso(vraw.get('fx_date', ''))
                    if fx is None or fx <= 0 or fx_date > as_of or (as_of - fx_date).days > config['max_fx_age_days'] or not source_url_valid(vraw.get('fx_source_url')):
                        raise ValueError('currency mismatch requires dated sourced positive FX rate')
                cap_reporting = cap * scale * fx
                if not math.isfinite(cap_reporting):
                    raise ValueError('valuation conversion overflow')
                valuation = {'market_cap_reporting': cap_reporting}
                r['valuation_source'] = dict(vraw)
            except (ValueError, TypeError) as exc:
                r['warnings'].append(str(exc))
            metrics, signals = compute_metrics(o, p, c, valuation)
            r['metrics'], r['signals'] = metrics, signals
            known = sum(x is not None for x in signals.values())
            passes = sum(x is True for x in signals.values())
            r['f_score_known'], r['f_score_lower_bound'] = known, passes
            r['f_score'] = passes if known == 9 else None
            r['metric_coverage'] = sum(metrics[k] is not None for fields in BLOCK_FIELDS.values() for k in fields) / sum(len(fields) for fields in BLOCK_FIELDS.values())
            if c.get('audit_opinion') != 'unqualified':
                r['warnings'].append('audit opinion is missing or not unqualified; manual review required')
            if metrics['core_profit_share'] is not None and metrics['core_profit_share'] < .7:
                r['warnings'].append('reported profit contains material non-core contribution')
            if metrics['fcf'] is not None and metrics['fcf'] < 0:
                r['warnings'].append('negative FCF; verify expansion capex versus maintenance capex')
            if metrics['revenue_cagr_2y'] is not None and abs(metrics['revenue_cagr_2y']) > .8:
                r['warnings'].append('extreme revenue growth; verify acquisitions and consolidation scope')
            # Missing or weak critical facts are never filled with zero.
            critical = ('roe', 'cash_conversion', 'net_debt_to_cfo')
            required_positive = ('net_income', 'net_income_parent', 'core_income_parent', 'operating_cash_flow', 'equity_parent')
            if any(c[k] is None for k in required_positive) or known < 9 or valuation is None or c.get('audit_opinion') != 'unqualified':
                r['reasons'].append('critical data, complete F-score, valuation or audit review is required')
                continue
            failures = []
            for field in required_positive:
                if c[field] is None or c[field] <= 0:
                    failures.append(f'{field} is not positive')
            if not failures and any(metrics[k] is None for k in critical):
                r['reasons'].append('critical derived ratio is undefined; manual data review required')
                continue
            if metrics['roe'] is not None and metrics['roe'] < config['min_roe']:
                failures.append('ROE below research threshold')
            if metrics['cash_conversion'] is not None and metrics['cash_conversion'] < config['min_cash_conversion']:
                failures.append('cash conversion below research threshold')
            if metrics['net_debt_to_cfo'] is not None and metrics['net_debt_to_cfo'] > config['max_net_debt_to_cfo']:
                failures.append('net debt / operating cash flow above research threshold')
            if passes < config['min_f_score']:
                failures.append('financial health F-score below research threshold')
            r['status'] = 'excluded' if failures else 'watchlist'
            r['reasons'].extend(failures)
        except (ValueError, TypeError) as exc:
            r['reasons'].append(str(exc))
    # Valid non-financial peers set comparison distributions, before research gates.
    groups = defaultdict(list)
    for r in results:
        if r['metrics']:
            groups[(r['market'], r['sector'])].append(r)
    for group in groups.values():
        for r in group:
            r['peer_count'] = len(group)
            for block, fields in BLOCK_FIELDS.items():
                components = {}
                for metric, higher in fields.items():
                    value = r['metrics'][metric]
                    peers = [x['metrics'][metric] for x in group if x['metrics'][metric] is not None]
                    if value is not None and len(peers) >= config['min_peer_count']:
                        components[metric] = percentile(value, peers, higher)
                complete = len(components) == len(fields)
                r['blocks'][block] = {'score': statistics.mean(components.values()) if complete else None, 'components': components}
            if all(b['score'] is not None for b in r['blocks'].values()):
                r['score'] = sum(config['weights'][b] * r['blocks'][b]['score'] for b in BLOCK_FIELDS)
            else:
                r['warnings'].append('incomplete metrics or too few same-market/sector peers; no composite score')
            if r['status'] == 'watchlist':
                if r['score'] is not None and r['score'] >= config['candidate_score']:
                    r['status'] = 'candidate'
                    r['reasons'].append('passed health/quality gates and peer-relative research priority threshold')
                else:
                    r['reasons'].append('health/quality gates passed; lower priority or insufficient peer comparison')
            for metric in ('roe', 'core_earnings_yield', 'revenue_cagr_2y'):
                values = [x['metrics'][metric] for x in group if x['metrics'][metric] is not None]
                if len(values) >= config['min_peer_count'] and r['metrics'][metric] is not None:
                    median = statistics.median(values)
                    mad = statistics.median(abs(v - median) for v in values)
                    if mad > 0 and abs(r['metrics'][metric] - median) / (1.4826 * mad) > 5:
                        r['warnings'].append(f'{metric}: peer outlier; verify source (MAD rule, not ML prediction)')
            r['next_questions'] = [
                '现金流改善能否持续？核对回款、应收和预收变化。',
                '扣非口径与归母利润差额来自什么？核对附注。',
                '增长来自同店/销量/价格还是并购和合并范围变化？',
                '相对低估值是否反映客户集中、治理或行业下行？',
            ]
    # Rank is only meaningful inside a comparable cohort; no global A/H score rank.
    results.sort(key=lambda r: (r['market'], r['sector'], -(r['score'] if r['score'] is not None else -1), r['ticker']))
    return {'as_of': as_of.isoformat(), 'config': config, 'companies': results,
            'audit': audit, 'valuation_audit': value_audit, 'normalized_statements': clean_rows,
            'method': 'annual fundamental research priority; no fitted return model',
            'limitations': ['annual data only; verify latest interim filings before research decisions',
                            'net_income is reported consolidated profit, an A/H adaptation of original F-score',
                            'core_income_parent requires analyst/provider harmonization across A/H',
                            'peer scores depend on supplied universe; no claim of validated investment performance']}
