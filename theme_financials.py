"""Explainable research styles for business matches; no return model or orders."""
from __future__ import annotations

import copy
import math
from collections import defaultdict

from engine import difference, divide, iso, parse_statement, screen, validate_config, visible_versions

DEFAULT_STYLE_CONFIG = {
    'quality_min_roe': .12,
    'quality_min_cash_conversion': 1.0,
    'quality_min_revenue_cagr': .10,
    'quality_min_core_profit_cagr': .10,
    'quality_min_core_profit_share': .80,
    'quality_max_net_debt_to_cfo': 2.0,
    'value_min_percentile': 75.0,
    'value_min_peers': 5,
    'value_max_net_debt_to_cfo': 4.0,
    'improvement_min_signals': 2,
    'improvement_min_margin_change': .005,
}
STYLE_NAMES = ('quality_growth', 'relative_value', 'operating_improvement',
               'loss_watchlist', 'unclassified')


def validate_style_config(supplied=None):
    cfg = dict(DEFAULT_STYLE_CONFIG)
    if supplied is not None:
        if not isinstance(supplied, dict) or set(supplied) - set(cfg):
            raise ValueError('unknown or invalid style configuration')
        cfg.update(supplied)
    for key, value in cfg.items():
        try:
            valid = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid or value < 0:
            raise ValueError(f'invalid style setting: {key}')
    if type(cfg['value_min_peers']) is not int or cfg['value_min_peers'] < 5:
        raise ValueError('value_min_peers must be an integer >= 5')
    if type(cfg['improvement_min_signals']) is not int or not 1 <= cfg['improvement_min_signals'] <= 3:
        raise ValueError('improvement_min_signals must be 1, 2 or 3')
    if cfg['value_min_percentile'] > 100:
        raise ValueError('value_min_percentile must be <= 100')
    return cfg


def _check(metric, actual, op, threshold):
    passed = None if actual is None else {
        '>=': lambda: actual >= threshold, '<=': lambda: actual <= threshold,
        '>': lambda: actual > threshold,
    }[op]()
    return dict(metric=metric, actual=actual, operator=op, threshold=threshold, passed=passed)


def _all(checks):
    return all(c['passed'] is True for c in checks)


def attach_fundamentals(discovery, statement_rows=None, valuation_rows=None, *,
                        config=None, screen_config=None):
    """Classify matches using the entire supplied financial peer universe.

    A missing valuation does not block quality/growth or operating analysis.
    Relative value requires its own valid, current valuation and peer coverage.
    A reported loss is a risk flag even when other data require manual review.
    """
    cfg = validate_style_config(config)
    base = validate_config(screen_config)
    result = copy.deepcopy(discovery)
    cutoff = iso(result['as_of'])
    statements, values = list(statement_rows or []), list(valuation_rows or [])
    # Disable only valuation scoring for the operating assessment. The engine's
    # dates, audit, critical facts and continuity checks still apply unchanged.
    operating_cfg = dict(base, weights={'quality': .5, 'value': 0., 'growth': .3, 'balance': .2})
    value_cfg = dict(base, weights={'quality': .4, 'value': .25, 'growth': .2, 'balance': .15},
                     min_peer_count=max(base['min_peer_count'], cfg['value_min_peers']))
    op_report = screen(statements, values, cutoff, operating_cfg)
    value_report = screen(statements, values, cutoff, value_cfg)
    operating = {x['ticker']: x for x in op_report['companies']}
    valuation = {x['ticker']: x for x in value_report['companies']}
    histories = defaultdict(list)
    for row in op_report['normalized_statements']:
        histories[row['ticker']].append(row)
    # Separately retain a sourced reported loss with <3 years of history.
    visible, version_audit = visible_versions(statements, cutoff, 'period_end')
    invalid = {x['ticker'] for x in version_audit if x['reason'] != 'not_yet_available'}
    latest_raw = {}
    for row in visible:
        if row['ticker'] not in latest_raw or row['period_end'] > latest_raw[row['ticker']]['period_end']:
            latest_raw[row['ticker']] = row
    lists = {key: [] for key in STYLE_NAMES}
    for company in result['companies']:
        ticker = company['ticker']
        assessment, price_assessment = operating.get(ticker), valuation.get(ticker)
        financial = {'status': 'unclassified', 'styles': [], 'loss_making': None,
                     'checks': {}, 'metrics': {}, 'sources': [], 'risks': [], 'reasons': []}
        company['financial'] = financial
        if assessment is None:
            financial['reasons'].append('No financial statements supplied for this company.')
        elif any(assessment.get(k) != company.get(k) for k in ('market', 'sector')):
            financial['status'] = 'data_review'
            financial['reasons'].append('Company directory and financial market/sector disagree; no financial join.')
        else:
            financial.update(status=assessment['status'], metrics=assessment['metrics'],
                             sources=assessment['sources'], reasons=list(assessment['reasons']),
                             risks=list(assessment['warnings']),
                             period_end=assessment.get('period_end'), currency=assessment.get('currency'),
                             valuation_status=price_assessment['status'],
                             value_block=price_assessment['blocks'].get('value'),
                             valuation_source=price_assessment.get('valuation_source'),
                             peer_count=price_assessment['peer_count'])
            current = None
            if ticker not in invalid and ticker in latest_raw:
                try:
                    current = parse_statement(latest_raw[ticker])
                except (ValueError, TypeError):
                    pass
            if current is not None and current['net_income_parent'] is not None:
                financial['loss_making'] = current['net_income_parent'] < 0
                financial['reported_parent_profit'] = current['net_income_parent']
                financial['loss_source'] = {k: current[k] for k in
                                            ('period_end', 'available_at', 'source_url', 'filing_id')}
                if financial['loss_making']:
                    financial['styles'].append('loss_watchlist')
                    financial['risks'].append('Reported parent profit is negative; retained for manual research.')
            valid = assessment['status'] in ('candidate', 'watchlist', 'excluded')
            if valid:
                history = sorted(histories[ticker], key=lambda x: x['period_end'])
                prior, current = history[-2:]
                metrics = assessment['metrics']
                positive_checks = [_check(field, current[field], '>', 0) for field in
                                   ('net_income', 'net_income_parent', 'core_income_parent',
                                    'operating_cash_flow', 'equity_parent')]
                quality = positive_checks + [
                    _check('roe', metrics['roe'], '>=', cfg['quality_min_roe']),
                    _check('cash_conversion', metrics['cash_conversion'], '>=', cfg['quality_min_cash_conversion']),
                    _check('revenue_cagr_2y', metrics['revenue_cagr_2y'], '>=', cfg['quality_min_revenue_cagr']),
                    _check('core_profit_cagr_2y', metrics['core_profit_cagr_2y'], '>=', cfg['quality_min_core_profit_cagr']),
                    _check('core_profit_share', metrics['core_profit_share'], '>=', cfg['quality_min_core_profit_share']),
                    _check('net_debt_to_cfo', metrics['net_debt_to_cfo'], '<=', cfg['quality_max_net_debt_to_cfo']),
                ]
                financial['checks']['quality_growth'] = quality
                if _all(quality):
                    financial['styles'].append('quality_growth')
                vmetrics = price_assessment['metrics']
                value = positive_checks + [
                    _check('value_percentile', (price_assessment['blocks'].get('value') or {}).get('score'), '>=', cfg['value_min_percentile']),
                    _check('valuation_valid', price_assessment['status'] in ('candidate', 'watchlist', 'excluded'), '>=', True),
                    _check('same_market_sector_peers', price_assessment['peer_count'], '>=', cfg['value_min_peers']),
                    _check('core_earnings_yield', vmetrics.get('core_earnings_yield'), '>', 0),
                    _check('fcf_yield', vmetrics.get('fcf_yield'), '>', 0),
                    _check('net_debt_to_cfo', metrics['net_debt_to_cfo'], '<=', cfg['value_max_net_debt_to_cfo']),
                ]
                financial['checks']['relative_value'] = value
                if _all(value):
                    financial['styles'].append('relative_value')
                # Absolute profit change handles negative bases / loss narrowing.
                margins = [divide(x['net_income_parent'], x['revenue']) for x in (prior, current)]
                margin_change = difference(margins[1], margins[0])
                changes = {field: difference(current[field], prior[field])
                           for field in ('net_income_parent', 'operating_cash_flow')}
                improvements = [
                    _check('parent_profit_change', changes['net_income_parent'], '>', 0),
                    _check('parent_net_margin_change', margin_change, '>=', cfg['improvement_min_margin_change']),
                    _check('operating_cash_flow_change', changes['operating_cash_flow'], '>', 0),
                ]
                financial['checks']['operating_improvement'] = improvements
                financial['improvement_signal_count'] = sum(c['passed'] is True for c in improvements)
                if financial['improvement_signal_count'] >= cfg['improvement_min_signals']:
                    if financial['loss_making']:
                        financial['risks'].append('Operating improvement signals present, but losses persist; kept only in loss watchlist.')
                    else:
                        financial['styles'].append('operating_improvement')
        # Financial strength cannot turn an unverified or discontinued theme
        # association into a current-business research recommendation.
        financial['financial_style_candidates'] = [style for style in financial['styles']
                                                    if style != 'loss_watchlist']
        financial['business_status'] = company.get('business_status', 'not_assessed')
        if financial['business_status'] != 'current_business':
            held = [style for style in financial['styles'] if style != 'loss_watchlist']
            financial['styles'] = [style for style in financial['styles'] if style == 'loss_watchlist']
            if held:
                financial['reasons'].append('Financial thresholds pass, but current theme business is not established; styles withheld pending business review.')
        if not financial['styles']:
            financial['styles'].append('unclassified')
        for style in financial['styles']:
            lists[style].append(ticker)
    result['style_lists'] = lists
    result['style_config'] = cfg
    result['financial_audit'] = {
        'supplied_statement_companies': len(operating),
        'company_statuses': {ticker: {'market': row['market'], 'sector': row['sector'],
                                     'operating_status': row['status'],
                                     'valuation_status': valuation[ticker]['status']}
                             for ticker, row in sorted(operating.items())},
        'matched_companies_with_statements': sum(c['ticker'] in operating for c in result['companies']),
        'statement_versions': op_report['audit'], 'valuation_versions': value_report['valuation_audit'],
        'operating_config': operating_cfg, 'value_config': value_cfg,
        'peer_universe': 'All supplied financially eligible companies, not just theme matches.',
        'limitations': ['Research thresholds are configurable hypotheses, not calibrated return predictors.',
                        'Styles can overlap; relevance does not establish revenue exposure or investment quality.',
                        'Annual statements require source reconciliation and comparable accounting; interim changes need review.'],
    }
    return result
