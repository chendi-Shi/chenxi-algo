"""Deterministic synthetic inputs. These identifiers are not listed stocks."""
import copy
import csv
import json
import random
from pathlib import Path


def create_demo(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261009)
    statements, valuations, documents = [], [], []
    for market in ('A', 'HK'):
        for sector in ('Consumer', 'Industrials'):
            for index in range(14):
                ticker = f'DEMO-{market}-{sector[0]}{index:02d}'
                currency = 'CNY' if market == 'A' else 'HKD'
                for offset, year in enumerate((2023, 2024, 2025)):
                    assets = (1000 + 75 * index) * 1.08 ** offset
                    revenue = (850 + 40 * index) * (1.04 + .013 * index) ** offset
                    margin = .22 + .002 * index + .01 * offset
                    income = revenue * (.07 + .003 * index + .006 * offset)
                    debt = assets * (.28 - .04 * offset)
                    row = {
                        'ticker': ticker, 'name': f'SYNTHETIC {market} {sector} {index}',
                        'market': market, 'sector': sector, 'currency': currency,
                        'unit_scale': 1000000, 'scope': 'consolidated',
                        'period_start': f'{year}-01-01', 'period_end': f'{year}-12-31',
                        'available_at': f'{year + 1}-04-20', 'revision_id': 0,
                        'filing_id': f'{ticker}-{year}-v0',
                        'source_url': f'https://example.invalid/synthetic/{ticker}/{year}',
                        'net_income': round(income, 4),
                        'net_income_parent': round(income * .9, 4),
                        'core_income_parent': round(income * .9 * (.75 + .016 * index), 4),
                        'operating_cash_flow': round(income * (.75 + .035 * index + .03 * offset), 4),
                        'total_assets': round(assets, 4), 'equity_parent': round(assets * .6, 4),
                        'current_assets': round(assets * (.4 + .025 * offset), 4),
                        'current_liabilities': round(assets * (.28 - .015 * offset), 4),
                        'long_term_debt': round(debt * .7, 4),
                        'total_debt': round(debt, 4), 'cash': round(assets * (.08 + .01 * offset), 4),
                        'revenue': round(revenue, 4), 'cogs': round(revenue * (1 - margin), 4),
                        'capex': round(income * (.35 + rng.random() * .1), 4),
                        'ebit': round(income * 1.35, 4), 'interest_expense': round(debt * .04, 4),
                        'equity_issued': 0, 'audit_opinion': 'unqualified',
                    }
                    statements.append(row)
                valuations.append({
                    'ticker': ticker, 'snapshot_date': '2026-10-08',
                    'available_at': '2026-10-08', 'revision_id': 0,
                    'market_cap': round(row['core_income_parent'] * (18 - .45 * index), 4),
                    'currency': currency, 'unit_scale': 1000000, 'cap_scope': 'total_company',
                    'fx_to_reporting': '', 'fx_date': '', 'fx_source_url': '',
                    'source_url': f'https://example.invalid/synthetic/{ticker}/valuation',
                })
                documents.append({
                    'document_id': f'{ticker}-report-extract', 'ticker': ticker,
                    'available_at': '2026-04-20', 'source_type': 'annual_report', 'page': 88,
                    'source_url': row['source_url'],
                    'text': '【合成演示文本】经营现金流改善。主要客户集中度需核对；本年度无重大关联交易。资本开支包含扩产，产能利用率仍需跟踪。',
                })
    # Review cases exercise distinct branches; none represents a real company.
    template = [r for r in statements if r['ticker'] == 'DEMO-A-C08']
    value_template = next(v for v in valuations if v['ticker'] == 'DEMO-A-C08')
    for tag in ('MISSING', 'BANK', 'BADUNIT', 'FX', 'CONFLICT', 'WEAK'):
        ticker = f'DEMO-{tag}'
        versions = copy.deepcopy(template)
        for r in versions:
            r['ticker'], r['name'] = ticker, f'SYNTHETIC REVIEW {tag}'
            r['filing_id'] = f"{ticker}-{r['period_end']}"
            r['source_url'] = f'https://example.invalid/synthetic/{ticker}/annual'
            if tag == 'BANK':
                r['sector'] = 'Banks'
        if tag == 'MISSING':
            versions[-1]['operating_cash_flow'] = ''
        if tag == 'BADUNIT':
            versions[-1]['capex'] = -10
        if tag == 'WEAK':
            versions[-1]['operating_cash_flow'] = versions[-1]['net_income'] * .2
        statements.extend(versions)
        v = copy.deepcopy(value_template)
        v['ticker'] = ticker
        if tag == 'FX':
            v['currency'], v['fx_to_reporting'] = 'HKD', ''
        valuations.append(v)
        if tag == 'CONFLICT':
            duplicate = copy.deepcopy(versions[-1])
            duplicate['net_income'] *= 2
            statements.append(duplicate)
    # A later restatement must not overwrite data visible at the demo cutoff.
    future = copy.deepcopy(template[-1])
    future['available_at'], future['revision_id'] = '2026-11-01', 1
    future['net_income'] = 999999
    statements.append(future)
    documents.append({**documents[0], 'document_id': 'future-demo-evidence',
                      'available_at': '2026-11-01', 'text': '【未来合成演示文本】审计意见变更。'})
    for name, rows in (('statements.csv', statements), ('valuations.csv', valuations)):
        with (directory / name).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (directory / 'documents.jsonl').open('w', encoding='utf-8') as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + '\n')
    return {key: directory / filename for key, filename in {
        'statements': 'statements.csv', 'valuations': 'valuations.csv',
        'documents': 'documents.jsonl'}.items()}
