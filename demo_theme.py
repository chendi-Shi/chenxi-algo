"""Generate entirely fictional theme + financial fixtures for learning the CLI."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def create_theme_demo(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    companies, documents, statements, valuations = [], [], [], []
    for i in range(8):
        ticker = f'DEMO{i + 1:03d}'
        companies.append(dict(ticker=ticker, name=f'虚构机器人公司{i + 1}', market='A',
                              sector='Industrial Robotics', scope='manufacturing', universe_as_of='2026-01-01'))
        documents.append(dict(document_id=f'demo-{i}', ticker=ticker, available_at='2026-04-01',
                              source_url=f'https://example.invalid/demo/{ticker}', source_type='annual_report',
                              page=1, text='本公司主要从事工业机器人的研发、生产和销售。本段为虚构演示数据。'))
        for year, index in ((2023, 0), (2024, 1), (2025, 2)):
            profit = [20, 28, 40][index]
            if i == 6:
                profit = [-30, -20, -10][index]
            if i == 7:
                profit = [10, 10, 10][index]
            statements.append(dict(ticker=ticker, name=f'虚构机器人公司{i + 1}', market='A',
                sector='Industrial Robotics', currency='CNY', scope='consolidated',
                period_start=f'{year}-01-01', period_end=f'{year}-12-31', available_at=f'{year+1}-03-31',
                revision_id=0, source_url=f'https://example.invalid/demo/{ticker}/{year}', filing_id=f'{ticker}-{year}',
                unit_scale=1000000, net_income=profit, net_income_parent=profit, core_income_parent=profit * .9,
                operating_cash_flow=[30, 38, 50][index] if i != 7 else 15, total_assets=[220, 240, 260][index],
                equity_parent=[100, 120, 140][index], current_assets=[90, 110, 130][index], current_liabilities=40,
                long_term_debt=20, total_debt=30, cash=25, revenue=[100, 130, 170][index],
                cogs=[60, 75, 95][index], capex=10, ebit=profit + 5, interest_expense=2,
                equity_issued=0, audit_opinion='unqualified'))
        valuations.append(dict(ticker=ticker, snapshot_date='2026-10-09', available_at='2026-10-09',
            revision_id=0, market_cap=200 if i == 0 else 1000 + i * 100, unit_scale=1000000, currency='CNY',
            cap_scope='total_company', source_url=f'https://example.invalid/demo/cap/{ticker}'))
    (directory / 'companies.json').write_text(json.dumps(companies, ensure_ascii=False, indent=2), encoding='utf-8')
    (directory / 'documents.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in documents), encoding='utf-8')
    for filename, rows in (('statements.csv', statements), ('valuations.csv', valuations)):
        with (directory / filename).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return directory


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/theme_demo_inputs'))
    args = parser.parse_args()
    print(create_theme_demo(args.output).resolve())
