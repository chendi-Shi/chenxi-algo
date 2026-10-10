"""Verify frozen Li Ning source facts, accounting identities and original PDF rows.

Offline: python validation/enterprise_hk.py
Original PDF verification: python validation/enterprise_hk.py --check-pdfs
Fetch missing public source PDFs first: add --download (requires pypdf for checking).
No financial inputs or screening approval flags are changed by this script.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def verify_frozen(payload):
    checks = []

    def check(name, observed, expected):
        checks.append({'name': name, 'observed': observed, 'expected': expected,
                       'passed': observed == expected})

    facts = payload['facts']
    keys = [(f['fiscal_year'], f['field']) for f in facts]
    check('no_duplicate_field_year', len(keys), len(set(keys)))
    lookup = dict(zip(keys, facts))
    get = lambda year, field: lookup[(year, field)]['value_yuan']
    for f in facts:
        name = f"{f['fiscal_year']}:{f['field']}"
        check(name + ':source_url', f['url'], payload['sources'][f['source_id']]['url'])
        check(name + ':page_offset', f['pdf_page'] - f['report_page'], 2)
        if f['value_yuan'] is None:
            check(name + ':only_explicit_core_unknown', f['field'], 'core_income_parent')
            continue
        check(name + ':unit_scale', f['scale_to_yuan'], 1000)
        check(name + ':unit', f['source_unit'], 'RMB thousand')
        check(name + ':yuan', f['value_yuan'], f['source_value'] * f['scale_to_yuan'])
        check(name + ':availability', f['available_at'], payload['sources'][f['source_id']]['publication_date'])
        if 'source_column_years' in f:
            index = f['source_column_years'].index(f['fiscal_year'])
            check(name + ':year_column', f['source_value'], f['source_column_values'][index])
    for year in (2022, 2023, 2024):
        v = lambda field: get(year, field)
        check(f'{year}:balance_sheet', v('total_assets'), v('equity_consolidated') + v('total_liabilities'))
        check(f'{year}:gross_profit', v('gross_profit'), v('revenue') - v('cost_of_goods_sold'))
        check(f'{year}:profit_before_tax', v('profit_before_tax'), v('operating_profit') + v('finance_income') - v('gross_finance_expense') + v('equity_method_profit'))
        check(f'{year}:profit_after_tax', v('net_income_consolidated'), v('profit_before_tax') - v('income_tax_expense'))
        check(f'{year}:cash_components', v('cash_and_cash_equivalents'), v('cash_at_bank_and_on_hand') + v('deposit_original_maturity_under_three_months'))
        check(f'{year}:finance_expense_components', v('gross_finance_expense'),
              v('borrowing_interest_cost') + v('lease_interest_cost') + v('royalty_unwinding_cost') + v('other_finance_expense') + payload['finance_expense_reconciliation'][str(year)]['fx_loss_yuan'])
        check(f'{year}:bank_loan_flow_reconciles', v('bank_borrowings_received') - v('bank_borrowings_repaid'), v('bank_borrowings_at_year_end'))
        check(f'{year}:interest_paid_matches_accrued', v('borrowing_interest_paid'), v('borrowing_interest_cost'))
        check(f'{year}:issuance_occurred', v('share_option_issuance_proceeds') > 0, True)
        check(f'{year}:core_profit_preserved_unknown', v('core_income_parent'), None)
        check(f'{year}:zero_bank_debt_does_not_zero_interest', v('borrowing_interest_cost') > 0, True)
    for f in payload['derived_facts']:
        expected = sum(get(f['fiscal_year'], t['field']) * t['coefficient'] for t in f['formula_terms'])
        check(f"{f['fiscal_year']}:{f['field']}:formula", f['value_yuan'], expected)
    adjusted = payload['specific_adjustment']
    check('2023:single_gain_adjustment_reconciles', adjusted['company_adjusted_parent_profit_yuan'],
          adjusted['reported_parent_profit_yuan'] - adjusted['equity_method_oneoff_gain_yuan'])
    check('2022:parent_vs_total_equity', get(2022, 'equity_consolidated') - get(2022, 'equity_parent'), 2498000)
    check('2022:parent_vs_total_profit', get(2022, 'net_income_consolidated') - get(2022, 'net_income_parent'), -63000)
    check('2023:parent_vs_total_profit', get(2023, 'net_income_consolidated') - get(2023, 'net_income_parent'), 52000)
    check('automatic_engine_release_disabled', payload['release_to_engine'], False)
    old = json.loads((ROOT / 'hk_coverage.json').read_text(encoding='utf-8'))
    for year in (2022, 2023, 2024):
        for field in ('revenue', 'net_income_parent', 'total_assets', 'equity_parent', 'operating_cash_flow'):
            check(f'{year}:{field}:aggregator_vs_original', old['years'][str(year)][field]['value'], get(year, field))
    return checks


def compact(text):
    return re.sub(r'\s+', '', text)


def verify_pdfs(payload, download=False):
    from pypdf import PdfReader
    checks = []
    readers = {}
    for key, source in payload['sources'].items():
        path = ROOT / source['local_path']
        if not path.is_file() and download:
            content = urlopen(source['url'], timeout=60).read()
            if hashlib.sha256(content).hexdigest() != source['sha256']:
                raise ValueError(f'{key}: downloaded source hash differs from verified original')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        if not path.is_file():
            checks.append({'name':key + ':pdf_present', 'passed':False})
            continue
        match = hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
        checks.append({'name':key + ':pdf_sha256', 'passed':match})
        if match:
            readers[key] = PdfReader(path)
    page_cache = {}
    for fact in payload['facts']:
        if fact['value_yuan'] is None:
            continue  # Missing standardized earnings is a semantic conclusion, not a numeric row.
        key = fact['source_id']
        if key not in readers:
            continue
        page_key = key, fact['pdf_page']
        if page_key not in page_cache:
            page_cache[page_key] = readers[key].pages[fact['pdf_page'] - 1].extract_text()
        page = page_cache[page_key]
        if 'text_assertions' in fact:
            matched = all(compact(text) in compact(page) for text in fact['text_assertions'])
        else:
            def token(value):
                if value == 0:
                    return '–'
                rendered = f'{abs(value):,}'
                return '(' + rendered + ')' if fact['pdf_sign'] < 0 else rendered
            ending = ''.join(token(v) for v in fact['source_column_values'])
            label = compact(fact['table_label'])
            matched = any(label in compact(line) and compact(line).endswith(ending)
                          for line in page.splitlines())
            matched = matched and all(str(y) + '年' in compact(page) for y in fact['source_column_years'])
        checks.append({'name':f"{fact['fiscal_year']}:{fact['field']}:original_pdf_row", 'passed':matched,
                       'source_id':key,'pdf_page':fact['pdf_page']})
    # Specific adjustment is reported in yuan in prose, not RMB thousands.
    source = payload['specific_adjustment']
    if source['source_id'] in readers:
        page = compact(readers[source['source_id']].pages[source['pdf_page'] - 1].extract_text())
        checks.append({'name':'2023:company_specific_adjustment:pdf_prose',
                       'passed':'3,046,453,000' in page and '一次性收益' in page})
    gain_source = source['oneoff_gain_source']
    if gain_source['source_id'] in readers:
        page = compact(readers[gain_source['source_id']].pages[gain_source['pdf_page'] - 1].extract_text())
        checks.append({'name':'2023:equity_method_oneoff_gain:pdf_prose',
                       'passed':'140,457,000' in page and '一次性與經營無關的收益' in page})
    return checks


def mutation_checks(payload):
    """Ensure wrong units, zeroed leases, filled core profit and wrong year fail."""
    results=[]
    changes=[('wrong_scale','revenue','scale_to_yuan',1),
             ('zero_lease','lease_liabilities_current','value_yuan',0),
             ('filled_core','core_income_parent','value_yuan',123),
             ('wrong_year_column','revenue','source_column_values',[1,2])]
    for name, field, key, value in changes:
        broken=deepcopy(payload)
        fact=next(f for f in broken['facts'] if f['fiscal_year']==2024 and f['field']==field)
        fact[key]=value
        try:
            rejected=any(not check['passed'] for check in verify_frozen(broken))
        except (KeyError, TypeError, ValueError):
            rejected=True
        results.append({'name':name,'passed':rejected})
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-pdfs', action='store_true')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    if args.download and not args.check_pdfs:
        parser.error('--download requires --check-pdfs')
    path=ROOT/'enterprise_hk_facts.json'
    payload=json.loads(path.read_text(encoding='utf-8'))
    checks=verify_frozen(payload)
    pdf_checks=verify_pdfs(payload,args.download) if args.check_pdfs else []
    adversarial=mutation_checks(payload)
    all_checks=checks+pdf_checks+adversarial
    report={'company':'02331.HK','facts_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'frozen_fact_count':len(payload['facts']),'numeric_fact_count':sum(f['value_yuan'] is not None for f in payload['facts']),
            'derived_fact_count':len(payload['derived_facts']),
            'frozen_checks':checks,'original_pdf_checks':pdf_checks,
            'original_pdf_verification_requested':args.check_pdfs,
            'negative_control_checks':adversarial,
            'passed':all(c['passed'] for c in all_checks),
            'remaining_blockers':payload['remaining_blockers'],
            'automatically_released_to_screening':False}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'checks':len(all_checks),
                      'numeric_facts':report['numeric_fact_count'],'pdf_checks':len(pdf_checks),
                      'failures':[c for c in all_checks if not c['passed']],
                      'automatically_released_to_screening':False},ensure_ascii=True))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
