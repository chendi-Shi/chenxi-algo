"""One-command CLI, audit database and human-readable research output."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from demo import create_demo
from engine import config_from, iso, load_csv, screen
from evidence import process_documents, search_evidence

ROOT = Path(__file__).resolve().parent
STATUS_CN = {'candidate': '优先研究', 'watchlist': '观察清单', 'data_review': '数据待核验',
             'specialist_review': '行业专用分析', 'excluded': '未通过初筛'}
BLOCK_CN = {'quality': '盈利质量', 'value': '相对估值', 'growth': '增长持续性', 'balance': '资产负债表'}


def dump_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)


def load_search_space(path):
    """Read a finite JSON object without accepting nonstandard numeric constants."""
    def reject_constant(value):
        raise ValueError(f'search space contains non-finite JSON number: {value}')

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f'search space contains non-finite JSON number: {value}')
        return parsed

    space = json.loads(path.read_text(encoding='utf-8-sig'),
                       parse_constant=reject_constant, parse_float=finite_float)
    if not isinstance(space, dict):
        raise ValueError('--search-space must contain a JSON object')
    return space


def display(value, percentage=False):
    if value is None:
        return '未知'
    return f'{value:.1%}' if percentage else f'{value:.2f}'


def save_database(path, run_id, manifest, results, inputs, document_path=None):
    with sqlite3.connect(path) as con:
        con.executescript('''
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, as_of TEXT NOT NULL, manifest_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS raw_records (
                run_id TEXT NOT NULL, dataset TEXT NOT NULL, row_number INTEGER NOT NULL,
                payload_json TEXT NOT NULL, PRIMARY KEY (run_id, dataset, row_number));
            CREATE TABLE IF NOT EXISTS statements (
                run_id TEXT NOT NULL, ticker TEXT NOT NULL, period_end TEXT NOT NULL,
                available_at TEXT NOT NULL, filing_id TEXT NOT NULL, payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, ticker, period_end));
            CREATE TABLE IF NOT EXISTS companies (
                run_id TEXT NOT NULL, ticker TEXT NOT NULL, market TEXT, sector TEXT,
                status TEXT NOT NULL, score REAL, payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, ticker));
            CREATE TABLE IF NOT EXISTS evidence (
                run_id TEXT NOT NULL, document_id TEXT NOT NULL, ticker TEXT NOT NULL,
                source_url TEXT NOT NULL, available_at TEXT NOT NULL, payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, document_id));
            CREATE INDEX IF NOT EXISTS company_status ON companies (run_id, status, market, sector);
            CREATE INDEX IF NOT EXISTS evidence_ticker ON evidence (run_id, ticker);
        ''')
        con.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?)',
                    (run_id, results['as_of'], dump_json(manifest)))
        for dataset, rows in inputs.items():
            con.executemany('INSERT OR REPLACE INTO raw_records VALUES (?,?,?,?)',
                            ((run_id, dataset, i, dump_json(row)) for i, row in enumerate(rows, 1)))
        if document_path:
            # Preserve raw source lines, including rejected records, without executing them.
            lines = Path(document_path).read_text(encoding='utf-8-sig').splitlines()
            con.executemany('INSERT OR REPLACE INTO raw_records VALUES (?,?,?,?)',
                            ((run_id, 'documents', i, dump_json({'raw_line': line})) for i, line in enumerate(lines, 1)))
        for row in results['normalized_statements']:
            con.execute('INSERT OR REPLACE INTO statements VALUES (?,?,?,?,?,?)',
                        (run_id, row['ticker'], row['period_end'], row['available_at'], row['filing_id'], dump_json(row)))
        for row in results['companies']:
            con.execute('INSERT OR REPLACE INTO companies VALUES (?,?,?,?,?,?,?)',
                        (run_id, row['ticker'], row['market'], row['sector'], row['status'], row['score'], dump_json(row)))
        for row in results['documents']['evidence']:
            con.execute('INSERT OR REPLACE INTO evidence VALUES (?,?,?,?,?,?)',
                        (run_id, row['document_id'], row['ticker'], row['source_url'], row['available_at'], dump_json(row)))


def export_csv(path, companies):
    fields = ['ticker', 'name', 'market', 'sector', 'status', 'score', 'f_score',
              'f_score_lower_bound', 'f_score_known', 'peer_count', 'roe',
              'cash_conversion', 'core_earnings_yield', 'fcf_yield',
              'revenue_cagr_2y', 'core_profit_cagr_2y', 'net_debt_to_cfo',
              'interest_cover', 'reasons', 'warnings']
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for company in companies:
            row = {key: company.get(key, '') for key in fields}
            row.update({key: company['metrics'].get(key) for key in fields if key in company['metrics']})
            row['reasons'] = ' | '.join(company['reasons'])
            row['warnings'] = ' | '.join(company['warnings'])
            # Prevent spreadsheet programs from interpreting imported text as formulas.
            for key, value in row.items():
                if isinstance(value, str) and (value.lstrip().startswith(('=', '+', '-', '@')) or value.startswith(('\t', '\r', '\n'))):
                    row[key] = "'" + value
            writer.writerow(row)


def report_html(results, synthetic=False):
    escape = lambda value: html.escape(str(value), quote=True)
    counts = Counter(x['status'] for x in results['companies'])
    banner = '合成数据演示：所有公司、财务数字和文本均为虚构。' if synthetic else '使用用户提供的标准化数据；来源真实性与会计口径仍需研究员核验。'
    intro = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
    <title>基本面研究筛选</title><style>
    body{{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f5f6fa;color:#183042;max-width:1180px;margin:32px auto;padding:0 22px;line-height:1.65}}
    h1,h2{{color:#12374c}} .banner{{background:#fff1ce;padding:16px;border-radius:8px}}
    table{{border-collapse:collapse;width:100%;background:white;font-size:14px}} th,td{{padding:10px;border-bottom:1px solid #dce3eb;text-align:left}}
    th{{background:#e6eef5}} details{{background:white;margin:16px 0;padding:16px;border-radius:8px}} summary{{cursor:pointer;font-weight:600}}
    .muted{{color:#5b6c7e}} .chips{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}} .chip{{background:white;padding:12px 20px;border-radius:8px}}
    a{{color:#176491}} code{{background:#eaf0f5;padding:2px 4px}} blockquote{{border-left:3px solid #b9cfe1;padding-left:14px;white-space:pre-wrap}}
    </style><h1>基本面研究筛选 · A 股 / 港股</h1><p>数据截止：{escape(results['as_of'])} · 优先级只在同市场、同行业内比较</p>
    <div class="banner">{escape(banner)} 分数用于分配研究时间，尚未验证投资效果。</div>
    <div class="chips">'''
    for status in STATUS_CN:
        intro += f'<div class="chip">{STATUS_CN[status]} <b>{counts[status]}</b></div>'
    weight_text = ' · '.join(f'{BLOCK_CN[key]} {weight:.0%}' for key, weight in results['config']['weights'].items())
    intro += f'</div><p class="muted">{escape(weight_text)}。这是可修改的研究偏好，详见本次 JSON 配置。未知值保留为空；银行、保险及 REIT 使用行业专用分析。</p>'
    search = results.get('configuration_search')
    if search is not None:
        intro += '<h2>筛选参数搜索</h2>'
        intro += f'<p>{escape(search.get("market"))} / {escape(search.get("sector"))} · 方法 {escape(search.get("method"))} · 目标名单规模 {escape(search.get("target_count"))} · 状态 {escape(search.get("status"))} · 已评价配置 {escape(search.get("evaluations"))}</p>'
        intro += '<p class="muted">搜索寻找符合研究偏好和目标名单规模的门槛组合，不验证投资效果。本页主筛选结果仍使用原始配置；替代配置和候选名单见 search.json。</p>'
        best = search.get('best')
        if best:
            target_met = best.get('target_met', best.get('candidate_count') == search.get('target_count'))
            intro += '<p><b>最佳已评价配置：</b>' + ('达到目标规模' if target_met else '尚未达到目标规模') + ' · 候选数量 ' + escape(best.get('candidate_count')) + '</p>'
            intro += '<pre><code>' + escape(dump_json(best.get('config', {}))) + '</code></pre>'
            candidates = best.get('candidates', [])
            labels = [f'{item.get("ticker", "")} {item.get("name", "")}'.strip()
                      if isinstance(item, dict) else str(item) for item in candidates]
            intro += '<p><b>对应候选名单：</b>' + (escape('、'.join(labels)) if labels else '无') + '</p>'
        else:
            intro += '<p>当前搜索没有可展示的配置，详见 search.json 的限制与原因。</p>'
        if search.get('limitations'):
            intro += '<ul>' + ''.join(f'<li>{escape(item)}</li>' for item in search['limitations']) + '</ul>'
    intro += '<table><tr><th>公司</th><th>市场 / 行业</th><th>状态</th><th>研究分数</th><th>F-score</th><th>ROE</th><th>现金 / 利润</th><th>扣非盈利收益率</th></tr>'
    for c in results['companies']:
        m = c['metrics']
        fscore = str(c['f_score']) + '/9' if c['f_score'] is not None else f"未知（{c['f_score_known']}/9 项可用）"
        intro += f"<tr><td>{escape(c['ticker'])}<br><span class='muted'>{escape(c['name'])}</span></td><td>{escape(c['market'])} / {escape(c['sector'])}</td><td>{STATUS_CN[c['status']]}</td><td>{display(c['score'])}</td><td>{escape(fscore)}</td><td>{display(m.get('roe'), True)}</td><td>{display(m.get('cash_conversion'))}</td><td>{display(m.get('core_earnings_yield'), True)}</td></tr>"
    intro += '</table><h2>逐家公司复核</h2>'
    docs_by_ticker = {}
    for document in results['documents']['evidence']:
        docs_by_ticker.setdefault(document['ticker'], []).append(document)
    for c in results['companies']:
        intro += f"<details><summary>{escape(c['ticker'])} · {STATUS_CN[c['status']]} · {display(c['score'])}</summary>"
        for label, values in (('筛选理由', c['reasons']), ('数据与业务复核', c['warnings'])):
            if values:
                intro += f'<p><b>{label}</b></p><ul>' + ''.join(f'<li>{escape(v)}</li>' for v in values) + '</ul>'
        if c['blocks']:
            intro += '<p>' + ' · '.join(f'{BLOCK_CN[key]} {display(value["score"])}' for key, value in c['blocks'].items()) + '</p>'
        intro += '<table><tr><th>指标</th><th>原始计算结果</th></tr>'
        for key, value in c['metrics'].items():
            intro += f'<tr><td>{escape(key)}</td><td>{escape(display(value))}</td></tr>'
        intro += '</table><p><b>九项财务健康信号</b></p><ul>'
        for key, value in c['signals'].items():
            text = '未知' if value is None else ('通过' if value else '未通过')
            intro += f'<li>{escape(key)}：{text}</li>'
        intro += '</ul><p><b>财报来源</b></p><ul>'
        for source in c['sources']:
            intro += f'<li>{escape(source["period_end"])}，可用日期 {escape(source["available_at"])} · {escape(source["source_url"])} · {escape(source["filing_id"])}</li>'
        intro += '</ul>'
        if c.get('valuation_source'):
            intro += '<p><b>估值 / 汇率来源：</b>' + escape(c['valuation_source']) + '</p>'
        if c.get('ml_data_review'):
            intro += '<p><b>ML 数据复核（不参与评分）：</b>' + escape(c['ml_data_review']) + '</p>'
        for doc in docs_by_ticker.get(c['ticker'], []):
            intro += f'<p><b>待复核文本</b> {escape(doc["topics"])} · 页码 {escape(doc["page"])} · {escape(doc["source_url"])}</p><blockquote>{escape(doc["excerpt"])}</blockquote>'
        if c.get('next_questions'):
            intro += '<p><b>研究员跟进问题</b></p><ul>' + ''.join(f'<li>{escape(q)}</li>' for q in c['next_questions']) + '</ul>'
        intro += '</details>'
    intro += '<h2>数据处理记录</h2><p>' + escape(results['documents']['method']) + '</p>'
    intro += '<p>财务输入跳过 / 拒绝行：' + str(len(results['audit'])) + '；估值输入：' + str(len(results['valuation_audit'])) + '；文本：' + str(len(results['documents']['rejected'])) + '。详细原因见 audit.json。</p>'
    intro += '<p class="muted">年度模型尚未整合最近中报、TTM、分析师预测、银行专用指标或供应链实体匹配。应收、存货、治理、竞争优势和管理层判断需要原始资料及研究员判断。</p></html>'
    return intro


def main(argv=None):
    parser = argparse.ArgumentParser(description='A/H annual fundamental research screener')
    parser.add_argument('--demo', action='store_true', help='generate synthetic demonstration inputs')
    parser.add_argument('--statements', type=Path)
    parser.add_argument('--valuations', type=Path)
    parser.add_argument('--documents', type=Path)
    parser.add_argument('--as-of', required=True, help='inclusive end-of-day cutoff YYYY-MM-DD')
    parser.add_argument('--config', type=str, default=str(ROOT / 'config.json'))
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'demo')
    parser.add_argument('--query', help='optional lexical retrieval over evidence excerpts')
    parser.add_argument('--ml-cleaning', action='store_true', help='optional NumPy PCA review hints')
    parser.add_argument('--search-method', choices=('grid', 'beam'),
                        help='optional search for a research shortlist size; does not optimize returns')
    parser.add_argument('--search-market', choices=('A', 'HK'), help='explicit market for parameter search')
    parser.add_argument('--search-sector', help='explicit industry cohort for parameter search')
    parser.add_argument('--search-target-size', type=int, help='target candidate count (default: 5)')
    parser.add_argument('--search-beam-width', type=int, help='retained branches per layer (default: 5)')
    parser.add_argument('--search-max-evaluations', type=int, help='hard evaluation budget (default: 1000)')
    parser.add_argument('--search-top-k', type=int, help='number of alternatives to retain (default: 5)')
    parser.add_argument('--search-space', type=Path, help='optional JSON search-space object')
    args = parser.parse_args(argv)
    try:
        cutoff = iso(args.as_of)
        config = config_from(args.config)
        if args.search_method and (args.search_market is None or not args.search_sector or not args.search_sector.strip()):
            raise ValueError('--search-method requires explicit --search-market and --search-sector')
        if not args.search_method and any(value is not None for value in (
                args.search_market, args.search_sector, args.search_space, args.search_target_size,
                args.search_beam_width, args.search_max_evaluations, args.search_top_k)):
            raise ValueError('search options require --search-method')
        search_space = load_search_space(args.search_space) if args.search_space else None
        search_request = None
        if args.search_method:
            search_request = {
                'method': args.search_method, 'market': args.search_market,
                'sector': args.search_sector.strip(),
                'target_size': 5 if args.search_target_size is None else args.search_target_size,
                'beam_width': 5 if args.search_beam_width is None else args.search_beam_width,
                'max_evaluations': 1000 if args.search_max_evaluations is None else args.search_max_evaluations,
                'top_k': 5 if args.search_top_k is None else args.search_top_k, 'space': search_space,
            }
        if args.demo:
            if args.statements or args.valuations or args.documents:
                raise ValueError('--demo cannot be mixed with supplied input files')
            paths = create_demo(ROOT / 'examples')
            args.statements, args.valuations, args.documents = (paths[k] for k in ('statements', 'valuations', 'documents'))
        if args.statements is None or args.valuations is None:
            raise ValueError('provide --statements and --valuations, or use --demo')
        inputs = {'statements': load_csv(args.statements), 'valuations': load_csv(args.valuations)}
        if not inputs['statements']:
            raise ValueError('statement input is empty')
        results = screen(inputs['statements'], inputs['valuations'], cutoff, config)
        results['synthetic_demo'] = args.demo
        results['documents'] = process_documents(args.documents, cutoff) if args.documents else {
            'evidence': [], 'rejected': [], 'method': 'No document input supplied.'}
        known_tickers = {c['ticker'] for c in results['companies']}
        results['unmatched_document_tickers'] = sorted({d['ticker'] for d in results['documents']['evidence']} - known_tickers)
        results['ml_cleaning'] = {'status': 'not_requested'}
        if args.ml_cleaning:
            from anomaly import annotate_anomalies
            ml = annotate_anomalies(results['companies'])
            results['ml_cleaning'] = ml['summary']
            for c in results['companies']:
                c['ml_data_review'] = ml['annotations'].get(c['ticker'])
        if args.query:
            results['document_search'] = search_evidence(results['documents']['evidence'], args.query)
        if search_request:
            from search import search_configs
            results['configuration_search'] = search_configs(
                inputs['statements'], inputs['valuations'], cutoff, config,
                market=search_request['market'], sector=search_request['sector'],
                target_count=search_request['target_size'], method=search_request['method'],
                beam_width=search_request['beam_width'], max_evaluations=search_request['max_evaluations'],
                top_k=search_request['top_k'], space=search_request['space'])
        input_paths = {'statements': args.statements, 'valuations': args.valuations, 'documents': args.documents}
        hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in input_paths.items() if path}
        code_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in ROOT.glob('*.py')}
        numpy_version = None
        if args.ml_cleaning:
            try:
                numpy_version = importlib.metadata.version('numpy')
            except importlib.metadata.PackageNotFoundError:
                pass
        environment = {'python_version': sys.version, 'numpy_version': numpy_version,
                       'numpy_available': results['ml_cleaning'].get('numpy_available')}
        identity = {'as_of': cutoff.isoformat(), 'config': config, 'inputs': hashes, 'code': code_hashes,
                    'environment': environment,
                    'synthetic_demo': args.demo, 'ml_cleaning_requested': args.ml_cleaning, 'query': args.query,
                    'configuration_search_request': search_request}
        run_id = hashlib.sha256(dump_json(identity).encode('utf-8')).hexdigest()[:20]
        manifest = {'run_id': run_id, **identity, 'input_paths': {k: str(v.resolve()) for k, v in input_paths.items() if v},
                    'python_version': sys.version.split()[0]}
        results['run_id'] = run_id
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'results.json').write_text(dump_json(results), encoding='utf-8')
        (args.output / 'manifest.json').write_text(dump_json(manifest), encoding='utf-8')
        # Explicitly overwrite this generated artifact when search is disabled,
        # so reusing an output directory cannot display a previous run's proposal.
        (args.output / 'search.json').write_text(dump_json(results.get('configuration_search',
            {'status': 'not_requested', 'run_id': run_id})), encoding='utf-8')
        (args.output / 'audit.json').write_text(dump_json({
            'statements': results['audit'], 'valuations': results['valuation_audit'],
            'documents': results['documents']['rejected']}), encoding='utf-8')
        (args.output / 'report.html').write_text(report_html(results, args.demo), encoding='utf-8')
        export_csv(args.output / 'screen.csv', results['companies'])
        save_database(args.output / 'research.sqlite', run_id, manifest, results, inputs, args.documents)
        counts = Counter(c['status'] for c in results['companies'])
        print(dump_json({'run_id': run_id, 'synthetic_demo': args.demo,
                         'counts': dict(counts), 'report': str((args.output / 'report.html').resolve())}))
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f'Input/output error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
