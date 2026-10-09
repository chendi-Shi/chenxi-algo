# chenxi-algo：基本面股票筛选 Algo 项目

一个面向 A 股、港股非金融公司的 Python 投研工具。输入标准化年度财务数据与估值快照，输出可解释的研究候选池、来源记录和复核问题。适合 Data Analyst Intern 展示数据治理、财务理解、算法实现和投资团队交付能力。

已完成可运行原型：财务清洗 → 披露日期与版本选择 → 指标计算 → 同市场同行业比较 → 基本面筛选 → 文本证据整理 → SQLite 数据库与报告。当前示例全部是合成数据，项目没有验证投资效果。

## 一键运行

需要 Python 3.10+；核心流程只用标准库，不需要 API key。

```powershell
cd chenxi-algo
python run.py --demo --as-of 2026-10-09
```

可选 ML 清洗模块需要 NumPy。文本处理、基本面评分和数据库不依赖 NumPy。

```powershell
python -m pip install "numpy>=1.24,<3"
python run.py --demo --as-of 2026-10-09 --ml-cleaning --query "客户集中"
python -m unittest discover -s tests -v
```

运行后打开 `output/demo/report.html` 查看结果。`--demo` 每次确定性生成 `examples/` 下的虚构公司和财报，不下载真实股票数据。演示估值日期为 2026-10-08；修改截止日期可能按设计触发陈旧数据或未披露检查。

## 接入真实数据

从已有财务数据库导出并按 [DATA_DICTIONARY.md](DATA_DICTIONARY.md) 映射。不要直接把未经口径核对的免费接口输出当作统一数据库。

```powershell
python run.py --statements your_statements.csv --valuations your_valuations.csv --documents your_documents.jsonl --as-of 2026-10-09 --output output/real
```

`--documents` 可省略；文本应是已提取的财报、公告、访谈纪要或合法取得的另类数据，不直接读取 PDF。`available_at` 表示资料真正可被使用的日期，截止日期按当日结束、包含当天处理。历史重述保留原版本及新版披露日期。

每家公司需要连续三份年度合并财报。第三份为当期，前两份提供期初资产、平均权益和两年增长基期。季度及中报不能混入；币种改变、期间不连续、异常单位、冲突版本进入人工核验。

## 算法

1. **清洗及可用性校验**：统一金额单位；保留缺失值；拒绝 NaN、Infinity、不符合约定的符号和元数据；只选择截止日已披露的版本。财务源、估值源、FX 源均保留；链接是来源指针，程序没有替研究员验证原文真实性。
2. **财务健康**：适配版 Piotroski 九项信号。ROA 与现金流质量用同一期初资产，杠杆用平均资产，增发用发行事件字段。缺项为未知，完整 F-score 才给 0–9 分；另外保留已知项数与通过项下界。
3. **基本面指标**：归母 ROE、现金利润转换、扣非利润占比、扣非盈利收益率、FCF 收益率、收入和扣非利润两年 CAGR、净债务/CFO、利息覆盖率。FCF 为经营现金流减现金资本开支；请核对利息分类及租赁会计口径。
4. **可比公司排名**：只在同市场、同行业内比较，每个指标至少五个有效观测。采用带并列平均秩的百分位；完全相同的同业得 50。组内每个指标等权，四组默认质量 40%、估值 25%、增长 20%、资产负债表 15%。权重为研究偏好，可改 `config.json`；没有经过收益优化或有效性校准。
5. **初筛与分流**：需正利润、正经营现金流、正归母权益、ROE ≥ 8%、CFO/利润 ≥ 0.8、净债务/CFO ≤ 4、完整 F-score ≥ 6、无保留审计意见；之后同业综合分 ≥ 65 为优先研究，否则观察。数据缺失或审计意见需核验进入 `data_review`。银行、保险、金融及 REIT 进入 `specialist_review`，应先统一行业分类。
6. **文本及异常复核**：中英文 TF-IDF 检索与相似文本提示、关键词复核主题、原文摘录与页码。可选 PCA 在同市场同行业至少 12 个完整样本上提示异常财务关系。主题命中不等于确认风险，PCA 不判定错误或造假，两者都不改变财务分数。

模型不会把零利息费用变成“无限高覆盖率”，也不会把负利润基期制造为高 CAGR。这些结果保留未知，综合分可能不生成。一个完整但较差的公司与一个数据不完整的公司有不同状态；不要把待核验等同于公司质量差。

估值必须使用**全公司、全部股权类别总市值**，并转换成报表币种。A/H 两地上市、不同币种和不同股份类别需要数据提供方或研究员先核对；只输入港股流通股市值会低估估值分母。公司输入为证券标识，尚未实现 A/H 同发行人的实体合并。

## 交付文件

| 文件 | 用途 |
|---|---|
| `run.py` | CLI、数据落库、结果导出 |
| `engine.py` | 财务口径、九项信号、基本面初筛、同业评分 |
| `evidence.py` | 文本清洗、去重、可追溯证据、TF-IDF 检索 |
| `anomaly.py` | 可选 PCA 数据复核提示 |
| `config.json` | 阈值、权重和专用行业配置 |
| `examples/` | 62 家虚构公司、年度数据、估值和文本样例 |
| `tests/` | 财务口径、时间隔离、文本、ML、导出与数据库测试 |
| `reference/` | 锁定提交的 GitHub 原版源码及 MIT 许可证 |
| `RESEARCH.md` | GitHub 项目对比、源码缺陷及公式来源 |

每次运行导出 `screen.csv`、`results.json`、`report.html`、`audit.json`、`manifest.json` 和 `research.sqlite`。CSV 可查看，完整缺失状态、组成分数与引用在 JSON / 数据库中。

SQLite 包含 `runs`、`raw_records`、`statements`、`companies`、`evidence`。原始输入与清洗后记录并存，可按 `run_id`、公司、截止日期复核。运行标识包含输入文件哈希、代码哈希、配置、截止日、Python / 可选 NumPy 环境和查询参数，便于复现；重复同一运行会更新同一标识。

示例查询：

```sql
SELECT ticker, market, sector, score
FROM companies
WHERE run_id = :run_id AND status = 'candidate'
ORDER BY market, sector, score DESC;
```

## 原版到改进版

底座借鉴 [ayondey47/piotroski-f-score](https://github.com/ayondey47/piotroski-f-score/blob/d97f9d8612ef6f7c96989ecea2300dcd6a5c81dd/piotroski/model.py) 的透明九项信号结构，保留 [MIT 许可证](reference/LICENSE) 与原版源码快照。其他财务库用于源码对比，详见 [RESEARCH.md](RESEARCH.md)。

| 原版 | 本项目 |
|---|---|
| 两年数据直接算健康分 | 三年年度数据、统一资产分母、披露与重述版本选择 |
| 不完整/不合理数值可能影响判断 | 缺失为未知、有效覆盖率、冲突记录与审计日志 |
| 股数变化推断增发 | 显式普通股发行事件，避免拆股、送股、回购抵销影响 |
| 单一财务健康分 | 质量、估值、增长、资产负债表四组；输出原始指标 |
| 无市场与行业语境 | A/H 独立同业组；金融和 REIT 专用分流 |
| 一次性输出 | 原始/清洗数据落库、可复现 manifest、文本证据与研究问题 |

这里的“改进”指数据可靠性、会计口径和研究交付；未声称排名更能预测收益。

## JD 对应的实习项目表述

> Built an auditable fundamental research screener for A-share and Hong Kong companies, with point-in-time financial ingestion, accounting-quality checks, peer-relative screening, SQLite lineage, and optional PCA anomaly triage / bilingual TF-IDF evidence retrieval.

真实部署前优先对十家公司做财报人工对账，再扩充基金认可的同业样本。后续可接中报/TTM、应收存货质量、分业务收入、盈利预测与供应链实体匹配；当前版本只处理年度数据，不覆盖这些模块。

检验投资研究效果应使用独立人工标注或后续财务结果，例如异常发现率、核验耗时和后续盈利兑现情况。当前测试验证程序行为，不替代这些实证评估。
