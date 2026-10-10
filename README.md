# chenxi-algo：基本面股票筛选 Algo 项目

**v0.2 主流程：输入产品或主题 → 从公司资料库发现相关 A/H 公司 → 展示原文证据和业务阶段 → 分为优质成长、相对价值、经营改善及亏损观察名单。** 第一版聚焦科技与制造，使用 Python，搜索与财务门槛均可调整。

项目覆盖公开资料导入、非结构化文本清洗、主题检索、结构化财务计算、数据审计和研究名单交付。默认附带 9 家真实公司的公开证据样例，以及 6 份完整公开 PDF 的可下载清单。样例是有限语料验证；尚未接入全 A/H 上市公司资料库，不能声称全市场扫描或证明投资收益有效。

```powershell
python discover.py --query "机器人" --as-of 2026-10-10 --output output/robots
python discover.py --verify output/robots
```

查看 `output/robots/results.json` 中的匹配原因、来源链接、物理页码、逐项财务判定；`matches.csv` 是简表。默认真实证据样例没有配套完整财务历史，因此财务分类为待补资料，不编造估值或利润。接入真实财务 CSV 和完整 PDF 的步骤、算法公式与全部可调门槛见 **[THEME_ALGORITHM.md](THEME_ALGORITHM.md)**；来源见 [THEMATIC_SOURCES.md](THEMATIC_SOURCES.md)，实测结果及剩余限制见 **[THEME_VALIDATION.md](THEME_VALIDATION.md)**。

原有财务引擎、SQLite 输出和网格/束搜索继续保留，作为专题研究的辅助工具。旧版财务审计记录见 [ENTERPRISE_VALIDATION.md](ENTERPRISE_VALIDATION.md)，它与新主题检索的实测范围分别记录。

## 原有财务引擎示例

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
python validation/validate_real.py
python validation/verify_hk_coverage.py
```

运行后打开 `output/demo/report.html` 查看结果。`--demo` 每次确定性生成 `examples/` 下的虚构公司和财报，不下载真实股票数据。演示估值日期为 2026-10-08；修改截止日期可能按设计触发陈旧数据或未披露检查。

## 搜索筛选组合

项目现支持**网格搜索和束搜索**，比较更严格的门槛与不同模块权重，寻找接近指定规模的同业研究名单。搜索保留原始数据核验门槛，详细算法、约束和自定义空间见 [SEARCH.md](SEARCH.md)。

```sh
python run.py --demo --as-of 2026-10-09 --search-method grid --search-market A --search-sector Consumer --search-target-size 3 --output output/search-grid
python run.py --demo --as-of 2026-10-09 --search-method beam --search-market A --search-sector Consumer --search-target-size 3 --search-beam-width 5 --output output/search-beam
```

结果另存为 `search.json`；主筛选仍按原始配置输出。搜索优化研究名单规模和配置变化，投资有效性需独立验证。

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
5. **初筛与分流**：需正利润、正经营现金流、正归母权益、ROE ≥ 8%、CFO/利润 ≥ 0.8、净债务/CFO ≤ 4、无保留审计意见；之后同业综合分 ≥ 65 为优先研究，否则观察。F-score 默认仅作诊断，避免误排除经营稳定的成熟企业；设置 `min_f_score: 6` 才启用完整九项的硬筛。数据缺失或审计意见需核验进入 `data_review`。银行、保险、金融及 REIT 进入 `specialist_review`，应先统一行业分类。
6. **文本及异常复核**：中英文 TF-IDF 检索与相似文本提示、关键词复核主题、原文摘录与页码。可选 PCA 在同市场同行业至少 12 个完整样本上提示异常财务关系。主题命中不等于确认风险，PCA 不判定错误或造假，两者都不改变财务分数。

零利息费用仍不会产生无限覆盖率。如果已核验总有息债务为零、利息费用也为零，则以明确的无债务标记参与债务服务能力排序；全组同样无债时该项得中性 50。未知利息或有债但零利息需要复核。负利润基期不计算 CAGR，前期或当期权益非正不计算可比 ROE。

CFO/利润和扣非/归母的质量评分贡献分别在 2 和 1 封顶，原始数值完整保留；超过 3 或 1.5 时另提示核对营运资本、现金分类和非经常性损失。这是研究规则，尚未实证校准。权重为零的模块无需评分字段，但净债务等基础健康门槛仍然有效。数据不完整与财务较弱有不同状态。

如已确定某份年度报告应当可用，可配置 `expected_latest_period: "2025-12-31"`，缺失即人工复核。默认年报期末距截止日超过一年会提醒核对最新年报/中报，超过 550 天拒绝；具体应披露期间需按市场、公司财年及实际公告核验。

估值必须使用**全公司、全部股权类别总市值**，并转换成报表币种。A/H 两地上市、不同币种和不同股份类别需要数据提供方或研究员先核对；只输入港股流通股市值会低估估值分母。公司输入为证券标识，尚未实现 A/H 同发行人的实体合并。

## 交付文件

| 文件 | 用途 |
|---|---|
| `discover.py` / `theme_search.py` | 新主入口；主题扩展、BM25 检索、来源及业务阶段 |
| `theme_financials.py` / `theme_config.json` | 三类基本面研究风格、亏损观察、逐项门槛解释 |
| `ingest_theme.py` / `data/` | 公开 PDF/TXT 导入、真实证据与来源清单 |
| `THEME_ALGORITHM.md` / `THEMATIC_SOURCES.md` | 新流程使用方法、公式、边界、实测来源 |
| `run.py` | CLI、数据落库、结果导出 |
| `engine.py` | 财务口径、九项信号、基本面初筛、同业评分 |
| `search.py` / `SEARCH.md` | 受约束的网格与束搜索、研究名单规模目标及配置对照 |
| `evidence.py` | 文本清洗、去重、可追溯证据、TF-IDF 检索 |
| `anomaly.py` | 可选 PCA 数据复核提示 |
| `config.json` | 阈值、权重和专用行业配置 |
| `examples/` | 62 家虚构公司、年度数据、估值和文本样例 |
| `tests/` | 财务口径、时间隔离、文本、ML、导出与数据库测试 |
| `reference/` | 锁定提交的 GitHub 原版源码及 MIT 许可证 |
| `RESEARCH.md` | GitHub 项目对比、源码缺陷及公式来源 |
| `VALIDATION.md` / `validation/` | 真实财报对账、经济边界测试、权重敏感性、公开数据及可复现结果 |

`run.py` 每次运行导出 `screen.csv`、`results.json`、`report.html`、`audit.json`、`manifest.json`、`search.json` 和 `research.sqlite`。未启用搜索时，`search.json` 明确记录 `not_requested`，避免复用输出目录时残留旧方案。CSV 可查看，完整缺失状态、组成分数与引用在 JSON / 数据库中。

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

> Built an auditable fundamental research screener for A-share and Hong Kong companies, with availability-aware version selection, accounting-quality checks, peer-relative screening, SQLite lineage, and optional PCA anomaly triage / bilingual TF-IDF evidence retrieval. Reconciled selected public financial facts to original annual reports and evaluated screening sensitivity and data-coverage constraints.

真实部署前应扩充财报人工对账、统一债务与港股利润调整口径，并建立保留原始披露版本的数据库。后续可接中报/TTM、应收存货质量、分业务收入、盈利预测与供应链实体匹配；当前版本只处理年度数据。

本次小样本探索中，综合诊断分与 2025 年三项财务结果的秩相关均低于单独 ROE 对照；不同权重也会改变前两名。历史接口有后续更新，不能将这次重构当作严格回测。进一步验证应使用真实历史版本、多行业多年度样本，以及独立人工复核或后续盈利兑现结果。
