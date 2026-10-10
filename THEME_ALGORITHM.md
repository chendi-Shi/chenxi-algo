# 从产品线索寻找基本面研究公司

## 使用方法

需要 Python 3.10+。检索、财务分类和 TXT 导入只使用标准库；PDF 导入另需 `python -m pip install "pypdf>=5,<7"`。

```powershell
python discover.py --demo-corpus --query "光模块" --as-of 2026-10-10 --output output/optics
python discover.py --demo-corpus --query "AI服务器" --market HK --as-of 2026-10-10 --output output/ai-hk
python discover.py --verify output/optics
```

`--demo-corpus` 选择真实公开来源的短证据样例，9 家公司仅为开发验证集；不加该选项时必须明确输入公司库、文档库。下载并导入原文件全文后运行：

```powershell
python ingest_theme.py --manifest data/theme_ingest_manifest.json --output output/theme_full_documents.jsonl --download
python discover.py --query "机器人" --companies data/theme_companies.json --documents output/theme_full_documents.jsonl --as-of 2026-10-10 --output output/full-robots
```

PDF 清单覆盖 6 家，其余官网摘录仍在默认 JSONL 中；两个数据集覆盖范围不同。首次下载需要网络，来源内容变化导致 SHA256 不一致时会拒绝，需复核新版本后更新清单。扫描件无文本页被记录，尚无 OCR。完整正文只存本地 `output/` / `validation/raw/`，不提交到公有仓库。

自有资料使用 `--companies companies.json --documents documents.jsonl`。公司目录为 JSON 数组，必填 ticker、name、market（A/HK）、sector、scope（technology/manufacturing）、universe_as_of；证券代码与财务数据须一致，建议带交易所后缀。文档每行一个 JSON，必填 document_id、ticker、available_at、source_url、source_type、page、text。PDF 导入从 manifest 自动产生这些字段；TXT 页码为 1。目录与来源日期晚于截止日的记录不可用，元数据错误、重复冲突均进审计。官方名录采集与每日试跑见 [PRODUCTION.md](PRODUCTION.md)；全市场经核对的细分业务分类和跨上市地同发行人合并仍未完成。

接入财务 CSV，字段沿用 [DATA_DICTIONARY.md](DATA_DICTIONARY.md)：

```powershell
python discover.py --query "机器人" --companies companies.json --documents documents.jsonl --statements statements.csv --valuations valuations.csv --as-of 2026-10-10 --output output/research
```

`--statements` 和 `--valuations` 均可省略；只提供报表也能评估成长及改善，缺估值时不能进入相对价值名单。需要连续三份可比年度合并报表、真实可用日期及审计信息，不能混入季度数据。缺失字段保留未知；港股扣非口径、会计币种、A/H 全公司总市值仍需研究员核对。财务样本至少同市场同行业 5 家才能做相对价值比较，比较池是全部输入的合格财务公司，而非主题命中公司。

想直接看全部名单如何生成，可运行完全虚构的教学样例；不会给真实公司套用虚构财务：

```powershell
python demo_theme.py
python discover.py --query "机器人" --companies output/theme_demo_inputs/companies.json --documents output/theme_demo_inputs/documents.jsonl --statements output/theme_demo_inputs/statements.csv --valuations output/theme_demo_inputs/valuations.csv --as-of 2026-10-10 --output output/theme_demo
```

也可直接在 Python 中调用：

```python
from theme_search import discover_companies
from theme_financials import attach_fundamentals
matches = discover_companies(companies, documents, "机器人", "2026-10-10")
result = attach_fundamentals(matches, statements, valuations)
```

## 搜索怎么计算

1. 规范证券代码、筛选截止日可见资料、检查来源和元数据，拒绝同 ID 冲突，重复段落不增加公司分数。
2. 把输入主题展开成简繁中文、英文和产品别称。默认提供光模块、机器人、AI 服务器；可以在 `theme_config.json` 的 `retrieval.theme_dictionary` 加入 `{ "新主题": ["别称1", "英文名"] }`。
3. 对全文分句、处理长句窗口，保留字符起止位置；对有效文本片段做中文双字分词、英文词分词，建立倒排索引并计算 BM25。展示的证据必须实际命中主题或别称，单纯共享一两个字不足以召回。
4. 判断段落上下文：直接业务权重 1，产业链线索 0.7，规划/开发 0.35，不确定 0.3，明确否定 0。退出/停止、不涉及等否定不能独立召回；只提客户、对手或行业也不能据此确认公司自己经营。规则是启发式，所有关系均需人工核验，不构成已验证供销关系。
5. 公司分数取其最高加权段落分数，多次重复热词不会累加成更高公司总分。结果保留支持和否定证据、历史资料提示，按相关度排序；财务好坏不改变业务检索分数。

每个词的 BM25 贡献为：

`log(1 + (N - df + 0.5)/(df + 0.5)) × tf × (k1 + 1) / (tf + k1 × (1 - b + b × L/avgL))`

这里 N 是本次建立索引的有效文本片段数，df 是含该词的片段数，tf 是片段内词频，L 是片段长度；默认 k1=1.5、b=0.75。对扩展查询词求和，再乘上下文权重。分数只在同次查询和资料集内用于阅读顺序，不能理解为相关概率、收入占比或公司价值。BM25 来源：[Robertson & Zaragoza, 2009](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)。

## 怎样进入财务名单

以下默认值是可解释的研究假设，尚未用晨曦投资团队的历史判断校准；修改 `theme_config.json` 即可调整。公司可同时满足多个盈利风格，亏损公司单列。三个盈利风格还要求 `business_status=current_business`。规划、主体不确定、历史或存在否定冲突的公司保留在线索中，财务上通过的风格记录在 `financial_style_candidates`，供人工复核；亏损观察不因此消失。

提供 `--production-policy` 时，盈利风格还要求当前主题的经营支持证据满足指定时效、哈希和日期依据门槛。最新的一份无关公告不能替旧主题证据背书。支持原文保存在 `business_support_evidence`，输出同时给出 `theme_business_freshness` 和数据准入检查。

| 名单 | 默认条件 |
|---|---|
| 优质成长 quality_growth | 正归母/合并/扣非利润、正 CFO 与权益；ROE≥12%，CFO/合并利润≥1，收入和扣非利润两年 CAGR 均≥10%，扣非/归母≥80%，净债务/CFO≤2；全部成立 |
| 相对价值 relative_value | 正归母/合并/扣非利润、正 CFO 与权益；同市场同行业有效估值指标至少5家，扣非盈利收益率与FCF收益率平均百分位≥75，两项收益率都为正，净债务/CFO≤4，估值日期及币种合格 |
| 经营改善 operating_improvement | 经核验的连续年度数据；归母利润增加、归母净利率提高至少0.5个百分点、CFO增加，三项至少两项；当前不亏损 |
| 亏损观察 loss_watchlist | 最新已披露且可解析年报归母利润<0；即使历史不足或其他资料需复核也保留，并展示数据状态。亏损收窄只提示改善迹象，不进入盈利风格 |
| 未分类 unclassified | 财务资料不足、数据需复核，或已知财务未满足任何风格；具体原因、未知项及失败门槛分别记录 |

ROE＝当期归母利润/期初期末归母权益均值；净债务＝有息债务−现金；FCF＝CFO−现金资本开支；CAGR＝(当期/两年前)^(1/2)−1，负基期不计算。改善用绝对利润变化，避免把负基期除法包装成增长。异常单位、NaN/Infinity、缺债务/现金、非无保留审计、不可比期间等仍由原财务引擎拦截。缺估值不会阻断经营分析，但不能产生相对价值标签。

经营改善可能来自低基数或一次性事项，相对低估值可能反映风险；算法提供下一步研究名单，不输出买卖指令。尤其需要核验主题收入占比、客户关系和最新中报变化。

## 输出与验收

`results.json` 的 `all_companies` 保留所有匹配公司的证据和财务 checks，`companies` 是显示数量限制后的列表；文档证据本身的展示上限及是否截断另有标记。`all_matched_style_lists` 与全部公司对应，`style_lists` 与显示列表对应。`matches.csv` 为带业务状态的简表，`manifest.json` 保存输入及代码哈希、配置和参数；`completion.json` 最后写入，用 `--verify` 检查导出。单次入口复用目录会替换导出，持续运行请用 `research_job.py` 保留各次历史快照。

软件回归、真实段落标签测试与完整 PDF 召回分别检查。真实来源验证不会证明全市场准确率；完整文件新增命中需逐条审阅，不能把未标注公司默认为负例。验证脚本和结果在 `validation/`。上线供团队持续使用前仍需补齐实际研究股票池、财务口径映射、人工相关性标注及更新任务；本版提供可运行和可审计的研究原型。
